"""
src/logger.py — Experiment logging with wandb primary and CSV fallback.

Every metric emitted here has a corresponding CSV row, ensuring that every
claimed performance number in the paper is traceable to a saved file even
if wandb is unavailable (e.g. offline Kaggle kernels).

Run name format: {model_id}_{dataset}_{aug_p}_{seed}_{timestamp}
Example:         M2_kitti_p04_seed42_20251101_143022
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import RunConfig

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """
    Unified experiment logger.

    Primary backend: wandb (auto-disabled if wandb is not installed or
    WANDB_MODE=disabled is set).
    Fallback backend: CSV at results/logs/{run_id}/metrics.csv

    At training end, saves:
      - best checkpoint (by AP_hard on val set)
      - config YAML snapshot
      - confusion matrix as a wandb artefact / PNG

    Args:
        cfg:     RunConfig for the current experiment.
        dataset: Dataset name string (e.g. 'kitti', 'citypersons').
        project: wandb project name.
    """

    def __init__(
        self,
        cfg: RunConfig,
        dataset: str = "kitti",
        project: str = "occlusion-aware-detection",
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        aug_tag = f"p{int(cfg.aug_p * 10):02d}"
        self.run_id = f"{cfg.model_id}_{dataset}_{aug_tag}_seed{cfg.seed}_{ts}"

        self.log_dir = cfg.log_dir / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None  # initialised on first log

        self._best_ap_hard: float = -1.0
        self._best_epoch: int = -1
        self._wandb_run = None
        self._use_wandb = False

        self._init_wandb(project)

        # Save config snapshot
        self._save_config_snapshot()

        logger.info("ExperimentLogger initialised. run_id=%s", self.run_id)
        logger.info("Metrics CSV: %s", self._csv_path)

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _init_wandb(self, project: str) -> None:
        try:
            import wandb  # type: ignore

            if wandb.run is not None:
                wandb.finish()

            self._wandb_run = wandb.init(
                project=project,
                name=self.run_id,
                config={
                    "model_id": self.cfg.model_id,
                    "dataset": self.dataset,
                    "epochs": self.cfg.epochs,
                    "batch": self.cfg.batch,
                    "seed": self.cfg.seed,
                    "aug_p": self.cfg.aug_p,
                    "imgsz": self.cfg.imgsz,
                    "amp": self.cfg.amp,
                    "mode": self.cfg.mode.value,
                },
                resume="allow",
                id=self.run_id,
            )
            self._use_wandb = True
            logger.info("wandb initialised: %s", self._wandb_run.url)
        except Exception as exc:
            logger.warning("wandb unavailable (%s). Falling back to CSV.", exc)
            self._use_wandb = False

    def _save_config_snapshot(self) -> None:
        import yaml  # type: ignore

        snapshot = {
            "run_id": self.run_id,
            "model_id": self.cfg.model_id,
            "dataset": self.dataset,
            "mode": self.cfg.mode.value,
            "epochs": self.cfg.epochs,
            "batch": self.cfg.batch,
            "seed": self.cfg.seed,
            "aug_p": self.cfg.aug_p,
            "imgsz": self.cfg.imgsz,
            "amp": self.cfg.amp,
            "checkpoint_metric": self.cfg.checkpoint_metric,
            "data_root": str(self.cfg.data_root),
            "output_dir": str(self.cfg.output_dir),
        }
        config_path = self.log_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(snapshot, f, default_flow_style=False)

    # -----------------------------------------------------------------------
    # Per-epoch logging
    # -----------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        checkpoint_path: Optional[Path] = None,
    ) -> bool:
        """
        Log all metrics for one epoch.

        Metrics dict should contain (all optional, logged if present):
          Train losses:
            train/loss_total, train/loss_box, train/loss_cls,
            train/loss_dfl, train/loss_vis, train/loss_occ_consistency
          Val metrics:
            val/mAP_50, val/AP_easy, val/AP_mod, val/AP_hard,
            val/ORS, val/FN_rate_hard
          LR:
            lr
          Hardware:
            gpu_mem_gb

        Args:
            epoch:           Current epoch (1-indexed).
            metrics:         Dict of metric_name → float.
            checkpoint_path: If provided, copy it as best checkpoint when
                             AP_hard improves.

        Returns:
            True if a new best checkpoint was saved.
        """
        metrics["epoch"] = epoch
        metrics["timestamp"] = time.time()

        # --- CSV ---
        if self._csv_writer is None:
            fieldnames = sorted(metrics.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fieldnames, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(metrics)
        self._csv_file.flush()

        # --- wandb ---
        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb  # type: ignore
                self._wandb_run.log(metrics, step=epoch)
            except Exception as exc:
                logger.warning("wandb log failed: %s", exc)

        # --- best checkpoint ---
        ap_hard = metrics.get("val/AP_hard", -1.0)
        new_best = False
        if ap_hard > self._best_ap_hard and checkpoint_path is not None:
            self._best_ap_hard = ap_hard
            self._best_epoch = epoch
            best_path = self.cfg.checkpoint_dir / f"{self.cfg.model_id}_best.pt"
            best_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, best_path)
            logger.info(
                "New best AP_hard=%.2f at epoch %d → %s", ap_hard, epoch, best_path
            )
            new_best = True

        return new_best

    # -----------------------------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------------------------

    def log_confusion_matrix(self, cm: Any, class_names: list[str]) -> None:
        """
        Save confusion matrix as PNG and optionally log to wandb.

        Args:
            cm:           Numpy array (N, N) confusion matrix.
            class_names:  List of class label strings.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True label",
            xlabel="Predicted label",
            title=f"Confusion Matrix — {self.run_id}",
        )
        plt.tight_layout()
        cm_path = self.log_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)

        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb  # type: ignore
                self._wandb_run.log({"confusion_matrix": wandb.Image(str(cm_path))})
            except Exception as exc:
                logger.warning("wandb confusion matrix log failed: %s", exc)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a dict summarising the best result for this run."""
        return {
            "run_id": self.run_id,
            "best_AP_hard": self._best_ap_hard,
            "best_epoch": self._best_epoch,
            "log_dir": str(self.log_dir),
        }

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def finish(self) -> None:
        """Flush CSV, upload wandb artefacts, close handles."""
        self._csv_file.flush()
        self._csv_file.close()

        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb  # type: ignore

                # Upload CSV as artefact
                artifact = wandb.Artifact(
                    name=f"metrics-{self.run_id}",
                    type="metrics",
                )
                artifact.add_file(str(self._csv_path))
                self._wandb_run.log_artifact(artifact)
                self._wandb_run.finish()
            except Exception as exc:
                logger.warning("wandb finish failed: %s", exc)

        logger.info("ExperimentLogger finished. Summary: %s", json.dumps(self.summary()))

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()
