"""Experiment logger: W&B when available, CSV fallback otherwise."""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None


class ExperimentLogger:
    def __init__(self, run_name: str, config: dict, csv_path: Optional[Path] = None,
                 project: str = "occlusion-pedestrian", enabled: bool = True):
        self.run_name = run_name
        self.csv_path = csv_path
        self._enabled = enabled and bool(project)
        self._wandb_run = None
        self._fields: list[str] = []

        if wandb is not None and self._enabled:
            try:
                self._wandb_run = wandb.init(project=project, name=run_name, config=config)
            except Exception as e:  # pragma: no cover
                logger.warning("W&B init failed (%s); falling back to CSV", e)
                self._wandb_run = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)

    def log_epoch(self, metrics: Dict[str, Any]) -> None:
        if self._wandb_run is not None:
            self._wandb_run.log(metrics)
        if self.csv_path is not None:
            self._write_csv(metrics)

    def _write_csv(self, metrics: Dict[str, Any]) -> None:
        fields = list(metrics.keys())
        new_file = not self.csv_path.exists()
        if new_file or fields != self._fields:
            self._fields = fields
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            if new_file:
                w.writeheader()
            w.writerow({k: metrics.get(k, "") for k in self._fields})

    def finish(self) -> None:
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()
