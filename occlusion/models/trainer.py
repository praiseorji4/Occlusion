"""Unified Ultralytics trainer wired to the config-driven model + dataset.

One ``OcclusionDetectionTrainer`` covers every experiment row: it builds the
model via ``models.build`` (fusion / FEM / POPAM / visibility head), builds an
``OcclusionYOLODataset`` that optionally pairs depth and inserts occlusion
augmentation, and keeps the archive's OOM-retry + W&B/CSV logging.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .. import paths
from ..config import RunConfig
from ..data.dataset import build_dataset
from ..data.transforms import build_aug_config
from .build import build_model
from .fusion import patch_autobackend_warmup

try:
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import colorstr
except Exception:  # pragma: no cover - ultralytics is a hard dep at runtime
    DetectionTrainer = object
    colorstr = lambda s: s


def _aug_config(cfg: RunConfig) -> dict:
    if cfg.aug_strategy in (None, "none"):
        return {}
    return build_aug_config(cfg.aug_strategy, cfg.aug_p)


def _depth_root(cfg: RunConfig) -> Optional[Path]:
    return paths.depth_root(cfg.dataset) if cfg.uses_depth else None


class OcclusionDetectionTrainer(DetectionTrainer):
    """DetectionTrainer with occlusion augmentation, depth pairing and fusion."""

    def __init__(self, cfg: RunConfig, overrides: dict, _callbacks=None):
        self.cfg = cfg
        if cfg.uses_depth:
            patch_autobackend_warmup()
        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        self.data["channels"] = 3
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        build_model(self.cfg, model)
        if self.cfg.uses_depth:
            self.data["channels"] = 4
            self._persist_channels(4)
        return model

    def _persist_channels(self, ch: int) -> None:
        data_path = self.args.data
        if isinstance(data_path, str) and os.path.exists(data_path):
            with open(data_path) as f:
                d = yaml.safe_load(f) or {}
            d["channels"] = ch
            with open(data_path, "w") as f:
                yaml.dump(d, f, default_flow_style=False)

    def build_dataset(self, img_path, mode="train", batch=None):
        return build_dataset(
            self, img_path, mode, batch,
            aug_config=_aug_config(self.cfg),
            label_aware=self.cfg.label_aware,
            depth_root=_depth_root(self.cfg),
            seed=self.cfg.seed,
        )

    def get_validator(self):
        validator = super().get_validator()
        cfg = self.cfg

        def _build(img_path, mode="val", batch=None):
            return build_dataset(
                validator, img_path, mode, batch or self.args.batch,
                aug_config={}, label_aware=False,
                depth_root=_depth_root(cfg), seed=cfg.seed,
            )
        validator.build_dataset = _build
        if cfg.uses_depth:
            self._patch_validator_plots(validator)
        return validator

    def _patch_validator_plots(self, validator) -> None:
        def strip(batch):
            if isinstance(batch, dict) and "img" in batch:
                b = dict(batch)
                b["img"] = batch["img"][:, :3, :, :] if batch["img"].dim() == 4 else batch["img"][:3]
                return b
            return batch
        for name in ("plot_val_samples", "plot_predictions"):
            orig = getattr(validator, name, None)
            if orig is None:
                continue
            setattr(validator, name, lambda *a, _o=orig, **k: _o(strip(a[0]), *a[1:], **k))

    def plot_training_samples(self, batch, ni):
        if self.cfg.uses_depth and batch.get("img", None) is not None and batch["img"].dim() == 4:
            batch = {**batch, "img": batch["img"][:, :3, :, :]}
        super().plot_training_samples(batch, ni)


def build_trainer(cfg: RunConfig, extra: Optional[dict] = None) -> OcclusionDetectionTrainer:
    """Construct the trainer with ultralytics overrides derived from ``cfg``."""
    overrides = cfg.to_overrides()
    overrides["data"] = str(paths.dataset_yaml(cfg.dataset))
    overrides["project"] = str(paths.output_root() / "runs")
    overrides["name"] = cfg.run_name
    if extra:
        overrides.update(extra)
    return OcclusionDetectionTrainer(cfg, overrides)
