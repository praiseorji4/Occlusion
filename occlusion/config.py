"""Run configuration with YAML ``_base_`` inheritance.

Configs compose: an experiment lists one or more bases under ``_base_``; each
base is resolved recursively and merged left-to-right, then the experiment's
own keys win. Paths come from environment variables (see ``paths.py``), never
from hardcoded home directories.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

FUSION_TYPES = ("none", "early", "late_feature", "cross_attention")
AUG_STRATEGIES = (
    "none", "cutout", "random_erasing", "hide_and_seek", "gridcut",
    "body_part", "combined_all", "combined_best_two",
)


@dataclass
class RunConfig:
    model: str = "yolov8x"
    dataset: str = "kitti"
    experiment: str = "m0_baseline"

    imgsz: int = 640
    epochs: int = 150
    batch: int = 16
    optimizer: str = "SGD"
    lr0: float = 0.001
    lrf: float = 0.01
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    dropout: float = 0.0
    mosaic: float = 1.0
    scale: float = 0.5
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    cls_weight: float = 0.5
    box_weight: float = 7.5
    translate: float = 0.1
    fliplr: float = 0.5
    mixup: float = 0.0

    seed: int = 42
    amp: bool = True
    patience: int = 20
    workers: int = 8

    aug_strategy: str = "none"
    aug_p: float = 0.0
    label_aware: bool = False

    fusion_type: str = "none"
    fusion_scale: str = "P4"
    use_fem: bool = False
    use_popam: bool = False
    use_visibility_head: bool = False
    lambda_vis: float = 0.5
    lambda_occ_consistency: float = 0.1

    checkpoint_metric: str = "AP_hard"
    nc: int = 1
    names: dict = field(default_factory=lambda: {0: "pedestrian"})

    mode: str = "local"
    data_limit: Optional[int] = None
    checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode == "cloud":
            if self.batch == 16:
                self.batch = 32
        if self.fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got {self.fusion_type!r}")
        if self.aug_strategy not in AUG_STRATEGIES:
            raise ValueError(f"aug_strategy must be one of {AUG_STRATEGIES}, got {self.aug_strategy!r}")

    @property
    def uses_depth(self) -> bool:
        return self.fusion_type in ("early", "late_feature", "cross_attention")

    @property
    def run_name(self) -> str:
        return f"{self.experiment}_{self.dataset}_{self.model}_seed{self.seed}"

    def to_overrides(self) -> dict:
        """Ultralytics trainer override dict for the training knobs."""
        return {
            "model": f"{self.model}.pt",
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "warmup_epochs": self.warmup_epochs,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "patience": self.patience,
            "workers": self.workers,
            "amp": self.amp,
            "deterministic": True,
            "seed": self.seed,
            "hsv_h": self.hsv_h, "hsv_s": self.hsv_s, "hsv_v": self.hsv_v,
            "translate": self.translate, "fliplr": self.fliplr,
            "mixup": self.mixup, "mosaic": self.mosaic, "scale": self.scale,
            "cls": self.cls_weight, "box": self.box_weight,
        }


def _resolve_base(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    bases = raw.pop("_base_", None)
    merged: dict[str, Any] = {}
    if bases:
        if isinstance(bases, str):
            bases = [bases]
        for b in bases:
            base_path = Path(b)
            if not base_path.is_absolute():
                # Resolve relative to the configs root, not the file's own directory.
                resolved = _CONFIGS_DIR / base_path
                if not resolved.exists() and path.parent.joinpath(base_path).exists():
                    resolved = path.parent / base_path
                base_path = resolved
            merged.update(_resolve_base(base_path.resolve()))
    merged.update(raw)
    return merged


def _coerce(cfg: RunConfig, data: dict) -> None:
    valid = {f.name for f in fields(cfg)}
    for key, value in data.items():
        if key not in valid:
            continue
        setattr(cfg, key, value)


def load_config(yaml_path: str | Path, overrides: Optional[dict] = None) -> RunConfig:
    path = Path(yaml_path).resolve()
    merged = _resolve_base(path)
    if overrides:
        merged.update(overrides)
    cfg = RunConfig()
    _coerce(cfg, merged)
    cfg.__post_init__()
    return cfg


def configs_dir() -> Path:
    return _CONFIGS_DIR
