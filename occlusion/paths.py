"""Portable path resolution. No hardcoded home directories.

Resolve data and output roots from environment variables, falling back to
a local ``./data`` and ``./results`` layout so a fresh clone works with no
configuration. Override for cloud boxes:

    export OCCLUSION_DATA_ROOT=/kaggle/input/occlusion-data
    export OCCLUSION_OUTPUT_ROOT=/kaggle/working/results
"""
from __future__ import annotations

import os
from pathlib import Path

_DATA_ROOT = Path(os.environ.get("OCCLUSION_DATA_ROOT", "data"))
_OUTPUT_ROOT = Path(os.environ.get("OCCLUSION_OUTPUT_ROOT", "results"))


def data_root() -> Path:
    return _DATA_ROOT


def output_root() -> Path:
    return _OUTPUT_ROOT


def kitti_root() -> Path:
    return _DATA_ROOT / "kitti"


def citypersons_root() -> Path:
    return _DATA_ROOT / "citypersons"


def yolo_root(dataset: str) -> Path:
    return _DATA_ROOT / f"{dataset}_yolo"


def depth_root(dataset: str) -> Path:
    return _DATA_ROOT / f"{dataset}_depth"


def dataset_yaml(dataset: str) -> Path:
    return _DATA_ROOT / f"{dataset}_yolo.yaml"


def checkpoint_dir() -> Path:
    return _OUTPUT_ROOT / "checkpoints"


def log_dir() -> Path:
    return _OUTPUT_ROOT / "logs"


def figures_dir() -> Path:
    return _OUTPUT_ROOT / "figures"


def tables_dir() -> Path:
    return _OUTPUT_ROOT / "tables"


def narratives_dir() -> Path:
    return _OUTPUT_ROOT / "narratives"


def ensure_output_dirs() -> None:
    for d in (checkpoint_dir(), log_dir(), figures_dir(), tables_dir(), narratives_dir()):
        d.mkdir(parents=True, exist_ok=True)
