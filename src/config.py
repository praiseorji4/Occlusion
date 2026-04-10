"""
src/config.py — Run configuration and reproducibility utilities.

Serves hypothesis: all experiments must be fully reproducible (seed=42, ±0.1 AP).
Provides mode-switched paths so the same code runs locally, on Kaggle, and on Colab
without any path edits.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

class RunMode(Enum):
    """
    Execution context selector.

    LOCAL  — smoke-test on a laptop (small data, few epochs, CPU or single GPU).
    KAGGLE — full training on Kaggle T4/P100 GPUs (/kaggle/input paths).
    COLAB  — full training on Google Colab A100 (Drive checkpoints).
    EVAL   — load a checkpoint and run test-set evaluation only; no training.
    RESUME — continue an interrupted training run from the last checkpoint.
    """
    LOCAL  = "local"
    KAGGLE = "kaggle"
    COLAB  = "colab"
    EVAL   = "eval"
    RESUME = "resume"


# ---------------------------------------------------------------------------
# Per-mode defaults
# ---------------------------------------------------------------------------

_MODE_DEFAULTS: dict[RunMode, dict] = {
    RunMode.LOCAL: {
        "epochs": 3,
        "batch": 4,
        "data_root": Path("/home/chibueze/Documents/Projects/Occlusion/data"),
        "output_dir": Path("/home/chibueze/Documents/Projects/Occlusion/results"),
    },
    RunMode.KAGGLE: {
        "epochs": 100,
        "batch": 32,
        "data_root": Path("/kaggle/input/occlusion-data"),
        "output_dir": Path("/kaggle/working/results"),
    },
    RunMode.COLAB: {
        "epochs": 100,
        "batch": 32,
        "data_root": Path("/content/drive/MyDrive/occlusion/data"),
        "output_dir": Path("/content/drive/MyDrive/occlusion/results"),
    },
    RunMode.EVAL: {
        "epochs": 0,
        "batch": 16,
        "data_root": Path("/home/chibueze/Documents/Projects/Occlusion/data"),
        "output_dir": Path("/home/chibueze/Documents/Projects/Occlusion/results"),
    },
    RunMode.RESUME: {
        "epochs": 100,
        "batch": 4,
        "data_root": Path("/home/chibueze/Documents/Projects/Occlusion/data"),
        "output_dir": Path("/home/chibueze/Documents/Projects/Occlusion/results"),
    },
}


# ---------------------------------------------------------------------------
# RunConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """
    Centralised experiment configuration.

    All paths are derived from `mode`; never hardcode paths elsewhere.
    Override `data_root` / `output_dir` only when running on a non-standard
    cluster layout.

    Hypothesis served: reproducibility gate — seed controls all randomness,
    amp and batch size are set per-mode to avoid OOM without manual tuning.
    """
    mode: RunMode

    # Populated from _MODE_DEFAULTS if not explicitly set
    epochs: int = 0           # 0 = use mode default
    batch: int = 0            # 0 = use mode default
    _data_root: Optional[Path] = field(default=None, repr=False)
    _output_dir: Optional[Path] = field(default=None, repr=False)

    # Training knobs
    seed: int = 42
    imgsz: int = 640
    amp: bool = True
    checkpoint_metric: str = "AP_hard"

    # Model / experiment identity
    model_id: str = "M0"
    aug_p: float = 0.0

    # Limiting data for smoke tests (None = full dataset)
    data_limit: Optional[int] = None

    # Checkpoint path for EVAL / RESUME modes
    checkpoint_path: Optional[str] = None

    def __post_init__(self) -> None:
        defaults = _MODE_DEFAULTS[self.mode]
        if self.epochs == 0:
            self.epochs = defaults["epochs"]
        if self.batch == 0:
            self.batch = defaults["batch"]

    # --- path properties ----------------------------------------------------

    @property
    def data_root(self) -> Path:
        if self._data_root is not None:
            return self._data_root
        return _MODE_DEFAULTS[self.mode]["data_root"]

    @data_root.setter
    def data_root(self, value: Path) -> None:
        self._data_root = value

    @property
    def output_dir(self) -> Path:
        if self._output_dir is not None:
            return self._output_dir
        return _MODE_DEFAULTS[self.mode]["output_dir"]

    @output_dir.setter
    def output_dir(self, value: Path) -> None:
        self._output_dir = value

    # --- derived paths ------------------------------------------------------

    @property
    def kitti_root(self) -> Path:
        """Root of the KITTI dataset on disk."""
        return self.data_root / "kitti"

    @property
    def citypersons_root(self) -> Path:
        """Root of the CityPersons dataset on disk."""
        return self.data_root / "citypersons"

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def narratives_dir(self) -> Path:
        return self.output_dir / "narratives"

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [
            self.checkpoint_dir,
            self.log_dir,
            self.figures_dir,
            self.tables_dir,
            self.narratives_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def is_training(self) -> bool:
        return self.mode not in (RunMode.EVAL,)

    def __repr__(self) -> str:
        return (
            f"RunConfig(mode={self.mode.value}, model={self.model_id}, "
            f"epochs={self.epochs}, batch={self.batch}, seed={self.seed}, "
            f"aug_p={self.aug_p}, data_root={self.data_root})"
        )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = 42) -> None:
    """
    Lock down all sources of randomness for reproducibility.

    Verification: run M0 twice with seed=42 — val AP_hard must agree within ±0.1.
    Calls torch.use_deterministic_algorithms(True) which may slow some ops but
    guarantees bitwise reproducibility on the same hardware.

    Args:
        seed: Integer seed. Default 42 matches configs/base.yaml.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Config factory from YAML
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, overrides: Optional[dict] = None) -> RunConfig:
    """
    Load a RunConfig from a YAML file and apply any CLI overrides.

    The YAML must contain at least a `run_mode` key.  All other keys are
    optional and fall back to RunConfig defaults.

    Args:
        yaml_path: Path to a configs/*.yaml file.
        overrides: Dict of key→value pairs to override after loading YAML.

    Returns:
        Fully populated RunConfig.
    """
    import yaml  # local import to avoid hard dependency at module level

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw.update(overrides)

    mode = RunMode(raw.pop("run_mode", "local"))
    cfg = RunConfig(mode=mode)

    # Apply scalar fields
    for key in ("epochs", "batch", "seed", "imgsz", "amp",
                 "checkpoint_metric", "model_id", "aug_p",
                 "data_limit", "checkpoint_path"):
        if key in raw:
            setattr(cfg, key, raw[key])

    # Path overrides
    if "data_root" in raw:
        cfg.data_root = Path(raw["data_root"])
    if "output_dir" in raw:
        cfg.output_dir = Path(raw["output_dir"])

    return cfg
