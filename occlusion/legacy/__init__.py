"""Compatibility surface for the analysis notebooks.

Re-exports the symbols the notebooks import under their old names. The
self-contained analysis modules (``datasets``, ``augmentation``, ``plotting``,
``logger``) live here because they pre-date the Ultralytics-native pipeline and
have no direct equivalent in the rest of the package. Canonical training runs
through ``train.py`` and the ``occlusion`` package proper, not this module.
"""
from __future__ import annotations

from enum import Enum

from ..config import RunConfig, load_config
from ..data.depth import MiDaSDepthEstimator
from ..metrics import (
    compute_fn_rate_hard,
    compute_fps,
    compute_kitti_ap,
    compute_ors,
    sample_to_annotation,
)
from ..models.fusion import CrossAttentionFusion, MultiHeadCrossAttention
from ..models.visibility import VisibilityMapHead
from ..seeds import set_all_seeds
from .augmentation import LabelAwareCutout, LabelAwareGridMask, RealOccluderAugmentation
from .datasets import KITTIDataset, collate_fn
from .logger import ExperimentLogger
from .plotting import (
    create_narrative_template,
    plot_ablation_bar_chart,
    plot_augmentation_strength_sweep,
    plot_depth_map_comparison,
    plot_occlusion_performance_curve,
    plot_results_table,
)


class RunMode(str, Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    COLAB = "colab"


__all__ = [
    "RunMode", "RunConfig", "load_config", "set_all_seeds",
    "compute_kitti_ap", "compute_ors", "compute_fn_rate_hard", "compute_fps",
    "sample_to_annotation",
    "MiDaSDepthEstimator",
    "CrossAttentionFusion", "MultiHeadCrossAttention", "VisibilityMapHead",
    "KITTIDataset", "collate_fn",
    "LabelAwareCutout", "LabelAwareGridMask", "RealOccluderAugmentation",
    "ExperimentLogger",
    "create_narrative_template", "plot_ablation_bar_chart",
    "plot_augmentation_strength_sweep", "plot_occlusion_performance_curve",
    "plot_results_table", "plot_depth_map_comparison",
]
