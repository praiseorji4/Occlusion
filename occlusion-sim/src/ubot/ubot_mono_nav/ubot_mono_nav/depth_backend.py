r"""Metric depth from one RGB image, plus the per-camera calibration.

Deliberately thin. The source of truth for WHICH model and WHY is the Occlusion
repo's `occlusion/core/pipeline.py` and `occlusion/core/depth_scale.py`; this is
the robot-side copy so ubot does not import across projects.

--------------------------------------------------------------------------
THE MODEL CHOICE IS NOT ARBITRARY
--------------------------------------------------------------------------
Default is Metric-INDOOR-Small, not the outdoor head. Measured on the ubot's own
camera at ~0.19 m above the ground, against same-device stereo:

    resolution, d ln(z_dep)/d ln(z_true)     1-4 m      4-8 m
    Metric-Outdoor-Small  (DAv2 default)      0.52       0.03   <- saturated
    Metric-Outdoor-Base                       0.54       0.03
    Metric-Indoor-Small                       0.84       0.33

The outdoor head reads ~16 m for everything past 4 m from this viewpoint: it was
trained at car height. The indoor head tracks stereo to within ~20% over 1-4 m
with no correction at all. Past ~4 m it compresses too, which is why callers cap
the trusted range rather than believing the far readings.

--------------------------------------------------------------------------
CALIBRATION
--------------------------------------------------------------------------
`depth_scale.json` is produced by the Occlusion repo:

    python occlusion/eval/depth_affine.py --run <clip> --depth-model <id>

It carries z_dep = a * z_true + b, fitted against a metric reference, and this
applies the inverse. The file is the contract: no code is shared between the
research pipeline and the robot, so they cannot drift apart silently. If the fit
was refused (an ill-conditioned reference), the file holds `fit: null` and depth
is passed through unchanged -- with a warning, because uncalibrated metres
should never be mistaken for measured ones.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INDOOR_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
OUTDOOR_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


def load_affine(path: str | None) -> tuple[tuple[float, float] | None, str]:
    """((a, b), provenance) from a depth_scale.json, or (None, why not)."""
    if not path:
        return None, "uncalibrated (no depth_scale_json given)"
    p = Path(path).expanduser()
    if not p.exists():
        return None, f"uncalibrated ({p} not found)"
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as e:
        return None, f"uncalibrated ({p.name}: {e})"
    fit = doc.get("fit")
    if not fit:
        return None, f"uncalibrated ({p.name}: {doc.get('provenance', 'no fit')})"
    return (float(fit["a"]), float(fit["b"])), f"{p.name}: {doc.get('provenance', '')}"


class MonoDepth:
    """RGB uint8 (H, W, 3) in, metres float32 (h, w) out.

    The output is usually SMALLER than the input -- the network works at its own
    resolution and upsampling here would be wasted work. Callers must scale the
    intrinsics to the returned shape; `scale_intrinsics` does it.
    """

    def __init__(self, model_id: str = INDOOR_MODEL, size: int | None = None,
                 device: str | None = None, affine: tuple[float, float] | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        if size:
            self.proc.size = {"height": size, "width": size}
        self.size = size
        self.affine = affine
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        t = self.torch
        inp = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with t.no_grad():
            pred = self.model(**inp).predicted_depth
        depth = pred[0].float().cpu().numpy().astype(np.float32)
        if self.affine is not None:
            a, b = self.affine
            depth = ((depth - b) / a).astype(np.float32)
        return depth

    def describe(self) -> str:
        cal = "uncalibrated" if self.affine is None else \
            f"calibrated a={self.affine[0]:.3f} b={self.affine[1]:+.3f}"
        return (f"{self.model_id.split('/')[-1]} size={self.size or 'native'} "
                f"on {self.device}, {cal}")


def scale_intrinsics(fx: float, fy: float, cx: float, cy: float,
                     src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> tuple:
    """Intrinsics moved from the RGB resolution to the depth map's.

    Getting this wrong tilts every obstacle by a fixed factor, and nothing in
    the output looks obviously wrong -- so it is one call, used everywhere.
    """
    sx = dst_wh[0] / float(src_wh[0])
    sy = dst_wh[1] / float(src_wh[1])
    return fx * sx, fy * sy, cx * sx, cy * sy
