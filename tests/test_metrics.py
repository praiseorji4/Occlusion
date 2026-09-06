"""Metric sanity: perfect predictions score ~100 AP; FN-rate and IoU helpers."""
from __future__ import annotations

import numpy as np

from occlusion.metrics import _box_iou, compute_fn_rate_hard, compute_kitti_ap


def _annotations(n=10, imgsz=1.0):
    rng = np.random.default_rng(0)
    boxes = []
    for _ in range(n):
        cx, cy = rng.uniform(0.2, 0.8, 2)
        s = rng.uniform(0.08, 0.2)
        boxes.append([cx - s, cy - s, cx + s, cy + s])
    boxes = np.array(boxes, dtype=np.float32)
    return [{
        "image_id": "img0",
        "boxes": boxes,
        "height_px": np.full(n, 50.0, dtype=np.float32),
        "occlusion": np.zeros(n, dtype=np.int32),
        "truncation": np.zeros(n, dtype=np.float32),
    }]


def test_perfect_predictions_score_high_ap():
    anns = _annotations()
    preds = [{"image_id": "img0", "boxes": anns[0]["boxes"], "scores": np.ones(len(anns[0]["boxes"]))}]
    for tier in ("easy", "moderate", "hard"):
        ap = compute_kitti_ap(preds, anns, tier)
        assert ap > 97.0, f"perfect preds AP={ap:.2f} on {tier} (expected >97)"


def test_no_predictions_zero_ap():
    anns = _annotations()
    preds = [{"image_id": "img0", "boxes": np.empty((0, 4), dtype=np.float32), "scores": np.empty(0)}]
    assert compute_kitti_ap(preds, anns, "hard") == 0.0


def test_box_iou_self_is_one():
    box = np.array([0.1, 0.1, 0.4, 0.4], dtype=np.float32)
    iou = _box_iou(box, box[None])
    assert np.isclose(iou[0], 1.0)


def test_fn_rate_zero_for_perfect_predictions():
    anns = _annotations()
    anns[0]["occlusion"] = np.full(len(anns[0]["boxes"]), 2, dtype=np.int32)
    preds = [{"image_id": "img0", "boxes": anns[0]["boxes"],
              "scores": np.ones(len(anns[0]["boxes"]))}]
    assert compute_fn_rate_hard(preds, anns) == 0.0


def test_fn_rate_one_for_no_predictions():
    anns = _annotations()
    anns[0]["occlusion"] = np.full(len(anns[0]["boxes"]), 2, dtype=np.int32)
    preds = [{"image_id": "img0", "boxes": np.empty((0, 4), dtype=np.float32), "scores": np.empty(0)}]
    assert compute_fn_rate_hard(preds, anns) == 1.0
