"""Occlusion-aware detection metrics: KITTI 40-point AP, ORS, FN-rate, FPS.

Primary metric is ``AP_hard`` (KITTI 40-point interpolated AP on the Hard tier),
returned as a percentage 0–100. Boxes are normalised xyxy; annotations carry
``height_px``, ``occlusion`` (0–3), ``truncation``.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

DIFFICULTY_THRESHOLDS: Dict[str, dict] = {
    "easy": {"min_height": 40.0, "max_occlusion": 0, "max_truncation": 0.15},
    "moderate": {"min_height": 25.0, "max_occlusion": 1, "max_truncation": 0.30},
    "hard": {"min_height": 25.0, "max_occlusion": 2, "max_truncation": 0.50},
}

IOU_THRESHOLD = 0.5


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return np.where(union > 0, inter / union, 0.0)


def compute_kitti_ap(predictions: List[dict], annotations: List[dict], difficulty: str) -> float:
    """KITTI 40-point interpolated AP (%) for the Pedestrian class."""
    if difficulty not in DIFFICULTY_THRESHOLDS:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from {list(DIFFICULTY_THRESHOLDS)}")
    t = DIFFICULTY_THRESHOLDS[difficulty]
    ann_by_id = {a["image_id"]: a for a in annotations}

    all_scores: List[float] = []
    all_tp: List[int] = []
    all_fp: List[int] = []
    n_gt_total = 0

    for pred in predictions:
        ann = ann_by_id.get(pred["image_id"])
        if ann is None:
            continue
        pred_boxes = np.array(pred.get("boxes", []), dtype=np.float32)
        pred_scores = np.array(pred.get("scores", []), dtype=np.float32)
        gt_boxes = np.array(ann.get("boxes", []), dtype=np.float32)
        gt_heights = np.array(ann.get("height_px", []), dtype=np.float32)
        gt_occ = np.array(ann.get("occlusion", []), dtype=np.int32)
        gt_trunc = np.array(ann.get("truncation", []), dtype=np.float32)

        valid = np.ones(len(gt_boxes), dtype=bool)
        if len(gt_heights) == len(gt_boxes):
            valid &= gt_heights >= t["min_height"]
        if len(gt_occ) == len(gt_boxes):
            valid &= gt_occ <= t["max_occlusion"]
        if len(gt_trunc) == len(gt_boxes):
            valid &= gt_trunc <= t["max_truncation"]
        gt_valid = gt_boxes[valid] if len(gt_boxes) > 0 else gt_boxes
        n_gt = len(gt_valid)
        n_gt_total += n_gt
        matched = np.zeros(n_gt, dtype=bool)

        if len(pred_boxes) == 0:
            continue
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]
        for k, pb in enumerate(pred_boxes):
            if n_gt == 0:
                all_tp.append(0); all_fp.append(1)
            else:
                ious = _box_iou(pb, gt_valid)
                best = int(np.argmax(ious))
                if ious[best] >= IOU_THRESHOLD and not matched[best]:
                    matched[best] = True
                    all_tp.append(1); all_fp.append(0)
                else:
                    all_tp.append(0); all_fp.append(1)
            all_scores.append(float(pred_scores[k]))

    if n_gt_total == 0:
        return 0.0

    order = np.argsort(-np.array(all_scores, dtype=np.float32))
    tp = np.array(all_tp, dtype=np.float32)[order]
    fp = np.array(all_fp, dtype=np.float32)[order]
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / n_gt_total
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    ap = 0.0
    for r in np.linspace(0.0, 1.0, 41):
        mask = recall >= r
        ap += precision[mask].max() if mask.any() else 0.0
    return float(ap / 41 * 100.0)


def compute_ors(predictions: List[dict], annotations: List[dict], difficulty: str = "hard") -> float:
    """Occlusion Robustness Score: weighted AP across occlusion coverage bins."""
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    weights = [1.0 - b for b in bins]
    preds_by_cov = {float(e["coverage"]): e["preds"] for e in predictions}
    ors = 0.0
    for cov, w in zip(bins, weights):
        preds = preds_by_cov.get(cov, [])
        if not preds:
            continue
        ors += w * compute_kitti_ap(preds, annotations, difficulty)
    wsum = sum(weights)
    return ors / wsum if wsum > 0 else 0.0


def apply_synthetic_occlusion(image: torch.Tensor, coverage: float, seed: Optional[int] = None) -> torch.Tensor:
    if coverage <= 0:
        return image.clone()
    rng = np.random.RandomState(seed)
    _, H, W = image.shape
    side = min(int((H * W * coverage) ** 0.5), H, W)
    x1 = rng.randint(0, W - side + 1)
    y1 = rng.randint(0, H - side + 1)
    out = image.clone()
    out[:, y1:y1 + side, x1:x1 + side] = 0.0
    return out


def compute_fn_rate_hard(predictions: List[dict], annotations: List[dict]) -> float:
    """False-negative rate on largely-occluded (lvl=2) instances."""
    ann_by_id = {a["image_id"]: a for a in annotations}
    n_gt = n_missed = 0
    for pred in predictions:
        ann = ann_by_id.get(pred["image_id"])
        if ann is None:
            continue
        pred_boxes = np.array(pred.get("boxes", []), dtype=np.float32)
        pred_scores = np.array(pred.get("scores", []), dtype=np.float32)
        gt_boxes = np.array(ann.get("boxes", []), dtype=np.float32)
        gt_occ = np.array(ann.get("occlusion", []), dtype=np.int32)
        hard = gt_boxes[gt_occ == 2] if len(gt_boxes) > 0 else np.empty((0, 4))
        n_gt += len(hard)
        if len(hard) == 0:
            continue
        if len(pred_boxes) > 0:
            pred_boxes = pred_boxes[pred_scores > 0.5]
        detected = np.zeros(len(hard), dtype=bool)
        for pb in pred_boxes:
            ious = _box_iou(pb, hard)
            for i, iou in enumerate(ious):
                if iou >= IOU_THRESHOLD and not detected[i]:
                    detected[i] = True
        n_missed += int((~detected).sum())
    return n_missed / n_gt if n_gt > 0 else 0.0


def compute_fps(model: torch.nn.Module, input_size: Tuple[int, int] = (640, 640),
                n_runs: int = 200, warmup: int = 50, device: Optional[torch.device] = None) -> dict:
    device = device or next(model.parameters()).device
    model.eval()
    dummy = torch.zeros(1, 3, *input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {"mean_fps": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
            "std_fps": 1000.0 / mean_ms ** 2 * arr.std() if mean_ms > 0 else 0.0,
            "mean_ms": mean_ms}


def sample_to_annotation(sample: dict) -> dict:
    """Convert a KITTI/CP sample dict to the annotation format ``compute_kitti_ap`` expects."""
    return {
        "image_id":   sample["image_id"],
        "boxes":      sample["boxes"].numpy()      if len(sample["boxes"]) > 0 else np.empty((0, 4)),
        "height_px":  sample["height_px"].numpy()  if len(sample["height_px"]) > 0 else np.empty(0),
        "occlusion":  sample["occlusion_lvl"].numpy() if len(sample["occlusion_lvl"]) > 0 else np.empty(0, dtype=np.int32),
        "truncation": sample["truncation"].numpy() if len(sample["truncation"]) > 0 else np.empty(0),
    }
