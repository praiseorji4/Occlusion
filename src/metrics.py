"""
src/metrics.py — Evaluation metrics for occlusion-aware pedestrian detection.

Primary metric: AP_hard (KITTI 40-point interpolated AP on Hard tier).
  Hard tier: height ≥ 25px, occlusion ≤ 2, truncation ≤ 0.50, IoU ≥ 0.5.
  Returns percentage (0–100), NOT fraction (0–1).

Also provides:
  - compute_ors():          Occlusion Robustness Score
  - compute_fn_rate_hard(): False-negative rate on occlusion_lvl=2 instances
  - compute_fps():          Inference speed measurement

Unit test (mandatory before training):
  ap = compute_kitti_ap(perfect_preds, annotations, 'hard')
  assert ap > 97.0, f"Perfect predictions must score >97 on 40-pt, got {ap}"
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# KITTI difficulty thresholds
# ---------------------------------------------------------------------------

DIFFICULTY_THRESHOLDS: Dict[str, dict] = {
    "easy": {
        "min_height": 40.0,
        "max_occlusion": 0,
        "max_truncation": 0.15,
    },
    "moderate": {
        "min_height": 25.0,
        "max_occlusion": 1,
        "max_truncation": 0.30,
    },
    "hard": {
        "min_height": 25.0,
        "max_occlusion": 2,
        "max_truncation": 0.50,
    },
}

IOU_THRESHOLD = 0.5   # Pedestrian class


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU of a single box against an array of boxes.

    Args:
        box:   (4,) array [x1, y1, x2, y2] in any unit (must match boxes).
        boxes: (N, 4) array.

    Returns:
        (N,) array of IoU values.
    """
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    box_area   = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = box_area + boxes_area - inter_area
    return np.where(union > 0, inter_area / union, 0.0)


# ---------------------------------------------------------------------------
# KITTI 40-point AP
# ---------------------------------------------------------------------------

def compute_kitti_ap(
    predictions: List[dict],
    annotations: List[dict],
    difficulty: str,
) -> float:
    """
    Compute KITTI 40-point interpolated AP for the Pedestrian class.

    This follows the official KITTI evaluation protocol:
    https://www.cvlibs.net/datasets/kitti/eval_object.php

    Pred format (one dict per image):
      {
        'image_id':  str,
        'boxes':     np.ndarray (N, 4), normalised xyxy [0,1]
        'scores':    np.ndarray (N,)
      }

    Ann format (one dict per image):
      {
        'image_id':    str,
        'boxes':       np.ndarray (M, 4), normalised xyxy [0,1]
        'height_px':   np.ndarray (M,)
        'occlusion':   np.ndarray (M,)  int {0,1,2,3}
        'truncation':  np.ndarray (M,)  float
      }

    Args:
        predictions:  List of per-image prediction dicts.
        annotations:  List of per-image ground-truth dicts.
        difficulty:   'easy' | 'moderate' | 'hard'

    Returns:
        AP as percentage (0–100).  Raises ValueError for unknown difficulty.

    Note:
        40-point interpolation uses recall breakpoints at
        r = 0.0, 0.025, 0.05, …, 0.975, 1.0 (41 points → 40 intervals).
        Each point's precision = max precision at recall ≥ r.
        AP = mean of all 41 precision values × 100.
    """
    if difficulty not in DIFFICULTY_THRESHOLDS:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from {list(DIFFICULTY_THRESHOLDS)}")

    thresh = DIFFICULTY_THRESHOLDS[difficulty]
    min_h  = thresh["min_height"]
    max_occ = thresh["max_occlusion"]
    max_trunc = thresh["max_truncation"]

    # Index annotations by image_id
    ann_by_id: Dict[str, dict] = {a["image_id"]: a for a in annotations}

    # Collect all (score, tp, fp) tuples across the dataset
    all_scores: List[float] = []
    all_tp:     List[int]   = []
    all_fp:     List[int]   = []
    n_gt_total: int         = 0

    for pred in predictions:
        img_id = pred["image_id"]
        ann    = ann_by_id.get(img_id)
        if ann is None:
            continue

        pred_boxes  = np.array(pred.get("boxes", []),  dtype=np.float32)
        pred_scores = np.array(pred.get("scores", []), dtype=np.float32)

        gt_boxes    = np.array(ann.get("boxes",      []), dtype=np.float32)
        gt_heights  = np.array(ann.get("height_px",  []), dtype=np.float32)
        gt_occ      = np.array(ann.get("occlusion",  []), dtype=np.int32)
        gt_trunc    = np.array(ann.get("truncation", []), dtype=np.float32)

        # Filter GT to difficulty tier
        valid_gt = np.ones(len(gt_boxes), dtype=bool)
        if len(gt_heights) == len(gt_boxes):
            valid_gt &= gt_heights >= min_h
        if len(gt_occ) == len(gt_boxes):
            valid_gt &= gt_occ <= max_occ
        if len(gt_trunc) == len(gt_boxes):
            valid_gt &= gt_trunc <= max_trunc

        gt_boxes_valid = gt_boxes[valid_gt] if len(gt_boxes) > 0 else gt_boxes
        n_gt = len(gt_boxes_valid)
        n_gt_total += n_gt

        matched = np.zeros(n_gt, dtype=bool)

        if len(pred_boxes) == 0:
            continue

        # Sort predictions by descending score
        order = np.argsort(-pred_scores)
        pred_boxes  = pred_boxes[order]
        pred_scores = pred_scores[order]

        tp_list = []
        fp_list = []
        score_list = list(pred_scores)

        for pb in pred_boxes:
            if n_gt == 0:
                tp_list.append(0)
                fp_list.append(1)
                continue

            ious = _box_iou(pb, gt_boxes_valid)
            best_iou_idx = int(np.argmax(ious))
            best_iou     = ious[best_iou_idx]

            if best_iou >= IOU_THRESHOLD and not matched[best_iou_idx]:
                matched[best_iou_idx] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

        all_scores.extend(score_list)
        all_tp.extend(tp_list)
        all_fp.extend(fp_list)

    if n_gt_total == 0:
        return 0.0

    # Sort by descending score globally
    order = np.argsort(-np.array(all_scores, dtype=np.float32))
    tp_arr = np.array(all_tp, dtype=np.float32)[order]
    fp_arr = np.array(all_fp, dtype=np.float32)[order]

    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)

    recall    = cum_tp / n_gt_total
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    # 40-point interpolation: 41 recall breakpoints from 0 to 1 inclusive
    recall_thresholds = np.linspace(0.0, 1.0, 41)
    ap = 0.0
    for r_thresh in recall_thresholds:
        # Max precision at recall >= r_thresh
        mask = recall >= r_thresh
        p = precision[mask].max() if mask.any() else 0.0
        ap += p

    ap /= len(recall_thresholds)  # mean over 41 points
    return float(ap * 100.0)      # return as percentage


# ---------------------------------------------------------------------------
# Occlusion Robustness Score
# ---------------------------------------------------------------------------

def compute_ors(
    predictions: List[dict],
    annotations: List[dict],
    difficulty: str = "hard",
) -> float:
    """
    Occlusion Robustness Score (ORS).

    Applies synthetic occlusion patches to validation images at 9 coverage
    levels [0, 10, 20, 30, 40, 50, 60, 70, 80]% and measures AP at each.
    ORS = Σ w_i × AP_i,  where w_i = 1 - (bin_centre / 100).
    Higher ORS = more robust to occlusion.

    In this implementation, the caller is expected to provide pre-computed
    predictions at each occlusion coverage level.  The function aggregates
    them into ORS.

    Args:
        predictions: List of dicts, one per coverage level:
          {
            'coverage': float,   e.g. 0.0, 0.1, ..., 0.8
            'preds':    List[dict]  (standard pred format)
          }
        annotations: Standard annotation list.
        difficulty:  Difficulty tier for AP computation.

    Returns:
        ORS as a float (higher is better).
    """
    # Coverage bins: centre values as fractions
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    weights = [1.0 - b for b in bins]
    weight_sum = sum(weights)

    # Group predictions by coverage
    preds_by_coverage: Dict[float, List[dict]] = {}
    for entry in predictions:
        cov = float(entry["coverage"])
        preds_by_coverage[cov] = entry["preds"]

    ors = 0.0
    for cov, w in zip(bins, weights):
        preds = preds_by_coverage.get(cov, [])
        if not preds:
            continue
        ap = compute_kitti_ap(preds, annotations, difficulty)
        ors += w * ap

    return ors / weight_sum if weight_sum > 0 else 0.0


def apply_synthetic_occlusion(
    image: torch.Tensor,
    coverage: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply a random square occlusion patch to an image tensor.

    Used when computing ORS from scratch (not from pre-computed predictions).

    Args:
        image:    (3, H, W) float32 tensor.
        coverage: Fraction of image area to occlude [0, 1].
        seed:     Random seed for reproducibility.

    Returns:
        Occluded image tensor.
    """
    if coverage <= 0:
        return image.clone()

    rng = np.random.RandomState(seed)
    _, H, W = image.shape
    area = H * W * coverage
    side = int(area ** 0.5)
    side = min(side, H, W)

    x1 = rng.randint(0, W - side + 1)
    y1 = rng.randint(0, H - side + 1)

    result = image.clone()
    result[:, y1:y1 + side, x1:x1 + side] = 0.0
    return result


# ---------------------------------------------------------------------------
# False-negative rate on hard instances
# ---------------------------------------------------------------------------

def compute_fn_rate_hard(
    predictions: List[dict],
    annotations: List[dict],
) -> float:
    """
    False-negative rate on occlusion_lvl=2 (largely occluded) instances.

    Hypothesis H3: visibility head reduces FN rate on largely-occluded
    instances by ≥15%.

    FN rate = (# lvl-2 GT boxes not detected) / (# lvl-2 GT boxes total)

    A GT box is "detected" if any predicted box with score > 0.5 has IoU ≥ 0.5
    with it.

    Args:
        predictions: Standard prediction list.
        annotations: Standard annotation list with 'occlusion' field.

    Returns:
        FN rate as a fraction [0, 1].
    """
    ann_by_id = {a["image_id"]: a for a in annotations}

    n_gt_hard     = 0
    n_missed_hard = 0

    for pred in predictions:
        img_id = pred["image_id"]
        ann    = ann_by_id.get(img_id)
        if ann is None:
            continue

        pred_boxes  = np.array(pred.get("boxes",  []), dtype=np.float32)
        pred_scores = np.array(pred.get("scores", []), dtype=np.float32)

        gt_boxes = np.array(ann.get("boxes",     []), dtype=np.float32)
        gt_occ   = np.array(ann.get("occlusion", []), dtype=np.int32)

        # Only largely-occluded GT boxes (lvl=2)
        hard_mask = gt_occ == 2
        gt_boxes_hard = gt_boxes[hard_mask] if len(gt_boxes) > 0 else np.empty((0, 4))
        n_gt_hard += len(gt_boxes_hard)

        if len(gt_boxes_hard) == 0:
            continue

        # Filter predictions by confidence threshold
        if len(pred_boxes) > 0:
            conf_mask   = pred_scores > 0.5
            pred_boxes  = pred_boxes[conf_mask]

        detected = np.zeros(len(gt_boxes_hard), dtype=bool)
        for pb in pred_boxes:
            ious = _box_iou(pb, gt_boxes_hard)
            for i, iou in enumerate(ious):
                if iou >= IOU_THRESHOLD and not detected[i]:
                    detected[i] = True

        n_missed_hard += int((~detected).sum())

    if n_gt_hard == 0:
        return 0.0

    return n_missed_hard / n_gt_hard


# ---------------------------------------------------------------------------
# FPS measurement
# ---------------------------------------------------------------------------

def compute_fps(
    model: torch.nn.Module,
    input_size: Tuple[int, int] = (640, 640),
    n_runs: int = 200,
    warmup: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Measure model inference speed on a dummy input.

    Args:
        model:      Trained model in eval mode.
        input_size: (H, W) of the input tensor.
        n_runs:     Number of timed runs (warmup excluded).
        warmup:     Number of warmup runs before timing begins.
        device:     Device to run on. Defaults to model's device.

    Returns:
        Dict with keys: mean_fps, std_fps, mean_ms.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    dummy = torch.zeros(1, 3, *input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed runs
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    arr = np.array(latencies, dtype=np.float64)
    mean_ms = float(arr.mean())
    std_ms  = float(arr.std())

    return {
        "mean_fps": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
        "std_fps":  float(1000.0 / mean_ms ** 2 * std_ms) if mean_ms > 0 else 0.0,
        "mean_ms":  mean_ms,
    }


# ---------------------------------------------------------------------------
# Convenience: build annotation dict from dataset sample
# ---------------------------------------------------------------------------

def sample_to_annotation(sample: dict) -> dict:
    """
    Convert a dataset sample dict to the annotation format expected by metrics.

    Args:
        sample: Dict as returned by KITTIDataset or CityPersonsDataset.

    Returns:
        Annotation dict with image_id, boxes (numpy), height_px, occlusion, truncation.
    """
    return {
        "image_id":   sample["image_id"],
        "boxes":      sample["boxes"].numpy()      if len(sample["boxes"]) > 0 else np.empty((0, 4)),
        "height_px":  sample["height_px"].numpy()  if len(sample["height_px"]) > 0 else np.empty(0),
        "occlusion":  sample["occlusion_lvl"].numpy() if len(sample["occlusion_lvl"]) > 0 else np.empty(0, dtype=np.int32),
        "truncation": sample["truncation"].numpy() if len(sample["truncation"]) > 0 else np.empty(0),
    }
