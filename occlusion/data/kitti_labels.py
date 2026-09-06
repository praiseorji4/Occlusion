"""KITTI label parsing and YOLO conversion.

KITTI labels store occlusion / truncation / 3D fields that YOLO format cannot
represent. We keep a rich parser for evaluation (the 40-point AP metric needs
occlusion level, truncation, and pixel height) and a converter that writes
YOLO ``class cx cy w h`` files for Ultralytics training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch

KITTI_TRAIN_END = 5985
KITTI_VAL_END = 6732

CLASS_TO_IDX = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
NAMES = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def split_for_id(img_id: int) -> str | None:
    if img_id < KITTI_TRAIN_END:
        return "train"
    if img_id < KITTI_VAL_END:
        return "val"
    return "test"


def parse_label(label_path: Path, orig_W: int, orig_H: int) -> dict:
    """Parse a KITTI label file into rich per-instance arrays (normalised xyxy)."""
    boxes, labels, occ, truncs, heights, depth_zs = [], [], [], [], [], []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] not in CLASS_TO_IDX:
                    continue
                t = float(parts[1])
                o = int(parts[2])
                x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                z = float(parts[13])
                boxes.append([x1 / orig_W, y1 / orig_H, x2 / orig_W, y2 / orig_H])
                labels.append(CLASS_TO_IDX[parts[0]])
                occ.append(o)
                truncs.append(t)
                heights.append(y2 - y1)
                depth_zs.append(z)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "occlusion": torch.tensor(occ, dtype=torch.int64),
        "truncation": torch.tensor(truncs, dtype=torch.float32),
        "height_px": torch.tensor(heights, dtype=torch.float32),
        "depth_z": torch.tensor(depth_zs, dtype=torch.float32),
    }


def to_yolo_lines(ann: dict, orig_W: int, orig_H: int) -> list[str]:
    """Convert a rich KITTI annotation into YOLO ``class cx cy w h`` lines."""
    lines = []
    boxes = ann["boxes"]
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        lines.append(f"{int(ann['labels'][i])} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def empty_annotation(image_id: str) -> dict:
    z = torch.zeros(0)
    return {
        "image_id": image_id,
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.int64),
        "occlusion": torch.zeros(0, dtype=torch.int64),
        "truncation": torch.zeros(0, dtype=torch.float32),
        "height_px": torch.zeros(0, dtype=torch.float32),
        "depth_z": torch.zeros(0, dtype=torch.float32),
    }


def annotation_for(label_path: Path, image_id: str, orig_W: int, orig_H: int) -> dict:
    ann = parse_label(label_path, orig_W, orig_H)
    ann["image_id"] = image_id
    return ann
