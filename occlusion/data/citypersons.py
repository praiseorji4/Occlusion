"""CityPersons label parsing and YOLO conversion.

CityPersons annotations are JSON with continuous occlusion ratios and a
visible-part bounding box (``bboxVis``). Discrete occlusion levels are derived
for the metric tier filtering; ``ignore`` regions are dropped for training.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

NAMES = {0: "pedestrian"}


def continuous_to_discrete(occl: float) -> int:
    if occl < 0.10:
        return 0
    if occl < 0.35:
        return 1
    return 2


def parse_annotation(json_path: Path, orig_W: int, orig_H: int, imgsz: int = 640) -> dict:
    with open(json_path) as f:
        ann = json.load(f)

    boxes, occ_lvls, occ_ratios, bbox_vis, heights = [], [], [], [], []
    for bbox in ann.get("bboxes", []):
        lbl = bbox.get("lbl", "")
        if lbl != "pedestrian":
            continue
        x1, y1, w, h = bbox.get("bbox", [0, 0, 0, 0])
        x2, y2 = x1 + w, y1 + h
        vx1, vy1, vw, vh = bbox.get("bboxVis", bbox.get("bbox"))
        vx2, vy2 = vx1 + vw, vy1 + vh
        occl = float(bbox.get("occl", 0.0))

        boxes.append([x1 / orig_W, y1 / orig_H, x2 / orig_W, y2 / orig_H])
        bbox_vis.append([vx1 / orig_W, vy1 / orig_H, vx2 / orig_W, vy2 / orig_H])
        occ_lvls.append(continuous_to_discrete(occl))
        occ_ratios.append(occl)
        heights.append((y2 - y1) * imgsz)

    if not boxes:
        return _empty()
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    return {
        "boxes": boxes_t,
        "bbox_vis": torch.tensor(bbox_vis, dtype=torch.float32),
        "occlusion": torch.tensor(occ_lvls, dtype=torch.int64),
        "occlusion_ratio": torch.tensor(occ_ratios, dtype=torch.float32),
        "height_px": torch.tensor(heights, dtype=torch.float32),
        "labels": torch.zeros(len(boxes), dtype=torch.int64),
        "truncation": torch.zeros(len(boxes), dtype=torch.float32),
        "depth_z": torch.zeros(len(boxes), dtype=torch.float32),
    }


def to_yolo_lines(ann: dict) -> list[str]:
    lines = []
    for i in range(len(ann["boxes"])):
        x1, y1, x2, y2 = ann["boxes"][i].tolist()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _empty() -> dict:
    z4 = torch.zeros((0, 4), dtype=torch.float32)
    z = torch.zeros(0)
    return {
        "boxes": z4, "bbox_vis": z4,
        "occlusion": torch.zeros(0, dtype=torch.int64),
        "occlusion_ratio": z, "height_px": z,
        "labels": torch.zeros(0, dtype=torch.int64),
        "truncation": z, "depth_z": z,
    }
