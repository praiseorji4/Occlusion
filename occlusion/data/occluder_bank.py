"""CityPersons occluder patch bank for realistic occlusion augmentation.

Patches are the occluder region of partly-occluded pedestrians: the full bbox
crop with the visible portion zeroed. Used by ``RealOccluderAugmentation``.
"""
from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MIN_OCCL = 0.2
MIN_PATCH_DIM = 8


class OccluderBank:
    def __init__(self, bank_path: Path) -> None:
        self.bank_path = Path(bank_path)
        self._patches: List[dict] = []
        if self.bank_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.bank_path, "rb") as f:
            self._patches = pickle.load(f)
        logger.info("OccluderBank: %d patches from %s", len(self._patches), self.bank_path)

    def __len__(self) -> int:
        return len(self._patches)

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Optional[np.ndarray]:
        if not self._patches:
            return None
        if rng is None:
            entry = random.choice(self._patches)
        else:
            entry = self._patches[rng.randint(0, len(self._patches))]
        return entry["patch"].copy()

    def build(self, cp_root: Path, split: str = "train", max_patches: int = 10000) -> None:
        ann_root = cp_root / "gtBbox_cityPersons_trainval" / split
        image_root = cp_root / "leftImg8bit_trainvaltest" / split
        if not ann_root.exists():
            raise FileNotFoundError(f"CityPersons annotation dir not found: {ann_root}")

        patches: List[dict] = []
        for json_path in sorted(ann_root.rglob("*.json")):
            city = json_path.name.split("_")[0]
            img_stem = json_path.stem.replace("_gtBbox_cityPersons_annotation", "_leftImg8bit")
            img_path = image_root / city / f"{img_stem}.png"
            if not img_path.exists():
                continue
            rgb = cv2.imread(str(img_path))
            if rgb is None:
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            H, W = rgb.shape[:2]
            with open(json_path) as f:
                ann = json.load(f)

            for bbox in ann.get("bboxes", []):
                if bbox.get("lbl") != "pedestrian":
                    continue
                occl = float(bbox.get("occl", 0.0))
                if occl <= MIN_OCCL:
                    continue
                x1, y1, bw, bh = [int(v) for v in bbox.get("bbox", [0, 0, 0, 0])]
                vx1, vy1, vw, vh = [int(v) for v in bbox.get("bboxVis", bbox.get("bbox"))]
                x2, y2 = min(W, x1 + bw), min(H, y1 + bh)
                x1, y1 = max(0, x1), max(0, y1)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = rgb[y1:y2, x1:x2].copy()
                ch, cw = crop.shape[:2]
                if ch < MIN_PATCH_DIM or cw < MIN_PATCH_DIM:
                    continue
                rvx1, rvy1 = max(0, vx1 - x1), max(0, vy1 - y1)
                rvx2, rvy2 = min(cw, vx1 + vw - x1), min(ch, vy1 + vh - y1)
                if rvx2 > rvx1 and rvy2 > rvy1:
                    crop[rvy1:rvy2, rvx1:rvx2] = 0
                patches.append({"patch": crop, "occl": occl, "source_city": city})

        if len(patches) > max_patches:
            random.shuffle(patches)
            patches = patches[:max_patches]

        self._patches = patches
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bank_path, "wb") as f:
            pickle.dump(patches, f)
        logger.info("OccluderBank built: %d patches -> %s", len(self._patches), self.bank_path)
