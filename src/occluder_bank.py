"""
src/occluder_bank.py — CityPersons occluder patch extractor.

Builds a bank of realistic occluder patches for use by RealOccluderAugmentation.
Each patch is the "occluder region" of a partly-occluded pedestrian: the
difference between the full bounding box and the visible bounding box.

Only instances with occl > 0.2 are included (meaningful occlusion).

The bank is saved as a pickle file at:
  data/citypersons/occluder_bank.pkl

Format:
  List of dicts:
    {
      'patch':       np.ndarray (H, W, 3) uint8 RGB
      'occl':        float  raw occlusion ratio
      'source_city': str
      'source_stem': str
    }

Usage:
  python src/occluder_bank.py --data_root /path/to/data
"""

from __future__ import annotations

import json
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Minimum occlusion ratio to include a patch in the bank
MIN_OCCL = 0.2

# Minimum patch dimensions (pixels) — discard tiny patches
MIN_PATCH_DIM = 8


class OccluderBank:
    """
    Repository of realistic occluder patches extracted from CityPersons.

    Patches come from the difference region between bbox_full and bbox_vis:
    the area that IS annotated as a pedestrian but IS NOT visible in the
    scene — i.e. the occluding object is covering this region.

    In practice, we extract the full bounding box crop and mask out the
    visible portion, leaving the occluder region.  This is pasted onto
    occlusion_lvl=0 instances to simulate realistic partial occlusion.

    Args:
        bank_path: Path to the .pkl file.  If it doesn't exist, call build().
    """

    def __init__(self, bank_path: Path) -> None:
        self.bank_path = Path(bank_path)
        self._patches: List[dict] = []
        if self.bank_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.bank_path, "rb") as f:
            self._patches = pickle.load(f)
        logger.info("OccluderBank loaded: %d patches from %s", len(self._patches), self.bank_path)

    def __len__(self) -> int:
        return len(self._patches)

    def sample(self, rng: Optional[np.random.RandomState] = None) -> Optional[np.ndarray]:
        """
        Sample a random occluder patch from the bank.

        Args:
            rng: Optional random state for reproducibility.

        Returns:
            (H, W, 3) uint8 RGB array, or None if bank is empty.
        """
        if not self._patches:
            return None
        if rng is None:
            entry = random.choice(self._patches)
        else:
            entry = self._patches[rng.randint(0, len(self._patches))]
        return entry["patch"].copy()

    def build(
        self,
        cp_root: Path,
        split: str = "train",
        max_patches: int = 10000,
    ) -> None:
        """
        Build the occluder bank from CityPersons annotations.

        Iterates over all 'pedestrian' instances with occl > MIN_OCCL.
        Extracts the occluder region crop (bbox_full minus bbox_vis region).

        Args:
            cp_root:     Path to data/citypersons/
            split:       Which annotation split to use ('train' recommended).
            max_patches: Cap on total patches (random subsample if exceeded).
        """
        ann_root   = cp_root / "gtBbox_cityPersons_trainval" / split
        image_root = cp_root / "leftImg8bit_trainvaltest"    / split

        if not ann_root.exists():
            raise FileNotFoundError(f"CityPersons annotation dir not found: {ann_root}")

        patches: List[dict] = []

        for json_path in sorted(ann_root.rglob("*.json")):
            city = json_path.name.split("_")[0]
            stem = json_path.stem
            img_stem = stem.replace("_gtBbox_cityPersons_annotation", "_leftImg8bit")
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

                b    = bbox.get("bbox",    [0, 0, 0, 0])
                bvis = bbox.get("bboxVis", b)

                x1, y1, bw, bh = [int(v) for v in b]
                vx1, vy1, vw, vh = [int(v) for v in bvis]

                x2, y2   = x1 + bw, y1 + bh
                vx2, vy2 = vx1 + vw, vy1 + vh

                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Extract full bbox crop
                full_crop = rgb[y1:y2, x1:x2].copy()
                crop_h, crop_w = full_crop.shape[:2]
                if crop_h < MIN_PATCH_DIM or crop_w < MIN_PATCH_DIM:
                    continue

                # Mask out the visible portion within the crop
                # relative coords of visible region within full crop:
                rel_vx1 = max(0, vx1 - x1)
                rel_vy1 = max(0, vy1 - y1)
                rel_vx2 = min(crop_w, vx2 - x1)
                rel_vy2 = min(crop_h, vy2 - y1)

                occluder_patch = full_crop.copy()
                # Zero out the confirmed-visible region (we only want the occluder)
                if rel_vx2 > rel_vx1 and rel_vy2 > rel_vy1:
                    occluder_patch[rel_vy1:rel_vy2, rel_vx1:rel_vx2] = 0

                patches.append({
                    "patch":       occluder_patch,
                    "occl":        occl,
                    "source_city": city,
                    "source_stem": stem,
                })

        if len(patches) > max_patches:
            random.shuffle(patches)
            patches = patches[:max_patches]

        self._patches = patches
        self.bank_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bank_path, "wb") as f:
            pickle.dump(patches, f)

        logger.info(
            "OccluderBank built: %d patches → %s", len(self._patches), self.bank_path
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CityPersons occluder bank.")
    parser.add_argument("--data_root",   type=Path,  required=True)
    parser.add_argument("--split",       default="train")
    parser.add_argument("--max_patches", type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cp_root   = args.data_root / "citypersons"
    bank_path = cp_root / "occluder_bank.pkl"

    bank = OccluderBank(bank_path)
    bank.build(cp_root, split=args.split, max_patches=args.max_patches)
    print(f"Done. Bank contains {len(bank)} patches.")
