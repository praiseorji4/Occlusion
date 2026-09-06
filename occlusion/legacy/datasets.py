"""
src/datasets.py — KITTI and CityPersons dataset classes.

Both return a unified sample dict schema so downstream code is dataset-agnostic.
Depth maps are loaded from precomputed .npy files (src/depth.py creates them).
If depth files are missing the loader returns zero tensors and logs a warning
ONCE per missing directory (not once per sample, to avoid log spam).

Unit tests (run before any training):
  assert dataset[0]['image'].shape == (3, 640, 640)
  assert dataset[0]['depth'].shape == (1, 640, 640)
  assert dataset[0]['occlusion_lvl'].max() <= 3
  assert not torch.isnan(dataset[0]['image']).any()
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ImageNet normalisation constants
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# KITTI split boundaries (image_id integers)
KITTI_TRAIN_END = 5985
KITTI_VAL_END   = 6732

# CityPersons city → split mapping
_CP_CITY_SPLIT: Dict[str, str] = {}
for _c in ("aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf",
           "erfurt", "hamburg", "hanover", "jena", "krefeld", "monchengladbach",
           "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich"):
    _CP_CITY_SPLIT[_c] = "train"
for _c in ("frankfurt", "lindau", "munster"):
    _CP_CITY_SPLIT[_c] = "val"
for _c in ("berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"):
    _CP_CITY_SPLIT[_c] = "test"


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _load_image_as_tensor(path: Path, imgsz: int) -> torch.Tensor:
    """
    Load an image, resize to (imgsz, imgsz), apply ImageNet normalisation.

    Returns:
        Float32 tensor of shape (3, imgsz, imgsz).
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    return t


def _load_depth(npy_path: Path, imgsz: int) -> torch.Tensor:
    """
    Load a precomputed MiDaS depth map from .npy and resize to (1, imgsz, imgsz).

    Returns zeros with a logged warning if the file is missing.
    """
    if not npy_path.exists():
        return torch.zeros(1, imgsz, imgsz, dtype=torch.float32)
    depth = np.load(str(npy_path)).astype(np.float32)
    depth_t = torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
    if depth_t.shape[1] != imgsz or depth_t.shape[2] != imgsz:
        depth_t = F.interpolate(
            depth_t.unsqueeze(0), size=(imgsz, imgsz), mode="bilinear", align_corners=False
        ).squeeze(0)
    return depth_t


def _load_conf(npy_path: Path, imgsz: int) -> torch.Tensor:
    """Load a precomputed depth confidence map. Returns zeros if missing."""
    return _load_depth(npy_path, imgsz)  # same shape logic


def _xyxy_normalise(x1: float, y1: float, x2: float, y2: float,
                    W: int, H: int) -> Tuple[float, float, float, float]:
    return x1 / W, y1 / H, x2 / W, y2 / H


# ---------------------------------------------------------------------------
# KITTI Dataset
# ---------------------------------------------------------------------------

class KITTIDataset(Dataset):
    """
    KITTI Pedestrian detection dataset.

    Serves hypotheses H1–H3 by providing per-annotation occlusion levels that
    the augmentation and metrics modules consume.

    Depth maps (precomputed by MiDaSDepthEstimator) live at:
      {kitti_root}/depth_hybrid/training/image_2/{image_id}.npy
      {kitti_root}/depth_conf/training/image_2/{image_id}.npy

    If they don't exist yet, depth/depth_conf are returned as zero tensors and
    a warning is logged ONCE per missing directory.

    Args:
        kitti_root: Path to data/kitti/
        split:      'train' | 'val' | 'test'
        imgsz:      Target image size (square).
        data_limit: If set, only use the first N samples (smoke-test mode).
    """

    # Track which missing depth dirs we've warned about (class-level)
    _warned_depth_dirs: set = set()

    def __init__(
        self,
        kitti_root: Path,
        split: str,
        imgsz: int = 640,
        data_limit: Optional[int] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.kitti_root = Path(kitti_root)
        self.split = split
        self.imgsz = imgsz

        self._image_dir = self.kitti_root / "data_object_image_2" / "training" / "image_2"
        self._label_dir = self.kitti_root / "data_object_label_2" / "training" / "label_2"
        self._depth_dir = self.kitti_root / "depth_hybrid" / "training" / "image_2"
        self._conf_dir  = self.kitti_root / "depth_conf"   / "training" / "image_2"

        self._warn_if_depth_missing()

        self._image_ids = self._collect_ids()
        if data_limit is not None:
            self._image_ids = self._image_ids[:data_limit]

        logger.info("KITTIDataset [%s]: %d images", split, len(self._image_ids))

    def _warn_if_depth_missing(self) -> None:
        for d in (self._depth_dir, self._conf_dir):
            key = str(d)
            if not d.exists() and key not in KITTIDataset._warned_depth_dirs:
                warnings.warn(
                    f"Depth directory not found: {d}. "
                    "Returning zero depth tensors. Run src/depth.py to precompute.",
                    UserWarning,
                    stacklevel=3,
                )
                KITTIDataset._warned_depth_dirs.add(key)

    def _collect_ids(self) -> List[str]:
        if not self._image_dir.exists():
            logger.warning("KITTI image dir not found: %s", self._image_dir)
            return []
        ids = []
        for p in sorted(self._image_dir.glob("*.png")):
            img_id = int(p.stem)
            if self.split == "train" and img_id < KITTI_TRAIN_END:
                ids.append(p.stem)
            elif self.split == "val" and KITTI_TRAIN_END <= img_id < KITTI_VAL_END:
                ids.append(p.stem)
            elif self.split == "test" and img_id >= KITTI_VAL_END:
                ids.append(p.stem)
        return ids

    def _parse_label(self, label_path: Path, orig_W: int, orig_H: int) -> dict:
        """
        Parse a KITTI label file and return per-annotation arrays.

        KITTI format (space-separated):
          class trunc occ alpha x1 y1 x2 y2 h w l X Y Z ry
        occlusion: 0=fully visible, 1=partly, 2=largely, 3=unknown.

        Only 'Pedestrian' class annotations are returned.
        """
        boxes, labels, occ_lvls, truncs, heights, depth_zs = [], [], [], [], [], []

        if not label_path.exists():
            return self._empty_annotations()

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] != "Pedestrian":
                    continue

                trunc    = float(parts[1])
                occ      = int(parts[2])
                x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                h_3d     = float(parts[8])   # 3D object height in metres (not pixel height)
                depth_z  = float(parts[13])  # camera-coord Z in metres

                # Pixel height of the 2D bounding box
                height_px = y2 - y1

                nx1, ny1, nx2, ny2 = _xyxy_normalise(x1, y1, x2, y2, orig_W, orig_H)

                boxes.append([nx1, ny1, nx2, ny2])
                labels.append(0)        # single class: Pedestrian = 0
                occ_lvls.append(occ)
                truncs.append(trunc)
                heights.append(height_px)
                depth_zs.append(depth_z)

        if not boxes:
            return self._empty_annotations()

        return {
            "boxes":         torch.tensor(boxes,    dtype=torch.float32),
            "labels":        torch.tensor(labels,   dtype=torch.int64),
            "occlusion_lvl": torch.tensor(occ_lvls, dtype=torch.int64),
            "truncation":    torch.tensor(truncs,   dtype=torch.float32),
            "height_px":     torch.tensor(heights,  dtype=torch.float32),
            "depth_z":       torch.tensor(depth_zs, dtype=torch.float32),
        }

    @staticmethod
    def _empty_annotations() -> dict:
        return {
            "boxes":         torch.zeros((0, 4), dtype=torch.float32),
            "labels":        torch.zeros(0,      dtype=torch.int64),
            "occlusion_lvl": torch.zeros(0,      dtype=torch.int64),
            "truncation":    torch.zeros(0,      dtype=torch.float32),
            "height_px":     torch.zeros(0,      dtype=torch.float32),
            "depth_z":       torch.zeros(0,      dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self._image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self._image_ids[idx]

        # --- Image ---
        img_path = self._image_dir / f"{img_id}.png"
        raw = cv2.imread(str(img_path))
        if raw is None:
            raise FileNotFoundError(f"KITTI image not found: {img_path}")
        orig_H, orig_W = raw.shape[:2]

        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(raw_rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD

        # --- Depth + confidence ---
        depth      = _load_depth(self._depth_dir / f"{img_id}.npy", self.imgsz)
        depth_conf = _load_conf(self._conf_dir   / f"{img_id}_conf.npy", self.imgsz)
        depth_mask = (depth_conf > 0.1)  # (1, H, W) bool

        # --- Annotations ---
        ann = self._parse_label(
            self._label_dir / f"{img_id}.txt", orig_W, orig_H
        )

        return {
            "image":         image,
            "depth":         depth,
            "depth_conf":    depth_conf,
            "depth_mask":    depth_mask,
            "boxes":         ann["boxes"],
            "labels":        ann["labels"],
            "occlusion_lvl": ann["occlusion_lvl"],
            "truncation":    ann["truncation"],
            "height_px":     ann["height_px"],
            "depth_z":       ann["depth_z"],
            "image_id":      img_id,
            "split":         self.split,
        }


# ---------------------------------------------------------------------------
# CityPersons Dataset
# ---------------------------------------------------------------------------

class CityPersonsDataset(Dataset):
    """
    CityPersons pedestrian detection dataset.

    Extends the unified schema with:
      - occlusion_ratio: continuous [0,1] occlusion ratio
      - bbox_vis:        normalised xyxy of the visible part of each box
      - ignore_mask:     (H, W) bool; True where loss should be zeroed

    Occlusion level mapping (discrete from continuous):
      occl < 0.10  → level 0
      occl < 0.35  → level 1
      occl ≥ 0.35  → level 2

    Ignore regions (lbl='ignore') are rasterised into ignore_mask and must be
    excluded from loss computation in the training loop.

    Depth maps live at:
      {cp_root}/depth_hybrid/train/{city}/{image_stem}.npy
      {cp_root}/depth_conf/train/{city}/{image_stem}.npy

    Args:
        cp_root:    Path to data/citypersons/
        split:      'train' | 'val' | 'test'
        imgsz:      Target image size (square).
        data_limit: If set, only use the first N samples.
    """

    _warned_depth_dirs: set = set()

    def __init__(
        self,
        cp_root: Path,
        split: str,
        imgsz: int = 640,
        data_limit: Optional[int] = None,
    ) -> None:
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.cp_root = Path(cp_root)
        self.split = split
        self.imgsz = imgsz

        self._ann_root   = self.cp_root / "gtBbox_cityPersons_trainval"
        self._image_root = self.cp_root / "leftImg8bit_trainvaltest"
        self._depth_root = self.cp_root / "depth_hybrid"
        self._conf_root  = self.cp_root / "depth_conf"

        self._samples = self._collect_samples()
        if data_limit is not None:
            self._samples = self._samples[:data_limit]

        logger.info("CityPersonsDataset [%s]: %d images", split, len(self._samples))

    def _collect_samples(self) -> List[dict]:
        """
        Scan annotation directories and build a list of
        {json_path, image_path, depth_path, conf_path, city, stem} dicts.
        """
        samples = []

        # CityPersons annotations are stored under train/ and val/ subdirs
        for ann_subdir in ("train", "val"):
            p = self._ann_root / ann_subdir
            if not p.exists():
                continue
            for json_path in sorted(p.rglob("*.json")):
                city = json_path.name.split("_")[0]
                if _CP_CITY_SPLIT.get(city) != self.split:
                    continue

                stem = json_path.stem  # e.g. 'frankfurt_000000_000294_gtBbox_cityPersons_annotation'
                # Derive the image stem: replace annotation suffix with leftImg8bit
                img_stem = stem.replace(
                    "_gtBbox_cityPersons_annotation", "_leftImg8bit"
                )

                image_path = (
                    self._image_root / ann_subdir / city / f"{img_stem}.png"
                )
                depth_path = (
                    self._depth_root / ann_subdir / city / f"{img_stem}.npy"
                )
                conf_path = (
                    self._conf_root / ann_subdir / city / f"{img_stem}.npy"
                )

                if not image_path.exists():
                    # Try alternative directory structure
                    image_path = self._find_image(city, img_stem)

                samples.append({
                    "json_path":  json_path,
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "conf_path":  conf_path,
                    "city":       city,
                    "stem":       stem,
                })

        return samples

    def _find_image(self, city: str, img_stem: str) -> Path:
        """Try multiple possible image path layouts."""
        candidates = [
            self._image_root / "train" / city / f"{img_stem}.png",
            self._image_root / "val"   / city / f"{img_stem}.png",
            self._image_root / city / f"{img_stem}.png",
        ]
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]  # return first even if missing (will raise on load)

    @staticmethod
    def _continuous_to_discrete_occ(occl: float) -> int:
        if occl < 0.10:
            return 0
        elif occl < 0.35:
            return 1
        return 2

    def _parse_annotations(
        self,
        json_path: Path,
        orig_W: int,
        orig_H: int,
    ) -> dict:
        """
        Parse a CityPersons JSON annotation file.

        Returns per-instance tensors and a rasterised ignore_mask.
        """
        with open(json_path) as f:
            ann = json.load(f)

        boxes, labels, occ_lvls, occ_ratios, bbox_vis_list, truncs = [], [], [], [], [], []
        ignore_regions: List[List[float]] = []

        for bbox in ann.get("bboxes", []):
            lbl  = bbox.get("lbl", "")
            b    = bbox.get("bbox", [0, 0, 0, 0])   # [x, y, w, h]
            bvis = bbox.get("bboxVis", b)            # [x, y, w, h]
            occl = float(bbox.get("occl", 0.0))

            x1, y1, w, h = b
            x2, y2 = x1 + w, y1 + h

            if lbl == "ignore":
                ignore_regions.append([x1, y1, x2, y2])
                continue

            if lbl != "pedestrian":
                continue

            vx1, vy1, vw, vh = bvis
            vx2, vy2 = vx1 + vw, vy1 + vh

            nx1, ny1, nx2, ny2 = _xyxy_normalise(x1, y1, x2, y2, orig_W, orig_H)
            vnx1, vny1, vnx2, vny2 = _xyxy_normalise(vx1, vy1, vx2, vy2, orig_W, orig_H)

            boxes.append([nx1, ny1, nx2, ny2])
            bbox_vis_list.append([vnx1, vny1, vnx2, vny2])
            labels.append(0)
            occ_lvls.append(self._continuous_to_discrete_occ(occl))
            occ_ratios.append(occl)
            truncs.append(0.0)  # CityPersons does not have a truncation field

        # --- Rasterise ignore mask ---
        ignore_mask = torch.zeros(self.imgsz, self.imgsz, dtype=torch.bool)
        scale_x = self.imgsz / orig_W
        scale_y = self.imgsz / orig_H
        for ix1, iy1, ix2, iy2 in ignore_regions:
            rx1 = max(0, int(ix1 * scale_x))
            ry1 = max(0, int(iy1 * scale_y))
            rx2 = min(self.imgsz, int(ix2 * scale_x))
            ry2 = min(self.imgsz, int(iy2 * scale_y))
            ignore_mask[ry1:ry2, rx1:rx2] = True

        if not boxes:
            return {
                "boxes":          torch.zeros((0, 4), dtype=torch.float32),
                "labels":         torch.zeros(0,      dtype=torch.int64),
                "occlusion_lvl":  torch.zeros(0,      dtype=torch.int64),
                "occlusion_ratio": torch.zeros(0,     dtype=torch.float32),
                "bbox_vis":       torch.zeros((0, 4), dtype=torch.float32),
                "truncation":     torch.zeros(0,      dtype=torch.float32),
                "height_px":      torch.zeros(0,      dtype=torch.float32),
                "depth_z":        torch.zeros(0,      dtype=torch.float32),
                "ignore_mask":    ignore_mask,
            }

        boxes_t     = torch.tensor(boxes,        dtype=torch.float32)
        bbox_vis_t  = torch.tensor(bbox_vis_list, dtype=torch.float32)
        # Pixel height from normalised bbox
        height_px   = (boxes_t[:, 3] - boxes_t[:, 1]) * self.imgsz

        return {
            "boxes":           boxes_t,
            "labels":          torch.tensor(labels,     dtype=torch.int64),
            "occlusion_lvl":   torch.tensor(occ_lvls,   dtype=torch.int64),
            "occlusion_ratio": torch.tensor(occ_ratios, dtype=torch.float32),
            "bbox_vis":        bbox_vis_t,
            "truncation":      torch.tensor(truncs,     dtype=torch.float32),
            "height_px":       height_px,
            "depth_z":         torch.zeros(len(boxes),  dtype=torch.float32),  # CP has no LiDAR
            "ignore_mask":     ignore_mask,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]

        # --- Image ---
        raw = cv2.imread(str(s["image_path"]))
        if raw is None:
            raise FileNotFoundError(f"CityPersons image not found: {s['image_path']}")
        orig_H, orig_W = raw.shape[:2]
        raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(raw_rgb, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD

        # --- Depth + confidence ---
        depth      = _load_depth(s["depth_path"], self.imgsz)
        depth_conf = _load_conf(s["conf_path"],   self.imgsz)
        depth_mask = (depth_conf > 0.1)

        # --- Annotations ---
        ann = self._parse_annotations(s["json_path"], orig_W, orig_H)

        return {
            "image":           image,
            "depth":           depth,
            "depth_conf":      depth_conf,
            "depth_mask":      depth_mask,
            "boxes":           ann["boxes"],
            "labels":          ann["labels"],
            "occlusion_lvl":   ann["occlusion_lvl"],
            "occlusion_ratio": ann["occlusion_ratio"],
            "bbox_vis":        ann["bbox_vis"],
            "truncation":      ann["truncation"],
            "height_px":       ann["height_px"],
            "depth_z":         ann["depth_z"],
            "ignore_mask":     ann["ignore_mask"],
            "image_id":        s["stem"],
            "split":           self.split,
        }


# ---------------------------------------------------------------------------
# Collate function (handles variable-length annotation lists)
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    """
    Custom collate for variable-length annotation lists.
    Images, depth, depth_conf, depth_mask, ignore_mask are stacked normally.
    Box/label tensors are kept as a list (one tensor per sample).
    """
    keys_to_stack = ["image", "depth", "depth_conf", "depth_mask"]
    keys_as_list  = ["boxes", "labels", "occlusion_lvl", "truncation",
                     "height_px", "depth_z"]

    out: dict = {}

    for k in keys_to_stack:
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])

    for k in keys_as_list:
        if k in batch[0]:
            out[k] = [b[k] for b in batch]

    # Optional CP-specific fields
    for k in ("occlusion_ratio", "bbox_vis", "ignore_mask"):
        if k in batch[0]:
            if k == "ignore_mask":
                out[k] = torch.stack([b[k] for b in batch])
            else:
                out[k] = [b[k] for b in batch]

    out["image_id"] = [b["image_id"] for b in batch]
    out["split"]    = [b["split"]    for b in batch]

    return out
