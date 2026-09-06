"""Occlusion augmentation transforms for the Ultralytics data pipeline.

Ported from the experiment scripts with the box-format fix preserved: at the
point these transforms run, ``instances.bboxes`` is ``[cx, cy, w, h]`` in
pixels, not ``[x1, y1, x2, y2]``. We explicitly convert to xyxy before reading
and restore the original format so the next transform still gets what it expects.

Strategies operate on numpy HWC images and normalised ``[cx, cy, w, h]`` boxes.
If per-box occlusion levels are supplied and ``label_aware`` is set, heavy
augmentation is skipped on already-occluded boxes (lvl >= 2); otherwise the
strategy applies to all boxes (blind, matching the original experiments).
"""
from __future__ import annotations

import random
from typing import Optional

import cv2
import numpy as np
from ultralytics.data.augment import BaseTransform


def _yolo_to_pixel(bboxes_norm, img_h, img_w):
    if bboxes_norm.shape[0] == 0:
        return bboxes_norm.copy()
    xc = bboxes_norm[:, 0] * img_w
    yc = bboxes_norm[:, 1] * img_h
    bw = bboxes_norm[:, 2] * img_w
    bh = bboxes_norm[:, 3] * img_h
    x1 = np.clip(xc - bw / 2, 0, img_w - 1).astype(int)
    y1 = np.clip(yc - bh / 2, 0, img_h - 1).astype(int)
    x2 = np.clip(xc + bw / 2, 0, img_w - 1).astype(int)
    y2 = np.clip(yc + bh / 2, 0, img_h - 1).astype(int)
    return np.stack([x1, y1, x2, y2], axis=1)


def _sort_by_area_ascending(bboxes_pixel):
    areas = (bboxes_pixel[:, 2] - bboxes_pixel[:, 0]) * (bboxes_pixel[:, 3] - bboxes_pixel[:, 1])
    return np.argsort(areas)


def _visible_ratio(x1, y1, x2, y2, mask):
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)
    return 1.0 - mask[y1:y2, x1:x2].sum() / (h * w)


class _AugBase:
    def __init__(self, probability=0.5, seed=None):
        self.probability = probability
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def _apply_to_bbox(self, img, x1, y1, x2, y2, mask):
        raise NotImplementedError

    def __call__(self, image, bboxes_norm, occlusion_lvls=None, label_aware=False):
        if self._rng.random() > self.probability:
            return image, bboxes_norm
        img = image.copy()
        img_h, img_w = img.shape[:2]
        if bboxes_norm.shape[0] == 0:
            return img, bboxes_norm
        bboxes_pixel = _yolo_to_pixel(bboxes_norm, img_h, img_w)
        order = _sort_by_area_ascending(bboxes_pixel)
        mask = np.zeros((img_h, img_w), dtype=bool)
        for idx in order:
            if label_aware and occlusion_lvls is not None and int(occlusion_lvls[idx]) >= 2:
                continue
            x1, y1, x2, y2 = bboxes_pixel[idx]
            if x2 <= x1 or y2 <= y1:
                continue
            img = self._apply_to_bbox(img, x1, y1, x2, y2, mask)
        return img, bboxes_norm


class Cutout(_AugBase):
    def __init__(self, probability=0.5, num_patches=2, patch_ratio_range=(0.15, 0.35),
                 min_visible_ratio=0.5, seed=None):
        super().__init__(probability, seed)
        self.num_patches = num_patches
        self.patch_ratio_range = patch_ratio_range
        self.min_visible_ratio = min_visible_ratio

    def _apply_to_bbox(self, img, x1, y1, x2, y2, mask):
        bw, bh = x2 - x1, y2 - y1
        for _ in range(self.num_patches):
            if _visible_ratio(x1, y1, x2, y2, mask) < self.min_visible_ratio:
                break
            ratio = self._rng.uniform(*self.patch_ratio_range)
            ar = self._rng.uniform(0.5, 2.0)
            ph = max(1, min(int(np.sqrt(ratio * bw * bh / ar)), bh))
            pw = max(1, min(int(ph * ar), bw))
            ox = self._rng.integers(0, max(1, bw - pw))
            oy = self._rng.integers(0, max(1, bh - ph))
            px1, py1, px2, py2 = x1 + ox, y1 + oy, x1 + ox + pw, y1 + oy + ph
            t = mask.copy()
            t[py1:py2, px1:px2] = True
            if _visible_ratio(x1, y1, x2, y2, t) < self.min_visible_ratio:
                continue
            img[py1:py2, px1:px2] = 0
            mask[py1:py2, px1:px2] = True
        return img


class RandomErasing(_AugBase):
    def __init__(self, probability=0.5, num_patches=2, patch_ratio_range=(0.15, 0.35),
                 min_visible_ratio=0.5, fill_mode="random", seed=None):
        super().__init__(probability, seed)
        self.num_patches = num_patches
        self.patch_ratio_range = patch_ratio_range
        self.min_visible_ratio = min_visible_ratio
        self.fill_mode = fill_mode

    def _apply_to_bbox(self, img, x1, y1, x2, y2, mask):
        bw, bh = x2 - x1, y2 - y1
        mc = img[y1:y2, x1:x2].mean(axis=(0, 1))
        for _ in range(self.num_patches):
            if _visible_ratio(x1, y1, x2, y2, mask) < self.min_visible_ratio:
                break
            ratio = self._rng.uniform(*self.patch_ratio_range)
            ar = self._rng.uniform(0.5, 2.0)
            ph = max(1, min(int(np.sqrt(ratio * bw * bh / ar)), bh))
            pw = max(1, min(int(ph * ar), bw))
            ox = self._rng.integers(0, max(1, bw - pw))
            oy = self._rng.integers(0, max(1, bh - ph))
            px1, py1, px2, py2 = x1 + ox, y1 + oy, x1 + ox + pw, y1 + oy + ph
            t = mask.copy()
            t[py1:py2, px1:px2] = True
            if _visible_ratio(x1, y1, x2, y2, t) < self.min_visible_ratio:
                continue
            if self.fill_mode == "random":
                fill = self._rng.integers(0, 256, (ph, pw, 3), dtype=np.uint8)
            elif self.fill_mode == "mean":
                fill = np.full((ph, pw, 3), mc, dtype=np.uint8)
            else:
                fill = np.zeros((ph, pw, 3), dtype=np.uint8)
            img[py1:py2, px1:px2] = fill
            mask[py1:py2, px1:px2] = True
        return img


class HideAndSeek(_AugBase):
    def __init__(self, probability=0.5, grid_size=4, hide_prob=0.3,
                 min_visible_ratio=0.5, seed=None):
        super().__init__(probability, seed)
        self.grid_size = grid_size
        self.hide_prob = hide_prob
        self.min_visible_ratio = min_visible_ratio

    def _apply_to_bbox(self, img, x1, y1, x2, y2, mask):
        bw, bh = x2 - x1, y2 - y1
        S = self.grid_size
        cw, ch = max(1, bw // S), max(1, bh // S)
        total = S * S
        hidden, max_hidden = 0, int(total * (1 - self.min_visible_ratio))
        cells = list(range(total))
        self._py_rng.shuffle(cells)
        for ci in cells:
            if hidden >= max_hidden:
                break
            if self._rng.random() > self.hide_prob:
                continue
            row, col = divmod(ci, S)
            cx1, cy1 = x1 + col * cw, y1 + row * ch
            cx2, cy2 = min(cx1 + cw, x2), min(cy1 + ch, y2)
            t = mask.copy()
            t[cy1:cy2, cx1:cx2] = True
            if _visible_ratio(x1, y1, x2, y2, t) < self.min_visible_ratio:
                continue
            img[cy1:cy2, cx1:cx2] = 0
            mask[cy1:cy2, cx1:cx2] = True
            hidden += 1
        return img


class GridCut(_AugBase):
    def __init__(self, probability=0.5, d_range=(64, 128), ratio=0.4, rotate=True,
                 min_visible_ratio=0.5, seed=None):
        super().__init__(probability, seed)
        self.d_range = d_range
        self.ratio = ratio
        self.rotate = rotate
        self.min_visible_ratio = min_visible_ratio

    def _grid_mask(self, h, w, d, angle):
        diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2))) + d
        mask = np.ones((diag, diag), dtype=bool)
        sw = max(1, int(d * self.ratio))
        for s in range(0, diag, d):
            mask[s:s + sw, :] = False
            mask[:, s:s + sw] = False
        if angle != 0:
            M = cv2.getRotationMatrix2D((diag // 2, diag // 2), angle, 1.0)
            mask = cv2.warpAffine(mask.astype(np.uint8), M, (diag, diag),
                                  flags=cv2.INTER_NEAREST, borderValue=1).astype(bool)
        top, left = (diag - h) // 2, (diag - w) // 2
        return mask[top:top + h, left:left + w]

    def _apply_to_bbox(self, img, x1, y1, x2, y2, mask):
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            return img
        d = int(self._rng.integers(self.d_range[0], self.d_range[1] + 1))
        angle = float(self._rng.uniform(-45, 45)) if self.rotate else 0.0
        keep = self._grid_mask(bh, bw, d, angle)
        occ = ~keep
        t = mask.copy()
        t[y1:y2, x1:x2] |= occ
        if _visible_ratio(x1, y1, x2, y2, t) < self.min_visible_ratio:
            occ_h = np.zeros((bh, bw), dtype=bool)
            sw = max(1, int(d * self.ratio))
            for s in range(0, bh, d):
                occ_h[s:s + sw, :] = True
            t2 = mask.copy()
            t2[y1:y2, x1:x2] |= occ_h
            if _visible_ratio(x1, y1, x2, y2, t2) >= self.min_visible_ratio:
                occ = occ_h
            else:
                return img
        region = img[y1:y2, x1:x2].copy()
        region[occ] = 0
        img[y1:y2, x1:x2] = region
        mask[y1:y2, x1:x2] |= occ
        return img


_AUG_MAP = {
    "cutout": Cutout,
    "random_erasing": RandomErasing,
    "hide_and_seek": HideAndSeek,
    "gridcut": GridCut,
}

DEFAULT_CONFIGS = {
    "cutout": {"probability": 0.5, "num_patches": 2,
               "patch_ratio_range": [0.15, 0.35], "min_visible_ratio": 0.5},
    "random_erasing": {"probability": 0.5, "num_patches": 2,
                       "patch_ratio_range": [0.15, 0.35], "min_visible_ratio": 0.5},
    "hide_and_seek": {"probability": 0.5, "grid_size": 4, "hide_prob": 0.3, "min_visible_ratio": 0.5},
    "gridcut": {"probability": 0.5, "d_range": [64, 128], "ratio": 0.4, "min_visible_ratio": 0.5},
}


def build_aug_config(strategy: str, p: float) -> dict:
    """Build the aug config dict for a strategy name, scaled by probability p."""
    if strategy == "none":
        return {}
    if strategy in DEFAULT_CONFIGS:
        cfg = dict(DEFAULT_CONFIGS[strategy])
        cfg["probability"] = p
        return {strategy: cfg}
    if strategy == "combined_all":
        return {name: {**c, "probability": p} for name, c in DEFAULT_CONFIGS.items()}
    if strategy == "combined_best_two":
        cfg = {name: {**DEFAULT_CONFIGS[name], "probability": p}
               for name in ("cutout", "gridcut")}
        return cfg
    raise ValueError(f"Unknown aug strategy: {strategy}")


class OcclusionAugPipeline:
    def __init__(self, aug_config: dict, label_aware: bool = False, seed: int = 42):
        self.label_aware = label_aware
        self.augs = []
        for i, (name, params) in enumerate(aug_config.items()):
            if name not in _AUG_MAP:
                raise ValueError(f"Unknown aug: {name}")
            self.augs.append(_AUG_MAP[name](**{**params, "seed": seed + i}))

    def __call__(self, image, bboxes_norm, occlusion_lvls=None):
        for aug in self.augs:
            image, bboxes_norm = aug(image, bboxes_norm, occlusion_lvls, self.label_aware)
        return image, bboxes_norm


class OcclusionAugTransform(BaseTransform):
    """Insert occlusion augmentation into the Ultralytics transform stack.

    Reads bboxes from ``labels['instances']``, converting to xyxy first (the
    box-format fix) then to normalised cx,cy,w,h. Optionally consults per-box
    occlusion levels for label-aware gating.
    """

    def __init__(self, aug_config: dict, label_aware: bool = False, seed: int = 42):
        super().__init__()
        self.pipeline = OcclusionAugPipeline(aug_config, label_aware=label_aware, seed=seed)

    def __call__(self, labels):
        img = labels["img"]
        img_h, img_w = img.shape[:2]
        instances = labels.get("instances")
        occlusion_lvls = labels.get("occlusion_lvls", None)

        if instances is not None and len(instances) > 0:
            original_format = instances._bboxes.format
            instances.convert_bbox(format="xyxy")
            bboxes_xyxy = instances.bboxes
            xc = ((bboxes_xyxy[:, 0] + bboxes_xyxy[:, 2]) / 2) / img_w
            yc = ((bboxes_xyxy[:, 1] + bboxes_xyxy[:, 3]) / 2) / img_h
            bw = (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]) / img_w
            bh = (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]) / img_h
            bboxes_norm = np.stack([xc, yc, bw, bh], axis=1).astype(np.float32)
            instances.convert_bbox(format=original_format)
        else:
            bboxes_norm = np.zeros((0, 4), dtype=np.float32)

        img_aug, _ = self.pipeline(img, bboxes_norm, occlusion_lvls)
        labels["img"] = img_aug
        return labels
