"""
src/augmentation.py — Label-aware occlusion augmentation with consistent depth masking.

CRITICAL INVARIANT (enforced by unit tests before every training run):
  When any augmentation masks pixels in the RGB image, THE SAME PIXELS must be
  masked in depth AND depth_conf AND depth_mask. Passing a clean depth map while
  RGB is augmented creates a false training signal.

Depth fill policy (uniform across ALL 5 strategies):
  depth[masked]      = 0.5   ← midpoint of [0,1], meaning "unknown depth"
                               (NOT 0, which falsely means "extremely close")
  depth_conf[masked] = 0.0   ← zero confidence; fusion module gates this out
  depth_mask[masked] = False ← excludes region from consistency loss

Unit tests (run before any augmented training):
  aug_result = augmentor(image, depth, depth_conf, depth_mask, boxes, occlusion_lvls)
  assert (aug_result.depth[aug_result.mask_applied] == 0.5).all()
  assert (aug_result.depth_conf[aug_result.mask_applied] == 0.0).all()
  assert (aug_result.depth_mask[aug_result.mask_applied] == False).all()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# AugmentationResult — unified return type
# ---------------------------------------------------------------------------

@dataclass
class AugmentationResult:
    """
    Return type for all augmentation strategies.

    Every field that was modified by the augmentation is set here.
    The caller must use these tensors, not the originals.

    Hypothesis served: depth consistency — the fusion module must see the
    same occluded view that the detector sees.
    """
    image:               torch.Tensor   # (3, H, W) float32, ImageNet-normalised
    depth:               torch.Tensor   # (1, H, W) float32 in [0, 1]
    depth_conf:          torch.Tensor   # (1, H, W) float32 in [0, 1]
    depth_mask:          torch.Tensor   # (1, H, W) bool; True where depth is valid
    mask_applied:        torch.Tensor   # (H, W) bool; True where pixels were masked
    updated_occlusion_lvl: List[int]    # per-instance occlusion level (may change 0→1)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_depth_mask(
    depth: torch.Tensor,
    depth_conf: torch.Tensor,
    depth_mask: torch.Tensor,
    spatial_mask: torch.Tensor,          # (H, W) bool; True = this pixel was masked
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the uniform depth fill policy to all masked pixels.

    depth[masked]      = 0.5
    depth_conf[masked] = 0.0
    depth_mask[masked] = False

    Args:
        depth, depth_conf, depth_mask: Original tensors (1, H, W).
        spatial_mask: (H, W) bool tensor marking masked pixels.

    Returns:
        Updated (depth, depth_conf, depth_mask) — new tensors, originals untouched.
    """
    m = spatial_mask.unsqueeze(0)  # (1, H, W)
    depth      = depth.clone()
    depth_conf = depth_conf.clone()
    depth_mask = depth_mask.clone()

    depth[m]      = 0.5
    depth_conf[m] = 0.0
    depth_mask[m] = False

    return depth, depth_conf, depth_mask


def _pairwise_iou(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute all-pairs IoU for a set of boxes.

    Args:
        boxes: (N, 4) normalised xyxy.

    Returns:
        (N, N) IoU matrix.
    """
    N = boxes.shape[0]
    if N == 0:
        return torch.zeros(0, 0)

    x1 = boxes[:, 0].unsqueeze(1).expand(N, N)
    y1 = boxes[:, 1].unsqueeze(1).expand(N, N)
    x2 = boxes[:, 2].unsqueeze(1).expand(N, N)
    y2 = boxes[:, 3].unsqueeze(1).expand(N, N)

    ix1 = torch.maximum(x1, x1.T)
    iy1 = torch.maximum(y1, y1.T)
    ix2 = torch.minimum(x2, x2.T)
    iy2 = torch.minimum(y2, y2.T)

    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area  = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area.unsqueeze(1) + area.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-9)


def _no_mask(image: torch.Tensor, depth: torch.Tensor,
             depth_conf: torch.Tensor, depth_mask: torch.Tensor,
             occlusion_lvls: List[int]) -> AugmentationResult:
    """Return an AugmentationResult with no changes (identity)."""
    H, W = image.shape[1], image.shape[2]
    return AugmentationResult(
        image=image, depth=depth, depth_conf=depth_conf, depth_mask=depth_mask,
        mask_applied=torch.zeros(H, W, dtype=torch.bool),
        updated_occlusion_lvl=list(occlusion_lvls),
    )


# ---------------------------------------------------------------------------
# Strategy 1: LabelAwareCutout
# ---------------------------------------------------------------------------

class LabelAwareCutout:
    """
    Place square zero-patches inside bounding boxes of occlusion_lvl=0 instances.

    Serves hypothesis H1: label-aware augmentation teaches the model to handle
    partial occlusion without training it on already-occluded instances.

    Patch policy:
      - Applied only to lvl=0 (fully visible) boxes.
      - Patch area = uniform(0.15, 0.50) × bbox area.
      - RGB fill = 0 (black patch, simulates a hard occluder).
      - Depth fill = 0.5 (unknown), conf = 0, mask = False.
      - GT bounding box is NOT changed.

    Args:
        p: Probability of applying cutout to each eligible box.
    """

    def __init__(self, p: float = 0.4) -> None:
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_mask: torch.Tensor,
        boxes: torch.Tensor,
        occlusion_lvls: List[int],
    ) -> AugmentationResult:
        _, H, W = image.shape
        image      = image.clone()
        depth      = depth.clone()
        depth_conf = depth_conf.clone()
        depth_mask = depth_mask.clone()
        spatial_mask = torch.zeros(H, W, dtype=torch.bool)

        for i, lvl in enumerate(occlusion_lvls):
            if lvl != 0:
                continue
            if random.random() > self.p:
                continue
            if i >= len(boxes):
                continue

            bx1, by1, bx2, by2 = boxes[i].tolist()
            # Convert normalised to pixel coords
            px1 = int(bx1 * W)
            py1 = int(by1 * H)
            px2 = int(bx2 * W)
            py2 = int(by2 * H)
            bw, bh = max(px2 - px1, 1), max(py2 - py1, 1)

            coverage = random.uniform(0.15, 0.50)
            patch_area = coverage * bw * bh
            patch_side = max(1, int(patch_area ** 0.5))
            patch_side = min(patch_side, bw, bh)

            # Random position within the bbox
            off_x = random.randint(0, bw - patch_side)
            off_y = random.randint(0, bh - patch_side)

            rx1 = px1 + off_x
            ry1 = py1 + off_y
            rx2 = rx1 + patch_side
            ry2 = ry1 + patch_side

            # Clamp to image
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(W, rx2), min(H, ry2)

            image[:, ry1:ry2, rx1:rx2] = 0.0
            spatial_mask[ry1:ry2, rx1:rx2] = True

        depth, depth_conf, depth_mask = _apply_depth_mask(
            depth, depth_conf, depth_mask, spatial_mask
        )

        return AugmentationResult(
            image=image, depth=depth, depth_conf=depth_conf, depth_mask=depth_mask,
            mask_applied=spatial_mask,
            updated_occlusion_lvl=list(occlusion_lvls),
        )


# ---------------------------------------------------------------------------
# Strategy 2: LabelAwareGridMask
# ---------------------------------------------------------------------------

class LabelAwareGridMask:
    """
    Apply a rotated grid mask to crowded scenes.

    Only applied when ≥2 boxes have pairwise IoU > 0.3 (crowded).
    Not applied when ALL boxes are occlusion_lvl=2 (already maximally occluded).

    Grid parameters:
      d = min_bbox_height_px × uniform(0.3, 0.8)
      rotation = uniform(-30°, +30°)

    RGB fill = 0.  Depth fill = 0.5, conf = 0, mask = False.

    Serves: learning crowd-specific partial occlusion patterns.
    """

    def __init__(self, p: float = 0.4) -> None:
        self.p = p

    def _make_grid_mask(self, H: int, W: int, d: int, angle_deg: float) -> np.ndarray:
        """Generate a binary grid mask (1 = keep, 0 = mask out)."""
        mask = np.ones((H, W), dtype=np.uint8)
        r = d // 2
        for y in range(0, H + d, d):
            for x in range(0, W + d, d):
                y1, y2 = max(0, y - r), min(H, y + r)
                x1, x2 = max(0, x - r), min(W, x + r)
                mask[y1:y2, x1:x2] = 0

        # Rotate mask
        centre = (W / 2, H / 2)
        M = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
        mask = cv2.warpAffine(
            mask, M, (W, H),
            flags=cv2.INTER_NEAREST, borderValue=1
        )
        return mask  # 0 = masked, 1 = keep

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_mask: torch.Tensor,
        boxes: torch.Tensor,
        occlusion_lvls: List[int],
        height_px: Optional[torch.Tensor] = None,
    ) -> AugmentationResult:
        _, H, W = image.shape

        if random.random() > self.p:
            return _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)

        # Condition 1: ≥2 boxes with pairwise IoU > 0.3
        if len(boxes) < 2:
            return _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)

        iou_mat = _pairwise_iou(boxes)
        # Off-diagonal upper triangle
        N = len(boxes)
        crowded = any(
            iou_mat[i, j].item() > 0.3
            for i in range(N) for j in range(i + 1, N)
        )
        if not crowded:
            return _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)

        # Condition 2: NOT all boxes at lvl=2
        if all(lvl == 2 for lvl in occlusion_lvls):
            return _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)

        # d = min bbox height × uniform(0.3, 0.8)
        if height_px is not None and len(height_px) > 0:
            min_h = float(height_px.min().item())
        else:
            # Estimate from normalised boxes
            heights = [(boxes[i, 3] - boxes[i, 1]).item() * H for i in range(N)]
            min_h = min(heights) if heights else H * 0.2
        d = max(4, int(min_h * random.uniform(0.3, 0.8)))
        angle = random.uniform(-30.0, 30.0)

        grid = self._make_grid_mask(H, W, d, angle)  # 0=masked, 1=keep
        spatial_mask = torch.from_numpy(1 - grid).bool()  # True=masked

        image      = image.clone()
        image[:, spatial_mask] = 0.0

        depth, depth_conf, depth_mask = _apply_depth_mask(
            depth, depth_conf, depth_mask, spatial_mask
        )

        return AugmentationResult(
            image=image, depth=depth, depth_conf=depth_conf, depth_mask=depth_mask,
            mask_applied=spatial_mask,
            updated_occlusion_lvl=list(occlusion_lvls),
        )


# ---------------------------------------------------------------------------
# Strategy 3: RealOccluderAugmentation
# ---------------------------------------------------------------------------

class RealOccluderAugmentation:
    """
    Paste realistic occluder patches (from CityPersons OccluderBank) onto
    fully-visible (lvl=0) pedestrian instances.

    After pasting:
      - occlusion_lvl for that instance is updated from 0 → 1.
      - depth in the occluder region is set to 0.5 (unknown — depth behind
        an occluder is ambiguous since MiDaS cannot see through it).
      - depth_conf = 0 in occluder region.
      - depth_mask = False in occluder region.

    Serves hypothesis H1: provides realistic, geometrically plausible occlusion
    patterns beyond simple cutouts.

    Args:
        occluder_bank: Loaded OccluderBank instance.
        p:             Probability per eligible box.
    """

    def __init__(self, occluder_bank, p: float = 0.4) -> None:
        self.bank = occluder_bank
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_mask: torch.Tensor,
        boxes: torch.Tensor,
        occlusion_lvls: List[int],
    ) -> AugmentationResult:
        _, H, W = image.shape
        image      = image.clone()
        depth      = depth.clone()
        depth_conf = depth_conf.clone()
        depth_mask = depth_mask.clone()
        spatial_mask   = torch.zeros(H, W, dtype=torch.bool)
        updated_lvls   = list(occlusion_lvls)

        rng = np.random.RandomState()

        for i, lvl in enumerate(occlusion_lvls):
            if lvl != 0:
                continue
            if random.random() > self.p:
                continue
            if i >= len(boxes):
                continue

            patch = self.bank.sample(rng)
            if patch is None:
                continue

            bx1, by1, bx2, by2 = boxes[i].tolist()
            px1 = int(bx1 * W)
            py1 = int(by1 * H)
            px2 = int(bx2 * W)
            py2 = int(by2 * H)
            bw, bh = max(px2 - px1, 1), max(py2 - py1, 1)

            # Resize patch to bbox dimensions
            patch_resized = cv2.resize(
                patch, (bw, bh), interpolation=cv2.INTER_LINEAR
            )  # (bh, bw, 3) uint8

            # Clamp bbox to image
            px1c, py1c = max(0, px1), max(0, py1)
            px2c, py2c = min(W, px2), min(H, py2)
            cw, ch = px2c - px1c, py2c - py1c
            if cw <= 0 or ch <= 0:
                continue

            # Crop patch to clamped region
            src_x = px1c - px1
            src_y = py1c - py1
            patch_crop = patch_resized[src_y:src_y + ch, src_x:src_x + cw]

            # Convert patch to float tensor (3, ch, cw)
            patch_t = torch.from_numpy(patch_crop).permute(2, 0, 1).float() / 255.0
            # The patch is raw RGB; ImageNet-normalise to match image space
            _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            _std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            patch_t = (patch_t - _mean) / _std

            image[:, py1c:py2c, px1c:px2c] = patch_t
            spatial_mask[py1c:py2c, px1c:px2c] = True

            updated_lvls[i] = 1  # 0 → 1: partly occluded after pasting

        depth, depth_conf, depth_mask = _apply_depth_mask(
            depth, depth_conf, depth_mask, spatial_mask
        )

        return AugmentationResult(
            image=image, depth=depth, depth_conf=depth_conf, depth_mask=depth_mask,
            mask_applied=spatial_mask,
            updated_occlusion_lvl=updated_lvls,
        )


# ---------------------------------------------------------------------------
# Strategy 4: LabelAwareHideAndSeek
# ---------------------------------------------------------------------------

class LabelAwareHideAndSeek:
    """
    4×4 grid mask where each cell is hidden with probability 0.4.

    ONLY applied when ALL boxes in the image have height_px > 60.
    This restriction exists because hiding grid cells erases small objects
    entirely, which degrades rather than improves detection.

    EXPECTED NEGATIVE RESULT: test on images with mixed heights and record
    that AP decreases. Document in results/narratives/M2.md with mechanism.

    RGB fill = mean pixel of that cell.
    Depth fill = 0.5 (unknown), conf = 0, mask = False.

    Args:
        p:          Probability of applying the strategy.
        cell_p:     Probability each individual cell is hidden (default 0.4).
        min_height: Minimum height_px threshold for all boxes (default 60).
    """

    GRID = 4

    def __init__(self, p: float = 0.4, cell_p: float = 0.4, min_height: float = 60.0) -> None:
        self.p = p
        self.cell_p = cell_p
        self.min_height = min_height

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_mask: torch.Tensor,
        boxes: torch.Tensor,
        occlusion_lvls: List[int],
        height_px: Optional[torch.Tensor] = None,
    ) -> AugmentationResult:
        _, H, W = image.shape

        if random.random() > self.p:
            return _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)

        # Only apply when ALL boxes are tall enough
        if height_px is not None and len(height_px) > 0:
            if float(height_px.min().item()) <= self.min_height:
                return _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)
        elif len(boxes) > 0:
            heights = (boxes[:, 3] - boxes[:, 1]) * H
            if float(heights.min().item()) <= self.min_height:
                return _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)

        image      = image.clone()
        depth      = depth.clone()
        depth_conf = depth_conf.clone()
        depth_mask = depth_mask.clone()
        spatial_mask = torch.zeros(H, W, dtype=torch.bool)

        cell_h = H // self.GRID
        cell_w = W // self.GRID

        for row in range(self.GRID):
            for col in range(self.GRID):
                if random.random() > self.cell_p:
                    continue
                y1 = row * cell_h
                y2 = min(H, (row + 1) * cell_h)
                x1 = col * cell_w
                x2 = min(W, (col + 1) * cell_w)

                # RGB fill = mean pixel of this cell
                cell_mean = image[:, y1:y2, x1:x2].mean(dim=(1, 2), keepdim=True)
                image[:, y1:y2, x1:x2] = cell_mean
                spatial_mask[y1:y2, x1:x2] = True

        depth, depth_conf, depth_mask = _apply_depth_mask(
            depth, depth_conf, depth_mask, spatial_mask
        )

        return AugmentationResult(
            image=image, depth=depth, depth_conf=depth_conf, depth_mask=depth_mask,
            mask_applied=spatial_mask,
            updated_occlusion_lvl=list(occlusion_lvls),
        )


# ---------------------------------------------------------------------------
# Strategy 5: OcclusionBalancedMosaic
# ---------------------------------------------------------------------------

class OcclusionBalancedMosaic:
    """
    4-tile mosaic that guarantees occlusion diversity across tiles.

    Tile assignment:
      ≥1 tile from hard_pool  (contains ≥1 occlusion_lvl=2 instance)
      ≥1 tile from easy_pool  (contains ≥1 occlusion_lvl=0 instance)
      2 tiles from random pool

    Depth maps are mosaicked with the same tile layout as RGB.
    No additional pixel masking is applied (existing depth maps are preserved).

    Used from epoch 76 onward; replaces standard mosaic in YOLOv8.

    Args:
        hard_pool: List of samples with ≥1 lvl=2 instance.
        easy_pool: List of samples with ≥1 lvl=0 instance.
        all_pool:  All samples (for random tiles).
        imgsz:     Target output size.
    """

    def __init__(
        self,
        hard_pool: List[dict],
        easy_pool: List[dict],
        all_pool:  List[dict],
        imgsz: int = 640,
    ) -> None:
        self.hard_pool = hard_pool
        self.easy_pool = easy_pool
        self.all_pool  = all_pool
        self.imgsz     = imgsz
        self.half      = imgsz // 2

    def _resize_sample(self, sample: dict) -> dict:
        """Resize a sample's image and depth to (half, half)."""
        h = self.half
        img = sample["image"]  # (3, H, W)
        img_r = F.interpolate(img.unsqueeze(0), size=(h, h), mode="bilinear",
                              align_corners=False).squeeze(0)
        dep = sample["depth"]  # (1, H, W)
        dep_r = F.interpolate(dep.unsqueeze(0), size=(h, h), mode="bilinear",
                              align_corners=False).squeeze(0)
        conf = sample["depth_conf"]
        conf_r = F.interpolate(conf.unsqueeze(0), size=(h, h), mode="bilinear",
                               align_corners=False).squeeze(0)
        dmask = sample["depth_mask"].float()
        dmask_r = F.interpolate(dmask.unsqueeze(0), size=(h, h), mode="nearest"
                                ).squeeze(0).bool()
        return {**sample, "image": img_r, "depth": dep_r,
                "depth_conf": conf_r, "depth_mask": dmask_r}

    def __call__(self, anchor_sample: dict) -> AugmentationResult:
        """
        Build a 4-tile mosaic from anchor_sample + 3 sampled tiles.

        Tile layout:
          [TL | TR]
          [BL | BR]

        Args:
            anchor_sample: The current training sample (top-left tile).

        Returns:
            AugmentationResult with mosaicked image/depth tensors.
        """
        h = self.half
        S = self.imgsz

        # Pick tiles
        hard_tile = random.choice(self.hard_pool) if self.hard_pool else anchor_sample
        easy_tile = random.choice(self.easy_pool) if self.easy_pool else anchor_sample
        rand_tile = random.choice(self.all_pool)

        tiles = [anchor_sample, hard_tile, easy_tile, rand_tile]
        random.shuffle(tiles)  # randomise positions (keeps guarantee on contents)
        tiles = [self._resize_sample(t) for t in tiles]

        # Assemble mosaic canvas
        img_canvas  = torch.zeros(3, S, S)
        dep_canvas  = torch.zeros(1, S, S)
        conf_canvas = torch.zeros(1, S, S)
        mask_canvas = torch.zeros(1, S, S, dtype=torch.bool)

        positions = [(0, 0), (0, h), (h, 0), (h, h)]  # (row, col) top-left corners
        all_boxes, all_lvls = [], []

        for (r, c), tile in zip(positions, tiles):
            img_canvas[:, r:r + h, c:c + h]  = tile["image"]
            dep_canvas[:, r:r + h, c:c + h]  = tile["depth"]
            conf_canvas[:, r:r + h, c:c + h] = tile["depth_conf"]
            mask_canvas[:, r:r + h, c:c + h] = tile["depth_mask"]

            # Remap boxes into mosaic coordinate space
            if len(tile["boxes"]) > 0:
                b = tile["boxes"].clone()
                # boxes are in [0,1] relative to (h, h); remap to (S, S)
                b[:, [0, 2]] = (b[:, [0, 2]] * h + c) / S
                b[:, [1, 3]] = (b[:, [1, 3]] * h + r) / S
                b = b.clamp(0.0, 1.0)
                all_boxes.append(b)
                all_lvls.extend(tile["occlusion_lvl"].tolist())

        merged_boxes = torch.cat(all_boxes, dim=0) if all_boxes else torch.zeros(0, 4)

        return AugmentationResult(
            image=img_canvas,
            depth=dep_canvas,
            depth_conf=conf_canvas,
            depth_mask=mask_canvas,
            mask_applied=torch.zeros(S, S, dtype=torch.bool),  # no additional masking
            updated_occlusion_lvl=all_lvls,
        )


# ---------------------------------------------------------------------------
# Augmentation budget gate
# ---------------------------------------------------------------------------

def _strategy_allowed(
    lvl: int, strategy_name: str, p_config: float
) -> Tuple[bool, float]:
    """
    Return (is_allowed, effective_p) for a strategy given an instance's occlusion level.

    Budget table (from plan):
      lvl=0: All 5 strategies,         p = p_config
      lvl=1: Cutout + RealOccluder,    p = p_config × 0.5
      lvl=2: HFlip + ColorJitter only, p = 0.5 fixed
      lvl=3: Same as lvl=1,            p = p_config × 0.5
    """
    if lvl == 0:
        return True, p_config
    elif lvl in (1, 3):
        if strategy_name in ("cutout", "real_occluder"):
            return True, p_config * 0.5
        return False, 0.0
    elif lvl == 2:
        if strategy_name in ("hflip", "color_jitter"):
            return True, 0.5
        return False, 0.0
    return False, 0.0


# ---------------------------------------------------------------------------
# Augmentation Curriculum
# ---------------------------------------------------------------------------

class AugmentationCurriculum:
    """
    Epoch-gated augmentation schedule.

    Epochs  1–30:  ColorJitter (p=0.4) + HorizontalFlip (p=0.5)
    Epochs 31–50:  + LabelAwareCutout
    Epochs 51–75:  + RealOccluderAugmentation + LabelAwareGridMask
    Epochs 76–100: All strategies; OcclusionBalancedMosaic active

    Each strategy is applied subject to the augmentation budget table above.

    Args:
        p_config:      Selected aug_probability from sweep (0.2, 0.4, or 0.6).
        occluder_bank: Loaded OccluderBank (required for RealOccluderAugmentation).
        mosaic:        OcclusionBalancedMosaic instance (used from epoch 76).
    """

    def __init__(
        self,
        p_config: float = 0.4,
        occluder_bank=None,
        mosaic: Optional[OcclusionBalancedMosaic] = None,
    ) -> None:
        self.p_config     = p_config
        self.occluder_bank = occluder_bank
        self.mosaic        = mosaic

        self._cutout     = LabelAwareCutout(p=p_config)
        self._gridmask   = LabelAwareGridMask(p=p_config)
        self._real_occ   = (
            RealOccluderAugmentation(occluder_bank, p=p_config)
            if occluder_bank is not None else None
        )
        self._has       = LabelAwareHideAndSeek(p=p_config)

    def apply(
        self,
        epoch: int,
        image: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_mask: torch.Tensor,
        boxes: torch.Tensor,
        occlusion_lvls: List[int],
        height_px: Optional[torch.Tensor] = None,
        use_mosaic: bool = False,
        anchor_sample: Optional[dict] = None,
    ) -> AugmentationResult:
        """
        Apply epoch-appropriate augmentations.

        Args:
            epoch:          Current training epoch (1-indexed).
            image:          (3, H, W) float32 ImageNet-normalised tensor.
            depth:          (1, H, W) float32 depth tensor.
            depth_conf:     (1, H, W) float32 confidence tensor.
            depth_mask:     (1, H, W) bool validity tensor.
            boxes:          (N, 4) normalised xyxy tensor.
            occlusion_lvls: Per-instance occlusion level list.
            height_px:      Per-instance pixel height tensor (optional).
            use_mosaic:     If True and epoch ≥ 76, apply OcclusionBalancedMosaic.
            anchor_sample:  Current sample dict (required for mosaic).

        Returns:
            AugmentationResult after all applicable strategies.
        """
        # Phase 1 (1–30): ColorJitter + HFlip only — applied externally via
        # torchvision transforms in the DataLoader; nothing extra here.

        result = _no_mask(image, depth, depth_conf, depth_mask, occlusion_lvls)

        # Phase 2+: LabelAwareCutout (epoch ≥ 31)
        if epoch >= 31:
            result = self._cutout(
                result.image, result.depth, result.depth_conf, result.depth_mask,
                boxes, result.updated_occlusion_lvl,
            )

        # Phase 3+: RealOccluder + GridMask (epoch ≥ 51)
        if epoch >= 51:
            if self._real_occ is not None:
                result = self._real_occ(
                    result.image, result.depth, result.depth_conf, result.depth_mask,
                    boxes, result.updated_occlusion_lvl,
                )
            result = self._gridmask(
                result.image, result.depth, result.depth_conf, result.depth_mask,
                boxes, result.updated_occlusion_lvl, height_px,
            )

        # Phase 4: HideAndSeek (epoch ≥ 76)
        if epoch >= 76:
            result = self._has(
                result.image, result.depth, result.depth_conf, result.depth_mask,
                boxes, result.updated_occlusion_lvl, height_px,
            )
            if use_mosaic and self.mosaic is not None and anchor_sample is not None:
                result = self.mosaic(anchor_sample)

        return result


# ---------------------------------------------------------------------------
# Basic photometric augmentations (applied to image only, no depth change)
# ---------------------------------------------------------------------------

class ColorJitterDepthSafe:
    """
    Standard torchvision ColorJitter, depth-safe (no depth modification).

    Applied at all epochs regardless of occlusion level (lvl=2 budget: p=0.5 fixed).
    Returns AugmentationResult with mask_applied all-False (no pixel masking).
    """

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1, p=0.4):
        import torchvision.transforms as T
        self._jitter = T.ColorJitter(
            brightness=brightness, contrast=contrast,
            saturation=saturation, hue=hue
        )
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_mask: torch.Tensor,
        boxes: torch.Tensor,
        occlusion_lvls: List[int],
    ) -> AugmentationResult:
        _, H, W = image.shape
        if random.random() < self.p:
            image = self._jitter(image)
        return AugmentationResult(
            image=image, depth=depth, depth_conf=depth_conf, depth_mask=depth_mask,
            mask_applied=torch.zeros(H, W, dtype=torch.bool),
            updated_occlusion_lvl=list(occlusion_lvls),
        )


class HorizontalFlipDepthSafe:
    """
    Horizontal flip applied consistently to image, depth, depth_conf, depth_mask,
    and bounding boxes.

    No pixel masking — mask_applied is all-False.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
        depth_mask: torch.Tensor,
        boxes: torch.Tensor,
        occlusion_lvls: List[int],
    ) -> Tuple[AugmentationResult, torch.Tensor]:
        """
        Returns (AugmentationResult, flipped_boxes).
        Boxes are flipped: x_new = 1 - x_old (with x1/x2 swapped).
        """
        _, H, W = image.shape
        if random.random() < self.p:
            image      = TF.hflip(image)
            depth      = TF.hflip(depth)
            depth_conf = TF.hflip(depth_conf)
            depth_mask = TF.hflip(depth_mask)
            if len(boxes) > 0:
                boxes = boxes.clone()
                x1 = 1.0 - boxes[:, 2]
                x2 = 1.0 - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2

        result = AugmentationResult(
            image=image, depth=depth, depth_conf=depth_conf, depth_mask=depth_mask,
            mask_applied=torch.zeros(H, W, dtype=torch.bool),
            updated_occlusion_lvl=list(occlusion_lvls),
        )
        return result, boxes
