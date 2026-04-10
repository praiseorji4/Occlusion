"""
src/modules.py — Architecture modules: FEM, POPAM, GhostCSP, VisibilityMapHead.

FEM (Feature Enhancement Module):
  Multi-scale receptive field expansion via three parallel dilated depthwise
  separable convolution branches.  NOT attention.  Residual connection mandatory.
  Insertion: after C2f blocks at backbone positions 3 and 5.

POPAM (Dual-Pooling Attention Module):
  Channel attention combining GlobalAveragePooling and GlobalMaxPooling.
  NOT part-based.  Preserves peak activations when only one limb is visible.
  Insertion: top of FPN neck, before first detection scale.

GhostCSP:
  Efficient CSP variant using Ghost convolutions.  Recovers the FLOPs budget
  spent on cross-attention in the FPN neck.

VisibilityMapHead:
  Auxiliary head predicting per-pixel visibility probability V in [0, 1].
  Supervised by V_gt constructed from annotation occlusion fields.
  Loss: BCE(V_pred, V_gt) + λ_occ * L_occ_consistency.
  λ_occ=0 is a documented negative result (V_pred collapses to ≈0.5).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Depthwise-separable convolution building block
# ---------------------------------------------------------------------------

class DSConv(nn.Module):
    """
    Depthwise-separable convolution: depthwise → pointwise.

    Args:
        in_ch:    Input channels.
        out_ch:   Output channels.
        kernel:   Kernel size for the depthwise conv.
        dilation: Dilation rate for the depthwise conv.
        padding:  Padding (auto-computed if None).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        dilation: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = dilation * (kernel - 1) // 2
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel,
            padding=padding, dilation=dilation, groups=in_ch, bias=False
        )
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.pw(self.dw(x)))


# ---------------------------------------------------------------------------
# FEM — Feature Enhancement Module
# ---------------------------------------------------------------------------

class FEM(nn.Module):
    """
    Feature Enhancement Module (from POP-YOLOv8).

    Three parallel dilated depthwise-separable convolution branches capture
    three scales simultaneously:
      Branch 1 (dilation=1): local texture and fine detail
      Branch 2 (dilation=2): medium context, partial-occlusion gaps
      Branch 3 (dilation=3): large receptive field, scene context / depth ambiguity

    Output = SiLU(BN(Conv1×1(Concat(B1, B2, B3)))) + F
                                                      ^--- residual (mandatory)

    The residual is mandatory: when only a tiny fragment of an object is visible,
    dilated branches may produce weak responses.  The residual ensures gradient
    flow is not blocked in these extreme cases.

    Insertion points in YOLOv8s: after C2f blocks at backbone positions 3 and 5.

    Args:
        channels: Number of input (and output) channels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        branch_ch = channels // 4  # each branch reduces to C/4

        # Branch 1: dilation=1 (local texture)
        self.b1 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1, bias=False),
            DSConv(branch_ch, branch_ch, kernel=3, dilation=1),
            nn.SiLU(inplace=True),
        )
        # Branch 2: dilation=2 (medium context)
        self.b2 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1, bias=False),
            DSConv(branch_ch, branch_ch, kernel=3, dilation=2),
            nn.SiLU(inplace=True),
        )
        # Branch 3: dilation=3 (large context)
        self.b3 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, 1, bias=False),
            DSConv(branch_ch, branch_ch, kernel=3, dilation=3),
            nn.SiLU(inplace=True),
        )

        # Fusion: 3 × branch_ch → channels
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        fused = self.fuse(torch.cat([b1, b2, b3], dim=1))
        return fused + x   # residual


# ---------------------------------------------------------------------------
# POPAM — Dual-Pooling Attention Module
# ---------------------------------------------------------------------------

class POPAM(nn.Module):
    """
    Dual-Pooling Attention Module (from POP-YOLOv8).

    Channel attention combining Global Average Pooling and Global Max Pooling.
    NOT part-based, NOT spatial attention.

    GAP captures holistic per-channel statistics — useful when an object is
    mostly visible.
    GMP captures the single strongest activation per channel — critical when
    only one limb or shoulder is visible, because the peak response is preserved
    even when the average is diluted by background.

    Architecture:
      gap = AdaptiveAvgPool2d(1)(F)    → (B, C)
      gmp = AdaptiveMaxPool2d(1)(F)    → (B, C)
      combined = Concat([gap, gmp])    → (B, 2C)
      attention = Sigmoid(Linear2(ReLU(Linear1(combined))))  → (B, C), reduction=16
      Output = F × attention.unsqueeze(-1).unsqueeze(-1)

    Insertion point: top of FPN neck, before the first detection scale.

    Args:
        channels:  Number of input channels.
        reduction: Channel reduction ratio for the MLP (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        gap = self.gap(x).view(B, C)
        gmp = self.gmp(x).view(B, C)
        combined = torch.cat([gap, gmp], dim=1)   # (B, 2C)
        attention = self.mlp(combined)             # (B, C)
        return x * attention.view(B, C, 1, 1)


# ---------------------------------------------------------------------------
# GhostConv + GhostCSP
# ---------------------------------------------------------------------------

class GhostConv(nn.Module):
    """
    Ghost convolution: produces 'phantom' feature maps via cheap linear operations.

    Primary conv produces C/2 feature maps; a cheap depthwise conv generates
    another C/2 maps from the primary output.  Total cost ≈ half of standard conv.

    Args:
        in_ch:  Input channels.
        out_ch: Output channels.
        kernel: Kernel size for primary conv.
        ratio:  Ratio of ghost maps to primary maps (default 2).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 1, ratio: int = 2) -> None:
        super().__init__()
        init_ch = out_ch // ratio
        new_ch  = init_ch * (ratio - 1)

        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.SiLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, 3, padding=1, groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        cheap   = self.cheap(primary)
        return torch.cat([primary, cheap], dim=1)


class GhostCSP(nn.Module):
    """
    Ghost Cross-Stage Partial module.

    Splits feature map into two paths:
      Path 1: GhostConv sequence (feature extraction)
      Path 2: Shortcut (identity)
    Outputs are concatenated and projected.

    Used in the FPN neck to recover efficiency budget spent on cross-attention.

    Args:
        in_ch:  Input channels.
        out_ch: Output channels.
        n:      Number of GhostConv repetitions.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int = 1) -> None:
        super().__init__()
        mid = out_ch // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[GhostConv(mid, mid) for _ in range(n)])
        self.cv3 = nn.Sequential(
            nn.Conv2d(mid * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.blocks(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


# ---------------------------------------------------------------------------
# VisibilityMapHead
# ---------------------------------------------------------------------------

class VisibilityMapHead(nn.Module):
    """
    Auxiliary head predicting per-pixel visibility probability V ∈ [0, 1].

    Supervises with V_gt derived from annotation occlusion fields:

    KITTI (discrete):
      occlusion_lvl=0 → V_gt = 1.00  (fully visible)
      occlusion_lvl=1 → V_gt = 0.50  (partly occluded)
      occlusion_lvl=2 → V_gt = 0.15  (largely occluded)
      occlusion_lvl=3 → V_gt = 0.50  (unknown — conservative)
      background      → V_gt = 1.00
      overlapping boxes → V_gt = min(V_gt values at that pixel)

    CityPersons (continuous — richer supervision):
      V_gt[inside bbox_full] = 1 - occlusion_ratio
      V_gt[inside bbox_vis]  = 1.0  (confirmed visible)

    Total loss:
      L_vis = BCE(V_pred, V_gt)
      L_occ_consistency = mean((depth_conf * (1 - V_pred))²)
        ^ Where depth is confident (real surface), V_pred should be high.
          Penalises predicting "occluded" where depth confirms a surface exists.
          λ_occ MUST NOT be 0 — λ_occ=0 is a documented negative result:
          V_pred collapses to ≈0.5 (minimum BCE without structural constraint).

      L_total = L_det + λ_vis * L_vis + λ_occ * L_occ_consistency

    λ_vis schedule: 0.5 for epochs 1–30, linear ramp to 1.0 over 31–60, constant 1.0 after.
    λ_occ: 0.1 constant.

    Architecture:
      Input: feature map from FPN P3 scale (B, C, H/8, W/8)
      3 × Conv3×3(C → 64) → Conv1×1(64 → 1) → Sigmoid
      Upsampled to (B, 1, H, W) via bilinear interpolation.

    Args:
        in_channels: Channel count of the input feature map (FPN P3 scale).
    """

    def __init__(self, in_channels: int = 256) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        """
        Args:
            feat:        (B, C, H', W') FPN P3 feature map.
            target_size: (H, W) to upsample to. If None, returns at feature map resolution.

        Returns:
            (B, 1, H, W) visibility probability map.
        """
        v = self.head(feat)
        if target_size is not None:
            v = F.interpolate(v, size=target_size, mode="bilinear", align_corners=False)
        return v

    @staticmethod
    def build_vgt_kitti(
        boxes: torch.Tensor,
        occlusion_lvls: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Construct the KITTI V_gt visibility map for one image.

        Args:
            boxes:         (N, 4) normalised xyxy boxes.
            occlusion_lvls: (N,) int64 occlusion level {0,1,2,3}.
            height, width: Output map dimensions.

        Returns:
            (1, height, width) float32 V_gt in [0, 1].
        """
        _OCC_TO_VGT = {0: 1.00, 1: 0.50, 2: 0.15, 3: 0.50}
        v_gt = torch.ones(1, height, width, dtype=torch.float32)

        for i in range(len(boxes)):
            lvl = int(occlusion_lvls[i].item())
            val = _OCC_TO_VGT.get(lvl, 0.50)

            bx1, by1, bx2, by2 = boxes[i].tolist()
            px1 = max(0, int(bx1 * width))
            py1 = max(0, int(by1 * height))
            px2 = min(width,  int(bx2 * width))
            py2 = min(height, int(by2 * height))

            # Apply min (overlapping boxes take the harder value)
            v_gt[0, py1:py2, px1:px2] = torch.minimum(
                v_gt[0, py1:py2, px1:px2],
                torch.tensor(val, dtype=torch.float32),
            )

        return v_gt

    @staticmethod
    def build_vgt_citypersons(
        boxes: torch.Tensor,
        bbox_vis: torch.Tensor,
        occlusion_ratios: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """
        Construct the CityPersons V_gt visibility map for one image.
        Uses continuous occlusion ratios (richer supervision than KITTI discrete).

        Args:
            boxes:            (N, 4) normalised xyxy full bboxes.
            bbox_vis:         (N, 4) normalised xyxy visible bboxes.
            occlusion_ratios: (N,) float32 in [0, 1].
            height, width:    Output map dimensions.

        Returns:
            (1, height, width) float32 V_gt.
        """
        v_gt = torch.ones(1, height, width, dtype=torch.float32)

        for i in range(len(boxes)):
            occ_r = float(occlusion_ratios[i].item())

            # Full bbox region: V_gt = 1 - occlusion_ratio
            bx1, by1, bx2, by2 = boxes[i].tolist()
            px1 = max(0, int(bx1 * width))
            py1 = max(0, int(by1 * height))
            px2 = min(width,  int(bx2 * width))
            py2 = min(height, int(by2 * height))
            v_gt[0, py1:py2, px1:px2] = torch.minimum(
                v_gt[0, py1:py2, px1:px2],
                torch.tensor(1.0 - occ_r, dtype=torch.float32),
            )

            # Visible bbox region: confirmed visible → V_gt = 1.0
            vx1, vy1, vx2, vy2 = bbox_vis[i].tolist()
            vpx1 = max(0, int(vx1 * width))
            vpy1 = max(0, int(vy1 * height))
            vpx2 = min(width,  int(vx2 * width))
            vpy2 = min(height, int(vy2 * height))
            v_gt[0, vpy1:vpy2, vpx1:vpx2] = 1.0

        return v_gt

    @staticmethod
    def compute_loss(
        v_pred: torch.Tensor,
        v_gt: torch.Tensor,
        depth_conf: torch.Tensor,
        lambda_occ: float = 0.1,
    ) -> dict:
        """
        Compute visibility head losses.

        Args:
            v_pred:     (B, 1, H, W) predicted visibility.
            v_gt:       (B, 1, H, W) ground-truth visibility.
            depth_conf: (B, 1, H, W) depth confidence map.
            lambda_occ: Weight for consistency loss (must NOT be 0).

        Returns:
            Dict with keys: loss_vis, loss_occ_consistency, loss_total.
        """
        loss_vis = F.binary_cross_entropy(v_pred, v_gt)

        # Occ consistency: where depth is confident, V_pred should be high.
        # (depth_conf * (1 - V_pred))² penalises predicting "occluded" at
        # a pixel where a physical surface is confirmed by depth.
        loss_occ_consistency = ((depth_conf * (1.0 - v_pred)) ** 2).mean()

        loss_total = loss_vis + lambda_occ * loss_occ_consistency

        return {
            "loss_vis":             loss_vis,
            "loss_occ_consistency": loss_occ_consistency,
            "loss_total":           loss_total,
        }

    @staticmethod
    def get_lambda_vis(epoch: int) -> float:
        """
        λ_vis schedule:
          Epochs 1–30:  0.5
          Epochs 31–60: linear ramp 0.5 → 1.0
          Epochs 61+:   1.0
        """
        if epoch <= 30:
            return 0.5
        elif epoch <= 60:
            return 0.5 + 0.5 * (epoch - 30) / 30.0
        return 1.0
