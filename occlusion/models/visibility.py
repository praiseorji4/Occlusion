"""Auxiliary per-pixel visibility head and its V_gt builders + loss."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_OCC_TO_VGT = {0: 1.0, 1: 0.5, 2: 0.15, 3: 0.5}


class VisibilityMapHead(nn.Module):
    """Predict V ∈ [0, 1] from the P3 feature map; upsample to image size."""

    def __init__(self, in_channels: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 1, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor, target_size: Optional[tuple] = None) -> torch.Tensor:
        v = self.head(feat)
        if target_size is not None:
            v = F.interpolate(v, size=target_size, mode="bilinear", align_corners=False)
        return v

    @staticmethod
    def build_vgt_kitti(boxes, occlusion_lvls, height, width):
        v_gt = torch.ones(1, height, width, dtype=torch.float32)
        for i in range(len(boxes)):
            val = _OCC_TO_VGT.get(int(occlusion_lvls[i].item()), 0.5)
            bx1, by1, bx2, by2 = boxes[i].tolist()
            x1 = max(0, int(bx1 * width)); y1 = max(0, int(by1 * height))
            x2 = min(width, int(bx2 * width)); y2 = min(height, int(by2 * height))
            v_gt[0, y1:y2, x1:x2] = torch.minimum(v_gt[0, y1:y2, x1:x2],
                                                  torch.tensor(val, dtype=torch.float32))
        return v_gt

    @staticmethod
    def build_vgt_citypersons(boxes, bbox_vis, occlusion_ratios, height, width):
        v_gt = torch.ones(1, height, width, dtype=torch.float32)
        for i in range(len(boxes)):
            occ_r = float(occlusion_ratios[i].item())
            bx1, by1, bx2, by2 = boxes[i].tolist()
            x1 = max(0, int(bx1 * width)); y1 = max(0, int(by1 * height))
            x2 = min(width, int(bx2 * width)); y2 = min(height, int(by2 * height))
            v_gt[0, y1:y2, x1:x2] = torch.minimum(v_gt[0, y1:y2, x1:x2],
                                                  torch.tensor(1.0 - occ_r, dtype=torch.float32))
            vx1, vy1, vx2, vy2 = bbox_vis[i].tolist()
            v_gt[0, max(0, int(vy1 * height)):min(height, int(vy2 * height)),
                     max(0, int(vx1 * width)):min(width, int(vx2 * width))] = 1.0
        return v_gt

    @staticmethod
    def compute_loss(v_pred, v_gt, depth_conf=None, lambda_occ: float = 0.1):
        loss_vis = F.binary_cross_entropy(v_pred, v_gt)
        if depth_conf is not None:
            loss_occ = ((depth_conf * (1.0 - v_pred)) ** 2).mean()
        else:
            loss_occ = torch.zeros((), device=v_pred.device)
        return {"loss_vis": loss_vis, "loss_occ_consistency": loss_occ,
                "loss_total": loss_vis + lambda_occ * loss_occ}

    @staticmethod
    def get_lambda_vis(epoch: int) -> float:
        if epoch <= 30:
            return 0.5
        if epoch <= 60:
            return 0.5 + 0.5 * (epoch - 30) / 30.0
        return 1.0
