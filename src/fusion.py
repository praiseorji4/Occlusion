"""
src/fusion.py — RGB-D fusion strategies for the YOLOv8 FPN neck.

Three strategies tested across experiment variants M4–M8:

EarlyFusion (M4):
  Concatenate depth as a 4th input channel.  Adapt the first backbone
  Conv2d from (B,3,H,W) to (B,4,H,W) by weight surgery.

LateFusion (M5):
  Run a separate depth backbone branch in parallel.  Merge final
  predictions via Weighted Box Fusion (RGB weight=0.7, depth weight=0.3).

CrossAttentionFusion (M6/M7/M8):
  Confidence-map-gated cross-attention at P4 (40×40 = 1600 tokens only).
  P3 (80×80 = 6400 tokens) is a DOCUMENTED NEGATIVE RESULT — test once,
  record memory usage, document in narrative.

  Step 1 — Depth gating at all scales (P3, P4, P5):
    G_i = sigmoid(BN(Conv1×1(depth_features_i)))
    F_i = rgb_features_i * G_i + rgb_features_i   (residual gate)

  Step 2 — Cross-attention at P4 only:
    Q = Linear(rgb_P4_flat)
    K = Linear(depth_P4_flat)
    V = Linear(depth_P4_flat)
    A = softmax(Q @ K.T / sqrt(C))
    F4_attn = rgb_P4 + Proj(attended.reshape(B, C, H4, W4))

  Step 3 — Confidence masking at all scales (spatially varying):
    C_i = interpolate(depth_conf_map, (H_i, W_i))
    F_final_i = F_fused_i * C_i + rgb_features_i * (1 - C_i)
    → Where conf=0 (textureless wall, sky): pure RGB fallback — automatic.
    → Where conf=1 (reliable surface):     fully fused output.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Early Fusion — M4
# ---------------------------------------------------------------------------

class EarlyFusion(nn.Module):
    """
    Early RGB-D fusion: depth concatenated as a 4th input channel.

    Adapts the first Conv2d of the YOLOv8 backbone from (B,3,H,W) to
    (B,4,H,W) via weight surgery:
      - Copy pretrained RGB weights into the first 3 channels.
      - Initialise the 4th channel (depth) to zero so training starts from
        the pretrained RGB baseline.

    Hypothesis served: tests whether raw channel-level fusion before any
    feature extraction is sufficient (expected: worse than cross-attention
    because depth and RGB have very different statistics at pixel level).

    Args:
        original_conv: The first Conv2d from the pretrained YOLOv8 backbone.
    """

    def __init__(self, original_conv: nn.Conv2d) -> None:
        super().__init__()
        out_ch = original_conv.out_channels
        k      = original_conv.kernel_size
        s      = original_conv.stride
        p      = original_conv.padding

        # New conv: 4 input channels
        self.conv = nn.Conv2d(4, out_ch, k, stride=s, padding=p, bias=False)

        # Weight surgery: copy RGB weights, zero depth channel
        with torch.no_grad():
            self.conv.weight[:, :3, ...] = original_conv.weight.data.clone()
            self.conv.weight[:, 3:, ...] = 0.0

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb:   (B, 3, H, W) ImageNet-normalised.
            depth: (B, 1, H, W) in [0, 1].

        Returns:
            (B, out_ch, H', W') fused feature map.
        """
        x = torch.cat([rgb, depth], dim=1)  # (B, 4, H, W)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Late Fusion — M5
# ---------------------------------------------------------------------------

class LateFusionWBF(nn.Module):
    """
    Late fusion via Weighted Box Fusion (WBF) on parallel RGB and depth predictions.

    Each branch produces its own set of detections.  WBF merges them
    using per-branch confidence weights (RGB=0.7, depth=0.3).

    This module is a post-processing step — it does not modify backbone features.
    The depth branch uses the same YOLOv8 architecture but with 1-channel input.

    Hypothesis served: tests whether separate modality-specific detectors can
    be combined effectively vs. feature-level fusion (expected: late fusion
    misses context where depth disambiguates overlapping boxes mid-feature-extraction).

    Args:
        rgb_weight:   Confidence weight for RGB branch predictions.
        depth_weight: Confidence weight for depth branch predictions.
        iou_thr:      IoU threshold for WBF box clustering.
        skip_box_thr: Minimum score to include a box in WBF.
    """

    def __init__(
        self,
        rgb_weight: float = 0.7,
        depth_weight: float = 0.3,
        iou_thr: float = 0.55,
        skip_box_thr: float = 0.05,
    ) -> None:
        super().__init__()
        self.rgb_weight   = rgb_weight
        self.depth_weight = depth_weight
        self.iou_thr      = iou_thr
        self.skip_box_thr = skip_box_thr

    def forward(
        self,
        rgb_preds: List[dict],
        depth_preds: List[dict],
    ) -> List[dict]:
        """
        Merge RGB and depth predictions via WBF.

        Args:
            rgb_preds:   List of per-image dicts with 'boxes' (N,4), 'scores' (N,), 'labels' (N,).
            depth_preds: Same format.

        Returns:
            Merged list of per-image prediction dicts.
        """
        try:
            from ensemble_boxes import weighted_boxes_fusion  # type: ignore
        except ImportError:
            raise ImportError(
                "ensemble-boxes not installed. Run: pip install ensemble-boxes"
            )

        merged = []
        for rgb_p, dep_p in zip(rgb_preds, depth_preds):
            boxes_list  = [rgb_p["boxes"].tolist(),  dep_p["boxes"].tolist()]
            scores_list = [rgb_p["scores"].tolist(), dep_p["scores"].tolist()]
            labels_list = [rgb_p["labels"].tolist(), dep_p["labels"].tolist()]
            weights     = [self.rgb_weight, self.depth_weight]

            m_boxes, m_scores, m_labels = weighted_boxes_fusion(
                boxes_list, scores_list, labels_list,
                weights=weights,
                iou_thr=self.iou_thr,
                skip_box_thr=self.skip_box_thr,
            )
            merged.append({
                "boxes":  torch.tensor(m_boxes,  dtype=torch.float32),
                "scores": torch.tensor(m_scores, dtype=torch.float32),
                "labels": torch.tensor(m_labels, dtype=torch.int64),
            })

        return merged


# ---------------------------------------------------------------------------
# Cross-Attention Fusion — M6/M7/M8
# ---------------------------------------------------------------------------

class DepthGatingLayer(nn.Module):
    """
    Per-scale residual depth gate (applied at P3, P4, P5).

    G = sigmoid(BN(Conv1×1(depth_features)))
    output = rgb_features * G + rgb_features   (residual)

    The residual ensures that if G≈0 (depth completely unreliable),
    the output falls back to the original RGB features.

    Args:
        channels: Feature map channel count at this FPN scale.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(
        self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor
    ) -> torch.Tensor:
        G = self.gate(depth_feat)
        return rgb_feat * G + rgb_feat   # residual gating


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention between RGB and depth feature maps.

    Applied at P4 ONLY (40×40 = 1600 tokens at imgsz=640).
    P3 (80×80 = 6400 tokens) would produce a 6400×6400 attention matrix
    → OOM at batch=16.  DOCUMENTED NEGATIVE RESULT: test once, record
    memory, document in results/narratives.

    Query  = Linear(rgb_P4_flat)
    Key/V  = Linear(depth_P4_flat)
    Output = rgb_P4 + Proj(attended.reshape(B, C, H4, W4))

    Args:
        channels:  Feature channel count at P4.
        num_heads: Number of attention heads.
    """

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def forward(
        self, rgb_feat: torch.Tensor, depth_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_feat:   (B, C, H, W) RGB feature map at P4.
            depth_feat: (B, C, H, W) Depth feature map at P4.

        Returns:
            (B, C, H, W) attended feature map.
        """
        B, C, H, W = rgb_feat.shape
        N = H * W  # sequence length = 1600 at P4 for imgsz=640

        # Flatten spatial dims: (B, C, H, W) → (B, N, C)
        rgb_flat   = rgb_feat.flatten(2).transpose(1, 2)    # (B, N, C)
        depth_flat = depth_feat.flatten(2).transpose(1, 2)  # (B, N, C)

        Q = self.q_proj(rgb_flat)    # (B, N, C)
        K = self.k_proj(depth_flat)  # (B, N, C)
        V = self.v_proj(depth_flat)  # (B, N, C)

        # Reshape for multi-head
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, h, N, d)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale   # (B, h, N, N)
        attn = attn.softmax(dim=-1)
        attended = (attn @ V)                            # (B, h, N, d)
        attended = attended.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, C)
        attended = self.out_proj(attended)

        # Reshape back to spatial: (B, N, C) → (B, C, H, W)
        attended = attended.transpose(1, 2).view(B, C, H, W)

        # Residual: add to RGB features
        return rgb_feat + attended


class ConfidenceMaskingLayer(nn.Module):
    """
    Spatially-varying confidence masking at a given FPN scale.

    C_i = interpolate(depth_conf_map, (H_i, W_i))
    F_final = F_fused * C_i + rgb_features * (1 - C_i)

    Where C_i = 0 (textureless region, sky, water — MiDaS unreliable):
      output = pure RGB features  (automatic fallback)
    Where C_i = 1 (surface with strong texture — MiDaS reliable):
      output = fully fused features

    Unlike a scalar α, this is spatially adaptive — no hand-tuned threshold.
    """

    def forward(
        self,
        fused_feat: torch.Tensor,
        rgb_feat: torch.Tensor,
        depth_conf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_feat:  (B, C, H_i, W_i) fused features.
            rgb_feat:    (B, C, H_i, W_i) original RGB features.
            depth_conf:  (B, 1, H_orig, W_orig) confidence map (full resolution).

        Returns:
            (B, C, H_i, W_i) confidence-masked output.
        """
        _, _, Hi, Wi = fused_feat.shape
        C_i = F.interpolate(
            depth_conf, size=(Hi, Wi), mode="bilinear", align_corners=False
        )  # (B, 1, Hi, Wi)
        return fused_feat * C_i + rgb_feat * (1.0 - C_i)


class CrossAttentionFusion(nn.Module):
    """
    Full confidence-map-gated cross-attention fusion for M6/M7/M8.

    Three-step pipeline applied at the YOLOv8 FPN neck:

    Step 1 — Depth gating at all scales (P3, P4, P5):
      G_i = sigmoid(BN(Conv1×1(depth_features_i)))
      F_i = rgb_features_i * G_i + rgb_features_i

    Step 2 — Cross-attention at P4 ONLY (40×40):
      F4_attn = rgb_P4 + Proj(MultiHeadAttention(Q=rgb_P4, K=V=depth_P4))

    Step 3 — Confidence masking at all scales:
      F_final_i = F_gated_i * C_i + rgb_features_i * (1 - C_i)
      (C_i is the depth confidence map, bilinearly interpolated to scale i)

    Args:
        channels_p3: Channel count at P3 scale (default 128 for YOLOv8s).
        channels_p4: Channel count at P4 scale (default 256 for YOLOv8s).
        channels_p5: Channel count at P5 scale (default 512 for YOLOv8s).
        num_heads:   Attention heads for cross-attention.

    Note on P3 OOM (documented negative result):
        At imgsz=640, P3 is 80×80 = 6400 tokens.
        The attention matrix is 6400×6400 = 40.96M elements per head.
        At 8 heads, batch=16: ~5.2GB for attention alone → OOM.
        This is tested ONCE, memory usage is recorded, and documented
        in results/narratives/M6.md as a negative result.
    """

    def __init__(
        self,
        channels_p3: int = 128,
        channels_p4: int = 256,
        channels_p5: int = 512,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        # Depth encoder: project 1-channel depth maps to match FPN channel counts
        self.depth_enc_p3 = nn.Sequential(
            nn.Conv2d(1, channels_p3, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_p3),
            nn.SiLU(inplace=True),
        )
        self.depth_enc_p4 = nn.Sequential(
            nn.Conv2d(1, channels_p4, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_p4),
            nn.SiLU(inplace=True),
        )
        self.depth_enc_p5 = nn.Sequential(
            nn.Conv2d(1, channels_p5, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels_p5),
            nn.SiLU(inplace=True),
        )

        # Step 1: depth gating at all scales
        self.gate_p3 = DepthGatingLayer(channels_p3)
        self.gate_p4 = DepthGatingLayer(channels_p4)
        self.gate_p5 = DepthGatingLayer(channels_p5)

        # Step 2: cross-attention at P4 only
        self.xattn_p4 = MultiHeadCrossAttention(channels_p4, num_heads=num_heads)

        # Step 3: confidence masking
        self.conf_mask = ConfidenceMaskingLayer()

    def _encode_depth_at_scale(
        self,
        depth: torch.Tensor,
        encoder: nn.Module,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """Resize depth to FPN scale and encode to matching channel count."""
        depth_resized = F.interpolate(
            depth, size=size, mode="bilinear", align_corners=False
        )
        return encoder(depth_resized)

    def forward(
        self,
        rgb_p3: torch.Tensor,
        rgb_p4: torch.Tensor,
        rgb_p5: torch.Tensor,
        depth: torch.Tensor,
        depth_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply full cross-attention fusion at the FPN neck.

        Args:
            rgb_p3:     (B, C3, H3, W3) RGB features at P3 scale.
            rgb_p4:     (B, C4, H4, W4) RGB features at P4 scale.
            rgb_p5:     (B, C5, H5, W5) RGB features at P5 scale.
            depth:      (B, 1, H, W)    Precomputed depth map (full resolution).
            depth_conf: (B, 1, H, W)    Depth confidence map (full resolution).

        Returns:
            Tuple of fused (p3, p4, p5) feature tensors.
        """
        # --- Step 1: Depth gating at all scales ---
        d_p3 = self._encode_depth_at_scale(depth, self.depth_enc_p3, rgb_p3.shape[2:])
        d_p4 = self._encode_depth_at_scale(depth, self.depth_enc_p4, rgb_p4.shape[2:])
        d_p5 = self._encode_depth_at_scale(depth, self.depth_enc_p5, rgb_p5.shape[2:])

        gated_p3 = self.gate_p3(rgb_p3, d_p3)
        gated_p4 = self.gate_p4(rgb_p4, d_p4)
        gated_p5 = self.gate_p5(rgb_p5, d_p5)

        # --- Step 2: Cross-attention at P4 ONLY ---
        # gated_p4 is already: rgb_p4 * G + rgb_p4
        # Apply cross-attention on top: gated_p4 + attended(Q=rgb_P4, KV=depth_P4)
        fused_p4 = self.xattn_p4(gated_p4, d_p4)

        # --- Step 3: Confidence masking at all scales ---
        out_p3 = self.conf_mask(gated_p3, rgb_p3, depth_conf)
        out_p4 = self.conf_mask(fused_p4,  rgb_p4, depth_conf)
        out_p5 = self.conf_mask(gated_p5, rgb_p5, depth_conf)

        return out_p3, out_p4, out_p5


# ---------------------------------------------------------------------------
# Fusion factory
# ---------------------------------------------------------------------------

def build_fusion(
    fusion_type: str,
    **kwargs,
) -> nn.Module:
    """
    Instantiate a fusion module by name.

    Args:
        fusion_type: 'early' | 'late' | 'cross_attention'
        **kwargs:    Forwarded to the fusion module constructor.

    Returns:
        Initialised fusion module.
    """
    if fusion_type == "early":
        return EarlyFusion(**kwargs)
    elif fusion_type == "late":
        return LateFusionWBF(**kwargs)
    elif fusion_type == "cross_attention":
        return CrossAttentionFusion(**kwargs)
    else:
        raise ValueError(
            f"Unknown fusion_type '{fusion_type}'. "
            "Choose from: 'early', 'late', 'cross_attention'."
        )
