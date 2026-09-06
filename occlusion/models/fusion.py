"""RGB-D fusion strategies.

``late_feature`` is the proven archive path: a parallel 1-channel depth backbone
fused into the RGB backbone at P3/P4/P5 via 1x1 convs. ``early`` adapts the
first conv to 4 input channels. ``cross_attention`` is the novel path —
depth-gated multi-head cross-attention at P4 with confidence masking.
"""
from __future__ import annotations

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel

BACKBONE_END = 10
FUSION_INDICES = [4, 6, 9]


def run_stream(layers, x, save, y=None):
    y = {} if y is None else y
    for m in layers:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        if m.i in save:
            y[m.i] = x
    return x, y


def reshape_first_conv_to_1ch(model) -> None:
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    out_ch, in_ch, kh, kw = first_conv.weight.shape
    new_conv = nn.Conv2d(1, out_ch, kernel_size=first_conv.kernel_size,
                         stride=first_conv.stride, padding=first_conv.padding,
                         bias=first_conv.bias is not None)
    new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data.clone()
    for parent in model.modules():
        for name, child in parent.named_children():
            if child is first_conv:
                setattr(parent, name, new_conv)
                return


def patch_autobackend_warmup() -> None:
    from ultralytics.nn.autobackend import AutoBackend
    if getattr(AutoBackend.warmup, "_patched", False):
        return
    original = AutoBackend.warmup

    def warmup(self, *args, **kwargs):
        imgsz = kwargs.get("imgsz", args[0] if args else (1, 3, 640, 640))
        ch = getattr(self.model, "_input_channels", imgsz[1])
        kwargs["imgsz"] = (imgsz[0], ch) + tuple(imgsz[2:])
        return original(self, **kwargs)

    warmup._patched = True
    AutoBackend.warmup = warmup


class LateFusionModel(DetectionModel):
    """DetectionModel with parallel RGB + depth backbones fused at P3/P4/P5."""

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        rgb, depth = x[:, :3], x[:, 3:4]
        _, y_rgb = run_stream(self.rgb_backbone, rgb, self.save)
        _, y_depth = run_stream(self.depth_backbone, depth, self.save)
        y = {idx: fuse(torch.cat([y_rgb[idx], y_depth[idx]], dim=1))
             for idx, fuse in zip(FUSION_INDICES, self.fusion_modules)}
        x_cur, _ = run_stream(self.neck_head, y[FUSION_INDICES[-1]], self.save, y=y)
        return x_cur


def build_late_fusion(model: DetectionModel, imgsz: int = 640) -> None:
    """Split ``model`` into fused RGB/depth backbones + shared neck/head."""
    rgb_backbone = model.model[:BACKBONE_END]
    depth_backbone = copy.deepcopy(rgb_backbone)
    reshape_first_conv_to_1ch(depth_backbone)
    neck_head = model.model[BACKBONE_END:]
    save = model.save

    with torch.no_grad():
        _, y = run_stream(rgb_backbone, torch.zeros(1, 3, imgsz, imgsz), save)
    fusion_modules = nn.ModuleList(
        nn.Conv2d(y[idx].shape[1] * 2, y[idx].shape[1], kernel_size=1) for idx in FUSION_INDICES)

    model.rgb_backbone = rgb_backbone
    model.depth_backbone = depth_backbone
    model.neck_head = neck_head
    model.fusion_modules = fusion_modules
    model._input_channels = 4
    model.__class__ = LateFusionModel


class EarlyFusion:
    """Weight-surgery the first backbone conv from 3 to 4 input channels."""

    @staticmethod
    def apply(model: DetectionModel) -> None:
        first_conv = None
        for parent in model.modules():
            for name, child in parent.named_children():
                if isinstance(child, nn.Conv2d) and first_conv is None:
                    first_conv = child
                    out_ch, in_ch = child.weight.shape[0], child.weight.shape[1]
                    new_conv = nn.Conv2d(4, out_ch, kernel_size=child.kernel_size,
                                         stride=child.stride, padding=child.padding,
                                         bias=child.bias is not None)
                    with torch.no_grad():
                        new_conv.weight[:, :3] = child.weight.data.clone()
                        new_conv.weight[:, 3:] = 0.0
                        if child.bias is not None:
                            new_conv.bias.data = child.bias.data.clone()
                    setattr(parent, name, new_conv)
        model._input_channels = 4


class DepthGatingLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.Sigmoid())

    def forward(self, rgb_feat, depth_feat):
        g = self.gate(depth_feat)
        return rgb_feat * g + rgb_feat


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

    def forward(self, rgb_feat, depth_feat):
        B, C, H, W = rgb_feat.shape
        N = H * W
        rgb = rgb_feat.flatten(2).transpose(1, 2)
        dep = depth_feat.flatten(2).transpose(1, 2)
        q = self.q_proj(rgb).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(dep).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(dep).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        attended = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        attended = self.out_proj(attended).transpose(1, 2).view(B, C, H, W)
        return rgb_feat + attended


class ConfidenceMaskingLayer(nn.Module):
    def forward(self, fused_feat, rgb_feat, depth_conf):
        _, _, Hi, Wi = fused_feat.shape
        c = F.interpolate(depth_conf, size=(Hi, Wi), mode="bilinear", align_corners=False)
        return fused_feat * c + rgb_feat * (1.0 - c)


class CrossAttentionFusion(nn.Module):
    """Depth-gated P4 cross-attention with spatially-varying confidence masking."""

    def __init__(self, channels_p3=128, channels_p4=256, channels_p5=512, num_heads=8):
        super().__init__()
        self.enc_p3 = self._enc(1, channels_p3)
        self.enc_p4 = self._enc(1, channels_p4)
        self.enc_p5 = self._enc(1, channels_p5)
        self.gate_p3 = DepthGatingLayer(channels_p3)
        self.gate_p4 = DepthGatingLayer(channels_p4)
        self.gate_p5 = DepthGatingLayer(channels_p5)
        self.xattn_p4 = MultiHeadCrossAttention(channels_p4, num_heads)
        self.conf_mask = ConfidenceMaskingLayer()

    @staticmethod
    def _enc(in_ch, out_ch):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                             nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))

    def _depth_at(self, depth, encoder, size):
        return encoder(F.interpolate(depth, size=size, mode="bilinear", align_corners=False))

    def forward(self, rgb_p3, rgb_p4, rgb_p5, depth, depth_conf):
        d3 = self._depth_at(depth, self.enc_p3, rgb_p3.shape[2:])
        d4 = self._depth_at(depth, self.enc_p4, rgb_p4.shape[2:])
        d5 = self._depth_at(depth, self.enc_p5, rgb_p5.shape[2:])
        g3 = self.gate_p3(rgb_p3, d3)
        g4 = self.gate_p4(rgb_p4, d4)
        g5 = self.gate_p5(rgb_p5, d5)
        f4 = self.xattn_p4(g4, d4)
        out_p3 = self.conf_mask(g3, rgb_p3, depth_conf)
        out_p4 = self.conf_mask(f4, rgb_p4, depth_conf)
        out_p5 = self.conf_mask(g5, rgb_p5, depth_conf)
        return out_p3, out_p4, out_p5
