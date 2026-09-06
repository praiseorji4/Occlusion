"""FEM, POPAM, GhostCSP — the novel YOLOv8 building blocks.

FEM expands the receptive field with three parallel dilated depthwise-separable
branches (residual). POPAM is a dual avg/max-pool channel-attention gate. Both
are inserted into the backbone/neck by ``models.build`` without re-indexing the
Ultralytics layer graph (target layers are wrapped in a passthrough chain).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, padding=None):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel - 1) // 2
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, padding=padding,
                            dilation=dilation, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class FEM(nn.Module):
    """Three-branch dilated feature enhancement with a mandatory residual."""

    def __init__(self, channels: int):
        super().__init__()
        b = channels // 4
        self.b1 = nn.Sequential(nn.Conv2d(channels, b, 1, bias=False),
                                DSConv(b, b, 3, dilation=1), nn.SiLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(channels, b, 1, bias=False),
                                DSConv(b, b, 3, dilation=2), nn.SiLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(channels, b, 1, bias=False),
                                DSConv(b, b, 3, dilation=3), nn.SiLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(b * 3, channels, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)) + x


class POPAM(nn.Module):
    """Dual avg/max-pool channel attention; preserves peak activations."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(channels * 2, mid, bias=False),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(mid, channels, bias=False),
                                 nn.Sigmoid())

    def forward(self, x):
        B, C, _, _ = x.shape
        cat = torch.cat([self.gap(x).view(B, C), self.gmp(x).view(B, C)], dim=1)
        return x * self.mlp(cat).view(B, C, 1, 1)


class GhostConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, ratio=2):
        super().__init__()
        init_ch = out_ch // ratio
        new_ch = init_ch * (ratio - 1)
        self.primary = nn.Sequential(nn.Conv2d(in_ch, init_ch, kernel, padding=kernel // 2, bias=False),
                                     nn.BatchNorm2d(init_ch), nn.SiLU(inplace=True))
        self.cheap = nn.Sequential(nn.Conv2d(init_ch, new_ch, 3, padding=1, groups=init_ch, bias=False),
                                   nn.BatchNorm2d(new_ch), nn.SiLU(inplace=True))

    def forward(self, x):
        return torch.cat([self.primary(x), self.cheap(self.primary(x))], dim=1)


class GhostCSP(nn.Module):
    def __init__(self, in_ch, out_ch, n=1):
        super().__init__()
        mid = out_ch // 2
        self.cv1 = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False),
                                 nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.cv2 = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False),
                                 nn.BatchNorm2d(mid), nn.SiLU(inplace=True))
        self.blocks = nn.Sequential(*[GhostConv(mid, mid) for _ in range(n)])
        self.cv3 = nn.Sequential(nn.Conv2d(mid * 2, out_ch, 1, bias=False),
                                 nn.BatchNorm2d(out_ch), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.cv3(torch.cat([self.blocks(self.cv1(x)), self.cv2(x)], dim=1))


class _Chained(nn.Module):
    """Run a sequence of modules under one Ultralytics graph slot.

    Holds the original layer's ``f``/``i``/``np`` so the routing in
    ``_predict_once`` keeps working without re-indexing every layer.
    """

    def __init__(self, *mods):
        super().__init__()
        self.mods = nn.ModuleList(mods)
        self.f = -1
        self.i = -1
        self.np = getattr(mods[0], "np", 0)

    def forward(self, x):
        for m in self.mods:
            x = m(x)
        return x


def wrap_layer(model, index: int, *extras: nn.Module) -> None:
    """Replace ``model.model[index]`` with ``[original, *extras]`` chained."""
    seq = model.model
    original = seq[index]
    chained = _Chained(original, *extras)
    chained.f = getattr(original, "f", -1)
    chained.i = getattr(original, "i", index)
    chained.np = getattr(original, "np", 0)
    seq[index] = chained
