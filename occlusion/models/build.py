"""Assemble a config-driven DetectionModel: fusion + FEM/POPAM + visibility head.

Called from the trainer's ``get_model`` after Ultralytics builds the base
YOLOv8 weights. FEM/POPAM are wrapped into the layer graph without re-indexing
(see ``modules.wrap_layer``); fusion is applied by re-pointing ``_predict_once``
where depth must be split off from the input.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from .fusion import (
    BACKBONE_END,
    CrossAttentionFusion,
    EarlyFusion,
    build_late_fusion,
)
from .modules import FEM, POPAM, wrap_layer
from .visibility import VisibilityMapHead


class CrossAttentionDetectionModel(DetectionModel):
    """DetectionModel with depth-gated P4 cross-attention fused at the neck."""

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        rgb, depth = x[:, :3], x[:, 3:4]
        conf = torch.ones_like(depth)
        head = self.model[-1]
        body = list(self.model[:-1])
        y = []
        out = rgb
        for m in body:
            if m.f != -1:
                out = y[m.f] if isinstance(m.f, int) else [out if j == -1 else y[j] for j in m.f]
            out = m(out)
            y.append(out if m.i in self.save else None)
        p_idx = head.f if isinstance(head.f, (list, tuple)) else [head.f]
        feats = []
        for j in p_idx:
            abs_idx = len(self.model) + j if j < 0 else j
            feats.append(y[abs_idx])
        p3, p4, p5 = feats
        f3, f4, f5 = self.cross_attn(p3, p4, p5, depth, conf)
        return head([f3, f4, f5])


def _probe_backbone_channels(model: DetectionModel, imgsz: int) -> dict:
    """Run the backbone once and return {layer_index: output_channels}."""
    chans = {}
    y = []
    out = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        for m in model.model[:BACKBONE_END]:
            if m.f != -1:
                out = y[m.f] if isinstance(m.f, int) else [out if j == -1 else y[j] for j in m.f]
            out = m(out)
            chans[m.i] = int(out.shape[1])
            y.append(out if m.i in model.save else None)
    return chans


def _insert_fem(model: DetectionModel, imgsz: int, indices=(4, 6)) -> None:
    chans = _probe_backbone_channels(model, imgsz)
    for idx in indices:
        ch = chans.get(idx)
        if ch is None:
            continue
        wrap_layer(model, idx, FEM(ch))


def _insert_popam(model: DetectionModel, imgsz: int) -> None:
    chans = _probe_backbone_channels(model, imgsz)
    idx = BACKBONE_END - 1
    ch = chans.get(idx)
    if ch is None:
        return
    wrap_layer(model, idx, POPAM(ch))


def build_model(cfg, model: DetectionModel) -> DetectionModel:
    """Mutate ``model`` in place per ``cfg`` and return it."""
    if cfg.use_fem:
        _insert_fem(model, cfg.imgsz)
    if cfg.use_popam:
        _insert_popam(model, cfg.imgsz)

    if cfg.fusion_type == "late_feature":
        build_late_fusion(model, imgsz=cfg.imgsz)
    elif cfg.fusion_type == "early":
        EarlyFusion.apply(model)
    elif cfg.fusion_type == "cross_attention":
        _wire_cross_attention(model, cfg)
    elif cfg.fusion_type in ("none", None):
        pass
    else:
        raise ValueError(f"Unknown fusion_type: {cfg.fusion_type}")

    if cfg.use_visibility_head:
        p3_ch = _detect_p_channels(model, cfg.imgsz)[0]
        model.visibility_head = VisibilityMapHead(in_channels=p3_ch)
        model.lambda_vis = cfg.lambda_vis
        model.lambda_occ_consistency = cfg.lambda_occ_consistency

    return model


def _wire_cross_attention(model: DetectionModel, cfg) -> None:
    p3_ch, p4_ch, p5_ch = _detect_p_channels(model, cfg.imgsz)
    model.cross_attn = CrossAttentionFusion(p3_ch, p4_ch, p5_ch)
    model._input_channels = 4
    model.__class__ = CrossAttentionDetectionModel


def _detect_p_channels(model: DetectionModel, imgsz: int):
    """Probe P3/P4/P5 channel counts by running the model up to the Detect head."""
    head = model.model[-1]
    p_idx = head.f if isinstance(head.f, (list, tuple)) else [head.f]
    body = list(model.model[:-1])
    with torch.no_grad():
        y = []
        out = torch.zeros(1, 3, imgsz, imgsz)
        for m in body:
            if m.f != -1:
                out = y[m.f] if isinstance(m.f, int) else [out if j == -1 else y[j] for j in m.f]
            out = m(out)
            y.append(out if m.i in model.save else None)
    chs = []
    n = len(model.model)
    for j in p_idx:
        abs_idx = n + j if j < 0 else j
        chs.append(int(y[abs_idx].shape[1]) if abs_idx < len(y) and y[abs_idx] is not None else None)
    return tuple(chs)
