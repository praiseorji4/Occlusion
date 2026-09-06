"""Ultralytics-native datasets for occlusion augmentation and RGB-D fusion.

One config-switched ``YOLODataset`` subclass handles both proven archive paths:
depth pairing (4-channel images for late / early fusion) and occlusion
augmentation (inserted into the transform stack with the box-format fix).
Per-box occlusion levels are attached from a ``.occ.npy`` sidecar written by
``preprocess.py`` so label-aware gating can skip already-occluded boxes; when the
sidecar is absent or mosaic has reordered boxes, augmentation falls back to
blind (the original experiment behaviour).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics.data.augment import RandomHSV
from ultralytics.data.dataset import YOLODataset

from .transforms import OcclusionAugTransform


def load_depth_channel(im: np.ndarray, im_file: str, depth_root: Optional[Path]) -> np.ndarray:
    """Return a single uint8 depth channel resized to ``im``."""
    h, w = im.shape[:2]
    depth = None
    if depth_root is not None:
        stem = Path(im_file).stem
        for cand in (depth_root / f"{stem}.npy", depth_root / stem[: -4] if stem.endswith("_leftImg8bit") else None):
            if cand and cand.exists():
                arr = np.load(str(cand))
                depth = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                break
    if depth is None:
        depth = np.full((h, w), 128, dtype=np.uint8)
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    return depth


class DepthAwareRandomHSV:
    """Wrap ``RandomHSV`` so the depth channel survives the colour jitter."""

    def __init__(self, hsv: RandomHSV):
        self._hsv = hsv

    def __call__(self, labels):
        img = labels["img"]
        depth = img[..., 3:4]
        labels["img"] = np.ascontiguousarray(img[..., :3])
        labels = self._hsv(labels)
        labels["img"] = np.dstack([labels["img"], depth])
        return labels


class OcclusionYOLODataset(YOLODataset):
    """YOLO dataset with optional depth pairing and occlusion augmentation."""

    def __init__(self, *args, aug_config: Optional[dict] = None, label_aware: bool = False,
                 depth_root: Optional[Path] = None, seed: int = 42, **kwargs):
        self._aug_config = aug_config or {}
        self._label_aware = label_aware
        self._depth_root = Path(depth_root) if depth_root else None
        self._seed = seed
        super().__init__(*args, **kwargs)

    def load_image(self, i, rect_mode=True):
        im, hw_original, hw_resized = super().load_image(i, rect_mode=rect_mode)
        if self._depth_root is not None:
            depth = load_depth_channel(im, self.im_files[i], self._depth_root)
            im = np.dstack([im, depth])
        return im, hw_original, hw_resized

    def _occ_levels_for(self, labels) -> Optional[np.ndarray]:
        if not self._label_aware:
            return None
        im_file = labels.get("im_file")
        if not im_file:
            return None
        label_path = Path(str(im_file))
        sidecar = label_path.with_name(label_path.stem + ".occ.npy")
        if not sidecar.exists():
            return None
        try:
            lvls = np.load(str(sidecar))
        except Exception:
            return None
        instances = labels.get("instances")
        n = len(instances) if instances is not None else 0
        if lvls.shape[0] != n:
            return None
        return lvls

    def build_transforms(self, hyp=None):
        transforms = super().build_transforms(hyp)
        if hasattr(transforms, "transforms"):
            new_list = []
            for t in transforms.transforms:
                if self._depth_root is not None and isinstance(t, RandomHSV):
                    new_list.append(DepthAwareRandomHSV(t))
                else:
                    new_list.append(t)
            if self.augment and self._aug_config:
                occ_t = OcclusionAugTransform(self._aug_config, label_aware=self._label_aware,
                                              seed=self._seed)
                if transforms.transforms:
                    new_list.insert(-1, occ_t)
                else:
                    new_list.append(occ_t)
            transforms.transforms = new_list
        return transforms

    def __getitem__(self, index):
        labels = super().__getitem__(index)
        if self.augment and self._label_aware and isinstance(labels, dict):
            lvls = self._occ_levels_for(labels)
            if lvls is not None:
                labels["occlusion_lvls"] = lvls
        return labels


def build_dataset(owner, img_path, mode, batch, *,
                  aug_config: Optional[dict] = None, label_aware: bool = False,
                  depth_root: Optional[Path] = None, seed: int = 42):
    """Build an ``OcclusionYOLODataset`` matching the ultralytics trainer contract."""
    try:
        from ultralytics.utils.torch_utils import de_parallel
    except ImportError:
        def de_parallel(model):
            return model.module if hasattr(model, "module") else model

    from ultralytics.utils import colorstr

    model = getattr(owner, "model", None)
    gs = max(int(de_parallel(model).stride.max()) if model else 0, 32)
    use_aug = aug_config if mode == "train" else None
    return OcclusionYOLODataset(
        img_path=img_path, imgsz=owner.args.imgsz, batch_size=batch,
        augment=mode == "train", hyp=owner.args,
        rect=owner.args.rect or (mode == "val"),
        cache=owner.args.cache or None, single_cls=owner.args.single_cls or False,
        stride=int(gs), pad=0.0 if mode == "train" else 0.5,
        prefix=colorstr(f"{mode}: "), task=owner.args.task,
        classes=owner.args.classes, data=owner.data,
        fraction=owner.args.fraction if mode == "train" else 1.0,
        aug_config=use_aug, label_aware=label_aware, depth_root=depth_root, seed=seed,
    )
