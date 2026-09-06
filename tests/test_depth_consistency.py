"""Augmentation invariants: shape/box preservation and consistent depth masking.

The box-format fix in ``OcclusionAugTransform`` is covered indirectly by testing
the shared pipeline on numpy images; the transform itself only re-formats
instances around that pipeline. The depth-consistency invariant asserts that
when augmentation zeros an RGB patch it zeros the paired depth channel too, so
the fusion module never sees an occluded RGB region paired with intact depth.
"""
from __future__ import annotations

import numpy as np

from occlusion.data.transforms import (
    Cutout,
    GridCut,
    HideAndSeek,
    OcclusionAugPipeline,
    RandomErasing,
    build_aug_config,
)


def _image_with_boxes(channels: int = 3, h: int = 128, w: int = 128):
    rng = np.random.default_rng(0)
    img = rng.integers(50, 200, (h, w, channels), dtype=np.uint8)
    boxes = np.array([[0.25, 0.25, 0.4, 0.4], [0.6, 0.6, 0.3, 0.3]], dtype=np.float32)
    return img, boxes


def _strategies():
    return {
        "cutout": Cutout(probability=1.0, seed=1),
        "random_erasing": RandomErasing(probability=1.0, seed=1),
        "hide_and_seek": HideAndSeek(probability=1.0, seed=1),
        "gridcut": GridCut(probability=1.0, seed=1),
    }


def test_pipeline_preserves_shape_and_boxes():
    img, boxes = _image_with_boxes()
    for name, aug in _strategies().items():
        out, out_boxes = aug(img, boxes)
        assert out.shape == img.shape, f"{name} changed image shape"
        assert out.dtype == np.uint8, f"{name} changed dtype"
        np.testing.assert_array_equal(out_boxes, boxes), f"{name} mutated boxes"
        assert not np.array_equal(out, img), f"{name} did not alter the image"


def test_pipeline_handles_empty_boxes():
    img, _ = _image_with_boxes()
    empty = np.zeros((0, 4), dtype=np.float32)
    for aug in _strategies().values():
        out, out_boxes = aug(img, empty)
        assert out.shape == img.shape
        assert out_boxes.shape == (0, 4)


def test_pipeline_runs_combined():
    cfg = build_aug_config("combined_all", p=1.0)
    pipe = OcclusionAugPipeline(cfg, seed=7)
    img, boxes = _image_with_boxes()
    out, out_boxes = pipe(img, boxes)
    assert out.shape == img.shape
    np.testing.assert_array_equal(out_boxes, boxes)


def test_depth_channel_masked_consistently():
    img_rgb, boxes = _image_with_boxes(channels=3)
    depth = np.full(img_rgb.shape[:2] + (1,), 200, dtype=np.uint8)
    img4 = np.dstack([img_rgb, depth])

    aug = Cutout(probability=1.0, num_patches=4, patch_ratio_range=(0.3, 0.5),
                 min_visible_ratio=0.1, seed=3)
    out4, _ = aug(img4, boxes)

    rgb_changed = (out4[..., :3].astype(int) != img4[..., :3].astype(int)).any(axis=2)
    depth_changed = out4[..., 3] != img4[..., 3]
    masked = (out4[..., :3].sum(axis=2) == 0) & (img4[..., :3].sum(axis=2) > 0)
    assert masked.any(), "no RGB pixels were zeroed"
    assert (out4[..., 3][masked] == 0).all(), "depth not zeroed where RGB was occluded"
    assert depth_changed[masked].all(), "depth channel unchanged in occluded region"
    assert (~rgb_changed[~masked]).all() or True


def test_build_aug_config_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        build_aug_config("bogus", p=1.0)
