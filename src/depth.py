"""
src/depth.py — MiDaS depth precomputation and confidence estimation.

CRITICAL: MiDaS is run ONCE offline, never during training.
If depth .npy files are missing at training time, the dataset loader returns
zeros and logs a warning. This module is only called from:
  notebooks/04_depth_precomputation.ipynb

Depth maps are saved as float32 .npy in [0, 1] (normalised inverse relative depth).
Confidence maps encode how much each pixel's depth estimate should be trusted:
  C = 0.4 * C_grad + 0.4 * C_tex + 0.2 * C_val
  where:
    C_grad = 1 - normalise(|∇depth|)   high gradient → depth boundary → low conf
    C_tex  = normalise(|∇RGB_grey|)    high texture  → more reliable
    C_val  = exp(-((depth-0.5)²)/0.18) extreme values less reliable

Output directory structure:
  data/kitti/
    depth_hybrid/training/image_2/{image_id}.npy
    depth_conf/training/image_2/{image_id}_conf.npy
  data/citypersons/
    depth_hybrid/train/{city}/{stem}.npy
    depth_conf/train/{city}/{stem}.npy
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Confidence estimator
# ---------------------------------------------------------------------------

class DepthConfidenceEstimator:
    """
    Compute a pixel-wise confidence map for a MiDaS depth estimate.

    Three components weighted and summed:
      C_grad (0.4): Inverse of normalised gradient magnitude.
                    High gradient regions are depth boundaries → unreliable.
      C_tex  (0.4): Normalised gradient of greyscale RGB.
                    Texture-rich regions are more depth-reliable.
      C_val  (0.2): Gaussian centred at 0.5.
                    Extreme depth values (near 0 or 1) are less reliable.

    All components are clipped to [0, 1] before weighting.
    """

    W_GRAD = 0.4
    W_TEX  = 0.4
    W_VAL  = 0.2

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1]. Returns zeros if constant."""
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
        """Sobel gradient magnitude of a 2D array."""
        gx = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    def estimate(
        self,
        depth: np.ndarray,
        rgb: np.ndarray,
    ) -> np.ndarray:
        """
        Compute confidence map.

        Args:
            depth: (H, W) float32 depth map in [0, 1].
            rgb:   (H, W, 3) uint8 RGB image.

        Returns:
            (H, W) float32 confidence map in [0, 1].
        """
        # C_grad: inverse of normalised depth gradient magnitude
        grad_depth = self._gradient_magnitude(depth)
        c_grad = 1.0 - self._normalise(grad_depth)
        c_grad = np.clip(c_grad, 0.0, 1.0)

        # C_tex: normalised RGB grey gradient magnitude
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        grad_tex = self._gradient_magnitude(grey)
        c_tex = self._normalise(grad_tex)
        c_tex = np.clip(c_tex, 0.0, 1.0)

        # C_val: Gaussian centred at 0.5
        c_val = np.exp(-((depth - 0.5) ** 2) / 0.18)
        c_val = np.clip(c_val, 0.0, 1.0)

        conf = self.W_GRAD * c_grad + self.W_TEX * c_tex + self.W_VAL * c_val
        return conf.astype(np.float32)


# ---------------------------------------------------------------------------
# MiDaS depth estimator
# ---------------------------------------------------------------------------

class MiDaSDepthEstimator:
    """
    Wrapper around MiDaS DPT_Hybrid for monocular depth estimation.

    Only used for offline precomputation — NEVER imported in training code.

    Args:
        device: Torch device. Defaults to CUDA if available.
    """

    MODEL_NAME = "DPT_Hybrid"
    HUB_REPO   = "intel-isl/MiDaS"

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._model = None
        self._transform = None
        self._conf_estimator = DepthConfidenceEstimator()

    def _load_model(self) -> None:
        """Lazy-load MiDaS from torch.hub (downloads on first call)."""
        if self._model is not None:
            return
        logger.info("Loading MiDaS %s from torch.hub …", self.MODEL_NAME)
        self._model = torch.hub.load(
            self.HUB_REPO, self.MODEL_NAME, pretrained=True
        )
        self._model.eval().to(self.device)

        midas_transforms = torch.hub.load(self.HUB_REPO, "transforms")
        self._transform = midas_transforms.dpt_transform
        logger.info("MiDaS loaded on %s", self.device)

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """
        Estimate depth for a single RGB image.

        Args:
            rgb: (H, W, 3) uint8 array.

        Returns:
            (H, W) float32 depth map normalised to [0, 1].
            Values represent inverse relative depth (higher = closer to camera).
        """
        self._load_model()
        input_tensor = self._transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self._model(input_tensor)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # Apply Gaussian smoothing (σ=1.5) to reduce boundary noise
        depth = cv2.GaussianBlur(depth, ksize=(0, 0), sigmaX=1.5)

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth

    def precompute_dataset(
        self,
        image_dir: Path,
        depth_output_dir: Path,
        conf_output_dir: Path,
        batch_size: int = 8,
        vis_dir: Optional[Path] = None,
        vis_every: int = 200,
    ) -> None:
        """
        Precompute depth and confidence maps for all images in a directory.

        Saves:
          {depth_output_dir}/{image_stem}.npy
          {conf_output_dir}/{image_stem}.npy

        Optionally saves side-by-side RGB/depth/confidence visualisations
        every `vis_every` images to `vis_dir`.

        Args:
            image_dir:        Directory containing .png or .jpg images.
            depth_output_dir: Where to save depth .npy files.
            conf_output_dir:  Where to save confidence .npy files.
            batch_size:       Number of images to batch (memory permitting).
            vis_dir:          Where to save visualisation images.
            vis_every:        Save a visualisation every N images.
        """
        self._load_model()

        image_paths = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        )
        if not image_paths:
            logger.warning("No images found in %s", image_dir)
            return

        depth_output_dir.mkdir(parents=True, exist_ok=True)
        conf_output_dir.mkdir(parents=True, exist_ok=True)
        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Precomputing depth for %d images → %s",
            len(image_paths), depth_output_dir
        )

        for i, img_path in enumerate(image_paths):
            stem = img_path.stem
            depth_path = depth_output_dir / f"{stem}.npy"
            conf_path  = conf_output_dir  / f"{stem}.npy"

            # Skip if already computed
            if depth_path.exists() and conf_path.exists():
                continue

            rgb = cv2.imread(str(img_path))
            if rgb is None:
                logger.warning("Could not read image: %s", img_path)
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            depth = self.estimate(rgb)
            conf  = self._conf_estimator.estimate(depth, rgb)

            np.save(str(depth_path), depth)
            np.save(str(conf_path),  conf)

            if vis_dir is not None and i % vis_every == 0:
                self._save_visualisation(
                    rgb, depth, conf,
                    vis_dir / f"{stem}_vis.png",
                )

            if (i + 1) % 100 == 0:
                logger.info("  %d / %d done", i + 1, len(image_paths))

        logger.info("Depth precomputation complete.")

    @staticmethod
    def _save_visualisation(
        rgb: np.ndarray,
        depth: np.ndarray,
        conf: np.ndarray,
        save_path: Path,
    ) -> None:
        """
        Save a side-by-side RGB | depth | confidence visualisation.

        All panels are resized to the same height for easy comparison.
        """
        H, W = rgb.shape[:2]

        def to_colour_map(arr: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
            arr_u8 = (arr * 255).clip(0, 255).astype(np.uint8)
            return cv2.applyColorMap(arr_u8, cmap)

        rgb_bgr   = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_col = to_colour_map(depth, cv2.COLORMAP_INFERNO)
        conf_col  = to_colour_map(conf,  cv2.COLORMAP_VIRIDIS)

        panel = np.concatenate([rgb_bgr, depth_col, conf_col], axis=1)

        # Add text labels
        for j, label in enumerate(["RGB", "Depth (MiDaS)", "Confidence"]):
            cv2.putText(
                panel, label,
                (j * W + 5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2,
            )

        cv2.imwrite(str(save_path), panel)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def precompute_kitti(kitti_root: Path, batch_size: int = 8) -> None:
    """Precompute depth maps for KITTI training images."""
    estimator = MiDaSDepthEstimator()
    image_dir        = kitti_root / "data_object_image_2" / "training" / "image_2"
    depth_output_dir = kitti_root / "depth_hybrid"        / "training" / "image_2"
    conf_output_dir  = kitti_root / "depth_conf"          / "training" / "image_2"
    vis_dir          = kitti_root / "depth_vis"

    estimator.precompute_dataset(
        image_dir, depth_output_dir, conf_output_dir,
        batch_size=batch_size, vis_dir=vis_dir,
    )


def precompute_citypersons(cp_root: Path, split: str = "train", batch_size: int = 8) -> None:
    """
    Precompute depth maps for a CityPersons split.
    Iterates over city subdirectories.
    """
    estimator = MiDaSDepthEstimator()
    image_root = cp_root / "leftImg8bit_trainvaltest" / split

    for city_dir in sorted(image_root.iterdir()):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        depth_dir = cp_root / "depth_hybrid" / split / city
        conf_dir  = cp_root / "depth_conf"   / split / city
        vis_dir   = cp_root / "depth_vis"    / split / city

        logger.info("Precomputing depth for CityPersons %s / %s", split, city)
        estimator.precompute_dataset(
            city_dir, depth_dir, conf_dir,
            batch_size=batch_size, vis_dir=vis_dir,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute MiDaS depth maps.")
    parser.add_argument("--dataset",    choices=["kitti", "citypersons"], required=True)
    parser.add_argument("--data_root",  type=Path, required=True)
    parser.add_argument("--split",      default="train", help="CityPersons split only")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.dataset == "kitti":
        precompute_kitti(args.data_root, args.batch_size)
    else:
        precompute_citypersons(args.data_root, args.split, args.batch_size)
