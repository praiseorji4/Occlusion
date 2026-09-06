"""MiDaS monocular depth precomputation and confidence estimation.

Run once offline (``preprocess.py``). Depth is saved as float32 ``.npy`` in
[0, 1] (inverse relative depth); confidence as a separate ``.npy``. Both
fusion paths consume these: late-feature stacks depth as a 4th channel,
cross-attention uses depth + confidence for gating and masking.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DepthConfidenceEstimator:
    """Pixel-wise confidence for a MiDaS depth estimate.

    C = 0.4 * (1 - |grad depth|) + 0.4 * |grad grey| + 0.2 * gauss(depth, 0.5).
    Depth boundaries and textureless regions get low confidence.
    """
    W_GRAD, W_TEX, W_VAL = 0.4, 0.4, 0.2

    @staticmethod
    def _norm(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-8:
            return np.zeros_like(arr)
        return (arr - lo) / (hi - lo)

    @staticmethod
    def _grad(arr: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    def estimate(self, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        c_grad = np.clip(1.0 - self._norm(self._grad(depth)), 0.0, 1.0)
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        c_tex = np.clip(self._norm(self._grad(grey)), 0.0, 1.0)
        c_val = np.clip(np.exp(-((depth - 0.5) ** 2) / 0.18), 0.0, 1.0)
        return (self.W_GRAD * c_grad + self.W_TEX * c_tex + self.W_VAL * c_val).astype(np.float32)


class MiDaSDepthEstimator:
    """Lazy MiDaS DPT_Hybrid wrapper for offline depth precomputation."""

    MODEL_NAME = "DPT_Hybrid"
    HUB_REPO = "intel-isl/MiDaS"

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._model = None
        self._transform = None
        self._conf = DepthConfidenceEstimator()

    def _load(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading MiDaS %s from torch.hub", self.MODEL_NAME)
        self._model = torch.hub.load(self.HUB_REPO, self.MODEL_NAME, pretrained=True)
        self._model.eval().to(self.device)
        transforms = torch.hub.load(self.HUB_REPO, "transforms")
        self._transform = transforms.dpt_transform

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        self._load()
        x = self._transform(rgb).to(self.device)
        with torch.no_grad():
            pred = self._model(x)
            pred = F.interpolate(pred.unsqueeze(1), size=rgb.shape[:2],
                                 mode="bicubic", align_corners=False).squeeze()
        depth = cv2.GaussianBlur(pred.cpu().numpy().astype(np.float32), (0, 0), 1.5)
        lo, hi = depth.min(), depth.max()
        if hi - lo > 1e-8:
            depth = (depth - lo) / (hi - lo)
        else:
            depth = np.zeros_like(depth)
        return depth

    def precompute_directory(
        self, image_dir: Path, depth_dir: Path, conf_dir: Path,
        vis_dir: Optional[Path] = None, vis_every: int = 200,
    ) -> None:
        self._load()
        image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
        if not image_paths:
            logger.warning("No images in %s", image_dir)
            return
        depth_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)
        if vis_dir is not None:
            vis_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Precomputing depth for %d images -> %s", len(image_paths), depth_dir)

        for i, img_path in enumerate(image_paths):
            stem = img_path.stem
            depth_path, conf_path = depth_dir / f"{stem}.npy", conf_dir / f"{stem}.npy"
            if depth_path.exists() and conf_path.exists():
                continue
            rgb = cv2.imread(str(img_path))
            if rgb is None:
                logger.warning("Unreadable image: %s", img_path)
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = self.estimate(rgb)
            conf = self._conf.estimate(depth, rgb)
            np.save(str(depth_path), depth)
            np.save(str(conf_path), conf)
            if vis_dir is not None and i % vis_every == 0:
                self._save_vis(rgb, depth, conf, vis_dir / f"{stem}_vis.png")
            if (i + 1) % 100 == 0:
                logger.info("  %d / %d", i + 1, len(image_paths))
        logger.info("Depth precomputation complete.")

    @staticmethod
    def _save_vis(rgb: np.ndarray, depth: np.ndarray, conf: np.ndarray, path: Path) -> None:
        def cmap(arr, c):
            return cv2.applyColorMap((arr * 255).clip(0, 255).astype(np.uint8), c)
        panel = np.concatenate([
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            cmap(depth, cv2.COLORMAP_INFERNO),
            cmap(conf, cv2.COLORMAP_VIRIDIS),
        ], axis=1)
        cv2.imwrite(str(path), panel)
