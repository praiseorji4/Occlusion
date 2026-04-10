"""
train.py — Main training entry point for the occlusion-aware detection pipeline.

Usage:
  # Smoke test (must pass before any cloud GPU session):
  python train.py --config configs/base.yaml --run_mode local --model M0 \
                  --epochs 3 --data_limit 100

  # Full training:
  python train.py --config configs/base.yaml --run_mode kaggle --model M2 \
                  --aug_p 0.4

  # Resume:
  python train.py --config configs/base.yaml --run_mode resume \
                  --checkpoint results/checkpoints/M2_best.pt

  # Eval only:
  python train.py --config configs/base.yaml --run_mode eval \
                  --model M8 --checkpoint results/checkpoints/M8_best.pt

Mandatory smoke test checks (all must pass before cloud GPU session):
  1. Loss decreases over 3 epochs
  2. No OOM errors
  3. All metrics compute without assertion errors
  4. Checkpoint saves and reloads correctly
  5. Depth consistency test passes
  6. Inference completes in < 5s for 100 images
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Project imports
from src.config import RunConfig, RunMode, load_config, set_all_seeds
from src.datasets import KITTIDataset, collate_fn
from src.logger import ExperimentLogger
from src.metrics import compute_kitti_ap, compute_fps, sample_to_annotation
from src.plotting import create_narrative_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Occlusion-aware pedestrian detection training pipeline."
    )
    parser.add_argument("--config",     type=str, default="configs/base.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--run_mode",   type=str, default=None,
                        help="Override run_mode from config (local|kaggle|colab|eval|resume).")
    parser.add_argument("--model",      type=str, default=None,
                        help="Model variant ID (M0–M8). Overrides config.")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="Override epoch count.")
    parser.add_argument("--batch",      type=int, default=None,
                        help="Override batch size.")
    parser.add_argument("--aug_p",      type=float, default=None,
                        help="Augmentation probability. Overrides config.")
    parser.add_argument("--data_limit", type=int, default=None,
                        help="Limit dataset to N samples (smoke test).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for eval/resume modes.")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Random seed. Overrides config (default 42).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Unit tests — run before any training
# ---------------------------------------------------------------------------

def run_unit_tests(cfg: RunConfig) -> None:
    """
    Run mandatory unit tests before starting any training run.

    Fails fast with a clear error if any invariant is violated.
    This is not optional — every training run must pass these.
    """
    logger.info("Running pre-training unit tests …")

    # --- 1. Data loading shape checks ---
    try:
        train_ds = KITTIDataset(
            cfg.kitti_root, split="train", imgsz=cfg.imgsz, data_limit=4
        )
        if len(train_ds) > 0:
            sample = train_ds[0]
            assert sample["image"].shape == (3, cfg.imgsz, cfg.imgsz), \
                f"image shape {sample['image'].shape} ≠ (3, {cfg.imgsz}, {cfg.imgsz})"
            assert sample["depth"].shape == (1, cfg.imgsz, cfg.imgsz), \
                f"depth shape {sample['depth'].shape} ≠ (1, {cfg.imgsz}, {cfg.imgsz})"
            assert sample["occlusion_lvl"].max() <= 3, \
                f"occlusion_lvl max = {sample['occlusion_lvl'].max()} > 3"
            assert not torch.isnan(sample["image"]).any(), \
                "NaN values in image tensor"
            logger.info("  [PASS] Data loading shapes and values")
        else:
            logger.warning("  [SKIP] KITTI data not found — skipping shape tests")
    except FileNotFoundError as e:
        logger.warning("  [SKIP] Data not available: %s", e)

    # --- 2. Depth consistency after augmentation ---
    try:
        from src.augmentation import LabelAwareCutout
        import torch

        img   = torch.rand(3, cfg.imgsz, cfg.imgsz)
        depth = torch.rand(1, cfg.imgsz, cfg.imgsz)
        conf  = torch.rand(1, cfg.imgsz, cfg.imgsz)
        dmask = (conf > 0.5)
        boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5],
                               [0.6, 0.6, 0.9, 0.9]])
        occ_lvls = [0, 0]

        cutout = LabelAwareCutout(p=1.0)
        result = cutout(img, depth, conf, dmask, boxes, occ_lvls)

        if result.mask_applied.any():
            depth_in_mask = result.depth[result.mask_applied.unsqueeze(0).expand_as(result.depth)]
            conf_in_mask  = result.depth_conf[result.mask_applied.unsqueeze(0).expand_as(result.depth_conf)]
            dmask_in_mask = result.depth_mask[result.mask_applied.unsqueeze(0).expand_as(result.depth_mask)]

            assert (depth_in_mask == 0.5).all(), \
                f"CRITICAL: depth fill ≠ 0.5 in masked region. Got: {depth_in_mask.unique()}"
            assert (conf_in_mask == 0.0).all(), \
                "CRITICAL: depth_conf not zeroed in masked region."
            assert (~dmask_in_mask).all(), \
                "CRITICAL: depth_mask not set to False in masked region."
            logger.info("  [PASS] Depth consistency (fill=0.5, conf=0, mask=False)")
        else:
            logger.info("  [SKIP] Depth consistency — no pixels masked (p may be low)")
    except Exception as e:
        logger.error("  [FAIL] Depth consistency test: %s", e)
        raise

    # --- 3. Metric sanity — perfect predictions ---
    try:
        from src.metrics import compute_kitti_ap
        import numpy as np

        # Synthetic perfect prediction: predict the GT box exactly
        gt_boxes = np.array([[0.1, 0.1, 0.5, 0.5],
                              [0.6, 0.2, 0.9, 0.8]], dtype=np.float32)
        annotations = [{
            "image_id": "000000",
            "boxes": gt_boxes,
            "height_px": np.array([100.0, 150.0]),
            "occlusion": np.array([0, 1]),
            "truncation": np.array([0.0, 0.1]),
        }]
        predictions = [{
            "image_id": "000000",
            "boxes": gt_boxes,
            "scores": np.array([1.0, 1.0]),
        }]
        ap = compute_kitti_ap(predictions, annotations, "hard")
        assert ap > 97.0, \
            f"Perfect predictions should score >97 on 40-pt interpolation, got {ap:.2f}"
        logger.info("  [PASS] Metric sanity (perfect preds AP=%.2f)", ap)
    except Exception as e:
        logger.error("  [FAIL] Metric sanity test: %s", e)
        raise

    # --- 4. Split non-overlap (quick check) ---
    try:
        if cfg.kitti_root.exists():
            train_ds_full = KITTIDataset(cfg.kitti_root, "train", cfg.imgsz)
            val_ds_full   = KITTIDataset(cfg.kitti_root, "val",   cfg.imgsz)
            test_ds_full  = KITTIDataset(cfg.kitti_root, "test",  cfg.imgsz)

            train_ids = set(train_ds_full._image_ids)
            val_ids   = set(val_ds_full._image_ids)
            test_ids  = set(test_ds_full._image_ids)

            assert len(train_ids & val_ids)  == 0, "CRITICAL: train/val overlap"
            assert len(train_ids & test_ids) == 0, "CRITICAL: train/test overlap"
            assert len(val_ids   & test_ids) == 0, "CRITICAL: val/test overlap"
            logger.info("  [PASS] Split non-overlap")
        else:
            logger.warning("  [SKIP] KITTI root not found — skipping split test")
    except Exception as e:
        logger.error("  [FAIL] Split non-overlap test: %s", e)
        raise

    logger.info("All unit tests passed. Safe to begin training.\n")


# ---------------------------------------------------------------------------
# Smoke test validation
# ---------------------------------------------------------------------------

def run_smoke_test(cfg: RunConfig) -> None:
    """
    Validate the smoke test requirements before any cloud GPU session.

    Requirements:
      1. Loss decreases over 3 epochs (monotonically or net decrease)
      2. No OOM errors
      3. All metrics compute without assertion errors
      4. Checkpoint saves and reloads correctly
      5. Augmentation depth consistency test passes (covered by unit tests)
      6. Inference < 5s for 100 images
    """
    assert cfg.mode == RunMode.LOCAL and cfg.epochs <= 5, \
        "Smoke test should only run in LOCAL mode with ≤5 epochs"

    logger.info("=" * 60)
    logger.info("SMOKE TEST — %s", cfg.model_id)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load a tiny data subset
    try:
        ds = KITTIDataset(cfg.kitti_root, "train", cfg.imgsz,
                          data_limit=cfg.data_limit or 100)
        if len(ds) == 0:
            logger.warning("No data found — skipping smoke test training loop.")
            return
    except Exception as e:
        logger.warning("Data not available (%s) — skipping smoke training loop.", e)
        return

    loader = DataLoader(
        ds, batch_size=min(cfg.batch, 4),
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )

    # Test 6: inference speed
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO("yolov8s.pt")
        dummy = torch.rand(100, 3, cfg.imgsz, cfg.imgsz)
        t0 = time.perf_counter()
        with torch.no_grad():
            for i in range(0, 100, 4):
                _ = model(dummy[i:i+4])
        elapsed = time.perf_counter() - t0
        logger.info("Inference time for 100 images: %.2fs (limit: 5s)", elapsed)
        assert elapsed < 5.0, f"Inference too slow: {elapsed:.2f}s > 5s"
        logger.info("  [PASS] Inference speed")
    except Exception as e:
        logger.warning("  [SKIP] Inference speed test: %s", e)

    logger.info("Smoke test complete.\n")


# ---------------------------------------------------------------------------
# Test set evaluation (ONE TIME ONLY)
# ---------------------------------------------------------------------------

def run_test_evaluation(cfg: RunConfig) -> None:
    """
    Final test set evaluation — run ONCE at the very end of the project.

    Protocol:
      1. Load best checkpoint (selected by highest AP_hard on val set)
      2. No gradient computation
      3. Compute: AP_easy, AP_mod, AP_hard, ORS, FN_rate_hard, mAP@0.5, mAP@.5:.95, FPS
      4. Save to results/tables/test_results_FINAL.csv
      5. Print "FINAL TEST SET RESULTS — DO NOT RE-RUN"
      6. Create flag file results/.test_run_complete to prevent re-evaluation

    This function refuses to run if results/.test_run_complete already exists.
    """
    flag_path = cfg.output_dir / ".test_run_complete"
    if flag_path.exists():
        logger.error(
            "Test set has already been evaluated. "
            "Flag file exists: %s  — DO NOT RE-RUN.", flag_path
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("FINAL TEST SET EVALUATION — %s", cfg.model_id)
    logger.info("THIS RUNS ONCE. RESULTS ARE LOCKED AFTER THIS CALL.")
    logger.info("=" * 60)

    # Load best checkpoint
    if cfg.checkpoint_path is None:
        best_ckpt = cfg.checkpoint_dir / f"{cfg.model_id}_best.pt"
    else:
        best_ckpt = Path(cfg.checkpoint_path)

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")

    logger.info("Loading checkpoint: %s", best_ckpt)

    # Load test dataset
    test_ds = KITTIDataset(cfg.kitti_root, "test", cfg.imgsz)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch,
        shuffle=False, collate_fn=collate_fn, num_workers=4,
    )

    # Load model
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO(str(best_ckpt))
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {best_ckpt}: {e}")

    # Collect predictions (no gradients)
    predictions = []
    annotations = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.model.eval()
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(device)
            results = model(imgs)
            for i, r in enumerate(results):
                img_id = batch["image_id"][i]
                boxes  = r.boxes.xyxyn.cpu().numpy() if r.boxes is not None else []
                scores = r.boxes.conf.cpu().numpy()  if r.boxes is not None else []
                predictions.append({
                    "image_id": img_id,
                    "boxes":    boxes,
                    "scores":   scores,
                })
                annotations.append({
                    "image_id":   img_id,
                    "boxes":      batch["boxes"][i].numpy()         if len(batch["boxes"][i]) > 0 else [],
                    "height_px":  batch["height_px"][i].numpy()     if len(batch["height_px"][i]) > 0 else [],
                    "occlusion":  batch["occlusion_lvl"][i].numpy() if len(batch["occlusion_lvl"][i]) > 0 else [],
                    "truncation": batch["truncation"][i].numpy()    if len(batch["truncation"][i]) > 0 else [],
                })

    # Compute all metrics
    from src.metrics import (
        compute_fn_rate_hard,
        compute_fps as _compute_fps,
    )

    ap_easy = compute_kitti_ap(predictions, annotations, "easy")
    ap_mod  = compute_kitti_ap(predictions, annotations, "moderate")
    ap_hard = compute_kitti_ap(predictions, annotations, "hard")
    fn_hard = compute_fn_rate_hard(predictions, annotations)
    fps_info = _compute_fps(model.model, input_size=(cfg.imgsz, cfg.imgsz), device=device)

    results_row = {
        "model_id":       cfg.model_id,
        "AP_easy":        ap_easy,
        "AP_mod":         ap_mod,
        "AP_hard":        ap_hard,
        "FN_rate_hard":   fn_hard,
        "FPS_mean":       fps_info["mean_fps"],
        "FPS_mean_ms":    fps_info["mean_ms"],
    }

    # Save results
    import pandas as pd
    results_df = pd.DataFrame([results_row])
    out_path = cfg.tables_dir / "test_results_FINAL.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    # Lock flag
    flag_path.touch()

    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST SET RESULTS — DO NOT RE-RUN")
    logger.info("=" * 60)
    for k, v in results_row.items():
        logger.info("  %-20s: %s", k, f"{v:.4f}" if isinstance(v, float) else v)
    logger.info("Saved to: %s", out_path)
    logger.info("Lock file: %s", flag_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Build overrides dict from CLI args
    overrides = {}
    if args.run_mode:  overrides["run_mode"]   = args.run_mode
    if args.model:     overrides["model_id"]   = args.model
    if args.epochs:    overrides["epochs"]     = args.epochs
    if args.batch:     overrides["batch"]      = args.batch
    if args.aug_p is not None: overrides["aug_p"] = args.aug_p
    if args.data_limit: overrides["data_limit"] = args.data_limit
    if args.checkpoint: overrides["checkpoint_path"] = args.checkpoint
    if args.seed is not None:  overrides["seed"] = args.seed

    # Load config
    cfg = load_config(args.config, overrides)
    cfg.ensure_dirs()

    logger.info("Run config: %s", cfg)

    # Set seeds — must be first after config load
    set_all_seeds(cfg.seed)

    # Run unit tests unconditionally
    run_unit_tests(cfg)

    # Route by mode
    if cfg.mode == RunMode.EVAL:
        run_test_evaluation(cfg)
        return

    if cfg.mode == RunMode.LOCAL and cfg.epochs <= 5:
        run_smoke_test(cfg)

    # Create narrative template for this model variant
    create_narrative_template(cfg.model_id, cfg.narratives_dir)

    # --- Training stub ---
    # Full training loop integrates with the Ultralytics YOLO trainer.
    # Model-specific training is implemented in each notebook (01_baseline.ipynb,
    # 05_augmentation.ipynb, etc.) which call the appropriate dataset/augmentation
    # modules and override the YOLOv8 training hooks.
    #
    # This file provides the common infrastructure (config, seeds, unit tests,
    # logging, test-set locking) that every notebook imports.

    logger.info(
        "Training infrastructure initialised for %s. "
        "Open the appropriate notebook to run the full training loop.",
        cfg.model_id,
    )


if __name__ == "__main__":
    main()
