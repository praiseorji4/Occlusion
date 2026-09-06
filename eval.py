"""Evaluate a checkpoint: ultralytics val + KITTI 40-point AP -> CSV.

    python eval.py --checkpoint runs/.../weights/best.pt --config configs/experiment/m2_kitti_cutout.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from occlusion import paths
from occlusion.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths.ensure_output_dirs()

    from ultralytics import YOLO
    model = YOLO(str(args.checkpoint))
    metrics = model.val(data=str(paths.dataset_yaml(cfg.dataset)), split=args.split,
                        imgsz=cfg.imgsz, device=args.device, verbose=True)

    row = {
        "run": cfg.run_name,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "fusion_type": cfg.fusion_type,
        "aug_strategy": cfg.aug_strategy,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }

    out = paths.tables_dir() / f"eval_{cfg.run_name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"[eval] {cfg.run_name}: mAP50={row['mAP50']:.4f} mAP50-95={row['mAP50_95']:.4f} -> {out}")


if __name__ == "__main__":
    main()
