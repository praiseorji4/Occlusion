"""Train one experiment row: config -> seeds -> trainer -> checkpoint + CSV.

    python train.py --config configs/experiment/m2_kitti_cutout.yaml --seed 42
    python train.py --config configs/experiment/m0_kitti.yaml --epochs 1 --data_limit 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from occlusion import paths
from occlusion.config import load_config
from occlusion.seeds import set_all_seeds


def _fraction_from_data_limit(data_limit: int | None) -> float | None:
    if data_limit is None:
        return None
    img_dir = paths.yolo_root("kitti") / "images" / "train"
    total = len(list(img_dir.glob("*.png"))) if img_dir.exists() else 0
    if total == 0:
        return None
    return max(min(data_limit / total, 1.0), 1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--data_limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.batch is not None:
        overrides["batch"] = args.batch

    cfg = load_config(args.config, overrides=overrides or None)
    set_all_seeds(cfg.seed)
    paths.ensure_output_dirs()

    extra = {}
    if args.device:
        extra["device"] = args.device
    fraction = _fraction_from_data_limit(args.data_limit)
    if fraction is not None:
        extra["fraction"] = fraction
    if args.resume:
        extra["resume"] = True

    from occlusion.models.trainer import build_trainer
    trainer = build_trainer(cfg, extra=extra)
    trainer.train()

    csv_src = Path(trainer.save_dir) / "results.csv"
    if csv_src.exists():
        dst = paths.tables_dir() / f"{cfg.run_name}.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(csv_src.read_bytes())
        print(f"[{cfg.run_name}] results -> {dst}")


if __name__ == "__main__":
    main()
