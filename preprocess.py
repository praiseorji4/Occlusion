"""Convert KITTI / CityPersons to YOLO format and precompute MiDaS depth.

Outputs a ``{dataset}_yolo/`` tree with ``images/{train,val,test}`` and
``labels/{train,val,test}``, a ``{dataset}_yolo.yaml`` for ultralytics, a
``.occ.npy`` sidecar per label (per-box occlusion levels for label-aware
augmentation), and ``{dataset}_depth/`` MiDaS maps when ``--depth`` is set.

    python preprocess.py --dataset kitti
    python preprocess.py --dataset citypersons --depth
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from occlusion import paths
from occlusion.data import citypersons as cp
from occlusion.data import kitti_labels as kl

CP_TRAIN_CITIES = {
    "aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf",
    "erfurt", "hamburg", "hanover", "jena", "krefeld", "monchengladbach",
    "strasbourg", "stuttgart", "tubingen", "ulm", "weimar", "zurich",
}
CP_VAL_CITIES = {"frankfurt", "lindau", "munster"}
CP_TEST_CITIES = {"berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich"}


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _write_label(label_path: Path, lines: list[str], occ_lvls: np.ndarray | None) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    if occ_lvls is not None:
        np.save(str(label_path.with_suffix(".occ.npy")), occ_lvls.astype(np.int64))


def _img_size(img_path: Path) -> tuple[int, int]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Unreadable image: {img_path}")
    return img.shape[1], img.shape[0]


def process_kitti(out_root: Path) -> dict:
    raw = paths.kitti_root()
    img_dir = raw / "data_object_image_2" / "training" / "image_2"
    lbl_dir = raw / "data_object_label_2" / "training" / "label_2"
    if not img_dir.exists():
        raise FileNotFoundError(f"KITTI images not found: {img_dir}")

    splits = {"train": [], "val": [], "test": []}
    for img_path in sorted(img_dir.glob("*.png")):
        splits[kl.split_for_id(int(img_path.stem))].append(img_path.stem)

    counts = {s: 0 for s in splits}
    for split, stems in splits.items():
        for stem in stems:
            img_path = img_dir / f"{stem}.png"
            lbl_path = lbl_dir / f"{stem}.txt"
            w, h = _img_size(img_path)
            ann = kl.parse_label(lbl_path, w, h) if lbl_path.exists() else kl.empty_annotation(stem)
            _link_or_copy(img_path, out_root / "images" / split / img_path.name)
            _write_label(out_root / "labels" / split / f"{stem}.txt",
                         kl.to_yolo_lines(ann, w, h), ann["occlusion"].numpy())
            counts[split] += 1
    return counts


def process_citypersons(out_root: Path) -> dict:
    raw = paths.citypersons_root()
    ann_root = raw / "gtBbox_cityPersons_trainval"
    image_root = raw / "leftImg8bit_trainvaltest"
    if not ann_root.exists():
        raise FileNotFoundError(f"CityPersons annotations not found: {ann_root}")

    city_split = {}
    for cities, split in ((CP_TRAIN_CITIES, "train"), (CP_VAL_CITIES, "val"),
                          (CP_TEST_CITIES, "test")):
        for c in cities:
            city_split[c] = split

    counts = {"train": 0, "val": 0, "test": 0}
    for subdir in ("train", "val"):
        for json_path in sorted((ann_root / subdir).rglob("*.json")):
            city = json_path.name.split("_")[0]
            split = city_split.get(city)
            if split is None:
                continue
            img_stem = json_path.stem.replace("_gtBbox_cityPersons_annotation", "_leftImg8bit")
            img_path = image_root / subdir / city / f"{img_stem}.png"
            if not img_path.exists():
                continue
            w, h = _img_size(img_path)
            ann = cp.parse_annotation(json_path, w, h)
            _link_or_copy(img_path, out_root / "images" / split / img_path.name)
            _write_label(out_root / "labels" / split / f"{img_stem}.txt",
                         cp.to_yolo_lines(ann), ann["occlusion"].numpy())
            counts[split] += 1
    return counts


def write_yaml(dataset: str, out_root: Path, names: dict) -> Path:
    yaml_path = paths.dataset_yaml(dataset)
    data = {
        "path": str(out_root.resolve()),
        "nc": len(names),
        "names": names,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


def precompute_depth(dataset: str) -> None:
    from occlusion.data.depth import MiDaSDepthEstimator

    out_root = paths.yolo_root(dataset)
    depth_dir = paths.depth_root(dataset)
    image_dir = out_root / "images" / "train"
    if not image_dir.exists():
        return
    estimator = MiDaSDepthEstimator()
    estimator.precompute_directory(image_dir, depth_dir, depth_dir.parent / f"{dataset}_depth_conf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["kitti", "citypersons"])
    parser.add_argument("--depth", action="store_true", help="precompute MiDaS depth maps")
    args = parser.parse_args()

    out_root = paths.yolo_root(args.dataset)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dataset == "kitti":
        counts = process_kitti(out_root)
        names = kl.NAMES
    else:
        counts = process_citypersons(out_root)
        names = cp.NAMES

    yaml_path = write_yaml(args.dataset, out_root, names)
    print(f"[{args.dataset}] YOLO conversion: {counts} -> {out_root}")
    print(f"[{args.dataset}] dataset yaml: {yaml_path}")

    if args.depth:
        print(f"[{args.dataset}] precomputing MiDaS depth...")
        precompute_depth(args.dataset)


if __name__ == "__main__":
    main()
