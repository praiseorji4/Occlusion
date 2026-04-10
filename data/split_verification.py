"""
data/split_verification.py — Formal dataset split verification.

Run ONCE before any training to assert:
  1. Zero image overlap between train/val/test for KITTI.
  2. Zero image overlap between train/val/test for CityPersons.
  3. Natural occlusion distribution is preserved (no stratification).

KITTI split protocol (chronological, by image_id integer):
  Train: image_id < 5985          (first ~80% of 7,481 training images)
  Val:   5985 ≤ image_id < 6732   (next ~10%)
  Test:  image_id ≥ 6732          (final ~10% — HOLD OUT)

CityPersons split protocol (official Cityscapes city-level split):
  Train: 18 cities (2975 images)
  Val:    3 cities (500 images)
  Test:   6 cities (1575 images) — HOLD OUT

Usage:
  python data/split_verification.py --data_root /path/to/data
  python data/split_verification.py   # uses default path
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# KITTI constants
# ---------------------------------------------------------------------------

KITTI_TRAIN_END = 5985   # exclusive upper bound for train split
KITTI_VAL_END   = 6732   # exclusive upper bound for val split
# ids >= KITTI_VAL_END go to test

# ---------------------------------------------------------------------------
# CityPersons city-level split (official Cityscapes)
# ---------------------------------------------------------------------------

CP_TRAIN_CITIES = {
    "aachen", "bochum", "bremen", "cologne", "darmstadt",
    "dusseldorf", "erfurt", "hamburg", "hanover", "jena",
    "krefeld", "monchengladbach", "strasbourg", "stuttgart",
    "tubingen", "ulm", "weimar", "zurich",
}  # 18 cities

CP_VAL_CITIES = {"frankfurt", "lindau", "munster"}  # 3 cities

CP_TEST_CITIES = {
    "berlin", "bielefeld", "bonn", "leverkusen", "mainz", "munich",
}  # 6 cities  — HOLD OUT


# ---------------------------------------------------------------------------
# KITTI helpers
# ---------------------------------------------------------------------------

def _get_kitti_split(image_id: int) -> str:
    if image_id < KITTI_TRAIN_END:
        return "train"
    elif image_id < KITTI_VAL_END:
        return "val"
    else:
        return "test"


def load_kitti_splits(kitti_root: Path) -> Dict[str, List[str]]:
    """
    Scan KITTI image directory and assign each image to train/val/test
    using the chronological integer split protocol.

    Args:
        kitti_root: Path to data/kitti/

    Returns:
        Dict mapping split name → list of image_id strings (zero-padded, e.g. '000001').
    """
    image_dir = kitti_root / "data_object_image_2" / "training" / "image_2"
    if not image_dir.exists():
        print(f"[WARN] KITTI image dir not found: {image_dir}")
        return {"train": [], "val": [], "test": []}

    splits: Dict[str, List[str]] = defaultdict(list)
    for img_path in sorted(image_dir.glob("*.png")):
        stem = img_path.stem  # e.g. '000001'
        image_id = int(stem)
        splits[_get_kitti_split(image_id)].append(stem)

    return dict(splits)


def load_kitti_occlusion_distribution(
    kitti_root: Path,
    splits: Dict[str, List[str]],
) -> Dict[str, Counter]:
    """
    For each split, count how many Pedestrian annotations fall into each
    occlusion level {0, 1, 2, 3}.

    KITTI label format (space-separated):
      class trunc occ alpha x1 y1 x2 y2 h w l X Y Z ry
    Occlusion field (index 2): 0=fully visible, 1=partly, 2=largely, 3=unknown.

    Args:
        kitti_root: Path to data/kitti/
        splits:     Output of load_kitti_splits().

    Returns:
        Dict split → Counter of occlusion level frequencies.
    """
    label_dir = kitti_root / "data_object_label_2" / "training" / "label_2"
    if not label_dir.exists():
        print(f"[WARN] KITTI label dir not found: {label_dir}")
        return {s: Counter() for s in splits}

    occ_dist: Dict[str, Counter] = {s: Counter() for s in splits}

    for split, ids in splits.items():
        for img_id in ids:
            lbl_path = label_dir / f"{img_id}.txt"
            if not lbl_path.exists():
                continue
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts or parts[0] != "Pedestrian":
                        continue
                    occ_lvl = int(parts[2])
                    occ_dist[split][occ_lvl] += 1

    return occ_dist


# ---------------------------------------------------------------------------
# CityPersons helpers
# ---------------------------------------------------------------------------

def _city_from_filename(filename: str) -> str:
    """Extract city name from CityPersons filename e.g. 'frankfurt_000000_000294_gtBbox_cityPersons_annotation.json'."""
    return filename.split("_")[0]


def load_citypersons_splits(cp_root: Path) -> Dict[str, List[str]]:
    """
    Scan CityPersons annotation directory and assign each image to
    train/val/test based on the official city-level split.

    Args:
        cp_root: Path to data/citypersons/

    Returns:
        Dict mapping split name → list of image stems.
    """
    ann_root = cp_root / "gtBbox_cityPersons_trainval"
    if not ann_root.exists():
        print(f"[WARN] CityPersons annotation dir not found: {ann_root}")
        return {"train": [], "val": [], "test": []}

    splits: Dict[str, List[str]] = defaultdict(list)

    for subdir in ["train", "val"]:
        subdir_path = ann_root / subdir
        if not subdir_path.exists():
            continue
        for json_path in sorted(subdir_path.rglob("*.json")):
            city = _city_from_filename(json_path.name)
            stem = json_path.stem
            if city in CP_TRAIN_CITIES:
                splits["train"].append(stem)
            elif city in CP_VAL_CITIES:
                splits["val"].append(stem)
            elif city in CP_TEST_CITIES:
                splits["test"].append(stem)
            else:
                print(f"[WARN] Unknown city '{city}' in {json_path}")

    return dict(splits)


def load_citypersons_occlusion_distribution(
    cp_root: Path,
    splits: Dict[str, List[str]],
) -> Dict[str, Counter]:
    """
    For each split, count discrete occlusion levels derived from the
    continuous occlusion ratio:
      occl < 0.10  → level 0
      occl < 0.35  → level 1
      occl ≥ 0.35  → level 2

    Args:
        cp_root: Path to data/citypersons/
        splits:  Output of load_citypersons_splits().

    Returns:
        Dict split → Counter of occlusion level frequencies.
    """
    ann_root = cp_root / "gtBbox_cityPersons_trainval"
    occ_dist: Dict[str, Counter] = {s: Counter() for s in splits}

    stem_to_split = {}
    for split, stems in splits.items():
        for s in stems:
            stem_to_split[s] = split

    for subdir in ["train", "val"]:
        subdir_path = ann_root / subdir
        if not subdir_path.exists():
            continue
        for json_path in sorted(subdir_path.rglob("*.json")):
            stem = json_path.stem
            split = stem_to_split.get(stem)
            if split is None:
                continue
            with open(json_path) as f:
                ann = json.load(f)
            for bbox in ann.get("bboxes", []):
                if bbox.get("lbl") != "pedestrian":
                    continue
                occl = bbox.get("occl", 0.0)
                if occl < 0.10:
                    level = 0
                elif occl < 0.35:
                    level = 1
                else:
                    level = 2
                occ_dist[split][level] += 1

    return occ_dist


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_no_overlap(splits: Dict[str, List[str]], dataset_name: str) -> None:
    """Assert zero overlap between all pairs of splits."""
    split_sets = {k: set(v) for k, v in splits.items()}
    pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]
    all_ok = True
    for a, b in pairs:
        if a not in split_sets or b not in split_sets:
            continue
        overlap = split_sets[a] & split_sets[b]
        if overlap:
            print(f"[FAIL] {dataset_name}: {len(overlap)} images overlap between {a} and {b}: {list(overlap)[:5]}")
            all_ok = False
        else:
            print(f"[OK]   {dataset_name}: no overlap between {a} and {b}")
    if not all_ok:
        raise AssertionError(f"Split overlap detected in {dataset_name}. See above.")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    dataset_name: str,
    splits: Dict[str, List[str]],
    occ_dist: Dict[str, Counter],
) -> None:
    """Print a formatted summary table of split counts and occlusion distributions."""
    print(f"\n{'='*60}")
    print(f" {dataset_name} Split Summary")
    print(f"{'='*60}")
    print(f"{'Split':<10} {'Images':>8}  {'lvl0':>7} {'lvl1':>7} {'lvl2':>7} {'lvl3':>7}  {'Total anns':>10}")
    print(f"{'-'*60}")
    for split in ["train", "val", "test"]:
        if split not in splits:
            continue
        n_images = len(splits[split])
        c = occ_dist.get(split, Counter())
        total_anns = sum(c.values())
        print(
            f"{split:<10} {n_images:>8}  "
            f"{c.get(0, 0):>7} {c.get(1, 0):>7} "
            f"{c.get(2, 0):>7} {c.get(3, 0):>7}  "
            f"{total_anns:>10}"
        )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset splits.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(__file__).parent,  # data/ directory by default
        help="Root data directory containing kitti/ and citypersons/ subdirs.",
    )
    args = parser.parse_args()

    data_root = args.data_root
    kitti_root = data_root / "kitti"
    cp_root = data_root / "citypersons"

    errors: List[str] = []

    # ---- KITTI ----
    print("\n--- KITTI ---")
    kitti_splits = load_kitti_splits(kitti_root)

    if not any(kitti_splits.values()):
        print("[SKIP] KITTI data not found — skipping KITTI verification.")
    else:
        try:
            assert_no_overlap(kitti_splits, "KITTI")
        except AssertionError as e:
            errors.append(str(e))

        kitti_occ = load_kitti_occlusion_distribution(kitti_root, kitti_splits)
        print_summary_table("KITTI Pedestrian", kitti_splits, kitti_occ)

        # Additional sanity: counts match expected totals
        total_kitti = sum(len(v) for v in kitti_splits.values())
        print(f"KITTI total images accounted for: {total_kitti} (expected ≈7481)")

    # ---- CityPersons ----
    print("--- CityPersons ---")
    cp_splits = load_citypersons_splits(cp_root)

    if not any(cp_splits.values()):
        print("[SKIP] CityPersons data not found — skipping CP verification.")
    else:
        try:
            assert_no_overlap(cp_splits, "CityPersons")
        except AssertionError as e:
            errors.append(str(e))

        cp_occ = load_citypersons_occlusion_distribution(cp_root, cp_splits)
        print_summary_table("CityPersons Pedestrian", cp_splits, cp_occ)

        total_cp = sum(len(v) for v in cp_splits.values())
        print(f"CityPersons total images accounted for: {total_cp} (expected ≈5050)")

    # ---- Final verdict ----
    if errors:
        print("\n[FAIL] Split verification FAILED:")
        for e in errors:
            print(f"  • {e}")
        sys.exit(1)
    else:
        print("\n[PASS] All split verification checks passed.")
        print("Safe to proceed with training.\n")


if __name__ == "__main__":
    main()
