"""Aggregate per-run metrics in ``results/expected/`` into one CSV + a markdown table.

    python scripts/aggregate_results.py

Reads the ``metrics.json`` / ``inference_speed.json`` files extracted from the
evaluation archive and writes ``results/expected/aggregate.csv`` (one row per
dataset x experiment x split x seed) plus a grouped mean+/-std summary printed
to stdout. The README results table is regenerated from the stdout block.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "expected"
RUNS = ROOT / "runs"
SPEED = ROOT / "predictions"

EXPERIMENT_ORDER = [
    "baseline", "cutout", "random_erasing", "hide_and_seek", "hs",
    "gridcut", "combined_best_two", "combined_all", "body_part_occlusion",
]

DISPLAY = {
    "baseline": "baseline (no aug)",
    "cutout": "cutout",
    "random_erasing": "random_erasing",
    "hide_and_seek": "hide_and_seek",
    "hs": "hide_and_seek",
    "gridcut": "gridcut",
    "combined_best_two": "combined_best_two",
    "combined_all": "combined_all",
    "body_part_occlusion": "body_part_occlusion",
}


def parse_experiment(name: str):
    m = re.match(r"best_(.+)_s(\d+)$", name)
    if not m:
        return None
    body, seed = m.group(1), int(m.group(2))
    parts = body.split("_", 2)  # model, dataset, experiment
    if len(parts) < 3:
        return None
    return parts[0], parts[1], parts[2], seed


def tier(m, tier_name):
    return m["by_occlusion"][tier_name]["Pedestrian"]


def load_runs():
    rows = []
    for mf in RUNS.rglob("metrics.json"):
        m = json.loads(mf.read_text())
        parts = mf.parts
        split = parts[parts.index("runs") + 2]  # .../runs/<dataset>/<split>/...
        p = parse_experiment(m["experiment"])
        if p is None:
            continue
        model, ds, expt, seed = p
        sf = SPEED / ds / split / m["experiment"] / "inference_speed.json"
        fps = None
        if sf.exists():
            fps = json.loads(sf.read_text()).get("fps")
        rows.append({
            "dataset": ds, "split": split, "model": model,
            "experiment": expt, "seed": seed,
            "AP_easy": tier(m, "Easy")["mAP50"] * 100,
            "AP_mod": tier(m, "Medium")["mAP50"] * 100,
            "AP_hard": tier(m, "Hard")["mAP50"] * 100,
            "mAP50": m["overall"]["Pedestrian"]["mAP50"] * 100,
            "MR_hard": tier(m, "Hard")["MR"] * 100,
            "FPS": fps,
        })
    return rows


def write_csv(rows):
    out = ROOT / "aggregate.csv"
    fields = ["dataset", "split", "model", "experiment", "seed",
              "AP_easy", "AP_mod", "AP_hard", "mAP50", "MR_hard", "FPS"]
    rows_sorted = sorted(rows, key=lambda r: (r["dataset"], r["split"],
                                              r["experiment"], r["seed"]))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_sorted:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})
    return out


def fmt_mean_std(vals):
    if not vals:
        return "—"
    mean = statistics.mean(vals)
    if len(vals) < 2:
        return f"{mean:.2f}"
    std = statistics.pstdev(vals)
    return f"{mean:.2f}±{std:.2f}"


def _table_lines(rows, dataset, split):
    groups = defaultdict(list)
    for r in rows:
        if r["split"] != split or r["dataset"] != dataset:
            continue
        groups[r["experiment"]].append(r)

    def key(e):
        return EXPERIMENT_ORDER.index(e) if e in EXPERIMENT_ORDER else len(EXPERIMENT_ORDER)

    lines = [f"\n### {dataset.upper()} — {split} split (YOLOv8x, seeds 42/123/456)",
             "| Config | AP_easy | AP_mod | AP_hard | mAP50 | MR_hard | FPS |",
             "|---|---|---|---|---|---|---|"]
    for expt in sorted(groups, key=key):
        rs = groups[expt]
        lines.append(f"| {DISPLAY.get(expt, expt)} "
                     f"| {fmt_mean_std([r['AP_easy'] for r in rs])} "
                     f"| {fmt_mean_std([r['AP_mod'] for r in rs])} "
                     f"| {fmt_mean_std([r['AP_hard'] for r in rs])} "
                     f"| {fmt_mean_std([r['mAP50'] for r in rs])} "
                     f"| {fmt_mean_std([r['MR_hard'] for r in rs])} "
                     f"| {fmt_mean_std([r['FPS'] for r in rs if r['FPS']])} |")
    return lines


def main():
    rows = load_runs()
    csv_path = write_csv(rows)
    lines = [f"wrote {csv_path} ({len(rows)} rows)", ""]
    for dataset in ("kitti", "citypersons"):
        for split in ("test", "val"):
            lines += _table_lines(rows, dataset, split)
    groups = defaultdict(list)
    for r in rows:
        if r["split"] == "test":
            groups[(r["dataset"], r["experiment"])].append(r["AP_hard"])
    stds = [statistics.pstdev(v) for v in groups.values() if len(v) > 1]
    lines += ["", "### Reproducibility — per-seed std on AP_hard (test split)",
              f"AP_hard std range across experiments: "
              f"{min(stds):.2f}–{max(stds):.2f} (mean {statistics.mean(stds):.2f})"]
    text = "\n".join(lines)
    (ROOT / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
