"""Paper figures, read from saved CSVs only."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_ablation_bar_chart(csv_path: Path, out_path: Path, label_col: str,
                            value_col: str, title: str = "") -> None:
    rows = read_csv(csv_path)
    labels = [r[label_col] for r in rows]
    values = [float(r[value_col]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylabel(value_col)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    _save(fig, out_path)


def plot_occlusion_performance_curve(csv_path: Path, out_path: Path,
                                     coverage_col: str = "coverage",
                                     ap_col: str = "AP_hard") -> None:
    rows = read_csv(csv_path)
    cov = [float(r[coverage_col]) for r in rows]
    ap = [float(r[ap_col]) for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cov, ap, marker="o")
    ax.set_xlabel("Occlusion coverage")
    ax.set_ylabel(ap_col)
    ax.set_ylim(0, 100)
    _save(fig, out_path)


def plot_results_table(csv_path: Path, out_path: Path, title: str = "Results") -> None:
    rows = read_csv(csv_path)
    if not rows:
        return
    cols = list(rows[0].keys())
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(cols)), 0.5 * (len(rows) + 1)))
    ax.axis("off")
    table = ax.table(cellText=[[r[c] for c in cols] for r in rows],
                     colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax.set_title(title)
    _save(fig, out_path)
