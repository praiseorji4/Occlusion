"""
src/plotting.py — All required paper figures.

Every function saves outputs as both .png and .pdf to results/figures/.
Figures are generated from results/tables/val_results_all_models.csv and
results/tables/test_results_FINAL.csv — never from in-memory dicts alone,
so every plot is reproducible from the saved CSV files.

Required plots:
  1. plot_occlusion_performance_curve  — per-difficulty AP across all model variants
  2. plot_augmentation_strength_sweep  — p vs AP_easy/mod/hard/mAP@0.5 (4 lines)
  3. plot_ablation_bar_chart           — ΔAP_hard when each M8 component removed
  4. plot_results_table                — LaTeX + CSV full results table
  5. plot_depth_map_comparison         — 4-panel RGB|depth|occluded depth|mask
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for headless servers)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Figure saving helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, save_path: Path) -> None:
    """Save figure as both PNG (150 dpi) and PDF (vector)."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path.with_suffix(".png")), dpi=150, bbox_inches="tight")
    fig.savefig(str(save_path.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Occlusion performance curve
# ---------------------------------------------------------------------------

def plot_occlusion_performance_curve(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Path,
    include_m0_hard: bool = True,
) -> None:
    """
    Per-difficulty AP for all model variants on one figure.

    Serves: visual summary of how each successive component reduces the
    Easy→Moderate→Hard performance drop.

    Args:
        results_dict: {model_id: {'AP_easy': float, 'AP_mod': float, 'AP_hard': float}}
                      Model IDs to plot: M0, M2, M4, M6, M7, M8 (+ M0_hard dashed).
        save_path:    Path without extension; .png and .pdf are appended.
        include_m0_hard: If True, plot M0_hard as a dashed line.

    Layout:
        x-axis: Difficulty tier (Easy, Moderate, Hard)
        y-axis: AP (%)
        One line per model variant.
        M0_hard plotted as dashed.
        Text annotations on M0 line: "ΔX.X AP: [mechanism]"
    """
    variants = ["M0", "M2", "M4", "M6", "M7", "M8"]
    colours = {
        "M0": "#4e79a7",
        "M2": "#f28e2b",
        "M4": "#e15759",
        "M6": "#76b7b2",
        "M7": "#59a14f",
        "M8": "#edc948",
        "M0_hard": "#b07aa1",
    }
    tiers = ["Easy", "Moderate", "Hard"]
    keys  = ["AP_easy", "AP_mod", "AP_hard"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_id in variants:
        if model_id not in results_dict:
            continue
        r = results_dict[model_id]
        aps = [r.get(k, 0.0) for k in keys]
        ax.plot(tiers, aps, marker="o", label=model_id,
                color=colours.get(model_id, "grey"), linewidth=2)

    if include_m0_hard and "M0_hard" in results_dict:
        r = results_dict["M0_hard"]
        aps = [r.get(k, 0.0) for k in keys]
        ax.plot(tiers, aps, marker="s", linestyle="--",
                label="M0_hard", color=colours["M0_hard"], linewidth=2)

    # Annotate Easy→Mod and Mod→Hard drops on M0
    if "M0" in results_dict:
        r = results_dict["M0"]
        easy, mod, hard = r.get("AP_easy", 0), r.get("AP_mod", 0), r.get("AP_hard", 0)
        d1 = easy - mod
        d2 = mod  - hard
        ax.annotate(
            f"Δ{d1:.1f} AP\n(height & truncation filter)",
            xy=("Moderate", mod), xytext=(0.32, mod + 3),
            textcoords=("axes fraction", "data"),
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )
        ax.annotate(
            f"Δ{d2:.1f} AP\n(occlusion lvl ≤2 filter)",
            xy=("Hard", hard), xytext=(0.68, hard + 3),
            textcoords=("axes fraction", "data"),
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", lw=0.8),
        )

    ax.set_xlabel("Difficulty Tier", fontsize=12)
    ax.set_ylabel("AP (%)", fontsize=12)
    ax.set_title("Per-Difficulty AP — All Model Variants", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Augmentation strength sweep
# ---------------------------------------------------------------------------

def plot_augmentation_strength_sweep(
    aug_results: Dict[float, Dict[str, float]],
    optimal_p: float,
    save_path: Path,
) -> None:
    """
    Augmentation probability p vs AP at all difficulty tiers + mAP@0.5.

    Correction 6 requirement: plot 4 separate lines so the tradeoff is visible.
    If AP_hard ↑ while AP_easy ↓ at a given p → acceptable (intended effect).
    If all tiers ↓ → augmentation too aggressive.

    Args:
        aug_results: {p_value: {'AP_easy': float, 'AP_mod': float,
                                'AP_hard': float, 'mAP_50': float}}
        optimal_p:   The selected p value (marked with vertical dashed line).
        save_path:   Path without extension.
    """
    ps = sorted(aug_results.keys())

    easy  = [aug_results[p].get("AP_easy",  0.0) for p in ps]
    mod   = [aug_results[p].get("AP_mod",   0.0) for p in ps]
    hard  = [aug_results[p].get("AP_hard",  0.0) for p in ps]
    map50 = [aug_results[p].get("mAP_50",   0.0) for p in ps]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(ps, hard, marker="o", color="#e15759", linewidth=2.5,
             label="AP_hard (primary)")
    ax1.plot(ps, mod,  marker="^", color="#4e79a7", linewidth=1.8,
             linestyle="dotted", label="AP_moderate")
    ax1.plot(ps, easy, marker="s", color="#59a14f", linewidth=1.8,
             linestyle="dashed", label="AP_easy")

    ax2.plot(ps, map50, marker="D", color="#b07aa1", linewidth=1.8,
             linestyle=(0, (3, 1, 1, 1)), label="mAP@0.5")

    # Vertical line at optimal p
    ax1.axvline(optimal_p, color="black", linestyle="--", linewidth=1.2)

    # Find the AP_easy drop at optimal_p vs the next lower p for annotation
    p_list = sorted(ps)
    if len(p_list) >= 2:
        opt_idx = p_list.index(optimal_p) if optimal_p in p_list else -1
        if opt_idx >= 0:
            ap_easy_at_opt  = aug_results[optimal_p].get("AP_easy",  0.0)
            ap_hard_at_opt  = aug_results[optimal_p].get("AP_hard",  0.0)
            baseline_easy   = aug_results[p_list[0]].get("AP_easy",  0.0)
            easy_drop       = baseline_easy - ap_easy_at_opt
            ax1.text(
                optimal_p + 0.01, ap_hard_at_opt - 4,
                f"Selected p={optimal_p:.1f}\nHighest AP_hard\n"
                f"AP_easy drop: {easy_drop:.1f} pts",
                fontsize=8, ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.9),
            )

    ax1.set_xlabel("Augmentation Probability p", fontsize=12)
    ax1.set_ylabel("AP (%)", fontsize=12, color="black")
    ax2.set_ylabel("mAP@0.5 (%)", fontsize=12, color="#b07aa1")
    ax2.tick_params(axis="y", labelcolor="#b07aa1")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower left")

    ax1.set_title(
        "Augmentation Strength vs AP — Label-Aware Augmentation",
        fontsize=12, fontweight="bold"
    )
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticks(ps)

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation_bar_chart(
    ablation_results: Dict[str, float],
    save_path: Path,
) -> None:
    """
    Horizontal bar chart: ΔAP_hard when each component is removed from M8.

    Colour-coded by category:
      depth=teal, augmentation=orange, architecture=blue.
    Sorted from largest drop to smallest.

    Args:
        ablation_results: {component_name: delta_AP_hard}
                          delta is negative (AP drops when component removed).
                          Example: {'CrossAttnFusion': -3.2, 'LabelAugment': -2.1, ...}
        save_path:        Path without extension.
    """
    # Category mapping
    depth_components = {"CrossAttnFusion", "EarlyDepth", "LateDepth", "DepthConf"}
    aug_components   = {"LabelAugment", "RealOccluder", "GridMask", "Cutout", "Mosaic"}
    arch_components  = {"FEM", "POPAM", "GhostCSP", "VisHead"}

    def get_colour(name: str) -> str:
        if name in depth_components:
            return "#2ca02c"   # teal-green
        elif name in aug_components:
            return "#ff7f0e"   # orange
        elif name in arch_components:
            return "#1f77b4"   # blue
        return "#7f7f7f"

    # Sort by absolute drop (largest first)
    sorted_items = sorted(ablation_results.items(), key=lambda x: x[1])  # most negative first
    names  = [k for k, _ in sorted_items]
    deltas = [v for _, v in sorted_items]
    colours = [get_colour(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.55)))

    bars = ax.barh(names, deltas, color=colours, height=0.6, edgecolor="white")

    # Value labels
    for bar, val in zip(bars, deltas):
        ax.text(
            val - 0.1, bar.get_y() + bar.get_height() / 2,
            f"{val:+.1f}",
            ha="right", va="center", fontsize=9, fontweight="bold",
        )

    # Legend patches
    legend_patches = [
        mpatches.Patch(color="#2ca02c", label="Depth"),
        mpatches.Patch(color="#ff7f0e", label="Augmentation"),
        mpatches.Patch(color="#1f77b4", label="Architecture"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔAP_hard (%)", fontsize=12)
    ax.set_title(
        "Component Contribution: ΔAP_hard (Removed from FullSystem)",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()  # largest drop at top

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Results table (LaTeX + CSV)
# ---------------------------------------------------------------------------

def plot_results_table(
    all_results: List[Dict],
    save_path: Path,
) -> pd.DataFrame:
    """
    Generate a full results table and export as CSV + LaTeX.

    Columns: Model | AP_easy | AP_mod | AP_hard | ORS | FPS | Notes
    Bold the best value per numeric column.

    Args:
        all_results: List of dicts, one per model:
          {
            'Model':   str,
            'AP_easy': float,
            'AP_mod':  float,
            'AP_hard': float,
            'ORS':     float,
            'FPS':     float,
            'Notes':   str   (optional)
          }
        save_path: Path without extension; .csv and .tex are appended.

    Returns:
        pd.DataFrame of the results table.
    """
    columns = ["Model", "AP_easy", "AP_mod", "AP_hard", "ORS", "FPS", "Notes"]
    df = pd.DataFrame(all_results)

    # Ensure all columns exist
    for col in columns:
        if col not in df.columns:
            df[col] = "" if col in ("Model", "Notes") else 0.0

    df = df[columns]

    # Save CSV
    csv_path = save_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    # Generate LaTeX with bolded best per numeric column
    numeric_cols = ["AP_easy", "AP_mod", "AP_hard", "ORS", "FPS"]
    best_vals = {col: df[col].max() for col in numeric_cols}

    def _fmt(val, col: str) -> str:
        if not isinstance(val, (int, float)):
            return str(val)
        s = f"{val:.1f}"
        if col in best_vals and abs(val - best_vals[col]) < 1e-4:
            s = f"\\textbf{{{s}}}"
        return s

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{KITTI Pedestrian Detection Results}",
        "\\label{tab:results}",
        "\\begin{tabular}{lcccccl}",
        "\\toprule",
        "Model & AP$_{easy}$ & AP$_{mod}$ & AP$_{hard}$ & ORS & FPS & Notes \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [str(row["Model"])]
        for col in numeric_cols:
            cells.append(_fmt(row[col], col))
        cells.append(str(row.get("Notes", "")))
        lines.append(" & ".join(cells) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    latex_str = "\n".join(lines)

    tex_path = save_path.with_suffix(".tex")
    with open(tex_path, "w") as f:
        f.write(latex_str)

    return df


# ---------------------------------------------------------------------------
# 5. Depth map comparison (4-panel)
# ---------------------------------------------------------------------------

def plot_depth_map_comparison(
    rgb: torch.Tensor,
    depth_clean: torch.Tensor,
    depth_occluded: torch.Tensor,
    mask: torch.Tensor,
    save_path: Path,
) -> None:
    """
    4-panel figure: RGB | Clean depth | Occluded depth | Mask overlay.

    Used during augmentation testing to verify depth consistency is applied
    correctly.  Call this for 20 examples and save to
    results/figures/depth_consistency_check/.

    Args:
        rgb:            (3, H, W) float32 ImageNet-normalised tensor.
        depth_clean:    (1, H, W) float32 depth before augmentation.
        depth_occluded: (1, H, W) float32 depth after augmentation (masked region = 0.5).
        mask:           (H, W) bool tensor; True = pixels that were masked.
        save_path:      Full path for the saved figure.
    """
    import matplotlib.gridspec as gridspec

    # De-normalise RGB for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    rgb_disp = (rgb * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    depth_clean_np    = depth_clean.squeeze(0).numpy()
    depth_occluded_np = depth_occluded.squeeze(0).numpy()
    mask_np           = mask.numpy().astype(np.float32)

    fig = plt.figure(figsize=(14, 4))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)

    def add_panel(idx: int, img, title: str, cmap=None, vmin=None, vmax=None):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")

    add_panel(0, rgb_disp,         "RGB")
    add_panel(1, depth_clean_np,   "Clean Depth",    cmap="inferno", vmin=0, vmax=1)
    add_panel(2, depth_occluded_np,"Occluded Depth", cmap="inferno", vmin=0, vmax=1)

    # Mask overlay: RGB with red tint on masked region
    overlay = rgb_disp.copy()
    overlay[mask_np.astype(bool), 0] = 1.0  # red channel
    overlay[mask_np.astype(bool), 1] = 0.0
    overlay[mask_np.astype(bool), 2] = 0.0
    add_panel(3, overlay.clip(0, 1), "Mask Overlay")

    # Add text confirming the fill value
    n_masked = int(mask_np.sum())
    fig.text(
        0.5, -0.02,
        f"Masked pixels: {n_masked} | depth fill=0.5 (unknown) | conf fill=0.0",
        ha="center", fontsize=9, color="grey",
    )

    _save(fig, save_path)


# ---------------------------------------------------------------------------
# Narrative template generator
# ---------------------------------------------------------------------------

def create_narrative_template(model_id: str, output_dir: Path) -> Path:
    """
    Create a blank narrative .md file for a model variant.

    The narrative becomes the paper's ablation section when filled in.
    Fill in AP values and mechanism explanations after each training run.

    Args:
        model_id:   e.g. 'M0', 'M2', 'M8'.
        output_dir: results/narratives/

    Returns:
        Path to the created .md file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_id}.md"

    if path.exists():
        return path  # don't overwrite existing narratives

    content = f"""# {model_id} — [Variant Name]

## AP_hard: [FILL IN after training]

## Per-difficulty results
| Tier     | AP (%) |
|----------|--------|
| Easy     | [FILL] |
| Moderate | [FILL] |
| Hard     | [FILL] |

## Why AP drops Easy → Moderate → Hard
AP drops from Easy ([X]) to Moderate ([Y]) because [mechanism: e.g. smaller
instances enter the evaluation pool at ≥25px, increasing the detection difficulty].

AP drops further from Moderate ([Y]) to Hard ([Z]) because [mechanism: e.g.
occlusion_lvl=2 instances account for most of the Hard-exclusive annotations,
and the RGB-only model has no depth cue to distinguish overlapping pedestrians].

## What this model cannot do
[Specific limitation tied to the failure mode — not generic language.
Example: "Cannot distinguish a largely-occluded pedestrian from a background
region at occlusion level 2, because both produce similarly sparse RGB texture."]

## What the next variant is expected to fix
[Specific prediction tied to the change being made in the next variant.
Example: "M3 adds FEM + POPAM to the backbone. FEM's dilated branches should
capture the partial-edge features that correlate with occluded limbs at dilation=3."]

## What actually happened after adding the next variant
[Fill in after the next variant is trained. Was the prediction correct?
If not, explain why — this is a first-class result, not a failure to hide.]

## Negative results for this variant
[If any planned negative result was confirmed here (e.g. HideAndSeek hurts
small objects), document: the metric delta, the mechanism, and whether the
prediction was correct.]
"""
    path.write_text(content)
    return path
