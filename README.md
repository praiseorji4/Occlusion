# Occlusion-Robust Pedestrian Detection

A reproducible, config-driven codebase for occlusion-robust pedestrian detection on
KITTI and CityPersons. It unifies two contributions into one pipeline:

1. **Proven augmentation + RGB-D late-fusion ablations** — label-aware occlusion
   augmentation (Cutout, RandomErasing, HideAndSeek, GridCut, combined) and
   feature-level RGB-D late fusion (parallel 1-channel depth backbone fused at
   P3/P4/P5 with 1×1 convs).
2. **Novel modules** — FEM (Feature Enhancement Module), POPAM (dual-pooling
   attention), CrossAttentionFusion (depth-gated P4 cross-attention with
   confidence masking), and an auxiliary VisibilityMapHead.

**Model scope:** YOLOv8x and YOLOv8n only. **Datasets:** KITTI + CityPersons.
**Primary metric:** KITTI 40-point interpolated AP on the Hard tier (`AP_hard`).

## Structure

```
occlusion/                # installable package
  config.py               # RunConfig + YAML _base_ inheritance (drives everything)
  seeds.py                # deterministic seeding
  paths.py                # env-var path resolution (no hardcoded home dirs)
  metrics.py              # KITTI 40-pt AP, ORS, FN-rate, FPS
  logging.py / figures.py # W&B+CSV logging, paper figures from CSV
  data/                   # KITTI/CP parsers, MiDaS depth, occluder bank, transforms, dataset
  models/                 # FEM/POPAM, fusion (late_feature/early/cross_attention), build, trainer
configs/                  # base + dataset/ + model/ + experiment/*.yaml
train.py  eval.py  preprocess.py
tests/                    # pytest: config, metrics, split, depth-consistency
data/split_verification.py
notebooks/                # analysis / paper figures only (not the training path)
```

## Setup

```bash
pip install -e .
# pip install -r requirements.txt  # pinned alternative
```

Python ≥ 3.10. Pinned dependencies (notably `ultralytics==8.3.39`) are in
`requirements.txt`. The box-format fix and the `_predict_once` overrides depend
on Ultralytics internals; treat version upgrades as breaking.

## Data preparation

Point the code at your data with env vars (defaults: `./data`, `./results`):

```bash
export OCCLUSION_DATA_ROOT=/path/to/data
export OCCLUSION_OUTPUT_ROOT=/path/to/results
```

Place raw KITTI under `$OCCLUSION_DATA_ROOT/kitti/` (with
`data_object_image_2/` and `data_object_label_2/`) and CityPersons under
`$OCCLUSION_DATA_ROOT/citypersons/` (with `gtBbox_cityPersons_trainval/` and
`leftImg8bit_trainvaltest/`). Then:

```bash
python preprocess.py --dataset kitti
python preprocess.py --dataset citypersons --depth   # MiDaS depth maps (slow)
python data/split_verification.py                     # assert zero split overlap
```

This writes `{dataset}_yolo/{images,labels}/{train,val,test}/`, a
`{dataset}_yolo.yaml`, per-label `.occ.npy` occlusion sidecars (for label-aware
augmentation), and `{dataset}_depth/` when `--depth` is set.

## Training

```bash
# Baseline (RGB only, no augmentation)
python train.py --config configs/experiment/m0_kitti.yaml --seed 42

# Label-aware Cutout augmentation
python train.py --config configs/experiment/m2_kitti_cutout.yaml --seed 42

# RGB-D late fusion (proven archive path)
python train.py --config configs/experiment/latefusion_kitti.yaml --seed 42

# Novel full system (FEM + POPAM + cross-attention + visibility head)
python train.py --config configs/experiment/full_system_kitti.yaml --seed 42
```

Multi-seed (reproducibility gate): repeat with `--seed 123` and `--seed 456`.
Across the completed runs, per-seed std on AP_hard is 0.16–1.69 AP (mean 0.78) —
report mean ± std rather than a single seed. Override any knob on the CLI:
`--epochs`, `--batch`, `--device`, `--data_limit N` (subsamples training).

Smoke test (no real data needed beyond the preprocessing step):

```bash
python train.py --config configs/experiment/m0_kitti.yaml --epochs 1 --data_limit 64
```

## Evaluation

```bash
python eval.py --checkpoint $OCCLUSION_OUTPUT_ROOT/runs/<run>/weights/best.pt \
               --config configs/experiment/m2_kitti_cutout.yaml
```

Writes `results/tables/eval_<run>.csv` (mAP50, mAP50-95, precision, recall).
For the KITTI 40-point tiered AP (`AP_easy/mod/hard`) use `occlusion.metrics`
directly with predictions + the rich KITTI annotations.

## Results

All numbers are Pedestrian mAP@0.5 (%) on the KITTI Easy / Moderate / Hard
occlusion tiers; `MR_hard` is the miss-rate on the Hard tier. Every row is the
mean ± std over three seeds (42 / 123 / 456), test split. Per-seed values and
bootstrap CIs are in `results/expected/` (raw `metrics.json`) and
`results/expected/aggregate.csv`; regenerate the tables with
`python scripts/aggregate_results.py`.

### KITTI — test split (YOLOv8x)

| Config | AP_easy | AP_mod | AP_hard | mAP50 | MR_hard | FPS |
|---|---|---|---|---|---|---|
| baseline (no aug) | 95.45±0.52 | 76.11±1.55 | 46.21±1.69 | 88.09±0.51 | 42.68±0.68 | 22.55±3.13 |
| cutout | 95.46±0.27 | 77.81±0.79 | 48.64±0.97 | 88.86±0.24 | 40.06±0.41 | 24.76±0.05 |
| random_erasing | 95.79±0.16 | 75.92±0.55 | 48.71±1.06 | 88.62±0.21 | 40.46±0.99 | 24.86±0.02 |
| hide_and_seek | 95.46±0.84 | 77.63±0.78 | 48.92±0.64 | 88.30±0.43 | 41.73±0.46 | 24.67±0.09 |
| gridcut | 95.62±0.28 | 76.98±1.43 | 50.98±0.26 | 89.03±0.34 | 39.00±1.64 | 24.71±0.11 |
| combined_best_two | 96.31±0.43 | 75.79±0.89 | 47.93±1.50 | 88.73±0.60 | 42.05±2.08 | 24.93±0.02 |
| combined_all | 95.06±0.49 | 76.33±1.65 | 47.68±0.25 | 88.39±0.41 | 41.17±0.16 | 24.80±0.03 |

### CityPersons — test split (YOLOv8x)

| Config | AP_easy | AP_mod | AP_hard | mAP50 | MR_hard | FPS |
|---|---|---|---|---|---|---|
| baseline (no aug) | 79.31±0.22 | 70.64±0.34 | 39.40±0.28 | 71.54±0.19 | 77.34±0.18 | 8.26±1.43 |
| cutout | 78.97±0.35 | 71.00±0.31 | 39.52±0.16 | 71.57±0.18 | 77.13±0.15 | 9.27±0.01 |
| random_erasing | 78.54±0.23 | 69.88±0.23 | 38.00±0.85 | 71.08±0.35 | 78.86±0.44 | 9.27±0.01 |
| hide_and_seek | 79.84±0.01 | 70.97±0.49 | 39.00±0.87 | 71.66±0.22 | 77.56±0.90 | 9.23±0.02 |
| gridcut | 79.30±0.14 | 70.66±0.08 | 38.95±0.77 | 71.41±0.01 | 77.80±0.77 | 9.25±0.03 |
| combined_best_two | 79.29±0.32 | 69.79±0.38 | 38.11±0.58 | 71.03±0.32 | 78.39±0.56 | 9.27±0.02 |
| combined_all | 79.08±0.23 | 70.03±0.60 | 37.49±1.20 | 70.87±0.45 | 78.77±0.92 | 9.28±0.02 |
| body_part_occlusion | 79.51±0.39 | 71.26±0.31 | 38.27±0.58 | 71.57±0.40 | 78.30±0.37 | 8.97±0.39 |

**Pending (will be added as runs complete):** RGB-D `late_feature` / `early`
fusion, FEM + POPAM, `cross_attention` fusion, the auxiliary visibility head,
and the YOLOv8n rows. These configs build and forward-pass; training is the
remaining step (see Notes).

**Reproducibility:** per-seed std on AP_hard (test) is 0.16–1.69 AP (mean
0.78). The KITTI baseline row is the noisiest (seed 42 lands at 43.84 vs 47.12
/ 47.67); treat single-seed numbers as ±1–2 AP. Val-split tables are in
`results/expected/summary.md`.

## Reproducibility

- Pinned dependencies; deterministic seeding (`occlusion.seeds`).
- No hardcoded paths — all roots via `OCCLUSION_DATA_ROOT` / `OCCLUSION_OUTPUT_ROOT`.
- `pytest -q` runs config, metrics, split, and depth-consistency checks.
- Configs use `_base_` inheritance; every experiment config resolves to a
  concrete `build_model` plan (see `tests/test_config.py`).

## Notes / known scope

- The `cross_attention` fusion and the visibility head have not been trained
  before (only the `late_feature` path has). They build and forward-pass; full
  training is the expected debugging cycle. The visibility-head loss is not yet
  wired into the Ultralytics loss flow.
- YOLOv12x is out of scope and not ported.
- `notebooks/` are analysis/figure artifacts. They import from
  `occlusion.legacy`, a compatibility shim that re-exports the analysis modules
  (datasets, augmentation, plotting, logger) alongside the package's metrics,
  fusion, and depth symbols. The canonical training path is `train.py`.

## License

MIT (see `LICENSE`).
