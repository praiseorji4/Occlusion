# Expected results

Per-run evaluation outputs for the augmentation ablations (YOLOv8x, KITTI and
CityPersons, seeds 42/123/456, val + test splits). Source: the evaluation
archive; reproduced here so the README tables are auditable without re-training.

- `runs/<dataset>/<split>/<experiment>/metrics.json` — per-class and per-tier
  (Easy/Medium/Hard) Pedestrian mAP@0.5, miss-rate, recall, bootstrap CIs.
- `predictions/<dataset>/<split>/<experiment>/inference_speed.json` — FPS.
- `aggregate.csv` — one row per run (dataset, split, experiment, seed, tiers).
- `summary.md` — mean ± std tables (regenerate with `scripts/aggregate_results.py`).

The RGB-D fusion, FEM/POPAM, cross-attention, visibility-head, and YOLOv8n runs
are not yet trained; their rows will be added here as they complete.
