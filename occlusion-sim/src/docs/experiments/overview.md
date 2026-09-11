# Experiments Overview

This page documents the experimental history of the ubot (Horizon) differential-drive robot,
including the physical environments that have been mapped with SLAM Toolbox, the evolution of
Nav2 parameters through real on-robot tuning, and the evidence of active development visible
in the workspace.

All maps are stored in the workspace root `/home/chibueze/uni-bot/` (one level above `src/`).

---

## Environments Mapped

Five distinct environments have been fully mapped and serialised with SLAM Toolbox Online Async.
Each produces four files: a PGM occupancy image and YAML metadata (used by the Nav2 map server),
plus a `.data` + `.posegraph` pair (used by SLAM Toolbox for serialisation and localisation
resumption).

| Environment | Files | Map metadata | Notes |
|---|---|---|---|
| `studio_1` | `studio_1_save.{pgm,yaml}` + `studio_1_serial.{data,posegraph}` | ⚠️ metadata not read — inspect `studio_1_save.yaml` directly | First saved environment |
| `studio_2` | `studio_2_save.{pgm,yaml}` + `studio_2_serial.{data,posegraph}` | ⚠️ metadata not read | Second environment |
| `studio_3` | `studio_3_save.{pgm,yaml}` + `studio_3_serial.{data,posegraph}` | resolution=0.050 m, origin=[-3.844, -2.408, 0], occupied_thresh=0.65, free_thresh=0.196 | **Current default** — hardcoded in `mapper_params_online_async.yaml` |
| `Tee_map` | `Tee_map_save.{pgm,yaml}` + `Tee_map_serial.{data,posegraph}` | ⚠️ metadata not read | T-shaped corridor or junction area |
| `defence_map` | `defence_map_save.{pgm,yaml}` + `defence_map_serial.{data,posegraph}` | ⚠️ metadata not read | Possibly mapped during a project demonstration or defence session |

### studio_3 map metadata (verified)

`studio_3_save.yaml` was read directly; the values below are confirmed ground truth:

```yaml
image: studio_3_save.pgm
mode: trinary
resolution: 0.050
origin: [-3.844, -2.408, 0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
```

The `map_file_name` in
`src/ubot/ubot_bringup/config/mapper_params_online_async.yaml` is hardcoded to:

```
/home/chibueze/uni-bot/studio_3_serial
```

(no file extension — SLAM Toolbox appends `.posegraph` automatically when loading.)
The SLAM Toolbox mode in that file is currently set to `mapping`; change it to `localization`
to resume navigation from this existing map without re-mapping.

---

## Development Timeline Evidence

TF tree snapshots captured by `ros2 run tf2_tools view_frames` accumulate in the workspace root
as `frames_<timestamp>.gv` / `frames_<timestamp>.pdf` pairs. The collection spans:

- **Earliest capture**: `frames_2026-04-17_*.gv`
- **Most recent capture**: `frames_2026-06-26_17.33.00.gv`

This represents roughly **two months of active development** visible through successive TF tree
snapshots. The most recent capture documents the current TF tree as described in
`architecture/tf_tree.md`. The five distinct saved maps, together with these dated captures,
confirm that full SLAM mapping has been achieved in multiple physical environments and that the
robot has been actively used for research throughout this period.

---

## Nav2 Tuning History

`nav2_params.yaml` contains inline changelog comments that record parameter changes made
during real robot runs. These comments are the authoritative record of tuning decisions;
the table below is derived directly from them.

| Parameter | Original value | Current value | Motivation |
|---|---|---|---|
| `BaseObstacle.scale` (DWB critic) | 0.02 | 0.5 | Was effectively ~1600x weaker than path critics, nearly ignored; robot was not adequately penalised for approaching obstacles |
| Local costmap `inflation_radius` | 0.05 m | 0.30 m | Was less than robot radius (0.15 m) — unsafe; robot would plan paths through lethal cells |
| Global costmap `inflation_radius` | 0.20 m | 0.30 m | Increased to match robot radius for consistent clearance margins |
| `planner_server` `tolerance` | 0.5 m | 0.25 m | Robot was stopping far from intended goals; coarser tolerance was acceptable on large open areas but unsuitable for confined spaces |
| `planner_server` `use_astar` | `false` (Dijkstra) | `true` (A*) | A* produces the same path quality as Dijkstra on these maps while being faster to compute |
| `velocity_smoother` `feedback` | `OPEN_LOOP` | `CLOSED_LOOP` | Actual odometry now feeds the smoother, giving smoother acceleration profiles in practice |
| `max_vel_x` | 0.5 m/s | 0.15 m/s | More conservative speed appropriate for indoor use; reduces risk of overshooting narrow spaces |

The pre-fix state of these parameters is preserved in
`src/ubot/ubot_bringup/config/original_nav2_params.yaml` for historical comparison.
That file is not referenced by any current launch file and should not be confused with the
live configuration.

---

## What Constitutes Successful Navigation

The workspace provides strong evidence that:

1. **Full SLAM mapping has been achieved in five environments.** Each map required the robot to
   traverse the environment while SLAM Toolbox Online Async built and loop-closed the occupancy
   grid in real time.

2. **Nav2 goal navigation has been demonstrated and iteratively improved.** The inline changelog
   comments in `nav2_params.yaml` record real failure modes (unsafe inflation radius, ignored
   obstacle critic, robot stopping short of goals) that were observed during actual runs and
   subsequently fixed.

3. **The system is actively being used for research.** Two months of TF tree snapshots and five
   maps from distinct physical locations confirm sustained operation, not a one-off demo.

---

## How to Switch Environments

### Use an existing map in localization mode

1. Open `src/ubot/ubot_bringup/config/mapper_params_online_async.yaml`.
2. Change `map_file_name` to point to the target environment's serial save:

   ```yaml
   map_file_name: /home/chibueze/uni-bot/<environment_name>_serial
   ```

3. Change `mode` from `mapping` to `localization`:

   ```yaml
   mode: localization
   ```

4. Relaunch the SLAM Toolbox node. It will load the `.posegraph` file and broadcast the
   `map` → `odom` transform from the saved pose graph.

### Map a new environment

Leave `mode: mapping` and update `map_file_name` to the desired output path for the new
serial save. After mapping, save via the SLAM Toolbox service (see `experiments/logging.md`).

---

## Data Not Yet Collected

The following data would strengthen quantitative claims about navigation performance but
have not been recorded to date. See `research/gaps.md` for the full list of research gaps.

- **ROS 2 bags of navigation runs** — no bag recordings of live navigation sessions have been
  identified in the workspace.
- **Quantitative path-tracking error** — no measurements of cross-track error or goal-reaching
  accuracy under controlled conditions.
- **Odometry repeatability data** — no repeated straight-line or rotation trials recorded for
  statistical analysis of wheel odometry drift.

These gaps are the primary targets for the logging protocol described in
`experiments/logging.md`.
