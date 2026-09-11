# original_nav2_params.yaml

## Purpose

This is a historical "before" snapshot of the Nav2 configuration, preserved to document the parameter values that existed prior to the tuning fixes applied in `nav2_params.yaml`. It is **not used by any current launch file** and should not be loaded on the real robot or in simulation. Its sole purpose is as a reference for understanding what changed and why.

> Do not confuse this file with `nav2_params.yaml` (the live real-robot config) or `sim_nav2_params.yaml` (the simulation config). Neither launch file references `original_nav2_params.yaml`.

## What is different from the current config

The following table shows all confirmed parameter differences between `original_nav2_params.yaml` and `nav2_params.yaml`. Values were read directly from both files.

| Parameter | original_nav2_params.yaml | nav2_params.yaml (current) | Change rationale |
|---|---|---|---|
| `FollowPath.max_vel_x` | `0.5` m/s | `0.15` m/s | Major safety reduction — robot was moving at 3× the current cautious speed |
| `FollowPath.max_vel_theta` | `1.0` rad/s | `0.6` rad/s | Reduced turn rate for stability |
| `FollowPath.max_speed_xy` | `0.5` m/s | `0.15` m/s | Matches `max_vel_x` |
| `FollowPath.BaseObstacle.scale` | `0.02` | `0.5` | Was ~1600× weaker than path critics; obstacles were nearly ignored |
| `local_costmap.inflation_radius` | `0.45` m | `0.30` m | Was larger than robot radius — but the inline comment in current config says `was: 0.05`, meaning the actual historical value at time of the fix was `0.05`, not `0.45`. The `original_nav2_params.yaml` file itself shows `0.45`. |
| `global_costmap.inflation_radius` | `0.5` m | `0.30` m | Decreased; real fix comment says `was: 0.2` — the original file shows `0.5` |
| `planner_server.GridBased.tolerance` | `0.5` m | `0.25` m | Robot stopped too far from goal |
| `planner_server.GridBased.use_astar` | `false` (Dijkstra) | `true` (A*) | Faster with same path quality |
| `velocity_smoother.feedback` | `"OPEN_LOOP"` | `"CLOSED_LOOP"` | Now uses actual odometry for smoother acceleration |
| `velocity_smoother.max_velocity[0]` | `0.5` m/s | `0.15` m/s | Matches reduced `max_vel_x` |
| `velocity_smoother.max_velocity[2]` | `1.0` rad/s | `0.6` rad/s | Matches reduced `max_vel_theta` |

> **Note on inflation radius discrepancy:** The `# was: ...` inline comments in `nav2_params.yaml` state the previous local inflation was `0.05` and global was `0.2`. The `original_nav2_params.yaml` file shows `0.45` (local) and `0.5` (global). This suggests the file was edited at an intermediate stage — the `original_nav2_params.yaml` may not represent the exact state at the moment the inline-comment fixes were made. Use the inline comments in `nav2_params.yaml` as the authoritative "before" values for those two parameters.

## Parameters that are identical

The following servers and parameters are unchanged between the original and current config:

- `bt_navigator`: all parameters identical
- `controller_server.controller_frequency`: `20.0` Hz
- `controller_server.progress_checker`: `required_movement_radius: 0.5`, `movement_time_allowance: 10.0`
- `controller_server.general_goal_checker`: `xy_goal_tolerance: 0.15`, `yaw_goal_tolerance: 0.15`
- DWB critic list and all scales except `BaseObstacle`
- `smoother_server`: identical
- `behavior_server`: identical
- `collision_monitor`: identical
- `map_saver`: identical
- `docking_server`: identical

## Detailed parameter listing (original values)

### controller_server / FollowPath

| Parameter | Original value |
|---|---|
| `max_vel_x` | `0.5` m/s |
| `max_vel_theta` | `1.0` rad/s |
| `max_speed_xy` | `0.5` m/s |
| `BaseObstacle.scale` | `0.02` |

All other DWB params (sim_time, critic scales, vx_samples, etc.) are identical to current config.

### local_costmap

| Parameter | Original value |
|---|---|
| `inflation_layer.inflation_radius` | `0.45` m |
| `inflation_layer.cost_scaling_factor` | `3.0` |

All other local_costmap parameters are identical to current config.

### global_costmap

| Parameter | Original value |
|---|---|
| `inflation_layer.inflation_radius` | `0.5` m |
| `inflation_layer.cost_scaling_factor` | `3.0` (original) vs `1.5` (current) |

### planner_server

| Parameter | Original value |
|---|---|
| `GridBased.tolerance` | `0.5` m |
| `GridBased.use_astar` | `false` |

### velocity_smoother

| Parameter | Original value |
|---|---|
| `feedback` | `"OPEN_LOOP"` |
| `max_velocity` | `[0.5, 0.0, 1.0]` |

Note that `original_nav2_params.yaml` does **not** have an `odom_topic` field in `velocity_smoother` — this field was added in the current config when feedback was switched to `CLOSED_LOOP`.

## Usage

This file is **not loaded by any launch file**. It exists purely as documentation of the pre-fix state. Do not add it to any launch configuration.

## Notes / Known issues

- The `controller_server` section in `original_nav2_params.yaml` also lacks the `odom_topic` field that exists in `nav2_params.yaml`. This field was added when the current config was created.
- The `max_vel_x: 0.5` original value represents a significant safety risk for the real robot — at that speed, the robot covers 50 cm per second with the original weak obstacle avoidance (`BaseObstacle.scale: 0.02`). The current `max_vel_x: 0.15` with `BaseObstacle.scale: 0.5` is a much safer combination.

## See Also

- [`nav2_params.yaml`](nav2_params.md) — Current real-robot Nav2 configuration
- [`sim_nav2_params.yaml`](sim_nav2_params.md) — Simulation config (partially carries original values for some parameters)
