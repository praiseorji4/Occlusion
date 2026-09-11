# sim_nav2_params.yaml

## Purpose

Nav2 stack configuration for the **Gazebo simulation**. It is structurally identical to `nav2_params.yaml` but adapted for simulation: `use_sim_time: True` throughout, a different odometry topic, and different costmap inflation radii tuned for the simulation environment. This file does **not** contain the `# was: ...` resolved-issue comments present in the real robot config — it appears to be a partially-updated copy reflecting different tuning choices.

## Key differences from `nav2_params.yaml` (real robot)

| Parameter | Real robot | Simulation | Notes |
|---|---|---|---|
| `use_sim_time` | `False` (all servers) | `True` (all servers) | Must use Gazebo `/clock` |
| `odom_topic` (bt_navigator) | `/diff_drive_controller/odom` | `/odometry/filtered` | Sim uses EKF output |
| `odom_topic` (controller_server) | `/diff_drive_controller/odom` | `/odometry/filtered` | Sim uses EKF output |
| `local inflation_radius` | `0.30` m | `0.55` m | Larger buffer in sim for stability |
| `global inflation_layer.cost_scaling_factor` | `1.5` | `3.0` | Sharper global costmap in sim |
| `global inflation_radius` | `0.30` m | `0.2` m | Smaller than real robot |
| `velocity_smoother.feedback` | `"CLOSED_LOOP"` | `"OPEN_LOOP"` | Sim uses open-loop smoothing |
| `velocity_smoother.odom_topic` | `/diff_drive_controller/odom` | `/odometry/filtered` | |
| `velocity_smoother.max_velocity[0]` | `0.15` m/s | `0.15` m/s | Same |
| `planner.tolerance` | `0.25` m | `0.5` m | Looser goal tolerance in sim |
| `planner.use_astar` | `true` | `false` (Dijkstra) | Sim keeps original planner |
| `FollowPath.BaseObstacle.scale` | `0.5` | `0.02` | Real robot fix NOT applied to sim |

> **Important:** The `BaseObstacle.scale` fix (0.02 → 0.5) and the planner improvements (`tolerance`, `use_astar`) have **not** been applied to `sim_nav2_params.yaml`. The simulation config retains the older pre-fix values. If the same navigation quality improvements are desired in simulation, these parameters should be updated.

## Parameters

### bt_navigator

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `True` | |
| `global_frame` | `map` | |
| `robot_base_frame` | `base_footprint` | |
| `odom_topic` | `/odometry/filtered` | EKF-fused odometry output from `ekf_filter_node` running with `sim_ekf.yaml`. Not raw diff_drive_controller output. |
| `bt_loop_duration` | `10` ms | |
| `filter_duration` | `0.3` s | |
| `default_server_timeout` | `20` ms | |
| `wait_for_service_timeout` | `1000` ms | |
| `introspection_mode` | `"disabled"` | |
| `navigators` | `["navigate_to_pose", "navigate_through_poses"]` | Same as real robot. |

---

### controller_server

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `True` | |
| `controller_frequency` | `20.0` Hz | |
| `odom_topic` | `/odometry/filtered` | EKF-fused odometry. |
| `failure_tolerance` | `0.3` s | |

#### FollowPath (DWBLocalPlanner) — sim-specific values

| Parameter | Value | Notes |
|---|---|---|
| `max_vel_x` | `0.15` m/s | Same as real robot. |
| `max_vel_theta` | `0.6` rad/s | Same. |
| `sim_time` | `1.7` s | Same. |
| `vx_samples` | `20` | Same. |
| `vtheta_samples` | `20` | Same. |
| `xy_goal_tolerance` | `0.15` m | Same. |
| `BaseObstacle.scale` | `0.02` | **Not fixed** — retains original weak obstacle avoidance weight. Real robot uses `0.5`. |
| `PathAlign.scale` | `32.0` | Same as real. |
| `GoalAlign.scale` | `24.0` | Same as real. |
| `PathDist.scale` | `32.0` | Same as real. |
| `GoalDist.scale` | `24.0` | Same as real. |
| `RotateToGoal.scale` | `32.0` | Same as real. |

---

### local_costmap

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `True` | |
| `update_frequency` | `5.0` Hz | |
| `publish_frequency` | `2.0` Hz | |
| `global_frame` | `odom` | |
| `rolling_window` | `true` | |
| `width` | `3` m | |
| `height` | `3` m | |
| `resolution` | `0.05` m/cell | |
| `robot_radius` | `0.15` m | |
| `plugins` | `["voxel_layer", "inflation_layer"]` | |

#### local_costmap — inflation_layer

| Parameter | Value | Description |
|---|---|---|
| `cost_scaling_factor` | `3.0` | |
| `inflation_radius` | `0.55` m | **Larger than real robot (0.30 m).** In simulation, a larger inflation buffer compensates for sensor noise and Gazebo physics contact dynamics. |

#### local_costmap — voxel_layer (scan source)

Identical configuration to real robot: `/scan`, `LaserScan`, `raytrace_max_range: 3.0`, `obstacle_max_range: 2.5`.

---

### global_costmap

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `True` | |
| `update_frequency` | `1.0` Hz | |
| `global_frame` | `map` | |
| `robot_radius` | `0.15` m | |
| `resolution` | `0.05` m/cell | |
| `plugins` | `["static_layer", "obstacle_layer", "inflation_layer"]` | |

#### global_costmap — inflation_layer

| Parameter | Value | Description |
|---|---|---|
| `cost_scaling_factor` | `3.0` | Sharper than real robot's global costmap (`1.5`). |
| `inflation_radius` | `0.2` m | **Smaller than real robot (0.30 m).** In simulation the global planner can use tighter margins since the environment is deterministic. |

---

### planner_server (NavfnPlanner)

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `True` | |
| `GridBased.plugin` | `nav2_navfn_planner::NavfnPlanner` | |
| `GridBased.tolerance` | `0.5` m | **Retains original value** — real robot was fixed to `0.25` m but sim was not updated. |
| `GridBased.use_astar` | `false` | **Retains Dijkstra** — real robot was updated to A* but sim was not. |
| `GridBased.allow_unknown` | `true` | |

---

### smoother_server

Same as real robot: `SimpleSmoother`, `tolerance: 1.0e-10`, `max_its: 1000`, `do_refinement: True`.

---

### behavior_server

Same as real robot: all five plugins, same acceleration/speed limits, `local_frame: odom`, `global_frame: map`.

---

### velocity_smoother

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `True` | |
| `smoothing_frequency` | `20.0` Hz | |
| `feedback` | `"OPEN_LOOP"` | **Retains open-loop mode** — real robot was updated to `"CLOSED_LOOP"` but sim was not. |
| `max_velocity` | `[0.15, 0.0, 0.6]` | Same as real robot. |
| `min_velocity` | `[-0.5, 0.0, -1.0]` | |
| `max_accel` | `[2.5, 0.0, 3.2]` | |
| `max_decel` | `[-2.5, 0.0, -3.2]` | |
| `odom_topic` | `"/odometry/filtered"` | EKF output (note: OPEN_LOOP mode means this topic is monitored but not strictly required for the smoothing algorithm — it would be used if feedback mode were switched). |

---

### collision_monitor

Same configuration as real robot: `FootprintApproach`, `time_before_collision: 1.2`, `/scan` source with `min_height: 0.15`, `max_height: 2.0`.

---

### map_saver

Same as real robot: `free_thresh_default: 0.25`, `occupied_thresh_default: 0.65`.

---

### docking_server

Identical to real robot configuration with `use_sim_time: True`. Same informational note applies — no physical docking hardware.

---

## Running Nav2 in simulation

Two prerequisites must be running before Nav2 can receive valid odometry:

1. **`sim.launch.py`** — starts Gazebo with the robot URDF, spawns `sim_ubot_controllers.yaml` controllers, and publishes `/diff_drive_controller/odom`.
2. **`ekf_filter_node`** (using `sim_ekf.yaml`) — fuses `/diff_drive_controller/odom` and `/imu/data` into `/odometry/filtered` and publishes the `odom → base_footprint` TF.

If the EKF is not running, `bt_navigator` and `controller_server` will not receive valid odometry on `/odometry/filtered`, and Nav2 will fail.

## Usage

Loaded by the simulation Nav2 launch configuration. Not used by `real_robot.launch.py`.

## Notes / Known issues

- `BaseObstacle.scale: 0.02` — same weak obstacle avoidance as the pre-fix real robot config. If obstacle avoidance performance is poor in simulation, update to `0.5`.
- `planner.tolerance: 0.5` and `use_astar: false` — real robot improvements not back-ported to sim.
- `velocity_smoother.feedback: "OPEN_LOOP"` — sim does not use closed-loop smoothing. This may be acceptable in simulation where command tracking is ideal.
- Global inflation radius (0.2 m) is smaller than local (0.55 m), which is an unusual configuration. This appears to be an intentional tuning choice for simulation but the rationale is not documented in the YAML.

## See Also

- [`nav2_params.yaml`](nav2_params.md) — Real robot variant with resolved issues
- [`sim_ekf.yaml`](sim_ekf.md) — EKF that produces the `/odometry/filtered` topic this config depends on
- [`sim_ubot_controllers.yaml`](sim_ubot_controllers.md) — Controller config that produces `/diff_drive_controller/odom`
