# nav2_params.yaml

## Purpose

Nav2 stack configuration for the **real robot**. This is the primary navigation parameter file — 354 lines covering every Nav2 server. It has been actively tuned and contains five inline changelog comments (`# was: ...`) documenting resolved issues. All LiDAR observation sources reference `/scan` (the LD19 LiDAR topic).

## Resolved issues

The following parameters were changed from their original values. Each is flagged inline in the YAML with a `# was: ...` comment:

| Parameter | Original value | Current value | Reason (from inline comment) |
|---|---|---|---|
| `FollowPath.BaseObstacle.scale` | `0.02` | `0.5` | Was ~1600× weaker than path critics; obstacles were nearly ignored |
| `local_costmap.inflation_layer.inflation_radius` | `0.05` m | `0.30` m | Was less than robot radius (0.15 m) — unsafe |
| `global_costmap.inflation_layer.inflation_radius` | `0.2` m | `0.30` m | Increased to match robot radius |
| `planner_server.GridBased.tolerance` | `0.5` m | `0.25` m | Too coarse; robot stopped far from intended goal |
| `planner_server.GridBased.use_astar` | `false` (Dijkstra) | `true` (A*) | A* is faster than Dijkstra with same path quality |
| `velocity_smoother.feedback` | `"OPEN_LOOP"` | `"CLOSED_LOOP"` | Now uses actual odometry for smoother acceleration |

See [`original_nav2_params.yaml`](original_nav2_params.md) for the complete pre-fix snapshot.

## Parameters

### bt_navigator

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | Wall-clock time for real robot. |
| `global_frame` | `map` | Navigation operates in the map frame. |
| `robot_base_frame` | `base_footprint` | Robot's base frame for goal poses and footprint queries. |
| `odom_topic` | `/diff_drive_controller/odom` | Raw wheel odometry source. Used by the navigator for progress monitoring. |
| `bt_loop_duration` | `10` ms | Behavior tree tick rate. |
| `filter_duration` | `0.3` s | Duration of the velocity filter window. |
| `default_server_timeout` | `20` ms | Default action server timeout. |
| `wait_for_service_timeout` | `1000` ms | How long to wait for action servers to become available. |
| `introspection_mode` | `"disabled"` | Nav2 introspection/lifecycle monitoring disabled. |
| `navigators` | `["navigate_to_pose", "navigate_through_poses"]` | Active navigator plugins. |
| `navigate_to_pose.plugin` | `nav2_bt_navigator::NavigateToPoseNavigator` | Standard point-to-point navigation. |
| `navigate_to_pose.enable_groot_monitoring` | `false` | Groot BT visualizer not used. |
| `navigate_to_pose.groot_server_port` | `1667` | Port unused since monitoring is disabled. |
| `navigate_through_poses.plugin` | `nav2_bt_navigator::NavigateThroughPosesNavigator` | Waypoint-sequence navigation. |
| `navigate_through_poses.enable_groot_monitoring` | `false` | |
| `navigate_through_poses.groot_server_port` | `1669` | |

`error_code_name_prefixes` lists the Nav2 action namespaces tracked for error codes: `assisted_teleop`, `backup`, `compute_path`, `drive_on_heading`, `follow_path`, `nav_thru_poses`, `nav_to_pose`, `spin`, `smoother`, `wait`.

---

### controller_server

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | |
| `controller_frequency` | `20.0` Hz | Rate at which the local planner produces velocity commands. |
| `costmap_update_timeout` | `0.30` s | Maximum time to wait for a costmap update before failure. |
| `min_x_velocity_threshold` | `0.001` m/s | Below this, x velocity is treated as zero. |
| `odom_topic` | `/diff_drive_controller/odom` | Odometry source for the controller. |
| `min_y_velocity_threshold` | `0.5` m/s | Below this, y velocity is treated as zero. (Large value is appropriate for a non-holonomic robot.) |
| `min_theta_velocity_threshold` | `0.001` rad/s | Below this, angular velocity is treated as zero. |
| `failure_tolerance` | `0.3` s | How long to tolerate controller failure before aborting. |
| `controller_plugins` | `["FollowPath"]` | Active controller plugin. |
| `path_handler_plugins` | `["PathHandler"]` | |
| `use_realtime_priority` | `false` | No RT thread priority requested. |
| `speed_limit_topic` | `"speed_limit"` | Topic for dynamic speed limiting. |

#### progress_checker

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_controller::SimpleProgressChecker` | |
| `required_movement_radius` | `0.5` m | Robot must move at least this far within `movement_time_allowance`. |
| `movement_time_allowance` | `10.0` s | Time window for progress check. If no movement, navigation fails. |

#### general_goal_checker

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_controller::SimpleGoalChecker` | |
| `stateful` | `True` | Remembers that xy goal was reached when checking yaw. |
| `xy_goal_tolerance` | `0.15` m | Goal is reached when position error is within 15 cm. |
| `yaw_goal_tolerance` | `0.15` rad | Goal is reached when heading error is within ~8.6°. |
| `path_length_tolerance` | `1.0` m | ⚠️ Could not be determined from source — requires runtime inspection to confirm its effect. |

#### PathHandler

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_controller::FeasiblePathHandler` | Filters global path to a feasible local segment. |
| `prune_distance` | `2.0` m | Distance ahead of robot along path kept for local following. |
| `enforce_path_inversion` | `False` | Does not force reversal detection. |
| `enforce_path_rotation` | `False` | |
| `inversion_xy_tolerance` | `0.2` m | Tolerance for detecting path inversion point. |
| `inversion_yaw_tolerance` | `0.4` rad | |
| `minimum_rotation_angle` | `0.785` rad (~45°) | Minimum angle triggering a rotation behavior. |
| `reject_unit_path` | `False` | Does not reject very short paths. |

#### FollowPath (DWBLocalPlanner)

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `dwb_core::DWBLocalPlanner` | Dynamic Window B local planner. |
| `debug_trajectory_details` | `True` | Publishes trajectory debug info. |
| `min_vel_x` | `0.0` m/s | Minimum forward velocity (robot can stop). |
| `min_vel_y` | `0.0` m/s | No lateral velocity for diff-drive. |
| `max_vel_x` | `0.15` m/s | Maximum forward speed. Deliberately conservative — real robot max. |
| `max_vel_y` | `0.0` m/s | |
| `max_vel_theta` | `0.6` rad/s | Maximum turn rate for planning. |
| `min_speed_xy` | `0.0` m/s | |
| `max_speed_xy` | `0.15` m/s | Matches `max_vel_x`. |
| `min_speed_theta` | `0.0` rad/s | |
| `acc_lim_x` | `2.5` m/s² | Forward acceleration limit for trajectory sampling. |
| `acc_lim_y` | `0.0` m/s² | |
| `acc_lim_theta` | `3.2` rad/s² | Angular acceleration limit. |
| `decel_lim_x` | `-2.5` m/s² | Deceleration limit. |
| `decel_lim_y` | `0.0` m/s² | |
| `decel_lim_theta` | `-3.2` rad/s² | |
| `vx_samples` | `20` | Number of linear velocity samples in DW window. |
| `vy_samples` | `5` | (Unused for diff-drive but required by DWB.) |
| `vtheta_samples` | `20` | Number of angular velocity samples. |
| `sim_time` | `1.7` s | How far ahead DWB simulates each trajectory candidate. |
| `linear_granularity` | `0.05` m | Spatial resolution for trajectory simulation steps. |
| `angular_granularity` | `0.025` rad | Angular resolution for trajectory simulation. |
| `transform_tolerance` | `0.2` s | TF lookup time tolerance. |
| `xy_goal_tolerance` | `0.15` m | Matches `general_goal_checker`. |
| `trans_stopped_velocity` | `0.25` m/s | Below this, robot is considered "stopped" for RotateToGoal critic. |
| `short_circuit_trajectory_evaluation` | `True` | Stops evaluating a trajectory once it hits a lethal cost. |
| `stateful` | `True` | |

#### DWB critics

Active critics: `["RotateToGoal", "Oscillation", "BaseObstacle", "GoalAlign", "PathAlign", "PathDist", "GoalDist"]`

| Critic | Scale | Notes |
|---|---|---|
| `BaseObstacle` | `0.5` | **Resolved issue:** was `0.02` — was ~1600× weaker than path critics, meaning obstacles were nearly ignored during path following. Increased 25× to `0.5`. |
| `PathAlign` | `32.0` | Forward point distance: `0.1` m. Rewards trajectories aligned with the global path direction. |
| `GoalAlign` | `24.0` | Forward point distance: `0.1` m. Rewards heading alignment at goal. |
| `PathDist` | `32.0` | Penalises lateral distance from the global path. |
| `GoalDist` | `24.0` | Penalises distance from goal. |
| `RotateToGoal` | `32.0` | Slowing factor: `5.0`. Lookahead time: `-1.0` (disabled). Triggers in-place rotation when near goal. |
| `Oscillation` | (default scale) | Penalises back-and-forth velocity oscillation. |

---

### local_costmap

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | |
| `update_frequency` | `5.0` Hz | How often the costmap is recomputed from sensor data. |
| `publish_frequency` | `2.0` Hz | How often the costmap is published for visualisation. |
| `global_frame` | `odom` | Local costmap is in the odometry frame (rolling window follows robot). |
| `robot_base_frame` | `base_footprint` | |
| `rolling_window` | `true` | Window moves with the robot rather than being fixed in map. |
| `width` | `3` m | Size of local costmap window. |
| `height` | `3` m | |
| `resolution` | `0.05` m/cell | Each cell is 5 cm × 5 cm. |
| `robot_radius` | `0.15` m | Circular robot footprint radius used for collision checks. |
| `plugins` | `["voxel_layer", "inflation_layer"]` | Active layers. |
| `always_send_full_costmap` | `True` | Publishes complete costmap on every update, not just diffs. |
| `introspection_mode` | `"disabled"` | |

#### local_costmap — inflation_layer

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_costmap_2d::InflationLayer` | |
| `cost_scaling_factor` | `3.0` | Exponential decay rate of cost away from obstacles. Higher = sharper falloff. |
| `inflation_radius` | `0.30` m | **Resolved issue:** was `0.05` m — this was less than the robot's own radius (0.15 m), making the robot effectively invisible to collision avoidance. Now set to 2× robot radius. |

#### local_costmap — voxel_layer (scan source)

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_costmap_2d::VoxelLayer` | 3D voxel representation collapsed to 2D costmap. |
| `enabled` | `True` | |
| `publish_voxel_map` | `True` | Publishes 3D voxel map for debugging. |
| `origin_z` | `0.0` m | Voxel column starts at ground level. |
| `z_resolution` | `0.05` m | Vertical cell height. |
| `z_voxels` | `16` | Columns are 16 cells tall (0.8 m total). |
| `max_obstacle_height` | `2.0` m | Obstacles above this are ignored. |
| `mark_threshold` | `0` | Any voxel with at least 1 mark triggers obstacle. |
| `observation_sources` | `scan` | |
| `scan.topic` | `/scan` | LD19 LiDAR scan topic. |
| `scan.max_obstacle_height` | `2.0` m | |
| `scan.clearing` | `True` | Free space is cleared when scan rays pass through. |
| `scan.marking` | `True` | Obstacles are marked when hit by scan rays. |
| `scan.data_type` | `"LaserScan"` | |
| `scan.raytrace_max_range` | `3.0` m | Free space cleared up to 3 m along each ray. |
| `scan.raytrace_min_range` | `0.0` m | |
| `scan.obstacle_max_range` | `2.5` m | Obstacles only marked within 2.5 m. |
| `scan.obstacle_min_range` | `0.0` m | |

---

### global_costmap

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | |
| `update_frequency` | `1.0` Hz | Updated less frequently than local costmap — covers entire map. |
| `publish_frequency` | `1.0` Hz | |
| `global_frame` | `map` | Fixed in the map frame. |
| `robot_base_frame` | `base_footprint` | |
| `robot_radius` | `0.15` m | |
| `resolution` | `0.05` m/cell | Same cell size as local costmap. |
| `track_unknown_space` | `true` | Unknown cells are represented; planner avoids them unless `allow_unknown` is set. |
| `plugins` | `["static_layer", "obstacle_layer", "inflation_layer"]` | |
| `always_send_full_costmap` | `True` | |
| `introspection_mode` | `"disabled"` | |

#### global_costmap — static_layer

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_costmap_2d::StaticLayer` | Loads the occupancy grid from the map server. |
| `map_subscribe_transient_local` | `True` | Uses transient-local QoS to receive the map even if the subscription starts after the map was published. |

#### global_costmap — obstacle_layer

Same observation source configuration as the local voxel_layer scan source (same topic `/scan`, same ranges and data type).

#### global_costmap — inflation_layer

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_costmap_2d::InflationLayer` | |
| `cost_scaling_factor` | `1.5` | Gentler falloff than local costmap (3.0) — global planner benefits from wider cost gradients to steer around obstacles. |
| `inflation_radius` | `0.30` m | **Resolved issue:** was `0.2` m — increased to match robot radius and align with local costmap. |

---

### planner_server

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | |
| `allow_partial_planning` | `false` | Path planning must reach the goal or fail — no partial paths. |
| `expected_planner_frequency` | `20.0` Hz | |
| `planner_plugins` | `["GridBased"]` | |
| `costmap_update_timeout` | `1.0` s | |
| `introspection_mode` | `"disabled"` | |

#### GridBased (NavfnPlanner)

| Parameter | Value | Description |
|---|---|---|
| `plugin` | `nav2_navfn_planner::NavfnPlanner` | Navigation function global planner. |
| `tolerance` | `0.25` m | **Resolved issue:** was `0.5` m — too coarse; robot stopped 50 cm from intended goal. Now 25 cm. |
| `use_astar` | `true` | **Resolved issue:** was `false` (Dijkstra). A* runs faster than Dijkstra with equivalent path quality for this planner. |
| `allow_unknown` | `true` | Planner may navigate through unknown space. |

---

### smoother_server

| Parameter | Value | Description |
|---|---|---|
| `smoother_plugins` | `["simple_smoother"]` | |
| `simple_smoother.plugin` | `nav2_smoother::SimpleSmoother` | Post-processes the global path for smoothness. |
| `simple_smoother.tolerance` | `1.0e-10` | Convergence tolerance for the iterative smoother. |
| `simple_smoother.max_its` | `1000` | Maximum iterations. |
| `simple_smoother.refinement_num` | `2` | Number of refinement passes. |
| `simple_smoother.do_refinement` | `True` | Enables path refinement. |
| `simple_smoother.enforce_path_inversion` | `True` | Prevents path from reversing direction. |

---

### behavior_server

| Parameter | Value | Description |
|---|---|---|
| `behavior_plugins` | `["spin", "backup", "drive_on_heading", "assisted_teleop", "wait"]` | Recovery and auxiliary behaviors. |
| `cycle_frequency` | `10.0` Hz | Behavior execution rate. |
| `local_frame` | `odom` | |
| `global_frame` | `map` | |
| `robot_base_frame` | `base_footprint` | |
| `transform_tolerance` | `0.1` s | |
| `simulate_ahead_time` | `2.0` s | |
| `max_rotational_vel` | `1.0` rad/s | Maximum spin speed during recovery behaviors. |
| `min_rotational_vel` | `0.4` rad/s | Minimum spin speed. |
| `rotational_acc_lim` | `3.2` rad/s² | Angular acceleration for recovery behaviors. |
| `backup.acceleration_limit` | `2.5` m/s² | |
| `backup.deceleration_limit` | `-2.5` m/s² | |
| `backup.minimum_speed` | `0.10` m/s | |
| `drive_on_heading.acceleration_limit` | `2.5` m/s² | |
| `drive_on_heading.deceleration_limit` | `-2.5` m/s² | |
| `drive_on_heading.minimum_speed` | `0.10` m/s | |

---

### waypoint_follower

| Parameter | Value | Description |
|---|---|---|
| `loop_rate` | `20` Hz | |
| `stop_on_failure` | `false` | Continues to next waypoint if one fails. |
| `waypoint_task_executor_plugin` | `"wait_at_waypoint"` | |
| `wait_at_waypoint.enabled` | `True` | |
| `wait_at_waypoint.waypoint_pause_duration` | `200` ms | Brief pause at each waypoint. |

---

### velocity_smoother

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | |
| `smoothing_frequency` | `20.0` Hz | Rate at which velocity commands are smoothed. |
| `scale_velocities` | `False` | Does not scale velocity components proportionally. |
| `feedback` | `"CLOSED_LOOP"` | **Resolved issue:** was `"OPEN_LOOP"`. Now reads actual odometry to apply acceleration limits, producing smoother velocity profiles. |
| `max_velocity` | `[0.15, 0.0, 0.6]` | [x, y, theta] — matches DWB planner limits. |
| `min_velocity` | `[-0.5, 0.0, -1.0]` | |
| `max_accel` | `[2.5, 0.0, 3.2]` | |
| `max_decel` | `[-2.5, 0.0, -3.2]` | |
| `odom_topic` | `"/diff_drive_controller/odom"` | Odometry source for CLOSED_LOOP feedback. |
| `odom_duration` | `0.1` s | Time window for odometry averaging. |
| `deadband_velocity` | `[0.0, 0.0, 0.0]` | No deadband applied. |
| `velocity_timeout` | `1.0` s | Commands older than this are discarded. |

---

### collision_monitor

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | |
| `enabled` | `True` | Active safety layer. |
| `base_frame_id` | `"base_footprint"` | |
| `odom_frame_id` | `"odom"` | |
| `cmd_vel_in_topic` | `"cmd_vel_smoothed"` | Receives already-smoothed velocity commands. |
| `cmd_vel_out_topic` | `"cmd_vel"` | Outputs velocity with collision-modified limits. |
| `state_topic` | `"collision_monitor_state"` | Publishes current monitor state. |
| `transform_tolerance` | `0.2` s | |
| `source_timeout` | `1.0` s | LiDAR scan must arrive within this window. |
| `base_shift_correction` | `True` | Accounts for robot motion during sensor data collection. |
| `stop_pub_timeout` | `2.0` s | How long to keep publishing zero velocity after stop command. |
| `polygons` | `["FootprintApproach"]` | |

#### FootprintApproach polygon

| Parameter | Value | Description |
|---|---|---|
| `type` | `"polygon"` | Defined as a polygon shape (uses dynamic footprint from costmap). |
| `action_type` | `"approach"` | Reduces speed as robot approaches obstacle, scaling velocity linearly with `time_before_collision`. |
| `footprint_topic` | `"local_costmap/published_footprint"` | Uses dynamic footprint published by local costmap. |
| `time_before_collision` | `1.2` s | Safety time horizon: robot is slowed if a collision would occur within 1.2 s at current velocity. |
| `simulation_time_step` | `0.1` s | Time step for collision prediction. |
| `min_points` | `6` | Minimum LiDAR points to trigger approach action. |
| `visualize` | `False` | Polygon not published for visualisation. |

#### collision_monitor — scan source

| Parameter | Value | Description |
|---|---|---|
| `type` | `"scan"` | |
| `topic` | `"/scan"` | LD19 LiDAR. |
| `min_height` | `0.15` m | Ignores returns below 15 cm (floor reflections). |
| `max_height` | `2.0` m | |

---

### map_saver

| Parameter | Value | Description |
|---|---|---|
| `save_map_timeout` | `5.0` s | |
| `free_thresh_default` | `0.25` | Cells with occupancy probability below 0.25 are saved as free. |
| `occupied_thresh_default` | `0.65` | Cells above 0.65 are saved as occupied. Matches SLAM Toolbox saves (confirmed in `studio_3_save.yaml`). |
| `map_subscribe_transient_local` | `True` | |

---

### docking_server

> **Informational note (Issue #9):** The docking server is fully configured but no physical docking station or charging hardware has been found anywhere else in this workspace. This is likely an aspirational or placeholder configuration. It should not be exercised without physical hardware.

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `False` | |
| `controller_frequency` | `50.0` Hz | High-frequency control for precise dock approach. |
| `initial_perception_timeout` | `5.0` s | Time to detect the dock before aborting. |
| `wait_charge_timeout` | `5.0` s | Time to wait for charging confirmation. |
| `dock_approach_timeout` | `30.0` s | Maximum time allowed for the full dock approach. |
| `undock_linear_tolerance` | `0.05` m | |
| `undock_angular_tolerance` | `0.1` rad | |
| `max_retries` | `3` | |
| `base_frame` | `"base_footprint"` | |
| `fixed_frame` | `"odom"` | |
| `dock_prestaging_tolerance` | `0.5` m | |
| `dock_plugins` | `["simple_charging_dock"]` | |
| `simple_charging_dock.plugin` | `opennav_docking::SimpleChargingDock` | |
| `simple_charging_dock.docking_threshold` | `0.05` m | Contact distance threshold. |
| `simple_charging_dock.staging_x_offset` | `-0.7` m | Pre-dock staging position (0.7 m behind dock). |
| `simple_charging_dock.use_external_detection_pose` | `false` | Uses internal perception. |
| `simple_charging_dock.use_battery_status` | `false` | No battery topic monitored. |
| `simple_charging_dock.use_stall_detection` | `false` | No stall current detection. |
| `controller.k_phi` | `3.0` | Heading proportional gain for dock approach controller. |
| `controller.k_delta` | `2.0` | Lateral error gain. |
| `controller.v_linear_min` | `0.15` m/s | Minimum approach speed. |
| `controller.v_linear_max` | `0.15` m/s | Maximum approach speed (same as min — constant approach). |

---

## Usage

Loaded by `real_robot.launch.py` and any Nav2 bringup launch that targets the real robot. Not used by `sim.launch.py` — that uses `sim_nav2_params.yaml`.

## Notes / Known issues

- **Issue #9 (Low/Informational):** `docking_server` is configured but no docking hardware exists in the workspace.
- Velocity smoother and DWB `max_vel_x` are both `0.15` m/s. This is deliberately conservative for the real robot. The original config had `0.5` m/s (see `original_nav2_params.yaml`).
- All `/scan` observation sources are confirmed correct — old audit claim that costmaps subscribed to `/lidar` is resolved.
- `controller_server.odom_topic` and `velocity_smoother.odom_topic` both point to `/diff_drive_controller/odom` (raw wheel odometry). If the EKF is enabled in the future, these may need updating to `/odometry/filtered`.

## See Also

- [`original_nav2_params.yaml`](original_nav2_params.md) — Historical "before" snapshot
- [`sim_nav2_params.yaml`](sim_nav2_params.md) — Simulation variant
- [`ubot_controllers.yaml`](ubot_controllers.md) — Controller config that publishes `/diff_drive_controller/odom`
- [`real_ekf.yaml`](real_ekf.md) — EKF config (currently not launched)
