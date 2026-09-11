# mapper_params_online_async.yaml

## Purpose

SLAM Toolbox configuration for online asynchronous mapping. In this mode, the node processes incoming laser scans as they arrive (asynchronously) and builds a map in real time while the robot moves. This is the primary configuration used for mapping new environments with the ubot robot.

The file currently has `mode: mapping` (not `localization`), making it the active SLAM mapping config. A `map_file_name` is set pointing to a previously saved map (`studio_3_serial`) — this is used only when switching to `localization` mode and is inactive during mapping.

## What `.data` and `.posegraph` files are

When SLAM Toolbox saves a map (via the `slam_toolbox/SaveMap` service or `use_map_saver: true`), it creates two serialized files:

- **`.posegraph`** — The pose graph: nodes (robot poses at which scans were taken), edges (spatial constraints between poses), and loop closure constraints. This is the internal SLAM representation used to reconstruct or extend the map.
- **`.data`** — The scan data associated with each pose graph node — the raw or compressed laser scan readings. Paired with the `.posegraph`, this allows the full map to be reloaded for continued mapping or localization.

Additionally, a standard ROS occupancy grid can be saved as `.pgm` + `.yaml` (via `map_saver_cli`). The workspace contains five saved maps from real robot runs:

| Map name | Files present |
|---|---|
| `studio_1_save` | `.pgm`, `.yaml`, `.data`, `.posegraph` |
| `studio_2_serial` | `.data`, `.posegraph` |
| `studio_3_serial` | `.data`, `.posegraph` (referenced by `map_file_name`) |
| `Tee_map` | ⚠️ Could not be determined from source — requires runtime inspection |
| `defence_map` | ⚠️ Could not be determined from source — requires runtime inspection |

The `studio_3_serial` map has confirmed resolution `0.050` m/cell, origin `[-3.844, -2.408, 0]`, `occupied_thresh=0.65`, `free_thresh=0.196` (from `studio_3_save.yaml`).

## Parameters

### Plugin and solver

| Parameter | Value | Description |
|---|---|---|
| `solver_plugin` | `solver_plugins::CeresSolver` | Uses Google Ceres Solver for pose graph optimization. Ceres is more robust to large maps than the alternative (g2o/Karto). |
| `ceres_linear_solver` | `SPARSE_NORMAL_CHOLESKY` | Sparse Cholesky factorization — efficient for sparse pose graphs. |
| `ceres_preconditioner` | `SCHUR_JACOBI` | Jacobi-based Schur complement preconditioner. |
| `ceres_trust_strategy` | `LEVENBERG_MARQUARDT` | Levenberg-Marquardt trust region method for nonlinear optimization. |
| `ceres_dogleg_type` | `TRADITIONAL_DOGLEG` | Traditional dogleg step within the trust region. |
| `ceres_loss_function` | `None` | No robust loss function (standard least squares). |

### ROS / frame parameters

| Parameter | Value | Description |
|---|---|---|
| `odom_frame` | `odom` | Frame name for odometry. Must match `odom_frame_id` in `ubot_controllers.yaml`. |
| `map_frame` | `map` | Frame name for the SLAM map. |
| `base_frame` | `base_footprint` | Robot base frame. Must match `base_frame_id` in `ubot_controllers.yaml`. |
| `scan_topic` | `/scan` | LiDAR input topic. Matches the LD19 LiDAR topic name published by `ldlidar_node`. |
| `use_map_saver` | `true` | Enables the internal map saver service. |
| `mode` | `mapping` | Active SLAM mode. Alternative is `localization` (load existing map, localize only). |

### Map file (localization mode only)

| Parameter | Value | Description |
|---|---|---|
| `map_file_name` | `/home/chibueze/uni-bot/studio_3_serial` | **Hardcoded absolute path** to a previously saved SLAM Toolbox serialized map (no file extension — SLAM Toolbox automatically appends `.posegraph` when loading). Used only when `mode: localization`. |
| `map_start_at_dock` | `true` | When loading a map for localization, start pose is assumed to be the map origin (dock position). |

> **Important — hardcoded path:** `map_file_name` is an absolute path tied to the machine where the maps were recorded (`/home/chibueze/uni-bot/`). When switching environments, deploying on a different machine, or loading a different map, this path **must be updated manually**. Failure to update it will cause SLAM Toolbox to fail to load the map when switching to localization mode.

### QoS override

| Parameter | Value | Description |
|---|---|---|
| `/scan subscription reliability` | `reliable` | Ensures no scan messages are dropped. |
| `/scan subscription durability` | `volatile` | Does not request historical scan messages. |

### Timing and transform parameters

| Parameter | Value | Description |
|---|---|---|
| `debug_logging` | `false` | Disables verbose debug output. |
| `throttle_scans` | `1` | Process every scan (no throttling). |
| `transform_publish_period` | `0.02` s | TF `map → odom` is published at **50 Hz** (1/0.02). This high rate ensures Nav2 and other consumers always have a fresh transform. If set to `0`, the transform is never published. |
| `map_update_interval` | `5.0` s | Occupancy grid map is recomputed and published every 5 seconds. |
| `minimum_time_interval` | `0.5` s | Minimum time between processing successive scans. Even if scans arrive faster, SLAM Toolbox will skip until 0.5 s has elapsed. |
| `transform_timeout` | `0.2` s | Maximum wait for a required TF transform before giving up. |
| `tf_buffer_duration` | `30.0` s | TF buffer size. 30 seconds accommodates delayed or reordered transforms. |
| `stack_size_to_use` | `40000000` bytes (40 MB) | Increased stack size for serializing large maps. The comment in the file states: "program needs a larger stack size to serialize large maps." |
| `enable_interactive_mode` | `true` | Enables the SLAM Toolbox interactive tools (loop closure correction, pose graph editing via RViz plugin). |

### Map resolution and range

| Parameter | Value | Description |
|---|---|---|
| `resolution` | `0.05` m/cell | Each cell in the occupancy grid is 5 cm × 5 cm. Matches costmap resolution in `nav2_params.yaml`. |
| `max_laser_range` | `12.0` m | Maximum range used for map rasterization. The LD19 LiDAR has a rated range of ~12 m. Scans beyond this are ignored. |

### Scan acceptance criteria

| Parameter | Value | Description |
|---|---|---|
| `minimum_travel_distance` | `0.5` m | Robot must travel at least 0.5 m before a new scan is added to the pose graph. Prevents over-dense graphs when the robot is stationary. |
| `minimum_travel_heading` | `0.5` rad (~28.6°) | Robot must rotate at least 0.5 rad before a new scan is accepted (if distance threshold not met). |
| `scan_buffer_size` | `10` | Number of recent scans kept in memory for scan matching. |
| `scan_buffer_maximum_scan_distance` | `10.0` m | Maximum distance of buffered scans from current pose. |

### Scan matching

| Parameter | Value | Description |
|---|---|---|
| `use_scan_matching` | `true` | Enables scan-to-map matching for pose correction. |
| `use_scan_barycenter` | `true` | Uses scan centroid for matching computations. |
| `link_match_minimum_response_fine` | `0.1` | Minimum correlation response to accept a fine scan match. |
| `link_scan_maximum_distance` | `1.5` m | Maximum distance between scans to attempt matching. |

### Loop closure

| Parameter | Value | Description |
|---|---|---|
| `do_loop_closing` | `true` | Loop closure is enabled. The pose graph optimizer corrects accumulated drift when a previously visited location is recognized. |
| `loop_search_maximum_distance` | `3.0` m | Search radius for loop closure candidates. |
| `loop_match_minimum_chain_size` | `10` | Minimum number of consecutive nodes in a loop candidate chain. |
| `loop_match_maximum_variance_coarse` | `3.0` | Maximum variance for coarse loop match acceptance. |
| `loop_match_minimum_response_coarse` | `0.35` | Minimum correlation response for coarse loop closure. |
| `loop_match_minimum_response_fine` | `0.45` | Minimum correlation response for fine loop closure confirmation. |

### Correlation parameters (scan matching search space)

| Parameter | Value | Description |
|---|---|---|
| `correlation_search_space_dimension` | `0.5` m | Size of the search window for scan correlation during pose estimation. |
| `correlation_search_space_resolution` | `0.01` m | Resolution of the correlation search grid. |
| `correlation_search_space_smear_deviation` | `0.1` | Gaussian smear applied to correlation space. |

### Loop closure correlation parameters

| Parameter | Value | Description |
|---|---|---|
| `loop_search_space_dimension` | `8.0` m | Larger search window for loop closure correlation. |
| `loop_search_space_resolution` | `0.05` m | Coarser resolution for loop search. |
| `loop_search_space_smear_deviation` | `0.03` | Tighter smear for loop closure. |

### Scan matcher parameters

| Parameter | Value | Description |
|---|---|---|
| `distance_variance_penalty` | `0.5` | Penalty applied for distance uncertainty in scan matching. |
| `angle_variance_penalty` | `1.0` | Penalty for angular uncertainty. |
| `fine_search_angle_offset` | `0.00349` rad (~0.2°) | Angular offset for fine search pass. |
| `coarse_search_angle_offset` | `0.349` rad (~20°) | Angular offset for coarse search pass. |
| `coarse_angle_resolution` | `0.0349` rad (~2°) | Angular resolution for coarse scan matching. |
| `minimum_angle_penalty` | `0.9` | Floor on the angular penalty score. |
| `minimum_distance_penalty` | `0.5` | Floor on the distance penalty score. |
| `use_response_expansion` | `true` | Expands the search if initial correlation response is poor. |
| `min_pass_through` | `2` | Minimum number of laser passes through a cell to mark it free. |
| `occupancy_threshold` | `0.1` | Occupancy probability threshold for marking cells. |

## Usage

Loaded by any SLAM Toolbox launch configuration targeting the real robot. Typical invocation:

```bash
ros2 launch slam_toolbox online_async_launch.py \
    params_file:=<path_to>/mapper_params_online_async.yaml
```

Or via a dedicated bringup launch file that includes this config.

To switch from mapping to localization with the `studio_3_serial` map:
1. Change `mode: mapping` to `mode: localization`
2. Verify `map_file_name` points to the correct map path
3. Relaunch

## Notes / Known issues

- **Hardcoded path:** `map_file_name: /home/chibueze/uni-bot/studio_3_serial` — this path must be updated when deploying to a different machine or when using a different saved map. This is an environment-specific value that should ideally be passed as a launch argument.
- `transform_publish_period: 0.02` (50 Hz) produces `map → odom` at a higher rate than the Nav2 controller frequency (20 Hz), which is correct — the transform should always be fresher than the consumers' cycle time.
- `minimum_travel_distance: 0.5` and `minimum_travel_heading: 0.5` set a fairly coarse scan acceptance threshold. In tight spaces with sharp turns, the 0.5 m distance requirement may mean too few scans are added. Reduce if mapping quality in constrained areas is poor.
- `enable_interactive_mode: true` enables the RViz SLAM Toolbox panel for manual loop closure adjustment — useful during mapping sessions but adds computational overhead.
- With `stack_size_to_use: 40000000`, this node requests 40 MB of stack. This is documented in the file itself as necessary for large maps.

## See Also

- [`nav2_params.yaml`](nav2_params.md) — Nav2 config that depends on the `map` frame produced by SLAM Toolbox
- [`ubot_controllers.yaml`](ubot_controllers.md) — Provides `odom` frame that SLAM Toolbox consumes as input
- Saved maps in `/home/chibueze/uni-bot/`: `studio_1_save`, `studio_2_serial`, `studio_3_serial`, `Tee_map`, `defence_map`
