# ubot_bringup

The launch and configuration coordinator for the ubot robot. This package contains no compiled code — it is a pure `ament_cmake` package whose sole purpose is to install launch files and configuration YAMLs into the ROS 2 share directory.

## Overview

| Field | Value |
|---|---|
| Package name | `ubot_bringup` |
| Version | `0.0.0` |
| License | BSD-3-Clause |
| Build type | `ament_cmake` |
| Maintainer | chibueze (praiseorji4@gmail.com) |
| Exec dependency | `robot_localization` (declared but EKF node is currently commented out) |

The `CMakeLists.txt` installs three directories:

```cmake
install(
  DIRECTORY config launch worlds
  DESTINATION share/${PROJECT_NAME}
)
```

No library or executable targets are built.

## Launch Files

Four launch files are under `launch/`:

| File | What It Launches | State |
|---|---|---|
| `real_robot.launch.py` | Full real-robot bringup (RSP, controller_manager, joint_state_broadcaster, diff_drive_controller, twist_stamper, ldlidar_node). BNO055 and EKF are defined but excluded. | Functional — 6 nodes active, 2 intentionally excluded (see Known Issues) |
| `laptop_slam_nav2.launch.py` | SLAM Toolbox (online_async), RViz2 (slam.rviz), Nav2 bringup (navigation_launch.py). Designed to run on the laptop while `real_robot.launch.py` runs on the robot. | Functional — all three inclusions are active |
| `sim.launch.py` | Simulation bringup (robot_state_publisher with `use_gazebo:=true`, Gazebo, gz_ros2_control, controller spawners, twist_stamper, Nav2, RViz2). Contains approximately 120 lines of dead/commented-out code. | Partially working — core sim path functional; dead code is a maintainability concern |
| `test_wheels.launch.py` | ⚠️ Empty (1 line). Will fail or do nothing if launched. | **Non-functional** |

### real_robot.launch.py — Node Details

The returned `LaunchDescription` contains exactly 6 entries, in order:

1. **`node_robot_state_publisher`** (`robot_state_publisher/robot_state_publisher`) — `robot_description` generated via `xacro ubot_robot.urdf.xacro use_gazebo:=false`. `use_sim_time: False`.
2. **`controller_manager`** (`controller_manager/ros2_control_node`) — params: `robot_description` + `config/ubot_controllers.yaml`. `use_sim_time: False`.
3. **`joint_state_broadcaster_spawner`** — `TimerAction(period=5.0)` → spawner for `joint_state_broadcaster`.
4. **`diff_drive_controller_spawner`** — `TimerAction(period=6.0)` → spawner for `diff_drive_controller`.
5. **`twist_stamper`** (`twist_stamper/twist_stamper`) — remaps `/cmd_vel` → `/cmd_vel_in`, `/diff_drive_controller/cmd_vel` → `/cmd_vel_out`. `frame_id: base_footprint`.
6. **`ldlidar_node`** (`ldlidar_stl_ros2/ldlidar_stl_ros2_node`) — LD19, `/dev/ttyUSB1`, 230400 baud, `lidar_link` frame.

**Defined but excluded** (commented out in the `LaunchDescription` list):

- `bno055_node` — defined at lines ~101–107 with params from `bno055_params.yaml`; the line `# bno055_node,` at ~line 125 is commented out.
- `ekf_node` — the entire `Node()` block is commented out (lines ~110–116); would launch `robot_localization/ekf_node` named `ekf_filter_node` with `real_ekf.yaml`.

### laptop_slam_nav2.launch.py — Node Details

Includes three launch actions (all active):

1. **`slam_toolbox`** — `online_async_launch.py` with `config/mapper_params_online_async.yaml`, `use_sim_time: false`.
2. **`rviz_node`** — `rviz2/rviz2` with `rviz/slam.rviz` config, `use_sim_time: False`.
3. **`nav2`** — `nav2_bringup/navigation_launch.py` with `config/nav2_params.yaml`, `use_sim_time: false`.

Note: Despite the filename containing "Nav2", all three components including Nav2 are active. This is not the case for `sim.launch.py` which has more complex conditional inclusion.

## Configuration Files

Nine YAML files under `config/`:

| File | Purpose |
|---|---|
| `ubot_controllers.yaml` | `controller_manager` configuration: `update_rate: 30` Hz, `diff_drive_controller` (front wheels only, `wheel_separation: 0.264204`, `wheel_radius: 0.033`), `joint_state_broadcaster`. Includes embedded odometry debugging checklist and PlotJuggler signal list. |
| `nav2_params.yaml` | Nav2 stack configuration for real robot: DWB local planner, NavFnPlanner (A*), voxel + inflation costmap layers, collision monitor, velocity smoother (CLOSED_LOOP), docking server. Fully tuned current state. |
| `sim_nav2_params.yaml` | Nav2 configuration for simulation: `use_sim_time: true`, `odom_topic: /odometry/filtered` (expects EKF output, unlike real robot). Different inflation radii. |
| `original_nav2_params.yaml` | Historical "before tuning" snapshot of `nav2_params.yaml`. Not referenced by any active launch file. Kept for comparison. |
| `real_ekf.yaml` | `robot_localization` EKF configuration: `frequency: 30` Hz, `two_d_mode: true`, fuses `odom0=/diff_drive_controller/odom` and `imu0=/bno055/imu` (yaw angular velocity only). **Not currently launched** — `ekf_node` is commented out in `real_robot.launch.py`. |
| `sim_ekf.yaml` | EKF configuration for simulation: `use_sim_time: true`, `frequency: 50` Hz, `imu0: /imu/data`. |
| `sim_ubot_controllers.yaml` | Controller configuration for simulation: `update_rate: 50` Hz, all 4 wheels included in `diff_drive_controller` wheel lists (unlike real robot which uses front wheels only). |
| `bno055_params.yaml` | IMU deployment config: `connection_type: i2c`, `i2c_bus: 1`, `frame_id: imu_link`, `placement_axis_remap: P2`, `data_query_frequency: 100`, calibration offsets populated but `set_offsets: false`. |
| `mapper_params_online_async.yaml` | SLAM Toolbox online async mapping: `mode: mapping`, CeresSolver, `odom_frame: odom`, `map_frame: map`, `base_frame: base_footprint`, `map_resolution: 0.05` m, `max_laser_range: 12` m, loop closure enabled. References prior save path `/home/chibueze/uni-bot/studio_3_serial`. |

## Known Issues

### BNO055 and EKF Not Active (Medium-High)

`bno055_node` and `ekf_node` are both defined in `real_robot.launch.py` but excluded from the returned `LaunchDescription`. The real robot navigates on wheel odometry alone. All configuration files are written and ready — enabling them requires uncommenting two lines in `real_robot.launch.py`. See [bno055](bno055.md) for IMU node details.

### `has_velocity_limits` / `has_acceleration_limits` Flags Missing (Medium)

`ubot_controllers.yaml` sets velocity and acceleration limits for `diff_drive_controller`:

```yaml
linear.x.max_velocity: 0.5
linear.x.max_acceleration: 1.0
angular.z.max_velocity: 2.0
angular.z.max_acceleration: 2.0
```

However, `has_velocity_limits: true` and `has_acceleration_limits: true` boolean flags are never set anywhere in the file. Depending on the installed `diff_drive_controller` version, limits may be silently ignored without these flags. Verify against the installed `ros2_control` Jazzy version.

### `position_feedback` Potentially Deprecated (Requires Verification)

`ubot_controllers.yaml` line 86 sets `position_feedback: true`. This parameter name may be renamed or deprecated in the installed version. ⚠️ Requires runtime inspection to confirm.

### `test_wheels.launch.py` is Empty

The file contains only 1 line and has no effective content. It will fail or produce no output if launched. Should either be populated or removed.

### `sim.launch.py` Contains ~120 Lines of Dead Code

Multiple commented-out node blocks exist in `sim.launch.py`. This is a maintainability concern but does not affect runtime behavior of the active code path.

### `laptop_slam_nav2.launch.py` — Nav2 Active Despite Earlier Audit Claim

An earlier audit (WORKSPACE_AUDIT.md issue #20) noted that Nav2 was commented out. As of the current source read, Nav2 is **fully active** in this file — all three entries (`slam_toolbox`, `rviz_node`, `nav2`) appear in the returned `LaunchDescription`. The audit issue is resolved.

## See Also

- [bno055](bno055.md) — IMU driver (currently excluded from launch)
- [ldlidar_stl_ros2](ldlidar_stl_ros2.md) — active LiDAR driver
- [ubot_control](ubot_control.md) — hardware interface loaded by controller_manager
- [ubot_description](ubot_description.md) — URDF/xacro model used by robot_state_publisher
- [ubot_debugger](ubot_debugger.md) — diagnostic node (not included in any launch file)
