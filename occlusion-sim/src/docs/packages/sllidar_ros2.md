# sllidar_ros2

Vendored RPLiDAR driver from Slamtec. This package is present in the workspace but is **not currently used** on the real robot.

## Overview

| Field | Value |
|---|---|
| Package name | `sllidar_ros2` |
| Version | `1.0.1` |
| License | BSD |
| Build type | `ament_cmake` |
| Upstream maintainer | Slamtec ROS Maintainer (ros@slamtec.com) |
| Status | NOT used on real robot — `real_robot.launch.py` uses `ldlidar_stl_ros2` instead |

## What is an RPLiDAR?

RPLiDAR is Slamtec's family of 2D laser rangefinders (A-series, S-series, C-series, T-series). They use a rotating laser triangulation or ToF approach to produce 360° planar scans. The `sllidar_ros2` driver communicates with the sensor over USB-serial and publishes a `sensor_msgs/LaserScan` message. Topic name and frame ID are configurable via launch-file parameters.

## Why This Package is in the Workspace

The workspace likely included `sllidar_ros2` during an earlier phase of development or as a candidate driver. The live robot uses an LD19 LiDAR (LDROBOT product), which requires `ldlidar_stl_ros2`. The two drivers use incompatible protocols and are not interchangeable without configuration changes.

## Executables

Two executables are built:

| Executable | Source | Role |
|---|---|---|
| `sllidar_node` | `src/sllidar_node.cpp` | Main driver node — reads scan data and publishes `/scan`. |
| `sllidar_client` | `src/sllidar_client.cpp` | Utility client (motor control, etc.). |

## Topics

If `sllidar_node` were run, it would publish:

| Topic | Message Type | Description |
|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | 2D laser scan. Topic name and `frame_id` are set via launch-file parameters. |

⚠️ The exact default topic name and QoS cannot be determined from source alone — requires runtime inspection of `sllidar_node.cpp`.

## Launch Files

The package ships 24 launch files under `launch/`. They follow two patterns:

- **Driver-only** (`sllidar_<model>_launch.py`): launches `sllidar_node` with model-specific defaults.
- **Driver + RViz viewer** (`view_sllidar_<model>_launch.py`): launches the driver plus an RViz2 instance.

### Supported Models

| Driver launch | Viewer launch | Model |
|---|---|---|
| `sllidar_a1_launch.py` | `view_sllidar_a1_launch.py` | RPLIDAR A1 |
| `sllidar_a2m7_launch.py` | `view_sllidar_a2m7_launch.py` | RPLIDAR A2 (M7 variant) |
| `sllidar_a2m8_launch.py` | `view_sllidar_a2m8_launch.py` | RPLIDAR A2 (M8 variant) |
| `sllidar_a2m12_launch .py` | `view_sllidar_a2m12_launch.py` | RPLIDAR A2 (M12 variant) |
| `sllidar_a3_launch.py` | `view_sllidar_a3_launch.py` | RPLIDAR A3 |
| `sllidar_c1_launch.py` | `view_sllidar_c1_launch.py` | RPLIDAR C1 |
| `sllidar_s1_launch.py` | `view_sllidar_s1_launch.py` | RPLIDAR S1 |
| `sllidar_s1_tcp_launch.py` | `view_sllidar_s1_tcp_launch.py` | RPLIDAR S1 (TCP/IP transport) |
| `sllidar_s2_launch.py` | `view_sllidar_s2_launch.py` | RPLIDAR S2 |
| `sllidar_s2e_launch.py` | `view_sllidar_s2e_launch.py` | RPLIDAR S2E |
| `sllidar_s3_launch.py` | `view_sllidar_s3_launch.py` | RPLIDAR S3 |
| `sllidar_t1_launch.py` | `view_sllidar_t1_launch.py` | RPLIDAR T1 |

Note: `sllidar_a2m12_launch .py` has a trailing space in the filename as it exists on disk.

## Build

```
ament_cmake, C++14
Dependencies: rclcpp, sensor_msgs, std_srvs
SDK: ./sdk/ (Slamtec RPLIDAR SDK, compiled from arch/linux, hal, dataunpacker subdirs)
```

## If You Ever Switch to RPLiDAR

To activate `sllidar_ros2` on the real robot in place of `ldlidar_stl_ros2`:

1. **`real_robot.launch.py`**: Replace the `ldlidar_node` `Node()` block with one that launches `sllidar_node` from `sllidar_ros2`. Use the appropriate model-specific launch as a reference.
2. **`frame_id`**: Update the `frame_id` parameter in the node invocation to `lidar_link` (to match the URDF), unless the TF setup is changed.
3. **URDF**: `ubot_lidar.urdf.xacro` defines `lidar_link` — no change needed if `frame_id` is kept consistent.
4. **SLAM Toolbox**: `mapper_params_online_async.yaml` references scan data implicitly through the `/scan` topic — no change needed if topic name stays `/scan`.
5. **Serial port**: RPLiDAR typically uses `/dev/ttyUSB0` by default. Since `/dev/ttyUSB0` is already used by the ESP32 motor controller, a different port or USB hub port assignment is required to avoid conflict.

## See Also

- [ldlidar_stl_ros2](ldlidar_stl_ros2.md) — the active LiDAR driver on the real robot
- [ubot_bringup](ubot_bringup.md) — `real_robot.launch.py` that selects which driver to use
- [ubot_description](ubot_description.md) — `lidar_link` frame in the URDF
