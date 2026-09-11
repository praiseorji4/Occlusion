# sllidar_node

## Overview

`sllidar_node` is the ROS 2 driver for Slamtec RPLiDAR sensors (A1, A2, A3, S1, S2, S3, C1, T1
and variants).  It is provided by the vendored `sllidar_ros2` package (v1.0.1, BSD licence,
upstream: Slamtec / Shanghai Slamtec Co., Ltd.).  No modifications have been made to this package
in the ubot workspace.

> **This driver is NOT currently used on the ubot robot.**
>
> `real_robot.launch.py` launches `ldlidar_stl_ros2_node` for the LDROBOT LD19 sensor that is
> physically installed.  The `sllidar_ros2` package is present in the workspace as an alternative
> driver for use if the hardware is ever switched to a Slamtec RPLiDAR sensor.

| Attribute | Value |
|---|---|
| Package | `sllidar_ros2` (v1.0.1, BSD) |
| Executable | `sllidar_node` |
| ROS node name (from source) | `sllidar_node` |
| Supported hardware | Slamtec RPLiDAR A1, A2, A3, S1, S2, S2E, S3, C1, T1 |

## Parameters

All parameters are declared in `sllidar_ros2/src/sllidar_node.cpp` (`SLlidarNode::init_param()`).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `channel_type` | string | `'serial'` | Transport: `'serial'`, `'tcp'`, or `'udp'` |
| `serial_port` | string | `'/dev/ttyUSB0'` | Serial device path (serial mode) |
| `serial_baudrate` | int | `1000000` | A1/A2 typically 115200; A3 typically 256000; S-series often 1000000 |
| `frame_id` | string | `'laser_frame'` | Header frame_id in published LaserScan — **see warning below** |
| `inverted` | bool | `false` | Invert scan data direction |
| `angle_compensate` | bool | `false` | Enable angle compensation |
| `scan_mode` | string | `''` (empty) | Scan mode name (model-dependent); empty = firmware default |
| `scan_frequency` | float | `10.0` Hz (serial/tcp), `20.0` Hz (udp) | Target scan frequency |
| `tcp_ip` | string | `'192.168.0.7'` | TCP mode only |
| `tcp_port` | int | `20108` | TCP mode only |
| `udp_ip` | string | `'192.168.11.2'` | UDP mode only |
| `udp_port` | int | `8089` | UDP mode only |

### Per-model baud rates (from sllidar_a1_launch.py and source comments)

| Model | Typical `serial_baudrate` |
|---|---|
| A1 | `115200` |
| A2 (M7/M8) | `115200` |
| A2 (M12) | `115200` |
| A3 | `256000` |
| S1, S2, S3 | `1000000` |
| C1 | `460800` |
| T1 | `1000000` |

## Published Topics

| Topic | Message Type | QoS | frame_id |
|---|---|---|---|
| `scan` | `sensor_msgs/LaserScan` | `KeepLast(10)` | Value of `frame_id` parameter (default `'laser_frame'`) |

## Services

The node advertises two motor-control services (from `sllidar_node.cpp`):

| Service | Type | Behaviour |
|---|---|---|
| `stop_motor` | `std_srvs/srv/Empty` | Stops the LiDAR motor |
| `start_motor` | `std_srvs/srv/Empty` | Starts the LiDAR motor and resumes scanning |

## Subscribed Topics

None.

## Lifecycle / Operation

`SLlidarNode` is a plain `rclcpp::Node`.  The `work_loop()` method (called from `main()`):

1. Calls `init_param()` to declare and read all parameters.
2. Creates the SLLIDAR driver and the appropriate channel (serial, TCP, or UDP).
3. Connects to the device; logs an error and returns `-1` if connection fails.
4. Queries device info (firmware/hardware version, serial number) and health status.
5. Starts scanning.
6. Spins, calling `grabScanDataHq()` and `publish_scan()` to fill and publish `LaserScan` messages.

The scan message uses `range_min = 0.05` m (from source).  `range_max` is model-dependent and
determined at runtime from the driver.

## Available Launch Files

The `sllidar_ros2/launch/` directory contains 20 launch files covering all supported models.  None
are referenced in any ubot bringup launch file.  Representative examples:

| File | Model | Default serial_baudrate | Default scan_mode |
|---|---|---|---|
| `sllidar_a1_launch.py` | RPLiDAR A1 | `115200` | `'Sensitivity'` |
| `sllidar_a2m7_launch.py` | RPLiDAR A2 M7 | `115200` | — |
| `sllidar_a3_launch.py` | RPLiDAR A3 | `256000` | — |
| `sllidar_s1_launch.py` | RPLiDAR S1 | `1000000` | — |
| `sllidar_s1_tcp_launch.py` | RPLiDAR S1 (TCP) | n/a | — |

## When to Use This Driver

Use `sllidar_ros2` instead of `ldlidar_stl_ros2` if the physical sensor is replaced with a
Slamtec RPLiDAR (A1, A2, A3, S1, or similar).

To switch:

1. In `real_robot.launch.py`, replace the `ldlidar_node` `Node()` block with an appropriate
   `sllidar_ros2` `Node()` block (or use one of the package launch files).
2. Set the correct `serial_port` and `serial_baudrate` for the new sensor model.
3. **Critically:** set `frame_id` to `'lidar_link'` — this is the frame used in the ubot URDF
   (`ubot_lidar.urdf.xacro`) and confirmed in the captured TF tree.  The driver's default
   `'laser_frame'` does **not** exist in the ubot TF tree and will cause all costmap observations
   to be silently dropped.

> ⚠️ **frame_id warning:** If you launch `sllidar_node` without explicitly setting
> `frame_id: 'lidar_link'`, the published `/scan` messages will carry `frame_id: 'laser_frame'`.
> `tf2` cannot transform this to `odom` or `base_footprint`, so Nav2 costmaps will discard all
> scan observations and the robot will navigate blind.  Always pass `frame_id: 'lidar_link'`
> when integrating this driver into the ubot stack.

## Known Issues

- This driver is not wired into `real_robot.launch.py`.  Running it alongside the current
  `ldlidar_node` would create two publishers on the `/scan` topic, confusing Nav2.
- The `sllidar_a1_launch.py` default `frame_id` is `'laser'` (not `'laser_frame'` or
  `'lidar_link'`); each model launch file may use a different default.  Always verify and override.

## See Also

- [`../nodes/ldlidar_stl_ros2_node.md`](ldlidar_stl_ros2_node.md) — the currently active LiDAR driver
- [`../packages/sllidar_ros2.md`](../packages/sllidar_ros2.md) — full package page
- [`../hardware/sensors.md`](../hardware/sensors.md) — physical sensor selection and wiring
- [`../launch/real_robot.launch.py.md`](../launch/real_robot.launch.py.md) — where the active LiDAR node is launched
