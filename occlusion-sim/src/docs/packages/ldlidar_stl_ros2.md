# ldlidar_stl_ros2

Vendored LiDAR driver from SHENZHEN LDROBOT CO., LTD. This is the **live, active** driver on the real robot, used to read data from the LD19 LiDAR unit.

## Overview

| Field | Value |
|---|---|
| Package name | `ldlidar_stl_ros2` |
| Version | `3.0.0` |
| License | MIT |
| Build type | `ament_cmake` |
| Upstream maintainer | SHENZHEN LDROBOT CO., LTD. (support@ldrobot.com) |
| Real robot device | `/dev/ttyUSB1` at 230400 baud |
| LiDAR model in use | LD19 (product_name: `LDLiDAR_LD19`) |

## Role in the Real Robot

`ldlidar_stl_ros2` is the active scan source in `real_robot.launch.py`. The `sllidar_ros2` package is present in the workspace but is **not** used. See [sllidar_ros2](sllidar_ros2.md) for details.

The node is launched directly from `real_robot.launch.py` with inline parameters — it does **not** use any launch file from within this package itself.

## Executable

Built from `src/demo.cpp` + the bundled `ldlidar_driver/` C++ library:

```
ldlidar_stl_ros2_node
```

Install path: `lib/ldlidar_stl_ros2/ldlidar_stl_ros2_node`

## Topics

| Topic | Message Type | QoS | Description |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | ⚠️ Could not be determined from source — requires runtime inspection | 360° 2D laser scan from the LD19. Frame ID is `lidar_link` as configured in `real_robot.launch.py`. |

## Parameters (as used in real_robot.launch.py)

These are the exact parameter values passed inline in `real_robot.launch.py` to the `ldlidar_node`:

| Parameter | Type | Value (real robot) | Description |
|---|---|---|---|
| `product_name` | string | `LDLiDAR_LD19` | Selects the driver variant / protocol for the LD19 model. |
| `topic_name` | string | `scan` | Published topic name (no leading `/`; becomes `/scan`). |
| `frame_id` | string | `lidar_link` | TF frame ID stamped on every scan message. Matches URDF `lidar_link`. |
| `port_name` | string | `/dev/ttyUSB1` | Serial device for the LiDAR. |
| `port_baudrate` | int | `230400` | Serial baud rate. |
| `laser_scan_dir` | bool | `True` | `True` = counterclockwise scan direction. |
| `enable_angle_crop_func` | bool | `False` | Disable angle cropping — full 360° scan published. |
| `angle_crop_min` | float | `0.0` | Minimum crop angle in degrees (unused; `enable_angle_crop_func=False`). |
| `angle_crop_max` | float | `0.0` | Maximum crop angle in degrees (unused). |

The node name in `real_robot.launch.py` is `ldlidar_node`.

## How Parameters Are Passed (Important)

`real_robot.launch.py` instantiates the node directly:

```python
ldlidar_node = Node(
    package='ldlidar_stl_ros2',
    executable='ldlidar_stl_ros2_node',
    name='ldlidar_node',
    parameters=[
        {'product_name': 'LDLiDAR_LD19'},
        {'topic_name': 'scan'},
        {'frame_id': 'lidar_link'},
        {'port_name': '/dev/ttyUSB1'},
        {'port_baudrate': 230400},
        {'laser_scan_dir': True},
        {'enable_angle_crop_func': False},
        {'angle_crop_min': 0.0},
        {'angle_crop_max': 0.0},
    ]
)
```

None of the six launch files bundled in this package are used by the ubot bringup — they all default to `/dev/ttyUSB0` and a different TF frame (`base_laser`), which would conflict with the URDF's `lidar_link`.

## Bundled Launch Files

The package ships six launch files under `launch/`. None of these are invoked by `real_robot.launch.py`.

| File | Target model | Notes |
|---|---|---|
| `ld06.launch.py` | LD06 | Driver node + `base_link → base_laser` static TF. Port: `/dev/ttyUSB0`. |
| `ld19.launch.py` | LD19 | Driver node + `base_link → base_laser` static TF. Port: `/dev/ttyUSB0`, frame: `base_laser` — differs from ubot deployment (`lidar_link`). |
| `stl27l.launch.py` | STL27L | Driver node for STL27L model. |
| `viewer_ld06.launch.py` | LD06 | Driver + RViz2 viewer preset. |
| `viewer_ld19.launch.py` | LD19 | Driver + RViz2 viewer preset. |
| `viewer_stl27l.launch.py` | STL27L | Driver + RViz2 viewer preset. |

## Build

```
ament_cmake, C++14
Dependencies: rclcpp, sensor_msgs
Links: pthread
```

Driver source is in `ldlidar_driver/` and compiled from six subdirectory globs (core, dataprocess, filter, logger, networkcom, serialcom). Installed artifacts: the node binary and the `launch/` + `rviz2/` directories.

## Serial Port Assignment

On the real robot the two USB serial devices are assigned as follows:

| Device | Connection |
|---|---|
| `/dev/ttyUSB0` | ESP32 NodeMCU (motor controller) — used by `ubot_control` |
| `/dev/ttyUSB1` | LD19 LiDAR — used by `ldlidar_stl_ros2` |

There is no port conflict. The old audit issue claiming a conflict (#1) is resolved.

## Switching to a Different LiDAR

If this package is ever replaced with `sllidar_ros2` (e.g., when switching to an RPLiDAR model):

1. Update the `Node()` block in `real_robot.launch.py` to use `sllidar_ros2/sllidar_node`.
2. Update `frame_id` to match whatever frame the new driver publishes.
3. Update `lidar_link` in `ubot_lidar.urdf.xacro` if the physical mount changes.
4. Update `mapper_params_online_async.yaml` (`odom_frame` / scan source) if the topic name changes.

## See Also

- [sllidar_ros2](sllidar_ros2.md) — alternative RPLiDAR driver (not currently active)
- [ubot_bringup](ubot_bringup.md) — `real_robot.launch.py` where this node is instantiated
- [ubot_description](ubot_description.md) — `lidar_link` frame definition in the URDF
