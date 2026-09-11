# ldlidar_stl_ros2 Launch Files

The `ldlidar_stl_ros2` package ships 6 launch files covering 3 LiDAR models, each with a plain variant and a viewer variant (plain + RViz2). These are vendored launch files from LDROBOT and are **not used by `real_robot.launch.py`** — the main bringup launches the LiDAR node directly with inline parameters instead.

Source directory: `/home/chibueze/uni-bot/src/ldlidar_stl_ros2/launch/`

## Launch file reference

| Launch file | Target model | `product_name` | Port default | Baud default | Scan dir | Also opens |
|---|---|---|---|---|---|---|
| `ld06.launch.py` | LD06 | `LDLiDAR_LD06` | `/dev/ttyUSB0` | 230400 | `True` (CCW) | nothing |
| `ld19.launch.py` | LD19 | `LDLiDAR_LD19` | `/dev/ttyUSB0` | 230400 | `True` (CCW) | nothing |
| `stl27l.launch.py` | STL27L | `LDLiDAR_STL27L` | `/dev/ttyUSB0` | 921600 | `False` (CW) | nothing |
| `viewer_ld06.launch.py` | LD06 | *(includes ld06.launch.py)* | `/dev/ttyUSB0` | 230400 | `True` | RViz2 |
| `viewer_ld19.launch.py` | LD19 | *(includes ld19.launch.py)* | `/dev/ttyUSB0` | 230400 | `True` | RViz2 |
| `viewer_stl27l.launch.py` | STL27L | *(includes stl27l.launch.py)* | `/dev/ttyUSB0` | 921600 | `False` | RViz2 |

## Common parameter pattern

Every plain launch file (ld06, ld19, stl27l) launches two nodes:

1. **`ldlidar_stl_ros2_node`** — the LiDAR driver, with these parameters:

| Parameter | Description | LD06 default | LD19 default | STL27L default |
|---|---|---|---|---|
| `product_name` | Model identifier string | `LDLiDAR_LD06` | `LDLiDAR_LD19` | `LDLiDAR_STL27L` |
| `topic_name` | Published scan topic | `scan` | `scan` | `scan` |
| `frame_id` | TF frame for scan | `base_laser` | `base_laser` | `base_laser` |
| `port_name` | Serial device path | `/dev/ttyUSB0` | `/dev/ttyUSB0` | `/dev/ttyUSB0` |
| `port_baudrate` | Serial baud rate | `230400` | `230400` | `921600` |
| `laser_scan_dir` | `True`=CCW, `False`=CW | `True` | `True` | `False` |
| `enable_angle_crop_func` | Enable angle masking | `False` | `False` | `False` |
| `angle_crop_min` | Mask start angle (deg) | `135.0` | `135.0` | `0.0` |
| `angle_crop_max` | Mask end angle (deg) | `225.0` | `225.0` | `0.0` |

2. **`static_transform_publisher`** (tf2_ros) — broadcasts a fixed transform `base_link → base_laser` at `[0, 0, 0.18]` (18 cm height, no rotation).

   Node names: `base_link_to_base_laser_ld06`, `base_link_to_base_laser_ld19`, `base_link_to_base_laser_stl27l` respectively.

The viewer variants include the plain launch file and add an `rviz2` node with the package's bundled `ldlidar.rviz` config (`ldlidar_stl_ros2/rviz2/ldlidar.rviz`).

## Quick start (standalone testing)

```bash
# Test LD19 only (no RViz):
ros2 launch ldlidar_stl_ros2 ld19.launch.py

# Test LD19 with RViz visualisation:
ros2 launch ldlidar_stl_ros2 viewer_ld19.launch.py
```

## How real_robot.launch.py differs

`real_robot.launch.py` does **not** use these launch files. Instead it launches `ldlidar_stl_ros2_node` directly as a `Node()` with inline parameters. The differences from the vendor defaults are:

| Parameter | Vendor default (`ld19.launch.py`) | real_robot.launch.py |
|---|---|---|
| `frame_id` | `base_laser` | `lidar_link` |
| `port_name` | `/dev/ttyUSB0` | `/dev/ttyUSB1` |
| `angle_crop_min` | `135.0` | `0.0` |
| `angle_crop_max` | `225.0` | `0.0` |
| Static TF publisher | Yes (base_link → base_laser) | No (URDF handles base_link → lidar_link) |

The `frame_id` change (`lidar_link` vs `base_laser`) is the critical one: it must match the URDF link name. The URDF defines `lidar_link` at `[-0.1425, 0, 0.2147]` relative to `base_link`. If you use the vendor launch files for testing, the `/scan` messages will carry `frame_id: base_laser` which will be inconsistent with the robot's URDF TF tree (which has no `base_laser` frame).

## When to use these launch files

Use the vendor launch files when:

- **Standalone LiDAR testing** — you want to quickly verify the LiDAR hardware is functioning without bringing up the full robot stack. The viewer variants add RViz for immediate scan visualisation.
- **Different LiDAR model** — you have an LD06 or STL27L connected and want to confirm it works before modifying `real_robot.launch.py`.
- **Debugging scan data** — isolating the LiDAR from the controller stack to rule out interference.

Do not use them as a substitute for `real_robot.launch.py` — the `frame_id` and port differ, and there is no robot_state_publisher, controller_manager, or twist_stamper.

## Known issues

None specific to these vendor launch files. See [real_robot.launch.py](real_robot.md) for the production LiDAR configuration.

## See also

- [real_robot.launch.py](real_robot.md) — production LiDAR bringup with `frame_id: lidar_link`, port `/dev/ttyUSB1`
- [sllidar_launches.md](sllidar_launches.md) — RPLiDAR (sllidar_ros2) equivalent
- [Nodes: ldlidar_stl_ros2_node](../nodes/ldlidar_stl_ros2_node.md)
- [Hardware: LiDAR sensor](../hardware/sensors.md)
