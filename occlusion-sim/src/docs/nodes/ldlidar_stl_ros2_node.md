# ldlidar_stl_ros2_node

## Overview

`ldlidar_stl_ros2_node` is the LiDAR driver node for LDROBOT LiDAR products (LD06, LD19, STL27L).
It is provided by the vendored `ldlidar_stl_ros2` package (v3.0.0, MIT licence, upstream:
SHENZHEN LDROBOT CO., LTD.).  No modifications have been made to this package in the ubot
workspace.

The **LD19** model is the sensor physically installed on this robot.  The node is launched directly
from `real_robot.launch.py` with inline parameters — it is **not** launched via the package's own
`ld19.launch.py`.

| Attribute | Value |
|---|---|
| ROS node name | `ldlidar_node` (set explicitly in `real_robot.launch.py`) |
| Executable | `ldlidar_stl_ros2_node` |
| Package | `ldlidar_stl_ros2` (v3.0.0, MIT) |
| C++ node name | `ldlidar_published` (from `rclcpp::Node("ldlidar_published")` in `demo.cpp`) |
| Serial port | `/dev/ttyUSB1` @ 230400 baud |

> **Note on naming:** The ROS node name `ldlidar_node` is assigned by `real_robot.launch.py`
> (`name='ldlidar_node'`), which overrides the name `LD19` used in the package's own
> `ld19.launch.py` and the internal C++ node name `ldlidar_published`.

## How it is Launched

The node is launched inline in `real_robot.launch.py` (entry #6 in the returned
`LaunchDescription`).  The full parameter block as set in `real_robot.launch.py`:

```python
Node(
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

## Parameters

All parameters are declared in `ldlidar_stl_ros2/src/demo.cpp`.

| Parameter | Type | Value in real_robot.launch.py | Notes |
|---|---|---|---|
| `product_name` | string | `LDLiDAR_LD19` | Selects LD type; valid values: `LDLiDAR_LD06`, `LDLiDAR_LD19`, `LDLiDAR_STL27L` |
| `topic_name` | string | `scan` | Name of the published LaserScan topic |
| `frame_id` | string | `lidar_link` | Must match the lidar link name in the URDF TF tree (see below) |
| `port_name` | string | `/dev/ttyUSB1` | Serial device for LD19 |
| `port_baudrate` | int | `230400` | LD19 native baud rate |
| `laser_scan_dir` | bool | `True` | `True` = counterclockwise; `False` = clockwise |
| `enable_angle_crop_func` | bool | `False` | If `True`, sets points in `[angle_crop_min, angle_crop_max]` to NaN |
| `angle_crop_min` | double | `0.0` | Degrees; only used when `enable_angle_crop_func: True` |
| `angle_crop_max` | double | `0.0` | Degrees; only used when `enable_angle_crop_func: True` |

### Additional parameters from source (demo.cpp defaults)

These parameters exist in the source but are not overridden in `real_robot.launch.py`:

| Parameter | Source default | Notes |
|---|---|---|
| `range_min` | `0.02` m | Hard-coded in `ToLaserscanMessagePublish()`, not a declared ROS parameter |
| `range_max` | `25` m | Hard-coded in `ToLaserscanMessagePublish()`, not a declared ROS parameter |
| Scan frequency | ⚠️ Could not be determined from source — requires runtime inspection | The LD19 spin frequency is hardware-determined and obtained via `ldlidarnode->GetLidarScanFreq()` at runtime |

The source publishes at a `rclcpp::WallRate` of 10 Hz, but scan messages are only emitted when
`ldlidarnode->GetLaserScanData()` returns `LidarStatus::NORMAL` — the effective topic rate matches
the hardware spin frequency of the LD19.

## Published Topics

| Topic | Message Type | QoS | frame_id |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | depth 10 | `lidar_link` |

The topic name is taken from the `topic_name` parameter (`'scan'`), which the node passes directly
to `node->create_publisher<sensor_msgs::msg::LaserScan>(topic_name, 10)`.

### TF frame requirement

The `frame_id` parameter must match the `lidar_link` frame broadcast in the URDF TF tree.  From
the captured TF tree (`frames_2026-06-26_17.33.00.gv`):

```
base_link -> lidar_link   (static)
```

The `ubot_lidar.urdf.xacro` places `lidar_link` at origin `[-0.1425, 0, 0.2147]` relative to
`base_link`.  The `real_robot.launch.py` parameter `frame_id=lidar_link` matches this — do not
change it without updating the URDF.

The vendored `ld19.launch.py` uses `frame_id: 'base_laser'` with a separate static TF publisher.
That launch file is **not** used in this workspace; `real_robot.launch.py` uses `lidar_link`
directly because the URDF already defines the correct static transform.

## Subscribed Topics

None.

## Services

None.

## Lifecycle / Operation

The node is a plain `rclcpp::Node` (not a lifecycle node).  Startup sequence:

1. Declares and reads all parameters.
2. Creates an `LDLidarDriver` instance and calls `Start(LD_19, port_name, port_baudrate, COMM_SERIAL_MODE)`.
3. Waits up to 3000 ms for communication connect; exits with error if timeout.
4. Enters a `WallRate(10)` loop calling `GetLaserScanData()` with a 1500 ms per-scan timeout.
5. On `LidarStatus::NORMAL`: calls `ToLaserscanMessagePublish()` which fills and publishes a
   `LaserScan` message with angle range `[0, 2π]`, `range_min=0.02`, `range_max=25`.
6. On `DATA_TIME_OUT`: logs an error and continues.

The node uses an internal filter algorithm enabled via `ldlidarnode->EnableFilterAlgorithnmProcess(true)`.

## Known Issues

- Serial port `/dev/ttyUSB1` is assigned to the LD19.  The ESP32 occupies `/dev/ttyUSB0`.  The
  USB enumeration order can change if devices are unplugged and replugged.  Use udev rules to
  assign stable symlinks (e.g. `/dev/ldlidar`, `/dev/esp32`) if port flipping becomes a problem.
- The node exits (via `exit(EXIT_FAILURE)`) if the serial port cannot be opened or if
  communication does not connect within 3 seconds.  There is no automatic retry.

## See Also

- [`../launch/real_robot.launch.py.md`](../launch/real_robot.launch.py.md) — full launch file documentation
- [`../packages/ldlidar_stl_ros2.md`](../packages/ldlidar_stl_ros2.md) — package page
- [`../hardware/sensors.md`](../hardware/sensors.md) — physical sensor details and wiring
- [`../nodes/sllidar_node.md`](sllidar_node.md) — the alternative Slamtec driver (not currently used)
