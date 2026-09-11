# sllidar_ros2 Launch Files

The `sllidar_ros2` package ships 24 launch files covering 12 RPLiDAR models, each with a plain driver variant and an RViz viewer variant. This package is **not used by `real_robot.launch.py`** — the ubot currently uses an LDROBOT LD19 (`ldlidar_stl_ros2`). These files are present because the package was vendored as a potential alternative.

Source directory: `/home/chibueze/uni-bot/src/sllidar_ros2/launch/`

## Launch file reference

| Launch file | Target model | Default port | Default baud | Default scan mode | Channel |
|---|---|---|---|---|---|
| `sllidar_a1_launch.py` | RPLIDAR A1 | `/dev/ttyUSB0` | 115200 | Sensitivity | serial |
| `sllidar_a2m7_launch.py` | RPLIDAR A2 (M7) | `/dev/ttyUSB0` | 256000 | Sensitivity | serial |
| `sllidar_a2m8_launch.py` | RPLIDAR A2 (M8) | `/dev/ttyUSB0` | 115200 | Sensitivity | serial |
| `sllidar_a2m12_launch.py` | RPLIDAR A2 (M12) | `/dev/ttyUSB0` | 256000 | Sensitivity | serial |
| `sllidar_a3_launch.py` | RPLIDAR A3 | `/dev/ttyUSB0` | 256000 | Sensitivity | serial |
| `sllidar_c1_launch.py` | RPLIDAR C1 | `/dev/ttyUSB0` | 460800 | Standard | serial |
| `sllidar_s1_launch.py` | RPLIDAR S1 | `/dev/ttyUSB0` | 256000 | *(no scan_mode param)* | serial |
| `sllidar_s1_tcp_launch.py` | RPLIDAR S1 (TCP) | *(no serial port)* | *(N/A)* | Sensitivity | tcp (IP: 192.168.0.7, port: 20108) |
| `sllidar_s2_launch.py` | RPLIDAR S2 | `/dev/ttyUSB0` | 1000000 | DenseBoost | serial |
| `sllidar_s2e_launch.py` | RPLIDAR S2E | *(no serial port)* | *(N/A)* | Sensitivity | udp (IP: 192.168.11.2, port: 8089) |
| `sllidar_s3_launch.py` | RPLIDAR S3 | `/dev/ttyUSB0` | 1000000 | DenseBoost | serial |
| `sllidar_t1_launch.py` | RPLIDAR T1 | *(no serial port)* | *(N/A)* | Sensitivity | udp (IP: 192.168.11.2, port: 8089) |
| `view_sllidar_a1_launch.py` | RPLIDAR A1 | — | — | — | serial + RViz |
| `view_sllidar_a2m7_launch.py` | RPLIDAR A2 (M7) | — | — | — | serial + RViz |
| `view_sllidar_a2m8_launch.py` | RPLIDAR A2 (M8) | — | — | — | serial + RViz |
| `view_sllidar_a2m12_launch.py` | RPLIDAR A2 (M12) | — | — | — | serial + RViz |
| `view_sllidar_a3_launch.py` | RPLIDAR A3 | — | — | — | serial + RViz |
| `view_sllidar_c1_launch.py` | RPLIDAR C1 | — | — | — | serial + RViz |
| `view_sllidar_s1_launch.py` | RPLIDAR S1 | — | — | — | serial + RViz |
| `view_sllidar_s1_tcp_launch.py` | RPLIDAR S1 (TCP) | — | — | — | tcp + RViz |
| `view_sllidar_s2_launch.py` | RPLIDAR S2 | — | — | — | serial + RViz |
| `view_sllidar_s2e_launch.py` | RPLIDAR S2E | — | — | — | udp + RViz |
| `view_sllidar_s3_launch.py` | RPLIDAR S3 | — | — | — | serial + RViz |
| `view_sllidar_t1_launch.py` | RPLIDAR T1 | — | — | — | udp + RViz |

## Common parameter pattern

All serial-connected launch files declare the same set of `DeclareLaunchArgument` entries and pass them to a single `sllidar_node`:

| Argument | Description | Default (model-dependent) |
|---|---|---|
| `channel_type` | Connection type (`serial`, `tcp`, `udp`) | `serial` |
| `serial_port` | USB device path | `/dev/ttyUSB0` |
| `serial_baudrate` | Baud rate | Model-specific (see table above) |
| `frame_id` | TF frame for published scans | `laser` |
| `inverted` | Invert scan data (`true`/`false`) | `false` |
| `angle_compensate` | Enable angle compensation | `true` |
| `scan_mode` | Scan quality mode (Sensitivity / Standard / DenseBoost) | Model-specific |

Network-connected variants (S1 TCP, S2E UDP, T1 UDP) replace `serial_port` / `serial_baudrate` with `tcp_ip` / `tcp_port` or `udp_ip` / `udp_port`.

All variants publish a single topic: `/scan` (`sensor_msgs/LaserScan`) with `frame_id: laser`.

## Quick start

```bash
# Run A1 (serial, 115200 baud):
ros2 launch sllidar_ros2 sllidar_a1_launch.py

# Run A1 with custom port and RViz:
ros2 launch sllidar_ros2 view_sllidar_a1_launch.py serial_port:=/dev/ttyUSB1

# Run S2 (serial, 1 Mbit/s):
ros2 launch sllidar_ros2 sllidar_s2_launch.py
```

## sllidar_ros2 is NOT used by real_robot.launch.py

The physical ubot uses an **LDROBOT LD19**, which is driven by `ldlidar_stl_ros2`. The `sllidar_ros2` package handles **Slamtec RPLiDAR** units (A-series, C-series, S-series, T-series). These are different hardware from different manufacturers.

The `sllidar_ros2` package is present in the workspace as a vendored dependency available for future use. No `sllidar_node` appears in any of the ubot bringup launch files.

## When you would switch to sllidar_ros2

If the physical LD19 LiDAR were replaced with a Slamtec RPLiDAR unit, the following changes would be required:

### 1. real_robot.launch.py

Replace the `ldlidar_stl_ros2_node` block with an appropriate `sllidar_node` block:

```python
# Remove:
ldlidar_node = Node(
    package='ldlidar_stl_ros2',
    executable='ldlidar_stl_ros2_node',
    name='ldlidar_node',
    ...
)

# Add:
sllidar_node = Node(
    package='sllidar_ros2',
    executable='sllidar_node',
    name='sllidar_node',
    parameters=[{
        'channel_type': 'serial',
        'serial_port': '/dev/ttyUSB1',       # keep same USB port
        'serial_baudrate': 115200,            # match your RPLiDAR model
        'frame_id': 'lidar_link',             # CRITICAL: match URDF frame name
        'inverted': False,
        'angle_compensate': True,
        'scan_mode': 'Sensitivity'
    }]
)
```

### 2. URDF — no change needed for frame_id

The URDF already defines `lidar_link` at the correct physical position. The only change required in the URDF would be if the physical mounting position or orientation of the new sensor differs from the LD19.

### 3. SLAM Toolbox config (mapper_params_online_async.yaml)

The `scan_topic` field defaults to `/scan` which both drivers publish to — no change needed.

Check `max_laser_range` against the new sensor's rated range (LD19 ≈ 12 m; RPLiDAR A1 ≈ 12 m; A3 ≈ 25 m; S2 ≈ 30 m).

### 4. Verify frame_id is consistent

The sllidar launch files default to `frame_id: laser`, not `lidar_link`. When launching sllidar via `real_robot.launch.py`, always explicitly set `'frame_id': 'lidar_link'` to keep the scan consistent with the URDF TF tree. If you use a standalone sllidar vendor launch file for testing, you will need to either pass `frame_id:=lidar_link` or expect a TF frame mismatch in the full stack.

## Known issues

None specific to the sllidar_ros2 vendor launch files. The package is correctly vendored and functional for its intended hardware.

## See also

- [ldlidar_launches.md](ldlidar_launches.md) — the active LiDAR driver (LDROBOT LD19)
- [real_robot.launch.py](real_robot.md) — production bringup using ldlidar_stl_ros2
- [Nodes: sllidar_node](../nodes/sllidar_node.md)
- [Hardware: LiDAR sensor](../hardware/sensors.md)
