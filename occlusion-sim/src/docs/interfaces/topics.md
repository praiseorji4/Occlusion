# Topic Reference

This page documents every confirmed ROS 2 topic in the ubot workspace. Topics are grouped by subsystem. The "Active?" column reflects a default `ros2 launch ubot_bringup real_robot.launch.py` run — topics marked inactive exist in config or code but are not published or subscribed by any node in that launch.

---

## Active nodes on a default real-robot launch

The following nodes are started by `real_robot.launch.py`:

| Node | Package |
|---|---|
| `robot_state_publisher` | robot_state_publisher |
| `ros2_control_node` (controller_manager) | controller_manager |
| `joint_state_broadcaster` (spawned at t+5 s) | joint_state_broadcaster |
| `diff_drive_controller` (spawned at t+6 s) | diff_drive_controller |
| `twist_stamper` | twist_stamper |
| `ldlidar_node` | ldlidar_stl_ros2 |

Nodes that are **not** started by default: `bno055_node` (defined but commented out), `ekf_filter_node` (fully commented out block), `diag_publisher` (no entry in this launch file at all — must be run manually).

---

## Section 1: Drive / control topics

These topics are active on a default real-robot launch once both controller spawners have completed (approximately 6 seconds after launch).

| Topic | Message Type | Publisher | Subscriber(s) | QoS | Active? |
|---|---|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/Twist` | External (Nav2 collision_monitor, teleop, etc.) | `twist_stamper` | Default (reliable, depth 10) | Yes — input side of twist_stamper |
| `/diff_drive_controller/cmd_vel` | `geometry_msgs/TwistStamped` | `twist_stamper` (remapped from `/cmd_vel_out`) | `diff_drive_controller` | Default | Yes |
| `/diff_drive_controller/odom` | `nav_msgs/Odometry` | `diff_drive_controller` | Nav2 bt_navigator, velocity_smoother, ekf_filter_node (when enabled) | Reliable, depth 10 | Yes |
| `/joint_states` | `sensor_msgs/JointState` | `joint_state_broadcaster` | `robot_state_publisher` | Reliable, depth 10 | Yes — all 4 wheel joints published; rear joints mirror front position/velocity |
| `/robot_description` | `std_msgs/String` | `robot_state_publisher` | `controller_manager`, rviz2 | Transient-local, depth 1 | Yes |
| `/tf` | `tf2_msgs/TFMessage` | `diff_drive_controller` (odom→base_footprint, ~30 Hz), `robot_state_publisher` (wheel joint TFs, ~15 Hz) | All TF listeners | Best-effort, depth 100 | Yes |
| `/tf_static` | `tf2_msgs/TFMessage` | `robot_state_publisher` | All TF listeners | Transient-local, reliable | Yes — static transforms: base_footprint→base_link, base_link→{camera_link, imu_link, lidar_link} and camera sub-frames |

**Notes:**

- `/cmd_vel` carries an **unstamped** `Twist`. The `twist_stamper` node converts it to a stamped `TwistStamped` (frame_id = `base_footprint`) on `/diff_drive_controller/cmd_vel` as required by the diff_drive_controller. The remapping inside real_robot.launch.py is: `cmd_vel_in` → `/cmd_vel`, `cmd_vel_out` → `/diff_drive_controller/cmd_vel`.
- Wheel joint TF broadcast rate observed at ~15.26 Hz in a captured TF snapshot despite `controller_manager.update_rate: 30 Hz` and `diff_drive_controller.publish_rate: 30.0`. Root cause is not determined from source alone — flag for runtime inspection.
- The `odom` frame is broadcast at ~30.27 Hz (matching configured rate), confirming `diff_drive_controller` with `enable_odom_tf: true` is the active odom→base_footprint broadcaster.
- `/ubot/diagnostics` (`std_msgs/Float32MultiArray`) is an **optional** topic published by `UbotHardware` when `diag_publish_rate` is non-zero. It is disabled by default (`diag_publish_rate: 0` in `ubot_ros2_control.xacro`). Enable it by setting the parameter to a non-zero integer representing every Nth read cycle.

---

## Section 2: Sensing topics

> **Warning: BNO055 topics are inactive on a default real-robot launch.**
> `bno055_node` is defined in `real_robot.launch.py` but the line adding it to the `LaunchDescription` is commented out (line 125). None of the `/bno055/*` topics are published during a normal bringup run. See [issue #3](../issues.md) and the [BNO055 node page](../nodes/bno055_node.md).

| Topic | Message Type | Publisher | QoS | Frame ID | Active? |
|---|---|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | `ldlidar_node` (ldlidar_stl_ros2) | Default (reliable, depth 10) | `lidar_link` | Yes |
| `/bno055/imu_raw` | `sensor_msgs/Imu` | `bno055_node` | depth=10 | `imu_link` (from `frame_id` param) | No — bno055_node not in bringup launch |
| `/bno055/imu` | `sensor_msgs/Imu` | `bno055_node` | depth=10 | `imu_link` | No — bno055_node not in bringup launch |
| `/bno055/mag` | `sensor_msgs/MagneticField` | `bno055_node` | depth=10 | `imu_link` | No — bno055_node not in bringup launch |
| `/bno055/grav` | `geometry_msgs/Vector3` | `bno055_node` | depth=10 | N/A (Vector3 has no header) | No — bno055_node not in bringup launch |
| `/bno055/temp` | `sensor_msgs/Temperature` | `bno055_node` | depth=10 | `imu_link` | No — bno055_node not in bringup launch |
| `/bno055/calib_status` | `std_msgs/String` | `bno055_node` | depth=10 | N/A | No — bno055_node not in bringup launch |

**LiDAR (`/scan`) configuration details:**

- Sensor: LD19 (LDLiDAR_LD19), connected on `/dev/ttyUSB1` at 230400 baud.
- Frame ID: `lidar_link` (URDF origin: `[-0.1425, 0, 0.2147]` relative to `base_link`).
- Angle cropping disabled (`enable_angle_crop_func: False`).
- Nav2 local costmap and global costmap both subscribe to `/scan`.
- SLAM Toolbox also subscribes to `/scan` for mapping.

**BNO055 topic details (when `bno055_node` is running):**

- Topic prefix is configurable via the `ros_topic_prefix` parameter (default `'bno055/'`).
- `/bno055/imu_raw`: raw accelerometer + gyroscope data; no sensor fusion. `orientation_covariance` field is set from `variance_orientation` parameter (upstream TODO notes this field's use here is questionable).
- `/bno055/imu`: fusion output with quaternion orientation. Quaternion normalization is done by hand (upstream TODO: replace with standard `normalize()` function).
- `/bno055/grav`: gravity vector only, published as `geometry_msgs/Vector3` (no header — message type limitation, not a bug).
- `/bno055/calib_status`: JSON string with keys `sys`, `gyro`, `accel`, `mag`, each 0–3 (0 = uncalibrated, 3 = fully calibrated). Published on every sensor read cycle.
- EKF configuration (`real_ekf.yaml`) is set up to fuse `/bno055/imu` (yaw angular velocity only) with `/diff_drive_controller/odom`, but the EKF node is currently not launched.

---

## Section 3: Localisation / navigation topics

These topics are active only when the relevant Nav2 or SLAM nodes are running. None are started by `real_robot.launch.py` itself — they require a separate `ros2 launch` invocation.

| Topic | Message Type | Publisher | Notes |
|---|---|---|---|
| `/odometry/filtered` | `nav_msgs/Odometry` | `ekf_filter_node` (robot_localization) | Inactive — `ekf_node` block is entirely commented out in `real_robot.launch.py`. Would fuse wheel odometry + IMU yaw. `sim_nav2_params.yaml` uses this as its odom source; real robot nav2 uses `/diff_drive_controller/odom` directly. |
| `/map` | `nav_msgs/OccupancyGrid` | SLAM Toolbox (mapping mode) or map_server (localisation mode) | Published when SLAM or nav2 map_server is running. Not active during plain bringup. SLAM Toolbox config: `mapper_params_online_async.yaml`, resolution 0.05 m, max laser range 12 m. |
| `/plan` | `nav_msgs/Path` | `planner_server` (Nav2) | Published when Nav2 is running and a navigation goal is active. NavFn planner with A*, tolerance 0.25 m. |
| `/cmd_vel_smoothed` | `geometry_msgs/Twist` | `velocity_smoother` (Nav2) | Output of velocity smoother. In Nav2 collision_monitor config, this is the `cmd_vel_in_topic`; collision_monitor then publishes the final safe velocity on `/cmd_vel`. Feedback mode: `CLOSED_LOOP` (uses `/diff_drive_controller/odom`). |

**Notes:**

- The full Nav2 topic graph (BT navigator, controller, planner, recovery servers) is documented in [architecture/ros_graph.md](../architecture/ros_graph.md). Standard Nav2 topics not listed here are unchanged from upstream Nav2 defaults.
- When Nav2 is active, `/cmd_vel` is driven by the collision_monitor node (which reads `/cmd_vel_smoothed` from Nav2, applies safety checks, and re-publishes on `/cmd_vel`).

---

## Section 4: Diagnostic topics (manual run only, not in bringup)

These 14 topics are published exclusively by `diag_publisher` (`ubot_debugger` package, node name `horizon_diag_publisher`). This node is **not** included in any launch file and must be started manually:

```bash
ros2 run ubot_debugger diag_publisher \
  --ros-args -p serial_port:=/dev/ttyUSB0 -p publish_rate:=10.0
```

> **Caution:** `diag_publisher` uses the same serial port as `UbotHardware` (`/dev/ttyUSB0`). Both cannot hold the port simultaneously. Run `diag_publisher` only when `ros2_control_node` / `controller_manager` is **not** running, or use the ESP32 Serial2 diagnostic stream on `/dev/ttyUSB1` with PlotJuggler instead.
>
> Keep `publish_rate` at or below 15 Hz to avoid saturating the serial command channel (code comment in `diag_publisher.py`).

All 14 topics use message type `std_msgs/Float64`, QoS depth=10.

| Topic | Value description |
|---|---|
| `/horizon/diag/enc/left` | Raw encoder tick count, left front wheel (cumulative, reset by `r` command) |
| `/horizon/diag/enc/right` | Raw encoder tick count, right front wheel (cumulative) |
| `/horizon/diag/rpm/left_target` | PID target RPM, left wheel (computed from ticks-per-frame setpoint) |
| `/horizon/diag/rpm/right_target` | PID target RPM, right wheel |
| `/horizon/diag/rpm/left_actual` | Low-pass-filtered actual RPM, left (Jimeno filter: `0.854*prev + 0.0728*raw + 0.0728*raw_prev`) |
| `/horizon/diag/rpm/right_actual` | Low-pass-filtered actual RPM, right |
| `/horizon/diag/pid/left_error` | RPM tracking error (target − actual), left |
| `/horizon/diag/pid/right_error` | RPM tracking error (target − actual), right |
| `/horizon/diag/pid/left_integral` | PID integral accumulator, left (anti-windup clamped ±50.0) |
| `/horizon/diag/pid/right_integral` | PID integral accumulator, right (anti-windup clamped ±50.0) |
| `/horizon/diag/pid/left_output` | Final PWM output value, left (range −255 to +255) |
| `/horizon/diag/pid/right_output` | Final PWM output value, right (range −255 to +255) |
| `/horizon/diag/vel/linear` | Computed body linear velocity in m/s: `(vL + vR) / 2` |
| `/horizon/diag/vel/angular` | Computed body angular velocity in rad/s: `(vR − vL) / wheel_separation` |

These values are sourced from the ESP32 `q` command diagnostic snapshot, which is the same 14-field `"D ..."` frame emitted continuously on Serial2 (GPIO16). The `diag_publisher` node polls `q` via the Serial1 (USB) command channel at the configured rate and republishes each field as a separate Float64 topic, making them individually plottable in PlotJuggler.

**Source files:**
- `src/ubot/ubot_debugger/ubot_debugger/diag_publisher.py` — ROS 2 node
- `src/Ros-esp32_bridge/diff_controller.h` — `emitDiagnostics()` function defines the exact 14-field format
