# Sensors

This page covers each sensor on the ubot platform: its physical interface, ROS integration, URDF frame, and operational status.

---

## Summary Table

| Sensor | Physical interface | ROS topic(s) | Frame ID | Active on real robot? | Driver package |
|---|---|---|---|---|---|
| LDLidar LD19 | USB serial `/dev/ttyUSB1` @ 230400 | `/scan` | `lidar_link` | Yes | `ldlidar_stl_ros2` (v3.0.0, MIT) |
| BNO055 IMU | I2C | `bno055/imu`, `bno055/imu_raw`, `bno055/mag`, `bno055/grav`, `bno055/temp`, `bno055/calib_status` | `imu_link` | No (configured, not launched) | `bno055` (v0.5.0, BSD) |
| RGB-D camera | — | None on real robot | `camera_link`, `camera_depth_frame`, `camera_optical_frame`, `camera_rgb_frame` | No (URDF + Gazebo only) | None |
| Ultrasonic (×2) | — | None | `left_ultrasonic_link`, `right_ultrasonic_link` | No (URDF placeholder) | None |

---

## LDLidar LD19

**Status: ACTIVE on real robot**

### Physical

The LD19 is a 2D rotating laser rangefinder. It spins continuously and produces a full 360° scan at approximately 8 Hz with 720 evenly-distributed sample points per revolution.

| Specification | Value |
|---|---|
| Type | 2D rotating LiDAR |
| Spin rate | ~8 Hz |
| Range | 12 m (typical) |
| Samples per scan | 720 |
| Field of view | 360° |

### ROS Integration

| Property | Value |
|---|---|
| Topic | `/scan` |
| Message type | `sensor_msgs/LaserScan` |
| Frame ID | `lidar_link` |
| Connection | `/dev/ttyUSB1` @ 230400 baud (USB serial) |
| Angle cropping | Disabled (`enable_angle_crop_func=False` in `real_robot.launch.py`) |

The node is launched as `ldlidar_node` in `real_robot.launch.py` with `product_name=LDLiDAR_LD19`, `laser_scan_dir=True`, `angle_crop_min=0.0`, `angle_crop_max=0.0`.

### URDF Position

Defined in `ubot_lidar.urdf.xacro`. The lidar\_link is fixed to `base_link` at:

```
origin xyz="-0.1425  0.0  0.2154"
```

The sensor is rear-mounted and elevated above the chassis top. The 0.2154 m height is computed as `base_height + lidar_height/2` (0.2025 + 0.01290 = 0.21540 m).

### Driver

Package `ldlidar_stl_ros2` (vendored, v3.0.0, MIT). The package is cloned under `src/ldlidar_stl_ros2/`.

### Gazebo Model

When `use_gazebo:=true`, a `gpu_lidar` sensor is attached to `lidar_link` in `ubot_gazebo.urdf.xacro`. It publishes to `/scan` at 10 Hz with 720 samples, range 0.2–30 m, and Gaussian noise (stddev=0.01). The sim update rate (10 Hz) and max range (30 m) differ intentionally from the real hardware spec.

---

## BNO055 IMU

**Status: CONFIGURED, not active on real robot**

### Physical

The Bosch BNO055 is a 9-DOF absolute orientation sensor integrating a triaxial accelerometer, gyroscope, and magnetometer with an onboard ARM Cortex-M0 running sensor fusion.

| Specification | Value |
|---|---|
| Sensor | Bosch BNO055 |
| DOF | 9 (accel + gyro + mag) |
| Interface | I2C |
| Operation mode | NDOF (Bosch internal Madgwick/Kalman 9-DOF fusion) |
| Mount position | P2 (configured in `bno055_params.yaml` — applies axis-remap registers `0x24`, `0x06`) |

### ROS Integration

| Topic | Message type | Description |
|---|---|---|
| `bno055/imu` | `sensor_msgs/Imu` | Fused orientation quaternion (normalized by driver), fused linear acceleration, raw angular velocity |
| `bno055/imu_raw` | `sensor_msgs/Imu` | Raw accelerometer + gyroscope data; orientation\_covariance field reused (see known issues) |
| `bno055/mag` | `sensor_msgs/MagneticField` | Magnetometer readings |
| `bno055/grav` | `geometry_msgs/Vector3` | Gravity vector (no covariance field — message type limitation) |
| `bno055/temp` | `sensor_msgs/Temperature` | Onboard temperature |
| `bno055/calib_status` | `std_msgs/String` | JSON: `{"sys":N,"gyro":N,"accel":N,"mag":N}`, scale 0–3 per Bosch spec |

Service: `bno055/calibration_request` (`std_srvs/Trigger`) — switches to config mode, reads calibration offsets, switches back to NDOF, returns offsets as a string.

All topics use QoS depth=10. The topic prefix (`bno055/`) is configurable via the `ros_topic_prefix` parameter.

Frame ID for all topics: `imu_link`.

### URDF Position

Defined in `ubot_imu.urdf.xacro`. The imu\_link is fixed to `base_link` at:

```
origin xyz="-0.0900  -0.0465  0.18"
```

Values computed from xacro expressions: `imu_x = -(base_length/2 - 0.115)` = −0.090 m, `imu_y = -(0.165 - base_width/2)` = −0.04645 m, `imu_z = 0.18`.

### Driver

Package `bno055` (vendored, v0.5.0, BSD). Cloned under `src/bno055/`. The `SensorService` class handles I2C communication and publishes all topics; it is composed into `Bno055Node` (not a Node subclass itself).

Driver startup: on I2C communication failure in `configure()`, the driver calls `sys.exit(1)` immediately with no retry. An invalid `placement_axis_remap` value in the params YAML would raise an unhandled `KeyError` (see `issues.md` issue #10).

### Status and How to Enable

The `bno055_node` is **defined** in `real_robot.launch.py` (lines ~101–107) with params loaded from `ubot_bringup/config/bno055_params.yaml`. However, the line that appends it to the `LaunchDescription` is commented out (line 125: `# bno055_node,`).

The `ekf_node` (`robot_localization/ekf_node`) that would fuse IMU yaw rate with wheel odometry into `/odometry/filtered` is also fully commented out.

To enable the IMU on the real robot:

1. Uncomment the `bno055_node` append line in `real_robot.launch.py`.
2. Optionally uncomment the `ekf_node` block to enable IMU-fused odometry.
3. If enabling the EKF, update the Nav2 odom topic from `/diff_drive_controller/odom` to `/odometry/filtered` (see `real_ekf.yaml`).

### Gazebo Model

When `use_gazebo:=true`, an `imu` sensor is attached to `imu_link` in `ubot_gazebo.urdf.xacro`. It publishes to `/imu` at 100 Hz. Note the sim topic (`/imu`) differs from the real driver topic prefix (`bno055/imu`).

---

## RGB-D Camera

**Status: URDF and Gazebo only — no real robot driver**

### Physical

The physical model could not be determined from source. The camera URDF comment references an Intel RealSense D435 mass value (0.072 kg) and the file has been designed to match a depth camera form factor (90 mm wide, 25 mm tall), but this is not confirmed in any launch file or configuration. Requires runtime/hardware inspection.

### URDF Frames

Defined in `ubot_camera.urdf.xacro`. The camera\_link is fixed to `base_link` at:

```
origin xyz="0.2050  0.0  0.1626"
```

Where `camera_x = base_length/2 = 0.41/2 = 0.205` m — front-facing, at the forward edge of the base\_link.

| Frame | Parent | Joint | Origin (relative to parent) |
|---|---|---|---|
| `camera_link` | `base_link` | `camera_joint` (fixed) | (0.205, 0, 0.1626) |
| `camera_depth_frame` | `camera_link` | `camera_depth_joint` (fixed) | (0, 0, 0) — same as camera\_link |
| `camera_optical_frame` | `camera_depth_frame` | `camera_optical_joint` (fixed) | (0, 0, 0) |
| `camera_rgb_frame` | `camera_link` | `camera_rgb_joint` (fixed) | (0, 0, 0) rpy=(−π/2, 0, −π/2) |

`camera_depth_frame` follows ROS REP-103 (X right, Y down, Z forward). `camera_rgb_frame` is rotated `(−π/2, 0, −π/2)` for standard camera optical convention. All static TF frames are published by `robot_state_publisher` on every launch.

### ROS Integration

No camera driver node exists in any launch file. Static TF frames are broadcast but no image, depth, or point-cloud topics are published on the real robot.

### Gazebo Model

When `use_gazebo:=true`, an `rgbd_camera` sensor is attached to `camera_link` in `ubot_gazebo.urdf.xacro`:

| Parameter | Value |
|---|---|
| Update rate | 30 Hz |
| Image size | 640 × 480 px (R8G8B8) |
| Horizontal FOV | 1.211 rad |
| Near / far clip | 0.1 m / 10.0 m |
| Noise | Gaussian (mean=0.0, stddev=0.007) |
| Frame ID | `camera_depth_frame` |

---

## Ultrasonic Sensors

**Status: URDF placeholder only**

### Physical

Two links are defined in `ubot_ultrasonic.urdf.xacro` corresponding to HC-SR04-style ultrasonic modules (8 g, 50 × 14 × 20 mm). The comment in the file (`HC-SR04 module ~8 g`) suggests the HC-SR04 as the intended device.

| Link | Parent | Joint | Position (x, y, z) |
|---|---|---|---|
| `left_ultrasonic_link` | `base_link` | `left_ultrasonic_joint` (fixed) | (−0.0476, 0.0917, 0.0370) |
| `right_ultrasonic_link` | `base_link` | `right_ultrasonic_joint` (fixed) | (−0.0476, −0.0917, 0.0370) |

Both use the `ultrasonic.stl` mesh for visualization. The `right_ultrasonic_link` visual origin is offset `(0.040, 0.057, 0.019)` with rpy `(0, 0, 0)` while the left is `(−0.040, −0.057, 0.019)` with rpy `(0, 0, π)` — mirrored.

Note: these link names do **not** use the `prefix` argument (a minor inconsistency in the xacro macro).

### ROS Integration

None. There is no ROS driver node, no published topic, and the sensors are not referenced in Nav2 costmap observation sources. They are not wired into any part of the active software stack.

### Status

URDF-only placeholder. No evidence of active hardware or software integration exists in the workspace. See `gaps.md` for tracking.
