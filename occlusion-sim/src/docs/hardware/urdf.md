# URDF Reference

The robot description is defined as a set of xacro files under `src/ubot/ubot_description/urdf/`.
The top-level entry point is `body/ubot_robot.urdf.xacro`.

---

## Top-Level xacro Arguments

Defined in `ubot_robot.urdf.xacro` (`<robot name="ubot">`):

| Argument | Default | Description |
|---|---|---|
| `robot_name` | `ubot_robot` | Robot instance name (passed through but not used to prefix links in the current macros) |
| `prefix` | `""` | Link/joint name prefix, passed to all macros. Empty on the real robot; used for multi-robot namespacing. |
| `use_gazebo` | `false` | When `true`: activates `gz_ros2_control/GazeboSimSystem` hardware plugin and includes all Gazebo sensor plugins. When `false`: activates `ubot_control/UbotHardware` real-hardware plugin. |

**Include order** (in file): `ubot_base`, `ubot_lidar`, `ubot_ultrasonic`, `ubot_camera`, `ubot_imu`, `ubot_wheel`, `ubot_ros2_control`, `ubot_gazebo`.

**Macro invocation order**: `ubot_ros2_control` (always) → `ubot_gazebo` (only if `use_gazebo:=true`) → `ubot_base` → `ubot_wheel` → `ubot_lidar` → `ubot_ultrasonic` → `ubot_camera` → `ubot_imu`.

---

## Link and Joint Tree

All numeric origins are relative to the parent link. Values are computed from xacro property expressions; see individual xacro files for the raw formulas.

| Link | Parent Link | Joint Name | Joint Type | Origin (x, y, z) m | Mass | Geometry |
|---|---|---|---|---|---|---|
| `base_footprint` | — | — | root | — | — | virtual frame |
| `base_link` | `base_footprint` | `base_joint` | fixed | (0, 0, 0.0142) | 3.0 kg | box 0.41 × 0.2371 × 0.2025 m |
| `front_left_wheel_link` | `base_link` | `front_left_wheel_joint` | continuous | (0.0993, 0.1321, 0.0188) | 0.4 kg | cylinder r=0.033 m, l=0.0264 m |
| `front_right_wheel_link` | `base_link` | `front_right_wheel_joint` | continuous | (0.0993, −0.1321, 0.0188) | 0.4 kg | cylinder r=0.033 m, l=0.0264 m |
| `rear_left_wheel_link` | `base_link` | `rear_left_wheel_joint` | continuous | (−0.1485, 0.1321, 0.0188) | 0.4 kg | cylinder r=0.033 m, l=0.0264 m |
| `rear_right_wheel_link` | `base_link` | `rear_right_wheel_joint` | continuous | (−0.1485, −0.1321, 0.0188) | 0.4 kg | cylinder r=0.033 m, l=0.0264 m |
| `lidar_link` | `base_link` | `lidar_joint` | fixed | (−0.1425, 0, 0.2154) | 0.170 kg | cylinder r=0.0175 m, l=0.0129 m |
| `imu_link` | `base_link` | `imu_joint` | fixed | (−0.0900, −0.0465, 0.18) | 0.050 kg | box 0.0259 × 0.0155 × 0.015 m |
| `camera_link` | `base_link` | `camera_joint` | fixed | (0.2050, 0, 0.1626) | 0.072 kg | box 0.025 × 0.090 × 0.025 m |
| `camera_depth_frame` | `camera_link` | `camera_depth_joint` | fixed | (0, 0, 0) | — | virtual frame |
| `camera_optical_frame` | `camera_depth_frame` | `camera_optical_joint` | fixed | (0, 0, 0) | — | virtual frame |
| `camera_rgb_frame` | `camera_link` | `camera_rgb_joint` | fixed | (0, 0, 0) rpy=(−π/2, 0, −π/2) | — | virtual frame |
| `left_ultrasonic_link` | `base_link` | `left_ultrasonic_joint` | fixed | (−0.0476, 0.0917, 0.0370) | 0.008 kg | mesh (ultrasonic.stl) |
| `right_ultrasonic_link` | `base_link` | `right_ultrasonic_joint` | fixed | (−0.0476, −0.0917, 0.0370) | 0.008 kg | mesh (ultrasonic.stl) |

**Notes:**
- `base_joint` z-offset = `wheel_radius − wheel_zoff` = 0.033 − 0.0188 = 0.0142 m. This positions `base_link` so that the wheel bottoms sit on the ground plane.
- All four wheel joint axes are `[0, 1, 0]` (roll about Y).
- `camera_depth_frame` follows ROS REP-103 convention (X right, Y down, Z forward). `camera_rgb_frame` is rotated `(−π/2, 0, −π/2)` relative to `camera_link` to match a standard camera optical orientation.
- `left_ultrasonic_link` and `right_ultrasonic_link` do not use the `prefix` argument in their link names (a minor inconsistency in `ubot_ultrasonic.urdf.xacro`).

---

## `ros2_control` Tag

Defined in `ubot_ros2_control.xacro`, macro `ubot_ros2_control(prefix, use_gazebo)`.

> **Structural rule (from file header comment):** `<joint>` tags must be **siblings** of `<hardware>`, not children of it. The Gazebo plugin block lives inside `<hardware>` as a `<plugin>` tag only.

### Hardware Plugin Selection

| Condition | Plugin |
|---|---|
| `use_gazebo:=false` (real hardware) | `ubot_control/UbotHardware` |
| `use_gazebo:=true` (simulation) | `gz_ros2_control/GazeboSimSystem` |

### Real Hardware Plugin Parameters (`ubot_control/UbotHardware`)

| Parameter | Value | Notes |
|---|---|---|
| `left_wheel_name` | `front_left_wheel_joint` | Must match URDF joint name exactly |
| `right_wheel_name` | `front_right_wheel_joint` | Must match URDF joint name exactly |
| `enc_counts_per_rev_left` | `3956` | Must match `ENC_CPR_LEFT` in `diff_controller.h` |
| `enc_counts_per_rev_right` | `3956` | Must match `ENC_CPR_RIGHT` in `diff_controller.h` |
| `wheel_separation` | `0.264204` | Must match `WHEEL_SEPARATION_M` in `diff_controller.h` |
| `wheel_radius` | `0.033` | Must match `WHEEL_RADIUS_M` in `diff_controller.h` |
| `serial_device` | `/dev/ttyUSB0` | Check `ls /dev/ttyUSB*` if assignment changes after reboot |
| `baud_rate` | `115200` | |
| `serial_timeout_ms` | `1000` | |
| `loop_rate` | `30` | Must match `PID_RATE` in `diff_controller.h` |
| `diag_publish_rate` | `0` | 0 = disabled; N = publish `/ubot/diagnostics` every N `read()` cycles |

### Joint Interface Table

| Joint | Real hardware command | Real hardware state | Sim command | Sim state |
|---|---|---|---|---|
| `front_left_wheel_joint` | velocity | position, velocity | velocity | position, velocity |
| `front_right_wheel_joint` | velocity | position, velocity | velocity | position, velocity |
| `rear_left_wheel_joint` | — (state only) | position, velocity | velocity | position, velocity |
| `rear_right_wheel_joint` | — (state only) | position, velocity | velocity | position, velocity |

On real hardware, `UbotHardware` mirrors the front wheel state into the rear wheel state interfaces (shared doubles in the C++ implementation). Rear joints receive no independent velocity commands on real hardware.

In simulation (`use_gazebo:=true`), `sim_ubot_controllers.yaml` lists all four joints in the diff\_drive\_controller wheel lists, because Gazebo provides a command interface for every joint.

---

## Gazebo Plugins

The following plugins are only active when `use_gazebo:=true`. They are defined in `ubot_gazebo.urdf.xacro`.

### gz\_ros2\_control

```xml
<plugin filename="gz_ros2_control-system"
        name="gz_ros2_control::GazeboSimROS2ControlPlugin">
  <parameters>$(find ubot_bringup)/config/sim_ubot_controllers.yaml</parameters>
</plugin>
```

Also requires the `gz-sim-sensors-system` plugin with `ogre2` render engine for sensor simulation.

### gpu\_lidar (LiDAR sensor)

Attached to `lidar_link`. Publishes to `/scan`.

| Parameter | Value |
|---|---|
| Topic | `/scan` |
| Update rate | 10 Hz |
| Horizontal samples | 720 |
| Min angle | −π rad |
| Max angle | +π rad |
| Min range | 0.2 m |
| Max range | 30.0 m |
| Range resolution | 0.01 m |
| Noise type | Gaussian (mean=0.0, stddev=0.01) |
| Frame ID | `lidar_link` |

Note: the sim LiDAR update rate (10 Hz) and max range (30 m) differ from the real LD19 hardware spec (~8 Hz, 12 m).

### rgbd\_camera

Attached to `camera_link`. Frame ID: `camera_depth_frame`.

| Parameter | Value |
|---|---|
| Topic | `camera` (prefixed) |
| Update rate | 30 Hz |
| Image width | 640 px |
| Image height | 480 px |
| Format | R8G8B8 |
| Horizontal FOV | 1.211 rad (~69°) |
| Near clip | 0.1 m |
| Far clip | 10.0 m |
| Noise type | Gaussian (mean=0.0, stddev=0.007) |

### IMU sensor

Attached to `imu_link`. Publishes to `/imu`.

| Parameter | Value |
|---|---|
| Topic | `/imu` |
| Update rate | 100 Hz |
| Angular velocity noise stddev | 0.009 rad/s (all axes) |
| Angular velocity bias mean | 0.00075 rad/s |
| Angular velocity bias stddev | 0.005 rad/s |
| Linear acceleration noise stddev | 0.017 m/s² (all axes) |
| Linear acceleration bias mean | 0.1 m/s² |
| Linear acceleration bias stddev | 0.001 m/s² |
| Frame ID | `imu_link` |

Note: the sim IMU topic is `/imu`; the real BNO055 driver publishes to `bno055/imu`.

### Wheel contact properties

All four wheel links receive `mu1=1.1`, `mu2=1.1`, `kp=1000000.0`, `kd=10.0`, `minDepth=0.001`, `maxVel=0.1`, `fdir1=[1,0,0]`.

---

## Dead File Note

`src/ubot/ubot_description/urdf/gazebo/gazebo_control.xacro` exists in the repository but is **not included** anywhere in the xacro include tree. It contains a standalone differential-drive plugin block that predates the current `gz_ros2_control` approach. It is unreferenced and has no effect. See `issues.md` issue #5.

---

## Previewing the URDF

```bash
# Launch RViz with joint state publisher GUI (requires installed workspace):
ros2 launch ubot_bringup display.launch.py

# Validate the expanded URDF without launching RViz:
xacro src/ubot/ubot_description/urdf/body/ubot_robot.urdf.xacro use_gazebo:=false \
  | check_urdf

# Expand with Gazebo plugins enabled:
xacro src/ubot/ubot_description/urdf/body/ubot_robot.urdf.xacro use_gazebo:=true \
  | check_urdf
```
