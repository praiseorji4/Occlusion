# real_ekf.yaml

## Purpose

`robot_localization` Extended Kalman Filter configuration for the **real robot**. When active, `ekf_filter_node` fuses wheel odometry from `/diff_drive_controller/odom` with IMU yaw rate from the BNO055 (`/bno055/imu`) to produce a filtered pose estimate on `/odometry/filtered`. This estimate is more robust to wheel slip than raw encoder odometry alone.

> **Current status — NOT LAUNCHED (Issue #3):** The `ekf_filter_node` block is fully commented out in `real_robot.launch.py` (lines ~110-116). The BNO055 node is also excluded from the launch (see `bno055_params.yaml` documentation). The robot currently navigates using raw wheel odometry only from `diff_drive_controller`. All configuration in this file is ready to enable but requires the associated launch file changes.

## Parameters

### ekf_filter_node

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `false` | Wall-clock time — correct for real hardware. |
| `frequency` | `30.0` Hz | EKF update and publish rate. Matches `controller_manager.update_rate` in `ubot_controllers.yaml` — one EKF update per control cycle. |
| `two_d_mode` | `true` | Constrains the filter to a 2D plane. Ignores z, roll, and pitch states. Appropriate for a ground robot. |
| `odom_frame` | `odom` | Name of the odometry frame. |
| `base_link_frame` | `base_footprint` | Name of the robot base frame. |
| `world_frame` | `odom` | World-fixed reference frame. Setting this to `odom` (not `map`) means the filter produces a continuous `odom → base_footprint` transform, not a map-aligned one. SLAM/AMCL handles `map → odom`. |
| `publish_tf` | `true` | EKF publishes the `odom → base_footprint` TF transform. **See TF conflict warning below.** |

### Odometry input (odom0)

| Parameter | Value | Description |
|---|---|---|
| `odom0` | `/diff_drive_controller/odom` | Wheel encoder odometry source. |
| `odom0_differential` | `false` | Uses absolute pose and velocity directly — not differential (change-per-step). |
| `odom0_relative` | `false` | Measurements are in the global/odom frame, not relative to previous pose. |

#### odom0_config — 15-element fusion mask

The 15 elements correspond to: [x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw, ax, ay, az]

```
odom0_config: [true,  false, false,
               false, false, true,
               true,  false, false,
               false, false, true,
               false, false, false]
```

| Index | State | Fused | Rationale |
|---|---|---|---|
| 0 | x | **yes** | Encoder x position is reliable. |
| 1 | y | no | Diff-drive cannot measure lateral position directly. |
| 2 | z | no | Ground robot — no vertical motion. |
| 3 | roll | no | `two_d_mode` — ignored. |
| 4 | pitch | no | `two_d_mode` — ignored. |
| 5 | yaw | **yes** | Encoder-derived heading is useful. |
| 6 | vx | **yes** | Forward velocity from encoders is reliable. |
| 7 | vy | no | Zero for diff-drive. |
| 8 | vz | no | Ground robot. |
| 9 | vroll | no | |
| 10 | vpitch | no | |
| 11 | vyaw | **yes** | Angular velocity from encoders. |
| 12 | ax | no | Not in odometry message. |
| 13 | ay | no | |
| 14 | az | no | |

**Fused from odom0: x (pose), yaw (pose), vx (twist), vyaw (twist)**

### IMU input (imu0)

| Parameter | Value | Description |
|---|---|---|
| `imu0` | `/bno055/imu` | BNO055 IMU topic. Published by `bno055_node` when running. |
| `imu0_differential` | `false` | |
| `imu0_remove_gravitational_acceleration` | `true` | Removes the ~9.81 m/s² gravitational component from linear acceleration readings before fusion. Required when `two_d_mode` is true to avoid corrupting the filter. |

#### imu0_config — 15-element fusion mask

```
imu0_config: [false, false, false,
              false, false, false,
              false, false, false,
              false, false, true,
              false, false, false]
```

Only one field is fused: **index 11 = vyaw** (angular velocity about z-axis). The BNO055's integrated orientation (quaternion) is not fused from this source — only the raw angular velocity z is used. This is a conservative choice that avoids issues with BNO055 absolute orientation drift or NDOF fusion artifacts corrupting the EKF state.

**Fused from imu0: vyaw only**

## What `/odometry/filtered` will contain

When this EKF is running, `/odometry/filtered` will be a `nav_msgs/Odometry` message with:
- **Pose (x, y, yaw):** fused from encoder x/yaw + IMU angular velocity integration
- **Twist (vx, vyaw):** fused from encoder vx + IMU vyaw
- **Covariance:** reduced compared to raw odometry due to dual-sensor fusion
- **Frame:** `odom` → `base_footprint`

## TF conflict — critical

With `publish_tf: true` in this EKF config AND `enable_odom_tf: true` in `ubot_controllers.yaml`, both nodes would publish the `odom → base_footprint` transform simultaneously. This will cause TF tree corruption and unpredictable navigation behavior.

**Before enabling this EKF node, set `enable_odom_tf: false` in `ubot_controllers.yaml`.**

The TF tree currently shows `odom → base_footprint` at ~30.27 Hz published by `diff_drive_controller`. With the EKF enabled, this edge should be published by `ekf_filter_node` at 30.0 Hz instead.

## Usage

Configured as a commented-out block in `real_robot.launch.py`:

```python
# ekf_node = Node(
#     package='robot_localization',
#     executable='ekf_node',
#     name='ekf_filter_node',
#     parameters=[real_ekf.yaml]
# )
```

To enable: uncomment this block, add `ekf_node` to the `LaunchDescription`, ensure `bno055_node` is also active, and set `enable_odom_tf: false` in `ubot_controllers.yaml`.

## Notes / Known issues

- **Issue #3 (Medium-High):** EKF is fully configured but not launched. The robot currently navigates on wheel-odometry-only with no IMU fusion.
- The BNO055 IMU must also be active (`bno055_node` currently commented out in `real_robot.launch.py`) for `imu0` fusion to work. Without IMU data, the EKF will degrade to odometry-only filtering with increased covariance.
- `frequency: 30.0` Hz matches the wheel odometry publish rate (`publish_rate: 30.0` in `ubot_controllers.yaml`) and the ESP32 PID rate. This alignment ensures the EKF is never waiting for odometry updates.
- `world_frame: odom` — the EKF does not produce a `map → odom` transform. SLAM Toolbox or AMCL handles that separately.

## See Also

- [`sim_ekf.yaml`](sim_ekf.md) — Simulation variant (50 Hz, `/imu/data` source)
- [`bno055_params.yaml`](bno055_params.md) — BNO055 configuration that produces `/bno055/imu`
- [`ubot_controllers.yaml`](ubot_controllers.md) — Must have `enable_odom_tf: false` when EKF is enabled
- `src/ubot/ubot_bringup/launch/real_robot.launch.py` — Contains the commented-out `ekf_node` block
