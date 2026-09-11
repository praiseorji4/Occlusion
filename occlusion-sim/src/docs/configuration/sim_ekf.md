# sim_ekf.yaml

## Purpose

`robot_localization` Extended Kalman Filter configuration for the **Gazebo simulation**. Functionally equivalent to `real_ekf.yaml` but adapted for simulation: runs at 50 Hz (matching the sim controller rate), uses `use_sim_time: true`, and subscribes to the Gazebo simulated IMU topic `/imu/data` instead of the physical BNO055 topic `/bno055/imu`.

The `/odometry/filtered` output of this EKF is consumed by `sim_nav2_params.yaml` as the primary odometry source for all Nav2 servers.

## Differences from `real_ekf.yaml`

| Parameter | real_ekf.yaml | sim_ekf.yaml |
|---|---|---|
| `use_sim_time` | `false` | `true` |
| `frequency` | `30.0` Hz | `50.0` Hz |
| `imu0` topic | `/bno055/imu` | `/imu/data` |
| `odom0_config` description | "x, yaw pose + vx, vyaw twist" | "x, yaw pose + vx, vyaw twist" (identical mask) |

## Parameters

### ekf_filter_node

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `true` | Uses Gazebo `/clock` topic. Required for any node running in simulation. |
| `frequency` | `50.0` Hz | EKF update and publish rate. Matches `sim_ubot_controllers.yaml` `update_rate: 50` and `diff_drive_controller.publish_rate: 50.0`. |
| `two_d_mode` | `true` | Constrains to 2D plane — same as real robot. |
| `odom_frame` | `odom` | |
| `base_link_frame` | `base_footprint` | |
| `world_frame` | `odom` | |
| `publish_tf` | `true` | EKF publishes `odom → base_footprint`. This is why `sim_ubot_controllers.yaml` sets `enable_odom_tf: false` — to prevent a dual-publisher TF conflict. |

### Odometry input (odom0)

| Parameter | Value | Description |
|---|---|---|
| `odom0` | `/diff_drive_controller/odom` | Same wheel odometry source as real robot. Published by `diff_drive_controller` running with `sim_ubot_controllers.yaml`. |
| `odom0_differential` | `false` | |
| `odom0_relative` | `false` | |

#### odom0_config

```
odom0_config: [true,  false, false,   # x only, not y
               false, false, true,    # yaw only
               true,  false, false,   # vx only
               false, false, true,    # vyaw only
               false, false, false]
```

Identical to `real_ekf.yaml`. The inline comments in `sim_ekf.yaml` explicitly annotate each row:

| Index | State | Fused |
|---|---|---|
| 0 | x | **yes** |
| 1 | y | no |
| 2 | z | no |
| 3–5 | roll, pitch, yaw | yaw **yes** |
| 6 | vx | **yes** |
| 7 | vy | no |
| 8 | vz | no |
| 9–11 | vroll, vpitch, vyaw | vyaw **yes** |
| 12–14 | ax, ay, az | no |

**Fused from odom0: x, yaw, vx, vyaw** (same as real robot)

### IMU input (imu0)

| Parameter | Value | Description |
|---|---|---|
| `imu0` | `/imu/data` | Gazebo simulated IMU topic. Published by the `ubot_gazebo.urdf.xacro` IMU sensor at 100 Hz with Gaussian noise (bias/noise stddevs configured in the xacro). This is NOT the BNO055 physical sensor topic. |
| `imu0_differential` | `false` | |
| `imu0_remove_gravitational_acceleration` | `true` | Same as real robot — removes simulated gravitational component. |

#### imu0_config

```
imu0_config: [false, false, false,
              false, false, false,
              false, false, false,
              false, false, true,
              false, false, false]
```

Identical to `real_ekf.yaml` — only angular velocity z (index 11, vyaw) is fused from the IMU. The sim file comments: "Avoids noisy linear accelerations in 2D mode."

**Fused from imu0: vyaw only**

## What `/odometry/filtered` will contain in simulation

When this EKF is running, `/odometry/filtered` provides:
- **Pose (x, y, yaw):** fused from Gazebo encoder simulation and IMU
- **Twist (vx, vyaw):** fused from encoder and IMU angular rate
- **Published at 50 Hz** — higher rate than real robot (30 Hz)
- **Frame:** `odom → base_footprint` (via `publish_tf: true`)

`sim_nav2_params.yaml` subscribes to `/odometry/filtered` for both `bt_navigator.odom_topic` and `controller_server.odom_topic`, replacing the raw `/diff_drive_controller/odom`.

## Usage

This file is loaded by the EKF node when running in simulation. The typical sim launch sequence:

1. `sim.launch.py` — starts Gazebo, robot URDF (`use_gazebo:=true`), `sim_ubot_controllers.yaml`
2. `ekf_filter_node` with `sim_ekf.yaml` — fuses odometry, publishes `/odometry/filtered` and `odom → base_footprint` TF
3. Nav2 launch with `sim_nav2_params.yaml` — uses `/odometry/filtered`

## Notes / Known issues

- The Gazebo IMU sensor (`ubot_gazebo.urdf.xacro`) publishes at 100 Hz — the EKF at 50 Hz will receive 2 IMU samples per update cycle, which is fine (robot_localization handles multiple measurements between filter updates).
- `imu0_remove_gravitational_acceleration: true` is important even in simulation because Gazebo's IMU model includes the gravitational acceleration component in the linear acceleration field.
- Unlike `real_ekf.yaml`, there is no TF conflict issue to worry about if only the sim stack is running — `sim_ubot_controllers.yaml` already sets `enable_odom_tf: false`.

## See Also

- [`real_ekf.yaml`](real_ekf.md) — Real robot variant (30 Hz, `/bno055/imu`)
- [`sim_ubot_controllers.yaml`](sim_ubot_controllers.md) — Sets `enable_odom_tf: false` to hand TF responsibility to this EKF
- [`sim_nav2_params.yaml`](sim_nav2_params.md) — Consumes `/odometry/filtered` produced by this EKF
- `src/ubot/ubot_description/urdf/ubot_gazebo.urdf.xacro` — Defines the simulated IMU sensor publishing on `/imu/data`
