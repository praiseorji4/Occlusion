# ubot_controllers.yaml

## Purpose

Real-hardware ros2_control controller configuration for the ubot differential-drive robot (internally called "Horizon" in the file header — same physical robot, different naming context; see [Robot identity](../architecture/ros_graph.md)). This file is loaded by `controller_manager` at startup and governs the DiffDriveController, JointStateBroadcaster, and all wheel-odometry parameters.

## Three-way sync requirement

Three sources must always agree on wheel geometry and control rate:

| Value | This file | `diff_controller.h` | `ubot_ros2_control.xacro` |
|---|---|---|---|
| `wheel_separation` | `0.264204` | `WHEEL_SEPARATION_M=0.264204f` | `wheel_separation=0.264204` |
| `wheel_radius` | `0.033` | `WHEEL_RADIUS_M=0.033f` | `wheel_radius=0.033` |
| Control rate | `update_rate: 30` Hz | `PID_RATE=30` Hz | (no explicit rate) |

Changing any of these in isolation will break odometry or cause encoder/command mismatches.

## Parameters

### controller_manager

| Parameter | Value | Description |
|---|---|---|
| `update_rate` | `30` Hz | Main control loop rate. **Must match `PID_RATE` in `diff_controller.h`** — this is the rate at which the hardware interface calls `read()`/`write()` and at which the ESP32 fires its PID interrupt. |
| `diff_drive_controller.type` | `diff_drive_controller/DiffDriveController` | Plugin class for the differential drive controller. |
| `joint_state_broadcaster.type` | `joint_state_broadcaster/JointStateBroadcaster` | Plugin class for broadcasting all joint states. |

### diff_drive_controller

#### Wheel assignment

| Parameter | Value | Description |
|---|---|---|
| `left_wheel_names` | `["front_left_wheel_joint"]` | Only the front-left joint. The rear-left joint has no encoder and no command interface on real hardware — it is a state-only mirror of the front. |
| `right_wheel_names` | `["front_right_wheel_joint"]` | Only the front-right joint, for the same reason. |

The file header explains this directly: "diff_drive_controller sees only the two front wheel joints — those are the ones with real encoder feedback. The rear joints have state interfaces (mirrored from front) but no command interfaces, so they are correctly excluded from the wheel lists below."

#### Wheel geometry (calibration constants)

| Parameter | Value | Description |
|---|---|---|
| `wheel_separation` | `0.264204` m | Center-to-center lateral distance between left and right wheels. Calibrate by spinning the robot exactly 360° in place and checking that yaw returns to ≈ 0.0 rad. |
| `wheel_radius` | `0.033` m | Wheel radius. Calibrate by driving exactly 1 m straight and checking `pose.position.x` on `/diff_drive_controller/odom`. |

#### Velocity and acceleration limits

| Parameter | Value | Description |
|---|---|---|
| `linear.x.max_velocity` | `0.5` m/s | Maximum forward speed commanded to the controller. |
| `linear.x.min_velocity` | `-0.5` m/s | Maximum reverse speed. |
| `linear.x.max_acceleration` | `1.0` m/s² | Maximum forward acceleration. |
| `linear.x.min_acceleration` | `-1.0` m/s² | Maximum forward deceleration. |
| `angular.z.max_velocity` | `2.0` rad/s | Maximum angular (turn) speed. |
| `angular.z.min_velocity` | `-2.0` rad/s | Maximum angular speed in reverse direction. |
| `angular.z.max_acceleration` | `2.0` rad/s² | Maximum angular acceleration. |
| `angular.z.min_acceleration` | `-2.0` rad/s² | Maximum angular deceleration. |

> **Known issue (M2):** The `has_velocity_limits: true` and `has_acceleration_limits: true` boolean flags are **never set** in this file, despite the limit values above being present. Depending on the installed version of `ros2_control`, limits configured without these flags may be silently ignored. Verify against the installed `diff_drive_controller` version — flag behavior changed across releases. See issues list item #4.

#### Odometry configuration

| Parameter | Value | Description |
|---|---|---|
| `odom_frame_id` | `odom` | Frame name for the odometry origin. Published as the parent of `base_footprint` in the TF tree. |
| `base_frame_id` | `base_footprint` | Frame name for the robot base. |
| `enable_odom_tf` | `true` | Controller publishes the `odom → base_footprint` TF transform directly. Confirmed at ~30.27 Hz in the captured TF tree (`frames_2026-06-26_17.33.00.gv`). **Important:** if `ekf_filter_node` is enabled, this must be set to `false` to avoid a dual-publisher TF conflict. |
| `open_loop` | `false` | Odometry is computed from encoder counts, not by integrating velocity commands. The file comment states: "MUST be false — use encoder feedback, not cmd_vel." |
| `publish_rate` | `30.0` Hz | Rate at which `/diff_drive_controller/odom` and the TF transform are published. Matches `update_rate` so every control cycle produces one odometry update. |
| `position_feedback` | `true` | Uses encoder position (tick count) for odometry rather than velocity. Improves accuracy at low speeds and when velocity commands are noisy. **Note:** this parameter name may be deprecated or renamed in some installed versions of `diff_drive_controller`. ⚠️ Verify against the installed ros2_control distro version — do not assume this is a hard bug. See issues list item #17. |

#### Covariance matrices

```
pose_covariance_diagonal:  [0.001, 0.001, 1e6, 1e6, 1e6, 0.01]
twist_covariance_diagonal: [0.001, 1e6,   1e6, 1e6, 1e6, 0.01]
```

The six elements correspond to [x, y, z, roll, pitch, yaw]:

| Index | Axis | Pose value | Twist value | Rationale |
|---|---|---|---|---|
| 0 | x | `0.001` | `0.001` | Encoder-derived; reasonably accurate. |
| 1 | y | `0.001` | `1e6` | Pose y is measured by encoder; lateral velocity is effectively zero for diff-drive. |
| 2 | z | `1e6` | `1e6` | Ground robot — no z motion; "infinite uncertainty" is deliberate. |
| 3 | roll | `1e6` | `1e6` | Ground robot — cannot measure roll; deliberate infinite uncertainty. |
| 4 | pitch | `1e6` | `1e6` | Ground robot — cannot measure pitch; deliberate infinite uncertainty. |
| 5 | yaw | `0.01` | `0.01` | Encoders give reasonable heading estimation. **This is not a placeholder value — it is a reasoned choice.** |

The file explicitly comments: "indices 2 (z), 3 (roll), 4 (pitch) are set to 1e6 (effectively infinite uncertainty) because a ground robot cannot measure them. Leave index 5 (yaw) at 0.01 — encoders give reasonable heading."

These values feed into `robot_localization` EKF if enabled in the future.

### joint_state_broadcaster

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `false` | Uses wall-clock time; correct for real hardware. |

The broadcaster publishes all four joints (`front_left`, `front_right`, `rear_left`, `rear_right`). The rear joints report the same position/velocity as their front counterparts because they share the same memory locations in the `UbotHardware` interface.

## Embedded odometry debugging checklist

The following checklist is reproduced directly from the comments in `ubot_controllers.yaml`:

**If `/odom` drifts, work through these in order:**

1. **`wheel_radius`**: Drive exactly 1 m straight. Check `/diff_drive_controller/odom pose.position.x`.
   - Too short → increase radius.
   - Too far → decrease radius.

2. **`wheel_separation`**: Spin exactly 360° in place. Check that yaw returns to ≈ 0.0 rad.
   - Drifts positive → separation too small, increase it.
   - Drifts negative → separation too large, decrease it.

3. Both values must match the ESP32 firmware (`diff_controller.h`) and the URDF (`ubot_ros2_control.xacro`) — keep all three in sync.

## PlotJuggler signals for odometry debugging

Subscribe to these topics (from the file's embedded comment):

| Topic | Fields of interest |
|---|---|
| `/diff_drive_controller/odom` | `pose.position.x/y`, `twist.twist.linear.x` |
| `/joint_states` | velocity for all 4 joints |
| `/ubot/diagnostics` (if enabled) | per-wheel RPM setpoint vs actual |
| Serial2 stream `/dev/ttyUSB1` | full 14-field diagnostic frame at 30 Hz |

The 14-field Serial2 diagnostic frame format (from `diff_controller.h` / `emitDiagnostics()`):
`leftEnc, rightEnc, leftTargetRpm, rightTargetRpm, leftRpmFiltered, rightRpmFiltered, leftError, rightError, leftIntegral, rightIntegral, leftOutput, rightOutput, linVel, angVel`

## Usage

This file is loaded exclusively by `real_robot.launch.py`:

```python
controller_manager:
  params: [robot_description, "ubot_bringup/config/ubot_controllers.yaml"]
```

It is **not** used by `sim.launch.py` — that uses `sim_ubot_controllers.yaml`.

## Notes / Known issues

- **Issue #4 (Medium):** `has_velocity_limits` / `has_acceleration_limits` flags are absent. Limit values may be silently ignored depending on the installed `ros2_control` version.
- **Issue #17 (inspection required):** `position_feedback: true` — parameter name may be deprecated in newer `diff_drive_controller` versions. Verify at runtime.
- **TF observation (Issue #13):** The captured TF tree shows wheel-joint TF at ~15.26 Hz — approximately half the configured 30 Hz `update_rate`. Root cause is not determinable from source alone. ⚠️ Requires runtime inspection to confirm whether this is a `joint_state_broadcaster` internal behavior or a `view_frames` sampling artifact.
- The `enable_odom_tf: true` setting will conflict with `ekf_filter_node`'s `publish_tf: true` if the EKF is enabled. Before enabling the EKF (currently commented out in `real_robot.launch.py`), set `enable_odom_tf: false` here.

## See Also

- [`real_ekf.yaml`](real_ekf.md) — EKF configuration that consumes `/diff_drive_controller/odom`
- [`sim_ubot_controllers.yaml`](sim_ubot_controllers.md) — Simulation variant (50 Hz, 4 wheels)
- `src/Ros-esp32_bridge/diff_controller.h` — ESP32 firmware constants that must stay in sync
- `src/ubot/ubot_description/urdf/ubot_ros2_control.xacro` — URDF hardware interface parameters that must stay in sync
- `src/ubot/ubot_control/ubot_hardware_interface.cpp` — `UbotHardware` plugin that reads this config
