# sim_ubot_controllers.yaml

## Purpose

Simulation-mode ros2_control controller configuration, loaded when the robot runs inside Gazebo (via `sim.launch.py`). It registers the same controller types as the real-hardware configuration but differs in update rate, wheel assignment, and the addition of an IMU sensor broadcaster — reflecting the different capabilities of the Gazebo simulation versus physical hardware.

## Key differences from `ubot_controllers.yaml` (real robot)

| Parameter | Real robot (`ubot_controllers.yaml`) | Simulation (`sim_ubot_controllers.yaml`) |
|---|---|---|
| `update_rate` | 30 Hz | **50 Hz** |
| `use_sim_time` (controller_manager) | not set (defaults false) | **true** |
| Left wheel names | `["front_left_wheel_joint"]` | `["front_left_wheel_joint", "rear_left_wheel_joint"]` |
| Right wheel names | `["front_right_wheel_joint"]` | `["front_right_wheel_joint", "rear_right_wheel_joint"]` |
| `publish_rate` | 30.0 Hz | **50.0 Hz** |
| `enable_odom_tf` | `true` | **`false`** |
| Covariance yaw (idx 5) | `0.01` | **`0.03`** |
| IMU broadcaster | not present | **`imu_sensor_broadcaster` included** |
| Velocity limit style | flat keys (`linear.x.max_velocity`) | nested YAML (`linear: x: max_velocity:`) |
| `has_velocity_limits` | not set | **`true`** (set in this file) |

## Parameters

### controller_manager

| Parameter | Value | Description |
|---|---|---|
| `update_rate` | `50` Hz | Main control loop rate for simulation. Higher than real robot (30 Hz) because Gazebo can run faster than real time and the sim physics benefits from finer granularity. |
| `use_sim_time` | `true` | Instructs all spawned controllers to use `/clock` topic instead of wall time. Required for Gazebo operation. |
| `diff_drive_controller.type` | `diff_drive_controller/DiffDriveController` | Same plugin as real robot. |
| `joint_state_broadcaster.type` | `joint_state_broadcaster/JointStateBroadcaster` | Same plugin as real robot. |
| `imu_sensor_broadcaster.type` | `imu_sensor_broadcaster/IMUSensorBroadcaster` | Additional broadcaster present only in sim. Publishes the Gazebo simulated IMU as a ROS topic. |

### diff_drive_controller

#### Wheel assignment (all 4 wheels)

| Parameter | Value | Description |
|---|---|---|
| `left_wheel_names` | `["front_left_wheel_joint", "rear_left_wheel_joint"]` | All left wheels commanded. |
| `right_wheel_names` | `["front_right_wheel_joint", "rear_right_wheel_joint"]` | All right wheels commanded. |

**Why 4 wheels in simulation vs front-only on real hardware:**

In `ubot_ros2_control.xacro`, rear wheel joints receive a `command_interface(velocity)` only when `use_gazebo:=true` is passed to the xacro. On real hardware, rear joints have state interfaces only (mirrored from front). When Gazebo loads via `gz_ros2_control/GazeboSimSystem`, it allocates command interfaces to every joint that declares one in the xacro, so all four joints are fully actuatable in simulation. Adding the rear joints to the diff_drive_controller's wheel lists gives proper four-wheel-drive torque distribution in the physics simulation.

#### Wheel geometry

| Parameter | Value | Description |
|---|---|---|
| `wheel_separation` | `0.264204` m | Identical to real robot — same physical geometry modelled in URDF. |
| `wheel_radius` | `0.033` m | Identical to real robot. |

#### Odometry configuration

| Parameter | Value | Description |
|---|---|---|
| `publish_rate` | `50.0` Hz | Matches the 50 Hz `update_rate`. |
| `odom_frame_id` | `odom` | Same as real robot. |
| `base_frame_id` | `base_footprint` | Same as real robot. |
| `enable_odom_tf` | `false` | The sim EKF node (`ekf_filter_node` with `sim_ekf.yaml`) publishes the `odom → base_footprint` TF. Disabling it here prevents a dual-publisher conflict. |
| `open_loop` | `false` | Uses Gazebo encoder simulation, not cmd_vel integration. |

#### Velocity limits

| Parameter | Value | Description |
|---|---|---|
| `linear.x.has_velocity_limits` | `true` | Explicitly set — this is the correct way to enable limits. Contrast with `ubot_controllers.yaml` where this flag is absent (Issue #4). |
| `linear.x.max_velocity` | `1.0` m/s | Higher than real robot's 0.5 m/s — simulation allows faster exploration. |
| `linear.x.min_velocity` | `-1.0` m/s | |
| `angular.z.has_velocity_limits` | `true` | Explicitly set. |
| `angular.z.max_velocity` | `2.0` rad/s | Same as real robot. |
| `angular.z.min_velocity` | `-2.0` rad/s | |

#### Covariance matrices

```
pose_covariance_diagonal:  [0.001, 0.001, 0.001, 0.001, 0.001, 0.03]
twist_covariance_diagonal: [0.001, 0.001, 0.001, 0.001, 0.001, 0.03]
```

Unlike the real robot config, z/roll/pitch indices are `0.001` (not `1e6`) and yaw is `0.03` (not `0.01`). In simulation all axes are "known" by the physics engine, so small uniform values are acceptable. The slightly larger yaw value (0.03 vs 0.01) acknowledges that simulation odometry is less precise than real encoder data.

### joint_state_broadcaster

| Parameter | Value | Description |
|---|---|---|
| `use_sim_time` | `true` | Uses Gazebo clock. |
| `qos_overrides./joint_states.publisher.durability` | `volatile` | Prevents stale joint state messages accumulating. |
| `qos_overrides./joint_states.publisher.reliability` | `reliable` | Ensures joint states are not dropped in simulation. |

### imu_sensor_broadcaster

| Parameter | Value | Description |
|---|---|---|
| `frame_id` | `imu_link` | TF frame for the IMU data. Matches `ubot_imu.urdf.xacro` which places `imu_link` at `[-0.0101, 0.0148, 0.18]` relative to `base_link`. |
| `sensor_name` | `imu_sensor` | Name of the hardware interface sensor registered in the `gz_ros2_control/GazeboSimSystem` plugin. |

## Usage

Loaded by `sim.launch.py` (Gazebo simulation launch) when `use_gazebo:=true` is passed to the URDF xacro. Not used by any real-hardware launch file.

## Notes / Known issues

- The `has_velocity_limits: true` flags are correctly set here, unlike `ubot_controllers.yaml` (real robot). This is a discrepancy to note when porting changes between the two config files.
- `enable_odom_tf: false` is essential — the sim EKF (`sim_ekf.yaml`) takes ownership of the `odom → base_footprint` transform.
- `sim_nav2_params.yaml` uses `odom_topic: /odometry/filtered` (EKF output), not `/diff_drive_controller/odom` directly — both `sim_ekf.yaml` and this controllers file must be running for Nav2 to receive valid odometry in simulation.

## See Also

- [`ubot_controllers.yaml`](ubot_controllers.md) — Real hardware variant
- [`sim_ekf.yaml`](sim_ekf.md) — EKF configuration for simulation that consumes `/diff_drive_controller/odom` and publishes the TF
- [`sim_nav2_params.yaml`](sim_nav2_params.md) — Nav2 configuration for simulation, uses `/odometry/filtered`
- `src/ubot/ubot_description/urdf/ubot_ros2_control.xacro` — Declares which joints get command interfaces depending on `use_gazebo`
