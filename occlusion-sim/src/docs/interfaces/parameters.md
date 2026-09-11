# Parameter Reference

This page is a cross-reference of all important tunable parameters in the ubot workspace, grouped by subsystem. For each parameter: what it is, where it is set, what it affects, and any critical constraints.

---

## Critical sync requirement — wheel geometry

**Three files must always agree on wheel geometry.** If they disagree, odometry will be wrong and navigation will fail silently.

| File | Role |
|---|---|
| `src/Ros-esp32_bridge/diff_controller.h` | ESP32 firmware — used to compute target RPM from ticks and to calculate `linVel`/`angVel` for the Serial2 diagnostic stream |
| `src/ubot/ubot_description/urdf/ubot_ros2_control.xacro` | URDF hardware plugin params — read by `UbotHardware` at startup to configure the ROS-side odometry computation |
| `src/ubot/ubot_bringup/config/ubot_controllers.yaml` | diff_drive_controller params — used by the ros2_control controller to compute `/diff_drive_controller/odom` |

When any of these three values changes in one file, it must be updated in **all three** before reflashing or relaunching. Refer to the [wheel calibration guide](../guides/calibration.md) for the procedure to measure and verify these values.

---

## Section 1: Wheel geometry

These four parameters define the physical drive system. They must be kept in sync across the three files listed above.

| Parameter | Value | File | What it affects | Constraint |
|---|---|---|---|---|
| `WHEEL_RADIUS_M` / `wheel_radius` | `0.033` m | `diff_controller.h`, `ubot_ros2_control.xacro`, `ubot_controllers.yaml` | Converts encoder ticks → linear distance; scale of `/diff_drive_controller/odom` pose | Measure with a tape across the physical tyre. Must match in all three files. |
| `WHEEL_SEPARATION_M` / `wheel_separation` | `0.264204` m | `diff_controller.h`, `ubot_ros2_control.xacro`, `ubot_controllers.yaml` | Converts differential wheel speed → angular velocity; heading accuracy of odometry | Measure centre-to-centre of the driven wheels. Must match in all three files. |
| `ENC_CPR_LEFT` / `ENC_CPR_RIGHT` / `enc_counts_per_rev_left` / `enc_counts_per_rev_right` | `3956` | `diff_controller.h`, `ubot_ros2_control.xacro` | Converts raw encoder ticks → revolutions → RPM and linear position. Not in ubot_controllers.yaml (controller uses wheel_radius instead). | Determined by encoder hardware + gearbox ratio. Verify by commanding one full wheel revolution and reading encoder ticks via the `e` serial command. |
| `PID_RATE` / `loop_rate` / `controller_manager.update_rate` | `30` Hz | `diff_controller.h` (`PID_RATE=30`), `ubot_ros2_control.xacro` (`loop_rate=30`), `ubot_controllers.yaml` (`update_rate: 30`) | ESP32 PID tick interval, ROS hardware interface read/write rate, controller manager update rate | All three must match. Changing the rate affects PID integral/derivative scaling — re-tune gains after any change. |

**Odometry debugging procedure** (from `ubot_controllers.yaml` embedded comment):

1. Drive 1 m straight: check `wheel_radius` via `/diff_drive_controller/odom` `pose.position.x`.
2. Spin 360°: check `wheel_separation` via `/diff_drive_controller/odom` yaw.
3. Use PlotJuggler signals: `/diff_drive_controller/odom`, `/joint_states`, `/ubot/diagnostics` (if enabled), Serial2 stream on `/dev/ttyUSB1`.

---

## Section 2: PID parameters (ESP32 firmware)

These parameters live exclusively in the ESP32 firmware (`diff_controller.h`). Changing them requires re-flashing the ESP32 via the Arduino IDE (or equivalent). Alternatively, the `u`, `l`, and `f` serial commands can update gains at runtime without reflashing (changes are not persisted across power cycles).

> **Warning:** The code comment at `diff_controller.h` line 44 explicitly marks the current gains as **starting values pending re-tuning after a bug fix** ("Starting gains — re-tune after flashing (previous values were calibrated with a double min_pwm bug)"). These gains have not been validated as final production values.

| Parameter | Value | Notes |
|---|---|---|
| `leftKp` / `rightKp` | `5.0` | Proportional gain. Both wheels set identically. |
| `leftKi` / `rightKi` | `9.0` | Integral gain. Both wheels set identically. High integral — watch for windup at speed transitions. |
| `leftKd` / `rightKd` | `0.0` | Derivative gain. Disabled (zero). |
| `leftMinPwm` / `rightMinPwm` | `50` | Dead-zone offset PWM value, applied as a feedforward kick: `output = sign(target) * minPwm + PID_output`. Helps overcome motor stiction. |
| Integral anti-windup clamp | `±50.0` | Integral accumulator is clamped to this range every tick inside `doPID()`. |
| Output clamp | `±255` | Final PWM value clamped to Arduino analogWrite range. |
| Control rate | `30` Hz (`PID_RATE`) | `PID_DT_S = 1/30 ≈ 0.0333 s`. Used in integral and derivative terms. |
| Auto-stop timeout | `10000` ms (`CMD_TIMEOUT_MS`) | If no `m` (motor command) is received for 10 s, firmware brakes both motors and calls `resetPID()`. Matches `UbotHardware` — note the orphaned `hacker.ino` used 500 ms, but that firmware is not active. |

**Runtime gain update via serial (without reflashing):**

```
# Update both wheels (legacy order: Kp Kd Ki Min — note different order from 'l'/'f' commands):
u 5.0:0.0:9.0:50

# Update left wheel only (order: Kp Ki Kd Min):
l 5.0:9.0:0.0:50

# Update right wheel only (order: Kp Ki Kd Min):
f 5.0:9.0:0.0:50
```

Changes take effect immediately but are lost on power cycle or firmware reset.

---

## Section 3: Nav2 key tunable parameters

These parameters are in `src/ubot/ubot_bringup/config/nav2_params.yaml`. The values below are the **current fixed values** (post bug-fix; see inline changelog comments in the file for previous values). For the full parameter set see [configuration/nav2_params.md](../configuration/nav2_params.md).

| Parameter | File | Current value | Notes |
|---|---|---|---|
| `max_vel_x` | nav2_params.yaml (`controller_server`) | `0.15` m/s | Maximum forward speed. Conservative — robot is small (0.41 m body) and uses wheel odometry only. |
| `max_vel_theta` | nav2_params.yaml (`controller_server`) | `0.6` rad/s | Maximum angular speed during navigation. |
| `max_speed_xy` | nav2_params.yaml (`controller_server`) | `0.15` m/s | DWB overall speed cap, consistent with `max_vel_x`. |
| `sim_time` | nav2_params.yaml (`controller_server` / DWB) | `1.7` s | DWB forward simulation horizon. Longer = smoother paths but more compute. |
| `xy_goal_tolerance` | nav2_params.yaml (`controller_server` + `general_goal_checker`) | `0.15` m | How close the robot must be to the goal position to consider it reached. |
| `yaw_goal_tolerance` | nav2_params.yaml (`general_goal_checker`) | `0.15` rad | Angular tolerance at goal. |
| `local inflation_radius` | nav2_params.yaml (`local_costmap`, `inflation_layer`) | `0.30` m | Inflation around obstacles in the local costmap. Previously `0.05` m — less than the robot radius, which was unsafe. Now matches `robot_radius: 0.15` with a safety margin. |
| `global inflation_radius` | nav2_params.yaml (`global_costmap`, `inflation_layer`) | `0.30` m | Inflation around obstacles in the global costmap. Previously `0.20` m. |
| `local cost_scaling_factor` | nav2_params.yaml (`local_costmap`) | `3.0` | Exponential decay rate of inflation cost. Higher = tighter penalty near obstacles. |
| `global cost_scaling_factor` | nav2_params.yaml (`global_costmap`) | `1.5` | Slower decay in global costmap, allowing wider planning margins. |
| `planner tolerance` | nav2_params.yaml (`planner_server`) | `0.25` m | NavFn planner goal tolerance. Previously `0.5` m — too coarse. |
| `use_astar` | nav2_params.yaml (`planner_server`) | `true` | A* enabled (previously false/Dijkstra). A* is faster with the same path quality for this environment size. |
| DWB `BaseObstacle.scale` | nav2_params.yaml (`controller_server`) | `0.5` | Previously `0.02` — ~1600x weaker than the path critics, causing the robot to near-ignore obstacles. Now at a meaningful weight. |
| DWB `PathAlign.scale` | nav2_params.yaml (`controller_server`) | `32.0` | Weight for aligning robot heading with the planned path. |
| DWB `GoalAlign.scale` | nav2_params.yaml (`controller_server`) | `24.0` | Weight for aligning with the goal heading. |
| DWB `PathDist.scale` | nav2_params.yaml (`controller_server`) | `32.0` | Weight for minimising distance from the planned path. |
| DWB `GoalDist.scale` | nav2_params.yaml (`controller_server`) | `24.0` | Weight for minimising distance to the goal. |
| DWB `RotateToGoal.scale` | nav2_params.yaml (`controller_server`) | `32.0` | Weight for rotating in-place to align with goal heading at the end of a path. |
| `velocity_smoother feedback` | nav2_params.yaml (`velocity_smoother`) | `"CLOSED_LOOP"` | Uses `/diff_drive_controller/odom` for feedback. Previously `OPEN_LOOP`. |
| `collision_monitor time_before_collision` | nav2_params.yaml (`collision_monitor`) | `1.2` s | Approach polygon: collision_monitor slows or stops the robot when an obstacle is predicted to be within this time window. |

---

## Section 4: Serial device paths

These must match the physical USB device assignments on the robot's Raspberry Pi. USB device numbers (`ttyUSB0`, `ttyUSB1`) can change if devices are unplugged and re-plugged or if plug-in order changes. Consider adding udev rules to fix device paths by USB hardware address or serial number.

| Parameter | File | Default | What connects here | Constraint |
|---|---|---|---|---|
| `serial_device` | `src/ubot/ubot_description/urdf/ubot_ros2_control.xacro` (hardware plugin param) | `/dev/ttyUSB0` | ESP32 NodeMCU via USB — Serial1 command/response channel at 115200 baud | Must not be shared with any other process while `ros2_control_node` is running. `diag_publisher.py` also uses this port — run one or the other, not both simultaneously. |
| `port_name` | `src/ubot/ubot_bringup/launch/real_robot.launch.py` (`ldlidar_node` param) | `/dev/ttyUSB1` | LD19 LiDAR UART at 230400 baud | If `/scan` topic is absent, verify this device exists: `ls -la /dev/ttyUSB*`. |

**ESP32 Serial2 stream:** The ESP32 also emits a continuous 14-field diagnostic stream on its Serial2 pin (TX = GPIO16) at 115200 baud. This is a one-way output intended for PlotJuggler connection on a second laptop or via a second USB-serial adapter. It carries the same data as the `q` command but without polling. It is **not** read by any ROS node.

---

## Section 5: BNO055 parameters

These parameters are set in `src/ubot/ubot_bringup/config/bno055_params.yaml` (the deployment config owned by `ubot_bringup`, distinct from the vendored package defaults). They are only relevant when `bno055_node` is running.

| Parameter | Value | What it does |
|---|---|---|
| `placement_axis_remap` | `'P2'` | Selects the physical mounting orientation of the BNO055 on the robot. `P2` maps to register bytes `[0x24, 0x06]` (axis remap config + axis remap sign). This must match how the chip is physically oriented relative to the `imu_link` frame. Valid values: P0–P7 (Bosch BNO055 datasheet, Table 3-4). An invalid value causes an **unhandled `KeyError`** at node startup — there is no friendly error message. |
| `connection_type` | UART or I2C (set in bno055_params.yaml) | Selects the physical connection method. Must match the wiring on the robot. |
| `operation_mode` | NDOF (default) | Sets the BNO055 fusion mode. NDOF uses accelerometer + gyroscope + magnetometer for full 9-DOF fusion including absolute orientation. Required for the `/bno055/imu` quaternion output. |
| `set_offsets` | false (default) | If true, writes hardcoded calibration offsets from params at startup. Set to true and populate `offset_acc`, `offset_mag`, `offset_gyr` fields after capturing values from `/bno055/calibration_request` to skip the in-session calibration warm-up period. |

---

## Section 6: SLAM Toolbox parameters

These parameters are in `src/ubot/ubot_bringup/config/mapper_params_online_async.yaml`.

| Parameter | Value | Notes |
|---|---|---|
| `resolution` | `0.05` m/cell | Map cell size. Matches Nav2 costmap resolution. Do not change without also updating Nav2 costmap resolutions. |
| `max_laser_range` | `12.0` m | Maximum range used from `/scan`. The LD19 LiDAR has a rated range of ~12 m — this is near the hardware limit. |
| `map_file_name` | `/home/chibueze/uni-bot/studio_3_serial` | **Hardcoded absolute path** — this path is specific to the development machine and will break on any other system or if maps are moved. This is a known limitation (see [issues.md](../issues.md)). Update before deploying to a different machine. |
| `transform_publish_period` | `0.02` s | TF publish rate for the `map` → `odom` transform: 50 Hz. Higher than the 30 Hz controller rate, providing smooth map transforms. |
| `mode` | `"mapping"` | Online async mapping mode. Switch to `"localization"` for pure localisation against a saved map. |
| `solver_plugin` | `solver_plugins::CeresSolver` | Pose graph optimisation backend. Ceres is the recommended default for online SLAM Toolbox. |
| `loop_closure_enabled` | `true` | Enables loop closure detection and correction. Critical for consistent long-session maps. |

---

## EKF parameters (inactive — for reference)

These parameters are in `src/ubot/ubot_bringup/config/real_ekf.yaml`. The EKF node is currently commented out in `real_robot.launch.py` (see [issue #3](../issues.md)).

| Parameter | Value | Notes |
|---|---|---|
| `frequency` | `30` Hz | EKF update rate. Matched to controller_manager update_rate. |
| `two_d_mode` | `true` | Constrains the filter to 2D — appropriate for a ground robot. |
| `odom0` | `/diff_drive_controller/odom` | Primary odometry source. Fuses x/y pose and forward/angular velocity. |
| `imu0` | `/bno055/imu` | IMU source. Only yaw angular velocity is fused (other axes masked out). |
| `imu_remove_gravitational_acceleration` | `true` | Strips gravity from accelerometer before fusion. |
| `publish_tf` | `true` | EKF would broadcast `odom` → `base_footprint`. Currently `diff_drive_controller` broadcasts this TF since EKF is disabled — enabling the EKF requires disabling `enable_odom_tf` in `ubot_controllers.yaml` to avoid two nodes broadcasting the same TF edge. |
