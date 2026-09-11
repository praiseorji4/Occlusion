# ubot_hardware_interface

## Overview

`UbotHardware` is a **`ros2_control` hardware interface plugin**, not a standalone ROS 2 node.  It
runs *inside* the `controller_manager` process and is loaded automatically when the
`ros2_control_node` starts with the URDF that contains the `<ros2_control>` block.

| Attribute | Value |
|---|---|
| Plugin name | `ubot_control/UbotHardware` |
| Base class | `hardware_interface::SystemInterface` |
| Package | `ubot_control` (ament_cmake, v0.0.0, BSD-3-Clause) |
| Transport | libserial over USB UART to ESP32 (`/dev/ttyUSB0` @ 115200 baud) |
| Drive config | 4WD; only front wheels carry encoders; rear wheels are mechanically slaved |

## Hardware Parameters

All parameters are read from the `<ros2_control>` URDF block (defined in
`ubot_description/urdf/ros2_control/ubot_ros2_control.xacro`) inside `on_init()`.

| URDF parameter | Type | Value in URDF | Notes |
|---|---|---|---|
| `left_wheel_name` | string | `front_left_wheel_joint` | Must match joint name in URDF |
| `right_wheel_name` | string | `front_right_wheel_joint` | Must match joint name in URDF |
| `enc_counts_per_rev_left` | int | `3956` | Must match `ENC_CPR_LEFT` in `diff_controller.h` |
| `enc_counts_per_rev_right` | int | `3956` | Must match `ENC_CPR_RIGHT` in `diff_controller.h` |
| `wheel_separation` | double | `0.264204` | m; must match `WHEEL_SEPARATION_M` in `diff_controller.h` and `wheel_separation` in `ubot_controllers.yaml` |
| `wheel_radius` | double | `0.033` | m; must match `WHEEL_RADIUS_M` in `diff_controller.h` and `wheel_radius` in `ubot_controllers.yaml` |
| `serial_device` | string | `/dev/ttyUSB0` | ESP32 USB UART |
| `baud_rate` | int | `115200` | Must match ESP32 `Serial` baud |
| `serial_timeout_ms` | int | `1000` | libserial read timeout |
| `loop_rate` | double | `30.0` | Hz; **must match `PID_RATE` in `diff_controller.h` (30 Hz)** — used for ticks-per-frame conversion |
| `diag_publish_rate` | int | `0` | 0 = disabled; N = publish `/ubot/diagnostics` every N `read()` cycles |

### Three-way synchronisation table

These three values must be kept in sync across firmware, URDF, and controller YAML.  A mismatch
causes odometry errors.

| Constant | `diff_controller.h` | `ubot_ros2_control.xacro` | `ubot_controllers.yaml` |
|---|---|---|---|
| Wheel separation | `WHEEL_SEPARATION_M = 0.264204f` | `wheel_separation = 0.264204` | `wheel_separation: 0.264204` |
| Wheel radius | `WHEEL_RADIUS_M = 0.033f` | `wheel_radius = 0.033` | `wheel_radius: 0.033` |
| Encoder CPR | `ENC_CPR_LEFT = ENC_CPR_RIGHT = 3956` | `enc_counts_per_rev_left/right = 3956` | n/a (firmware only) |
| Loop/PID rate | `PID_RATE = 30 Hz` | `loop_rate = 30` | `update_rate: 30` |

## Interfaces Exported to ros2_control

### State interfaces (8 total)

`export_state_interfaces()` returns position and velocity for all four wheels.  Rear wheel
state doubles point to the same memory as the front wheel `Wheel` objects — they are literal
mirrors.

| Joint | Interface | Backing storage |
|---|---|---|
| `front_left_wheel_joint` | `position` | `wheel_l_.pos` |
| `front_left_wheel_joint` | `velocity` | `wheel_l_.vel` |
| `front_right_wheel_joint` | `position` | `wheel_r_.pos` |
| `front_right_wheel_joint` | `velocity` | `wheel_r_.vel` |
| `rear_left_wheel_joint` | `position` | `wheel_l_.pos` (mirror) |
| `rear_left_wheel_joint` | `velocity` | `wheel_l_.vel` (mirror) |
| `rear_right_wheel_joint` | `position` | `wheel_r_.pos` (mirror) |
| `rear_right_wheel_joint` | `velocity` | `wheel_r_.vel` (mirror) |

### Command interfaces (2 total)

`export_command_interfaces()` only exposes velocity commands for the front wheels.  The rear joints
have no command interface on real hardware.

| Joint | Interface | Backing storage |
|---|---|---|
| `front_left_wheel_joint` | `velocity` | `wheel_l_.cmd` (rad/s) |
| `front_right_wheel_joint` | `velocity` | `wheel_r_.cmd` (rad/s) |

## Published Topics

| Topic | Message Type | QoS | Condition |
|---|---|---|---|
| `/ubot/diagnostics` | `std_msgs/Float32MultiArray` | depth 10 | Only when `diag_publish_rate > 0` |

The `Float32MultiArray` contains 14 fields in the same order as the ESP32 `'q'` response:
`[l_enc, r_enc, l_tgt_rpm, r_tgt_rpm, l_rpm, r_rpm, l_err, r_err, l_int, r_int, l_out, r_out, lin_vel, ang_vel]`.

## Subscribed Topics

None (hardware interfaces do not subscribe directly).

## Services

None (hardware interfaces do not create services directly).

## Lifecycle / Operation

### Hardware interface within the controller_manager pipeline

```mermaid
flowchart TD
    subgraph controller_manager process
        CM[controller_manager\nros2_control_node]
        CM -->|loads plugin| HW[UbotHardware\non_init]
        HW --> CFG[on_configure\nopen serial /dev/ttyUSB0]
        CFG --> ACT[on_activate\nzero state, seed encoders\nset_motor_values 0,0]
        ACT --> LOOP[30 Hz control loop]
        LOOP --> READ[read()\nsend 'e\\r'\nparse encoder delta\nupdate pos/vel state\noptional 'q\\r' for /ubot/diagnostics]
        LOOP --> WRITE[write()\nconvert rad/s → ticks/frame\nsend 'm L R\\r']
        LOOP --> READ
        ACT --> DEACT[on_deactivate\nset_motor_values 0,0]
        DEACT --> CLEAN[on_cleanup\ndisconnect serial]
    end
    subgraph Controllers
        JBS[joint_state_broadcaster\nreads state interfaces\npublishes /joint_states]
        DDC[diff_drive_controller\nreads /diff_drive_controller/cmd_vel\nwrites command interfaces\npublishes /diff_drive_controller/odom]
    end
    READ -->|state interfaces| JBS
    READ -->|state interfaces| DDC
    DDC -->|command interfaces| WRITE
    HW <-->|libserial\n/dev/ttyUSB0\n115200 baud| ESP32[ESP32\ndiff_controller.h]
```

### Lifecycle callbacks

**`on_init(params)`**
- Called by `controller_manager` during plugin loading.
- Reads all URDF parameters listed in the [Hardware Parameters](#hardware-parameters) table.
- Calls `Wheel::setup()` for `wheel_l_` and `wheel_r_` to compute `rads_per_count = 2π / enc_counts_per_rev`.
- Validates that exactly 4 joints are declared and each has exactly 2 state interfaces.
- Returns `ERROR` and logs a fatal message if validation fails.

**`on_configure(previous_state)`**
- Opens the serial connection to the ESP32 using `ArduinoComms::connect()` (libserial).
- Returns `ERROR` if the port cannot be opened (logs fatal, does not exit the process — the
  `controller_manager` handles the failure).
- If `diag_publish_rate > 0`, creates the `/ubot/diagnostics` publisher.

**`on_activate(previous_state)`**
- Zeros all kinematic state (`enc`, `pos`, `vel`, `cmd`) on both `Wheel` objects.
- Reads current encoder tick counts from the ESP32 and stores them as `last_enc` baselines so
  the first `read()` delta is zero rather than a large jump.
- Sends `m 0 0` to stop the robot.

**`on_deactivate(previous_state)`**
- Sends `m 0 0` to stop the robot.

**`on_cleanup(previous_state)`**
- Calls `ArduinoComms::disconnect()` to close the serial port.
- Nulls the diagnostics publisher.

### read() flow

Called at `loop_rate` Hz (30 Hz) by the `controller_manager` update loop:

1. Sends `'e\r'` to the ESP32 via `ArduinoComms::read_encoder_values()`.
2. Parses the response `"<lEnc> <rEnc>\r\n"` into signed integers.
3. Calls `Wheel::update_position()` on both wheels:
   - `diff = enc - last_enc`
   - `pos += diff * rads_per_count`
   - `last_enc = enc`
4. Computes `vel = (pos_new - pos_old) / dt` (where `dt = period.seconds()`).
5. If `diag_publish_rate > 0` and `diag_cycle_count_ >= diag_publish_rate`:
   - Sends `'q\r'`, parses `DiagSnapshot`, publishes `Float32MultiArray` on `/ubot/diagnostics`.

### write() flow

Called at `loop_rate` Hz immediately after `read()`:

1. Reads `wheel_l_.cmd` and `wheel_r_.cmd` (velocity setpoints in rad/s, written by `diff_drive_controller`).
2. Converts: `ticks_per_frame = (cmd / 2π) * enc_counts_per_rev / loop_rate`
3. Sends `"m <L_tpf> <R_tpf>\r"` via `ArduinoComms::set_motor_values()`.

The ESP32 `doPID()` function computes PWM from the received `TargetTicksPerFrame` value.

### Serial command protocol summary

| Command | Response | Purpose |
|---|---|---|
| `e\r` | `<lEnc> <rEnc>\r\n` | Read encoder ticks |
| `m L R\r` | `OK\r\n` | Set motor ticks-per-frame (signed int) |
| `q\r` | `D <14 fields>\r\n` | Full diagnostic snapshot |
| `u Kp:Kd:Ki:Min\r` | `OK\r\n` | Update both wheel PIDs (legacy order) |
| `l Kp:Ki:Kd:Min\r` | `OK\r\n` | Update left PID only |
| `f Kp:Ki:Kd:Min\r` | `OK\r\n` | Update right PID only |

## Known Issues

| # | Issue | Severity |
|---|---|---|
| 4 | `ubot_controllers.yaml` sets velocity/acceleration limits but never sets `has_velocity_limits: true` / `has_acceleration_limits: true`.  Depending on installed `diff_drive_controller` version, limits may be silently ignored. | Medium |
| 8 | ESP32 PID gains are labelled "Starting gains — re-tune after flashing (previous values were calibrated with a double min_pwm bug)" in `diff_controller.h` line 44.  Current gains (Kp=5.0, Ki=9.0, Kd=0.0) are self-documented as unvalidated. | Low (informational) |
| 13 | Wheel joint TF broadcast rate (~15.26 Hz) observed at approximately half the configured 30 Hz `update_rate`.  Root cause not determined from source alone. | ⚠️ Could not be determined from source — requires runtime inspection |

## See Also

- [`../packages/ubot_control.md`](../packages/ubot_control.md) — full package page
- [`../configuration/ubot_controllers.md`](../configuration/ubot_controllers.md) — controller YAML
- [`../launch/real_robot.launch.py.md`](../launch/real_robot.launch.py.md) — how `controller_manager` is launched
- [`../architecture/ros_graph.md`](../architecture/ros_graph.md) — full ROS graph including controller pipeline
