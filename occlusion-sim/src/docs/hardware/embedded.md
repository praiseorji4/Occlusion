# Embedded Firmware

This page documents both ESP32 firmware implementations in the workspace: the **LIVE** serial-bridge firmware wired to the current ROS2 graph, and the **ORPHANED** micro-ROS firmware that exists in the workspace but is incompatible with the ros2_control architecture.

---

## Firmware summary

| | LIVE firmware | ORPHANED firmware |
|---|---|---|
| **File** | `src/Ros-esp32_bridge/Ros-esp32_bridge.ino` | `src/esp32/hacker/hacker.ino` |
| **Header** | "Serial bridge: ESP32 <-> ROS 2 (Jazzy)" | "Phase 6: Pure Motor Controller — RPM PID, no odometry" |
| **Named robot** | (ubot / Horizon) | "Horizon Mini Robot" |
| **Architecture** | Text-command serial ↔ ROS2 via `UbotHardware` plugin | micro-ROS (rclc + micro_ros_arduino), direct `/cmd_vel` subscriber |
| **Interfaces ROS2 via** | `ubot_control/UbotHardware` (ros2_control) | micro-ROS executor, bypasses ros2_control entirely |
| **Command channel** | `Serial` (USB GPIO1/3) @115200 via libserial | micro-ROS agent over USB (separate transport) |
| **PID rate** | 30 Hz | 50 Hz (CONTROL_INTERVAL_MS=20) |
| **PID gains** | Kp=5.0, Ki=9.0, Kd=0.0 | Kp=20.0, Ki=1.5, Kd=0.0 |
| **Anti-windup** | ±50.0 | ±30.0 |
| **Auto-stop timeout** | 10 000 ms | 500 ms |
| **Min PWM** | 50 | 50 |
| **Encoder CPR** | 3956 per wheel | 3956 per wheel |
| **Wheel geometry** | radius=0.033 m, sep=0.264204 m | radius=0.033 m, sep=0.264204 m |
| **ROS topics** | None directly (mediated by ros2_control) | `/horizon/left_encoder`, `/horizon/right_encoder`, `/horizon/left_vel_ms`, `/horizon/right_vel_ms` |
| **Node name (ROS2)** | n/a | `mini_esp32_node` |

---

## LIVE firmware: `Ros-esp32_bridge.ino`

### Hardware

- **MCU**: ESP32 NodeMCU
- **Motor driver**: 2× BTS7960 43A H-bridge (one per side, each driving 2 wheels mechanically coupled)
- **Drive**: 4WD, front-wheel encoders only; rear wheels are mechanically slaved to the front axle on each side
- **Serial channels**:
  - `Serial` (USB, GPIO1/3) @115200 baud — command/response channel for `UbotHardware`
  - `Serial2` (TX GPIO16) @115200 baud — diagnostic-only stream (`/dev/ttyUSB1` on host, feeds PlotJuggler)

### Serial command protocol (`commands.h`)

Commands arrive as ASCII characters, with space-separated arguments, terminated by `\r` (CR, char 13). The parser reads chars into a buffer, splits on space, and calls `runCommand()` on CR.

| Char | Arguments | Response | Description |
|---|---|---|---|
| `b` | — | `"<baudrate>\r\n"` | Return current baud rate |
| `e` | — | `"<leftTicks> <rightTicks>\r\n"` | Read current encoder counts |
| `r` | — | (none) | Reset encoders + PID state |
| `m` | `L R` (ticks/frame, int) | (none) | Set motor speed targets — main control command from `UbotHardware::write()` |
| `o` | `L R` (PWM int ±255) | (none) | Raw PWM bypass — directly sets PWM without PID |
| `u` | `Kp:Kd:Ki:Min` | (none) | Update BOTH PIDs (legacy order: Kp, **Kd**, Ki, Min) |
| `l` | `Kp:Ki:Kd:Min` | (none) | Update left PID (standard order) |
| `f` | `Kp:Ki:Kd:Min` | (none) | Update right PID (standard order) |
| `q` | — | 14-field "D ..." line (same as Serial2) | Diagnostic snapshot — used by `diag_publisher.py` |

> **Note on `'u'` command order**: the legacy `'u'` command uses the order `Kp:Kd:Ki` (not `Kp:Ki:Kd`). The newer `'l'` and `'f'` commands use `Kp:Ki:Kd`. Passing the wrong order via `'u'` will silently apply Kd as Ki and vice versa.

### `diff_controller.h` — PID implementation (30 lines key excerpt)

```
File: src/Ros-esp32_bridge/diff_controller.h (210 lines)
```

**Robot geometry constants** (must stay in sync with `ubot_controllers.yaml` and `ubot_ros2_control.xacro`):

| Constant | Value |
|---|---|
| `WHEEL_RADIUS_M` | 0.033 f |
| `WHEEL_SEPARATION_M` | 0.264204 f |
| `WHEEL_CIRC_M` | 2π × 0.033 = 0.20735 m |
| `ENC_CPR_LEFT` | 3956 |
| `ENC_CPR_RIGHT` | 3956 |
| `PID_RATE` | 30 Hz |
| `PID_DT_S` | 1/30 ≈ 0.0333 s |
| `CMD_TIMEOUT_MS` | 10 000 ms (10 s auto-stop) |

**PID gains** (line 44 comment: _"Starting gains — re-tune after flashing (previous values were calibrated with a double min_pwm bug)"_):

| Gain | Left | Right |
|---|---|---|
| Kp | 5.0 | 5.0 |
| Ki | 9.0 | 9.0 |
| Kd | 0.0 | 0.0 |
| MinPwm | 50 | 50 |

> **Known limitation**: these are explicitly self-documented as _starting values_ not yet re-tuned. A previous `min_pwm` double-application bug was fixed but the gains from before that fix are no longer valid. Re-tuning is required — see [Calibration Guide](../guides/calibration.md).

**RPM filtering — Jimeno low-pass filter**:
```
rpm_filtered = 0.854 × rpm_filtered + 0.0728 × raw_rpm + 0.0728 × rpm_prev
```
This is a second-order IIR filter. The coefficients (0.854, 0.0728, 0.0728) produce ~0.854 pole, rolling-window smoothing. The sum 0.854 + 0.0728 + 0.0728 = 0.9996 ≈ 1.0 (unity gain with slight attenuation).

**PID loop per tick (updatePID(), called at 30 Hz)**:

1. Read encoders atomically (`noInterrupts()` / `interrupts()`)
2. Convert `TargetTicksPerFrame` to RPM: `rpm_target = (ticks / CPR) × 60 × PID_RATE`
3. `getRpm()` — Jimeno filter on both wheels
4. **10-second watchdog**: if `cmdTimerActive && elapsed ≥ 10000ms` → `setMotorBrakes(200, 200)` + `resetPID()` + log to Serial2
5. If not moving and RPM nonzero → `resetPID()`
6. If both targets == 0 → `setMotorBrakes()` + `resetPID()`
7. `doPID()` — `error = target_rpm − rpm_filtered`, `integral += error × dt`, integral clamped ±50, `derivative = (error − last_error) / dt`, `output = clamp(Kp×e + Ki×i + Kd×d, −255, 255)`
8. **Feedforward dead-zone offset**: `cmd = clamp(sign(output) × minPwm + output, −255, 255)` — adds the minimum PWM needed to overcome static friction in the direction of motion
9. `setMotorSpeeds(l_cmd, r_cmd)` → drives BTS7960 via ledc API
10. Every 15 ticks (~2 Hz): debug line to Serial2
11. `emitDiagnostics()` → full 14-field "D ..." line to Serial2

**Diagnostics serial stream (Serial2, GPIO16, 115200)**:

Each PID tick emits one `"D ..."` line:
```
D <leftEnc> <rightEnc> <leftTargetRpm> <rightTargetRpm> \
  <leftRpmFiltered> <rightRpmFiltered> \
  <leftError> <rightError> \
  <leftIntegral> <rightIntegral> \
  <leftOutput> <rightOutput> \
  <linVel_ms> <angVel_rads>
```
15 whitespace-separated tokens (token 0 = `'D'`). `diag_publisher.py` sends `'q\r'` to request this same line via the command channel instead of reading Serial2.

Every 15 ticks (~2 Hz), a human-readable summary is also sent:
```
TGT L:<rpm> R:<rpm> | ACT L:<rpm> R:<rpm> | TICKS L:<n> R:<n> | PWM L:<n> R:<n>
```

### Encoder driver (`encoder_driver.h/.ino`)

The encoder ISRs decode quadrature signals using a 16-entry lookup table (standard Gray-code decoding for 4-state transitions). The table approach is efficient on the ESP32 and handles both direction transitions cleanly.

Pins:
| Signal | Left | Right |
|---|---|---|
| ENC_A | 18 | 23 |
| ENC_B | 19 | 22 |

### Motor driver (`DualBTS7960MotorShieldESP32.h/.cpp`)

The BTS7960 H-bridge is controlled via PWM (ESP32 ledc API):
- Frequency: 30 kHz
- Resolution: 8-bit (0-255)
- `setMotorSpeeds(L, R)`: positive = forward (RPWM active, LPWM low), negative = reverse (LPWM active, RPWM low)
- `setMotorBrakes(L, R)`: both RPWM and LPWM driven high simultaneously → braking

---

## ORPHANED firmware: `hacker.ino`

> **Do not flash this firmware** when using the ros2_control / UbotHardware path. If flashed, the hardware interface will fail to communicate (wrong serial protocol) and two ROS2 nodes will attempt to command the same motors via incompatible paths.

### Why it exists

This appears to be an earlier or parallel development track exploring micro-ROS as an alternative to the text-command serial bridge. It operates as a first-class ROS2 participant (via micro-ROS agent), subscribing to `/cmd_vel` directly and publishing wheel feedback, without needing a C++ hardware_interface plugin on the host side.

The approach is architecturally valid but was apparently superseded in favour of the ros2_control / serial-bridge approach (which integrates with Nav2's velocity command pipeline, joint_state_broadcaster, and the standard diff_drive_controller).

### Key differences from live firmware

| Aspect | `Ros-esp32_bridge.ino` (live) | `hacker.ino` (orphaned) |
|---|---|---|
| ROS2 integration point | Host-side `UbotHardware` plugin reads/writes serial | micro-ROS on ESP32 connects directly to ROS2 graph |
| `/cmd_vel` consumption | Via `diff_drive_controller` → `UbotHardware::write()` → serial `'m'` command | Direct micro-ROS subscription on ESP32 |
| Odometry | Computed by `diff_drive_controller` from encoder state | "Phase 6: no odometry" (explicitly none) |
| Diagnostics topic | `/horizon/diag/*` via diag_publisher on host | `/horizon/left_vel_ms`, `/horizon/right_vel_ms` directly |
| Auto-stop timeout | 10 000 ms | 500 ms (much more aggressive) |
| PID rate | 30 Hz | 50 Hz |
| Gains | Kp=5, Ki=9, Kd=0 | Kp=20, Ki=1.5, Kd=0 |

### Internal bug in `hacker.ino`

**Lines 34-35** (header comment block):
```
// BTS7960 Right: RPWM→2 LPWM→15 R_EN→4 L_EN→17
```

**Actual `#define`s** (~line 40):
```cpp
#define RIGHT_RPWM 32
#define RIGHT_LPWM 33
#define RIGHT_R_EN  4
#define RIGHT_L_EN 17
```

`RIGHT_RPWM` and `RIGHT_LPWM` are documented as pins 2 and 15 in the comment but defined as 32 and 33 in code. If anyone were to wire the hardware based on the comment rather than the actual code, the right-side H-bridge would be wired incorrectly and the motor would not respond. Since this file is not flashed in the current system, the impact is documentation confusion only.

---

## Three-way sync requirement

Three sources must stay in perfect agreement on wheel geometry and control rate. When changing any of these values, update **all three**:

| Parameter | `diff_controller.h` | `ubot_ros2_control.xacro` | `ubot_controllers.yaml` |
|---|---|---|---|
| Wheel radius | `WHEEL_RADIUS_M 0.033f` | `wheel_radius 0.033` | `wheel_radius: 0.033` |
| Wheel separation | `WHEEL_SEPARATION_M 0.264204f` | `wheel_separation 0.264204` | `wheel_separation: 0.264204` |
| CPR | `ENC_CPR_LEFT/RIGHT 3956` | `enc_counts_per_rev_left/right 3956` | n/a |
| Control rate | `PID_RATE 30` | `loop_rate 30` | `update_rate: 30` |

The odometry debugging checklist in `ubot_controllers.yaml` also provides a physical verification procedure — see [Calibration Guide](../guides/calibration.md).
