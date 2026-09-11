# diag_publisher

## Overview

`DiagPublisher` is a standalone ROS 2 node (class name `DiagPublisher`, ROS node name
`'horizon_diag_publisher'`) in the `ubot_debugger` package.  It polls the ESP32's `'q'` diagnostic
command over the same `/dev/ttyUSB0` serial port used by `ubot_hardware_interface` and republishes
each of the 14 diagnostic fields as individual `std_msgs/Float64` topics under the
`/horizon/diag/` namespace.  This allows PlotJuggler (or any ROS 2 subscriber) to plot the full
PID state of both drive wheels in real time.

**Package:** `ubot_debugger` (ament_python, v0.0.0, licence field is literally `TODO` — see issue #16)

**Executable (from `setup.py`):** `diag_publisher`

**Deployment status.** This node is NOT included in any launch file (`real_robot.launch.py` or
otherwise).  It must be started manually.  See [Known Issues](#known-issues).

## How to Run

```bash
# With defaults (serial_port=/dev/ttyUSB0, baud_rate=115200, publish_rate=10.0):
ros2 run ubot_debugger diag_publisher

# Override parameters:
ros2 run ubot_debugger diag_publisher \
  --ros-args -p serial_port:=/dev/ttyUSB0 -p baud_rate:=115200 -p publish_rate:=10.0
```

> **Warning:** Keep `publish_rate` at or below 15 Hz.  The node sends a `'q\r'` command to the
> ESP32 on every timer tick and waits for a response.  Higher rates can saturate the single shared
> serial command channel, causing encoder-read timeouts in `ubot_hardware_interface`.

## Parameters

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `serial_port` | string | `/dev/ttyUSB0` | Serial device connected to ESP32 USB UART |
| `baud_rate` | int | `115200` | Must match ESP32 firmware (matches `commands.h`) |
| `publish_rate` | float | `10.0` | Hz; keep ≤ 15 — see warning above |

## Published Topics

All 14 topics use message type `std_msgs/Float64` and QoS depth 10.

| Topic | ESP32 field | Range / Unit |
|---|---|---|
| `/horizon/diag/enc/left` | `leftEnc` | raw encoder tick count (signed long) |
| `/horizon/diag/enc/right` | `rightEnc` | raw encoder tick count (signed long) |
| `/horizon/diag/rpm/left_target` | `leftTargetRpm` | PID target RPM |
| `/horizon/diag/rpm/right_target` | `rightTargetRpm` | PID target RPM |
| `/horizon/diag/rpm/left_actual` | `leftRpmFiltered` | Jimeno low-pass filtered actual RPM |
| `/horizon/diag/rpm/right_actual` | `rightRpmFiltered` | Jimeno low-pass filtered actual RPM |
| `/horizon/diag/pid/left_error` | `leftError` | RPM error (target − actual) |
| `/horizon/diag/pid/right_error` | `rightError` | RPM error (target − actual) |
| `/horizon/diag/pid/left_integral` | `leftIntegral` | PID integral accumulator (clamped ±50) |
| `/horizon/diag/pid/right_integral` | `rightIntegral` | PID integral accumulator (clamped ±50) |
| `/horizon/diag/pid/left_output` | `leftOutput` | Final PWM command (clamped −255..255) |
| `/horizon/diag/pid/right_output` | `rightOutput` | Final PWM command (clamped −255..255) |
| `/horizon/diag/vel/linear` | `linVel` | Body linear velocity, m/s |
| `/horizon/diag/vel/angular` | `angVel` | Body angular velocity, rad/s |

These 14 fields correspond exactly to the 14-field diagnostic line that the ESP32 firmware
(`diff_controller.h: emitDiagnostics()`) emits on `Serial2` (GPIO16, `/dev/ttyUSB1`) and also
returns synchronously on Serial in response to the `'q'` command.

## Subscribed Topics

None.

## Services

None.

## Lifecycle / Operation

### Data flow

```mermaid
flowchart LR
    A[ESP32 firmware\ndiff_controller.h\nemitDiagnostics] -->|Serial USB\n/dev/ttyUSB0\n115200 baud| B[DiagPublisher\nhorizon_diag_publisher]
    B -->|'q\\r' command sent\nevery 1/publish_rate s| A
    B --> C[/horizon/diag/enc/left\n/horizon/diag/enc/right]
    B --> D[/horizon/diag/rpm/left_target\n/horizon/diag/rpm/right_target\n/horizon/diag/rpm/left_actual\n/horizon/diag/rpm/right_actual]
    B --> E[/horizon/diag/pid/left_error\n/horizon/diag/pid/right_error\n/horizon/diag/pid/left_integral\n/horizon/diag/pid/right_integral\n/horizon/diag/pid/left_output\n/horizon/diag/pid/right_output]
    B --> F[/horizon/diag/vel/linear\n/horizon/diag/vel/angular]
    C --> G[PlotJuggler]
    D --> G
    E --> G
    F --> G
```

### Polling mechanism

A `rclpy` timer fires every `1.0 / publish_rate` seconds (default every 0.1 s).  On each tick:

1. `_send_command('q\r')` flushes the serial input buffer, writes `'q\r'` (two bytes), then calls
   `readline()` with a 1-second timeout.
2. The response is decoded as ASCII.
3. `_parse_diag(line)` validates the response:
   - The line must start with `'D'`.
   - After splitting on whitespace the result must have exactly **15 tokens** (`'D'` + 14 floats).
   - Any `ValueError` during `float()` conversion is caught.
   - On any validation failure: `logger.warn()` + return `None` (publish skipped for this cycle).
4. On success, each of the 14 float values is published to its corresponding topic.

### Startup and teardown

- **Startup failure:** If `serial.Serial()` raises `SerialException` (port absent, permission
  denied, etc.), the node calls `logger.fatal()` and raises `SystemExit(1)`.  There is no retry
  loop.
- **Runtime serial errors:** Caught in `_send_command()` as `SerialException`, logged as a
  warning, and the current poll cycle is skipped.
- **Teardown:** `destroy_node()` closes the serial port cleanly.

### Relationship to the ESP32 Serial2 stream

The ESP32 firmware sends the same 14-field diagnostic line autonomously on `Serial2` (TX GPIO16,
intended wiring to `/dev/ttyUSB1`) at approximately 2 Hz (every 15 PID ticks at 30 Hz).  This node
instead queries the same data synchronously via the `'q'` command on the main `Serial` USB port,
which means it does **not** require a second serial adapter.  The field order and format are
identical to the `Serial2` stream.

## Known Issues

| # | Issue | Severity |
|---|---|---|
| 6 | Node is not included in any launch file; must be run manually | Low |
| 6 | Fatal exit on serial open failure — no retry/recovery | Low |
| — | `diag_publisher` shares `/dev/ttyUSB0` with `ubot_hardware_interface`.  Both send `'q\r'`/`'e\r'` commands to the ESP32.  Running both simultaneously at high `publish_rate` can cause command collisions and missed encoder reads in the hardware interface.  Recommend `publish_rate` ≤ 10 Hz during normal operation. | Medium |

## See Also

- [`../packages/ubot_debugger.md`](../packages/ubot_debugger.md) — package page
- [`../nodes/ubot_hardware_interface.md`](ubot_hardware_interface.md) — the hardware interface that shares `/dev/ttyUSB0`
- [ESP32 firmware `diff_controller.h`](../../Ros-esp32_bridge/diff_controller.h) — `emitDiagnostics()` source
