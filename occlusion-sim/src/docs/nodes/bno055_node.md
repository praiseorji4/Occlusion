# bno055_node

## Overview

The `bno055` node is a ROS 2 driver for the Bosch BNO055 9-axis IMU sensor.  It is provided by the
vendored `bno055` package (v0.5.0, BSD licence, upstream: Flynneva), which supports both I2C and UART
transports.

**Naming note.** The ROS node name (set in the `Node()` constructor in `bno055.py`) is `'bno055'`.
The executable registered in `setup.py` is also `bno055`.

**Deployment status.** This node is *defined* in `real_robot.launch.py` (lines ~101-107) but the
line that adds it to the returned `LaunchDescription` is commented out (`# bno055_node,` at line
~125).  The EKF node (`ekf_filter_node`) that would consume IMU data is also fully commented out.
The robot therefore runs on wheel-odometry only until both blocks are uncommented.  See
[Known Issues](#known-issues) and [`real_robot.launch.py`](../launch/real_robot.launch.py.md).

**Executable invocation** (manual):

```bash
ros2 run bno055 bno055 --ros-args --params-file \
  /home/chibueze/uni-bot/src/ubot/ubot_bringup/config/bno055_params.yaml
```

The `bno055_params.yaml` in `ubot_bringup/config/` is the deployment config that sets
`placement_axis_remap: 'P2'` for this robot's mount orientation.

## Architecture

`Bno055Node` is the ROS node class (subclass of `rclpy.node.Node`).  It composes a
`SensorService` object (not itself a Node) which owns all publishers, the service, and the
hardware communication logic.  The connector (`I2C` or `UART`) is selected by the
`connection_type` parameter.

## Parameters

All parameters are declared in `bno055/bno055/params/NodeParameters.py`.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `ros_topic_prefix` | string | `'bno055/'` | Prepended to every published topic name |
| `connection_type` | string | `'uart'` | `'uart'` or `'i2c'` |
| `uart_port` | string | `'/dev/ttyUSB0'` | UART only |
| `uart_baudrate` | int | `115200` | UART only |
| `uart_timeout` | float | `0.1` | Seconds; UART only |
| `i2c_bus` | int | `0` | I2C only |
| `i2c_addr` | int | `0x28` | I2C only |
| `frame_id` | string | `'bno055'` | Header frame_id for all stamped topics |
| `data_query_frequency` | int | `10` | Hz — how often `get_sensor_data()` is called |
| `calib_status_frequency` | float | `0.1` | Hz — how often `get_calib_status()` is called |
| `operation_mode` | int | `0x0C` | BNO055 operation mode register value (NDOF = 0x0C) |
| `placement_axis_remap` | string | `'P1'` | Mount position key; ubot deployment uses `'P2'` |
| `acc_factor` | float | `100.0` | Divisor applied to raw accelerometer counts |
| `mag_factor` | float | `16000000.0` | Divisor applied to raw magnetometer counts |
| `gyr_factor` | float | `900.0` | Divisor applied to raw gyroscope counts |
| `grav_factor` | float | `100.0` | Divisor applied to raw gravity vector counts |
| `set_offsets` | bool | `False` | If true, writes `offset_acc/mag/gyr` on startup |
| `offset_acc` | int[] | `DEFAULT_OFFSET_ACC` | Accelerometer calibration offsets |
| `offset_mag` | int[]  | `DEFAULT_OFFSET_MAG` | Magnetometer calibration offsets |
| `offset_gyr` | int[]  | `DEFAULT_OFFSET_GYR` | Gyroscope calibration offsets |
| `radius_acc` | int | `DEFAULT_RADIUS_ACC` | Accelerometer radius |
| `radius_mag` | int | `DEFAULT_RADIUS_MAG` | Magnetometer radius |
| `variance_acc` | float[3] | `DEFAULT_VARIANCE_ACC` | Linear acceleration covariance diagonal |
| `variance_angular_vel` | float[3] | `DEFAULT_VARIANCE_ANGULAR_VEL` | Angular velocity covariance diagonal |
| `variance_orientation` | float[3] | `DEFAULT_VARIANCE_ORIENTATION` | Orientation covariance diagonal |
| `variance_mag` | float[3] | `DEFAULT_VARIANCE_MAG` | Magnetic field covariance diagonal |

`DEFAULT_*` constants are defined in `bno055/bno055/registers.py`; their numeric values require
inspection of that file.

## Published Topics

All publishers are created in `SensorService.__init__()` with `QoSProfile(depth=10)`.  The topic
name is `{ros_topic_prefix}{suffix}` where `ros_topic_prefix` defaults to `'bno055/'`.

| Topic | Message Type | frame_id | Notes |
|---|---|---|---|
| `bno055/imu_raw` | `sensor_msgs/Imu` | `param.frame_id` | Raw accelerometer + gyroscope data; orientation_covariance field reused from `variance_orientation` param (see Known Issues) |
| `bno055/imu` | `sensor_msgs/Imu` | `param.frame_id` | Filtered IMU data; quaternion orientation manually normalised; orientation_covariance copied from `imu_raw` |
| `bno055/mag` | `sensor_msgs/MagneticField` | `param.frame_id` | Magnetometer data |
| `bno055/grav` | `geometry_msgs/Vector3` | n/a | Gravity vector; `geometry_msgs/Vector3` has no header/covariance field — this is a message-type limitation, not a bug |
| `bno055/temp` | `sensor_msgs/Temperature` | `param.frame_id` | Temperature in degrees Celsius |
| `bno055/calib_status` | `std_msgs/String` | n/a | JSON string: `{"sys": 0-3, "gyro": 0-3, "accel": 0-3, "mag": 0-3}` where 3 = fully calibrated |

## Subscribed Topics

None.

## Services

| Service | Type | Behaviour |
|---|---|---|
| `bno055/calibration_request` | `example_interfaces/srv/Trigger` (imported as `Trigger`) | Switches sensor to config mode, reads current calibration offset registers, switches back to NDOF mode, returns offsets as a string in `response.message`; `response.success` is always `True` |

**Calling the service:**

```bash
ros2 service call /bno055/calibration_request example_interfaces/srv/Trigger "{}"
```

## Lifecycle / Operation

### Node startup sequence

```mermaid
flowchart TD
    A[ros2 run bno055 bno055] --> B[Bno055Node.__init__\nsuper().__init__('bno055')]
    B --> C[node.setup()\nNodeParameters declared]
    C --> D{connection_type?}
    D -- uart --> E[UART connector created\nuart_port / uart_baudrate / uart_timeout]
    D -- i2c --> F[I2C connector created\ni2c_bus / i2c_addr]
    E --> G[connector.connect()]
    F --> G
    G --> H[SensorService created\nPublishers + Service registered]
    H --> I[sensor.configure()]
    I --> J{Chip ID\nverified?}
    J -- No --> K[logger.error\nsys.exit(1)]
    J -- Yes --> L[Set config mode\nnormal power\npage 0\ntrigger 0x00\nunit_sel 0x83]
    L --> M[Axis remap written\nmount_positions key\nfrom placement_axis_remap]
    M --> N{set_offsets\ntrue?}
    N -- Yes --> O[Write calib offsets\nto sensor registers]
    N -- No --> P[Set operation_mode]
    O --> P
    P --> Q[Timers created\ndata_query_timer\nstatus_timer]
    Q --> R[rclpy.spin]
    R --> S[data_query_timer fires\nat data_query_frequency Hz\nget_sensor_data()]
    R --> T[status_timer fires\nat calib_status_frequency Hz\nget_calib_status()]
```

### configure()

`SensorService.configure()` (in `bno055/bno055/sensor/SensorService.py`):

1. Reads chip ID register `BNO055_CHIP_ID_ADDR` via the connector.  If the received ID does not
   match `BNO055_ID`, or if any exception is raised, logs an error and calls `sys.exit(1)` — hard
   exit, no retry or recovery.
2. Sets operation mode → `OPERATION_MODE_CONFIG`.
3. Sets power mode → `POWER_MODE_NORMAL`.
4. Sets register page → `0x00`.
5. Writes `BNO055_SYS_TRIGGER_ADDR` → `0x00`.
6. Writes `BNO055_UNIT_SEL_ADDR` → `0x83` (m/s², radians, degrees C, quaternion units).
7. Looks up axis-remap bytes from the `mount_positions` dict keyed by `placement_axis_remap`:

   | Key | Bytes (hex) |
   |---|---|
   | `P0` | `\x21\x04` |
   | `P1` | `\x24\x00` |
   | `P2` | `\x24\x06` ← ubot deployment value |
   | `P3` | `\x21\x02` |
   | `P4` | `\x24\x03` |
   | `P5` | `\x21\x02` |
   | `P6` | `\x21\x07` |
   | `P7` | `\x24\x05` |

8. Optionally writes calibration offsets if `set_offsets` is `True`.
9. Sets the final operation mode from the `operation_mode` parameter.

### get_sensor_data()

Called on every `data_query_timer` tick (rate = `data_query_frequency` Hz, default 10 Hz).

Reads **45 bytes** starting from register `BNO055_ACCEL_DATA_X_LSB_ADDR` in a single transaction.
The byte layout used:

| Bytes | Field |
|---|---|
| 0–1 | Accel X LSB/MSB |
| 2–3 | Accel Y LSB/MSB |
| 4–5 | Accel Z LSB/MSB |
| 6–7 | Mag X LSB/MSB |
| 8–9 | Mag Y LSB/MSB |
| 10–11 | Mag Z LSB/MSB |
| 12–13 | Gyro X LSB/MSB |
| 14–15 | Gyro Y LSB/MSB |
| 16–17 | Gyro Z LSB/MSB |
| 24–25 | Quat W LSB/MSB |
| 26–27 | Quat X LSB/MSB |
| 28–29 | Quat Y LSB/MSB |
| 30–31 | Quat Z LSB/MSB |
| 32–37 | Linear accel X/Y/Z (filtered) |
| 38–43 | Gravity vector X/Y/Z |
| 44 | Temperature |

Publishes `bno055/imu_raw`, `bno055/imu`, `bno055/mag`, `bno055/grav`, and `bno055/temp`.

### get_calib_status()

Called on every `status_timer` tick (rate = `calib_status_frequency` Hz, default 0.1 Hz).

Reads **1 byte** from `BNO055_CALIB_STAT_ADDR`.  Extracts four 2-bit fields by bit-shifting:

```
sys   = (byte >> 6) & 0x03
gyro  = (byte >> 4) & 0x03
accel = (byte >> 2) & 0x03
mag   =  byte       & 0x03
```

Each field ranges 0–3 (0 = uncalibrated, 3 = fully calibrated).  Published as a JSON string on
`bno055/calib_status`.

## Known Issues

| # | Issue | Severity |
|---|---|---|
| 3 | Node is defined but excluded from `real_robot.launch.py` — robot runs without IMU fusion | Medium-High |
| 10 | `configure()` does an unguarded dict lookup on `mount_positions`: an invalid `placement_axis_remap` value (e.g. `'P8'`) raises an unhandled `KeyError` instead of a friendly error | Low |
| 11 | Quaternion normalisation in `get_sensor_data()` is hand-rolled (`sqrt(x²+y²+z²+w²)`) rather than using a standard library call. Source TODO at line ~200: `"TODO(flynneva): replace with standard normalize() function"` | Low |
| 12 | `orientation_covariance` on `bno055/imu` is copied from `imu_raw_msg.orientation_covariance` rather than computed separately. TODOs at lines ~154 and ~157/189 ask: `"TODO: do headers need sequence counters now?"` and `"TODO: make this an option to publish?"` | Low |

## See Also

- [`../configuration/bno055_params.md`](../configuration/bno055_params.md) — deployment YAML (`placement_axis_remap: 'P2'`)
- [`../launch/real_robot.launch.py.md`](../launch/real_robot.launch.py.md) — shows the commented-out `bno055_node` block
- [`../configuration/real_ekf.md`](../configuration/real_ekf.md) — EKF config that consumes `bno055/imu`
- [`../packages/bno055.md`](../packages/bno055.md) — full package page
