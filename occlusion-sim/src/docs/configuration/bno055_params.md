# bno055_params.yaml

## Purpose

Deployment configuration for the BNO055 IMU node on the **real robot**. This file lives in `ubot_bringup/config/` and overrides the vendored package's own default parameters. It specifies the physical connection type, I2C bus and address, sensor operation mode, axis remapping for the robot's specific IMU mounting orientation, and pre-recorded calibration offsets.

> **Current status — NOT LAUNCHED (Issue #3):** `bno055_node` is defined but commented out of `real_robot.launch.py`. This config file is ready for use but `bno055_node` is not running on the real robot by default. It must be enabled explicitly.

This documentation covers **only** the `ubot_bringup`-owned deployment config. The vendored package's own default YAML files (`bno055_params.yaml`, `bno055_params_i2c.yaml`, `bno055_params_test.yaml` in `src/bno055/`) are separate and are not part of the ubot deployment config.

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `ros_topic_prefix` | `"bno055/"` | All published topics are prefixed with this string. Results in topics: `/bno055/imu`, `/bno055/imu_raw`, `/bno055/mag`, `/bno055/grav`, `/bno055/temp`, `/bno055/calib_status`. |
| `connection_type` | `"i2c"` | Physical connection method. The BNO055 is connected via I2C, not UART. |
| `i2c_bus` | `1` | I2C bus number. On Raspberry Pi, bus 1 corresponds to GPIO pins 2 (SDA) and 3 (SCL). |
| `i2c_addr` | `0x28` | I2C address of the BNO055. This is the default address when the ADR pin is pulled low. (Alternate address when ADR=HIGH is `0x29`.) |
| `data_query_frequency` | `100` | Hz at which the driver polls the BNO055 for sensor data. |
| `calib_status_frequency` | `0.1` | Hz at which calibration status is published on `/bno055/calib_status`. Low frequency is appropriate — calibration state changes slowly. |
| `frame_id` | `"imu_link"` | TF frame attached to published IMU messages. Matches the `imu_link` frame defined in `ubot_imu.urdf.xacro` at position `[-0.0101, 0.0148, 0.18]` relative to `base_link`. |
| `operation_mode` | `0x0C` | BNO055 register value for **NDOF (Nine Degrees Of Freedom) fusion mode**. In this mode the BNO055 uses its internal sensor fusion algorithm to compute absolute orientation using accelerometer, gyroscope, and magnetometer together. |
| `placement_axis_remap` | `"P2"` | Axis remapping profile for the physical mounting orientation. See "Axis remap" section below. |
| `acc_factor` | `100.0` | Divisor applied to raw accelerometer register values to convert to m/s². |
| `mag_factor` | `16000000.0` | Divisor for magnetometer raw values. |
| `gyr_factor` | `900.0` | Divisor for gyroscope raw values to convert to rad/s. |
| `grav_factor` | `100.0` | Divisor for gravity vector raw values. |
| `set_offsets` | `false` | **Currently false** — the calibration offsets below are defined but NOT applied at startup. Set to `true` to write the pre-recorded offsets to the BNO055 during initialization, enabling faster calibration convergence. |
| `offset_acc` | `[0xFFEC, 0x00A5, 0xFFE8]` | Pre-recorded accelerometer calibration offsets [x, y, z]. These are signed 16-bit values from a previous calibration session. |
| `offset_mag` | `[0xFFB4, 0xFE9E, 0x027D]` | Pre-recorded magnetometer calibration offsets [x, y, z]. |
| `offset_gyr` | `[0x0002, 0xFFFF, 0xFFFF]` | Pre-recorded gyroscope calibration offsets [x, y, z]. |

### Commented-out variance parameters

The following parameters are defined but commented out:

```yaml
# variance_acc: [0.0, 0.0, 0.0]
# variance_angular_vel: [0.0, 0.0, 0.0]
# variance_orientation: [0.0, 0.0, 0.0]
# variance_mag: [0.0, 0.0, 0.0]
```

These would set the diagonal covariance values published in the IMU message headers. With all zeros, the covariance matrices in published messages use the BNO055 driver's internal defaults (or zeros, which downstream nodes such as `robot_localization` interpret as "unknown"). If EKF fusion quality is poor, enabling and tuning these is a diagnostic step.

## Axis remap (P2)

The BNO055 supports eight predefined axis remapping configurations (P0–P7) to handle different physical mounting orientations. The `placement_axis_remap: "P2"` setting is looked up in `SensorService.py`'s `mount_positions` dictionary and writes the corresponding axis remap and sign registers to the BNO055.

**P2 physical meaning:** ⚠️ Could not be determined from source — the exact axis-to-axis and sign mapping for P2 requires inspection of the `mount_positions` dictionary in `src/bno055/bno055/sensor/SensorService.py`. The dictionary maps `'P0'`–`'P7'` to BNO055 register byte values. What is known:

- P2 is selected because the BNO055 is mounted in a non-default orientation on the ubot chassis (IMU at `[-0.0101, 0.0148, 0.18]` in `ubot_imu.urdf.xacro`).
- The purpose of axis remapping is to ensure that the coordinate system reported by the BNO055 matches the ROS convention: x forward, y left, z up (REP-103), aligned with `base_link`.
- Without the correct remap, IMU orientation would be physically incorrect relative to the robot frame, causing EKF fusion errors.

The vendored BNO055 driver will raise an unhandled `KeyError` if `placement_axis_remap` is set to a value outside `'P0'`–`'P7'` — see Issue #10.

## Operation mode detail

`operation_mode: 0x0C` = **NDOF** (register value `0x0C` per BNO055 datasheet).

In NDOF mode:
- The internal MCU runs a sensor fusion algorithm combining accelerometer, gyroscope, and magnetometer.
- The BNO055 publishes absolute orientation as a calibrated quaternion.
- The `bno055` driver publishes this as `sensor_msgs/Imu` on `/bno055/imu` (after manual normalization — see Issue #11).
- Calibration is required for accurate magnetometer-based heading. The `/bno055/calib_status` topic reports calibration progress on a 0–3 scale for each sensor subsystem.

## Calibration offsets

The offsets in this file were recorded from a previous calibration session:

| Sensor | X offset | Y offset | Z offset |
|---|---|---|---|
| Accelerometer | `0xFFEC` (−20) | `0x00A5` (165) | `0xFFE8` (−24) |
| Magnetometer | `0xFFB4` (−76) | `0xFE9E` (−354) | `0x027D` (637) |
| Gyroscope | `0x0002` (2) | `0xFFFF` (−1) | `0xFFFF` (−1) |

These are signed 16-bit register values in hexadecimal. To apply them at startup, change `set_offsets: false` to `set_offsets: true`. If the physical environment has changed significantly since these were recorded (e.g., different room, nearby ferromagnetic objects), the magnetometer offsets may need to be recalibrated.

## Usage

Referenced in `real_robot.launch.py` by the `bno055_node` definition:

```python
bno055_node = Node(
    package='bno055',
    executable='bno055',
    name='bno055',
    parameters=[bno055_params.yaml],
    output='screen'
)
# bno055_node,  ← this line is commented out in the LaunchDescription
```

Can also be launched standalone via the package's own launch file:
`ros2 launch bno055 bno055.launch.py`

## Notes / Known issues

- **Issue #3 (Medium-High):** `bno055_node` is not included in `real_robot.launch.py`. Enable by uncommenting the node line in that file.
- **Issue #10 (Low):** `SensorService.py` performs an unguarded dict lookup on `placement_axis_remap`. An invalid value (e.g., `"P9"`) will raise `KeyError` and crash the node with no friendly error message.
- **Issue #11 (Low/TODO):** The driver normalizes the BNO055 quaternion with hand-rolled code rather than a standard library function (`TODO` at ~line 200 of `SensorService.py`). The normalization is functionally correct but non-idiomatic.
- **Issue #12 (Low/TODO):** `orientation_covariance` is reused for both raw and filtered IMU messages. There are TODOs (~lines 157, 189 of `SensorService.py`) to make this configurable.
- `set_offsets: false` means every power cycle requires the BNO055 to re-calibrate from scratch. For operational use, consider setting `set_offsets: true` with the pre-recorded offsets to reduce startup time.
- The BNO055 will enter calibration mode from a standing start in NDOF mode. Full magnetometer calibration typically requires moving the robot in a figure-8 pattern.

## See Also

- [`real_ekf.yaml`](real_ekf.md) — EKF that consumes `/bno055/imu`
- `src/bno055/bno055/sensor/SensorService.py` — Driver implementation with `mount_positions` dict (P2 register values), `configure()`, and `get_sensor_data()`
- `src/ubot/ubot_description/urdf/ubot_imu.urdf.xacro` — URDF definition of `imu_link` frame
- `src/ubot/ubot_bringup/launch/real_robot.launch.py` — Contains commented-out `bno055_node` block
