# Calibration

This guide covers the three calibration procedures required for accurate robot operation: wheel geometry, IMU, and PID gain tuning. Performing these in order will give you the best results.

---

## Overview: The Three-File Rule

Wheel geometry constants exist in **three places** and must always be kept in sync:

| File | Constants |
|---|---|
| `/home/chibueze/uni-bot/src/Ros-esp32_bridge/diff_controller.h` | `WHEEL_RADIUS_M`, `WHEEL_SEPARATION_M` |
| `/home/chibueze/uni-bot/src/ubot/ubot_description/urdf/ros2_control/ubot_ros2_control.xacro` | `wheel_radius`, `wheel_separation` |
| `/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/ubot_controllers.yaml` | `wheel_radius`, `wheel_separation` |

**Current values** (verified from source):

```
WHEEL_RADIUS_M     = 0.033 m
WHEEL_SEPARATION_M = 0.264204 m
```

Changing only one or two of the three files will cause odometry inconsistency. After changing `diff_controller.h` you must reflash the ESP32.

---

## 1. Wheel Geometry Calibration

Perform these tests with the robot on a flat surface. Run `real_robot.launch.py` first so that odometry is active.

### 1a. Wheel Radius Calibration

The wheel radius controls how far the robot thinks it has travelled per encoder tick.

**Procedure**:

1. Mark a start position on the floor. Point the robot straight ahead.
2. Drive the robot exactly 1 metre forward (use a tape measure on the floor).
3. In a second terminal, check the reported distance:
   ```bash
   ros2 topic echo /diff_drive_controller/odom --once
   ```
   Read `pose.pose.position.x`.

**Interpretation**:

| Observed `position.x` | Action |
|---|---|
| Less than 1.0 m | Robot under-reports distance → **increase** `wheel_radius` |
| More than 1.0 m | Robot over-reports distance → **decrease** `wheel_radius` |
| ~1.0 m (within ±0.01 m) | Radius is calibrated |

**Update procedure** (after determining the correct value):

1. Edit `diff_controller.h`:
   ```cpp
   #define WHEEL_RADIUS_M  0.033f   // change to your calibrated value
   ```
   Reflash the ESP32 with the updated firmware.

2. Edit `ubot_ros2_control.xacro` — update the `wheel_radius` parameter in the `<ros2_control>` hardware tag.

3. Edit `ubot_controllers.yaml`:
   ```yaml
   wheel_radius: 0.033   # change to your calibrated value
   ```

4. Rebuild and re-source:
   ```bash
   cd /home/chibueze/uni-bot
   colcon build --packages-select ubot_bringup ubot_description ubot_control
   source install/setup.bash
   ```

5. Repeat the 1 m test to confirm.

### 1b. Wheel Separation Calibration

The wheel separation controls how much the robot thinks it has rotated per difference in left/right wheel travel.

**Procedure**:

1. Place the robot on the floor with a reference mark for its initial heading.
2. Command it to spin exactly 360° in place. Using teleop, send a pure rotation command and time it carefully, or use:
   ```bash
   ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist \
     "{linear: {x: 0.0}, angular: {z: 0.5}}"
   ```
   Then calculate the duration needed for 360° at 0.5 rad/s (≈12.6 seconds) and stop.
3. Check the reported yaw:
   ```bash
   ros2 topic echo /diff_drive_controller/odom --once
   ```
   Read `pose.pose.orientation` (convert to yaw, or watch for it approaching 0.0 after a full 360° spin since the robot starts at yaw=0).

**Interpretation**:

| Observed yaw drift after 360° | Action |
|---|---|
| Positive (rotated more than 360°) | Separation too **small** → **increase** `wheel_separation` |
| Negative (rotated less than 360°) | Separation too **large** → **decrease** `wheel_separation` |
| Near 0.0 rad | Separation is calibrated |

**Update procedure**: Same three-file update as for wheel radius, following the same rebuild steps.

### 1c. Iteration

These two calibrations interact slightly. After adjusting wheel separation, re-run the 1 m straight test to confirm radius is still correct, and vice versa. Two or three iterations is typically sufficient.

---

## 2. BNO055 IMU Calibration

The BNO055 IMU is defined in the launch file but **currently commented out** in `real_robot.launch.py`. To work with it, start it standalone:

```bash
ros2 launch bno055 bno055.launch.py
```

The deployment configuration used by `real_robot.launch.py` (when uncommented) is:
```
/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/bno055_params.yaml
```
This config sets `placement_axis_remap: 'P2'` for the physical mounting orientation.

### 2a. Monitor Calibration Status

```bash
ros2 topic echo /bno055/calib_status
```

The topic publishes a JSON string with four calibration scores, each ranging from 0 (uncalibrated) to 3 (fully calibrated):

```json
{"sys": 0, "gyro": 3, "accel": 0, "mag": 1}
```

Target: all four values at **3**. At minimum, `gyro: 3` is required before the IMU data is useful for navigation.

### 2b. Calibration Motions

| Sensor | How to calibrate |
|---|---|
| **Gyroscope** | Keep the robot **completely still** for several seconds |
| **Accelerometer** | Tilt the robot into several different stable orientations (at least 6 positions) |
| **Magnetometer** | Slowly rotate the robot through a full 360°, away from metal objects |

The system calibration (`sys`) reaches 3 automatically once all individual sensors reach 3.

### 2c. Save Calibration Offsets

Once all scores reach 3, call the calibration service to retrieve the offset values:

```bash
ros2 service call /bno055/calibration_request std_srvs/srv/Trigger {}
```

The response message contains the calibration offset values as a string. Copy these values into `bno055_params.yaml` under the `calibration_offset_*` parameters and set `set_offsets: true`. The BNO055 will then apply the stored offsets on each startup, skipping the motion-based calibration routine.

### 2d. Enabling IMU + EKF Fusion

After calibrating the IMU, you can enable the full EKF pipeline:

1. In `real_robot.launch.py`, uncomment `bno055_node` (line ~125) and `ekf_node` (lines ~110-116).
2. The EKF configuration is in `/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/real_ekf.yaml`:
   - Fuses `/diff_drive_controller/odom` (wheel odometry) with `/bno055/imu` (yaw angular velocity)
   - Publishes the filtered result to `/odometry/filtered`
   - Runs at 30 Hz in `two_d_mode`
3. Rebuild `ubot_bringup` and re-source after any launch file changes.

---

## 3. PID Gain Re-Tuning

**Context**: The current PID gains in `diff_controller.h` are explicitly marked as **starting values pending re-tuning**:

```cpp
// Starting gains — re-tune after flashing
// (previous values were calibrated with a double min_pwm bug)
float leftKp  = 5.0f;  float leftKi  = 9.0f;  float leftKd  = 0.0f;
float rightKp = 5.0f;  float rightKi = 9.0f;  float rightKd = 0.0f;
```

The gains are not validated final values. If you observe oscillation, wheel speed overshoot, or poor tracking, re-tune using the procedure below.

### 3a. Enable Diagnostic Streaming

```bash
ros2 run ubot_debugger diag_publisher
```

This polls the ESP32's `q` command at 10 Hz and publishes individual `Float64` topics.

### 3b. Open PlotJuggler and Subscribe

Subscribe to these topics to observe the PID step response:

```
/horizon/diag/rpm/left_target    — commanded RPM (left wheel)
/horizon/diag/rpm/left_actual    — filtered RPM feedback (left wheel)
/horizon/diag/pid/left_error     — tracking error (left wheel)
/horizon/diag/rpm/right_target   — commanded RPM (right wheel)
/horizon/diag/rpm/right_actual   — filtered RPM feedback (right wheel)
/horizon/diag/pid/right_error    — tracking error (right wheel)
```

### 3c. Drive and Observe

Drive the robot at a known constant velocity (e.g. 0.1 m/s forward) and observe the step response:

- **Overshoot** (actual exceeds target then drops): `Kp` too high, or `Ki` too high
- **Slow settling** (actual slowly creeps to target): `Kp` too low
- **Steady-state error** (actual never quite reaches target): `Ki` too low
- **Oscillation** (actual bounces around target): `Kd` needed, or `Kp`/`Ki` too high

### 3d. Update Gains at Runtime (No Reflash Required)

Use the serial commands to update gains immediately for testing. The command format depends on which wheel:

**Update left wheel** (`l` command — format: `Kp:Ki:Kd:Min`):
```bash
# Using picocom (must close ros2_control first, or use the q-command approach)
picocom /dev/ttyUSB0 -b 115200
l 6.0:10.0:0.0:50
```

**Update right wheel** (`f` command — format: `Kp:Ki:Kd:Min`):
```bash
f 6.0:10.0:0.0:50
```

> **Warning**: The `u` command updates **both** wheels simultaneously but uses a **legacy argument order** (`Kp:Kd:Ki:Min` — note `Kd` and `Ki` swapped). Use `l` and `f` separately to avoid confusion.

> **Note on serial port access**: If `ros2_control_node` is running, it holds `/dev/ttyUSB0`. The `diag_publisher` uses the `q` read-only command alongside it. To send gain-update commands interactively, you may need to stop `ros2_control_node` first, or implement a service in `ubot_control` to forward commands.

### 3e. Make Gains Permanent

Runtime gain updates via serial commands are lost on ESP32 reset. To make them permanent:

1. Edit `diff_controller.h`:
   ```cpp
   float leftKp  = 6.0f;  float leftKi  = 10.0f;  float leftKd  = 0.0f;
   float rightKp = 6.0f;  float rightKi = 10.0f;  float rightKd = 0.0f;
   ```

2. Reflash the ESP32 with the updated firmware (`Ros-esp32_bridge.ino`).

3. Verify in PlotJuggler that the step response is improved.

### 3f. Tuning Targets

- Minimal overshoot (< 10% of target RPM)
- Fast settling time (< 0.5 s for typical velocity commands)
- Low steady-state error (< 2 RPM at operating speeds)
- No oscillation at constant velocity

---

## See Also

- [Running the Full Stack](running.md) — operational procedure
- [Tuning](tuning.md) — Nav2 planner and costmap parameter tuning
- [Troubleshooting](troubleshooting.md) — fault finding for odometry and motor issues
