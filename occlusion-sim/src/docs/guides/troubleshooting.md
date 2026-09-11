# Troubleshooting

This guide is structured as a decision tree. Start at the top of each section and work through the numbered steps in order — each step either resolves the issue or narrows it down for the next step.

All commands assume the workspace is sourced:
```bash
source /opt/ros/jazzy/setup.bash
source /home/chibueze/uni-bot/install/setup.bash
```

---

## Problem: Robot Does Not Move After Launch

### Step 1 — Is `diff_drive_controller` spawned and active?

Check the terminal running `real_robot.launch.py`. The diff_drive_controller spawner fires at **6 seconds** after launch. Look for:

```
Successfully activated 'diff_drive_controller'
```

If this message never appears, the spawner failed. Check whether the `ros2_control_node` itself started (look for `UbotHardware` initialisation messages). If the controller failed to activate, proceed to Step 4.

### Step 2 — Is `/cmd_vel` being published?

```bash
ros2 topic echo /cmd_vel
```

If no messages appear, the source of velocity commands is not running:
- **Teleop**: is `teleop_twist_keyboard` running in another terminal with focus?
- **Nav2**: is `nav2_bringup` running and has a goal been sent?

### Step 3 — Is `/diff_drive_controller/cmd_vel` receiving stamped commands?

```bash
ros2 topic echo /diff_drive_controller/cmd_vel
```

If `/cmd_vel` has messages but `/diff_drive_controller/cmd_vel` does not, the `twist_stamper` has a problem. Verify it is running:

```bash
ros2 node list | grep twist_stamper
```

The `twist_stamper` node is started by `real_robot.launch.py`. It subscribes to `/cmd_vel` (remapped from `/cmd_vel_in`) and publishes to `/diff_drive_controller/cmd_vel` (remapped from `/cmd_vel_out`) with `frame_id=base_footprint`.

### Step 4 — Is the hardware interface connected?

In the `real_robot.launch.py` terminal, look for errors containing `UbotHardware` or `serial`. A serial connection failure looks like:

```
[ros2_control_node] [ERROR] [UbotHardware]: Failed to open serial port /dev/ttyUSB0
```

If you see this, proceed to Step 5.

### Step 5 — Is `/dev/ttyUSB0` accessible?

```bash
ls -la /dev/ttyUSB*
```

If `/dev/ttyUSB0` is missing, check the USB cable between the laptop and the ESP32. If it appears but is inaccessible:

```bash
groups $USER    # must include 'dialout'
```

If `dialout` is not in your groups:
```bash
sudo usermod -aG dialout $USER
# Log out and back in for the change to take effect
```

### Step 6 — Quick serial test: does the ESP32 respond?

Stop `real_robot.launch.py` first (only one process can hold the port). Then:

```bash
picocom /dev/ttyUSB0 -b 115200
```

Type `e` and press Enter. You should receive two numbers:

```
0 0
```

If you get no response or garbled output:
- The ESP32 may not have the correct firmware flashed (`Ros-esp32_bridge.ino`).
- The baud rate must be **115200**.
- Check that `Serial` (USB, GPIO 1/3) is the port being used — not `Serial2` (TX GPIO 16 only, diagnostic stream to `/dev/ttyUSB1`).

Press `Ctrl-A Ctrl-X` to exit picocom.

---

## Problem: LiDAR Not Showing in RViz

### Step 1 — Is `/scan` publishing?

```bash
ros2 topic hz /scan
```

Expected: approximately **10 Hz**. If there is no output, the `ldlidar_node` is not running or failed to open the serial port.

Check the `real_robot.launch.py` terminal for messages from `ldlidar_node`. A successful connection looks like:

```
[ldlidar_node] LDLiDAR_LD19 connected on /dev/ttyUSB1 @ 230400
```

### Step 2 — Is the LiDAR device present?

```bash
ls /dev/ttyUSB1
```

If it does not exist, the LDLidar LD19 is not connected or not powered. Check the USB cable and spin up the physical unit.

If present but `/scan` is still not publishing, the `ldlidar_node` may have failed silently. Restart `real_robot.launch.py` and check for error messages at launch.

### Step 3 — Is the correct frame ID configured in RViz?

The LiDAR publishes scans with `frame_id: lidar_link` (set in `real_robot.launch.py`). In RViz, the `LaserScan` display must be configured to use the topic `/scan` and the fixed frame must be `odom` or `map` (with TF active). If you see "No transform from [lidar_link] to [odom]", the TF tree is broken — check that `real_robot.launch.py` is running and the `robot_state_publisher` has loaded the URDF.

---

## Problem: `diag_publisher` Exits Immediately

The `ubot_debugger` diagnostic publisher exits with `SystemExit(1)` if it cannot open the serial port:

```bash
ros2 run ubot_debugger diag_publisher
```

Log message on failure:
```
[FATAL] [diag_publisher]: Failed to open serial port /dev/ttyUSB0
```

### Diagnosis

**Is another process holding `/dev/ttyUSB0`?**

When `real_robot.launch.py` is running, the `UbotHardware` hardware interface holds `/dev/ttyUSB0`. The `diag_publisher` sends the `q` command (a read-only diagnostic snapshot) to the same port. In principle, both can coexist since `diag_publisher` only reads responses to `q` queries. However, if the hardware interface has an exclusive lock on the port, `diag_publisher` will fail.

Check which process owns the port:

```bash
fuser /dev/ttyUSB0
```

If only `ros2_control_node` holds it, try running `diag_publisher` — it may still work via the same file descriptor. If it fails, you have two options:

1. Use the `Serial2` diagnostic stream instead: connect a USB-serial adapter to the ESP32's TX GPIO 16 pin and read from `/dev/ttyUSB1` (the same port the LDLidar uses — you would need a USB hub or to reassign ports).
2. Stop `real_robot.launch.py`, run `diag_publisher` standalone for diagnostic capture, then restart the full stack.

The `diag_publisher` does **not** retry on failure and does **not** auto-restart. If the port becomes available, you must restart the node manually.

---

## Problem: Odometry Drifts Badly

### Step 1 — Run the 1 m straight test

Drive exactly 1 m forward and check:

```bash
ros2 topic echo /diff_drive_controller/odom --once
```

If `pose.pose.position.x` differs significantly from 1.0 m, `wheel_radius` is wrong. See [Calibration — Wheel Radius Calibration](calibration.md).

### Step 2 — Run the 360° spin test

Spin the robot exactly 360° in place and check whether yaw returns to ~0.0 rad. Significant drift means `wheel_separation` is wrong. See [Calibration — Wheel Separation Calibration](calibration.md).

### Step 3 — Verify all three files are in sync

The calibration constants must match in all three locations:

| File | What to check |
|---|---|
| `diff_controller.h` | `WHEEL_RADIUS_M`, `WHEEL_SEPARATION_M` |
| `ubot_ros2_control.xacro` | `wheel_radius`, `wheel_separation` params |
| `ubot_controllers.yaml` | `wheel_radius`, `wheel_separation` |

Current values: `wheel_radius=0.033`, `wheel_separation=0.264204`. If any of these three differ from each other, odometry will drift even if each file individually seems reasonable.

### Step 4 — Confirm `open_loop: false`

```bash
grep open_loop /home/chibueze/uni-bot/src/ubot/ubot_bringup/config/ubot_controllers.yaml
```

This must show `open_loop: false`. If it shows `true`, the controller ignores encoder feedback and integrates the commanded velocity instead — odometry will drift immediately even with correct geometry.

### Step 5 — Consider enabling EKF + BNO055

The `robot_localization` EKF node and BNO055 IMU are fully configured but **commented out** in `real_robot.launch.py`. Enabling IMU fusion would significantly reduce long-term heading drift. See [Calibration — Enabling IMU + EKF Fusion](calibration.md).

---

## Problem: Nav2 Fails to Plan

### Step 1 — Is there a map?

```bash
ros2 topic echo /map --once
```

If no message arrives, SLAM Toolbox is not running or has not yet built a map. Start SLAM Toolbox (see [Running — Terminal 2](running.md)) and drive the robot around the environment until a map appears in RViz.

### Step 2 — Is the `map → odom` TF present?

```bash
ros2 run tf2_tools view_frames
```

Open the generated `frames.pdf` (or `frames.gv`). You must see the chain:

```
map → odom → base_footprint
```

If `map` is absent, SLAM Toolbox is not broadcasting that TF edge. Check that SLAM is running and has received at least a few scans.

### Step 3 — Is `/scan` publishing?

```bash
ros2 topic hz /scan
```

Both the local and global costmaps subscribe to `/scan` for obstacle detection. If the LiDAR is not publishing, costmaps will be empty and the planner may report no valid path or plan through obstacles. Fix the LiDAR issue first (see above).

### Step 4 — Inflation radius vs robot radius

Verify the values in `nav2_params.yaml`:

```yaml
robot_radius: 0.15            # m
inflation_radius: 0.30        # m  (local costmap)
```

`inflation_radius` must be greater than or equal to `robot_radius` (0.15 m). If `inflation_radius` is smaller, the planner may route the robot through spaces where it will physically collide. The current value (0.30 m) is safe.

If Nav2 reports "No valid path found" in a space the robot should fit through, the inflation radius may be too large. Try reducing it incrementally (e.g. to 0.20 m) and re-testing.

---

## Problem: BNO055 Node Crashes on Start

The BNO055 node performs hardware communication during `configure()` and exits hard on failure.

### Step 1 — Check `placement_axis_remap`

In `/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/bno055_params.yaml`, the `placement_axis_remap` parameter must be one of `P0` through `P7`. An invalid value causes an unhandled `KeyError` in `SensorService.py` at startup. The current deployment config uses `P2`.

### Step 2 — Verify I2C device is accessible

```bash
i2cdetect -y 1
```

The BNO055 should appear at address `0x28` (ADDR pin low) or `0x29` (ADDR pin high). If neither address shows, check:
- I2C wiring (SDA/SCL connected correctly)
- 3.3V power to the BNO055 module
- Whether the I2C bus number is correct (the `i2c_bus` parameter in `bno055_params.yaml`)

### Step 3 — Chip ID verification failure

The `SensorService.configure()` method calls `sys.exit(1)` if the chip ID read from the BNO055 does not match the expected value. This means the I2C communication is physically working but returning wrong data. Causes:
- Wrong I2C address configured (check `i2c_address` in `bno055_params.yaml`)
- Hardware fault on the BNO055 module
- I2C bus noise / missing pull-up resistors

---

## Problem: ESP32 Motors Stop Unexpectedly

### Symptom

Motors brake and stop mid-operation. The `real_robot.launch.py` terminal (which reads `Serial2` output via `/dev/ttyUSB1`) or a separate terminal monitoring the serial stream may show:

```
--- 10s timeout: stopping motors ---
```

### Cause: 10-second auto-stop watchdog

The ESP32 firmware (`diff_controller.h`) contains a watchdog timer:

```cpp
#define CMD_TIMEOUT_MS  10000UL   // 10 seconds
```

If the firmware receives no motor command for 10 consecutive seconds, it calls:
```cpp
setMotorBrakes(200, 200);
resetPID();
```

and logs the timeout message to `Serial2`.

### Diagnosis

1. **Is `diff_drive_controller` running?**
   ```bash
   ros2 node list | grep controller
   ```
   The `diff_drive_controller` must be active and sending commands to the hardware interface.

2. **Is `/diff_drive_controller/cmd_vel` receiving commands?**
   ```bash
   ros2 topic hz /diff_drive_controller/cmd_vel
   ```
   If the controller is running but nothing is publishing to this topic (no teleop, no Nav2 goal), the watchdog will fire after 10 seconds. This is **expected behaviour** — the firmware is designed to stop the robot if ROS communication is lost.

3. **Is `twist_stamper` running?**
   The `diff_drive_controller` requires stamped `TwistStamped` messages. If `twist_stamper` is not running, velocity commands from teleop will be on `/cmd_vel` (plain `Twist`) but not forwarded to `/diff_drive_controller/cmd_vel`.

4. **Was the robot intentionally stopped?**
   When teleop is used and all keys are released, `teleop_twist_keyboard` sends zero-velocity commands. After 10 seconds of zero commands, the watchdog fires — this is normal. The robot will resume when it receives a non-zero command again.

---

## See Also

- [Getting Started](getting_started.md) — hardware setup and first launch
- [Running the Full Stack](running.md) — complete operational procedure
- [Calibration](calibration.md) — wheel geometry and PID tuning
- [Tuning](tuning.md) — Nav2 and controller parameter tuning
