# Getting Started

This guide is for first-time users. It walks you from unboxing to driving the robot in under ten minutes.

---

## What You Need

### Hardware

| Item | Notes |
|---|---|
| ubot robot (Horizon chassis) | 4WD, BTS7960 motor drivers, front-wheel-only encoders |
| USB cable (USB-A to USB-micro/B) | Connects laptop to ESP32 NodeMCU on `/dev/ttyUSB0` |
| LDLidar LD19 | Connected via USB on `/dev/ttyUSB1` at 230400 baud |
| Power supply / battery | Must power both the ESP32 and motor drivers |

### Software

- **ROS 2 Jazzy** installed and sourced (`/opt/ros/jazzy`)
- **Workspace built** — see [Building from Source](building.md) if you have not done this yet
- `teleop_twist_keyboard` installed (`sudo apt install ros-jazzy-teleop-twist-keyboard`)
- `libserial-dev` installed (required by the `ubot_control` hardware interface)

---

## 5-Minute Sanity Check

Run these before every session to catch hardware problems early.

### 1. Verify serial ports are present

```bash
ls -la /dev/ttyUSB*
```

Expected output:

```
crw-rw---- 1 root dialout ... /dev/ttyUSB0   # ESP32 (command channel)
crw-rw---- 1 root dialout ... /dev/ttyUSB1   # LDLidar LD19
```

If either device is missing, check USB connections and power. If you see `Permission denied` when accessing these ports, add yourself to the `dialout` group:

```bash
sudo usermod -aG dialout $USER   # then log out and back in
```

### 2. Confirm the ESP32 responds

```bash
picocom /dev/ttyUSB0 -b 115200
```

Type `e` then press Enter. You should receive two numbers representing the left and right encoder counts, for example:

```
0 0
```

Type `Ctrl-A Ctrl-X` to exit picocom.

If you get no response, or garbled output, the ESP32 firmware may not be running or the baud rate is wrong (must be 115200).

### 3. Confirm the LiDAR enumerates

```bash
ls -la /dev/ttyUSB1
```

The device must exist before launch. The LD19 driver connects at 230400 baud — the launch file sets this automatically.

---

## Your First Launch

Source the workspace, then start everything with one command:

```bash
source /opt/ros/jazzy/setup.bash
source /home/chibueze/uni-bot/install/setup.bash

ros2 launch ubot_bringup real_robot.launch.py
```

This starts six components:

| Component | Delay | What it does |
|---|---|---|
| `robot_state_publisher` | Immediate | Publishes URDF robot model and static TF frames |
| `ros2_control_node` (controller_manager) | Immediate | Loads `UbotHardware` plugin, opens `/dev/ttyUSB0` at 115200 |
| `joint_state_broadcaster` spawner | 5 s | Begins publishing `/joint_states` for all 4 wheels |
| `diff_drive_controller` spawner | 6 s | Activates wheel odometry and accepts velocity commands |
| `twist_stamper` | Immediate | Stamps plain `/cmd_vel` → `/diff_drive_controller/cmd_vel` |
| `ldlidar_node` | Immediate | Opens `/dev/ttyUSB1`, publishes `/scan` (lidar_link frame) |

Wait approximately 8 seconds after launch before expecting the robot to respond to commands.

---

## How to Verify It Is Working

Open a second terminal (with the workspace sourced) and run these checks:

### Check all topics are active

```bash
ros2 topic list
```

Key topics that must appear:

```
/diff_drive_controller/cmd_vel
/diff_drive_controller/odom
/joint_states
/scan
/tf
/tf_static
```

### Confirm odometry is publishing

```bash
ros2 topic echo /diff_drive_controller/odom
```

You should see an `Odometry` message updating at ~30 Hz with `pose.position.x/y` starting near 0.0.

### Confirm the LiDAR is streaming

```bash
ros2 topic hz /scan
```

Expected rate: ~10 Hz (the LD19 runs at 10 Hz scan rate in the current configuration).

---

## Teleoperation

The `diff_drive_controller` expects **stamped** `TwistStamped` commands on `/diff_drive_controller/cmd_vel`. The `twist_stamper` node (already launched by `real_robot.launch.py`) handles this automatically.

Run teleop in a new terminal:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

The command flow is:

```
teleop_twist_keyboard  -->  /cmd_vel  (Twist)
        |
   twist_stamper  (adds timestamp, frame_id=base_footprint)
        |
        v
/diff_drive_controller/cmd_vel  (TwistStamped)
        |
   diff_drive_controller  -->  ESP32  -->  motors
```

**Important**: The 10-second auto-stop watchdog in the ESP32 firmware (`CMD_TIMEOUT_MS = 10000 ms`) will brake the motors if no command is received for 10 seconds. Keep teleop running or send periodic commands to prevent this.

---

## When Things Go Wrong

If the robot does not respond, or sensors are missing, see [Troubleshooting](troubleshooting.md) for a step-by-step decision tree covering:

- Robot does not move after launch
- LiDAR not showing in RViz
- Odometry drifts badly
- Nav2 fails to plan
- ESP32 motors stop unexpectedly

---

## See Also

- [Building from Source](building.md) — workspace setup and build instructions
- [Running the Full Stack](running.md) — SLAM, Nav2, and map saving
- [Calibration](calibration.md) — wheel geometry and IMU calibration
- [Troubleshooting](troubleshooting.md) — decision-tree fault finding
