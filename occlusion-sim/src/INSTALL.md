# INSTALL — ubot Workspace

## Prerequisites

| Requirement | Version |
|---|---|
| Ubuntu | 24.04 LTS (Noble) |
| ROS2 | Jazzy Jalisco |
| colcon | latest (`pip install colcon-common-extensions`) |
| Python | ≥ 3.10 |
| micro-ROS / ESP32 toolchain | Arduino IDE or PlatformIO (for embedded firmware only) |

---

## 1. ROS2 Jazzy

Follow the official install instructions. Recommended: desktop install.

```bash
# Verify after install
ros2 --version   # should print "ros2cli ... Jazzy"
```

---

## 2. System dependencies

```bash
sudo apt update && sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-pip \
  libserial-dev \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-robot-localization \
  ros-jazzy-nav2-bringup \
  ros-jazzy-slam-toolbox \
  ros-jazzy-twist-stamper \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-diff-drive-controller
```

---

## 3. Python package dependencies

```bash
pip install pyserial mkdocs mkdocs-material mkdocs-mermaid2-plugin
```

---

## 4. Clone or locate the workspace

```bash
# If you are reading this file you have already located the workspace root.
# The workspace root is the parent of this file's directory:
#   /home/chibueze/uni-bot/        ← colcon workspace root
#     src/                         ← this file is here
#       ubot/                      ← first-party packages
#       bno055/                    ← vendored IMU driver
#       ldlidar_stl_ros2/          ← vendored LDLidar driver (LIVE)
#       sllidar_ros2/              ← vendored RPLidar driver
#       Ros-esp32_bridge/          ← LIVE ESP32 firmware (Arduino sketch)
#       esp32/hacker/              ← ORPHANED micro-ROS firmware — do NOT flash
```

---

## 5. rosdep

```bash
cd /home/chibueze/uni-bot
rosdep install --from-paths src --ignore-src -r -y
```

---

## 6. Build

```bash
cd /home/chibueze/uni-bot
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

Expected output: 7 packages finished. Any `stderr` about `position_feedback` or deprecated parameter names is non-fatal; see [Known Issues](docs/architecture/issues.md).

---

## 7. Source the install space

Add this to your `~/.bashrc` or run before every session:

```bash
source /opt/ros/jazzy/setup.bash
source /home/chibueze/uni-bot/install/setup.bash
```

---

## 8. udev rules (serial devices)

The real-robot bringup expects:

| Device | Default path | Usage |
|---|---|---|
| ESP32 NodeMCU (USB-serial) | `/dev/ttyUSB0` | ros2_control hardware interface + diag_publisher |
| LDLidar LD19 | `/dev/ttyUSB1` | ldlidar_stl_ros2_node |

Create persistent symlinks (optional but recommended):

```bash
# Find the exact ttyUSB* assigned at boot, then create udev rules:
sudo nano /etc/udev/rules.d/99-ubot.rules
```

Example rule content:
```
# ESP32 NodeMCU (CH340 chip)
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="ttyUSB_ESP32"
# LDLidar LD19 (CP2102 chip)
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="ttyUSB_LIDAR"
```

Update `ubot_ros2_control.xacro` and `real_robot.launch.py` if you change from the default `/dev/ttyUSB0` / `/dev/ttyUSB1` paths.

---

## 9. Flash the ESP32

Flash only the LIVE firmware, located at:
```
src/Ros-esp32_bridge/Ros-esp32_bridge.ino
```

**Do NOT flash** `src/esp32/hacker/hacker.ino` — this is an orphaned micro-ROS firmware that bypasses ros2_control and conflicts with the current ROS2 graph. See [Embedded Firmware](docs/hardware/embedded.md) for details.

Required Arduino libraries: none beyond ESP32 board support and the built-in `Serial`/`Serial2` peripherals. The firmware uses the `DualBTS7960MotorShieldESP32` driver included in the same directory.

---

## 10. Verify hardware connections

After flashing and powering the robot:

```bash
# Confirm ESP32 is enumerated
ls /dev/ttyUSB*

# Quick serial check — should respond with baudrate number
ros2 run ubot_control check_serial  # or: picocom /dev/ttyUSB0 -b 115200
# type 'b' + Enter — should return "115200"
# type 'e' + Enter — should return "<leftTicks> <rightTicks>"

# Confirm LDLidar
ls /dev/ttyUSB1    # should exist when LD19 is connected
```

---

## 11. Launch (real robot)

```bash
# Terminal 1 — core bringup (ros2_control + LiDAR + twist_stamper)
ros2 launch ubot_bringup real_robot.launch.py

# Terminal 2 — SLAM mapping
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=$(ros2 pkg prefix ubot_bringup)/share/ubot_bringup/config/mapper_params_online_async.yaml

# Terminal 3 — Nav2
ros2 launch nav2_bringup bringup_launch.py \
  params_file:=$(ros2 pkg prefix ubot_bringup)/share/ubot_bringup/config/nav2_params.yaml \
  use_sim_time:=False

# Terminal 4 — Teleoperation
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

See [Running Guide](docs/guides/running.md) for the full step-by-step procedure including map saving.

---

## 12. Build the documentation site

```bash
cd /home/chibueze/uni-bot/src
mkdocs build --strict            # validate and build to site/
mkdocs serve                     # live-reload dev server at http://127.0.0.1:8000
```
