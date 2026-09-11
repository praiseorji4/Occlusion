# Building from Source

This guide walks through cloning the workspace and building all 7 packages from scratch on a machine that already has ROS 2 Jazzy installed.

---

## Prerequisites

Install system dependencies before building:

```bash
# ROS 2 Jazzy base (if not already installed — follow official ROS 2 docs)
# https://docs.ros.org/en/jazzy/Installation.html

# colcon build tool
sudo apt install python3-colcon-common-extensions

# libserial — required by ubot_control (UbotHardware hardware interface)
sudo apt install libserial-dev

# pyserial — required by ubot_debugger (diag_publisher.py)
pip install pyserial
# or: sudo apt install python3-serial

# rosdep (for resolving ROS package dependencies)
sudo apt install python3-rosdep
sudo rosdep init      # only needed once per machine
rosdep update
```

---

## Workspace Location

The workspace root is `/home/chibueze/uni-bot`. All `colcon` commands must be run from this directory, **not** from inside `src/`.

```
/home/chibueze/uni-bot/
├── src/                    ← all packages live here
│   ├── bno055/             ← vendored IMU driver (Flynneva)
│   ├── ldlidar_stl_ros2/   ← vendored LDLidar LD19 driver
│   ├── sllidar_ros2/       ← vendored RPLiDAR driver (not used on real robot)
│   └── ubot/
│       ├── ubot_bringup/   ← launch files + config YAMLs
│       ├── ubot_control/   ← ros2_control hardware interface (UbotHardware)
│       ├── ubot_debugger/  ← diagnostic publisher
│       └── ubot_description/ ← URDF/xacro, meshes, RViz configs
├── build/
├── install/
└── log/
```

---

## Resolve Dependencies

From the workspace root:

```bash
cd /home/chibueze/uni-bot

source /opt/ros/jazzy/setup.bash

rosdep install --from-paths src --ignore-src -r -y
```

This reads each package's `package.xml` and installs missing ROS dependencies. The `--ignore-src` flag skips packages that are already present in `src/` (i.e., the vendored drivers). The `-r` flag continues past packages whose dependencies cannot be resolved automatically.

---

## Build

```bash
cd /home/chibueze/uni-bot

colcon build --symlink-install
```

`--symlink-install` creates symlinks instead of copying Python scripts and config files, so edits to source files under `src/` take effect immediately without rebuilding.

> **Important**: Always run `colcon` from `/home/chibueze/uni-bot`, never from inside `src/`. Running colcon from `src/` will produce a broken build with install paths in the wrong location.

### Expected Output

A successful build prints a summary like:

```
Summary: 7 packages finished [<time>]
```

The 7 packages are: `bno055`, `ldlidar_stl_ros2`, `sllidar_ros2`, `ubot_bringup`, `ubot_control`, `ubot_debugger`, `ubot_description`.

---

## Sourcing the Workspace

After every successful build, source both overlays in each new terminal:

```bash
source /opt/ros/jazzy/setup.bash
source /home/chibueze/uni-bot/install/setup.bash
```

Add these two lines to your `~/.bashrc` to avoid repeating them:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source /home/chibueze/uni-bot/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Building Individual Packages

To rebuild a single package after making changes:

```bash
cd /home/chibueze/uni-bot

colcon build --packages-select ubot_control
```

Common single-package builds:

```bash
colcon build --packages-select ubot_control      # hardware interface (C++)
colcon build --packages-select ubot_bringup      # launch files + config
colcon build --packages-select ubot_description  # URDF/xacro changes
colcon build --packages-select ubot_debugger     # diagnostic publisher (Python)
```

After building a C++ package (`ubot_control`, `ubot_description`, `ubot_bringup`), re-source the workspace:

```bash
source /home/chibueze/uni-bot/install/setup.bash
```

Python packages (`ubot_debugger`, `bno055`) with `--symlink-install` do not require re-sourcing for script changes — only for `package.xml` / `setup.py` changes.

---

## Common Build Errors and Fixes

### `libserial not found` / `Could not find PkgConfig module: libserial`

```bash
sudo apt install libserial-dev
```

Then re-run `colcon build --packages-select ubot_control`.

### `Could not find package ros2_control` or missing `controller_manager`

```bash
sudo apt install ros-jazzy-ros2-control ros-jazzy-ros2-controllers
```

### `Could not find package diff_drive_controller`

```bash
sudo apt install ros-jazzy-diff-drive-controller
```

### `Could not find package slam_toolbox`

```bash
sudo apt install ros-jazzy-slam-toolbox
```

### `Could not find package nav2_bringup` or missing Nav2 packages

```bash
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup
```

### `Could not find package twist_stamper`

```bash
sudo apt install ros-jazzy-twist-stamper
```

### `Could not find package robot_localization`

```bash
sudo apt install ros-jazzy-robot-localization
```

### Python import errors during build (`ModuleNotFoundError: pyserial`)

```bash
pip install pyserial
# or
sudo apt install python3-serial
```

### `colcon build` produces `0 packages finished`

You are likely running colcon from inside `src/` or a subdirectory. Change to the workspace root:

```bash
cd /home/chibueze/uni-bot
colcon build --symlink-install
```

---

## See Also

- [Getting Started](getting_started.md) — hardware setup and first launch
- [Running the Full Stack](running.md) — operational procedure after a successful build
- [Troubleshooting](troubleshooting.md) — fault finding for runtime issues
