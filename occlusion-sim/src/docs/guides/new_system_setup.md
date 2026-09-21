# New system setup — every install the project needs

Complete install list for bringing up a new machine, derived from three sources: the
**ROS 2 Development Issues & Fixes log** (23 pages, Uni-Bot, v1.2), the **declared dependencies**
in every `package.xml` and launch file in this workspace, and the **failures actually hit** during
the WSL and Pi bring-ups.

Ordered so that each step's prerequisites are already satisfied — several of the log's incidents
were caused purely by running the right command too early.

> Supersedes the shorter `src/INSTALL.md`, which predates the mono-nav and eval packages and
> still references the old `/home/chibueze/uni-bot` path.

---

## 0. Pick the machine role

Not every machine needs everything. Four roles appear in this project:

| Role | What it runs | Needs |
|---|---|---|
| **A. Robot (Raspberry Pi 5, arm64)** | hardware interface, drivers, SLAM, nav2 | §1–§6, §8, §9 — **no Gazebo** |
| **B. Dev / sim (WSL2 or Ubuntu desktop)** | Gazebo, RViz, the eval harness | §1–§5, §7, §9, §10 |
| **C. GPU server** | monocular depth inference | §1–§3, §8.3 (torch/transformers) |
| **D. Viewer laptop** | RViz only, against the robot | §1–§4, §9 |

Roles B and D both need **§10 WSL mirrored networking** if they run under WSL, or DDS discovery
to the robot silently fails.

---

## 1. Fix the apt sources first — before installing anything

Two separate incidents in the log (#2 and #18) trace to the same root cause, and both present as
impossible-looking dependency errors:

```
libc6-dev : Depends: libc6 (= 2.39-0ubuntu8) but 2.39-0ubuntu8.5 is to be installed
dpkg-dev : Depends: bzip2 but it is not installable
E: Unable to correct problems, you have held broken packages.
```

Ubuntu 24.04 **Server** images (the usual Pi image) use the deb822 format and ship with only
`noble` and `noble-security` enabled. Without `noble-updates`, apt cannot see the package
revisions ROS 2 depends on, so perfectly valid packages look uninstallable.

```bash
sudo nano /etc/apt/sources.list.d/ubuntu.sources
```

The **first** block must read (on arm64; on x86-64 use `http://archive.ubuntu.com/ubuntu/`):

```
Types: deb
URIs: http://ports.ubuntu.com/ubuntu-ports/
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
```

Leave the second (`noble-security`) block alone. Then:

```bash
sudo apt clean && sudo apt update && sudo apt full-upgrade -y
```

**Do this before the ROS install, not after.** Every later step assumes it.

---

## 2. Locale, ROS apt repository, and ROS 2 Jazzy

```bash
sudo apt install -y locales && sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && export LANG=en_US.UTF-8
```

```bash
sudo apt install -y software-properties-common curl gnupg lsb-release
sudo add-apt-repository -y universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
```

Then the base install — **desktop** on roles B and D (it brings RViz and the GUI stack),
**ros-base** on the robot and the GPU server:

```bash
# Roles B, D
sudo apt install -y ros-jazzy-desktop ros-dev-tools
```
```bash
# Roles A, C
sudo apt install -y ros-jazzy-ros-base ros-dev-tools
```

---

## 3. rosdep — upgrade pip *first*

Log issue #1: `rosdep install` fails on a fresh system, and the fix is ordered, not optional —
rosdep needs a current pip to fetch its package manifests before initialisation.

```bash
sudo apt install -y python3-pip python3-rosdep python3-colcon-common-extensions
python3 -m pip install --upgrade pip --break-system-packages
sudo rosdep init          # harmless "already exists" if re-run
rosdep update
```

Once the workspace is cloned, this resolves everything declared in the manifests and is the
correct fix for the log's "executable not found on the PATH" launch failure (#8), which is a
stale rosdep cache rather than a launch-file bug:

```bash
cd ~/occl_ws
rosdep update --rosdistro=$ROS_DISTRO
rosdep install --from-paths src -i -y --rosdistro jazzy
```

---

## 4. ROS 2 packages the workspace actually uses

Grouped by what pulls them in. All are `apt`, all prefixed `ros-jazzy-`.

### 4.1 Control stack — needed wherever the robot moves (A and B)

```bash
sudo apt install -y \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-controller-manager \
  ros-jazzy-hardware-interface \
  ros-jazzy-diff-drive-controller \
  ros-jazzy-joint-state-broadcaster \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-xacro \
  ros-jazzy-twist-stamper \
  ros-jazzy-teleop-twist-keyboard
```

`hardware_interface` missing is the log's issue #6 (`Findhardware_interface.cmake` not found).
`twist-stamper` is issue #15 and is **not optional on Jazzy**: `diff_drive_controller` requires
`TwistStamped`, while `teleop_twist_keyboard` publishes unstamped `Twist`. Without the stamper
the robot simply does not move, with no error.

### 4.2 Localisation, SLAM and navigation (A, B)

```bash
sudo apt install -y \
  ros-jazzy-robot-localization \
  ros-jazzy-slam-toolbox \
  ros-jazzy-navigation2 \
  ros-jazzy-nav2-bringup \
  ros-jazzy-nav2-msgs \
  ros-jazzy-nav2-common
```

### 4.3 Sensors, perception and messages (A, B)

```bash
sudo apt install -y \
  ros-jazzy-imu-tools \
  ros-jazzy-rviz-imu-plugin \
  ros-jazzy-cv-bridge \
  ros-jazzy-image-transport \
  ros-jazzy-image-transport-plugins \
  ros-jazzy-v4l2-camera \
  ros-jazzy-example-interfaces \
  ros-jazzy-tf-transformations \
  ros-jazzy-tf2-tools \
  ros-jazzy-rosbag2 \
  ros-jazzy-rosbag2-storage-mcap
```

`rviz-imu-plugin` is the one whose absence produces the RViz error
`the class rviz_imu_plugin/Imu ... does not exist`, seen in our own launch logs.
`example-interfaces` is the log's `ModuleNotFoundError: No module named 'example_interfaces'`.
`rosbag2` is required by `ubot_eval`, which imports `rosbag2_py` directly.

### 4.4 Simulation — role B only

```bash
sudo apt install -y \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-interfaces \
  ros-jazzy-gz-ros2-control
```

`ros-jazzy-ros-gz` pulls **Gazebo Harmonic** (gz-sim 8) as a vendored dependency on Jazzy — do
not add the upstream Gazebo apt repository as well, or you get two copies of the sensors system
and a segfault a few seconds after spawn (a comment to this effect is already in
`worlds/sonoma_occlusion.sdf`).

`ros-gz-interfaces` is required for contact sensing: `/bumper/contacts` is
`ros_gz_interfaces/msg/Contacts`.

---

## 5. System libraries (non-ROS)

```bash
sudo apt install -y \
  libserial-dev \
  libi2c-dev \
  python3-serial \
  python3-smbus \
  python3-numpy python3-yaml \
  git build-essential cmake
```

- `libserial-dev` — the ESP32/Arduino serial link used by `ubot_control` (log #6)
- `libi2c-dev` — otherwise `fatal error: i2c/smbus.h: No such file or directory` (log #10)
- `python3-serial` — **not** `python-serial`, which does not exist; ROS 2 is Python 3 only
  (log #4). Test a link with `python3 -m serial.tools.miniterm /dev/ttyUSB0 57600`

---

## 6. Drivers built from source (role A)

Not available as apt packages. `src/ubot/ubot.repos` lists the lidar and IMU drivers; launch
files additionally reference `ldlidar_stl_ros2`, `sllidar_ros2` and `oradar_lidar`.

**The robot currently uses the MS200 (Oradar).** Its upstream package ships configured for ROS 1
and will not build until switched — log issues on pages 20–21:

```bash
# 1. src/MS200_ros/CMakeLists.txt line 7
set(COMPILE_METHOD COLCON)

# 2. use the ROS 2 manifest the package already ships
cp src/MS200_ros/package_ros2.xml src/MS200_ros/package.xml

# 3. clear the poisoned CMake cache, or the stale value persists
rm -rf build/oradar_lidar install/oradar_lidar
```

Verify before building — this is the check that catches it early:

```bash
colcon list | grep oradar   # must read (ros.ament_cmake), NOT (ros.catkin)
```

In the build log, `ROS Not Found, Ros Support is turned Off!` is expected and harmless (that is
ROS 1); the line that must now appear is `ROS2 Found. ROS2 Support is turned On!`.

---

## 7. Python for the eval and analysis stack (role B)

```bash
sudo apt install -y python3-matplotlib python3-scipy python3-pandas
```

`ubot_eval` and `research/nav_eval/` need numpy, scipy, pandas, matplotlib and pyyaml only —
no new dependencies beyond these.

---

## 8. Camera and depth model

### 8.1 OAK-D Lite — pin depthai below 3

```bash
python3 -m pip install "depthai<3" --break-system-packages
```

**The version pin is required.** depthai v3 removed `ColorCamera` and `XLinkOut`, which this
project's `oak_rgb_node` uses; installing the latest gives an import-time failure. 2.33.0.0 is
the known-good version here.

### 8.2 udev rule — without it the device is invisible to a non-root user

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' \
  | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Plug the OAK-D into a **USB 3** port. On USB 2 it enumerates and then dies mid-stream with an
XLINK error under load — diagnosed on this machine with both USB 3 ports sitting empty.

### 8.3 Monocular depth model (role C, and B if running locally)

```bash
python3 -m pip install torch torchvision transformers --break-system-packages
```

On the GPU server install the CUDA build of torch for the installed driver, not the default
wheel. The model is `Depth-Anything-V2-Metric-Indoor-Small`, fetched on first run.

---

## 9. Post-install configuration — the steps people skip

### 9.1 Serial port access, and the reboot that is not optional

```bash
sudo adduser $USER dialout
sudo reboot
```

Without the reboot the group membership is not active, and you get
`avrdude: ser_open(): can't open device "/dev/ttyUSB0": Permission denied` (log #3). The log's
own lesson list calls this out as wasted time.

### 9.2 Source ROS in the shell

```bash
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
echo 'source ~/occl_ws/install/setup.bash' >> ~/.bashrc
```

Omitting the first line is exactly the `colcon build` failure where `ament_cmake` is "not found"
— hit during the WSL bring-up.

### 9.3 SSH logins need `.bash_profile` (log #13)

SSH starts a **login** shell, which reads `~/.bash_profile` and not `~/.bashrc`, so `ros2` is
missing over SSH while working fine locally:

```bash
cat >> ~/.bash_profile <<'EOF'
if [ -f ~/.bashrc ]; then . ~/.bashrc; fi
EOF
```

### 9.4 Check the serial port names before blaming anything else

```bash
ls -l /dev/tty*     # typically /dev/ttyUSB0 lidar, /dev/ttyACM0 the MCU
```

Wrong port assignments produce garbage encoder data and a shaking TF tree with no lidar — which
looks like a URDF or driver bug and is not (log #16).

### 9.5 One publisher of `odom -> base_footprint`

If EKF is used, the controller must not also publish the transform, or the robot flickers in
RViz as two nodes fight over the pose:

```yaml
diff_drive_controller:
  ros__parameters:
    enable_odom_tf: false      # the EKF is the only publisher
```

---

## 10. WSL only — mirrored networking

Without this, WSL sits behind a NAT'd adapter: `.local` names do not resolve and, more
importantly, **DDS discovery to the robot does not work**, so RViz in WSL never sees the Pi's
topics. Create `C:\Users\<you>\.wslconfig`:

```
[wsl2]
networkingMode=mirrored

[experimental]
hostAddressLoopback=true
```

Then `wsl --shutdown` from PowerShell and reopen. See
`ASAP/WSL2_Networking_local_Resolution_Fix.pdf` for the full diagnosis.

---

## 11. Build and verify

```bash
cd ~/occl_ws
rosdep install --from-paths src -i -y --rosdistro jazzy
colcon build --symlink-install
source install/setup.bash
```

Verification ladder — each step only meaningful if the previous passed:

```bash
ros2 --version                      # ros2cli ... jazzy
ros2 pkg list | grep -E "ubot_(description|bringup|control|mono_nav|eval)"
xacro src/ubot/ubot_description/urdf/body/ubot_robot.urdf.xacro use_gazebo:=true > /tmp/t.urdf
gz sdf -p /tmp/t.urdf | grep -c "collision name="     # role B: sanity-checks the URDF -> SDF path
ros2 launch ubot_description display.launch.py        # model in RViz, no mesh errors
ros2 launch ubot_bringup sim.launch.py world:=apartment world_name:=apartment   # role B
```

If `ros2 node list` comes back empty while `ros2 topic list` works, that is the CLI daemon, not
your system (log, page 17):

```bash
ros2 daemon stop && ros2 daemon start
```

---

## 12. Things the log records that are *not* installs

Worth reading before debugging, because each cost real time and none is fixed by a package:

- **Mesh URIs must be `package://`, never `file://` or `model://`** (log #12, #21). A `file://`
  path that works on one machine breaks on every other install location.
- **A Gazebo sensor needs the sensors system plugin**, or the topic exists and stays empty
  forever (log #20) — the same class of failure as the contact sensor needing
  `gz-sim-contact-system`, documented in `nav_eval_plan.md`.
- **Motor wiring direction** — two weeks lost to a control loop that was correct while the
  motors turned the wrong way (log #14). Verify direction conventions before tuning PID.
- **Firmware serial protocol must match the driver's parse format** — `m l:r` versus `m l r`
  gives "failed to parse token 0" and one wheel that never turns (log, page 20).
