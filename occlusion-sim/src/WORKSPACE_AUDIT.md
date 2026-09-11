# Workspace Audit: /home/chibueze/uni-bot/src

Comprehensive review of the ubot ROS 2 workspace (excluding `esp` directory). **17 issues found** — organized by severity below.

---

## CRITICAL — Will cause immediate failure or incorrect behavior

### 1. Serial port conflict: RPLIDAR and ESP32 both on /dev/ttyUSB0 (real robot)
- **Where**: `ubot_bringup/launch/real_robot.launch.py:75` (RPLIDAR) and `ubot_description/urdf/ros2_control/ubot_ros2_control.xacro:43` (ESP32)
- **Problem**: Two processes trying to own the same serial port. Whichever opens last will fail with `OSError: [Errno 13] Permission denied: '/dev/ttyUSB0'`.
- **Impact**: Real robot will not boot. Hardware interface or LiDAR node will crash on startup.
- **Fix**: 
  1. Apply udev rules from `sllidar_ros2/scripts/rplidar.rules` to create a persistent `/dev/rplidar` symlink
  2. Update `real_robot.launch.py:75` to use `'serial_port': '/dev/rplidar'`
  3. Create a similar udev rule for the ESP32 (identify USB VID:PID) and update `ubot_ros2_control.xacro:43` to point to it (e.g., `/dev/esp32`)

### 2. Nav2 LiDAR topic mismatch — real robot costmaps receive no obstacles
- **Where**: `ubot_bringup/config/nav2_params.yaml:148,177`
- **Details**: 
  - RPLIDAR A1 publishes to `/scan` (sllidar driver default)
  - Both `local_costmap` (line 148) and `global_costmap` (line 177) subscribe to `/lidar`
- **Problem**: Nav2 costmaps never receive scan data. The robot will navigate treating the world as empty.
- **Impact**: Robot collision detection fails. Navigation will be unsafe.
- **Fix**: In `nav2_params.yaml`, change `topic: /lidar` to `topic: /scan` in both costmap observation sources.

---

## HIGH — Significant runtime problems

### 3. BNO055 IMU node not launched in real_robot.launch.py
- **Where**: `ubot_bringup/launch/real_robot.launch.py` — no BNO055 node declaration
- **Problem**: `/imu/data` topic won't exist on the real robot. If using SLAM Toolbox, it won't have IMU for scan matching refinement.
- **Impact**: Reduced localization accuracy if SLAM tries to fuse IMU.
- **Fix**: Add BNO055 node to `real_robot.launch.py` (copy from `bno055/launch/bno055.launch.py`). If using EKF on the real robot, also create a `real_ekf.yaml` (similar to `sim_ekf.yaml` but with `use_sim_time: false`) and add the EKF node to `laptop_slam_nav2.launch.py`.

### 4. bno055/package.xml missing sensor_msgs dependency
- **Where**: `bno055/package.xml` — exec_depends section
- **Details**: `SensorService.py` publishes `sensor_msgs/msg/Imu`, `sensor_msgs/msg/MagneticField`, `sensor_msgs/msg/Temperature`
- **Problem**: Only `std_msgs` and `example_interfaces` declared. On a clean install, `rosdep install` won't pull `sensor_msgs`.
- **Impact**: Build will succeed but runtime import will fail when bno055 tries to use sensor message types.
- **Fix**: Add to `bno055/package.xml`:
  ```xml
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  ```

### 5. ubot_bringup/package.xml missing exec_depends
- **Where**: `ubot_bringup/package.xml` — only `robot_localization` declared
- **Missing**: `twist_stamper`, `slam_toolbox`, `nav2_bringup`, `sllidar_ros2`, `controller_manager`, `robot_state_publisher`, `joint_state_broadcaster`, `diff_drive_controller`, `rviz2`
- **Problem**: `rosdep install --from-paths src` will miss all these. Leads to "package not found" errors at launch time.
- **Impact**: Deployment and CI/CD fragile.
- **Fix**: Add all above as `<exec_depend>` entries.

### 6. Velocity limits not activated in ubot_controllers.yaml (real robot)
- **Where**: `ubot_bringup/config/ubot_controllers.yaml:64-71`
- **Details**: Sets `linear.x.max_velocity: 0.5` but never sets `has_velocity_limits: true`. `sim_ubot_controllers.yaml` explicitly sets this flag.
- **Problem**: Depending on diff_drive_controller version, velocity limits may be silently ignored.
- **Impact**: Real robot may exceed 0.5 m/s limit, causing loss of control or motor damage.
- **Fix**: Add under `linear.x` and `angular.z` blocks:
  ```yaml
  has_velocity_limits: true
  has_acceleration_limits: true
  ```

---

## MEDIUM — Functional issues or confusing behavior

### 7. Dead /odom bridge in sim.launch.py
- **Where**: `ubot_bringup/launch/sim.launch.py:57` — bridges argument
- **Problem**: Bridges Gazebo `/odom` → ROS `/odom`. But no Gazebo plugin publishes to `/odom`. The standalone DiffDrive plugin in `gazebo_control.xacro` is NOT included in the URDF. ros2_control publishes to `/diff_drive_controller/odom` (not `/odom`).
- **Impact**: Bridge receives nothing, dead weight; doesn't cause errors but increases complexity.
- **Fix**: Remove `/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry` line from bridge arguments.

### 8. Filename with space: sllidar_a2m12_launch .py
- **Where**: `sllidar_ros2/launch/sllidar_a2m12_launch .py` (note space before `.py`)
- **Problem**: Python launch discovery will not find this file.
- **Impact**: Cannot launch this LiDAR model variant; import will fail.
- **Fix**: Rename to `sllidar_a2m12_launch.py`.

### 9. Nav2 commented out in laptop_slam_nav2.launch.py
- **Where**: `ubot_bringup/launch/laptop_slam_nav2.launch.py:~54` — `# nav2`
- **Problem**: File name suggests "slam_nav2" but Nav2 is never started. Only SLAM Toolbox and RViz run.
- **Impact**: Unclear if this is intentional or oversight; confuses users.
- **Fix**: Either uncomment Nav2 if it should be included, or rename to `laptop_slam.launch.py` to match actual functionality. Add a comment explaining if this is deliberate.

### 10. test_wheels.launch.py is empty
- **Where**: `ubot_bringup/launch/test_wheels.launch.py` — 0 bytes
- **Problem**: `ros2 launch ubot_bringup test_wheels.launch.py` crashes with `SyntaxError: unexpected EOF while parsing`.
- **Impact**: Cannot use this launch file; unclear if it's work-in-progress or dead code.
- **Fix**: Either implement it (e.g., load controllers, spawn diff_drive, allow manual testing via `ros2 topic pub`) or delete it.

### 11. position_feedback parameter may be deprecated
- **Where**: `ubot_bringup/config/ubot_controllers.yaml:86` — `position_feedback: true`
- **Problem**: This parameter was removed in newer diff_drive_controller releases.
- **Impact**: May emit deprecation warnings or be silently ignored depending on installed version.
- **Fix**: Verify with your diff_drive_controller version. If not supported, remove the line.

### 12. Dead file: gazebo_control.xacro not referenced
- **Where**: `ubot_description/urdf/gazebo/gazebo_control.xacro`
- **Problem**: Defines a standalone Gazebo DiffDrive plugin but is not included anywhere in `ubot_robot.urdf.xacro` or any other xacro file.
- **Impact**: Leftover code; if ever accidentally included, would conflict with `gz_ros2_control`. Confuses readers.
- **Fix**: Delete the file.

### 13. Large commented-out block in sim.launch.py
- **Where**: `ubot_bringup/launch/sim.launch.py:209-329`
- **Problem**: ~120 lines of dead code (old launch file version). No functional impact but degrades readability.
- **Fix**: Delete lines 209-329.

### 14. ubot_description/package.xml missing rviz2 exec_depend
- **Where**: `ubot_description/package.xml` — depends section
- **Details**: `display.launch.py` launches `rviz2` and `joint_state_publisher_gui`
- **Problem**: These are not declared as dependencies. On clean install, they won't be available.
- **Fix**: Add to `package.xml`:
  ```xml
  <exec_depend>rviz2</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  ```

---

## LOW — Style and minor inconsistencies

### 15. Placeholder descriptions in package.xml
- `ubot_bringup/package.xml`: `<description>TODO: Package description</description>`
- `ubot_description/package.xml`: `<description>TODO: Package description</description>`
- **Fix**: Replace with meaningful descriptions.

### 16. All custom ubot packages at version 0.0.0
- `ubot_bringup`, `ubot_control`, `ubot_description`
- **Fix**: Consider bumping to at least `0.1.0` for release readiness.

### 17. Hard-coded /dev/ttyUSB0 paths (addressed by issue #1)
- `ubot_ros2_control.xacro:43`, `real_robot.launch.py:75`
- **Fix**: Resolve via udev rules as described in issue #1.

---

## Priority Order for Fixes

**Fix immediately:**
1. Issue #1 — Serial port conflict (blocks real robot startup)
2. Issue #2 — Nav2 topic mismatch (blocks real robot navigation)

**Fix before deployment:**
3. Issue #3 — Missing IMU launch
4. Issue #4 — Missing sensor_msgs dependency
5. Issue #5 — Missing ubot_bringup dependencies
6. Issue #6 — Velocity limits not activated

**Fix for quality:**
7-14. Medium-severity issues (style, dead code, missing depends)
15-17. Low-severity cleanup

---

## Summary Table

| # | Severity | File(s) | Type | Fix |
|---|----------|---------|------|-----|
| 1 | CRITICAL | real_robot.launch.py, ubot_ros2_control.xacro | Config | Udev rules + use `/dev/rplidar` |
| 2 | CRITICAL | nav2_params.yaml | Config | Change `/lidar` → `/scan` |
| 3 | HIGH | real_robot.launch.py | Launch | Add BNO055 node |
| 4 | HIGH | bno055/package.xml | Manifest | Add sensor_msgs |
| 5 | HIGH | ubot_bringup/package.xml | Manifest | Add exec_depends |
| 6 | HIGH | ubot_controllers.yaml | Config | Add has_velocity_limits |
| 7 | MEDIUM | sim.launch.py | Launch | Remove dead /odom bridge |
| 8 | MEDIUM | sllidar_a2m12_launch .py | Filename | Rename (remove space) |
| 9 | MEDIUM | laptop_slam_nav2.launch.py | Launch | Uncomment nav2 or rename |
| 10 | MEDIUM | test_wheels.launch.py | Launch | Implement or delete |
| 11 | MEDIUM | ubot_controllers.yaml | Config | Remove position_feedback? |
| 12 | MEDIUM | gazebo_control.xacro | File | Delete |
| 13 | MEDIUM | sim.launch.py | Launch | Delete dead code |
| 14 | MEDIUM | ubot_description/package.xml | Manifest | Add rviz2 depend |
| 15 | LOW | various package.xml | Manifest | Replace TODO text |
| 16 | LOW | various package.xml | Manifest | Update versions |
| 17 | LOW | various | Config | Document ports |

---

**Generated by workspace audit.**
