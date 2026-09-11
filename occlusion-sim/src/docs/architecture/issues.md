# Known Issues

This page catalogues issues found by directly reading the current workspace source. Every item is independently verified against the current state of the files cited — items from the pre-existing `WORKSPACE_AUDIT.md` were re-checked, and two of its claims were found to be already resolved. The audit document itself is now partially stale and should not be used as a reference; this page supersedes it.

Severity key: **Critical** — system will not function; **High** — significant capability loss; **Medium** — operational limitation or real risk; **Low** — cosmetic/maintenance concern; **Informational** — not a bug, just noteworthy context.

---

## MEDIUM-HIGH severity

### M1 — BNO055 IMU and EKF not launched on real robot

**File**: `src/ubot/ubot_bringup/launch/real_robot.launch.py`, lines 101–125  
**Status**: Confirmed open

The BNO055 node (`bno055_node`, lines 101-107) is defined but its addition to the `LaunchDescription` return is commented out (line 125: `# bno055_node,`). The EKF node (`ekf_filter_node`, lines 110-116) is entirely commented out.

As a result, the real robot currently runs on **wheel odometry alone**. The BNO055/EKF config is fully written (`real_ekf.yaml`, `bno055_params.yaml`) and ready to enable, but the final `LaunchDescription` excludes these nodes.

**Impact**: Nav2 operates with no IMU fusion. Heading estimation accumulates encoder drift with no correction. Long-duration runs in environments with wheel slip (carpet, ramps) will exhibit significant odometry divergence.

**Fix**:
```python
# In real_robot.launch.py, change the return to include both nodes:
return LaunchDescription([
    node_robot_state_publisher,
    controller_manager,
    joint_state_broadcaster_spawner,
    diff_drive_controller_spawner,
    twist_stamper,
    ldlidar_node,
    bno055_node,      # ← uncomment
    ekf_node,         # ← uncomment
])
```

**Additional required step**: when EKF is enabled, `diff_drive_controller` must stop broadcasting its own `odom→base_footprint` TF to avoid a conflict. Set `enable_odom_tf: false` in `ubot_controllers.yaml` and let the EKF's `publish_tf: true` (in `real_ekf.yaml`) be the sole broadcaster for that edge. Also update Nav2's `odom_topic` in `nav2_params.yaml` from `/diff_drive_controller/odom` to `/odometry/filtered`.

---

## MEDIUM severity

### M2 — Velocity/acceleration limits may be silently ignored

**File**: `src/ubot/ubot_bringup/config/ubot_controllers.yaml`, lines 64-71  
**Status**: Confirmed present — runtime behaviour depends on installed ros2_control version

The YAML sets `linear.x.max_velocity: 0.5`, `linear.x.max_acceleration: 1.0`, `angular.z.max_velocity: 2.0`, etc., but the boolean flags `has_velocity_limits: true` and `has_acceleration_limits: true` are never set anywhere in the file.

In some versions of `diff_drive_controller`, the numeric limit values are only enforced when the corresponding `has_*_limits` flag is explicitly `true`. If the flag is absent (defaults to `false` in those versions), the limits are silently ignored and the controller imposes no software cap on wheel speeds.

**Impact**: the robot may receive velocity commands beyond what the motors can follow, causing encoder slip and odometry error.

**Fix**:
```yaml
# In diff_drive_controller section of ubot_controllers.yaml, add:
linear.x.has_velocity_limits:      true
linear.x.has_acceleration_limits:  true
angular.z.has_velocity_limits:     true
angular.z.has_acceleration_limits: true
```

> ⚠️ Verify against your installed `ros2_controllers` version — the flag semantics changed between ros2_control 3.x (Humble) and later releases. Run `ros2 pkg show ros2_controllers` to check.

### M3 — Two incompatible ESP32 firmwares coexist in the workspace

**Files**: `src/Ros-esp32_bridge/Ros-esp32_bridge.ino` (LIVE) and `src/esp32/hacker/hacker.ino` (ORPHANED)  
**Status**: Architectural concern

`hacker.ino` is a complete, self-contained micro-ROS firmware that subscribes directly to `/cmd_vel` and publishes `/horizon/*` velocity/encoder topics. It completely bypasses `ros2_control`. If flashed to the same ESP32 that `UbotHardware` expects to talk to via text-command serial, the system would have two competing control paths for the same motors, and the serial protocol would mismatch entirely.

Additionally, `hacker.ino` itself contains a doc/code mismatch: its header comment (lines 34-35) states `RIGHT_RPWM→2, LPWM→15` but the actual `#define`s are `RIGHT_RPWM=32, RIGHT_LPWM=33`.

**Impact**: low immediate risk (the orphaned file is not a Makefile target or build output), but high confusion risk for anyone new to the codebase who might flash the wrong firmware.

**Fix**: add a prominent `DO_NOT_FLASH.md` or `README.md` to `esp32/hacker/` explaining the conflict. Consider moving the file out of `src/` or deleting it if it provides no ongoing research value.

### M4 — `test_wheels.launch.py` is empty — will fail if launched

**File**: `src/ubot/ubot_bringup/launch/test_wheels.launch.py`  
**Status**: Confirmed open

The file contains no content (0 effective bytes / 1 line). Attempting to run `ros2 launch ubot_bringup test_wheels.launch.py` will fail with a Python error since there is no `generate_launch_description()` function.

**Fix**: either implement the intended wheel-test launch, or delete the file and remove it from CMakeLists.txt install targets.

---

## LOW severity

### L1 — `gazebo_control.xacro` is a dead/unreferenced file

**File**: `src/ubot/ubot_description/urdf/gazebo/gazebo_control.xacro`  
**Status**: Confirmed open

This file defines a standalone Gazebo classic `DiffDrive` plugin. It is not included by any xacro in the current include tree (confirmed by tracing `ubot_robot.urdf.xacro` through all its includes). The Gazebo integration is now handled by `gz_ros2_control/GazeboSimSystem` in `ubot_ros2_control.xacro`.

**Impact**: none at runtime; maintenance confusion only.

### L2 — `diag_publisher.py` not wired into any bringup launch file

**File**: `src/ubot/ubot_debugger/ubot_debugger/diag_publisher.py`  
**Status**: Confirmed open (newly identified — not in old audit)

The `horizon_diag_publisher` node is a useful diagnostic tool that polls the ESP32 `'q'` command and republishes 14 signals for PlotJuggler. However, it has no `Node()` block in `real_robot.launch.py` or any other launch file. It must be run manually:

```bash
ros2 run ubot_debugger diag_publisher
```

**Impact**: users unfamiliar with the workspace may not discover this tool. On first serial open failure, the node exits immediately (`SystemExit(1)`) with no retry — see [Troubleshooting](../guides/troubleshooting.md) if the node exits on startup.

### L3 — `laptop_slam_nav2.launch.py` filename implies Nav2 runs, but Nav2 is commented out

**File**: `src/ubot/ubot_bringup/launch/laptop_slam_nav2.launch.py`  
**Status**: Confirmed open

The launch file name and its `Include` blocks suggest it should bring up both SLAM and Nav2 from the laptop side. The Nav2 inclusion is commented out in the current file, meaning only SLAM (or SLAM's dependencies) would actually launch.

**Impact**: potentially misleading to users expecting navigation capability when using this file.

### L4 — `hacker.ino` header comment contradicts pin #defines

**File**: `src/esp32/hacker/hacker.ino`, lines 34-35 vs line ~40  
**Status**: Confirmed (doc bug in an orphaned file)

Header comment states: `BTS7960 Right: RPWM→2 LPWM→15 R_EN→4 L_EN→17`.  
Actual `#define`s: `RIGHT_RPWM=32`, `RIGHT_LPWM=33`, `RIGHT_R_EN=4`, `RIGHT_L_EN=17`.

RPWM and LPWM values are wrong in the comment (2/15 vs 32/33). Since this file is not the live firmware, there is no immediate hardware risk.

### L5 — PID gains in live firmware marked as "untuned starting gains"

**File**: `src/Ros-esp32_bridge/diff_controller.h`, line 44  
**Status**: Self-documented known limitation

The comment on line 44 reads: _"Starting gains — re-tune after flashing (previous values were calibrated with a double min_pwm bug)"_. This explicitly states the current gains (Kp=5, Ki=9, Kd=0) are starting values, not validated tuned values — the previous calibration was invalidated by a now-fixed dead-zone bug.

**Impact**: the robot may exhibit suboptimal speed tracking. See [Calibration Guide](../guides/calibration.md) for the re-tuning procedure.

### L6 — `ubot_debugger` package.xml license field is literally "TODO"

**File**: `src/ubot/ubot_debugger/package.xml`  
**Status**: Confirmed open (cosmetic)

The `<license>` tag contains the string `"TODO"` rather than an actual SPDX license identifier.

### L7 — `bno055` axis-remap dict lookup is unguarded

**File**: `src/bno055/bno055/sensor/SensorService.py`, `configure()` method  
**Status**: Confirmed open

The `mount_positions` dict maps keys `'P0'`..'`P7'` to axis remap register values. The dict lookup `mount_positions[placement_axis_remap_param]` is unguarded — if `placement_axis_remap` is set to an invalid value in the YAML (e.g. `'P9'` or a typo), the node will crash with an unhandled `KeyError` rather than logging a useful error.

**Fix**: add a `try/except KeyError` around the lookup and raise/log a meaningful message.

### L8 — BNO055 quaternion normalization is hand-rolled

**File**: `src/bno055/bno055/sensor/SensorService.py`, ~line 200  
**Status**: TODO in source code

Comment in source: `"TODO(flynneva): replace with standard normalize() function"`. The code manually normalizes the quaternion from the BNO055 rather than using a library function. The hand-rolled implementation is likely mathematically correct but is a code-smell.

### L9 — BNO055 orientation_covariance field reused as filtered covariance

**File**: `src/bno055/bno055/sensor/SensorService.py`, lines ~157/189  
**Status**: TODO in source code (two identical comments)

The `imu_raw` message reuses the `orientation_covariance` field in a non-standard way. TODO comments at lines ~157 and ~189 note this and suggest making it configurable.

### L10 — `docking_server` configured with no physical docking hardware in workspace

**File**: `src/ubot/ubot_bringup/config/nav2_params.yaml`, lines 328-354  
**Status**: Informational / aspirational config

`docking_server` is configured with `simple_charging_dock` plugin (`opennav_docking::SimpleChargingDock`, `use_battery_status: false`, `use_stall_detection: false`). No docking station hardware, IR beacon, or charger hardware driver appears anywhere else in the workspace. The config appears aspirational or pre-ported from a Nav2 defaults template.

**Impact**: none at runtime (docking actions simply won't be used). Remove from nav2_params.yaml or implement the hardware if autonomous charging is intended.

---

## INFORMATIONAL

### I1 — Wheel TF rate at ~15.26 Hz despite 30 Hz controller configuration

See [TF Tree — Observed Discrepancies](tf_tree.md#2-wheel-tf-rate-1526-hz-is-half-the-configured-controller-rate-30-hz).

> ⚠️ Could not be determined from source — requires runtime inspection. Use `ros2 topic hz /joint_states` to confirm actual rate.

### I2 — No `map` frame in most recent TF capture

See [TF Tree](tf_tree.md#1-no-map-frame-at-capture-time). SLAM was not running at capture time — not a persistent bug.

### I3 — `WORKSPACE_AUDIT.md` is partially stale

**File**: `src/WORKSPACE_AUDIT.md`  
The pre-existing 17-item audit doc has at least two resolved items presented as open:
- Its Issue #1 (RPLiDAR/ESP32 port conflict on `/dev/ttyUSB0`) — **resolved**: `real_robot.launch.py` now uses ldlidar on `/dev/ttyUSB1`.
- Its Issue #2 (Nav2 costmaps subscribing to `/lidar`) — **resolved**: `nav2_params.yaml` uses `/scan` throughout.

Use this page (generated from current source) rather than `WORKSPACE_AUDIT.md` as the authoritative issue list.

### I4 — `position_feedback` param potentially deprecated

**File**: `src/ubot/ubot_bringup/config/ubot_controllers.yaml`, line 86  
The `position_feedback: true` parameter name may be deprecated in newer versions of `diff_drive_controller`. The parameter enables encoder-based position feedback rather than velocity-based odometry integration, which is correct for this hardware. If you see warnings about unknown parameters on launch, verify against your installed ros2_controllers version.

> ⚠️ Could not be determined from source — requires runtime inspection.

### I5 — `sim.launch.py` contains ~120 lines of commented-out dead code

**File**: `src/ubot/ubot_bringup/launch/sim.launch.py`  
Large sections of commented-out code (prior architectures, duplicate node definitions) reduce readability. Not functionally harmful. Recommend a cleanup pass.
