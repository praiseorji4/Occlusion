# Research Gaps

This page identifies ten capability gaps in the current ubot platform. Each gap is grounded in specific evidence from the source code and configuration files.

---

## G1 — Sensor fusion not deployed: IMU + EKF commented out

**Evidence**: `src/ubot/ubot_bringup/launch/real_robot.launch.py`, lines 101-116 and 125. The BNO055 node and `ekf_filter_node` are defined but excluded from the returned `LaunchDescription`. `real_ekf.yaml` configures a 30 Hz two-dimensional EKF fusing `/diff_drive_controller/odom` (x, y, vx, vyaw) and `/bno055/imu` (angular velocity z only).

**Technical gap**: the robot currently localises using wheel odometry alone. Quadrature encoders provide position feedback at 30 Hz but accumulate heading error that grows with distance. The BNO055 NDOF mode provides a fused yaw estimate from gyroscope + accelerometer + magnetometer — the gyroscope component (short-term accurate) would correct high-frequency heading drift while the wheel odometry (low-frequency drift) corrects IMU bias. Without fusion, long straight runs and tight rotations are susceptible to significant pose error, especially on low-traction surfaces.

**Quantitative gap**: unaided wheel-odometry heading error is on the order of 1-5° per metre of travel in typical differential-drive systems; fused odometry with a good IMU can reduce this to <0.5°/m in similar conditions. No measurement of the current ubot's drift rate exists in the workspace.

**Prerequisite work**: re-enabling the BNO055 + EKF requires resolving the TF conflict (both `diff_drive_controller` and `ekf_filter_node` attempt to broadcast `odom→base_footprint` — must set `enable_odom_tf: false` in `ubot_controllers.yaml` first). See [Known Issues M1](../architecture/issues.md#m1--bno055-imu-and-ekf-not-launched-on-real-robot).

---

## G2 — Camera integration absent on real robot

**Evidence**: `src/ubot/ubot_description/urdf/sensors/ubot_camera.urdf.xacro` defines `camera_link`, `camera_depth_frame`, `camera_optical_frame`, and `camera_rgb_frame`. The Gazebo description (`ubot_gazebo.urdf.xacro`) includes an `rgbd_camera` plugin publishing 640×480 images. However, no ROS2 camera driver node (e.g. `realsense2_camera`, `ros2_usb_cam`) appears in any launch file or package manifest for real-robot operation.

**Technical gap**: the physical RGB-D camera is mechanically mounted (at [0.205, 0, 0.163] m from base_link) and is part of the TF tree (static transform published by robot_state_publisher), but produces no data on the real robot. This means:
- No visual obstacle detection to complement the 2D LiDAR scan.
- No depth-based costmap layer (Nav2's `depth_layer` or point-cloud obstacles).
- No potential for visual odometry (e.g. RTAB-Map, ORB-SLAM3) to augment or replace wheel odometry.
- No RGB image for semantic tasks (object detection, person following).

**Prerequisite work**: identify the physical camera model, install its ROS2 driver, add a node to `real_robot.launch.py`, and decide whether to add a `PointCloudLayer` to Nav2 costmaps.

---

## G3 — Ultrasonic sensors in URDF have no driver or ROS2 topic

**Evidence**: `src/ubot/ubot_description/urdf/sensors/ubot_ultrasonic.urdf.xacro` defines at least two ultrasonic sensor links. No ROS2 driver node, no subscriber, and no observation source entry in `nav2_params.yaml` references ultrasonic data.

**Technical gap**: ultrasonic sensors are typically used for close-range obstacle detection (<0.5 m) where LiDAR often has blind spots (minimum range). If ultrasonic hardware is physically present on the robot, it provides no collision avoidance benefit in the current system. The Nav2 `collision_monitor` is configured for LiDAR scan only.

**Prerequisite work**: verify whether ultrasonic hardware is physically installed. If so, implement an ESP32-side reading loop (sensors are short-range, low-baud) and add a `sensor_msgs/Range` publisher, or add distance readings to the `'q'` diagnostic frame from the ESP32.

---

## G4 — No wheel slip detection or compensation

**Evidence**: `src/Ros-esp32_bridge/diff_controller.h` implements a velocity PID based on encoder ticks. The encoder CPR (3956) is high, but there is no mechanism to detect or flag wheel slip — the controller receives encoder counts and converts them directly to velocity estimates regardless of whether traction was maintained.

**Technical gap**: for a 4WD robot with only front-wheel encoders, rear wheels are mechanically slaved but their slip is unobservable. Even on the front wheels, encoder-based velocity estimation cannot distinguish between the wheel rotating (contact) and the wheel spinning in place (slip). On surfaces where slip occurs (carpet, ramps, wet floors), odometry diverges from actual robot displacement.

**Approaches**: (a) cross-validate wheel odometry with IMU-derived acceleration/velocity (this is what the EKF in G1 partly addresses), (b) add contact force estimation (requires additional hardware), (c) use visual odometry as an independent displacement estimate.

**Specific to this codebase**: the Jimeno low-pass filter in `diff_controller.h` (`rpm_filtered = 0.854×rpm_filtered + 0.0728×raw_rpm + 0.0728×rpm_prev`) smooths RPM noise but does not detect slip events (large instantaneous delta between raw_rpm and filtered).

---

## G5 — PID gains not empirically validated after dead-zone bug fix

**Evidence**: `src/Ros-esp32_bridge/diff_controller.h`, line 44 comment: _"Starting gains — re-tune after flashing (previous values were calibrated with a double min_pwm bug)"_. Current gains: Kp=5.0, Ki=9.0, Kd=0.0.

**Technical gap**: the dead-zone feedforward in the current `updatePID()` applies `sign(output) × minPwm + output` to overcome static motor friction. A previous version apparently applied `minPwm` twice in the control path. The current gains were tuned against the bugged version and are therefore uncalibrated for the corrected system. The robot may exhibit integral windup, oscillation, or speed-tracking error that would not exist with properly tuned gains.

**Quantitative gap**: no step-response or steady-state RPM-tracking data exists in the workspace. A systematic Ziegler-Nichols or model-based gain sweep is needed.

**Supporting infrastructure present**: `ubot_controllers.yaml` includes a PlotJuggler signal list (`/ubot/diagnostics`, Serial2 `/dev/ttyUSB1` 14-field frame at 30 Hz) for real-time PID observation. The infrastructure to measure and iterate on gains exists; the tuning work has not been done.

---

## G6 — No long-duration autonomy / lifecycle management

**Evidence**: the bringup launch sequence (`real_robot.launch.py`) has no restart logic, no watchdog beyond the ESP32's 10-second motor timeout, and no node lifecycle management (nodes are launched as "unmanaged" nodes, not `rclcpp_lifecycle::LifecycleNode`). `diag_publisher.py` exits immediately on a serial open failure with `SystemExit(1)` and no recovery.

**Technical gap**: for autonomous operation beyond a few minutes, nodes must handle: (a) hardware interface disconnection and reconnection, (b) sensor brownouts (e.g. IMU I2C error), (c) controller failure modes (e.g. diff_drive_controller crashes and is not respawned). None of these are handled.

**Specific evidence**: `SensorService.configure()` calls `sys.exit(1)` on BNO055 communication failure (no retry). `DiagPublisher.__init__` calls `SystemExit(1)` on serial open failure (no retry). `UbotHardware::on_init()` — lifetime not confirmed from source without reading the full CPP, but the standard ros2_control lifecycle does not auto-restart on `ERROR_STATE`.

---

## G7 — No semantic or 3D mapping

**Evidence**: `mapper_params_online_async.yaml` configures SLAM Toolbox in 2D mode (LaserScan input, occupancy grid output, map resolution=0.05m, max_laser_range=12m). The RGB-D camera is not used for mapping.

**Technical gap**: all five saved maps (studio_1 through defence_map) are 2D occupancy grids. The system has no capability for: (a) semantic labels on map cells (rooms, doors, obstacles by class), (b) 3D volumetric mapping (needed for navigating multi-level environments or detecting table-height obstacles), (c) dynamic landmark detection for relocalization without loop closure.

**Opportunity**: the RGB-D camera URDF is already in the TF tree. Adding an RTAB-Map or point-cloud-based map layer would provide 3D context without new hardware. The BNO055 pitch/roll data (not currently used in the EKF — `real_ekf.yaml` fuses only yaw angular velocity) could provide attitude ground truth for slope detection.

---

## G8 — Dynamic obstacle handling is reactive only

**Evidence**: `nav2_params.yaml` configures `collision_monitor` with `FootprintApproach` polygon (type=polygon, action_type=approach, time_before_collision=1.2s) and `observation_sources: ["scan"]`. The `controller_server` uses the DWB local planner with `Oscillation` and `BaseObstacle` critics.

**Technical gap**: the current Nav2 configuration detects and avoids static and momentarily-present obstacles from LiDAR scan data, but has no predictive model for moving obstacles (people, other robots). The `time_before_collision` parameter provides a fixed time horizon for velocity reduction, but velocity extrapolation for moving obstacles is not configured.

**Specific gap in code**: the DWB `BaseObstacle` critic was previously configured at `scale: 0.02` (nearly ignored) and is now at `scale: 0.5`. While this is a significant improvement, DWB's trajectory evaluation for dynamic obstacles requires either (a) the `CriticFunction` interface with a dynamic costmap layer, or (b) an MPPI-based controller (Nav2 supports `nav2_mppi_controller`) that can explicitly predict obstacle motion.

---

## G9 — No quantitative navigation benchmarking

**Evidence**: the workspace contains 5 saved maps and evidence of physical navigation runs, but no benchmark dataset, no logged `/diff_drive_controller/odom` bags, no path-tracking error measurements, and no repeatability studies.

**Technical gap**: it is not possible from the workspace alone to answer: (a) what is the steady-state odometry error after a 10-metre run? (b) what is the navigation success rate to a goal 3m away? (c) how does heading error accumulate over time? The PlotJuggler signals and Serial2 diagnostic stream provide real-time visibility but there are no recorded datasets.

**Infrastructure partially present**: `ros2 bag record` is standard ROS2 infrastructure. The diagnostic topics and `/diff_drive_controller/odom` provide the signals needed. `experiments/logging.md` should define a standard logging protocol.

---

## G10 — Simulation/reality gap not characterised

**Evidence**: `sim.launch.py` + `sim_ubot_controllers.yaml` + `sim_nav2_params.yaml` provide a Gazebo simulation with `gz_ros2_control`, `gpu_lidar` (720 samples, 10 Hz), `rgbd_camera` (640×480), and IMU (100 Hz, Gaussian noise model). The sim controller uses 50 Hz update rate and includes all 4 wheels in the diff_drive_controller (unlike the real robot's front-wheel-only). Different Nav2 inflation radii (`local: 0.55` vs real `0.30`).

**Technical gap**: the simulation and real-robot configurations differ in at least 4 ways:
1. Controller update rate: 50 Hz (sim) vs 30 Hz (real).
2. Diff drive wheel configuration: 4 wheels (sim) vs 2 front wheels (real).
3. Nav2 local inflation radius: 0.55 m (sim) vs 0.30 m (real).
4. Odometry source: `/odometry/filtered` (sim, EKF active) vs `/diff_drive_controller/odom` (real, EKF inactive).

Policies tuned in simulation may not transfer directly to real hardware. The sim-to-real gap for this platform has not been formally characterised.
