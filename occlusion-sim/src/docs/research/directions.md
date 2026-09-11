# Research Directions

Ten open research directions for the ubot platform. Each direction is grounded in a specific capability gap identified from source code analysis and leads to a concrete research contribution.

---

## D1 — IMU-wheel odometry fusion with calibrated noise models

**Contribution claim**: empirically calibrated sensor noise models for the BNO055 + 3956-CPR encoder pair in the ubot configuration, and a validated EKF fusion pipeline delivering localisation quality suitable for sustained autonomous navigation.

**Why it's open**: the ubot's `real_ekf.yaml` and `bno055_params.yaml` are fully written but the system has never run with fusion active (`real_robot.launch.py` has both nodes commented out). The covariance values in `ubot_controllers.yaml` (`pose_covariance_diagonal: [0.001, 0.001, 1e6, 1e6, 1e6, 0.01]`) are engineering estimates, not measured from ground truth. The BNO055 mount position (`P2` per bno055_params.yaml) affects axis mapping, and IMU calibration offsets require physical calibration before fusion converges reliably.

**Codebase connection**: `real_ekf.yaml` (frequency=30Hz, two_d_mode=true, odom0=`/diff_drive_controller/odom` fusing x/yaw, imu0=`/bno055/imu` fusing only angular velocity yaw); `bno055/sensor/SensorService.py` `get_calib_status()` (sys/gyro/accel/mag calibration levels); `diff_controller.h` (30Hz PID provides velocity ground truth for noise characterisation).

**Technical approach**: (1) Record simultaneous runs with a ground-truth reference (e.g. checkerboard motion capture or a high-quality total-station). (2) Fit a stationary noise model for the BNO055 gyroscope (Allan variance analysis). (3) Characterise encoder noise as a function of surface, speed, and heading. (4) Tune EKF covariance matrices to minimise RMSE vs ground truth. (5) Enable fusion in `real_robot.launch.py` following the TF conflict resolution procedure in [Known Issues M1](../architecture/issues.md).

**ROS2 implementation sketch**:
```bash
# 1. Enable BNO055 and EKF (fix TF conflict first):
#    - Set enable_odom_tf: false in ubot_controllers.yaml
#    - Uncomment bno055_node and ekf_node in real_robot.launch.py
# 2. Collect noise data:
ros2 bag record /bno055/imu /bno055/calib_status /diff_drive_controller/odom -o imu_noise_bag
# 3. Analyse (Python, with rosbags + numpy):
#    Read bag, compute Allan variance on gyro z-axis
# 4. Tune covariances in real_ekf.yaml, iterate
# 5. Call calibration service once per session:
ros2 service call /bno055/calibration_request std_srvs/srv/Trigger
```

**Evaluation protocol**: Run the robot 10 times along a 3-metre straight path. Measure end-pose error (x, y, yaw) for wheel-only vs fused localisation. Report mean ± std. Also evaluate heading error accumulation across a 360° spin.

**Expected contribution**: validated noise parameters for BNO055 + quadrature encoder pair; enabled EKF configuration ready for Nav2 integration; measurement of localisation improvement from fusion.

**Target venue**: IEEE ICRA workshop, ROS-Industrial conference, or as a dataset contribution.

**Estimated effort**: 2-4 weeks (1 week physical experiments, 1-2 weeks calibration and tuning, 1 week evaluation + writeup).

---

## D2 — Empirical PID gain optimisation for the BTS7960 dead-zone correction

**Contribution claim**: a systematic methodology for tuning ESP32 RPM-PID gains for brushed DC motors with dead-zone feedforward correction, producing validated gains for the ubot BTS7960 configuration.

**Why it's open**: `diff_controller.h` line 44 explicitly documents that current gains (Kp=5, Ki=9, Kd=0) are _"starting gains — re-tune after flashing (previous values were calibrated with a double min_pwm bug)"_. The fixed dead-zone offset (`sign(output) × minPwm + output`, minPwm=50) changes the effective plant gain. No step-response dataset, Bode plot, or closed-loop performance metrics exist in the workspace. The Jimeno low-pass filter (α=0.854, β=0.0728) introduces phase lag that interacts with the PID derivative term.

**Codebase connection**: `diff_controller.h` (full PID + Jimeno filter source); `/horizon/diag/pid/*` topics from `diag_publisher.py` (provides left/right error, integral, output at 10Hz over ROS2); Serial2 stream at GPIO16 (14-field frame at 30Hz for offline analysis); `ubot_controllers.yaml` (PlotJuggler signal list embedded as comments).

**Technical approach**: (1) Record step-response data (`'m <target> <target>'` command, record Serial2 stream). (2) Identify open-loop transfer function from PWM to RPM (motor + H-bridge dead zone). (3) Design PID gains using Ziegler-Nichols or ITAE criteria on the identified model, accounting for the Jimeno filter's phase. (4) Validate closed-loop step response and steady-state error. (5) Update `leftKp`, `leftKi`, `leftKd` in `diff_controller.h` and reflash.

**ROS2 implementation sketch**:
```bash
# Collect step responses via diagnostic node:
ros2 run ubot_debugger diag_publisher --ros-args \
  -p publish_rate:=15.0 -p serial_port:=/dev/ttyUSB0
ros2 bag record /horizon/diag/rpm/left_actual /horizon/diag/rpm/left_target \
  /horizon/diag/pid/left_error /horizon/diag/pid/left_output -o pid_tuning_bag
# Send step command via ros2_control (or directly via 'm' serial command)
```

**Evaluation protocol**: measure rise time, overshoot, settling time, and steady-state RPM tracking error at three target speeds (low/mid/max). Compare left vs right wheel symmetry. Test at multiple surface types (wood floor, carpet).

**Expected contribution**: tuned gain set for BTS7960 + ESP32 30Hz PID with dead-zone correction; characterisation of Jimeno filter effect on control bandwidth.

**Target venue**: IEEE IROS poster, university robotics day.

**Estimated effort**: 1-2 weeks.

---

## D3 — Visual odometry integration as wheel-slip detector

**Contribution claim**: a fusion architecture combining wheel encoder odometry, IMU, and RGB-D visual odometry (from the ubot's on-board camera) to detect and compensate for wheel slip events.

**Why it's open**: the RGB-D camera is fully described in the URDF and TF tree (`camera_link` at [0.205, 0, 0.163] from base_link, camera_depth_frame, camera_optical_frame) and present in Gazebo, but has no driver or processing node in the real-robot bringup (Gap G2). No visual odometry pipeline exists. Wheel slip is undetectable from encoders alone (Gap G4).

**Codebase connection**: `ubot_camera.urdf.xacro` (TF frame definitions for camera_link tree); `ubot_gazebo.urdf.xacro` (rgbd_camera plugin at 640×480, confirming physical camera model compatibility); `real_ekf.yaml` (EKF already has `odom0` input wired; a second odometry source `odom1` can be added for visual odometry without changing the EKF architecture).

**Technical approach**: (1) Add RGB-D camera driver to `real_robot.launch.py` (e.g. `ros2_usb_cam` or `realsense2_camera` depending on hardware model). (2) Run a visual odometry node (e.g. RTAB-Map odometry, or `depth_image_proc` + feature matching) publishing `nav_msgs/Odometry` on `/camera/odom`. (3) Extend `real_ekf.yaml` to fuse `odom1=/camera/odom`. (4) Implement a slip-detection monitor: when wheel odometry and visual odometry diverge by more than a threshold, flag a slip event and temporarily down-weight wheel odometry covariance.

**ROS2 implementation sketch**:
```yaml
# Addition to real_ekf.yaml:
odom1: /camera/odom
odom1_config: [true, true, false, false, false, false, true, false, false, false, false, true]
odom1_queue_size: 5
odom1_differential: false
```

**Evaluation protocol**: deliberately drive over low-traction surface (foam mat). Compare trajectory reconstruction with (a) wheels only, (b) wheels + IMU, (c) wheels + IMU + visual odometry. Quantify slip events detected.

**Expected contribution**: slip-detection middleware for differential-drive robots; validated three-source EKF configuration; dataset of slip events with ground-truth comparison.

**Target venue**: IEEE RA-L or ICRA.

**Estimated effort**: 4-8 weeks (hardware integration week 1, algorithm integration weeks 2-4, experiments weeks 5-6, writing weeks 7-8).

---

## D4 — Evaluation of MPPI vs DWB for a low-speed indoor differential-drive robot

**Contribution claim**: a comparative evaluation of Model Predictive Path Integral control (nav2_mppi_controller) and the Dynamic Window Approach (dwb_core::DWBLocalPlanner) for the ubot's operating envelope: indoor environments, max_vel_x=0.15 m/s, tight corridors, 2D scan-only perception.

**Why it's open**: the current Nav2 configuration uses DWB (`FollowPath plugin: "dwb_core::DWBLocalPlanner"`) with a manually-tuned critic set. DWB's trajectory rollout is scored by a critic ensemble; MPPI samples trajectories from a learned/configured cost function and can handle non-Gaussian cost landscapes. At low speeds and in narrow corridors, the two controllers may perform differently. The ubot's physical configuration (max 0.15 m/s, 0.264 m track, 0.33 m wheel radius) and DWB parameter choices (`sim_time: 1.7s`, 20×20 velocity samples) are fully specified in `nav2_params.yaml`, making a fair comparison reproducible.

**Codebase connection**: `nav2_params.yaml` (DWB FollowPath config: all critic scales, vx_samples, vtheta_samples, sim_time, goal tolerances); `sim.launch.py` + `sim_nav2_params.yaml` (Gazebo environment where controller swap can be validated before physical testing); `mapper_params_online_async.yaml` (studio_3 map can serve as repeatable benchmark environment).

**Technical approach**: (1) Create a Nav2 config variant with `nav2_mppi_controller::MPPIController` in place of DWBLocalPlanner. (2) Define a benchmark: 5 goal positions in the studio_3 map, each run 10 times. (3) Metrics: path length, time-to-goal, minimum obstacle clearance, number of recovery behaviours triggered. (4) Run both controllers in Gazebo first, then on real robot.

**ROS2 implementation sketch**:
```yaml
# Alternate FollowPath plugin in nav2_params_mppi.yaml:
FollowPath:
  plugin: "nav2_mppi_controller::MPPIController"
  time_steps: 56
  model_dt: 0.05
  batch_size: 2000
  vx_max: 0.15
  vx_min: -0.05
  vy_max: 0.0
  wz_max: 0.6
  iteration_count: 1
```

**Evaluation protocol**: 10 trials per controller per goal point (studio_3 environment). Record `/diff_drive_controller/odom`, `/plan`, `/local_costmap/costmap_raw`, Nav2 action result. Report success rate, path length ratio (actual vs planned), goal time.

**Expected contribution**: controller comparison data for small low-speed indoor robots; recommended MPPI parameters for this class of robot; open benchmark protocol for other ubot-class platforms.

**Target venue**: IEEE IROS, ROS-Con proceedings.

**Estimated effort**: 3-5 weeks.

---

## D5 — Adaptive velocity limits from real-time battery state estimation

**Contribution claim**: a battery-aware velocity limit scheduler that reduces `max_velocity` in Nav2's `velocity_smoother` as battery voltage drops, preventing the ESP32 from receiving commands that produce dead-zone stalling at low voltages.

**Why it's open**: the BTS7960 H-bridge provides no current or voltage telemetry back to the ESP32 firmware in the current implementation. `diag_publisher.py` publishes 14 diagnostic fields but none include battery voltage. At low battery, the effective dead-zone threshold rises, making `minPwm=50` insufficient to overcome static friction at commanded speeds — the motor stalls while the PID integral accumulates. Nav2's `velocity_smoother` is configured with fixed limits (`max_velocity: [0.15, 0.0, 0.6]`).

**Codebase connection**: `diff_controller.h` (`updatePID()` and `emitDiagnostics()` — adding a battery ADC read here would be the lowest-latency instrumentation point); `diag_publisher.py` (adding a 15th field to the `'q'` response and a 15th publisher would extend the diagnostics pipeline without breaking the existing 14-field contract); `nav2_params.yaml` (`velocity_smoother` with `odom_topic: "/diff_drive_controller/odom"` and `speed_limit_topic: "speed_limit"` — the controller_server is already configured to subscribe to this topic for speed limiting).

**Technical approach**: (1) Add an ADC read on an ESP32 GPIO pin connected to a voltage divider across the battery. (2) Include battery voltage in the `'q'` diagnostic response as field 15. (3) Extend `diag_publisher.py` to parse and publish field 15. (4) Implement a `battery_speed_limiter` ROS2 node subscribing to `/horizon/diag/battery_voltage` and publishing `std_msgs/Float32` on `speed_limit` topic based on a piecewise-linear voltage-to-speed-fraction map.

**ROS2 implementation sketch**:
```python
# Pseudocode for battery_speed_limiter node:
if voltage > 12.0:  speed_fraction = 1.0
elif voltage > 11.0: speed_fraction = 0.75
elif voltage > 10.5: speed_fraction = 0.5
else:               speed_fraction = 0.25  # and emit warning
pub.publish(Float32(data=speed_fraction))
# Nav2 velocity_smoother + controller_server consume speed_limit topic directly
```

**Evaluation protocol**: discharge battery through full range while recording `speed_limit`, `/diff_drive_controller/odom` twist, and `/horizon/diag/pid/left_output`. Confirm that stall events (encoder velocity near zero despite nonzero velocity command) are reduced by adaptive speed limiting.

**Expected contribution**: low-cost battery-aware velocity scheduling for brushed-DC differential drives; characterisation of BTS7960 voltage-to-stall relationship.

**Target venue**: IEEE RO-MAN or ICRA workshop on reliable autonomy.

**Estimated effort**: 2-3 weeks (hardware instrumentation, firmware extension, ROS2 node, experiments).

---

## D6 — Systematic SLAM Toolbox parameter study for small indoor environments

**Contribution claim**: an empirical parameter sensitivity study for SLAM Toolbox online-async mode on the ubot's LD19 LiDAR configuration (8 Hz, 12 m range, 720-sample scans), identifying the combinations of `minimum_travel_distance`, `minimum_travel_heading`, `loop_search_maximum_distance`, and `Ceres` solver parameters that produce the most consistent maps in small indoor spaces (studio environments, <100 m²).

**Why it's open**: `mapper_params_online_async.yaml` uses default-adjacent parameters (map resolution=0.05m, transform_publish_period=0.02s, loop closure enabled, map file path hardcoded to `/home/chibueze/uni-bot/studio_3_serial`). Five real maps exist in the workspace but no systematic parameter variation has been recorded. The ubot's 8 Hz scan rate and max 12m range are at the lower end of the LD19's capability, potentially leaving loop closure performance on the table.

**Codebase connection**: `mapper_params_online_async.yaml` (full SLAM Toolbox config, solver settings, frame IDs, map resolution); saved maps in `/home/chibueze/uni-bot/` (studio_1/2/3, Tee_map, defence_map — ground truth for visual map quality comparison); frames_*.gv TF captures (provide timing evidence for transform_publish_period effect).

**Technical approach**: (1) Define a repeatable mapping trajectory in studio_3 (e.g. three loops around the perimeter). (2) Run SLAM with a grid of parameter combinations (3×3×3 = 27 runs minimum). (3) Evaluate map quality: global consistency (start/end-point error in pose graph), loop closure success rate, wall linearity (extract wall segments and measure straightness). (4) Compare to the five existing saved maps.

**Evaluation protocol**: quantitative map quality scores (ATE — Absolute Trajectory Error, if ground truth is available; otherwise relative endpoint error on closed loops). Report as a table of parameter → quality score.

**Expected contribution**: parameter recommendations for LD19 + SLAM Toolbox in small indoor spaces; reusable mapping trajectory protocol; reference maps for the documented environments.

**Target venue**: IEEE IROS mapping workshop, or as an extended technical report.

**Estimated effort**: 3-4 weeks (protocol design, data collection, analysis, writeup).

---

## D7 — Quantitative localisation benchmarking with and without loop closure

**Contribution claim**: a repeatable localisation benchmark using the ubot's real saved maps, characterising absolute trajectory error with SLAM Toolbox loop closure enabled vs disabled in the studio_3 environment.

**Why it's open**: no localisation benchmark data exists in the workspace. `mapper_params_online_async.yaml` has loop closure enabled (`do_loop_closing: true` is the default for online-async mode), but the impact has not been measured for this environment/sensor combination. The five saved maps exist as qualitative evidence of successful operation but provide no quantitative pose-error data.

**Codebase connection**: `mapper_params_online_async.yaml` (Ceres solver config, map resolution, loop closure settings — the toggle is `enable_interactive_mode` and `scan_matcher_ceres` solver params); saved maps + posegraph files (`studio_3_serial.data`, `studio_3_serial.posegraph`) — posegraph files contain the raw factor graph from which trajectory ATE can be computed offline.

**Technical approach**: (1) Define a benchmark trajectory in studio_3 (e.g. one rectangle). (2) Record `ros2 bag` of `/tf`, `/scan`, `/diff_drive_controller/odom` for 5 traversals. (3) Run SLAM Toolbox in localization mode (`localization_mode: true`) with loop closure enabled vs disabled. (4) Compute ATE from the posegraph vs reference trajectory. (5) If external reference is unavailable, use start/end-point closure error as a proxy.

**Evaluation protocol**: ATE (m) and rotational error (°) at the end of each traversal, mean ± std over 5 runs. Separate analysis for straight corridors vs open areas where loop closure is more valuable.

**Expected contribution**: quantitative localisation quality data for LD19 + SLAM Toolbox in a real indoor environment; evidence for whether loop closure provides measurable benefit in sub-100 m² spaces.

**Target venue**: IEEE ICRA workshop on mobile robot navigation benchmarks.

**Estimated effort**: 2-3 weeks.

---

## D8 — Dynamic obstacle prediction for pedestrian environments using velocity estimation

**Contribution claim**: a lightweight moving-obstacle velocity estimator integrated as a Nav2 costmap layer, using consecutive LiDAR scans to estimate obstacle velocity and project a time-expanded footprint for cost inflation, improving navigation safety in pedestrian environments.

**Why it's open**: Nav2's current `collision_monitor` uses a fixed `time_before_collision: 1.2s` horizon with no object velocity estimation (Gap G8). The `BaseObstacle` DWB critic scores trajectories against a static costmap snapshot. LiDAR scan differentiation to estimate obstacle velocity is well-studied but not implemented in the Nav2 default costmap plugin set.

**Codebase connection**: `nav2_params.yaml` (collision_monitor `FootprintApproach` polygon, BaseObstacle.scale=0.5, observation_sources=[scan]→/scan); ldlidar_node publishing `/scan` at 8 Hz (sufficient temporal resolution for pedestrian velocity estimation in short-range corridors); `local_costmap` `voxel_layer` (existing costmap infrastructure into which a new obstacle_layer with velocity expansion can be added without architectural changes).

**Technical approach**: (1) Implement a `VelocityObstacleLayer` as a Nav2 costmap plugin. (2) Track point clusters across consecutive `/scan` messages (ICP or centroid tracking). (3) Estimate velocity vector per cluster. (4) Inflate each tracked cluster's cost proportionally to `velocity × time_horizon`, projecting its predicted position. (5) Register plugin in `local_costmap.plugins` list in nav2_params.yaml.

**ROS2 implementation sketch**:
```cpp
// Nav2 custom costmap plugin skeleton
class VelocityObstacleLayer : public nav2_costmap_2d::Layer {
  void updateCosts(nav2_costmap_2d::Costmap2D& master_grid, ...) override {
    // cluster LaserScan, track centroids, project forward, inflate cost
  }
};
```

**Evaluation protocol**: run robot along corridor with a person walking perpendicular to the path. Count: (a) near-misses (person within 0.3m of robot), (b) unnecessary stops (robot stops when person is >1m away). Compare static vs velocity-aware costmap. Test at pedestrian approach speeds of 0.5, 1.0, 1.5 m/s.

**Expected contribution**: open-source Nav2 costmap plugin for velocity-aware obstacle inflation; evaluation protocol for pedestrian avoidance with a small indoor robot.

**Target venue**: IEEE RA-L with IROS presentation option.

**Estimated effort**: 6-10 weeks (plugin implementation 3-4 weeks, evaluation 2-3 weeks, writing 2-3 weeks).

---

## D9 — Characterising encoder-based odometry repeatability across surfaces and speeds

**Contribution claim**: a rigorous characterisation of wheel odometry repeatability for the ubot's 3956-CPR encoder configuration across five surface types and three speed levels, establishing the conditions under which wheel-only localisation is acceptable vs when sensor fusion is necessary.

**Why it's open**: the encoder CPR (3956) is high for a robot of this class, suggesting good theoretical resolution. However, no experimental data on actual odometry repeatability across surfaces, speeds, or distances exists in the workspace. The ubot_controllers.yaml covariance values (`pose_covariance_diagonal: [0.001, 0.001, 1e6, 1e6, 1e6, 0.01]`) are engineering estimates without empirical backing.

**Codebase connection**: `/diff_drive_controller/odom` (primary odometry output); `ubot_controllers.yaml` (embedded debugging checklist: drive 1m straight, spin 360° — this paper extends this into a systematic experiment); `diag_publisher.py` `/horizon/diag/enc/left` and `/horizon/diag/enc/right` (raw encoder counts for noise floor analysis).

**Technical approach**: (1) Fixed 1m straight runs × 20 trials on each of: smooth tile, wood floor, carpet, textured concrete, outdoor tarmac. (2) Fixed 360° rotations × 20 trials on the same surfaces. (3) Measure end-pose error vs start. (4) Vary speed: 0.05, 0.10, 0.15 m/s. (5) Compute mean and standard deviation of x, y, yaw error. (6) Fit a noise model and compare to the current covariance matrix.

**Evaluation protocol**: RMSE of position error over 20 trials per condition. Report speed × surface condition matrix. Propose updated covariance values for `ubot_controllers.yaml` based on measured noise.

**Expected contribution**: empirically grounded odometry noise models; surface-conditional operating recommendations; updated covariance parameters for the robot_localization EKF.

**Target venue**: IEEE IROS, or as a dataset paper with open bag recordings.

**Estimated effort**: 2-3 weeks (experiments 1-2 weeks, analysis and writing 1 week).

---

## D10 — Towards multi-session mapping: merging studio maps with SLAM Toolbox serialisation

**Contribution claim**: a multi-session mapping methodology using SLAM Toolbox's serialisation format (`.data` + `.posegraph` files), demonstrated on the five ubot environment maps, enabling incremental expansion and long-term map maintenance without full re-mapping.

**Why it's open**: the workspace contains five separately saved SLAM Toolbox maps (studio_1 through defence_map), each captured in an independent session. SLAM Toolbox supports loading a serialised map and continuing in mapping mode — the `studio_3_serial` path is hardcoded in `mapper_params_online_async.yaml` — but merging or linking maps across sessions (e.g. connecting studio_2 with the Tee_map if they share a corridor) has not been explored. No multi-session or lifelong mapping pipeline exists in the codebase.

**Codebase connection**: `mapper_params_online_async.yaml` (`map_file_name: /home/chibueze/uni-bot/studio_3_serial` — the hardcoded serialisation path used for load/save); saved posegraph files across all five environments (these encode the full pose graph with loop closure edges — the raw material for merging); `laptop_slam_nav2.launch.py` (intended to run SLAM from a laptop, potentially on a different ROS domain or network — relevant if multi-session work uses a more capable compute node).

**Technical approach**: (1) Document the SLAM Toolbox serialisation format (`.data` JSON + `.posegraph` protobuf structure) as seen in the five saved files. (2) Identify physically overlapping regions between pairs of environments (studio_2 and Tee_map may share entry areas). (3) Use SLAM Toolbox's `merge_maps` service or manual pose-graph editing to merge submaps. (4) Validate merged map quality by navigating across the merged region.

**Evaluation protocol**: navigation success rate in the merged map, measured by path planning success to goals requiring cross-submap traversal. Visual comparison of merged vs individually-saved map quality in the overlap region.

**Expected contribution**: documented multi-session mapping workflow for SLAM Toolbox with serialised maps; merged map artefacts for the five ubot environments; protocol adaptable to other ros2/slam_toolbox deployments.

**Target venue**: IEEE ICRA, or as a contribution to the SLAM Toolbox open-source project.

**Estimated effort**: 3-5 weeks.
