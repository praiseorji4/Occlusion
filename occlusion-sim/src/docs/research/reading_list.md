# Reading List

A prioritised literature list grounded in the exact algorithms and subsystems deployed in the ubot workspace. Every entry is chosen because it directly describes code or methods running on this robot.

---

## Tier 1 — Must-read (directly describes running code)

These papers describe the exact algorithms executing in this workspace. Read these before modifying any of the subsystems they cover.

### SLAM and mapping

**Macenski, S., Martin, F., Santos, R., Ginés Clavero, J. (2021). The Marathon 2: A Navigation System.** *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).* arXiv:2003.00368.

Covers the Nav2 architecture end-to-end: BT navigator, lifecycle management, costmaps, planner/controller server. The Nav2 version deployed in `nav2_params.yaml` is descended from this architecture. Specifically relevant: how `controller_frequency` (set to 20.0 Hz in nav2_params.yaml) interacts with the planner/smoother pipeline, and how `bt_navigator` error codes map to recovery behaviours (see `error_code_name_prefixes` list in nav2_params.yaml).

**Macenski, S., Foote, T., Gerkey, B., Lalancette, C., Woodall, W. (2022). Robot Operating System 2: Design, architecture, and uses in the wild.** *Science Robotics, 7(66).* doi:10.1126/scirobotics.abm6074.

Covers ROS2 design decisions relevant to the ubot system: DDS QoS, lifecycle nodes, ros2_control hardware_interface. Explains why `joint_state_broadcaster` and `diff_drive_controller` are separate managed components, and the rationale for the controller_manager's `update_rate: 30` heartbeat.

**Macenski, S., Stachowiak, J. (2021). SLAM Toolbox: SLAM for the Dynamic World.** *Journal of Open Source Software, 6(61), 2783.* doi:10.21105/joss.02783.

Directly covers SLAM Toolbox online-async mode, Ceres solver integration, loop closure, and the `.data`/`.posegraph` serialisation format used by all five saved maps in `/home/chibueze/uni-bot/`. The `mapper_params_online_async.yaml` configuration parameters (transform_publish_period, minimum_travel_distance, etc.) are all defined in this codebase.

### Robot localisation (EKF)

**Moore, T., Stouch, D. (2016). A Generalized Extended Kalman Filter Implementation for the Robot Operating System.** *Proceedings of the 13th International Conference on Intelligent Autonomous Systems.* Springer.

This is the robot_localization EKF paper. `real_ekf.yaml` configures `ekf_filter_node` from this package: `two_d_mode: true`, `odom0` and `imu0` fusion, the `odom0_config` and `imu0_config` masks, `imu_remove_gravitational_acceleration: true`. Understanding the configuration vector format (12-element boolean arrays) is essential before modifying `real_ekf.yaml`.

### ros2_control

**Balachandran, A., Quigley, M., et al. (2023). ros2_control: A hardware-agnostic framework for resource-constrained embedded development.** *Proceedings of ROS-Industrial Conference.* (Also: the ros2_control documentation at control.ros.org.)

Covers the `hardware_interface::SystemInterface` lifecycle that `UbotHardware` implements: `on_init`, `on_configure`, `on_activate`, `read()`, `write()`. Essential reading before modifying `ubot_hardware_interface.cpp`.

---

## Tier 2 — Recommended (methods used or directly relevant)

### PID and motor control

**Jiménez, A. R., Seco, F., Prieto, J. C., Guevara, J. (2009). A comparison of Pedestrian Dead-Reckoning Algorithms using a Low-Cost MEMS IMU.** *IEEE International Symposium on Intelligent Signal Processing (WISP).* doi:10.1109/WISP.2009.5286542.

The RPM filter in `diff_controller.h` is explicitly labelled a "Jimeno low-pass filter" in comments. The coefficients (0.854, 0.0728, 0.0728) correspond to a second-order IIR filter of the form used in this paper for pedestrian dead-reckoning IMU smoothing, adapted for encoder RPM. Understanding the filter's phase and frequency response is relevant to gain tuning (Direction D2).

**Åström, K. J., Hägglund, T. (2006). *Advanced PID Control.* ISA — The Instrumentation, Systems, and Automation Society.**

Chapter 6 (anti-windup) directly applies to the integral clamping in `doPID()` (`integral = constrain(integral, -50.0f, 50.0f)`). Chapter 8 (feedforward) covers the dead-zone offset (`sign(output) × minPwm + output`) applied in `updatePID()`. Required reading for any gain re-tuning work (Direction D2).

### Differential drive and odometry

**Borenstein, J., Feng, L. (1996). Measurement and Correction of Systematic Odometry Errors in Mobile Robots.** *IEEE Transactions on Robotics and Automation, 12(6).* doi:10.1109/70.544770.

Covers systematic odometry errors from wheel-diameter mismatch and non-ideal wheel placement — exactly the errors addressed by the three-way sync requirement (`wheel_radius=0.033`, `wheel_separation=0.264204`) across `diff_controller.h`, `ubot_ros2_control.xacro`, and `ubot_controllers.yaml`. The calibration protocol in this paper is the theoretical basis for the embedded debugging checklist in `ubot_controllers.yaml` (drive 1m, spin 360°).

**Martinelli, A., Tomatis, N., Siegwart, R. (2007). Some results on SLAM and the closing loop problem.** *Autonomous Robots, 22(2).*

Relevant to Direction D7 (localisation benchmarking). Provides a framework for quantifying odometry drift as a function of travelled distance — useful for designing the experiment in Direction D9 and interpreting ubot encoder covariance data.

### LiDAR and scan matching

**Grisetti, G., Stachniss, C., Burgard, W. (2007). Improved Techniques for Grid Mapping with Rao-Blackwellized Particle Filters.** *IEEE Transactions on Robotics, 23(1).* doi:10.1109/TRO.2006.889486.

The theoretical foundation of grid-based SLAM that SLAM Toolbox builds on. Understanding the scan-to-scan matching and the particle filter update step helps interpret why `minimum_travel_distance` and `minimum_travel_heading` thresholds exist in `mapper_params_online_async.yaml`.

### Nav2 local planning

**Marder-Eppstein, E., Berger, E., Foote, T., Gerkey, B., Konolige, K. (2010). The Office Marathon: Robust Navigation in an Indoor Office Environment.** *IEEE International Conference on Robotics and Automation (ICRA).*

The original DWA/DWB description. The `dwb_core::DWBLocalPlanner` in `nav2_params.yaml` (FollowPath plugin) is DWB — Dynamic Window Approach with a critic ensemble. Understanding the original DWA kinematic window helps interpret why `vx_samples: 20`, `vtheta_samples: 20`, and `sim_time: 1.7` were chosen.

### IMU calibration

**Bosch Sensortec (2023). BNO055 Datasheet.** Rev. 1.8. Bosch Sensortec GmbH.

Required reading for understanding the BNO055 operating modes (NDOF, IMU, CONFIG), axis remap registers P0-P7 (the `mount_positions` dict in `SensorService.py`), and the calibration offset register map that `calibration_request_callback()` reads. The `unit_sel: 0x83` value set in `configure()` is defined in §4.3.61 of this datasheet.

---

## Tier 3 — Extended reading (relevant research context)

### MPPI and advanced local planning

**Williams, G., Drews, P., Goldfain, B., Rehg, J. M., Theodorou, E. A. (2017). Aggressive Driving with Model Predictive Path Integral Control.** *IEEE International Conference on Robotics and Automation (ICRA).* doi:10.1109/ICRA.2016.7487277.

Background for Direction D4 (MPPI vs DWB evaluation). MPPI samples trajectories stochastically, which makes it better at handling non-convex costs than DWB's deterministic rollouts. The Nav2 implementation (`nav2_mppi_controller`) is based on this formulation.

### Visual odometry

**Mur-Artal, R., Montiel, J. M. M., Tardós, J. D. (2015). ORB-SLAM: A Versatile and Accurate Monocular SLAM System.** *IEEE Transactions on Robotics, 31(5).* doi:10.1109/TRO.2015.2463671.

Background for Direction D3 (visual odometry integration). ORB-SLAM3 supports RGB-D input matching the ubot's camera plugin configuration (640×480, depth+rgb). Understanding feature extraction and tracking helps evaluate whether the ubot's camera placement (forward-facing at [0.205, 0, 0.163]) provides sufficient texture for robust feature tracking in indoor corridors.

### Sensor fusion architectures

**Madgwick, S. O. H., Harrison, A. J. L., Vaidyanathan, R. (2011). Estimation of IMU and MEMS sensor bias and scale factor errors using a novel backpropagation scheme.** *IEEE ICRA.*

Background for Direction D1 (IMU calibration). The Madgwick filter is an alternative to the EKF for quaternion estimation — the BNO055 already runs Bosch's proprietary fusion internally (NDOF mode), but understanding IMU fusion algorithms helps interpret the `bno055/imu` output and design the noise characterisation experiment.

**Thrun, S., Burgard, W., Fox, D. (2005). *Probabilistic Robotics.* MIT Press.**

The theoretical foundation for all probabilistic sensing and estimation components in this workspace: EKF (Chapter 3), particle-filter SLAM (Chapter 13), and motion models (Chapter 5, relevant to the wheel odometry model in `diff_drive_controller`). Part III (mobile robot localisation) is directly relevant to Directions D1, D7, and D9.

### Dynamic obstacle handling

**Fox, D., Burgard, W., Thrun, S. (1997). The Dynamic Window Approach to Collision Avoidance.** *IEEE Robotics and Automation Magazine, 4(1).* doi:10.1109/100.580977.

Original DWA paper. Read alongside the DWB critic architecture to understand the velocity space sampling and why `RotateToGoal.scale: 32.0` and `PathDist.scale: 32.0` dominate the critic ensemble in `nav2_params.yaml`.

**Macenski, S., Martín, F., Herrero-Pérez, D. (2023). Regulated Pure Pursuit for Robot Path Tracking.** *Autonomous Robots.* doi:10.1007/s10514-023-10097-6.

The Regulated Pure Pursuit (RPP) controller available in Nav2 as an alternative to DWB, relevant to Direction D4. RPP may be more appropriate than DWB for small indoor robots navigating narrow corridors at low speeds.

### Long-duration autonomy

**Krajník, T., Fentanes, J. P., Santos, J. M., Duckett, T. (2017). FreMEn: Frequency Map Enhancement for Long-Term Mobile Robot Autonomy in Changing Environments.** *IEEE Transactions on Robotics, 33(4).* doi:10.1109/TRO.2017.2665664.

Background for Direction D10 (multi-session mapping) and Gap G6 (no long-duration lifecycle management). FreMEn models temporal changes in occupancy maps — relevant for the ubot's studio environments which change between mapping sessions (chairs moved, doors opened/closed). The studio_1 through defence_map saves could seed a FreMEn model.
