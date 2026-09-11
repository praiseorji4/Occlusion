# Research Overview

The ubot platform is a research-grade differential-drive robot used for autonomous mobile robotics research. This section documents the system's research context, known capability gaps, open research directions, and a prioritised literature reading list grounded in the actual technology stack deployed.

---

## Platform research identity

| Dimension | Current state |
|---|---|
| **Navigation** | Full Nav2 stack (bt_navigator, DWB local planner, NavfnPlanner/A*, velocity_smoother, collision_monitor) |
| **Mapping** | SLAM Toolbox online-async (Ceres solver, loop closure enabled) — multiple real maps saved |
| **Localisation** | Wheel odometry only on real robot (EKF/IMU not yet enabled); EKF + BNO055 config ready |
| **Sensing** | LDLidar LD19 (2D laser); BNO055 IMU (configured, inactive on real robot); RGB-D camera (URDF + Gazebo, no real-robot driver active) |
| **Control** | ros2_control + diff_drive_controller; ESP32 30 Hz RPM PID with Jimeno RPM filtering and dead-zone feedforward |
| **Simulation** | Gazebo (gz-sim) with gz_ros2_control, gpu_lidar, rgbd_camera, IMU plugins |
| **Maps collected** | 5 real environments: studio_1, studio_2, studio_3, Tee_map, defence_map |

---

## What works today

Based on the saved map artefacts (`studio_1` through `defence_map` in `/home/chibueze/uni-bot/`) and the presence of a fully configured Nav2 stack with inline changelog comments documenting real tuning decisions, the following capabilities have been demonstrated on real hardware:

1. **2D SLAM mapping** in multiple indoor environments (studios, corridors).
2. **Teleoperation** via keyboard with twist_stamper converting Twist→TwistStamped for ros2_control.
3. **Map saving and reloading** via slam_toolbox serialisation (.data + .posegraph files).
4. **Closed-loop diff-drive control** at 30 Hz through the full ros2_control stack.
5. **Nav2 goal navigation** (tuning history visible in nav2_params.yaml inline comments: BaseObstacle.scale fix, inflation radius correction, A* enabled, CLOSED_LOOP velocity smoother).
6. **LiDAR-based obstacle avoidance** via Nav2 costmaps (voxel_layer in local, obstacle_layer in global).
7. **Diagnostic telemetry** via Serial2 / `diag_publisher.py` for ESP32 PID state.

---

## What is not yet working (top gaps)

1. **IMU fusion**: BNO055 and robot_localization EKF are configured but commented out of `real_robot.launch.py`. The robot runs on wheel odometry only — susceptible to heading drift during long-duration runs and low-traction surfaces.
2. **Camera integration on real robot**: RGB-D camera is in the URDF and Gazebo (640×480, rgbd_camera plugin), but no real-robot camera driver or sensor processing pipeline exists in the workspace.
3. **Ultrasonic sensor integration**: ultrasonic links are defined in the URDF but no driver, no published topics.
4. **PID re-tuning**: current gains (Kp=5, Ki=9, Kd=0) are explicitly marked as untuned starting values in `diff_controller.h`. No empirically tuned gain set exists in the workspace.
5. **Docking / charging**: configured in nav2_params.yaml but no hardware or detection pipeline exists.

See [Gaps](gaps.md) for a comprehensive analysis and [Known Issues](../architecture/issues.md) for actionable fixes.

---

## Research environment

The workspace contains evidence of active research use across multiple sessions:
- 5 distinct saved SLAM maps across different physical environments
- TF tree captures spanning 2026-04-17 through 2026-06-26 (>2 months of development history visible through `frames_*.gv` files)
- Nav2 parameter tuning history documented inline (`# was: ...` comments in nav2_params.yaml)
- Serial2 diagnostic streaming and PlotJuggler integration for real-time PID observation
- Explicit odometry debugging checklist embedded in `ubot_controllers.yaml`

This is consistent with a robot in active research development: core navigation works, but sensor fusion, perception pipelines, and long-duration reliability are the open frontiers.

---

## Navigation to detailed research pages

| Page | Content |
|---|---|
| [Gaps](gaps.md) | 10 capability gaps with technical analysis |
| [Research Directions](directions.md) | 10 open research problems, each with contribution claim, technical approach, ROS2 implementation sketch, and evaluation protocol |
| [Reading List](reading_list.md) | 3-tier prioritised literature grounded in the exact algorithms deployed (Jimeno RPM filter, DWB local planner, SLAM Toolbox, robot_localization) |
