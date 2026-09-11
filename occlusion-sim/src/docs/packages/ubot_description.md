# ubot_description

The model definition package for the ubot robot. It contains all URDF/xacro files, meshes, and RViz configurations. This package has no compiled code — it is a pure data package that installs its directories into the ROS 2 share path.

## Overview

| Field | Value |
|---|---|
| Package name | `ubot_description` |
| Version | `0.0.0` |
| License | BSD-3-Clause |
| Build type | `ament_cmake` |
| Maintainer | chibueze (praiseorji4@gmail.com) |
| ROS package dependencies | `urdf`, `xacro` |
| Robot name (URDF) | `ubot` (`<robot name="ubot">`) |

## Install Targets

`CMakeLists.txt` installs four directories:

```cmake
install(
  DIRECTORY launch meshes urdf rviz
  DESTINATION share/${PROJECT_NAME}
)
```

## Generating robot_description

The robot description is generated at launch time via the `xacro` command. The `use_gazebo` argument selects between real-hardware and simulation plugin paths:

```bash
# Real robot (use_gazebo:=false — default):
xacro urdf/body/ubot_robot.urdf.xacro use_gazebo:=false

# Simulation:
xacro urdf/body/ubot_robot.urdf.xacro use_gazebo:=true
```

Both `real_robot.launch.py` and `sim.launch.py` use `xacro Command` substitution to generate this at launch time, not at build time.

## Top-Level xacro: ubot_robot.urdf.xacro

**Path:** `urdf/body/ubot_robot.urdf.xacro`

### xacro Arguments

| Argument | Default | Description |
|---|---|---|
| `robot_name` | `ubot_robot` | Robot name (not the URDF `<robot name>` attribute; passed as xacro argument). |
| `prefix` | `""` | Joint/link name prefix (for multi-robot or namespaced deployments). |
| `use_gazebo` | `false` | Selects hardware vs. simulation plugin in `ubot_ros2_control.xacro` and conditionally includes `ubot_gazebo`. |

### Include Order

The file includes 8 sub-xacros in this order:

1. `urdf/body/ubot_base.urdf.xacro`
2. `urdf/sensors/ubot_lidar.urdf.xacro`
3. `urdf/sensors/ubot_ultrasonic.urdf.xacro`
4. `urdf/sensors/ubot_camera.urdf.xacro`
5. `urdf/sensors/ubot_imu.urdf.xacro`
6. `urdf/body/ubot_wheel.urdf.xacro`
7. `urdf/ros2_control/ubot_ros2_control.xacro`
8. `urdf/gazebo/ubot_gazebo.urdf.xacro`

### Macro Invocation Order

After includes, macros are invoked in this order (all receive `prefix`):

1. `ubot_ros2_control(prefix, use_gazebo)` — always invoked
2. `ubot_gazebo(prefix)` — invoked only if `use_gazebo=true`
3. `ubot_base(prefix)`
4. `ubot_wheel(prefix)`
5. `ubot_lidar(prefix)`
6. `ubot_ultrasonic(prefix)`
7. `ubot_camera(prefix)`
8. `ubot_imu(prefix)`

## URDF File Inventory

### Body

| File | Content |
|---|---|
| `urdf/body/ubot_robot.urdf.xacro` | Top-level entry point. `<robot name="ubot">`. Defines xacro arguments and orchestrates all includes and macro calls. |
| `urdf/body/ubot_base.urdf.xacro` | `base_footprint` link, `base_link` box (0.41 × 0.2371 × 0.2025 m, mass 3.0 kg), and the fixed `base_footprint → base_link` joint. |
| `urdf/body/ubot_wheel.urdf.xacro` | 4 continuous wheel joints: `front_left_wheel_joint`, `front_right_wheel_joint`, `rear_left_wheel_joint`, `rear_right_wheel_joint`. `wheel_radius=0.033`, `wheel_separation=0.264204`, `wheel_mass=0.4 kg`. No caster wheel. |

### Sensors

| File | Content |
|---|---|
| `urdf/sensors/ubot_lidar.urdf.xacro` | `lidar_link` at origin `[-0.1425, 0, 0.2147]` relative to `base_link`. Fixed joint. |
| `urdf/sensors/ubot_imu.urdf.xacro` | `imu_link` at `[-0.0101, 0.0148, 0.18]` relative to `base_link`. Fixed joint. |
| `urdf/sensors/ubot_camera.urdf.xacro` | `camera_link` at `[0.205, 0, 0.162618]`. Child frames: `camera_depth_frame`, `camera_optical_frame` (child of depth), `camera_rgb_frame`. |
| `urdf/sensors/ubot_ultrasonic.urdf.xacro` | Left and right ultrasonic sensor links. No live ROS driver or topic confirmed wired — likely aspirational hardware. |

### ros2_control

| File | Content |
|---|---|
| `urdf/ros2_control/ubot_ros2_control.xacro` | 105-line macro `ubot_ros2_control(prefix, use_gazebo)`. Defines the `<ros2_control>` block. Real hardware: plugin `ubot_control/UbotHardware` with all 10 parameters. Sim: plugin `gz_ros2_control/GazeboSimSystem`. Front joints always get `command_interface(velocity)` + both state interfaces. Rear joints: command_interface added only for `use_gazebo=true`; both state interfaces always present. |

### Gazebo

| File | Content |
|---|---|
| `urdf/gazebo/ubot_gazebo.urdf.xacro` | Macro `ubot_gazebo(prefix)`. Adds the `gz_ros2_control` Gazebo plugin. Defines a `gpu_lidar` sensor on `lidar_link` (→ `/scan`, 10 Hz, 720 samples), an `rgbd_camera` (640×480) on `camera_link`, and an `imu` sensor at 100 Hz with Gaussian bias/noise model. |
| `urdf/gazebo/ubot_camera_gazebo.urdf.xacro` | Gazebo-specific camera plugin/sensor properties. |
| `urdf/gazebo/gazebo_control.xacro` | **Dead file** — standalone DiffDrive plugin from an earlier architecture. Not included anywhere in the xacro include tree. Should be removed or clearly marked as obsolete. |

## TF Frame Summary

The static frames defined by this URDF (when `use_gazebo:=false`):

```
base_footprint
└── base_link (fixed)
    ├── front_left_wheel_link  (continuous joint)
    ├── front_right_wheel_link (continuous joint)
    ├── rear_left_wheel_link   (continuous joint)
    ├── rear_right_wheel_link  (continuous joint)
    ├── lidar_link             (fixed, at [-0.1425, 0, 0.2147])
    ├── imu_link               (fixed, at [-0.0101, 0.0148, 0.18])
    └── camera_link            (fixed, at [0.205, 0, 0.162618])
        ├── camera_depth_frame (fixed)
        │   └── camera_optical_frame (fixed)
        └── camera_rgb_frame   (fixed)
```

This matches the captured TF tree from `frames_2026-06-26_17.33.00.gv` — confirmed real runtime ground truth.

## Mesh Files

Seven STL meshes are in `meshes/`:

| File | Description |
|---|---|
| `latest.stl` | Most recent chassis model. |
| `chassis.stl` | Original chassis mesh. |
| `chassis_rotated.stl` | Chassis mesh with corrected orientation. |
| `wheel.stl` | Single wheel mesh (used for all 4 wheel links). |
| `lidar.stl` | LiDAR housing mesh. |
| `camera.stl` | Camera module mesh. |
| `ultrasonic.stl` | Ultrasonic sensor mesh. |

## RViz Configurations

| File | Purpose |
|---|---|
| `rviz/display.rviz` | General display configuration for robot visualization. Used by the `display.launch.py` launch file. |
| `rviz/slam.rviz` | SLAM-focused configuration. Used by `laptop_slam_nav2.launch.py`. |

## Known Issues

### `gazebo_control.xacro` is a Dead File (Low)

`urdf/gazebo/gazebo_control.xacro` is never included anywhere in the xacro include tree. It contains a standalone DiffDrive plugin from an earlier architecture that predates the current `gz_ros2_control` approach. It is a source of potential confusion and should be removed or moved to an `_archive/` subdirectory.

### Ultrasonic Sensor Links Without an Active Driver

`ubot_ultrasonic.urdf.xacro` defines left and right ultrasonic links in the URDF, but no ROS 2 driver node or published topic for ultrasonic data was found anywhere in the workspace. These links are likely aspirational hardware or placeholders.

## See Also

- [ubot_control](ubot_control.md) — hardware interface plugin referenced in `ubot_ros2_control.xacro`
- [ubot_bringup](ubot_bringup.md) — launch files that process this xacro to generate `robot_description`
- [ldlidar_stl_ros2](ldlidar_stl_ros2.md) — driver that publishes to `lidar_link` frame at `/scan`
- [bno055](bno055.md) — IMU driver that publishes to `imu_link` frame
