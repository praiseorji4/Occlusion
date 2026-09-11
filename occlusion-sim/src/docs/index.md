# ubot — ROS2 Differential-Drive Robot

This site documents the `ubot` ROS2 workspace, a 4-wheel differential-drive mobile robot platform developed for autonomous navigation research using ROS2 Jazzy, ros2_control, Nav2, and SLAM Toolbox.

---

## Robot snapshot

| Property | Value |
|---|---|
| Platform | 4WD differential drive, front-wheel encoders only |
| Compute | Raspberry Pi (host) + ESP32 NodeMCU (motor controller) |
| Chassis | base_link: 0.41 × 0.237 × 0.203 m, mass 3.0 kg |
| Wheel radius | 0.033 m |
| Wheel separation (track) | 0.264204 m |
| Encoder resolution | 3956 CPR per wheel |
| Control loop rate | 30 Hz (Pi) / 30 Hz PID (ESP32) |
| LiDAR | LDLidar LD19 — `/scan`, 230400 baud, `/dev/ttyUSB1` |
| IMU | Bosch BNO055, I2C — `/bno055/imu` (configured, not yet active on real robot) |
| ESP32 firmware | `Ros-esp32_bridge.ino` — text-command serial bridge on `/dev/ttyUSB0` |
| ROS2 distro | Jazzy Jalisco (inferred) |

---

## Quick links

| Want to… | Go to |
|---|---|
| Understand the full system | [Architecture Overview](architecture/overview.md) |
| See all ROS topics/services | [Topics](interfaces/topics.md) · [Services](interfaces/services.md) |
| Launch the robot | [Running Guide](guides/running.md) |
| Build from source | [Building Guide](guides/building.md) · [INSTALL.md](../INSTALL.md) |
| Calibrate odometry | [Calibration Guide](guides/calibration.md) |
| Fix something broken | [Troubleshooting](guides/troubleshooting.md) · [Known Issues](architecture/issues.md) |
| Explore the TF tree | [TF Tree](architecture/tf_tree.md) |
| Understand the firmware | [Embedded Firmware](hardware/embedded.md) |
| Read the research context | [Research Overview](research/overview.md) |

---

## Workspace packages

```
/home/chibueze/uni-bot/src/
├── bno055/                  BNO055 IMU driver (vendored, Python)
├── ldlidar_stl_ros2/        LDLidar LD19 driver (vendored, C++) — LIVE on real robot
├── sllidar_ros2/            RPLiDAR driver (vendored, C++) — not used on real robot currently
├── Ros-esp32_bridge/        LIVE ESP32 firmware (Arduino sketch, text-command serial protocol)
├── esp32/hacker/            ORPHANED micro-ROS firmware — do NOT flash, conflicts with ros2_control
└── ubot/
    ├── ubot_bringup/        Launch files + YAML configs
    ├── ubot_control/        ros2_control hardware_interface plugin (UbotHardware)
    ├── ubot_debugger/       Diagnostics publisher node (Python)
    └── ubot_description/    URDF/xacro, meshes, RViz configs
```

---

## Key architectural decisions

- **ros2_control over micro-ROS**: the live system uses a text-command serial bridge (`Ros-esp32_bridge.ino`) that the `UbotHardware` plugin reads/writes via libserial. An alternative micro-ROS firmware (`hacker.ino`) exists in the workspace but is not wired to the ros2_control graph — see [Embedded Firmware](hardware/embedded.md) for the full comparison.
- **Front-encoder-only odometry**: the chassis has four driven wheels but encoders only on the front pair. The `diff_drive_controller` is configured for front wheels only; rear joints are state-only mirrors. This simplifies wiring at the cost of slightly noisier rear-wheel slip estimation.
- **No active IMU fusion on real robot yet**: the BNO055 driver and robot_localization EKF are fully configured in `real_robot.launch.py` but commented out of the launch return. The robot currently navigates on wheel odometry alone. See [Known Issues](architecture/issues.md).
- **Multiple saved environments**: real SLAM Toolbox maps are saved in `/home/chibueze/uni-bot/` (studio_1, studio_2, studio_3, Tee_map, defence_map), evidence of active research use in multiple spaces.

---

## Documentation generated

This site was generated after exhaustive Phase 0 source reading — every topic name, parameter value, frame name, and command string matches current source exactly. Where a value could not be confirmed from static analysis alone, the text uses the marker:

> ⚠️ Could not be determined from source — requires runtime inspection.
