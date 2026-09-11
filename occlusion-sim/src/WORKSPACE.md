# Workspace Folder Structure

Root: `/home/chibueze/uni-bot/src`
*(The `Ros-esp32_bridge/` folder is excluded from this description.)*

---

```
src/
│
├── bno055/                                   BNO055 IMU sensor ROS2 driver package (Python)
│   ├── AUTHORS                               List of package contributors
│   ├── CHANGELOG.rst                         Version history and release notes
│   ├── CONTRIBUTING.md                       Guidelines for contributing to the package
│   ├── LICENSE                               BSD license text
│   ├── LEGACY_LICENSE                        Previous license version
│   ├── README.md                             Package overview, setup, and usage instructions
│   ├── requirements.txt                      Python dependencies
│   ├── setup.cfg                             Additional Python package setup options
│   ├── setup.py                              Package build and installation script
│   ├── package.xml                           ROS2 package manifest (name, deps, maintainer)
│   │
│   ├── .github/workflows/
│   │   ├── build_test.yml                    CI workflow: build and run tests on push/PR
│   │   ├── docs.yml                          CI workflow: auto-generate Sphinx documentation
│   │   └── greetings.yml                     CI workflow: greet first-time contributors
│   │
│   ├── bno055/                               Main Python source package
│   │   ├── __init__.py                       Package initializer
│   │   ├── bno055.py                         Core ROS2 node — reads sensor data and publishes IMU messages
│   │   ├── registers.py                      BNO055 register address constants and calibration maps
│   │   │
│   │   ├── connectors/                       Hardware communication adapters
│   │   │   ├── __init__.py
│   │   │   ├── Connector.py                  Abstract base class defining the connector interface
│   │   │   ├── i2c.py                        I2C communication connector implementation
│   │   │   └── uart.py                       UART/serial communication connector implementation
│   │   │
│   │   ├── error_handling/
│   │   │   ├── __init__.py
│   │   │   └── exceptions.py                 Custom exception classes for sensor errors
│   │   │
│   │   ├── params/
│   │   │   ├── __init__.py
│   │   │   ├── NodeParameters.py             Loads and manages ROS2 node parameters at runtime
│   │   │   ├── bno055_params.yaml            Default sensor parameters (frame, frequency, etc.)
│   │   │   ├── bno055_params_i2c.yaml        I2C-specific parameter overrides
│   │   │   └── bno055_params_test.yaml       Parameters used during automated testing
│   │   │
│   │   └── sensor/
│   │       ├── __init__.py
│   │       └── SensorService.py              High-level service: reads raw registers, computes
│   │                                         orientation/acceleration, fills ROS2 Imu message fields
│   │
│   ├── launch/
│   │   └── bno055.launch.py                  ROS2 launch file to start the BNO055 driver node
│   │
│   ├── resource/
│   │   └── bno055/                           ROS2 ament resource marker directory
│   │
│   └── docs/                                 Sphinx API documentation source
│       ├── conf.py                           Sphinx configuration (theme, extensions, version)
│       ├── index.rst                         Documentation root / table of contents
│       ├── bno055.rst                        Auto-doc page for the main bno055 module
│       ├── connectors.rst                    Auto-doc page for connector modules
│       ├── error_handling.rst                Auto-doc page for exception classes
│       ├── params.rst                        Auto-doc page for parameter management
│       ├── sensor.rst                        Auto-doc page for the sensor service
│       ├── registers.rst                     Auto-doc page for register definitions
│       ├── Makefile                          Build script for docs on Linux/macOS
│       └── make.bat                          Build script for docs on Windows
│
├── sllidar_ros2/                             SLAMTEC RPLiDAR / SLLiDAR ROS2 driver package (C++)
│   ├── CMakeLists.txt                        CMake build rules for the ROS2 node and SDK
│   ├── LICENSE                               Package license
│   ├── README.md                             Setup, wiring, and usage instructions for supported models
│   ├── package.xml                           ROS2 package manifest
│   ├── rplidar_A1.png                        Reference photo of the A1 LiDAR module
│   ├── rplidar_A2.png                        Reference photo of the A2 LiDAR module
│   │
│   ├── launch/                               ROS2 launch files — one per supported LiDAR model
│   │   ├── sllidar_a1_launch.py              Launch config for RPLIDAR A1
│   │   ├── sllidar_a2m7_launch.py            Launch config for RPLIDAR A2M7
│   │   ├── sllidar_a2m8_launch.py            Launch config for RPLIDAR A2M8
│   │   ├── sllidar_a2m12_launch .py          Launch config for RPLIDAR A2M12
│   │   ├── sllidar_a3_launch.py              Launch config for RPLIDAR A3
│   │   ├── sllidar_c1_launch.py              Launch config for RPLIDAR C1
│   │   ├── sllidar_s1_launch.py              Launch config for SLLiDAR S1 (serial)
│   │   ├── sllidar_s1_tcp_launch.py          Launch config for SLLiDAR S1 (TCP)
│   │   ├── sllidar_s2_launch.py              Launch config for SLLiDAR S2
│   │   ├── sllidar_s2e_launch.py             Launch config for SLLiDAR S2E
│   │   ├── sllidar_s3_launch.py              Launch config for SLLiDAR S3
│   │   ├── sllidar_t1_launch.py              Launch config for SLLiDAR T1
│   │   ├── view_sllidar_a1_launch.py         Launches driver + RViz for A1
│   │   ├── view_sllidar_a2m7_launch.py       Launches driver + RViz for A2M7
│   │   ├── view_sllidar_a2m8_launch.py       Launches driver + RViz for A2M8
│   │   ├── view_sllidar_a2m12_launch.py      Launches driver + RViz for A2M12
│   │   ├── view_sllidar_a3_launch.py         Launches driver + RViz for A3
│   │   ├── view_sllidar_c1_launch.py         Launches driver + RViz for C1
│   │   ├── view_sllidar_s1_launch.py         Launches driver + RViz for S1 (serial)
│   │   ├── view_sllidar_s1_tcp_launch.py     Launches driver + RViz for S1 (TCP)
│   │   ├── view_sllidar_s2_launch.py         Launches driver + RViz for S2
│   │   ├── view_sllidar_s2e_launch.py        Launches driver + RViz for S2E
│   │   ├── view_sllidar_s3_launch.py         Launches driver + RViz for S3
│   │   └── view_sllidar_t1_launch.py         Launches driver + RViz for T1
│   │
│   ├── rviz/
│   │   └── sllidar_ros2.rviz                 RViz2 config for real-time LiDAR scan visualization
│   │
│   ├── scripts/
│   │   ├── rplidar.rules                     Linux udev rules granting non-root USB access
│   │   ├── create_udev_rules.sh              Installs the udev rules to /etc/udev/rules.d/
│   │   └── delete_udev_rules.sh              Removes the udev rules
│   │
│   ├── src/
│   │   ├── sllidar_node.cpp                  Main ROS2 driver node: communicates with LiDAR via SDK,
│   │   │                                     publishes sensor_msgs/LaserScan on /scan
│   │   └── sllidar_client.cpp                Standalone client utility for direct SDK interaction
│   │
│   └── sdk/                                  SLAMTEC LiDAR C++ SDK (vendored)
│       ├── Makefile                          SDK standalone build rules
│       │
│       ├── include/                          Public SDK headers
│       │   ├── rplidar.h                     Top-level RPLIDAR header (includes all sub-headers)
│       │   ├── rplidar_cmd.h                 RPLIDAR command byte definitions
│       │   ├── rplidar_driver.h              RPLIDAR driver class interface
│       │   ├── rplidar_protocol.h            RPLIDAR wire protocol structures
│       │   ├── rptypes.h                     RPLIDAR primitive type aliases
│       │   ├── sl_lidar.h                    Top-level SLLiDAR header
│       │   ├── sl_lidar_cmd.h                SLLiDAR command definitions
│       │   ├── sl_lidar_driver.h             SLLiDAR driver interface
│       │   ├── sl_lidar_driver_impl.h        Driver implementation details exposed to the SDK
│       │   ├── sl_lidar_protocol.h           SLLiDAR protocol structures
│       │   ├── sl_types.h                    SLLiDAR type definitions
│       │   └── sl_crc.h                      CRC utility declarations
│       │
│       └── src/                              SDK implementation
│           ├── rplidar_driver.cpp            RPLIDAR legacy driver implementation
│           ├── sdkcommon.h                   Internal SDK common utilities
│           ├── sl_lidar_driver.cpp           Main SLLiDAR driver logic (scan start/stop, data poll)
│           ├── sl_async_transceiver.cpp      Asynchronous packet send/receive engine
│           ├── sl_async_transceiver.h
│           ├── sl_crc.cpp                    CRC-32 checksum implementation
│           ├── sl_lidarprotocol_codec.cpp    Protocol frame encoder and decoder
│           ├── sl_lidarprotocol_codec.h
│           ├── sl_serial_channel.cpp         Serial port channel (wraps arch serial code)
│           ├── sl_tcp_channel.cpp            TCP socket channel
│           ├── sl_udp_channel.cpp            UDP socket channel
│           │
│           ├── arch/                         Platform-specific low-level code
│           │   ├── linux/
│           │   │   ├── arch_linux.h          Linux platform macros
│           │   │   ├── net_serial.cpp        Linux serial port open/read/write
│           │   │   ├── net_serial.h
│           │   │   ├── net_socket.cpp        Linux TCP/UDP socket implementation
│           │   │   ├── thread.hpp            POSIX thread wrapper
│           │   │   ├── timer.cpp             Monotonic timer implementation
│           │   │   └── timer.h
│           │   ├── macOS/
│           │   │   ├── arch_macOS.h          macOS platform macros
│           │   │   ├── net_serial.cpp        macOS serial port implementation
│           │   │   ├── net_serial.h
│           │   │   ├── net_socket.cpp        macOS socket implementation
│           │   │   ├── thread.hpp            macOS thread wrapper
│           │   │   ├── timer.cpp
│           │   │   └── timer.h
│           │   └── win32/
│           │       ├── arch_win32.h          Windows platform macros
│           │       ├── net_serial.cpp        Windows COM port implementation
│           │       ├── net_serial.h
│           │       ├── net_socket.cpp        Windows Winsock2 implementation
│           │       ├── timer.cpp
│           │       ├── timer.h
│           │       └── winthread.hpp         Windows thread wrapper
│           │
│           ├── hal/                          Hardware Abstraction Layer — OS-neutral interfaces
│           │   ├── abs_rxtx.h                Abstract RX/TX byte-stream interface
│           │   ├── assert.h                  Assertion macro (maps to platform assert)
│           │   ├── byteops.h                 Byte-level read/write helpers
│           │   ├── byteorder.h               Endianness conversion utilities
│           │   ├── event.h                   Cross-platform event/signal primitive
│           │   ├── locker.h                  Cross-platform mutex / lock guard
│           │   ├── socket.h                  Abstract socket interface
│           │   ├── thread.cpp                Thread base implementation
│           │   ├── thread.h                  Cross-platform thread interface
│           │   ├── types.h                   HAL primitive types (u8, u16, u32, etc.)
│           │   ├── util.h                    Miscellaneous utilities
│           │   └── waiter.h                  Timed-wait helper
│           │
│           └── dataunpacker/                 Parses raw LiDAR scan packets into point data
│               ├── dataunpacker.cpp          Entry point: dispatches packets to the right handler
│               ├── dataunpacker.h
│               ├── dataunnpacker_commondef.h Common data-type definitions for unpacker
│               ├── dataunnpacker_internal.h  Internal unpacker state structures
│               ├── dataupacker_namespace.h   Namespace/alias declarations
│               └── unpacker/
│                   ├── handler_capsules.cpp  Handles "capsule" (dense) scan packet format
│                   ├── handler_capsules.h
│                   ├── handler_hqnode.cpp    Handles high-quality (HQ) node packet format
│                   ├── handler_hqnode.h
│                   ├── handler_normalnode.cpp Handles standard node packet format
│                   └── handler_normalnode.h
│
├── ubot/                                     UBot differential-drive robot — ROS2 workspace packages
│   ├── .gitignore                            Git ignore rules for the ubot workspace
│   ├── ubot.repos                            VCS tool manifest listing external git repos to clone
│   │
│   ├── ubot_bringup/                         Robot-level launch and configuration package
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── LICENSE
│   │   │
│   │   ├── launch/
│   │   │   ├── real_robot.launch.py          Starts all nodes needed to run on physical hardware
│   │   │   │                                 (hardware interface, LiDAR, IMU, robot state publisher)
│   │   │   ├── sim.launch.py                 Starts simulation stack in Gazebo
│   │   │   ├── laptop_slam_nav2.launch.py    Launches SLAM Toolbox + Nav2 navigation on the laptop
│   │   │   └── test_wheels.launch.py         Diagnostic launch for testing wheel encoder feedback
│   │   │
│   │   ├── config/
│   │   │   ├── ubot_controllers.yaml         ros2_control controller configs for real robot
│   │   │   │                                 (diff_drive_controller, joint_state_broadcaster)
│   │   │   ├── sim_ubot_controllers.yaml     Same controller configs tuned for simulation
│   │   │   ├── nav2_params.yaml              Nav2 stack parameters (costmaps, planner, controller)
│   │   │   ├── sim_nav2_params.yaml          Nav2 parameters adjusted for simulation
│   │   │   ├── mapper_params_online_async.yaml SLAM Toolbox online async mapping parameters
│   │   │   └── sim_ekf.yaml                  robot_localization EKF parameters for simulation
│   │   │
│   │   └── worlds/
│   │       └── basic.sdf                     Gazebo SDF world file — simple empty arena
│   │
│   ├── ubot_control/                         ros2_control hardware interface package (C++)
│   │   ├── CMakeLists.txt
│   │   ├── package.xml
│   │   ├── LICENSE
│   │   ├── ubot_control.xml                  Plugin descriptor registering the hardware interface
│   │   │                                     with the ros2_control plugin loader
│   │   │
│   │   ├── include/ubot_control/
│   │   │   ├── ubot_hardware_interface.hpp   Declares UbotHardwareInterface — the ros2_control
│   │   │   │                                 SystemInterface that bridges ROS2 commands to hardware
│   │   │   ├── wheel.hpp                     Wheel struct: stores encoder ticks, velocity, and
│   │   │   │                                 computes radians from tick counts
│   │   │   └── arduino_comms.hpp             Serial communication class: sends velocity commands
│   │   │                                     to and reads encoder feedback from the Arduino/ESP32
│   │   │
│   │   └── src/
│   │       └── ubot_hardware_interface.cpp   Implements on_init, on_activate, read, write lifecycle
│   │                                         methods; reads joint states and writes velocity commands
│   │
│   └── ubot_description/                     Robot model and visualization package
│       ├── CMakeLists.txt
│       ├── package.xml
│       ├── LICENSE
│       │
│       ├── launch/
│       │   └── display.launch.py             Launches robot_state_publisher + RViz to visualize URDF
│       │
│       ├── meshes/                           STL 3D models for robot parts
│       │   ├── chassis.stl                   Main chassis body
│       │   ├── chassis_rotated.stl           Rotated variant of the chassis
│       │   ├── wheel.stl                     Drive wheel
│       │   ├── camera.stl                    Camera bracket/mount
│       │   ├── lidar.stl                     LiDAR bracket/mount
│       │   ├── ultrasonic.stl                Ultrasonic sensor bracket
│       │   └── latest.stl                    Latest chassis revision (WIP)
│       │
│       ├── rviz/
│       │   ├── display.rviz                  RViz2 config for basic robot model display
│       │   └── slam.rviz                     RViz2 config for SLAM (map + robot + LiDAR scan)
│       │
│       └── urdf/                             URDF/Xacro robot description files
│           ├── body/
│           │   ├── ubot_robot.urdf.xacro     Top-level robot xacro — includes all sub-xacros
│           │   ├── ubot_base.urdf.xacro      Chassis link, caster wheel, and base_footprint
│           │   └── ubot_wheel.urdf.xacro     Left/right wheel links, joints, and inertia macros
│           │
│           ├── sensors/
│           │   ├── ubot_lidar.urdf.xacro     LiDAR link and fixed joint to the chassis
│           │   ├── ubot_imu.urdf.xacro       IMU link and fixed joint, plus sensor origin
│           │   ├── ubot_camera.urdf.xacro    Camera link, optical frame, and joint
│           │   └── ubot_ultrasonic.urdf.xacro Ultrasonic sensor link and joint
│           │
│           ├── gazebo/
│           │   ├── ubot_gazebo.urdf.xacro    Gazebo material colors and physics plugin params
│           │   ├── ubot_camera_gazebo.urdf.xacro Gazebo camera sensor plugin config
│           │   └── gazebo_control.xacro      Loads the gazebo_ros2_control plugin
│           │
│           └── ros2_control/
│               └── ubot_ros2_control.xacro   ros2_control <hardware> tag wiring the URDF to
│                                             either the real hardware interface or a Gazebo mock
│
└── .vscode/
    └── settings.json                         VS Code workspace settings (file associations,
                                              linting, C++ IntelliSense paths, etc.)
```

---

## Package Summary

| Package | Language | Role |
|---|---|---|
| `bno055` | Python | ROS2 driver for the BNO055 9-DOF IMU (I2C or UART) |
| `sllidar_ros2` | C++ | ROS2 driver + vendored SDK for SLAMTEC RPLiDAR / SLLiDAR family |
| `ubot_bringup` | Python (launch) | Top-level robot launch files and all YAML configuration |
| `ubot_control` | C++ | ros2_control hardware interface linking ROS2 to the motor controller |
| `ubot_description` | Xacro / STL | URDF robot model, meshes, and RViz configs |
