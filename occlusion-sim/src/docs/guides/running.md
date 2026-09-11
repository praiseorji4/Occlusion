# Running the Full Stack

This guide covers the complete operational procedure for the ubot robot — from hardware pre-flight to SLAM mapping, Nav2 navigation, teleoperation, and simulation.

---

## Hardware Checklist (Pre-Launch)

Verify all of the following before running any launch command:

- [ ] ESP32 NodeMCU USB cable connected; appears as `/dev/ttyUSB0`
- [ ] LDLidar LD19 USB cable connected; appears as `/dev/ttyUSB1`
- [ ] Robot powered on; motor drivers armed (BTS7960 enable signals active)
- [ ] Workspace sourced in every terminal you will use:
  ```bash
  source /opt/ros/jazzy/setup.bash
  source /home/chibueze/uni-bot/install/setup.bash
  ```

Quick serial-port check:

```bash
ls -la /dev/ttyUSB*
```

You must see both `/dev/ttyUSB0` and `/dev/ttyUSB1` before proceeding.

---

## Terminal 1 — Core Bringup

```bash
ros2 launch ubot_bringup real_robot.launch.py
```

This single launch file starts the entire hardware stack. Here is the startup sequence and what to look for:

| Time | Event | Log message to look for |
|---|---|---|
| 0 s | `robot_state_publisher` starts | `Received urdf xml string` |
| 0 s | `ros2_control_node` starts, `UbotHardware` opens `/dev/ttyUSB0` | Serial connection messages |
| 0 s | `twist_stamper` starts | Node ready |
| 0 s | `ldlidar_node` starts, opens `/dev/ttyUSB1` at 230400 | `LDLiDAR_LD19` initialisation messages |
| 5 s | `joint_state_broadcaster` spawner fires | `Successfully activated 'joint_state_broadcaster'` |
| 6 s | `diff_drive_controller` spawner fires | `Successfully activated 'diff_drive_controller'` |

**Success indicators**: After ~8 seconds you should see no error messages, and `/diff_drive_controller/odom` and `/scan` topics should be active (verify in Terminal 2).

**Important**: The `bno055` IMU node and the `ekf_filter_node` (robot_localization) are defined in the launch file but are **commented out** — the robot currently navigates on wheel odometry only. To enable them, see [Calibration — BNO055 IMU Calibration](calibration.md) and uncomment the relevant lines in `real_robot.launch.py`.

---

## Terminal 2 — SLAM Mapping

Run SLAM Toolbox in online async mapping mode. The parameter file is in the installed workspace:

```bash
ros2 launch slam_toolbox online_async_launch.py \
  slam_params_file:=/home/chibueze/uni-bot/install/ubot_bringup/share/ubot_bringup/config/mapper_params_online_async.yaml
```

Key parameters in `mapper_params_online_async.yaml`:

| Parameter | Value | Meaning |
|---|---|---|
| `mode` | `mapping` | Active mapping (not localisation) |
| `scan_topic` | `/scan` | Must match LDLidar output topic |
| `odom_frame` | `odom` | Must match diff_drive_controller |
| `base_frame` | `base_footprint` | Must match URDF |
| `map_file_name` | `/home/chibueze/uni-bot/studio_3_serial` | **Update this for a new environment** — it is used for serialised save/load, not for the live map |
| `minimum_travel_distance` | `0.5` m | Robot must move 0.5 m before a new scan is accepted |
| `minimum_travel_heading` | `0.5` rad | Robot must rotate 0.5 rad before a new scan is accepted |

> **Note on `map_file_name`**: This path refers to a prior SLAM Toolbox serialised save (`.posegraph` / `.data` files). It is used when switching to `localization` mode to resume a previous session. For a fresh mapping run in a new space, update this path to reflect your new environment name, for example `/home/chibueze/uni-bot/my_lab`.

SLAM broadcasts the `map` → `odom` TF edge. Once SLAM is running, verify with:

```bash
ros2 run tf2_tools view_frames
```

You should see `map -> odom -> base_footprint` in the output PDF.

---

## Terminal 3 — Navigation (Nav2)

```bash
ros2 launch nav2_bringup bringup_launch.py \
  params_file:=/home/chibueze/uni-bot/install/ubot_bringup/share/ubot_bringup/config/nav2_params.yaml \
  use_sim_time:=False
```

Nav2 requires an active `map` frame (from SLAM or AMCL) before it can plan paths. Key topics Nav2 depends on:

- `/scan` — laser scans for costmap obstacle updates
- `/diff_drive_controller/odom` — odometry for the `bt_navigator` and `velocity_smoother`
- `map` → `odom` TF — provided by SLAM Toolbox

The `collision_monitor` reads `/scan` and publishes velocity-limited commands. The `velocity_smoother` uses `CLOSED_LOOP` feedback from `/diff_drive_controller/odom` for smooth acceleration.

---

## Terminal 4 — Teleoperation or Goal Sending

### Keyboard teleoperation

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Commands flow: `/cmd_vel` (Twist) → `twist_stamper` → `/diff_drive_controller/cmd_vel` (TwistStamped).

The `twist_stamper` is already running (started by `real_robot.launch.py`) — teleop publishes to plain `/cmd_vel` and the stamper handles the conversion automatically.

### Sending a Nav2 navigation goal

In RViz (see below), use the `Nav2 Goal` tool to click a destination on the map. Alternatively, from the command line:

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

---

## Saving a Map

After mapping an environment, save the serialised SLAM Toolbox map:

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: '/home/chibueze/uni-bot/my_new_map'}}"
```

This writes `my_new_map.posegraph` and `my_new_map.data`. To also save a standard occupancy grid (`.pgm` + `.yaml`) for Nav2:

```bash
ros2 run nav2_map_server map_saver_cli -f /home/chibueze/uni-bot/my_new_map
```

Previously saved maps are stored at `/home/chibueze/uni-bot/` (e.g. `studio_1_save`, `studio_2_save`, `studio_3_serial`, `Tee_map`, `defence_map`).

---

## RViz Visualisation

The workspace provides a pre-configured RViz display launch. The `display.launch.py` file was not found at the expected path during documentation preparation:

⚠️ `display.launch.py` — could not be determined from source; check available launch files with:

```bash
ros2 launch ubot_bringup --show-args
```

As a fallback, launch RViz manually:

```bash
rviz2
```

Add these displays:
- **RobotModel** (topic `/robot_description`)
- **LaserScan** (topic `/scan`, frame `lidar_link`)
- **Odometry** (topic `/diff_drive_controller/odom`)
- **Map** (topic `/map`)
- **TF**

---

## Optional: Diagnostics

The `diag_publisher` node polls the ESP32's `q` diagnostic command and republishes 14 fields as individual `Float64` topics. It is **not** started by `real_robot.launch.py` — run it manually:

```bash
ros2 run ubot_debugger diag_publisher
```

Default parameters: `serial_port=/dev/ttyUSB0`, `baud_rate=115200`, `publish_rate=10.0 Hz`.

> **Warning**: The node exits immediately with `SystemExit(1)` if it cannot open the serial port. Only one process can hold `/dev/ttyUSB0` at a time. The `diag_publisher` uses the `q` command (a read-only diagnostic snapshot) alongside the running `UbotHardware` interface — this should coexist, but if you see an immediate exit, check whether another process has the port locked (see [Troubleshooting — diag_publisher exits immediately](troubleshooting.md)).

Topics published under `/horizon/diag/*`:

```
/horizon/diag/enc/left            /horizon/diag/enc/right
/horizon/diag/rpm/left_target     /horizon/diag/rpm/right_target
/horizon/diag/rpm/left_actual     /horizon/diag/rpm/right_actual
/horizon/diag/pid/left_error      /horizon/diag/pid/right_error
/horizon/diag/pid/left_integral   /horizon/diag/pid/right_integral
/horizon/diag/pid/left_output     /horizon/diag/pid/right_output
/horizon/diag/vel/linear          /horizon/diag/vel/angular
```

Visualise in PlotJuggler by subscribing to these topics.

---

## Simulation (Gazebo)

To run in simulation without physical hardware:

```bash
ros2 launch ubot_bringup sim.launch.py
```

This uses `use_gazebo:=true` in the URDF, which activates the `gz_ros2_control/GazeboSimSystem` hardware interface instead of `UbotHardware`, and enables command interfaces on all four wheels (not just the front pair). The simulation uses `use_sim_time:=true`.

For Nav2 in simulation, use `sim_nav2_params.yaml` instead of `nav2_params.yaml`:

```bash
ros2 launch nav2_bringup bringup_launch.py \
  params_file:=/home/chibueze/uni-bot/install/ubot_bringup/share/ubot_bringup/config/sim_nav2_params.yaml \
  use_sim_time:=True
```

---

## See Also

- [Getting Started](getting_started.md) — first-time hardware setup
- [Building from Source](building.md) — workspace build instructions
- [Calibration](calibration.md) — wheel geometry and IMU calibration
- [Tuning](tuning.md) — Nav2 and controller parameter tuning
- [Troubleshooting](troubleshooting.md) — fault finding
