# Experiment Logging Protocol

This page defines a standardised logging protocol for future ubot experiments. The goal is to
produce reproducible, comparable datasets that can support quantitative analysis of odometry
accuracy, navigation performance, and PID controller behaviour.

---

## Bag Recording Commands

All `ros2 bag record` commands use shell date substitution to timestamp the output directory
automatically. Run them from any working directory; bags are stored under
`/home/chibueze/uni-bot/bags/` by convention (create this directory before recording).

### Odometry experiments

Records wheel odometry, joint states, and the full TF tree. Use for straight-line repeatability
trials and wheel-radius / wheel-separation calibration (see the odometry debugging checklist
embedded in `src/ubot/ubot_bringup/config/ubot_controllers.yaml`).

```bash
ros2 bag record \
  /diff_drive_controller/odom \
  /joint_states \
  /tf \
  /tf_static \
  -o /home/chibueze/uni-bot/bags/odometry_$(date +%Y%m%d_%H%M%S)
```

### Navigation experiments

Records the full navigation data set: odometry, laser scan, occupancy map, planned path,
velocity commands (both the Nav2-issued `/cmd_vel` and the post-twist-stamper
`/diff_drive_controller/cmd_vel`), and TF. Sufficient to replay a navigation run in RViz and
compute cross-track error offline.

```bash
ros2 bag record \
  /diff_drive_controller/odom \
  /scan \
  /map \
  /plan \
  /tf /tf_static \
  /cmd_vel \
  /diff_drive_controller/cmd_vel \
  -o /home/chibueze/uni-bot/bags/navigation_$(date +%Y%m%d_%H%M%S)
```

### PID tuning

The `diag_publisher` node must be started manually before recording (it is not included in
`real_robot.launch.py`). Keep `publish_rate` at or below 15 Hz to avoid saturating the
shared `/dev/ttyUSB0` serial command channel.

```bash
# Terminal 1 — start the diagnostic publisher:
ros2 run ubot_debugger diag_publisher

# Terminal 2 — record all 14 diagnostic topics:
ros2 bag record \
  /horizon/diag/rpm/left_actual \
  /horizon/diag/rpm/right_actual \
  /horizon/diag/rpm/left_target \
  /horizon/diag/rpm/right_target \
  /horizon/diag/pid/left_error \
  /horizon/diag/pid/right_error \
  /horizon/diag/pid/left_integral \
  /horizon/diag/pid/right_integral \
  /horizon/diag/pid/left_output \
  /horizon/diag/pid/right_output \
  /horizon/diag/enc/left \
  /horizon/diag/enc/right \
  /horizon/diag/vel/linear \
  /horizon/diag/vel/angular \
  -o /home/chibueze/uni-bot/bags/pid_$(date +%Y%m%d_%H%M%S)
```

---

## Real-Time Monitoring with PlotJuggler

PlotJuggler can stream ROS 2 topics live during a run. Signals recommended in the
`ubot_controllers.yaml` embedded comments:

| Signal | Fields of interest |
|---|---|
| `/diff_drive_controller/odom` | `pose.position.x`, `pose.position.y`, `twist.twist.linear.x` |
| `/joint_states` | `velocity[0..3]` for all four wheel joints |
| `/ubot/diagnostics` | Float32MultiArray — only published when `diag_publish_rate > 0` in the `<ros2_control>` xacro tag (default: 0 = disabled) |
| `/horizon/diag/*` | All 14 topics from `diag_publisher.py` — requires manual `ros2 run ubot_debugger diag_publisher` |

---

## Serial2 Diagnostic Stream (Non-ROS)

The ESP32 firmware (`Ros-esp32_bridge.ino`) continuously emits a 14-field diagnostic line on
`Serial2` (TX = GPIO16) at 115200 baud, independent of the ROS command channel (`Serial`,
USB, GPIO1/3). This stream is also what `diag_publisher.py` queries via the `q` command.

**Frame format** (emitted at 30 Hz by `emitDiagnostics()` in `diff_controller.h`):

```
D <leftEnc> <rightEnc> <leftTargetRpm> <rightTargetRpm> <leftRpmFiltered> <rightRpmFiltered>
  <leftError> <rightError> <leftIntegral> <rightIntegral> <leftOutput> <rightOutput>
  <linVel_ms> <angVel_rads>
```

Fields (in order):

| # | Field | Type | Description |
|---|---|---|---|
| 1 | `leftEnc` | long | Left encoder tick count |
| 2 | `rightEnc` | long | Right encoder tick count |
| 3 | `leftTargetRpm` | float | Commanded left wheel RPM |
| 4 | `rightTargetRpm` | float | Commanded right wheel RPM |
| 5 | `leftRpmFiltered` | float | Filtered actual left RPM (Jimeno low-pass) |
| 6 | `rightRpmFiltered` | float | Filtered actual right RPM |
| 7 | `leftError` | float | PID error, left wheel |
| 8 | `rightError` | float | PID error, right wheel |
| 9 | `leftIntegral` | float | PID integral accumulator, left (clamped ±50) |
| 10 | `rightIntegral` | float | PID integral accumulator, right (clamped ±50) |
| 11 | `leftOutput` | float | PID output (pre-feedforward), left |
| 12 | `rightOutput` | float | PID output (pre-feedforward), right |
| 13 | `linVel_ms` | float | Estimated linear velocity, m/s |
| 14 | `angVel_rads` | float | Estimated angular velocity, rad/s |

To read this stream independently of the ROS command channel, connect a second USB-serial
adapter to GPIO16 and open it at 115200 baud. PlotJuggler can ingest the stream with a
custom CSV/string parser, or you can bridge it into ROS using a lightweight Python node that
reads the serial port and republishes each field as `std_msgs/Float64`.

---

## Map Saving

### Via SLAM Toolbox service (recommended)

When SLAM Toolbox is running in `mapping` mode, call its `save_map` service to write all
four output files atomically:

```bash
ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap \
  "{name: {data: '/home/chibueze/uni-bot/<environment_name>_serial'}}"
```

This produces:

- `<environment_name>_save.pgm` — occupancy image for Nav2 map server
- `<environment_name>_save.yaml` — map metadata (resolution, origin, thresholds)
- `<environment_name>_serial.data` — SLAM Toolbox serialised scan data
- `<environment_name>_serial.posegraph` — pose graph for localisation resumption

### Nav2 map_saver (alternative)

If only the Nav2 map server representation is needed (PGM + YAML, no SLAM Toolbox
serialisation):

```bash
ros2 run nav2_map_server map_saver_cli \
  -f /home/chibueze/uni-bot/<environment_name>_save
```

This produces only the `.pgm` and `.yaml` files; the SLAM Toolbox `.data` /
`.posegraph` pair will not be present, so localisation resumption via SLAM Toolbox
`localization` mode will not be possible unless the session is still active.

---

## TF Tree Snapshot

The `frames_*.gv` files already present in `/home/chibueze/uni-bot/` were produced by:

```bash
ros2 run tf2_tools view_frames
```

Run this at any point during a live session to capture the current TF tree topology and
edge rates. The command writes `frames_<timestamp>.gv` and `frames_<timestamp>.pdf` to the
current working directory. Run it from `/home/chibueze/uni-bot/` to keep captures alongside
the existing collection.

The resulting `.gv` files are also useful for debugging: compare captures taken before and
after a configuration change to confirm that a new transform edge (e.g., `map` → `odom`
from SLAM Toolbox) is being broadcast.

---

## Naming Conventions

| Artefact | Convention | Example |
|---|---|---|
| Bag directories | `<type>_YYYYMMDD_HHMMSS/` | `navigation_20260630_143200/` |
| SLAM Toolbox serial save | `<environment_name>_serial` (no extension) | `studio_4_serial` |
| Nav2 map server save | `<environment_name>_save` | `studio_4_save` |
| TF snapshots | `frames_<timestamp>.gv/.pdf` (auto-named by tf2_tools) | `frames_2026-06-30_14.32.00.gv` |

Storage locations:

- Bags: `/home/chibueze/uni-bot/bags/` — create this directory before first use.
- Maps: `/home/chibueze/uni-bot/` — workspace root, alongside the existing five maps.
- TF snapshots: `/home/chibueze/uni-bot/` — workspace root, alongside the existing captures.

---

## Minimum Experiment Metadata

Record the following alongside every bag, in a plain-text or Markdown file placed inside
the bag directory. Without this metadata a bag is difficult to interpret weeks later.

| Field | Where to find it | Example |
|---|---|---|
| Date and time | System clock | 2026-06-30 14:32 |
| Environment | Physical location | Studio 3, Building B, Room 204 |
| Surface type | Direct observation | Smooth concrete / carpet / tiles |
| Nav2 params version | `git log --oneline -1` or `sha256sum nav2_params.yaml` | `a3f1c2d` |
| PID gains in effect | `src/Ros-esp32_bridge/diff_controller.h` lines 40–50 | Kp=5.0, Ki=9.0, Kd=0.0, MinPwm=50 |
| Wheel geometry | Must match across three files (see note below) | radius=0.033 m, separation=0.264204 m |
| BNO055 enabled | `real_robot.launch.py` — is `bno055_node` uncommented? | No (currently commented out) |
| EKF enabled | `real_robot.launch.py` — is `ekf_node` uncommented? | No (currently commented out) |
| SLAM Toolbox mode | `mapper_params_online_async.yaml` `mode:` field | `mapping` |
| Active map | `mapper_params_online_async.yaml` `map_file_name:` | `/home/chibueze/uni-bot/studio_3_serial` |

### Wheel geometry consistency check

The wheel radius (0.033 m) and wheel separation (0.264204 m) must be consistent across all
three locations before recording. A mismatch will cause silent odometry errors that are
difficult to diagnose post-hoc:

1. `src/Ros-esp32_bridge/diff_controller.h` — `WHEEL_RADIUS_M`, `WHEEL_SEPARATION_M`
2. `src/ubot/ubot_bringup/config/ubot_controllers.yaml` — `wheel_radius`, `wheel_separation`
3. `src/ubot/ubot_description/urdf/components/ubot_wheel.urdf.xacro` — `wheel_radius`,
   `wheel_separation` xacro properties

If these differ, resolve the discrepancy and reflash / rebuild before recording.
