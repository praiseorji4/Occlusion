<!--
  Working notes for the ubot packages: what was measured, what was tried, and why
  things are the way they are. Kept beside the code so it travels with it.

  CANONICAL LOCATION (2026-09-15): these packages are worked on in ONE place -
  this checkout. Earlier there were copies under Horizon/ and Documents/Chibueze/uni-bot/
  that drifted apart and had to be merged by hand; do not make another.
  The robot (horizon.local) tracks Unipod-Robotics/uni-bot and is deployed to by rsync.
-->

# uni-bot / Horizon — context for Claude

ROS 2 **Jazzy** workspace for the Horizon robot (4WD skid-steer, ESP32 motor bridge,
LD06 lidar, BNO055 IMU, OAK-D Lite + RPi cam v1). Main packages: `src/ubot/*`.

## Status as of 2026-09-11 (work done on Windows, not yet built/tested)

`ubot_description` was switched from box/cylinder placeholders to the real
**Inspection Robot 01** Fusion 360 body. Frame/joint names were kept identical so
ros2_control, Nav2, EKF, SLAM, gz bridges and RViz configs are unchanged.

- Meshes: `ubot_description/meshes/inspection/*.stl` (mm, Fusion world coords, scale 0.001).
- `base_link`: centred between the 4 wheels at chassis underside, X forward.
  `base_footprint` → `base_link` z = 0.066375. Mesh origin offset x = -0.16238.
- Wheels: joints at (±0.1275, ±0.1132, -0.033875), axis +Y, r 0.0325, width 0.0264.
  Collision = cylinders; base collision = 2 boxes (body + mast).
- `lidar_link` (LD06 centre): (0.030584, 0, 0.296434).
- `camera_link` = **OAK-D Lite** (RGB lens centre): (0.167769, 0, 0.085654), pitch +0.0714 rad (4.09° down).
  Kept existing camera_depth/rgb/optical child frames → `/camera/*` sim topics unchanged.
- New `rpi_camera_link` (`ubot_rpi_camera.urdf.xacro`): (0.163777, 0, 0.032609), pitch +0.2188 rad (12.54° down),
  plus `rpi_camera_optical_frame`. No Gazebo sensor for it yet.
- Tilted links use visual origins = inverse of joint transform (verified numerically, <0.01 mm).
- Masses from CAD (PLA solid, no battery/Pi): base 1.7576 kg, wheels 0.0442, lidar 0.0428, OAK-D 0.061.
- Ultrasonics commented out in `ubot_robot.urdf.xacro` (not on this body).
- Fixed xacro global-property clash: base COM props renamed `base_com_*`, IMU `imu_com_*`
  (previously base_link silently used the IMU's COM).
- `ubot_bringup/config/sim_ubot_controllers.yaml`: wheel_separation 0.2264, wheel_radius 0.0325.
- Backups of the previous description/config: `../_backups/` (outside the workspace).

The fusion2urdf (ROS 1) export in `../Inspection_Robot_Fixed_description/` has a constant
joint/COM offset bug (-16.4, +3.4, +19.7 mm) — do NOT use its joint values; meshes are fine.

## Deliberately NOT changed — pending
1. Real-robot wheel geometry still 0.264204 / 0.033 in `ubot_controllers.yaml`,
   `ubot_ros2_control.xacro` AND ESP32 `src/Ros-esp32_bridge/diff_controller.h`.
   Recalibrate on the new body (1 m straight / 360° spin test) and update all three together.
2. IMU pose (`ubot_imu.urdf.xacro`) still old placeholder (-0.09, -0.04645, 0.18) — floats above
   the new roof; needs measuring.
3. Base mass is CAD-only; set real weight if sim dynamics matter.
4. Optional: Gazebo camera sensor + bridge for `rpi_camera_link`.

## Simulation world (2026-09-12)
- `ubot_bringup/worlds/apartment.sdf` — indoor corridor/house world, `<world name='apartment'>`.
  Building = Fuel "Apartment" (OpenRobotics, CC-BY 4.0) vendored to `ubot_bringup/models/apartment`
  (Fuel copy alone renders untextured: its DAE names textures by bare filename, so the vendored
  copy duplicates the .jpgs next to the .dae). Two halls ~7 x 20 m, walls slice cleanly at z=0.3.
- `sim.launch.py` now takes `world` (default apartment), `spawn_x/y/z/yaw` (default 5.0/0/0.1/1.5708,
  an open spot with 4 m clearance facing down the right-hand hall) and appends
  `GZ_SIM_RESOURCE_PATH=<share>/ubot_bringup/models` so `model://apartment` resolves.
  `-world` for the spawner comes from the `world` arg — world FILE name must equal `<world name>`.
- `ubot_bringup/CMakeLists.txt` now installs `models` too.
- Verified headless: world loads clean, robot spawns, /scan returns 720 beams (min 3.22 m).
- Old `basic.sdf` still works: `ros2 launch ubot_bringup sim.launch.py world:=basic spawn_x:=0.0 spawn_yaw:=0.0`.

## Skid-steer sim fix (2026-09-12)
Symptom: robot drove fine in Gazebo but in RViz spun/drifted "to the right".
Cause: `ubot_gazebo.urdf.xacro` wheel_gazebo had mu1=mu2=1.1. A 4-wheel skid-steer can only
turn by scrubbing tyres sideways; with equal lateral grip the wheels STALLED (measured: wheel
velocity 0, Gazebo yaw change 0.00 deg) while diff_drive_controller kept integrating the
commanded rotation -> phantom yaw in TF only.
- Fix 1: `mu2` 1.1 -> 0.05 (mu1 stays 1.0, fdir1 1 0 0 = rolling direction).
- Fix 2: sim `wheel_separation` must be the EFFECTIVE value, not geometric (0.2264 would
  over-report rotation by ~15%). Sim now uses the SAME 0.264204 as the real robot, with
  mu2 tuned to 0.12 so that value is correct in sim (verified ratio 1.001).
  mu2 -> effective separation (noisy, ~8% per-trial spread): 0.05 -> 0.261, 0.12 -> 0.264,
  0.30 -> 0.274, 1.1 -> wheels lock. Retune mu2 if the real robot is recalibrated.
- Verified after fix: straight 1.620 m truth == 1.620 m odom, 0 sideways, 0 yaw;
  spin ratio actual/commanded = 1.007.
- Test scripts used live in the scratch dir (run_drive.sh / calib3.py pattern): spawn robot
  headless, publish TwistStamped to /diff_drive_controller/cmd_vel with SIM-TIME stamps
  (wall-clock stamps => "Velocity command timed out. Braking."), compare `gz model -m ubot -p`
  ground truth against /diff_drive_controller/odom.

## Drive train mirrors the real robot (2026-09-12)
Real: only the FRONT motors are commanded and only they have encoders; the ESP32 mirrors each
front command to the rear motor on the same side. The sim now matches exactly:
- `ubot_wheel.urdf.xacro`: rear joints carry `<mimic joint="front_*_wheel_joint" multiplier="1.0"/>`.
  gz_ros2_control logs "is mimicking joint ... with multiplier: 1" and copies the command.
  The `[Err] Physics.cc ... does not support mimic constraints` line is EXPECTED and harmless:
  DART makes no physical constraint, the mirroring happens at the command level (like the ESP32).
- `ubot_ros2_control.xacro`: rear joints are state-only in sim AND real (a mimic joint must not
  expose a command interface).
- `sim_ubot_controllers.yaml`: left/right wheel names are front joints only, same as real.
- Verified: all 4 wheels turn (+-2.4 rad/s) from a front-only command; straight 1.584 m truth ==
  1.584 m odom; spin ratio 1.001.
- NOT tested on real hardware: UbotHardware may need a look now that rear joints are URDF mimic
  joints (they should still be state-only, so it is expected to be fine).

## Real-robot drivetrain findings (2026-09-12)
Firmware bug FIXED and flashed: the ESP32 'm' handler zeroed the PID integral on every
message, and ros2_control streams 'm' at 30 Hz -> the loop was P-only, every wheel ~20-25%
below target for ever (output == Kp*error exactly; reported integral 0.30 == one cycle).
- Fix in src/Ros-esp32_bridge: clear integral only on start-from-rest or direction flip;
  conditional integration (anti-windup); integral clamped to 255/Ki.
- After: straight tracking error 9.34 -> 0.52 RPM; free spin 68.8 -> 88.5 RPM (target 90).
  Kp 5 / Ki 9 track cleanly - no gain changes needed.
- NOT changed, flagged: the 10 s runaway timer never fires (cmdStartTime refreshed every
  'm'); min_pwm is passed to doPID() and never used.
- Firmware lives ONLY on the laptop (two copies, both synced: Horizon/... and
  Documents/Chibueze/uni-bot/uni-bot/...). It is not in git or on the Pi.

Spin stall = TORQUE LIMIT, not a fault. Raw-PWM matrix ('o' command) showed every motor
healthy both directions, alone and together (66-76 rpm), both-forward and both-reverse fine.
Only spins collapse: same PWM gives 16-26 rpm because the tyres must scrub. Marginal system
-> asymmetric stalls (CCW stalled the left, CW crawled on both).
- Left needs more PWM for the same speed as load grows: 4% (raised) -> 12% (straight) ->
  36% (loaded spin). Motors match unloaded, so suspect weight distribution, not drag.
- wheel_separation 0.805 is INVALID (measured with the left stalled: the robot pivoted
  around it, odometry integrated the right wheel).
- angular.z.max_velocity was 0.8 = fiction; measured limit ~0.3 rad/s (verified 0.3: 0%
  saturation, wheels within 0.6%). Now capped at 0.3 in ubot_controllers.yaml.
- Calibrate with ARCS not in-place spins:
    python3 ~/calibrate_drive.py arc --target 180 --speed 0.3 --linear 0.12
  then apply --kind spin --measured <real_deg> --odom <odom_deg>.
- Current real values: wheel_radius 0.032340 (1 m test: odom 1.000 vs tape 0.980),
  wheel_separation 0.805 PENDING replacement from the arc test.

Robot-side diagnostic tools (on the Pi, ~/): calibrate_drive.py (straight|spin|arc|apply),
wheel_diag.py (target vs actual RPM, PWM, saturation, integral - needs diag_publish_rate>0,
now 3), esp_raw_test.py and esp_dir_test.py (raw PWM via /dev/esp32, bypasses ros2_control;
stop real_robot.launch.py first).

## ubot_mono_nav merged in (2026-09-14)
Camera-only (lidarless) navigation package, authored in the OTHER workspace copy
(Documents/Chibueze/uni-bot/uni-bot) and ported here, which is the copy WSL builds.
The two copies had diverged: that one predates ALL the CAD body work, so a straight copy
would have (a) reverted mu2 to 1.1 and re-broken the wheel lock, (b) replaced the OAK-D
CAD mesh with the old RealSense box and referenced base_length (gone -> xacro failure),
(c) re-enabled ultrasonics and dropped rpi_camera_link. Ported the real improvements only:
- `ubot_mono_nav/` copied wholesale (7 modules, 4 launch files, 2 configs).
- `ubot_camera.urdf.xacro`: added `pitch` param (POSITIVE = UP) on top of the CAD 0.0714
  rad down tilt -> net tilt = 0.0714 - pitch. The mesh visual origin is now computed
  symbolically (cos/sin) so it lands correctly at ANY tilt - verified at pitch 0 and 0.059.
  Also took the frame-semantics fix: camera_optical_frame and camera_rgb_frame are now
  REAL optical frames parented to camera_link; camera_depth_frame stays body-aligned.
- `ubot_robot.urdf.xacro`: `camera_pitch` arg wired through to the camera macro.
- `ubot_gazebo.urdf.xacro` + `ubot_camera_gazebo.urdf.xacro`: gz_frame_id -> camera_optical_frame.
- `sim_mono.launch.py`: was hard-coded to basic.sdf and `-world sensors`; now takes the same
  world/spawn_x/y/z/yaw args as sim.launch.py (default apartment, 5.0/0/0.1/1.5708) and
  appends GZ_SIM_RESOURCE_PATH for model://apartment.
Verified in WSL: 3 packages build; scan_geometry --selftest 10/10; xacro+check_urdf pass at
pitch 0 AND 0.059; depth_to_scan and scan_watchdog start and log correctly.
torch/transformers/depthai are lazy imports - only mono_depth_node (laptop/GPU) and
oak_rgb_node (Pi) need them; none are installed in WSL.

Camera geometry RESOLVED (2026-09-14): the CAD is right - OAK-D lens centre 0.152 m above
the floor, 0.168 m forward, tilted 4.09 deg DOWN; the Pi cam is the lower one at 0.099 m.
The 0.186 m / 3.4 deg UP figures were stale (written when the URDF still said 0 pitch).
Fixed everywhere:
- `real_robot_mono.launch.py`: camera_pitch default 0.059 -> **0.0**. This mattered: with the
  CAD tilt now live, 0.059 would have cancelled most of the real tilt (net 0.71 deg instead
  of 4.09 deg down). camera_pitch is a CORRECTION on top of the URDF, not the tilt itself.
- `mono_perception.yaml` + `depth_to_scan_node.py` fallback: x 0.168, z 0.152,
  fallback_pitch_up_deg -4.09 (negative because the camera looks DOWN).
- Verified the fallback and the TF/URDF transform now agree: rotation diff 1.6e-5, translation
  0.2 mm, and a point 2 m ahead of the lens lands at the same base coords either way.
- Geometry note: looking 4.09 deg down from 0.152 m, the CENTRE image row meets the floor at
  ~2.13 m. Longer ranges come from the upper rows; the 0.05-0.60 m height band discards the
  floor either way.

OPEN, flagged not changed:
- `nav2_params_mono.yaml` still uses `robot_radius: 0.15` in both costmaps. The lidar params
  use the measured rectangular footprint (half-diagonal is really 0.219 m).
- `tf_transformations` is an exec_depend in package.xml but never imported.
- TWO diverged workspace copies is the real hazard here. Consider git + one checkout.

## Environment
- Dev/test on Windows via WSL2 Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic (gz-sim 8.15).
- WSL workspace `~/ubot_ws` with `src/ubot` symlinked to the Windows checkout.
- NOTE: `ubot_robot.urdf.xacro` includes parts via `$(find ubot_description)`, i.e. the INSTALLED
  copy — always `colcon build` after editing URDF or RViz shows the old model.

## Next steps
```bash
colcon build --packages-select ubot_description ubot_bringup
source install/setup.bash
xacro src/ubot/ubot_description/urdf/body/ubot_robot.urdf.xacro > /tmp/ubot.urdf && check_urdf /tmp/ubot.urdf
ros2 launch ubot_description display.launch.py
```
In RViz: wheels should spin in place without wobble; camera frames point forward, slightly down.
Then try `ros2 launch ubot_bringup sim.launch.py`.
