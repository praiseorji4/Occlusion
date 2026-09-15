# ubot_mono_nav — navigating on the RGB camera alone

The LiDAR does two jobs on this robot. This package replaces one of them.

| job | LiDAR stack | this package |
|---|---|---|
| obstacles for the costmaps | `/scan` → `voxel_layer` / `obstacle_layer` | RGB → depth → `/scan_mono` → same layers |
| localisation and mapping | `slam_toolbox`, `map → odom` | **not replaced** — everything runs in `odom` |

So the robot navigates relative to where it started. No map, no loop closure, and
goals are in the `odom` frame. That is enough to show the camera driving the
planner, which is the point.

**Nothing in `ubot_bringup` was edited.** `real_robot.launch.py` and
`laptop_slam_nav2.launch.py` still run the LiDAR stack exactly as before, so the
two can be compared on the same course.

## The chain

```
RGB image ──> mono_depth_node ──> /mono/depth ──> depth_to_scan ──> /scan_mono ──> nav2
                (laptop)          32FC1 metres      (geometry)       LaserScan      costmaps

nav2 ──> velocity_smoother ──> collision_monitor ──> /cmd_vel_raw ──> scan_watchdog ──> /cmd_vel
```

`scan_watchdog` is last on purpose: if the camera pipeline stalls, it zeroes the
command. Without it, nav2's last order keeps executing while perception is frozen.

## Run it in simulation

```bash
colcon build --packages-select ubot_mono_nav ubot_description
source install/setup.bash
ros2 launch ubot_mono_nav sim_mono.launch.py
```

Gazebo's LiDAR is still published on `/scan`, unused by navigation — it is the
reference to grade `/scan_mono` against. In rviz, add both.

```bash
ros2 topic hz /mono/depth /scan_mono          # depth ~5 Hz, scan the same
ros2 topic echo /scan_mono --field ranges     # +inf where nothing was seen
```

## Run it on the robot

**On the Pi** (streams JPEG, drives wheels, no LiDAR):

```bash
ros2 launch ubot_mono_nav real_robot_mono.launch.py
```

`camera_pitch` is a *correction* in radians, positive up, on top of the tilt the
URDF already carries: the CamCase holds the OAK-D 4.09° down, 0.152 m off the
floor, which was checked against the robot. So the default is 0 and you only set
it if the mount is physically bent. Re-measure with `calibrate_pose.py` after
touching the mount, or every camera obstacle lands at the wrong height.

**On the laptop** (depth network, nav2, rviz):

```bash
ros2 launch ubot_mono_nav mono_nav_laptop.launch.py \
    depth_scale_json:=/path/to/depth_scale.json
```

## Calibrate the depth first

Monocular metric depth is wrong by roughly a factor of two at this viewpoint
until it is calibrated. Produce the file from the Occlusion repo:

```bash
python occlusion/eval/depth_affine.py --run runs/<clip> \
    --depth-model depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
```

It writes `<clip>/depth_scale.json`, which is the contract between the research
pipeline and the robot: no code is shared, so the two cannot drift apart. Without
it the nodes still run and say loudly that the metres are uncalibrated — fine for
looking at rviz, not for driving.

**Indoor, not outdoor.** From a camera 0.19 m off the ground the outdoor metric
head stops resolving depth past ~4 m (it reads ~16 m for everything); the indoor
head tracks stereo to within ~20% over 1–4 m. Measurements are in
`depth_backend.py`.

## What it will not do

- **Sees 65° ahead, and nothing else.** No sides, no rear. `spin` and `backup`
  are removed from the behaviour list because both move the robot blind.
- **Trusts 1–4 m.** Past that the scan reads `+inf`, so nav2 neither marks the
  space blocked nor clears it as free.
- **200–500 ms of latency** over WiFi plus inference. Keep speeds low; the
  watchdog bounds the damage when it stalls.

## Testing

Runs anywhere, no ROS and no robot:

```bash
python -m ubot_mono_nav.scan_geometry --selftest
```

Ten constructed scenes, including the two that matter: floor alone must yield no
obstacles, and an obstacle to the left must land at a positive angle.

On the robot, before driving: put something at a tape-measured 1, 2, 3 and 4 m
and compare `/scan_mono` against the tape. Then kill `mono_depth_node` while
driving and confirm the robot stops within half a second.
