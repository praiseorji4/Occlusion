# Tuning

This guide covers parameter tuning for the Nav2 navigation stack and the differential drive controller. All values documented here are the **current state** as read directly from source files — inline changelog comments in the YAML files record what was changed from the original defaults and why.

Configuration files referenced:
- Nav2: `/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/nav2_params.yaml`
- Controller: `/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/ubot_controllers.yaml`
- SLAM: `/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/mapper_params_online_async.yaml`

---

## DWB Local Planner Critic Tuning

The `controller_server` uses `dwb_core::DWBLocalPlanner` with the following critics active:

```
RotateToGoal, Oscillation, BaseObstacle, GoalAlign, PathAlign, PathDist, GoalDist
```

### Current Critic Scales

| Critic | Current Scale | Notes |
|---|---|---|
| `RotateToGoal` | 32.0 | High — strongly prefers rotating toward goal at close range |
| `PathAlign` | 32.0 | High — prefers trajectories aligned with the planned path |
| `PathDist` | 32.0 | High — prefers trajectories that stay close to the planned path |
| `GoalAlign` | 24.0 | Medium-high — prefers heading toward the goal |
| `GoalDist` | 24.0 | Medium-high — prefers moving toward the goal position |
| `Oscillation` | default | Penalises oscillation between forward/backward motion |
| `BaseObstacle` | **0.5** | Low — penalises proximity to obstacles |

**History**: `BaseObstacle.scale` was originally `0.02`, making it approximately 1600x weaker than the path critics. At that scale the obstacle penalty was effectively ignored and the robot could plan paths that came dangerously close to obstacles while still scoring well. It was raised to `0.5`.

If the robot still cuts corners too aggressively near obstacles, consider raising `BaseObstacle.scale` further (try 1.0–2.0). If the robot becomes overly cautious and avoids too-large areas, reduce it.

### Trajectory Rollout Parameters

| Parameter | Current Value | Effect |
|---|---|---|
| `sim_time` | 1.7 s | Trajectory simulation horizon. Decrease for tighter spaces (but may cause jerky motion); increase for smoother paths in open areas |
| `vx_samples` | 20 | Number of linear velocity samples. Increase for better path coverage; decrease if planning is slow |
| `vtheta_samples` | 20 | Number of angular velocity samples. Same trade-off as `vx_samples` |
| `transform_tolerance` | 0.2 s | TF age tolerance. Increase if you see `Could not get robot pose` errors |

---

## Inflation Radius Tuning

The inflation layer pads obstacles in the costmap. The `inflation_radius` must be at least equal to `robot_radius` to prevent the planner from routing the robot through spaces where it would collide.

| Parameter | Current Value | Original | Notes |
|---|---|---|---|
| `robot_radius` | 0.15 m | 0.15 m | Actual robot half-width |
| Local `inflation_radius` | **0.30 m** | 0.05 m | Was less than robot_radius — unsafe; now 2× robot_radius |
| Global `inflation_radius` | **0.30 m** | 0.20 m | Increased to match local costmap |
| Local `cost_scaling_factor` | 3.0 | — | How steeply cost falls off from obstacle |
| Global `cost_scaling_factor` | 1.5 | — | Lower = gentler cost gradient = planner stays further away |

**Rule**: `inflation_radius >= robot_radius`. The current 0.30 m (2× robot_radius) gives a safety margin. In very tight environments where the robot genuinely needs to navigate through narrow gaps, you could decrease it, but never below `robot_radius` (0.15 m).

If the robot refuses to navigate through corridors it should fit through, the inflation radius may be too large. Decrease it incrementally (e.g. 0.25 m, 0.20 m) and test.

---

## Goal Tolerance Tuning

These parameters control how close to the goal pose the robot must get before Nav2 considers the goal achieved.

| Parameter | Current Value | Original | Notes |
|---|---|---|---|
| `xy_goal_tolerance` | **0.15 m** | 0.5 m | Distance from goal position |
| `yaw_goal_tolerance` | 0.15 rad | — | Heading error at goal |
| Planner `tolerance` | **0.25 m** | 0.5 m | How close to the goal the global planner must reach |

**History**: The planner tolerance was 0.5 m originally — at this value the robot would stop up to 0.5 m from the intended goal, which was too imprecise for most tasks. It was reduced to 0.25 m.

**Trade-off**: Tighter tolerances produce more accurate goal reaching but are harder to achieve, particularly with wheel-odometry-only localisation (no IMU fusion currently active). If the robot spins at the goal trying to reach a tight tolerance, consider loosening `yaw_goal_tolerance` first (e.g. 0.25 rad).

---

## Velocity Limits

Velocity limits appear in **two places** and should match:

### In `nav2_params.yaml` (controller_server and velocity_smoother)

```yaml
# controller_server
max_vel_x:     0.15   # m/s
min_vel_x:    -0.5
max_vel_theta: 0.6    # rad/s
max_speed_xy:  0.15   # m/s (overall speed cap)

# velocity_smoother
max_velocity:  [0.15, 0.0, 0.6]   # [linear_x, linear_y, angular_z]
min_velocity:  [-0.5, 0.0, -1.0]
max_accel:     [2.5, 0.0, 3.2]
```

### In `ubot_controllers.yaml` (diff_drive_controller)

```yaml
linear.x.max_velocity:   0.5   # note: controller-level limit, higher than Nav2 limit
angular.z.max_velocity:  2.0
```

The `nav2_params.yaml` limits are more conservative (0.15 m/s vs 0.5 m/s in the controller). Nav2 will never command faster than 0.15 m/s. The controller-level limits act as a hard safety cap if anything bypasses Nav2.

**Velocity smoother feedback**: The `velocity_smoother` uses `CLOSED_LOOP` feedback mode, reading actual odometry from `/diff_drive_controller/odom` to smooth acceleration. This was changed from `OPEN_LOOP` (which used only commanded velocities) to prevent jerk on uneven surfaces.

---

## SLAM Toolbox Tuning

Configuration: `/home/chibueze/uni-bot/src/ubot/ubot_bringup/config/mapper_params_online_async.yaml`

| Parameter | Current Value | Effect |
|---|---|---|
| `map_file_name` | `/home/chibueze/uni-bot/studio_3_serial` | **Must update for each new environment** — used for serialised save/load |
| `minimum_travel_distance` | 0.5 m | Robot must move at least 0.5 m before a new scan is used for map update. Decrease for slow-speed mapping; increase to reduce map update frequency |
| `minimum_travel_heading` | 0.5 rad | Robot must rotate at least 0.5 rad before a new scan is used. Decrease if the robot misses features while rotating slowly |
| `map_update_interval` | 5.0 s | How often the map image is regenerated |
| `resolution` | 0.05 m | Map cell size. Smaller = more detail but larger map files |
| `max_laser_range` | 12.0 m | Maximum range used from LiDAR scans (LD19 max is ~12 m) |
| `transform_publish_period` | 0.02 s (50 Hz) | Rate of `map` → `odom` TF. Match to or higher than `diff_drive_controller.publish_rate` (30 Hz) |

---

## A* vs Dijkstra (Global Planner)

The global planner (`nav2_navfn_planner::NavfnPlanner`) is configured with:

```yaml
use_astar: true   # was: false (Dijkstra)
```

A* was enabled because it is faster than Dijkstra for goal-directed planning on the costmap sizes used. Path quality is equivalent. If you observe incorrect path choices in maps with many obstacles, temporarily switch back to `use_astar: false` (Dijkstra) to rule out heuristic issues.

---

## Applying Parameter Changes

Nav2 and SLAM Toolbox parameters are read at launch time. To apply changes:

1. Edit the YAML file.
2. Rebuild `ubot_bringup` (required to update the installed copy):
   ```bash
   cd /home/chibueze/uni-bot
   colcon build --packages-select ubot_bringup
   source install/setup.bash
   ```
3. Re-launch the affected node or the full stack.

For the `ubot_controllers.yaml` parameters (controller update rate, wheel geometry, velocity limits), the controller_manager must be restarted to pick up changes.

---

## See Also

- [Calibration](calibration.md) — wheel geometry and PID gain tuning
- [Running the Full Stack](running.md) — operational procedure
- [Troubleshooting](troubleshooting.md) — fault finding for Nav2 and planning issues
