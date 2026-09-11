# Action Reference

## No custom actions defined

This workspace defines **no custom ROS 2 action interface types** (`.action` files). No custom action servers or action clients are implemented in any of the ubot packages (`ubot_bringup`, `ubot_control`, `ubot_description`, `ubot_debugger`).

All action-based navigation functionality is provided by standard Nav2 action servers. These are available when Nav2 is running alongside the bringup launch.

---

## Standard Nav2 actions (configured for this robot)

The following Nav2 action servers are configured in `nav2_params.yaml` and are available during a Nav2 session. The Nav2 configuration is documented in full in [configuration/nav2_params.md](../configuration/nav2_params.md). For action interface definitions, see the [Nav2 documentation](https://docs.nav2.org/) and the `nav2_msgs` package.

### Navigation actions (`bt_navigator`)

#### `/navigate_to_pose`

| Field | Value |
|---|---|
| Action type | `nav2_msgs/action/NavigateToPose` |
| Server node | `bt_navigator` |
| Configured navigator | `navigate_to_pose` |

Sends the robot to a single goal pose specified in the `map` frame. The BT navigator runs a behaviour tree that calls the planner server (NavFn/A*) to compute a path and the controller server (DWB local planner) to follow it. On completion, the robot is within `xy_goal_tolerance: 0.15 m` and `yaw_goal_tolerance: 0.15 rad` of the goal.

**Example call:**

```bash
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5, z: 0.0}, \
  orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}}"
```

#### `/navigate_through_poses`

| Field | Value |
|---|---|
| Action type | `nav2_msgs/action/NavigateThroughPoses` |
| Server node | `bt_navigator` |
| Configured navigator | `navigate_through_poses` |

Sends the robot through a sequence of intermediate waypoints before reaching a final goal. Uses the same planner and controller as `navigate_to_pose`.

---

### Path following action (`controller_server`)

#### `/follow_path`

| Field | Value |
|---|---|
| Action type | `nav2_msgs/action/FollowPath` |
| Server node | `controller_server` |
| Controller plugin | `dwb_core::DWBLocalPlanner` |

Follows a pre-computed path. This is called internally by the BT navigator — you would typically not call it directly. The DWB controller runs at 20 Hz (`controller_frequency: 20.0`). Key DWB parameters: `max_vel_x: 0.15 m/s`, `max_vel_theta: 0.6 rad/s`, `sim_time: 1.7 s`.

---

### Behaviour / recovery actions (`behavior_server`)

The following recovery actions are all served by the `behavior_server` node and are triggered automatically by Nav2 behaviour trees when navigation fails or is blocked. They can also be called directly for testing.

| Action | Action Type | Description |
|---|---|---|
| `/spin` | `nav2_msgs/action/Spin` | Rotate the robot in-place by a specified angle. Useful for recovering from a poor initial heading estimate. |
| `/backup` | `nav2_msgs/action/BackUp` | Drive the robot backwards by a specified distance and speed. Parameters: `acceleration_limit: 2.5`, `minimum_speed: 0.10`. |
| `/drive_on_heading` | `nav2_msgs/action/DriveOnHeading` | Drive the robot forward on its current heading by a specified distance. Same acceleration/speed parameters as backup. |
| `/assisted_teleop` | `nav2_msgs/action/AssistedTeleop` | Pass teleop commands through Nav2's collision avoidance layer — useful for manual recovery while keeping safety constraints active. |
| `/wait` | `nav2_msgs/action/Wait` | Wait in place for a specified duration before retrying navigation. |

**Behavior server frame configuration:** `local_frame: odom`, `global_frame: map`.

---

### Smoothing action (`smoother_server`)

| Action | Action Type | Plugin |
|---|---|---|
| `/smooth_path` | `nav2_msgs/action/SmoothPath` | `nav2_smoother::SimpleSmoother` |

Post-processes a computed path to reduce sharp direction changes. Configuration: `tolerance: 1e-10`, `max_its: 1000`, `do_refinement: True`. Called internally by the BT navigator.

---

## Further reading

- [Nav2 action interface definitions](https://docs.nav2.org/configuration/packages/configuring-bt-navigator.html) — full action `.action` file specifications.
- [Nav2 behaviour trees](https://docs.nav2.org/behavior_trees/index.html) — how the BT navigator chains the above actions.
- [ubot Nav2 configuration](../configuration/nav2_params.md) — robot-specific parameter values.
