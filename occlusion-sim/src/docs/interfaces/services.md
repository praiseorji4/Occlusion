# Service Reference

This page documents ROS 2 services available in the ubot workspace, grouped by origin: custom services defined in this workspace, standard ros2_control services, and standard Nav2 services.

No custom service interface types (`.srv` files) are defined in this workspace. The one custom-behaviour service (`/bno055/calibration_request`) uses the standard `std_srvs/Trigger` type.

---

## Custom workspace services

### `/bno055/calibration_request`

| Field | Value |
|---|---|
| Service type | `std_srvs/Trigger` (via `example_interfaces/srv/Trigger` import in source) |
| Server node | `bno055_node` |
| Source file | `src/bno055/bno055/sensor/SensorService.py`, `calibration_request_callback()` |
| Availability | **Only when `bno055_node` is running.** Not available on a default `real_robot.launch.py` launch — bno055_node is currently commented out. |

**What it does:**

When called, the service handler performs the following sequence synchronously:

1. Switches the BNO055 into **config mode** (`OPERATION_MODE_CONFIG`) — sensor fusion pauses during this window.
2. Waits 25 ms for the mode transition to complete.
3. Reads all calibration offset registers: accelerometer offsets (X/Y/Z) and radius, magnetometer offsets (X/Y/Z) and radius, gyroscope offsets (X/Y/Z).
4. Switches the BNO055 back to **NDOF mode** (`OPERATION_MODE_NDOF`) — full sensor fusion resumes.
5. Returns `success: True` with `message` set to a Python string representation of the calibration data dictionary.

The `response.message` field contains a dict-like string with keys: `accel_offset` (x/y/z), `accel_radius`, `mag_offset` (x/y/z), `mag_radius`, `gyro_offset` (x/y/z).

**How to call it:**

```bash
ros2 service call /bno055/calibration_request std_srvs/srv/Trigger '{}'
```

**When to use it:**

Call this service once per session before starting a mapping or localisation run, to verify that the BNO055 has reached acceptable calibration levels. Cross-reference the returned offsets against the `/bno055/calib_status` topic (all fields should read 3 for a fully calibrated state). You can optionally copy the returned offsets into `bno055_params.yaml` under the `offset_acc`, `offset_mag`, `offset_gyr` fields and set `set_offsets: true` so the calibration is reloaded automatically on the next startup.

**Important note:** The service causes a brief interruption to sensor fusion (config mode window). Do not call it while active navigation is in progress. The IMU will resume normal output immediately after the call returns.

---

## Standard ros2_control services

The following services are provided by `controller_manager` and the spawned controllers. They are standard ros2_control services — no custom implementation exists in this workspace. Consult the [ros2_control documentation](https://control.ros.org/rolling/doc/ros2_control/controller_manager/doc/userdoc.html) for full interface definitions and usage.

| Service | Type | Provider | Purpose |
|---|---|---|---|
| `/controller_manager/configure_controller` | `controller_manager_msgs/srv/ConfigureController` | controller_manager | Move a controller to the configured (inactive) state |
| `/controller_manager/list_controllers` | `controller_manager_msgs/srv/ListControllers` | controller_manager | List all loaded controllers and their states |
| `/controller_manager/list_controller_types` | `controller_manager_msgs/srv/ListControllerTypes` | controller_manager | List available controller plugin types |
| `/controller_manager/list_hardware_components` | `controller_manager_msgs/srv/ListHardwareComponents` | controller_manager | List hardware interface components and their state |
| `/controller_manager/list_hardware_interfaces` | `controller_manager_msgs/srv/ListHardwareInterfaces` | controller_manager | List all command and state interfaces |
| `/controller_manager/load_controller` | `controller_manager_msgs/srv/LoadController` | controller_manager | Load a controller plugin by name |
| `/controller_manager/reload_controller_libraries` | `controller_manager_msgs/srv/ReloadControllerLibraries` | controller_manager | Reload pluginlib controller libraries |
| `/controller_manager/set_hardware_component_state` | `controller_manager_msgs/srv/SetHardwareComponentState` | controller_manager | Transition a hardware component between lifecycle states |
| `/controller_manager/switch_controller` | `controller_manager_msgs/srv/SwitchController` | controller_manager | Activate or deactivate one or more controllers atomically |
| `/controller_manager/unload_controller` | `controller_manager_msgs/srv/UnloadController` | controller_manager | Unload a named controller |
| `/diff_drive_controller/describe_parameters` | `rcl_interfaces/srv/DescribeParameters` | diff_drive_controller | Describe all controller parameters |
| `/diff_drive_controller/get_parameter_types` | `rcl_interfaces/srv/GetParameterTypes` | diff_drive_controller | Get parameter types by name |
| `/diff_drive_controller/get_parameters` | `rcl_interfaces/srv/GetParameters` | diff_drive_controller | Read current parameter values |
| `/diff_drive_controller/list_parameters` | `rcl_interfaces/srv/ListParameters` | diff_drive_controller | List all parameter names |
| `/diff_drive_controller/set_parameters` | `rcl_interfaces/srv/SetParameters` | diff_drive_controller | Set one or more parameters at runtime |
| `/diff_drive_controller/set_parameters_atomically` | `rcl_interfaces/srv/SetParametersAtomically` | diff_drive_controller | Set multiple parameters in one atomic transaction |

**Commonly useful calls:**

```bash
# Check which controllers are active
ros2 service call /controller_manager/list_controllers \
  controller_manager_msgs/srv/ListControllers '{}'

# Check the hardware interface state (useful when UbotHardware fails to configure)
ros2 service call /controller_manager/list_hardware_components \
  controller_manager_msgs/srv/ListHardwareComponents '{}'

# Deactivate diff_drive_controller (e.g. to safely test raw motor commands)
ros2 service call /controller_manager/switch_controller \
  controller_manager_msgs/srv/SwitchController \
  '{deactivate_controllers: [diff_drive_controller], strictness: 1}'
```

---

## Standard Nav2 services

Nav2 exposes a number of lifecycle management and parameter services when its nodes are running. These are all standard Nav2 services — consult the [Nav2 documentation](https://docs.nav2.org/) for full descriptions.

| Service | Provider node | Purpose |
|---|---|---|
| `/bt_navigator/get_parameters` | bt_navigator | Read BT navigator parameters |
| `/controller_server/get_parameters` | controller_server | Read controller (DWB) parameters |
| `/planner_server/get_parameters` | planner_server | Read planner (NavFn) parameters |
| `/map_server/load_map` | map_server | Load a saved map at runtime |
| `/map_server/save_map` | map_saver | Save the current map to disk |
| `/lifecycle_manager_navigation/manage_nodes` | lifecycle_manager | Start/stop Nav2 node group |
| `/slam_toolbox/save_map` | slam_toolbox | Save SLAM map to `.pgm`/`.yaml` |
| `/slam_toolbox/serialize_map` | slam_toolbox | Serialize map to `.data`/`.posegraph` |

Nav2 actions (navigate_to_pose, follow_path, etc.) are documented separately in [actions.md](actions.md).

---

## Summary: No custom action servers or action clients

No custom ROS 2 action interfaces are defined in this workspace. See [actions.md](actions.md) for the standard Nav2 actions that are configured and available when Nav2 is running.
