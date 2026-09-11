# test_wheels.launch.py

**This file is empty and will crash if launched.** Do not attempt to use it without implementing it first.

Source: `/home/chibueze/uni-bot/src/ubot/ubot_bringup/launch/test_wheels.launch.py`

## Quick start

```bash
# DO NOT RUN — this file is empty and will produce a Python error:
# AttributeError: module has no attribute 'generate_launch_description'
ros2 launch ubot_bringup test_wheels.launch.py
```

## Current state

The file contains a single blank line and no Python content. There is no `generate_launch_description()` function defined. Attempting to launch it will result in a `launch` framework error because the required entry point is missing.

This is tracked as issue [#m4](../issues.md#m4).

## What a wheel test launch file would typically contain

A functional wheel test launch for a differential-drive robot like the ubot would typically include some combination of:

1. **Manual motor command publisher** — a node or script that sends known velocity commands to `/diff_drive_controller/cmd_vel` (e.g., drive forward at 0.1 m/s for 2 seconds, stop).

2. **Hardware bringup subset** — the minimum set needed to drive wheels:
   - `robot_state_publisher` (for URDF)
   - `controller_manager` with `UbotHardware`
   - `joint_state_broadcaster` spawner
   - `diff_drive_controller` spawner

3. **Diagnostic subscriber** — either RViz with `/joint_states` visible, or a direct `ros2 topic echo /joint_states` to verify encoder feedback is returning non-zero values.

4. **Sequenced test pattern** — left-wheel-only, right-wheel-only, both forward, both reverse, to isolate per-wheel issues.

A minimal implementation would look like:

```python
# EXAMPLE — not currently implemented in this file
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction

def generate_launch_description():
    # ... controller_manager, joint_state_broadcaster, diff_drive_controller ...
    # ... TimerAction to publish a test velocity after controllers are ready ...
    return LaunchDescription([...])
```

## Suggested fix

Choose one of:

**Option A — Implement it**: Add a proper `generate_launch_description()` that starts the hardware interface and optionally a test velocity publisher. Reference `real_robot.launch.py` for the controller bringup pattern, then add a timed velocity command node.

**Option B — Remove it**: If standalone wheel testing is handled manually (e.g., `ros2 topic pub /cmd_vel ...` after bringup), remove the file from `CMakeLists.txt` and delete it to eliminate the misleading entry point.

## Known issues

| Ref | Description | Severity |
|---|---|---|
| [#m4](../issues.md#m4) | `test_wheels.launch.py` is an empty file (1 blank line). Will crash if launched with `AttributeError`. | Medium |

## See also

- [real_robot.launch.py](real_robot.md) — contains the working controller bringup pattern
- [Hardware: UbotHardware interface](../packages/ubot_control.md)
- [Packages: ubot_debugger](../packages/ubot_debugger.md) — diagnostic tool for wheel RPM/PID monitoring
