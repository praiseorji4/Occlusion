#!/usr/bin/env bash
# Shared helpers for the trial scripts. Source it; do not run it.
#
#   source "$(dirname "$0")/lib.sh"
#
# WS defaults to the WSL build workspace and can be overridden:
#   WS=/ws ./run_episode.sh          # inside the container
WS="${WS:-$HOME/occl_ws}"

# ---------------------------------------------------------------------------
# kill_stack - remove EVERY node of a previous run, then PROVE it.
#
# `pkill -f nav2` is not enough: opennav_docking's binary path contains no
# "nav2", so it survived, stayed ACTIVE, and the next lifecycle_manager got
#   Unable to start transition 1 from current state active
# and aborted the whole bringup - which reads like a params or frame fault and
# is neither. Two Gazebo servers once ran at the same time for the same reason,
# fighting over /clock and flooding nav2 with "jump back in time".
#
# Returns non-zero if anything survived, so callers can refuse to launch onto a
# dirty process table rather than measure a mixture of two runs.
# ---------------------------------------------------------------------------
kill_stack() {
    for pat in 'gz sim' 'ruby.*gz' parameter_bridge ros_gz nav2 opennav \
               static_transform_publisher lifecycle_manager twist_stamper \
               controller_manager robot_state_publisher ekf_node slam_toolbox \
               'ros2 launch' rviz2 spawner component_container map_saver; do
        pkill -9 -f "$pat" 2>/dev/null
    done
    sleep 6
    local left
    left=$(pgrep -af 'gz sim|opennav|nav2_controller|static_transform_publisher' \
           2>/dev/null | grep -v 'pgrep\|bash ' || true)
    if [ -n "$left" ]; then
        echo "WARNING: survivors after cleanup:" >&2
        echo "$left" >&2
        return 1
    fi
    echo "stack clean"
}

# ---------------------------------------------------------------------------
# wait_bringup <logfile> [max_polls]
#
# Waits for nav2's lifecycle_manager to report every node active. Returns 1 on
# an explicit bringup failure and prints why, instead of timing out silently.
# ---------------------------------------------------------------------------
wait_bringup() {
    local log="$1" max="${2:-40}" n=0
    while [ "$n" -lt "$max" ]; do
        if grep -aq 'Managed nodes are active' "$log" 2>/dev/null; then
            echo "bringup OK after ~$((n * 5))s"
            return 0
        fi
        if grep -aq 'Failed to bring up' "$log" 2>/dev/null; then
            echo "BRINGUP FAILED:" >&2
            grep -aiE 'Failed to change state|Unable to start transition' "$log" \
                | head -5 >&2
            return 1
        fi
        sleep 5
        n=$((n + 1))
    done
    echo "bringup TIMED OUT after $((max * 5))s" >&2
    tail -15 "$log" >&2
    return 1
}

# ---------------------------------------------------------------------------
# ros_env - source ROS and the workspace. Both need `set +u` relief: their
# generated setup scripts read AMENT_TRACE_SETUP_FILES and COLCON_TRACE without
# defaulting them, which kills any script using `set -u`.
# ---------------------------------------------------------------------------
ros_env() {
    set +u
    source /opt/ros/jazzy/setup.bash
    [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
    set -u
}

# The apartment spawn pose. THE SPAWN POSE BELONGS TO THE WORLD: sim.launch.py
# defaults to Sonoma's pit lane (272, -136, 3.45), so passing world:=apartment
# alone puts the robot ~300 m outside the building where the LiDAR honestly
# reads .inf on every beam and the camera sees nothing. No error is printed.
APARTMENT_SPAWN=(spawn_x:=5.0 spawn_y:=0.0 spawn_z:=0.1 spawn_yaw:=1.5708)
