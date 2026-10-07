#!/usr/bin/env bash
# Map a world with SLAM, then measure every course's shortest path in it.
#
#   ./map_world.sh parking_garage
#   ./map_world.sh apartment
#
# The robot is driven to every SPAWN and every GOAL of every course in that
# world, taken from courses.yaml - so coverage of the things being measured is
# guaranteed by construction rather than by a hand-written waypoint list that
# drifts out of date when a course is added.
#
# The map frame's origin is the pose this pass STARTS from, which is why that
# pose is passed on to measure_courses.py. Get it wrong and every course reports
# "no path" while the map is perfectly good.
set -uo pipefail
cd "$(dirname "$0")"
source ./lib.sh
ros_env

WORLD="${1:-apartment}"
# Everything this script prints also goes to a file. When the pass failed,
# the only output that survived was the last command's, and the stages
# before it were the ones worth reading.
PASS_LOG="${PASS_LOG:-/tmp/map_world_${WORLD}.out}"
exec > >(tee "$PASS_LOG") 2>&1
echo "(full log: $PASS_LOG)"
MAPS="${MAPS:-$WS/maps}"
mkdir -p "$MAPS"

# World file, <world name> and the mapping start pose all come from courses.yaml.
read -r WORLD_FILE WORLD_NAME SX SY SZ SYAW < <(python3 - "$WORLD" <<'PY'
import sys, yaml
from pathlib import Path
try:
    from ament_index_python.packages import get_package_share_directory
    cfg = Path(get_package_share_directory("ubot_eval")) / "config"
except Exception:
    cfg = Path(__file__).resolve().parent.parent / "config"
doc = yaml.safe_load((cfg / "courses.yaml").read_text())
w = doc["worlds"][sys.argv[1]]
# Map from the first course's spawn: it is a pose the robot is known to fit in.
first = next(iter(w["courses"].values()))
s = first["spawn"]
print(w["world"], w["world_name"], s["x"], s["y"], s.get("z", 0.1), s["yaw"])
PY
)
echo "world=$WORLD file=$WORLD_FILE name=$WORLD_NAME spawn=($SX, $SY, $SYAW)"

# DRIVE WITH THE CAMPAIGN'S nav2 CONFIG, NOT sim_nav2_params.yaml. SLAM only
# needs the robot to MOVE; it does not need nav2 to plan in the map frame.
# The map-framed file never received the fixes that made navigation reliable
# (BT action timeouts, costmap timeout, goal tolerances, a planning window
# wider than 10 m). With it the apartment pass reached 1 of 4 waypoints -
# "Timed out while waiting for action server to acknowledge goal request for
# follow_path". The odom-framed campaign config reached 5 of 5. Goals go out
# in `odom`, which starts at the spawn pose exactly as the SLAM map does.
NAV2_PARAMS="$WS/install/ubot_bringup/share/ubot_bringup/config/nav2_params_lidar_odom.yaml"

kill_stack || exit 1
nohup ros2 launch ubot_bringup sim.launch.py \
    world:="$WORLD_FILE" world_name:="$WORLD_NAME" \
    spawn_x:="$SX" spawn_y:="$SY" spawn_z:="$SZ" spawn_yaw:="$SYAW" \
    use_slam:=true use_nav2:=true use_rviz:=false headless:=true \
    params_file:="$NAV2_PARAMS" \
    ${UBOT_HEADLESS_RENDERING:+headless_rendering:=true} \
    > "/tmp/map_${WORLD}.log" 2>&1 &
echo "sim pid $!"
wait_bringup "/tmp/map_${WORLD}.log" 60 || exit 1

python3 ./drive_courses.py --world "$WORLD" --origin "$SX" "$SY" "$SYAW" \
    --frame odom --save-start "$MAPS/$WORLD.start.json" || exit 1

echo "=== saving the map ==="
# save_map_timeout: map_saver_cli defaults to ~2 s, and slam_toolbox only
# republishes /map on its update interval, so the saver routinely gives up with
# "Failed to spin map subscription" on a map that exists perfectly well. The
# apartment pass got lucky; the garage did not.
#
# The comments sit ABOVE the command deliberately: a comment line between a
# trailing backslash and its continuation ENDS the continuation, and the next
# line then runs as its own command - "--ros-args: command not found", which is
# exactly what happened here, AFTER the map had already been saved.
ros2 run nav2_map_server map_saver_cli -f "$MAPS/$WORLD" \
    --ros-args -p use_sim_time:=true -p save_map_timeout:=30.0 || exit 1
ls -l "$MAPS/$WORLD".{yaml,pgm}

kill_stack || true

echo
python3 ./measure_courses.py --world "$WORLD" \
    --map "$MAPS/$WORLD.yaml" --origin "$SX" "$SY" "$SYAW" \
    --start-json "$MAPS/$WORLD.start.json"
