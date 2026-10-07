#!/usr/bin/env bash
# Build the occupancy grid that the SPL denominator is measured on.
#
#   ./mapping_pass.sh                    # apartment
#
# Drives the course area with SLAM on, then saves map.pgm/map.yaml. Unmapped
# cells count as BLOCKED in shortest_path.py, so thin coverage shows up as "no
# path" rather than as a quietly wrong number - drive more, do not relax the
# assumption.
#
# Afterwards:
#   python3 -m ubot_eval.shortest_path --map ~/occl_ws/maps/apartment.yaml \
#       --start 5.0 0.0 --goal 5.0 3.0
set -uo pipefail
cd "$(dirname "$0")"
source ./lib.sh

WORLD="${WORLD:-apartment}"
MAPS="${MAPS:-$WS/maps}"
mkdir -p "$MAPS"

ros_env
kill_stack || exit 1

nohup ros2 launch ubot_bringup sim.launch.py \
    world:="$WORLD" world_name:="$WORLD" "${APARTMENT_SPAWN[@]}" \
    use_slam:=true use_nav2:=true use_rviz:=false headless:=true \
    > /tmp/map_sim.log 2>&1 &
echo "sim pid $!"
wait_bringup /tmp/map_sim.log 48 || exit 1

python3 - <<"PY"
import math, rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

# Covers both halls and the corner, so hall_3m, hall_10m and corner_turn all
# sit inside mapped space.
WAYPOINTS = [(5.0, 3.0, 1.5708), (5.0, 8.0, 1.5708), (5.0, 11.0, 1.5708),
             (3.0, 9.0, 3.1416), (1.5, 8.0, 3.1416), (1.5, 4.0, -1.5708),
             (3.0, 1.0, -1.5708), (5.0, 0.5, -1.5708)]

rclpy.init()
node = Node("mapping_drive")
client = ActionClient(node, NavigateToPose, "navigate_to_pose")
if not client.wait_for_server(timeout_sec=90.0):
    raise SystemExit("navigate_to_pose never appeared")

for i, (x, y, yaw) in enumerate(WAYPOINTS, 1):
    goal = NavigateToPose.Goal()
    p = PoseStamped()
    p.header.frame_id = "map"          # SLAM publishes map->odom in this pass
    p.header.stamp = node.get_clock().now().to_msg()
    p.pose.position.x, p.pose.position.y = x, y
    p.pose.orientation.z = math.sin(yaw / 2.0)
    p.pose.orientation.w = math.cos(yaw / 2.0)
    goal.pose = p
    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=30.0)
    handle = send.result()
    if handle is None or not handle.accepted:
        print(f"  waypoint {i} ({x}, {y}) NOT ACCEPTED - coverage will be thinner")
        continue
    res = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res, timeout_sec=180.0)
    print(f"  waypoint {i} ({x}, {y}) done")
node.destroy_node()
rclpy.shutdown()
PY

echo "=== saving the map ==="
ros2 run nav2_map_server map_saver_cli -f "$MAPS/$WORLD" \
    --ros-args -p use_sim_time:=true
ls -l "$MAPS"
