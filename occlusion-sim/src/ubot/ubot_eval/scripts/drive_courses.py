#!/usr/bin/env python3
"""Drive to every course spawn and goal in a world, so SLAM sees all of it.

    python3 drive_courses.py --world parking_garage --origin 0 0 0

Goals are sent in the `map` frame, which SLAM anchors at the pose this mapping
pass started from - so world coordinates from courses.yaml are converted before
being sent. Sending them unconverted aims at a different place entirely, which
is the same mistake the episode runner used to make with odom-framed goals.

A waypoint that nav2 refuses or fails is reported and SKIPPED rather than
aborting the pass: partial coverage still produces a usable map for the courses
that were reached, and measure_courses.py will say "no path" for the rest
instead of inventing a number.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from measure_courses import load_courses, world_to_map   # noqa: E402


def record_map_start(node, out_path, spawn_world):
    """Look up where the robot actually is in SLAM's `map` frame, and save it.

    WHY. The measurement converts courses.yaml's WORLD coordinates into the
    saved map, and it used to ASSUME the map frame starts at the spawn pose
    facing the robot's heading. That held for one pass and not the next: when
    nav2 drove in `odom`, slam_toolbox's map came out aligned with the WORLD,
    90 degrees from the assumption, and every course was measured against the
    wrong cells ("85% unmapped" for a hall the robot had just driven down).
    Recording the real pose removes the assumption instead of swapping it for
    another one.
    """
    import json
    import time
    import rclpy
    from tf2_ros import Buffer, TransformListener

    buf = Buffer()
    TransformListener(buf, node)
    deadline = time.time() + 60.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        try:
            t = buf.lookup_transform("map", "base_footprint", rclpy.time.Time())
        except Exception:
            continue
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        doc = {"map_pose": [t.transform.translation.x, t.transform.translation.y, yaw],
               "world_pose": list(spawn_world)}
        Path(out_path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"  map-frame start pose {doc['map_pose']} -> {out_path}", flush=True)
        return
    print("  WARNING: no map->base_footprint within 60 s; start pose NOT recorded",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--origin", nargs=3, type=float, required=True,
                    metavar=("X", "Y", "YAW"))
    ap.add_argument("--timeout-s", type=float, default=240.0)
    # "odom" when nav2 runs the odom-framed campaign config while SLAM maps in
    # the background (see map_world.sh). Both frames start at the spawn pose,
    # so the same world->frame conversion is correct for either.
    ap.add_argument("--frame", default="map", choices=["map", "odom"])
    ap.add_argument("--save-start", default=None,
                    help="write the MAP-frame start pose here, as json")
    # Close enough for MAPPING: a goal only has to pull the robot through the
    # area. Waiting out a full timeout 0.16 m from every goal cost 240 s per
    # waypoint and ended the pass before the far hall was scanned.
    ap.add_argument("--near-m", type=float, default=0.5)
    args = ap.parse_args()
    origin = tuple(args.origin)
    args.timeout_s = min(args.timeout_s, 120.0)

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from nav2_msgs.action import NavigateToPose
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import PoseStamped

    # Visit every spawn and every goal, de-duplicated and ordered nearest-first
    # from the start so the drive does not criss-cross the world.
    points = []
    for name, spawn, goal in load_courses(args.world):
        for label, p in ((f"{name}:spawn", spawn), (f"{name}:goal", goal)):
            if not any(math.dist(p, q) < 0.3 for _, q in points):
                points.append((label, p))
    points.sort(key=lambda lp: math.dist(lp[1], (origin[0], origin[1])))

    rclpy.init()
    node = Node("drive_courses")
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    if args.save_start:
        record_map_start(node, args.save_start, origin)

    if not client.wait_for_server(timeout_sec=90.0):
        print("navigate_to_pose never appeared", file=sys.stderr)
        raise SystemExit(1)

    reached = 0
    for label, world_pt in points:
        mx, my = world_to_map(world_pt, origin)
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = args.frame
        pose.header.stamp = node.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = mx, my
        pose.pose.orientation.w = 1.0
        goal.pose = pose

        print(f"  -> {label:<22} world={world_pt}  map=({mx:.2f}, {my:.2f})",
              flush=True)
        near = {"hit": False}

        def _fb(msg, near=near):
            d = getattr(msg.feedback, "distance_remaining", None)
            if d is not None and 0.0 < d <= args.near_m:
                near["hit"] = True

        send = client.send_goal_async(goal, feedback_callback=_fb)
        rclpy.spin_until_future_complete(node, send, timeout_sec=60.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            print(f"     NOT ACCEPTED - skipping (coverage will be thinner)")
            continue
        result = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result, timeout_sec=args.timeout_s)
        # `done()` only means the action FINISHED - an aborted goal is done too.
        # Reporting that as success made a pass where the robot never left the
        # start look like four reached waypoints, and the resulting map covered
        # 8.8 x 11.25 m of a world the courses span 19 m of. Check the status.
        if not result.done():
            handle.cancel_goal_async()
            if near["hit"]:
                reached += 1
                print(f"     within {args.near_m} m - good enough for mapping")
            else:
                print("     timed out - skipping")
            continue
        status = getattr(result.result(), "status", None)
        if status == GoalStatus.STATUS_SUCCEEDED:
            reached += 1
            print("     reached")
        else:
            name = {GoalStatus.STATUS_ABORTED: "ABORTED",
                    GoalStatus.STATUS_CANCELED: "CANCELED"}.get(status, f"status {status}")
            print(f"     {name} - skipping (this area will be thin on the map)")

    node.destroy_node()
    rclpy.shutdown()
    print(f"visited {reached} of {len(points)} waypoints")
    raise SystemExit(0 if reached else 1)


if __name__ == "__main__":
    main()
