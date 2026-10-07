#!/usr/bin/env bash
# Every course's shortest path, from the saved map, ready to paste into
# courses.yaml.
#
#   ./measure_courses.sh ~/occl_ws/maps/apartment.yaml
#
# Prints "no path" for any course the mapping pass did not cover well enough.
# That is the honest answer: unmapped space is not free space, and a course
# whose reference path cannot be measured keeps shortest_path_m: null, which
# makes schema.validate refuse to score a success on it rather than record an
# SPL of zero.
set -uo pipefail
cd "$(dirname "$0")"
source ./lib.sh
ros_env

MAP="${1:-$WS/maps/apartment.yaml}"
[ -f "$MAP" ] || { echo "no map at $MAP - run mapping_pass.sh first"; exit 1; }

python3 - "$MAP" <<"PY"
import sys
from ubot_eval.shortest_path import shortest_path_m
import math

MAP = sys.argv[1]

# (course, start_xy, goal_xy) for the apartment world, from courses.yaml.
COURSES = [
    ("hall_3m",     (5.0, 0.0), (5.0, 3.0)),
    ("hall_10m",    (5.0, 0.0), (5.0, 10.0)),
    ("corner_turn", (5.0, 0.0), (1.5, 8.0)),
]

# THE SAVED MAP IS IN SLAM'S `map` FRAME, NOT THE GAZEBO WORLD FRAME.
# slam_toolbox starts its map at the robot's initial pose, so the world point
# (5, 0) with yaw pi/2 is map (0, 0) with yaw 0. courses.yaml is in WORLD
# coordinates, so querying it directly asks about points far outside the mapped
# area and every course comes back "no path" - which looks like thin coverage
# and is not.
#
# Assumes SLAM's origin is exactly the spawn pose; loop-closure drift during the
# pass makes that approximate, which is acceptable for a reference path but is
# an assumption, not a measurement.
MAP_ORIGIN_WORLD = (5.0, 0.0, math.pi / 2)


def world_to_map(p):
    ox, oy, oyaw = MAP_ORIGIN_WORLD
    dx, dy = p[0] - ox, p[1] - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    return (c * dx - s * dy, s * dx + c * dy)


COURSES = [(name, world_to_map(a), world_to_map(b)) for name, a, b in COURSES]

print(f"map: {MAP}")
print(f"map frame origin in world: {MAP_ORIGIN_WORLD}\n")
print(f"{'course':<14}{'shortest_path_m':>17}{'straight':>11}{'ratio':>8}")
# UNKNOWN CELLS COUNT AS FREE HERE, and obstacles are still inflated by the
# footprint. A slam_toolbox map of a corridor is mostly unknown - 148688 unknown
# against 37407 free on this one - because the LiDAR only ever hit surfaces, so
# treating unknown as blocked (and then DILATING it) seals every corridor and
# every course reports "no path" even when both endpoints are free.
#
# The cost of this choice: a reference path may cut through space the robot
# never observed, which would make the denominator too SHORT and so the SPL too
# LOW - it understates the robot rather than flattering it. The check that it
# has not gone wrong is the ratio column: a reference path far shorter than the
# straight line, or below 1.0, means it found a shortcut through an unmapped
# wall and the number must not be used.
for name, start, goal in COURSES:
    metres = shortest_path_m(MAP, start, goal, unknown_is_blocked=False)
    straight = math.dist(start, goal)
    if metres is None:
        print(f"{name:<14}{'no path':>17}{straight:>11.3f}{'-':>8}"
              "   <- course not covered by the map, or blocked for this footprint")
    else:
        print(f"{name:<14}{metres:>17.4f}{straight:>11.3f}{metres / straight:>8.3f}")

print("\nPaste into ubot_eval/config/courses.yaml, and DELETE the")
print("`shortest_path_provisional: true` flag from any course you replace.")
PY
