#!/usr/bin/env python3
"""Shortest path for every course in a world, from a saved occupancy grid.

    python3 measure_courses.py --world apartment \
        --map ~/occl_ws/maps/apartment.yaml --origin 5.0 0.0 1.5708

    python3 measure_courses.py --selftest

Replaces the apartment-only shell version: courses come from courses.yaml, so a
new world needs no new code.

--origin IS THE POSE THE MAPPING PASS STARTED FROM, and it matters.
slam_toolbox puts the map frame's origin at the robot's initial pose, while
courses.yaml states everything in WORLD coordinates. Querying world coordinates
against a SLAM map without that transform asks about points far outside the
mapped area, and every course comes back "no path" - which looks exactly like
thin coverage and is not. For the parking garage the mapping spawn is
(0, 0, yaw 0), so the two frames coincide and the transform is the identity;
for the apartment it is (5, 0, pi/2) and very much is not.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # ubot_eval package

from ubot_eval.shortest_path import shortest_path_m   # noqa: E402


def world_to_map(point, origin):
    """A WORLD (x, y) in the SLAM map frame, given the mapping start pose."""
    ox, oy, oyaw = origin
    dx, dy = point[0] - ox, point[1] - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    return (c * dx - s * dy, s * dx + c * dy)


def origin_from_start(start_json) -> tuple:
    """world->map as an (x, y, yaw) origin for world_to_map, from RECORDED poses.

    p_map = R(dyaw) * (p_world - world_pose) + map_pose, rewritten into the
    origin form world_to_map implements: world_to_map(p, o) = R(-oyaw)(p - o).
    So oyaw = -dyaw and o = world_pose - R(-dyaw) * map_pose.
    """
    import json
    doc = json.loads(Path(start_json).read_text(encoding="utf-8"))
    wx, wy, wyaw = doc["world_pose"]
    mx, my, myaw = doc["map_pose"]
    dyaw = myaw - wyaw
    c, s = math.cos(-dyaw), math.sin(-dyaw)
    return (wx - (c * mx - s * my), wy - (s * mx + c * my), -dyaw)


def load_courses(world: str, config_dir=None) -> list:
    """[(course, spawn_xy, goal_xy)] for one world, from courses.yaml."""
    import yaml
    if config_dir is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            config_dir = Path(get_package_share_directory("ubot_eval")) / "config"
        except Exception:
            config_dir = HERE.parent / "config"
    doc = yaml.safe_load((Path(config_dir) / "courses.yaml").read_text(encoding="utf-8"))
    world_doc = doc["worlds"][world]
    out = []
    for name, course in world_doc["courses"].items():
        out.append((name,
                    (float(course["spawn"]["x"]), float(course["spawn"]["y"])),
                    (float(course["goal"]["x"]), float(course["goal"]["y"]))))
    return out


#: Above this share of UNKNOWN cells along the course, the measurement is refused.
MAX_UNKNOWN_FRACTION = 0.20


def unknown_fraction(map_yaml, a_map, b_map) -> float:
    """Share of cells on the straight segment a->b that the map never observed.

    WHY THIS EXISTS. Unknown cells count as passable (a SLAM map of a corridor
    is mostly unknown, and blocking them seals every route). The price is that
    a corridor the mapping pass never reached returns the straight line through
    it - walls included - as a confident "shortest path". A failed pass that
    reached 1 of 4 waypoints produced ratio 1.000 for EVERY course, corner_turn
    included, which looks like a flawless map and is the opposite.
    """
    from ubot_eval.shortest_path import UNKNOWN, load_map, world_to_cell
    grid, res, origin = load_map(map_yaml)
    h, w = len(grid), len(grid[0])
    n = max(2, int(math.dist(a_map, b_map) / res))
    unknown = total = 0
    for i in range(n + 1):
        t = i / n
        x = a_map[0] + (b_map[0] - a_map[0]) * t
        y = a_map[1] + (b_map[1] - a_map[1]) * t
        r, c = world_to_cell(x, y, res, origin, h)
        total += 1
        if not (0 <= r < h and 0 <= c < w) or grid[r][c] == UNKNOWN:
            unknown += 1
    return unknown / total if total else 1.0


def measure(world, map_yaml, origin, *, unknown_is_blocked=False) -> list:
    """[(course, metres|None, straight, ratio|None, unknown_frac)] per course.

    metres is None both for "no path" and for "too little of this was mapped to
    say" - a refusal, never a straight line dressed up as a measurement.
    """
    rows = []
    for name, spawn, goal in load_courses(world):
        a, b = world_to_map(spawn, origin), world_to_map(goal, origin)
        frac = unknown_fraction(map_yaml, a, b)
        metres = None
        if frac <= MAX_UNKNOWN_FRACTION:
            metres = shortest_path_m(map_yaml, a, b,
                                     unknown_is_blocked=unknown_is_blocked)
        straight = math.dist(spawn, goal)
        ratio = (metres / straight) if (metres and straight) else None
        rows.append((name, metres, straight, ratio, frac))
    return rows


def render(world, map_yaml, origin, rows) -> str:
    lines = [f"world: {world}",
             f"map:   {map_yaml}",
             f"map frame origin in world: {origin}",
             ""]
    lines.append(f"{'course':<16}{'shortest_path_m':>17}{'straight':>11}{'ratio':>8}{'unmapped':>10}")
    for name, metres, straight, ratio, frac in rows:
        if metres is None and frac > MAX_UNKNOWN_FRACTION:
            lines.append(f"{name:<16}{'REFUSED':>17}{straight:>11.3f}{'-':>8}{frac:>9.0%}"
                         "   <- the mapping pass did not cover this course")
            continue
        if metres is None:
            lines.append(f"{name:<16}{'no path':>17}{straight:>11.3f}{'-':>8}"
                         "   <- not covered by the map, or blocked for this footprint")
        else:
            flag = ""
            # A reference SHORTER than the straight line means the path cut a
            # corner through space the robot cannot actually pass, which would
            # inflate every SPL computed from it.
            if ratio is not None and ratio < 0.999:
                flag = "   <- SHORTER than straight line: suspect, do not use"
            lines.append(f"{name:<16}{metres:>17.4f}{straight:>11.3f}{ratio:>8.3f}{frac:>9.0%}{flag}")
    lines.append("")
    lines.append("Paste into ubot_eval/config/courses.yaml, and delete any")
    lines.append("`shortest_path_provisional: true` on the courses you replace:")
    lines.append("")
    for name, metres, _, _, _ in rows:
        if metres is not None:
            lines.append(f"        # MEASURED from {Path(map_yaml).name}")
            lines.append(f"        shortest_path_m: {metres}    # {name}")
    return "\n".join(lines)


def selftest() -> int:
    print("SELFTEST  measure_courses (frame transform only, no map needed)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<56} {detail}")

    # The garage: mapping starts at the world origin, so the frames coincide.
    ident = world_to_map((8.0, 0.0), (0.0, 0.0, 0.0))
    check("an origin of (0,0,0) leaves world coordinates unchanged",
          abs(ident[0] - 8.0) < 1e-9 and abs(ident[1]) < 1e-9, str(ident))

    # The apartment: spawn (5, 0) facing +y, so world (5, 3) is 3 m straight
    # ahead - map (3, 0).
    apt = world_to_map((5.0, 3.0), (5.0, 0.0, math.pi / 2))
    check("apartment world (5,3) becomes map (3,0)",
          abs(apt[0] - 3.0) < 1e-9 and abs(apt[1]) < 1e-9, str(apt))
    check("...and the spawn itself becomes the map origin",
          all(abs(v) < 1e-9 for v in world_to_map((5.0, 0.0), (5.0, 0.0, math.pi / 2))))

    # Distances must survive the transform.
    a = world_to_map((-4.0, 0.0), (0.0, 0.0, math.pi))
    b = world_to_map((-11.0, 0.0), (0.0, 0.0, math.pi))
    check("a rotated frame preserves distance",
          abs(math.dist(a, b) - 7.0) < 1e-9, f"{math.dist(a, b):.3f} m")

    import json
    import tempfile
    for label, map_yaw, expect in (
            ("recorded poses reproduce the old heading-aligned case", 0.0, (3.0, 0.0)),
            ("a WORLD-aligned map (the case that broke) maps (5,3) to (0,3)",
             math.pi / 2, (0.0, 3.0))):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"world_pose": [5.0, 0.0, math.pi / 2],
                       "map_pose": [0.0, 0.0, map_yaw]}, fh)
        got = world_to_map((5.0, 3.0), origin_from_start(fh.name))
        check(label, abs(got[0] - expect[0]) < 1e-9 and abs(got[1] - expect[1]) < 1e-9,
              str(tuple(round(v, 6) for v in got)))

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--world")
    ap.add_argument("--map")
    ap.add_argument("--origin", nargs=3, type=float, metavar=("X", "Y", "YAW"),
                    help="the pose the MAPPING PASS started from, in world coords")
    ap.add_argument("--start-json", default=None,
                    help="recorded map+world start pose; preferred over --origin")
    ap.add_argument("--block-unknown", action="store_true",
                    help="treat unmapped cells as obstacles (seals sparse SLAM maps)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.start_json and Path(args.start_json).exists():
        args.origin = origin_from_start(args.start_json)
        print(f"(origin derived from the recorded start pose: {args.origin})")
    if not (args.world and args.map and args.origin):
        ap.error("--world, --map and --origin are required")

    rows = measure(args.world, args.map, tuple(args.origin),
                   unknown_is_blocked=args.block_unknown)
    print(render(args.world, args.map, tuple(args.origin), rows))
    sys.exit(0 if all(r[1] is not None for r in rows) else 1)


if __name__ == "__main__":
    main()
