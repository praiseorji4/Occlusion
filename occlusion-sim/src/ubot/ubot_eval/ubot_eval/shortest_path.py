"""The shortest path a robot of this size could have taken - the SPL denominator.

    python3 -m ubot_eval.shortest_path --selftest
    python3 -m ubot_eval.shortest_path --map maps/apartment.yaml \
        --start 5.0 0.0 --goal 5.0 3.0

WHY NOT THE STRAIGHT LINE
-------------------------
SPL divides the optimal path by the actual one, so a straight-line denominator
makes every robot that goes round a wall look worse than it is, and makes a
robot that cuts a corner it could not physically fit through look better than
possible (SPL > 1). The denominator has to be the shortest path that RESPECTS
THE ROBOT'S WIDTH, measured on the same geometry the robot drove.

WHY THE PATH IS STRING-PULLED
-----------------------------
8-connected A* can only move in 45-degree steps, so a straight run at any other
angle comes out as a staircase. The classic worst case is ~8% long. Feeding that
into SPL would scale every score down by roughly that factor - invisibly, and
differently for different courses, which is worse than being uniformly wrong.
String-pulling replaces the staircase with the straight segments it approximates
whenever the line between two path points is clear of obstacles.

WHY INFLATION USES THE HALF-DIAGONAL
------------------------------------
The footprint is rectangular (357 x 252 mm), so the half-diagonal 0.219 m is the
radius of the smallest circle containing it. Inflating by that treats the robot
as that circle: it can refuse a gap the real robot could just fit through at the
right heading, which makes the denominator slightly CONSERVATIVE (longer). That
is the safe direction - it cannot manufacture an SPL above 1 - and it matches
what nav2's own costmap inflation does.

WHAT IS MEASURED AND WHAT IS ASSUMED
------------------------------------
The grid comes from a real mapping pass, so unmapped space is unknown, not free.
Unknown cells are treated as BLOCKED: a path through a region the robot never
observed is not a path anyone demonstrated. If that makes a course unreachable,
the answer is a better mapping pass, not a more generous assumption.
"""
from __future__ import annotations

import argparse
import heapq
import math
import sys
from pathlib import Path

# ROS map_server convention: cells are probability 0-100 after scaling, and the
# PGM stores 0 (occupied) to 255 (free) unless `negate` flips it.
DEFAULT_OCCUPIED_THRESH = 0.65
DEFAULT_FREE_THRESH = 0.196
FOOTPRINT_HALF_DIAGONAL_M = 0.219       # 357 x 252 mm body; see the module note

UNKNOWN = -1
FREE = 0
OCCUPIED = 1


def read_pgm(path):
    """Minimal binary PGM (P5) reader: (width, height, maxval, bytes).

    Written out rather than pulled from PIL so this module has no dependency
    beyond the standard library - it has to run in the container, in WSL and on
    a laptop without ROS.
    """
    data = Path(path).read_bytes()
    fields = []
    i = 0
    while len(fields) < 4:
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":                       # comment to end of line
            while i < len(data) and data[i:i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        start = i
        while i < len(data) and not data[i:i + 1].isspace():
            i += 1
        fields.append(data[start:i])
    magic, width, height, maxval = fields
    if magic != b"P5":
        raise ValueError(f"{path}: only binary PGM (P5) is supported, got {magic!r}")
    i += 1                                              # single whitespace after maxval
    w, h, mx = int(width), int(height), int(maxval)
    pixels = data[i:i + w * h]
    if len(pixels) != w * h:
        raise ValueError(f"{path}: expected {w * h} pixels, found {len(pixels)}")
    return w, h, mx, pixels


def classify(pixels, w, h, maxval, *, negate=False,
             occupied_thresh=DEFAULT_OCCUPIED_THRESH,
             free_thresh=DEFAULT_FREE_THRESH):
    """PGM bytes -> rows of FREE / OCCUPIED / UNKNOWN, row 0 = TOP of the image.

    Follows map_server: p = (maxval - value) / maxval, or value / maxval when
    negate is set. p above occupied_thresh is an obstacle, below free_thresh is
    free, and everything between is unknown.
    """
    grid = []
    for row in range(h):
        out = []
        for col in range(w):
            value = pixels[row * w + col]
            p = value / maxval if negate else (maxval - value) / maxval
            if p > occupied_thresh:
                out.append(OCCUPIED)
            elif p < free_thresh:
                out.append(FREE)
            else:
                out.append(UNKNOWN)
        grid.append(out)
    return grid


def load_map(yaml_path):
    """A map_server map.yaml plus its image -> (grid, resolution, origin).

    `grid` is indexed [row][col] with row 0 at the TOP of the image, which is
    the HIGHEST y in world coordinates - the flip that makes hand-checked map
    coordinates come out mirrored if it is missed.
    """
    import yaml

    yaml_path = Path(yaml_path)
    meta = yaml.safe_load(yaml_path.read_text())
    image = Path(meta["image"])
    if not image.is_absolute():
        image = yaml_path.parent / image
    w, h, maxval, pixels = read_pgm(image)
    grid = classify(pixels, w, h, maxval,
                    negate=bool(meta.get("negate", 0)),
                    occupied_thresh=float(meta.get("occupied_thresh",
                                                   DEFAULT_OCCUPIED_THRESH)),
                    free_thresh=float(meta.get("free_thresh", DEFAULT_FREE_THRESH)))
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    return grid, float(meta["resolution"]), (float(origin[0]), float(origin[1]))


def blocked_mask(grid, resolution, radius_m, *, unknown_is_blocked=True):
    """Cells the robot's CENTRE cannot occupy, after inflating by `radius_m`.

    Unknown space counts as blocked by default: see the module note. This is a
    plain dilation by a disc, which is what a circular footprint means.
    """
    h, w = len(grid), len(grid[0])
    hard = [[(cell == OCCUPIED) or (unknown_is_blocked and cell == UNKNOWN)
             for cell in row] for row in grid]
    r_cells = int(math.ceil(radius_m / resolution))
    if r_cells <= 0:
        return hard

    offsets = [(dr, dc)
               for dr in range(-r_cells, r_cells + 1)
               for dc in range(-r_cells, r_cells + 1)
               if (dr * dr + dc * dc) * resolution * resolution <= radius_m * radius_m]

    out = [[False] * w for _ in range(h)]
    for row in range(h):
        for col in range(w):
            if not hard[row][col]:
                continue
            for dr, dc in offsets:
                rr, cc = row + dr, col + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = True
    # The border is blocked too: a path must not leave the mapped area.
    for row in range(h):
        for col in range(w):
            if row < r_cells or col < r_cells or row >= h - r_cells or col >= w - r_cells:
                out[row][col] = out[row][col] or hard[row][col]
    return out


def world_to_cell(x, y, resolution, origin, height):
    """(x, y) metres -> (row, col). Row 0 is the top of the image = highest y."""
    col = int((x - origin[0]) / resolution)
    row = height - 1 - int((y - origin[1]) / resolution)
    return row, col


def cell_to_world(row, col, resolution, origin, height):
    """Cell CENTRE in metres."""
    x = origin[0] + (col + 0.5) * resolution
    y = origin[1] + (height - 1 - row + 0.5) * resolution
    return x, y


def astar(blocked, start, goal):
    """8-connected A* over a boolean blocked grid. Returns cells, or None."""
    h, w = len(blocked), len(blocked[0])

    def ok(rc):
        r, c = rc
        return 0 <= r < h and 0 <= c < w and not blocked[r][c]

    if not ok(start) or not ok(goal):
        return None

    def heuristic(a, b):
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        # Octile distance: exact for 8-connected movement, so it never
        # overestimates and A* stays optimal on the grid.
        return (max(dr, dc) - min(dr, dc)) + math.sqrt(2) * min(dr, dc)

    open_heap = [(heuristic(start, goal), 0.0, start)]
    came_from = {}
    best = {start: 0.0}
    seen = set()
    while open_heap:
        _, cost, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]
        if current in seen:
            continue
        seen.add(current)
        r, c = current
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if not ok(nxt):
                    continue
                if dr and dc:
                    # No cutting between two diagonally touching obstacles: the
                    # robot cannot squeeze through a corner that is closed.
                    if blocked[r + dr][c] or blocked[r][c + dc]:
                        continue
                    step = math.sqrt(2)
                else:
                    step = 1.0
                new_cost = cost + step
                if new_cost < best.get(nxt, float("inf")):
                    best[nxt] = new_cost
                    came_from[nxt] = current
                    heapq.heappush(open_heap,
                                   (new_cost + heuristic(nxt, goal), new_cost, nxt))
    return None


def line_of_sight(blocked, a, b):
    """Is every cell on the segment a->b free? Supercover walk, so it cannot
    slip diagonally between two obstacles."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r1 > r0 else -1
    sc = 1 if c1 > c0 else -1
    r, c = r0, c0
    if blocked[r][c]:
        return False
    err = dr - dc
    while (r, c) != (r1, c1):
        e2 = 2 * err
        stepped_r = stepped_c = False
        if e2 > -dc:
            err -= dc
            r += sr
            stepped_r = True
        if e2 < dr:
            err += dr
            c += sc
            stepped_c = True
        if stepped_r and stepped_c:
            # Moving diagonally: both orthogonal neighbours must be clear.
            if blocked[r][c - sc] or blocked[r - sr][c]:
                return False
        if blocked[r][c]:
            return False
    return True


def string_pull(blocked, path):
    """Drop intermediate cells whenever the straight line stays clear.

    Without this, A*'s 45-degree staircase reports a straight run at an
    arbitrary angle as up to ~8% longer than it is.
    """
    if not path or len(path) < 3:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(blocked, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def path_length_m(path, resolution):
    return sum(math.dist(path[k], path[k + 1]) for k in range(len(path) - 1)) * resolution


def shortest_path_m(map_yaml, start_xy, goal_xy, *,
                    radius_m=FOOTPRINT_HALF_DIAGONAL_M, unknown_is_blocked=True):
    """Metres, or None when no path exists for a robot this size.

    None is a refusal, not a zero: `courses.yaml` keeps null and schema.validate
    then declines to score a success on that course.
    """
    grid, resolution, origin = load_map(map_yaml)
    height = len(grid)
    blocked = blocked_mask(grid, resolution, radius_m,
                           unknown_is_blocked=unknown_is_blocked)
    start = world_to_cell(start_xy[0], start_xy[1], resolution, origin, height)
    goal = world_to_cell(goal_xy[0], goal_xy[1], resolution, origin, height)
    raw = astar(blocked, start, goal)
    if raw is None:
        return None
    return round(path_length_m(string_pull(blocked, raw), resolution), 4)


# --------------------------------------------------------------------------- #
def _grid_from_art(rows):
    """'.' free, '#' occupied, '?' unknown."""
    lookup = {".": FREE, "#": OCCUPIED, "?": UNKNOWN}
    return [[lookup[ch] for ch in row] for row in rows]


def selftest() -> int:
    print("SELFTEST  ubot_eval.shortest_path (A* + string pull, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<54} {detail}")

    res = 0.05

    # Open 2 x 2 m room, no inflation: a straight 1 m run must measure 1 m.
    open_room = _grid_from_art(["." * 40 for _ in range(40)])
    blocked = blocked_mask(open_room, res, 0.0)
    a = world_to_cell(0.5, 0.5, res, (0.0, 0.0), 40)
    b = world_to_cell(1.5, 0.5, res, (0.0, 0.0), 40)
    straight = path_length_m(string_pull(blocked, astar(blocked, a, b)), res)
    check("straight run measures its true length", abs(straight - 1.0) < 1e-9,
          f"{straight:.4f} m")

    # THE REASON STRING-PULLING EXISTS. A shallow diagonal is a staircase to
    # 8-connected A*. Raw, it is long; pulled, it is the true distance.
    c = world_to_cell(1.5, 0.9, res, (0.0, 0.0), 40)
    raw = astar(blocked, a, c)
    raw_len = path_length_m(raw, res)
    pulled_len = path_length_m(string_pull(blocked, raw), res)
    true_len = math.dist((0.5, 0.5), (1.5, 0.9))
    check("raw 8-connected A* overestimates a shallow diagonal",
          raw_len > true_len * 1.02, f"raw {raw_len:.4f} vs true {true_len:.4f}")
    check("string pull recovers the true distance",
          abs(pulled_len - true_len) < 0.02,
          f"pulled {pulled_len:.4f} vs true {true_len:.4f}")
    check("string pull never lengthens a path", pulled_len <= raw_len + 1e-9)

    # A wall with a gap: the path must go through the gap, not through the wall,
    # so it is longer than the straight line.
    rows = ["." * 40 for _ in range(18)]
    rows += ["#" * 18 + ".." + "#" * 20]
    rows += ["." * 40 for _ in range(21)]
    walled = _grid_from_art(rows)
    wb = blocked_mask(walled, res, 0.0)
    start = world_to_cell(0.5, 0.2, res, (0.0, 0.0), 40)
    goal = world_to_cell(0.5, 1.8, res, (0.0, 0.0), 40)
    detour = path_length_m(string_pull(wb, astar(wb, start, goal)), res)
    check("a wall forces a detour longer than the straight line",
          detour > math.dist((0.5, 0.2), (0.5, 1.8)) + 0.05,
          f"{detour:.4f} m vs straight {math.dist((0.5, 0.2), (0.5, 1.8)):.4f} m")

    # THE ROBOT'S WIDTH DECIDES WHETHER A GAP IS A GAP. The gap above is 2 cells
    # = 0.10 m wide; a 0.219 m-radius robot cannot use it.
    wide = blocked_mask(walled, res, FOOTPRINT_HALF_DIAGONAL_M)
    check("a gap narrower than the robot is not a path",
          astar(wide, start, goal) is None,
          "0.10 m gap vs 0.438 m body")

    # Unknown space is not free space.
    with_unknown = _grid_from_art(
        ["." * 40 for _ in range(18)] + ["?" * 40] + ["." * 40 for _ in range(21)])
    ub = blocked_mask(with_unknown, res, 0.0, unknown_is_blocked=True)
    check("unmapped space blocks the path rather than being assumed clear",
          astar(ub, start, goal) is None)
    fb = blocked_mask(with_unknown, res, 0.0, unknown_is_blocked=False)
    check("...and the assumption is switchable, for comparison",
          astar(fb, start, goal) is not None)

    # No path at all is None, never 0.0 - a 0.0 denominator would make SPL
    # infinite or silently zero depending on the guard that caught it.
    sealed = _grid_from_art(
        ["." * 40 for _ in range(18)] + ["#" * 40] + ["." * 40 for _ in range(21)])
    sb = blocked_mask(sealed, res, 0.0)
    check("a sealed wall gives None, not 0.0", astar(sb, start, goal) is None)

    # The world <-> cell round trip, including the y flip.
    for (x, y) in ((0.5, 0.5), (1.9, 0.1), (0.1, 1.9)):
        r, cc = world_to_cell(x, y, res, (0.0, 0.0), 40)
        bx, by = cell_to_world(r, cc, res, (0.0, 0.0), 40)
        if abs(bx - x) > res or abs(by - y) > res:
            check(f"world->cell->world round trip at ({x}, {y})", False,
                  f"got ({bx:.3f}, {by:.3f})")
            break
    else:
        check("world->cell->world round trips within one cell", True)

    # A non-zero map origin must shift the answer, not the length.
    shifted = path_length_m(string_pull(blocked, astar(
        blocked,
        world_to_cell(-9.5, -9.5, res, (-10.0, -10.0), 40),
        world_to_cell(-8.5, -9.5, res, (-10.0, -10.0), 40))), res)
    check("a negative map origin does not change a length",
          abs(shifted - 1.0) < 1e-9, f"{shifted:.4f} m")

    # Diagonal corner-cutting: two obstacles touching at a corner are not a gap.
    corner = _grid_from_art(["..#.", ".#..", "....", "...."])
    cb = blocked_mask(corner, res, 0.0)
    path = astar(cb, (0, 0), (0, 3))
    if path:
        squeezed = any(path[k] == (1, 2) and path[k - 1] == (0, 1)
                       for k in range(1, len(path)))
        check("A* does not squeeze between diagonally touching obstacles",
              not squeezed)
    else:
        check("A* does not squeeze between diagonally touching obstacles", True,
              "no path at all")

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--map")
    ap.add_argument("--start", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--goal", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--radius", type=float, default=FOOTPRINT_HALF_DIAGONAL_M)
    ap.add_argument("--allow-unknown", action="store_true",
                    help="treat unmapped cells as free (reports a path nobody demonstrated)")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not (args.map and args.start and args.goal):
        ap.error("--map, --start and --goal are required")

    metres = shortest_path_m(args.map, tuple(args.start), tuple(args.goal),
                             radius_m=args.radius,
                             unknown_is_blocked=not args.allow_unknown)
    if metres is None:
        print("no path: the goal is unreachable for a robot of this size on this map")
        sys.exit(1)
    straight = math.dist(tuple(args.start), tuple(args.goal))
    print(f"shortest_path_m: {metres}")
    print(f"straight line:   {straight:.4f}   (ratio {metres / straight:.3f})")


if __name__ == "__main__":
    main()
