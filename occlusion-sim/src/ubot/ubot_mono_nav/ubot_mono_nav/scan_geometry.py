r"""Depth image -> LaserScan, the geometry only. No ROS, no models, numpy alone.

    python -m ubot_mono_nav.scan_geometry --selftest      # runs anywhere

This is the piece that decides where the robot thinks obstacles are, so it is
kept free of ROS and of the depth network: it can be tested on a laptop with no
robot, against scenes whose answers are known by construction. `depth_to_scan.py`
is a thin ROS wrapper around it.

--------------------------------------------------------------------------
FRAMES
--------------------------------------------------------------------------
Two conventions meet here and mixing them is the classic way to put obstacles
90 degrees out (the ubot URDF already has one frame mislabelled -- see
ubot_camera.urdf.xacro, where camera_depth_frame claims to be optical but is
attached with rpy="0 0 0").

    optical (REP-104)   X right,   Y DOWN, Z FORWARD   <- depth images
    base (REP-103)      X forward, Y LEFT, Z UP        <- the robot

`optical_to_base` builds the rotation for a camera that is level except for a
pitch, which is the ubot's case. The ROS node prefers the real transform from
TF and falls back to this.

--------------------------------------------------------------------------
WHY A HEIGHT BAND
--------------------------------------------------------------------------
The camera sits ~0.19 m off the ground and looks slightly up, so most of the
image is FLOOR. Floor pixels are perfectly good depth returns, and turning them
into obstacles would make the robot brake at the ground in front of it. Only
points between `height_min` and `height_max` above the base plane count.

--------------------------------------------------------------------------
WHY UNKNOWN IS +inf AND NOT range_max
--------------------------------------------------------------------------
Monocular depth is trustworthy over a limited band (1-4 m for the indoor DAv2
head at this viewpoint; beyond that it saturates). Past `trusted_range` we do
not know what is there. In a LaserScan, +inf means "nothing detected along this
ray" and nav2 will CLEAR along it; a finite number would mark an obstacle. So
this returns +inf and lets the caller cap `range_max` at the trusted distance,
so nav2 never clears space the camera cannot actually see.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["optical_to_base", "depth_to_scan", "ScanSpec"]


class ScanSpec:
    """The LaserScan angular grid, and the ranges that go with it."""

    def __init__(self, angle_min: float = -math.pi / 4, angle_max: float = math.pi / 4,
                 angle_increment: float = math.radians(0.5),
                 range_min: float = 0.15, range_max: float = 4.0):
        if angle_max <= angle_min:
            raise ValueError("angle_max must exceed angle_min")
        self.angle_min = float(angle_min)
        self.angle_max = float(angle_max)
        self.angle_increment = float(angle_increment)
        self.range_min = float(range_min)
        self.range_max = float(range_max)

    @property
    def n_bins(self) -> int:
        return int(round((self.angle_max - self.angle_min) / self.angle_increment)) + 1


def optical_to_base(pitch_up: float = 0.0, xyz=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) taking a point in the camera's OPTICAL frame into the base frame.

    pitch_up : radians, positive when the camera looks UP. On the ubot the OAK-D
               is tilted DOWN 4.09 deg by the CamCase, i.e. pitch_up is negative;
               the URDF carries that and depth_to_scan reads it from TF.
    xyz      : camera origin in the base frame, metres.

    With no pitch, optical Z (forward) -> base X, optical X (right) -> base -Y,
    optical Y (down) -> base -Z. Pitching the camera up rotates about the base
    Y axis (which points LEFT), so looking up is a NEGATIVE rotation about it.
    """
    base_from_optical = np.array([[0.0, 0.0, 1.0],
                                  [-1.0, 0.0, 0.0],
                                  [0.0, -1.0, 0.0]])
    c, s = math.cos(-pitch_up), math.sin(-pitch_up)
    r_y = np.array([[c, 0.0, s],
                    [0.0, 1.0, 0.0],
                    [-s, 0.0, c]])
    return r_y @ base_from_optical, np.asarray(xyz, dtype=float)


def depth_to_scan(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                  rot: np.ndarray, trans: np.ndarray, spec: ScanSpec | None = None,
                  height_min: float = 0.05, height_max: float = 0.60,
                  trusted_range: float = 4.0, min_depth: float = 0.3,
                  stride: int = 2, min_points: int = 2) -> np.ndarray:
    """Nearest obstacle per angular bin, as LaserScan `ranges`.

    depth  : (H, W) float metres in the optical frame. 0 and non-finite are
             "no measurement" and are dropped, never treated as an obstacle at
             zero distance.
    fx..cy : intrinsics for THIS depth image's resolution. A depth map that is
             smaller than the RGB frame needs its intrinsics scaled first; the
             ROS node does that from camera_info.
    rot, trans : optical -> base, from TF or `optical_to_base`.

    Returns an array of `spec.n_bins` ranges, +inf where nothing was seen.
    """
    spec = spec or ScanSpec()
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth must be 2-D, got shape {d.shape}")
    if stride > 1:
        d = d[::stride, ::stride]
        fx, fy = fx / stride, fy / stride
        cx, cy = cx / stride, cy / stride

    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(d) & (d > max(min_depth, 0.0)) & (d <= trusted_range)
    if not ok.any():
        return np.full(spec.n_bins, np.inf, dtype=np.float32)

    z = d[ok].astype(np.float64)
    x = (u[ok] - cx) * z / fx
    y = (v[ok] - cy) * z / fy
    pts = np.column_stack([x, y, z]) @ rot.T + trans          # -> base frame

    # Height band: drop the floor the camera is mostly looking at, and drop
    # anything the robot can pass under.
    keep = (pts[:, 2] >= height_min) & (pts[:, 2] <= height_max)
    if not keep.any():
        return np.full(spec.n_bins, np.inf, dtype=np.float32)
    pts = pts[keep]

    rng = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.arctan2(pts[:, 1], pts[:, 0])
    good = (rng >= spec.range_min) & (rng <= min(spec.range_max, trusted_range)) \
        & (ang >= spec.angle_min) & (ang <= spec.angle_max)
    if not good.any():
        return np.full(spec.n_bins, np.inf, dtype=np.float32)
    rng, ang = rng[good], ang[good]

    idx = np.rint((ang - spec.angle_min) / spec.angle_increment).astype(int)
    np.clip(idx, 0, spec.n_bins - 1, out=idx)

    ranges = np.full(spec.n_bins, np.inf, dtype=np.float64)
    np.minimum.at(ranges, idx, rng)
    # A single stray pixel should not stop the robot; require a few agreeing
    # returns in a bin before it counts as an obstacle.
    if min_points > 1:
        counts = np.zeros(spec.n_bins, dtype=np.int64)
        np.add.at(counts, idx, 1)
        ranges[counts < min_points] = np.inf
    return ranges.astype(np.float32)


# --------------------------------------------------------------------------- #
def _render(scene, fx, fy, cx, cy, h, w, rot, trans) -> np.ndarray:
    """Depth image of a synthetic scene, for the selftest.

    Casts one ray per pixel into the base frame and keeps the nearest hit, so
    the test scenes are built from geometry rather than from the code under
    test. `scene` is a list of ("plane", axis, value) with axis 0=X, 2=Z.
    """
    v, u = np.mgrid[0:h, 0:w]
    dirs = np.column_stack([((u - cx) / fx).ravel(), ((v - cy) / fy).ravel(),
                            np.ones(u.size)])
    dirs_base = dirs @ rot.T                     # directions: rotation only
    depth = np.full(u.size, np.inf)
    for kind, axis, value in scene:
        if kind != "plane":
            raise ValueError(kind)
        denom = dirs_base[:, axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (value - trans[axis]) / denom
        hit = np.isfinite(s) & (s > 0)
        # s scales a direction whose optical Z is 1, so s IS the depth
        depth = np.where(hit & (s < depth), s, depth)
    return depth.reshape(h, w).astype(np.float32)


def selftest() -> int:
    """Scenes whose answers follow from the geometry, not from this code."""
    print("SELFTEST  depth -> laser scan geometry (no ROS, no models)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<46} {detail}")

    # An OAK-D Lite at the ubot's mount: 65 deg horizontal, 0.19 m up.
    W, H = 320, 180
    FX = FY = (W / 2) / math.tan(math.radians(65.0 / 2))
    CX, CY = W / 2, H / 2
    spec = ScanSpec(angle_min=-math.radians(32.5), angle_max=math.radians(32.5),
                    angle_increment=math.radians(1.0), range_max=4.0)
    CAM_H = 0.19

    def run(scene, pitch_deg=0.0, **kw):
        rot, trans = optical_to_base(math.radians(pitch_deg), (0.10, 0.0, CAM_H))
        depth = _render(scene, FX, FY, CX, CY, H, W, rot, trans)
        return depth_to_scan(depth, FX, FY, CX, CY, rot, trans, spec, **kw), depth

    centre = spec.n_bins // 2
    floor = ("plane", 2, 0.0)                     # base Z = 0
    wall2 = ("plane", 0, 2.0)                     # base X = 2 m, straight ahead

    # 1. a wall 2 m ahead, with the floor also in view.
    # Ranges are measured from the BASE origin, not from the lens: the points
    # are transformed into the base frame before the range is taken, so the
    # camera's 0.10 m forward offset is already accounted for. That is what
    # lets the LaserScan be published in base_footprint.
    r, _ = run([floor, wall2])
    check("wall at base X=2 m reads 2 m from base origin", abs(r[centre] - 2.00) < 0.05,
          f"centre bin {r[centre]:.2f} m")
    off = np.degrees(spec.angle_min + centre // 2 * spec.angle_increment)
    check("off-axis bins read farther (wall is flat, not curved)",
          r[centre // 2] > r[centre], f"at {off:.0f} deg: {r[centre // 2]:.2f} m")

    # 2. THE ONE THAT MATTERS: floor alone must produce no obstacle at all
    r, d = run([floor])
    check("floor alone -> no obstacles (nothing to brake for)",
          np.all(~np.isfinite(r)), f"valid depth in {100 * np.mean(np.isfinite(d)):.0f}% of pixels")

    # 3. the same wall with the camera pitched up, as the real mount is
    r, _ = run([floor, wall2], pitch_deg=10.0)
    check("pitched up 10 deg, wall still at 2 m", abs(r[centre] - 2.00) < 0.05,
          f"centre bin {r[centre]:.2f} m")

    # 4. past the trusted range: unknown, NOT free and NOT an obstacle
    r, _ = run([floor, ("plane", 0, 6.0)], trusted_range=4.0)
    check("wall at 6 m with 4 m trust -> +inf everywhere",
          np.all(~np.isfinite(r)), "unknown, so nav2 neither marks nor clears it")

    # 5. a low box on the floor, the case the height band must NOT filter out
    r, _ = run([floor, ("plane", 0, 1.5)], height_max=0.60)
    check("obstacle at 1.5 m is seen", abs(r[centre] - 1.50) < 0.05,
          f"centre bin {r[centre]:.2f} m")

    # 6. overhead clearance: a ceiling at 1.2 m must be ignored
    r, _ = run([floor, ("plane", 2, 1.2)], height_max=0.60)
    check("ceiling above height_max ignored", np.all(~np.isfinite(r)))

    # 7. zeros and NaNs are missing data, never obstacles at 0 m
    rot, trans = optical_to_base(0.0, (0.10, 0.0, CAM_H))
    d = _render([floor, wall2], FX, FY, CX, CY, H, W, rot, trans)
    d[:, :W // 2] = 0.0
    d[0:10, :] = np.nan
    r = depth_to_scan(d, FX, FY, CX, CY, rot, trans, spec)
    finite = r[np.isfinite(r)]
    check("zero and NaN depth dropped, not read as 0 m",
          finite.size > 0 and finite.min() > spec.range_min,
          f"nearest {finite.min():.2f} m")

    # 8. a lone hot pixel must not stop the robot
    d = np.full((H, W), np.inf, dtype=np.float32)
    d[H // 2, W // 2] = 1.0
    r = depth_to_scan(d, FX, FY, CX, CY, rot, trans, spec, min_points=2)
    check("single stray pixel rejected (min_points)", np.all(~np.isfinite(r)))

    # 9. the frame convention itself: a wall to the LEFT must land at a
    #    positive angle, which is where REP-103 puts left
    r, _ = run([floor, ("plane", 1, 1.0)])       # base Y = +1 m is to the left
    seen = np.where(np.isfinite(r))[0]
    left_half = seen > centre
    check("obstacle on the left appears at positive angles",
          seen.size > 0 and left_half.mean() > 0.9,
          f"{seen.size} bins, {100 * left_half.mean():.0f}% left of centre")

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
