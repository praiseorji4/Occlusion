r"""Where "where the robot actually was" comes from.

    python3 -m ubot_eval.truth_source --selftest       # pure maths, no ROS

--------------------------------------------------------------------------
WHY NOT ODOMETRY
--------------------------------------------------------------------------
Every path metric is a comparison against truth, and on this stack odometry is
the thing under suspicion: there is no SLAM, so nav2 believes odom, and when
odom drifts nav2 reports SUCCEEDED from somewhere else entirely. That failure
has its own outcome class (`false_success`) and it can only be detected with an
independent reference.

    sim   Gazebo's own pose stream. SceneBroadcaster already publishes
          /world/<world name>/dynamic_pose/info, so no world edit is needed.
    real  There is none. EKF odom plus an operator's tape-measured final pose,
          with `odom_drift_m` carried beside every path metric, never hidden.

--------------------------------------------------------------------------
TWO TRAPS
--------------------------------------------------------------------------
1. The topic carries the **<world name> inside the sdf**, which is not the file
   name: `parking_garage_occlusion.sdf` declares `<world name='parking_garage'>`
   and `sonoma_occlusion.sdf` declares `sensors`. Bridging the wrong one gives a
   ROS topic that exists and never receives a message — the same silent failure
   the contact bridge hit. `sim.launch.py` substitutes `world_name` for this
   reason; `SimTruth` refuses to start rather than wait forever on a dead topic.
2. `gz model -m ubot -p` blocks about a second per call, so it cannot sample a
   trajectory. It is still the independent check on the stream: one sample per
   episode, and a disagreement over 2 cm means the bridge is reporting some
   other model's pose. Refuse the episode rather than record a plausible lie.

--------------------------------------------------------------------------
PATH LENGTH IS NOT THE SUM OF THE SAMPLES
--------------------------------------------------------------------------
The pose stream runs at ~60 Hz and jitters by a few millimetres while the robot
is stationary. Summing raw consecutive distances turns that jitter into metres
of phantom travel, which inflates the denominator of SPL and flatters nothing:
it just makes every number wrong. `path_length` decimates, smooths, and counts
only steps taken while the robot was demonstrably moving. The selftest pins a
stationary robot under 0.05 m AND a 0.03 m/s crawl at its true length, which a
plain minimum-step rule cannot do at once - see `path_length` for the numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

DEFAULT_MODEL = "ubot"
CROSS_CHECK_TOLERANCE_M = 0.02
DECIMATE_HZ = 10.0
#: Rejects float noise only. The real work is done by the speed gate; a floor
#: large enough to stop jitter (5 mm at 10 Hz) also silently deletes any travel
#: below 0.05 m/s, which is a third of this robot's cruise.
MIN_STEP_M = 0.001
#: Moving-average window over positions. Half a second divides +-5 mm jitter by
#: sqrt(5) while a 0.15 m/s robot still moves ~75 mm within it.
SMOOTH_WINDOW_S = 0.5


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Yaw only; the robot is planar and roll/pitch are noise on a flat floor."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def decimate(samples, target_hz: float = DECIMATE_HZ):
    """Keep at most one sample per 1/target_hz, first and last always kept.

    `samples` are (t, x, y, yaw). Decimation is by TIME, not by index, so a
    stream that stutters does not silently change the effective rate.
    """
    samples = list(samples)
    if len(samples) < 3:
        return samples
    period = 1.0 / float(target_hz)
    out = [samples[0]]
    for s in samples[1:-1]:
        # 1e-9 of slack: six ticks of a 60 Hz stream is 0.09999999999999999 s,
        # which is "not yet 0.1" in binary, and every sixth sample would be
        # dropped for no reason anyone would ever find.
        if s[0] - out[-1][0] >= period - 1e-9:
            out.append(s)
    out.append(samples[-1])
    return out


def smooth(samples, window_s: float = SMOOTH_WINDOW_S):
    """Centred moving average of the positions, with the endpoints preserved.

    WHY THIS EXISTS, given the plan prescribes only decimate + min-step:
    that pair does NOT remove the jitter it is aimed at. Two random points in a
    +-5 mm box are typically 5-7 mm apart, which clears a 5 mm threshold, so the
    steps are summed. Measured on the selftest fixture - 60 s stationary at
    60 Hz - raw gives 18.7 m and decimate + min-step still gives 2.17 m, against
    a requirement of < 0.05 m. Averaging first divides the noise by sqrt(window),
    which puts it under the threshold and lets the min-step rule work.

    The window shrinks symmetrically towards each end (width 1 at the endpoints),
    so the first and last poses are exact and a straight line keeps its length.
    """
    samples = list(samples)
    n = len(samples)
    if n < 3:
        return samples
    dts = sorted(b[0] - a[0] for a, b in zip(samples, samples[1:]))
    dt = dts[len(dts) // 2]
    if dt <= 0:
        return samples
    half_max = max(0, int(round((window_s / 2.0) / dt)))
    if half_max == 0:
        return samples

    out = []
    for i, s in enumerate(samples):
        half = min(i, n - 1 - i, half_max)
        if half == 0:
            out.append(s)
            continue
        window = samples[i - half:i + half + 1]
        k = len(window)
        out.append((s[0], sum(w[1] for w in window) / k,
                    sum(w[2] for w in window) / k) + tuple(s[3:]))
    return out


#: Below this the robot is not travelling, it is jittering. Seven times under
#: the 0.15 m/s cruise and comfortably under the slowest useful crawl, so real
#: motion is never gated out - both directions are pinned in the selftest.
MIN_SPEED_MS = 0.02
#: Speed is measured over this span, not between adjacent samples. Jitter can
#: look fast for 0.1 s; it cannot sustain 2 cm/s for a whole second.
SPEED_WINDOW_S = 1.0


def sustained_speeds(pts, window_s: float = SPEED_WINDOW_S):
    """Speed at each sample, measured across a +-window_s/2 span of real travel.

    Net displacement over the span divided by the span, so noise that wanders
    and comes back reads as slow - which it is.
    """
    n = len(pts)
    if n < 2:
        return [0.0] * n
    times = [p[0] for p in pts]
    half = window_s / 2.0
    out = []
    lo = hi = 0
    for i, t in enumerate(times):
        while times[lo] < t - half:          # both pointers only advance
            lo += 1
        hi = max(hi, i)
        while hi + 1 < n and times[hi + 1] <= t + half:
            hi += 1
        span = times[hi] - times[lo]
        out.append(math.dist(pts[lo][1:3], pts[hi][1:3]) / span if span > 0 else 0.0)
    return out


def path_length(samples, target_hz: float = DECIMATE_HZ,
                min_step_m: float = MIN_STEP_M,
                window_s: float = SMOOTH_WINDOW_S,
                min_speed_ms: float = MIN_SPEED_MS,
                speed_window_s: float = SPEED_WINDOW_S) -> float:
    """Distance travelled while actually moving.

    Measured on two fixtures that a single threshold cannot satisfy together -
    60 s of a stationary robot with +-5 mm pose noise at 60 Hz (must read under
    0.05 m) and 60 s of a genuine 0.03 m/s crawl (must read 1.8 m):

        method                                  jitter     crawl
        raw sum                                 18.7 m     1.80 m
        decimate 10 Hz + 5 mm minimum step       2.17 m    0.00 m   <- the plan's
        + moving average                         0.08 m    0.00 m      recipe
        + per-step speed gate                    0.00 m    0.00 m
        + SUSTAINED speed gate (this)            0.00 m    1.80 m

    The minimum-step rule is the trap. At 10 Hz a 5 mm floor is a 0.05 m/s speed
    floor, which deletes any travel slower than a third of cruise - silently,
    and in the direction that flatters SPL. Gating on speed sustained over a
    second separates the two cases on physics instead: jitter never holds
    2 cm/s, and a crawl does. `min_step_m` survives only to reject float noise.

    Rotation in place contributes nothing, which is correct: SPL compares
    distance covered, and spinning covers none.
    """
    pts = smooth(decimate(samples, target_hz), window_s)
    if len(pts) < 2:
        return 0.0
    speeds = sustained_speeds(pts, speed_window_s)
    total = 0.0
    for i in range(len(pts) - 1):
        d = math.dist(pts[i][1:3], pts[i + 1][1:3])
        if d < min_step_m:
            continue
        # Either end of the step counting as moving keeps starts and stops whole.
        if max(speeds[i], speeds[i + 1]) < min_speed_ms:
            continue
        total += d
    return total


def pose_error(a, b) -> float:
    """Planar distance between two (x, y, ...) poses."""
    return math.dist((a[0], a[1]), (b[0], b[1]))


class SimTruth:
    """Ground truth from Gazebo's pose stream, via /gt/tf.

    Subscribes rather than polls, so a whole trajectory is available. ROS
    imports are lazy: this module must import for its selftest on a machine
    with no ROS.
    """

    def __init__(self, node, *, model: str = DEFAULT_MODEL, topic: str = "/gt/tf",
                 world_name: str | None = None):
        from tf2_msgs.msg import TFMessage
        self.node = node
        self.model = model
        self.topic = topic
        self.world_name = world_name
        self.samples: list = []
        self._latest = None
        #: Every child_frame_id seen, so a timeout can say what DID arrive.
        #: Without this, "no ground truth" is indistinguishable from "plenty of
        #: ground truth under a name you did not expect", which is exactly the
        #: failure that cost a launch cycle to find.
        self.seen_frames: dict = {}
        node.create_subscription(TFMessage, topic, self._on_tf, 50)

    def _rank(self, child: str, parent: str) -> int:
        """How much we want this frame as the robot's pose. 0 means: not ours.

        ONLY the model's pose in a world frame counts. The link poses published
        alongside it are expressed RELATIVE TO THE MODEL - measured:

            'ubot'                 xyz=(5.0, 0.0, -0.02)  parent='apartment'
            'ubot/base_footprint'  xyz=(0.0, 0.0,  0.0)   parent='ubot'

        so treating a link as a fallback would report the robot parked at the
        origin for the whole episode, with a path length of zero and a
        `false_success` for every goal. It looks like data, which is why it is
        refused here rather than ranked lower.
        """
        if child == self.model and parent != self.model:
            return 3
        return 0

    def _on_tf(self, msg) -> None:
        best, best_rank = None, 0
        for tr in msg.transforms:
            child = tr.child_frame_id
            self.seen_frames[child] = self.seen_frames.get(child, 0) + 1
            rank = self._rank(child, tr.header.frame_id)
            if rank > best_rank:
                best, best_rank = tr, rank
        if best is None:
            return
        t = best.header.stamp.sec + best.header.stamp.nanosec * 1e-9
        q = best.transform.rotation
        p = best.transform.translation
        self._latest = (t, p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w))
        self.samples.append(self._latest)

    def wait_for_stream(self, timeout_s: float = 10.0) -> None:
        """Fail loudly, naming what arrived, rather than producing no truth."""
        import rclpy
        deadline = time.time() + timeout_s
        while time.time() < deadline and self._latest is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        if self._latest is not None:
            return

        if self.seen_frames:
            names = ", ".join(f"{n!r} x{c}" for n, c in
                              sorted(self.seen_frames.items(), key=lambda kv: -kv[1])[:6])
            if any(n.startswith(f"{self.model}/") for n in self.seen_frames):
                raise RuntimeError(
                    f"{self.topic} carries this model's LINK poses but not its model "
                    f"pose. Frames seen: {names}.\nLink poses are relative to the "
                    "model (base_footprint reads 0,0,0), so they cannot be used as "
                    "ground truth. Set <publish_model_pose>true</publish_model_pose> "
                    "on the PosePublisher in ubot_gazebo.urdf.xacro.")
            raise RuntimeError(
                f"{self.topic} is publishing, but nothing on it is model {self.model!r}. "
                f"Frames seen: {names}.\n"
                "An EMPTY name means the bridge is fed /world/<w>/dynamic_pose/info, "
                "whose poses carry no per-pose header data, so ros_gz_bridge cannot "
                "fill child_frame_id. Bridge /model/<model>/pose instead - the "
                "PosePublisher in ubot_gazebo.urdf.xacro sets those frame ids.")
        raise RuntimeError(
            f"no messages at all on {self.topic} after {timeout_s:.0f}s. Expected the "
            "bridge line\n    /model/" + self.model + "/pose@tf2_msgs/msg/TFMessage"
            "[gz.msgs.Pose_V\nremapped to /gt/tf, and the PosePublisher plugin on the "
            "model in ubot_gazebo.urdf.xacro.")

    def pose(self):
        """(x, y, yaw) or None. The signature `readiness.RosReadiness` expects."""
        return None if self._latest is None else self._latest[1:]

    def trajectory(self):
        return list(self.samples)

    def path_length(self) -> float:
        return path_length(self.samples)

    def cross_check(self, cli_pose, tolerance_m: float = CROSS_CHECK_TOLERANCE_M):
        """Compare one stream sample against `gz model -m <model> -p`.

        Returns (ok, error_m). A disagreement means the stream is reporting a
        different entity, and every metric downstream would be measured against
        the wrong body.
        """
        here = self.pose()
        if here is None or cli_pose is None:
            return False, float("nan")
        err = pose_error(here, cli_pose)
        return err <= tolerance_m, err

    def write_trajectory(self, episode_dir) -> None:
        from pathlib import Path
        path = Path(episode_dir) / "trajectory.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for t, x, y, yaw in self.samples:
                fh.write(json.dumps({"t": t, "x": x, "y": y, "yaw": yaw}) + "\n")


class OdomTruth:
    """The real robot's stand-in: EKF odom, honestly labelled.

    Same API as `SimTruth` so the episode runner does not branch, but
    `is_ground_truth` is False and `odom_drift_m` (operator tape measure minus
    odom at the end of the run) rides along with every metric derived from it.
    """

    is_ground_truth = False

    def __init__(self, node, *, topic: str = "/odometry/filtered"):
        from nav_msgs.msg import Odometry
        self.node = node
        self.topic = topic
        self.samples: list = []
        self._latest = None
        self.odom_drift_m: float | None = None
        node.create_subscription(Odometry, topic, self._on_odom, 50)

    def _on_odom(self, msg) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._latest = (t, p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w))
        self.samples.append(self._latest)

    def pose(self):
        return None if self._latest is None else self._latest[1:]

    def trajectory(self):
        return list(self.samples)

    def path_length(self) -> float:
        return path_length(self.samples)

    def set_measured_final_pose(self, x: float, y: float) -> float:
        """Operator's tape measure. Returns the drift it implies."""
        here = self.pose()
        self.odom_drift_m = float("nan") if here is None else pose_error(here, (x, y))
        return self.odom_drift_m


SimTruth.is_ground_truth = True


# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("SELFTEST  ubot_eval.truth_source (pure maths, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<50} {detail}")

    # yaw
    check("yaw of identity is 0", abs(yaw_from_quat(0, 0, 0, 1)) < 1e-12)
    q90 = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    check("yaw of a 90 deg rotation", abs(yaw_from_quat(*q90) - math.pi / 2) < 1e-9,
          f"{math.degrees(yaw_from_quat(*q90)):.1f} deg")

    # a clean 10 m straight line at 60 Hz
    straight = [(i / 60.0, i * (10.0 / 600), 0.0, 0.0) for i in range(601)]
    check("straight 10 m measures 10 m", abs(path_length(straight) - 10.0) < 0.01,
          f"{path_length(straight):.3f} m")

    # THE jitter trap: 60 s stationary, +-5 mm, 60 Hz
    import random
    rng = random.Random(0)
    jitter = [(i / 60.0, rng.uniform(-0.005, 0.005), rng.uniform(-0.005, 0.005), 0.0)
              for i in range(3600)]
    raw = sum(math.dist((a[1], a[2]), (b[1], b[2]))
              for a, b in zip(jitter, jitter[1:]))
    filt = path_length(jitter)
    check("60 s of +-5 mm jitter stays under 0.05 m", filt < 0.05,
          f"{filt:.3f} m filtered, {raw:.1f} m raw")

    # decimation is by time and keeps the endpoints
    # 601 samples spanning 10 s at 60 Hz -> ~101 at 10 Hz, not ~60: the rate is
    # samples per SECOND, and the span is 10 s. (This assertion was wrong first
    # time round and the selftest caught it.)
    dec = decimate(straight, 10.0)
    check("60 Hz over 10 s decimates to ~101 samples",
          95 <= len(dec) <= 105, f"{len(dec)} of {len(straight)}")
    check("decimation keeps first and last sample",
          dec[0] == straight[0] and dec[-1] == straight[-1])
    check("a 2-sample trajectory survives decimation", len(decimate(straight[:2])) == 2)

    # a real move plus jitter still measures the move
    moving = [(i / 60.0, i * (5.0 / 600) + rng.uniform(-0.002, 0.002),
               rng.uniform(-0.002, 0.002), 0.0) for i in range(601)]
    check("5 m with noise measures ~5 m", abs(path_length(moving) - 5.0) < 0.2,
          f"{path_length(moving):.3f} m")

    # The other half of the speed gate: a genuine crawl must NOT be gated out.
    # 0.03 m/s is a fifth of cruise and still well above the 0.02 m/s floor, so
    # this fails the moment someone raises the threshold to chase jitter.
    crawl = [(i / 60.0, i * (0.03 / 60.0), 0.0, 0.0) for i in range(3601)]
    check("60 s crawl at 0.03 m/s still measures 1.8 m",
          abs(path_length(crawl) - 1.8) < 0.05, f"{path_length(crawl):.3f} m")

    # and a pure rotation contributes no distance, which is what SPL wants
    spin = [(i / 60.0, 0.0, 0.0, i * (math.pi / 600)) for i in range(601)]
    check("rotation in place adds no path length", path_length(spin) < 0.01,
          f"{path_length(spin):.4f} m")

    # cross-check tolerance
    class _Fake(SimTruth):
        def __init__(self, pose, model="ubot"):     # no ROS, no node
            self._latest = (0.0,) + tuple(pose)
            self.samples = [self._latest]
            self.model = model
            self.seen_frames = {}

    ok_flag, err = _Fake((1.000, 2.000, 0.0)).cross_check((1.005, 2.010, 0.0))
    check("11 mm disagreement accepted", ok_flag and err < CROSS_CHECK_TOLERANCE_M,
          f"{err * 100:.1f} cm")
    ok_flag, err = _Fake((1.0, 2.0, 0.0)).cross_check((1.5, 2.0, 0.0))
    check("50 cm disagreement refused (wrong entity)", not ok_flag, f"{err * 100:.0f} cm")
    ok_flag, err = _Fake((1.0, 2.0, 0.0)).cross_check(None)
    check("missing CLI sample is a refusal, not a pass", not ok_flag)

    # WHICH FRAME IS THE ROBOT. Measured on the live topic: the model pose is
    # world-referenced, the link poses are model-relative and read (0, 0, 0).
    f = _Fake((0.0, 0.0, 0.0))
    check("the model pose in a world frame is accepted",
          f._rank("ubot", "apartment") == 3)
    check("a model-RELATIVE link pose is refused, not used as a fallback",
          f._rank("ubot/base_footprint", "ubot") == 0,
          "it reads 0,0,0 - a whole episode parked at the origin")
    check("another entity is refused", f._rank("pedestrian_3", "apartment") == 0)
    check("an empty frame id is refused",
          f._rank("", "") == 0, "the unnamed dynamic_pose/info case")

    # and the selection picks the world pose out of a mixed message
    from types import SimpleNamespace as NS

    def _tf(child, parent, x, y):
        return NS(child_frame_id=child,
                  header=NS(frame_id=parent, stamp=NS(sec=1, nanosec=0)),
                  transform=NS(translation=NS(x=x, y=y, z=0.0),
                               rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0)))

    f2 = _Fake((0.0, 0.0, 0.0))
    f2._latest, f2.samples = None, []
    f2._on_tf(NS(transforms=[_tf("ubot/base_footprint", "ubot", 0.0, 0.0),
                             _tf("ubot", "apartment", 5.0, 3.0),
                             _tf("pedestrian_3", "apartment", 1.0, 1.0)]))
    check("mixed message -> the robot's world pose wins",
          f2.pose() == (5.0, 3.0, 0.0), str(f2.pose()))
    check("every frame seen is recorded for diagnostics",
          len(f2.seen_frames) == 3 and f2.seen_frames["pedestrian_3"] == 1)

    # the API readiness depends on
    check("pose() returns (x, y, yaw) for the motion proof",
          _Fake((1.0, 2.0, 0.5)).pose() == (1.0, 2.0, 0.5))
    check("sim truth is labelled ground truth, odom is not",
          SimTruth.is_ground_truth and not OdomTruth.is_ground_truth)

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    sys.exit(selftest() if args.selftest else ap.print_help())


if __name__ == "__main__":
    main()
