r"""Readiness gate: prove the robot CAN move before an episode is allowed to count.

    python3 -m ubot_eval.readiness --selftest      # decision logic, no ROS

--------------------------------------------------------------------------
WHY A MOTION PROOF, AND NOT JUST STATUS CHECKS
--------------------------------------------------------------------------
Every status check on this platform passes while the wheels are locked. It has
actually happened: a lateral-friction setting stalled the wheels in simulation
while `diff_drive_controller` kept integrating the commanded motion, so
odometry advanced, TF advanced, controllers reported `active`, nav2 was happy —
and the robot had not moved a millimetre (see src/ubot/CLAUDE.md, "Skid-steer
sim fix"). An episode run in that state is not a navigation failure; it is a
measurement of nothing, and scoring it as a failure is silent corruption.

So the last check is empirical: nudge the robot 0.6 s at 0.05 m/s and require
more than 10 mm of GROUND-TRUTH displacement. Odometry is not allowed to answer
this question, because odometry is the thing that lies.

A failed gate writes `outcome: not_started` — never `timeout`, never a failure.

--------------------------------------------------------------------------
STRUCTURE
--------------------------------------------------------------------------
The decision logic here is pure: `gate()` takes a callable that returns
`CheckResult`s and decides. The ROS collectors live in `RosReadiness`, whose
imports are lazy, so this module imports and selftests on a machine with no ROS.
That split is what lets the gate be tested AGAINST FAILURE, which the plan
demands: a gate never exercised in its failing state is not a gate.

--------------------------------------------------------------------------
TWO TRAPS THIS PLATFORM HAS ALREADY SPRUNG
--------------------------------------------------------------------------
* `TwistStamped` on `/diff_drive_controller/cmd_vel` must carry a SIM-TIME
  stamp. A wall-clock stamp yields "Velocity command timed out. Braking." and
  the nudge silently does nothing — which would read as locked wheels.
* The nudge must be undone. `stop()` publishes zero and is called in a `finally`,
  because leaving a velocity command live would have the robot creeping into the
  course before nav2 takes over.
"""
from __future__ import annotations

import argparse
import sys
import math
import time
from dataclasses import asdict, dataclass, field

#: Order matters: the first failure is the one reported, so cheap structural
#: checks come before the motion proof, which costs a second and moves the robot.
CHECK_ORDER = ("controllers", "tf", "scan", "nav2", "motion_proof")

REQUIRED_CONTROLLERS = ("joint_state_broadcaster", "diff_drive_controller")
REQUIRED_NAV2_NODES = ("controller_server", "planner_server", "behavior_server",
                       "bt_navigator", "smoother_server", "velocity_smoother",
                       "collision_monitor", "waypoint_follower")

NUDGE_SPEED_MS = 0.05
NUDGE_SECONDS = 0.6
MIN_DISPLACEMENT_M = 0.01
MIN_FINITE_SCANS = 3
#: Below this angular span a scan is a narrow forward sensor (the camera's
#: 64 deg), where an all-inf scan is legitimate. Above it, a 360 deg LiDAR,
#: where an all-inf scan indoors means the sensor is not working.
NARROW_FOV_RAD = 3.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    measured: float | None = None

    def __post_init__(self) -> None:
        if self.name not in CHECK_ORDER:
            raise ValueError(f"unknown check {self.name!r}; expected one of {CHECK_ORDER}")


@dataclass
class ReadinessReport:
    passed: bool
    failed_check: str | None
    attempts: int
    checks: list = field(default_factory=list)
    motion_proof_m: float | None = None

    def to_dict(self) -> dict:
        """Shape written into `EpisodeSummary.readiness`."""
        return dict(passed=self.passed, failed_check=self.failed_check,
                    attempts=self.attempts, motion_proof_m=self.motion_proof_m,
                    checks=[asdict(c) for c in self.checks])

    def line(self) -> str:
        if self.passed:
            return (f"readiness PASSED on attempt {self.attempts}"
                    + (f" (moved {self.motion_proof_m:.3f} m)"
                       if self.motion_proof_m is not None else ""))
        failed = next((c for c in self.checks if not c.passed), None)
        return (f"readiness FAILED after {self.attempts} attempt(s) at "
                f"{self.failed_check!r}: {failed.detail if failed else 'no detail'}")


def decide(results) -> ReadinessReport:
    """One attempt's verdict. First failure in CHECK_ORDER wins and is named."""
    results = list(results)
    by_name = {r.name: r for r in results}
    motion = by_name.get("motion_proof")
    for name in CHECK_ORDER:
        r = by_name.get(name)
        if r is None:
            return ReadinessReport(False, name, 1, results,
                                   motion.measured if motion else None)
        if not r.passed:
            return ReadinessReport(False, name, 1, results,
                                   motion.measured if motion else None)
    return ReadinessReport(True, None, 1, results, motion.measured if motion else None)


def gate(collect, attempts: int = 3, delay_s: float = 2.0, sleep=time.sleep) -> ReadinessReport:
    """Run `collect()` up to `attempts` times; return the first passing verdict.

    Retries exist because bring-up is a race, not because failure is acceptable:
    controllers spawn on a timer, nav2 activates asynchronously, and the first
    scan arrives when the sensor feels like it. What must NOT be retried away is
    a genuine inability to move, which is why the motion proof runs last and its
    failure is reported by name.
    """
    report = None
    for attempt in range(1, max(1, attempts) + 1):
        report = decide(collect())
        report.attempts = attempt
        if report.passed:
            return report
        if attempt < attempts:
            sleep(delay_s)
    return report


# --------------------------------------------------------------------------- #
class RosReadiness:
    """The live checks. All ROS imports are lazy so this module stays importable.

    `pose_fn` must return ground truth — `truth_source.SimTruth.pose()` in
    simulation. On the real robot there is none, so the caller passes EKF odom
    and the resulting `motion_proof_m` inherits its drift; that is recorded in
    the summary rather than hidden.
    """

    def __init__(self, node, *, pose_fn, scan_topic: str = "/scan_mono",
                 cmd_topic: str = "/diff_drive_controller/cmd_vel",
                 base_frame: str = "base_footprint", odom_frame: str = "odom",
                 controllers=REQUIRED_CONTROLLERS, nav2_nodes=REQUIRED_NAV2_NODES):
        self.node = node
        self.pose_fn = pose_fn
        self.scan_topic = scan_topic
        self.cmd_topic = cmd_topic
        self.base_frame = base_frame
        self.odom_frame = odom_frame
        self.controllers = tuple(controllers)
        self.nav2_nodes = tuple(nav2_nodes)
        self._scans = []

        from geometry_msgs.msg import TwistStamped
        from sensor_msgs.msg import LaserScan
        self._TwistStamped = TwistStamped
        self._pub = node.create_publisher(TwistStamped, cmd_topic, 10)
        node.create_subscription(LaserScan, scan_topic, self._on_scan, 10)

    # -- collectors ---------------------------------------------------------
    def _on_scan(self, msg) -> None:
        import math
        finite = sum(1 for r in msg.ranges if math.isfinite(r))
        self._scans.append(finite)
        self._scans = self._scans[-20:]
        # The angular span decides what an EMPTY scan means; see check_scan.
        self._scan_span_rad = float(msg.angle_max - msg.angle_min)

    def check_controllers(self) -> CheckResult:
        """Ask the service, never the log. Log scraping cannot see a later crash."""
        from controller_manager_msgs.srv import ListControllers
        cli = self.node.create_client(ListControllers, "/controller_manager/list_controllers")
        if not cli.wait_for_service(timeout_sec=5.0):
            return CheckResult("controllers", False, "controller_manager service absent")
        import rclpy
        fut = cli.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=5.0)
        if not fut.done() or fut.result() is None:
            return CheckResult("controllers", False, "list_controllers did not answer")
        state = {c.name: c.state for c in fut.result().controller}
        missing = [c for c in self.controllers if state.get(c) != "active"]
        if missing:
            return CheckResult("controllers", False,
                               f"not active: {', '.join(missing)} (saw {state or 'nothing'})")
        return CheckResult("controllers", True, f"{len(self.controllers)} controllers active")

    def check_tf(self, wall_timeout_s: float = 15.0) -> CheckResult:
        """odom -> base_footprint must exist AND advance; a frozen TF is not ready.

        Polls until the stamp CHANGES rather than comparing two samples a fixed
        wall-clock interval apart. Headless Gazebo on CPU runs at a real-time
        factor around 0.2, so a 0.4 s wall gap is under 0.1 s of sim time - often
        less than one publish interval, which reads as a frozen TF on a perfectly
        healthy robot. Measured: the sim clock reached 12.8 s during a 55 s
        settle. Everything here is in sim time; only the give-up is wall time.
        """
        import rclpy
        from tf2_ros import Buffer, TransformListener
        buf = Buffer()
        TransformListener(buf, self.node)
        first = None
        deadline = time.time() + wall_timeout_s
        while time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            try:
                tf = buf.lookup_transform(self.odom_frame, self.base_frame,
                                          rclpy.time.Time())
            except Exception:
                continue
            stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            if first is None:
                first = stamp
            elif stamp > first:
                return CheckResult("tf", True,
                                   f"TF advancing (+{stamp - first:.3f}s of sim time)")
        if first is None:
            return CheckResult("tf", False,
                               f"no {self.odom_frame} -> {self.base_frame} transform in "
                               f"{wall_timeout_s:.0f}s")
        return CheckResult("tf", False,
                           f"TF stuck at t={first:.3f} for {wall_timeout_s:.0f}s of wall "
                           "clock: the simulation is paused, or nothing is publishing odom")

    def check_scan(self) -> CheckResult:
        """Scans arriving, and - for a 360-degree sensor - actually hitting things.

        WHAT AN EMPTY SCAN MEANS DEPENDS ON THE FIELD OF VIEW, so this check
        does too:

        * A 360-degree LiDAR indoors ALWAYS sees something. An all-inf scan from
          one means the sensor is not really working, and that is exactly how
          this check caught a robot spawned 300 m outside the apartment and a
          gpu_lidar with no GL context - both published at full rate with every
          range inf.
        * A 64-degree forward camera scan does not. `hall_3m` is "a clear 3 m hop
          down the right-hand hall": the floor is below depth_to_scan's
          0.05-0.60 m height band and the far walls are past its 4 m trusted
          range, so an EMPTY scan is the correct output of a healthy sensor.
          Requiring finite returns there refuses the camera arm for driving down
          an empty corridor, which is the thing it is supposed to do.

        The narrow case is not unguarded: docker/run.sh check renders a frame and
        fails if its pixel variance is zero, which is what would catch a black
        camera. Here the requirement is that scans ARRIVE, on time.
        """
        import rclpy
        deadline = time.time() + 10.0
        while time.time() < deadline and len(self._scans) < MIN_FINITE_SCANS:
            rclpy.spin_once(self.node, timeout_sec=0.2)

        span = getattr(self, "_scan_span_rad", None)
        narrow = span is not None and span < NARROW_FOV_RAD

        if len(self._scans) < MIN_FINITE_SCANS:
            return CheckResult("scan", False,
                               f"{len(self._scans)} scans on {self.scan_topic}, "
                               f"need {MIN_FINITE_SCANS} - nothing is publishing")

        usable = [n for n in self._scans if n > 0]
        if narrow:
            return CheckResult(
                "scan", True,
                f"{len(self._scans)} scans on {self.scan_topic} "
                f"({math.degrees(span):.0f} deg FOV), "
                f"{usable[-1] if usable else 0} finite beams "
                f"(empty is legitimate at this FOV)")

        if len(usable) < MIN_FINITE_SCANS:
            return CheckResult("scan", False,
                               f"{len(usable)} scans with a finite range on "
                               f"{self.scan_topic}, need {MIN_FINITE_SCANS} "
                               f"(a {math.degrees(span):.0f} deg sensor indoors "
                               f"must see something)" if span else
                               f"{len(usable)} scans with a finite range on "
                               f"{self.scan_topic}, need {MIN_FINITE_SCANS}")
        return CheckResult("scan", True,
                           f"{len(usable)} usable scans, {usable[-1]} finite beams")

    def check_nav2(self) -> CheckResult:
        from lifecycle_msgs.srv import GetState
        import rclpy
        inactive = []
        for name in self.nav2_nodes:
            cli = self.node.create_client(GetState, f"/{name}/get_state")
            if not cli.wait_for_service(timeout_sec=3.0):
                inactive.append(f"{name}(absent)")
                continue
            fut = cli.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self.node, fut, timeout_sec=3.0)
            label = fut.result().current_state.label if fut.done() and fut.result() else "?"
            if label != "active":
                inactive.append(f"{name}({label})")
        if inactive:
            return CheckResult("nav2", False, "not active: " + ", ".join(inactive))

        # `active` is NOT the same as "will accept a goal". Observed: every node
        # reported active, the episode sent its goal, and bt_navigator aborted
        # 0.8 s later with
        #   Timed out while waiting for action server to acknowledge goal
        #   request for follow_path
        # because controller_server's action server was not serving yet - the
        # local costmap was still waiting for odom->base_footprint ("Tf has two
        # or more unconnected trees"). That produced `aborted` with a 0.09 m
        # path, which looks like a navigation failure and is really a bring-up
        # race. The gate exists precisely so that does not get recorded as a
        # result, so it has to wait for the action servers themselves.
        missing = self._action_servers_missing()
        if missing:
            return CheckResult("nav2", False,
                               "nodes active but action server(s) not accepting "
                               "goals: " + ", ".join(missing))
        return CheckResult("nav2", True,
                           f"{len(self.nav2_nodes)} nav2 nodes active, "
                           f"action servers accepting goals")

    def _action_servers_missing(self, timeout_s: float = 10.0) -> list:
        """Which of the action servers the episode depends on are not up yet.

        navigate_to_pose is what the episode calls; follow_path is what
        bt_navigator calls underneath and is the one that was late.
        """
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose, FollowPath

        missing = []
        for name, action_type in (("navigate_to_pose", NavigateToPose),
                                  ("follow_path", FollowPath)):
            client = ActionClient(self.node, action_type, name)
            try:
                if not client.wait_for_server(timeout_sec=timeout_s):
                    missing.append(name)
            finally:
                client.destroy()
        return missing

    def motion_proof(self) -> CheckResult:
        """Nudge and measure. The only check that cannot pass with locked wheels."""
        import math
        import rclpy
        start = self.pose_fn()
        if start is None:
            return CheckResult("motion_proof", False, "no pose source for the proof")
        # Drive for NUDGE_SECONDS of SIMULATED time, not wall time. Headless
        # Gazebo on CPU runs at a real-time factor near 0.2, so a 0.6 s wall-clock
        # nudge is ~0.14 s of driving: the robot moves ~2 mm, the proof reads
        # "the wheels are not turning", and a perfectly healthy robot is refused.
        # Measured exactly that: commanded 30 mm, moved 2.2 mm.
        # The wall-clock cap is the backstop for a genuinely paused clock, which
        # would otherwise loop here for ever.
        wall_cap = time.time() + NUDGE_SECONDS * 30
        try:
            msg = self._TwistStamped()
            msg.header.frame_id = self.base_frame
            msg.twist.linear.x = NUDGE_SPEED_MS
            deadline_ns = (self.node.get_clock().now().nanoseconds
                           + NUDGE_SECONDS * 1e9)
            while (self.node.get_clock().now().nanoseconds < deadline_ns
                   and time.time() < wall_cap):
                # Sim time, from the node clock: a wall-clock stamp is rejected
                # by the controller as stale and the robot never moves.
                msg.header.stamp = self.node.get_clock().now().to_msg()
                self._pub.publish(msg)
                rclpy.spin_once(self.node, timeout_sec=0.05)
        finally:
            self.stop()
        # Let the pose stream catch up, again in sim time.
        settle_ns = self.node.get_clock().now().nanoseconds + 0.5e9
        while (self.node.get_clock().now().nanoseconds < settle_ns
               and time.time() < wall_cap):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        end = self.pose_fn()
        if end is None:
            return CheckResult("motion_proof", False, "pose source died during the proof")
        moved = math.dist(start[:2], end[:2])
        if moved <= MIN_DISPLACEMENT_M:
            return CheckResult("motion_proof", False,
                               f"commanded {NUDGE_SPEED_MS} m/s for {NUDGE_SECONDS}s and "
                               f"moved {moved * 1000:.1f} mm: the wheels are not turning",
                               moved)
        return CheckResult("motion_proof", True, f"moved {moved * 1000:.1f} mm", moved)

    def stop(self) -> None:
        msg = self._TwistStamped()
        msg.header.frame_id = self.base_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        for _ in range(3):
            self._pub.publish(msg)

    def collect(self):
        """One attempt's worth of checks, cheapest first, stopping at the first failure.

        Short-circuits deliberately: there is no point nudging a robot whose
        controllers are not up, and the motion proof is the only check with a
        side effect.
        """
        results = []
        for fn in (self.check_controllers, self.check_tf, self.check_scan,
                   self.check_nav2, self.motion_proof):
            r = fn()
            results.append(r)
            if not r.passed:
                break
        return results


# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("SELFTEST  ubot_eval.readiness (decision logic, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<50} {detail}")

    def ready(**overrides):
        base = {n: True for n in CHECK_ORDER}
        base.update(overrides)
        out = []
        for n in CHECK_ORDER:
            passed = base[n]
            measured = (0.031 if passed else 0.003) if n == "motion_proof" else None
            out.append(CheckResult(n, passed, "ok" if passed else "forced failure", measured))
        return out

    r = decide(ready())
    check("all checks pass -> passed, nothing named",
          r.passed and r.failed_check is None, f"moved {r.motion_proof_m:.3f} m")

    r = decide(ready(controllers=False))
    check("controllers down -> names 'controllers'", r.failed_check == "controllers", r.line()[:46])
    r = decide(ready(tf=False))
    check("frozen TF -> names 'tf'", r.failed_check == "tf")
    r = decide(ready(scan=False))
    check("no usable scans -> names 'scan'", r.failed_check == "scan")
    r = decide(ready(nav2=False))
    check("nav2 not active -> names 'nav2'", r.failed_check == "nav2")

    # FIELD OF VIEW DECIDES WHAT AN EMPTY SCAN MEANS. This relaxes a check that
    # has caught real faults, so both directions are pinned here.
    class _FakeScanner:
        def __init__(self, span_rad, finite_counts):
            self.scan_topic = "/scan_test"
            self._scans = list(finite_counts)
            self._scan_span_rad = span_rad
            self.node = None

        check_scan = RosReadiness.check_scan

        def _spin(self, *a, **k):
            pass

    import types
    fake_rclpy = types.SimpleNamespace(spin_once=lambda *a, **k: None)
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict)         else __builtins__.__import__

    def fake_import(name, *a, **k):
        return fake_rclpy if name == "rclpy" else real_import(name, *a, **k)

    if isinstance(__builtins__, dict):
        __builtins__["__import__"] = fake_import
    else:
        __builtins__.__import__ = fake_import
    try:
        # A 360 deg LiDAR seeing nothing indoors is BROKEN - this is what caught
        # the robot spawned 300 m outside the map and the GL-less gpu_lidar.
        wide_empty = _FakeScanner(2 * math.pi, [0, 0, 0, 0]).check_scan()
        check("360 deg scan with no finite range still FAILS",
              not wide_empty.passed, wide_empty.detail[:60])

        wide_ok = _FakeScanner(2 * math.pi, [275, 275, 275]).check_scan()
        check("360 deg scan with returns passes", wide_ok.passed)

        # A 64 deg camera scan down an empty corridor is WORKING.
        narrow_empty = _FakeScanner(math.radians(64), [0, 0, 0, 0]).check_scan()
        check("64 deg scan with no finite range PASSES (empty corridor)",
              narrow_empty.passed, narrow_empty.detail[:60])

        # ...but only if it is actually publishing.
        narrow_silent = _FakeScanner(math.radians(64), []).check_scan()
        check("a silent narrow scan still fails - nothing is publishing",
              not narrow_silent.passed, narrow_silent.detail[:60])
    finally:
        if isinstance(__builtins__, dict):
            __builtins__["__import__"] = real_import
        else:
            __builtins__.__import__ = real_import

    # THE test the plan demands: everything green, wheels locked.
    r = decide(ready(motion_proof=False))
    check("locked wheels caught by the motion proof",
          r.failed_check == "motion_proof" and not r.passed,
          f"measured {r.motion_proof_m * 1000:.1f} mm")

    r = decide(ready(controllers=False, motion_proof=False))
    check("first failure in order wins", r.failed_check == "controllers")

    partial = [CheckResult("controllers", True, "ok")]
    r = decide(partial)
    check("a check that never ran fails the gate", r.failed_check == "tf")

    # retries: fails once, then passes
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        return ready() if state["n"] >= 2 else ready(nav2=False)

    r = gate(flaky, attempts=3, delay_s=0, sleep=lambda _s: None)
    check("retry lets a slow bring-up pass", r.passed and r.attempts == 2,
          f"attempts={r.attempts}")

    state["n"] = 0

    def never():
        state["n"] += 1
        return ready(motion_proof=False)

    r = gate(never, attempts=3, delay_s=0, sleep=lambda _s: None)
    check("a real inability to move is not retried away",
          not r.passed and r.attempts == 3 and r.failed_check == "motion_proof")

    d = r.to_dict()
    check("to_dict matches the summary's readiness field",
          set(d) == {"passed", "failed_check", "attempts", "motion_proof_m", "checks"}
          and d["checks"][0]["name"] == "controllers")

    # the contract with schema: a failed gate is never scored as a failure
    from ubot_eval.schema import classify
    outcome, _ = classify(readiness_ok=r.passed, collided=False, nav2_status=None,
                          gt_distance_to_goal_m=None, timed_out=True)
    check("failed gate -> not_started, not timeout", outcome == "not_started", outcome)

    try:
        CheckResult("typo_check", True)
        check("unknown check name rejected", False)
    except ValueError:
        check("unknown check name rejected", True)

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
