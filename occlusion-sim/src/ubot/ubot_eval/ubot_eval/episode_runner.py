r"""One episode: one goal attempt, gated, recorded, and classified.

    python3 -m ubot_eval.episode_runner --selftest        # summary logic, no ROS

    ros2 run ubot_eval episode_runner --ros-args \
        -p campaign:=pilot -p arm:=mono_baseline -p world:=apartment \
        -p world_name:=apartment -p course:=hall_3m -p trial:=1 \
        -p goal_x:=3.0 -p goal_y:=0.0 -p goal_yaw:=0.0

--------------------------------------------------------------------------
THE SHAPE OF AN EPISODE
--------------------------------------------------------------------------
    readiness gate  ->  start bag  ->  NavigateToPose  ->  classify  ->  write

Every step can end the episode, and the outcome says which one did. The gate
comes first because an episode run on a robot that cannot move is not a
navigation failure, it is a measurement of nothing: that writes `not_started`,
which is excluded from the success rate and counted separately.

--------------------------------------------------------------------------
GOALS ARE IN `odom`
--------------------------------------------------------------------------
There is no SLAM and no map in this configuration, so `map` does not exist and a
goal expressed in it would never resolve. Goals are relative to wherever the
robot started, which is also why `campaign.py` respawns at a fixed pose between
episodes rather than trusting the robot to return.

THE GROUND TRUTH IS IN THE GAZEBO WORLD FRAME, THE GOAL IS NOT. Comparing the
two directly is meaningless, and it is meaningless in a way that LOOKS like a
result: a robot that drove the course perfectly was reported 3.465 m from its
goal and written down as `false_success`, because the world pose (5, 3) was
being measured against the odom point (3, 0). `goal_in_world` converts the goal
using the world->odom transform captured at episode start. That transform is
fixed - odom does not move once odometry has started - so capturing it once is
correct, and odometry drift still shows up exactly where it should: as the
robot believing it arrived while ground truth says otherwise.

--------------------------------------------------------------------------
WHY THE BAG IS A SUBPROCESS
--------------------------------------------------------------------------
`ros2 bag record` is the command documented in docs/experiments/logging.md, it
survives this node crashing, and it needs no rosbag2_py API compatibility. The
episode owns the process and stops it in a `finally`, so an aborted run still
leaves a playable bag.

--------------------------------------------------------------------------
WHAT IS PURE HERE
--------------------------------------------------------------------------
`build_summary` turns observations into the verdict and the summary, and it
takes those observations as data. That is the part worth testing, and it is
tested with no ROS, no Gazebo and no robot - including the case where the gate
failed and the metrics must NOT be written.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import types
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ubot_eval import contact as contact_mod
from ubot_eval import schema
from ubot_eval.schema import EpisodeSummary, bag_dirname, classify, episode_path, spl_term

#: Recorded for every episode. `/scan` and `/scan_mono` are both listed so the
#: lidar and camera arms produce identically shaped bags, and the scan
#: comparison in the research tree can read one bag per episode.
BAG_TOPICS = ("/tf", "/tf_static", "/gt/tf", "/odometry/filtered",
              "/diff_drive_controller/odom", "/scan", "/scan_mono",
              "/bumper/contacts", "/plan", "/cmd_vel", "/cmd_vel_raw",
              "/diff_drive_controller/cmd_vel", "/local_costmap/costmap",
              "/navigate_to_pose/_action/feedback",
              # /clock is small and makes the real-time factor directly readable
              # instead of inferred from (receive time, header stamp) pairs.
              "/clock")

#: Image streams, recorded ONLY when UBOT_BAG_IMAGES is set.
#:
#: They bracket the perception chain and are the only way to say WHICH node is
#: the bottleneck:
#:     image 30 Hz, depth 30 Hz, scan 1.6 Hz -> depth_to_scan is slow
#:     image 30 Hz, depth 1.6 Hz             -> the depth node is slow
#:
#: They are NOT recorded by default because they change the experiment. At
#: 640x480 and 30 Hz, /camera/rgb/image_raw alone is ~27 MB/s - one 110 s
#: episode produced a 2.7 GB bag - and that write load competes with Gazebo and
#: nav2 for the same cores. Turned on for every arm, BOTH arms regressed
#: together: lidar_ref, which had reached the goal on every previous run, fell
#: to a 0.21 m path and an abort, matching mono_baseline almost exactly. A
#: diagnostic that perturbs what it measures is worse than no diagnostic.
#:
#:     UBOT_BAG_IMAGES=1   # when diagnosing the perception chain, not in a campaign
BAG_TOPICS_IMAGES = ("/camera/rgb/image_raw", "/mono/depth")

DEFAULT_TIMEOUT_S = 180.0
#: Perception is "dead" if no scan arrives for this long while an episode runs.
SCAN_STALE_S = 2.0


@dataclass
class EpisodeSpec:
    campaign: str
    arm: str
    world: str
    course: str
    trial: int
    goal_x: float
    goal_y: float
    goal_yaw: float = 0.0
    runs_root: str = "runs"
    venue: str = "sim"
    timeout_s: float = DEFAULT_TIMEOUT_S
    shortest_path_m: float | None = None
    goal_frame: str = "odom"

    @property
    def directory(self) -> Path:
        return episode_path(self.runs_root, self.campaign, self.world,
                            self.course, self.arm, self.trial)


@dataclass
class Observations:
    """Everything the episode measured. Deliberately plain data."""

    readiness: dict = field(default_factory=dict)
    nav2_status: str | None = None
    gt_distance_to_goal_m: float | None = None
    gt_available: bool = False
    path_length_m: float | None = None
    time_to_goal_s: float | None = None
    timed_out: bool = False
    recoveries: int | None = None
    min_clearance_m: float | None = None
    collisions: dict = field(default_factory=dict)
    perception: dict = field(default_factory=lambda: {"ok": True})
    truth: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    notes: str = ""


def build_summary(spec: EpisodeSpec, obs: Observations) -> EpisodeSummary:
    """Observations -> verdict -> summary. Pure, and the reason this is testable.

    Two rules live here rather than in the caller, because both are ways a
    campaign silently corrupts itself:

    * a `not_started` episode carries NO metrics. The robot never ran, so a path
      length or a time would be fiction, and `schema.validate` refuses them.
    * a collided episode still records `reached_goal`, so the safety-versus-
      completion trade-off stays visible instead of collapsing into one number.
    """
    collided = bool((obs.collisions or {}).get("count", 0))
    readiness_ok = bool((obs.readiness or {}).get("passed", False))
    outcome, reason = classify(
        readiness_ok=readiness_ok,
        collided=collided,
        nav2_status=obs.nav2_status,
        gt_distance_to_goal_m=obs.gt_distance_to_goal_m,
        timed_out=obs.timed_out,
        perception_ok=bool((obs.perception or {}).get("ok", True)),
    )

    reached_goal = (obs.gt_distance_to_goal_m is not None
                    and obs.gt_distance_to_goal_m <= schema.GOAL_TOLERANCE_M)
    if obs.gt_distance_to_goal_m is None:
        reached_goal = (obs.nav2_status or "").upper() == "SUCCEEDED"

    warnings: list[str] = []
    spl = None
    distance_ratio = None
    if outcome != "not_started":
        spl, warn = spl_term(outcome in schema.SUCCESS_OUTCOMES,
                             spec.shortest_path_m, obs.path_length_m)
        if warn:
            warnings.append(warn)
        if spec.shortest_path_m and obs.path_length_m and spec.shortest_path_m > 0:
            distance_ratio = obs.path_length_m / spec.shortest_path_m

    summary = EpisodeSummary(
        campaign=spec.campaign, arm=spec.arm, world=spec.world, course=spec.course,
        trial=spec.trial, outcome=outcome, outcome_reason=reason, venue=spec.venue,
        goal=dict(x=spec.goal_x, y=spec.goal_y, yaw=spec.goal_yaw, frame=spec.goal_frame),
        nav2_status=obs.nav2_status,
        reached_goal=bool(reached_goal), gt_checked=bool(obs.gt_available),
        gt_distance_to_goal_m=obs.gt_distance_to_goal_m,
        readiness=obs.readiness or {}, truth=obs.truth or {},
        perception=obs.perception or {}, collisions=obs.collisions or {},
        artifacts=obs.artifacts or {}, provenance=obs.provenance or {},
        warnings=warnings, notes=obs.notes)

    if outcome == "not_started":
        # Never write metrics for an episode that never ran.
        summary.path_length_m = None
        summary.shortest_path_m = None
        summary.time_to_goal_s = None
        summary.spl = None
        summary.distance_ratio = None
        summary.recoveries = None
        summary.min_clearance_m = None
    else:
        summary.path_length_m = obs.path_length_m
        summary.shortest_path_m = spec.shortest_path_m
        summary.time_to_goal_s = obs.time_to_goal_s
        summary.spl = spl
        summary.distance_ratio = distance_ratio
        summary.recoveries = obs.recoveries
        summary.min_clearance_m = obs.min_clearance_m
    return summary


def goal_in_world(goal_xy, world_T_odom):
    """Put a goal expressed in `odom` into the Gazebo world frame.

    `world_T_odom` is (x, y, yaw): where the odom ORIGIN sits in the world. It
    is NOT the robot's start pose, though the two coincide when odometry is
    zeroed at the spawn point and the robot has not moved since. Readiness
    nudges the robot ~30 mm before the episode starts, so they are not identical
    and the distinction is worth keeping.

    Returns (x, y) in the world frame.
    """
    ox, oy, oyaw = world_T_odom
    gx, gy = goal_xy
    c, s = math.cos(oyaw), math.sin(oyaw)
    return (ox + c * gx - s * gy,
            oy + s * gx + c * gy)


def world_to_odom(goal_world, world_T_odom):
    """A WORLD point expressed in `odom` - the inverse of `goal_in_world`.

    courses.yaml states goals in WORLD coordinates, next to the world-frame
    spawn pose, which is the only way they can be written down before a run
    exists. nav2 is given them in `odom`, because without SLAM there is no map
    frame. Skipping this conversion sends the robot to the odom point that
    happens to share the world point's numbers: an observed episode asked for
    odom (5, 3) instead of odom (0.03, 3.0), which is outside the 10 m rolling
    costmap, and the planner failed with
      worldToMap failed: mx,my: 206,169, size_x,size_y: 200,200
    """
    ox, oy, oyaw = world_T_odom
    dx, dy = goal_world[0] - ox, goal_world[1] - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    return (c * dx - s * dy, s * dx + c * dy)


def compose_world_T_odom(world_pose, odom_pose):
    """world->odom from one simultaneous pair of poses of the SAME body.

    `world_pose` is the robot in the Gazebo world (ground truth); `odom_pose` is
    the same robot as odometry reports it. Both are (x, y, yaw). Returns the
    odom origin's pose in the world, as (x, y, yaw).
    """
    wx, wy, wyaw = world_pose
    ox, oy, oyaw = odom_pose
    yaw = wyaw - oyaw
    c, s = math.cos(yaw), math.sin(yaw)
    # world = T * odom  =>  T = world * inv(odom)
    return (wx - (c * ox - s * oy),
            wy - (s * ox + c * oy),
            yaw)


def gz_model_pose(model: str, world: str, timeout_s: float = 5.0):
    """(x, y) from the Gazebo CLI. Blocks ~1 s, so this is a cross-check only.

    The pose stream is what samples a trajectory; this exists to catch the
    stream reporting a different entity, which would make every path metric a
    measurement of the wrong body.
    """
    # No -w/--world: `gz model` has no such flag, and passing it makes the call
    # fail with empty output. A failed cross-check is not an error anywhere - it
    # reads as "unverified" - so this stayed invisible until the summary showed
    # cross_check_cli: null on a run where everything else worked.
    try:
        out = subprocess.run(["gz", "model", "-m", model, "-p"],
                             capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return None
    # Real output, captured from gz-tools on Jazzy:
    #
    #     Model: [18]
    #       - Name: ubot
    #       - Pose [ XYZ (m) ] [ RPY (rad) ]:
    #         [5.000000 0.000000 -0.020001]
    #         [0.000000 0.000000 1.570800]
    #
    # WHITESPACE separated - not commas, not pipes. The XYZ line comes first and
    # RPY second, so the first bracketed row of numbers wins. The `- Pose [ XYZ`
    # header does not match because it starts with a dash, and SDF warnings start
    # with "Warning".
    nums = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        parts = line.strip("[]").replace("|", " ").replace(",", " ").split()
        try:
            nums = [float(v) for v in parts[:3]]
            break
        except ValueError:
            continue
    return (nums[0], nums[1]) if len(nums) >= 2 else None


def bag_topics(env=None) -> tuple:
    """The topics to record. Images only when explicitly asked for."""
    import os as _os
    env = _os.environ if env is None else env
    if str(env.get("UBOT_BAG_IMAGES", "")).lower() in ("1", "true", "yes"):
        return BAG_TOPICS + BAG_TOPICS_IMAGES
    return BAG_TOPICS


class BagRecorder:
    """`ros2 bag record` as an owned subprocess. Stops in a finally, always."""

    def __init__(self, episode_dir: Path, topics=None, when=None):
        self.path = Path(episode_dir) / bag_dirname(when or datetime.now())
        topics = bag_topics() if topics is None else topics
        self.topics = list(topics)
        self.proc = None

    def start(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ros2", "bag", "record", "-o", str(self.path), *self.topics]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        return self.path

    def stop(self, grace_s: float = 5.0) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None


# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("SELFTEST  ubot_eval.episode_runner (summary logic, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<52} {detail}")

    spec = EpisodeSpec(campaign="pilot", arm="mono_baseline", world="apartment",
                       course="hall_3m", trial=3, goal_x=3.0, goal_y=0.0,
                       shortest_path_m=3.0)
    passed_gate = {"passed": True, "failed_check": None, "attempts": 1,
                   "motion_proof_m": 0.031}

    s = build_summary(spec, Observations(
        readiness=passed_gate, nav2_status="SUCCEEDED", gt_distance_to_goal_m=0.04,
        gt_available=True, path_length_m=3.3, time_to_goal_s=26.0))
    check("clean arrival -> reached, SPL 0.91",
          s.outcome == "reached" and abs(s.spl - 3.0 / 3.3) < 1e-9,
          f"spl={s.spl:.3f} ratio={s.distance_ratio:.3f}")
    check("summary validates", not schema.validate(s.to_dict()))

    s = build_summary(spec, Observations(
        readiness=passed_gate, nav2_status="SUCCEEDED", gt_distance_to_goal_m=1.8,
        gt_available=True, path_length_m=4.0, time_to_goal_s=40.0))
    check("nav2 SUCCEEDED 1.8 m away -> false_success, SPL 0",
          s.outcome == "false_success" and s.spl == 0.0 and not s.reached_goal)

    s = build_summary(spec, Observations(
        readiness=passed_gate, nav2_status="SUCCEEDED", gt_distance_to_goal_m=0.05,
        gt_available=True, path_length_m=3.1, time_to_goal_s=30.0,
        collisions=contact_mod.summarize([1.0, 1.02], path_length_m=3.1)))
    check("collision outranks arrival, but reached_goal is kept",
          s.outcome == "collided" and s.reached_goal and s.spl == 0.0,
          f"{s.collisions['count']} event(s), reached_goal={s.reached_goal}")
    check("collided summary validates", not schema.validate(s.to_dict()))

    failed_gate = {"passed": False, "failed_check": "motion_proof", "attempts": 3,
                   "motion_proof_m": 0.003}
    s = build_summary(spec, Observations(
        readiness=failed_gate, nav2_status=None, timed_out=True,
        path_length_m=0.4, time_to_goal_s=180.0, recoveries=2))
    check("failed gate -> not_started, never timeout", s.outcome == "not_started", s.outcome)
    check("not_started carries NO metrics",
          s.path_length_m is None and s.spl is None and s.time_to_goal_s is None
          and s.recoveries is None)
    check("not_started summary validates", not schema.validate(s.to_dict()))

    s = build_summary(spec, Observations(
        readiness=passed_gate, nav2_status=None, timed_out=True,
        path_length_m=2.0, time_to_goal_s=180.0))
    check("clock cap -> timeout with metrics kept",
          s.outcome == "timeout" and s.spl == 0.0 and s.path_length_m == 2.0)

    s = build_summary(spec, Observations(
        readiness=passed_gate, nav2_status="SUCCEEDED", gt_distance_to_goal_m=None,
        gt_available=False, path_length_m=3.2, time_to_goal_s=25.0))
    check("no ground truth -> reached but gt_checked False",
          s.outcome == "reached" and not s.gt_checked and s.reached_goal)

    s = build_summary(spec, Observations(
        readiness=passed_gate, nav2_status="SUCCEEDED", gt_distance_to_goal_m=0.03,
        gt_available=True, path_length_m=2.5, time_to_goal_s=20.0))
    check("path shorter than the reference warns", any("reference path" in w
                                                       for w in s.warnings),
          (s.warnings or ["none"])[0][:40] + "...")

    # a goal with no reference path still classifies, but cannot claim an SPL
    bare = EpisodeSpec(campaign="pilot", arm="a", world="w", course="c", trial=1,
                       goal_x=1.0, goal_y=0.0)
    s = build_summary(bare, Observations(
        readiness=passed_gate, nav2_status="SUCCEEDED", gt_distance_to_goal_m=0.02,
        gt_available=True, path_length_m=1.1, time_to_goal_s=9.0))
    check("success without a reference path is caught by validate",
          s.outcome == "reached" and bool(schema.validate(s.to_dict())),
          "validate refuses to let it become a silent SPL 0")

    # PERCEPTION WINDOW. The bug: worst_gap_s was a running max seeded at the
    # first scan ever seen, so bring-up counted against navigation. A real
    # episode was failed with worst_gap_s=12.366 while its bag - covering the
    # navigation window - had a worst gap of 0.93 s.
    class _GapProbe:
        """Just the gap bookkeeping, driven by a fake clock."""
        scan_topic = "/scan_mono"

        def __init__(self):
            self.t = 0.0
            self.last_scan_t = None
            self.worst_gap_s = 0.0
            self.startup_worst_gap_s = 0.0
            self._gaps = []
            self._nav_worst_gap_s = 0.0
            self._nav_worst_gap_at_s = None
            self._gap_t0 = None
            self._gap_window_open = False

        _now = lambda self: self.t
        reset_perception_window = EpisodeNode.reset_perception_window
        perception_report = EpisodeNode.perception_report
        _on_scan = EpisodeNode._on_scan

        def scan_at(self, t):
            # A plain local class, NOT types.SimpleNamespace: selftest() imports
            # `types` further down, which makes the name function-local for the
            # whole function, so referencing it here raises NameError.
            class _Scan:
                ranges = []
                range_min = 0.0
            self.t = t
            self._on_scan(_Scan())

    probe = _GapProbe()
    for t in (0.0, 12.4, 12.5):          # a 12.4 s gap during BRING-UP
        probe.scan_at(t)
    probe.reset_perception_window()
    for i in range(1, 21):               # then a healthy 2.5 Hz stream
        probe.scan_at(12.5 + i * 0.4)
    rep = probe.perception_report()
    check("a bring-up gap no longer fails the episode",
          rep["ok"] is True, f"nav worst {rep['worst_gap_s']}s")
    check("...and is still reported, as startup_worst_gap_s",
          abs(rep["startup_worst_gap_s"] - 12.4) < 1e-6,
          f"{rep['startup_worst_gap_s']}s")
    check("the navigation window reports the real rate",
          abs(rep["rate_hz"] - 2.5) < 0.01, f"{rep['rate_hz']} Hz")
    check("a distribution is reported, not one number",
          rep["median_gap_s"] is not None and rep["p95_gap_s"] is not None
          and rep["gaps_seen"] == 20,
          f"median {rep['median_gap_s']}, p95 {rep['p95_gap_s']}, n={rep['gaps_seen']}")

    # A stall INSIDE the window must still fail, and be located.
    probe = _GapProbe()
    probe.scan_at(0.0)
    probe.reset_perception_window()
    for i in range(1, 6):
        probe.scan_at(i * 0.4)
    probe.scan_at(2.0 + 9.0)             # 9 s hole mid-navigation
    rep = probe.perception_report()
    check("a stall during navigation still fails the episode", rep["ok"] is False,
          f"worst {rep['worst_gap_s']}s")
    check("...and says WHEN it happened",
          rep["worst_gap_at_s"] is not None and rep["worst_gap_at_s"] > 10.0,
          f"at {rep['worst_gap_at_s']}s")

    # Frame maths. THE REGRESSION THIS PREVENTS, with the real numbers: the
    # robot spawns at world (5, 0) facing +y (yaw pi/2), odometry starts at
    # zero, and the goal is 3 m straight ahead as (3, 0) in `odom`. The robot
    # drives the course perfectly and ends at world (5, 3). Comparing the world
    # pose against the raw odom goal gives dist((5,3),(3,0)) = 3.61 m, and the
    # episode is written down as `false_success` - a perfect run recorded as a
    # navigation failure. Through the transform the distance is 0.
    start_world = (5.0, 0.0, math.pi / 2)
    odom_at_start = (0.0, 0.0, 0.0)
    T = compose_world_T_odom(start_world, odom_at_start)
    gw = goal_in_world((3.0, 0.0), T)
    check("odom goal 3 m ahead maps to world (5, 3)",
          abs(gw[0] - 5.0) < 1e-9 and abs(gw[1] - 3.0) < 1e-9, str(gw))
    check("a perfect run measures 0 m from the goal, not 3.61 m",
          math.dist((5.0, 3.0), gw) < 1e-9,
          f"{math.dist((5.0, 3.0), gw):.4f} m")
    check("the naive comparison really would have said ~3.6 m",
          abs(math.dist((5.0, 3.0), (3.0, 0.0)) - 3.6056) < 1e-3)

    # Odometry that has drifted: odom says the robot is at the goal, truth says
    # it is a metre short. That MUST still read as a miss.
    T2 = compose_world_T_odom((5.0, 0.0, math.pi / 2), (0.0, 0.0, 0.0))
    check("drift is still caught: truth 1 m short of the goal reads 1 m",
          abs(math.dist((5.0, 2.0), goal_in_world((3.0, 0.0), T2)) - 1.0) < 1e-9)

    # THE CAMPAIGN BUG. courses.yaml gives hall_3m's goal as world (5, 3) and
    # the spawn as world (5, 0) facing +y. Sent as-is under an odom header that
    # means odom (5, 3), which is 5 m to the robot's RIGHT and outside the 10 m
    # rolling costmap - the planner failed with "worldToMap failed: mx,my:
    # 206,169, size_x,size_y: 200,200" and every campaign episode aborted.
    T_camp = compose_world_T_odom((5.0, 0.0, math.pi / 2), (0.0, 0.0, 0.0))
    g_odom = world_to_odom((5.0, 3.0), T_camp)
    check("the world goal (5, 3) becomes odom (3, 0), straight ahead",
          abs(g_odom[0] - 3.0) < 1e-9 and abs(g_odom[1]) < 1e-9, str(g_odom))
    check("...and is NOT sent as odom (5, 3), 5 m off to the side",
          math.dist(g_odom, (5.0, 3.0)) > 1.0)
    check("world_to_odom inverts goal_in_world exactly",
          all(abs(a - b) < 1e-9 for a, b in
              zip(goal_in_world(world_to_odom((1.5, 8.0), T_camp), T_camp),
                  (1.5, 8.0))))

    # A non-zero odom reading at episode start, which is what the readiness
    # nudge leaves behind: truth (5, 0.03) facing +y while odom already says
    # 0.03 m travelled. The odom ORIGIN is therefore world (5, 0), and a goal
    # 3.03 m along odom x is world y = 3.03. Getting the sign wrong and adding
    # the offset twice would give 3.06.
    T3 = compose_world_T_odom((5.0, 0.03, math.pi / 2), (0.03, 0.0, 0.0))
    check("odom origin lands at the world point odometry was zeroed from",
          abs(T3[0] - 5.0) < 1e-9 and abs(T3[1]) < 1e-9, str(T3))
    check("odom origin offset is applied once, not doubled",
          abs(goal_in_world((3.03, 0.0), T3)[1] - 3.03) < 1e-6,
          str(goal_in_world((3.03, 0.0), T3)))

    # The CLI cross-check parser. THIS FIXTURE IS REAL - captured verbatim from
    # `gz model -m ubot -p` on Jazzy, not written from memory. The first version
    # of this test used an invented pipe-separated format, so it passed while the
    # parser returned None against the actual whitespace-separated output; and a
    # None cross-check is "unverified", not a failure, so nothing in the episode
    # summary gave the mistake away. Do not edit this string to match the parser.
    import types
    real_run = subprocess.run
    gz_text = """Warning [Utils.cc:132] [/sdf/model[@name="ubot"]] is deprecated.
Requesting state for world [apartment]...

Model: [18]
  - Name: ubot
  - Pose [ XYZ (m) ] [ RPY (rad) ]:
    [5.000000 0.020000 -0.020001]
    [0.000000 0.000000 1.570800]
"""
    try:
        subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout=gz_text, returncode=0)
        pose = gz_model_pose("ubot", "apartment")
        check("gz CLI pose parsed from REAL whitespace-separated output",
              pose is not None and abs(pose[0] - 5.0) < 1e-9 and abs(pose[1] - 0.02) < 1e-9,
              str(pose))
        check("the RPY row is not mistaken for XYZ",
              pose is not None and abs(pose[0] - 0.0) > 1e-6)
        subprocess.run = lambda *a, **k: types.SimpleNamespace(stdout="no such model\n",
                                                              returncode=1)
        check("unparseable output gives None, not a wrong pose",
              gz_model_pose("ubot", "apartment") is None)
    finally:
        subprocess.run = real_run

    b = BagRecorder(Path("/tmp/x"), when=datetime(2026, 9, 16, 14, 32, 0))
    check("bag directory follows logging.md",
          b.path.name == "navigation_20260916_143200", b.path.name)
    check("both scans are recorded, so arms produce identical bags",
          "/scan" in BAG_TOPICS and "/scan_mono" in BAG_TOPICS)
    check("ground truth is in the bag", "/gt/tf" in BAG_TOPICS)
    # IMAGE STREAMS MUST BE OPT-IN. Recording /camera/rgb/image_raw for every
    # arm put ~27 MB/s of writes beside Gazebo and nav2, and BOTH arms regressed
    # together - lidar_ref went from reaching the goal every time to a 0.21 m
    # path and an abort. The measurement must not move the thing measured.
    check("images are NOT recorded by default",
          all(t not in bag_topics({}) for t in BAG_TOPICS_IMAGES))
    check("...and are recorded when explicitly asked for",
          all(t in bag_topics({"UBOT_BAG_IMAGES": "1"}) for t in BAG_TOPICS_IMAGES))
    check("the default set still has what every episode needs",
          all(t in bag_topics({}) for t in ("/gt/tf", "/scan", "/scan_mono", "/clock")))

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
class EpisodeNode:
    """The live half. Every ROS import is lazy so the module selftests anywhere.

    Owns exactly one goal attempt and returns the summary for it. It does not
    decide anything itself: it collects observations and hands them to
    `build_summary`, which is the tested part.
    """

    def __init__(self, spec: EpisodeSpec, *, scan_topic: str = "/scan_mono",
                 model: str = "ubot", world_name: str | None = None,
                 odom_topic: str = "/odometry/filtered"):
        import rclpy
        from rclpy.node import Node
        from rclpy.action import ActionClient
        from nav2_msgs.action import NavigateToPose
        from sensor_msgs.msg import LaserScan
        from nav_msgs.msg import Odometry

        from ubot_eval.contact import ContactMonitor
        from ubot_eval.readiness import RosReadiness, gate
        from ubot_eval.truth_source import OdomTruth, SimTruth

        self.rclpy = rclpy
        self.spec = spec
        self.scan_topic = scan_topic
        self.model = model
        self.world_name = world_name or spec.world
        self.node = Node("episode_runner")
        self.node.set_parameters([rclpy.parameter.Parameter(
            "use_sim_time", rclpy.Parameter.Type.BOOL, spec.venue == "sim")])

        self.truth = (SimTruth(self.node, model=model, world_name=self.world_name)
                      if spec.venue == "sim" else OdomTruth(self.node))
        self.contacts = ContactMonitor(self.node)
        self.readiness = RosReadiness(self.node, pose_fn=self.truth.pose,
                                      scan_topic=scan_topic)
        self._gate = gate
        self._client = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        self._NavigateToPose = NavigateToPose

        # Perception watchdog and clearance, both straight off the scan the arm
        # actually navigates on.
        self.last_scan_t: float | None = None
        self.worst_gap_s = 0.0
        self.startup_worst_gap_s = 0.0
        self._gaps: list = []
        self._nav_worst_gap_s = 0.0
        self._nav_worst_gap_at_s: float | None = None
        self._gap_t0: float | None = None
        self._gap_window_open = False
        self.min_clearance_m: float | None = None
        self.recoveries = 0
        self.distance_remaining: float | None = None
        self.node.create_subscription(LaserScan, scan_topic, self._on_scan, 10)

        # The robot as ODOMETRY sees it, needed to place the odom-framed goal in
        # the world for the ground-truth check. This is the same topic nav2
        # plans against, so the goal is converted through the very estimate nav2
        # is trusting - which is what makes a drift-induced arrival visible.
        self.odom_pose: tuple[float, float, float] | None = None
        self.node.create_subscription(Odometry, odom_topic, self._on_odom, 10)

    # -- callbacks ----------------------------------------------------------
    def _on_odom(self, msg) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_pose = (float(p.x), float(p.y), float(yaw))

    def _on_scan(self, msg) -> None:
        now = self._now()
        if self.last_scan_t is not None:
            gap = now - self.last_scan_t
            self.worst_gap_s = max(self.worst_gap_s, gap)
            # Every gap, not just the worst, and only those inside the
            # NAVIGATION window - see reset_perception_window().
            if self._gap_window_open:
                self._gaps.append(gap)
                if gap > self._nav_worst_gap_s:
                    self._nav_worst_gap_s = gap
                    # `self._gap_t0 or now` would be WRONG: a t0 of exactly 0.0
                    # is falsy, so the offset collapsed to zero and every gap
                    # was reported as happening at the start of the window.
                    base = self._gap_t0 if self._gap_t0 is not None else now
                    self._nav_worst_gap_at_s = now - base
        self.last_scan_t = now
        finite = [r for r in msg.ranges if math.isfinite(r) and r > msg.range_min]
        if finite:
            nearest = min(finite)
            if self.min_clearance_m is None or nearest < self.min_clearance_m:
                self.min_clearance_m = float(nearest)

    def _on_feedback(self, msg) -> None:
        fb = msg.feedback
        self.recoveries = int(getattr(fb, "number_of_recoveries", 0) or 0)
        self.distance_remaining = float(getattr(fb, "distance_remaining", 0.0) or 0.0)

    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def reset_perception_window(self) -> None:
        """Start measuring scan gaps from HERE - the moment navigation begins.

        WHY THIS EXISTS. `worst_gap_s` used to be a running maximum seeded at
        the very first scan, so it included bring-up: nav2 activating, the
        readiness nudge, the depth model loading its weights. A camera episode
        was condemned with `perception ok=False, worst_gap_s=12.366` while its
        BAG - which starts at the same moment as this window - showed a worst
        gap of 0.93 s and a lidar running at exactly 10.00 Hz. The 12.4 s gap
        was real, but it happened before the robot was ever asked to move, and
        judging navigation by it is measuring the wrong interval.

        The pre-navigation maximum is still reported, as `startup_worst_gap_s`,
        because a stack that takes 12 s to produce its first scans is worth
        knowing about - it is just not a navigation failure.
        """
        self.startup_worst_gap_s = self.worst_gap_s
        self.worst_gap_s = 0.0
        self._gaps = []
        self._nav_worst_gap_s = 0.0
        self._nav_worst_gap_at_s = None
        self._gap_t0 = self._now()
        self._gap_window_open = True

    def perception_report(self) -> dict:
        """Scan timing over the navigation window: a distribution, not one number."""
        gaps = sorted(self._gaps)
        n = len(gaps)

        def pct(q):
            if not gaps:
                return None
            if n == 1:
                return round(gaps[0], 3)
            pos = (n - 1) * q
            lo, hi = math.floor(pos), math.ceil(pos)
            val = gaps[lo] + (gaps[hi] - gaps[lo]) * (pos - lo) if lo != hi else gaps[lo]
            return round(val, 3)

        return dict(
            ok=self._nav_worst_gap_s < SCAN_STALE_S,
            topic=self.scan_topic,
            scans_seen=self.last_scan_t is not None,
            worst_gap_s=round(self._nav_worst_gap_s, 3),
            worst_gap_at_s=(None if self._nav_worst_gap_at_s is None
                            else round(self._nav_worst_gap_at_s, 2)),
            median_gap_s=pct(0.50),
            p95_gap_s=pct(0.95),
            gaps_seen=n,
            rate_hz=(round(n / sum(gaps), 2) if gaps and sum(gaps) > 0 else None),
            # Bring-up, measured before navigation and reported separately so it
            # cannot be mistaken for a navigation fault.
            startup_worst_gap_s=round(getattr(self, "startup_worst_gap_s", 0.0), 3),
        )

    # -- the episode --------------------------------------------------------
    def _goal_msg(self, world_T_odom=None):
        """The nav2 goal, always published in `odom`.

        A spec whose goal_frame is `world` is CONVERTED here rather than
        relabelled: courses.yaml writes goals in world coordinates because that
        is the frame the spawn pose is in, and nav2 has no map frame to resolve
        them against. Sending the world numbers under an odom header aims at a
        different place entirely.
        """
        from geometry_msgs.msg import PoseStamped
        goal = self._NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = "odom"
        pose.header.stamp = self.node.get_clock().now().to_msg()

        x, y, yaw = self.spec.goal_x, self.spec.goal_y, self.spec.goal_yaw
        if self.spec.goal_frame == "world":
            if world_T_odom is None:
                raise RuntimeError(
                    "a world-frame goal needs world_T_odom, which needs one "
                    "ground-truth pose and one odometry sample. Refusing to "
                    "send the world numbers under an odom header.")
            x, y = world_to_odom((x, y), world_T_odom)
            yaw = yaw - world_T_odom[2]

        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        goal.pose = pose
        self._goal_odom = (float(x), float(y), float(yaw))
        return goal

    def _cross_check_truth(self) -> dict:
        """One CLI sample against the stream; refuse if they disagree by >2 cm."""
        info = dict(source="sim_gt" if self.spec.venue == "sim" else "odom",
                    topic=getattr(self.truth, "topic", None),
                    is_ground_truth=bool(getattr(self.truth, "is_ground_truth", False)))
        if self.spec.venue != "sim":
            return info
        cli = gz_model_pose(self.model, self.world_name)
        ok, err = self.truth.cross_check(cli)
        # THREE states, not two. "Could not check" is not "disagreed":
        #   True  - the stream and the CLI agree; the pose is verified
        #   False - they disagree; the stream is reporting another entity and
        #           every path metric would be measured against the wrong body
        #   None  - no CLI sample (gz unavailable); unverified, but not wrong
        # Collapsing None into False refuses healthy episodes; collapsing it into
        # True quietly drops the only check that catches a mis-identified robot.
        info.update(cross_check_error_m=None if err != err else round(err, 4),
                    cross_check_ok=None if cli is None else bool(ok),
                    cross_check_cli=list(cli) if cli else None)
        return info

    def run(self) -> EpisodeSummary:
        rclpy = self.rclpy
        spec = self.spec
        obs = Observations()

        if spec.venue == "sim":
            # 60 s, not 15: this is the FIRST thing that touches the stack, so
            # it absorbs whatever is left of bring-up. At 15 s an episode gave up
            # on /gt/tf while the bridge was still starting and wrote no summary
            # at all - a host-speed artefact recorded as a missing episode.
            self.truth.wait_for_stream(timeout_s=60.0)

        report = self._gate(self.readiness.collect, attempts=3, delay_s=2.0)
        obs.readiness = report.to_dict()
        self.node.get_logger().info(report.line())

        obs.truth = self._cross_check_truth()
        if obs.readiness.get("passed") and obs.truth.get("cross_check_ok") is False:
            # Measured disagreement: the stream is tracking something that is not
            # this robot, so every path metric would describe the wrong body.
            # Not a navigation failure - we could not measure, so we did not run.
            obs.readiness = dict(obs.readiness, passed=False,
                                 failed_check="truth_cross_check")
            self.node.get_logger().error(
                f"ground truth disagrees with the CLI by "
                f"{obs.truth.get('cross_check_error_m')} m; refusing the episode")
        elif obs.truth.get("cross_check_ok") is None and spec.venue == "sim":
            # Unverified is not wrong. The episode runs, and the summary carries
            # cross_check_ok: null so the analysis can see which episodes had
            # their ground truth confirmed and which merely had it.
            self.node.get_logger().warn(
                "ground truth could not be cross-checked (`gz model` gave no pose); "
                "running anyway, recorded as unverified")

        if not obs.readiness.get("passed"):
            summary = build_summary(spec, obs)
            schema.write_summary(spec.directory, summary)
            return summary

        recorder = BagRecorder(spec.directory)
        start_pose = self.truth.pose()
        # world->odom, captured once at the start. Both poses describe the same
        # body at the same moment, so this is the fixed offset between the two
        # frames; anything that changes afterwards is drift, which is the thing
        # being measured, not something to compensate for.
        world_T_odom = None
        odom_at_start = self.odom_pose      # snapshot: self.odom_pose keeps moving
        if start_pose is not None and odom_at_start is not None:
            world_T_odom = compose_world_T_odom(start_pose, odom_at_start)
        t0 = self._now()
        self.contacts.reset()
        try:
            obs.artifacts = {"bag": recorder.start().name}
            # Measure perception over the SAME interval the bag covers, so the
            # summary and the recording can be checked against each other.
            self.reset_perception_window()
            if not self._client.wait_for_server(timeout_sec=20.0):
                obs.nav2_status = "NO_SERVER"
            else:
                obs.nav2_status = self._navigate(t0, world_T_odom)
        finally:
            recorder.stop()

        obs.time_to_goal_s = round(self._now() - t0, 3)
        obs.timed_out = obs.time_to_goal_s >= spec.timeout_s
        obs.path_length_m = round(self.truth.path_length(), 4)
        obs.recoveries = self.recoveries
        obs.min_clearance_m = self.min_clearance_m
        obs.collisions = self.contacts.summary(obs.path_length_m)
        obs.perception = self.perception_report()
        end = self.truth.pose()
        obs.gt_available = bool(getattr(self.truth, "is_ground_truth", False)) and end is not None
        goal_world = None
        if end is not None and spec.goal_frame == "world":
            # Already in the frame ground truth is measured in: compare directly
            # rather than round-tripping it through odom and back.
            goal_world = (spec.goal_x, spec.goal_y)
            obs.gt_distance_to_goal_m = round(math.dist((end[0], end[1]), goal_world), 4)
        elif end is not None and world_T_odom is not None:
            # The goal is in `odom`; ground truth is in the Gazebo world. Compare
            # them only after putting both in the same frame.
            goal_world = goal_in_world((spec.goal_x, spec.goal_y), world_T_odom)
            obs.gt_distance_to_goal_m = round(math.dist((end[0], end[1]), goal_world), 4)
        elif end is not None:
            # No odometry sample means no way to relate the frames. Refuse to
            # produce the number rather than compute a meaningless one: a bogus
            # distance here silently becomes `false_success` or a fake arrival.
            obs.gt_distance_to_goal_m = None
            obs.warnings = list(obs.warnings or []) + [
                "no odometry sample at episode start: the odom-framed goal could "
                "not be placed in the world frame, so gt_distance_to_goal_m is "
                "not reported"]
        obs.provenance = dict(scan_topic=self.scan_topic, model=self.model,
                              world_name=self.world_name,
                              start_pose=list(start_pose) if start_pose else None,
                              # The START value, snapshotted before navigating.
                              # Recording the live self.odom_pose here labelled
                              # the FINAL reading as the start one: a run showed
                              # odom_start_pose 2.8555 on a 2.89 m course, which
                              # makes the transform look wrong when it is not.
                              odom_start_pose=list(odom_at_start) if odom_at_start else None,
                              odom_end_pose=list(self.odom_pose) if self.odom_pose else None,
                              world_T_odom=list(world_T_odom) if world_T_odom else None,
                              goal_world=list(goal_world) if goal_world else None)

        summary = build_summary(spec, obs)
        schema.write_summary(spec.directory, summary)
        if hasattr(self.truth, "write_trajectory"):
            self.truth.write_trajectory(spec.directory)
        self.node.get_logger().info(
            f"{summary.outcome}: {summary.outcome_reason} "
            f"(path {obs.path_length_m} m, {obs.collisions.get('count', 0)} collision(s))")
        return summary

    def _navigate(self, t0: float, world_T_odom=None) -> str:
        """Send the goal, pump callbacks until it finishes or the clock runs out."""
        rclpy = self.rclpy
        from action_msgs.msg import GoalStatus

        send = self._client.send_goal_async(self._goal_msg(world_T_odom),
                                            feedback_callback=self._on_feedback)
        rclpy.spin_until_future_complete(self.node, send, timeout_sec=15.0)
        handle = send.result() if send.done() else None
        if handle is None or not handle.accepted:
            return "REJECTED"

        result_future = handle.get_result_async()
        while rclpy.ok() and not result_future.done():
            if self._now() - t0 >= self.spec.timeout_s:
                handle.cancel_goal_async()
                rclpy.spin_once(self.node, timeout_sec=1.0)
                return "TIMEOUT"
            rclpy.spin_once(self.node, timeout_sec=0.1)
        status = result_future.result().status if result_future.done() else None
        return {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
                GoalStatus.STATUS_ABORTED: "ABORTED",
                GoalStatus.STATUS_CANCELED: "CANCELED"}.get(status, f"STATUS_{status}")

    def destroy(self) -> None:
        self.node.destroy_node()


def run_episode(spec: EpisodeSpec, **kwargs) -> EpisodeSummary:
    """Spin up rclpy, run one episode, always tear down."""
    import rclpy
    rclpy.init()
    node = None
    try:
        node = EpisodeNode(spec, **kwargs)
        return node.run()
    finally:
        if node is not None:
            node.destroy()
        rclpy.try_shutdown()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--campaign", default="pilot")
    ap.add_argument("--arm", default="mono_baseline")
    ap.add_argument("--world", default="apartment")
    ap.add_argument("--world-name", default=None,
                    help="<world name> inside the sdf; defaults to --world")
    ap.add_argument("--course", default="hall_3m")
    ap.add_argument("--trial", type=int, default=1)
    ap.add_argument("--goal-x", type=float, default=3.0)
    ap.add_argument("--goal-frame", default="odom", choices=["odom", "world"],
                    help="frame the goal coordinates are in. courses.yaml uses "
                         "world; a world goal is converted to odom before it is "
                         "sent, never relabelled.")
    ap.add_argument("--goal-y", type=float, default=0.0)
    ap.add_argument("--goal-yaw", type=float, default=0.0)
    ap.add_argument("--shortest-path-m", type=float, default=None)
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--venue", default="sim", choices=["sim", "real"])
    ap.add_argument("--scan-topic", default="/scan_mono")
    args, _ = ap.parse_known_args()

    if args.selftest:
        sys.exit(selftest())

    spec = EpisodeSpec(campaign=args.campaign, arm=args.arm, world=args.world,
                       course=args.course, trial=args.trial, goal_x=args.goal_x,
                       goal_y=args.goal_y, goal_yaw=args.goal_yaw,
                       runs_root=args.runs_root, venue=args.venue,
                       goal_frame=args.goal_frame,
                       timeout_s=args.timeout_s,
                       shortest_path_m=args.shortest_path_m)
    summary = run_episode(spec, scan_topic=args.scan_topic,
                          world_name=args.world_name or args.world)
    print(f"{summary.outcome}: {summary.outcome_reason}")
    sys.exit(0 if summary.outcome in schema.SUCCESS_OUTCOMES else 1)


if __name__ == "__main__":
    main()
