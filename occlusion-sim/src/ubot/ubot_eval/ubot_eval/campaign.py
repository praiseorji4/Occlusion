r"""The trial matrix: which episodes to run, in what order, and what is left.

    python3 -m ubot_eval.campaign --selftest          # planning logic, no ROS
    python3 -m ubot_eval.campaign --selftest-all      # every module, in order
    python3 -m ubot_eval.campaign --plan --trials 25  # what a campaign would run
    python3 -m ubot_eval.campaign --power --pairs 75  # BEFORE collecting anything

--------------------------------------------------------------------------
THE UNIT IS THE PAIR, NOT THE EPISODE
--------------------------------------------------------------------------
`pair_key` is world/course/trial and deliberately excludes the arm, because the
statistics resample pairs: the same course and trial index run under every arm.
So the plan is built pair-first, and `check_pairing` refuses a plan where the
arms do not all cover the same set of keys. An arm missing a pair is not a
smaller sample, it is a broken pairing.

--------------------------------------------------------------------------
ORDER IS SHUFFLED WITHIN A PAIR, NEVER ACROSS
--------------------------------------------------------------------------
A full matrix is ~19 hours of simulation. Anything that drifts over that time -
thermal throttling, a memory leak in Gazebo, the machine being used for
something else - would otherwise land on whichever arm runs last and look like
an effect. Arms are shuffled inside each pair_key with a fixed seed, so the
order is reproducible and drift is spread across arms rather than aligned with
them.

--------------------------------------------------------------------------
RESUME IS BY EVIDENCE, NOT BY COUNTING
--------------------------------------------------------------------------
`--resume` skips an episode only when its `episode.json` exists AND validates.
A half-written or corrupt summary is re-run, because the alternative is a
campaign that silently contains a truncated episode.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ubot_eval import schema

MODULES_IN_ORDER = ("schema", "readiness", "truth_source", "contact",
                    "episode_runner")

DEFAULT_SEED = 20260916


def default_config_dir() -> Path:
    """The installed `share/ubot_eval/config`, or the source tree when not built.

    `__file__`-relative paths point at site-packages once the package is
    installed, where the yaml files are NOT - setup.py installs them under
    share/. Getting this wrong makes --plan work only from the source tree.
    """
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("ubot_eval")) / "config"
    except Exception:
        return Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True)
class PlannedEpisode:
    world: str          # the key in courses.yaml, e.g. "parking_garage"
    world_file: str     # the .sdf file name
    world_name: str     # the <world name> inside it
    course: str
    trial: int
    arm: str
    launch: str
    goal: dict
    spawn: dict
    shortest_path_m: float | None
    timeout_s: float

    @property
    def pair_key(self) -> str:
        return schema.pair_key(self.world, self.course, self.trial)

    def directory(self, runs_root, campaign: str) -> Path:
        return schema.episode_path(runs_root, campaign, self.world,
                                   self.course, self.arm, self.trial)


def load_config(config_dir=None) -> tuple[dict, dict]:
    import yaml
    config_dir = Path(config_dir or default_config_dir())
    courses = yaml.safe_load((config_dir / "courses.yaml").read_text(encoding="utf-8"))
    arms = yaml.safe_load((config_dir / "arms.yaml").read_text(encoding="utf-8"))
    return courses, arms


def resolve_arm(name: str, arms: dict) -> dict:
    """An arm's parameters, with `inherits` merged left to right.

    mono_AB is defined as the union of mono_A and mono_B rather than a third
    copy of their values, so a change to A cannot silently fail to reach AB.
    """
    spec = dict(arms["arms"][name])
    params: dict = {}
    for parent in spec.get("inherits", []):
        params.update(resolve_arm(parent, arms).get("params", {}))
    params.update(spec.get("params", {}) or {})
    spec["params"] = params
    if "launch" not in spec:
        for parent in spec.get("inherits", []):
            spec["launch"] = arms["arms"][parent]["launch"]
            break
    return spec


def build_plan(courses: dict, arms: dict, *, trials: int, arm_names=None,
               worlds=None, courses_filter=None, seed: int = DEFAULT_SEED,
               shuffle_arms: bool = True) -> list:
    """Every episode to run, pair by pair, arms shuffled inside each pair."""
    arm_names = list(arm_names or arms["arms"].keys())
    plan: list[PlannedEpisode] = []
    for world_key, world in courses["worlds"].items():
        if worlds and world_key not in worlds:
            continue
        for course_name, course in world["courses"].items():
            if courses_filter and course_name not in courses_filter:
                continue
            for trial in range(1, trials + 1):
                key = schema.pair_key(world_key, course_name, trial)
                order = list(arm_names)
                if shuffle_arms:
                    # Seeded per pair_key: reproducible, and different pairs get
                    # different orders so drift cannot align with one arm.
                    random.Random(f"{seed}:{key}").shuffle(order)
                for arm in order:
                    spec = resolve_arm(arm, arms)
                    plan.append(PlannedEpisode(
                        world=world_key, world_file=world["world"],
                        world_name=world["world_name"], course=course_name,
                        trial=trial, arm=arm, launch=spec["launch"],
                        goal=course["goal"], spawn=course["spawn"],
                        shortest_path_m=course.get("shortest_path_m"),
                        timeout_s=float(course.get("timeout_s", 180))))
    return plan


def check_pairing(plan, expected_arms=None) -> list:
    """Problems that would break the paired statistics. Empty list means usable.

    Compares every arm against the UNION of pair keys, not against the biggest
    arm: measuring against the biggest cannot see an arm that is missing
    entirely, and an arm that never ran is exactly the failure that silently
    halves a paired comparison. `expected_arms` catches that case outright.
    """
    by_arm: dict = {}
    for ep in plan:
        by_arm.setdefault(ep.arm, set()).add(ep.pair_key)
    problems = []
    if not by_arm:
        return ["empty plan"]

    if expected_arms:
        for arm in expected_arms:
            if arm not in by_arm:
                problems.append(f"{arm} has no episodes at all")

    all_keys = set().union(*by_arm.values())
    for arm, keys in sorted(by_arm.items()):
        missing = all_keys - keys
        if missing:
            problems.append(f"{arm} is missing {len(missing)} pair(s), "
                            f"e.g. {sorted(missing)[0]}")
    return problems


def completed(plan, runs_root, campaign: str) -> set:
    """Pair/arm identifiers whose summary exists, validates, AND has data.

    A `not_started` episode does NOT count as complete. The readiness gate
    writes it precisely to say the robot could not run and the episode measured
    nothing about navigation, so banking it would leave a permanent hole: the
    pair survives for one arm and not the other, and paired statistics resample
    pairs. Observed at 1 refusal in 6 episodes from a controller-spawn race, so
    on a 300-episode matrix this is not a corner case - it would silently delete
    a sixth of the sample.

    Retrying costs one episode; keeping it costs a pair, permanently.
    """
    done = set()
    for ep in plan:
        path = ep.directory(runs_root, campaign) / "episode.json"
        if not path.exists():
            continue
        try:
            summary = schema.read_summary(path)  # validates; raises if unusable
        except Exception:
            continue                             # corrupt -> re-run it
        outcome = getattr(summary, "outcome", None) or (
            summary.get("outcome") if isinstance(summary, dict) else None)
        if outcome == "not_started":
            continue                             # refused -> not data -> re-run
        done.add((ep.pair_key, ep.arm))
    return done


def remaining(plan, runs_root, campaign: str) -> list:
    done = completed(plan, runs_root, campaign)
    return [ep for ep in plan if (ep.pair_key, ep.arm) not in done]


def detectable_sr_difference(pairs: int, discordance: float = 0.30,
                             alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest success-rate difference McNemar can detect, at `pairs` pairs.

    Run this BEFORE collecting. If the effect you care about is smaller than
    this, the answer is more trials, not more analysis.

    Paired proportions, so what matters is the DISCORDANT pairs - the ones where
    the arms disagree. `discordance` is their assumed share; 0.30 is typical when
    two arms differ by 10-20 points at a success rate near 0.8.
    """
    z_alpha = 1.959963985   # two-sided 0.05
    z_beta = 0.841621234    # 0.80
    return (z_alpha + z_beta) * math.sqrt(discordance / max(pairs, 1))


# Settings that must be IDENTICAL in both arms' nav2 params, because each one
# is an advantage if one arm gets it and the other does not. The value itself is
# a judgement call; the two files agreeing is not.
PARITY_KEYS = (
    ("bt_navigator.ros__parameters.default_server_timeout",
     "a longer fuse buys retries before an abort"),
    ("bt_navigator.ros__parameters.wait_for_service_timeout",
     "same, for service discovery"),
    ("controller_server.ros__parameters.costmap_update_timeout",
     "how long the controller waits for a costmap before aborting; more patience "
     "is an advantage, and it is what the camera arm needed"),
    ("controller_server.ros__parameters.general_goal_checker.xy_goal_tolerance",
     "a looser goal checker means stopping sooner: better SPL and better time"),
    ("controller_server.ros__parameters.general_goal_checker.yaw_goal_tolerance",
     "as above, for heading"),
    ("global_costmap.global_costmap.ros__parameters.global_frame",
     "planning in different frames makes the arms incomparable"),
    ("global_costmap.global_costmap.ros__parameters.width",
     "a bigger global costmap is a longer planning horizon"),
    ("global_costmap.global_costmap.ros__parameters.height",
     "as above"),
    ("local_costmap.local_costmap.ros__parameters.global_frame",
     "as above, locally"),
)


def _dig(tree, dotted):
    node = tree
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return KeyError
        node = node[part]
    return node


def parity_violations(lidar_params: dict, mono_params: dict) -> list:
    """Parity-critical settings where the two arms' params disagree.

    The experiment asks what dropping the LiDAR costs. Any setting that helps
    navigation and differs between the arms contaminates that answer, and the
    difference is invisible in the results - it just shows up as one arm being
    better. Comments asking the next person to keep two files in step are not
    enough, so this is checked.

    Settings that are deliberately NOT here: the observation source, the range
    caps and the behaviour list. Those SHOULD differ - they are the sensor and
    its consequences, which is the thing being measured.
    """
    out = []
    for key, why in PARITY_KEYS:
        a, b = _dig(lidar_params, key), _dig(mono_params, key)
        if a is KeyError and b is KeyError:
            continue        # neither sets it: both take nav2's default
        if a != b:
            out.append(f"{key}: lidar={a!r} mono={b!r} ({why})")
    return out


def check_params_parity(lidar_path=None, mono_path=None) -> list:
    """parity_violations against the installed params files."""
    import yaml
    from ament_index_python.packages import get_package_share_directory

    if lidar_path is None:
        lidar_path = (Path(get_package_share_directory("ubot_bringup"))
                      / "config" / "nav2_params_lidar_odom.yaml")
    if mono_path is None:
        mono_path = (Path(get_package_share_directory("ubot_mono_nav"))
                     / "config" / "nav2_params_mono.yaml")
    with open(lidar_path) as fh:
        lidar = yaml.safe_load(fh)
    with open(mono_path) as fh:
        mono = yaml.safe_load(fh)
    return parity_violations(lidar, mono)


def run_selftests(modules=MODULES_IN_ORDER) -> int:
    """Every module's selftest, in dependency order. Reports all, fails if any did."""
    failed = []
    for name in modules:
        proc = subprocess.run([sys.executable, "-m", f"ubot_eval.{name}", "--selftest"],
                              capture_output=True, text=True)
        tail = (proc.stdout.strip().splitlines() or ["no output"])[-1]
        print(f"  {name:<16} {tail}")
        if proc.returncode != 0:
            failed.append(name)
    print(f"\n{len(modules) - len(failed)} of {len(modules)} modules passed")
    return 1 if failed else 0


# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #

#: Keys in an arm's `params` that are LAUNCH arguments rather than node
#: parameters. Anything else needs a parameter overlay, which the launch files
#: do not accept yet - `unsupported_overrides` says so out loud rather than
#: starting a run whose arm silently equals the baseline.
LAUNCH_ARG_KEYS = ("use_slam", "use_nav2", "use_rviz", "headless",
                   "start_gazebo", "sensor_profile")

#: How long to let a stack come up before the episode's own readiness gate takes
#: over. The gate retries, so this only has to be in the right neighbourhood.
# A full stack - Gazebo, the bridge, ros2_control, nav2's eight lifecycle nodes -
# needs longer than this on a CPU-bound host. 25 s left the episode asking for
# ground truth while the bridge was still starting, and it gave up. The episode
# re-checks readiness anyway, so a generous settle costs one wait per episode and
# buys the difference between a measurement and a bring-up race.
SETTLE_S = 45.0


def _arg(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def arm_scan_topic(arm_name: str, arm_spec: dict) -> str:
    """Which scan the arm navigates on - and so which one readiness watches."""
    explicit = arm_spec.get("scan_topic")
    if explicit:
        return explicit
    return "/scan" if arm_spec.get("launch", "").startswith("ubot_bringup") else "/scan_mono"


def unsupported_overrides(arm_spec: dict) -> list:
    """Node-parameter overrides this harness cannot apply yet.

    Part 2's arms (mono_A, mono_B, mono_AB) need a nav2 parameter overlay, which
    means a `params_file` argument on the mono launch. Until that exists, running
    those arms would produce episodes labelled mono_A that are bit-identical to
    the baseline - the worst possible outcome, because the result would look like
    "the improvement does nothing".
    """
    return sorted(k for k in (arm_spec.get("params") or {}) if k not in LAUNCH_ARG_KEYS)


def launch_command(ep: PlannedEpisode, arm_spec: dict, headless: bool = True) -> list:
    """The `ros2 launch` command for one episode's stack."""
    pkg, launch_file = ep.launch.split("/", 1)
    cmd = ["ros2", "launch", pkg, launch_file,
           f"world:={ep.world_file}",
           f"world_name:={ep.world_name}",
           f"spawn_x:={_arg(ep.spawn['x'])}",
           f"spawn_y:={_arg(ep.spawn['y'])}",
           f"spawn_z:={_arg(ep.spawn.get('z', 0.1))}",
           f"spawn_yaw:={_arg(ep.spawn['yaw'])}"]
    for key, value in (arm_spec.get("params") or {}).items():
        if key in LAUNCH_ARG_KEYS:
            cmd.append(f"{key}:={_arg(value)}")
    if headless and not any(c.startswith("headless:=") for c in cmd):
        cmd.append("headless:=true")
        # rviz is a GUI for a human watching one run. In a campaign it is dead
        # weight that competes for the same cores as Gazebo and nav2, and it
        # died with SIGABRT mid-episode on a headless host anyway.
        if not any(c.startswith("use_rviz:=") for c in cmd):
            cmd.append("use_rviz:=false")
    # Where there is no display at all - a container - ogre2 must render
    # offscreen through EGL or Gazebo segfaults on startup and every sensor
    # topic stays silent. This is host-dependent, not arm-dependent, so it comes
    # from the environment rather than from arms.yaml: setting it per arm would
    # be a parity risk for no reason.
    if os.environ.get("UBOT_HEADLESS_RENDERING", "").lower() in ("1", "true", "yes"):
        if not any(c.startswith("headless_rendering:=") for c in cmd):
            cmd.append("headless_rendering:=true")
    return cmd


def episode_command(ep: PlannedEpisode, arm_spec: dict, *, campaign: str,
                    runs_root: str, venue: str = "sim") -> list:
    return ["ros2", "run", "ubot_eval", "episode_runner",
            "--campaign", campaign, "--arm", ep.arm, "--world", ep.world,
            "--world-name", ep.world_name, "--course", ep.course,
            "--trial", str(ep.trial), "--goal-x", str(ep.goal["x"]),
            "--goal-y", str(ep.goal["y"]), "--goal-yaw", str(ep.goal.get("yaw", 0.0)),
            "--timeout-s", str(ep.timeout_s), "--runs-root", str(runs_root),
            "--venue", venue, "--scan-topic", arm_scan_topic(ep.arm, arm_spec),
            # courses.yaml states goals in WORLD coordinates, beside the
            # world-frame spawn pose. episode_runner converts to odom using the
            # transform it captures at episode start. Without this flag the
            # world numbers go out under an odom header and the robot aims at a
            # different point - observed as
            #   GridBased plugin failed to plan from (3.66, 1.40) to (5.00, 3.00)
            #   worldToMap failed: mx,my: 206,169, size_x,size_y: 200,200
            # because odom (5, 3) is outside the 10 m rolling costmap.
            "--goal-frame", "world",
            *(["--shortest-path-m", str(ep.shortest_path_m)]
              if ep.shortest_path_m else [])]


#: Every node a stack leaves behind, by the pattern that actually matches it.
#: `nav2` alone is NOT enough: opennav_docking's binary path contains no "nav2",
#: so it survived, stayed ACTIVE, and the next episode's lifecycle_manager got
#: "Unable to start transition 1 from current state active" and aborted the whole
#: bring-up - reported as a stack failure and recorded as `not_started`.
SURVIVOR_PATTERNS = (
    "gz sim", "ruby.*gz", "parameter_bridge", "ros_gz", "nav2", "opennav",
    "static_transform_publisher", "lifecycle_manager", "twist_stamper",
    "controller_manager", "robot_state_publisher", "ekf_node", "slam_toolbox",
    "rviz2", "component_container",
)


def kill_survivors(settle_s: float = 5.0) -> list:
    """Kill leftovers from a previous stack. Returns whatever refused to die.

    Between episodes this is the difference between a measurement and a mixture
    of two runs. `LaunchedStack.stop` signals its own process group, which is
    the right thing and usually enough, but a node that ignored SIGINT, or a
    stack from an interrupted campaign, outlives it.

    Deliberately NOT matching this process: the patterns name nodes, never
    "python" or "ubot_eval", so a campaign cannot kill itself - which this
    project has already done once with `pkill -f ros2`.
    """
    import time as _time
    for pattern in SURVIVOR_PATTERNS:
        subprocess.run(["pkill", "-9", "-f", pattern],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _time.sleep(settle_s)
    check = subprocess.run(
        ["pgrep", "-af", "gz sim|opennav|nav2_controller|static_transform_publisher"],
        capture_output=True, text=True)
    return [line for line in check.stdout.splitlines()
            if "pgrep" not in line and "ubot_eval" not in line]


class LaunchedStack:
    """One `ros2 launch`, owned as a PROCESS GROUP so it can be killed whole.

    `ros2 launch` spawns Gazebo, nav2, rviz and more; terminating just the parent
    leaves those orphaned, and the next episode then talks to a stale stack. So
    the child gets its own session and the whole group is signalled.

    Not `pkill -f sim.launch.py`: run from `bash -lc`, that pattern matches the
    shell's own command line and kills the shell - which this project has
    already been bitten by.
    """

    def __init__(self, cmd, log_path):
        self.cmd = list(cmd)
        self.log_path = Path(log_path)
        self.proc = None

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        # start_new_session=True ALREADY calls setsid() in the child. Passing
        # preexec_fn=os.setsid as well made it call setsid() a SECOND time, from
        # a process that was by then already a session leader, which fails with
        # EPERM - and Popen reports that only as
        #   subprocess.SubprocessError: Exception occurred in preexec_fn.
        # Every episode died on launch, on every machine. The two settings do
        # the same job; keep the one that cannot fail.
        self.proc = subprocess.Popen(
            self.cmd, stdout=self._log, stderr=subprocess.STDOUT,
            start_new_session=True)          # its own process group; see above
        return self

    def wait_until_ready(self, timeout_s: float = 240.0,
                         ready="Managed nodes are active",
                         failed="Failed to bring up") -> str:
        """Block until nav2 says it is up. Returns "ready", "failed" or "timeout".

        A FIXED SLEEP IS NOT ENOUGH ON A SLOW HOST. At 25 s the episode asked
        for ground truth while the bridge was still starting; at 45 s the
        controller spawner was still reporting "Failed to acquire lock in 20
        seconds" and lifecycle_manager had given up on smoother_server. The
        stack takes as long as it takes, and it says so in its own log - so wait
        for the statement instead of guessing the duration.

        Returning "failed" rather than raising lets the caller record a refusal
        (`not_started`) instead of a navigation failure.
        """
        import time as _time
        deadline = _time.time() + timeout_s
        while _time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return "failed"
            try:
                text = self.log_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if ready in text:
                return "ready"
            if failed in text:
                return "failed"
            _time.sleep(2.0)
        return "timeout"

    def stop(self, grace_s: float = 15.0) -> None:
        import os
        import signal
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGINT)    # let launch shut nodes down cleanly
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait(timeout=grace_s)
        except (ProcessLookupError, PermissionError):
            pass
        finally:
            try:
                self._log.close()
            except Exception:
                pass
            self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def run_one(ep: PlannedEpisode, arm_spec: dict, *, campaign: str, runs_root: str,
            headless: bool = True, settle_s: float = SETTLE_S,
            venue: str = "sim", dry_run: bool = False) -> dict:
    """Launch a stack, run one episode against it, tear the stack down."""
    import time
    directory = ep.directory(runs_root, campaign)
    directory.mkdir(parents=True, exist_ok=True)
    launch = launch_command(ep, arm_spec, headless=headless)
    episode = episode_command(ep, arm_spec, campaign=campaign, runs_root=runs_root,
                              venue=venue)
    if dry_run:
        return {"outcome": "dry_run", "launch": launch, "episode": episode}

    # A stack inherits whatever the last one left running. See kill_survivors.
    left = kill_survivors()
    if left:
        print(f"      warning: {len(left)} process(es) survived cleanup")

    with LaunchedStack(launch, directory / "stack.log") as stack:
        # Wait for the stack to SAY it is up, with the fixed settle only as a
        # floor for stacks that never print the marker (use_nav2:=false).
        state = stack.wait_until_ready(timeout_s=max(settle_s * 6, 240.0))
        if state != "ready":
            print(f"      stack {state} before the episode started")
        time.sleep(min(settle_s, 10.0))   # let the last nodes settle after bringup
        with open(directory / "episode.log", "w", encoding="utf-8") as log:
            # WALL SECONDS, AND THE EPISODE'S BUDGET IS SIM SECONDS. The
            # episode caps navigation at ep.timeout_s of SIMULATED time; this
            # timeout is wall time. Headless Gazebo on a CPU-bound host runs at
            # a real-time factor near 0.25, so a 90 s sim budget is ~360 s of
            # wall clock - and `ep.timeout_s + settle_s + 180` killed the
            # episode at 315 s, before its own timeout could fire. That turns a
            # slow host into a missing episode with no summary at all.
            #
            # This is only a backstop against a hung process; the episode's own
            # clock decides the outcome, so it can afford to be generous.
            subprocess.run(episode, stdout=log, stderr=subprocess.STDOUT,
                           timeout=max(900.0, ep.timeout_s * 8 + 600))
    try:
        return schema.read_summary(directory)
    except Exception as exc:                   # no summary == the runner died
        return {"outcome": "no_summary", "reason": str(exc)}


def run_campaign(plan, arms: dict, *, campaign: str, runs_root: str,
                 headless: bool = True, settle_s: float = SETTLE_S,
                 venue: str = "sim", resume: bool = True,
                 stop_on_reference_failure: bool = True, dry_run: bool = False):
    """Every episode in order, with the stop-the-line rule.

    `lidar_ref` failing a clear course means the COURSE is broken, not the
    camera - a broken fixture produces a full matrix of meaningless numbers, so
    the run stops rather than collecting 19 hours of them.
    """
    todo = remaining(plan, runs_root, campaign) if resume else list(plan)
    results = []
    print(f"{len(todo)} episode(s) to run"
          + (f", {len(plan) - len(todo)} already done" if resume else ""))
    for i, ep in enumerate(todo, 1):
        spec = resolve_arm(ep.arm, arms)
        blocked = unsupported_overrides(spec)
        if blocked:
            print(f"  [{i}/{len(todo)}] SKIP {ep.pair_key} {ep.arm}: needs a parameter "
                  f"overlay for {blocked[:3]}{'...' if len(blocked) > 3 else ''}")
            results.append({"outcome": "not_runnable", "arm": ep.arm})
            continue
        print(f"  [{i}/{len(todo)}] {ep.pair_key} {ep.arm} ...", flush=True)
        summary = run_one(ep, spec, campaign=campaign, runs_root=runs_root,
                          headless=headless, settle_s=settle_s, venue=venue,
                          dry_run=dry_run)
        outcome = summary.get("outcome", "?")
        print(f"      -> {outcome}")
        results.append(summary)
        if (stop_on_reference_failure and ep.arm == "lidar_ref"
                and outcome not in ("reached", "dry_run")):
            print(f"\nSTOPPING: the lidar reference failed on {ep.pair_key} "
                  f"({outcome}). That means the COURSE is broken, not the camera. "
                  "Fix the fixture before collecting anything else.")
            break
    return results


# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("SELFTEST  ubot_eval.campaign (planning logic, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<52} {detail}")

    # -- arm parity ---------------------------------------------------------
    def _params(server_timeout=200, xy_tol=0.10, frame="odom", width=10,
                source="scan", obstacle_range=2.5):
        return {
            "bt_navigator": {"ros__parameters": {
                "default_server_timeout": server_timeout,
                "wait_for_service_timeout": 5000}},
            "controller_server": {"ros__parameters": {
                "general_goal_checker": {"xy_goal_tolerance": xy_tol,
                                         "yaw_goal_tolerance": 0.15}}},
            "global_costmap": {"global_costmap": {"ros__parameters": {
                "global_frame": frame, "width": width, "height": 10,
                "obstacle_layer": {"observation_sources": source,
                                   source: {"obstacle_max_range": obstacle_range}}}}},
            "local_costmap": {"local_costmap": {"ros__parameters": {
                "global_frame": "odom"}}},
        }

    check("identical params have no parity violations",
          parity_violations(_params(), _params()) == [])
    check("a looser goal checker in ONE arm is caught",
          len(parity_violations(_params(xy_tol=0.10), _params(xy_tol=0.25))) == 1,
          str(parity_violations(_params(xy_tol=0.10), _params(xy_tol=0.25))))
    check("a longer BT timeout in ONE arm is caught",
          len(parity_violations(_params(server_timeout=200),
                                _params(server_timeout=20))) == 1)
    check("planning in different frames is caught",
          any("global_frame" in v for v in
              parity_violations(_params(frame="odom"), _params(frame="map"))))
    check("a bigger planning horizon in one arm is caught",
          any("width" in v for v in
              parity_violations(_params(width=10), _params(width=30))))
    # The sensor and its consequences MUST be allowed to differ - that is the
    # experiment. A check that forced these equal would defeat the point.
    check("the observation source is allowed to differ",
          parity_violations(_params(source="scan"),
                            _params(source="mono_scan")) == [])
    check("the range caps are allowed to differ",
          parity_violations(_params(obstacle_range=2.5),
                            _params(obstacle_range=3.0)) == [])

    courses = {
        "schema_version": 1,
        "worlds": {
            "apartment": {"world": "apartment", "world_name": "apartment",
                          "courses": {
                              "hall_3m": {"spawn": {"x": 5.0, "y": 0.0, "z": 0.1, "yaw": 1.57},
                                          "goal": {"x": 5.0, "y": 3.0, "yaw": 1.57},
                                          "shortest_path_m": 3.0, "timeout_s": 90}}},
            "parking_garage": {"world": "parking_garage_occlusion",
                               "world_name": "parking_garage",
                               "courses": {
                                   "aisle_8m": {"spawn": {"x": 0.0, "y": 0.0, "z": 0.1, "yaw": 0.0},
                                                "goal": {"x": 8.0, "y": 0.0, "yaw": 0.0},
                                                "shortest_path_m": 8.0, "timeout_s": 180}}}}}
    arms = {"arms": {
        "mono_baseline": {"launch": "L", "params": {}},
        "mono_A": {"launch": "L", "params": {"a": 1}},
        "mono_B": {"launch": "L", "params": {"b": 2}},
        "mono_AB": {"launch": "L", "inherits": ["mono_A", "mono_B"]},
        "lidar_ref": {"launch": "M", "params": {}},
        "mono_baseline_dup": {"launch": "L", "params": {}}}}
    all_arms = list(arms["arms"])

    plan = build_plan(courses, arms, trials=3)
    check("plan size = worlds x courses x trials x arms",
          len(plan) == 2 * 1 * 3 * 6, f"{len(plan)} episodes")
    check("every arm covers every pair", not check_pairing(plan, all_arms))

    keys = {ep.pair_key for ep in plan}
    check("pair_key excludes the arm", len(keys) == 6, sorted(keys)[:2])
    check("world_name is carried separately from the file name",
          any(ep.world_file == "parking_garage_occlusion"
              and ep.world_name == "parking_garage" for ep in plan))

    # determinism and de-correlation
    again = build_plan(courses, arms, trials=3)
    check("same seed gives the identical order",
          [e.arm for e in plan] == [e.arm for e in again])
    other = build_plan(courses, arms, trials=3, seed=DEFAULT_SEED + 1)
    check("a different seed reorders", [e.arm for e in plan] != [e.arm for e in other])
    first_of_each = {}
    for ep in plan:
        first_of_each.setdefault(ep.pair_key, ep.arm)
    check("arms do not all start the same pair (drift cannot align)",
          len(set(first_of_each.values())) > 1, str(sorted(set(first_of_each.values()))[:3]))
    per_pair = {}
    for ep in plan:
        per_pair.setdefault(ep.pair_key, []).append(ep.arm)
    check("every pair still contains every arm exactly once",
          all(sorted(v) == sorted(all_arms) for v in per_pair.values()))

    ab = resolve_arm("mono_AB", arms)
    check("mono_AB inherits both parents' parameters",
          ab["params"] == {"a": 1, "b": 2} and ab["launch"] == "L", str(ab["params"]))

    # resume
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        target = plan[0]
        d = target.directory(tmp, "c1")
        s = schema.EpisodeSummary(
            campaign="c1", arm=target.arm, world=target.world, course=target.course,
            trial=target.trial, outcome="reached", outcome_reason="ok",
            path_length_m=3.2, shortest_path_m=3.0, spl=0.9375)
        schema.write_summary(d, s)
        left = remaining(plan, tmp, "c1")
        check("resume skips a completed episode", len(left) == len(plan) - 1,
              f"{len(left)} of {len(plan)} left")

        (d / "episode.json").write_text('{"schema_version": 1, "outcome": "bogus"}',
                                        encoding="utf-8")
        left = remaining(plan, tmp, "c1")
        check("a corrupt summary is re-run, not skipped", len(left) == len(plan))

    # power, before collecting
    d75 = detectable_sr_difference(75)
    check("75 pairs detect ~15-18 points, as the plan states",
          0.14 <= d75 <= 0.20, f"{d75 * 100:.1f} points at 80% power")
    d300 = detectable_sr_difference(300)
    check("four times the pairs halves the detectable effect",
          abs(d300 - d75 / 2) < 0.005, f"{d300 * 100:.1f} points at 300 pairs")

    # filters
    smoke = build_plan(courses, arms, trials=2, arm_names=["lidar_ref", "mono_baseline"],
                       worlds=["apartment"], courses_filter=["hall_3m"])
    check("smoke selection is 2 arms x 2 trials", len(smoke) == 4, f"{len(smoke)} episodes")

    # broken pairings, both shapes
    dropped_arm = [ep for ep in plan if ep.arm != "mono_A"]
    problems = check_pairing(dropped_arm, all_arms)
    check("an arm that never ran is caught",
          any("mono_A has no episodes" in p for p in problems),
          (problems or ["none"])[0][:44])
    check("...and is NOT caught without the expected arm list",
          not check_pairing(dropped_arm),
          "which is why --plan passes the arm list in")

    dropped_one = [ep for ep in plan if not (ep.arm == "mono_B"
                                             and ep.pair_key.endswith("/01"))]
    problems = check_pairing(dropped_one)
    check("an arm missing a single pair is caught by the union",
          any("mono_B is missing" in p for p in problems),
          (problems or ["none"])[0][:44])

    # ---- running it -------------------------------------------------------
    runnable = {"arms": {
        "mono_baseline": {"launch": "ubot_mono_nav/sim_mono.launch.py", "params": {}},
        "lidar_ref": {"launch": "ubot_bringup/sim.launch.py",
                      "params": {"use_slam": False}},
        "mono_A": {"launch": "ubot_mono_nav/sim_mono.launch.py",
                   "params": {"depth_to_scan.ros__parameters.min_points": 8}}}}
    garage = build_plan(courses, runnable, trials=1, arm_names=["mono_baseline"],
                        worlds=["parking_garage"], shuffle_arms=False)[0]
    cmd = launch_command(garage, resolve_arm("mono_baseline", runnable))
    check("launch uses the world FILE and the <world name> separately",
          "world:=parking_garage_occlusion" in cmd and "world_name:=parking_garage" in cmd,
          "the trap that hangs the spawner")
    check("spawn pose is passed through", "spawn_x:=0.0" in cmd and "spawn_yaw:=0.0" in cmd)
    check("headless by default for a campaign", "headless:=true" in cmd)

    lidar_ep = build_plan(courses, runnable, trials=1, arm_names=["lidar_ref"],
                          worlds=["apartment"], shuffle_arms=False)[0]
    lcmd = launch_command(lidar_ep, resolve_arm("lidar_ref", runnable))
    check("an arm's launch ARGS are applied", "use_slam:=false" in lcmd,
          "odom frame for both arms")
    check("the lidar arm launches ubot_bringup, not the mono stack",
          "ubot_bringup" in lcmd and "sim.launch.py" in lcmd)

    check("lidar arm watches /scan, mono arms /scan_mono",
          arm_scan_topic("lidar_ref", resolve_arm("lidar_ref", runnable)) == "/scan"
          and arm_scan_topic("mono_baseline",
                             resolve_arm("mono_baseline", runnable)) == "/scan_mono")

    ecmd = episode_command(lidar_ep, resolve_arm("lidar_ref", runnable),
                           campaign="c", runs_root="/tmp/r")
    check("episode command carries the arm's scan topic",
          "--scan-topic" in ecmd and ecmd[ecmd.index("--scan-topic") + 1] == "/scan")
    check("no --shortest-path-m when the course has none",
          "--shortest-path-m" not in ecmd or lidar_ep.shortest_path_m is not None)

    check("node-parameter arms are refused, not silently run as the baseline",
          unsupported_overrides(resolve_arm("mono_A", runnable))
          == ["depth_to_scan.ros__parameters.min_points"],
          "mono_A needs a params overlay that the launch files lack")
    check("launch-argument arms are runnable",
          unsupported_overrides(resolve_arm("lidar_ref", runnable)) == []
          and unsupported_overrides(resolve_arm("mono_baseline", runnable)) == [])

    stack = LaunchedStack(["true"], "/tmp/does-not-matter.log")
    stack.stop()                       # never started
    check("stopping a stack that never started is a no-op", True)
    # Assert the MECHANISM, not the absence of a word: "pkill" appears in the
    # docstring that explains why it is not used, and grepping the module for it
    # fails on the explanation rather than on any call.
    import inspect
    teardown = inspect.getsource(LaunchedStack.stop)
    check("teardown signals the process group, never pkill",
          "killpg" in teardown and "pkill" not in teardown,
          "orphaned Gazebo would poison the next episode")
    launcher = inspect.getsource(LaunchedStack.start)
    check("the stack gets its own session to make that possible",
          "start_new_session=True" in launcher)
    # start_new_session ALREADY setsid()s the child. Adding preexec_fn=os.setsid
    # on top made it setsid() twice, the second call failed with EPERM, and
    # Popen surfaced that only as "Exception occurred in preexec_fn" - every
    # episode died on launch. A real campaign caught it; this keeps it caught.
    # Comments stripped first: start() carries a note EXPLAINING why preexec_fn
    # is not used, and a plain substring search fails on the explanation rather
    # than on any argument - the same trap as the pkill check above.
    launcher_code = "\n".join(line.split("#", 1)[0]
                              for line in launcher.splitlines())
    check("and does NOT also setsid via preexec_fn, which always fails",
          "preexec_fn" not in launcher_code,
          "start_new_session + preexec_fn=os.setsid = EPERM on every launch")

    # Prove it launches, rather than trusting the source read above.
    probe = LaunchedStack(["true"], "/tmp/ubot_eval_launch_probe.log")
    try:
        probe.start()
        probe.proc.wait(timeout=10)
        check("a stack actually starts under the real Popen settings", True)
    except Exception as exc:                                  # pragma: no cover
        check("a stack actually starts under the real Popen settings", False, repr(exc))
    finally:
        probe.stop()

    dry = run_one(garage, resolve_arm("mono_baseline", runnable), campaign="c",
                  runs_root="/tmp/ubot_eval_dry", dry_run=True)
    check("dry run returns both commands and launches nothing",
          dry["outcome"] == "dry_run" and dry["launch"][0] == "ros2"
          and dry["episode"][3] == "episode_runner")

    # the stop-the-line rule, with run_one swapped out
    # RESUME MUST RETRY REFUSALS. A `not_started` episode is the gate saying the
    # robot could not run; banking it deletes that pair from the paired
    # statistics permanently.
    import json as _json
    import tempfile as _tempfile

    def _write_summary(directory, outcome):
        directory.mkdir(parents=True, exist_ok=True)
        doc = dict(schema_version=1, campaign="c", arm="lidar_ref",
                   world="apartment", course="hall_3m", trial=1, venue="sim",
                   outcome=outcome, outcome_reason="x",
                   goal={"x": 3.0, "y": 0.0, "yaw": 0.0, "frame": "odom"},
                   reached_goal=False, gt_checked=False,
                   readiness={"passed": outcome != "not_started"},
                   truth={}, perception={}, collisions={}, artifacts={},
                   provenance={}, warnings=[], notes="",
                   created_utc="2026-09-16T00:00:00+00:00")
        (directory / "episode.json").write_text(_json.dumps(doc), encoding="utf-8")

    with _tempfile.TemporaryDirectory() as _tmp:
        _root = Path(_tmp)
        _plan = build_plan(courses, arms, trials=1, arm_names=["lidar_ref"],
                           worlds=["apartment"], courses_filter=["hall_3m"])
        _ep = _plan[0]
        _write_summary(_ep.directory(_root, "c"), "not_started")
        check("a REFUSED episode is NOT treated as done",
              (_ep.pair_key, _ep.arm) not in completed(_plan, _root, "c"),
              "otherwise --resume deletes that pair for good")
        _write_summary(_ep.directory(_root, "c"), "aborted")
        check("a real failure IS done - it is data, not a refusal",
              (_ep.pair_key, _ep.arm) in completed(_plan, _root, "c"))

    import sys as _sys
    module = _sys.modules[__name__]
    real_run_one = module.run_one
    calls = []

    def fake_run_one(ep, spec, **kw):
        calls.append(ep.arm)
        return {"outcome": "timeout" if ep.arm == "lidar_ref" else "reached"}

    try:
        module.run_one = fake_run_one
        pair_plan = build_plan(courses, runnable, trials=2,
                               arm_names=["lidar_ref", "mono_baseline"],
                               worlds=["apartment"], shuffle_arms=False)
        results = run_campaign(pair_plan, runnable, campaign="c",
                               runs_root="/tmp/ubot_eval_dry", resume=False)
        check("a lidar_ref failure stops the line immediately",
              len(results) == 1 and calls == ["lidar_ref"],
              f"ran {len(calls)} episode(s) of {len(pair_plan)}")
    finally:
        module.run_one = real_run_one

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-all", action="store_true",
                    help="every module's selftest, in dependency order")
    ap.add_argument("--plan", action="store_true", help="print the episode plan")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--pairs", type=int, default=75)
    ap.add_argument("--discordance", type=float, default=0.30)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--config", default=None, help="default: the installed share/config")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--campaign", default="pilot")
    ap.add_argument("--smoke", action="store_true", help="the pre-campaign smoke selection")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run", action="store_true",
                    help="actually launch the stack and run the episodes")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --run: print the commands, launch nothing")
    ap.add_argument("--settle-s", type=float, default=SETTLE_S,
                    help="seconds to let a stack come up before the readiness gate")
    ap.add_argument("--gui", action="store_true", help="show Gazebo (default headless)")
    ap.add_argument("--venue", default="sim", choices=["sim", "real"])
    ap.add_argument("--keep-going", action="store_true",
                    help="do NOT stop when the lidar reference fails a course")
    # Filters, so a single-arm reliability run is a normal thing to ask for
    # rather than an edit to arms.yaml. A one-arm campaign has no pairing to
    # check and no deltas to report - it measures that arm's own reliability.
    ap.add_argument("--arms", default=None,
                    help="comma-separated arm names; default every arm in arms.yaml")
    ap.add_argument("--worlds", default=None,
                    help="comma-separated worlds; default every world in courses.yaml. "
                         "Courses whose shortest_path_m is still null cannot score a "
                         "success, so restrict to measured worlds for a real campaign.")
    ap.add_argument("--courses", default=None,
                    help="comma-separated course names within those worlds")
    args = ap.parse_args()

    def _split(value):
        return [v.strip() for v in value.split(",") if v.strip()] if value else None

    arm_filter = _split(args.arms)
    world_filter = _split(args.worlds)
    course_filter = _split(args.courses)

    if args.selftest:
        sys.exit(selftest())
    if args.selftest_all:
        print("SELFTEST ALL  ubot_eval, in dependency order")
        sys.exit(run_selftests())
    if args.power:
        d = detectable_sr_difference(args.pairs, args.discordance)
        print(f"{args.pairs} pairs, discordance {args.discordance:.2f}: "
              f"detectable success-rate difference {d * 100:.1f} points "
              f"(two-sided 0.05, 80% power)")
        print("If the effect you care about is smaller, collect more trials - "
              "no analysis recovers power that was never there.")
        return
    if args.plan:
        courses, arms = load_config(args.config)
        if args.smoke:
            s = courses["smoke"]
            plan = build_plan(courses, arms, trials=s["trials"], arm_names=s["arms"],
                              worlds=[s["world"]], courses_filter=[s["course"]])
            expected = s["arms"]
        else:
            plan = build_plan(courses, arms, trials=args.trials,
                              arm_names=arm_filter, worlds=world_filter,
                              courses_filter=course_filter)
            expected = arm_filter or list(arms["arms"])
        problems = check_pairing(plan, expected)
        if problems:
            sys.exit("broken pairing: " + "; ".join(problems))
        if args.resume:
            plan = remaining(plan, args.runs_root, args.campaign)
        print(f"{len(plan)} episodes, {len({e.pair_key for e in plan})} pairs")
        for ep in plan[:20]:
            print(f"  {ep.pair_key:<28} {ep.arm:<20} {ep.world_file}"
                  f" (<world name> {ep.world_name})")
        if len(plan) > 20:
            print(f"  ... and {len(plan) - 20} more")
        return
    if args.run:
        courses, arms = load_config(args.config)
        if args.smoke:
            s = courses["smoke"]
            plan = build_plan(courses, arms, trials=s["trials"], arm_names=s["arms"],
                              worlds=[s["world"]], courses_filter=[s["course"]])
            expected = s["arms"]
        else:
            plan = build_plan(courses, arms, trials=args.trials,
                              arm_names=arm_filter, worlds=world_filter,
                              courses_filter=course_filter)
            expected = arm_filter or list(arms["arms"])
        problems = check_pairing(plan, expected)
        if problems:
            sys.exit("broken pairing: " + "; ".join(problems))
        results = run_campaign(plan, arms, campaign=args.campaign,
                               runs_root=args.runs_root, headless=not args.gui,
                               settle_s=args.settle_s, venue=args.venue,
                               resume=args.resume, dry_run=args.dry_run,
                               stop_on_reference_failure=not args.keep_going)
        counts: dict = {}
        for r in results:
            counts[r.get("outcome", "?")] = counts.get(r.get("outcome", "?"), 0) + 1
        print("\noutcomes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        print(f"summaries under {Path(args.runs_root) / args.campaign}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
