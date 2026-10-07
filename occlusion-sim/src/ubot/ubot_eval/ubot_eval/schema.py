r"""The run-directory contract: what one navigation episode records, and what
counts as which outcome.

    python3 -m ubot_eval.schema --selftest        # no ROS, no robot, no numpy

--------------------------------------------------------------------------
WHY THIS MODULE IS STDLIB-ONLY
--------------------------------------------------------------------------
Two trees share these files and neither imports the other (see
docs/research/nav_eval_plan.md 5.1): `ubot_eval` (ROS) writes run directories,
`Occlusion/research/nav_eval/` (no ROS) reads them. The wire format is the
directory on disk, versioned by `schema_version`. So this module must import on
a machine with neither ROS nor numpy installed, and it must never grow a
dependency that only one side has.

--------------------------------------------------------------------------
LAYOUT
--------------------------------------------------------------------------
    <runs>/<campaign>/<world>/<course>/<arm>/trial_<NN>/
        episode.json                  this schema
        navigation_YYYYMMDD_HHMMSS/   the bag (docs/experiments/logging.md naming)
        scan_pairs.jsonl              mono vs reference scan, if logged
        depth_pairs/                  oakd.py run-dir layout, if logged

`pair_key` is `world/course/trial`, NOT including the arm: it is the unit the
paired bootstrap resamples on, so every arm must produce the same set of keys.

--------------------------------------------------------------------------
THE OUTCOME LADDER, FIRST MATCH WINS
--------------------------------------------------------------------------
    not_started -> collided -> reached -> false_success -> aborted
                -> timeout -> stalled -> degraded

Three of these exist because of specific ways this platform lies:

* `not_started` — the readiness gate failed. Scoring a robot that could not move
  as a navigation failure is silent corruption, so this is never a timeout and
  never a failure; it is excluded and counted separately.
* `false_success` — nav2 reported SUCCEEDED but ground truth says the robot is
  more than `goal_tolerance_m` away. With no SLAM, nav2 believes odometry, and
  odometry is exactly what drifts. Counted as a failure.
* `collided` — safety dominates, so it outranks `reached`. `reached_goal` is
  recorded separately on the same episode so the safety-versus-completion
  trade-off stays visible (the MonoNav reporting frame, 2311.14100).

KNOWN TENSION, deliberately kept: `degraded` (perception died) sits LAST, so an
episode whose camera died and then ran out of clock is scored `timeout`. That is
the plan's order and it is kept, but every raw signal feeding the decision is
stored in the summary, so analysis can re-classify without re-running anything.
`classify()` is pure and takes those signals explicitly for the same reason.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# First match wins. Order is load-bearing; see the module docstring.
OUTCOMES = ("not_started", "collided", "reached", "false_success",
            "aborted", "timeout", "stalled", "degraded")

#: Counts as a success in SR and contributes a non-zero SPL term.
SUCCESS_OUTCOMES = frozenset({"reached"})

#: Excluded from the success-rate denominator by default, and reported separately.
#: `not_started` never ran; `degraded` ran without working perception.
EXCLUDED_FROM_SR = frozenset({"not_started", "degraded"})

#: Ground-truth distance within which nav2's SUCCEEDED is believed.
GOAL_TOLERANCE_M = 0.15

#: `p_i < l_i` by more than this is a broken reference path, not a fast robot.
SPL_UNDERRUN_WARN = 0.02


def pair_key(world: str, course: str, trial: int) -> str:
    """The bootstrap's resampling unit. Arm is deliberately absent."""
    return f"{world}/{course}/{int(trial):02d}"


def episode_path(runs_root, campaign: str, world: str, course: str,
                 arm: str, trial: int) -> Path:
    return (Path(runs_root) / campaign / world / course / arm
            / f"trial_{int(trial):02d}")


def bag_dirname(when: datetime | None = None) -> str:
    """`navigation_YYYYMMDD_HHMMSS`, the convention in docs/experiments/logging.md."""
    return f"navigation_{(when or datetime.now()):%Y%m%d_%H%M%S}"


def spl_term(success: bool, shortest_m: float | None,
             actual_m: float | None) -> tuple[float, str | None]:
    """One episode's SPL contribution, and a warning if the geometry is impossible.

    SPL = mean over ALL episodes of  success * l / max(p, l)  -- failures
    contribute 0 and are still in the denominator. Averaging over successes only
    is the classic way to report an SPL that flatters a robot which rarely
    arrives, and the selftest below pins that down.
    """
    if not success:
        return 0.0, None
    if not shortest_m or not actual_m or shortest_m <= 0 or actual_m <= 0:
        return 0.0, "success with no usable path lengths; SPL term forced to 0"
    warn = None
    if actual_m < shortest_m * (1.0 - SPL_UNDERRUN_WARN):
        # Clamping keeps SPL <= 1, but the clamp is not the finding: a robot
        # cannot beat the shortest path, so the reference path is wrong.
        warn = (f"actual path {actual_m:.3f} m is under the reference "
                f"{shortest_m:.3f} m by more than {SPL_UNDERRUN_WARN:.0%}: "
                "the reference path is probably wrong")
    return float(shortest_m / max(actual_m, shortest_m)), warn


def classify(*, readiness_ok: bool, collided: bool, nav2_status: str | None,
             gt_distance_to_goal_m: float | None, timed_out: bool,
             perception_ok: bool = True, moved_recently: bool | None = None,
             goal_tolerance_m: float = GOAL_TOLERANCE_M) -> tuple[str, str]:
    """(outcome, reason). Pure: every input is an observation, nothing is read.

    `gt_distance_to_goal_m` None means no ground truth was available (the real
    robot). `reached` is then taken on nav2's word, and the summary records
    `gt_checked: false` so no one later mistakes it for a verified arrival.
    """
    if not readiness_ok:
        return "not_started", "readiness gate failed; the robot never ran"
    if collided:
        return "collided", "contact sensor fired during the episode"

    status = (nav2_status or "").upper()
    if status == "SUCCEEDED":
        if gt_distance_to_goal_m is None:
            return "reached", "nav2 SUCCEEDED; no ground truth to verify it"
        if gt_distance_to_goal_m <= goal_tolerance_m:
            return "reached", (f"nav2 SUCCEEDED and ground truth agrees "
                               f"({gt_distance_to_goal_m:.3f} m from goal)")
        return "false_success", (
            f"nav2 SUCCEEDED but ground truth is {gt_distance_to_goal_m:.3f} m "
            f"from the goal (> {goal_tolerance_m:.2f} m): odometry drifted")
    if status in ("ABORTED", "CANCELED", "CANCELLED", "REJECTED"):
        return "aborted", f"nav2 returned {status}"
    if timed_out:
        return "timeout", "episode wall-clock cap reached"
    if moved_recently is False:
        return "stalled", "no motion within the stall window while still active"
    if not perception_ok:
        return "degraded", "perception stopped delivering during the episode"
    return "aborted", (f"unclassified end state (nav2 status {status or 'none'}); "
                       "recorded as aborted so it cannot be mistaken for success")


@dataclass
class EpisodeSummary:
    """One goal attempt. Written as `episode.json` in the episode directory."""

    campaign: str
    arm: str
    world: str
    course: str
    trial: int
    outcome: str
    outcome_reason: str = ""

    schema_version: int = SCHEMA_VERSION
    created_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    venue: str = "sim"                      # "sim" or "real"

    goal: dict = field(default_factory=dict)          # x, y, yaw, frame
    nav2_status: str | None = None

    #: Recorded even when `outcome` is `collided`, so safety-versus-completion
    #: stays visible instead of being collapsed into one number.
    reached_goal: bool = False
    gt_checked: bool = False
    gt_distance_to_goal_m: float | None = None

    readiness: dict = field(default_factory=dict)     # passed, failed_check, attempts, motion_proof_m
    truth: dict = field(default_factory=dict)         # source, topic, cross_check_error_m, odom_drift_m
    perception: dict = field(default_factory=dict)    # ok, stale_events, worst_gap_s

    path_length_m: float | None = None
    shortest_path_m: float | None = None
    distance_ratio: float | None = None
    spl: float | None = None
    time_to_goal_s: float | None = None
    min_clearance_m: float | None = None
    recoveries: int | None = None
    collisions: dict = field(default_factory=dict)    # count, events, first_s, per_metre

    artifacts: dict = field(default_factory=dict)     # bag, scan_pairs, depth_pairs
    provenance: dict = field(default_factory=dict)    # git_sha, config hashes, ros_distro
    warnings: list = field(default_factory=list)
    notes: str = ""

    @property
    def pair_key(self) -> str:
        return pair_key(self.world, self.course, self.trial)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pair_key"] = self.pair_key          # written out; derived, never read back
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EpisodeSummary:
        d = dict(d)
        d.pop("pair_key", None)
        known = {f for f in cls.__dataclass_fields__}          # noqa: F821
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown fields for schema {SCHEMA_VERSION}: {sorted(unknown)}")
        return cls(**d)


def validate(d: dict) -> list[str]:
    """Problems with a summary dict. Empty list means usable.

    Catches the corruptions that would quietly change a result: an episode that
    never ran carrying metrics, a collision recorded under a non-collided
    outcome, a success with no reference path (so its SPL term is a silent 0).
    """
    problems: list[str] = []
    ver = d.get("schema_version")
    if ver is None:
        problems.append("no schema_version")
    elif int(ver) > SCHEMA_VERSION:
        problems.append(f"schema_version {ver} is newer than this reader ({SCHEMA_VERSION})")

    outcome = d.get("outcome")
    if outcome not in OUTCOMES:
        problems.append(f"outcome {outcome!r} is not one of {OUTCOMES}")

    for key in ("campaign", "arm", "world", "course"):
        if not d.get(key):
            problems.append(f"missing {key}")
    if not isinstance(d.get("trial"), int):
        problems.append("trial must be an int")

    if d.get("world") and d.get("course") and isinstance(d.get("trial"), int):
        expect = pair_key(d["world"], d["course"], d["trial"])
        if "pair_key" in d and d["pair_key"] != expect:
            problems.append(f"pair_key {d['pair_key']!r} disagrees with fields ({expect!r})")

    collisions = d.get("collisions") or {}
    if collisions.get("count", 0) and outcome not in ("collided", "not_started"):
        problems.append(f"{collisions['count']} collisions recorded but outcome is {outcome!r}")

    if outcome == "not_started":
        for key in ("path_length_m", "time_to_goal_s", "spl"):
            if d.get(key) is not None:
                problems.append(f"not_started episode carries {key}; it never ran")

    if outcome in SUCCESS_OUTCOMES:
        if not d.get("shortest_path_m"):
            problems.append("success without shortest_path_m: its SPL term would be a silent 0")
        if not d.get("path_length_m"):
            problems.append("success without path_length_m")

    if outcome == "false_success" and d.get("gt_distance_to_goal_m") is None:
        problems.append("false_success requires a ground-truth distance")

    return problems


def write_summary(episode_dir, summary: EpisodeSummary) -> Path:
    """Atomically write `episode.json`. Refuses to write an invalid summary.

    Atomic because a campaign is resumable: `campaign.py --resume` decides what
    to skip by reading these files, and a half-written one after a crash would
    make it skip an episode that never happened.
    """
    d = summary.to_dict()
    problems = validate(d)
    if problems:
        raise ValueError("refusing to write an invalid summary: " + "; ".join(problems))
    episode_dir = Path(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    path = episode_dir / "episode.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_summary(episode_dir) -> dict:
    """The summary dict, validated. Raises on anything a reader cannot trust."""
    path = Path(episode_dir)
    if path.is_dir():
        path = path / "episode.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    problems = validate(d)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return d


def iter_summaries(runs_root):
    """Every readable episode under a runs root, in sorted path order."""
    for path in sorted(Path(runs_root).rglob("episode.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def selftest() -> int:
    """Cases from nav_eval_plan.md 6, plus the corruption guards."""
    print("SELFTEST  ubot_eval.schema (no ROS, no numpy)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<52} {detail}")

    # ---- SPL term ----
    v, w = spl_term(True, 10.0, 10.0)
    check("straight 10 m run -> SPL term 1.0", abs(v - 1.0) < 1e-9 and w is None, f"{v:.3f}")
    v, _ = spl_term(True, 10.0, 20.0)
    check("20 m taken to a 10 m goal -> 0.5", abs(v - 0.5) < 1e-9, f"{v:.3f}")
    v, _ = spl_term(False, 10.0, 10.0)
    check("failure contributes 0, not omitted", v == 0.0, f"{v:.3f}")
    # 2 of 4 successful, each perfect: SR 0.5 AND SPL 0.5 over ALL episodes
    terms = [spl_term(True, 10.0, 10.0)[0], spl_term(True, 10.0, 10.0)[0],
             spl_term(False, 10.0, 31.0)[0], spl_term(False, None, 12.0)[0]]
    check("2 of 4 successes -> SPL 0.5 over all episodes",
          abs(sum(terms) / len(terms) - 0.5) < 1e-9,
          "averaging over successes only would give 1.0")
    v, w = spl_term(True, 10.0, 9.0)
    check("p < l clamps AND warns", abs(v - 1.0) < 1e-9 and w is not None,
          (w or "")[:34] + "...")

    # ---- the outcome ladder ----
    o, _ = classify(readiness_ok=False, collided=False, nav2_status=None,
                    gt_distance_to_goal_m=None, timed_out=True)
    check("readiness failure -> not_started, never timeout", o == "not_started", o)
    o, _ = classify(readiness_ok=True, collided=True, nav2_status="SUCCEEDED",
                    gt_distance_to_goal_m=0.05, timed_out=False)
    check("collision outranks arrival", o == "collided", o)
    o, _ = classify(readiness_ok=True, collided=False, nav2_status="SUCCEEDED",
                    gt_distance_to_goal_m=0.05, timed_out=False)
    check("SUCCEEDED and truth agrees -> reached", o == "reached", o)
    o, r = classify(readiness_ok=True, collided=False, nav2_status="SUCCEEDED",
                    gt_distance_to_goal_m=1.8, timed_out=False)
    check("SUCCEEDED but 1.8 m away -> false_success", o == "false_success", o)
    o, r = classify(readiness_ok=True, collided=False, nav2_status="SUCCEEDED",
                    gt_distance_to_goal_m=None, timed_out=False)
    check("no ground truth -> reached, reason says unverified",
          o == "reached" and "no ground truth" in r, o)
    o, _ = classify(readiness_ok=True, collided=False, nav2_status="ABORTED",
                    gt_distance_to_goal_m=3.0, timed_out=False)
    check("nav2 ABORTED -> aborted", o == "aborted", o)
    o, _ = classify(readiness_ok=True, collided=False, nav2_status=None,
                    gt_distance_to_goal_m=3.0, timed_out=True)
    check("clock cap -> timeout", o == "timeout", o)
    o, _ = classify(readiness_ok=True, collided=False, nav2_status=None,
                    gt_distance_to_goal_m=3.0, timed_out=False, moved_recently=False)
    check("still active but not moving -> stalled", o == "stalled", o)
    o, _ = classify(readiness_ok=True, collided=False, nav2_status=None,
                    gt_distance_to_goal_m=3.0, timed_out=False, perception_ok=False)
    check("perception died -> degraded", o == "degraded", o)
    o, _ = classify(readiness_ok=True, collided=False, nav2_status="",
                    gt_distance_to_goal_m=None, timed_out=False)
    check("unclassifiable never becomes a success", o == "aborted", o)

    # ---- round trip and validation ----
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ep = episode_path(tmp, "pilot", "apartment", "hall_3m", "mono_baseline", 7)
        s = EpisodeSummary(
            campaign="pilot", arm="mono_baseline", world="apartment",
            course="hall_3m", trial=7, outcome="reached",
            outcome_reason="ground truth agrees", reached_goal=True, gt_checked=True,
            gt_distance_to_goal_m=0.04, path_length_m=3.4, shortest_path_m=3.0,
            spl=spl_term(True, 3.0, 3.4)[0])
        p = write_summary(ep, s)
        back = read_summary(ep)
        check("write/read round trip", back["arm"] == "mono_baseline" and p.exists(),
              str(p.name))
        check("pair_key is world/course/trial, no arm",
              back["pair_key"] == "apartment/hall_3m/07", back["pair_key"])
        check("no .tmp file left behind", not list(Path(ep).glob("*.tmp")))

        again = EpisodeSummary.from_dict({k: v for k, v in back.items()})
        check("from_dict ignores the derived pair_key", again.trial == 7)

        bad = dict(back, schema_version=SCHEMA_VERSION + 1)
        check("a newer schema is refused, not guessed",
              any("newer" in x for x in validate(bad)))

    # corruption guards
    base = dict(schema_version=1, campaign="c", arm="a", world="w", course="x",
                trial=1, outcome="reached", shortest_path_m=3.0, path_length_m=3.2)
    check("success needs a reference path",
          any("SPL term" in x for x in validate(dict(base, shortest_path_m=None))))
    check("collisions under a non-collided outcome are caught",
          any("collisions recorded" in x for x in
              validate(dict(base, collisions={"count": 2}))))
    check("not_started carrying metrics is caught",
          any("never ran" in x for x in
              validate(dict(base, outcome="not_started", shortest_path_m=None,
                            path_length_m=None, time_to_goal_s=12.0))))
    check("false_success without ground truth is caught",
          any("requires a ground-truth" in x for x in
              validate(dict(base, outcome="false_success"))))
    check("bag name follows logging.md",
          bag_dirname(datetime(2026, 9, 16, 14, 32, 0)) == "navigation_20260916_143200",
          bag_dirname(datetime(2026, 9, 16, 14, 32, 0)))

    print("\n" + ("SELFTEST PASSED" if ok else "SELFTEST FAILED"))
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--show", metavar="EPISODE_DIR", help="validate and print one summary")
    args = ap.parse_args()
    if args.show:
        print(json.dumps(read_summary(args.show), indent=2, sort_keys=True))
        return
    sys.exit(selftest() if args.selftest else ap.print_help())


if __name__ == "__main__":
    main()
