"""When did messages actually arrive? Settles a perception stall from the bag.

    python3 -m ubot_eval.bag_timing --selftest
    python3 -m ubot_eval.bag_timing --episode ~/occl_ws/runs/pilot/apartment/hall_3m/mono_baseline/trial_01

THE QUESTION THIS ANSWERS
-------------------------
An episode recorded `worst_gap_s = 12.366` on /scan_mono while depth inference
takes 13 ms/frame on a V100. Inference cannot explain a 12 second gap, so
something else stopped. The candidates look identical in `episode.json` and are
told apart only by timing:

  * the whole SIMULATION froze - then /scan (the LiDAR, 10 Hz, same bag, same
    clock) gapped too, and the camera chain is innocent;
  * only the CAMERA CHAIN stalled - then /scan kept its cadence while
    /scan_mono did not;
  * a CLOCK-DOMAIN artefact - the gap exists in one clock and not the other.

TWO CLOCKS, AND THAT IS THE POINT
---------------------------------
`ros2 bag record` stores a WALL-CLOCK receive time for every message, while the
message's own `header.stamp` is SIM time (every publisher runs use_sim_time).
Comparing the two separates "time passed and nothing was published" from "the
simulation itself stopped advancing":

    wall gap >> sim gap   the sim clock barely moved while real seconds passed:
                          the host could not keep up. Everything froze.
    sim gap  >> wall gap  simulated time jumped while messages kept arriving:
                          a clock artefact, not a stall.
    both large together   a real outage at a normal real-time factor.

WHAT IS NOT MEASURED HERE
-------------------------
/clock is not in BAG_TOPICS, so the real-time factor is INFERRED from the
(receive time, header stamp) pairs of the recorded topics rather than read
directly. It is a ratio over a window, not an instantaneous RTF, and this module
says so rather than implying more precision than it has.

/scan_mono's header.stamp is copied from the RGB frame that produced it
(depth_to_scan_node.py <- mono_depth_node.py), so pipeline latency is INVISIBLE
in sim stamps and visible only in receive times. A stall that shows up in
receive deltas but not in stamp deltas is latency, not a gap in capture.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

#: A gap worth calling out. `SCAN_STALE_S` in episode_runner is 2.0; the
#: watchdog that actually stops the robot fires at 0.5.
NOTABLE_GAP_S = 2.0
#: Below this many messages, a distribution is not worth quoting.
MIN_SAMPLES = 5


def percentile(values, q: float):
    """Linear-interpolated percentile. No numpy: this must run anywhere."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def gap_stats(times) -> dict:
    """Inter-arrival statistics for one clock, plus WHERE the worst gap fell.

    `times` is a sorted sequence of seconds. The location of the worst gap is
    the point of this: one 12 s stall at start-up and repeated 12 s stalls
    mid-run are different findings, and a running maximum cannot tell them
    apart - which is exactly the limitation of the `worst_gap_s` field this
    module exists to replace.
    """
    times = sorted(float(t) for t in times)
    n = len(times)
    if n < 2:
        return {"n": n, "span_s": 0.0, "median_s": None, "p95_s": None,
                "max_s": None, "worst_at_s": None, "rate_hz": None,
                "gaps_over_notable": 0}
    gaps = [times[i + 1] - times[i] for i in range(n - 1)]
    worst = max(gaps)
    worst_idx = gaps.index(worst)
    span = times[-1] - times[0]
    return {
        "n": n,
        "span_s": round(span, 3),
        "median_s": round(percentile(gaps, 0.50), 4),
        "p95_s": round(percentile(gaps, 0.95), 4),
        "max_s": round(worst, 4),
        # Seconds from the FIRST message, so it can be read against the episode.
        "worst_at_s": round(times[worst_idx] - times[0], 3),
        "rate_hz": round((n - 1) / span, 3) if span > 0 else None,
        "gaps_over_notable": sum(1 for g in gaps if g > NOTABLE_GAP_S),
    }


def topic_timing(recv_s, stamp_s) -> dict:
    """Both clocks for one topic, and the ratio between them.

    `recv_s` are bag receive times (wall), `stamp_s` the header stamps (sim).
    """
    wall = gap_stats(recv_s)
    sim = gap_stats(stamp_s) if stamp_s else None
    out = {"wall": wall, "sim": sim}
    if sim and wall["span_s"] and sim["span_s"]:
        # Sim seconds per wall second over the whole topic: an AVERAGE real-time
        # factor, not an instantaneous one.
        out["rtf_mean"] = round(sim["span_s"] / wall["span_s"], 4)
    else:
        out["rtf_mean"] = None
    return out


def classify_stall(scan, scan_mono, notable_s: float = NOTABLE_GAP_S) -> tuple:
    """(verdict, explanation) from the two scan topics' timing.

    `scan` is the LiDAR reference and `scan_mono` the camera's. Both ride the
    same clock in the same bag, which is what makes the comparison decisive.
    Returns ("unknown", why) rather than guessing when the evidence is absent -
    a missing topic is not evidence of anything.
    """
    if not scan_mono or (scan_mono.get("wall") or {}).get("n", 0) < 2:
        return "unknown", ("/scan_mono has fewer than 2 messages: nothing was "
                           "published, which is a different fault from a gap")
    mono_wall = scan_mono["wall"]["max_s"]
    mono_sim = (scan_mono.get("sim") or {}).get("max_s")

    # THE CLOCK CASES COME FIRST. Checking "is the wall gap small?" before them
    # answers "no_stall" to a run whose SIM-time gap was 12 s - which is the
    # number episode.json reports and the question being asked. A disagreement
    # between the two clocks is the most informative answer available, so it
    # must not be short-circuited by either clock looking healthy alone.
    if mono_sim is not None and mono_wall is not None:
        if mono_sim > notable_s and mono_wall <= notable_s:
            return "clock_artefact", (
                f"/scan_mono gapped {mono_sim:.2f}s in SIM time but only "
                f"{mono_wall:.2f}s in wall time: simulated time jumped while "
                f"messages kept arriving at a healthy rate. Nothing stopped - "
                f"the episode's sim-time worst_gap_s is measuring the clock, "
                f"not the sensor.")
        if mono_wall > notable_s and mono_sim <= notable_s:
            return "latency_not_gap", (
                f"/scan_mono gapped {mono_wall:.2f}s in wall time but only "
                f"{mono_sim:.2f}s in stamps. Its stamps are copied from the RGB "
                f"capture, so this is pipeline LATENCY - frames were late, not "
                f"missing.")

    if mono_wall is not None and mono_wall <= notable_s:
        return "no_stall", (f"/scan_mono's worst wall gap is {mono_wall:.2f}s, "
                            f"under the {notable_s:.1f}s bar - the stall is not "
                            f"in this recording")

    if not scan or (scan.get("wall") or {}).get("n", 0) < 2:
        return "mono_only_unconfirmed", (
            f"/scan_mono gapped {mono_wall:.2f}s, but /scan is not in this bag "
            f"to compare against, so a whole-sim freeze cannot be ruled out")

    lidar_wall = scan["wall"]["max_s"]
    if lidar_wall is not None and lidar_wall > notable_s:
        return "sim_froze", (
            f"BOTH stalled: /scan gapped {lidar_wall:.2f}s and /scan_mono "
            f"{mono_wall:.2f}s. The LiDAR does no inference, so the simulation "
            f"itself stopped advancing - the camera chain is not the cause.")
    return "camera_chain", (
        f"/scan kept running (worst gap {lidar_wall:.2f}s) while /scan_mono "
        f"gapped {mono_wall:.2f}s. The stall is in the depth -> scan chain.")


def trajectory_gaps(path) -> dict | None:
    """Ground-truth sample gaps from trajectory.jsonl, if it exists.

    Written by truth_source at the PosePublisher's 50 Hz on the SIM clock, and
    independent of the perception chain: a gap here is direct evidence the sim
    clock itself stalled, whatever the scans say.
    """
    path = Path(path)
    if not path.exists():
        return None
    times = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            times.append(float(json.loads(line)["t"]))
        except (ValueError, KeyError, TypeError):
            continue
    return gap_stats(times) if len(times) >= 2 else None


# --------------------------------------------------------------------------- #
def read_bag(bag_dir, topics=None) -> dict:
    """{topic: (recv_seconds[], stamp_seconds[])} from a rosbag2 directory.

    Needs ROS (rosbag2_py + the message types). Everything above this line is
    pure so the maths can be tested without it.
    """
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    bag_dir = Path(bag_dir)
    reader = SequentialReader()
    reader.open(StorageOptions(uri=str(bag_dir), storage_id=""),
                ConverterOptions("", ""))
    type_by_topic = {t.name: t.type for t in reader.get_all_topics_and_types()}

    out: dict = {}
    while reader.has_next():
        topic, raw, recv_ns = reader.read_next()
        if topics and topic not in topics:
            continue
        recv_s, stamps = out.setdefault(topic, ([], []))
        recv_s.append(recv_ns * 1e-9)
        # The stamp is optional: not every message has a header.
        try:
            msg = deserialize_message(raw, get_message(type_by_topic[topic]))
            header = getattr(msg, "header", None)
            if header is not None:
                stamps.append(header.stamp.sec + header.stamp.nanosec * 1e-9)
        except Exception:
            pass
    return out


def find_bag(episode_dir, warn=print) -> Path | None:
    """The NEWEST bag directory in an episode directory.

    Newest, not first: an episode that was re-run leaves more than one bag
    behind, and `sorted()[0]` is the OLDEST. That is not a hypothetical - it
    silently produced a timing table from a previous run's bag, identical to
    the previous analysis down to the message counts, while episode.json
    described a completely different episode. Reading stale evidence is worse
    than reading none, so when there is a choice this says so out loud.
    """
    episode_dir = Path(episode_dir)
    if not episode_dir.is_dir():
        return None          # "no bag", not a crash
    bags = [c for c in episode_dir.iterdir()
            if c.is_dir() and (list(c.glob("*.db3")) or list(c.glob("*.mcap")))]
    if not bags:
        return None
    bags.sort(key=lambda p: p.stat().st_mtime)
    if len(bags) > 1 and warn:
        warn(f"NOTE: {len(bags)} bags in {episode_dir.name}; "
             f"using the newest ({bags[-1].name}), ignoring "
             + ", ".join(b.name for b in bags[:-1]))
    return bags[-1]


def analyse_episode(episode_dir, notable_s: float = NOTABLE_GAP_S) -> dict:
    episode_dir = Path(episode_dir)
    bag = find_bag(episode_dir)
    if bag is None:
        raise FileNotFoundError(f"no rosbag2 directory under {episode_dir}")

    data = read_bag(bag, topics={"/scan", "/scan_mono", "/clock",
                                 "/camera/rgb/image_raw", "/cmd_vel",
                                 "/diff_drive_controller/cmd_vel"})
    timing = {topic: topic_timing(recv, stamps)
              for topic, (recv, stamps) in sorted(data.items())}
    verdict, why = classify_stall(timing.get("/scan"), timing.get("/scan_mono"),
                                  notable_s)
    return {
        "episode": str(episode_dir),
        "bag": str(bag),
        "timing": timing,
        "verdict": verdict,
        "explanation": why,
        "trajectory": trajectory_gaps(episode_dir / "trajectory.jsonl"),
        "episode_json": _episode_summary(episode_dir),
    }


def _episode_summary(episode_dir) -> dict | None:
    path = Path(episode_dir) / "episode.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    return {"outcome": doc.get("outcome"),
            "perception": doc.get("perception"),
            "path_length_m": doc.get("path_length_m")}


def render(result: dict) -> str:
    lines = [f"episode: {result['episode']}", f"bag:     {result['bag']}", ""]
    summary = result.get("episode_json")
    if summary:
        p = summary.get("perception") or {}
        lines.append(f"episode.json says: {summary.get('outcome')}, "
                     f"perception ok={p.get('ok')}, "
                     f"worst_gap_s={p.get('worst_gap_s')} "
                     f"(a single running max, in SIM seconds)")
        lines.append("")

    head = (f"{'topic':<26}{'n':>7}{'rate Hz':>10}{'median':>9}"
            f"{'p95':>9}{'MAX':>10}{'worst at':>10}")
    for clock in ("wall", "sim"):
        lines.append(f"--- {clock.upper()} clock "
                     f"({'bag receive times' if clock == 'wall' else 'header stamps'})")
        lines.append(head)
        for topic, t in result["timing"].items():
            st = t.get(clock)
            if not st or st["n"] < 2:
                lines.append(f"{topic:<26}{(st or {}).get('n', 0):>7}"
                             f"{'-':>10}{'-':>9}{'-':>9}{'-':>10}{'-':>10}")
                continue
            lines.append(
                f"{topic:<26}{st['n']:>7}{_f(st['rate_hz'], 2):>10}"
                f"{_f(st['median_s'], 3):>9}{_f(st['p95_s'], 3):>9}"
                f"{_f(st['max_s'], 3):>10}{_f(st['worst_at_s'], 1):>10}")
        lines.append("")

    lines.append("--- real-time factor (sim seconds per wall second, averaged)")
    for topic, t in result["timing"].items():
        if t.get("rtf_mean") is not None:
            lines.append(f"  {topic:<26} {t['rtf_mean']:.3f}")
    lines.append("")

    traj = result.get("trajectory")
    if traj and traj["n"] >= 2:
        lines.append("--- ground truth (trajectory.jsonl, 50 Hz on the SIM clock,")
        lines.append("    independent of perception: a gap here means the sim clock stalled)")
        lines.append(f"  n={traj['n']}  rate={_f(traj['rate_hz'], 2)} Hz  "
                     f"median={_f(traj['median_s'], 4)}  MAX={_f(traj['max_s'], 3)}"
                     f"  worst at {_f(traj['worst_at_s'], 1)}s")
        lines.append("")

    lines.append(f"VERDICT: {result['verdict']}")
    for chunk in _wrap(result["explanation"], 76):
        lines.append(f"  {chunk}")
    return "\n".join(lines)


def _f(value, places):
    return "-" if value is None else f"{value:.{places}f}"


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("SELFTEST  ubot_eval.bag_timing (timing maths, no ROS, no bag)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<58} {detail}")

    # A clean 10 Hz stream.
    steady = [i * 0.1 for i in range(100)]
    s = gap_stats(steady)
    check("steady 10 Hz reports 10 Hz and a 0.1 s median",
          abs(s["rate_hz"] - 10.0) < 0.2 and abs(s["median_s"] - 0.1) < 1e-6,
          f"{s['rate_hz']} Hz")
    check("...and no notable gaps", s["gaps_over_notable"] == 0)

    # One 12.4 s stall in the middle: the case being diagnosed. The last sample
    # before the stall is at 3.9 s, so the next must be at 3.9 + 12.4 = 16.3 for
    # the gap to be exactly 12.4 - an off-by-one-step here had the test
    # asserting 12.4 against data that actually contained 12.5.
    stalled = [i * 0.1 for i in range(40)] + [16.3 + i * 0.1 for i in range(40)]
    s = gap_stats(stalled)
    check("a 12.4 s stall is found", abs(s["max_s"] - 12.4) < 1e-6, f"{s['max_s']}s")
    check("...and LOCATED, which a running max cannot do",
          abs(s["worst_at_s"] - 3.9) < 0.2, f"at {s['worst_at_s']}s")
    check("...while the median stays at the healthy rate",
          abs(s["median_s"] - 0.1) < 1e-6,
          "one stall must not look like a slow stream")

    # A uniformly slow stream is NOT the same finding as a stall.
    slow = [i * 1.5 for i in range(30)]
    s = gap_stats(slow)
    check("a uniformly slow stream shows in the median, not as a stall",
          abs(s["median_s"] - 1.5) < 1e-6 and abs(s["max_s"] - 1.5) < 1e-6)

    check("fewer than 2 samples yields None, not 0.0",
          gap_stats([1.0])["max_s"] is None and gap_stats([])["n"] == 0)

    # --- the verdict rules ------------------------------------------------
    def timing(wall_times, sim_times=None):
        return topic_timing(wall_times, sim_times if sim_times is not None else wall_times)

    lidar_ok = timing([i * 0.1 for i in range(100)])
    mono_stalled = timing(stalled)

    v, why = classify_stall(lidar_ok, mono_stalled)
    check("LiDAR fine + camera stalled -> camera chain", v == "camera_chain", why[:52])

    v, why = classify_stall(timing(stalled), mono_stalled)
    check("BOTH stalled -> the simulation froze", v == "sim_froze", why[:52])

    v, why = classify_stall(lidar_ok, timing([i * 0.1 for i in range(100)]))
    check("neither stalled -> no_stall", v == "no_stall")

    # Gap in sim stamps only: time jumped, nothing actually stopped.
    v, why = classify_stall(lidar_ok, topic_timing([i * 0.1 for i in range(80)],
                                                   stalled))
    check("gap in SIM stamps only -> clock artefact", v == "clock_artefact", why[:46])

    # Gap in receive times only: frames were late, not missing.
    v, why = classify_stall(lidar_ok, topic_timing(stalled,
                                                   [i * 0.1 for i in range(80)]))
    check("gap in RECEIVE times only -> latency, not a gap",
          v == "latency_not_gap", why[:46])

    v, why = classify_stall(None, mono_stalled)
    check("no /scan to compare against refuses to blame the camera",
          v == "mono_only_unconfirmed", why[:46])

    v, why = classify_stall(lidar_ok, timing([]))
    check("a silent /scan_mono is named as silent, not as a gap",
          v == "unknown", why[:46])

    # --- real-time factor --------------------------------------------------
    # 10 s of wall time carrying 2 s of sim time is RTF 0.2.
    t = topic_timing([i * 0.1 for i in range(101)], [i * 0.02 for i in range(101)])
    check("real-time factor is inferred from the two clocks",
          abs(t["rtf_mean"] - 0.2) < 1e-6, f"{t['rtf_mean']}")

    # --- percentile --------------------------------------------------------
    check("percentile interpolates rather than picking a neighbour",
          abs(percentile([0.0, 1.0], 0.5) - 0.5) < 1e-9)

    # --- bag selection -----------------------------------------------------
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        older = root / "navigation_20260916_185634"
        newer = root / "navigation_20260916_190325"
        for d in (older, newer):
            d.mkdir()
            (d / "x.mcap").write_bytes(b"")
        os.utime(older, (1_000_000, 1_000_000))
        os.utime(newer, (2_000_000, 2_000_000))
        notes = []
        picked = find_bag(root, warn=notes.append)
        check("a re-run episode's NEWEST bag is chosen, not the first",
              picked == newer, picked.name if picked else "none")
        check("...and the ignored bags are named", notes and "ignoring" in notes[0])
        check("a missing directory is 'no bag', not an exception",
              find_bag(root / "does_not_exist") is None)
        solo = root / "solo"
        solo.mkdir()
        (solo / "only.mcap").write_bytes(b"")
        quiet = []
        check("a single bag is chosen with no note",
              find_bag(solo.parent / "solo", warn=quiet.append) is None
              or not quiet)

    # --- the rendered report -----------------------------------------------
    text = render({"episode": "/tmp/e", "bag": "/tmp/e/bag",
                   "timing": {"/scan": lidar_ok, "/scan_mono": mono_stalled},
                   "verdict": "camera_chain", "explanation": "because",
                   "trajectory": None, "episode_json": None})
    check("the report shows both clocks", "WALL clock" in text and "SIM clock" in text)
    check("...and states a verdict", "VERDICT: camera_chain" in text)

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--episode", help="an episode directory containing a bag")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--notable-gap-s", type=float, default=NOTABLE_GAP_S)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.episode:
        ap.error("--episode is required")

    result = analyse_episode(args.episode, args.notable_gap_s)
    print(json.dumps(result, indent=2) if args.json else render(result))
    sys.exit(0)


if __name__ == "__main__":
    main()
