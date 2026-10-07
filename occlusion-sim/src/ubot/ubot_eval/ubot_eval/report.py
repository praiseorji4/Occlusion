"""The results table, built only from episode.json files on disk.

    python3 -m ubot_eval.report --runs-root ~/occl_ws/runs
    python3 -m ubot_eval.report --runs-root ~/occl_ws/runs --markdown
    python3 -m ubot_eval.report --selftest

WHAT IS AND IS NOT REPORTED
---------------------------
* `not_started` episodes are counted and shown SEPARATELY. They are refusals -
  the gate found the robot could not run - and folding them into the success
  rate would turn a broken bring-up into evidence about navigation.
* SPL is averaged over episodes that have one. An episode with no reference path
  has no SPL, and it is left out of that mean rather than counted as zero. The
  count is printed next to the mean so a mean over three episodes cannot be
  mistaken for a mean over thirty.
* A course whose `shortest_path_m` is provisional taints every SPL derived from
  it. Those episodes are flagged, and the table says so in a footer rather than
  printing a clean-looking number.
* `false_success` is its own outcome: nav2 said SUCCEEDED and ground truth
  disagreed. It counts as a failure for success rate, and it is shown separately
  because it means something different from a timeout.

WHY PAIRED
----------
Arms are compared on shared `pair_key`s (world/course/trial). A mean over
different courses per arm is not a comparison, so the delta table only uses
pair_keys BOTH arms actually completed, and says how many that was.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SUCCESS = ("reached",)
REFUSALS = ("not_started",)
#: McNemar works on the pairs where the arms DISAGREE. Fewer than this and
#: no success-rate difference is separable from noise.
MIN_DISCORDANT_PAIRS = 6


def load_episodes(runs_root) -> list:
    """Every episode.json under runs_root, as dicts, with the path attached."""
    out = []
    for path in sorted(Path(runs_root).rglob("episode.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  skipping unreadable {path}: {exc}", file=sys.stderr)
            continue
        data["_path"] = str(path)
        out.append(data)
    return out


def _mean(values):
    values = [v for v in values if v is not None]
    return (sum(values) / len(values)) if values else None


def summarise_arm(episodes) -> dict:
    """One arm's row. Counts first, then means over what is actually there."""
    attempted = [e for e in episodes if e.get("outcome") not in REFUSALS]
    refused = [e for e in episodes if e.get("outcome") in REFUSALS]
    reached = [e for e in attempted if e.get("outcome") in SUCCESS]
    spls = [e.get("spl") for e in reached if e.get("spl") is not None]
    times = [e.get("time_to_goal_s") for e in reached]
    paths = [e.get("path_length_m") for e in reached]
    collisions = [(e.get("collisions") or {}).get("count") or 0 for e in attempted]

    outcomes = defaultdict(int)
    for e in episodes:
        outcomes[e.get("outcome", "?")] += 1

    return {
        "episodes": len(episodes),
        "attempted": len(attempted),
        "not_started": len(refused),
        "reached": len(reached),
        "success_rate": (len(reached) / len(attempted)) if attempted else None,
        "spl_mean": _mean(spls),
        "spl_n": len(spls),
        "time_mean_s": _mean(times),
        "path_mean_m": _mean(paths),
        "collisions_total": sum(collisions),
        "outcomes": dict(outcomes),
        "provisional_spl": sum(1 for e in reached
                               if e.get("shortest_path_provisional")),
    }


def by_arm(episodes) -> dict:
    groups = defaultdict(list)
    for e in episodes:
        groups[e.get("arm", "?")].append(e)
    return {arm: summarise_arm(eps) for arm, eps in sorted(groups.items())}


def paired_deltas(episodes, arm_a: str, arm_b: str) -> dict:
    """arm_b minus arm_a over pair_keys BOTH completed.

    Unpaired episodes are excluded and counted, because a delta taken over
    different courses is not a delta.
    """
    index = defaultdict(dict)
    for e in episodes:
        key = e.get("pair_key")
        if key and e.get("arm"):
            index[key][e["arm"]] = e

    shared = [k for k, v in index.items() if arm_a in v and arm_b in v]
    if not shared:
        return {"pairs": 0, "note": f"no pair_key completed by both "
                                    f"{arm_a} and {arm_b}"}

    def ok(e):
        return 1.0 if e.get("outcome") in SUCCESS else 0.0

    sr_a = _mean([ok(index[k][arm_a]) for k in shared])
    sr_b = _mean([ok(index[k][arm_b]) for k in shared])
    spl_pairs = [(index[k][arm_a].get("spl"), index[k][arm_b].get("spl"))
                 for k in shared]
    spl_pairs = [(a, b) for a, b in spl_pairs if a is not None and b is not None]

    # Discordant pairs are what McNemar actually tests on.
    discordant = sum(1 for k in shared
                     if ok(index[k][arm_a]) != ok(index[k][arm_b]))

    return {
        "pairs": len(shared),
        "success_rate_a": sr_a,
        "success_rate_b": sr_b,
        "success_rate_delta": (sr_b - sr_a) if None not in (sr_a, sr_b) else None,
        "discordant_pairs": discordant,
        "spl_delta": _mean([b - a for a, b in spl_pairs]) if spl_pairs else None,
        "spl_pairs": len(spl_pairs),
    }


def _fmt(value, spec=".3f", dash="-"):
    return dash if value is None else format(value, spec)


def render_table(rows: dict, *, markdown=False) -> str:
    head = ["arm", "eps", "run", "reach", "SR", "SPL", "SPL n", "time s",
            "path m", "coll", "refused"]
    body = []
    for arm, r in rows.items():
        body.append([
            arm, str(r["episodes"]), str(r["attempted"]), str(r["reached"]),
            _fmt(r["success_rate"], ".2f"), _fmt(r["spl_mean"]), str(r["spl_n"]),
            _fmt(r["time_mean_s"], ".1f"), _fmt(r["path_mean_m"], ".2f"),
            str(r["collisions_total"]), str(r["not_started"]),
        ])
    widths = [max(len(head[i]), *(len(row[i]) for row in body)) if body
              else len(head[i]) for i in range(len(head))]

    if markdown:
        out = ["| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |",
               "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        out += ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
                for row in body]
        return "\n".join(out)

    out = ["  ".join(h.ljust(w) for h, w in zip(head, widths))]
    out.append("  ".join("-" * w for w in widths))
    out += ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in body]
    return "\n".join(out)


def build_report(runs_root, *, comparisons=None, markdown=False) -> str:
    episodes = load_episodes(runs_root)
    if not episodes:
        return (f"No episode.json under {runs_root}.\n"
                "Nothing to report - this is not an empty result, it is no data.")

    rows = by_arm(episodes)
    lines = [f"EPISODES: {len(episodes)} under {runs_root}", ""]
    lines.append(render_table(rows, markdown=markdown))
    lines.append("")

    lines.append("OUTCOMES")
    for arm, r in rows.items():
        detail = ", ".join(f"{k}={v}" for k, v in sorted(r["outcomes"].items()))
        lines.append(f"  {arm}: {detail}")
    lines.append("")

    underpowered = []
    if comparisons:
        lines.append("PAIRED DELTAS (arm_b - arm_a, shared pair_keys only)")
        for arm_a, arm_b in comparisons:
            d = paired_deltas(episodes, arm_a, arm_b)
            if not d.get("pairs"):
                lines.append(f"  {arm_a} -> {arm_b}: {d.get('note')}")
                continue
            # McNemar tests the DISCORDANT pairs, so that is the sample size
            # that matters - not the episode count. Below ~6 discordant pairs a
            # difference of any size is indistinguishable from chance, and a
            # delta printed without that warning invites exactly the wrong
            # conclusion.
            if d["discordant_pairs"] < MIN_DISCORDANT_PAIRS:
                underpowered.append(
                    f"{arm_a} -> {arm_b}: {d['discordant_pairs']} discordant pair(s) "
                    f"of {d['pairs']}. The success-rate delta "
                    f"({_fmt(d['success_rate_delta'], '+.2f')}) CANNOT be "
                    f"distinguished from chance at this sample size; it is a "
                    f"direction, not a result.")
            lines.append(
                f"  {arm_a} -> {arm_b}: pairs={d['pairs']} "
                f"SR {_fmt(d['success_rate_a'], '.2f')} -> "
                f"{_fmt(d['success_rate_b'], '.2f')} "
                f"(delta {_fmt(d['success_rate_delta'], '+.2f')}, "
                f"discordant {d['discordant_pairs']}) "
                f"SPL delta {_fmt(d['spl_delta'], '+.3f')} (n={d['spl_pairs']})")
        lines.append("")

    provisional = sum(r["provisional_spl"] for r in rows.values())
    caveats = []
    if provisional:
        caveats.append(
            f"{provisional} successful episode(s) used a PROVISIONAL "
            "shortest_path_m (straight-line placeholder). Every SPL derived "
            "from them is not reportable until the mapping pass and A* run: "
            "scripts/mapping_pass.sh then scripts/measure_courses.sh.")
    caveats.extend(underpowered)
    thin = [a for a, r in rows.items() if r["spl_n"] and r["spl_n"] < 5]
    if thin:
        caveats.append(f"SPL mean over fewer than 5 episodes for: {', '.join(thin)}.")
    refused = sum(r["not_started"] for r in rows.values())
    if refused:
        caveats.append(
            f"{refused} episode(s) were REFUSED by the readiness gate "
            "(not_started). They are excluded from success rate on purpose: "
            "a robot that could not run measured nothing about navigation.")
    if caveats:
        lines.append("CAVEATS")
        lines += [f"  * {c}" for c in caveats]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("SELFTEST  ubot_eval.report (aggregation only, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<56} {detail}")

    def ep(arm, key, outcome, spl=None, t=None, path=None, coll=0, prov=False):
        return {"arm": arm, "pair_key": key, "outcome": outcome, "spl": spl,
                "time_to_goal_s": t, "path_length_m": path,
                "collisions": {"count": coll},
                "shortest_path_provisional": prov}

    eps = [
        ep("lidar_ref", "w/c/01", "reached", 0.9, 20.0, 3.0),
        ep("lidar_ref", "w/c/02", "reached", 0.8, 25.0, 3.2),
        ep("lidar_ref", "w/c/03", "not_started"),
        ep("mono_baseline", "w/c/01", "reached", 0.7, 30.0, 3.4),
        ep("mono_baseline", "w/c/02", "aborted"),
        ep("mono_baseline", "w/c/03", "not_started"),
    ]
    rows = by_arm(eps)

    check("not_started is excluded from the success rate",
          rows["lidar_ref"]["attempted"] == 2 and
          abs(rows["lidar_ref"]["success_rate"] - 1.0) < 1e-9,
          f"attempted={rows['lidar_ref']['attempted']}")
    check("...and is still counted and shown",
          rows["lidar_ref"]["not_started"] == 1)
    check("a failed arm's success rate reflects the failure",
          abs(rows["mono_baseline"]["success_rate"] - 0.5) < 1e-9,
          _fmt(rows["mono_baseline"]["success_rate"], ".2f"))
    check("SPL mean ignores episodes without one, and reports n",
          rows["mono_baseline"]["spl_n"] == 1 and
          abs(rows["mono_baseline"]["spl_mean"] - 0.7) < 1e-9)

    # An SPL of None must never be silently read as zero.
    none_spl = by_arm([ep("x", "w/c/01", "reached", None, 10.0, 2.0)])["x"]
    check("a success with no SPL does not become SPL 0.0",
          none_spl["spl_mean"] is None and none_spl["spl_n"] == 0)

    text_thin = build_report_from(eps, comparisons=[("lidar_ref", "mono_baseline")])
    check("a delta from too few discordant pairs is called out",
          "CANNOT be distinguished from chance" in text_thin)

    d = paired_deltas(eps, "lidar_ref", "mono_baseline")
    check("deltas use only pair_keys both arms completed", d["pairs"] == 3,
          f"pairs={d['pairs']}")
    check("the success-rate delta has the right sign",
          d["success_rate_delta"] is not None and d["success_rate_delta"] < 0,
          _fmt(d["success_rate_delta"], "+.2f"))
    check("discordant pairs are counted for McNemar",
          d["discordant_pairs"] == 1, str(d["discordant_pairs"]))

    lonely = paired_deltas(
        [ep("a", "w/c/01", "reached", 0.9), ep("b", "w/c/02", "reached", 0.9)],
        "a", "b")
    check("no shared pair_key is refused, not averaged anyway",
          lonely["pairs"] == 0 and "note" in lonely)

    text = build_report_from(eps, comparisons=[("lidar_ref", "mono_baseline")])
    check("provisional SPL is called out in the footer",
          "PROVISIONAL" in build_report_from(
              [ep("a", "w/c/01", "reached", 0.9, prov=True)]).upper())
    check("refusals are called out in the footer", "REFUSED" in text)
    check("the table renders every arm",
          "lidar_ref" in text and "mono_baseline" in text)

    empty = build_report_from([])
    check("no data says so instead of printing an empty table",
          "no data" in empty.lower())

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def build_report_from(episodes, *, comparisons=None, markdown=False) -> str:
    """build_report's body, against a list already in memory (for tests)."""
    if not episodes:
        return "Nothing to report - this is not an empty result, it is no data."
    rows = by_arm(episodes)
    lines = [f"EPISODES: {len(episodes)}", "", render_table(rows, markdown=markdown), ""]
    lines.append("OUTCOMES")
    for arm, r in rows.items():
        lines.append(f"  {arm}: " + ", ".join(f"{k}={v}" for k, v
                                              in sorted(r["outcomes"].items())))
    underpowered = []
    if comparisons:
        lines += ["", "PAIRED DELTAS (arm_b - arm_a, shared pair_keys only)"]
        for a, b in comparisons:
            d = paired_deltas(episodes, a, b)
            lines.append(f"  {a} -> {b}: " + (d.get("note") or
                         f"pairs={d['pairs']} delta "
                         f"{_fmt(d['success_rate_delta'], '+.2f')}"))
            if d.get("pairs") and d["discordant_pairs"] < MIN_DISCORDANT_PAIRS:
                underpowered.append(
                    f"{a} -> {b}: {d['discordant_pairs']} discordant pair(s); the "
                    f"delta CANNOT be distinguished from chance at this sample size.")
    caveats_extra = underpowered
    prov = sum(r["provisional_spl"] for r in rows.values())
    refused = sum(r["not_started"] for r in rows.values())
    caveats = list(caveats_extra)
    if prov:
        caveats.append(f"{prov} success(es) used a PROVISIONAL shortest_path_m; "
                       "those SPLs are not reportable.")
    if refused:
        caveats.append(f"{refused} episode(s) were REFUSED by the readiness gate.")
    if caveats:
        lines += ["", "CAVEATS"] + [f"  * {c}" for c in caveats]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--compare", action="append", default=[],
                    metavar="ARM_A:ARM_B", help="repeatable paired comparison")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    comparisons = [tuple(c.split(":", 1)) for c in args.compare if ":" in c]
    if not comparisons:
        comparisons = [("mono_baseline", "mono_A"),
                       ("mono_baseline", "mono_B"),
                       ("mono_baseline", "mono_AB"),
                       ("mono_baseline", "lidar_ref"),
                       ("mono_baseline", "mono_baseline_dup")]
    print(build_report(args.runs_root, comparisons=comparisons,
                       markdown=args.markdown))


if __name__ == "__main__":
    main()
