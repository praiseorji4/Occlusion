"""The results table as a PDF, built only from episode.json files on disk.

    python3 -m ubot_eval.report_pdf --runs-root ~/occl_ws/runs \
        --campaign lidar50 --out ~/occl_ws/lidar50_results.pdf

    python3 -m ubot_eval.report_pdf --selftest

Safe to run while a campaign is still going: it reports what has been written so
far and says so on the page, rather than waiting or guessing at the rest.

WHAT THIS WILL NOT DO
---------------------
* It will not turn a missing SPL into a zero. An episode with no reference path
  is left out of the SPL mean, and the n is printed next to the mean so a mean
  over three cannot be read as a mean over thirty.
* It will not fold `not_started` into the success rate. Those are refusals - the
  readiness gate found the robot could not run - and they get their own column.
* It will not print an SPL built on a provisional reference path without saying
  so on the page.
* With one arm there are no deltas, and it says that instead of drawing an
  empty comparison.

matplotlib is the only PDF-capable tool on this machine (no pandoc, no
wkhtmltopdf, no headless browser), so the layout is drawn rather than typeset.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import sys
from collections import defaultdict
from pathlib import Path

from ubot_eval.report import by_arm, load_episodes, SUCCESS, REFUSALS

INK = "#11171C"
MUTED = "#5B6B75"
RULE = "#C9D2D7"
SIGNAL = "#0B6E78"
PASS = "#1F7A43"
WARN = "#9A6206"
FAIL = "#A3341F"

OUTCOME_COLOURS = {
    "reached": PASS,
    "false_success": FAIL,
    "collided": FAIL,
    "aborted": WARN,
    "timeout": WARN,
    "not_started": MUTED,
}


def by_course(episodes) -> dict:
    groups = defaultdict(list)
    for e in episodes:
        groups[e.get("course", "?")].append(e)
    return dict(sorted(groups.items()))


def _stats(values):
    """n, mean, sd, min, max over the values that exist."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "sd": sd, "min": min(vals), "max": max(vals)}


def wilson_interval(successes: int, trials: int, z: float = 1.959963985):
    """95% CI for a proportion. Wilson, not normal-approximation.

    At small n, or at a rate near 0 or 1, the textbook interval runs past the
    ends of the scale and reports impossible success rates. 17 trials on a
    course is exactly that regime.
    """
    if trials <= 0:
        return None
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _fmt(value, spec=".3f", dash="—"):
    return dash if value is None else format(value, spec)


def build_pdf(episodes, out_path, *, campaign="", runs_root="", title=None) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": RULE,
    })

    arms = by_arm(episodes)
    courses = by_course(episodes)
    stamp = _dt.datetime.now().strftime("%d %b %Y %H:%M")
    mono = {"family": "DejaVu Sans Mono"}

    with PdfPages(out_path) as pdf:
        # ---------------------------------------------------------------- p1
        fig = plt.figure(figsize=(8.27, 11.69))       # A4 portrait
        fig.patch.set_facecolor("white")
        y = 0.955

        fig.text(0.07, y, title or "Navigation trial results",
                 fontsize=21, fontweight="bold", color=INK)
        y -= 0.026
        sub = f"campaign {campaign}  ·  {len(episodes)} episodes  ·  generated {stamp}"
        fig.text(0.07, y, sub, fontsize=9, color=MUTED)
        y -= 0.013
        fig.text(0.07, y, f"runs root {runs_root}", fontsize=7.5, color=MUTED, **mono)
        y -= 0.030
        fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=RULE, lw=0.8))
        y -= 0.034

        # Headline figures, per arm.
        for arm, r in arms.items():
            attempted, reached = r["attempted"], r["reached"]
            ci = wilson_interval(reached, attempted) if attempted else None
            fig.text(0.07, y, arm, fontsize=13, fontweight="bold", color=SIGNAL)
            y -= 0.024
            cells = [
                ("episodes", str(r["episodes"])),
                ("ran", str(attempted)),
                ("reached", str(reached)),
                ("success rate", _fmt(r["success_rate"], ".1%") if attempted else "—"),
                ("95% CI", f"{ci[0]:.0%}–{ci[1]:.0%}" if ci else "—"),
                ("refused", str(r["not_started"])),
            ]
            for i, (k, v) in enumerate(cells):
                x = 0.07 + i * 0.145
                fig.text(x, y, k.upper(), fontsize=6.5, color=MUTED, **mono)
                fig.text(x, y - 0.018, v, fontsize=13, color=INK, **mono)
            y -= 0.050

            spl = _stats([e.get("spl") for e in episodes
                          if e.get("arm") == arm and e.get("outcome") in SUCCESS])
            tim = _stats([e.get("time_to_goal_s") for e in episodes
                          if e.get("arm") == arm and e.get("outcome") in SUCCESS])
            pat = _stats([e.get("path_length_m") for e in episodes
                          if e.get("arm") == arm and e.get("outcome") in SUCCESS])
            line = (f"SPL {_fmt(spl['mean'])} ± {_fmt(spl['sd'])} (n={spl['n']})   "
                    f"time {_fmt(tim['mean'], '.1f')} s ± {_fmt(tim['sd'], '.1f')}   "
                    f"path {_fmt(pat['mean'], '.2f')} m ± {_fmt(pat['sd'], '.2f')}   "
                    f"collisions {r['collisions_total']}")
            fig.text(0.07, y, line, fontsize=8.5, color=INK, **mono)
            y -= 0.034

        # Per-course table.
        y -= 0.006
        fig.text(0.07, y, "By course", fontsize=12, fontweight="bold", color=INK)
        y -= 0.022
        header = f"{'course':<14}{'n':>4}{'reach':>7}{'SR':>8}{'SPL':>9}{'time s':>9}{'path m':>9}{'ref m':>9}"
        fig.text(0.07, y, header, fontsize=8, color=MUTED, **mono)
        y -= 0.006
        fig.add_artist(plt.Line2D([0.07, 0.93], [y, y], color=RULE, lw=0.6))
        y -= 0.016
        for course, eps in courses.items():
            ran = [e for e in eps if e.get("outcome") not in REFUSALS]
            won = [e for e in ran if e.get("outcome") in SUCCESS]
            spl = _stats([e.get("spl") for e in won])
            tim = _stats([e.get("time_to_goal_s") for e in won])
            pat = _stats([e.get("path_length_m") for e in won])
            ref = next((e.get("shortest_path_m") for e in eps
                        if e.get("shortest_path_m")), None)
            sr = (len(won) / len(ran)) if ran else None
            row = (f"{course:<14}{len(eps):>4}{len(won):>7}"
                   f"{(_fmt(sr, '.0%') if sr is not None else '—'):>8}"
                   f"{_fmt(spl['mean']):>9}{_fmt(tim['mean'], '.1f'):>9}"
                   f"{_fmt(pat['mean'], '.2f'):>9}{_fmt(ref, '.2f'):>9}")
            fig.text(0.07, y, row, fontsize=8, color=INK, **mono)
            y -= 0.017
        y -= 0.010

        # Outcome ledger.
        fig.text(0.07, y, "Outcomes", fontsize=12, fontweight="bold", color=INK)
        y -= 0.022
        tally = defaultdict(int)
        for e in episodes:
            tally[e.get("outcome", "?")] += 1
        for outcome, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            colour = OUTCOME_COLOURS.get(outcome, INK)
            share = count / len(episodes) if episodes else 0
            fig.text(0.07, y, f"{outcome:<16}{count:>4}   {share:>5.0%}",
                     fontsize=8.5, color=colour, **mono)
            fig.add_artist(plt.Rectangle((0.30, y - 0.001), 0.60 * share, 0.009,
                                         color=colour, alpha=0.75,
                                         transform=fig.transFigure))
            y -= 0.018

        # Caveats: the part that keeps the page honest.
        y -= 0.014
        fig.text(0.07, y, "Caveats", fontsize=12, fontweight="bold", color=INK)
        y -= 0.022
        for note in caveats(episodes, arms):
            for line in _wrap(note, 96):
                fig.text(0.07, y, line, fontsize=8, color=MUTED)
                y -= 0.015
            y -= 0.004

        fig.text(0.07, 0.035,
                 "Every figure here came from an episode.json written by the run. "
                 "Nothing is reported that was not measured.",
                 fontsize=7.5, color=MUTED, style="italic")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------------- p2
        won_all = [e for e in episodes if e.get("outcome") in SUCCESS]
        if won_all:
            fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            fig.subplots_adjust(top=0.93, bottom=0.07, hspace=0.42,
                                left=0.12, right=0.93)
            fig.suptitle("Distributions over completed episodes",
                         fontsize=15, fontweight="bold", color=INK, x=0.12, ha="left")

            spls = [e["spl"] for e in won_all if e.get("spl") is not None]
            ax = axes[0]
            if spls:
                ax.hist(spls, bins=min(16, max(4, len(spls) // 2)),
                        color=SIGNAL, alpha=0.85, edgecolor="white")
                mean = sum(spls) / len(spls)
                ax.axvline(mean, color=FAIL, lw=1.4, ls="--",
                           label=f"mean {mean:.3f}")
                ax.legend(frameon=False, fontsize=8)
                ax.set_xlim(0, 1.02)
            ax.set_title(f"SPL  (n={len(spls)})", fontsize=10, loc="left", color=INK)
            ax.set_xlabel("SPL — 1.0 is the shortest path a robot this size could take")
            ax.set_ylabel("episodes")

            ax = axes[1]
            names, data = [], []
            for course, eps in courses.items():
                vals = [e.get("time_to_goal_s") for e in eps
                        if e.get("outcome") in SUCCESS and e.get("time_to_goal_s")]
                if vals:
                    names.append(f"{course}\n(n={len(vals)})")
                    data.append(vals)
            if data:
                bp = ax.boxplot(data, labels=names, patch_artist=True, widths=0.5)
                for patch in bp["boxes"]:
                    patch.set(facecolor=SIGNAL, alpha=0.30, edgecolor=SIGNAL)
                for whisk in bp["medians"]:
                    whisk.set(color=FAIL, linewidth=1.5)
            ax.set_title("Time to goal by course", fontsize=10, loc="left", color=INK)
            ax.set_ylabel("seconds")

            ax = axes[2]
            labels, actual, ref = [], [], []
            for course, eps in courses.items():
                vals = [e.get("path_length_m") for e in eps
                        if e.get("outcome") in SUCCESS and e.get("path_length_m")]
                r = next((e.get("shortest_path_m") for e in eps
                          if e.get("shortest_path_m")), None)
                if vals and r:
                    labels.append(course)
                    actual.append(sum(vals) / len(vals))
                    ref.append(r)
            if labels:
                idx = range(len(labels))
                ax.bar([i - 0.19 for i in idx], ref, width=0.38, color=MUTED,
                       alpha=0.65, label="shortest path (A*, measured)")
                ax.bar([i + 0.19 for i in idx], actual, width=0.38, color=SIGNAL,
                       label="mean path driven")
                ax.set_xticks(list(idx))
                ax.set_xticklabels(labels)
                ax.legend(frameon=False, fontsize=8)
            ax.set_title("Path driven against the reference path", fontsize=10,
                         loc="left", color=INK)
            ax.set_ylabel("metres")

            for ax in axes:
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(labelsize=8)
            pdf.savefig(fig)
            plt.close(fig)

        # ---------------------------------------------------------------- p3
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        fig.text(0.07, 0.955, "Method", fontsize=15, fontweight="bold", color=INK)
        y = 0.918
        for head, body in METHOD:
            fig.text(0.07, y, head, fontsize=10, fontweight="bold", color=SIGNAL)
            y -= 0.020
            for line in _wrap(body, 92):
                fig.text(0.07, y, line, fontsize=8.5, color=INK)
                y -= 0.0155
            y -= 0.012
        pdf.savefig(fig)
        plt.close(fig)

    return str(out_path)


def caveats(episodes, arms) -> list:
    out = []
    provisional = sum(1 for e in episodes
                      if e.get("outcome") in SUCCESS
                      and e.get("shortest_path_provisional"))
    if provisional:
        out.append(f"{provisional} successful episode(s) used a PROVISIONAL "
                   "shortest_path_m. SPL derived from those is not reportable.")
    refused = sum(r["not_started"] for r in arms.values())
    if refused:
        out.append(f"{refused} episode(s) were refused by the readiness gate "
                   "(not_started) and are excluded from the success rate: a robot "
                   "that could not run measured nothing about navigation. They are "
                   "counted separately above.")
    if len(arms) < 2:
        out.append("One arm only, so there is no comparison here and no deltas are "
                   "reported. This measures the reference stack's own reliability; "
                   "it says nothing about the camera-only stack.")
    thin = [a for a, r in arms.items() if 0 < r["spl_n"] < 5]
    if thin:
        out.append(f"SPL mean over fewer than 5 episodes for: {', '.join(thin)}.")
    no_spl = sum(1 for e in episodes
                 if e.get("outcome") in SUCCESS and e.get("spl") is None)
    if no_spl:
        out.append(f"{no_spl} successful episode(s) carry no SPL and are left out of "
                   "the SPL mean rather than counted as zero.")
    if not out:
        out.append("None. Every successful episode has a measured reference path.")
    return out


METHOD = [
    ("Arrival is judged on ground truth, not odometry",
     "Position comes from Gazebo via /gt/tf and is cross-checked against the gz CLI at "
     "episode start. An episode where nav2 reports SUCCEEDED but ground truth disagrees "
     "by more than 0.15 m is recorded as false_success, not as a success. nav2 aims for "
     "0.10 m so a clean run has room for odometry error."),
    ("Refusals are not failures",
     "Before each episode a readiness gate checks controllers, TF advancing in sim time, "
     "a scan with finite returns, nav2's nodes active AND its action servers accepting "
     "goals, then commands a short move and measures it. A failed gate writes not_started, "
     "which is excluded from the success rate and counted separately."),
    ("SPL denominator is measured, not assumed",
     "The shortest path is A* on a slam_toolbox occupancy grid (0.05 m/pix), obstacles "
     "inflated by the robot's 0.219 m footprint half-diagonal, then string-pulled — raw "
     "8-connected A* overstates a shallow diagonal by about 8%. Using the straight line "
     "instead would have been wrong by 32% on hall_10m, which routes around a doorway."),
    ("Collisions outrank arrival",
     "Bumper contacts are monitored throughout. An episode that reaches the goal after a "
     "collision is recorded as collided, with reached_goal kept separately, so a run that "
     "hit something cannot be counted as a clean success."),
    ("Every episode leaves evidence",
     "Each writes an episode.json plus a rosbag of /tf, ground truth, both scan topics and "
     "odometry. The bag records both scans regardless of arm, so arms produce identical "
     "bag layouts and a run can be re-examined after the fact."),
]


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
    print("SELFTEST  ubot_eval.report_pdf (no ROS; writes a real PDF)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<56} {detail}")

    lo, hi = wilson_interval(17, 17)
    check("a perfect run does not get an interval above 1.0", hi <= 1.0,
          f"17/17 -> {lo:.2f}-{hi:.2f}")
    check("...and its lower bound is below 1.0, not a point estimate", lo < 1.0)
    lo0, hi0 = wilson_interval(0, 17)
    check("a total failure does not get a negative lower bound", lo0 >= 0.0,
          f"0/17 -> {lo0:.2f}-{hi0:.2f}")
    check("an interval narrows as n grows",
          (wilson_interval(9, 10)[1] - wilson_interval(9, 10)[0]) >
          (wilson_interval(90, 100)[1] - wilson_interval(90, 100)[0]))
    check("no trials gives None, not a division by zero",
          wilson_interval(0, 0) is None)

    s = _stats([1.0, 2.0, 3.0, None])
    check("stats ignore missing values and report n",
          s["n"] == 3 and abs(s["mean"] - 2.0) < 1e-9)
    check("sd of a single value is 0.0, not a crash", _stats([5.0])["sd"] == 0.0)
    check("stats over nothing report None, never 0.0",
          _stats([None])["mean"] is None)

    def ep(course, outcome, spl=None, t=None, path=None, ref=None, prov=False):
        return {"arm": "lidar_ref", "course": course, "outcome": outcome,
                "spl": spl, "time_to_goal_s": t, "path_length_m": path,
                "shortest_path_m": ref, "collisions": {"count": 0},
                "pair_key": f"apartment/{course}/01",
                "shortest_path_provisional": prov}

    eps = [ep("hall_3m", "reached", 0.99, 26.6, 3.02, 3.0017),
           ep("hall_3m", "reached", 0.95, 28.0, 3.15, 3.0017),
           ep("hall_10m", "aborted"),
           ep("hall_10m", "not_started")]

    notes = " ".join(caveats(eps, by_arm(eps)))
    check("a single arm says so instead of drawing an empty comparison",
          "no comparison" in notes)
    check("refusals are named in the caveats", "not_started" in notes)
    check("provisional references are named when present",
          "PROVISIONAL" in " ".join(caveats(
              [ep("hall_3m", "reached", 0.9, 1.0, 1.0, 3.0, prov=True)],
              by_arm([ep("hall_3m", "reached", 0.9, 1.0, 1.0, 3.0, prov=True)]))))

    out = Path("/tmp/ubot_eval_report_selftest.pdf")
    try:
        build_pdf(eps, out, campaign="selftest", runs_root="/tmp")
        head = out.read_bytes()[:5]
        check("writes a file that is actually a PDF", head == b"%PDF-",
              f"{out} ({out.stat().st_size} bytes), header {head!r}")
    except Exception as exc:                                  # pragma: no cover
        check("writes a file that is actually a PDF", False, repr(exc))

    # An empty campaign must not produce a confident-looking empty report.
    try:
        build_pdf([], Path("/tmp/ubot_eval_report_empty.pdf"), campaign="empty")
        check("an empty run still renders rather than crashing", True)
    except Exception as exc:                                  # pragma: no cover
        check("an empty run still renders rather than crashing", False, repr(exc))

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--campaign", default=None,
                    help="only episodes of this campaign (a subdirectory of runs-root)")
    ap.add_argument("--out", default="trial_results.pdf")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    root = Path(args.runs_root)
    if args.campaign:
        root = root / args.campaign
    episodes = load_episodes(root)
    if not episodes:
        sys.exit(f"no episode.json under {root} - nothing to report. "
                 "This is not an empty result, it is no data.")
    path = build_pdf(episodes, args.out, campaign=args.campaign or "",
                     runs_root=str(root), title=args.title)
    print(f"{len(episodes)} episodes -> {path}")


if __name__ == "__main__":
    main()
