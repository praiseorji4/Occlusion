#!/usr/bin/env python3
"""Fit the monocular depth model against Gazebo's true depth -> depth_scale.json.

    # with a mono stack running:
    python3 calibrate_sim_depth.py --out ~/occl_ws/config/depth_scale_sim.json

    python3 calibrate_sim_depth.py --selftest        # the maths, no ROS

WHY SIM GETS A BETTER REFERENCE THAN THE ROBOT
----------------------------------------------
On the real robot the reference is OAK-D stereo, or people's apparent height,
and both are sparse and noisy - `occlusion/eval/depth_affine.py` in the research
tree does that job. In simulation Gazebo already publishes the TRUE depth of
every pixel on /camera/depth/image_raw, bridged next to the RGB the model sees.
It is dense, metric, and spans the whole range, so the fit is better conditioned
than anything available on hardware. Use this in sim; use depth_affine.py for
the robot. The two produce the same file.

WHAT IS BEING FITTED
--------------------
    z_model = a * z_true + b

mono_depth_node inverts it as (z_model - b) / a, so `a` is the gain and `b` the
offset, exactly as `occlusion/core/depth_scale.py` defines them. The research
tree's warning applies here too: the gain cancels in the occlusion SCORE, but
the offset does not, and an uncalibrated model at this viewpoint is wrong by
roughly a factor of 2 - which is why the node refuses to call itself calibrated
and says so in its log.

THIS REFUSES RATHER THAN GUESSING
---------------------------------
If the robot never moved, every sample is at the same distance and `a` and `b`
cannot be separated: any line through one point fits. The gates below are the
same ones depth_scale.fit_affine uses - a minimum number of points, and a
minimum SPREAD of reference distances both absolutely and relative to the
median. A refusal names what was wrong. Drive the robot somewhere with near and
far surfaces in view and run it again.

The code is not imported from the research repo on purpose: those two trees
share a FILE FORMAT, not a library (see Occlusion/CLAUDE.md).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

MIN_POINTS = 500
MIN_SPAN_M = 1.0          # absolute spread of reference depths, 5th-95th pct
MIN_SPAN_FRAC = 0.25      # ...and relative to the median
TRIM_ITERS = 3
TRIM_KEEP = 0.67          # drop the worst third by residual, then refit


def fit_affine(z_true, z_model):
    """(fit dict, provenance) for z_model = a*z_true + b, or (None, why not).

    Trimmed least squares: outliers here are not noise, they are whole objects
    the model got wrong, so a plain fit would be dragged by them.
    """
    z_true = np.asarray(z_true, float).ravel()
    z_model = np.asarray(z_model, float).ravel()
    ok = np.isfinite(z_true) & np.isfinite(z_model) & (z_true > 0) & (z_model > 0)
    z_true, z_model = z_true[ok], z_model[ok]
    n = int(len(z_true))
    if n < MIN_POINTS:
        return None, f"refused: {n} usable points, need {MIN_POINTS}"

    lo, hi = np.percentile(z_true, [5, 95])
    span = float(hi - lo)
    med = float(np.median(z_true))
    frac = span / med if med > 0 else 0.0
    if span < MIN_SPAN_M or frac < MIN_SPAN_FRAC:
        return None, (f"refused: reference spans only {span:.2f} m "
                      f"({frac:.0%} of its {med:.2f} m median); a and b are not "
                      f"separable below {MIN_SPAN_M:.1f} m and {MIN_SPAN_FRAC:.0%}. "
                      f"Drive somewhere with near AND far surfaces in view.")

    x, y = z_true, z_model
    a = b = float("nan")
    for _ in range(TRIM_ITERS):
        A = np.column_stack([x, np.ones_like(x)])
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        resid = np.abs(y - (a * x + b))
        keep = resid <= np.percentile(resid, 100 * TRIM_KEEP)
        if keep.sum() < max(MIN_POINTS // 10, 10):
            break
        x, y = x[keep], y[keep]

    if not np.isfinite(a) or a <= 0:
        return None, f"refused: degenerate fit (a = {a})"

    # Error in METRES after correction, which is the number that matters.
    corrected = (z_model - b) / a
    resid_m = float(np.median(np.abs(corrected - z_true)))
    return (dict(a=round(a, 6), b=round(b, 6), n=n, n_inliers=int(len(x)),
                 span_m=round(span, 3), median_abs_resid_m=round(resid_m, 4)),
            f"gazebo depth truth, {n} points over {span:.2f} m")


def sample_pair(model_img, true_img, per_frame=400, rng=None, max_range=10.0):
    """Matched (z_true, z_model) samples from one frame pair.

    The two images are different sizes - the network works at its own resolution
    - so points are matched in NORMALISED coordinates rather than by index.
    """
    rng = rng or np.random.default_rng(0)
    mh, mw = model_img.shape[:2]
    th, tw = true_img.shape[:2]
    ys = rng.uniform(0.15, 0.95, per_frame)     # skip the top: mostly ceiling/sky
    xs = rng.uniform(0.05, 0.95, per_frame)
    m = model_img[(ys * mh).astype(int).clip(0, mh - 1),
                  (xs * mw).astype(int).clip(0, mw - 1)]
    t = true_img[(ys * th).astype(int).clip(0, th - 1),
                 (xs * tw).astype(int).clip(0, tw - 1)]
    ok = np.isfinite(m) & np.isfinite(t) & (t > 0.1) & (t < max_range) & (m > 0)
    return t[ok], m[ok]


def write_json(out_path, fit, provenance, extra=None):
    doc = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source": "gazebo_depth_truth",
        "provenance": provenance,
        "fit": fit,
    }
    if extra:
        doc.update(extra)
    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
def collect(args) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    def to_array(msg):
        if msg.encoding not in ("32FC1", "16UC1"):
            raise ValueError(f"unexpected depth encoding {msg.encoding}")
        dtype = np.float32 if msg.encoding == "32FC1" else np.uint16
        arr = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        return arr.astype(np.float32) / (1000.0 if msg.encoding == "16UC1" else 1.0)

    rclpy.init()
    node = Node("calibrate_sim_depth")
    state = {"model": None, "true": None}
    truth_pts, model_pts = [], []
    rng = np.random.default_rng(12345)

    def on_model(msg):
        state["model"] = to_array(msg)

    def on_true(msg):
        state["true"] = to_array(msg)
        if state["model"] is None:
            return
        t, m = sample_pair(state["model"], state["true"],
                           per_frame=args.per_frame, rng=rng,
                           max_range=args.max_range)
        truth_pts.append(t)
        model_pts.append(m)

    # Both sides are sensor topics; mono/depth is published RELIABLE by
    # mono_depth_node, and a BEST_EFFORT subscriber accepts a RELIABLE publisher,
    # so sensor QoS is safe for both.
    node.create_subscription(Image, args.model_topic, on_model, qos_profile_sensor_data)
    node.create_subscription(Image, args.truth_topic, on_true, qos_profile_sensor_data)

    print(f"listening: {args.model_topic} (model) vs {args.truth_topic} (truth)")
    print(f"collecting {args.frames} frame pairs; drive the robot around for spread")
    deadline = node.get_clock().now().nanoseconds + args.timeout_s * 1e9
    while rclpy.ok() and len(truth_pts) < args.frames:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.get_clock().now().nanoseconds > deadline:
            print(f"timed out after {args.timeout_s:.0f}s with "
                  f"{len(truth_pts)} frame pair(s)")
            break
    node.destroy_node()
    rclpy.shutdown()

    if not truth_pts:
        print("NO frame pairs. Is the mono stack running, and does "
              f"{args.truth_topic} exist? (the depth camera is bridged by "
              "sim.launch.py / sim_mono.launch.py)")
        return 1

    z_true = np.concatenate(truth_pts)
    z_model = np.concatenate(model_pts)
    fit, prov = fit_affine(z_true, z_model)
    print(f"\n{len(truth_pts)} frame pairs, {len(z_true)} points")
    if fit is None:
        print(prov)
        print("\nNothing written. A refusal is the honest outcome here: with the "
              "reference bunched at one distance, any (a, b) fits and the number "
              "would be meaningless.")
        return 1

    out = write_json(args.out, fit, prov,
                     extra={"frames": len(truth_pts),
                            "model_topic": args.model_topic,
                            "truth_topic": args.truth_topic})
    print(f"a = {fit['a']:.4f}   b = {fit['b']:+.4f}")
    print(f"median error after correction: {fit['median_abs_resid_m']:.3f} m "
          f"over {fit['span_m']:.2f} m of range")
    print(f"\nwrote {out}\n")
    print("Use it:")
    print(f"  ros2 launch ubot_mono_nav sim_mono.launch.py depth_scale_json:={out}")
    print("or set depth_scale_json in ubot_mono_nav/config/mono_perception.yaml")
    return 0


def selftest() -> int:
    print("SELFTEST  calibrate_sim_depth (fit + gates, no ROS)")
    ok = True

    def check(name, got, detail=""):
        nonlocal ok
        ok &= bool(got)
        print(f"  {'OK ' if got else '** '} {name:<56} {detail}")

    rng = np.random.default_rng(7)
    truth = rng.uniform(0.8, 8.0, 4000)

    # A known gain and offset must come back.
    fit, prov = fit_affine(truth, 2.13 * truth - 0.42)
    check("recovers a known gain and offset",
          fit and abs(fit["a"] - 2.13) < 1e-3 and abs(fit["b"] + 0.42) < 1e-3,
          f"a={fit['a']:.4f} b={fit['b']:+.4f}" if fit else prov)

    # With noise, and with 20% of points being whole wrong objects.
    noisy = 2.13 * truth - 0.42 + rng.normal(0, 0.05, truth.size)
    spoiled = noisy.copy()
    spoiled[:800] += rng.uniform(3.0, 6.0, 800)
    fit, _ = fit_affine(truth, spoiled)
    check("trimming survives 20% gross outliers",
          fit and abs(fit["a"] - 2.13) < 0.05,
          f"a={fit['a']:.4f}" if fit else "refused")

    # THE GATE THAT MATTERS: a stationary robot sees one distance.
    bunched = rng.uniform(2.40, 2.60, 4000)
    fit, prov = fit_affine(bunched, 2.13 * bunched - 0.42)
    check("a bunched reference is REFUSED, not fitted", fit is None, prov[:64])

    fit, prov = fit_affine(truth[:50], 2.13 * truth[:50])
    check("too few points is refused", fit is None, prov[:48])

    fit, prov = fit_affine(truth, -truth)
    check("a negative gain is refused, not written", fit is None, prov[:48])

    # The reported error is in metres after correction, not raw residual.
    fit, _ = fit_affine(truth, 2.0 * truth + 0.0)
    check("zero-error fit reports ~0 m residual",
          fit and fit["median_abs_resid_m"] < 1e-6,
          f"{fit['median_abs_resid_m']:.2e} m" if fit else "refused")

    # Sampling across two different image sizes must line up.
    true_img = np.tile(np.linspace(1.0, 5.0, 64, dtype=np.float32), (48, 1))
    model_img = 2.0 * np.tile(np.linspace(1.0, 5.0, 100, dtype=np.float32), (80, 1))
    t, m = sample_pair(model_img, true_img, per_frame=500,
                       rng=np.random.default_rng(3))
    check("samples match across differing resolutions",
          len(t) > 400 and np.median(np.abs(m / t - 2.0)) < 0.1,
          f"n={len(t)}, median ratio {np.median(m / t):.3f}")

    # The written file must be what load_affine reads.
    out = write_json("/tmp/ubot_depth_scale_selftest.json",
                     {"a": 2.13, "b": -0.42}, "selftest")
    doc = json.loads(Path(out).read_text())
    check("written file has the keys load_affine needs",
          doc.get("fit", {}).get("a") == 2.13 and "provenance" in doc)

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model-topic", default="/mono/depth")
    ap.add_argument("--truth-topic", default="/camera/depth/image_raw")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--per-frame", type=int, default=400)
    ap.add_argument("--max-range", type=float, default=10.0)
    ap.add_argument("--timeout-s", type=float, default=180.0)
    ap.add_argument("--out", default="depth_scale_sim.json")
    args = ap.parse_args()
    sys.exit(selftest() if args.selftest else collect(args))


if __name__ == "__main__":
    main()
