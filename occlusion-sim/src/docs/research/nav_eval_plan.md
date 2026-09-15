# Camera-only navigation evaluation — plan and handoff

Self-contained brief for continuing this work in a fresh session. Written 16 Sep 2026.
Everything an incoming agent needs is here; no prior conversation context is required.

---

## 0. Read this first

### Two separate repositories

| Tree | Git | Contains |
|---|---|---|
| `C:\Users\pator\Documents\Chibueze\Occlusion-sim` | remote `praiseorji4/Occlusion`, branch **`mono-nav`** | the ROS workspace at `occlusion-sim/src/ubot` — this is the **only** place the robot is worked on |
| `C:\Users\pator\Documents\Chibueze\Occlusion` | **local only** | `research/` (papers, analysis code, the experiment PDF) and the `occlusion/` Python package |

They are *not* the same repository despite the similar names. The plan calls for code in both.

### ⚠ The research tree is unversioned

`Occlusion` has **zero commits, no remote, and 2055 staged files**. The twelve papers, the paper
index, the experimental-setup PDF and all the analysis code exist on one disk and nowhere else.
Ask the user before committing (it may be deliberate — there may be large or private files in
that 2055), but do not build more work on top of it without raising this.

### Working environment

- Windows host; ROS runs in **WSL2 Ubuntu 24.04**, ROS 2 Jazzy, Gazebo Harmonic (gz-sim 8.15).
- WSL workspace is `~/occl_ws`, which **builds from the Windows tree** — edit the files under
  `Occlusion-sim/...`, then `colcon build` in `~/occl_ws`.
- `$(find ubot_description)` resolves to the **install space**. Editing a xacro and re-running
  `xacro` without rebuilding silently uses the stale installed copy. This cost an hour; rebuild
  before every verification.
- In **non-interactive** WSL shells (i.e. anything an agent runs via `wsl -e bash -lc`),
  `ros2 topic list` returns zero topics and subscribers receive nothing — including on `/scan`.
  `ros2 topic pub` and `gz topic -e` work fine. Use `--no-daemon` for listing, prefer the `gz`
  side for verification, and hand genuinely ROS-side checks to the user's interactive terminal.
- `pkill -f "sim.launch.py"` from inside `bash -lc` **kills the shell itself**, because the
  pattern matches its own command line. Always put such scripts in a file and run the file.

---

## 1. What the system is

`ubot_mono_nav` navigates a four-wheel skid-steer inspection robot from a single RGB camera:
monocular metric depth (Depth-Anything-V2-Metric-Indoor-Small) → a synthetic `LaserScan` on
`/scan_mono` (66 bins, 65° forward, height band 0.05–0.60 m, trust band 1–4 m) → unmodified nav2.
There is **no SLAM**: everything is in the `odom` frame. The robot also carries an MS200 2D LiDAR
at z = 0.296 m and can run a conventional LiDAR stack, which makes a paired control group
possible.

Key geometry and calibration (all measured, do not "correct" them):

- wheel radius 0.032340 m; geometric track 0.226 m
- **effective** wheel separation: **0.55 m on the real robot**, **0.2504 m in simulation**. Not a
  bug — a skid-steer scrubs its tyres to rotate, and the real limiter is motor torque while the
  simulated one is friction. The sim physically cannot reproduce 0.55. Real config lives in
  `ubot_bringup/config/ubot_controllers.yaml`, sim in `sim_ubot_controllers.yaml`.
- angular cap 0.3 rad/s (measurements must be taken *below* the cap or the cap silently clamps
  the command and corrupts the ratio)
- the firmware's `diff_controller.h` value is telemetry-only and does not affect calibration

---

## 2. Why this work exists

`research/papers/rgb_only_navigation_benchmarks.pdf` surveys the comparable field and ends with
this project's own measurement gaps: results rest on **3 trials** where comparable works run
10–100; **SPL is not measured**; **collision rate is not measured**; and the **paired LiDAR
baseline has never been run**. The review argues the defensible contributions are (a) a trust
band derived from mount geometry rather than tuned, and (b) a same-course LiDAR control group,
which only arXiv 2604.03096 among the surveyed works performs.

The instrumentation gap is total: `ubot_mono_nav` ships four executables, none of which record
anything, and there is no `NavigateToPose` client anywhere in the workspace.
`src/docs/research/gaps.md:97` (G9) states this; `docs/research/directions.md:108` (D4) and
`docs/experiments/logging.md` describe what should be measured. Neither is implemented.

**Target: a full paper.** Primary evidence in simulation, real corridor runs as sim-to-real
validation. Depth model runs on the GPU server.

---

## 3. Status

| Part | State |
|---|---|
| 1 — literature | ✅ done, 12 papers fetched and index written |
| 0 — collision detection | ✅ implemented and committed; **one check still open** |
| 2 — navigation improvements A and B | ⬜ not started |
| 3 — the experiment harness | ⬜ not started |

### Part 1 ✅

Twelve PDFs in `Occlusion/research/papers/`, each opened and its title checked against its arXiv
id (filenames not taken on trust). `papers/README.md` indexes them by tier with the specific
numbers each of our metrics will sit against. Highlights:

- **2604.03096** — the closest comparison in the field: Depth Anything V2 benchmarked directly
  against LiDAR *on the same courses*. Sim 20 m: 97% SR mono vs 67% LiDAR; real: both 100% SR,
  mono 22% lower SPL; hard sim 30 m: mono SR falls to 10%. Its failure mode is **persistent
  phantom obstacles**, which is exactly what improvement A risks causing.
- **MonoNav (2311.14100)** — established the safety-vs-completion reporting frame: collision rate
  4× lower at 22% lower goal completion.
- **1904.01201 / 1911.00357 (Habitat, DD-PPO)** — the canonical cost of dropping depth.
  Gibson SPL: RGB 0.46, RGB-D 0.70, depth 0.79.
- **2603.25937** — uses LiDAR for ground-truth pose only, never as navigation input. This is the
  protocol to copy for real-robot runs.

**Two are paywalled and were not fetched**: IEEE 9602346, and Springer `s40430-025-06053-3`,
whose MAE 0.056 m / RMSE 0.082 m is the figure our depth metrics get compared against. The review
marks it abstract-sourced; keep that caveat until the full text is read.

### Part 0 ✅ — one check open

Was a blocker: no world loaded `gz-sim-contact-system` and the robot declared no contact sensor,
so **the simulator could not detect a collision at all** and the central safety metric did not
exist. Implemented in commit `4f2d549` on `mono-nav`: Contact system in all five worlds, a
contact sensor on `base_link`, bridged and remapped to `/bumper/contacts`.

**Chassis only, not the wheels.** The wheels are the only part touching the floor in normal
driving, so every reported contact is already a real collision and there is no ground-contact
filter to get subtly wrong. The chassis has 18 mm clearance (underside box stops at z = −0.048,
floor sits at −0.066 under `base_link`).

Two silent traps found while verifying, both now documented in the xacro:

1. **Fixed-joint lumping renames every collision.** `base_link` is joined to `base_footprint` by
   a fixed joint and urdf → sdf lumps them, rewriting collisions to
   `base_footprint_fixed_joint_lump__base_link_collision{,_1,_2}`. A sensor referencing the
   declared names loads cleanly, matches nothing, and reports zero collisions forever — which
   reads exactly like a robot that never crashes. Re-check after any collision change:
   `gz sdf -p robot.urdf | grep "collision name="`
2. **The Contact system ignores the sensor's `<topic>`** and publishes on the scoped
   `world/<world>/model/ubot/link/base_footprint/sensor/base_contact_sensor/contact`. The bridge
   needs the world name substituted in; a short name yields a ROS topic that exists, is
   subscribed, and never receives a message.

**Verified on the Gazebo side:** 0 contacts parked, 0 across 10.5 m of open driving, and firing
on wall impact with collision names confirming chassis hull against the apartment wall.

**Open:** delivery through the bridge into ROS. `/bumper/contacts` exists with one RELIABLE
publisher and the bridge is a connected gz-side subscriber, but no subscriber receives anything
in a non-interactive WSL shell — the `/scan` control fails identically, so that harness cannot
distinguish a broken bridge from a broken shell. The user was asked to run, interactively:

```bash
ros2 launch ubot_bringup sim.launch.py world:=apartment world_name:=apartment \
    spawn_x:=5.0 spawn_y:=10.62 spawn_z:=0.1 spawn_yaw:=1.5708
```
```bash
ros2 topic hz /bumper/contacts
```

That spawn places the robot against the wall so contacts fire immediately — it is also the fast
fixture for `contact.py --selftest` (driving to the wall takes ~100 s). **~50 Hz → Part 0 is
fully verified.** **"no new messages" → the bridge does not deliver `Contacts`; write a small
relay node rather than fighting `ros_gz_bridge`.** Ask the user for this result before assuming
either way. Debouncing the ~50 Hz raw stream into one event per impact belongs in `contact.py`;
one wall hit produced 3033 raw messages.

---

## 4. Part 2 — navigation improvements

Both target documented failure modes, both toggle from parameters, neither touches nav2 itself.
Defaults must preserve current behaviour so the baseline arm is bit-identical to today's stack.

### A. Blind-spot safety
65° forward, nothing to the sides or rear. `observation_persistence: 0.0` in both the local voxel
layer and the global obstacle layer means an obstacle is forgotten the moment it leaves view, and
`Spin`/`BackUp` are absent because both move the robot blind.

- **A1 — obstacle memory.** `observation_persistence` 0.0 → 3.0 s local, 8.0 s global; retention
  is the ablation parameter. The stale-vs-forgotten trade-off is exactly 2604.03096's phantom
  obstacle failure, so it must be measured, not assumed.
- **A2 — restore `Spin`, guarded.** This is the point of A: with persistence, obstacles seen
  before the spin are still in the costmap during it, so `Spin`'s own `simulate_ahead_time` check
  finally has something to check against. New tree `navigate_to_pose_mono_A.xml`, with
  `max_rotational_vel` capped at 0.25 to match DWB. **`BackUp` stays removed unconditionally** —
  persistence cannot conjure a rear sensor.
- **A3 — turn-rate gating** (only if A1+A2 prove insufficient). Scale angular velocity by how
  recently the swept sector was observed, in `scan_watchdog_node.py`, already the last hop before
  `/cmd_vel`.

Split `A_persist` from `A_persist_spin` if budget allows, so the effect is attributable.

### B. Phantom obstacles and depth trust
`scan_geometry.depth_to_scan` accepts any bin with ≥ `min_points` returns, with no temporal
filtering and no notion of frame trustworthiness. Textureless walls and glare produce confident,
wrong, near obstacles.

| parameter | baseline (= today) | B-on | meaning |
|---|---|---|---|
| `min_points` | 2 | 8 | agreeing returns per angular bin; at 2, one near stray plus one far real return passes and reports the stray |
| `temporal_window` / `temporal_min_hits` | 1 / 1 | 3 / 2 | a bin marks only if ≥2 of the last 3 frames agree within `temporal_tol_m` 0.25 |
| `validity_gate` | `none` | `texture` | reject pixels whose local RGB gradient is below `texture_min_grad` |
| `glare_max_luma` | 255 | 245 | saturated pixels give confident garbage depth |
| `range_confidence_max_m` | 4.0 | 3.5 | tighter trust band |

Put the logic in `scan_geometry.py` (pure numpy) so it is covered by that module's existing
`--selftest` and testable with no robot. `depth_to_scan_node` gains an optional synchronised RGB
subscription, made only when the gate is enabled. Re-run `scan_geometry --selftest` afterwards:
it already encodes the floor-is-not-an-obstacle and stray-pixel cases that B could break.

---

## 5. Part 3 — the experiment

### 5.1 Where the code goes

Hard rule: **`ubot_eval` never imports the research code, and the research code never imports
ROS.** The wire format between them is the run directory on disk, versioned by `schema_version`.

| Path | Contents |
|---|---|
| `Occlusion-sim/occlusion-sim/src/ubot/ubot_eval/` | ROS: `episode_runner.py`, `campaign.py`, `readiness.py`, `truth_source.py`, `contact.py`, `schema.py`, `scan_pair_logger.py`, `depth_pair_logger.py`, plus `config/courses.yaml`, `config/arms.yaml`, nav2 overlays, the A-arm behaviour tree |
| `Occlusion/research/nav_eval/` | Analysis: `load.py`, `metrics_nav.py`, `shortest_path.py`, `scan_compare.py`, `depth_metrics.py`, `episode_bootstrap.py`, `aggregate.py`, `make_nav_report.py` |

`nav_eval` sits beside `distance_analysis/` and is imported the same way; add `NAV_RUNS` /
`NAV_EVAL` to `paths.py`. No new dependencies — numpy, scipy, pandas, matplotlib and pyyaml are
already in use, and A* is ~40 lines of `heapq`.

### 5.2 Episode runner

One episode = one goal attempt via the `NavigateToPose` action, goal in `odom` (no map exists).
Log the action feedback stream (it supplies `number_of_recoveries` and `distance_remaining` for
free), `/plan`, `/cmd_vel`, odom, both scans and contacts, as a bag following
`docs/experiments/logging.md` naming, plus a JSON summary.

**Readiness is a hard gate, not a hope.** Controller states via
`/controller_manager/list_controllers` (the service, not log scraping), TF advancing, ≥3 scans
with a finite range, nav2 lifecycle nodes active — then a **motion proof**: a 0.6 s, 0.05 m/s
nudge that must produce >0.01 m of ground-truth displacement. Every other check passes while the
wheels are locked, which is a failure this platform has actually exhibited (a lateral-friction
setting stalled the wheels while odometry kept integrating motion that never happened). A failed
gate after retries writes `outcome: not_started` — **never a timeout or a failure**, because
scoring a robot that could not move as a navigation failure is silent corruption.

**Outcome classes**, first match wins:

`not_started` → `collided` (safety dominates; log `reached_goal` separately so the trade-off
stays visible) → `reached` → **`false_success`** (nav2 SUCCEEDED but ground truth says >0.15 m
from goal — nav2 believed odom and odom was wrong; a real and expected failure mode in a no-SLAM
stack, counted as a failure) → `aborted` → `timeout` → `stalled` → `degraded` (perception died;
excluded from SR by default, counted and reported).

**Ground truth.** Sim: bridge `/world/<world>/dynamic_pose/info` to `/gt/tf` — `SceneBroadcaster`
is already loaded, so no world edit is needed. This replaces polling `gz model -m ubot -p`, which
blocks ~1 s per call and cannot sample a trajectory. The world name is baked into the topic, so
cross-check one sample per episode against the CLI and refuse if they disagree by >2 cm.
Real: no ground truth exists — log EKF odom plus an operator-entered tape-measured final pose,
and carry `odom_drift_m` beside every real path metric.

### 5.3 Metrics

SR, SPL, collision rate (per-episode **and** pooled per metre), distance ratio (median over
successes only; failures reported separately as `wander_ratio`), goal completion, plus
time-to-goal, minimum clearance and recovery count — exactly D4's list, so one harness serves
both.

Two traps: decimate the 60 Hz trajectory to 10 Hz and drop sub-5 mm steps before summing path
length, or pose jitter while stationary accumulates phantom metres; and **warn rather than
silently clamp** when `p_i < l_i` by more than 2%, because that means the reference path is
wrong, not that the robot beat geometry.

**Shortest path `l_i`:** build an occupancy grid per world by running the existing `slam_toolbox`
stack once in a manual mapping pass — ground-truth LiDAR, independent of both arms, reusing
`mapper_params_online_async.yaml` — then 8-connected A* inflated by the 0.219 m footprint
half-diagonal, **followed by a string-pulling pass**. The smoothing is not cosmetic: 8-connected
A* overestimates diagonals by ~8%, which would inflate every SPL. Real: tape-measured per course
once, since `l_i` is a property of the course, not the trial.

### 5.4 Scan comparison

Rebin by **min-pooling** the 720-beam LiDAR onto the 66 mono bins — the only reduction that
preserves "nearest obstacle in this bin", which is what the mono scan means. Treat `+inf` as a
**category, not a number**: both finite → range error; LiDAR finite, mono `+inf` → **miss**; mono
finite, LiDAR beyond band → **phantom** (improvement B's target metric). Report `near_miss_rate`
and `phantom_rate_near` (<1 m) separately — a miss at 3.8 m is nearly harmless, one at 0.8 m is
the collision.

**The confound that must not be ignored:** the LiDAR sits at z = 0.296 m while the mono height
band is 0.05–0.60 m, so anything shorter than ~0.30 m is invisible to the reference and scores as
a phantom when the camera is *right*. In sim the honest fix exists — the rgbd camera's
**ground-truth depth image** run through `depth_to_scan` with identical parameters, isolating
depth error from the scan builder. **GT depth is the primary sim reference, LiDAR the
secondary.** On the real robot the LiDAR is all there is, so state the confound.

### 5.5 Depth metrics

MAE, RMSE, AbsRel, δ₁ banded 1–2 / 2–3 / 3–4 m, with `<1 m` and `>4 m` rows flagged `untrusted`
rather than folded into the total. Reuse `occlusion.core.depth_scale.resolve_affine` /
`apply_affine`, and **refuse to emit metrics when the affine is unresolved** — unscaled output is
off by ~2× at this viewpoint, so publishing MAE on it would be nonsense. Run `band_table` first
as a gate on the house 15% flatness threshold. GT is the sim rgbd depth; on the real robot it is
OAK-D stereo, labelled `stereo_ref`, **not** `gt`. Log depth pairs in the same run-dir layout as
`occlusion/capture/oakd.py` so existing tools read them unchanged.

### 5.6 Statistics

Reuse `research/distance_analysis/bootstrap.py::paired_bca_delta` directly — it takes plain
aligned arrays and is already unit-agnostic; only its callers hardcode KITTI columns. Resample on
`pair_key = world/course/trial`, same fixed seed across arms to preserve the paired draws.
**Never resample per frame or per detection**: frames within an episode are heavily
autocorrelated and would shrink CIs by roughly √(frames per episode), manufacturing significance.

Three metrics are not means: **SR** is a paired proportion where BCa degenerates when an arm is
all-success or all-failure — catch that and report an exact **McNemar** rather than fabricating
an interval; **collisions per metre** is a ratio of sums needing numerator and denominator
resampled together; **median distance ratio** needs a `statistic=` parameter threaded through
`paired_bca_delta` as a backwards-compatible extension, not a copy.

Run `--power` *before* collecting: at 75 pairs the detectable SR difference at 80% power is
roughly 15–18 points. If the target effect is smaller, the answer is more trials, not more
analysis.

### 5.7 Trial matrix

2×2 factorial plus references: `mono_baseline`, `mono_A`, `mono_B`, `mono_AB`, `lidar_ref`, and a
**negative control `mono_baseline_dup`** — the baseline under a second name, whose delta against
the baseline must have a CI spanning zero. If it doesn't, the harness has an arm-correlated
artefact and every other number is suspect. Cheapest insurance in the design; put the result in
the paper.

Worlds: `apartment` (corridor) and `parking_garage_occlusion` (pillars, parked vehicles, a
pedestrian crossing at x = −11.0 — occlusion-rich, and the reason that world exists).
25 trials per arm per world, `--shuffle-arms` within each `pair_key` so ~19 h of sim drift cannot
correlate with arm. `campaign.py --resume` skips completed episodes.

Real validation: `lidar_ref`, `mono_baseline`, `mono_AB` × 3 goals × 3 reps.

Cap the sim at the real robot's measured 0.25 rad/s (already configured) so trajectories
transfer, and state in the write-up that the sim cannot reproduce the real scrub, which bounds
sim-to-real claims about turning.

### 5.8 What cannot be measured in each venue

| Claim | Reliable in | What to do in the other |
|---|---|---|
| SPL, distance ratio, path length | **sim** (ground-truth pose) | real: report with `odom_drift_m` beside it, never in a headline table |
| collision rate | **sim** (contact sensor) | real: operator video annotation, reported separately, never pooled |
| depth MAE/RMSE/AbsRel/δ₁ vs true GT | **sim** (rgbd GT depth) | real: OAK-D stereo as `stereo_ref`, 1–4 m only |
| scan fidelity isolated from the builder | **sim** (GT depth through the same code) | real: LiDAR, carrying the z = 0.296 m confound |
| does it work on the real robot at all | **real** | sim SR is not evidence about the real corridor |

---

## 6. Verification

Every module gets a `--selftest` (house convention; there is no pytest suite), and
`campaign.py --selftest-all` runs them in dependency order.

**Synthetic, no ROS:** straight 10 m run → SPL exactly 1.0; 20 m detour to a 10 m goal → 0.5;
2 of 4 episodes successful → SR 0.5 **and** SPL 0.5 (catches the classic bug of averaging SPL
over successes only); `p < l` → clamped **and warned**; nav2-SUCCEEDED-but-1.8 m-away →
`false_success`, the single test that justifies having ground truth at all; 60 s of ±5 mm jitter
while stationary → path length < 0.05 m.

**Geometry:** A* diagonal within 0.5% of √2·L after smoothing, and 8.2% before it (asserting the
smoother actually ran); a 0.30 m gap with a 0.219 m radius returns *no path* with provenance, not
zero. Scan comparison: `+inf` never enters the range error; an all-`inf` mono scan gives
`miss_rate = 1.0` and `range_mae = None` with a reason, not `0.0`. Depth: `pred = 1.2·gt` →
AbsRel 0.2, δ₁ = 1; `1.3·gt` → δ₁ = 0; band edges half-open.

**Stats:** identical arms → CI contains 0 in ≥93% of 200 replications; large effect → excludes 0;
all-success → `None` plus McNemar provenance; same input twice → bit-identical output.

**In the loop:** readiness must be made to *fail on purpose* (inject the spawner race by delaying
the robot description) and name the failing check — a gate never tested against failure is not a
gate. Truth source: a scripted 1 m straight line and 90° arc agreeing with the CLI to <2 cm.
Contact: the wall-hit and open-floor pair from Part 0.

**End-to-end smoke** before every campaign: 3 m goal, `mono_baseline` + `lidar_ref`, 2 trials,
~5 minutes. A `lidar_ref` failure on a 3 m clear hop means the *course* is broken, not the
camera — stop the line.

**Then a 10-episode pilot** of `lidar_ref` vs `mono_baseline` before committing to the full
matrix, to confirm episode duration, disk per bag, and whether the CIs are narrow enough at 25
trials to justify the ~19 hours.

---

## 7. Known issues not caused by this work

- **dartsim ignores the rear-wheel mimic joints.** The launch logs
  `Attempting to create a mimic constraint for joint [rear_left_wheel_joint] but the chosen
  physics engine does not support mimic constraints`. The rear wheels are dragged by the chassis
  rather than driven, which undercuts the "only front wheels commanded, rears follow" behaviour
  the sim is supposed to mirror. Does not affect contact sensing. **Unresolved — raise with the
  user before the campaign**, since it changes the very skid-steer dynamics being measured.
- `RTPS_TRANSPORT_SHM Error: Failed init_port fastrtps_port7002` appears on nearly every node,
  including in the user's interactive terminal where nav2 activates cleanly. Noisy rather than
  fatal there; fatal for subscribers in agent-run non-interactive shells.
- The Pi is offline, still tracks `Unipod-Robotics/uni-bot`, and has three uncommitted files;
  it needs re-pointing to `Occlusion` after those are diffed.
- The stale `uni-bot/mono-nav` branch has no decided fate.
- User-run items outstanding: udev rule on the Pi, `python3-pip` + torch/transformers in WSL,
  move the OAK-D to a USB 3 port (it is currently on USB 2 with both USB 3 ports empty).

---

## 8. Immediate next steps

1. **Get the `ros2 topic hz /bumper/contacts` result from the user** and close Part 0, or write
   the relay node if the bridge does not deliver.
2. Raise the **unversioned research tree** and the **dartsim mimic** issue.
3. Start `ubot_eval`: `schema.py`, then `readiness.py` (including the motion proof and its
   deliberate-failure test), then `truth_source.py`, then `episode_runner.py`.
4. Only after the end-to-end smoke passes, implement Part 2's arms.

Work measurement-first throughout: this project's house style is that every claim is verified
numerically — mesh bounding boxes, TF echoes, ground-truth vs odom comparisons, hash checks —
and a number that was not measured is not reported.
