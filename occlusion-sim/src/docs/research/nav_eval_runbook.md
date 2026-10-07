# Navigation trials — runbook

Everything here runs without Claude. Commands are copy-paste.

Two machines:

| | where | what it can do |
|---|---|---|
| **WSL** | `~/occl_ws` (symlink to `Occlusion-sim/occlusion-sim/src/ubot`) | `lidar_ref` only — no GPU, no torch, so no camera arm |
| **riftvm** | `ssh carla-gpu` → `~/occl_ws`, container `ubot_trials:jazzy` | everything, 2× V100 |

> **Use the ssh alias `carla-gpu`, not `toyibat@66.172.10.85`.** Only the alias
> carries `IdentityFile ~/.ssh/id_carla`; the bare `user@host` matches no `Host`
> block, silently falls back to password auth and hangs with no output.

---

## 1. One episode, locally (5 min)

```bash
cd ~/occl_ws/src/ubot/ubot_eval/scripts
./run_episode.sh
```

Expect `outcome: reached`, `path_length_m` ≈ 3.0, `spl` ≈ 0.99.
Override with env vars: `ARM=lidar_ref COURSE=hall_3m TRIAL=2 ./run_episode.sh`.

If it prints **`refusing to launch onto a dirty process table`**, a previous run
left nodes alive. Run `source lib.sh && kill_stack` and look at what it names.

---

## 2. The SPL denominator (do this before reporting any SPL)

SPL divides the optimal path by the actual one. Right now `hall_3m` uses a
**provisional straight-line 3.0 m** and every other course is `null`. Any SPL
built on the provisional value is **not reportable**.

```bash
cd ~/occl_ws/src/ubot/ubot_eval/scripts
./mapping_pass.sh                               # ~15-25 min, drives + saves the map
./measure_courses.sh ~/occl_ws/maps/apartment.yaml
```

Paste the numbers into `ubot_eval/config/courses.yaml` and **delete the
`shortest_path_provisional: true` flag** from each course you replace.

`no path` for a course means the mapping pass did not cover it (unmapped space
counts as blocked, deliberately). Drive more waypoints — do not relax the
assumption.

Check the maths itself any time: `python3 -m ubot_eval.shortest_path --selftest`.

---

## 3. The campaign on riftvm

```bash
ssh carla-gpu
cd occl_ws/src/ubot/ubot_eval/docker

./run.sh check      # GPU, torch, depth timing, AND a rendered frame. Do this first.
./run.sh smoke      # 2 arms x 1 course x 2 trials
./run.sh campaign   # the matrix; --resume is on, so it is safe to re-run
```

**`./run.sh check` must pass before any campaign.** It is the only thing that
catches a silent rendering failure — Gazebo without a GL runtime segfaults and
every sensor topic exists but stays empty, which downstream looks like a bad
depth model rather than a dead renderer.

To sync local edits up first:

```bash
scp -r ~/occl_ws/src/ubot/. carla-gpu:occl_ws/src/ubot/
```

Results land in the bind-mounted `runs/`. Bring them back with:

```bash
rsync -av carla-gpu:occl_ws/runs/ ~/occl_ws/runs/
```

---

## 4. The results table

```bash
python3 -m ubot_eval.report --runs-root ~/occl_ws/runs
python3 -m ubot_eval.report --runs-root ~/occl_ws/runs --markdown   # for slides
```

Reads only `episode.json` files, so it is safe to run mid-campaign. It reports
`not_started` **separately** (a refused episode measured nothing about
navigation), never treats a missing SPL as zero, compares arms only on
`pair_key`s both completed, and prints a CAVEATS footer naming any provisional
SPL or thin sample.

---

## 5. Things that fail silently — check these first when something is odd

| Symptom | Cause |
|---|---|
| LiDAR publishes at full rate, every range `.inf` | spawn pose not passed with `world:=` — robot is ~300 m outside the map. Always pass `spawn_x/y/z/yaw`. |
| Camera topics exist, no frames ever | no GL runtime / no `headless_rendering:=true`. Gazebo segfaulted at startup. |
| Whole nav2 bringup aborts | a zombie `opennav_docking` from a previous run is still ACTIVE. `pkill -f nav2` does **not** match it. Use `kill_stack`. |
| Every success recorded as `false_success` | goal is in `odom`, ground truth is in the Gazebo world — they must be compared through `world_T_odom`. |
| `aborted` after ~0.8 s with a tiny path | BT action-server timeout too short for a CPU-starved host. |
| ssh command produces no output at all | wrong ssh target — use `carla-gpu`. |

---

## 6. Self-tests (all offline, no ROS or Gazebo needed)

```bash
cd ~/occl_ws/src/ubot/ubot_eval
python3 ubot_eval/schema.py --selftest
python3 ubot_eval/readiness.py --selftest
python3 ubot_eval/truth_source.py --selftest
python3 ubot_eval/contact.py --selftest
python3 ubot_eval/episode_runner.py --selftest
python3 ubot_eval/campaign.py --selftest
python3 ubot_eval/shortest_path.py --selftest
python3 ubot_eval/report.py --selftest
```

`campaign.py --selftest` includes the **arm-parity check**: it fails if the two
arms' nav2 params disagree on anything that would advantage one of them (BT
timeouts, goal tolerances, planning frame, costmap size), while allowing the
observation source and range caps to differ — those are the thing being
measured.

Against the live config:

```bash
python3 -c "from ubot_eval.campaign import check_params_parity as c; print(c() or 'PARITY OK')"
```
