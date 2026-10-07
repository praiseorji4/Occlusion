#!/usr/bin/env bash
# One episode, end to end, from a verified-clean process table.
#
#   ./run_episode.sh                                   # lidar_ref, hall_3m
#   ARM=lidar_ref COURSE=hall_3m TRIAL=2 ./run_episode.sh
#
# Prints the episode.json THIS run wrote - never an older one. That matters:
# `find -newermt '-8 minutes'` once showed me a previous run's success after the
# current run had crashed before writing anything.
set -uo pipefail
cd "$(dirname "$0")"
source ./lib.sh

ARM="${ARM:-lidar_ref}"
COURSE="${COURSE:-hall_3m}"
TRIAL="${TRIAL:-1}"
SCAN="${SCAN:-/scan}"            # /scan_mono for the camera arms
SHORTEST="${SHORTEST:-3.0}"      # PROVISIONAL for hall_3m; see courses.yaml
RUNS="${RUNS:-$WS/runs}"
HR="${UBOT_HEADLESS_RENDERING:-}"

ros_env
kill_stack || { echo "refusing to launch onto a dirty process table"; exit 1; }

EXTRA=()
[ -n "$HR" ] && EXTRA+=(headless_rendering:=true)

nohup ros2 launch ubot_bringup sim.launch.py \
    world:=apartment world_name:=apartment "${APARTMENT_SPAWN[@]}" \
    use_slam:=false use_rviz:=false headless:=true "${EXTRA[@]}" \
    > /tmp/episode_sim.log 2>&1 &
echo "sim pid $!"

wait_bringup /tmp/episode_sim.log || exit 1

STAMP=$(date +%s)
timeout 900 python3 -m ubot_eval.episode_runner \
    --arm "$ARM" --course "$COURSE" --trial "$TRIAL" \
    --scan-topic "$SCAN" --shortest-path-m "$SHORTEST" \
    --runs-root "$RUNS" 2>&1 | tail -6
rc=${PIPESTATUS[0]}
echo "=== episode exit: $rc ==="

newest=$(find "$RUNS" -name 'episode.json' -newermt "@$STAMP" 2>/dev/null | head -1)
if [ -z "$newest" ]; then
    echo "NO episode.json was written by this run"
    exit 1
fi
python3 - "$newest" <<"PY"
import json, sys
d = json.load(open(sys.argv[1]))
for k in ("outcome", "outcome_reason", "reached_goal", "path_length_m",
          "shortest_path_m", "spl", "gt_distance_to_goal_m", "time_to_goal_s",
          "recoveries", "warnings"):
    print(f"  {k}: {d.get(k)}")
print("  collisions:", (d.get("collisions") or {}).get("count"))
PY
