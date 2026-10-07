#!/usr/bin/env bash
# 51 lidar_ref episodes on the three MEASURED apartment courses.
#
#   ./lidar50.sh                 # start, or resume where it stopped
#   tail -f /tmp/lidar50.log     # watch
#
# This is a RELIABILITY run of one arm, not a comparison: there is no second arm
# to pair against, so there are no deltas to report. It answers "how often does
# the reference stack complete these courses, and how efficiently".
#
# --keep-going is deliberate. The default stops the line when lidar_ref fails,
# which is right for a smoke test guarding a 19-hour matrix; here a failure is
# part of the measurement, so it is recorded and the run continues.
#
# --resume skips episodes whose episode.json exists AND validates, so it is safe
# to re-run after an interruption.
#
# Only apartment: the parking_garage courses still have shortest_path_m null, so
# a success there cannot be scored and would be thrown away.
set -uo pipefail
cd "$(dirname "$0")"
source ./lib.sh
ros_env

TRIALS="${TRIALS:-17}"
CAMPAIGN="${CAMPAIGN:-lidar50}"
RUNS="${RUNS:-$WS/runs}"

kill_stack || echo "continuing anyway: campaign.py owns each stack as a process group"

echo "starting $CAMPAIGN: 3 courses x $TRIALS trials = $((3 * TRIALS)) episodes"
echo "runs-root $RUNS"
date

# `setsid` matters. campaign.py gives every stack its own process group with
# os.setsid() so teardown can signal the group instead of pkill-ing by name. If
# THIS process is already a process-group leader - which it is when the script
# is started with `nohup ... &` - that call fails with EPERM and the campaign
# dies on episode 1 with "Exception occurred in preexec_fn". Running the
# campaign in a new session makes it a plain member of one, so its children can
# lead their own groups.
setsid python3 -m ubot_eval.campaign --run --resume --keep-going \
    --arms lidar_ref --worlds apartment \
    --trials "$TRIALS" --campaign "$CAMPAIGN" --runs-root "$RUNS"
rc=$?

date
echo "=== campaign exit: $rc ==="
echo
python3 -m ubot_eval.report --runs-root "$RUNS" --compare lidar_ref:lidar_ref
exit "$rc"
