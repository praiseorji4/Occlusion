#!/usr/bin/env bash
# The results PDF. Safe to run while the campaign is still going - it reports
# what has been written so far and prints the episode count on the page.
#
#   ./make_pdf.sh                                  # campaign lidar50
#   CAMPAIGN=pilot OUT=~/pilot.pdf ./make_pdf.sh
#
# The PDF lands in the Windows-visible workspace by default, so it can be opened
# without copying anything out of WSL.
set -uo pipefail
cd "$(dirname "$0")"
source ./lib.sh
ros_env

CAMPAIGN="${CAMPAIGN:-lidar50}"
RUNS="${RUNS:-$WS/runs}"
OUT="${OUT:-$WS/src/ubot/ubot_eval/${CAMPAIGN}_results.pdf}"
TITLE="${TITLE:-LiDAR reference stack — 51-episode reliability run}"

python3 -m ubot_eval.report_pdf \
    --runs-root "$RUNS" --campaign "$CAMPAIGN" \
    --out "$OUT" --title "$TITLE" || exit 1

echo
echo "text version:"
python3 -m ubot_eval.report --runs-root "$RUNS/$CAMPAIGN"
