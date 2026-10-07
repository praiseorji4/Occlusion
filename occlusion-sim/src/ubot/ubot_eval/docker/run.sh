#!/usr/bin/env bash
# Run the trials container on a GPU box.
#
#   ./run.sh check       GPU, torch, and A RENDERED FRAME - run this first
#   ./run.sh calibrate   fit the depth model -> depth_scale.json (camera arms)
#   ./run.sh smoke       the pre-campaign smoke selection
#   ./run.sh campaign    the full matrix, DETACHED - survives the ssh dropping
#   ./run.sh progress    how far along, and what the outcomes look like so far
#   ./run.sh results     the table and the PDF; safe to run mid-campaign
#   ./run.sh shell       interactive
#
# Typical session:
#   ./run.sh check && ./run.sh smoke        # both must look right first
#   ./run.sh campaign --trials 25           # hours; detached
#   ./run.sh progress                       # whenever
#   ./run.sh results                        # whenever; writes runs/results.pdf
#
# WS defaults to the workspace three levels up (src/ubot/ubot_eval/docker -> ws).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${WS:-$(cd "$HERE/../../../.." && pwd)}"
IMAGE="${IMAGE:-ubot_trials:jazzy}"
RUNS="${RUNS:-$WS/runs}"
CONTAINER="${CONTAINER:-ubot_campaign}"
mkdir -p "$RUNS"

# --shm-size: Fast DDS puts its segments in /dev/shm, and docker's 64 MB default
#   makes it fail to open ports - the "RTPS_TRANSPORT_SHM ... open_and_lock_file
#   failed" line, after which discovery quietly does not work.
# --runtime=nvidia, NOT --gpus all. On this host `--gpus all` does not give the
# container the driver's graphics stack: torch sees the GPUs and ogre2 still
# cannot render, which is the exact split that makes a rendering failure look
# like a depth-model problem. The legacy nvidia runtime honours
# NVIDIA_DRIVER_CAPABILITIES=...,graphics and injects libEGL_nvidia. Found by
# the user on riftvm; keep both this flag and the capability list together.
DOCKER_ARGS=(
    --rm --runtime=nvidia
    --shm-size=2g
    --network host
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
    # No display in here, so every stack this container launches must render
    # offscreen through EGL. campaign.py reads this and appends
    # headless_rendering:=true; without it Gazebo segfaults on startup and every
    # sensor topic exists but stays silent.
    -e UBOT_HEADLESS_RENDERING=1
    -v "$WS/src:/ws/src"
    -v "$RUNS:/ws/runs"
    -w /ws
)

cmd="${1:-shell}"; shift || true

case "$cmd" in
shell)
    exec docker run -it "${DOCKER_ARGS[@]}" "$IMAGE" bash
    ;;

check)
    # Everything that must be true before a campaign is worth starting. The
    # render test is the important one: a container can have a perfectly healthy
    # nvidia-smi and still produce BLACK camera frames, which reach the depth
    # model, become a LaserScan, and look like a bad depth model rather than a
    # broken renderer.
    exec docker run -i "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc '
set -e
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== torch ==="
python3 -c "import torch; print(torch.__version__, \"cuda:\", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")"
echo "=== depth model timing (target well under 1 s/frame) ==="
python3 - <<"PY"
import time, numpy as np, torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
m = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
proc = AutoImageProcessor.from_pretrained(m)
net = AutoModelForDepthEstimation.from_pretrained(m).to("cuda" if torch.cuda.is_available() else "cpu").eval()
img = (np.random.rand(360, 640, 3) * 255).astype("uint8")
for _ in range(2):
    inp = proc(images=img, return_tensors="pt").to(net.device)
    with torch.no_grad():
        net(**inp)
t0 = time.time()
for _ in range(5):
    inp = proc(images=img, return_tensors="pt").to(net.device)
    with torch.no_grad():
        net(**inp)
print(f"  {(time.time()-t0)/5*1000:.0f} ms/frame on {net.device}")
PY
echo "=== RENDER TEST: is the camera producing real pixels? ==="
source /ws/install/setup.bash
# SPAWN POSE GOES WITH THE WORLD. sim.launch.py defaults to the Sonoma pit lane
# (272, -136, 3.45); passing world:=apartment alone drops the robot ~300 m
# outside the building, where the camera legitimately sees nothing. That looks
# exactly like a rendering failure and is not one.
setsid ros2 launch ubot_bringup sim.launch.py world:=apartment world_name:=apartment \
    spawn_x:=5.0 spawn_y:=0.0 spawn_z:=0.1 spawn_yaw:=1.5708 \
    headless:=true headless_rendering:=true \
    use_rviz:=false use_nav2:=false use_slam:=false \
    > /tmp/render_stack.log 2>&1 &
STACK=$!
sleep 45
# set +e around the probe: under `set -e` a failing probe kills this shell
# immediately and the diagnostic block below never runs, which is how the first
# two attempts reported "no camera frames at all" and nothing else.
set +e
python3 - <<"PY"
import numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
rclpy.init(); n = Node("render_check"); got = []
n.create_subscription(Image, "/camera/rgb/image_raw", lambda m: got.append(m), 5)
end = n.get_clock().now().nanoseconds + 60e9
while rclpy.ok() and not got and n.get_clock().now().nanoseconds < end:
    rclpy.spin_once(n, timeout_sec=0.2)
if not got:
    raise SystemExit("FAIL: no camera frames at all")
a = np.frombuffer(got[0].data, dtype=np.uint8)
print(f"  frame {got[0].width}x{got[0].height}, mean {a.mean():.1f}, std {a.std():.1f}")
if a.std() < 1.0:
    raise SystemExit("FAIL: the frame is uniform - EGL fell back to software or is blank. "
                     "Check NVIDIA_DRIVER_CAPABILITIES includes graphics.")
print("  OK: real pixels")
PY
RENDER_RC=$?
set -e
if [ "$RENDER_RC" -ne 0 ]; then
    # "No frames at all" usually means the STACK did not come up, which is a
    # different problem from a black frame. Say which, instead of leaving the
    # next person to guess at a container that has already been removed.
    echo "--- gz / bridge / spawn errors from the stack log ---"
    grep -aiE "error|failed|unable|cannot|no such|segfault|EGL|ogre|libGL" \
        /tmp/render_stack.log | head -25
    echo "--- last 15 lines ---"
    tail -15 /tmp/render_stack.log
    echo "--- camera topics present ---"
    ros2 topic list 2>/dev/null | grep -i camera || echo "   (no camera topics)"
fi
kill -INT -$STACK 2>/dev/null || true; sleep 5; kill -KILL -$STACK 2>/dev/null || true
exit "$RENDER_RC"
'
    ;;

calibrate)
    # Fit the depth model against Gazebo's TRUE depth and write depth_scale.json.
    #
    # This is not optional polish. Uncalibrated, the model's ranges are wrong by
    # roughly 2x at this viewpoint, so every point depth_to_scan projects lands
    # at the wrong HEIGHT, falls outside its 0.05-0.60 m band, and the scan comes
    # out with no finite range at all - readiness then refuses the episode with
    # "0 scans with a finite range on /scan_mono" even though the whole chain is
    # running. The camera arm cannot navigate until this file exists.
    #
    # The robot is driven in a slow square: a and b are only separable if the
    # reference spans a range of distances, and the tool refuses a bunched
    # reference rather than returning a meaningless number.
    exec docker run -i "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc '
source /ws/install/setup.bash
setsid ros2 launch ubot_mono_nav sim_mono.launch.py     world:=apartment world_name:=apartment     spawn_x:=5.0 spawn_y:=0.0 spawn_z:=0.1 spawn_yaw:=1.5708     headless:=true headless_rendering:=true use_rviz:=false     > /tmp/calib_stack.log 2>&1 &
STACK=$!
echo "waiting for the depth chain..."
sleep 60

( python3 - <<"PY"
import time, rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
rclpy.init(); n = Node("calib_driver")
pub = n.create_publisher(Twist, "/cmd_vel", 10)
t0 = time.time()
while rclpy.ok() and time.time() - t0 < 150:
    t = Twist()
    phase = (time.time() - t0) % 20.0
    t.linear.x = 0.10 if phase < 12 else 0.0
    t.angular.z = 0.0 if phase < 12 else 0.25
    pub.publish(t); time.sleep(0.1)
pub.publish(Twist())
PY
) &
DRIVER=$!

python3 /ws/src/ubot/ubot_eval/scripts/calibrate_sim_depth.py     --frames 80 --timeout-s 200 --out /ws/runs/depth_scale_sim.json
RC=$?
kill $DRIVER 2>/dev/null || true
kill -INT -$STACK 2>/dev/null || true; sleep 5; kill -KILL -$STACK 2>/dev/null || true
if [ $RC -ne 0 ]; then
    echo "--- depth chain log ---"
    grep -aiE "mono_depth_node|depth_to_scan" /tmp/calib_stack.log | tail -12
fi
exit $RC'
    ;;

smoke)
    exec docker run -i "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc '
source /ws/install/setup.bash
python3 -m ubot_eval.campaign --run --smoke --runs-root /ws/runs "$@"' -- "$@"
    ;;

campaign)
    # DETACHED, on purpose. A matrix runs for hours and `docker run -i` is tied
    # to the ssh session that started it: this project has already lost a run to
    # a dropped connection, and sshd itself gets starved while Gazebo and nav2
    # saturate the cores, so the connection dropping is the NORMAL case, not an
    # accident. Detached means the campaign survives it.
    #
    # HOST_UID/HOST_GID: the container runs as root, so everything it writes
    # into the bind-mounted runs/ is root-owned and the host user cannot even
    # delete it. The container chowns the tree back on the way out.
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$CONTAINER"         -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)"         "${DOCKER_ARGS[@]}" "$IMAGE" bash -lc '
source /ws/install/setup.bash
python3 -m ubot_eval.campaign --run --resume --runs-root /ws/runs "$@"
rc=$?
chown -R "${HOST_UID}:${HOST_GID}" /ws/runs 2>/dev/null || true
exit $rc' -- "$@"
    echo "campaign started detached as '$CONTAINER'."
    echo
    echo "  progress:  $HERE/run.sh progress"
    echo "  results:   $HERE/run.sh results"
    echo "  stop:      docker stop $CONTAINER"
    echo
    echo "It survives this ssh session closing. --resume is on, so re-running"
    echo "this command after an interruption continues where it stopped."
    ;;

progress)
    # Safe to run at any time, including while the campaign is going.
    if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        echo "STATUS: running ($(docker ps --filter "name=$CONTAINER" --format '{{.Status}}'))"
    elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
        echo "STATUS: finished ($(docker ps -a --filter "name=$CONTAINER" --format '{{.Status}}'))"
    else
        echo "STATUS: no campaign container named '$CONTAINER'"
    fi
    echo "load:   $(uptime | sed 's/.*load/load/')"
    echo
    echo "--- episodes written so far ---"
    find "$RUNS" -name episode.json 2>/dev/null | wc -l | sed 's/^/  total: /'
    find "$RUNS" -name episode.json 2>/dev/null -exec grep -ho '"outcome": *"[a-z_]*"' {} +         | sort | uniq -c | sed 's/^/  /'
    echo
    echo "--- last 15 lines from the campaign ---"
    docker logs --tail 15 "$CONTAINER" 2>&1 || true
    ;;

results)
    # Reads only the episode.json files, so it works mid-campaign too.
    docker run -i --rm --network host         -v "$WS/src:/ws/src" -v "$RUNS:/ws/runs" -w /ws "$IMAGE" bash -lc '
source /ws/install/setup.bash
python3 -m ubot_eval.report --runs-root /ws/runs "$@"
echo
echo "=== writing the PDF ==="
python3 -m ubot_eval.report_pdf --runs-root /ws/runs     --out /ws/runs/results.pdf --title "ubot navigation trials"
chown -R "'"$(id -u)"':'"$(id -g)"'" /ws/runs 2>/dev/null || true' -- "$@"
    echo
    echo "PDF: $RUNS/results.pdf"
    ;;

*)
    echo "usage: run.sh {check|calibrate|smoke|campaign|progress|results|shell} [args]" >&2
    exit 2
    ;;
esac
