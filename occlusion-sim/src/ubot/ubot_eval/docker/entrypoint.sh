#!/usr/bin/env bash
# Container entrypoint: source ROS, build the mounted workspace if needed, run.
#
# The build is here rather than in the image so that editing a node does not
# mean rebuilding a multi-gigabyte image. It is skipped when install/ already
# exists, so a campaign restart does not pay for it.
set -euo pipefail

# ROS's setup scripts read AMENT_TRACE_SETUP_FILES and other variables without
# defaulting them, so `set -u` kills the container before anything runs:
#   /opt/ros/jazzy/setup.bash: line 8: AMENT_TRACE_SETUP_FILES: unbound variable
# Relax -u across the sourcing only, then restore it for our own code, where an
# unset variable really is a bug worth stopping for.
set +u
source /opt/ros/jazzy/setup.bash
set -u

if [[ -d /ws/src && ! -f /ws/install/setup.bash ]]; then
    echo "[entrypoint] building the workspace (first run)..."
    cd /ws && colcon build --symlink-install \
        --packages-select ubot_description ubot_bringup ubot_control ubot_mono_nav ubot_eval
fi

if [[ -f /ws/install/setup.bash ]]; then
    # Same -u problem as the ROS setup above, one layer down: colcon's generated
    # setup.bash reads COLCON_TRACE without defaulting it. Fixing only the first
    # source moved the failure here rather than removing it.
    set +u
    source /ws/install/setup.bash
    set -u
fi

# mono_depth_node needs torch; everything else needs only rclpy. Putting the
# venv first on PATH keeps both importable in the same process.
export PATH="${DEPTH_VENV:-/opt/depthenv}/bin:${PATH}"

# PATH IS NOT ENOUGH. ROS installs each node as an executable whose shebang is
# the SYSTEM python (/usr/bin/python3), so launching mono_depth_node ignores the
# venv on PATH entirely and dies with
#   ModuleNotFoundError: No module named 'torch'
# taking the whole camera stack down with it. The venv was created with
# --system-site-packages, so the traffic only has to go the other way: put its
# site-packages on PYTHONPATH and the system interpreter can import torch while
# rclpy keeps coming from /opt/ros.
_DEPTH_SITE=$(ls -d "${DEPTH_VENV:-/opt/depthenv}"/lib/python*/site-packages 2>/dev/null | head -1)
if [[ -n "${_DEPTH_SITE}" ]]; then
    export PYTHONPATH="${_DEPTH_SITE}:${PYTHONPATH:-}"
fi

exec "$@"
