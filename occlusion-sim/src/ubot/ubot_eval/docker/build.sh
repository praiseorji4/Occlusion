#!/usr/bin/env bash
# Build the trials image. Run from anywhere; the context is this directory.
#
#   ./build.sh                 # build
#   ./build.sh --no-cache      # rebuild from scratch (e.g. new torch/CUDA)
#
# Needs docker WITHOUT sudo:  sudo usermod -aG docker $USER   (then re-login)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-ubot_trials:jazzy}"

if ! docker info >/dev/null 2>&1; then
    echo "docker is not usable as $(id -un)." >&2
    echo "  sudo usermod -aG docker $(id -un)   # then log out and back in" >&2
    exit 1
fi

echo "building $IMAGE from $HERE"
docker build "$@" -t "$IMAGE" "$HERE"

echo
echo "built $IMAGE. Next:"
echo "  $HERE/run.sh check      # GPU, torch, and the render test - do this first"
echo "  $HERE/run.sh smoke      # two arms, one course, two trials"
