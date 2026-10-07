# Navigation trials on a GPU box: ROS 2 Jazzy + Gazebo Harmonic + Depth Anything.
#
# Built for riftvm (2x Tesla V100, 10 CPUs). The reason this image exists is that
# WSL has no GPU passthrough, so mono_depth_node runs at ~1-2 s/frame there and
# every camera arm of the experiment is unrunnable.
#
#   build:  docker/build.sh
#   run:    docker/run.sh            (needs --gpus all; see the EGL note below)
#
# ---------------------------------------------------------------------------
# THE TRAP THIS IMAGE IS SHAPED AROUND: BLACK CAMERA FRAMES
# ---------------------------------------------------------------------------
# Gazebo's camera sensor renders through ogre2. Without a working EGL/GL vendor
# inside the container it silently falls back to software or produces BLACK
# frames - and a black frame does not error. It reaches the depth model, which
# returns a confident, uniform, wrong depth map, which becomes a LaserScan, which
# the robot navigates by. The failure looks like "the depth model is bad".
#
# So: the NVIDIA EGL vendor file is required, and docker/run.sh checks a rendered
# frame's pixel variance before any campaign is allowed to start.
FROM ros:jazzy-ros-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# --- ROS side -------------------------------------------------------------
# ros-gz brings Gazebo Harmonic; gz-ros2-control is what actually drives the
# wheels in sim, and is easy to forget because it is pulled in by the robot's
# xacro rather than by any launch file.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz \
        ros-jazzy-gz-ros2-control \
        ros-jazzy-ros2-control \
        ros-jazzy-ros2-controllers \
        ros-jazzy-nav2-bringup \
        ros-jazzy-slam-toolbox \
        ros-jazzy-robot-localization \
        ros-jazzy-twist-stamper \
        ros-jazzy-rosbag2 \
        ros-jazzy-rosbag2-storage-mcap \
        ros-jazzy-tf-transformations \
        # ubot_control talks to the real robot's ESP32 over serial and is not
        # used by a sim trial, but it is in the workspace and colcon builds it.
        # Installing the dependency keeps the container's package set identical
        # to the local workspace; excluding the package instead would mean the
        # two build different things, which is how "works locally, fails in the
        # container" starts.
        libserial-dev \
        # THE GL/EGL RUNTIME. ros:jazzy-ros-base ships none of it, so ogre2
        # cannot even create a render system: it logs "Unable to load Ogre
        # Plugin[RenderSystem_GL3Plus]" and "unable to find OpenGL 3+ Rendering
        # Subsystem", then SEGFAULTS (exit 139) and takes the whole sim down.
        # The camera and LiDAR topics still exist because the bridge is up, so
        # the only symptom is that nothing ever publishes on them.
        # libglvnd is the dispatch layer that lets the NVIDIA driver's
        # libEGL_nvidia (injected by nvidia-container-runtime when
        # NVIDIA_DRIVER_CAPABILITIES includes `graphics`) be found at runtime.
        libglvnd0 \
        libgl1 \
        libglx0 \
        libegl1 \
        libgles2 \
        libopengl0 \
        libglx-mesa0 \
        libgl1-mesa-dri \
        libxrandr2 \
        libxi6 \
        libxcursor1 \
        libxinerama1 \
        python3-colcon-common-extensions \
        python3-pip python3-venv \
        git rsync xz-utils \
    && rm -rf /var/lib/apt/lists/*

# --- GPU rendering --------------------------------------------------------
# NVIDIA_DRIVER_CAPABILITIES must include `graphics` (not just `compute`), or
# the GPU is visible to torch while ogre2 still cannot render - the exact split
# that produces black camera frames with a healthy-looking nvidia-smi.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    QT_QPA_PLATFORM=offscreen \
    GZ_SIM_RESOURCE_PATH=/ws/install/ubot_bringup/share/ubot_bringup/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglvnd0 libglx0 \
    && rm -rf /var/lib/apt/lists/*

# --- the depth model ------------------------------------------------------
# A venv with system site packages: torch must sit beside rclpy, because
# mono_depth_node imports both in one process. Ubuntu 24.04 refuses a bare
# `pip install` into the system python, and --break-system-packages here would
# put CUDA wheels somewhere apt can later stand on.
ENV DEPTH_VENV=/opt/depthenv
RUN python3 -m venv --system-site-packages $DEPTH_VENV \
    && $DEPTH_VENV/bin/pip install --no-cache-dir --upgrade pip \
    && $DEPTH_VENV/bin/pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && $DEPTH_VENV/bin/pip install --no-cache-dir transformers pillow numpy
# torchvision is NOT optional despite nothing importing it by name: transformers
# builds the *fast* image processor on it, and AutoImageProcessor for this model
# resolves to the fast path, so without it the weights-caching step below dies on
# "AutoImageProcessor requires the Torchvision library". It must come from the
# same cu124 index as torch or pip pulls a CPU build and drags a second torch in.

# Bake the weights in. Without this every episode's first frame waits on a
# download, and a campaign is hundreds of episodes.
ENV HF_HOME=/opt/hf
RUN $DEPTH_VENV/bin/python -c "\
from transformers import AutoImageProcessor, AutoModelForDepthEstimation; \
m='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'; \
AutoImageProcessor.from_pretrained(m); AutoModelForDepthEstimation.from_pretrained(m); \
print('weights cached')"

# --- the workspace --------------------------------------------------------
# Source is bind-mounted at /ws/src so a code change does not mean a rebuild of
# the image; the entrypoint builds it if install/ is missing.
WORKDIR /ws
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
