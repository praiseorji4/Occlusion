r"""RGB image -> metric depth image.

    ros2 run ubot_mono_nav mono_depth_node --ros-args \
        -p depth_scale_json:=/path/to/depth_scale.json

Subscribes : ~/image (sensor_msgs/Image), ~/camera_info
Publishes  : ~/depth (Image, 32FC1, metres), ~/depth/camera_info

Runs on the LAPTOP, not the Pi: Depth Anything on a Pi 5 CPU is seconds per
frame, which cannot drive a planner. The Pi streams RGB (see oak_rgb_node).

The published depth keeps the RGB image's frame_id, because the depth is
computed from that image and shares its optical frame. On the ubot, check which
frame the driver actually stamps: `camera_depth_frame` in the URDF is NOT an
optical frame despite its comment (it is attached rpy="0 0 0"), while
`camera_rgb_frame` is. Use `override_frame_id` if the driver stamps the wrong
one -- an obstacle map built in a 90-degree-rotated frame looks plausible and is
entirely wrong.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from ubot_mono_nav.depth_backend import INDOOR_MODEL, MonoDepth, load_affine, scale_intrinsics


class MonoDepthNode(Node):
    def __init__(self) -> None:
        super().__init__('mono_depth_node')
        self.declare_parameter('model_id', INDOOR_MODEL)
        self.declare_parameter('input_size', 0)          # 0 = the model's own
        self.declare_parameter('device', '')             # '' = cuda if present
        self.declare_parameter('depth_scale_json', '')
        self.declare_parameter('max_rate_hz', 5.0)
        self.declare_parameter('override_frame_id', '')

        size = int(self.get_parameter('input_size').value) or None
        device = str(self.get_parameter('device').value) or None
        affine, prov = load_affine(str(self.get_parameter('depth_scale_json').value))
        self.min_period = 1.0 / max(float(self.get_parameter('max_rate_hz').value), 0.1)
        self.override_frame = str(self.get_parameter('override_frame_id').value)

        self.bridge = CvBridge()
        self.info: CameraInfo | None = None
        self.last_stamp = 0.0
        self.n = 0

        self.get_logger().info(f'loading {self.get_parameter("model_id").value} ...')
        self.depth = MonoDepth(str(self.get_parameter('model_id').value), size, device, affine)
        self.get_logger().info(self.depth.describe())
        if affine is None:
            self.get_logger().warn(
                f'DEPTH IS {prov}. Ranges are the model\'s own units, which at this '
                'viewpoint are wrong by a factor of ~2. Fine for a first look, NOT for '
                'driving. Produce a depth_scale.json with occlusion/eval/depth_affine.py.')
        else:
            self.get_logger().info(f'calibration: {prov}')

        self.pub = self.create_publisher(Image, '~/depth', qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, '~/depth/camera_info',
                                              qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '~/camera_info', self._on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '~/image', self._on_image, qos_profile_sensor_data)

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def _on_image(self, msg: Image) -> None:
        # Rate-limit by wall clock: inference is slower than the camera, and a
        # queue of stale frames is worse than a lower frame rate.
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_stamp < self.min_period:
            return
        if self.info is None:
            self.get_logger().warn('no camera_info yet; cannot scale intrinsics',
                                   throttle_duration_sec=5.0)
            return
        self.last_stamp = now

        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        depth = self.depth(np.ascontiguousarray(rgb))

        out = self.bridge.cv2_to_imgmsg(depth, encoding='32FC1')
        out.header = msg.header
        if self.override_frame:
            out.header.frame_id = self.override_frame
        self.pub.publish(out)

        # camera_info for the DEPTH image, which is smaller than the RGB one
        fx, fy, cx, cy = scale_intrinsics(
            self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5],
            (self.info.width, self.info.height), (depth.shape[1], depth.shape[0]))
        info = CameraInfo()
        info.header = out.header
        info.width, info.height = depth.shape[1], depth.shape[0]
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        self.pub_info.publish(info)

        self.n += 1
        if self.n % 20 == 0:
            finite = np.isfinite(depth)
            self.get_logger().info(
                f'{self.n} frames, {depth.shape[1]}x{depth.shape[0]}, '
                f'{depth[finite].min():.1f}-{depth[finite].max():.1f} m')


def main() -> None:
    rclpy.init()
    node = MonoDepthNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl-C and `ros2 launch` shutdown are normal exits, not faults. Without
        # catching the second one, every stop dumps a traceback into the robot log.
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
