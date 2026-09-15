r"""Metric depth image -> LaserScan that nav2 can plan around.

Subscribes : ~/depth (Image 32FC1, metres), ~/depth/camera_info
Publishes  : ~/scan (sensor_msgs/LaserScan), in `base_frame`

The geometry lives in `scan_geometry.py`, which has no ROS dependency and a
selftest that runs on any machine. This node only moves data and resolves the
camera-to-base transform.

WHY A LaserScan AND NOT A PointCloud2
-------------------------------------
nav2 CLEARS free space by ray-tracing a scan. A 65-degree camera leaves most of
the costmap unobserved, so without clearing, obstacles seen once would persist
for ever and the robot would slowly wall itself in. A PointCloud2 source marks
well and clears poorly; a LaserScan does both.

RANGES BEYOND THE TRUSTED DISTANCE ARE +inf
-------------------------------------------
`range_max` is set to `trusted_range`, and anything further reads +inf, meaning
"nothing detected". nav2 clears along those rays only out to range_max, so space
the camera cannot judge is neither marked as blocked nor cleared as free.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from tf2_ros import Buffer, TransformListener

from ubot_mono_nav.scan_geometry import ScanSpec, depth_to_scan, optical_to_base


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class DepthToScanNode(Node):
    def __init__(self) -> None:
        super().__init__('depth_to_scan')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('height_min', 0.05)
        self.declare_parameter('height_max', 0.60)
        self.declare_parameter('trusted_range', 4.0)
        self.declare_parameter('min_depth', 0.3)
        self.declare_parameter('angle_min_deg', -32.5)
        self.declare_parameter('angle_max_deg', 32.5)
        self.declare_parameter('angle_increment_deg', 1.0)
        self.declare_parameter('range_min', 0.15)
        self.declare_parameter('stride', 2)
        self.declare_parameter('min_points', 2)
        # Fallback pose, used only when TF has no camera transform. Matches the
        # OAK-D as built: 0.168 m forward, 0.152 m above the floor, tilted 4.09
        # degrees DOWN by the CamCase, so pitch_up is negative. Used only when TF
        # carries no camera transform; the URDF is the authority.
        self.declare_parameter('use_tf', True)
        self.declare_parameter('fallback_pitch_up_deg', -4.09)
        self.declare_parameter('fallback_xyz', [0.168, 0.0, 0.152])

        p = self.get_parameter
        self.base_frame = str(p('base_frame').value)
        self.spec = ScanSpec(
            angle_min=math.radians(float(p('angle_min_deg').value)),
            angle_max=math.radians(float(p('angle_max_deg').value)),
            angle_increment=math.radians(float(p('angle_increment_deg').value)),
            range_min=float(p('range_min').value),
            range_max=float(p('trusted_range').value))
        self.use_tf = bool(p('use_tf').value)
        self.fallback = optical_to_base(
            math.radians(float(p('fallback_pitch_up_deg').value)),
            [float(v) for v in p('fallback_xyz').value])

        self.bridge = CvBridge()
        self.info: CameraInfo | None = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.warned_tf = False

        self.pub = self.create_publisher(LaserScan, '~/scan', qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '~/depth/camera_info', self._on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '~/depth', self._on_depth, qos_profile_sensor_data)
        self.get_logger().info(
            f'scan in {self.base_frame}: {self.spec.n_bins} bins over '
            f'{p("angle_min_deg").value:.0f}..{p("angle_max_deg").value:.0f} deg, '
            f'trusted to {p("trusted_range").value:.1f} m, '
            f'height band {p("height_min").value:.2f}-{p("height_max").value:.2f} m')

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def _lookup(self, source_frame: str, stamp) -> tuple[np.ndarray, np.ndarray]:
        """optical -> base, from TF, falling back to the measured mount."""
        if self.use_tf:
            try:
                tf = self.tf_buffer.lookup_transform(self.base_frame, source_frame, stamp)
                q = tf.transform.rotation
                t = tf.transform.translation
                return _quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])
            except Exception as e:                      # tf2 raises several types
                if not self.warned_tf:
                    self.get_logger().warn(
                        f'no TF {self.base_frame} <- {source_frame} ({e}); using the '
                        'fallback mount pose. Obstacles will be wrong if the real '
                        'mount differs.')
                    self.warned_tf = True
        return self.fallback

    def _on_depth(self, msg: Image) -> None:
        if self.info is None:
            self.get_logger().warn('no depth camera_info yet', throttle_duration_sec=5.0)
            return
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        rot, trans = self._lookup(msg.header.frame_id, msg.header.stamp)
        p = self.get_parameter
        ranges = depth_to_scan(
            depth, self.info.k[0], self.info.k[4], self.info.k[2], self.info.k[5],
            rot, trans, self.spec,
            height_min=float(p('height_min').value),
            height_max=float(p('height_max').value),
            trusted_range=float(p('trusted_range').value),
            min_depth=float(p('min_depth').value),
            stride=int(p('stride').value),
            min_points=int(p('min_points').value))

        scan = LaserScan()
        scan.header.stamp = msg.header.stamp
        scan.header.frame_id = self.base_frame
        scan.angle_min = self.spec.angle_min
        scan.angle_max = self.spec.angle_max
        scan.angle_increment = self.spec.angle_increment
        scan.range_min = self.spec.range_min
        scan.range_max = self.spec.range_max
        scan.ranges = [float(v) for v in ranges]
        self.pub.publish(scan)


def main() -> None:
    rclpy.init()
    node = DepthToScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
