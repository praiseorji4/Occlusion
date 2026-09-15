r"""OAK-D Lite RGB publisher, for the Pi. Colour only -- the stereo stays off.

Publishes : ~/image_raw (Image bgr8) or ~/image_raw/compressed, and ~/camera_info

This is the whole robot-side addition for LiDAR-less navigation. Stereo depth is
deliberately NOT enabled: the point of the exercise is that a single RGB image
drives the planner, and the stereo pair is what we keep in reserve to grade the
result against.

CAMERA INFO COMES FROM THE DEVICE
---------------------------------
`readCalibration()` gives this unit's own intrinsics, scaled here to whatever
resolution is streamed. Do not substitute datasheet numbers: the catalogue
figure for this camera (69 deg horizontal at 4:3) disagrees with the measured
one (65 deg at 16:9), and the ground-plane check on the robot backs the measured
value.

BANDWIDTH
---------
640x360 JPEG at 5 Hz is roughly 0.5 MB/s, which WiFi carries comfortably. Raw
bgr8 at the same size and rate is ~3.5 MB/s and will drop frames on a busy
network, so `compressed` defaults to true.
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class OakRgbNode(Node):
    def __init__(self) -> None:
        super().__init__('oak_rgb_node')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 360)
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('compressed', True)
        self.declare_parameter('jpeg_quality', 80)
        # camera_rgb_frame, not camera_depth_frame: the latter is attached with
        # rpy="0 0 0" in the ubot URDF and is not an optical frame despite its
        # comment. Stamping the wrong one rotates every obstacle by 90 degrees.
        self.declare_parameter('frame_id', 'camera_rgb_frame')

        p = self.get_parameter
        self.w = int(p('width').value)
        self.h = int(p('height').value)
        self.fps = float(p('fps').value)
        self.compressed = bool(p('compressed').value)
        self.quality = int(p('jpeg_quality').value)
        self.frame_id = str(p('frame_id').value)

        if self.compressed:
            self.pub_img = self.create_publisher(CompressedImage, '~/image_raw/compressed',
                                                 qos_profile_sensor_data)
        else:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.pub_img = self.create_publisher(Image, '~/image_raw', qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, '~/camera_info',
                                              qos_profile_sensor_data)

        self._start_device()
        self.create_timer(1.0 / max(self.fps, 1.0), self._tick)

    def _start_device(self) -> None:
        import depthai as dai
        self.dai = dai
        pipeline = dai.Pipeline()
        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setIspScale(self.w, 1920)          # 1080p -> requested width
        cam.setFps(self.fps)
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName('rgb')
        cam.isp.link(xout.input)

        self.device = dai.Device(pipeline)
        self.queue = self.device.getOutputQueue('rgb', 2, False)

        calib = self.device.readCalibration()
        k = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 1280, 720)
        sx, sy = self.w / 1280.0, self.h / 720.0
        self.k = [k[0][0] * sx, 0.0, k[0][2] * sx,
                  0.0, k[1][1] * sy, k[1][2] * sy,
                  0.0, 0.0, 1.0]
        self.get_logger().info(
            f'OAK-D RGB {self.w}x{self.h} @ {self.fps:g} fps, '
            f'f_y {self.k[4]:.1f} px, frame {self.frame_id}, '
            f'{"jpeg" if self.compressed else "raw"}')

    def _tick(self) -> None:
        pkt = self.queue.tryGet()
        if pkt is None:
            return
        frame = pkt.getCvFrame()
        stamp = self.get_clock().now().to_msg()

        if self.compressed:
            import cv2
            ok, buf = cv2.imencode('.jpg', frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                self.get_logger().warn('jpeg encode failed', throttle_duration_sec=5.0)
                return
            msg = CompressedImage()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.format = 'jpeg'
            msg.data = buf.tobytes()
        else:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
        self.pub_img.publish(msg)

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width, info.height = self.w, self.h
        info.k = self.k
        info.p = [self.k[0], 0.0, self.k[2], 0.0,
                  0.0, self.k[4], self.k[5], 0.0,
                  0.0, 0.0, 1.0, 0.0]
        info.distortion_model = 'plumb_bob'
        self.pub_info.publish(info)

    def destroy_node(self) -> bool:
        if hasattr(self, 'device'):
            self.device.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OakRgbNode()
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
