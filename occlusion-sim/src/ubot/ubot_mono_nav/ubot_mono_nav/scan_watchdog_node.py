r"""Stop the robot when camera perception stalls.

    nav2  ->  /cmd_vel_raw  ->  [watchdog]  ->  /cmd_vel  ->  twist_stamper

Camera obstacles arrive over WiFi after a network hop and a neural network. That
pipeline WILL stall -- a dropped packet, a laptop that swaps, a model reload. The
failure mode without a watchdog is the worst one available: nav2's last command
keeps executing, so the robot drives on with perception frozen.

While `~/scan` is fresher than `stale_after`, commands pass through untouched.
Once it goes stale, this publishes zeros at `publish_rate` until it recovers. It
also refuses to pass anything through before the first scan ever arrives, so a
mistimed launch cannot roll the robot away.

Deliberately a separate node, not a nav2 plugin: it must keep running and keep
publishing zeros even if the nav2 stack is the thing that died.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanWatchdog(Node):
    def __init__(self) -> None:
        super().__init__('scan_watchdog')
        self.declare_parameter('stale_after', 0.5)
        self.declare_parameter('publish_rate', 20.0)
        self.stale_after = float(self.get_parameter('stale_after').value)
        rate = max(float(self.get_parameter('publish_rate').value), 1.0)

        self.last_scan: float | None = None
        self.blocked = True                     # nothing passes until a scan arrives

        self.pub = self.create_publisher(Twist, '~/cmd_vel_out', 10)
        self.create_subscription(Twist, '~/cmd_vel_in', self._on_cmd, 10)
        self.create_subscription(LaserScan, '~/scan', self._on_scan, qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            f'watchdog armed: commands pass only while the scan is < {self.stale_after:.2f}s old')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _fresh(self) -> bool:
        return self.last_scan is not None and (self._now() - self.last_scan) <= self.stale_after

    def _on_scan(self, _msg: LaserScan) -> None:
        first = self.last_scan is None
        self.last_scan = self._now()
        if first:
            self.get_logger().info('first scan received; commands enabled')

    def _on_cmd(self, msg: Twist) -> None:
        if self._fresh():
            self.blocked = False
            self.pub.publish(msg)
        elif not self.blocked:
            self.blocked = True
            self.pub.publish(Twist())
            self.get_logger().warn('scan is stale -- stopping the robot')

    def _tick(self) -> None:
        # Keep asserting zero while stale: one stop message can be dropped, and
        # a driver with a command timeout of its own needs to keep hearing it.
        if not self._fresh():
            if not self.blocked:
                self.blocked = True
                self.get_logger().warn('scan is stale -- stopping the robot')
            self.pub.publish(Twist())


def main() -> None:
    rclpy.init()
    node = ScanWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
