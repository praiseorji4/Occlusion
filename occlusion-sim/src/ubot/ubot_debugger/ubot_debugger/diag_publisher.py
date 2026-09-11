#!/usr/bin/env python3
"""
diag_publisher.py  —  Horizon robot diagnostic publisher

Polls the ESP32 'q' command over serial at a configurable rate and
publishes each field as a named ROS 2 topic so PlotJuggler can plot
them individually.

Published topics (all std_msgs/Float64):
  /horizon/diag/enc/left              raw encoder tick count, left
  /horizon/diag/enc/right             raw encoder tick count, right
  /horizon/diag/rpm/left_target       PID target RPM, left
  /horizon/diag/rpm/right_target      PID target RPM, right
  /horizon/diag/rpm/left_actual       filtered actual RPM, left
  /horizon/diag/rpm/right_actual      filtered actual RPM, right
  /horizon/diag/pid/left_error        RPM error (target - actual), left
  /horizon/diag/pid/right_error       RPM error (target - actual), right
  /horizon/diag/pid/left_integral     integral accumulator, left
  /horizon/diag/pid/right_integral    integral accumulator, right
  /horizon/diag/pid/left_output       final PWM output (-255..255), left
  /horizon/diag/pid/right_output      final PWM output (-255..255), right
  /horizon/diag/vel/linear            computed body linear velocity  (m/s)
  /horizon/diag/vel/angular           computed body angular velocity (rad/s)

Usage:
  ros2 run ubot_bringup diag_publisher \
    --ros-args -p serial_port:=/dev/ttyUSB0 -p publish_rate:=10.0

  Or via the launch file:
  ros2 launch ubot_bringup real_robot.launch.py

Parameters:
  serial_port   (string)  default: /dev/ttyUSB0
  baud_rate     (int)     default: 115200
  publish_rate  (float)   default: 10.0  Hz  (keep <= 15 to avoid
                          saturating the serial command channel)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

import serial
import time


class DiagPublisher(Node):

    def __init__(self):
        super().__init__('horizon_diag_publisher')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter('serial_port',  '/dev/ttyUSB0')
        self.declare_parameter('baud_rate',    115200)
        self.declare_parameter('publish_rate', 10.0)

        port  = self.get_parameter('serial_port').value
        baud  = self.get_parameter('baud_rate').value
        rate  = self.get_parameter('publish_rate').value

        # ── Serial connection ─────────────────────────────────────
        try:
            self._ser = serial.Serial(port, baud, timeout=1.0)
            time.sleep(0.1)          # let the port settle
            self._ser.reset_input_buffer()
            self.get_logger().info(f'Serial open: {port} @ {baud}')
        except serial.SerialException as e:
            self.get_logger().fatal(f'Cannot open serial port: {e}')
            raise SystemExit(1)

        # ── Publishers — one per diagnostic field ─────────────────
        def pub(topic):
            return self.create_publisher(Float64, topic, 10)

        self._pubs = {
            'l_enc':     pub('/horizon/diag/enc/left'),
            'r_enc':     pub('/horizon/diag/enc/right'),
            'l_tgt':     pub('/horizon/diag/rpm/left_target'),
            'r_tgt':     pub('/horizon/diag/rpm/right_target'),
            'l_rpm':     pub('/horizon/diag/rpm/left_actual'),
            'r_rpm':     pub('/horizon/diag/rpm/right_actual'),
            'l_err':     pub('/horizon/diag/pid/left_error'),
            'r_err':     pub('/horizon/diag/pid/right_error'),
            'l_int':     pub('/horizon/diag/pid/left_integral'),
            'r_int':     pub('/horizon/diag/pid/right_integral'),
            'l_out':     pub('/horizon/diag/pid/left_output'),
            'r_out':     pub('/horizon/diag/pid/right_output'),
            'lin_vel':   pub('/horizon/diag/vel/linear'),
            'ang_vel':   pub('/horizon/diag/vel/angular'),
        }

        # ── Timer ─────────────────────────────────────────────────
        period = 1.0 / rate
        self.create_timer(period, self._poll_and_publish)
        self.get_logger().info(
            f'Diagnostic publisher running at {rate} Hz')

    # ── Core poll/publish cycle ───────────────────────────────────

    def _poll_and_publish(self):
        """Send 'q', parse the 14-field response, publish each field."""
        line = self._send_command('q\r')
        if line is None:
            return

        fields = self._parse_diag(line)
        if fields is None:
            return

        keys = [
            'l_enc', 'r_enc',
            'l_tgt', 'r_tgt',
            'l_rpm', 'r_rpm',
            'l_err', 'r_err',
            'l_int', 'r_int',
            'l_out', 'r_out',
            'lin_vel', 'ang_vel',
        ]
        for key, value in zip(keys, fields):
            msg = Float64()
            msg.data = value
            self._pubs[key].publish(msg)

    # ── Serial helpers ────────────────────────────────────────────

    def _send_command(self, cmd: str) -> 'str | None':
        """Flush, write command, read one line back."""
        try:
            self._ser.reset_input_buffer()
            self._ser.write(cmd.encode())
            raw = self._ser.readline()           # blocks up to timeout
            return raw.decode('ascii', errors='replace').strip()
        except serial.SerialException as e:
            self.get_logger().warn(f'Serial error: {e}')
            return None

    def _parse_diag(self, line: str) -> 'list[float] | None':
        """
        Parse a 'D field1 field2 ... field14' line.
        Returns a list of 14 floats, or None on any error.
        """
        if not line.startswith('D'):
            self.get_logger().debug(f'Unexpected response: {line!r}')
            return None

        parts = line.split()          # ['D', f1, f2, ..., f14]
        if len(parts) != 15:          # 'D' + 14 fields
            self.get_logger().warn(
                f'Expected 15 tokens, got {len(parts)}: {line!r}')
            return None

        try:
            return [float(p) for p in parts[1:]]
        except ValueError as e:
            self.get_logger().warn(f'Parse error: {e}  line={line!r}')
            return None

    # ── Cleanup ───────────────────────────────────────────────────

    def destroy_node(self):
        if self._ser and self._ser.is_open:
            self._ser.close()
        super().destroy_node()


# ── Entry point ───────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DiagPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()