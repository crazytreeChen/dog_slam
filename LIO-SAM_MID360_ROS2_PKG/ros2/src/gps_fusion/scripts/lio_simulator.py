#!/usr/bin/env python3
"""
LIO 里程计模拟节点 — 生成圆形轨迹 Odometry，模拟机器人移动

用法:
  ros2 run gps_fusion lio_simulator.py --ros-args \
      -p topic:=/Odometry -p rate:=10.0 -p radius:=30.0 -p speed:=1.5
"""

import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist, Vector3


class LioSimulator(Node):
    def __init__(self):
        super().__init__('lio_simulator')

        self.declare_parameter('topic', '/Odometry')
        self.declare_parameter('rate', 10.0)
        self.declare_parameter('radius', 30.0)
        self.declare_parameter('speed', 1.5)
        self.declare_parameter('frame_id', 'world')

        self._rate = self.get_parameter('rate').value
        self._radius = self.get_parameter('radius').value
        self._speed = self.get_parameter('speed').value
        self._frame_id = self.get_parameter('frame_id').value

        self._angle = 0.0
        self._angle_step = self._speed / self._radius / self._rate
        self._seq = 0

        self._pub = self.create_publisher(
            Odometry, self.get_parameter('topic').value, 10)
        self._timer = self.create_timer(1.0 / self._rate, self._publish)

        circum = 2.0 * math.pi * self._radius
        lap_time = circum / self._speed if self._speed > 0 else float('inf')
        self.get_logger().info(
            f'LIO 模拟器已启动: topic={self.get_parameter("topic").value}, '
            f'radius={self._radius}m, speed={self._speed}m/s, '
            f'周长={circum:.0f}m, 圈时≈{lap_time:.0f}s')

    def _publish(self):
        now = self.get_clock().now()
        x = self._radius * math.cos(self._angle)
        y = self._radius * math.sin(self._angle)
        heading = self._angle + math.pi / 2

        msg = Odometry()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._frame_id
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position = Point(
            x=x + 0.02 * (0.5 - (self._seq % 100) / 100),
            y=y + 0.02 * (0.5 - (self._seq % 130) / 100),
            z=0.0,
        )
        msg.pose.pose.orientation = Quaternion(
            x=0.0, y=0.0,
            z=math.sin(heading / 2),
            w=math.cos(heading / 2),
        )
        msg.twist.twist.linear = Vector3(x=self._speed, y=0.0, z=0.0)

        self._pub.publish(msg)
        self._seq += 1
        self._angle = (self._angle + self._angle_step) % (2.0 * math.pi)


def main(args=None):
    rclpy.init(args=args)
    node = LioSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
