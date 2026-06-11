#!/usr/bin/env python3
"""
人工设置AMCL初始位姿节点

支持两种使用方式：
1. ROS2参数方式（可作为LifecycleNode被launch文件管理）：
   ros2 run nav2_dog_slam set_initial_pose --ros-args -p x:=2.0 -p y:=3.5 -p yaw_deg:=90.0

2. 命令行参数方式（一次性发布后退出）：
   ros2 run nav2_dog_slam set_initial_pose --x 2.0 --y 3.5 --yaw 1.57

发布到 /initialpose 后自动退出，适合人工开机校准流程。
"""

import sys
import argparse
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion


class InitialPosePublisher(Node):
    def __init__(self, x=None, y=None, yaw=None, yaw_deg=None):
        super().__init__('set_initial_pose')

        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('yaw_deg', 0.0)

        self.publisher = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # 优先使用命令行参数，否则从ROS参数读取
        px = x if x is not None else self.get_parameter('x').value
        py = y if y is not None else self.get_parameter('y').value
        pyaw_deg = yaw_deg if yaw_deg is not None else self.get_parameter('yaw_deg').value

        # 如果 yaw 以弧度直接传入，使用弧度
        if yaw is not None:
            pyaw = yaw
        else:
            pyaw = math.radians(pyaw_deg)

        self.has_published = False
        self.timer = self.create_timer(0.5, lambda: self._publish_once(px, py, pyaw))

    def _publish_once(self, x, y, yaw):
        if self.has_published:
            return

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        msg.pose.pose.position = Point(x=x, y=y, z=0.0)
        msg.pose.pose.orientation = Quaternion(
            x=0.0,
            y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )

        # 协方差：x方差=0.25, y方差=0.25, yaw方差=0.0685
        cov = [0.0] * 36
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = 0.06853892326654787
        msg.pose.covariance = cov

        self.publisher.publish(msg)
        self.get_logger().info(
            f'Published initial pose to /initialpose: '
            f'x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f}deg'
        )
        self.has_published = True

        # 发布后等0.5秒再退出，确保消息送达
        self.create_timer(0.5, lambda: rclpy.shutdown())


def main(args=None):
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='Set AMCL initial pose manually'
    )
    parser.add_argument('--x', type=float, default=None,
                        help='X position in map frame')
    parser.add_argument('--y', type=float, default=None,
                        help='Y position in map frame')
    parser.add_argument('--yaw', type=float, default=None,
                        help='Yaw angle in radians')
    parser.add_argument('--yaw-deg', type=float, default=None,
                        help='Yaw angle in degrees (ROS param style)')

    # 分离ROS参数和自定义参数
    parsed, remaining = parser.parse_known_args(args)

    rclpy.init(args=remaining)

    node = InitialPosePublisher(
        x=parsed.x, y=parsed.y,
        yaw=parsed.yaw, yaw_deg=parsed.yaw_deg
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
