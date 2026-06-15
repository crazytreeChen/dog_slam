#!/usr/bin/env python3
import sys
import math
import argparse
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Import common message types
try:
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
except ImportError:
    print("Error: Could not import ROS 2 message types. Make sure you sourced ROS 2.")
    sys.exit(1)


class PoseComparator(Node):
    def __init__(self, topic1, topic2, rate, align):
        super().__init__('pose_comparator')
        self.topic1 = topic1
        self.topic2 = topic2
        self.rate = rate
        self.align = align

        self.pose1 = None
        self.pose2 = None
        self.timestamp1 = None
        self.timestamp2 = None

        # Alignment offset (T_offset: Topic1 -> Topic2)
        # We will compute these on the first received message pair
        self.aligned = False
        self.delta_yaw = 0.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.get_logger().info(f"正在启动位姿偏差对比器...")
        self.get_logger().info(f"话题 1: {self.topic1}")
        self.get_logger().info(f"话题 2: {self.topic2}")
        self.get_logger().info(f"对齐初始偏移: {'开启 (计算相对跟踪漂移)' if self.align else '关闭 (直接坐标比对)'}")

        # Set up a timer to detect topic types and subscribe
        self.sub_timer = self.create_timer(1.0, self._try_subscribe)
        self.sub1 = None
        self.sub2 = None

        # Compare loop timer
        self.compare_timer = self.create_timer(self.rate, self._compare_loop)

    def _try_subscribe(self):
        # Dynamically inspect active topic types from ROS graph
        topic_names_and_types = self.get_topic_names_and_types()
        type_map = {t: types for t, types in topic_names_and_types}

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)

        # Topic 1 subscription
        if self.sub1 is None and self.topic1 in type_map:
            t_class = self._get_msg_class(type_map[self.topic1][0])
            if t_class:
                self.sub1 = self.create_subscription(
                    t_class, self.topic1, lambda msg: self._cb(msg, 1), qos
                )
                self.get_logger().info(f"成功订阅 话题1 [{self.topic1}]，类型: {t_class.__name__}")

        # Topic 2 subscription
        if self.sub2 is None and self.topic2 in type_map:
            t_class = self._get_msg_class(type_map[self.topic2][0])
            if t_class:
                self.sub2 = self.create_subscription(
                    t_class, self.topic2, lambda msg: self._cb(msg, 2), qos
                )
                self.get_logger().info(f"成功订阅 话题2 [{self.topic2}]，类型: {t_class.__name__}")

        # If both are subscribed, we can destroy the check timer
        if self.sub1 is not None and self.sub2 is not None:
            self.sub_timer.destroy()

    def _get_msg_class(self, type_str):
        if 'nav_msgs/msg/Odometry' in type_str or 'nav_msgs/Odometry' in type_str:
            return Odometry
        elif 'geometry_msgs/msg/PoseStamped' in type_str or 'geometry_msgs/PoseStamped' in type_str:
            return PoseStamped
        elif 'geometry_msgs/msg/PoseWithCovarianceStamped' in type_str or 'geometry_msgs/PoseWithCovarianceStamped' in type_str:
            return PoseWithCovarianceStamped
        self.get_logger().error(f"不支持的类型: {type_str}")
        return None

    def _cb(self, msg, id):
        # Extract pose and timestamp
        pose = None
        sec = 0
        nanosec = 0
        if hasattr(msg, 'header'):
            sec = msg.header.stamp.sec
            nanosec = msg.header.stamp.nanosec

        if isinstance(msg, Odometry):
            pose = msg.pose.pose
        elif isinstance(msg, PoseWithCovarianceStamped):
            pose = msg.pose.pose
        elif isinstance(msg, PoseStamped):
            pose = msg.pose

        if pose is not None:
            # Convert orientation to yaw
            q = pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            p_data = (pose.position.x, pose.position.y, pose.position.z, yaw)
            t_val = sec + nanosec * 1e-9

            if id == 1:
                self.pose1 = p_data
                self.timestamp1 = t_val
            else:
                self.pose2 = p_data
                self.timestamp2 = t_val

    def _compare_loop(self):
        if self.pose1 is None or self.pose2 is None:
            self.get_logger().info("正在等待两个话题的数据传入...", once=True)
            return

        # Check for timestamp synchronization (warning if older than 2.0 seconds)
        t_now = self.get_clock().now().nanoseconds / 1e9
        age1 = t_now - self.timestamp1
        age2 = t_now - self.timestamp2
        if age1 > 2.0 or age2 > 2.0:
            self.get_logger().warn(
                f"数据延时过大! 话题1延时: {age1:.1f}s, 话题2延时: {age2:.1f}s",
                throttle_duration_sec=3.0
            )

        x1, y1, z1, yaw1 = self.pose1
        x2, y2, z2, yaw2 = self.pose2

        # Initialize alignment offset on first message pair if alignment enabled
        # We only perform alignment if both messages are synchronized in time (e.g., within 2.0 seconds)
        if self.align and not self.aligned:
            time_diff = abs(self.timestamp1 - self.timestamp2)
            if time_diff > 2.0:
                self.get_logger().info(
                    f"正在等待时间戳同步以进行初始对齐 (当前话题1与话题2时间差: {time_diff:.1f}s，请控制机器人移动或发布初始位姿)...",
                    throttle_duration_sec=3.0
                )
                return

            self.delta_yaw = self._norm_angle(yaw2 - yaw1)
            # Rotate topic1 coordinates to match topic2's coordinate frame orientation
            rx1 = x1 * math.cos(self.delta_yaw) - y1 * math.sin(self.delta_yaw)
            ry1 = x1 * math.sin(self.delta_yaw) + y1 * math.cos(self.delta_yaw)
            # Compute translation offset
            self.offset_x = x2 - rx1
            self.offset_y = y2 - ry1
            self.aligned = True
            self.get_logger().info(
                f"[初始对齐计算完成] 坐标偏移 dx={self.offset_x:.3f}m, dy={self.offset_y:.3f}m, 角度偏置 d_yaw={math.degrees(self.delta_yaw):.1f}°"
            )

        # Apply alignment if enabled
        if self.align:
            # Map Pose 1 to Pose 2's coordinate system
            rx1 = x1 * math.cos(self.delta_yaw) - y1 * math.sin(self.delta_yaw) + self.offset_x
            ry1 = x1 * math.sin(self.delta_yaw) + y1 * math.cos(self.delta_yaw) + self.offset_y
            ryaw1 = self._norm_angle(yaw1 + self.delta_yaw)
            
            err_x = x2 - rx1
            err_y = y2 - ry1
            err_dist = math.sqrt(err_x*err_x + err_y*err_y)
            err_yaw = abs(self._norm_angle(yaw2 - ryaw1))
        else:
            # Direct absolute coordinate comparison
            err_x = x2 - x1
            err_y = y2 - y1
            err_dist = math.sqrt(err_x*err_x + err_y*err_y)
            err_yaw = abs(self._norm_angle(yaw2 - yaw1))

        # Output comparison report
        print("\n" + "="*50)
        print(f"【位姿偏差对比报告】 时间戳: {time.strftime('%H:%M:%S', time.localtime())}")
        print(f" 话题1 [{self.topic1}]: X={x1:.3f}, Y={y1:.3f}, Yaw={math.degrees(yaw1):.1f}°")
        print(f" 话题2 [{self.topic2}]: X={x2:.3f}, Y={y2:.3f}, Yaw={math.degrees(yaw2):.1f}°")
        print("-"*50)
        if self.align:
            print(f" 对齐后投影值: X'={rx1:.3f}, Y'={ry1:.3f}, Yaw'={math.degrees(ryaw1):.1f}°")
            print(f" 相对运动漂移误差：")
        else:
            print(f" 绝对坐标位置误差：")
        print(f"  ├─ 水平位置偏差 (2D): {err_dist:.3f} 米  (dx={err_x:.3f}m, dy={err_y:.3f}m)")
        print(f"  └─ 航向角度偏差 (Yaw): {math.degrees(err_yaw):.2f}°")
        print("="*50)

    @staticmethod
    def _norm_angle(a):
        while a > math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a


def main():
    parser = argparse.ArgumentParser(description="ROS 2 Pose and Odometry Topic Comparator")
    parser.add_argument('--topic1', type=str, default='/amcl_pose', help='First topic to compare (default: /amcl_pose)')
    parser.add_argument('--topic2', type=str, default='/lio/odom', help='Second topic to compare (default: /lio/odom)')
    parser.add_argument('--rate', type=float, default=1.0, help='Comparison rate in seconds (default: 1.0)')
    parser.add_argument('--no-align', action='store_false', dest='align', help='Disable automatic initial frame alignment')
    args, unknown = parser.parse_known_args()

    rclpy.init(args=unknown)
    node = PoseComparator(args.topic1, args.topic2, args.rate, args.align)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
