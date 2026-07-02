#!/usr/bin/env python3
"""
Mock TF 广播器 — 模拟 AMCL 漂移用于测试 rtk_pose_monitor 纠偏能力

发布一个可配置偏移的 static TF: map → base_footprint。
初始位姿 = GPS 推算值（无漂移），然后定时在 TF 上叠加模拟偏移量。

用法:
    ros2 run gps_fusion mock_tf_broadcaster.py --ros-args \
        -p drift_per_second:=0.3 -p init_x:=0.0 -p init_y:=0.0 -p init_yaw:=0.0

验证纠偏:
    1. 启动 rtk_simulator + gps_preprocessor + rtk_nav_bridge + 本节点
    2. 本节点 TF 慢慢漂移，超过 drift_threshold(默认2m) 时应触发 /initialpose
    3. ros2 topic echo /initialpose 查看纠偏消息

注意:
    本节点仅在室内模拟测试时使用，不应在生产环境运行。
    TF 以 map 为父帧、base_footprint 为子帧，与 AMCL 配置一致。
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster


def yaw_to_quat(yaw: float) -> Quaternion:
    half = yaw * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


class MockTfBroadcaster(Node):
    """发布带模拟漂移的 map→base_footprint TF"""

    def __init__(self):
        super().__init__('mock_tf_broadcaster')

        # ---- 参数 ----
        self.declare_parameter('init_x', 0.0)
        self.declare_parameter('init_y', 0.0)
        self.declare_parameter('init_yaw', 0.0)
        self.declare_parameter('drift_per_second', 0.3)   # 每秒漂移量（m）
        self.declare_parameter('drift_direction_deg', 135.0)  # 漂移方向（度, 0=正东）
        self.declare_parameter('rate', 10.0)               # 发布频率（Hz）
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')

        self._x = self.get_parameter('init_x').value
        self._y = self.get_parameter('init_y').value
        self._yaw = self.get_parameter('init_yaw').value
        self._drift = self.get_parameter('drift_per_second').value
        self._dir_rad = math.radians(
            self.get_parameter('drift_direction_deg').value)
        self._rate = self.get_parameter('rate').value
        self._map_frame = self.get_parameter('map_frame').value
        self._base_frame = self.get_parameter('base_frame').value

        self._dx_per_tick = self._drift * math.cos(self._dir_rad) / self._rate
        self._dy_per_tick = self._drift * math.sin(self._dir_rad) / self._rate

        self._broadcaster = TransformBroadcaster(self)

        self._timer = self.create_timer(1.0 / self._rate, self._publish)

        self._initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose',
            self._on_initialpose, 10,
        )

        self.get_logger().info(
            f'Mock TF ≈ AMCL 漂移模拟器已启动:\n'
            f'  初始位姿: ({self._x:.2f}, {self._y:.2f}, {math.degrees(self._yaw):.1f}°)\n'
            f'  漂移速率: {self._drift:.2f} m/s 方向: {math.degrees(self._dir_rad):.0f}°\n'
            f'  TF: {self._map_frame} → {self._base_frame}')

    def _publish(self):
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self._map_frame
        t.child_frame_id = self._base_frame
        t.transform.translation = Vector3(x=self._x, y=self._y, z=0.0)
        t.transform.rotation = yaw_to_quat(self._yaw)
        self._broadcaster.sendTransform(t)

        # 累加漂移
        self._x += self._dx_per_tick
        self._y += self._dy_per_tick

    def _on_initialpose(self, msg):
        self._x = msg.pose.pose.position.x
        self._y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self._yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        self.get_logger().info(
            f'收到 /initialpose 纠偏 → 复位到 ({self._x:.2f}, {self._y:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = MockTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
