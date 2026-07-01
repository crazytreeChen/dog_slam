#!/usr/bin/env python3
"""
导航阶段 AMCL 漂移监控节点

持续监控 AMCL 位姿（tf2 查询 map→base_footprint）与 GPS 推算位姿的偏差，
当偏差超过阈值时自动补发 /initialpose 纠偏。GPS 失效时自动退化为纯 AMCL。

零侵入原理:
    AMCL 标准行为订阅 /initialpose（PoseWithCovarianceStamped, frame_id=map），
    收到后重置粒子云。本节点只发布该话题，不修改 AMCL 任何配置。

坐标转换链（复用 gps_transform.rtk_to_map）:
    RTK lat/lon → UTM (E, N) → 减原点 (E₀,N₀) → 旋转 θ₀ → map (x, y)

容错状态机:
    MONITORING → (GPS连续N帧无效) → GPS_LOST → (GPS连续M帧有效) → MONITORING

用法:
    # 导航时启动（与 nav2_dog_slam 并行）
    ros2 launch gps_fusion rtk_nav_bridge.launch.py

    # 命名空间模式
    ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot
"""

import math
import os
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from sensor_msgs.msg import NavSatFix
from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener
import tf2_py as tf2

# 共享转换模块
from gps_fusion.gps_transform import (
    make_utm_transformer, latlon_to_utm, rtk_to_map,
    rtk_heading_to_map_yaw, compute_horizontal_accuracy,
    detect_rtk_quality, covariance_for_quality, load_map_origin,
)

# 尝试导入 RTK 消息类型（获取航向）
try:
    from robots_dog_msgs.msg import UniRtkPvh
    _HAS_UNI_RTK_PVH = True
except ImportError:
    UniRtkPvh = None
    _HAS_UNI_RTK_PVH = False


def _yaw_to_quat(yaw: float) -> Quaternion:
    """yaw (rad) → Quaternion"""
    half = yaw * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def _quat_to_yaw(q: Quaternion) -> float:
    """四元数 → yaw (rad)"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RtkPoseMonitor(Node):
    """导航阶段 AMCL 漂移监控器"""

    # 状态机
    MONITORING = 'MONITORING'
    GPS_LOST = 'GPS_LOST'

    def __init__(self):
        super().__init__('rtk_pose_monitor')

        # ---- 参数 ----
        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('fix_topic', '/fix_filtered')
        self.declare_parameter('rtk_topic', '/rtk_pvh')
        self.declare_parameter('use_rtk_heading', True)
        self.declare_parameter('map_origin_file', '')
        self.declare_parameter('ns', '')                          # 命名空间
        self.declare_parameter('map_frame', '')                   # 空则自动推导
        self.declare_parameter('base_frame', '')                  # 空则自动推导
        self.declare_parameter('drift_threshold', 2.0)            # 漂移阈值（m）
        self.declare_parameter('min_correction_interval', 15.0)   # 最小纠偏间隔（s）
        self.declare_parameter('monitor_rate', 2.0)               # 监控频率（Hz）
        self.declare_parameter('gps_loss_threshold', 5)           # 连续N帧无效→GPS_LOST
        self.declare_parameter('gps_recovery_threshold', 3)       # 连续M帧有效→恢复
        self.declare_parameter('min_accuracy', 5.0)               # GPS 最低精度（m）
        self.declare_parameter('cov_rtk_fix', 0.01)
        self.declare_parameter('cov_rtk_float', 0.1)
        self.declare_parameter('cov_dgps', 1.0)
        self.declare_parameter('cov_no_heading', 0.5)             # 无航向时 yaw 协方差

        self._utm_zone = self.get_parameter('utm_zone').value
        self._use_rtk_heading = self.get_parameter('use_rtk_heading').value
        self._drift_threshold = self.get_parameter('drift_threshold').value
        self._min_interval = self.get_parameter('min_correction_interval').value
        self._monitor_rate = self.get_parameter('monitor_rate').value
        self._gps_loss_threshold = self.get_parameter('gps_loss_threshold').value
        self._gps_recovery_threshold = self.get_parameter('gps_recovery_threshold').value
        self._min_accuracy = self.get_parameter('min_accuracy').value
        self._cov_no_heading = self.get_parameter('cov_no_heading').value

        # ---- 命名空间感知的 frame 名 ----
        ns = self.get_parameter('ns').value
        map_frame = self.get_parameter('map_frame').value
        base_frame = self.get_parameter('base_frame').value
        self._map_frame = map_frame if map_frame else (
            f'{ns}/map' if ns else 'map')
        self._base_frame = base_frame if base_frame else (
            f'{ns}/base_footprint' if ns else 'base_footprint')

        # ---- WGS84→UTM 转换器 ----
        self._to_utm = make_utm_transformer(self._utm_zone)

        # ---- 加载地图原点 ----
        origin_file = self.get_parameter('map_origin_file').value
        if not origin_file:
            origin_file = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'map_gps_origin.yaml')
        origin_data = load_map_origin(origin_file)
        if origin_data is None:
            self.get_logger().fatal(
                f'地图原点文件不存在或无效: {origin_file}\n'
                '请先在建图阶段运行 map_origin_recorder 记录原点 GPS')
            raise RuntimeError('map origin not configured')

        self._origin_utm, self._theta0_rad = origin_data
        e0, n0, a0 = self._origin_utm
        self.get_logger().info(
            f'地图原点已加载: UTM({e0:.2f}, {n0:.2f}), '
            f'heading={math.degrees(self._theta0_rad):.4f}°')

        # ---- TF2 ----
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- 状态 ----
        self._state = self.MONITORING
        self._gps_fail_count = 0
        self._gps_recover_count = 0
        self._latest_fix = None
        self._latest_rtk_heading = None
        self._last_correction_time = self.get_clock().now() - Duration(seconds=self._min_interval + 1)

        # ---- 订阅 ----
        self._fix_sub = self.create_subscription(
            NavSatFix,
            self.get_parameter('fix_topic').value,
            self._fix_callback, 10)

        self._rtk_sub = None
        rtk_topic = self.get_parameter('rtk_topic').value
        if self._use_rtk_heading and rtk_topic and _HAS_UNI_RTK_PVH:
            self._rtk_sub = self.create_subscription(
                UniRtkPvh, rtk_topic, self._rtk_callback, 10)
            self.get_logger().info(f'RTK 航向来源: {rtk_topic}')
        elif self._use_rtk_heading and not _HAS_UNI_RTK_PVH:
            self.get_logger().warn(
                'robots_dog_msgs 未安装，无法获取 RTK 航向，'
                '纠偏时 yaw 将使用 AMCL 当前值')
            self._use_rtk_heading = False

        # ---- 发布 /initialpose ----
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # ---- 监控定时器 ----
        self._monitor_timer = self.create_timer(
            1.0 / self._monitor_rate, self._monitor_loop)

        self.get_logger().info(
            f'RTK 位姿监控节点已启动 '
            f'(rate={self._monitor_rate}Hz, drift_threshold={self._drift_threshold}m, '
            f'min_interval={self._min_interval}s, '
            f'map_frame={self._map_frame}, base_frame={self._base_frame})')

    # ==================================================================
    #  回调
    # ==================================================================

    def _fix_callback(self, msg: NavSatFix):
        """更新最新 GPS 数据"""
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            self._latest_fix = None
            return
        h_acc = compute_horizontal_accuracy(msg)
        if h_acc > self._min_accuracy:
            self.get_logger().debug(
                f'GPS 精度不够: {h_acc:.3f}m > {self._min_accuracy}m')
            self._latest_fix = None
            return
        self._latest_fix = msg

    def _rtk_callback(self, msg):
        """从 RTK 原始消息获取航向"""
        try:
            heading = msg.heading
            if heading.heading_type not in (16, 34, 50):
                return
            if heading.sol_status not in (0, 2):
                return
            self._latest_rtk_heading = float(heading.heading_deg)
        except Exception:
            pass

    # ==================================================================
    #  监控主循环
    # ==================================================================

    def _monitor_loop(self):
        """监控主循环：比对 GPS 位姿与 AMCL 位姿"""
        gps_valid = self._latest_fix is not None

        # ---- 状态机转换 ----
        if self._state == self.MONITORING:
            if not gps_valid:
                self._gps_fail_count += 1
                if self._gps_fail_count >= self._gps_loss_threshold:
                    self._state = self.GPS_LOST
                    self._gps_recover_count = 0
                    self.get_logger().warn(
                        f'GPS 连续 {self._gps_fail_count} 帧无效，'
                        f'切换到 GPS_LOST 状态，停止纠偏')
            else:
                self._gps_fail_count = 0
        elif self._state == self.GPS_LOST:
            if gps_valid:
                self._gps_recover_count += 1
                if self._gps_recover_count >= self._gps_recovery_threshold:
                    self._state = self.MONITORING
                    self._gps_fail_count = 0
                    self.get_logger().info(
                        f'GPS 恢复（连续 {self._gps_recover_count} 帧有效），'
                        f'回到 MONITORING 状态')
            else:
                self._gps_recover_count = 0

        # GPS_LOST 状态不纠偏
        if self._state == self.GPS_LOST:
            return

        # GPS 无效不纠偏
        if self._latest_fix is None:
            return

        # ---- 查询 AMCL 位姿（map→base_footprint TF） ----
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
        except (TransformException, tf2.TransformException) as ex:
            self.get_logger().debug(
                f'TF 查询失败 {self._map_frame}→{self._base_frame}: {ex}')
            return

        amcl_x = tf_msg.transform.translation.x
        amcl_y = tf_msg.transform.translation.y
        amcl_yaw = _quat_to_yaw(tf_msg.transform.rotation)

        # ---- GPS 推算地图坐标 ----
        fix = self._latest_fix
        try:
            gps_x, gps_y = rtk_to_map(
                self._to_utm, fix.longitude, fix.latitude,
                self._origin_utm, self._theta0_rad)
        except Exception as ex:
            self.get_logger().warn(f'GPS→map 坐标转换失败: {ex}')
            return

        # NaN 防护
        if math.isnan(gps_x) or math.isinf(gps_x) or \
           math.isnan(gps_y) or math.isinf(gps_y):
            self.get_logger().warn('GPS map 坐标包含 NaN/Inf，跳过')
            return

        # ---- 偏差计算 ----
        drift = math.sqrt((gps_x - amcl_x) ** 2 + (gps_y - amcl_y) ** 2)

        # ---- 纠偏判定 ----
        now = self.get_clock().now()
        elapsed = (now - self._last_correction_time).nanoseconds * 1e-9

        if drift <= self._drift_threshold:
            self.get_logger().debug(
                f'drift={drift:.3f}m <= {self._drift_threshold}m，无需纠偏')
            return

        if elapsed < self._min_interval:
            self.get_logger().debug(
                f'drift={drift:.3f}m 超阈值，但距上次纠偏仅 {elapsed:.1f}s '
                f'< {self._min_interval}s，跳过')
            return

        # ---- 执行纠偏 ----
        quality = detect_rtk_quality(fix)
        pos_cov = covariance_for_quality(
            quality,
            cov_rtk_fix=self.get_parameter('cov_rtk_fix').value,
            cov_rtk_float=self.get_parameter('cov_rtk_float').value,
            cov_dgps=self.get_parameter('cov_dgps').value)

        # 航向处理
        if self._use_rtk_heading and self._latest_rtk_heading is not None:
            map_yaw = rtk_heading_to_map_yaw(
                self._latest_rtk_heading, self._theta0_rad)
            yaw_cov = pos_cov  # 有航向，小协方差
            hdg_src = 'rtk'
        else:
            map_yaw = amcl_yaw  # 无航向，用 AMCL 当前 yaw
            yaw_cov = self._cov_no_heading
            hdg_src = 'amcl'

        self._publish_initialpose(gps_x, gps_y, map_yaw, pos_cov, yaw_cov)
        self._last_correction_time = now

        self.get_logger().info(
            f'纠偏: drift={drift:.3f}m > {self._drift_threshold}m | '
            f'GPS map=({gps_x:.2f},{gps_y:.2f}) AMCL=({amcl_x:.2f},{amcl_y:.2f}) | '
            f'yaw_src={hdg_src} cov={pos_cov} quality={quality}')

    # ==================================================================
    #  发布
    # ==================================================================

    def _publish_initialpose(self, x, y, yaw, pos_cov, yaw_cov):
        """发布 /initialpose 纠偏"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame

        msg.pose.pose = Pose(
            position=Point(x=x, y=y, z=0.0),
            orientation=_yaw_to_quat(yaw),
        )

        cov = [0.0] * 36
        cov[0] = pos_cov    # x
        cov[7] = pos_cov    # y
        cov[35] = yaw_cov   # yaw
        msg.pose.covariance = cov

        self._initialpose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RtkPoseMonitor()
        rclpy.spin(node)
    except RuntimeError as e:
        if node:
            node.get_logger().error(str(e))
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
