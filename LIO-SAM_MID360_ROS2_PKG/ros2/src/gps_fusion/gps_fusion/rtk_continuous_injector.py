#!/usr/bin/env python3
"""连续平滑 GPS/RTK 注入节点（去建图依赖）。

用首次获取到的绝对话题 /initialpose（室外默认初始点位）作为机器人 map 位姿的
权威锚点，结合首个有效 RTK/GPS fix 建立 map↔经纬度↔yaw 坐标体系，之后持续、
平滑地推算机器人 map 位姿并发布 /<ns>/initialpose 纠偏（协方差自适应）。

设计要点：
- 不读 AMCL、不读 LIO-SAM、不依赖 map_gps_origin.yaml、不侵入任何其它模块。
- RTK 是绝对位姿唯一权威；LIO 里程计（/Odometry）仅做运动触发 / 断信号桥接 / yaw 平滑。
- 丝滑降级：逐帧按"当前帧有无可用航向"决定发布值；源切换只改位置协方差/阈值，
  绝不改 yaw，避免 AMCL 抖动。无航向时仅发位置、yaw 放大协方差交 AMCL 自收敛。

状态机：WAIT_ANCHOR → COLLECTING → INJECTING
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import (
    PoseWithCovarianceStamped, Pose, Point, Quaternion, PoseWithCovariance,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix

from gps_fusion.gps_transform import (
    make_utm_transformer, latlon_to_utm, rtk_to_map_anchored,
    compute_horizontal_accuracy, covariance_for_quality,
    circular_mean_heading, odom_pose_delta, decide_reinject,
    is_rtk_heading_valid, pos_type_to_quality, select_anchor_theta0,
)

try:
    from robots_dog_msgs.msg import UniRtkPvh
    _HAS_UNI_RTK_PVH = True
except ImportError:
    UniRtkPvh = None
    _HAS_UNI_RTK_PVH = False


WAIT_ANCHOR = 0
COLLECTING = 1
INJECTING = 2


def _yaw_to_quat(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


class RtkContinuousInjector(Node):
    """用首次 /initialpose 锚定，连续平滑注入 /initialpose 纠偏。"""

    def __init__(self):
        super().__init__('rtk_continuous_injector')
        self._declare_params()
        self._to_utm = make_utm_transformer(self._utm_zone)

        # 定位输入（均存为 NavSatFix，统一接口）
        self._latest_rtk_fix = None
        self._latest_rtk_quality = 'GPS'
        self._latest_rtk_heading = None
        self._last_rtk_time = None
        self._latest_gps_fix = None
        self._last_gps_time = None

        # 状态机
        self._state = WAIT_ANCHOR
        self._anchor_initialpose = None      # (mx0, my0, maw0)
        self._calib_samples = []             # [(e, n, heading_or_None)]
        self._calib_start_time = None
        self._anchor = None                  # (e0, n0, mx0, my0, theta0)

        # 注入跟踪
        self._last_pub_x = None
        self._last_pub_y = None
        self._last_pub_yaw = None
        self._last_source = None
        self._last_rtk_lost_time = None

        # LIO 累计（自上次发布）
        self._lio_dx = 0.0
        self._lio_dy = 0.0
        self._lio_dyaw = 0.0
        self._prev_odom_pose = None

        self._init_subscriptions()
        self._inject_timer = self.create_timer(
            1.0 / self._inject_rate, self._inject_loop)

        self.get_logger().info(
            '连续注入节点已启动 | 等待首次 /initialpose 锚定原点\n'
            f'  RTK 权威 / LIO 仅辅助(use_lio_deadreckon={self._use_lio_deadreckon})\n'
            f'  注入频率={self._inject_rate}Hz, 重发位移阈={self._reinject_motion_margin}m, '
            f'重发yaw阈={math.degrees(self._reinject_yaw_margin_rad):.1f}°\n'
            f'  map_frame={self._map_frame}, base_frame={self._base_frame}')

    # ==================================================================
    #  参数
    # ==================================================================
    def _declare_params(self):
        p = self.declare_parameter
        self._utm_zone = p('utm_zone', 50).value
        self._rtk_topic = p('rtk_topic', '/rtk_pvh').value
        self._gps_topic = p('gps_topic', '/fix').value
        self._lio_odom_topic = p('lio_odom_topic', '/Odometry').value
        self._map_frame = p('map_frame', 'map').value
        self._base_frame = p('base_frame', 'base_footprint').value
        self._inject_rate = float(p('inject_rate', 0.5).value)
        self._calib_sample_count = int(p('calib_sample_count', 5).value)
        self._max_collect_time = float(p('max_collect_time', 30.0).value)
        self._reinject_motion_margin = float(p('reinject_motion_margin', 0.3).value)
        self._reinject_yaw_margin_rad = math.radians(
            float(p('reinject_yaw_margin_deg', 5.0).value))
        self._enable_gps_fallback = bool(p('enable_gps_fallback', True).value)
        self._use_rtk_heading = bool(p('use_rtk_heading', True).value)
        self._use_lio_deadreckon = bool(p('use_lio_deadreckon', True).value)
        self._gps_stale_timeout = float(p('gps_stale_timeout', 3.0).value)
        self._min_accuracy = float(p('min_accuracy', 1.0).value)
        self._gps_min_accuracy = float(p('gps_min_accuracy', 10.0).value)
        self._gps_jump_threshold = float(p('gps_jump_threshold', 5.0).value)
        self._max_diff_age = float(p('max_diff_age', 5.0).value)
        self._max_heading_std = float(p('max_heading_std', 5.0).value)
        self._cov_rtk_fix = float(p('cov_rtk_fix', 0.01).value)
        self._cov_rtk_float = float(p('cov_rtk_float', 0.1).value)
        self._cov_dgps = float(p('cov_dgps', 1.0).value)
        self._cov_gps = float(p('cov_gps', 25.0).value)
        self._cov_no_heading = float(p('cov_no_heading', 0.5).value)
        self._cov_yaw_rtk = float(p('cov_yaw_rtk', 0.05).value)
        self._correction_cov_base = float(p('correction_cov_base', 0.25).value)
        self._lio_bridge_max_time = float(p('lio_bridge_max_time', 5.0).value)
        self._lio_bridge_max_dist = float(p('lio_bridge_max_dist', 10.0).value)

    def _init_subscriptions(self):
        if self._rtk_topic and _HAS_UNI_RTK_PVH:
            self.create_subscription(
                UniRtkPvh, self._rtk_topic, self._rtk_callback, 10)
            self.get_logger().info(f'RTK 数据源: {self._rtk_topic}')
        else:
            self.get_logger().error('UniRtkPvh 不可用或 rtk_topic 为空，RTK 输入失效')
            self._use_rtk_heading = False

        if self._enable_gps_fallback and self._gps_topic:
            self.create_subscription(
                NavSatFix, self._gps_topic, self._gps_callback, 10)
            self.get_logger().info(f'GPS fallback: {self._gps_topic}')

        if self._use_lio_deadreckon and self._lio_odom_topic:
            self.create_subscription(
                Odometry, self._lio_odom_topic, self._lio_odom_callback, 10)
            self.get_logger().info(f'LIO 里程计: {self._lio_odom_topic}')

        # 绝对话题 /initialpose 做锚点（nav2_web_control 始终发全局）
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose',
            self._on_initialpose_callback, 10)
        # 相对话题 initialpose → ROS2 自动加 ns 前缀，匹配 AMCL 订阅
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10)

    # ==================================================================
    #  回调
    # ==================================================================
    def _rtk_callback(self, msg):
        bestnav = msg.bestnav
        heading = msg.heading
        if bestnav.pos_type == 0 and getattr(bestnav, 'svs_num', -1) == 0:
            return  # 室内无信号，静默
        if not self._rtk_pos_valid(bestnav):
            return

        navsat = self._build_navsat(bestnav)
        h_acc = compute_horizontal_accuracy(navsat)
        if h_acc > self._min_accuracy:
            return

        self._latest_rtk_fix = navsat
        self._latest_rtk_quality = pos_type_to_quality(bestnav.pos_type)
        self._last_rtk_time = self.get_clock().now()

        hdg = None
        if (self._use_rtk_heading and
                is_rtk_heading_valid(heading.heading_type, heading.sol_status,
                                     getattr(heading, 'heading_std', 0.0),
                                     self._max_heading_std)):
            hdg = float(heading.heading_deg)
        self._latest_rtk_heading = hdg

        self._maybe_collect(navsat, 'rtk', hdg)

    def _gps_callback(self, msg: NavSatFix):
        if msg.status.status < 0:
            return
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return
        h_acc = compute_horizontal_accuracy(msg)
        if h_acc != float('inf') and h_acc > self._gps_min_accuracy:
            return
        self._latest_gps_fix = msg
        self._last_gps_time = self.get_clock().now()
        self._maybe_collect(msg, 'gps', None)

    def _lio_odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)
        cur = (pose.position.x, pose.position.y, yaw)
        if self._prev_odom_pose is None:
            self._prev_odom_pose = cur
            return
        if self._state == INJECTING:
            d = odom_pose_delta(self._prev_odom_pose, cur)
            self._lio_dx += d[0]
            self._lio_dy += d[1]
            self._lio_dyaw += d[2]
        self._prev_odom_pose = cur

    def _on_initialpose_callback(self, msg: PoseWithCovarianceStamped):
        if self._state != WAIT_ANCHOR:
            return  # 原点只标定一次
        q = msg.pose.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        self._anchor_initialpose = (
            msg.pose.pose.position.x, msg.pose.pose.position.y, yaw)
        self._state = COLLECTING
        self._calib_samples = []
        self._calib_start_time = self.get_clock().now()
        self.get_logger().info('收到 /initialpose，启动多帧锚点采集')

    # ==================================================================
    #  锚点
    # ==================================================================
    def _maybe_collect(self, fix: NavSatFix, source: str, heading):
        if self._state != COLLECTING or self._anchor_initialpose is None:
            return
        e, n = latlon_to_utm(self._to_utm, fix.longitude, fix.latitude)
        self._calib_samples.append((e, n, heading))
        if len(self._calib_samples) >= self._calib_sample_count:
            self._finalize_anchor()
            return
        # 超时则 best-effort 收尾（无航向时仍建立位置锚点）
        if (self.get_clock().now() - self._calib_start_time).nanoseconds * 1e-9 \
                > self._max_collect_time:
            self._finalize_anchor()

    def _finalize_anchor(self):
        e0 = sum(s[0] for s in self._calib_samples) / len(self._calib_samples)
        n0 = sum(s[1] for s in self._calib_samples) / len(self._calib_samples)
        headings = [s[2] for s in self._calib_samples if s[2] is not None]
        mx0, my0, maw0 = self._anchor_initialpose
        if headings:
            theta0 = select_anchor_theta0(maw0, circular_mean_heading(headings))
        else:
            theta0 = select_anchor_theta0(maw0, None)
            self.get_logger().warn(
                '锚点期无有效 RTK 航向，θ0 退化为 initialpose yaw，'
                '位置纠偏可能旋转偏差，待航向可用后请重新锚定')
        self._anchor = (e0, n0, mx0, my0, theta0)
        self._state = INJECTING
        self._prev_odom_pose = None
        self._reset_lio()
        self.get_logger().info(
            f'锚点建立完成: (e0={e0:.3f}, n0={n0:.3f}, '
            f'map=({mx0:.3f},{my0:.3f}), θ0={math.degrees(theta0):.2f}°), '
            f'heading={"有" if headings else "无"}')

    # ==================================================================
    #  注入
    # ==================================================================
    def _inject_loop(self):
        if self._state != INJECTING or self._anchor is None:
            return
        now = self.get_clock().now()
        fix, source, quality, heading = self._select_active_fix(now)

        if fix is None:
            if self._use_lio_deadreckon:
                self._maybe_publish_bridge(now)
            return

        map_x, map_y, map_yaw = rtk_to_map_anchored(
            self._to_utm, fix.longitude, fix.latitude, self._anchor, heading)
        if map_yaw is None:
            map_yaw = self._last_pub_yaw if self._last_pub_yaw is not None \
                else self._anchor_initialpose[2]

        moved = math.hypot(self._lio_dx, self._lio_dy)
        turned = abs(self._lio_dyaw)
        need = decide_reinject(
            moved, turned, self._reinject_motion_margin,
            self._reinject_yaw_margin_rad) or (source != self._last_source)
        if self._last_pub_x is None or need:
            self._publish(map_x, map_y, map_yaw, source, quality, heading is not None)

    def _select_active_fix(self, now):
        rtk_fresh = (self._latest_rtk_fix is not None and
                     self._last_rtk_time is not None and
                     (now - self._last_rtk_time).nanoseconds * 1e-9
                     <= self._gps_stale_timeout)
        if rtk_fresh:
            return self._latest_rtk_fix, 'rtk', self._latest_rtk_quality, \
                self._latest_rtk_heading
        gps_fresh = (self._enable_gps_fallback and
                     self._latest_gps_fix is not None and
                     self._last_gps_time is not None and
                     (now - self._last_gps_time).nanoseconds * 1e-9
                     <= self._gps_stale_timeout)
        if gps_fresh:
            return self._latest_gps_fix, 'gps', 'GPS', None
        return None, None, None, None

    def _maybe_publish_bridge(self, now):
        if self._last_pub_x is None:
            return
        if self._last_rtk_lost_time is None:
            self._last_rtk_lost_time = now
        lost = (now - self._last_rtk_lost_time).nanoseconds * 1e-9
        dist = math.hypot(self._lio_dx, self._lio_dy)
        if lost > self._lio_bridge_max_time or dist > self._lio_bridge_max_dist:
            return  # 超出桥接限幅，停止注入避免漂移累积
        self._publish(self._last_pub_x + self._lio_dx,
                      self._last_pub_y + self._lio_dy,
                      self._last_pub_yaw, 'lio_bridge', 'GPS', False)

    def _publish(self, x, y, yaw, source, quality, has_heading):
        pos_cov = max(covariance_for_quality(
            quality, self._cov_rtk_fix, self._cov_rtk_float,
            self._cov_dgps, self._cov_gps), self._correction_cov_base)
        yaw_cov = self._cov_yaw_rtk if has_heading else self._cov_no_heading
        cov = [pos_cov, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, pos_cov, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, yaw_cov, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw_cov]
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._map_frame
        msg.pose = PoseWithCovariance(
            pose=Pose(position=Point(x=x, y=y, z=0.0),
                      orientation=_yaw_to_quat(yaw)),
            covariance=cov)
        self._initialpose_pub.publish(msg)
        self._last_pub_x, self._last_pub_y, self._last_pub_yaw = x, y, yaw
        self._last_source = source
        self._last_rtk_lost_time = None
        self._reset_lio()
        self.get_logger().info(
            f'发布 /initialpose [{source}] x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.2f}° '
            f'(pos_cov={pos_cov:.3f}, yaw_cov={yaw_cov:.3f})')

    def _reset_lio(self):
        self._lio_dx = self._lio_dy = self._lio_dyaw = 0.0

    # ==================================================================
    #  RTK 位置有效性（复刻 rtk_pose_monitor 门禁）
    # ==================================================================
    def _rtk_pos_valid(self, bestnav) -> bool:
        if getattr(bestnav, 'p_sol_status', 0) not in (0, 2):
            return False
        if bestnav.pos_type not in (16, 17, 34, 50):
            return False
        if math.isnan(bestnav.latitude_deg) or math.isnan(bestnav.longitude_deg):
            return False
        diff_age = getattr(bestnav, 'diff_age_s', 0.0)
        if diff_age > self._max_diff_age:
            return False
        return True

    def _build_navsat(self, bestnav) -> NavSatFix:
        navsat = NavSatFix()
        navsat.header.frame_id = 'gps'
        navsat.latitude = bestnav.latitude_deg
        navsat.longitude = bestnav.longitude_deg
        navsat.altitude = bestnav.altitude_m if not math.isnan(
            bestnav.altitude_m) else 0.0
        navsat.status.status = {
            50: 4, 34: 5}.get(bestnav.pos_type, 2)
        navsat.position_covariance_type = 1
        lat_var = bestnav.lat_std ** 2 if not math.isnan(
            bestnav.lat_std) else 1.0
        lon_var = bestnav.lon_std ** 2 if not math.isnan(
            bestnav.lon_std) else 1.0
        hgt_var = bestnav.hgt_std ** 2 if not math.isnan(
            bestnav.hgt_std) else 4.0
        navsat.position_covariance = [
            lat_var, 0.0, 0.0, 0.0, lon_var, 0.0, 0.0, 0.0, hgt_var]
        return navsat


def main(args=None):
    rclpy.init(args=args)
    node = RtkContinuousInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
