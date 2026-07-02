#!/usr/bin/env python3
"""
RTK 位姿监控与自动纠偏节点

首次收到外部 /initialpose 时自动标定 RTK 原点（经纬度+朝向），
此后持续对比 RTK 推算位姿与当前位姿（tf map→base_footprint），
偏差超过阈值（默认 1m）时自动补发 /initialpose 纠偏。

零侵入原理:
    AMCL/定位节点订阅 /initialpose（PoseWithCovarianceStamped, frame_id=map），
    收到后重置位姿。本节点只发布该话题，不修改其他节点配置。

坐标转换链（复用 gps_transform.rtk_to_map）:
    RTK lat/lon → UTM (E, N) → 减原点 (E₀,N₀) → 旋转 θ₀ → map (x, y)

标定流程:
    收到第一条外部 /initialpose → 记录当前 RTK 经纬度+航向作为地图原点 → 开始监控纠偏

容错状态机:
    MONITORING → (GPS连续N帧无效) → GPS_LOST → (GPS连续M帧有效) → MONITORING

用法:
    # 直接启动（无需预标定，首次 /initialpose 自动标定）
    ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot

    # 手动启动节点
    ros2 run gps_fusion rtk_pose_monitor.py --ros-args \
        -p ns:=rkbot -p rtk_topic:=/rtk_pvh -p drift_threshold:=1.0
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from sensor_msgs.msg import NavSatFix
from tf2_ros import Buffer, TransformException
from tf2_ros.transform_listener import TransformListener

# 共享转换模块
from gps_fusion.gps_transform import (
    make_utm_transformer,
    latlon_to_utm, rtk_to_map,
    rtk_heading_to_map_yaw, compute_horizontal_accuracy,
    detect_rtk_quality, covariance_for_quality,
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
    """RTK 位姿监控与自动纠偏：首次 /initialpose 自动标定原点，持续监控纠偏"""

    # 状态机
    MONITORING = 'MONITORING'
    GPS_LOST = 'GPS_LOST'
    WAIT_ORIGIN = 'WAIT_ORIGIN'      # 等待首次 /initialpose 标定

    def __init__(self):
        super().__init__('rtk_pose_monitor')

        # ---- 参数 ----
        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('rtk_topic', '/rtk_pvh')
        self.declare_parameter('use_rtk_heading', True)
        self.declare_parameter('ns', '')                          # 命名空间
        self.declare_parameter('map_frame', '')                   # 空则自动推导
        self.declare_parameter('base_frame', '')                  # 空则自动推导
        self.declare_parameter('drift_threshold', 1.0)            # 漂移阈值（m）
        self.declare_parameter('min_correction_interval', 15.0)   # 最小纠偏间隔（s）
        self.declare_parameter('monitor_rate', 2.0)               # 监控频率（Hz）
        self.declare_parameter('gps_loss_threshold', 5)           # 连续N帧无效→GPS_LOST
        self.declare_parameter('gps_recovery_threshold', 3)       # 连续M帧有效→恢复
        self.declare_parameter('gps_stale_timeout', 3.0)          # RTK 数据超时（s），超时视为室内/无信号
        self.declare_parameter('min_accuracy', 5.0)               # RTK 最低精度（m）
        self.declare_parameter('status_report_interval', 30.0)    # 周期性状态汇报间隔（秒）
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
        self._gps_stale_timeout = self.get_parameter('gps_stale_timeout').value
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

        # ---- 地图原点：首次 /initialpose 自动标定，不从文件加载 ----
        self._origin_utm = None
        self._theta0_rad = None
        self._calibrated = False

        # ---- TF2 ----
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- 状态 ----
        self._state = self.WAIT_ORIGIN
        self._gps_fail_count = 0
        self._gps_recover_count = 0
        self._latest_fix = None           # NavSatFix（从 RTK 消息构造）
        self._latest_rtk_heading = None   # RTK 真北航向（度）
        self._last_fix_time = self.get_clock().now()  # 最后收到有效 RTK 数据的时间
        self._last_correction_time = self.get_clock().now() - Duration(seconds=self._min_interval + 1)

        # ---- RTK 诊断（限频日志） ----
        self._rtk_reject_count = 0
        self._rtk_reject_reason = None
        self._rtk_last_diag_time = 0.0

        # ---- 订阅：仅 RTK 原始数据（位置+航向均从此提取） ----
        rtk_topic = self.get_parameter('rtk_topic').value
        if rtk_topic and _HAS_UNI_RTK_PVH:
            self._rtk_sub = self.create_subscription(
                UniRtkPvh, rtk_topic, self._rtk_callback, 10)
            self.get_logger().info(
                f'RTK 数据源: {rtk_topic}（位置+航向均从此提取）')
        else:
            self.get_logger().error(
                'robots_dog_msgs 未安装或 rtk_topic 为空，RTK 输入不可用！')
            self._use_rtk_heading = False

        # ---- 发布 /<ns>/initialpose（纠偏输出） ----
        # 使用相对路径让 ROS2 自动添加命名空间前缀:
        #   ns=""     → /initialpose
        #   ns="rkbot" → /rkbot/initialpose
        # Nav2 AMCL 在命名空间内订阅 /<ns>/initialpose，必须匹配
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10)

        # ---- 订阅 /<ns>/initialpose（标定输入：首次收到即标定原点） ----
        self._initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, 'initialpose',
            self._on_initialpose_callback, 10,
        )

        # ---- 监控定时器 ----
        self._monitor_timer = self.create_timer(
            1.0 / self._monitor_rate, self._monitor_loop)

        self._last_status_time = self.get_clock().now()
        self._status_interval = self.get_parameter('status_report_interval').value
        self._status_timer = self.create_timer(1.0, self._status_report)

        self.get_logger().info(
            f'RTK 位姿监控节点已启动 | 等待首次 /initialpose 自动标定原点\n'
            f'  drift_threshold={self._drift_threshold}m, '
            f'min_interval={self._min_interval}s, rate={self._monitor_rate}Hz\n'
            f'  map_frame={self._map_frame}, base_frame={self._base_frame}')

    # ==================================================================
    #  回调
    # ==================================================================

    def _rtk_pos_valid_diag(self, bestnav, heading) -> bool:
        """检查 RTK 位置是否有效（以 heading_type 为统一质量指标），拒绝时输出诊断日志（限频）。"""
        try:
            p_sol = bestnav.p_sol_status
            heading_type = heading.heading_type
            lat = bestnav.latitude_deg
            lon = bestnav.longitude_deg
        except AttributeError as e:
            self._rtk_diag_log(
                'message_field_error',
                f'无法访问 bestnav/heading 字段: {e}。消息类型是否匹配？'
            )
            return False

        if p_sol not in (0, 2):
            self._rtk_diag_log(
                'sol_status_invalid',
                f'p_sol_status={p_sol} (期望0或2), heading_type={heading_type}'
            )
            return False

        if heading_type not in (16, 17, 34, 50):
            svs = getattr(bestnav, 'svs_num', '?')
            soln = getattr(bestnav, 'soln_svs_num', '?')
            self._rtk_diag_log(
                'position_not_converged',
                f'heading_type={heading_type} (期望16/17/34/50), '
                f'可见星={svs}, 解算星={soln}'
            )
            return False

        if math.isnan(lat) or math.isnan(lon):
            self._rtk_diag_log(
                'nan_coordinate',
                f'lat={lat}, lon={lon}'
            )
            return False

        return True

    def _rtk_diag_log(self, reason: str, detail: str):
        """限频输出 RTK 数据被拒绝的诊断日志。

        与 gps_preprocessor._rtk_diag_log 相同的限频策略：
        - 原因变化时立即输出
        - 每隔 1 秒输出一次
        - 每 60 次累计也输出
        """
        now = self.get_clock().now().nanoseconds * 1e-9
        self._rtk_reject_count += 1
        changed = (reason != self._rtk_reject_reason)
        if changed:
            self._rtk_reject_reason = reason
            self._rtk_reject_count = 1
        if changed or (now - self._rtk_last_diag_time) >= 1.0 or self._rtk_reject_count % 60 == 0:
            self._rtk_last_diag_time = now
            self.get_logger().warn(
                f'RTK 数据被过滤 [{self._rtk_reject_count}次]: {reason} — {detail}'
            )

    def _rtk_callback(self, msg):
        """从 RTK 原始消息提取位置（转 NavSatFix）和航向。

        UniRtkPvh → NavSatFix 映射（与 gps_preprocessor._rtk_to_navsat_fix 一致）：
          - bestnav.latitude_deg/longitude_deg/altitude_m → NavSatFix
          - heading_type 50→status=4(RTK_FIX), 34→5(RTK_FLOAT), 17→2(DGPS), 16→2(单点)
          - lat_std/lon_std/hgt_std → position_covariance
          - heading.heading_deg → _latest_rtk_heading（需 heading_type ∈ {16,17,34,50}）
        """
        bestnav = msg.bestnav

        # ---- 室内检测：无卫星信号 → 静默跳过 ----
        try:
            svs_num = bestnav.svs_num
        except AttributeError:
            svs_num = -1

        if bestnav.pos_type == 0 and svs_num == 0:
            return  # 室内无信号，静默

        heading = msg.heading

        # ---- 位置收敛检查（以 heading_type 为统一质量指标） ----
        if not self._rtk_pos_valid_diag(bestnav, heading):
            return

        # ---- 构造 NavSatFix（供监控循环使用） ----
        navsat = NavSatFix()
        navsat.header = bestnav.header if bestnav.header.stamp.sec else msg.header
        navsat.header.frame_id = 'gps'
        navsat.latitude = bestnav.latitude_deg
        navsat.longitude = bestnav.longitude_deg
        navsat.altitude = bestnav.altitude_m if not math.isnan(bestnav.altitude_m) else 0.0

        # heading_type → NavSatFix status（统一质量指标）
        if heading.heading_type == 50:
            navsat.status.status = 4   # RTK_FIX
        elif heading.heading_type == 34:
            navsat.status.status = 5   # RTK_FLOAT
        else:  # 17=DGPS, 16=单点
            navsat.status.status = 2   # DGPS/GBAS

        navsat.position_covariance_type = 1  # COVARIANCE_TYPE_KNOWN
        lat_var = bestnav.lat_std ** 2 if not math.isnan(bestnav.lat_std) else 1.0
        lon_var = bestnav.lon_std ** 2 if not math.isnan(bestnav.lon_std) else 1.0
        hgt_var = bestnav.hgt_std ** 2 if not math.isnan(bestnav.hgt_std) else 4.0
        navsat.position_covariance = [
            lat_var, 0.0, 0.0,
            0.0, lon_var, 0.0,
            0.0, 0.0, hgt_var,
        ]

        # 精度门槛检查
        h_acc = compute_horizontal_accuracy(navsat)
        if h_acc > self._min_accuracy:
            self._rtk_diag_log(
                'low_accuracy',
                f'h_acc={h_acc:.3f}m > threshold={self._min_accuracy}m '
                f'(heading_type={heading.heading_type})'
            )
            return

        # ---- 首次收到有效 RTK 数据时输出日志 ----
        if self._latest_fix is None:
            self.get_logger().info(
                f'收到首帧有效 RTK 数据: heading_type={heading.heading_type}, '
                f'lat={bestnav.latitude_deg:.8f}, lon={bestnav.longitude_deg:.8f}, '
                f'h_acc={h_acc:.3f}m'
            )
            self._rtk_reject_count = 0
            self._rtk_reject_reason = None

        self._latest_fix = navsat
        self._last_fix_time = self.get_clock().now()

        # ---- 航向提取 ----
        try:
            if heading.heading_type in (16, 17, 34, 50) and heading.sol_status in (0, 2):
                self._latest_rtk_heading = float(heading.heading_deg)
        except Exception:
            pass

    def _on_initialpose_callback(self, msg: PoseWithCovarianceStamped):
        """首次收到外部 /initialpose → 记录 RTK 位置作为地图原点，开始纠偏。

        后续再收到的 /initialpose（非本节点发布）会被忽略——原点只标定一次。
        标定公式与 rtk_initial_pose.py / calibrate_map_origin.py 一致。
        """
        if self._calibrated:
            return   # 已标定，忽略后续 /initialpose

        if self._latest_fix is None:
            self.get_logger().info(
                '收到 /initialpose 但尚未收到 RTK GPS 数据，'
                '无法标定，等待 GPS 信号...')
            return

        fix = self._latest_fix
        map_x = msg.pose.pose.position.x
        map_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        map_yaw = math.atan2(siny, cosy)

        e, n = latlon_to_utm(self._to_utm, fix.longitude, fix.latitude)

        if self._latest_rtk_heading is not None:
            theta0 = math.radians(self._latest_rtk_heading) - map_yaw
        else:
            theta0 = 0.0
            self.get_logger().warn('无 RTK 航向，使用 θ₀=0（地图与真北对齐）')

        cos_t0 = math.cos(theta0)
        sin_t0 = math.sin(theta0)
        e0 = e - map_x * cos_t0 + map_y * sin_t0
        n0 = n - map_x * sin_t0 - map_y * cos_t0
        a0 = fix.altitude if not math.isnan(fix.altitude) else 0.0

        self._origin_utm = (e0, n0, a0)
        self._theta0_rad = theta0
        self._calibrated = True
        self._state = self.MONITORING

        quality = detect_rtk_quality(fix)
        hdg_str = f'{self._latest_rtk_heading:.1f}°' if self._latest_rtk_heading else 'N/A'

        self.get_logger().info(
            f'========== 自动标定完成 ==========\n'
            f'  地图位姿: ({map_x:.2f}, {map_y:.2f}, {math.degrees(map_yaw):.1f}°)\n'
            f'  RTK: ({fix.latitude:.12f}, {fix.longitude:.12f}) heading={hdg_str}\n'
            f'  RTK 质量: {quality}\n'
            f'  原点 UTM: ({e0:.2f}, {n0:.2f}) heading={math.degrees(theta0):.4f}°\n'
            f'  开始纠偏监控 (threshold={self._drift_threshold}m)\n'
            f'===================================')

    # ==================================================================
    #  监控主循环
    # ==================================================================

    def _monitor_loop(self):
        """监控主循环：比对 GPS 位姿与当前位姿（tf）"""
        # 等待 /initialpose 标定完成
        if self._state == self.WAIT_ORIGIN:
            return

        # ---- RTK 数据超时检测：超过 N 秒无新数据 → 视为室内/无信号 ----
        now = self.get_clock().now()
        stale_duration = (now - self._last_fix_time).nanoseconds * 1e-9
        if self._latest_fix is not None and stale_duration > self._gps_stale_timeout:
            self._latest_fix = None
            self.get_logger().warn(
                f'RTK 数据超时 {stale_duration:.1f}s > {self._gps_stale_timeout}s，'
                f'视为室内/无信号，进入 GPS_LOST')

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

        if self._origin_utm is None:
            return

        if self._latest_fix is None:
            return

        # ---- 查询 AMCL 位姿（map→base_footprint TF） ----
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
        except TransformException as ex:
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

    def _status_report(self):
        """周期性状态汇报：当前位置、GPS偏差、运行状态"""
        now = self.get_clock().now()
        elapsed = (now - self._last_status_time).nanoseconds * 1e-9
        if elapsed < self._status_interval:
            return
        self._last_status_time = now

        origin_status = 'WAIT' if self._origin_utm is None else 'OK'

        if self._latest_fix is None:
            self.get_logger().info(
                f'[状态] GPS=无信号 | AMCL=-- | state={self._state} '
                f'origin={origin_status} indoor={self._state == self.GPS_LOST}')
            return

        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.3))
            amcl_x = tf_msg.transform.translation.x
            amcl_y = tf_msg.transform.translation.y
        except TransformException:
            amcl_x = float('nan')
            amcl_y = float('nan')

        try:
            gps_x, gps_y = rtk_to_map(
                self._to_utm,
                self._latest_fix.longitude,
                self._latest_fix.latitude,
                self._origin_utm, self._theta0_rad,
            )
        except Exception:
            gps_x = float('nan')
            gps_y = float('nan')

        quality = detect_rtk_quality(self._latest_fix)
        drift = math.sqrt((gps_x - amcl_x) ** 2 + (gps_y - amcl_y) ** 2) if not math.isnan(amcl_x) else float('nan')

        self.get_logger().info(
            f'[状态] GPS=({gps_x:.2f},{gps_y:.2f}) | '
            f'AMCL=({amcl_x:.2f},{amcl_y:.2f}) | '
            f'drift={drift:.2f}m | '
            f'Q={quality} | state={self._state} | origin={origin_status}')


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
