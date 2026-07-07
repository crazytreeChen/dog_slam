#!/usr/bin/env python3
"""
RTK 位姿监控与自动纠偏节点

首次收到外部 /initialpose 时自动建立 GPS/RTK 到 map 的坐标关系，
此后持续对比 GPS/RTK 推算位置与当前位姿（tf map→base_footprint），
偏差超过对应数据源阈值时自动补发 /initialpose 纠偏。

零侵入原理:
    AMCL/定位节点订阅 /initialpose（PoseWithCovarianceStamped, frame_id=map），
    收到后重置位姿。本节点只发布该话题，不修改其他节点配置。

坐标转换链（复用 gps_transform.rtk_to_map）:
    GPS/RTK lat/lon → UTM (E, N) → 减原点 (E₀,N₀) → 旋转 θ₀ → map (x, y)

标定方式（二选一）:
    1. 自动：收到第一条外部 /initialpose → 记录当前 GPS/RTK 位置关系 → 开始监控纠偏
    2. 主动：ros2 service call /start_gps_origin_calibration std_srvs/srv/Trigger
       → 以当前 AMCL+GPS 同步数据多帧标定地图原点 (E₀, N₀, θ₀)

航向策略:
    RTK 双天线可提供可信 heading，用于 RTK 模式 yaw 纠偏。
    普通 GPS fallback 不纠 yaw，只沿用当前 AMCL yaw；GPS 无航向时若需建立
    GPS→map 旋转关系，会等待机器人移动一段距离后用 GPS 位移方向与 map 位移方向估计。

容错状态机:
    MONITORING → (GPS连续N帧无效) → GPS_LOST → (GPS连续M帧有效) → MONITORING

用法:
    # 直接启动（无需预标定，首次 /initialpose 自动标定）
    ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot

    # 主动标定原点（以当前 AMCL+GPS 位置多帧标定）
    ros2 service call /start_gps_origin_calibration std_srvs/srv/Trigger
    # 带 namespace:
    ros2 service call /rkbot/start_gps_origin_calibration std_srvs/srv/Trigger

    # 手动启动节点
    ros2 run gps_fusion rtk_pose_monitor.py --ros-args \
        -p ns:=rkbot -p rtk_topic:=/rtk_pvh \
        -p rtk_drift_threshold:=1.0 -p gps_drift_threshold:=8.0
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Trigger
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

    SOURCE_RTK = 'rtk'
    SOURCE_GPS = 'gps'

    # 状态机
    MONITORING = 'MONITORING'
    GPS_LOST = 'GPS_LOST'
    WAIT_ORIGIN = 'WAIT_ORIGIN'      # 等待首次 /initialpose 标定
    COLLECTING = 'COLLECTING'        # 多帧采样标定中（采集N帧同步数据后计算原点）

    def __init__(self):
        super().__init__('rtk_pose_monitor')

        # ---- 参数 ----
        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('rtk_topic', '/rtk_pvh')
        self.declare_parameter('gps_topic', '/fix')
        self.declare_parameter('enable_gps_fallback', True)
        self.declare_parameter('use_rtk_heading', True)
        self.declare_parameter('ns', '')                          # 命名空间
        self.declare_parameter('map_frame', '')                   # 空则自动推导
        self.declare_parameter('base_frame', '')                  # 空则自动推导
        self.declare_parameter('drift_threshold', float('nan'))   # 旧参数兼容：映射到RTK阈值
        self.declare_parameter('rtk_drift_threshold', 1.0)        # RTK 纠偏阈值（m）
        self.declare_parameter('gps_drift_threshold', 8.0)        # GPS fallback 纠偏阈值（m）
        self.declare_parameter('min_correction_interval', 15.0)   # 最小纠偏间隔（s）
        self.declare_parameter('monitor_rate', 2.0)               # 监控频率（Hz）
        self.declare_parameter('gps_loss_threshold', 5)           # 连续N帧无效→GPS_LOST
        self.declare_parameter('gps_recovery_threshold', 3)       # 连续M帧有效→恢复
        self.declare_parameter('gps_stale_timeout', 3.0)          # RTK 数据超时（s），超时视为室内/无信号
        self.declare_parameter('min_accuracy', 5.0)               # RTK 最低精度（m）
        self.declare_parameter('gps_min_accuracy', 10.0)          # GPS fallback 最低精度（m）
        self.declare_parameter('status_report_interval', 30.0)    # 周期性状态汇报间隔（秒）
        self.declare_parameter('gps_jump_threshold', 5.0)         # GPS 位置跳变速度阈值（m/s），超过则视为抖动
        self.declare_parameter('cov_rtk_fix', 0.01)
        self.declare_parameter('cov_rtk_float', 0.1)
        self.declare_parameter('cov_dgps', 1.0)
        self.declare_parameter('cov_gps', 25.0)                   # GPS fallback 位置协方差（m²）
        self.declare_parameter('cov_no_heading', 0.5)             # 无航向时 yaw 协方差

        # ---- RTK 数据质量门禁 ----
        self.declare_parameter('max_diff_age', 5.0)               # 最大差分龄期（秒），超时视为差分修正过时
        self.declare_parameter('max_heading_std', 5.0)            # 最大航向标准差（度），超时丢弃RTK航向用AMCL兜底

        # ---- 多条件纠偏触发（解决"缺少契机"问题） ----
        self.declare_parameter('amcl_pose_topic', 'amcl_pose')    # AMCL 位姿（含协方差）
        self.declare_parameter('amcl_cov_threshold', 2.0)         # AMCL xy协方差膨胀阈值，超过则触发纠偏
        self.declare_parameter('cumulative_drift_window', 30.0)   # 累积偏差滑动窗口（秒）
        self.declare_parameter('cumulative_drift_threshold', 3.0) # 窗口内GPS/AMCL位移偏差阈值（m）
        self.declare_parameter('stationary_speed', 0.15)          # 静止判定速度（m/s）
        self.declare_parameter('stationary_drift_threshold', 1.0) # 静止时微调阈值（m），比运动时更低

        # ---- LIO 里程计（高频精确速度+方向，用于轨迹推算） ----
        self.declare_parameter('lio_odom_topic', '/Odometry')     # LIO 里程计话题
        self.declare_parameter('predict_window', 5.0)              # 前向预测窗口（秒）
        self.declare_parameter('predict_drift_threshold', 1.5)     # 预测位置 vs 实际位置偏差阈值（m）

        # ---- 多帧标定（提高地图原点精度，避免单帧标定的随机误差） ----
        self.declare_parameter('calib_sample_count', 10)           # 标定采集帧数
        self.declare_parameter('gps_calib_min_motion', 2.0)        # GPS无航向标定所需最小位移（m）
        self.declare_parameter('gps_calib_max_samples', 80)        # GPS无航向标定最多保留样本数
        self.declare_parameter('calib_max_std_position', 0.5)      # 标定位置标准差上限（m），超限告警
        self.declare_parameter('calib_max_std_heading', 2.0)       # 标定朝向标准差上限（°），超限告警

        # ---- 纠偏协方差策略（避免过度信任GPS推算位置） ----
        # 纠偏时发布的初始位姿协方差 = max(gps精度协方差, 标定误差², correction_cov_base)
        # 这样 AMCL 粒子云有一定散布空间，可通过 scan matching 自我精化
        self.declare_parameter('correction_cov_base', 0.25)      # 纠偏协方差下限（m²）

        self._utm_zone = self.get_parameter('utm_zone').value
        self._enable_gps_fallback = self.get_parameter('enable_gps_fallback').value
        self._use_rtk_heading = self.get_parameter('use_rtk_heading').value
        legacy_drift_threshold = self.get_parameter('drift_threshold').value
        self._rtk_drift_threshold = self.get_parameter('rtk_drift_threshold').value
        if not math.isnan(legacy_drift_threshold):
            self._rtk_drift_threshold = legacy_drift_threshold
            self.get_logger().warn(
                '参数 drift_threshold 已废弃，请改用 rtk_drift_threshold；'
                '本次已按旧参数覆盖 RTK 纠偏阈值')
        self._gps_drift_threshold = self.get_parameter('gps_drift_threshold').value
        self._min_interval = self.get_parameter('min_correction_interval').value
        self._monitor_rate = self.get_parameter('monitor_rate').value
        self._gps_loss_threshold = self.get_parameter('gps_loss_threshold').value
        self._gps_recovery_threshold = self.get_parameter('gps_recovery_threshold').value
        self._gps_stale_timeout = self.get_parameter('gps_stale_timeout').value
        self._min_accuracy = self.get_parameter('min_accuracy').value
        self._gps_min_accuracy = self.get_parameter('gps_min_accuracy').value
        self._gps_jump_threshold = self.get_parameter('gps_jump_threshold').value
        self._cov_gps = self.get_parameter('cov_gps').value
        self._cov_no_heading = self.get_parameter('cov_no_heading').value

        # ---- RTK 质量门禁 ----
        self._max_diff_age = self.get_parameter('max_diff_age').value
        self._max_heading_std = self.get_parameter('max_heading_std').value

        # ---- 多条件纠偏参数 ----
        self._amcl_cov_threshold = self.get_parameter('amcl_cov_threshold').value
        self._cumulative_window = self.get_parameter('cumulative_drift_window').value
        self._cumulative_threshold = self.get_parameter('cumulative_drift_threshold').value
        self._stationary_speed = self.get_parameter('stationary_speed').value
        self._stationary_drift_threshold = self.get_parameter('stationary_drift_threshold').value

        # ---- LIO 里程计参数 ----
        self._predict_window = self.get_parameter('predict_window').value
        self._predict_drift_threshold = self.get_parameter('predict_drift_threshold').value

        # ---- 多帧标定参数 ----
        self._calib_sample_count = self.get_parameter('calib_sample_count').value
        self._gps_calib_min_motion = self.get_parameter('gps_calib_min_motion').value
        self._gps_calib_max_samples = self.get_parameter('gps_calib_max_samples').value
        self._calib_max_std_position = self.get_parameter('calib_max_std_position').value
        self._calib_max_std_heading = self.get_parameter('calib_max_std_heading').value
        self._correction_cov_base = self.get_parameter('correction_cov_base').value

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

        # ---- 多帧标定状态（提高原点精度） ----
        self._calib_samples = []           # 采集的同步帧: [(amcl_x,amcl_y,amcl_yaw,utm_e,utm_n,heading_deg,source), ...]
        self._calib_source = None
        self._calib_position_std = 0.0     # 标定位置标准差（m），用于纠偏协方差膨胀
        self._calib_heading_std = 0.0      # 标定朝向标准差（°）

        # ---- TF2 ----
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ---- 状态 ----
        self._state = self.WAIT_ORIGIN
        self._gps_fail_count = 0
        self._gps_recover_count = 0
        self._latest_fix = None           # 当前选中的 NavSatFix（RTK 优先，GPS fallback）
        self._latest_fix_source = None
        self._latest_rtk_fix = None
        self._latest_gps_fix = None
        self._latest_rtk_heading = None   # RTK 真北航向（度）
        self._last_fix_time = None        # 最后收到当前有效定位数据的时间
        self._last_rtk_time = None
        self._last_gps_time = None
        self._last_correction_time = self.get_clock().now() - Duration(seconds=self._min_interval + 1)

        # ---- GPS 跳变检测 ----
        self._last_map_xy_by_source = {}       # source -> (x, y) 上一次有效地图坐标
        self._last_map_time_by_source = {}     # source -> 对应时间戳
        self._gps_jump_count = 0               # 累计跳变次数

        # ---- 累积偏差追踪（sliding window） ----
        # 每帧记录 (timestamp_s, gps_x, gps_y, amcl_x, amcl_y)
        self._traj_history = deque()
        self._traj_maxlen = int(self._cumulative_window * self._monitor_rate * 2)

        # ---- LIO 里程计状态（高频精确的速度+方向，替代 AMCL 差分） ----
        self._lio_linear_x = 0.0                # LIO 当前线速度（m/s）
        self._lio_linear_y = 0.0                # LIO 横向速度（m/s，通常≈0）
        self._lio_angular_z = 0.0               # LIO 当前角速度（rad/s）
        self._lio_heading = None                 # LIO 当前朝向（rad，从 odom quaternion 解算）
        self._lio_time = None                    # 最后收到 LIO odom 的时间戳（秒）
        self._lio_last_time = None               # 上一帧 LIO odom 时间戳（用于 dt 计算）
        self._robot_speed = 0.0                  # 当前合成速度（m/s）

        # ---- 前向轨迹预测（基于 LIO odom 逐帧积分） ----
        # 核心思想：LIO odom 在短时窗内局部精度极高（cm级），
        # 将每帧速度方向逐帧积分累积位置，与 AMCL 报告的位移对比，
        # 若位移不一致 → AMCL 在运动过程中匹配失败/跟丢了。
        self._lio_integrated_x = 0.0             # LIO odom 积分累积 X（自基准点起）
        self._lio_integrated_y = 0.0             # LIO odom 积分累积 Y（自基准点起）

        # 预测基准点（最近一次纠偏/建立基准时同时记录 LIO 和 AMCL 位置）
        self._pred_base_lio_x = None             # 基准时刻 LIO 积分 X
        self._pred_base_lio_y = None             # 基准时刻 LIO 积分 Y
        self._pred_base_amcl_x = None            # 基准时刻 AMCL X
        self._pred_base_amcl_y = None            # 基准时刻 AMCL Y
        self._pred_base_time = None              # 基准时间戳
        self._pred_lio_to_map_yaw = None         # 基准时刻 LIO odom 到 map 的旋转

        # 历史记录（保留用于调试）
        self._pred_history = deque()
        self._pred_maxlen = int(self._predict_window * self._monitor_rate * 2)

        # ---- AMCL 自身协方差（从 /amcl_pose 订阅） ----
        self._amcl_cov_xy = None                # max(cov[0], cov[7]), None=未订阅或未收到

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

        # ---- GPS fallback：RTK 不可用时使用普通 GPS 继续做低精度漂移监控 ----
        gps_topic = self.get_parameter('gps_topic').value
        self._gps_sub = None
        if self._enable_gps_fallback and gps_topic:
            self._gps_sub = self.create_subscription(
                NavSatFix, gps_topic, self._gps_callback, 10)
            self.get_logger().info(
                f'GPS fallback 数据源: {gps_topic} '
                f'(threshold={self._gps_drift_threshold}m, accuracy<={self._gps_min_accuracy}m)')

        # ---- 发布 /<ns>/initialpose（纠偏输出） ----
        # 使用相对路径让 ROS2 自动添加命名空间前缀:
        #   ns=""     → /initialpose
        #   ns="rkbot" → /rkbot/initialpose
        # Nav2 AMCL 在命名空间内订阅 /<ns>/initialpose，必须匹配
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10)

        # ---- 订阅 /initialpose（标定输入：首次收到即标定原点） ----
        # 使用绝对话题 /initialpose，因为 nav2_web_control 始终发布到全局话题：
        #   ns=""     → nav2_web_control 发 /initialpose，此处也收 /initialpose
        #   ns="rkbot" → nav2_web_control 仍发 /initialpose（全局），此处也收 /initialpose
        # 纠偏输出（_initialpose_pub）仍用相对话题，确保 ns 前缀自动匹配 AMCL
        self._initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose',
            self._on_initialpose_callback, 10,
        )

        # ---- 订阅 /<ns>/amcl_pose（获取 AMCL 自身协方差） ----
        amcl_pose_topic = self.get_parameter('amcl_pose_topic').value
        self._amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, amcl_pose_topic,
            self._amcl_pose_callback, 10,
        )
        self.get_logger().info(f'订阅 AMCL 协方差: {amcl_pose_topic}')

        # ---- 订阅 LIO 里程计（高频速度+方向，用于轨迹推算和静止检测） ----
        lio_odom_topic = self.get_parameter('lio_odom_topic').value
        self._lio_odom_sub = self.create_subscription(
            Odometry, lio_odom_topic,
            self._lio_odom_callback, 10,
        )
        self.get_logger().info(f'订阅 LIO 里程计: {lio_odom_topic}')

        # ---- 服务：主动标定原点（以当前 AMCL+GPS 位置多帧标定 E₀/N₀/θ₀） ----
        self._calib_srv = self.create_service(
            Trigger, 'start_gps_origin_calibration', self._on_active_calibration)

        # ---- 监控定时器 ----
        self._monitor_timer = self.create_timer(
            1.0 / self._monitor_rate, self._monitor_loop)

        self._last_status_time = self.get_clock().now()
        self._status_interval = self.get_parameter('status_report_interval').value
        self._status_timer = self.create_timer(1.0, self._status_report)

        self.get_logger().info(
            f'RTK/GPS 位姿监控节点已启动 | 等待首次 /initialpose 自动标定原点\n'
            f'  主动标定: ros2 service call /start_gps_origin_calibration std_srvs/srv/Trigger\n'
            f'  标定策略: 多帧采集 ({self._calib_sample_count}帧) → 平均值+标准差评估\n'
            f'  标定门限: position_std<{self._calib_max_std_position}m, heading_std<{self._calib_max_std_heading}°\n'
            f'  纠偏协方差: max(GPS精度, 标定误差², {self._correction_cov_base}m²下限)\n'
            f'  纠偏触发: RTK>{self._rtk_drift_threshold}m | GPS>{self._gps_drift_threshold}m | '
            f'amcl_cov>{self._amcl_cov_threshold} | '
            f'pred_drift>{self._predict_drift_threshold}m/{self._predict_window}s | '
            f'stationary(drift>{self._stationary_drift_threshold}m @ speed<{self._stationary_speed}m/s) | '
            f'cumulative>{self._cumulative_threshold}m/{self._cumulative_window}s\n'
            f'  LIO odom: {self.get_parameter("lio_odom_topic").value}\n'
            f'  min_interval={self._min_interval}s, rate={self._monitor_rate}Hz\n'
            f'  map_frame={self._map_frame}, base_frame={self._base_frame}')

    # ==================================================================
    #  回调
    # ==================================================================

    def _rtk_pos_valid_diag(self, bestnav) -> bool:
        """检查 RTK 位置是否有效（pos_type 为位置质量指标），拒绝时输出诊断日志（限频）。

        注意：pos_type 与 heading_type 共享表 0-4 枚举，但分别代表位置质量和航向质量。
        航向提取使用 heading_type 独立校验（见 _rtk_callback 中的航向提取段）。
        """
        try:
            p_sol = bestnav.p_sol_status
            pos_type = bestnav.pos_type
            lat = bestnav.latitude_deg
            lon = bestnav.longitude_deg
        except AttributeError as e:
            self._rtk_diag_log(
                'message_field_error',
                f'无法访问 bestnav 字段: {e}。消息类型是否匹配？'
            )
            return False

        if p_sol not in (0, 2):
            self._rtk_diag_log(
                'sol_status_invalid',
                f'p_sol_status={p_sol} (期望0或2), pos_type={pos_type}'
            )
            return False

        if pos_type not in (16, 17, 34, 50):
            svs = getattr(bestnav, 'svs_num', '?')
            soln = getattr(bestnav, 'soln_svs_num', '?')
            self._rtk_diag_log(
                'position_not_converged',
                f'pos_type={pos_type} (期望16/17/34/50), '
                f'可见星={svs}, 解算星={soln}'
            )
            return False

        if math.isnan(lat) or math.isnan(lon):
            self._rtk_diag_log(
                'nan_coordinate',
                f'lat={lat}, lon={lon}'
            )
            return False

        # 差分龄期检查（diff_age_s 过高说明差分修正数据过时）
        diff_age = getattr(bestnav, 'diff_age_s', 0.0)
        if diff_age > self._max_diff_age:
            self._rtk_diag_log(
                'diff_age_stale',
                f'diff_age={diff_age:.1f}s > {self._max_diff_age}s, pos_type={pos_type}'
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

    def _gps_fix_valid(self, msg: NavSatFix) -> bool:
        """检查普通 GPS fallback 数据是否可用于低精度纠偏。"""
        if msg.status.status < 0:
            return False
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return False
        if not (-90.0 <= msg.latitude <= 90.0):
            return False
        if not (-180.0 <= msg.longitude <= 180.0):
            return False
        h_acc = compute_horizontal_accuracy(msg)
        if h_acc != float('inf') and h_acc > self._gps_min_accuracy:
            return False
        return True

    def _gps_callback(self, msg: NavSatFix):
        """普通 GPS fallback 输入：仅在 RTK 不新鲜时参与标定和纠偏。"""
        if not self._gps_fix_valid(msg):
            return
        self._latest_gps_fix = msg
        self._last_gps_time = self.get_clock().now()
        if self._latest_fix is None:
            self.get_logger().info(
                f'收到首帧有效 GPS fallback: '
                f'lat={msg.latitude:.8f}, lon={msg.longitude:.8f}, '
                f'h_acc={compute_horizontal_accuracy(msg):.2f}m')

    def _select_active_fix(self):
        """选择当前定位输入：RTK 新鲜则优先 RTK，否则降级到 GPS。"""
        now = self.get_clock().now()
        rtk_fresh = (
            self._latest_rtk_fix is not None and
            self._last_rtk_time is not None and
            (now - self._last_rtk_time).nanoseconds * 1e-9 <= self._gps_stale_timeout
        )
        if rtk_fresh:
            return self._latest_rtk_fix, self.SOURCE_RTK, self._last_rtk_time

        gps_fresh = (
            self._enable_gps_fallback and
            self._latest_gps_fix is not None and
            self._last_gps_time is not None and
            (now - self._last_gps_time).nanoseconds * 1e-9 <= self._gps_stale_timeout
        )
        if gps_fresh:
            return self._latest_gps_fix, self.SOURCE_GPS, self._last_gps_time
        return None, None, None

    def _refresh_active_fix(self) -> bool:
        """刷新 _latest_fix，返回是否存在可用定位输入。"""
        fix, source, stamp = self._select_active_fix()
        if fix is None:
            if self._latest_fix is not None:
                self.get_logger().warn('RTK/GPS 数据均超时或无效，进入无定位输入状态')
            self._latest_fix = None
            self._latest_fix_source = None
            self._last_fix_time = None
            return False
        source_changed = self._latest_fix_source != source
        self._latest_fix = fix
        self._latest_fix_source = source
        self._last_fix_time = stamp
        if source_changed:
            threshold = self._threshold_for_source(source)
            self.get_logger().info(
                f'定位输入切换为 {source.upper()}，纠偏阈值={threshold:.1f}m')
        return True

    def _threshold_for_source(self, source: str) -> float:
        return (self._rtk_drift_threshold
                if source == self.SOURCE_RTK else self._gps_drift_threshold)

    def _rtk_callback(self, msg):
        """从 RTK 原始消息提取位置（转 NavSatFix）和航向。

        UniRtkPvh → NavSatFix 映射（与 gps_preprocessor._rtk_to_navsat_fix 一致）：
          - bestnav.latitude_deg/longitude_deg/altitude_m → NavSatFix
          - pos_type 50→status=4(RTK_FIX), 34→5(RTK_FLOAT), 17→2(DGPS), 16→2(单点)
          - lat_std/lon_std/hgt_std → position_covariance
          - heading.heading_deg → _latest_rtk_heading（需 heading_type ∈ {16,17,34,50}，独立于 pos_type）
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

        # ---- 位置收敛检查（pos_type 为位置质量指标） ----
        if not self._rtk_pos_valid_diag(bestnav):
            return

        # ---- 构造 NavSatFix（供监控循环使用） ----
        navsat = NavSatFix()
        navsat.header = bestnav.header if bestnav.header.stamp.sec else msg.header
        navsat.header.frame_id = 'gps'
        navsat.latitude = bestnav.latitude_deg
        navsat.longitude = bestnav.longitude_deg
        navsat.altitude = bestnav.altitude_m if not math.isnan(bestnav.altitude_m) else 0.0

        # pos_type → NavSatFix status（位置质量指标）
        # 注意：pos_type 可能因设备未注册激活而偏低（如始终=16），
        # 此时 heading_type 可独立达到 50（双天线基线收敛）
        if bestnav.pos_type == 50:
            navsat.status.status = 4   # RTK_FIX
        elif bestnav.pos_type == 34:
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
                f'(pos_type={bestnav.pos_type})'
            )
            return

        # ---- 首次收到有效 RTK 数据时输出日志 ----
        if self._latest_rtk_fix is None:
            self.get_logger().info(
                f'收到首帧有效 RTK 数据: pos_type={bestnav.pos_type}, '
                f'heading_type={heading.heading_type}, '
                f'lat={bestnav.latitude_deg:.8f}, lon={bestnav.longitude_deg:.8f}, '
                f'h_acc={h_acc:.3f}m'
            )
            self._rtk_reject_count = 0
            self._rtk_reject_reason = None

        self._latest_rtk_fix = navsat
        self._last_rtk_time = self.get_clock().now()

        # ---- 航向提取 ----
        try:
            if heading.heading_type in (16, 17, 34, 50) and heading.sol_status in (0, 2):
                h_std = getattr(heading, 'heading_std', 0.0)
                if h_std > self._max_heading_std:
                    # 航向标准差过大，丢弃 RTK 航向，纠偏时用 AMCL yaw 兜底
                    if self._latest_rtk_heading is not None:
                        self.get_logger().warn(
                            f'RTK航向标准差过大: heading_std={h_std:.1f}° > {self._max_heading_std}°, '
                            f'回退到 AMCL yaw')
                    self._latest_rtk_heading = None
                else:
                    self._latest_rtk_heading = float(heading.heading_deg)
        except Exception:
            pass

    def _on_initialpose_callback(self, msg: PoseWithCovarianceStamped):
        """首次收到外部 /initialpose → 启动多帧采集，完成后自动标定原点。

        不再使用单帧即刻标定（单帧 GPS/AMCL 的随机误差会导致原点偏差），
        改为采集 N 帧同步数据，每帧独立计算 (E₀, N₀, θ₀)，取平均 + 标准差评估。

        后续再收到的 /initialpose（非本节点发布）会被忽略——原点只标定一次。
        标定公式与 rtk_initial_pose.py / calibrate_map_origin.py 一致。
        """
        if self._calibrated:
            return   # 已标定，忽略后续 /initialpose

        if self._state == self.COLLECTING:
            return   # 正在采集中，忽略重复的 /initialpose

        if not self._refresh_active_fix():
            self.get_logger().info(
                '收到 /initialpose 但尚未收到有效 RTK/GPS 数据，'
                '等待定位输入后再标定...')
            return

        # 进入多帧采集模式（在 _monitor_loop 中逐步采集）
        self._calib_samples = []
        self._calib_source = self._latest_fix_source
        self._state = self.COLLECTING
        self.get_logger().info(
            f'收到 /initialpose，启动多帧标定采集 '
            f'(source={self._calib_source}, 目标 {self._calib_sample_count} 帧)')

    def _on_active_calibration(self, request, response):
        """服务回调：启动多帧标定采集，以当前 AMCL+GPS 数据计算地图原点。

        与自动标定 _on_initialpose_callback 公式一致，区别是：
          - 自动标定：由 /initialpose 触发（外部给了明确的 map 位姿）
          - 主动标定：由服务调用触发（以当前 AMCL 实际位姿为参考）

        采用多帧采集（而非单帧），避免单帧 GPS/AMCL 随机误差导致原点标定偏差。
        可重复标定——每次服务调用都会重新采集并计算原点。
        """
        if not self._refresh_active_fix():
            response.success = False
            response.message = '无有效 RTK/GPS 数据，无法标定。请确认定位输入已收敛'
            self.get_logger().warn(f'主动标定失败: {response.message}')
            return response

        # 进入多帧采集模式（在 _monitor_loop 中逐步采集）
        self._calib_samples = []
        self._calib_source = self._latest_fix_source
        self._state = self.COLLECTING
        self.get_logger().info(
            f'主动标定：启动多帧采集 '
            f'(source={self._calib_source}, 目标 {self._calib_sample_count} 帧)')

        response.success = True
        response.message = (
            f'标定采集中，目标 {self._calib_sample_count} 帧...')
        return response

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        """记录 AMCL 自身的位姿协方差（AMCL 不确定性的直接信号）。"""
        cov = msg.pose.covariance
        self._amcl_cov_xy = max(cov[0], cov[7])  # x, y 协方差取大者

    # ==================================================================
    #  多帧标定（提高地图原点精度）
    # ==================================================================

    def _collect_calibration_sample(self, amcl_x: float, amcl_y: float,
                                     amcl_yaw: float) -> bool:
        """采集一帧同步 AMCL+GPS+heading 数据用于多帧标定。

        每帧独立包含 (amcl_x, amcl_y, amcl_yaw, utm_e, utm_n, heading_deg)，
        后续 _finalize_calibration 中对每帧独立反算原点再取平均。

        Args:
            amcl_x, amcl_y: AMCL 当前地图坐标
            amcl_yaw: AMCL 当前朝向（rad）
        Returns:
            True 表示成功采集一帧
        """
        if self._latest_fix is None:
            return False

        if self._calib_source is None:
            self._calib_source = self._latest_fix_source
        elif self._latest_fix_source != self._calib_source:
            self.get_logger().warn(
                f'标定数据源从 {self._calib_source} 切换到 {self._latest_fix_source}，'
                '清空旧采样并重新标定')
            self._calib_samples = []
            self._calib_source = self._latest_fix_source

        fix = self._latest_fix
        try:
            e, n = latlon_to_utm(self._to_utm, fix.longitude, fix.latitude)
        except Exception:
            return False

        has_rtk_heading = (
            self._latest_fix_source == self.SOURCE_RTK and
            self._use_rtk_heading and
            self._latest_rtk_heading is not None
        )
        hdg = self._latest_rtk_heading if has_rtk_heading else None

        self._calib_samples.append(
            (amcl_x, amcl_y, amcl_yaw, e, n, hdg, self._latest_fix_source))
        while len(self._calib_samples) > self._gps_calib_max_samples:
            self._calib_samples.pop(0)
        return True

    def _estimate_theta0_without_heading(self):
        """GPS 无航向时，用机器人移动方向估计地图与 UTM 的旋转关系。"""
        if len(self._calib_samples) < 2:
            return None
        ax0, ay0, ayaw0, e0, n0, _, _ = self._calib_samples[0]
        ax1, ay1, ayaw1, e1, n1, _, _ = self._calib_samples[-1]
        amcl_dx = ax1 - ax0
        amcl_dy = ay1 - ay0
        utm_dx = e1 - e0
        utm_dy = n1 - n0
        amcl_dist = math.hypot(amcl_dx, amcl_dy)
        utm_dist = math.hypot(utm_dx, utm_dy)
        if min(amcl_dist, utm_dist) < self._gps_calib_min_motion:
            return None

        # rtk_to_map 使用 R(-theta0) 把 UTM 位移转到 map，故 theta0=utm_angle-map_angle。
        return math.atan2(utm_dy, utm_dx) - math.atan2(amcl_dy, amcl_dx)

    def _finalize_calibration(self) -> bool:
        """从采集的 N 帧同步数据计算地图原点 (E₀, N₀, θ₀) + 标准差评估。

        标定公式（与 calibrate_map_origin.py 一致）：
            对每帧 (ax, ay, ayaw, utm_e, utm_n, hdg_deg):
              θ₀ = hdg_rad - ayaw
              E₀ = utm_e - ax*cos(θ₀) + ay*sin(θ₀)
              N₀ = utm_n - ax*sin(θ₀) - ay*cos(θ₀)
            取各帧 E₀/N₀/θ₀ 的平均值作为最终标定结果。

        Returns:
            True 表示标定成功（即使标准差偏大也会标定，但会告警）
        """
        if len(self._calib_samples) == 0:
            self.get_logger().error('标定失败：无有效采样帧，请确认 GPS+AMCL 数据正常')
            self._state = self.WAIT_ORIGIN
            return False

        use_heading = any(sample[5] is not None for sample in self._calib_samples)
        fallback_theta0 = None
        if not use_heading:
            fallback_theta0 = self._estimate_theta0_without_heading()
            if fallback_theta0 is None:
                self.get_logger().info(
                    f'GPS fallback 标定等待足够位移: '
                    f'{len(self._calib_samples)}帧，'
                    f'需要机器人移动约 {self._gps_calib_min_motion:.1f}m')
                return False

        # 每帧独立计算 (E₀, N₀, θ₀)
        e0_list, n0_list, h0_list = [], [], []

        for ax, ay, ayaw, utm_e, utm_n, hdg_deg, _source in self._calib_samples:
            theta0 = math.radians(hdg_deg) - ayaw if hdg_deg is not None else fallback_theta0
            cos_t0 = math.cos(theta0)
            sin_t0 = math.sin(theta0)
            e0 = utm_e - ax * cos_t0 + ay * sin_t0
            n0 = utm_n - ax * sin_t0 - ay * cos_t0
            e0_list.append(e0)
            n0_list.append(n0)
            h0_list.append(math.degrees(theta0))

        n = len(e0_list)

        # 平均值
        e0_mean = sum(e0_list) / n
        n0_mean = sum(n0_list) / n
        h0_mean = sum(h0_list) / n

        # 标准差（≥ 2 帧才有意义）
        if n > 1:
            self._calib_position_std = math.sqrt(
                (sum((x - e0_mean) ** 2 for x in e0_list)
                 + sum((x - n0_mean) ** 2 for x in n0_list))
                / (2 * (n - 1))
            )
            self._calib_heading_std = math.sqrt(
                sum((x - h0_mean) ** 2 for x in h0_list) / (n - 1)
            )
        else:
            self._calib_position_std = 0.0
            self._calib_heading_std = 0.0

        # ---- 质量评估 ----
        pos_warn = self._calib_position_std > self._calib_max_std_position
        hdg_warn = self._calib_heading_std > self._calib_max_std_heading
        quality_issue = pos_warn or hdg_warn

        if quality_issue:
            self.get_logger().warn(
                f'⚠ 标定标准差偏大（原点精度可能不足，纠偏协方差已自动膨胀）:\n'
                f'  位置标准差: {self._calib_position_std:.3f}m '
                f'{"⚠ >" if pos_warn else "OK <"}{self._calib_max_std_position}m\n'
                f'  朝向标准差: {self._calib_heading_std:.2f}° '
                f'{"⚠ >" if hdg_warn else "OK <"}{self._calib_max_std_heading}°\n'
                f'  建议：确保 GPS 为 RTK_FIX 且 AMCL 已收敛后，重新标定')
        else:
            self.get_logger().info(
                f'✓ 标定标准差正常: '
                f'position={self._calib_position_std:.3f}m, '
                f'heading={self._calib_heading_std:.2f}°')

        # ---- 设置原点 ----
        a0 = 0.0  # 高度暂不处理
        self._origin_utm = (e0_mean, n0_mean, a0)
        self._theta0_rad = math.radians(h0_mean)
        self._calibrated = True
        self._state = self.MONITORING
        self._last_correction_time = self.get_clock().now()

        quality = detect_rtk_quality(self._latest_fix) if self._latest_fix else 'N/A'
        hdg_str = (f'{self._latest_rtk_heading:.1f}°'
                   if self._latest_rtk_heading else 'N/A')
        source = self._calib_source or self._latest_fix_source or 'unknown'

        self.get_logger().info(
            f'========== 多帧标定完成 ({n}帧) ==========\n'
            f'  原点 UTM: ({e0_mean:.2f}, {n0_mean:.2f})\n'
            f'  朝向 θ₀: {h0_mean:.4f}° (0=地图Y轴对齐真北)\n'
            f'  位置标准差: {self._calib_position_std:.3f}m '
            f'(±{self._calib_position_std * 2:.1f}m @ 2σ)\n'
            f'  朝向标准差: {self._calib_heading_std:.3f}° '
            f'(±{self._calib_heading_std * 2:.1f}° @ 2σ)\n'
            f'  数据源: {source.upper()} | 质量: {quality} | heading={hdg_str}\n'
            f'  纠偏协方差基准: {self._correction_cov_base}m²\n'
            f'  开始纠偏监控 (RTK>{self._rtk_drift_threshold}m, '
            f'GPS>{self._gps_drift_threshold}m)\n'
            f'===================================')

        self._calib_samples = []
        self._calib_source = None
        return True

    def _lio_odom_callback(self, msg: Odometry):
        """从 LIO 里程计提取速度、角速度、朝向，并逐帧积分累积位置。

        逐帧积分原理：
          - 每次回调计算 dt = 当前帧时间 - 上一帧时间
          - 用当前帧的速度和朝向（在 odom 坐标系下）积分：
              Δx = vx * dt * cos(heading) - vy * dt * sin(heading)
              Δy = vx * dt * sin(heading) + vy * dt * cos(heading)
          - 累积到 _lio_integrated_x/y（用于后续与 AMCL 位移对比）

        相比之前的"瞬时速度 × 总时间"快照外推，逐帧积分能正确处理
        转弯、变速等非匀速运动，短时窗（5-10s）内精度可达 cm 级。
        """
        twist = msg.twist.twist
        self._lio_linear_x = twist.linear.x
        self._lio_linear_y = twist.linear.y
        self._lio_angular_z = twist.angular.z
        self._robot_speed = math.sqrt(
            twist.linear.x ** 2 + twist.linear.y ** 2)

        # 从 odom 的 pose 解算当前朝向（用于轨迹积分）
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._lio_heading = math.atan2(siny, cosy)

        # 时间戳
        self._lio_time = (rclpy.time.Time.from_msg(msg.header.stamp)
                          .nanoseconds * 1e-9)

        # ---- 逐帧积分：累加 LIO odom 位移 ----
        if self._lio_last_time is not None and self._lio_heading is not None:
            dt = self._lio_time - self._lio_last_time
            # 仅当 dt 在合理范围（0 < dt < 0.5s）时积分，
            # 过大说明掉帧/重启，跳过避免异常跳变
            if 0.0 < dt < 0.5:
                heading = self._lio_heading
                # 在 odom 坐标系下：前进方向 = heading
                self._lio_integrated_x += (
                    self._lio_linear_x * math.cos(heading)
                    - self._lio_linear_y * math.sin(heading)
                ) * dt
                self._lio_integrated_y += (
                    self._lio_linear_x * math.sin(heading)
                    + self._lio_linear_y * math.cos(heading)
                ) * dt
        self._lio_last_time = self._lio_time

    # ==================================================================
    #  纠偏决策：多条件触发
    # ==================================================================

    def _should_correct(self, drift: float, amcl_x: float, amcl_y: float,
                        amcl_yaw: float, gps_x: float, gps_y: float,
                        source: str) -> (bool, str):
        """多条件纠偏决策：返回 (是否纠偏, 触发原因)。

        五个触发维度，按优先级排序：

        1. 瞬时大偏差：drift > 当前数据源阈值 → 紧急纠正
        2. AMCL 协方差膨胀：AMCL 自己都不确定位置了 → 用 GPS 注入确定性
        3. 前向预测偏差：LIO odom 推算的"应到位置"与 AMCL 实际位置偏差过大
           → AMCL 在移动过程中跟丢了（最重要的"移动中契机"）
        4. 静止窗口：机器人不动 + 有小偏差 → 低成本精准微调
        5. 累积位移偏差：滑动窗口内 GPS/AMCL 位移不一致 → 系统偏差
        """
        now = self.get_clock().now()
        elapsed = (now - self._last_correction_time).nanoseconds * 1e-9
        if elapsed < self._min_interval:
            return False, f'interval({elapsed:.1f}s<{self._min_interval}s)'

        source_threshold = self._threshold_for_source(source)

        # ---- 1. 瞬时大偏差 ----
        if drift > source_threshold:
            return True, f'instant_drift({source}:{drift:.2f}m>{source_threshold}m)'

        # ---- 2. AMCL 协方差膨胀 ----
        if (self._amcl_cov_xy is not None and
                self._amcl_cov_xy > self._amcl_cov_threshold and
                drift > source_threshold):
            return True, f'amcl_cov({self._amcl_cov_xy:.2f}>{self._amcl_cov_threshold})'

        # ---- 3. 前向预测偏差：LIO odom 推算位置 vs AMCL 实际位置 ----
        pred_drift = self._compute_prediction_drift(amcl_x, amcl_y, amcl_yaw)
        pred_threshold = max(self._predict_drift_threshold, source_threshold)
        if pred_drift > pred_threshold and drift > source_threshold:
            return True, f'pred_drift({pred_drift:.2f}m>{self._predict_drift_threshold}m, AMCL偏离预测轨迹)'

        # ---- 4. 静止窗口 ----
        stationary_threshold = max(self._stationary_drift_threshold, source_threshold)
        if self._robot_speed < self._stationary_speed and drift > stationary_threshold:
            return True, (
                f'stationary(speed={self._robot_speed:.2f}m/s<{self._stationary_speed}, '
                f'drift={drift:.2f}m>{stationary_threshold}m)')

        # ---- 5. 累积位移偏差 ----
        cum_drift = self._compute_cumulative_drift()
        cumulative_threshold = max(self._cumulative_threshold, source_threshold)
        if cum_drift > cumulative_threshold and drift > source_threshold:
            return True, f'cumulative({cum_drift:.2f}m>{cumulative_threshold}m over {self._cumulative_window}s)'

        return False, ''

    def _compute_prediction_drift(self, amcl_x: float, amcl_y: float,
                                  amcl_yaw: float = None) -> float:
        """前向预测偏差：比较 LIO odom 积分位移 与 AMCL 报告位移。

        原理：
          - LIO odom 在短时间窗口内局部精度极高（cm级，无全局漂移）
          - 以最近一次纠偏点为基准，同时记录 LIO 积分位置和 AMCL 位置
          - LIO 积分位移 = 当前 LIO 积分位置 - 基准 LIO 积分位置，并旋转到 map 坐标系
          - AMCL 报告位移 = 当前 AMCL 位置 - 基准 AMCL 位置
          - 两者偏差大 → AMCL 在运动过程中匹配失败/跟丢了

        与旧版"瞬时速度 × 总时间"快照外推的区别：
          旧版假设机器人匀速直线运动，转弯/变速时完全失效。
          新版通过 odom 回调中逐帧积分，正确追踪任意运动轨迹。

        这是解决"移动过程中缺少纠偏契机"的关键维度。
        """
        if self._lio_time is None or self._lio_heading is None:
            return 0.0

        now_s = self.get_clock().now().nanoseconds * 1e-9

        # 首次调用或纠偏后基准为空 → 建立基准
        if self._pred_base_amcl_x is None or self._pred_base_lio_x is None:
            self._pred_base_amcl_x = amcl_x
            self._pred_base_amcl_y = amcl_y
            self._pred_base_lio_x = self._lio_integrated_x
            self._pred_base_lio_y = self._lio_integrated_y
            self._pred_base_time = now_s
            if amcl_yaw is not None:
                self._pred_lio_to_map_yaw = amcl_yaw - self._lio_heading
            else:
                self._pred_lio_to_map_yaw = 0.0
            return 0.0

        dt = now_s - self._pred_base_time
        if dt < 1.0:  # 基准建立时间太短，累积位移太小无意义
            return 0.0

        # LIO 积分位移（odom 坐标系下机器人实际移动了多少）
        lio_dx = self._lio_integrated_x - self._pred_base_lio_x
        lio_dy = self._lio_integrated_y - self._pred_base_lio_y

        # 将 LIO odom 位移旋转到 map 坐标系，再与 AMCL map 位移比较。
        # 若 odom/map 轴本来一致，该角度为 0；若初始位姿带 yaw，这一步避免误判。
        if self._pred_lio_to_map_yaw is None:
            self._pred_lio_to_map_yaw = 0.0
        cos_t = math.cos(self._pred_lio_to_map_yaw)
        sin_t = math.sin(self._pred_lio_to_map_yaw)
        lio_map_dx = lio_dx * cos_t - lio_dy * sin_t
        lio_map_dy = lio_dx * sin_t + lio_dy * cos_t

        # AMCL 报告位移（地图坐标系下 AMCL 认为机器人移动了多少）
        amcl_dx = amcl_x - self._pred_base_amcl_x
        amcl_dy = amcl_y - self._pred_base_amcl_y

        # 位移向量差异（m），LIO 和 AMCL 报告的移动量应该一致
        drift = math.sqrt((lio_map_dx - amcl_dx) ** 2 + (lio_map_dy - amcl_dy) ** 2)

        # 同时检查绝对偏差（AMCL 可能停在原地，而机器人实际在移动）
        abs_drift = math.sqrt((amcl_x - self._pred_base_amcl_x) ** 2
                              + (amcl_y - self._pred_base_amcl_y) ** 2)

        # 如果 LIO 说走了很远但 AMCL 几乎没动 → AMCL 跟丢了，返回大偏差
        lio_travel = math.sqrt(lio_dx ** 2 + lio_dy ** 2)
        if lio_travel > 2.0 and abs_drift < 0.3:
            # AMCL 冻结了（scan matching 持续失败）
            return lio_travel

        return drift

    def _update_prediction_baseline(self, amcl_x: float, amcl_y: float,
                                    amcl_yaw: float = None):
        """纠偏后重置预测基准点（同时记录 LIO 积分位置和 AMCL 位置）。

        AMCL 已被 GPS 纠正到正确位置，从这里重新开始追踪位移。
        """
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._pred_base_amcl_x = amcl_x
        self._pred_base_amcl_y = amcl_y
        self._pred_base_lio_x = self._lio_integrated_x
        self._pred_base_lio_y = self._lio_integrated_y
        self._pred_base_time = now_s
        if amcl_yaw is not None and self._lio_heading is not None:
            self._pred_lio_to_map_yaw = amcl_yaw - self._lio_heading
        self._pred_history.clear()

    def _compute_cumulative_drift(self) -> float:
        """计算滑动窗口内 GPS 位移与 AMCL 位移的差异。

        返回 |gps_displacement - amcl_displacement|，单位米。
        若窗口内数据不足则返回 0.0。
        """
        if len(self._traj_history) < 2:
            return 0.0

        t0, gx0, gy0, ax0, ay0 = self._traj_history[0]
        tn, gxn, gyn, axn, ayn = self._traj_history[-1]

        dt = tn - t0
        if dt < 2.0:  # 窗口太短，不计算
            return 0.0

        gps_dist = math.sqrt((gxn - gx0) ** 2 + (gyn - gy0) ** 2)
        amcl_dist = math.sqrt((axn - ax0) ** 2 + (ayn - ay0) ** 2)
        return abs(gps_dist - amcl_dist)

    def _update_trajectory_history(self, gps_x: float, gps_y: float,
                                   amcl_x: float, amcl_y: float):
        """维护累积偏差滑动窗口。"""
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._traj_history.append((now_s, gps_x, gps_y, amcl_x, amcl_y))

        # 限制队列长度，防止内存增长
        while len(self._traj_history) > self._traj_maxlen:
            self._traj_history.popleft()

        # 按时间窗口修剪
        cutoff = now_s - self._cumulative_window
        while self._traj_history and self._traj_history[0][0] < cutoff:
            self._traj_history.popleft()

    # ==================================================================
    #  监控主循环
    # ==================================================================

    def _monitor_loop(self):
        """监控主循环：比对 GPS 位姿与当前位姿（tf）"""
        # 等待 /initialpose 标定完成
        if self._state == self.WAIT_ORIGIN:
            return

        # ---- 多帧标定采集（COLLECTING 状态） ----
        if self._state == self.COLLECTING:
            # 获取当前 AMCL 位姿
            try:
                tf_msg = self._tf_buffer.lookup_transform(
                    self._map_frame, self._base_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=0.5))
            except TransformException:
                return  # TF 不可用，等下一帧

            amcl_x = tf_msg.transform.translation.x
            amcl_y = tf_msg.transform.translation.y
            amcl_yaw = _quat_to_yaw(tf_msg.transform.rotation)

            # 定位输入有效性检查：RTK 优先，RTK 不可用则 GPS fallback
            if not self._refresh_active_fix():
                return

            # 采集一帧
            if self._collect_calibration_sample(amcl_x, amcl_y, amcl_yaw):
                n = len(self._calib_samples)
                if n == 1 or n % 5 == 0:
                    quality = detect_rtk_quality(self._latest_fix)
                    self.get_logger().info(
                        f'标定采集中... {n}/{self._calib_sample_count} '
                        f'(source={self._latest_fix_source} '
                        f'AMCL=({amcl_x:.1f},{amcl_y:.1f}) Q={quality})')

                if n >= self._calib_sample_count:
                    self._finalize_calibration()

            return  # 采集中不做纠偏

        gps_valid = self._refresh_active_fix()

        # ---- 状态机转换 ----
        if self._state == self.MONITORING:
            if not gps_valid:
                self._gps_fail_count += 1
                if self._gps_fail_count >= self._gps_loss_threshold:
                    self._state = self.GPS_LOST
                    self._gps_recover_count = 0
                    self.get_logger().warn(
                        f'RTK/GPS 连续 {self._gps_fail_count} 帧无效，'
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
                        f'RTK/GPS 恢复（连续 {self._gps_recover_count} 帧有效），'
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

        source = self._latest_fix_source or self.SOURCE_GPS

        # ---- GPS/RTK 位置跳变检测 ----
        # 比较当前 GPS 地图坐标与上一帧有效坐标的变化速度。
        # 若超过物理极限（默认 5m/s），视为 GPS 抖动，跳过本帧纠偏。
        now_monitor = self.get_clock().now()
        last_xy = self._last_map_xy_by_source.get(source)
        last_time = self._last_map_time_by_source.get(source)
        if last_xy is not None and last_time is not None:
            dx = gps_x - last_xy[0]
            dy = gps_y - last_xy[1]
            dt = (now_monitor - last_time).nanoseconds * 1e-9
            if dt > 0.1:  # 时间间隔足够才做速度判断
                velocity = math.sqrt(dx * dx + dy * dy) / dt
                if velocity > self._gps_jump_threshold:
                    self._gps_jump_count += 1
                    quality = detect_rtk_quality(fix)
                    self.get_logger().warn(
                        f'{source.upper()} 跳变 #{self._gps_jump_count}: '
                        f'vel={velocity:.1f}m/s > {self._gps_jump_threshold}m/s | '
                        f'GPS=({gps_x:.1f},{gps_y:.1f}) '
                        f'prev=({last_xy[0]:.1f},{last_xy[1]:.1f}) '
                        f'dt={dt:.1f}s Q={quality} | 跳过纠偏')
                    return  # 不纠偏，也不更新当前数据源的上一帧有效位置

        # 通过跳变检测 → 按数据源记录有效位置，避免 RTK/GPS 切换时互相误判
        self._last_map_xy_by_source[source] = (gps_x, gps_y)
        self._last_map_time_by_source[source] = now_monitor

        # ---- 偏差计算 ----
        drift = math.sqrt((gps_x - amcl_x) ** 2 + (gps_y - amcl_y) ** 2)

        # ---- 多条件纠偏决策 ----
        should_correct, reason = self._should_correct(
            drift, amcl_x, amcl_y, amcl_yaw, gps_x, gps_y, source)
        if not should_correct:
            # 非纠偏帧也更新累积窗口（用于累积偏差检测）
            self._update_trajectory_history(gps_x, gps_y, amcl_x, amcl_y)
            self.get_logger().debug(f'source={source}, drift={drift:.3f}m, speed={self._robot_speed:.2f}m/s, '
                                    f'amcl_cov={self._amcl_cov_xy}, cum_drift={self._compute_cumulative_drift():.2f}m, '
                                    f'skip({reason})')
            return

        # ---- 执行纠偏 ----
        now = self.get_clock().now()
        quality = detect_rtk_quality(fix)

        # 纠偏协方差策略（三级保护，避免过度信任 GPS 推算位置）：
        #   1. GPS 自身精度协方差（RTK_FIX=0.01, RTK_FLOAT=0.1, DGPS=1.0）
        #   2. 标定误差方差（calib_position_std²）
        #   3. 下限保护（correction_cov_base=0.25m²）
        # 取三者最大值 → AMCL 粒子云有一定散布空间，可通过 scan matching 自我精化
        if source == self.SOURCE_GPS:
            gps_cov = self._cov_gps
        else:
            gps_cov = covariance_for_quality(
                quality,
                cov_rtk_fix=self.get_parameter('cov_rtk_fix').value,
                cov_rtk_float=self.get_parameter('cov_rtk_float').value,
                cov_dgps=self.get_parameter('cov_dgps').value,
                cov_gps=self._cov_gps)
        calib_var = self._calib_position_std ** 2
        pos_cov = max(gps_cov, calib_var, self._correction_cov_base)

        # 航向处理
        if (source == self.SOURCE_RTK and
                self._use_rtk_heading and self._latest_rtk_heading is not None):
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

        # 纠偏后重置累积窗口 + 预测基准（以 GPS 纠偏后的位置为新起点）
        self._traj_history.clear()
        self._update_prediction_baseline(gps_x, gps_y, map_yaw)

        self.get_logger().info(
            f'纠偏 [{reason}]: source={source} drift={drift:.3f}m | '
            f'GPS map=({gps_x:.2f},{gps_y:.2f}) AMCL=({amcl_x:.2f},{amcl_y:.2f}) | '
            f'speed={self._robot_speed:.2f}m/s cov={pos_cov} '
            f'yaw_src={hdg_src} quality={quality}')

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
                f'[状态] 定位输入=无信号 | AMCL=-- | state={self._state} '
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
            f'[状态] source={self._latest_fix_source} map=({gps_x:.2f},{gps_y:.2f}) | '
            f'AMCL=({amcl_x:.2f},{amcl_y:.2f}) | '
            f'drift={drift:.2f}m | '
            f'Q={quality} | state={self._state} | origin={origin_status}'
            f'{" | jumps=" + str(self._gps_jump_count) if self._gps_jump_count > 0 else ""}')


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
