#!/usr/bin/env python3
"""
自动初始位姿校准节点 — 室内探索与避障改进版

主要流程：
  ① 开启服务 → 调用 /start_auto_calibration
  ② 进入 BOOT_DELAY 状态，静止等待 2 秒稳定数据
  ③ 进入 COLLECTING_SUBMAP1 状态，收集 30 帧雷达数据结合 Odom 合成 Submap 1
  ④ 进入 ROUGH_MATCHING 状态，使用分层粗精两阶段粒子搜索计算 Top N 候选位姿
  ⑤ 进入 SELECTING_ACTIVE_MOTION 状态，计算各安全方向的信息增益，选择最优探索动作
  ⑥ 进入 MOVING 状态，执行局部 P 控制器，并在运动中实时通过雷达避障监控
  ⑦ 运动结束（或因避障提前终止）进入 COLLECTING_SUBMAP2 状态，收集 30 帧合成 Submap 2
  ⑧ 进入 FILTERING 状态，传播候选位姿，利用 Submap 2 重新评分，判断是否唯一收敛
  ⑨ 若唯一收敛，发布 /initialpose 并转换到 DONE 状态；若未收敛，依据动态步长继续进行下轮探索
"""

import os
import sys
import math
import time
import logging
import importlib
import yaml
from datetime import datetime
from enum import Enum
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import LaserScan, NavSatFix
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion, Twist, PoseArray, PoseStamped
from std_srvs.srv import Trigger

try:
    from ament_index_python.packages import get_package_share_directory
    global_config_path = os.path.join(get_package_share_directory('global_config'), '../../src/global_config')
    if global_config_path not in sys.path:
        sys.path.insert(0, global_config_path)
    from global_config import NAV2_DEFAULT_MAP_FILE
except ImportError:
    NAV2_DEFAULT_MAP_FILE = "/home/ztl/slam_data/grid_map/map.yaml"


class IndoorPhase(Enum):
    IDLE = 0
    BOOT_DELAY = 1
    ROTATING_360 = 2
    COLLECTING_SUBMAP1 = 3
    ROUGH_MATCHING = 4
    SELECTING_ACTIVE_MOTION = 5
    MOVING = 6
    COLLECTING_SUBMAP2 = 7
    FILTERING = 8
    DONE = 9
    # ── 被动模式 ──
    PASSIVE_COLLECTING = 10
    PASSIVE_MATCHING = 11
    # ── 主动多步递推 ──
    ACTIVE_MULTISTEP = 12      # 逐帧ICP + 局部搜索, sigma衰减


class AutoInitialPoseCalibrator(Node):
    def __init__(self):
        super().__init__('auto_initial_pose_calibrator')

        # ────── 声明与读取参数 ──────
        self.declare_parameter('rtk_topic', '/rtk')
        self.declare_parameter('rtk_topic_type', 'robots_dog_msgs.msg.UniRtkPvh')
        self.declare_parameter('gps_topic', '/gps')
        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('scan_topic', 'scan')
        self.declare_parameter('odom_topic', 'lio/odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('outdoor_mode', True)
        self.declare_parameter('indoor_mode', True)
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('map_file', NAV2_DEFAULT_MAP_FILE)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        # 粗匹配/粒子匹配参数
        self.declare_parameter('rough_match_particles', 1500)
        self.declare_parameter('fine_match_particles_per_cluster', 50)
        self.declare_parameter('rough_match_max_beams', 60)
        self.declare_parameter('rough_top_n', 10)
        
        # 似然场参数
        self.declare_parameter('likelihood_max_dist', 2.0)
        self.declare_parameter('sigma_hit', 0.2)
        self.declare_parameter('z_hit', 0.8)
        self.declare_parameter('z_rand', 0.1)

        # 探索与控制避障参数
        self.declare_parameter('motion_distance', 0.8)
        self.declare_parameter('motion_angle_threshold_deg', 10.0)
        self.declare_parameter('filter_min_score_ratio', 0.3)
        self.declare_parameter('max_active_retry', 5)
        
        self.declare_parameter('max_linear_vel', 0.15)
        self.declare_parameter('max_angular_vel', 0.25)
        self.declare_parameter('kp_linear', 0.8)
        self.declare_parameter('kp_angular', 1.0)
        self.declare_parameter('min_safe_distance', 0.5)
        self.declare_parameter('collision_sector_angle_deg', 45.0)

        # 主动探索特征参数
        self.declare_parameter('ig_sample_dist_1', 1.0)
        self.declare_parameter('ig_sample_dist_2', 2.0)

        # 局部子图参数
        self.declare_parameter('submap_scan_count', 30)
        self.declare_parameter('submap_angle_resolution', 0.5)
        
        # 旋转全覆盖采集参数
        self.declare_parameter('rotation_enabled', True)
        self.declare_parameter('rotation_total_deg', 360.0)
        self.declare_parameter('rotation_angular_vel', 0.3)   # rad/s
        
        # 扫描离群点过滤参数（去除动态障碍物/杂点）
        self.declare_parameter('scan_outlier_filter', True)
        self.declare_parameter('scan_outlier_radius', 0.3)      # m, 邻域搜索半径
        self.declare_parameter('scan_outlier_min_neighbors', 3) # 至少需要的邻居点数

        # 多帧时序一致性过滤参数（利用帧间一致性剔除人/动物/动态障碍物）
        self.declare_parameter('temporal_merge_enabled', True)
        self.declare_parameter('temporal_merge_min_frames', 3)   # 至少来自多少不同帧才认定为静态
        self.declare_parameter('temporal_merge_radius', 0.15)    # m, 判断邻域一致性用的小半径

        # ICP 参数
        self.declare_parameter('icp_max_iterations', 15)
        self.declare_parameter('icp_tolerance_trans', 0.01)
        self.declare_parameter('icp_tolerance_rot_deg', 0.1)
        self.declare_parameter('icp_downsample_step', 5)

        # 运动验证参数
        self.declare_parameter('motion_verification_enabled', True)
        self.declare_parameter('motion_verification_threshold', 0.3)
        self.declare_parameter('motion_verification_angle_threshold_deg', 15.0)

        # 调试对比模式参数
        self.declare_parameter('auto_publish_initial_pose', False)
        self.declare_parameter('debug_comparison_mode', True)

        # ────── 开机自动启动与快速模式参数 ──────
        self.declare_parameter('auto_start', True)           # 开机收到地图+scan后自动触发校准
        self.declare_parameter('auto_start_delay', 3.0)      # 自动启动前的等待延迟 (秒)
        self.declare_parameter('quick_mode', False)           # 快速模式：跳过旋转+运动探索，仅做一次全局匹配

        # ────── 被动模式参数 ──────
        self.declare_parameter('passive_mode_enabled', True)    # 是否启用被动持续定位
        self.declare_parameter('passive_interval', 60.0)        # 被动匹配间隔 (秒)
        self.declare_parameter('passive_frame_count', 30)       # 每次被动匹配累积帧数
        self.declare_parameter('passive_buffer_max', 300)       # 循环缓冲区最大帧数

        # ────── 日志持久化参数 ──────
        self.declare_parameter('log_dir', '')                 # 日志持久化目录（为空则不写文件）
        self.declare_parameter('log_level', 'INFO')           # 文件日志级别

        # RTK室外参数
        self.declare_parameter('min_soln_svs', 4)
        self.declare_parameter('valid_pos_types', [34, 50])
        self.declare_parameter('valid_heading_types', [34, 50])
        self.declare_parameter('publish_rate', 0.5)

        # ───── 加载参数 ─────
        self.rtk_topic = self.get_parameter('rtk_topic').value
        self.rtk_topic_type = self.get_parameter('rtk_topic_type').value
        self.gps_topic = self.get_parameter('gps_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.map_frame = self.get_parameter('map_frame').value
        # 自动适配命名空间下的 topic 和 frame (例如 rkbot/map, rkbot/odom)
        ns = self.get_namespace().strip('/')
        if ns:
            if not self.map_frame.startswith(ns + '/'):
                self.map_frame = f"{ns}/{self.map_frame}"
            if not self.map_topic.startswith('/'):
                self.map_topic = f"/{ns}/{self.map_topic}"
            if not self.scan_topic.startswith('/'):
                self.scan_topic = f"/{ns}/{self.scan_topic}"
            if not self.odom_topic.startswith('/'):
                self.odom_topic = f"/{ns}/{self.odom_topic}"
        self.outdoor_mode = self.get_parameter('outdoor_mode').value
        self.indoor_mode = self.get_parameter('indoor_mode').value
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.map_file = self.get_parameter('map_file').value
        if not self.map_file:
            self.map_file = NAV2_DEFAULT_MAP_FILE
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.rough_particles = self.get_parameter('rough_match_particles').value
        self.fine_particles_per_cluster = self.get_parameter('fine_match_particles_per_cluster').value
        self.rough_beams = self.get_parameter('rough_match_max_beams').value
        self.top_n = self.get_parameter('rough_top_n').value
        
        self.likelihood_max_dist = self.get_parameter('likelihood_max_dist').value
        self.sigma_hit = self.get_parameter('sigma_hit').value
        self.z_hit = self.get_parameter('z_hit').value
        self.z_rand = self.get_parameter('z_rand').value

        self.base_motion_distance = self.get_parameter('motion_distance').value
        self.motion_angle_threshold = math.radians(self.get_parameter('motion_angle_threshold_deg').value)
        self.filter_min_ratio = self.get_parameter('filter_min_score_ratio').value
        self.max_active_retry = self.get_parameter('max_active_retry').value

        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.min_safe_distance = self.get_parameter('min_safe_distance').value
        self.collision_sector_angle = math.radians(self.get_parameter('collision_sector_angle_deg').value)

        self.ig_sample_dist_1 = self.get_parameter('ig_sample_dist_1').value
        self.ig_sample_dist_2 = self.get_parameter('ig_sample_dist_2').value

        self.submap_scan_count = self.get_parameter('submap_scan_count').value
        self.submap_angle_res = math.radians(self.get_parameter('submap_angle_resolution').value)
        
        # 旋转全覆盖参数
        self.rotation_enabled = self.get_parameter('rotation_enabled').value
        self.rotation_total_rad = math.radians(self.get_parameter('rotation_total_deg').value)
        self.rotation_angular_vel = self.get_parameter('rotation_angular_vel').value
        
        # 离群点过滤参数
        self.scan_outlier_filter = self.get_parameter('scan_outlier_filter').value
        self.scan_outlier_radius = self.get_parameter('scan_outlier_radius').value
        self.scan_outlier_min_neighbors = self.get_parameter('scan_outlier_min_neighbors').value

        # 多帧时序一致性过滤参数
        self.temporal_merge_enabled = self.get_parameter('temporal_merge_enabled').value
        self.temporal_merge_min_frames = self.get_parameter('temporal_merge_min_frames').value
        self.temporal_merge_radius = self.get_parameter('temporal_merge_radius').value

        self.publish_rate = self.get_parameter('publish_rate').value

        # ICP 参数读取
        self.icp_max_iterations = self.get_parameter('icp_max_iterations').value
        self.icp_tolerance_trans = self.get_parameter('icp_tolerance_trans').value
        self.icp_tolerance_rot_deg = self.get_parameter('icp_tolerance_rot_deg').value
        self.icp_downsample_step = self.get_parameter('icp_downsample_step').value

        # 运动验证参数读取
        self.motion_verification_enabled = self.get_parameter('motion_verification_enabled').value
        self.motion_verification_threshold = self.get_parameter('motion_verification_threshold').value
        self.motion_verification_angle_threshold_deg = self.get_parameter('motion_verification_angle_threshold_deg').value

        # 调试对比模式参数读取
        self.auto_publish_initial_pose = self.get_parameter('auto_publish_initial_pose').value
        self.debug_comparison_mode = self.get_parameter('debug_comparison_mode').value

        # 开机自动启动与快速模式参数读取
        self.auto_start = self.get_parameter('auto_start').value
        self.auto_start_delay = self.get_parameter('auto_start_delay').value
        self.quick_mode = self.get_parameter('quick_mode').value

        # ── 被动模式参数 ──
        self.passive_mode_enabled = self.get_parameter('passive_mode_enabled').value
        self.passive_interval = self.get_parameter('passive_interval').value
        self.passive_frame_count = self.get_parameter('passive_frame_count').value
        self.passive_buffer_max = self.get_parameter('passive_buffer_max').value

        # ────── 日志持久化初始化 ──────
        self._setup_file_logging()

        # 校准参数加载
        user_cal_file = self.get_parameter('calibration_file').value
        if user_cal_file:
            self.calibration_file = user_cal_file
        else:
            self.calibration_file = os.path.join(os.path.dirname(NAV2_DEFAULT_MAP_FILE), 'gps_map_calibration.yaml')
        self.calibration_tf = None
        self._load_calibration()

        # ────── 地图与似然场 ──────
        self.map_data = None
        self.map_info = None
        self.likelihood_field = None
        self.free_space_indices = None

        # ────── 传感器实时缓存 ──────
        self.current_scan = None
        self.current_odom = None
        self.current_rtk = None
        self.current_gps = None

        # ────── 室内状态变量 ──────
        self.indoor_phase = IndoorPhase.IDLE
        self.boot_start_time = None
        self.active_retry_count = 0
        self.candidates = []            # [(normalized_prob, x, y, yaw), ...]
        
        # 子图合并缓冲区（不依赖 odom，只用扫描消息 + 时间戳）
        self.scan_buffer = []           # [(scan_msg, timestamp)]
        self.submap_ready = False
        self.use_icp_for_submap = True # 使用 ICP 帧间匹配拼接子图
        
        # 旋转全覆盖采集状态
        self.rotation_start_yaw = None
        self.rotation_accumulated = 0.0
        self.rotation_start_time = None
        
        self.submap1 = None             # Synthesized LaserScan for stage 1
        self.submap1_ref_odom = None    # Odom reference at Submap 1 start
        self.submap2 = None             # Synthesized LaserScan for stage 2
        self.submap2_ref_odom = None    # Odom reference at Submap 2 start
        
        # 里程计数据验证
        self.last_odom_pose = None      # 上一帧 odom (x, y, yaw)
        self.max_odom_delta = 2.0       # 单帧最大允许里程计增量 (m)

        # 控制运动目标位姿
        self.target_odom_pose = None    # (x, y, yaw) in reference odom frame
        self.motion_start_odom = None   # Odom message at motion start
        self.motion_start_time = None

        # ────── Odom 与真值校对数据 ──────
        self.gt_received = False
        self.gt_pose = None             # (x, y, yaw) in map
        self.gt_ref_odom = None         # (x, y, yaw) in odom
        
        # ────── 上次校准结果（用于与手动设置对比） ──────
        self.last_calibrated_pose = None  # (x, y, yaw) in map

        # ────── 被动模式状态变量 ──────
        self.passive_scan_buffer = []      # [(scan_msg, timestamp)]
        self.passive_last_match_time = None # 上次被动匹配时间
        self.passive_best_pose = None       # 被动模式累积最佳位姿 (x, y, yaw)
        self.passive_pose_history = []      # [(timestamp, x, y, yaw, wall_cov), ...]

        # ────── QoS 配置 ──────
        be_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        # ────── 订阅器 ──────
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._map_cb, map_qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, be_qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_cb, be_qos)
        self.gps_sub = self.create_subscription(NavSatFix, self.gps_topic, self._gps_cb, be_qos)
        self._setup_rtk_sub(be_qos)

        # 校对 AMCL 话题
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb, 10)
        # 监听 RViz 2D Pose Estimate 触发真值标记或常规设置
        self.initialpose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._initialpose_cb, 10)

        # ────── 发布器 ──────
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        
        # 调试/可视化发布器
        self.candidates_pub = self.create_publisher(PoseArray, 'debug/candidates', 10)
        self.submap_scan_pub = self.create_publisher(LaserScan, 'debug/submap_scan', 10)
        self.odom_est_pose_pub = self.create_publisher(PoseStamped, 'debug/odom_est_pose', 10)
        # 调试模式：算法计算结果发布到该话题（不自动发布 /initialpose）
        self.debug_auto_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'debug/auto_initial_pose', 10)
        self.debug_pose_comparison_pub = self.create_publisher(PoseArray, 'debug/pose_comparison', 10)

        # ────── 服务 ──────
        self.srv_start = self.create_service(Trigger, 'start_auto_calibration', self._srv_start)
        self.srv_start_active = self.create_service(Trigger, 'start_active_calibration', self._srv_start_active)
        self.srv_start_passive = self.create_service(Trigger, 'start_passive_calibration', self._srv_start_passive)
        self.srv_stop_passive = self.create_service(Trigger, 'stop_passive_calibration', self._srv_stop_passive)
        self.srv_status = self.create_service(Trigger, 'auto_calibration_status', self._srv_status)
        self.srv_reset = self.create_service(Trigger, 'reset_calibration', self._srv_reset)
        # 调试模式服务
        self.srv_toggle_auto = self.create_service(Trigger, 'toggle_auto_publish', self._srv_toggle_auto_publish)
        from std_srvs.srv import SetBool
        self.srv_set_gt = self.create_service(SetBool, 'set_manual_ground_truth', self._srv_set_manual_ground_truth)

        self.detected_mode = None
        self.startup_time = self.get_clock().now()

        # ────── 开机自动启动状态 ──────
        self._auto_start_triggered = False          # 是否已触发自动启动
        self._auto_start_ready_time = None           # 所有数据就绪的时间点
        self._auto_start_map_received = False        # 是否已收到地图
        self._auto_start_scan_received = False       # 是否已收到扫描
        self._auto_start_odom_received = False       # 是否已收到里程计

        # ────── 定时器主循环 ──────
        self.indoor_timer = self.create_timer(0.1, self._indoor_loop)
        self.outdoor_timer = self.create_timer(self.publish_rate, self._outdoor_loop)
        self.mode_timer = self.create_timer(1.0, self._check_auto_mode)
        # 被动模式定时器
        self.passive_timer = self.create_timer(self.passive_interval, self._passive_timer_cb)

        # 延迟 1.5 秒检查是否订阅到网格地图，若无则使用本地地图文件进行回退加载
        self.map_check_timer = self.create_timer(1.5, self._check_map_and_fallback)

        self._logger.info('自动初始位姿校准器已就绪，正在自动检测并识别定位模式中...')

    def _setup_file_logging(self):
        """设置日志文件持久化：创建独立的 Python logging FileHandler，
        并通过包装方法 _flog() 同时输出到 ROS2 logger 和文件。
        
        ROS2 Humble 的 rclpy.logging.get_logger() 虽然返回 Logger 对象，
        但其底层 rcutils 日志系统不保证 addHandler 能正确路由，
        因此采用独立 file_logger + 包装方法的方式确保双写。
        """
        log_dir = self.get_parameter('log_dir').value
        log_level_str = self.get_parameter('log_level').value

        if not log_dir or not log_dir.strip():
            self.get_logger().info('[日志] log_dir 为空，跳过文件日志持久化')
            self._file_logger = None
            self._logger = self.get_logger()  # 无文件 logger 时，_logger 直接指向 ROS2 logger
            return

        # 创建日志目录
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            self.get_logger().error(f'[日志] 无法创建日志目录 {log_dir}: {e}')
            self._file_logger = None
            self._logger = self.get_logger()
            return

        # 日志文件名: auto_calibrator_{ns}_{timestamp}.log
        ns = self.get_namespace().strip('/')
        ns_suffix = f'_{ns}' if ns else ''
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'auto_calibrator{ns_suffix}_{timestamp}.log'
        log_path = os.path.join(log_dir, log_filename)

        # 映射日志级别
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARN': logging.WARNING,
            'ERROR': logging.ERROR,
        }
        file_level = level_map.get(log_level_str.upper(), logging.INFO)

        # 创建独立的 Python file logger（不依赖 rclpy 内部路由）
        self._file_logger = logging.getLogger(f'auto_calibrator_file.{timestamp}')
        self._file_logger.setLevel(file_level)
        self._file_logger.handlers.clear()  # 防止重复添加

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(file_level)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self._file_logger.addHandler(file_handler)

        # 存储引用防止被 GC
        self._file_handler = file_handler
        self._log_path = log_path

        # Monkey-patch self._logger 的方法，自动双写到文件
        self._patch_logger_for_file_output()

        self._flog(
            'info',
            f'[日志] 文件日志已启用: {log_path} (级别={log_level_str}, ns={ns or "无"})'
        )

    def _patch_logger_for_file_output(self):
        """创建日志代理对象并赋值给 self._logger，
        使所有 self._logger.info/warn/error/debug 调用同时输出到 ROS2 和文件。
        """
        ros2_logger = self.get_logger()

        if self._file_logger is None:
            self._logger = ros2_logger
            return

        file_logger = self._file_logger

        class _LoggerProxy:
            """代理 logger：所有 info/warn/error/debug 同时写 ROS2 和文件"""
            def __init__(self, ros2_logger, file_log):
                self._ros2 = ros2_logger
                self._file = file_log

            def info(self, msg, **kwargs):
                self._ros2.info(msg, **kwargs)
                self._file.info(msg)

            def warn(self, msg, **kwargs):
                self._ros2.warn(msg, **kwargs)
                self._file.warning(msg)

            def error(self, msg, **kwargs):
                self._ros2.error(msg, **kwargs)
                self._file.error(msg)

            def debug(self, msg, **kwargs):
                self._ros2.debug(msg, **kwargs)
                self._file.debug(msg)

            def __getattr__(self, name):
                return getattr(self._ros2, name)

        self._logger = _LoggerProxy(ros2_logger, file_logger)

    def _flog(self, level, msg):
        """双写日志：ROS2 logger + 文件 logger。
        
        当 self._logger 已经是代理对象时，只需调用 self._logger.info/warn/...
        即可自动双写。当代理尚未设置时，手动分别写入。
        
        Args:
            level: 'debug' / 'info' / 'warn' / 'error'
            msg: 日志消息字符串
        """
        # self._logger 已经是代理对象时，直接调用即可（代理内部会双写）
        if hasattr(self, '_logger'):
            log_method = {
                'debug': self._logger.debug,
                'info': self._logger.info,
                'warn': self._logger.warn,
                'error': self._logger.error,
            }.get(level, self._logger.info)
            log_method(msg)
            return

        # 回退：手动双写
        ros2_log = self.get_logger()
        {
            'debug': ros2_log.debug,
            'info': ros2_log.info,
            'warn': ros2_log.warn,
            'error': ros2_log.error,
        }.get(level, ros2_log.info)(msg)

        if self._file_logger is not None:
            {
                'debug': self._file_logger.debug,
                'info': self._file_logger.info,
                'warn': self._file_logger.warning,
                'error': self._file_logger.error,
            }.get(level, self._file_logger.info)(msg)

    def _check_map_and_fallback(self):
        # 销毁该定时器使其仅执行一次
        if hasattr(self, 'map_check_timer') and self.map_check_timer is not None:
            self.map_check_timer.cancel()
            self.destroy_timer(self.map_check_timer)
            self.map_check_timer = None

        if self.map_data is None:
            self._logger.info('启动后 1.5 秒内未接收到 ROS 话题地图数据，正在尝试从本地配置文件加载地图作为回退...')
            if self.map_file and os.path.exists(self.map_file):
                self._load_map_from_file(self.map_file)
            else:
                self._logger.warn(f'未找到本地地图文件: {self.map_file}，将继续等待 ROS 网格地图话题订阅发布...')

    # ================================================================
    #  开机自动启动逻辑
    # ================================================================
    def _check_auto_start_conditions(self):
        """每次收到传感器数据时检查是否所有数据就绪"""
        if self._auto_start_triggered:
            return
        if not self.auto_start:
            return
        # 只检查室内模式（室外由 _check_auto_mode 独立处理）
        if self.detected_mode == "OUTDOOR":
            return
        if self._auto_start_map_received and self._auto_start_scan_received and self._auto_start_odom_received:
            if self._auto_start_ready_time is None:
                self._auto_start_ready_time = self.get_clock().now()
                self._logger.info(
                    '[自动启动] 地图、激光扫描、里程计均已就绪，'
                    f'等待 {self.auto_start_delay:.1f}s 后自动触发初始位姿校准...'
                )

    def _try_auto_start_indoor(self):
        """在 indoor_loop 的 IDLE 状态中检查是否应该自动启动"""
        if not self.auto_start or self._auto_start_triggered:
            return
        if self._auto_start_ready_time is None:
            return
        if self.map_data is None or self.current_scan is None or self.current_odom is None:
            return

        # 等待延迟
        elapsed = (self.get_clock().now() - self._auto_start_ready_time).nanoseconds / 1e9
        if elapsed < self.auto_start_delay:
            return

        self._auto_start_triggered = True
        self._logger.info(
            f'[自动启动] 数据就绪 {elapsed:.1f}s，自动触发室内初始位姿校准...'
        )

        # 直接进入校准流程（模拟 _srv_start 的逻辑，但不检查数据）
        self.indoor_phase = IndoorPhase.BOOT_DELAY
        self.boot_start_time = self.get_clock().now()
        self.scan_buffer.clear()
        self.submap_ready = False
        self.candidates.clear()
        self.active_retry_count = 0

    # ================================================================
    #  快速模式：跳过旋转和运动探索，单帧采集+全局匹配+发布
    #  适用于开机后只需粗略初始位姿给 AMCL 的场景
    # ================================================================
    def _enter_quick_mode_collection(self):
        """快速模式入口：直接采集若干帧激光作为子图"""
        self._logger.info('[快速模式] 跳过旋转采集，直接累积静止帧用于全局匹配...')
        self.scan_buffer.clear()
        self.submap_ready = False
        self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP1

    def _load_map_from_file(self, yaml_path):
        if not os.path.exists(yaml_path):
            self._logger.error(f'地图配置文件不存在: {yaml_path}')
            return False
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            image_name = config.get('image', '')
            if not image_name:
                self._logger.error('地图配置文件未指定图像名称')
                return False
                
            img_path = os.path.join(os.path.dirname(yaml_path), image_name)
            if not os.path.exists(img_path):
                self._logger.error(f'地图图像文件不存在: {img_path}')
                return False
                
            import cv2
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                self._logger.error(f'读取地图图像失败: {img_path}')
                return False
                
            # 获取参数
            res = float(config.get('resolution', 0.05))
            origin = config.get('origin', [0.0, 0.0, 0.0])
            negate = int(config.get('negate', 0))
            occ_thresh = float(config.get('occupied_thresh', 0.65))
            free_thresh = float(config.get('free_thresh', 0.196))
            
            # ROS 2 栅格地图中，左下角为 (0,0)。而普通图像文件左上角为 (0,0)。
            # 需要上下翻转图像以与坐标系对齐。
            img_flipped = cv2.flip(img, 0)
            
            # 像素转换：
            # 黑色 0 -> p = 1.0 (占用)
            # 白色 255 -> p = 0.0 (空闲)
            # 计算概率 p
            p = (255.0 - img_flipped.astype(np.float32)) / 255.0
            
            map_data = np.full(img.shape, -1, dtype=np.int8)
            map_data[p > occ_thresh] = 100
            map_data[p < free_thresh] = 0
            
            from nav_msgs.msg import MapMetaData
            self.map_info = MapMetaData()
            self.map_info.resolution = res
            self.map_info.width = img.shape[1]
            self.map_info.height = img.shape[0]
            self.map_info.origin.position.x = float(origin[0])
            self.map_info.origin.position.y = float(origin[1])
            self.map_info.origin.position.z = float(origin[2])
            
            self.map_data = map_data
            self._build_likelihood()
            self._find_free_space()
            self._logger.info(f'从文件成功加载地图: {self.map_info.width}x{self.map_info.height} @ {res}m ({yaml_path})')
            return True
        except Exception as e:
            self._logger.error(f'从文件加载地图出错: {e}')
            return False

    # ================================================================
    #  ICP 扫描匹配相关方法
    # ================================================================
    def _scan_to_points(self, scan, apply_outlier_filter=False):
        """将 LaserScan 转换为 (N, 2) NumPy 点阵 (x, y in scan frame)
        apply_outlier_filter: 是否应用半径离群点过滤去除动态障碍物/杂点"""
        if scan is None:
            return np.empty((0, 2))
        ranges = scan.ranges
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        valid = (np.array(ranges) > scan.range_min) & (np.array(ranges) < scan.range_max)
        if not np.any(valid):
            return np.empty((0, 2))
        r_valid = np.array(ranges)[valid]
        a_valid = angles[valid]
        points = np.column_stack((r_valid * np.cos(a_valid), r_valid * np.sin(a_valid)))
        
        if apply_outlier_filter and self.scan_outlier_filter and len(points) > 10:
            points = self._filter_scan_outliers(points)
        return points
    
    def _filter_scan_outliers(self, points):
        """半径离群点过滤：去除邻域内邻居不足的孤立点（动态障碍物、杂点）
        输入: points (N,2) numpy array
        输出: 过滤后的 (M,2) numpy array"""
        if len(points) < self.scan_outlier_min_neighbors + 1:
            return points  # 点数太少，不过滤
        
        # 使用 KD-Tree 加速邻域搜索
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            # 查询每个点在 radius 内的邻居数
            counts = tree.query_ball_point(points, self.scan_outlier_radius, return_length=True)
            mask = np.array(counts) >= self.scan_outlier_min_neighbors
            n_removed = np.sum(~mask)
            if n_removed > 0:
                self._logger.debug(f'[离群点过滤] 移除 {n_removed}/{len(points)} 个离群点 (半径={self.scan_outlier_radius}m, 最少邻居={self.scan_outlier_min_neighbors})')
            return points[mask]
        except ImportError:
            # scipy 不可用，回退到简单距离阈值过滤
            self._logger.warn('[离群点过滤] scipy 未安装，使用简化过滤', throttle_duration_sec=10.0)
            # 简化版：去除距离超过中位数3倍标准差的点
            dists = np.sqrt(np.sum(points**2, axis=1))
            median_dist = np.median(dists)
            med_abs_dev = np.median(np.abs(dists - median_dist))
            if med_abs_dev > 0:
                mask = np.abs(dists - median_dist) < 3.0 * med_abs_dev * 1.4826
                return points[mask]
            return points

    def _temporal_consistency_filter(self, all_points, frame_ids):
        """
        多帧时序一致性过滤：剔除只在少数帧出现的动态障碍物（人、动物等）

        核心原理：
          - 静态墙壁/障碍物 → 多个不同帧的激光都会打到同一位置
          - 动态行人/动物 → 只在 1-2 帧出现在某位置，随后移走
          → 对每个点，检查其邻域内的点来自多少个不同帧，
            若少于 temporal_merge_min_frames 帧则剔除。

        输入：
          all_points: (N, 2) numpy array, 所有帧投影到参考帧系后的点坐标
          frame_ids:  (N,)  numpy array, 每个点所属的源帧索引 (0 ~ n_frames-1)
        输出：
          mask: (N,) boolean numpy array, True=保留, False=剔除
        """
        n_points = len(all_points)
        if n_points < self.temporal_merge_min_frames:
            self._logger.warn('[时序过滤] 总点数过少，跳过过滤')
            return np.ones(n_points, dtype=bool)

        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(all_points)
            # 对每个点查询半径内的邻居索引
            neighbors_list = tree.query_ball_point(
                all_points, self.temporal_merge_radius
            )

            # 统计每个点的邻居来自多少不同帧
            distinct_count = np.zeros(n_points, dtype=np.int32)
            for i in range(n_points):
                if len(neighbors_list[i]) == 0:
                    distinct_count[i] = 1  # 自邻域
                else:
                    distinct_count[i] = len(np.unique(frame_ids[neighbors_list[i]]))

            mask = distinct_count >= self.temporal_merge_min_frames
            n_kept = np.sum(mask)
            n_removed = n_points - n_kept

            self._logger.info(
                f'[时序过滤] 总点={n_points}, 保留={n_kept}({100*n_kept/max(1,n_points):.1f}%), '
                f'剔除动态/噪声={n_removed}, '
                f'(最少帧数={self.temporal_merge_min_frames}, 半径={self.temporal_merge_radius}m)'
            )

            # 按帧统计过滤效果
            unique_frames = np.unique(frame_ids)
            if len(unique_frames) <= 10:
                for fid in unique_frames:
                    fmask = frame_ids == fid
                    f_kept = np.sum(mask[fmask])
                    f_total = np.sum(fmask)
                    self._logger.debug(
                        f'    帧[{fid}]: 保留 {f_kept}/{f_total} '
                        f'({100*f_kept/max(1,f_total):.0f}%)'
                    )

            return mask

        except ImportError:
            self._logger.warn(
                '[时序过滤] scipy 未安装，回退到纯空间离群点过滤',
                throttle_duration_sec=10.0
            )
            # 回退：简单的空间离群点过滤
            if len(all_points) < 10:
                return np.ones(n_points, dtype=bool)

            # 对每个点找最近邻，若最近邻距离超过中位数的3倍则剔除
            dists = []
            for i in range(n_points):
                dx = all_points[:, 0] - all_points[i, 0]
                dy = all_points[:, 1] - all_points[i, 1]
                d = np.sqrt(dx*dx + dy*dy)
                d[i] = np.inf
                dists.append(np.min(d))
            dists = np.array(dists)
            med = np.median(dists)
            if med > 0:
                mask = dists < 3.0 * med
                return mask
            return np.ones(n_points, dtype=bool)

    def _icp_match(self, points1, points2, max_iter=15, tol_trans=0.01, tol_rot_deg=0.1):
        """
        简化的点对点 ICP 匹配
        输入：points1 (N,2) 参考帧点阵, points2 (M,2) 源帧点阵
        输出：(dx, dy, dyaw) 从 points2 到 points1 的变换 (points2 @ R + t -> points1)
        """
        if len(points1) < 3 or len(points2) < 3:
            return 0.0, 0.0, 0.0

        # 降采样
        step = max(1, len(points2) // 144)  # 降采样到约144点
        p2 = points2[::step]
        # 用角度有序特性做对应：直接按角度bin对应（两帧角度范围相同）
        # 更鲁棒做法：对 points1 每个点找 points2 最近邻（小数据集直接用暴力）
        p1 = points1

        T = np.eye(3)
        for i in range(max_iter):
            # 将 p2 用当前 T 变换到 p1 系
            p2_h = np.column_stack((p2, np.ones(len(p2))))
            p2_transformed = (T @ p2_h.T).T[:, :2]

            # 找对应点：对 p2_transformed 每个点，在 p1 中找最近邻
            # 暴力搜索（点数量少，可接受）
            dists = np.sum((p1[np.newaxis, :, :] - p2_transformed[:, np.newaxis, :]) ** 2, axis=2)
            idx = np.argmin(dists, axis=1)
            matched_p1 = p1[idx]

            # SVD 求解变换：p2 -> matched_p1
            centroid_p1 = np.mean(matched_p1, axis=0)
            centroid_p2 = np.mean(p2, axis=0)

            p1_centered = matched_p1 - centroid_p1
            p2_centered = p2 - centroid_p2

            H = p2_centered.T @ p1_centered
            try:
                U, S, Vt = np.linalg.svd(H)
            except np.linalg.LinAlgError:
                break
            R = Vt.T @ U.T
            # 处理反射矩阵
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            t = centroid_p1 - R @ centroid_p2

            T_new = np.eye(3)
            T_new[:2, :2] = R
            T_new[:2, 2] = t

            delta = np.linalg.norm(T_new - T)
            T = T_new
            if delta < tol_trans:
                break

        dx = T[0, 2]
        dy = T[1, 2]
        dyaw = math.atan2(T[1, 0], T[0, 0])
        return dx, dy, dyaw

    def _icp_align_scans(self, scan_list):
        """
        将多帧 LaserScan 对齐到第一帧参考系，返回累积变换列表
        输入：scan_list = [scan_msg1, scan_msg2, ...]
        输出：transforms = [(0,0,0), (dx1,dy1,dyaw1), (dx2,dy2,dyaw2), ...]
              每个变换将第 i 帧的点投影到第 0 帧参考系
        """
        if not scan_list:
            return []
        transforms = [(0.0, 0.0, 0.0)]  # 第一帧为单位变换
        prev_points = self._scan_to_points(scan_list[0])
        cum_T = np.eye(3)  # 累积变换：从当前帧到参考帧

        for i in range(1, len(scan_list)):
            curr_points = self._scan_to_points(scan_list[i])
            if len(prev_points) < 3 or len(curr_points) < 3:
                transforms.append(transforms[-1])  # 跳过，使用单位变换
                prev_points = curr_points
                continue

            dx, dy, dyaw = self._icp_match(prev_points, curr_points)
            # 从 prev 到 curr 的变换
            R = np.array([[math.cos(dyaw), -math.sin(dyaw)],
                          [math.sin(dyaw), math.cos(dyaw)]])
            T_step = np.eye(3)
            T_step[:2, :2] = R
            T_step[:2, 2] = [dx, dy]

            cum_T = cum_T @ np.linalg.inv(T_step)  # 累积：curr -> prev -> ... -> ref
            transforms.append((cum_T[0, 2], cum_T[1, 2], math.atan2(cum_T[1, 0], cum_T[0, 0])))
            prev_points = curr_points

        return transforms

    # ================================================================
    #  回调与基本数据读取
    # ================================================================
    def _map_cb(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self._build_likelihood()
        self._find_free_space()
        self._auto_start_map_received = True
        self._check_auto_start_conditions()
        self._logger.info(f'载入新网格地图: {msg.info.width}x{msg.info.height} @ {msg.info.resolution}m')

    def _build_likelihood(self):
        if self.map_data is None:
            return
        try:
            import cv2
            obs = (self.map_data == 100).astype(np.uint8)
            max_px = self.likelihood_max_dist / self.map_info.resolution
            dist = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            self.likelihood_field = np.clip(dist, 0, max_px).astype(np.float32) * self.map_info.resolution
            # ── 未知区域惩罚: 阻止优化器把扫描藏进灰色区域 ──
            self.likelihood_field[self.map_data == -1] = self.likelihood_max_dist
        except Exception as e:
            self._logger.error(f'似然场构建失败: {e}')


    def _score_points(self, points_odom, cx, cy, yaw):
        """向量化点云评分: O(N) with numpy, 替代逐束循环的 _score_scan.
        
        返回: (score, hit_rate, n_valid)
        """
        if self.likelihood_field is None or self.map_info is None:
            return -1e9, 0, 0
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        H, W = self.map_info.height, self.map_info.width

        c_y, s_y = math.cos(yaw), math.sin(yaw)
        # 变换到地图坐标系
        if points_odom.ndim == 2:
            mx = c_y * points_odom[:, 0] - s_y * points_odom[:, 1] + cx
            my = s_y * points_odom[:, 0] + c_y * points_odom[:, 1] + cy
        else:
            mx = np.array([c_y * points_odom[0] - s_y * points_odom[1] + cx])
            my = np.array([s_y * points_odom[0] + c_y * points_odom[1] + cy])

        ci = ((mx - ox) / res + 0.5).astype(np.int32)
        ri = ((my - oy) / res + 0.5).astype(np.int32)
        # ROS栅格: ri 越大 → y 越小, 需要翻转
        # ROS栅格: row 0 = y=0 (底部), 直接除res即可
        ri = ((my - oy) / res + 0.5).astype(np.int32)
        # ROS栅格: ri 越大 → y 越大, 无需翻转 (与离线NPZ一致)

        valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        nv = int(np.sum(valid))
        if nv < max(len(points_odom) * 0.10, 5):
            return -1e9, 0, nv

        dists = self.likelihood_field[ri[valid], ci[valid]]
        # Gaussian kernel 评分
        lf_score = float(np.mean(np.exp(-dists**2 / 0.045)))
        n_hit = int(np.sum(dists < 0.15))
        hit_rate = n_hit / nv
        return lf_score + hit_rate * 0.5, hit_rate, nv


    def _compute_wall_coverage(self, points_odom, cx, cy, yaw):
        """计算有效区域内的墙壁覆盖率 (排除灰色区域)"""
        if self.map_data is None:
            return 0.0, 0.0, 0.0
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        H, W = self.map_info.height, self.map_info.width

        c_y, s_y = math.cos(yaw), math.sin(yaw)
        mx = c_y * points_odom[:, 0] - s_y * points_odom[:, 1] + cx
        my = s_y * points_odom[:, 0] + c_y * points_odom[:, 1] + cy
        ci = ((mx - ox) / res + 0.5).astype(np.int32)
        ri = ((my - oy) / res + 0.5).astype(np.int32)
        valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        cells = self.map_data[ri[valid], ci[valid]]
        valid_cell = (cells != -1)
        n_valid = int(np.sum(valid_cell))
        n_total = len(points_odom)
        if n_valid < 5:
            return 0.0, float(n_valid) / n_total, 0.0
        w = int(np.sum(cells[valid_cell] == 100))
        f = int(np.sum(cells[valid_cell] == 0))
        return float(w) / (w + f), float(n_valid) / n_total, float(f) / (w + f)

    def _find_free_space(self):
        if self.map_data is None:
            return
        free = (self.map_data == 0)
        rows, cols = np.where(free)
        self.free_space_indices = np.stack([rows, cols], axis=1)

    def _scan_cb(self, msg):
        self.current_scan = msg
        self._auto_start_scan_received = True
        self._check_auto_start_conditions()
        
        # 首次收到 scan 时打印坐标系诊断信息
        if not hasattr(self, '_scan_frame_logged'):
            self._scan_frame_logged = True
            self._logger.info(f'═══ 坐标系诊断 ═══')
            self._logger.info(f'  /scan frame_id: "{msg.header.frame_id}"')
            self._logger.info(f'  地图 frame:     "{self.map_frame}"')
            self._logger.info(f'  FOV: [{math.degrees(msg.angle_min):.1f}°, {math.degrees(msg.angle_max):.1f}°]')
            self._logger.info(f'  角度分辨率: {math.degrees(msg.angle_increment):.3f}°')
            self._logger.info(f'  测距范围: [{msg.range_min}, {msg.range_max}]m')
            self._logger.info(f'  总光束: {len(msg.ranges)} 束')
            self._logger.info(f'═══ 请确认 /scan frame_id 与 base_footprint 一致，雷达安装偏移已在 URDF/TF 中标定 ═══')
        
        # 当处于子图累积状态时，将点云与此时的里程计匹配并缓存
        is_collecting = False
        if self.indoor_phase == IndoorPhase.COLLECTING_SUBMAP1:
            is_collecting = True
        elif self.indoor_phase == IndoorPhase.ROTATING_360:
            is_collecting = True
        elif self.indoor_phase == IndoorPhase.COLLECTING_SUBMAP2 and hasattr(self, '_submap2_collecting'):
            is_collecting = True

        # ── 被动模式: 循环缓冲区持续缓存所有扫描帧 ──
        if self.indoor_phase in (IndoorPhase.PASSIVE_COLLECTING, IndoorPhase.PASSIVE_MATCHING):
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.passive_scan_buffer.append((msg, ts))
            # 循环缓冲区: 超出上限时丢最旧的
            if len(self.passive_scan_buffer) > self.passive_buffer_max:
                self.passive_scan_buffer = self.passive_scan_buffer[-self.passive_buffer_max:]

        if is_collecting:
            if not self.submap_ready:
                # 只存储扫描消息和时间戳，不依赖 odom 坐标
                ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.scan_buffer.append((msg, ts))
                if len(self.scan_buffer) == 1:
                    phase_name = '旋转采集' if self.indoor_phase == IndoorPhase.ROTATING_360 else '子图构建'
                    self._logger.info(f'[{phase_name}] 开始累积扫描帧，使用 ICP 帧间匹配拼接（不依赖 odom）...')
                if len(self.scan_buffer) % 10 == 0:
                    self._logger.info(f'[子图构建] 已累积 {len(self.scan_buffer)} 帧')
                if len(self.scan_buffer) >= self.submap_scan_count:
                    self.submap_ready = True

    def _odom_cb(self, msg):
        curr_pos = msg.pose.pose.position
        curr_yaw = self._quat_to_yaw(msg.pose.pose.orientation)
        
        # 丢弃 NaN 数据
        if math.isnan(curr_pos.x) or math.isnan(curr_pos.y) or math.isnan(curr_yaw):
            return
        
        # 里程计数据验证：检测异常跳变（基于相对增量，与绝对值无关）
        if self.last_odom_pose is not None:
            last_x, last_y, last_yaw = self.last_odom_pose
            dx = curr_pos.x - last_x
            dy = curr_pos.y - last_y
            dist = math.sqrt(dx*dx + dy*dy)
            dyaw = abs(self._norm_angle(curr_yaw - last_yaw))
            
            # 如果单帧增量超过阈值，丢弃该帧数据
            if dist > self.max_odom_delta:
                self._logger.warn(f'[里程计验证] 检测到异常跳变: Δdist={dist:.2f}m，丢弃该帧')
                return
            # 里程计航向不可能单帧跳变超过90度
            if dyaw > math.radians(90.0):
                self._logger.warn(f'[里程计验证] 检测到航向跳变: {math.degrees(dyaw):.1f}°，丢弃该帧')
                return
        
        self.last_odom_pose = (curr_pos.x, curr_pos.y, curr_yaw)
        self.current_odom = msg
        self._auto_start_odom_received = True
        self._check_auto_start_conditions()
        # 实时计算并发布纯里程计推算的预计 map 坐标
        if self.gt_received and self.gt_pose is not None and self.gt_ref_odom is not None:
            curr_pos = msg.pose.pose.position
            curr_yaw = self._quat_to_yaw(msg.pose.pose.orientation)
            
            # 计算 odom 坐标系下的相对位移
            ref_x, ref_y, ref_yaw = self.gt_ref_odom
            dx = curr_pos.x - ref_x
            dy = curr_pos.y - ref_y
            
            # 旋转到起点 odom 坐标系
            rel_x = dx * math.cos(ref_yaw) + dy * math.sin(ref_yaw)
            rel_y = -dx * math.sin(ref_yaw) + dy * math.cos(ref_yaw)
            rel_yaw = self._norm_angle(curr_yaw - ref_yaw)
            
            # 递推到 Map 坐标系
            gt_x, gt_y, gt_yaw = self.gt_pose
            est_x = gt_x + rel_x * math.cos(gt_yaw) - rel_y * math.sin(gt_yaw)
            est_y = gt_y + rel_x * math.sin(gt_yaw) + rel_y * math.cos(gt_yaw)
            est_yaw = self._norm_angle(gt_yaw + rel_yaw)
            
            # 发布调试位姿
            est_msg = PoseStamped()
            est_msg.header.stamp = msg.header.stamp
            est_msg.header.frame_id = self.map_frame
            est_msg.pose.position = Point(x=est_x, y=est_y, z=0.0)
            qz = math.sin(est_yaw / 2.0)
            qw = math.cos(est_yaw / 2.0)
            est_msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            self.odom_est_pose_pub.publish(est_msg)

    def _amcl_cb(self, msg):
        # 记录 AMCL 位姿供偏差监控
        amcl_pos = msg.pose.pose.position
        amcl_yaw = self._quat_to_yaw(msg.pose.pose.orientation)
        self._record_amcl_pose(amcl_pos.x, amcl_pos.y, amcl_yaw)
            # 1. 获取当前定位点 (x, y, yaw)
            amcl_pos = msg.pose.pose.position
            amcl_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

            # 2. 计算此时由 odom 递推出来的理论位姿
            curr_pos = self.current_odom.pose.pose.position
            curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
            
            ref_x, ref_y, ref_yaw = self.gt_ref_odom
            dx = curr_pos.x - ref_x
            dy = curr_pos.y - ref_y
            
            rel_x = dx * math.cos(ref_yaw) + dy * math.sin(ref_yaw)
            rel_y = -dx * math.sin(ref_yaw) + dy * math.cos(ref_yaw)
            rel_yaw = self._norm_angle(curr_yaw - ref_yaw)
            
            gt_x, gt_y, gt_yaw = self.gt_pose
            est_x = gt_x + rel_x * math.cos(gt_yaw) - rel_y * math.sin(gt_yaw)
            est_y = gt_y + rel_x * math.sin(gt_yaw) + rel_y * math.cos(gt_yaw)
            est_yaw = self._norm_angle(gt_yaw + rel_yaw)

            # 3. 计算两者在 Map 坐标系下的绝对偏差
            err_dist = math.sqrt((amcl_pos.x - est_x)**2 + (amcl_pos.y - est_y)**2)
            err_yaw = abs(self._norm_angle(amcl_yaw - est_yaw))

            # 4. 频率限幅输出，避免刷屏 (1Hz 打印一次)
            if not hasattr(self, '_last_print_time'):
                self._last_print_time = 0.0
            now_sec = self.get_clock().now().nanoseconds / 1e9
            if now_sec - self._last_print_time > 1.0:
                self._logger.info(
                    f'[偏差对齐监测] 里程计递推: ({est_x:.2f}, {est_y:.2f}, {math.degrees(est_yaw):.1f}°), '
                    f'算法实际定位: ({amcl_pos.x:.2f}, {amcl_pos.y:.2f}, {math.degrees(amcl_yaw):.1f}°), '
                    f'累计偏差: 距离={err_dist:.3f}m, 角度={math.degrees(err_yaw):.1f}°'
                )
                self._last_print_time = now_sec

    def _initialpose_cb(self, msg):
        """若节点处于空闲状态，将手动标记作为 Ground Truth 用于偏差校准与调试"""
        if self.indoor_phase == IndoorPhase.IDLE:
            pos = msg.pose.pose.position
            yaw = self._quat_to_yaw(msg.pose.pose.orientation)
            
            if self.gt_received and self.gt_pose is not None:
                # 已经有真值起点，这次点击代表用户在终点进行二次人工校对
                if self.current_odom is not None and self.gt_ref_odom is not None:
                    curr_pos = self.current_odom.pose.pose.position
                    curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
                    
                    ref_x, ref_y, ref_yaw = self.gt_ref_odom
                    dx = curr_pos.x - ref_x
                    dy = curr_pos.y - ref_y
                    
                    rel_x = dx * math.cos(ref_yaw) + dy * math.sin(ref_yaw)
                    rel_y = -dx * math.sin(ref_yaw) + dy * math.cos(ref_yaw)
                    rel_yaw = self._norm_angle(curr_yaw - ref_yaw)
                    
                    gt_x, gt_y, gt_yaw = self.gt_pose
                    est_x = gt_x + rel_x * math.cos(gt_yaw) - rel_y * math.sin(gt_yaw)
                    est_y = gt_y + rel_x * math.sin(gt_yaw) + rel_y * math.cos(gt_yaw)
                    est_yaw = self._norm_angle(gt_yaw + rel_yaw)
                    
                    err_dist = math.sqrt((pos.x - est_x)**2 + (pos.y - est_y)**2)
                    err_yaw = abs(self._norm_angle(yaw - est_yaw))
                    
                    self._logger.info(
                        f'[人工终点核对] 第二次手动标记坐标: ({pos.x:.2f}, {pos.y:.2f}, {math.degrees(yaw):.1f}°), '
                        f'里程计递推理论坐标: ({est_x:.2f}, {est_y:.2f}, {math.degrees(est_yaw):.1f}°), '
                        f'实测累计偏差：距离={err_dist:.3f}m, 角度={math.degrees(err_yaw):.1f}°'
                    )
            else:
                # 初次点击，设定真值起点
                if self.current_odom is not None:
                    opos = self.current_odom.pose.pose.position
                    oyaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
                    self.gt_pose = (pos.x, pos.y, yaw)
                    self.gt_ref_odom = (opos.x, opos.y, oyaw)
                    self.gt_received = True
                    self._logger.info(f'[校对工具] 标记真值起点: Map({pos.x:.2f}, {pos.y:.2f}, {math.degrees(yaw):.1f}°), '
                                           f'记录基准 Odom({opos.x:.2f}, {opos.y:.2f})。后续将实时输出 odom 递推偏差。')
                    
                    # 如果之前有校准结果，自动对比偏差
                    if self.last_calibrated_pose is not None:
                        cal_x, cal_y, cal_yaw = self.last_calibrated_pose
                        err_x = pos.x - cal_x
                        err_y = pos.y - cal_y
                        err_dist = math.sqrt(err_x**2 + err_y**2)
                        err_yaw = abs(self._norm_angle(yaw - cal_yaw))
                        self._logger.info(
                            f'[校准精度对比] '
                            f'校准结果: ({cal_x:.2f}, {cal_y:.2f}, {math.degrees(cal_yaw):.1f}°), '
                            f'手动标记: ({pos.x:.2f}, {pos.y:.2f}, {math.degrees(yaw):.1f}°), '
                            f'偏差: Δx={err_x:.3f}m, Δy={err_y:.3f}m, 距离={err_dist:.3f}m, 角度={math.degrees(err_yaw):.1f}°'
                        )

    def _setup_rtk_sub(self, qos):
        try:
            from robots_dog_msgs.msg import UniRtkPvh
            self.rtk_sub = self.create_subscription(UniRtkPvh, self.rtk_topic, self._rtk_cb, qos)
        except ImportError:
            try:
                parts = self.rtk_topic_type.rsplit('.', 1)
                mod = importlib.import_module(parts[0])
                cls = getattr(mod, parts[1])
                self.rtk_sub = self.create_subscription(cls, self.rtk_topic, self._rtk_cb, qos)
            except Exception as e:
                self._logger.warning(f'无法动态订阅 RTK 话题: {e}')
                self.rtk_sub = None

    def _rtk_cb(self, msg):
        self.current_rtk = msg

    def _gps_cb(self, msg):
        self.current_gps = msg

    # ================================================================
    #  核心：子图拼接与合成
    # ================================================================
    # ================================================================
    #  旋转360°全覆盖采集
    # ================================================================
    def _do_rotation_collection(self):
        """旋转360°过程中持续采集扫描帧（由_scan_cb统一收集），确保子图覆盖全闭环空间"""
        if self.current_odom is None:
            self._logger.warn('[旋转采集] 里程计信号丢失，放弃旋转，直接构建子图。')
            self.cmd_vel_pub.publish(Twist())
            self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP1
            return
        
        curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
        # 累积旋转角
        delta_yaw = abs(self._norm_angle(curr_yaw - self.rotation_start_yaw))
        self.rotation_start_yaw = curr_yaw
        self.rotation_accumulated += delta_yaw
        
        # 防卡死超时保护（最多旋转90秒，即约27rad @ 0.3rad/s）
        elapsed = (self.get_clock().now() - self.rotation_start_time).nanoseconds / 1e9
        if elapsed > 90.0:
            self._logger.warn(f'[旋转采集] 超时 ({elapsed:.1f}s)，停止旋转。已采集 {len(self.scan_buffer)} 帧')
            self.cmd_vel_pub.publish(Twist())
            self.indoor_phase = IndoorPhase.ACTIVE_MULTISTEP
            self._multistep_frame_idx = 0
            return
        
        # 达到目标帧数则提前停止
        if self.submap_ready:
            self._logger.info(f'[旋转采集] 已达成目标帧数 ({len(self.scan_buffer)}/{self.submap_scan_count})')
            self.cmd_vel_pub.publish(Twist())
            self.indoor_phase = IndoorPhase.ACTIVE_MULTISTEP
            self._multistep_frame_idx = 0
            return
        
        # 是否完成全角度旋转
        if self.rotation_accumulated >= self.rotation_total_rad:
            self._logger.info(f'[旋转采集] 完成 {math.degrees(self.rotation_accumulated):.0f}° 全覆盖采集')
            self.cmd_vel_pub.publish(Twist())
            self.submap_ready = True
            self.indoor_phase = IndoorPhase.ACTIVE_MULTISTEP
            self._multistep_frame_idx = 0
            return
        
        # 碰撞安全检查
        if self.current_scan is not None:
            safe = self._check_local_direction_safety(0.0, 0.3)
            if not safe:
                self._logger.warn('[旋转采集] 前方检测到障碍物，终止旋转采集。')
                self.cmd_vel_pub.publish(Twist())
                self.indoor_phase = IndoorPhase.ACTIVE_MULTISTEP
                self._multistep_frame_idx = 0
                return
        
        # 持续旋转
        cmd = Twist()
        cmd.angular.z = float(self.rotation_angular_vel)
        self.cmd_vel_pub.publish(cmd)
    
    def _build_submap(self):
        """利用缓存的 scan_buffer，使用 ICP 帧间匹配合成为高精度激光帧"""
        if not self.scan_buffer:
            return None
        
        # 提取扫描消息列表
        scans = [item[0] for item in self.scan_buffer]
        num_beams = len(scans[0].ranges)
        
        if self.use_icp_for_submap:
            # 使用 ICP 计算帧间变换
            self._logger.info(f'[子图构建] 使用 ICP 帧间匹配拼接 {len(scans)} 帧...')
            # 计算每帧到第0帧的变换
            transforms = [(0.0, 0.0, 0.0)]  # 第0帧单位变换
            for i in range(1, len(scans)):
                pts_prev = self._scan_to_points(scans[i-1], apply_outlier_filter=True)
                pts_curr = self._scan_to_points(scans[i], apply_outlier_filter=True)
                if len(pts_prev) < 3 or len(pts_curr) < 3:
                    transforms.append(transforms[-1])  # 跳过
                    continue
                dx, dy, dyaw = self._icp_match(pts_prev, pts_curr)
                # 累积变换：从帧 i 到帧 0
                prev_dx, prev_dy, prev_dyaw = transforms[-1]
                cum_dx = prev_dx + dx * math.cos(prev_dyaw) - dy * math.sin(prev_dyaw)
                cum_dy = prev_dy + dx * math.sin(prev_dyaw) + dy * math.cos(prev_dyaw)
                cum_dyaw = self._norm_angle(prev_dyaw + dyaw)
                transforms.append((cum_dx, cum_dy, cum_dyaw))
            
            self._logger.info(f'[子图构建] ICP 变换计算完成: {transforms}')
        else:
            # 保留原有 odom 方式（作为备用）
            transforms = []
            for scan, sx, sy, syaw in self.scan_buffer:
                if transforms:
                    prev_rx, prev_ry, prev_ryaw = self.scan_buffer[len(transforms)][1:]
                    dx = sx - prev_rx
                    dy = sy - prev_ry
                    rel_x = dx * math.cos(prev_ryaw) + dy * math.sin(prev_ryaw)
                    rel_y = -dx * math.sin(prev_ryaw) + dy * math.cos(prev_ryaw)
                    rel_yaw = self._norm_angle(syaw - prev_ryaw)
                    # 累积到第一帧
                    prev_cum_dx, prev_cum_dy, prev_cum_dyaw = transforms[-1]
                    cum_dx = prev_cum_dx + rel_x * math.cos(prev_cum_dyaw) - rel_y * math.sin(prev_cum_dyaw)
                    cum_dy = prev_cum_dy + rel_x * math.sin(prev_cum_dyaw) + rel_y * math.cos(prev_cum_dyaw)
                    cum_dyaw = self._norm_angle(prev_cum_dyaw + rel_yaw)
                    transforms.append((cum_dx, cum_dy, cum_dyaw))
                else:
                    transforms.append((0.0, 0.0, 0.0))
        
        # ────── 多帧时序一致性合并（带动态物体过滤）──────
        if self.temporal_merge_enabled and len(scans) >= self.temporal_merge_min_frames:
            self._logger.info(
                f'[子图构建] 启用多帧时序一致性过滤 '
                f'(最少帧数={self.temporal_merge_min_frames}, 半径={self.temporal_merge_radius}m)...'
            )
            # 步骤1: 收集所有帧的有效点，带帧ID标签
            all_projected = []   # [(px, py), ...]
            all_frame_ids = []   # [frame_idx, ...]

            for i, (scan, _) in enumerate(self.scan_buffer):
                rel_x, rel_y, rel_yaw = transforms[i]
                for j in range(num_beams):
                    r = scan.ranges[j]
                    if not (scans[0].range_min < r < scans[0].range_max):
                        continue
                    beam_angle = scans[0].angle_min + j * scans[0].angle_increment
                    lx = r * math.cos(beam_angle)
                    ly = r * math.sin(beam_angle)
                    # 投影到参考帧系
                    px = rel_x + lx * math.cos(rel_yaw) - ly * math.sin(rel_yaw)
                    py = rel_y + lx * math.sin(rel_yaw) + ly * math.cos(rel_yaw)
                    all_projected.append((px, py))
                    all_frame_ids.append(i)

            if len(all_projected) < 10:
                self._logger.warn('[子图构建] 投影后有效点过少，回退到传统合并')
            else:
                proj_pts = np.array(all_projected)
                frame_ids = np.array(all_frame_ids)

                # 步骤2: 时序一致性过滤（剔除只在少数帧出现的动态点）
                keep_mask = self._temporal_consistency_filter(proj_pts, frame_ids)
                static_pts = proj_pts[keep_mask]

                self._logger.info(
                    f'[子图构建] 时序过滤后保留 {len(static_pts)}/{len(proj_pts)} 个静态点 '
                    f'({100*len(static_pts)/max(1,len(proj_pts)):.1f}%)'
                )

                # 步骤3: 将过滤后的静态点转换为 LaserScan (极坐标 bin + 取最近)
                merged_ranges = np.full(num_beams, scans[0].range_max, dtype=np.float32)
                for (px, py) in static_pts:
                    r_proj = math.sqrt(px*px + py*py)
                    theta_proj = math.atan2(py, px)
                    bin_idx = int(round((theta_proj - scans[0].angle_min) / scans[0].angle_increment))
                    if 0 <= bin_idx < num_beams:
                        if r_proj < merged_ranges[bin_idx]:
                            merged_ranges[bin_idx] = r_proj

                # 统计各角度bin填充情况
                filled_bins = np.sum(merged_ranges < scans[0].range_max)
                self._logger.info(
                    f'[子图构建] 合成 Scan: {filled_bins}/{num_beams} 个角度 bin 被填充 '
                    f'({100*filled_bins/num_beams:.1f}%)'
                )
        else:
            # ────── 传统合并（无时序过滤）──────
            merged_ranges = np.full(num_beams, scans[0].range_max, dtype=np.float32)

            for i, (scan, _) in enumerate(self.scan_buffer):
                rel_x, rel_y, rel_yaw = transforms[i]
                for j in range(num_beams):
                    r = scan.ranges[j]
                    if not (scans[0].range_min < r < scans[0].range_max):
                        continue
                    beam_angle = scans[0].angle_min + j * scans[0].angle_increment
                    lx = r * math.cos(beam_angle)
                    ly = r * math.sin(beam_angle)
                    px = rel_x + lx * math.cos(rel_yaw) - ly * math.sin(rel_yaw)
                    py = rel_y + lx * math.sin(rel_yaw) + ly * math.cos(rel_yaw)
                    r_proj = math.sqrt(px*px + py*py)
                    theta_proj = math.atan2(py, px)
                    bin_idx = int(round((theta_proj - scans[0].angle_min) / scans[0].angle_increment))
                    if 0 <= bin_idx < num_beams:
                        if r_proj < merged_ranges[bin_idx]:
                            merged_ranges[bin_idx] = r_proj

        # 封装成合成 LaserScan
        composite_scan = LaserScan()
        composite_scan.header.stamp = scans[0].header.stamp
        composite_scan.header.frame_id = scans[0].header.frame_id
        composite_scan.angle_min = scans[0].angle_min
        composite_scan.angle_max = scans[0].angle_max
        composite_scan.angle_increment = scans[0].angle_increment
        composite_scan.range_min = scans[0].range_min
        composite_scan.range_max = scans[0].range_max
        composite_scan.ranges = merged_ranges.tolist()
        
        self._logger.info(
            f'[子图构建] {len(self.scan_buffer)} 帧合并完成 '
            f'(ICP: {self.use_icp_for_submap}, 时序过滤: {self.temporal_merge_enabled})'
        )
        
        # 发布可视化
        self.submap_scan_pub.publish(composite_scan)
        
        return composite_scan

    # ================================================================
    #  室内校准核心状态机定时循环
    # ================================================================
    def _indoor_loop(self):
        if self.indoor_phase == IndoorPhase.IDLE:
            # 检查开机自动启动条件
            self._try_auto_start_indoor()
            return
            
        elif self.indoor_phase == IndoorPhase.BOOT_DELAY:
            # 静止 2 秒确保传感器和驱动队列填充完毕
            if (self.get_clock().now() - self.boot_start_time).nanoseconds > 2.0 * 1e9:
                self._logger.info('[状态机] 静止就绪。')
                if self.quick_mode:
                    self._enter_quick_mode_collection()
                elif self.rotation_enabled and self.current_odom is not None:
                    self._logger.info(f'[状态机] 开始旋转 {math.degrees(self.rotation_total_rad):.0f}° 全覆盖采集...')
                    self.scan_buffer.clear()
                    self.submap_ready = False
                    self.rotation_start_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
                    self.rotation_accumulated = 0.0
                    self.rotation_start_time = self.get_clock().now()
                    self.indoor_phase = IndoorPhase.ROTATING_360
                else:
                    self._logger.info('[状态机] 旋转已禁用或无里程计，直接构建第一子图 Submap 1...')
                    self.scan_buffer.clear()
                    self.submap_ready = False
                    self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP1
                    
        elif self.indoor_phase == IndoorPhase.ROTATING_360:
            self._do_rotation_collection()
                
        elif self.indoor_phase == IndoorPhase.COLLECTING_SUBMAP1:
            if self.submap_ready:
                self.submap1 = self._build_submap()
                self.scan_buffer.clear()
                self.submap_ready = False
                self._logger.info('[状态机] Submap 1 构建完成，开始分层粗精两阶段全局评分...')
                self.indoor_phase = IndoorPhase.ROUGH_MATCHING
                
        elif self.indoor_phase == IndoorPhase.ROUGH_MATCHING:
            self._do_hierarchical_matching()
            
        elif self.indoor_phase == IndoorPhase.SELECTING_ACTIVE_MOTION:
            self._do_active_motion_selection()
            
        elif self.indoor_phase == IndoorPhase.MOVING:
            self._do_control_loop_and_avoidance()
            
        elif self.indoor_phase == IndoorPhase.COLLECTING_SUBMAP2:
            # 停止发布速度并等待 0.5s 使车身稳定
            self.cmd_vel_pub.publish(Twist())
            if not hasattr(self, '_submap2_wait_start'):
                self._submap2_wait_start = self.get_clock().now()
                self._logger.info('[状态机] 停止运动，等待车身静止稳定 (0.5s)...')
            else:
                elapsed = (self.get_clock().now() - self._submap2_wait_start).nanoseconds / 1e9
                if elapsed > 0.5:
                    if not hasattr(self, '_submap2_collecting'):
                        self._submap2_collecting = True
                        self.scan_buffer.clear()
                        self.submap_ready = False
                        self._logger.info('[状态机] 车身已稳定，开始累积 Submap 2 雷达数据...')
                    else:
                        if self.submap_ready:
                            self.submap2 = self._build_submap()
                            self.scan_buffer.clear()
                            self.submap_ready = False
                            delattr(self, '_submap2_wait_start')
                            delattr(self, '_submap2_collecting')
                            self._logger.info('[状态机] Submap 2 构建完成，开始位姿传播与重评分...')
                            self.indoor_phase = IndoorPhase.FILTERING
                    
        elif self.indoor_phase == IndoorPhase.FILTERING:
            if self.motion_verification_enabled and not hasattr(self, '_motion_verified'):
                self._verify_motion_by_scan_matching()
                self._motion_verified = True
            self._do_filtering_and_propagation()
            if hasattr(self, '_motion_verified'):
                delattr(self, '_motion_verified')

        # ── 被动模式: 持续缓存扫描帧 ──
        elif self.indoor_phase == IndoorPhase.PASSIVE_COLLECTING:
            pass  # 扫描帧由 _scan_cb 自动缓存到 passive_scan_buffer

        elif self.indoor_phase == IndoorPhase.PASSIVE_MATCHING:
            self._do_passive_matching()

        elif self.indoor_phase == IndoorPhase.ACTIVE_MULTISTEP:
            self._do_multistep_matching()

    # ================================================================
    #  核心步骤 3：改进的网格全局搜索 + 精细局部搜索
    # ================================================================
    def _do_hierarchical_matching(self):
        if self.likelihood_field is None or self.map_data is None:
            self._logger.error('地图似然场尚未加载，重新等待...')
            self.indoor_phase = IndoorPhase.BOOT_DELAY
            return

        t0 = time.time()
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        mw = self.map_info.width * res
        mh = self.map_info.height * res

        # 从 submap1 提取点云
        submap_pts = self._scan_to_points(self.submap1)
        if len(submap_pts) < 10:
            self._logger.error('Submap1 点云为空')
            self.indoor_phase = IndoorPhase.BOOT_DELAY
            return

        # 降采样到 ~800 点用于快速搜索
        ds = max(1, len(submap_pts) // 800)
        pts_ds = submap_pts[::ds]

        # ── Phase 1: 网格全局搜索 (1.5m步长, 10°角步长) ──
        coarse_step = 1.5; angle_step = 10.0
        n_angles = int(360.0 / angle_step)
        xs = np.arange(ox + 2, ox + mw - 2, coarse_step)
        ys = np.arange(oy + 2, oy + mh - 2, coarse_step)

        self._logger.info(f'[全局网格搜索] {len(xs)}x{len(ys)}x{n_angles} = {len(xs)*len(ys)*n_angles} 次评估...')
        
        all_scores = []
        for ax in xs:
            for ay in ys:
                for adeg in range(n_angles):
                    sc, _, _ = self._score_points(pts_ds, ax, ay, math.radians(adeg * angle_step))
                    if sc > -1e8:
                        all_scores.append((sc, ax, ay, adeg * angle_step))

        all_scores.sort(key=lambda x: x[0], reverse=True)
        self._logger.info(f'  粗搜: {len(all_scores)} 个有效评分')

        # NMS 去冗余
        nms_candidates = []
        for sc, ax, ay, ad in all_scores:
            dup = any(math.sqrt((ax-cx)**2 + (ay-cy)**2) < 1.5
                      and abs(ad - ca) < 20 for _, cx, cy, ca in nms_candidates)
            if not dup:
                nms_candidates.append((sc, ax, ay, ad))
                if len(nms_candidates) >= self.top_n * 2:
                    break

        # ── Phase 2: 精细局部搜索 (±2.5m, 0.3m步长, ±15°, 3°步长) ──
        self._logger.info(f'  精搜: 对 Top-{min(3, len(nms_candidates))} 候选做局部搜索...')
        fine_results = []
        for rank, (_, hx, hy, had) in enumerate(nms_candidates[:3]):
            hyaw = math.radians(had)
            for dx in np.arange(-2.5, 2.51, 0.3):
                for dy in np.arange(-2.5, 2.51, 0.3):
                    for da in range(-15, 16, 3):
                        ax, ay = hx + dx, hy + dy
                        ayaw = hyaw + math.radians(da)
                        sc, _, _ = self._score_points(pts_ds, ax, ay, ayaw)
                        fine_results.append((sc, ax, ay, ayaw))

        fine_results.sort(key=lambda x: x[0], reverse=True)

        # 去冗余 + 概率归一化
        unique_candidates = []
        for sc, x, y, yaw in fine_results:
            redundant = any(math.sqrt((x-u[1])**2+(y-u[2])**2) < 0.3
                          and abs(self._norm_angle(yaw-u[3])) < math.radians(20)
                          for u in unique_candidates)
            if not redundant:
                unique_candidates.append((sc, x, y, yaw))
                if len(unique_candidates) >= self.top_n:
                    break

        scores_arr = np.array([u[0] for u in unique_candidates])
        max_sc = np.max(scores_arr)
        exp_scores = np.exp(scores_arr - max_sc)
        normalized_probs = exp_scores / np.sum(exp_scores)

        self.candidates = [(float(normalized_probs[i]), u[1], u[2], u[3])
                          for i, u in enumerate(unique_candidates)]

        elapsed = time.time() - t0
        self._logger.info(f'[网格匹配] 完成。耗时: {elapsed:.2f}s。Top-N 候选:')
        for i, (prob, x, y, yaw) in enumerate(self.candidates):
            wall_cov, cov, _ = self._compute_wall_coverage(submap_pts, x, y, yaw)
            self._logger.info(f'  #{i}: 概率={prob:.3f}, ({x:.2f},{y:.2f},{math.degrees(yaw):.1f}deg) '
                            f'wall={100*wall_cov:.0f}% cov={100*cov:.0f}%')

        self._publish_candidates()

        if self.quick_mode:
            self._handle_quick_mode_result(submap_pts)
        else:
            self.indoor_phase = IndoorPhase.SELECTING_ACTIVE_MOTION


    def _handle_quick_mode_result(self, scan_pts):
        """快速模式: 评估匹配质量后发布 (含不确定度信息)"""
        best_prob, best_x, best_y, best_yaw = self.candidates[0]
        second_prob = self.candidates[1][0] if len(self.candidates) > 1 else 0.0

        # 计算实际墙壁覆盖率
        wall_cov, coverage, _ = self._compute_wall_coverage(scan_pts, best_x, best_y, best_yaw)
        self._last_match_quality = {'wall_cov': wall_cov, 'coverage': coverage, 'score': best_prob}

        is_unique = best_prob > 0.5 and (second_prob == 0.0 or best_prob / max(second_prob, 1e-9) >= 1.5)

        if is_unique and wall_cov > 0.50:
            self._logger.info(f'[快速模式] 匹配唯一且可靠 (wall={100*wall_cov:.0f}%), 发布')
            self._publish_and_finish(best_x, best_y, best_yaw)
        else:
            self._logger.warn(
                f'[快速模式] 匹配质量: unique={is_unique} wall={100*wall_cov:.0f}% cov={100*coverage:.0f}%')
            self._logger.warn('[快速模式] 发布最佳结果供 AMCL 初步收敛，偏差监控将追踪后续对齐')
            self._publish_and_finish(best_x, best_y, best_yaw)

    # ================================================================
    #  多步递推匹配 (主动+被动共用)
    #  逐帧ICP → 预测 → 局部搜索 → sigma衰减 → 收敛
    # ================================================================
    def _do_multistep_matching(self):
        """逐帧递推匹配: 首帧全局搜索, 后续帧 ICP+局部搜索"""
        if self.likelihood_field is None or self.map_data is None:
            self.indoor_phase = IndoorPhase.BOOT_DELAY
            return

        scans = [item[0] for item in self.scan_buffer]
        n_frames = len(scans)
        if n_frames < 3:
            self._logger.warn('[多步递推] 帧数不足, 回退到合并匹配')
            self.indoor_phase = IndoorPhase.ROUGH_MATCHING
            return

        t0 = time.time()
        total_wall_hits = 0
        total_valid = 0

        # 初始化
        mu = None        # (x, y, yaw) 当前最佳估计 in map frame
        sigma = 5.0      # 位置不确定度 (m)
        sigma_angle = 20.0
        min_sigma = 0.5
        decay = 0.75
        step = max(1, n_frames // 20)  # 最多处理20帧, 跳帧加速

        self._logger.info(f'[多步递推] 开始, {n_frames}帧, 步长={step}')

        for i in range(0, n_frames, step):
            # 单帧提取点云
            frame_pts = self._scan_to_points(scans[i])
            if len(frame_pts) < 10:
                continue

            if mu is None:
                # ── 首帧: 全局搜索 ──
                ds0 = max(1, len(frame_pts) // 800)
                pts_ds = frame_pts[::ds0]
                res = self.map_info.resolution
                ox = self.map_info.origin.position.x
                oy = self.map_info.origin.position.y
                mw = self.map_info.width * res; mh = self.map_info.height * res
                xs = np.arange(ox + 2, ox + mw - 2, 2.0)
                ys = np.arange(oy + 2, oy + mh - 2, 2.0)

                best_sc = -1e9; best_pose = None
                for ax in xs:
                    for ay in ys:
                        for adeg in range(0, 360, 15):
                            sc, _, _ = self._score_points(pts_ds, ax, ay, math.radians(adeg))
                            if sc > best_sc: best_sc = sc; best_pose = (ax, ay, math.radians(adeg))

                if best_pose is None:
                    self._logger.error('[多步递推] 首帧全局搜索失败')
                    self.indoor_phase = IndoorPhase.ROUGH_MATCHING
                    return

                mu = best_pose
                wall_cov, cov, _ = self._compute_wall_coverage(frame_pts, mu[0], mu[1], mu[2])
                self._logger.info(f'  帧{i:02d}(首): 全局搜索 → ({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.0f}deg) '
                                f'wall={100*wall_cov:.0f}% σ={sigma:.1f}m')
            else:
                # ── 后续帧: ICP + 局部搜索 ──
                prev_pts = self._scan_to_points(scans[max(0, i - step)])
                dx_i, dy_i, dyaw_i = 0.0, 0.0, 0.0
                if len(prev_pts) > 10 and len(frame_pts) > 10:
                    dx_i, dy_i, dyaw_i = self._icp_match(prev_pts, frame_pts)
                    if abs(dyaw_i) > math.radians(30):  # 旋转太大, 放弃ICP
                        dx_i = dy_i = dyaw_i = 0.0

                # 预测 (ICP增量旋转到map系)
                c_m, s_m = math.cos(mu[2]), math.sin(mu[2])
                pred_x = mu[0] + c_m*dx_i - s_m*dy_i
                pred_y = mu[1] + s_m*dx_i + c_m*dy_i
                pred_yaw = mu[2] + dyaw_i

                # 局部搜索 (±sigma)
                search_r = min(sigma * 1.5, 3.0)
                angle_r = min(sigma_angle * 1.5, 20)
                ds_i = max(1, len(frame_pts) // 600)
                pts_local = frame_pts[::ds_i]

                best_sc = -1e9; best_local = (pred_x, pred_y, pred_yaw)
                for dx in np.arange(-search_r, search_r + 1e-5, max(0.2, search_r/7)):
                    for dy in np.arange(-search_r, search_r + 1e-5, max(0.2, search_r/7)):
                        for da in np.arange(-angle_r, angle_r + 1, max(2.0, angle_r/5)):
                            sc, _, _ = self._score_points(pts_local, pred_x+dx, pred_y+dy,
                                                          pred_yaw + math.radians(da))
                            if sc > best_sc:
                                best_sc = sc; best_local = (pred_x+dx, pred_y+dy, pred_yaw+math.radians(da))

                mu = best_local
                sigma = max(sigma * decay, min_sigma)
                sigma_angle = max(sigma_angle * decay, 2.0)
                wall_cov, _, _ = self._compute_wall_coverage(frame_pts, mu[0], mu[1], mu[2])
                total_wall_hits += int(wall_cov * 100)

                if i % (step * 5) == 0 or i == n_frames - 1:
                    self._logger.info(
                        f'  帧{i:02d}: ({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.0f}deg) '
                        f'wall={100*wall_cov:.0f}% σ={sigma:.2f}m')

            total_valid += 1

        elapsed = time.time() - t0
        avg_wall = total_wall_hits / max(total_valid, 1) / 100.0

        self._logger.info(
            f'[多步递推] 完成 ({elapsed:.1f}s, {total_valid}帧): '
            f'({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.1f}deg) '
            f'avg_wall={100*avg_wall:.0f}% final_σ={sigma:.2f}m')

        # 更新 candidates (兼容后续流程)
        self.candidates = [(1.0, mu[0], mu[1], mu[2])]
        self._last_match_quality = {'wall_cov': avg_wall, 'coverage': 0.5, 'score': 1.0}

        if self.quick_mode:
            self._publish_and_finish(mu[0], mu[1], mu[2])
        else:
            self._publish_candidates()
            self.indoor_phase = IndoorPhase.SELECTING_ACTIVE_MOTION
    def _passive_timer_cb(self):
        """被动模式定时器: 每隔 passive_interval 秒触发一次匹配"""
        if not self.passive_mode_enabled:
            return
        if self.map_data is None or self.current_scan is None:
            return
        # 主动模式移动中跳过 (运动期间扫描不可靠)
        if self.indoor_phase in (IndoorPhase.MOVING, IndoorPhase.ROTATING_360):
            return
        if self.indoor_phase not in (IndoorPhase.PASSIVE_COLLECTING, IndoorPhase.PASSIVE_MATCHING,
                                      IndoorPhase.IDLE, IndoorPhase.DONE):
            return
        self.indoor_phase = IndoorPhase.PASSIVE_MATCHING

    def _do_passive_matching(self):
        """被动匹配: 取缓冲区最近N帧 → 多步递推 (逐帧ICP + 局部搜索)"""
        if self.likelihood_field is None or not self.passive_scan_buffer:
            self.indoor_phase = IndoorPhase.PASSIVE_COLLECTING
            return

        t0 = time.time()
        recent = self.passive_scan_buffer[-self.passive_frame_count:]
        scans = [item[0] for item in recent]
        if len(scans) < 5:
            self.indoor_phase = IndoorPhase.PASSIVE_COLLECTING
            return

        # 多步递推: 逐帧处理
        mu = self.passive_best_pose  # 上次估计作为先验
        sigma = 2.0 if mu is not None else 5.0
        min_sigma = 0.3; decay = 0.8
        total_wall = 0; total_n = 0

        for i, scan in enumerate(scans):
            pts = self._scan_to_points(scan)
            if len(pts) < 10: continue

            if mu is None:
                # 首帧全局搜索
                ds0 = max(1, len(pts)//800); pts_ds = pts[::ds0]
                res = self.map_info.resolution
                ox = self.map_info.origin.position.x; oy = self.map_info.origin.position.y
                mw = self.map_info.width*res; mh = self.map_info.height*res
                best_sc = -1e9; best_pose = None
                for ax in np.arange(ox+2, ox+mw-2, 2.0):
                    for ay in np.arange(oy+2, oy+mh-2, 2.0):
                        for adeg in range(0, 360, 15):
                            sc, _, _ = self._score_points(pts_ds, ax, ay, math.radians(adeg))
                            if sc > best_sc: best_sc = sc; best_pose = (ax, ay, math.radians(adeg))
                if best_pose is None: continue
                mu = best_pose; method = "全局"
            else:
                # ICP + 局部搜索
                prev_pts = self._scan_to_points(scans[max(0,i-1)])
                dx_i = dy_i = dyaw_i = 0.0
                if len(prev_pts) > 10:
                    dx_i, dy_i, dyaw_i = self._icp_match(prev_pts, pts)
                    if abs(dyaw_i) > math.radians(30): dx_i = dy_i = dyaw_i = 0.0
                c_m, s_m = math.cos(mu[2]), math.sin(mu[2])
                pred_x = mu[0] + c_m*dx_i - s_m*dy_i
                pred_y = mu[1] + s_m*dx_i + c_m*dy_i
                pred_yaw = mu[2] + dyaw_i

                sr = min(sigma*1.5, 2.0); ar = min(sigma_angle:=10.0, 15)
                ds_i = max(1, len(pts)//600); pts_l = pts[::ds_i]
                best_sc = -1e9; best_l = (pred_x, pred_y, pred_yaw)
                for dx in np.arange(-sr, sr+1e-5, 0.3):
                    for dy in np.arange(-sr, sr+1e-5, 0.3):
                        for da in range(int(-ar), int(ar)+1, 2):
                            sc, _, _ = self._score_points(pts_l, pred_x+dx, pred_y+dy, pred_yaw+math.radians(da))
                            if sc > best_sc: best_sc = sc; best_l = (pred_x+dx, pred_y+dy, pred_yaw+math.radians(da))
                mu = best_l; sigma = max(sigma*decay, min_sigma); method = "局部"

            wc, _, _ = self._compute_wall_coverage(pts, mu[0], mu[1], mu[2])
            total_wall += wc; total_n += 1

        avg_wall = total_wall / max(total_n, 1)
        self.passive_best_pose = mu
        elapsed = time.time() - t0
        ts = self.get_clock().now().nanoseconds / 1e9
        self.passive_pose_history.append((ts, mu[0], mu[1], mu[2], avg_wall))

        if hasattr(self, 'debug_auto_pose_pub'):
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.map_frame
            msg.pose.pose.position = Point(x=mu[0], y=mu[1], z=0.0)
            qz = math.sin(mu[2]/2.0); qw = math.cos(mu[2]/2.0)
            msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            self.debug_auto_pose_pub.publish(msg)

        self._logger.info(
            f'[被动{method}] {elapsed:.1f}s {total_n}帧: ({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.1f}deg) '
            f'wall={100*avg_wall:.0f}% σ={sigma:.2f}m buf={len(self.passive_scan_buffer)}帧')
        self.indoor_phase = IndoorPhase.PASSIVE_COLLECTING

    def _start_passive_mode(self):
        """启动被动模式"""
        if not self.passive_mode_enabled:
            return
        # ── 互斥: 如果主动模式进行中, 拒绝启动 ──
        active_phases = {IndoorPhase.BOOT_DELAY, IndoorPhase.ROTATING_360,
                         IndoorPhase.COLLECTING_SUBMAP1, IndoorPhase.ROUGH_MATCHING,
                         IndoorPhase.SELECTING_ACTIVE_MOTION, IndoorPhase.MOVING,
                         IndoorPhase.COLLECTING_SUBMAP2, IndoorPhase.FILTERING}
        if self.indoor_phase in active_phases:
            self._logger.warn('[被动模式] 主动匹配进行中, 暂不启动被动模式')
            return
        self.passive_scan_buffer.clear()
        self.passive_best_pose = None
        self.passive_last_match_time = None
        self.indoor_phase = IndoorPhase.PASSIVE_COLLECTING
        self._logger.info(
            f'[被动模式] 已启动, 间隔={self.passive_interval}s, '
            f'每轮帧数={self.passive_frame_count}')

    def _stop_passive_mode(self):
        """停止被动模式"""
        self.indoor_phase = IndoorPhase.IDLE
        self._logger.info('[被动模式] 已停止')
    def _do_active_motion_selection(self):
        if not self.candidates:
            self._logger.error('候选 Pose 为空，无法进行主动运动规划，重置。')
            self._reset_indoor()
            return
            
        if self.current_scan is None:
            self._logger.warn('激光扫描未就位，等待数据...')
            return

        # 动态步长计算（根据候选 Pose 分布的信息熵）
        probs = np.array([c[0] for c in self.candidates])
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        # 熵值越大，说明歧义性高，采用大步长；熵值低，采用短步长精细对齐
        motion_dist = self.base_motion_distance
        if entropy < 1.5:
            motion_dist = self.base_motion_distance * 0.5
            self._logger.info(f'[运动选择] 信息熵偏低 ({entropy:.2f})，采用短步长探索: {motion_dist:.2f}m')
        else:
            self._logger.info(f'[运动选择] 信息熵高 ({entropy:.2f})，采用常规步长探索: {motion_dist:.2f}m')

        # 定义 8 个粗粒度朝向: 0, 45, 90, 135, 180, -135, -90, -45 (度)
        test_angles_deg = [0, 45, 90, 135, 180, -135, -90, -45]
        best_dir = None
        max_ig = -1.0
        
        # 遍历 8 个方向
        for deg in test_angles_deg:
            rad = math.radians(deg)
            
            # A. 局部安全检测（基于本体 /scan）
            is_safe = self._check_local_direction_safety(rad, motion_dist)
            if not is_safe:
                continue
                
            # B. 信息增益估算
            # 动作为：沿本地 rad 平移 motion_dist，同时旋转 45° (打破直线对称性)
            delta_rot = math.radians(45.0) if deg != 180 else 0.0 # 180度倒车不宜大旋转
            
            # 计算 Top-N 候选在此动作下预测点位似然场特征的差异度
            signatures = []
            for prob, x, y, yaw in self.candidates:
                # 递推预测全局位姿
                pred_x = x + motion_dist * math.cos(yaw + rad)
                pred_y = y + motion_dist * math.sin(yaw + rad)
                pred_yaw = self._norm_angle(yaw + delta_rot)
                
                # 提取 16 维特征向量 (似然场采样)
                sig = self._get_map_signature(pred_x, pred_y, pred_yaw)
                signatures.append(sig)
                
            # 计算方差和作为信息增益度量
            sig_matrix = np.array(signatures) # Shape: (N, 16)
            variance_sum = np.sum(np.var(sig_matrix, axis=0))
            
            self._logger.debug(f'方向 {deg}°: 局部安全=True, 信息增益 (方差和)={variance_sum:.4f}')
            
            if variance_sum > max_ig:
                max_ig = variance_sum
                best_dir = (rad, delta_rot, motion_dist)

        if best_dir is not None:
            rad, delta_rot, dist = best_dir
            self._logger.info(f'[运动决策] 最优安全探索动作: 相对朝向={math.degrees(rad):.1f}°, 旋转={math.degrees(delta_rot):.1f}°, 距离={dist:.2f}m, IG={max_ig:.4f}')
            
            # 设置控制目标里程计位姿
            if self.current_odom is not None:
                curr_pos = self.current_odom.pose.pose.position
                curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
                
                target_x = curr_pos.x + dist * math.cos(curr_yaw + rad)
                target_y = curr_pos.y + dist * math.sin(curr_yaw + rad)
                target_yaw = self._norm_angle(curr_yaw + delta_rot)
                
                self.target_odom_pose = (target_x, target_y, target_yaw)
                self.motion_start_odom = self.current_odom
                self.motion_start_time = self.get_clock().now()
                self.indoor_phase = IndoorPhase.MOVING
            else:
                self._logger.warn('里程计信号中断，无法启动移动。')
        else:
            # 如果没有安全的方向（如陷在狭窄空间）
            self._logger.warn('[运动决策] 未能找到安全的平移方向。强制尝试就地旋转 45° 以获取环境信息。')
            if self.current_odom is not None:
                curr_pos = self.current_odom.pose.pose.position
                curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
                self.target_odom_pose = (curr_pos.x, curr_pos.y, self._norm_angle(curr_yaw + math.radians(45.0)))
                self.motion_start_odom = self.current_odom
                self.motion_start_time = self.get_clock().now()
                self.indoor_phase = IndoorPhase.MOVING
            else:
                self._reset_indoor()

    def _check_local_direction_safety(self, local_angle, dist):
        """基于当前 /scan，检查局部坐标系方向是否安全"""
        if self.current_scan is None:
            return False
            
        ranges = self.current_scan.ranges
        num_beams = len(ranges)
        
        # 截取方向在 local_angle 左右 22.5 度范围内的所有光束
        angle_sector = math.radians(22.5)
        for i in range(num_beams):
            r = ranges[i]
            if not (self.current_scan.range_min < r < self.current_scan.range_max):
                continue
            
            beam_angle = self.current_scan.angle_min + i * self.current_scan.angle_increment
            angle_diff = abs(self._norm_angle(beam_angle - local_angle))
            
            if angle_diff <= angle_sector:
                # 如果测距值小于移动预定距离加停机冗余，则认为不安全
                if r < (dist + self.min_safe_distance):
                    return False
        return True

    def _get_map_signature(self, x, y, yaw):
        """以指定位姿为中心，以 2 个不同距离、8 个径向方向读取似然场数值，生成特征特征签名"""
        sig = []
        dists = [self.ig_sample_dist_1, self.ig_sample_dist_2]
        angles = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]
        
        for d in dists:
            for ang_deg in angles:
                rad = math.radians(ang_deg)
                px = x + d * math.cos(yaw + rad)
                py = y + d * math.sin(yaw + rad)
                
                # 映射到似然场索引
                if self.map_info is not None and self.likelihood_field is not None:
                    col = int((px - self.map_info.origin.position.x) / self.map_info.resolution)
                    row = int(self.map_info.height - 1 - (py - self.map_info.origin.position.y) / self.map_info.resolution)
                    
                    if 0 <= row < self.map_info.height and 0 <= col < self.map_info.width:
                        sig.append(float(self.likelihood_field[row, col]))
                    else:
                        sig.append(float(self.likelihood_max_dist))
                else:
                    sig.append(float(self.likelihood_max_dist))
        return sig

    # ================================================================
    #  核心步骤 6：控制器环路与实时激光避障
    # ================================================================
    def _do_control_loop_and_avoidance(self):
        if self.target_odom_pose is None or self.current_odom is None:
            self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
            return

        # 1. 检查超时限制（防卡死，12秒限制）
        time_elapsed = (self.get_clock().now() - self.motion_start_time).nanoseconds / 1e9
        if time_elapsed > 12.0:
            self._logger.warn('[主动控制] 移动超时限制，停止并开始下一步匹配。')
            self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
            self.scan_buffer.clear()
            self.submap_ready = False
            return

        # 2. 闭环误差计算 (局部坐标系)
        tx, ty, tyaw = self.target_odom_pose
        curr_pos = self.current_odom.pose.pose.position
        curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)

        dx = tx - curr_pos.x
        dy = ty - curr_pos.y
        dyaw = self._norm_angle(tyaw - curr_yaw)

        # 旋转至机器人局部系
        err_x = dx * math.cos(curr_yaw) + dy * math.sin(curr_yaw)
        err_y = -dx * math.sin(curr_yaw) + dy * math.cos(curr_yaw)

        dist_err = math.sqrt(err_x*err_x + err_y*err_y)

        # 3. 终点判定
        if dist_err < 0.05 and abs(dyaw) < math.radians(5.0):
            self._logger.info('[主动控制] 已精准抵达目标运动位姿。')
            self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
            self.scan_buffer.clear()
            self.submap_ready = False
            return

        # 4. 实时避障检测 (基于当前运动速度的局部方向锥角)
        # 前进速度方向角度
        move_dir = math.atan2(err_y, err_x)
        if self.current_scan is not None:
            ranges = self.current_scan.ranges
            num_beams = len(ranges)
            
            # 检测运动正前方锥角 ± 30 度
            avoid_sector = math.radians(30.0)
            for i in range(num_beams):
                r = ranges[i]
                if not (self.current_scan.range_min < r < self.current_scan.range_max):
                    continue
                beam_angle = self.current_scan.angle_min + i * self.current_scan.angle_increment
                diff = abs(self._norm_angle(beam_angle - move_dir))
                
                if diff <= avoid_sector:
                    if r < self.min_safe_distance:
                        self._logger.warn(f'[避障停机] 前方检测到障碍物距离过近 ({r:.2f}m)！紧急触发主动停止并进行子图构建。')
                        self.cmd_vel_pub.publish(Twist())
                        self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
                        self.scan_buffer.clear()
                        self.submap_ready = False
                        return

        # 5. P 控制器速度输出计算
        vx = self.kp_linear * err_x
        vy = self.kp_linear * err_y
        wz = self.kp_angular * dyaw

        # 速度限幅
        vx = np.clip(vx, -self.max_linear_vel, self.max_linear_vel)
        vy = np.clip(vy, -self.max_linear_vel, self.max_linear_vel)
        wz = np.clip(wz, -self.max_angular_vel, self.max_angular_vel)

        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(wz)
        self.cmd_vel_pub.publish(cmd)

    # ================================================================
    #  核心步骤 7 & 8：位姿传播与子图重评分
    # ================================================================
    def _do_filtering_and_propagation(self):
        if self.submap1 is None or self.submap2 is None:
            self._logger.error('子图数据丢失，重置状态。')
            self._reset_indoor()
            return

        # 计算运动位姿增量 — 完全使用控制器目标增量（不依赖 odom 绝对坐标）
        rel_x = 0.0
        rel_y = 0.0
        rel_yaw = 0.0
        
        if self.target_odom_pose is not None and self.motion_start_odom is not None:
            target_x, target_y, target_yaw = self.target_odom_pose
            start_pos = self.motion_start_odom.pose.pose.position
            start_yaw = self._quat_to_yaw(self.motion_start_odom.pose.pose.orientation)
            
            # 计算控制器目标在 odom 坐标系下的位移
            dx = target_x - start_pos.x
            dy = target_y - start_pos.y
            
            # 转到起始朝向的局部系（即机器人预期运动增量）
            rel_x = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
            rel_y = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)
            rel_yaw = self._norm_angle(target_yaw - start_yaw)
            
            self._logger.info(f'[位姿传播] 使用控制器目标增量: dx={rel_x:.3f}m, dy={rel_y:.3f}m, yaw={math.degrees(rel_yaw):.1f}°')
        else:
            # 无可用控制器目标，使用默认运动距离
            rel_x = self.base_motion_distance
            rel_y = 0.0
            rel_yaw = 0.0
            self._logger.warn('[位姿传播] 无控制器目标数据，使用默认运动距离')

        self._logger.info(f'[位姿传播] 最终增量: dx={rel_x:.3f}m, dy={rel_y:.3f}m, yaw={math.degrees(rel_yaw):.1f}°')

        # 对旧候选列表在 Map 系下进行坐标递推，并用 Submap 2 进行二次评分
        updated_candidates = []
        for prob, x, y, yaw in self.candidates:
            # 候选位姿递推
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            new_x = x + rel_x * cos_y - rel_y * sin_y
            new_y = y + rel_y * cos_y + rel_x * sin_y
            new_yaw = self._norm_angle(yaw + rel_yaw)
            
            # 使用 Submap 2 进行重打分
            score = self._score_scan(self.submap2, new_x, new_y, new_yaw, self.rough_beams)
            updated_candidates.append((score, new_x, new_y, new_yaw))

        # 概率归一化
        scores_arr = np.array([c[0] for c in updated_candidates])
        max_score = np.max(scores_arr)
        exp_scores = np.exp(scores_arr - max_score)
        normalized_probs = exp_scores / np.sum(exp_scores)

        self.candidates = []
        for i, c in enumerate(updated_candidates):
            self.candidates.append((float(normalized_probs[i]), c[1], c[2], c[3]))

        # 按概率从大到小排序
        self.candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 调试展示候选
        self._publish_candidates()
        
        self._logger.info('[位姿重评分] 完成。最新候选分布:')
        for i, (prob, x, y, yaw) in enumerate(self.candidates):
            self._logger.info(f'  #{i}: 概率={prob:.3f}, x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°')

        # 唯一性收敛判定
        best_prob = self.candidates[0][0]
        second_prob = self.candidates[1][0] if len(self.candidates) > 1 else 0.0
        
        if best_prob > 0.8 and (second_prob == 0.0 or best_prob / second_prob >= 2.0):
            # 唯一收敛！发布 /initialpose 并初始化定位
            _, cx, cy, cyaw = self.candidates[0]
            self._publish_and_finish(cx, cy, cyaw)
        else:
            if self.active_retry_count >= self.max_active_retry:
                self._logger.warn(f'[收敛检测] 已到达最大尝试次数 ({self.max_active_retry}) 仍未唯一收敛，重新搜索。')
                self._reset_indoor()
            else:
                self.active_retry_count += 1
                self._logger.info(f'[收敛检测] 尚未唯一收敛 (最佳概率={best_prob:.3f}, 次佳={second_prob:.3f})，开启第 {self.active_retry_count} 轮探索移动...')
                
                # 更新 Submap 1 参考基础（不依赖 odom）
                self.submap1 = self.submap2
                self.indoor_phase = IndoorPhase.SELECTING_ACTIVE_MOTION

    # ================================================================
    #  运动验证：用 Scan-to-Scan ICP 验证实际位移
    # ================================================================
    def _verify_motion_by_scan_matching(self):
        """
        用 Submap1 和 Submap2 做 scan-to-scan ICP 匹配，
        验证机器人实际位移与控制器目标增量的一致性。
        """
        if self.submap1 is None or self.submap2 is None:
            self._logger.warn('[运动验证] Submap1 或 Submap2 为空，跳过验证')
            return
        
        if self.target_odom_pose is None or self.motion_start_odom is None:
            self._logger.warn('[运动验证] 控制器目标数据缺失，跳过验证')
            return
        
        # 计算控制器目标增量（预期位移）
        target_x, target_y, target_yaw = self.target_odom_pose
        start_pos = self.motion_start_odom.pose.pose.position
        start_yaw = self._quat_to_yaw(self.motion_start_odom.pose.pose.orientation)
        
        dx = target_x - start_pos.x
        dy = target_y - start_pos.y
        rel_x = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
        rel_y = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)
        rel_yaw = self._norm_angle(target_yaw - start_yaw)
        
        # 用 ICP 计算 Submap1 -> Submap2 的实际变换
        points1 = self._scan_to_points(self.submap1)
        points2 = self._scan_to_points(self.submap2)
        
        if len(points1) < 3 or len(points2) < 3:
            self._logger.warn('[运动验证] 子图点云点数不足，跳过验证')
            return
        
        actual_dx, actual_dy, actual_dyaw = self._icp_match(points1, points2)
        
        # 比较预期位移与实际位移
        dist_expected = math.sqrt(rel_x**2 + rel_y**2)
        dist_actual = math.sqrt(actual_dx**2 + actual_dy**2)
        
        dist_error = abs(dist_actual - dist_expected)
        angle_error = abs(self._norm_angle(actual_dyaw - rel_yaw))
        
        self._logger.info(
            f'[运动验证] 控制器目标: ({rel_x:.3f}, {rel_y:.3f}, {math.degrees(rel_yaw):.1f}°), '
            f'ICP实际: ({actual_dx:.3f}, {actual_dy:.3f}, {math.degrees(actual_dyaw):.1f}°), '
            f'误差: 距离={dist_error:.3f}m, 角度={math.degrees(angle_error):.1f}°'
        )
        
        # 如果误差过大，记录警告
        if dist_error > self.motion_verification_threshold:
            self._logger.warn(
                f'[运动验证] 距离误差超限: {dist_error:.3f}m > {self.motion_verification_threshold:.3f}m'
            )
        if angle_error > math.radians(self.motion_verification_angle_threshold_deg):
            self._logger.warn(
                f'[运动验证] 角度误差超限: {math.degrees(angle_error):.1f}° > {self.motion_verification_angle_threshold_deg:.1f}°'
            )

    # ================================================================
    #  调试对比模式：算法结果 vs 手动真值
    # ================================================================
    def _compare_with_manual_ground_truth(self, algo_x, algo_y, algo_yaw):
        """
        对比算法计算结果与手动设置的真值，输出偏差日志。
        支持多轮累积统计。
        """
        if not self.gt_received or self.gt_pose is None:
            self._logger.info('[调试对比] 暂无手动真值，跳过对比')
            return
        
        gt_x, gt_y, gt_yaw = self.gt_pose
        err_dist = math.sqrt((algo_x - gt_x)**2 + (algo_y - gt_y)**2)
        err_yaw = abs(self._norm_angle(algo_yaw - gt_yaw))
        
        # 累积统计
        if not hasattr(self, "comparison_history"):
            self.comparison_history = []
        
        self.comparison_history.append({
            'algo': (algo_x, algo_y, algo_yaw),
            'gt': (gt_x, gt_y, gt_yaw),
            'err_dist': err_dist,
            'err_yaw': err_yaw,
            'timestamp': self.get_clock().now().nanoseconds / 1e9
        })
        
        self._logger.info(
            f'[调试对比] 算法结果: ({algo_x:.3f}, {algo_y:.3f}, {math.degrees(algo_yaw):.1f}°), '
            f'手动真值: ({gt_x:.3f}, {gt_y:.3f}, {math.degrees(gt_yaw):.1f}°), '
            f'偏差: 距离={err_dist:.3f}m, 角度={math.degrees(err_yaw):.1f}°'
        )
        
        # 每 5 轮输出一次统计
        if len(self.comparison_history) % 5 == 0:
            self._print_comparison_statistics()

    def _print_comparison_statistics(self):
        """输出多轮对比的统计结果（均值/方差/最大误差）"""
        if not hasattr(self, "comparison_history") or len(self.comparison_history) < 2:
            return
        
        history = self.comparison_history
        n = len(history)
        
        dist_errors = [h['err_dist'] for h in history]
        yaw_errors = [math.degrees(h['err_yaw']) for h in history]
        
        mean_dist = np.mean(dist_errors)
        std_dist = np.std(dist_errors)
        max_dist = np.max(dist_errors)
        
        mean_yaw = np.mean(yaw_errors)
        std_yaw = np.std(yaw_errors)
        max_yaw = np.max(yaw_errors)
        
        self._logger.info(
            f'[调试对比统计] 共 {n} 轮\n'
            f'  距离误差: 均值={mean_dist:.3f}m, 标准差={std_dist:.3f}m, 最大={max_dist:.3f}m\n'
            f'  角度误差: 均值={mean_yaw:.1f}°, 标准差={std_yaw:.1f}°, 最大={max_yaw:.1f}°'
        )

    # ================================================================
    #  激光评分实现 (似然场模型)
    # ================================================================
    def _score_scan(self, scan, x, y, yaw, max_beams):
        """Beam model 评分 (兼容旧接口, 内部调用 _score_points)"""
        if self.likelihood_field is None or self.map_info is None or scan is None:
            return -1e9
        
        pts = self._scan_to_points(scan)
        if len(pts) < 10:
            return -1e9
        # 降采样
        ds = max(1, len(pts) // max_beams)
        pts_ds = pts[::ds]
        sc, _, _ = self._score_points(pts_ds, x, y, yaw)
        return sc


    def _publish_and_finish(self, x, y, yaw):
        """发布初始位姿 (自适应协方差)"""
        quality = getattr(self, '_last_match_quality', {})
        wall_cov = quality.get('wall_cov', 0.5)
        coverage = quality.get('coverage', 0.5)
        pos_sigma = min(0.8 / max(wall_cov, 0.1) * max(1.0 - coverage, 0.3), 2.0)
        yaw_sigma = min(20.0 / max(wall_cov, 0.1) * (1.0 - coverage + 0.2), 30.0)

        cov = [0.0] * 36
        cov[0] = pos_sigma ** 2; cov[7] = pos_sigma ** 2; cov[35] = math.radians(yaw_sigma) ** 2

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position = Point(x=x, y=y, z=0.0)
        qz = math.sin(yaw / 2.0); qw = math.cos(yaw / 2.0)
        msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        msg.pose.covariance = cov

        self._compare_with_manual_ground_truth(x, y, yaw)

        if self.auto_publish_initial_pose:
            self.initialpose_pub.publish(msg)
            self._logger.info(
                f'[主动定位] 发布: ({x:.3f},{y:.3f},{math.degrees(yaw):.1f}deg) '
                f'sigma=({pos_sigma:.2f}m,{yaw_sigma:.1f}deg) '
                f'wall={100*wall_cov:.0f}% cov={100*coverage:.0f}%')
            self._start_deviation_monitor(x, y, yaw)
        else:
            if hasattr(self, 'debug_auto_pose_pub'):
                self.debug_auto_pose_pub.publish(msg)
            self._logger.info(f'[调试] sigma=({pos_sigma:.2f}m,{yaw_sigma:.1f}deg)')

        self.last_calibrated_pose = (x, y, yaw)
        self.indoor_phase = IndoorPhase.DONE
        self.cmd_vel_pub.publish(Twist())


    def _start_deviation_monitor(self, cal_x, cal_y, cal_yaw):
        """启动偏差监控"""
        self._deviation_ref = (cal_x, cal_y, cal_yaw)
        self._deviation_start_time = self.get_clock().now()
        self._deviation_count = 0; self._deviation_exceeded = False
        if not hasattr(self, '_deviation_timer') or self._deviation_timer is None:
            self._deviation_timer = self.create_timer(2.0, self._check_deviation)
        self._logger.info(f'[偏差监控] 启动, 基准: ({cal_x:.2f},{cal_y:.2f})')

    def _check_deviation(self):
        """每2秒检查AMCL vs 校准偏差"""
        if getattr(self, '_deviation_ref', None) is None or self.current_odom is None:
            return
        cal_x, cal_y, cal_yaw = self._deviation_ref
        elapsed = (self.get_clock().now() - self._deviation_start_time).nanoseconds / 1e9
        amcl_x = getattr(self, '_last_amcl_x', None)
        if amcl_x is None: return
        dev_dist = math.sqrt((amcl_x - cal_x)**2 + (self._last_amcl_y - cal_y)**2)
        self._deviation_count += 1
        if elapsed < 180 and dev_dist > 0.5 and not self._deviation_exceeded:
            self._logger.warn(f'[偏差告警] AMCL偏离 {dev_dist:.3f}m > 0.5m (开机{elapsed:.0f}s)')
            self._deviation_exceeded = True
        if self._deviation_count % 15 == 0:
            self._logger.info(f'[偏差监控] {elapsed:.0f}s, 偏差={dev_dist:.3f}m, ' +
                             f'AMCL=({amcl_x:.2f},{self._last_amcl_y:.2f})')

    def _publish_candidates(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = self.map_frame
        
        for prob, x, y, yaw in self.candidates:
            pose = Pose()
            pose.position = Point(x=x, y=y, z=0.0)
            pose.orientation = Quaternion(
                x=0.0, y=0.0, z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)
            )
            pose_array.poses.append(pose)
        self.candidates_pub.publish(pose_array)

    def _srv_start(self, req, resp):
        if self.map_data is None:
            if self.map_file and os.path.exists(self.map_file):
                self._logger.info('服务启动但话题未接收到地图，尝试从本地配置文件加载地图...')
                self._load_map_from_file(self.map_file)
            if self.map_data is None:
                resp.success = False
                resp.message = '全局栅格地图未就绪，且本地加载失败，校准失败'
                return resp
        if self.current_scan is None:
            resp.success = False
            resp.message = '激光雷达数据未就绪，校准失败'
            return resp

        # ── 互斥: 主动校准需要 odom; 被动模式不需要 ──
        if not self.passive_mode_enabled and self.current_odom is None:
            resp.success = False
            resp.message = '里程计未就绪，校准失败'
            return resp

        if self.passive_mode_enabled:
            # 被动模式: 直接启动持续采集, 不控制机器人运动
            self._start_passive_mode()
            resp.success = True
            resp.message = '被动持续定位已启动, 将在后台定时匹配'
            self._logger.info(f'[开始流程] {resp.message}')
            return resp

        self.indoor_phase = IndoorPhase.BOOT_DELAY
        self.boot_start_time = self.get_clock().now()
        self.scan_buffer.clear()
        self.submap_ready = False
        self.candidates.clear()
        self.active_retry_count = 0
        
        resp.success = True
        resp.message = '自动定位流程已成功触发启动，请保持机器人狗静止 2 秒。'
        self._logger.info(f'[开始流程] {resp.message}')
        return resp

    def _srv_start_active(self, req, resp):
        """手动启动主动校准 (旋转+移动探索)"""
        if self.map_data is None or self.current_scan is None or self.current_odom is None:
            resp.success = False
            resp.message = '地图/雷达/里程计未就绪'
            return resp
        self.indoor_phase = IndoorPhase.BOOT_DELAY
        self.boot_start_time = self.get_clock().now()
        self.scan_buffer.clear(); self.submap_ready = False
        self.candidates.clear(); self.active_retry_count = 0
        resp.success = True
        resp.message = '主动校准已启动，机器人将旋转360°并探索'
        return resp

    def _srv_start_passive(self, req, resp):
        """手动启动被动持续定位"""
        if self.map_data is None or self.current_scan is None:
            resp.success = False
            resp.message = '地图/雷达未就绪'
            return resp
        self._start_passive_mode()
        resp.success = True
        resp.message = f'被动定位已启动, 每 {self.passive_interval}s 匹配一次'
        return resp

    def _srv_stop_passive(self, req, resp):
        """停止被动持续定位"""
        self._stop_passive_mode()
        resp.success = True
        resp.message = '被动定位已停止'
        return resp

    def _srv_status(self, req, resp):
        msg = f'当前阶段: {self.indoor_phase.name}\n'
        msg += f'被动模式: {"启用" if self.passive_mode_enabled else "禁用"}\n'
        if self.passive_best_pose is not None:
            px, py, pyaw = self.passive_best_pose
            msg += f'被动最佳估计: ({px:.2f},{py:.2f},{math.degrees(pyaw):.1f}deg)\n'
        msg += f'被动缓存帧: {len(self.passive_scan_buffer)}\n'
        msg += f'主动重试轮数: {self.active_retry_count}\n'
        msg += f'候选Pose数: {len(self.candidates)}'
        resp.success = True
        resp.message = msg
        return resp

    def _srv_reset(self, req, resp):
        self._reset_indoor()
        resp.success = True
        resp.message = '已重置校准状态。'
        return resp

    def _srv_toggle_auto_publish(self, req, resp):
        """动态切换自动发布开关"""
        self.auto_publish_initial_pose = not self.auto_publish_initial_pose
        resp.success = True
        resp.message = f'自动发布已{"启用" if self.auto_publish_initial_pose else "禁用（调试模式）"}'
        self._logger.info(f'[调试模式] {resp.message}')
        return resp

    def _srv_set_manual_ground_truth(self, req, resp):
        """通过服务调用输入手动真值坐标"""
        if req.data:
            # req.data 为 true 时，使用当前算法结果作为真值（用于对比）
            if self.last_calibrated_pose is not None:
                self.gt_pose = self.last_calibrated_pose
                self.gt_received = True
                resp.success = True
                resp.message = f'已设置手动真值: {self.last_calibrated_pose}'
            else:
                resp.success = False
                resp.message = '无算法结果可用，请先运行校准'
        else:
            # req.data 为 false 时，清除真值
            self.gt_received = False
            self.gt_pose = None
            resp.success = True
            resp.message = '已清除手动真值'
        return resp

    def _reset_indoor(self):
        self.indoor_phase = IndoorPhase.IDLE
        self.scan_buffer.clear()
        self.submap_ready = False
        self.candidates.clear()
        self.active_retry_count = 0
        self.cmd_vel_pub.publish(Twist())
        self._logger.info('[重置] 室内自动位姿状态机已被重置并置为空闲。')

    # ================================================================
    #  室外 GPS/RTK 定位模式
    # ================================================================
    def _outdoor_loop(self):
        if not self.outdoor_mode or self.current_rtk is None:
            return
        try:
            heading = self.current_rtk.heading
            bestnav = self.current_rtk.bestnav
            if heading.heading_type not in [34, 50] or bestnav.pos_type not in [34, 50]:
                return
            if heading.sol_status != 0 or bestnav.p_sol_status != 0:
                return
            if self.calibration_tf is None:
                return
            
            tx = self.calibration_tf['translation']['x']
            ty = self.calibration_tf['translation']['y']
            yr = self.calibration_tf['rotation']['yaw']
            cx = math.cos(yr)
            sx = math.sin(yr)
            
            mx = bestnav.longitude_deg * cx - bestnav.latitude_deg * sx + tx
            my = bestnav.longitude_deg * sx + bestnav.latitude_deg * cx + ty
            myaw = self._norm_angle(math.radians(heading.heading_deg) + yr)
            
            cov = [0.0] * 36
            cov[0] = bestnav.lat_std ** 2
            cov[7] = bestnav.lon_std ** 2
            cov[35] = math.radians(heading.heading_std) ** 2
            
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.map_frame
            msg.pose.pose.position = Point(x=mx, y=my, z=0.0)
            qz = math.sin(myaw / 2.0)
            qw = math.cos(myaw / 2.0)
            msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            msg.pose.covariance = cov
            self.initialpose_pub.publish(msg)
        except Exception as e:
            self._logger.debug(f'室外 RTK 转换异常: {e}')

    # ================================================================
    #  常用工具方法
    # ================================================================
    @staticmethod
    def _quat_to_yaw(q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _norm_angle(a):
        while a > math.pi: a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    def _load_calibration(self):
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    self.calibration_tf = yaml.safe_load(f)
                self._logger.info(f'成功读取外参校准文件: {self.calibration_file}')
            except Exception as e:
                self._logger.error(f'读取外参文件出错: {e}')
        else:
            self._logger.warning(f'校准文件不存在: {self.calibration_file}')

    def _record_amcl_pose(self, x, y, yaw):
        """记录最新AMCL位姿 (供偏差监控使用)"""
        self._last_amcl_x = x; self._last_amcl_y = y; self._last_amcl_yaw = yaw

    def _check_auto_mode(self):
        # 仅在启用自动识别（即配置里 outdoor_mode=True 且 indoor_mode=True）时进行判断
        # 如果配置里只开启了一个，则强制为该模式
        if not self.outdoor_mode and self.indoor_mode:
            current_mode = "INDOOR"
        elif self.outdoor_mode and not self.indoor_mode:
            current_mode = "OUTDOOR"
        elif not self.outdoor_mode and not self.indoor_mode:
            current_mode = "NONE"
        else:
            # 双模开启时，进行自动识别
            # 判断 RTK 是否有效
            is_rtk_valid = False
            if self.current_rtk is not None:
                try:
                    heading = self.current_rtk.heading
                    bestnav = self.current_rtk.bestnav
                    if (heading.heading_type in [34, 50] and bestnav.pos_type in [34, 50] and
                        heading.sol_status == 0 and bestnav.p_sol_status == 0):
                        is_rtk_valid = True
                except Exception:
                    pass
            
            # 判断 GPS 是否有效
            is_gps_valid = False
            if self.current_gps is not None:
                try:
                    # status >= 0 代表有普通定位或更高级的定位
                    if self.current_gps.status.status >= 0:
                        is_gps_valid = True
                except Exception:
                    pass
            
            if is_rtk_valid or is_gps_valid:
                current_mode = "OUTDOOR"
            else:
                # 如果从启动到现在未满 5 秒，我们先保持 UNKNOWN 状态，避免传感器数据尚未接收完全时产生误判
                elapsed = (self.get_clock().now() - self.startup_time).nanoseconds / 1e9
                if elapsed < 5.0:
                    return
                current_mode = "INDOOR"

        # 如果检测到的模式发生改变，则打印日志
        if current_mode != self.detected_mode:
            self.detected_mode = current_mode
            if current_mode == "OUTDOOR":
                self._logger.info("[模式自动识别] 检测到有效 RTK/GPS 信号，系统当前运行于【室外模式】")
            elif current_mode == "INDOOR":
                self._logger.info("[模式自动识别] 未检测到有效 RTK/GPS 信号，系统当前运行于【室内模式】")
            elif current_mode == "NONE":
                self._logger.warn("[模式自动识别] 室内与室外模式均已关闭！")


def main(args=None):
    rclpy.init(args=args)
    node = AutoInitialPoseCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
