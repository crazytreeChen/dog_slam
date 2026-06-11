#!/usr/bin/env python3
"""
自动初始位姿校准节点 — 室内探索与避障改进版

主要流程：
  ① 开启服务 → 调用 /start_auto_calibration
  ② 进入 BOOT_DELAY 状态，静止等待 2 秒稳定数据
  ③ 进入 COLLECTING_SUBMAP1 状态，收集 20 帧雷达数据结合 Odom 合成 Submap 1
  ④ 进入 ROUGH_MATCHING 状态，使用分层粗精两阶段粒子搜索计算 Top N 候选位姿
  ⑤ 进入 SELECTING_ACTIVE_MOTION 状态，计算各安全方向的信息增益，选择最优探索动作
  ⑥ 进入 MOVING 状态，执行局部 P 控制器，并在运动中实时通过雷达避障监控
  ⑦ 运动结束（或因避障提前终止）进入 COLLECTING_SUBMAP2 状态，收集 20 帧合成 Submap 2
  ⑧ 进入 FILTERING 状态，传播候选位姿，利用 Submap 2 重新评分，判断是否唯一收敛
  ⑨ 若唯一收敛，发布 /initialpose 并转换到 DONE 状态；若未收敛，依据动态步长继续进行下轮探索
"""

import os
import sys
import math
import time
import importlib
import yaml
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
    global_config_path = os.path.join(get_package_share_directory('global_config'), '../src/global_config')
    if global_config_path not in sys.path:
        sys.path.insert(0, global_config_path)
    from global_config import NAV2_DEFAULT_MAP_FILE
except ImportError:
    NAV2_DEFAULT_MAP_FILE = "/home/ztl/slam_data/grid_map/map.yaml"


class IndoorPhase(Enum):
    IDLE = 0
    BOOT_DELAY = 1
    COLLECTING_SUBMAP1 = 2
    ROUGH_MATCHING = 3
    SELECTING_ACTIVE_MOTION = 4
    MOVING = 5
    COLLECTING_SUBMAP2 = 6
    FILTERING = 7
    DONE = 8


class AutoInitialPoseCalibrator(Node):
    def __init__(self):
        super().__init__('auto_initial_pose_calibrator')

        # ────── 声明与读取参数 ──────
        self.declare_parameter('rtk_topic', '/rtk')
        self.declare_parameter('rtk_topic_type', 'common_msgs.msg.RTK')
        self.declare_parameter('gps_topic', '/gps')
        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('outdoor_mode', True)
        self.declare_parameter('indoor_mode', True)
        self.declare_parameter('use_sim_time', False)
        self.declare_parameter('calibration_file', '')

        # 粗匹配/粒子匹配参数
        self.declare_parameter('rough_match_particles', 1500)
        self.declare_parameter('fine_match_particles_per_cluster', 30)
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
        self.declare_parameter('submap_scan_count', 20)
        self.declare_parameter('submap_angle_resolution', 0.5)

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
        self.outdoor_mode = self.get_parameter('outdoor_mode').value
        self.indoor_mode = self.get_parameter('indoor_mode').value
        self.use_sim_time = self.get_parameter('use_sim_time').value

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

        self.publish_rate = self.get_parameter('publish_rate').value

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
        
        # 子图合并缓冲区
        self.scan_buffer = []           # [(scan_msg, odom_x, odom_y, odom_yaw)]
        self.submap_ref_odom = None     # (x, y, yaw) at start of submap
        self.submap_ready = False
        
        self.submap1 = None             # Synthesized LaserScan for stage 1
        self.submap1_ref_odom = None    # Odom reference at Submap 1 start
        self.submap2 = None             # Synthesized LaserScan for stage 2
        self.submap2_ref_odom = None    # Odom reference at Submap 2 start

        # 控制运动目标位姿
        self.target_odom_pose = None    # (x, y, yaw) in reference odom frame
        self.motion_start_odom = None   # Odom message at motion start
        self.motion_start_time = None

        # ────── Odom 与真值校对数据 ──────
        self.gt_received = False
        self.gt_pose = None             # (x, y, yaw) in map
        self.gt_ref_odom = None         # (x, y, yaw) in odom

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
        self.amcl_sub = self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self._amcl_cb, 10)
        # 监听 RViz 2D Pose Estimate 触发真值标记或常规设置
        self.initialpose_sub = self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self._initialpose_cb, 10)

        # ────── 发布器 ──────
        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 调试/可视化发布器
        self.candidates_pub = self.create_publisher(PoseArray, '/debug/candidates', 10)
        self.submap_scan_pub = self.create_publisher(LaserScan, '/debug/submap_scan', 10)
        self.odom_est_pose_pub = self.create_publisher(PoseStamped, '/debug/odom_est_pose', 10)

        # ────── 服务 ──────
        self.srv_start = self.create_service(Trigger, 'start_auto_calibration', self._srv_start)
        self.srv_status = self.create_service(Trigger, 'auto_calibration_status', self._srv_status)
        self.srv_reset = self.create_service(Trigger, 'reset_calibration', self._srv_reset)

        # ────── 定时器主循环 ──────
        self.indoor_timer = self.create_timer(0.1, self._indoor_loop)
        self.outdoor_timer = self.create_timer(self.publish_rate, self._outdoor_loop)

        self.get_logger().info('自动初始位姿校准器已就绪。支持实时避障与安全主动探索。')

    # ================================================================
    #  回调与基本数据读取
    # ================================================================
    def _map_cb(self, msg):
        self.map_info = msg.info
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self._build_likelihood()
        self._find_free_space()
        self.get_logger().info(f'载入新网格地图: {msg.info.width}x{msg.info.height} @ {msg.info.resolution}m')

    def _build_likelihood(self):
        if self.map_data is None:
            return
        try:
            import cv2
            obs = (self.map_data == 100).astype(np.uint8)
            max_px = self.likelihood_max_dist / self.map_info.resolution
            dist = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            self.likelihood_field = np.clip(dist, 0, max_px).astype(np.float32)
        except Exception as e:
            self.get_logger().error(f'似然场构建失败: {e}')

    def _find_free_space(self):
        if self.map_data is None:
            return
        free = (self.map_data == 0)
        rows, cols = np.where(free)
        self.free_space_indices = np.stack([rows, cols], axis=1)

    def _scan_cb(self, msg):
        self.current_scan = msg
        # 当处于子图累积状态时，将点云与此时的里程计匹配并缓存
        is_collecting = False
        if self.indoor_phase == IndoorPhase.COLLECTING_SUBMAP1:
            is_collecting = True
        elif self.indoor_phase == IndoorPhase.COLLECTING_SUBMAP2 and hasattr(self, '_submap2_collecting'):
            is_collecting = True

        if is_collecting:
            if self.current_odom is not None and not self.submap_ready:
                pos = self.current_odom.pose.pose.position
                ori = self.current_odom.pose.pose.orientation
                yaw = self._quat_to_yaw(ori)
                self.scan_buffer.append((msg, pos.x, pos.y, yaw))
                if len(self.scan_buffer) == 1:
                    self.submap_ref_odom = (pos.x, pos.y, yaw)
                
                if len(self.scan_buffer) >= self.submap_scan_count:
                    self.submap_ready = True

    def _odom_cb(self, msg):
        self.current_odom = msg
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
        # 如果已标记真值，计算 AMCL 估计值与 Odom 推算值的偏差
        if self.gt_received and self.gt_pose is not None:
            # 订阅 /debug/odom_est_pose 的末端数据进行计算
            pass

    def _initialpose_cb(self, msg):
        """若节点处于空闲状态，将手动标记作为 Ground Truth 用于偏差校准与调试"""
        if self.indoor_phase == IndoorPhase.IDLE:
            pos = msg.pose.pose.position
            yaw = self._quat_to_yaw(msg.pose.pose.orientation)
            self.gt_pose = (pos.x, pos.y, yaw)
            
            if self.current_odom is not None:
                opos = self.current_odom.pose.pose.position
                oyaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)
                self.gt_ref_odom = (opos.x, opos.y, oyaw)
                self.gt_received = True
                self.get_logger().info(f'[校对工具] 标记真值起点: Map({pos.x:.2f}, {pos.y:.2f}, {math.degrees(yaw):.1f}°), '
                                       f'记录基准 Odom({opos.x:.2f}, {opos.y:.2f})。后续将实时输出 odom 递推偏差。')

    def _setup_rtk_sub(self, qos):
        try:
            from common_msgs.msg import RTK
            self.rtk_sub = self.create_subscription(RTK, self.rtk_topic, self._rtk_cb, qos)
        except ImportError:
            try:
                parts = self.rtk_topic_type.rsplit('.', 1)
                mod = importlib.import_module(parts[0])
                cls = getattr(mod, parts[1])
                self.rtk_sub = self.create_subscription(cls, self.rtk_topic, self._rtk_cb, qos)
            except Exception as e:
                self.get_logger().warning(f'无法动态订阅 RTK 话题: {e}')
                self.rtk_sub = None

    def _rtk_cb(self, msg):
        self.current_rtk = msg

    # ================================================================
    #  核心：子图拼接与合成
    # ================================================================
    def _build_submap(self):
        """利用缓存的 scan_buffer 和里程计数据合成为高精度激光帧"""
        if not self.scan_buffer:
            return None
        
        # 以第一帧为参考帧
        ref_scan, rx, ry, ryaw = self.scan_buffer[0]
        num_beams = len(ref_scan.ranges)
        
        # 初始化合并激光测距值为最大值
        merged_ranges = np.full(num_beams, ref_scan.range_max, dtype=np.float32)
        
        # 参考原点
        ref_origin = (rx, ry, ryaw)
        
        # 逐帧合并
        for scan, sx, sy, syaw in self.scan_buffer:
            dx = sx - rx
            dy = sy - ry
            
            # 转到参考帧坐标系
            rel_x = dx * math.cos(ryaw) + dy * math.sin(ryaw)
            rel_y = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
            rel_yaw = self._norm_angle(syaw - ryaw)
            
            # 投影激光击中点
            for j in range(num_beams):
                r = scan.ranges[j]
                if not (ref_scan.range_min < r < ref_scan.range_max):
                    continue
                
                beam_angle = ref_scan.angle_min + j * ref_scan.angle_increment
                lx = r * math.cos(beam_angle)
                ly = r * math.sin(beam_angle)
                
                # 投影到参考帧系下
                px = rel_x + lx * math.cos(rel_yaw) - ly * math.sin(rel_yaw)
                py = rel_y + lx * math.sin(rel_yaw) + ly * math.cos(rel_yaw)
                
                # 转为极坐标
                r_proj = math.sqrt(px*px + py*py)
                theta_proj = math.atan2(py, px)
                
                # 计算属于哪个合并 bin
                bin_idx = int(round((theta_proj - ref_scan.angle_min) / ref_scan.angle_increment))
                if 0 <= bin_idx < num_beams:
                    if r_proj < merged_ranges[bin_idx]:
                        merged_ranges[bin_idx] = r_proj

        # 封装成合成 LaserScan
        composite_scan = LaserScan()
        composite_scan.header.stamp = ref_scan.header.stamp
        composite_scan.header.frame_id = ref_scan.header.frame_id
        composite_scan.angle_min = ref_scan.angle_min
        composite_scan.angle_max = ref_scan.angle_max
        composite_scan.angle_increment = ref_scan.angle_increment
        composite_scan.range_min = ref_scan.range_min
        composite_scan.range_max = ref_scan.range_max
        composite_scan.ranges = merged_ranges.tolist()
        
        self.get_logger().info(f'[子图构建] 成功将 {len(self.scan_buffer)} 帧 Scan 合成为高精子图。')
        
        # 发布可视化
        self.submap_scan_pub.publish(composite_scan)
        
        return composite_scan

    # ================================================================
    #  室内校准核心状态机定时循环
    # ================================================================
    def _indoor_loop(self):
        if self.indoor_phase == IndoorPhase.IDLE:
            return
            
        elif self.indoor_phase == IndoorPhase.BOOT_DELAY:
            # 静止 2 秒确保传感器和驱动队列填充完毕
            if (self.get_clock().now() - self.boot_start_time).nanoseconds > 2.0 * 1e9:
                self.get_logger().info('[状态机] 静止就绪，开始构建第一子图 Submap 1...')
                self.scan_buffer.clear()
                self.submap_ready = False
                self.indoor_phase = IndoorPhase.COLLECTING_SUBMAP1
                
        elif self.indoor_phase == IndoorPhase.COLLECTING_SUBMAP1:
            if self.submap_ready:
                self.submap1 = self._build_submap()
                self.submap1_ref_odom = self.submap_ref_odom
                self.scan_buffer.clear()
                self.submap_ready = False
                self.get_logger().info('[状态机] Submap 1 构建完成，开始分层粗精两阶段全局评分...')
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
                self.get_logger().info('[状态机] 停止运动，等待车身静止稳定 (0.5s)...')
            else:
                elapsed = (self.get_clock().now() - self._submap2_wait_start).nanoseconds / 1e9
                if elapsed > 0.5:
                    if not hasattr(self, '_submap2_collecting'):
                        self._submap2_collecting = True
                        self.scan_buffer.clear()
                        self.submap_ready = False
                        self.get_logger().info('[状态机] 车身已稳定，开始累积 Submap 2 雷达数据...')
                    else:
                        if self.submap_ready:
                            self.submap2 = self._build_submap()
                            self.submap2_ref_odom = self.submap_ref_odom
                            self.scan_buffer.clear()
                            self.submap_ready = False
                            delattr(self, '_submap2_wait_start')
                            delattr(self, '_submap2_collecting')
                            self.get_logger().info('[状态机] Submap 2 构建完成，开始位姿传播与重评分...')
                            self.indoor_phase = IndoorPhase.FILTERING
                    
        elif self.indoor_phase == IndoorPhase.FILTERING:
            self._do_filtering_and_propagation()

    # ================================================================
    #  核心步骤 3：分层两阶段全局粒子匹配 (Coarse-to-Fine)
    # ================================================================
    def _do_hierarchical_matching(self):
        if self.likelihood_field is None or self.free_space_indices is None:
            self.get_logger().error('地图似然场尚未加载，重新等待...')
            self.indoor_phase = IndoorPhase.BOOT_DELAY
            self.boot_start_time = self.get_clock().now()
            return
            
        t0 = time.time()
        n_free = len(self.free_space_indices)
        
        # 阶段 1：粗匹配搜索
        n_coarse = min(self.rough_particles, n_free)
        coarse_indices = np.random.choice(n_free, size=n_coarse, replace=False)
        pixels = self.free_space_indices[coarse_indices]
        
        cx = (pixels[:, 1] + 0.5) * self.map_info.resolution + self.map_info.origin.position.x
        cy = (self.map_info.height - pixels[:, 0] - 0.5) * self.map_info.resolution + self.map_info.origin.position.y
        cyaw = np.random.uniform(-math.pi, math.pi, n_coarse)
        
        coarse_scores = np.zeros(n_coarse)
        # 用 submap1 作为全局匹配源
        for i in range(n_coarse):
            coarse_scores[i] = self._score_scan(self.submap1, cx[i], cy[i], cyaw[i], self.rough_beams)

        # 选出前 50 个高分粒子作为种子
        k_seeds = 50
        top_coarse_idx = np.argsort(coarse_scores)[::-1][:k_seeds]
        
        # 阶段 2：精细匹配（对每个高分粒子做局部粒子采样）
        fine_results = []
        for seed_idx in top_coarse_idx:
            sx, sy, syaw = cx[seed_idx], cy[seed_idx], cyaw[seed_idx]
            
            # 以种子为中心的高斯局部采样
            lx = np.random.normal(sx, 0.2, self.fine_particles_per_cluster)
            ly = np.random.normal(sy, 0.2, self.fine_particles_per_cluster)
            lyaw = np.random.normal(syaw, math.radians(15.0), self.fine_particles_per_cluster)
            
            for f in range(self.fine_particles_per_cluster):
                score = self._score_scan(self.submap1, lx[f], ly[f], lyaw[f], self.rough_beams)
                fine_results.append((score, lx[f], ly[f], lyaw[f]))

        # 按得分降序排列，选出 Top N 个候选 Pose 并进行概率归一化
        fine_results.sort(key=lambda x: x[0], reverse=True)
        
        # 去除空间极近的冗余候选，保持空间多样性
        unique_candidates = []
        for item in fine_results:
            sc, x, y, yaw = item
            is_redundant = False
            for u in unique_candidates:
                dist = math.sqrt((x - u[1])**2 + (y - u[2])**2)
                yaw_diff = abs(self._norm_angle(yaw - u[3]))
                if dist < 0.3 and yaw_diff < math.radians(20.0):
                    is_redundant = True
                    break
            if not is_redundant:
                unique_candidates.append((sc, x, y, yaw))
            if len(unique_candidates) >= self.top_n:
                break
                
        # 概率归一化 (使用 Softmax 思想对 log 似然进行转化)
        scores_arr = np.array([u[0] for u in unique_candidates])
        max_score = np.max(scores_arr)
        exp_scores = np.exp(scores_arr - max_score) # 缩放平移防溢出
        normalized_probs = exp_scores / np.sum(exp_scores)
        
        self.candidates = []
        for i, u in enumerate(unique_candidates):
            self.candidates.append((float(normalized_probs[i]), u[1], u[2], u[3]))
            
        elapsed = time.time() - t0
        self.get_logger().info(f'[分层匹配] 全局搜索完成。耗时: {elapsed:.2f}s。Top-N 候选:')
        for i, (prob, x, y, yaw) in enumerate(self.candidates):
            self.get_logger().info(f'  #{i}: 概率={prob:.3f}, x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°')
            
        # 发布候选 PoseArray
        self._publish_candidates()
        
        self.indoor_phase = IndoorPhase.SELECTING_ACTIVE_MOTION

    # ================================================================
    #  核心步骤 5：主动运动评估与选择 (安全过滤 + 信息增益)
    # ================================================================
    def _do_active_motion_selection(self):
        if not self.candidates:
            self.get_logger().error('候选 Pose 为空，无法进行主动运动规划，重置。')
            self._reset_indoor()
            return
            
        if self.current_scan is None:
            self.get_logger().warn('激光扫描未就位，等待数据...')
            return

        # 动态步长计算（根据候选 Pose 分布的信息熵）
        probs = np.array([c[0] for c in self.candidates])
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        # 熵值越大，说明歧义性高，采用大步长；熵值低，采用短步长精细对齐
        motion_dist = self.base_motion_distance
        if entropy < 1.5:
            motion_dist = self.base_motion_distance * 0.5
            self.get_logger().info(f'[运动选择] 信息熵偏低 ({entropy:.2f})，采用短步长探索: {motion_dist:.2f}m')
        else:
            self.get_logger().info(f'[运动选择] 信息熵高 ({entropy:.2f})，采用常规步长探索: {motion_dist:.2f}m')

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
            
            self.get_logger().debug(f'方向 {deg}°: 局部安全=True, 信息增益 (方差和)={variance_sum:.4f}')
            
            if variance_sum > max_ig:
                max_ig = variance_sum
                best_dir = (rad, delta_rot, motion_dist)

        if best_dir is not None:
            rad, delta_rot, dist = best_dir
            self.get_logger().info(f'[运动决策] 最优安全探索动作: 相对朝向={math.degrees(rad):.1f}°, 旋转={math.degrees(delta_rot):.1f}°, 距离={dist:.2f}m, IG={max_ig:.4f}')
            
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
                self.get_logger().warn('里程计信号中断，无法启动移动。')
        else:
            # 如果没有安全的方向（如陷在狭窄空间）
            self.get_logger().warn('[运动决策] 未能找到安全的平移方向。强制尝试就地旋转 45° 以获取环境信息。')
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
            self.get_logger().warn('[主动控制] 移动超时限制，停止并开始下一步匹配。')
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
            self.get_logger().info('[主动控制] 已精准抵达目标运动位姿。')
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
                        self.get_logger().warn(f'[避障停机] 前方检测到障碍物距离过近 ({r:.2f}m)！紧急触发主动停止并进行子图构建。')
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
        if self.submap1_ref_odom is None or self.submap2_ref_odom is None:
            self.get_logger().error('丢失子图参考里程计，重置状态。')
            self._reset_indoor()
            return

        # 计算 Submap 1 起点到 Submap 2 起点的实际 odom 位移 (O1 -> O2)
        o1_x, o1_y, o1_yaw = self.submap1_ref_odom
        o2_x, o2_y, o2_yaw = self.submap2_ref_odom

        dx = o2_x - o1_x
        dy = o2_y - o1_y
        
        # 局部系增量
        rel_x = dx * math.cos(o1_yaw) + dy * math.sin(o1_yaw)
        rel_y = -dx * math.sin(o1_yaw) + dy * math.cos(o1_yaw)
        rel_yaw = self._norm_angle(o2_yaw - o1_yaw)

        self.get_logger().info(f'[位姿传播] 里程计相对增量: dx={rel_x:.3f}m, dy={rel_y:.3f}m, yaw={math.degrees(rel_yaw):.1f}°')

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
        
        self.get_logger().info('[位姿重评分] 完成。最新候选分布:')
        for i, (prob, x, y, yaw) in enumerate(self.candidates):
            self.get_logger().info(f'  #{i}: 概率={prob:.3f}, x={x:.2f}, y={y:.2f}, yaw={math.degrees(yaw):.1f}°')

        # 唯一性收敛判定
        best_prob = self.candidates[0][0]
        second_prob = self.candidates[1][0] if len(self.candidates) > 1 else 0.0
        
        if best_prob > 0.8 and (second_prob == 0.0 or best_prob / second_prob >= 2.0):
            # 唯一收敛！发布 /initialpose 并初始化定位
            _, cx, cy, cyaw = self.candidates[0]
            self._publish_and_finish(cx, cy, cyaw)
        else:
            if self.active_retry_count >= self.max_active_retry:
                self.get_logger().warn(f'[收敛检测] 已到达最大尝试次数 ({self.max_active_retry}) 仍未唯一收敛，重新搜索。')
                self._reset_indoor()
            else:
                self.active_retry_count += 1
                self.get_logger().info(f'[收敛检测] 尚未唯一收敛 (最佳概率={best_prob:.3f}, 次佳={second_prob:.3f})，开启第 {self.active_retry_count} 轮探索移动...')
                
                # 更新 Submap 1 参考基础
                self.submap1 = self.submap2
                self.submap1_ref_odom = self.submap2_ref_odom
                self.indoor_phase = IndoorPhase.SELECTING_ACTIVE_MOTION

    # ================================================================
    #  激光评分实现 (似然场模型)
    # ================================================================
    def _score_scan(self, scan, x, y, yaw, max_beams):
        if self.likelihood_field is None or self.map_info is None or scan is None:
            return -1e9
            
        ranges = scan.ranges
        num_beams = len(ranges)
        step = max(1, num_beams // max_beams)

        z_hit_denom = 1.0 / (self.sigma_hit * math.sqrt(2.0 * math.pi))
        z_rand_mult = 1.0 / (scan.range_max - scan.range_min)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        total_log_likelihood = 0.0
        for i in range(0, num_beams, step):
            r = ranges[i]
            if not (scan.range_min < r < scan.range_max):
                continue
            ang = scan.angle_min + i * scan.angle_increment
            wx = x + cos_y * r * math.cos(ang) - sin_y * r * math.sin(ang)
            wy = y + sin_y * r * math.cos(ang) + cos_y * r * math.sin(ang)
            
            # 从地图似然场提取障碍距离像素坐标值
            col = int((wx - self.map_info.origin.position.x) / self.map_info.resolution)
            row = int(self.map_info.height - 1 - (wy - self.map_info.origin.position.y) / self.map_info.resolution)
            
            if 0 <= row < self.map_info.height and 0 <= col < self.map_info.width:
                dist_px = self.likelihood_field[row, col]
                dist_m = dist_px * self.map_info.resolution
                p_hit = z_hit_denom * math.exp(-0.5 * (dist_m / self.sigma_hit) ** 2)
                p = self.z_hit * p_hit + self.z_rand * z_rand_mult
                if p > 1e-12:
                    total_log_likelihood += math.log(p)
                else:
                    total_log_likelihood += math.log(1e-12)
            else:
                total_log_likelihood += math.log(1e-12)
        return total_log_likelihood

    # ================================================================
    #  发布和重置接口
    # ================================================================
    def _publish_and_finish(self, x, y, yaw):
        cov = [0.0] * 36
        cov[0] = 0.25
        cov[7] = 0.25
        cov[35] = 0.0685
        
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position = Point(x=x, y=y, z=0.0)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        msg.pose.covariance = cov
        
        self.initialpose_pub.publish(msg)
        self.get_logger().info(f'[主动定位] 成功唯一收敛！发布初始位姿: x={x:.3f}, y={y:.3f}, yaw={math.degrees(yaw):.1f}°')
        self.indoor_phase = IndoorPhase.DONE
        self.cmd_vel_pub.publish(Twist())

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
            resp.success = False
            resp.message = '全局栅格地图未就绪，校准失败'
            return resp
        if self.current_scan is None:
            resp.success = False
            resp.message = '激光雷达数据未就绪，校准失败'
            return resp
        if self.current_odom is None:
            resp.success = False
            resp.message = '里程计未就绪，校准失败'
            return resp

        self.indoor_phase = IndoorPhase.BOOT_DELAY
        self.boot_start_time = self.get_clock().now()
        self.scan_buffer.clear()
        self.submap_ready = False
        self.candidates.clear()
        self.active_retry_count = 0
        
        resp.success = True
        resp.message = '自动定位流程已成功触发启动，请保持机器人狗静止 2 秒。'
        self.get_logger().info(f'[开始流程] {resp.message}')
        return resp

    def _srv_status(self, req, resp):
        resp.success = True
        resp.message = (f'当前探索阶段: {self.indoor_phase.name}\n'
                        f'已累积重试轮数: {self.active_retry_count}\n'
                        f'当前候选 Pose 数量: {len(self.candidates)}')
        return resp

    def _srv_reset(self, req, resp):
        self._reset_indoor()
        resp.success = True
        resp.message = '已重置校准状态。'
        return resp

    def _reset_indoor(self):
        self.indoor_phase = IndoorPhase.IDLE
        self.scan_buffer.clear()
        self.submap_ready = False
        self.candidates.clear()
        self.active_retry_count = 0
        self.cmd_vel_pub.publish(Twist())
        self.get_logger().info('[重置] 室内自动位姿状态机已被重置并置为空闲。')

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
            self.get_logger().debug(f'室外 RTK 转换异常: {e}')

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
                self.get_logger().info(f'成功读取外参校准文件: {self.calibration_file}')
            except Exception as e:
                self.get_logger().error(f'读取外参文件出错: {e}')
        else:
            self.get_logger().warning(f'校准文件不存在: {self.calibration_file}')


def main(args=None):
    rclpy.init(args=args)
    node = AutoInitialPoseCalibrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
