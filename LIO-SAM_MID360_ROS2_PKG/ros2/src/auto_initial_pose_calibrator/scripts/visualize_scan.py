#!/usr/bin/env python3
"""
雷达扫描可视化脚本 — 旋转360° + 小范围移动 + 避障版

功能:
  1. 原地旋转360°，每隔约1秒采集一帧扫描
  2. 在当前位置1米范围内向8个方向依次移动探索
  3. 雷达避障：前方检测到障碍物自动停机并跳过该方向
  4. 采集完成后生成最终标注图

用法:
  ros2 run auto_initial_pose_calibrator visualize_scan.py
  ros2 run auto_initial_pose_calibrator visualize_scan.py --ros-args -p ns:=rkbot

输出:
  - /tmp/scan_viz/scan_on_map.png (最终标注图: 地图背景 + 扫描点云(绿) + 红色矩形框 + 机器人位置标记)
"""

import os
import sys
import math
import time
import yaml
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Twist

import tf2_ros

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    print("错误: 需要 matplotlib，请执行: pip3 install matplotlib")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("错误: 需要 opencv-python，请执行: pip3 install opencv-python")
    sys.exit(1)

# 加载 yaml 中的机器人差异化配置
ROBOT_CONFIGS = {}
try:
    from ament_index_python.packages import get_package_share_directory
    pkg_share = get_package_share_directory('auto_initial_pose_calibrator')
    yaml_path = os.path.join(pkg_share, 'config', 'auto_initial_pose_calibrator.yaml')
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)
        ROBOT_CONFIGS = raw['/**']['ros__parameters'].get('robot_configs', {})
except Exception:
    pass


class ScanCollector(Node):
    """旋转360° + 小范围移动 + 避障的扫描采集器"""

    def __init__(self):
        super().__init__('scan_collector')

        # ────── namespace 感知 ──────
        self.declare_parameter('ns', '')
        ns = self.get_parameter('ns').value.strip()
        config_key = ns if ns else 'default'
        robot_params = ROBOT_CONFIGS.get(config_key, {})
        if robot_params:
            self.get_logger().info(f'[config] 机器人配置: {config_key}')

        default_scan = robot_params.get('scan_topic', 'scan')
        default_map = robot_params.get('map_topic', 'map')
        default_odom = robot_params.get('odom_topic', 'lio/odom')

        # namespace 感知的话题前缀（与 auto_initial_pose_calibrator.py 保持一致）
        if ns:
            if not default_scan.startswith('/'):
                default_scan = f"/{ns}/{default_scan}"
            if not default_map.startswith('/'):
                default_map = f"/{ns}/{default_map}"
            if not default_odom.startswith('/'):
                default_odom = f"/{ns}/{default_odom}"

        self.declare_parameter('scan_topic', default_scan)
        self.declare_parameter('map_topic', default_map)
        self.declare_parameter('odom_topic', default_odom)
        self.declare_parameter('output_dir', '/tmp/scan_viz')
        self.declare_parameter('rotation_speed', 0.3)       # 旋转角速度 rad/s
        self.declare_parameter('rotation_total_deg', 360.0)  # 旋转总角度
        self.declare_parameter('save_interval', 1.0)         # 保存间隔秒
        self.declare_parameter('explore_radius', 1.0)        # 探索半径 m
        self.declare_parameter('min_safe_distance', 0.5)     # 避障安全距离 m
        self.declare_parameter('enable_explore', True)       # 是否启用小范围移动
        self.declare_parameter('enable_rotation', True)      # 是否启用旋转
        self.declare_parameter('target_frame', 'odom')       # TF目标坐标系

        self.scan_topic = self.get_parameter('scan_topic').value
        self.map_topic = self.get_parameter('map_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.output_dir = self.get_parameter('output_dir').value
        self.rotation_speed = self.get_parameter('rotation_speed').value
        self.rotation_total_rad = math.radians(self.get_parameter('rotation_total_deg').value)
        self.save_interval = self.get_parameter('save_interval').value
        self.explore_radius = self.get_parameter('explore_radius').value
        self.min_safe_distance = self.get_parameter('min_safe_distance').value
        self.enable_explore = self.get_parameter('enable_explore').value
        self.enable_rotation = self.get_parameter('enable_rotation').value
        self.target_frame = self.get_parameter('target_frame').value

        # ────── 状态变量 ──────
        self.current_scan = None
        self.current_odom = None
        self.map_data = None
        self.map_info = None
        self.saved_scans = []  # [(timestamp, scan_msg), ...]

        # ────── TF 变换（替代手工 odom R+t）──
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.scan_frame_id = None  # 首次收到 scan 时记录 frame_id

        # 旋转状态
        self.rotation_start_yaw = None
        self.rotation_accumulated = 0.0
        self.last_save_time = None

        # 移动探索状态: 8个方向 (0°,45°,90°,...,315°)
        self.explore_directions = [math.radians(d) for d in range(0, 360, 45)]
        self.current_direction_idx = 0
        self.moving = False
        self.target_odom_pose = None
        self.motion_start_odom = None
        self.motion_start_time = None

        # 阶段: rotation, explore, done
        self.phase = 'rotation' if self.enable_rotation else 'explore'

        # ────── QoS ──────
        be_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_cb, be_qos)
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self._map_cb, map_qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_cb, be_qos)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        os.makedirs(self.output_dir, exist_ok=True)

        # 主循环 10Hz
        self.timer = self.create_timer(0.1, self._main_loop)

        self.get_logger().info(f'[启动] scan={self.scan_topic}, odom={self.odom_topic}, map={self.map_topic}')
        self.get_logger().info(f'[启动] TF目标帧={self.target_frame}, 输出目录={self.output_dir}, '
                              f'旋转速度={self.rotation_speed}rad/s, '
                              f'探索半径={self.explore_radius}m, 避障距离={self.min_safe_distance}m')
        self.get_logger().info(f'[启动] 阶段: 旋转360° → 8方向探索 → 完成退出')
        self.get_logger().info('[启动] 等待 scan / odom / TF 数据就绪...')

    # ────── 回调 ──────
    def _scan_cb(self, msg):
        self.current_scan = msg
        if not hasattr(self, '_scan_first_logged'):
            self._scan_first_logged = True
            self.scan_frame_id = msg.header.frame_id.strip()
            self.get_logger().info(
                f'[scan] 已收到: frame_id="{msg.header.frame_id}", '
                f'目标变换系="{self.target_frame}", '
                f'FOV={math.degrees(msg.angle_max - msg.angle_min):.1f}°, '
                f'beams={len(msg.ranges)}, '
                f'(使用TF变换替代手工R+t)'
            )

    def _odom_cb(self, msg):
        self.current_odom = msg
        if not hasattr(self, '_odom_first_logged'):
            self._odom_first_logged = True
            self.get_logger().info(f'[odom] 已收到: frame_id={msg.header.frame_id}')

    def _map_cb(self, msg):
        self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.get_logger().info(f'[map] 已收到: {msg.info.width}x{msg.info.height} @ {msg.info.resolution}m')

    # ────── 工具 ──────
    @staticmethod
    def _quat_to_yaw(q):
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _norm_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def _scan_to_xy(self, scan):
        if scan is None:
            return np.empty((0, 2))
        ranges = np.array(scan.ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        valid = (ranges > scan.range_min) & (ranges < scan.range_max)
        if not np.any(valid):
            return np.empty((0, 2))
        r = ranges[valid]
        a = angles[valid]
        return np.column_stack((r * np.cos(a), r * np.sin(a)))

    # ────── 避障检测 ──────
    def _is_direction_safe(self, local_angle, distance):
        """检测局部方向是否安全（无近处障碍物）"""
        if self.current_scan is None:
            return False
        ranges = self.current_scan.ranges
        sector = math.radians(30.0)
        for i, r in enumerate(ranges):
            if not (self.current_scan.range_min < r < self.current_scan.range_max):
                continue
            beam_angle = self.current_scan.angle_min + i * self.current_scan.angle_increment
            if abs(self._norm_angle(beam_angle - local_angle)) <= sector:
                if r < distance + self.min_safe_distance:
                    return False
        return True

    def _check_front_obstacle(self):
        """检查前进方向是否有障碍物（避障停机用）"""
        if self.current_scan is None:
            return False
        ranges = self.current_scan.ranges
        sector = math.radians(30.0)
        for i, r in enumerate(ranges):
            if not (self.current_scan.range_min < r < self.current_scan.range_max):
                continue
            beam_angle = self.current_scan.angle_min + i * self.current_scan.angle_increment
            if abs(self._norm_angle(beam_angle)) <= sector:
                if r < self.min_safe_distance:
                    return True
        return False

    # ────── 主循环 ──────
    def _main_loop(self):
        if self.current_scan is None or self.current_odom is None:
            return  # 等待数据就绪

        if self.phase == 'rotation':
            self._do_rotation()
        elif self.phase == 'explore':
            self._do_explore()
        elif self.phase == 'done':
            pass  # 等待生成最终叠加图

    # ────── 阶段1：旋转360° ──────
    def _do_rotation(self):
        curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)

        # 初始化旋转起始角度
        if self.rotation_start_yaw is None:
            self.rotation_start_yaw = curr_yaw
            self.last_save_time = self.get_clock().now()
            self._last_yaw_log_time = self.get_clock().now()
            self.get_logger().info(f'[旋转] 开始360°旋转采集，角速度={self.rotation_speed}rad/s, '
                                   f'保存间隔={self.save_interval}s, 起始yaw={math.degrees(curr_yaw):.1f}°')

        # 累积旋转角度
        delta = abs(self._norm_angle(curr_yaw - self.rotation_start_yaw))
        self.rotation_start_yaw = curr_yaw
        self.rotation_accumulated += delta

        # 每2秒输出一次 yaw 变化诊断
        now = self.get_clock().now()
        if (now - self._last_yaw_log_time).nanoseconds > 2.0 * 1e9:
            self._last_yaw_log_time = now
            self.get_logger().info(
                f'[旋转诊断] 当前 yaw={math.degrees(curr_yaw):.1f}°, '
                f'扫␋计={math.degrees(self.rotation_accumulated):.0f}°, '
                f'odom位置=({self.current_odom.pose.pose.position.x:.2f}, '
                f'{self.current_odom.pose.pose.position.y:.2f}), '
                f'已保存{len(self.saved_scans)}帧'
            )

        # 累积旋转角度
        delta = abs(self._norm_angle(curr_yaw - self.rotation_start_yaw))
        self.rotation_start_yaw = curr_yaw
        self.rotation_accumulated += delta

        # 定时保存帧
        now = self.get_clock().now()
        elapsed = (now - self.last_save_time).nanoseconds / 1e9
        if elapsed >= self.save_interval:
            self._save_current_scan()
            self.last_save_time = now

        # 判断旋转完成
        if self.rotation_accumulated >= self.rotation_total_rad:
            self.cmd_vel_pub.publish(Twist())  # 停止
            self._save_current_scan()  # 保存最后一帧
            self.get_logger().info(f'[旋转] 完成！已转 {math.degrees(self.rotation_accumulated):.0f}°, '
                                   f'共保存 {len(self.saved_scans)} 帧')
            if self.enable_explore:
                self.phase = 'explore'
                self.current_direction_idx = 0
                self.moving = False
                self.get_logger().info('[探索] 开始8方向小范围移动探索...')
            else:
                self._finish()

        # 发布旋转速度
        cmd = Twist()
        cmd.angular.z = self.rotation_speed
        self.cmd_vel_pub.publish(cmd)

    # ────── 阶段2：8方向移动探索（自适应旋转优先）──
    def _do_explore(self):
        if self.current_direction_idx >= len(self.explore_directions):
            self._finish()
            return

        if not self.moving:
            # 选择下一个方向
            direction = self.explore_directions[self.current_direction_idx]
            direction_deg = math.degrees(direction)

            # 检查该方向是否安全
            if not self._is_direction_safe(direction, self.explore_radius):
                self.get_logger().warn(
                    f'[探索] 方向 {direction_deg:.0f}° 不安全，跳过 (方向{self.current_direction_idx+1}/8)')
                self.current_direction_idx += 1
                self._save_current_scan()
                return

            # 计算目标里程计位姿（相对于机器人当前位置+朝向）
            curr_pos = self.current_odom.pose.pose.position
            curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)

            tx = curr_pos.x + self.explore_radius * math.cos(curr_yaw + direction)
            ty = curr_pos.y + self.explore_radius * math.sin(curr_yaw + direction)
            self.target_odom_pose = (tx, ty)
            self.motion_start_odom = self.current_odom
            self.motion_start_time = self.get_clock().now()
            self.moving = True
            self._last_yaw_log_time = self.get_clock().now()

            self.get_logger().info(
                f'[探索] 方向 {direction_deg:.0f}° (方向{self.current_direction_idx+1}/8), '
                f'目标: ({tx:.2f}, {ty:.2f})')

        # ── 移动控制：旋转优先自适应（补偿 yaw 漂移）──
        if self.target_odom_pose is None or self.current_odom is None:
            self._reset_motion()
            return

        # 全局超时保护（15s）
        time_elapsed = (self.get_clock().now() - self.motion_start_time).nanoseconds / 1e9
        if time_elapsed > 15.0:
            self.get_logger().warn(
                f'[探索] 方向{self.current_direction_idx+1} 超时({time_elapsed:.1f}s)，跳过')
            self.cmd_vel_pub.publish(Twist())
            self._reset_motion()
            return

        # 避障停机
        if self._check_front_obstacle():
            self.get_logger().warn(f'[探索] 前方障碍物！停机，跳过方向{self.current_direction_idx+1}')
            self.cmd_vel_pub.publish(Twist())
            self._reset_motion()
            return

        # 获取当前位置和目标
        tx, ty = self.target_odom_pose
        curr_pos = self.current_odom.pose.pose.position
        curr_yaw = self._quat_to_yaw(self.current_odom.pose.pose.orientation)

        dx = tx - curr_pos.x
        dy = ty - curr_pos.y
        dist_to_target = math.sqrt(dx * dx + dy * dy)

        # 到达判定
        if dist_to_target < 0.15:
            self.get_logger().info(f'[探索] 方向{self.current_direction_idx+1} 到达目标')
            self.cmd_vel_pub.publish(Twist())
            self._reset_motion()
            return

        # 目标方向角（全局坐标系下）
        target_yaw = math.atan2(dy, dx)
        yaw_error = self._norm_angle(target_yaw - curr_yaw)
        yaw_error_abs = abs(yaw_error)

        # 2秒一次状态日志
        now = self.get_clock().now()
        if (now - self._last_yaw_log_time).nanoseconds > 2.0 * 1e9:
            self._last_yaw_log_time = now
            self.get_logger().info(
                f'[探索诊断] yaw={math.degrees(curr_yaw):.1f}° '
                f'目标yaw={math.degrees(target_yaw):.1f}° '
                f'误差={math.degrees(yaw_error):.1f}° '
                f'距离={dist_to_target:.2f}m '
                f'位置=({curr_pos.x:.2f},{curr_pos.y:.2f})')

        # ── 核心策略：旋转优先 ──
        # 非完整约束机器人必须先对准目标方向才能有效前进。
        # 阈值 0.25 rad (~14°)：低于此才允许前进，否则原地旋转。
        ROTATE_THRESHOLD = 0.25  # rad

        if yaw_error_abs > ROTATE_THRESHOLD:
            # 原地旋转，不前进
            wz = 0.6 * yaw_error                  # P控制（比之前更激进）
            wz = max(-0.4, min(0.4, wz))          # 限幅稍提高
            cmd = Twist()
            cmd.angular.z = wz
            self.cmd_vel_pub.publish(cmd)
            return

        # ── 对准后前进（恒速 + 纠偏）──
        # 使用恒速避免 P 控减速后被漂移推开
        vx = 0.2                                      # 恒速前进

        # yaw 纠偏（漂移补偿）
        wz = 0.8 * yaw_error                          # 较强纠偏
        wz = max(-0.35, min(0.35, wz))

        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.cmd_vel_pub.publish(cmd)

    def _reset_motion(self):
        """重置移动状态，前进到下一个方向"""
        self.moving = False
        self._save_current_scan()
        self.current_direction_idx += 1

    # ────── 保存当前帧（使用 TF 变换）──
    def _save_current_scan(self):
        if self.current_scan is None:
            return
        if self.scan_frame_id is None:
            self.get_logger().warn('[保存] scan frame_id 未知，跳过')
            return

        ts = self.get_clock().now().nanoseconds / 1e9
        ranges_copy = list(self.current_scan.ranges)

        # ── 核心：通过 TF 查询 scan_frame → target_frame 的精确变换 ──
        # 使用 scan 自身的时间戳，TF 会自动插值到该时刻
        try:
            scan_time = rclpy.time.Time.from_msg(self.current_scan.header.stamp)
            # 如果 stamp 为零，使用最新可用变换
            if scan_time.nanoseconds == 0:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, self.scan_frame_id, rclpy.time.Time())
            else:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, self.scan_frame_id, scan_time,
                    timeout=rclpy.duration.Duration(seconds=0.1))

            t = transform.transform.translation
            q = transform.transform.rotation
            yaw = self._quat_to_yaw(q)

            self.saved_scans.append({
                'ts': ts,
                'ranges': ranges_copy,
                'angle_min': self.current_scan.angle_min,
                'angle_increment': self.current_scan.angle_increment,
                'range_min': self.current_scan.range_min,
                'range_max': self.current_scan.range_max,
                # TF 变换参数（用于点云坐标转换）
                'tf_x': t.x,
                'tf_y': t.y,
                'tf_z': t.z,
                'tf_qx': q.x,
                'tf_qy': q.y,
                'tf_qz': q.z,
                'tf_qw': q.w,
                'tf_yaw': yaw,
                # 也保存 odom 原始值用于绘制朝向箭头等辅助信息
                'odom_x': self.current_odom.pose.pose.position.x if self.current_odom else t.x,
                'odom_y': self.current_odom.pose.pose.position.y if self.current_odom else t.y,
                'odom_yaw': self._quat_to_yaw(self.current_odom.pose.pose.orientation) if self.current_odom else yaw,
            })
            self.get_logger().info(
                f'[保存] 第 {len(self.saved_scans)} 帧已缓存 (TF), '
                f'{self.scan_frame_id}→{self.target_frame}: '
                f't=({t.x:.2f},{t.y:.2f}) yaw={math.degrees(yaw):.1f}°')

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'[保存] TF 查询失败 ({self.scan_frame_id}→{self.target_frame}): {e}')
            return

    # ────── 完成 ──────
    # ────── Scan-to-Map 配准算法 ──────
    def _build_likelihood_field(self):
        """从 OccupancyGrid 构建障碍物距离场

        使用 cv2.distanceTransform 计算每个像素到最近障碍物的距离（米）。
        用于评分函数查询扫描点到地图墙壁的接近程度。
        """
        if self.map_data is None or self.map_info is None:
            return None, None

        obs = (self.map_data == 100).astype(np.uint8)
        # distanceTransform: 非零像素（非障碍物）到最近零像素（障碍物）的距离
        dist_px = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        # 转换为米
        dist_m = dist_px * self.map_info.resolution
        self.get_logger().info(
            f'[似然场] 构建完成: {self.map_info.width}x{self.map_info.height}, '
            f'分辨率={self.map_info.resolution:.3f}m/pix, '
            f'最大距离={dist_m.max():.2f}m')
        return dist_m, None

    def _world_to_pixel(self, x, y):
        """世界坐标 (m) → 像素坐标 (row, col)"""
        res = self.map_info.resolution
        col = int((x - self.map_info.origin.position.x) / res)
        row = int(self.map_info.height - 1 - (y - self.map_info.origin.position.y) / res)
        return row, col

    def _score_points_at_pose(self, points, x, y, yaw):
        """向量化评分：近墙命中率 + 距离中位数（双指标）

        核心思想：激光扫描点本质上是"打在墙面上的点"，
        正确匹配位置应该让**大量点同时贴近地图障碍物边缘**。
        空白区域（无墙）的命中率≈0，自然被淘汰。

        Args:
            points: (N,2) numpy — 归一化后的扫描点（相对点云中心）
            x, y, yaw: 候选位姿（map frame 中点云中心的位置和朝向）

        Returns:
            score: float（越高越好，正值=有足够近墙命中）
        """
        if self.likelihood_field is None:
            return -1e9

        N = len(points)
        if N < 10:
            return -1e9

        # ── 1. 刚体变换（向量化）──
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        wx = cos_y * points[:, 0] - sin_y * points[:, 1] + x
        wy = sin_y * points[:, 0] + cos_y * points[:, 1] + y

        # ── 2. 世界坐标 → 像素坐标（向量化）──
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        H, W = self.map_info.height, self.map_info.width

        cols_f = (wx - ox) / res
        rows_f = H - 1 - (wy - oy) / res

        # ── 3. 边界掩码 ──
        valid = (cols_f >= 0) & (cols_f < W - 0.5) & (rows_f >= 0) & (rows_f < H - 0.5)

        # ── 4. 查询距离场（仅有效点）──
        n_valid = np.sum(valid)
        if n_valid < N * 0.3:  # 少于30%的点在地图范围内 → 无效
            return -1e9

        cols_i = np.clip(cols_f[valid].astype(np.int32), 0, W - 1)
        rows_i = np.clip(rows_f[valid].astype(np.int32), 0, H - 1)
        dists = self.likelihood_field[rows_i, cols_i]

        # ── 5. 双指标评分 ──
        hit_threshold = 0.5      # "近墙"判定距离 m
        good_threshold = 1.5     # "良好匹配"距离 m

        n_hits = np.sum(dists < hit_threshold)     # 近墙命中数
        n_good = np.sum(dists < good_threshold)     # 良好匹配数
        hit_rate = n_hits / N                       # 近墙命中率
        good_rate = n_good / N                      # 良好匹配率

        # 距离统计
        med_dist = float(np.median(dists))
        p90_dist = float(np.percentile(dists, 90))
        mean_dist = float(np.mean(dists))

        # 综合评分公式:
        #   主项: hit_rate * 100（命中率驱动，上限100）
        #   副项: good_rate * 20（良好率加分）
        #   惩罚: -med_dist（中位数距离越小越好）
        #   惩罚: -p90_dist * 0.3（90百分位惩罚离群点）
        score = (
            hit_rate * 100.0 +
            good_rate * 20.0 -
            med_dist -
            p90_dist * 0.3
        )

        return score

    # ────── Hu矩 形状匹配 ──────
    def _points_to_scan_contour(self, points, img_size=200, phys_size_m=20.0):
        """将扫描点渲染为封闭多边形，提取外轮廓用于Hu矩匹配

        核心逻辑：360° LiDAR 扫描点按角度排序后连接→封闭多边形→填充→提取轮廓。
        该轮廓的 Hu 矩对旋转/平移/缩放不变，适合与地图局部房间轮廓做形状比较。

        Args:
            points: (N,2) numpy — 已居中的扫描点
            img_size: 渲染图像边长 px
            phys_size_m: 图像覆盖的物理尺寸 m

        Returns:
            (contour, img) — OpenCV contour 和渲染后的二值图像
        """
        meters_per_px = phys_size_m / img_size
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        half = img_size // 2

        # 按角度排序（LiDAR 扫描点天然有序，但降采样后顺序可能被打乱）
        angles = np.arctan2(points[:, 1], points[:, 0])
        sorted_idx = np.argsort(angles)

        pts_px = []
        for idx in sorted_idx:
            px = int(points[idx, 0] / meters_per_px + half)
            py = int(half - points[idx, 1] / meters_per_px)
            if 0 <= px < img_size and 0 <= py < img_size:
                pts_px.append([px, py])

        if len(pts_px) < 3:
            return None, img

        pts_arr = np.array(pts_px, dtype=np.int32)

        # 画轮廓线 + 填充内部
        cv2.polylines(img, [pts_arr], isClosed=True, color=255, thickness=1)
        cv2.fillPoly(img, [pts_arr], 255)

        # 形态学闭操作：填补小间隙
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, img

        return max(contours, key=cv2.contourArea), img

    def _extract_map_contour_at(self, x, y, phys_size_m=20.0):
        """在地图位置 (x,y) 周围提取房间墙面轮廓（用于 Hu 矩比较）

        Args:
            x, y: 地图坐标系下的候选位置 (m)
            phys_size_m: 提取窗口物理尺寸 (m)，与 scan_contour 窗口一致

        Returns:
            (contour, img) — OpenCV contour 和提取窗口的二值图
        """
        res = self.map_info.resolution
        half_w_px = max(int(phys_size_m / 2 / res), 10)

        # 世界坐标 → 像素中心
        cx_px = int((x - self.map_info.origin.position.x) / res)
        cy_px = int(self.map_info.height - 1 -
                      (y - self.map_info.origin.position.y) / res)

        # 窗口边界
        r1 = max(0, cy_px - half_w_px)
        r2 = min(self.map_info.height, cy_px + half_w_px)
        c1 = max(0, cx_px - half_w_px)
        c2 = min(self.map_info.width, cx_px + half_w_px)

        if r2 - r1 < 10 or c2 - c1 < 10:
            return None, None

        roi = self.map_data[r1:r2, c1:c2]
        wall_binary = (roi == 100).astype(np.uint8) * 255

        if np.sum(wall_binary) < 50:  # 墙面太少 → 空白区域
            return None, None

        # 形态学：膨胀连接邻近墙面断点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        wall_binary = cv2.dilate(wall_binary, kernel, iterations=2)
        wall_binary = cv2.erode(wall_binary, kernel, iterations=1)

        contours, _ = cv2.findContours(wall_binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, wall_binary

        # 取面积最大的轮廓（主房间墙壁）
        main_contour = max(contours, key=cv2.contourArea)

        # 过滤太小或太大的轮廓
        min_area = 20   # 至少 20 px²
        if cv2.contourArea(main_contour) < min_area:
            return None, wall_binary

        return main_contour, wall_binary

    def _get_odom_to_map_transform(self):
        """查询最新的 odom → map 变换（用于把扫描数据从odom转到map坐标系）

        Returns:
            (odom_x_in_map, odom_y_in_map, odom_yaw_in_map) 或 None 失败时
        """
        try:
            t = self.tf_buffer.lookup_transform(
                'map', self.target_frame, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            tx = t.transform.translation.x
            ty = t.transform.translation.y
            yaw = self._quat_to_yaw(t.transform.rotation)
            self.get_logger().info(
                f'[TF] odom→map: t=({tx:.2f}, {ty:.2f}) yaw={math.degrees(yaw):.1f}°')
            return (tx, ty, yaw)
        except Exception as e:
            self.get_logger().warn(f'[TF] 无法查询 odom→map: {e}')
            return None

    def _merge_all_scan_points(self):
        """将所有保存的扫描帧合并为一个点云（odom/target_frame 系）"""
        all_points = []
        for saved in self.saved_scans:
            pts = self._scan_to_world_xy(saved)
            if len(pts) > 0:
                all_points.append(pts)
        if not all_points:
            return np.empty((0, 2))
        merged = np.vstack(all_points)
        # 降采样加速匹配
        step = max(1, len(merged) // 500)
        return merged[::step]

    def _merge_all_scan_points_map_frame(self):
        """将所有扫描帧合并并变换到 map 坐标系

        流程：scan_frame → odom(通过保存的TF) → map(通过最新TF)
        """
        tf_info = self._get_odom_to_map_transform()
        if tf_info is None:
            self.get_logger().warn('[配准] 无法获取 odom→map，回退到 odom 系点云')
            return self._merge_all_scan_points(), None

        odom_x, odom_y, odom_yaw = tf_info

        # 先得到 odom 系下的合并点云
        odom_points = self._merge_all_scan_points()
        if len(odom_points) == 0:
            return np.empty((0, 2)), tf_info

        # odom → map 刚体变换
        cos_o, sin_o = math.cos(odom_yaw), math.sin(odom_yaw)
        mx = cos_o * odom_points[:, 0] - sin_o * odom_points[:, 1] + odom_x
        my = sin_o * odom_points[:, 0] + cos_o * odom_points[:, 1] + odom_y
        map_points = np.stack([mx, my], axis=1)

        self.get_logger().info(
            f'[配准] 点云已转换到 map 系: {len(map_points)} 个点, '
            f'odom原点在map中的位置=({odom_x:.2f}, {odom_y:.2f})')
        return map_points, tf_info

    def _scan_to_map_match(self):
        """Scan-to-Map 图形配准：Hu矩粗定位 + 似然场精搜

        两阶段策略，利用 Hu 矩的旋转不变性大幅加速：
          Phase 1 (Hu矩粗搜): 逐位置网格，用 cv2.matchShapes 比较扫描轮廓与地图房间轮廓。
                             Hu 矩天然旋转/缩放不变 → 不需要遍历角度，极快。
          Phase 2 (似然场精搜): 在 Top-K 粗搜候选位置，遍历角度+细网格，用似然场精确评分。

        Returns:
            best_pose: (x, y, yaw) in map frame, 或 None 失败时
            match_score: 最佳评分值
        """
        self.get_logger().info('[配准] 开始 Scan-to-Map 配准 (Hu矩粗搜 + 似然场精搜)...')

        # ── 0. 准备 ──
        self.likelihood_field, _ = self._build_likelihood_field()
        if self.likelihood_field is None:
            self.get_logger().error('[配准] 地图不可用')
            return None, -1e9

        scan_points_odom = self._merge_all_scan_points()
        if len(scan_points_odom) < 20:
            self.get_logger().warn(f'[配准] 点云点数不足 ({len(scan_points_odom)})')
            return None, -1e9
        self.get_logger().info(f'[配准] 合并点云: {len(scan_points_odom)} 个点')

        # 居中归一化
        cx_odom = scan_points_odom[:, 0].mean()
        cy_odom = scan_points_odom[:, 1].mean()
        centered_pts = scan_points_odom.copy()
        centered_pts[:, 0] -= cx_odom
        centered_pts[:, 1] -= cy_odom

        t0 = time.time()

        # ── 1. 预计算扫描轮廓（Hu矩用）──
        scan_extent = max(
            centered_pts[:, 0].max() - centered_pts[:, 0].min(),
            centered_pts[:, 1].max() - centered_pts[:, 1].min())
        # 窗口至少 15m，确保扫描轮廓与地图房间有足够上下文比较
        scan_window_m = max(scan_extent * 1.5, 15.0)

        scan_contour, _ = self._points_to_scan_contour(
            centered_pts, img_size=200, phys_size_m=scan_window_m)

        if scan_contour is None:
            self.get_logger().warn('[配准] 扫描轮廓提取失败')
            return None, -1e9

        scan_area_px = cv2.contourArea(scan_contour)
        scan_hull = cv2.convexHull(scan_contour)
        scan_solidity = scan_area_px / max(cv2.contourArea(scan_hull), 1.0) if scan_area_px > 0 else 0

        self.get_logger().info(
            f'[配准] 扫描轮廓: 窗口={scan_window_m:.1f}m, '
            f'面积={scan_area_px:.0f}px², 凸度={scan_solidity:.2f}')

        # ══════════════════════════════════════════════════════════════
        # Phase 1: Hu矩粗搜 — 只搜位置，不搜角度 (Hu矩旋转不变)
        # ══════════════════════════════════════════════════════════════
        coarse_step_m = 1.5   # 位置采样步长 m（比之前更密，但免去角度乘数）
        n_keep = 5             # 保留 Top-K 进入 Phase 2

        res = self.map_info.resolution
        map_w_m = self.map_info.width * res
        map_h_m = self.map_info.height * res
        map_origin_x = self.map_info.origin.position.x
        map_origin_y = self.map_info.origin.position.y

        xs_coarse = np.arange(map_origin_x + 1.5, map_origin_x + map_w_m - 1.5, coarse_step_m)
        ys_coarse = np.arange(map_origin_y + 1.5, map_origin_y + map_h_m - 1.5, coarse_step_m)
        n_positions = len(xs_coarse) * len(ys_coarse)

        self.get_logger().info(
            f'[配准] Phase1 Hu矩粗搜: '
            f'网格 {len(xs_coarse)}x{len(ys_coarse)}={n_positions} 位置 '
            f'(窗口={scan_window_m:.1f}m, 免角度遍历)')

        # Top-K 小顶堆: (score, x, y)
        # score = -matchShapes距离 → 越大越好 → 用小顶堆保留最小的K个→K个最好的
        top_candidates = []  # [(score, x, y), ...] 升序, 保留最好的 K 个

        count_hu = 0
        for ax_val in xs_coarse:
            for ay_val in ys_coarse:
                count_hu += 1

                map_contour, _ = self._extract_map_contour_at(ax_val, ay_val, scan_window_m)
                if map_contour is None:
                    continue

                # matchShapes: 越小越相似 (I2方式用Hu矩不变量)
                dist = cv2.matchShapes(scan_contour, map_contour,
                                       cv2.CONTOURS_MATCH_I2, 0)

                # 额外过滤: 轮廓面积差异太大 → 惩罚
                map_area = cv2.contourArea(map_contour)
                area_ratio = min(map_area, scan_area_px) / max(map_area, scan_area_px, 1.0)
                penalty = (1.0 - area_ratio) * 2.0
                score = -(dist + penalty)  # 越高越好

                if len(top_candidates) < n_keep:
                    top_candidates.append((score, ax_val, ay_val))
                    top_candidates.sort(key=lambda x: x[0])  # 升序
                elif score > top_candidates[0][0]:
                    top_candidates[0] = (score, ax_val, ay_val)
                    top_candidates.sort(key=lambda x: x[0])

                # 进度日志
                if count_hu % 200 == 0:
                    elapsed = time.time() - t0
                    self.get_logger().info(
                        f'[Hu粗搜] {count_hu}/{n_positions} 位置 '
                        f'({elapsed:.1f}s), 当前Top1={top_candidates[-1][0]:.3f}')

        elapsed_hu = time.time() - t0
        if not top_candidates:
            self.get_logger().warn('[配准] Hu粗搜无有效候选（地图无房间轮廓？）')
            return None, -1e9

        # 排序从最好到最差
        top_candidates.sort(key=lambda x: x[0], reverse=True)

        self.get_logger().info(
            f'[配准] Hu粗搜完成 ({elapsed_hu:.1f}s, {count_hu}位置): '
            f'Top-K 候选:')
        for rank, (s, x, y) in enumerate(top_candidates):
            self.get_logger().info(
                f'  Hu #{rank}: score={s:.3f}, map=({x:.2f}, {y:.2f})')

        # ══════════════════════════════════════════════════════════════
        # Phase 2: 似然场精搜 — 在 Top-K 位置做全角度+细网格搜索
        # ══════════════════════════════════════════════════════════════
        angle_step_deg = 10.0   # 角度步长 °
        n_angles = int(360.0 / angle_step_deg)  # 36 个角度
        pos_radius_m = 2.0      # 位置微调半径 m
        n_pos_fine = 7          # 位置细搜格点数（7x7=49个点）

        best_score = -1e9
        best_pose = None
        all_fine_results = []

        t1 = time.time()
        for rank, (hu_score, hx, hy) in enumerate(top_candidates):
            xs_fine = np.linspace(hx - pos_radius_m, hx + pos_radius_m, n_pos_fine)
            ys_fine = np.linspace(hy - pos_radius_m, hy + pos_radius_m, n_pos_fine)

            for ax_val in xs_fine:
                for ay_val in ys_fine:
                    for adeg in range(n_angles):
                        ayaw = math.radians(adeg * angle_step_deg)
                        s = self._score_points_at_pose(
                            centered_pts,
                            ax_val + cx_odom * math.cos(ayaw) - cy_odom * math.sin(ayaw),
                            ay_val + cx_odom * math.sin(ayaw) + cy_odom * math.cos(ayaw),
                            ayaw)
                        if s > best_score:
                            best_score = s
                            best_pose = (ax_val, ay_val, ayaw)
                        all_fine_results.append((s, ax_val, ay_val, ayaw))

        elapsed = time.time() - t0
        elapsed_fine = time.time() - t1

        if best_pose is None:
            self.get_logger().warn('[配准] Phase2 无有效结果')
            return None, -1e9

        fx, fy, fyaw = best_pose
        full_fx = fx + cx_odom * math.cos(fyaw) - cy_odom * math.sin(fyaw)
        full_fy = fy + cx_odom * math.sin(fyaw) + cy_odom * math.cos(fyaw)

        total_evals = n_positions + len(top_candidates) * n_pos_fine * n_pos_fine * n_angles
        self.get_logger().info(
            f'[配准] 完成 ({elapsed:.1f}s total, Hu={elapsed_hu:.1f}s + '
            f'似然={elapsed_fine:.1f}s, ~{total_evals}评分): '
            f'最佳=({full_fx:.2f}, {full_fy:.2f}, {math.degrees(fyaw):.1f}°), '
            f'score={best_score:.2f}')

        # Top-5 精搜候选
        all_fine_results.sort(key=lambda x: x[0], reverse=True)
        for rank in range(min(5, len(all_fine_results))):
            s, x, y, yaw = all_fine_results[rank]
            rx = x + cx_odom * math.cos(yaw) - cy_odom * math.sin(yaw)
            ry = y + cx_odom * math.sin(yaw) + cy_odom * math.cos(yaw)
            self.get_logger().info(
                f'  精搜 #{rank}: score={s:.2f}, '
                f'map=({rx:.2f},{ry:.2f},{math.degrees(yaw):.1f}°)')

        return (full_fx, full_fy, fyaw), best_score

    def _plot_scan_on_map(self):
        """将扫描数据通过图形配准放到地图正确位置，画红框标区域+机器人位置

        流程：
          1. 调用 _scan_to_map_match() 做似然场全局搜索
          2. 用匹配结果将 odom 系点云转到 map 系
          3. 地图背景 + 扫描数据叠加 + 红色边界框 + 机器人位置标记
        """
        self.get_logger().info('[可视化] 开始 Scan-on-Map 图形配准标注...')

        # ── 0. 前置检查 ──
        if self.map_data is None or self.map_info is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, '无地图数据\n请确认 SLAM / map_server 正在运行',
                    ha='center', va='center', fontsize=16, color='red')
            ax.set_title('Scan on Map - Error')
            filepath = os.path.join(self.output_dir, 'scan_on_map.png')
            plt.savefig(filepath, dpi=150)
            plt.close()
            self.get_logger().warn('[可视化] 无地图数据')
            return None

        # ── 1. 图形配准（两阶段全局搜索）──
        match_result, score = self._scan_to_map_match()
        if match_result is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, '配准失败\n似然场搜索未找到有效匹配',
                    ha='center', va='center', fontsize=16, color='red')
            ax.set_title('Scan on Map - Match Failed')
            filepath = os.path.join(self.output_dir, 'scan_on_map.png')
            plt.savefig(filepath, dpi=150)
            plt.close()
            self.get_logger().warn('[可视化] 配准失败')
            return None

        robot_x, robot_y, robot_yaw = match_result

        # ── 2. 用配准结果将点云转到 map 系 ──
        odom_points = self._merge_all_scan_points()
        if len(odom_points) == 0:
            self.get_logger().warn('[可视化] 无扫描数据')
            return None

        # 点云中心（与 _scan_to_map_match 中的归一化一致）
        cx_odom = odom_points[:, 0].mean()
        cy_odom = odom_points[:, 1].mean()
        centered = odom_points.copy()
        centered[:, 0] -= cx_odom
        centered[:, 1] -= cy_odom

        # 用配准的 (robot_x, robot_y, robot_yaw) 变换到 map 系
        cos_r, sin_r = math.cos(robot_yaw), math.sin(robot_yaw)
        # 旋转 + 平移
        rx = cos_r * centered[:, 0] - sin_r * centered[:, 1]
        ry = sin_r * centered[:, 0] + cos_r * centered[:, 1]
        map_pts_x = rx + robot_x
        map_pts_y = ry + robot_y
        map_pts = np.stack([map_pts_x, map_pts_y], axis=1)

        # 也记录 TF 位置用于对比标注
        tf_info = self._get_odom_to_map_transform()

        # ── 3. 绘制 ──
        fig, ax = plt.subplots(figsize=(14, 12))
        ax.set_aspect('equal')

        # 地图背景
        map_display = np.zeros((self.map_info.height, self.map_info.width, 3), dtype=np.float32)
        map_display[self.map_data == 0] = [1.0, 1.0, 1.0]
        map_display[self.map_data == 100] = [0.0, 0.0, 0.0]
        map_display[self.map_data == -1] = [0.7, 0.7, 0.7]
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        extent = [origin_x, origin_x + self.map_info.width * res,
                  origin_y, origin_y + self.map_info.height * res]
        ax.imshow(map_display, origin='lower', extent=extent)

        # 扫描点云（绿色半透明）
        ax.scatter(map_pts[:, 0], map_pts[:, 1], s=2, c='lime', alpha=0.6,
                   edgecolors='none', label=f'Scan Data ({len(map_pts)} pts)')

        # ── 4. 红色边界框（包围所有扫描点）──
        x_min, x_max = map_pts[:, 0].min(), map_pts[:, 0].max()
        y_min, y_max = map_pts[:, 1].min(), map_pts[:, 1].max()
        pad = 0.5
        rect_x = x_min - pad; rect_w = (x_max - x_min) + 2*pad
        rect_y = y_min - pad; rect_h = (y_max - y_min) + 2*pad
        from matplotlib.patches import Rectangle
        red_rect = Rectangle((rect_x, rect_y), rect_w, rect_h,
                              linewidth=2.5, edgecolor='red', facecolor='none',
                              linestyle='-', zorder=9)
        ax.add_patch(red_rect)

        ax.text(rect_x, rect_y + rect_h + 0.3,
                f'{rect_w:.1f}m × {rect_h:.1f}m',
                fontsize=11, color='red', fontweight='bold',
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        # ── 5. 机器人位置（配准结果）──
        ax.plot(robot_x, robot_y, 'rX', markersize=20, markeredgewidth=4,
                zorder=15, label=f'Robot (matched) ({robot_x:.2f}, {robot_y:.2f})')

        # 朝向箭头
        arrow_len = 2.0
        ax.arrow(robot_x, robot_y,
                 arrow_len * math.cos(robot_yaw),
                 arrow_len * math.sin(robot_yaw),
                 head_width=0.4, head_length=0.25,
                 fc='red', ec='darkred', linewidth=2.5, zorder=15)

        # 机器人标签
        yaw_deg = math.degrees(robot_yaw)
        if yaw_deg < 0: yaw_deg += 360
        ax.annotate(f'({robot_x:.1f}, {robot_y:.1f})\nyaw={yaw_deg:.0f}°\nscore={score:.1f}',
                    xy=(robot_x, robot_y), xytext=(robot_x + 2.5, robot_y + 2),
                    fontsize=10, color='darkred', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

        # ── 如果 TF 可用，用蓝色标记 TF 位置作为对比 ──
        if tf_info is not None:
            tf_x, tf_y, tf_yaw = tf_info
            ax.plot(tf_x, tf_y, 'b+', markersize=18, markeredgewidth=3,
                    zorder=14, label=f'TF (drifted) ({tf_x:.2f}, {tf_y:.2f})')

            # TF 朝向箭头（蓝色）
            ax.arrow(tf_x, tf_y,
                     arrow_len * math.cos(tf_yaw),
                     arrow_len * math.sin(tf_yaw),
                     head_width=0.35, head_length=0.22,
                     fc='blue', ec='darkblue', linewidth=2.0, zorder=14, alpha=0.6)

            dx = robot_x - tf_x
            dy = robot_y - tf_y
            dist_err = math.sqrt(dx*dx + dy*dy)
            self.get_logger().info(
                f'[对比] 配准 vs TF: 偏差 Δ={dist_err:.2f}m '
                f'(TF→配准: dx={dx:.2f}m, dy={dy:.2f}m)')

        # ── 6. 视图范围：以机器人为中心 ──
        view_margin = 3.0
        ax_xlim = max(abs(x_max - robot_x), abs(x_min - robot_x)) + view_margin
        ax_ylim = max(abs(y_max - robot_y), abs(y_min - robot_y)) + view_margin
        view_r = max(ax_xlim, ax_ylim)
        ax.set_xlim(robot_x - view_r, robot_x + view_r)
        ax.set_ylim(robot_y - view_r, robot_y + view_r)

        method_label = "Likelihood Match" if tf_info is None else "Likelihood Match (vs TF)"
        ax.set_title(
            f'Scan on Map — Robot Position via {method_label}\n'
            f'Robot: ({robot_x:.2f}m, {robot_y:.2f}m, yaw={yaw_deg:.1f}°) | '
            f'Scan area: {rect_w:.1f}m×{rect_h:.1f}m | '
            f'{len(self.saved_scans)} frames | score={score:.1f}',
            fontsize=13)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.15)

        filepath = os.path.join(self.output_dir, 'scan_on_map.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f'[图片] Scan-on-Map 标注图: {filepath}')
        return match_result

    def _export_debug_data(self):
        """导出调试数据到 npz 文件，供离线算法测试

        导出内容:
          - scan_points_odom: (N,2) 合并后的扫描点云（odom系）
          - map_data: (H,W) 占据栅格地图
          - map_info: dict {resolution, width, height, origin_x, origin_y}
          - tf_odom_to_map: (x, y, yaw) odom→map 变换（真实值，用于验证）
          - saved_scans_raw: list of dict 原始帧数据
        """
        export_path = os.path.join(self.output_dir, 'debug_match_data.npz')

        # 1. 扫描点云 (odom系)
        scan_pts = self._merge_all_scan_points()

        # 2. 地图信息
        mi = self.map_info
        map_info_dict = {
            'resolution': mi.resolution,
            'width': mi.width,
            'height': mi.height,
            'origin_x': mi.origin.position.x,
            'origin_y': mi.origin.position.y,
        }

        # 3. odom→map TF（真实值）
        tf_info = self._get_odom_to_map_transform()
        tf_arr = np.array(tf_info) if tf_info else np.array([0, 0, 0])

        # 4. 保存每帧的TF和scan参数
        n_frames = len(self.saved_scans)
        frame_tfs = np.zeros((n_frames, 3))  # x, y, yaw
        frame_ranges_list = []
        for i, s in enumerate(self.saved_scans):
            frame_tfs[i] = [s['tf_x'], s['tf_y'], s['tf_yaw']]
            frame_ranges_list.append(s['ranges'])

        # 保存为 npz（支持 numpy 数组和 python 对象混存）
        np.savez_compressed(
            export_path,
            scan_points_odom=scan_pts,
            map_data=self.map_data,
            map_resolution=np.float64(map_info_dict['resolution']),
            map_width=np.int32(map_info_dict['width']),
            map_height=np.int32(map_info_dict['height']),
            map_origin_x=np.float64(map_info_dict['origin_x']),
            map_origin_y=np.float64(map_info_dict['origin_y']),
            tf_odom_to_map=tf_arr,
            frame_tfs=frame_tfs,
            # 用 pickle 保存非数组数据
            frame_angle_min=self.saved_scans[0]['angle_min'],
            frame_angle_increment=self.saved_scans[0]['angle_increment'],
            **{f'frame_ranges_{i}': np.array(r, dtype=np.float32)
               for i, r in enumerate(frame_ranges_list)},
        )

        file_size = os.path.getsize(export_path)
        self.get_logger().info(
            f'[导出] 调试数据: {export_path} '
            f'({file_size/1024:.0f}KB, {len(scan_pts)}扫描点, '
            f'{self.map_info.width}x{self.map_info.height}地图, '
            f'{n_frames}帧)')

    def _finish(self):
        self.cmd_vel_pub.publish(Twist())
        self.phase = 'done'
        self.get_logger().info(f'[完成] 共采集 {len(self.saved_scans)} 帧')

        # ── 导出调试数据（供离线算法测试）──
        try:
            self._export_debug_data()
        except Exception as e:
            self.get_logger().warn(f'[导出] 调试数据保存失败: {e}')

        # ── 只生成最终标注图：Scan-on-Map（TF变换 + 红框 + 机器人位置）──
        if len(self.saved_scans) >= 3 and self.map_data is not None:
            try:
                tf_info = self._plot_scan_on_map()
                if tf_info is not None:
                    bx, by, byaw = tf_info
                    self.get_logger().info(
                        f'[TF定位] 机器人在地图中的位姿: '
                        f'x={bx:.2f}m, y={by:.2f}m, yaw={math.degrees(byaw):.1f}°')
            except Exception as e:
                self.get_logger().warn(f'[可视化] scan_on_map 生成失败: {e}')
        else:
            if len(self.saved_scans) < 3:
                self.get_logger().warn(f'[可视化] 跳过: 帧数不足 ({len(self.saved_scans)} < 3)')
            if self.map_data is None:
                self.get_logger().warn('[可视化] 跳过: 无地图数据')

        self.get_logger().info(f'[完成] 结果图片已保存到 {self.output_dir}/scan_on_map.png')
        self.get_logger().info('[完成] 节点即将退出')
        # 延迟退出确保日志输出
        self.create_timer(0.5, lambda: rclpy.shutdown())

    # ────── 图片生成 ──────
    def _reconstruct_scan(self, saved):
        """从保存的数据重建 FakeScan 对象"""
        class FakeScan:
            pass
        s = FakeScan()
        s.ranges = saved['ranges']
        s.angle_min = saved['angle_min']
        s.angle_increment = saved['angle_increment']
        s.range_min = saved['range_min']
        s.range_max = saved['range_max']
        return s

    def _scan_to_world_xy(self, saved):
        """将一帧 scan 的点变换到目标坐标系下（使用 TF 变换矩阵）

        通过 TF 查询得到的 scan_frame → target_frame 刚体变换，
        自动处理任意坐标系关系（本地/全局），无需手动判断。
        """
        scan = self._reconstruct_scan(saved)
        points = self._scan_to_xy(scan)  # scan frame 下的点（原点=frame原点）
        if len(points) == 0:
            return np.empty((0, 2))

        # 使用保存的 TF 变换做刚体变换: R * p + t
        yaw = saved['tf_yaw']
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rot = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        points_world = (rot @ points.T).T
        points_world[:, 0] += saved['tf_x']
        points_world[:, 1] += saved['tf_y']
        return points_world

    def _generate_all_images(self):
        for i, saved in enumerate(self.saved_scans):
            scan = self._reconstruct_scan(saved)
            # 单帧图保持不变（只画当前帧）
            self._plot_single_scan(scan, i)
            if self.map_data is not None and self.map_info is not None:
                # scan_with_map: 右侧叠加 0~i 所有帧的历史数据
                self._plot_scan_with_map(i)

        # 生成多帧叠加总览
        if len(self.saved_scans) > 1:
            self._plot_overlay()

    def _plot_single_scan(self, scan, index):
        points = self._scan_to_xy(scan)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')

        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], s=1, c='blue', alpha=0.6, edgecolors='none')
        ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=3)

        # 方向线
        for deg in range(-180, 181, 30):
            rad = math.radians(deg)
            ax.plot([0, 8 * math.cos(rad)], [0, 8 * math.sin(rad)], 'gray', linewidth=0.3, alpha=0.3)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        ranges = np.array(scan.ranges)
        valid = (ranges > scan.range_min) & (ranges < scan.range_max)
        ax.set_title(f'Scan #{index:03d} | Valid: {np.sum(valid)}/{len(ranges)} beams')
        ax.grid(True, alpha=0.2)

        filepath = os.path.join(self.output_dir, f'scan_{index:03d}.png')
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f'[图片] {filepath}')

    def _plot_scan_with_map(self, index):
        """生成 scan vs map 对比图，右侧子图在 odom 世界坐标系下叠加 0~index 所有帧"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        ax1.set_aspect('equal')

        # ── 左侧：地图（不变）──
        map_display = np.zeros((self.map_info.height, self.map_info.width, 3), dtype=np.float32)
        map_display[self.map_data == 0] = [1.0, 1.0, 1.0]
        map_display[self.map_data == 100] = [0.0, 0.0, 0.0]
        map_display[self.map_data == -1] = [0.6, 0.6, 0.6]

        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        extent = [origin_x, origin_x + self.map_info.width * res,
                  origin_y, origin_y + self.map_info.height * res]
        ax1.imshow(map_display, origin='lower', extent=extent)
        ax1.set_title(f'Grid Map ({self.map_info.width}x{self.map_info.height})')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')

        # ── 右侧：累积叠加 0~index 所有帧（odom 世界坐标）──
        ax2.set_aspect('equal')

        n_frames = index + 1
        colors = plt.cm.cool(np.linspace(0.15, 1.0, max(n_frames, 1)))

        # 收集所有帧的世界坐标点，用于自动计算范围
        all_world_pts = []

        for i in range(n_frames):
            saved = self.saved_scans[i]
            pts = self._scan_to_world_xy(saved)
            if len(pts) > 0:
                all_world_pts.append(pts)
                alpha = 0.3 + 0.4 * (i / max(n_frames - 1, 1))
                sz = 0.5 + 1.0 * (i / max(n_frames - 1, 1))
                ax2.scatter(pts[:, 0], pts[:, 1], s=sz,
                            color=colors[i], alpha=alpha, edgecolors='none')

            # 绘制每帧的机器人朝向箭头
            arrow_len = 0.3
            ax2.arrow(saved['odom_x'], saved['odom_y'],
                      arrow_len * math.cos(saved['odom_yaw']),
                      arrow_len * math.sin(saved['odom_yaw']),
                      color=colors[i], width=0.02, alpha=0.4,
                      head_width=0.06, head_length=0.06)

        # 当前帧用红色醒目标记
        current_pts = self._scan_to_world_xy(self.saved_scans[index])
        if len(current_pts) > 0:
            ax2.scatter(current_pts[:, 0], current_pts[:, 1], s=1.5,
                        c='red', alpha=0.8, edgecolors='none',
                        label=f'Frame #{index:03d} (current)')

        # 机器人位置（第一帧的位置作为参考原点标记）
        first_pose = self.saved_scans[0]
        ax2.plot(first_pose['odom_x'], first_pose['odom_y'], 'g+',
                 markersize=15, markeredgewidth=3)

        # 自动调整坐标范围以包含所有点
        if all_world_pts:
            combined = np.vstack(all_world_pts)
            cx, cy = combined[:, 0].mean(), combined[:, 1].mean()
            half_w = max(10, (combined[:, 0].max() - combined[:, 0].min()) / 2 * 1.3)
            half_h = max(10, (combined[:, 1].max() - combined[:, 1].min()) / 2 * 1.3)
            ax2.set_xlim(cx - half_w, cx + half_w)
            ax2.set_ylim(cy - half_h, cy + half_h)

        coord_label = f'{self.target_frame} (TF transform)'
        ax2.set_title(f'Scan in {coord_label} Frame (frames 0-{index:03d}, total={n_frames})')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')

        fig.suptitle(f'Scan vs Map #{index:03d} — {n_frames} frames in {self.target_frame} frame (TF)', fontsize=14)
        filepath = os.path.join(self.output_dir, f'scan_with_map_{index:03d}.png')
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close()

    def _plot_overlay(self):
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_aspect('equal')

        n = len(self.saved_scans)
        colors = plt.cm.jet(np.linspace(0, 1, n))

        for i, saved in enumerate(self.saved_scans):
            pts = self._scan_to_world_xy(saved)
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=0.5, color=colors[i],
                           alpha=0.4, edgecolors='none')

            # 绘制每帧的机器人朝向箭头（所有帧都有 odom 位姿）
            arrow_len = 0.3
            ax.arrow(saved['odom_x'], saved['odom_y'],
                     arrow_len * math.cos(saved['odom_yaw']),
                     arrow_len * math.sin(saved['odom_yaw']),
                     color=colors[i], width=0.02, alpha=0.5,
                     head_width=0.08, head_length=0.08)

        # 起始位置
        first_pose = self.saved_scans[0]
        ax.plot(first_pose['odom_x'], first_pose['odom_y'], 'r+', markersize=20,
                markeredgewidth=4, label='Start')

        # 自动范围
        all_pts = [self._scan_to_world_xy(s) for s in self.saved_scans]
        all_pts = [p for p in all_pts if len(p) > 0]
        if all_pts:
            combined = np.vstack(all_pts)
            cx, cy = combined[:, 0].mean(), combined[:, 1].mean()
            hw = max(12, (combined[:, 0].max() - combined[:, 0].min()) / 2 * 1.3)
            hh = max(12, (combined[:, 1].max() - combined[:, 1].min()) / 2 * 1.3)
            ax.set_xlim(cx - hw, cx + hw)
            ax.set_ylim(cy - hh, cy + hh)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        coord_label = f'{self.target_frame} (TF transform)'
        ax.set_title(f'All {n} Scans Overlay in {coord_label}')
        ax.legend()
        ax.grid(True, alpha=0.2)

        filepath = os.path.join(self.output_dir, 'scan_overlay.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f'[图片] 叠加图: {filepath}')

        # 也生成地图叠加版
        if self.map_data is not None and self.map_info is not None:
            self._plot_overlay_with_map()

    def _plot_overlay_with_map(self):
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_aspect('equal')

        # 地图背景
        map_display = np.zeros((self.map_info.height, self.map_info.width, 3), dtype=np.float32)
        map_display[self.map_data == 0] = [1.0, 1.0, 1.0]
        map_display[self.map_data == 100] = [0.0, 0.0, 0.0]
        map_display[self.map_data == -1] = [0.6, 0.6, 0.6]
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        res = self.map_info.resolution
        extent = [origin_x, origin_x + self.map_info.width * res,
                  origin_y, origin_y + self.map_info.height * res]
        ax.imshow(map_display, origin='lower', extent=extent, alpha=0.5)

        n = len(self.saved_scans)
        colors = plt.cm.jet(np.linspace(0, 1, n))
        for i, saved in enumerate(self.saved_scans):
            pts = self._scan_to_world_xy(saved)
            if len(pts) > 0:
                ax.scatter(pts[:, 0], pts[:, 1], s=0.5, color=colors[i],
                           alpha=0.4, edgecolors='none')
            # 朝向箭头
            arrow_len = 0.3
            ax.arrow(saved['odom_x'], saved['odom_y'],
                     arrow_len * math.cos(saved['odom_yaw']),
                     arrow_len * math.sin(saved['odom_yaw']),
                     color=colors[i], width=0.02, alpha=0.5,
                     head_width=0.08, head_length=0.08)

        mode = f'{self.target_frame} (TF)'
        ax.set_title(f'All Scans Overlay on Grid Map ({n} frames, {mode})')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        filepath = os.path.join(self.output_dir, 'scan_overlay_with_map.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        self.get_logger().info(f'[图片] 地图叠加图: {filepath}')


def main(args=None):
    rclpy.init(args=args)
    node = ScanCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('用户中断，生成已采集的图片...')
        node._finish()
    except Exception as e:
        node.get_logger().error(f'异常: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
