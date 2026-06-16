#!/usr/bin/env python3
"""
雷达扫描可视化脚本 — 旋转360° + 小范围移动 + 避障版

功能:
  1. 原地旋转360°，每隔约1秒保存一帧扫描图
  2. 在当前位置1米范围内向8个方向依次移动探索
  3. 雷达避障：前方检测到障碍物自动停机并跳过该方向
  4. 所有帧保存为 PNG，同时生成叠加图

用法:
  ros2 run auto_initial_pose_calibrator visualize_scan.py
  ros2 run auto_initial_pose_calibrator visualize_scan.py --ros-args -p ns:=rkbot

输出:
  - /tmp/scan_viz/scan_000.png ... scan_NNN.png   (每帧独立图)
  - /tmp/scan_viz/scan_overlay.png               (所有帧叠加)
  - /tmp/scan_viz/scan_with_map_000.png ...       (叠加地图对比，需地图可用)
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
    def _finish(self):
        self.cmd_vel_pub.publish(Twist())
        self.phase = 'done'
        self.get_logger().info(f'[完成] 共采集 {len(self.saved_scans)} 帧，开始生成图片...')
        self._generate_all_images()
        self.get_logger().info(f'[完成] 所有图片已保存到 {self.output_dir}/')
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
