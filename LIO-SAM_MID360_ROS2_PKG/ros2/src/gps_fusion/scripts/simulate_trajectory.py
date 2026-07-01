#!/usr/bin/env python3
"""
GPS/RTK 轨迹仿真节点 — 围绕罗普特园区生成模拟 GPS/RTK 撒点数据。

模拟一条矩形巡逻路线，以约 1 m/s 的速度移动，输出三套数据：
1. sensor_msgs/NavSatFix 到 /fix 与 /fix_filtered，供 gps_preprocessor 消费；
2. robots_dog_msgs/UniRtkPvh 到 /rtk_pvh，与真实 RTK 接收端格式一致；
3. 经纬度 Path 到 /trajectory/lio_latlon、/trajectory/fused_latlon，
   以及 NavSatFix 到 /gps/current_latlon，供前端 WebSocket 可视化。

用法:
  ros2 run gps_fusion simulate_trajectory.py
  ros2 run gps_fusion simulate_trajectory.py --ros-args -p rate:=5.0
  ros2 run gps_fusion simulate_trajectory.py --ros-args -p speed:=2.0
  ros2 run gps_fusion simulate_trajectory.py --ros-args -p noise_std:=0.02

坐标系:
  - 罗普特园区中心: 24.6080°N, 118.0450°E（厦门市集美区凤岐路1888号）
  - 1°lat ≈ 111320m, 1°lon ≈ 111320×cos(24.608°) ≈ 101223m
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import math
import random
import time


# 尝试导入 RTK 自定义消息，机器人上已编译时可用；开发机未安装则自动跳过。
# msg 定义位于 extend/robots_dog_msgs/robots_dog_msgs/msg/
try:
    from robots_dog_msgs.msg import UniRtkPvh, UniHeading, UniBestNav
    _HAS_RTK_MSGS = True
except ImportError:
    UniRtkPvh = UniHeading = UniBestNav = None
    _HAS_RTK_MSGS = False


# ──────────────────── 罗普特园区参数 ────────────────────
CENTER_LAT = 24.6080   # 纬度
CENTER_LON = 118.0450  # 经度

# 1 米对应的经纬度偏移量
METERS_PER_DEG_LAT = 111320.0
METERS_PER_DEG_LON = 111320.0 * math.cos(math.radians(CENTER_LAT))  # ≈101223

# 巡逻路径：矩形环路 [ (lat°, lon°) ]，回到起点形成闭环
# 东 60m → 北 40m → 西 60m → 南 40m → 循环，一圈约 200m ≈ 200 秒
LOOP_WAYPOINTS = [
    (CENTER_LAT,                     CENTER_LON),                     # 起点
    (CENTER_LAT,                     CENTER_LON + 60 / METERS_PER_DEG_LON),   # 东 60m
    (CENTER_LAT + 40 / METERS_PER_DEG_LAT, CENTER_LON + 60 / METERS_PER_DEG_LON),  # 北 40m
    (CENTER_LAT + 40 / METERS_PER_DEG_LAT, CENTER_LON),              # 西 60m
    (CENTER_LAT,                     CENTER_LON),                     # 回起点
]


class GPSPatrolSimulator(Node):
    """以约 1 m/s 速度沿矩形巡逻路径移动，发布模拟 GPS 数据。"""

    def __init__(self):
        super().__init__('gps_patrol_simulator')

        # 参数
        self.declare_parameter('rate', 1.0)         # 发布频率 (Hz)
        self.declare_parameter('speed', 1.0)         # 移动速度 (m/s)
        self.declare_parameter('noise_std', 2.0)     # GPS 噪声标准差 (m)，0=无噪声(RTK模拟)
        self.declare_parameter('altitude', 15.0)     # 模拟海拔 (m)
        self.declare_parameter('rtk_topic', '/rtk_pvh')  # RTK 原始数据话题，空字符串=不发布

        rate = self.get_parameter('rate').value
        self.speed = self.get_parameter('speed').value
        self.noise_std = self.get_parameter('noise_std').value
        self.altitude = self.get_parameter('altitude').value
        self.rtk_topic = self.get_parameter('rtk_topic').value

        # 每步移动距离 (m)
        self.step_meters = self.speed / rate

        # 发布原始 GPS（/fix 供 gps_preprocessor 消费）
        self.gps_pub = self.create_publisher(NavSatFix, '/fix', 10)
        # 发布预处理后 GPS（跳过 gps_preprocessor 直接给 trajectory_server）
        self.filtered_pub = self.create_publisher(NavSatFix, '/fix_filtered', 10)

        # 发布 RTK 原始数据（robots_dog_msgs/UniRtkPvh）
        self.rtk_pub = None
        if self.rtk_topic and _HAS_RTK_MSGS:
            self.rtk_pub = self.create_publisher(UniRtkPvh, self.rtk_topic, 10)
        elif self.rtk_topic and not _HAS_RTK_MSGS:
            self.get_logger().warn(
                f'rtk_topic={self.rtk_topic} 已启用，但 robots_dog_msgs 未安装，'
                '不会发布 RTK 原始数据。'
            )

        # 发布经纬度轨迹 Path（与 trajectory_server 输出同名，
        # 供 trajectory_ws_server 直接转发到 WebSocket）
        self.lio_latlon_pub = self.create_publisher(Path, '/trajectory/lio_latlon', 10)
        self.fused_latlon_pub = self.create_publisher(Path, '/trajectory/fused_latlon', 10)
        # 另外发布原始 GPS 当前坐标（限频后转发给前端）
        self.gps_latlon_pub = self.create_publisher(NavSatFix, '/gps/current_latlon', 10)

        # 定时器
        period = 1.0 / rate
        self.timer = self.create_timer(period, self._step)

        # 巡逻状态
        self.waypoints = LOOP_WAYPOINTS
        self.wp_idx = 0              # 当前目标航点索引
        self.cur_lat = CENTER_LAT    # 当前纬度
        self.cur_lon = CENTER_LON    # 当前经度
        self.cur_yaw = 0.0           # 当前朝向（弧度，正东=0）
        self.elapsed = 0.0           # 仿真运行时间 (s)
        self.frame_seq = 0           # 帧序号

        # 轨迹记录
        self._trail = [(self.cur_lon, self.cur_lat)]

        # 计算总里程
        total = 0.0
        for i in range(len(self.waypoints) - 1):
            total += self._dist_deg(self.waypoints[i], self.waypoints[i + 1])
        total += self._dist_deg(self.waypoints[-1], self.waypoints[0])

        self.get_logger().info(
            'GPS 巡逻模拟器已启动'
            ' | 中心: (%.4f°N, %.4f°E)'
            ' | 速度: %.1f m/s'
            ' | 频率: %.1f Hz (每步 %.2fm)'
            ' | 噪声σ: %.1fm'
            ' | 巡逻路线: 矩形环路 %.0fm/圈'
            % (CENTER_LAT, CENTER_LON, self.speed, rate,
               self.step_meters, self.noise_std, total)
        )
        self.get_logger().info(
            '发布话题: /fix, /fix_filtered, /rtk_pvh (UniRtkPvh),'
            ' /trajectory/lio_latlon, /trajectory/fused_latlon, /gps/current_latlon'
        )

    # ────────────────── 几何工具 ──────────────────

    @staticmethod
    def _dist_deg(a, b):
        """两点间的球面距离 (m)。"""
        dlat = (b[0] - a[0]) * METERS_PER_DEG_LAT
        dlon = (b[1] - a[1]) * METERS_PER_DEG_LON
        return math.hypot(dlat, dlon)

    @staticmethod
    def _bearing(a, b):
        """从 a 到 b 的方位角（弧度，正东=0，逆时针）。"""
        dy = (b[0] - a[0]) * METERS_PER_DEG_LAT
        dx = (b[1] - a[1]) * METERS_PER_DEG_LON
        return math.atan2(dy, dx)

    def _make_latlon_path(self, trail, stamp):
        """将经纬度轨迹 [(lon, lat), ...] 转为 nav_msgs/Path。"""
        path = Path()
        path.header.frame_id = 'wgs84'
        path.header.stamp = stamp
        for (tlon, tlat) in trail:
            ps = PoseStamped()
            ps.pose.position.x = tlon
            ps.pose.position.y = tlat
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def _make_rtk_msg(self, lat, lon, altitude, yaw, now):
        """构造 robots_dog_msgs/UniRtkPvh 模拟数据。

        UniRtkPvh 只包含 header + heading(UniHeading) + bestnav(UniBestNav)，
        速度字段内置在 UniBestNav 中（hor_spd/trk_gnd/ver_spd 等）。
        """
        if not _HAS_RTK_MSGS:
            return None

        # 使用固定值模拟稳定 RTK 状态：50=整数解(固定解), 0=已解出
        pos_type = 50
        heading_type = 50
        sol_status = 0
        p_sol_status = 0

        heading = UniHeading()
        heading.header.frame_id = 'gps_link'
        heading.header.stamp = now
        heading.utc_time_s = time.time()
        heading.sol_status = sol_status
        heading.heading_type = heading_type
        heading.base_line = 2.0  # 基线长度，单位米
        heading.heading_deg = math.degrees(yaw) % 360.0
        heading.pitch_deg = 0.0
        heading.heading_std = 0.5 if self.noise_std > 0.1 else 0.05
        heading.pitch_std = 0.5
        heading.svs_num = 20
        heading.soln_svs_num = 18

        bestnav = UniBestNav()
        bestnav.header.frame_id = 'gps_link'
        bestnav.header.stamp = now
        bestnav.utc_time_s = time.time()
        bestnav.p_sol_status = p_sol_status
        bestnav.pos_type = pos_type
        bestnav.latitude_deg = lat
        bestnav.longitude_deg = lon
        bestnav.altitude_m = altitude
        bestnav.undulation = 0.0
        bestnav.lat_std = self.noise_std if self.noise_std > 0 else 0.02
        bestnav.lon_std = self.noise_std if self.noise_std > 0 else 0.02
        bestnav.hgt_std = 0.1
        bestnav.diff_age_s = 0.1
        bestnav.sol_age_s = 0.0
        bestnav.svs_num = 20
        bestnav.soln_svs_num = 18
        # 速度字段（UniBestNav 内置）
        bestnav.v_sol_status = 0
        bestnav.vel_type = 34
        bestnav.hor_spd = self.speed
        bestnav.trk_gnd = math.degrees(yaw) % 360.0
        bestnav.ver_spd = 0.0
        bestnav.ver_spd_std = 0.0
        bestnav.hor_spd_std = 0.05

        rtk = UniRtkPvh()
        rtk.header.frame_id = 'gps_link'
        rtk.header.stamp = now
        rtk.heading = heading
        rtk.bestnav = bestnav
        return rtk

    # ────────────────── 主循环 ──────────────────

    def _step(self):
        """每帧推进：沿路径移动 step_meters，发布 GPS 数据。"""
        self.elapsed += 1.0 / self.get_parameter('rate').value
        self.frame_seq += 1

        # 当前位置作为出发航点
        src = self.waypoints[self.wp_idx]
        # 目标航点（循环）
        dst_idx = (self.wp_idx + 1) % len(self.waypoints)
        dst = self.waypoints[dst_idx]

        # 距离与方向
        dist_to_dst = self._dist_deg(src, dst)
        bearing = self._bearing(src, dst)

        # 当前位置到 src 的已走距离
        dist_from_src = self._dist_deg(src, (self.cur_lat, self.cur_lon))

        # 推进 step_meters
        dist_from_src += self.step_meters
        self.cur_yaw = bearing

        # 检查是否需要切换到下一个航点
        while dist_from_src >= dist_to_dst and dist_to_dst > 0:
            dist_from_src -= dist_to_dst
            self.wp_idx = dst_idx
            src = self.waypoints[self.wp_idx]
            dst_idx = (self.wp_idx + 1) % len(self.waypoints)
            dst = self.waypoints[dst_idx]
            dist_to_dst = self._dist_deg(src, dst)
            bearing = self._bearing(src, dst)
            self.cur_yaw = bearing

            self.get_logger().info(
                '📍 到达航点 #%d (%.6f, %.6f) → 下一目标 #%d @ %.2fm [t=%.0fs]'
                % (self.wp_idx, src[0], src[1], dst_idx, dist_to_dst, self.elapsed),
                throttle_duration_sec=2.0,
            )

        # 插值当前位置
        if dist_to_dst > 0:
            ratio = dist_from_src / dist_to_dst
            lat = src[0] + (dst[0] - src[0]) * ratio
            lon = src[1] + (dst[1] - src[1]) * ratio
        else:
            lat, lon = src

        # GPS 噪声
        noise_lat_m = random.gauss(0, self.noise_std) if self.noise_std > 0 else 0.0
        noise_lon_m = random.gauss(0, self.noise_std) if self.noise_std > 0 else 0.0
        lat += noise_lat_m / METERS_PER_DEG_LAT
        lon += noise_lon_m / METERS_PER_DEG_LON

        self.cur_lat = lat
        self.cur_lon = lon

        # 记录轨迹
        self._trail.append((lon, lat))
        if len(self._trail) > 10000:
            self._trail.pop(0)

        # ── 发布 NavSatFix ──
        now = self.get_clock().now().to_msg()
        var = max(self.noise_std ** 2, 0.0004)  # 至少 0.02² (RTK精度)

        fix = NavSatFix()
        fix.header.frame_id = 'gps'
        fix.header.stamp = now
        fix.latitude = lat
        fix.longitude = lon
        fix.altitude = self.altitude + random.gauss(0, self.noise_std * 0.5)
        fix.status.status = 0 if self.noise_std > 1.0 else 4   # 0=无定位, 4=RTK_FIX
        fix.position_covariance_type = 1  # COVARIANCE_TYPE_KNOWN
        fix.position_covariance = [
            var, 0.0, 0.0,
            0.0, var, 0.0,
            0.0, 0.0, var * 4.0,
        ]

        self.gps_pub.publish(fix)

        # 同时发布到 /fix_filtered（跳过 gps_preprocessor 直接对接 trajectory_server）
        filtered = NavSatFix()
        filtered.header.frame_id = 'gps'
        filtered.header.stamp = now
        filtered.latitude = lat
        filtered.longitude = lon
        filtered.altitude = fix.altitude
        filtered.status = fix.status
        filtered.position_covariance = fix.position_covariance
        filtered.position_covariance_type = fix.position_covariance_type
        self.filtered_pub.publish(filtered)

        # ── 发布经纬度轨迹 Path ──
        # 用 GPS 轨迹同时作为 LIO 和 fused 可视化，让前端立即有数据
        lio_path = self._make_latlon_path(self._trail, now)
        fused_path = self._make_latlon_path(self._trail, now)
        self.lio_latlon_pub.publish(lio_path)
        self.fused_latlon_pub.publish(fused_path)

        # ── 发布当前 GPS 经纬度（前端当前位置标记） ──
        self.gps_latlon_pub.publish(filtered)

        # ── 发布 RTK 原始数据（robots_dog_msgs/UniRtkPvh） ──
        if self.rtk_pub is not None:
            rtk_msg = self._make_rtk_msg(lat, lon, fix.altitude, self.cur_yaw, now)
            if rtk_msg is not None:
                self.rtk_pub.publish(rtk_msg)

        # 每秒日志（throttle）
        self.get_logger().info(
            '📍 (%.6f, %.6f) 航向: %.0f° | 轨迹: %d pts | t=%.0fs'
            % (lat, lon, math.degrees(self.cur_yaw),
               len(self._trail), self.elapsed),
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GPSPatrolSimulator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            try:
                node.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            rclpy.shutdown()
        except (RuntimeError, KeyboardInterrupt):
            pass


if __name__ == '__main__':
    main()
