#!/usr/bin/env python3
"""
RTK数据模拟发布节点 - 模拟真实室外GPS/RTK信号（圆形轨迹，动态噪声）

模拟真实情况：
- 卫星数量动态变化（可见星 20-35，解算星 18-25），带时间相关性
- 位置精度随 pos_type 变化：DGPS(亚米级 std~0.5-3m)、FLOAT(0.05-0.5m)、FIX(0.01-0.05m)
- 精度指标带有随机抖动，模拟真实信号波动
- 航向角沿圆切线，航向精度与 pos_type 联动
- 高度带轻微噪声
- sol_status 和 pos_type 始终保持有效（通过预处理器过滤）
- 偶尔精度下降但仍保持在有效门槛内

用法:
  ros2 run gps_fusion rtk_simulator.py --ros-args -p pos_type:=16 -p radius:=80.0 -p speed:=0.6 -p rate:=1.0
  ros2 run gps_fusion rtk_simulator.py --ros-args -p pos_type:=50 -p radius:=100.0 -p speed:=1.0 -p rate:=5.0
  话题默认: /test/rtk_pvh（避免与真实GPS硬件 /rtk_pvh 冲突）
"""

import math
import random
import rclpy
from rclpy.node import Node

try:
    from robots_dog_msgs.msg import UniRtkPvh
    _HAS_RTK = True
except ImportError:
    UniRtkPvh = None
    _HAS_RTK = False


# ====== 真实室外 RTK 样本（作为圆心参考） ======
CENTER_LAT = 24.612603983011
CENTER_LON = 118.03419204406185
CENTER_ALT = 56.34729

# 在该纬度下：1° lat ≈ 111320 m, 1° lon ≈ 111320 × cos(lat) m
_M_PER_DEG_LAT = 111320.0
_M_PER_DEG_LON = 111320.0 * math.cos(math.radians(CENTER_LAT))


class RtkSimulator(Node):
    """模拟 RTK 数据发布节点 — 圆形轨迹 + 真实感噪声"""

    def __init__(self):
        super().__init__('rtk_simulator')

        if not _HAS_RTK:
            self.get_logger().fatal(
                'robots_dog_msgs 未安装，无法构造 UniRtkPvh 消息。请先编译 robots_dog_msgs 包。'
            )
            raise RuntimeError('robots_dog_msgs not available')

        # ---- 可调参数 ----
        self.declare_parameter('rate', 1.0)               # 发布频率 (Hz)
        self.declare_parameter('pos_type', 16)              # 定位类型: 16=单点, 17=DGPS, 34=FLOAT, 50=FIX
        self.declare_parameter('topic', '/test/rtk_pvh')
        self.declare_parameter('radius', 80.0)              # 圆形半径 (m)
        self.declare_parameter('speed', 0.6)                # 线速度 (m/s)
        self.declare_parameter('center_lat', CENTER_LAT)
        self.declare_parameter('center_lon', CENTER_LON)
        self.declare_parameter('center_alt', CENTER_ALT)

        self._rate = self.get_parameter('rate').value
        self._pos_type = self.get_parameter('pos_type').value
        self._topic = self.get_parameter('topic').value
        self._radius = self.get_parameter('radius').value
        self._speed = self.get_parameter('speed').value
        self._center_lat = self.get_parameter('center_lat').value
        self._center_lon = self.get_parameter('center_lon').value
        self._center_alt = self.get_parameter('center_alt').value

        self._pub = self.create_publisher(UniRtkPvh, self._topic, 10)
        self._timer = self.create_timer(1.0 / self._rate, self._publish)

        # ---- 运动状态 ----
        self._angle = 0.0
        self._angle_step = self._speed / self._radius / self._rate
        self._seq = 0

        # ---- 动态噪声状态 ----
        self._seed = random.randint(0, 10000)
        self._svs_base = 0    # 平滑可见星基准值
        self._init_noise_state()

        # 日志
        pos_label = {16: '单点', 17: 'DGPS', 34: 'RTK_FLOAT', 50: 'RTK_FIX'}.get(self._pos_type, '?')
        circum = 2.0 * math.pi * self._radius
        lap_time = circum / self._speed if self._speed > 0 else float('inf')

        self.get_logger().info(
            f'RTK模拟节点已启动: topic={self._topic}, rate={self._rate}Hz, pos_type={self._pos_type}({pos_label})'
        )
        self.get_logger().info(
            f'圆形轨迹: 圆心=({self._center_lat:.6f}, {self._center_lon:.6f}), '
            f'半径={self._radius:.0f}m, 速度={self._speed:.1f}m/s, '
            f'周长={circum:.0f}m, 圈时≈{lap_time:.0f}s'
        )
        self.get_logger().info(
            f'噪声参数: 1°lat={_M_PER_DEG_LAT:.0f}m, 1°lon={_M_PER_DEG_LON:.0f}m'
        )

    # ------------------------------------------------------------------
    # 噪声模型
    # ------------------------------------------------------------------

    def _init_noise_state(self):
        """初始化动态噪声基准值，后续平滑变化"""
        if self._pos_type == 50:
            # RTK_FIX: 高精度
            self._svs_base = random.uniform(22, 32)
            self._std_lat_range = (0.008, 0.03)     # lat_std 范围
            self._std_lon_range = (0.005, 0.02)
            self._std_hgt_range = (0.02, 0.08)
            self._heading_std_range = (0.5, 2.0)
        elif self._pos_type == 34:
            # RTK_FLOAT: 中等精度
            self._svs_base = random.uniform(18, 28)
            self._std_lat_range = (0.05, 0.3)
            self._std_lon_range = (0.03, 0.2)
            self._std_hgt_range = (0.3, 1.0)
            self._heading_std_range = (2.0, 8.0)
        else:
            # DGPS (16): 亚米级
            self._svs_base = random.uniform(16, 27)
            self._std_lat_range = (0.3, 3.0)       # h_accuracy 最大 ~sqrt(9+9)=4.2m, 远低于30m门槛
            self._std_lon_range = (0.2, 2.5)
            self._std_hgt_range = (0.5, 5.0)
            self._heading_std_range = (3.0, 15.0)

    def _smooth_random(self, base, scale, max_step):
        """生成平滑变化的随机值（带时间相关性），避免跳变"""
        # 使用正弦波+噪声混合，模拟缓慢变化的信号质量
        t = self._seq / max(1.0, self._rate)
        drift = math.sin(t * 0.05 + self._seed * 0.01) * scale * 0.5
        noise = (random.random() - 0.5) * scale * 0.5
        return base + drift + noise, max_step

    def _get_satellite_counts(self):
        """返回 (可见星数, 解算星数)，模拟真实波动

        真实场景：
        - 可见星数通常在 18-35 之间波动（取决于天空遮挡、卫星分布）
        - 解算星数 = 可见星 - 低仰角星(2-5颗) - 个别信号弱星(0-3颗)
        - 用缓慢漂移+随机噪声模拟
        """
        # 可见星数缓慢漂移 (摆动幅度 ±4)
        t = self._seq / max(1.0, self._rate)
        drift = math.sin(t * 0.02 + self._seed * 0.1) * 3.0
        jitter = random.gauss(0, 1.5)
        visible = int(round(self._svs_base + drift + jitter))
        visible = max(12, min(38, visible))

        # 解算星数：可见星减去一些不可用星 (2-7颗，取决于质量)
        if self._pos_type == 50:
            loss = random.randint(1, 3)   # RTK_FIX 模式下丢掉少
        elif self._pos_type == 34:
            loss = random.randint(2, 5)
        else:
            loss = random.randint(3, 7)

        solved = max(4, visible - loss)  # 至少4颗才能定位
        solved = min(solved, visible)

        return visible, solved

    def _get_position_std(self):
        """返回 (lat_std, lon_std, hgt_std)，模拟定位精度波动

        精度受多种因素影响：
        - 卫星几何分布 (DOP) — 占主导
        - 多路径效应 — 随机抖动
        - 大气延迟 — 缓慢漂移
        """
        t = self._seq / max(1.0, self._rate)

        # 基础精度（DOP缓慢漂移 + 随机噪声）
        lat_base = random.uniform(*self._std_lat_range)
        lon_base = random.uniform(*self._std_lon_range)
        hgt_base = random.uniform(*self._std_hgt_range)

        # 模拟 DOP 的周期性变化 (周期约5-10分钟)
        dop_factor = 0.7 + 0.3 * (0.5 + 0.5 * math.sin(t * 0.003 + self._seed * 0.05))

        lat_std = lat_base * dop_factor * (0.8 + random.random() * 0.4)
        lon_std = lon_base * dop_factor * (0.8 + random.random() * 0.4)
        hgt_std = hgt_base * dop_factor * (0.8 + random.random() * 0.4)

        # 偶尔出现精度尖峰（模拟瞬时干扰），概率 ~3-5%，但仍保持在门槛内
        if random.random() < 0.03:
            spike = 1.5 + random.random() * 1.5
            lat_std *= spike
            lon_std *= spike

        # 硬上限：保证 h_accuracy 低于 GPS 精度门槛
        # h_accuracy = sqrt(lat_std² + lon_std²)
        if self._pos_type == 16:
            max_h = 25.0  # 留余量，门槛30m
            h = math.sqrt(lat_std**2 + lon_std**2)
            if h > max_h:
                scale = max_h / h
                lat_std *= scale
                lon_std *= scale

        return lat_std, lon_std, hgt_std

    def _get_heading_quality(self):
        """返回 (heading_std, pitch_std, heading_type)

        航向精度：
        - 双天线基线越长越准（样本基线 23.3m → 正常~2°)
        - pos_type=50 时heading_type=50 (整数解航向)
        - pos_type=34 时heading_type=34 (浮点解)
        - pos_type=16 时heading_type=16 (DGPS级航向) 或偶尔 0(未收敛)
        """
        t = self._seq / max(1.0, self._rate)

        base_hstd = random.uniform(*self._heading_std_range)
        # 缓慢变化
        heading_std = base_hstd * (0.7 + 0.6 * random.random())

        # pitch精度通常比heading差一些（垂直基线短）
        pitch_std = heading_std * (1.2 + random.random() * 0.6)

        heading_type = self._pos_type  # 默认与pos_type一致

        # DGPS 模式下偶尔航向未收敛（概率~10%），但不影响位置
        if self._pos_type == 16 and random.random() < 0.10:
            heading_type = 0
            heading_std = random.uniform(90.0, 180.0)
            pitch_std = random.uniform(45.0, 90.0)

        return heading_std, pitch_std, heading_type

    # ------------------------------------------------------------------
    # 消息构造
    # ------------------------------------------------------------------

    def _build_heading(self, now, heading_deg, heading_std, pitch_std, heading_type, svs_num, soln_svs):
        """构造 UniHeading 子消息（模拟真实双天线航向数据）"""
        from robots_dog_msgs.msg import UniHeading

        h = UniHeading()
        h.header.stamp = now.to_msg()
        h.header.frame_id = 'WGS84'
        h.utc_time_s = now.nanoseconds * 1e-9

        h.sol_status = 0            # 0=已解出（有效）
        h.heading_type = heading_type
        h.base_line = 23.255 if self._pos_type >= 34 else random.uniform(10.0, 23.0)
        h.heading_deg = heading_deg
        h.pitch_deg = random.uniform(-1.0, 1.0)  # 轻微俯仰
        h.heading_std = heading_std
        h.pitch_std = pitch_std
        h.svs_num = svs_num
        h.soln_svs_num = soln_svs
        return h

    def _build_bestnav(self, now, lat, lon, lat_std, lon_std, hgt_std,
                       svs_num, soln_svs, track_deg):
        """构造 UniBestNav 子消息（模拟真实位置+速度解算数据）"""
        from robots_dog_msgs.msg import UniBestNav

        b = UniBestNav()
        b.header.stamp = now.to_msg()
        b.header.frame_id = 'WGS84'
        b.utc_time_s = now.nanoseconds * 1e-9

        b.p_sol_status = 0          # 0=已解出（有效）
        b.pos_type = self._pos_type
        b.latitude_deg = lat
        b.longitude_deg = lon
        b.altitude_m = self._center_alt + random.uniform(-0.5, 0.5)  # 高度微飘
        b.undulation = 9.3626
        b.lat_std = lat_std
        b.lon_std = lon_std
        b.hgt_std = hgt_std
        b.diff_age_s = 0.0 if self._pos_type >= 34 else random.uniform(0.5, 2.0)
        b.sol_age_s = 0.0
        b.svs_num = svs_num
        b.soln_svs_num = soln_svs

        b.v_sol_status = 0          # 速度解状态：已解出
        b.vel_type = 8 if self._pos_type >= 34 else 0
        b.hor_spd = self._speed + random.gauss(0, 0.02)   # 水平速度（带微噪）
        b.trk_gnd = track_deg                               # 地面航迹角
        b.ver_spd = random.gauss(0, 0.01)                   # 垂向速度~0
        b.ver_spd_std = hgt_std * 0.3
        b.hor_spd_std = 0.02 if self._pos_type >= 34 else 0.2
        return b

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def _publish(self):
        """定时发布一帧 UniRtkPvh（带真实感噪声）"""
        now = self.get_clock().now()

        # ---- 计算位置 ----
        dx = self._radius * math.cos(self._angle)   # 正东
        dy = self._radius * math.sin(self._angle)   # 正北
        lat = self._center_lat + dy / _M_PER_DEG_LAT
        lon = self._center_lon + dx / _M_PER_DEG_LON

        # ---- 航向沿圆切线 ----
        heading_deg = (math.degrees(self._angle) + 90.0) % 360.0
        track_deg = heading_deg  # 速度方向与航向一致（沿切线）

        # ---- 动态噪声 ----
        vis_svs, sol_svs = self._get_satellite_counts()
        lat_std, lon_std, hgt_std = self._get_position_std()
        hd_std, pitch_std, hd_type = self._get_heading_quality()

        # 计算水平精度因子
        h_accuracy = math.sqrt(lat_std**2 + lon_std**2)

        # ---- 构造消息 ----
        msg = UniRtkPvh()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'WGS84'
        msg.heading = self._build_heading(
            now, heading_deg, hd_std, pitch_std, hd_type, vis_svs, sol_svs)
        msg.bestnav = self._build_bestnav(
            now, lat, lon, lat_std, lon_std, hgt_std, vis_svs, sol_svs, track_deg)

        self._pub.publish(msg)
        self._seq += 1
        self._angle = (self._angle + self._angle_step) % (2.0 * math.pi)

        # ---- 日志（每10帧或首次）----
        pos_label = {16: '单点', 17: 'DGPS', 34: 'FLOAT', 50: 'FIX'}.get(self._pos_type, '?')
        if self._seq == 1 or self._seq % 10 == 0:
            self.get_logger().info(
                f'[{self._seq}] {pos_label} | '
                f'lat={lat:.7f} lon={lon:.7f} alt={self._center_alt:.1f}m | '
                f'h={heading_deg:.1f}° | '
                f'svs:{vis_svs}/{sol_svs} | '
                f'std: lat={lat_std:.3f}m lon={lon_std:.3f}m hgt={hgt_std:.3f}m '
                f'h_acc={h_accuracy:.3f}m | '
                f'hd_std={hd_std:.2f}° hd_type={hd_type}'
            )
        else:
            self.get_logger().debug(
                f'[{self._seq}] lat={lat:.7f} lon={lon:.7f} h={heading_deg:.1f}° svs:{vis_svs}/{sol_svs}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RtkSimulator()
        rclpy.spin(node)
    except RuntimeError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
