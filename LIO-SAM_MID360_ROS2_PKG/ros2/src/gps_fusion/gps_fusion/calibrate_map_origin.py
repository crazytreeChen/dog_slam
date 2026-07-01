#!/usr/bin/env python3
"""
地图原点 GPS + 朝向 标定工具

通过 AMCL pose + RTK 同步数据自动计算地图原点的 GPS 坐标和朝向，
保存到 map_gps_origin.yaml 供 rtk_initial_pose.py 使用。

原理（带旋转）:
    建图时机器人在地图中 (0, 0, 0)，真实航向 = θ₀（heading_deg）
    标定时已知 AMCL 位姿 (ax, ay, ayaw) + RTK (utm_e, utm_n, heading_deg)

    θ₀ = heading_deg - ayaw               (地图朝向 = 真实航向 - 地图航向)
    E₀ = utm_e - ax*cos(θ₀) + ay*sin(θ₀)  (旋转后的地图原点)
    N₀ = utm_n - ax*sin(θ₀) - ay*cos(θ₀)

导航时 RTK → map:
    dx = utm_e - E₀,  dy = utm_n - N₀
    map_x =  dx*cos(θ₀) + dy*sin(θ₀)
    map_y = -dx*sin(θ₀) + dy*cos(θ₀)
    map_yaw = heading_deg - θ₀

用法:
    # 自动模式：收集 N 帧同步数据，取平均
    ros2 run gps_fusion calibrate_map_origin.py --ros-args \
      -p amcl_pose_topic:=/rkbot/amcl_pose

    # 手动模式：直接提供已知位置
    ros2 run gps_fusion calibrate_map_origin.py --ros-args \
      -p manual_x:=5.0 -p manual_y:=-3.0 -p manual_trigger:=True

    # 无 RTK 航向时的降级模式（heading_deg=0）
    ros2 run gps_fusion calibrate_map_origin.py --ros-args \
      -p use_rtk_heading:=false
"""

import os
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
from sensor_msgs.msg import NavSatFix
from pyproj import Transformer

# 尝试导入 RTK 消息类型
try:
    from robots_dog_msgs.msg import UniRtkPvh
    _HAS_UNI_RTK_PVH = True
except ImportError:
    UniRtkPvh = None
    _HAS_UNI_RTK_PVH = False


def _deg_to_utm(transformer, lon, lat):
    e, n = transformer.transform(lon, lat)
    return e, n


def _utm_to_deg(transformer, e, n):
    lon, lat = transformer.transform(e, n)
    return lon, lat


def _quat_to_yaw(q: Quaternion) -> float:
    """四元数 → yaw 角 (rad, [-π, π])"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class CalibrateMapOrigin(Node):
    """AMCL + RTK 同步标定地图原点（含朝向）"""

    def __init__(self):
        super().__init__('calibrate_map_origin')

        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('amcl_pose_topic', '/amcl_pose')
        self.declare_parameter('fix_topic', '/fix_filtered')
        self.declare_parameter('rtk_topic', '/rtk_pvh')
        self.declare_parameter('sample_count', 20)       # 采集帧数
        self.declare_parameter('max_accuracy', 5.0)      # 最大可接受精度 (m)
        self.declare_parameter('output_file', '')
        self.declare_parameter('manual_x', float('nan'))
        self.declare_parameter('manual_y', float('nan'))
        self.declare_parameter('manual_trigger', False)
        self.declare_parameter('use_rtk_heading', True)  # 是否使用 RTK 航向

        self._utm_zone = self.get_parameter('utm_zone').value
        self._sample_count = self.get_parameter('sample_count').value
        self._max_accuracy = self.get_parameter('max_accuracy').value
        self._use_rtk_heading = self.get_parameter('use_rtk_heading').value

        utm_epsg = 32600 + abs(self._utm_zone) if self._utm_zone > 0 else 32700 + abs(self._utm_zone)
        self._to_utm = Transformer.from_crs('epsg:4326', f'epsg:{utm_epsg}', always_xy=True)
        self._to_wgs = Transformer.from_crs(f'epsg:{utm_epsg}', 'epsg:4326', always_xy=True)

        # 输出文件
        output_file = self.get_parameter('output_file').value
        if not output_file:
            output_file = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'map_gps_origin.yaml')
        self._output_file = os.path.abspath(output_file)

        # ---- 手动模式 ----
        manual_x = self.get_parameter('manual_x').value
        manual_y = self.get_parameter('manual_y').value
        self._manual_trigger = self.get_parameter('manual_trigger').value
        self._manual_yaw = 0.0  # 手动模式默认 yaw=0

        if self._manual_trigger and not (math.isnan(manual_x) or math.isnan(manual_y)):
            self._manual_mode = True
            self._manual_pose = (manual_x, manual_y)
        else:
            self._manual_mode = False

        # ---- 采样状态 ----
        # 每帧: (amcl_x, amcl_y, amcl_yaw, utm_e, utm_n, heading_deg)
        self._samples = []
        self._latest_amcl = None      # (x, y, yaw) 地图坐标+朝向
        self._latest_fix = None       # NavSatFix（位置+精度）
        self._latest_rtk_heading = None  # 真实航向 (度)
        self._samples_collected = False
        self._calibration_done = False

        # ---- 订阅 AMCL ----
        self._amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter('amcl_pose_topic').value,
            self._amcl_callback, 10,
        )

        # ---- 订阅 GPS 位置（/fix_filtered） ----
        self._fix_sub = self.create_subscription(
            NavSatFix,
            self.get_parameter('fix_topic').value,
            self._fix_callback, 10,
        )

        # ---- 订阅 RTK 原始数据获取航向 ----
        self._rtk_sub = None
        rtk_topic = self.get_parameter('rtk_topic').value
        if self._use_rtk_heading and rtk_topic and _HAS_UNI_RTK_PVH:
            self._rtk_sub = self.create_subscription(
                UniRtkPvh, rtk_topic, self._rtk_callback, 10)
            self.get_logger().info(f'RTK 航向来源: {rtk_topic}')
        elif self._use_rtk_heading and not _HAS_UNI_RTK_PVH:
            self.get_logger().warn(
                'robots_dog_msgs 未安装，无法获取 RTK 航向，'
                '将使用 heading_deg=0 标定（地图与真北对齐）')
            self._use_rtk_heading = False

        self.get_logger().info('标定节点已启动，等待 AMCL + GPS 同步数据...')
        self.get_logger().info(
            f'RTK 航向标定: {"启用" if self._use_rtk_heading else "禁用（地图与真北对齐）"}'
        )

        # ---- 定时打印状态 ----
        self._status_timer = self.create_timer(3.0, self._status_callback)

    # ==================================================================
    #  回调
    # ==================================================================

    def _amcl_callback(self, msg: PoseWithCovarianceStamped):
        self._latest_amcl = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            _quat_to_yaw(msg.pose.pose.orientation),
        )

    def _fix_callback(self, msg: NavSatFix):
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return
        h_acc = self._compute_h_accuracy(msg)
        if h_acc > self._max_accuracy:
            self.get_logger().debug(
                f'GPS 精度不够: {h_acc:.3f}m > {self._max_accuracy}m')
            return
        self._latest_fix = msg
        self._try_sample()

    def _rtk_callback(self, msg):
        """从原始 RTK 消息获取航向"""
        try:
            heading = msg.heading
            # heading_type: 16=DGPS, 34=浮点解, 50=整数解 → 有效
            if heading.heading_type not in (16, 34, 50):
                return
            if heading.sol_status not in (0, 2):
                return
            self._latest_rtk_heading = float(heading.heading_deg)
        except Exception:
            pass

    def _status_callback(self):
        if self._calibration_done or self._samples_collected:
            return
        n = len(self._samples)
        amcl_ok = self._latest_amcl is not None
        fix_ok = self._latest_fix is not None
        heading_ok = not self._use_rtk_heading or self._latest_rtk_heading is not None
        self.get_logger().info(
            f'采集进度: {n}/{self._sample_count} 帧, '
            f'AMCL={"✓" if amcl_ok else "✗"}, GPS={"✓" if fix_ok else "✗"}, '
            f'航向={"✓" if heading_ok else "✗"}'
        )
        if not amcl_ok:
            self.get_logger().warn(
                f'未收到 AMCL 数据 ({self.get_parameter("amcl_pose_topic").value})，'
                '请确认 AMCL 已启动并定位收敛')
        if not fix_ok:
            self.get_logger().warn(
                f'未收到有效 GPS 数据 ({self.get_parameter("fix_topic").value})，'
                '请确认 gps_preprocessor 已启动')
        if not heading_ok and self._use_rtk_heading:
            self.get_logger().warn(
                f'未收到 RTK 航向数据，请确认 /rtk_pvh 正在发布且航向已收敛')

    # ==================================================================
    #  采样与标定
    # ==================================================================

    def _try_sample(self):
        if self._samples_collected or self._calibration_done:
            return
        if self._latest_amcl is None or self._latest_fix is None:
            return

        # 手动模式
        if self._manual_mode:
            self._do_calibrate_manual()
            return

        # 需要航向时检查
        if self._use_rtk_heading and self._latest_rtk_heading is None:
            return

        # 累积采样
        ax, ay, ayaw = self._latest_amcl
        fix = self._latest_fix
        e, n = _deg_to_utm(self._to_utm, fix.longitude, fix.latitude)
        hdg = self._latest_rtk_heading if self._use_rtk_heading else 0.0
        self._samples.append((ax, ay, ayaw, e, n, hdg))

        n = len(self._samples)
        if n >= self._sample_count:
            self._samples_collected = True
            self._do_calibrate_auto()
        elif n % 5 == 0:
            self.get_logger().info(f'采样中... {n}/{self._sample_count}')

    def _do_calibrate_auto(self):
        """自动模式：带旋转的标定"""
        if len(self._samples) == 0:
            return

        # 对每帧独立计算 (E₀, N₀, θ₀)，再取平均
        e0_list, n0_list, h0_list = [], [], []

        for ax, ay, ayaw, utm_e, utm_n, hdg in self._samples:
            # θ₀ = RTK真北航向 - AMCL航向（地图朝向 = 真实 - 地图）
            theta0 = math.radians(hdg) - ayaw
            cos_t0 = math.cos(theta0)
            sin_t0 = math.sin(theta0)

            # E₀ = utm_e - ax*cos(θ₀) + ay*sin(θ₀)
            # N₀ = utm_n - ax*sin(θ₀) - ay*cos(θ₀)
            e0 = utm_e - ax * cos_t0 + ay * sin_t0
            n0 = utm_n - ax * sin_t0 - ay * cos_t0

            e0_list.append(e0)
            n0_list.append(n0)
            h0_list.append(math.degrees(theta0))

        e0 = sum(e0_list) / len(e0_list)
        n0 = sum(n0_list) / len(n0_list)
        h0 = sum(h0_list) / len(h0_list)

        # 标准差
        if len(e0_list) > 1:
            std_e = math.sqrt(sum((x - e0)**2 for x in e0_list) / (len(e0_list) - 1))
            std_n = math.sqrt(sum((x - n0)**2 for x in n0_list) / (len(n0_list) - 1))
            std_h = math.sqrt(sum((x - h0)**2 for x in h0_list) / (len(h0_list) - 1))
            self.get_logger().info(
                f'估计标准差: E={std_e:.3f}m, N={std_n:.3f}m, heading={std_h:.3f}°')

        self._save_result(e0, n0, h0)

    def _do_calibrate_manual(self):
        """手动模式：已知地图坐标 + 当前 GPS + RTK航向"""
        if self._latest_fix is None:
            self.get_logger().error('手动标定: 尚未收到有效 GPS 数据')
            return

        mx, my = self._manual_pose
        myaw = self._manual_yaw
        fix = self._latest_fix
        e, n = _deg_to_utm(self._to_utm, fix.longitude, fix.latitude)

        # 航向
        if self._use_rtk_heading and self._latest_rtk_heading is not None:
            hdg = self._latest_rtk_heading
            theta0 = math.radians(hdg) - myaw
        else:
            theta0 = 0.0

        cos_t0 = math.cos(theta0)
        sin_t0 = math.sin(theta0)

        e0 = e - mx * cos_t0 + my * sin_t0
        n0 = n - mx * sin_t0 - my * cos_t0
        h0 = math.degrees(theta0)

        self.get_logger().info(
            f'手动标定: 地图({mx:.2f}, {my:.2f}, {math.degrees(myaw):.1f}°) '
            f'+ RTK UTM({e:.2f}, {n:.2f}) '
            f'→ 原点 UTM({e0:.2f}, {n0:.2f}) heading={h0:.3f}°'
        )
        self._save_result(e0, n0, h0)

    def _save_result(self, e0, n0, heading_deg):
        """保存标定结果到 YAML

        注意：使用手动格式化字符串而非 yaml.dump，因为 yaml.dump 对 Python
        float 使用 repr() 输出，会导致精度丢失。
        例如 round(24.610000000000, 8) → 24.61 → yaml 写入 "24.61"（仅2位小数），
        纬度 0.01° 误差 ≈ 1.1km，对导航是灾难性的。
        """
        a0 = 0.0  # 暂不处理高度
        lon0, lat0 = _utm_to_deg(self._to_wgs, e0, n0)

        note = (
            f'由 calibrate_map_origin.py 自动生成 '
            f'(RTK航向={"是" if self._use_rtk_heading else "否"})'
        )
        yaml_content = (
            f'map_origin:\n'
            f'  latitude: {lat0:.8f}\n'
            f'  longitude: {lon0:.8f}\n'
            f'  altitude: {a0:.1f}\n'
            f'  heading_deg: {heading_deg:.4f}\n'
            f'  utm_easting: {e0:.3f}\n'
            f'  utm_northing: {n0:.3f}\n'
            f'  utm_zone: {self._utm_zone}\n'
            f'note: "{note}"\n'
        )

        os.makedirs(os.path.dirname(self._output_file), exist_ok=True)
        with open(self._output_file, 'w') as f:
            f.write(yaml_content)

        self._calibration_done = True

        print()
        print('=' * 60)
        print('  标定完成！')
        print(f'  地图原点 GPS:    ({lat0:.8f}, {lon0:.8f})')
        print(f'  地图朝向:         {heading_deg:.4f}° (0=正北, 正=东偏)')
        print(f'  地图原点 UTM:    ({e0:.3f}, {n0:.3f})')
        print(f'  已保存到:         {self._output_file}')
        print()
        print('  接下来运行:')
        print('    ros2 run gps_fusion rtk_initial_pose.py --ros-args \\')
        print(f'      -p map_origin_lat:={lat0:.8f} -p map_origin_lon:={lon0:.8f} \\')
        print(f'      -p map_origin_heading_deg:={heading_deg:.4f}')
        print()
        print('  或直接使用 YAML 文件:')
        print(f'    ros2 run gps_fusion rtk_initial_pose.py --ros-args \\')
        print(f'      -p map_origin_file:={self._output_file}')
        print('=' * 60)
        print()

    # ==================================================================
    #  辅助
    # ==================================================================

    @staticmethod
    def _compute_h_accuracy(msg):
        if msg.position_covariance_type > 0:
            return math.sqrt(msg.position_covariance[0] + msg.position_covariance[4])
        return float('inf')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CalibrateMapOrigin()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
