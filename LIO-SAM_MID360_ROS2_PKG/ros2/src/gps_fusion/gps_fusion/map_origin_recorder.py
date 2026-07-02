#!/usr/bin/env python3
"""
建图阶段地图原点 GPS 记录节点

机器人在 LIO 地图原点 (0,0,0) 启动建图时，机器人固定不动，此时采集 RTK
经纬度 + 真北航向作为地图原点 GPS 锚点，写入 map_gps_origin.yaml 供导航阶段使用。

使用场景:
    建图模式下无 AMCL、无 map frame（slam_toolbox/octomap 不发 map→odom TF），
    机器人停在原点不动，RTK 经纬度即为原点 GPS，RTK 航向即为地图朝向 θ₀。

触发方式:
    1. 自动模式（默认）: RTK 收敛后自动采集 sample_count 帧平均写入
    2. 手动模式: 调用 /gps_origin/record (std_srvs/Trigger) service 即时记录

用法:
    # 建图时自动记录（RTK 收敛后采集 10 帧平均）
    ros2 launch gps_fusion map_origin_record.launch.py

    # 手动触发记录（建图员确认 RTK 收敛后调用）
    ros2 service call /gps_origin/record std_srvs/srv/Trigger

    # 指定输出文件
    ros2 run gps_fusion map_origin_recorder.py --ros-args \
        -p output_file:=/tmp/map_origin.yaml
"""

import math
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from pyproj import Transformer

# 共享转换模块
from gps_fusion.gps_transform import (
    make_utm_transformer, make_wgs_transformer,
    latlon_to_utm, utm_to_latlon,
    compute_horizontal_accuracy, detect_rtk_quality,
    circular_mean_heading, write_map_origin_yaml,
)

# 尝试导入 RTK 消息类型（获取航向）
try:
    from robots_dog_msgs.msg import UniRtkPvh
    _HAS_UNI_RTK_PVH = True
except ImportError:
    UniRtkPvh = None
    _HAS_UNI_RTK_PVH = False


class MapOriginRecorder(Node):
    """建图阶段地图原点 GPS 记录器（单点记录 + 多帧平均）"""

    def __init__(self):
        super().__init__('map_origin_recorder')

        # ---- 参数 ----
        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('fix_topic', '/fix_filtered')
        self.declare_parameter('rtk_topic', '/rtk_pvh')
        self.declare_parameter('use_rtk_heading', True)
        self.declare_parameter('sample_count', 10)          # 自动模式采集帧数
        self.declare_parameter('min_accuracy', 5.0)         # 最低水平精度（m）
        self.declare_parameter('rtk_min_accuracy', 0.1)     # RTK 精度门槛
        self.declare_parameter('output_file', '')
        self.declare_parameter('auto_record', True)         # 自动记录模式
        self.declare_parameter('odom_topic', '/lio/robo/odom')
        self.declare_parameter('origin_distance_threshold', 1.0)
        self.declare_parameter('require_origin_odom', True)  # 模拟时设为 false 跳过 odom 检查

        self._utm_zone = self.get_parameter('utm_zone').value
        self._sample_count = self.get_parameter('sample_count').value
        self._min_accuracy = self.get_parameter('min_accuracy').value
        self._rtk_min_accuracy = self.get_parameter('rtk_min_accuracy').value
        self._use_rtk_heading = self.get_parameter('use_rtk_heading').value
        self._auto_record = self.get_parameter('auto_record').value
        self._odom_threshold = self.get_parameter('origin_distance_threshold').value
        self._require_origin_odom = self.get_parameter('require_origin_odom').value

        # 输出文件路径
        output_file = self.get_parameter('output_file').value
        if not output_file:
            output_file = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'map_gps_origin.yaml')
        self._output_file = os.path.abspath(output_file)

        # ---- WGS84↔UTM 转换器 ----
        self._to_utm = make_utm_transformer(self._utm_zone)
        self._to_wgs = make_wgs_transformer(self._utm_zone)

        # ---- 采样状态 ----
        self._lat_samples = []
        self._lon_samples = []
        self._heading_samples = []
        self._quality_samples = []
        self._latest_fix = None
        self._latest_rtk_heading = None
        self._record_done = False
        self._latest_odom_x = None
        self._latest_odom_y = None

        # ---- 订阅 /fix_filtered ----
        self._fix_sub = self.create_subscription(
            NavSatFix,
            self.get_parameter('fix_topic').value,
            self._fix_callback, 10)

        # ---- 订阅 /rtk_pvh（航向） ----
        self._rtk_sub = None
        rtk_topic = self.get_parameter('rtk_topic').value
        if self._use_rtk_heading and rtk_topic and _HAS_UNI_RTK_PVH:
            self._rtk_sub = self.create_subscription(
                UniRtkPvh, rtk_topic, self._rtk_callback, 10)
            self.get_logger().info(f'RTK 航向来源: {rtk_topic}')
        elif self._use_rtk_heading and not _HAS_UNI_RTK_PVH:
            self.get_logger().warn(
                'robots_dog_msgs 未安装，无法获取 RTK 航向，'
                '将使用 heading_deg=0 记录（地图与真北对齐）')
            self._use_rtk_heading = False

        self._odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_callback, 10,
        )

        # ---- 手动触发 service ----
        self._record_srv = self.create_service(
            Trigger, '/gps_origin/record', self._record_callback)

        # ---- 状态打印定时器 ----
        self._status_timer = self.create_timer(3.0, self._status_callback)

        self.get_logger().info(
            f'地图原点记录节点已启动 '
            f'(auto_record={self._auto_record}, samples={self._sample_count}, '
            f'min_accuracy={self._min_accuracy}m, '
            f'RTK航向={"启用" if self._use_rtk_heading else "禁用"})')
        self.get_logger().info(f'输出文件: {self._output_file}')

    # ==================================================================
    #  回调
    # ==================================================================

    def _fix_callback(self, msg: NavSatFix):
        """收到 /fix_filtered，检查精度后采集样本"""
        if self._record_done:
            return

        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            return

        h_acc = compute_horizontal_accuracy(msg)
        if h_acc > self._min_accuracy:
            self.get_logger().debug(
                f'GPS 精度不够: {h_acc:.3f}m > {self._min_accuracy}m')
            return

        self._latest_fix = msg

        # 自动模式：达到采样数后自动写入
        if self._auto_record and not self._use_rtk_heading:
            # 无航向模式：只需 GPS，直接采集
            self._collect_sample(msg, 0.0)
        elif self._auto_record and self._latest_rtk_heading is not None:
            # 有航向模式：等航向收敛后采集
            self._collect_sample(msg, self._latest_rtk_heading)

        if self._auto_record and len(self._lat_samples) >= self._sample_count:
            if not self._is_at_origin():
                ox = self._latest_odom_x
                oy = self._latest_odom_y
                if ox is not None and oy is not None:
                    self.get_logger().warn(
                        f'odom 距原点较远 (({ox:.2f},{oy:.2f}) > {self._odom_threshold}m)，'
                        f'等待机器人回到原点后再记录')
                else:
                    self.get_logger().debug('odom 数据未就绪，暂不记录')
                return
            self._do_record('auto')

    def _rtk_callback(self, msg):
        """从 RTK 原始消息获取航向"""
        try:
            heading = msg.heading
            # heading_type: 16=单点, 17=DGPS, 34=浮点解, 50=整数解 → 有效
            if heading.heading_type not in (16, 17, 34, 50):
                return
            if heading.sol_status not in (0, 2):
                return
            self._latest_rtk_heading = float(heading.heading_deg)
        except Exception:
            pass

    def _odom_callback(self, msg: Odometry):
        self._latest_odom_x = msg.pose.pose.position.x
        self._latest_odom_y = msg.pose.pose.position.y

    def _is_at_origin(self):
        if not self._require_origin_odom:
            return True
        if self._latest_odom_x is None:
            return False
        return math.hypot(self._latest_odom_x, self._latest_odom_y) <= self._odom_threshold

    def _record_callback(self, request, response):
        """手动触发记录 service"""
        if self._record_done:
            response.success = False
            response.message = '记录已完成，不可重复记录'
            return response

        if self._latest_fix is None:
            response.success = False
            response.message = '尚未收到有效 GPS 数据，请确认 RTK 已收敛'
            return response

        # 手动模式：用当前所有已采集样本，不足则取当前最新一帧
        if len(self._lat_samples) == 0:
            hdg = self._latest_rtk_heading if self._use_rtk_heading else 0.0
            self._collect_sample(self._latest_fix, hdg)

        self._do_record('manual')
        response.success = True
        lat = sum(self._lat_samples) / len(self._lat_samples)
        lon = sum(self._lon_samples) / len(self._lon_samples)
        response.message = (
            f'记录成功: lat={lat:.8f}, lon={lon:.8f}, '
            f'samples={len(self._lat_samples)}, '
            f'file={self._output_file}')
        return response

    def _status_callback(self):
        """定时打印采集状态"""
        if self._record_done:
            return
        n = len(self._lat_samples)
        fix_ok = self._latest_fix is not None
        hdg_ok = not self._use_rtk_heading or self._latest_rtk_heading is not None
        self.get_logger().info(
            f'采集进度: {n}/{self._sample_count} 帧, '
            f'GPS={"OK" if fix_ok else "WAIT"}, '
            f'航向={"OK" if hdg_ok else "WAIT"}, '
            f'odom={"({:.2f},{:.2f})".format(self._latest_odom_x, self._latest_odom_y) if self._latest_odom_x is not None else "WAIT"}')
        if not fix_ok:
            self.get_logger().warn(
                '未收到有效 GPS，请确认 RTK 已收敛且 gps_preprocessor 已启动')
        if not hdg_ok and self._use_rtk_heading:
            self.get_logger().warn('未收到 RTK 航向，请确认 /rtk_pvh 已收敛')

    # ==================================================================
    #  采集与记录
    # ==================================================================

    def _collect_sample(self, msg: NavSatFix, heading_deg: float):
        """采集一帧样本"""
        self._lat_samples.append(msg.latitude)
        self._lon_samples.append(msg.longitude)
        self._heading_samples.append(heading_deg)
        self._quality_samples.append(detect_rtk_quality(msg))

    def _do_record(self, source: str):
        """计算平均并写入 YAML"""
        if len(self._lat_samples) == 0:
            self.get_logger().error('无样本可记录')
            return

        # 经纬度算术平均（小范围 RTK 抖动，算术平均足够）
        lat0 = sum(self._lat_samples) / len(self._lat_samples)
        lon0 = sum(self._lon_samples) / len(self._lon_samples)
        # 航向环形平均（处理 0/360 环绕）
        if self._use_rtk_heading:
            heading0 = circular_mean_heading(self._heading_samples)
        else:
            heading0 = 0.0

        # RTK 质量取最优
        quality_priority = {'RTK_FIX': 4, 'RTK_FLOAT': 3, 'DGPS': 2, 'GPS': 1}
        best_quality = max(self._quality_samples,
                           key=lambda q: quality_priority.get(q, 0))

        # UTM 坐标（写入 YAML 供校验）
        e0, n0 = latlon_to_utm(self._to_utm, lon0, lat0)

        # 写入 YAML
        write_map_origin_yaml(
            yaml_path=self._output_file,
            lat=lat0, lon=lon0, altitude=0.0,
            heading_deg=heading0,
            utm_zone=self._utm_zone,
            utm_e=e0, utm_n=n0,
            source=source,
            sample_count=len(self._lat_samples),
            rtk_quality=best_quality,
        )

        self._record_done = True
        self._status_timer.cancel()

        self.get_logger().info('=' * 60)
        self.get_logger().info('  地图原点 GPS 记录完成！')
        self.get_logger().info(f'  经纬度: ({lat0:.8f}, {lon0:.8f})')
        self.get_logger().info(f'  朝向:   {heading0:.4f}° (0=正北)')
        self.get_logger().info(f'  UTM:    ({e0:.3f}, {n0:.3f}) zone={self._utm_zone}')
        self.get_logger().info(f'  质量:   {best_quality}, 样本数: {len(self._lat_samples)}')
        self.get_logger().info(f'  来源:   {source}')
        self.get_logger().info(f'  文件:   {self._output_file}')
        self.get_logger().info('=' * 60)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MapOriginRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
