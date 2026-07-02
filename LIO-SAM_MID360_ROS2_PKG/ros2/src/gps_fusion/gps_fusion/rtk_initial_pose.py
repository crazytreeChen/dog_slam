#!/usr/bin/env python3
"""
RTK 初始位姿发布节点（支持朝向）

将 RTK (lat/lon + 真北航向) 转换为地图坐标系 (map frame) 位姿，
发布 PoseWithCovarianceStamped 到 /initialpose 供 AMCL 使用。

转换链（带旋转）:
    RTK lat/lon → pyproj → UTM (E, N)
    RTK heading_deg → 真北航向 θ

    地图原点: UTM (E₀, N₀) + 朝向 θ₀ (heading_deg)

    dx = E - E₀,  dy = N - N₀
    map_x   =  dx * cos(θ₀) + dy * sin(θ₀)
    map_y   = -dx * sin(θ₀) + dy * cos(θ₀)
    map_yaw = θ - θ₀

地图原点 GPS 通过两种方式获取（按优先级）：
    1. 参数 map_origin_lat / map_origin_lon / map_origin_heading_deg
    2. YAML 文件 map_gps_origin.yaml（由 calibrate_map_origin.py 自动标定）

触发策略：
    - 首次收到有效 GPS + 航向后发布一次 initialpose
    - GPS 精度提升时重新发布
    - 无 RTK 航向时 yaw 为 0，协方差给大值让 AMCL 自己搜索朝向

用法:
    # 预配置模式
    ros2 run gps_fusion rtk_initial_pose.py --ros-args \
      -p map_origin_lat:=24.610 -p map_origin_lon:=118.030 \
      -p map_origin_heading_deg:=45.0

    # 自动标定模式（从 YAML 读取）
    ros2 run gps_fusion rtk_initial_pose.py --ros-args \
      -p map_origin_file:=/path/to/map_gps_origin.yaml

    # 建图时记录模式（不在导航中使用）
    ros2 run gps_fusion rtk_initial_pose.py --ros-args \
      -p record_only:=true
"""

import os
import math
import yaml
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
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


def _yaw_to_quat(yaw: float) -> Quaternion:
    """yaw (rad) → Quaternion"""
    half = yaw * 0.5
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


class RtkInitialPosePublisher(Node):
    """RTK → 地图初始位姿发布器（支持朝向旋转）"""

    def __init__(self):
        super().__init__('rtk_initial_pose')

        # ---- 参数 ----
        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('map_origin_lat', float('nan'))
        self.declare_parameter('map_origin_lon', float('nan'))
        self.declare_parameter('map_origin_alt', 0.0)
        self.declare_parameter('map_origin_heading_deg', float('nan'))
        self.declare_parameter('map_origin_file', '')
        self.declare_parameter('min_accuracy', 15.0)        # 最低水平精度门槛（m）
        self.declare_parameter('rtk_min_accuracy', 0.05)    # RTK固定解精度门槛（m）
        self.declare_parameter('publish_once', False)       # True=发布1次后停止
        self.declare_parameter('initialpose_topic', '/initialpose')
        self.declare_parameter('use_rtk_heading', True)     # 是否使用 RTK 航向
        self.declare_parameter('rtk_topic', '/rtk_pvh')     # RTK 原始数据话题（获取航向）
        self.declare_parameter('record_only', False)        # 建图模式：仅记录原点GPS，不发布initialpose

        self._utm_zone = self.get_parameter('utm_zone').value
        self._min_accuracy = self.get_parameter('min_accuracy').value
        self._rtk_min_accuracy = self.get_parameter('rtk_min_accuracy').value
        self._publish_once = self.get_parameter('publish_once').value
        self._initpose_topic = self.get_parameter('initialpose_topic').value
        self._use_rtk_hdg = self.get_parameter('use_rtk_heading').value
        self._record_only = self.get_parameter('record_only').value

        # ---- WGS84→UTM 转换器 ----
        utm_epsg = 32600 + abs(self._utm_zone) if self._utm_zone > 0 else 32700 + abs(self._utm_zone)
        self._transformer = Transformer.from_crs('epsg:4326', f'epsg:{utm_epsg}', always_xy=True)

        # ---- 确定地图原点（位置 + 朝向）----
        self._origin_utm, self._origin_heading_deg = self._resolve_map_origin()
        if self._origin_utm is None:
            self.get_logger().fatal(
                '地图原点 GPS 未配置！请设置 map_origin_lat/map_origin_lon 参数，'
                '或提供 map_origin_file YAML 文件。'
            )
            raise RuntimeError('map origin not configured')

        e0, n0, a0 = self._origin_utm
        h0 = self._origin_heading_deg
        self._theta0_rad = math.radians(h0)
        self.get_logger().info(
            f'地图原点: lat/lon → UTM({e0:.2f}, {n0:.2f}), '
            f'朝向={h0:.4f}° (0=正北)'
        )

        # ---- 状态 ----
        self._has_published = False
        self._best_accuracy_so_far = float('inf')
        self._latest_rtk_heading = None     # 真北航向（度）
        self._latest_fix = None             # 最近的 NavSatFix

        # ---- 订阅 /fix_filtered（位置） ----
        self._fix_sub = self.create_subscription(
            NavSatFix, '/fix_filtered', self._fix_callback, 10)

        # ---- 订阅 /rtk_pvh（航向） ----
        self._rtk_sub = None
        rtk_topic = self.get_parameter('rtk_topic').value
        if self._use_rtk_hdg and rtk_topic and _HAS_UNI_RTK_PVH:
            self._rtk_sub = self.create_subscription(
                UniRtkPvh, rtk_topic, self._rtk_callback, 10)
            self.get_logger().info(f'RTK 航向来源: {rtk_topic}')
        elif self._use_rtk_hdg and not _HAS_UNI_RTK_PVH:
            self.get_logger().warn(
                'robots_dog_msgs 未安装，无法获取 RTK 航向，'
                'initialpose 的 yaw 将设为 0（AMCL 自行搜索朝向）')
            self._use_rtk_hdg = False

        # ---- 发布 initialpose ----
        if not self._record_only:
            self._pose_pub = self.create_publisher(
                PoseWithCovarianceStamped, self._initpose_topic, 10)
        else:
            self._pose_pub = None
            self.get_logger().info(
                '建图记录模式：不发布 /initialpose，仅记录地图原点 GPS')

        self.get_logger().info(
            f'RTK 初始位姿节点已启动 '
            f'(精度门槛={self._min_accuracy}m, '
            f'RTK航向={"启用" if self._use_rtk_hdg else "禁用"}, '
            f'record_only={self._record_only})'
        )

    # ==================================================================
    #  地图原点获取
    # ==================================================================

    def _resolve_map_origin(self):
        """按优先级获取地图原点 (UTM坐标, 朝向角度)。

        返回 ((easting, northing, altitude), heading_deg) 或 (None, None)。
        """
        # 1. 参数直接指定
        lat = self.get_parameter('map_origin_lat').value
        lon = self.get_parameter('map_origin_lon').value
        alt = self.get_parameter('map_origin_alt').value
        hdg = self.get_parameter('map_origin_heading_deg').value

        if not (math.isnan(lat) or math.isnan(lon)):
            e, n = _deg_to_utm(self._transformer, lon, lat)
            h = hdg if not math.isnan(hdg) else 0.0
            self.get_logger().info(
                f'地图原点来自参数: ({lat:.6f}, {lon:.6f}) '
                f'heading={h:.4f}°'
            )
            return (e, n, alt), h

        # 2. YAML 文件
        yaml_path = self.get_parameter('map_origin_file').value
        if not yaml_path:
            default_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'map_gps_origin.yaml')
            if os.path.exists(default_path):
                yaml_path = default_path

        if yaml_path and os.path.exists(yaml_path):
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                origin = data.get('map_origin', {})
                lat = origin.get('latitude')
                lon = origin.get('longitude')
                alt = origin.get('altitude', 0.0)
                hdg = origin.get('heading_deg', 0.0)
                if lat is not None and lon is not None:
                    e, n = _deg_to_utm(self._transformer, lon, lat)
                    self.get_logger().info(
                        f'地图原点来自文件 {yaml_path}: '
                        f'({lat:.6f}, {lon:.6f}) heading={hdg:.4f}°'
                    )
                    return (e, n, alt), hdg
            except Exception as ex:
                self.get_logger().error(f'读取地图原点文件失败: {ex}')

        return None, None

    # ==================================================================
    #  回调
    # ==================================================================

    def _fix_callback(self, msg: NavSatFix):
        """收到 /fix_filtered 后尝试发布 initialpose"""
        self._latest_fix = msg
        self._try_publish()

    def _rtk_callback(self, msg):
        """从原始 RTK 消息获取航向"""
        try:
            heading = msg.heading
            if heading.heading_type not in (16, 17, 34, 50):
                return
            if heading.sol_status not in (0, 2):
                return
            self._latest_rtk_heading = float(heading.heading_deg)
        except Exception:
            pass

    # ==================================================================
    #  发布逻辑
    # ==================================================================

    def _try_publish(self):
        if self._latest_fix is None:
            return
        if self._publish_once and self._has_published:
            return

        msg = self._latest_fix

        # 检查位置精度
        h_acc = self._compute_horizontal_accuracy(msg)
        is_fix = self._is_rtk_fix(msg)

        if h_acc > self._min_accuracy:
            self.get_logger().debug(
                f'GPS精度不够: {h_acc:.3f}m > {self._min_accuracy}m')
            return
        if is_fix and h_acc > self._rtk_min_accuracy:
            return

        # 只在精度提升时重新发布
        if h_acc >= self._best_accuracy_so_far and self._has_published:
            return
        self._best_accuracy_so_far = h_acc

        # RTK → UTM
        e, n = _deg_to_utm(self._transformer, msg.longitude, msg.latitude)
        e0, n0, _ = self._origin_utm
        dx = e - e0
        dy = n - n0

        # 旋转到地图坐标系
        cos_t = math.cos(self._theta0_rad)
        sin_t = math.sin(self._theta0_rad)
        map_x = dx * cos_t + dy * sin_t
        map_y = -dx * sin_t + dy * cos_t

        # 朝向：RTK航向 − 地图旋转角
        # 有RTK航向 → 直接计算 map_yaw
        # 无RTK航向 → yaw=0，大方差让 AMCL 自己找
        if self._use_rtk_hdg and self._latest_rtk_heading is not None:
            rtk_yaw_rad = math.radians(self._latest_rtk_heading)
            map_yaw = rtk_yaw_rad - self._theta0_rad
            yaw_cov = 0.01  # 有航向观测，小方差
            heading_str = f'{math.degrees(map_yaw):.1f}°'
        else:
            map_yaw = 0.0
            yaw_cov = 0.5   # 无航向，大方差
            heading_str = '无(AMCL自行搜索)'

        # 建图模式：只记录原点数据，不发布
        if self._record_only:
            self.get_logger().info(
                f'[建图记录] 地图原点 GPS: 当前 robot 位于 '
                f'lat={msg.latitude:.8f}, lon={msg.longitude:.8f}, '
                f'对应 map 坐标 (0,0)，朝向='
                f'{self._latest_rtk_heading:.2f}°'
                if self._latest_rtk_heading is not None
                else '[建图记录] 记录中，RTK航向未收敛')
            self._has_published = True
            return

        # 发布
        self._publish_pose(map_x, map_y, map_yaw, h_acc, yaw_cov, is_fix)
        self._has_published = True

        self.get_logger().info(
            f'已发布 initialpose: map({map_x:.2f}, {map_y:.2f}, {heading_str}), '
            f'精度={h_acc:.3f}m, RTK固定解={is_fix}'
        )

    # ==================================================================
    #  辅助方法
    # ==================================================================

    def _compute_horizontal_accuracy(self, msg: NavSatFix) -> float:
        if msg.position_covariance_type > 0:
            return math.sqrt(
                msg.position_covariance[0] + msg.position_covariance[4]
            )
        return float('inf')

    def _is_rtk_fix(self, msg: NavSatFix) -> bool:
        return msg.status.status >= 4

    def _publish_pose(self, x, y, yaw, h_acc, yaw_cov, is_fix):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.pose = Pose(
            position=Point(x=x, y=y, z=0.0),
            orientation=_yaw_to_quat(yaw),
        )

        # 协方差
        cov = [0.0] * 36
        var_xy = h_acc ** 2
        cov[0] = var_xy      # x
        cov[7] = var_xy      # y
        cov[35] = yaw_cov    # yaw
        pose_msg.pose.covariance = cov

        self._pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RtkInitialPosePublisher()
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
