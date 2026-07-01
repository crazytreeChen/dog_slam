#!/usr/bin/env python3
"""
GPS轨迹数据发布节点 - gps_fusion 包

订阅 LIO 里程计和 GPS 融合里程计，积累轨迹点，
以 nav_msgs/Path 格式发布供前端通过 rosbridge 订阅。

用法:
  ros2 run gps_fusion trajectory_server.py
  ros2 run gps_fusion trajectory_server.py --ros-args -p publish_rate:=5.0
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix

from collections import deque
import threading
import pyproj


class TrajectoryStore:
    """线程安全的轨迹数据存储"""

    def __init__(self, max_points=10000):
        self.lock = threading.Lock()
        self.lio_points = deque(maxlen=max_points)
        self.fused_points = deque(maxlen=max_points)
        self.max_points = max_points
        self._lio_dirty = False
        self._fused_dirty = False

    def add_lio(self, x, y, t):
        with self.lock:
            self.lio_points.append((x, y, t))
            self._lio_dirty = True

    def add_fused(self, x, y, t):
        with self.lock:
            self.fused_points.append((x, y, t))
            self._fused_dirty = True

    def get_lio_path(self, frame_id='odom'):
        with self.lock:
            pts = list(self.lio_points)
            self._lio_dirty = False
        return _make_path(pts, frame_id)

    def get_fused_path(self, frame_id='odom'):
        with self.lock:
            pts = list(self.fused_points)
            self._fused_dirty = False
        return _make_path(pts, frame_id)

    def get_lio_raw(self):
        with self.lock:
            return list(self.lio_points)

    def get_fused_raw(self):
        with self.lock:
            return list(self.fused_points)


def _make_path(points, frame_id):
    """将点列表转为 nav_msgs/Path"""
    path = Path()
    path.header.frame_id = frame_id
    for (x, y, t) in points:
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        path.poses.append(pose)
    return path


class TrajectoryBroadcaster(Node):
    """订阅里程计 + 定时发布轨迹 Path"""

    def __init__(self):
        super().__init__('trajectory_broadcaster')

        self.declare_parameter('max_points', 10000)
        self.declare_parameter('publish_rate', 5.0)  # Hz
        self.declare_parameter('frame_id', 'odom')
        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('utm_hemisphere', 'N')

        max_points = self.get_parameter('max_points').value
        publish_rate = self.get_parameter('publish_rate').value
        self.frame_id = self.get_parameter('frame_id').value

        # UTM → WGS84 转换器（用于生成经纬度轨迹发布给前端地图）
        utm_zone = self.get_parameter('utm_zone').value
        utm_hem = self.get_parameter('utm_hemisphere').value
        proj_utm = '+proj=utm +zone=%d +%s +datum=WGS84 +units=m +no_defs' % (
            utm_zone, 'north' if utm_hem == 'N' else 'south')
        self._utm_to_wgs84 = pyproj.Transformer.from_proj(
            proj_utm, 'EPSG:4326', always_xy=True)
        self.get_logger().info('UTM→WGS84 转换器: zone=%d%s' % (utm_zone, utm_hem))

        self.store = TrajectoryStore(max_points=max_points)

        # 订阅 LIO 里程计
        self.lio_sub = self.create_subscription(
            Odometry, '/lio_odom', self._lio_callback, 10)

        # 订阅 GPS 融合里程计
        self.fused_sub = self.create_subscription(
            Odometry, '/odometry/gps_fused', self._fused_callback, 10)

        # 发布轨迹 Path（UTM 坐标）
        self.lio_path_pub = self.create_publisher(Path, '/trajectory/lio', 10)
        self.fused_path_pub = self.create_publisher(Path, '/trajectory/fused', 10)

        # 发布轨迹 Path（经纬度坐标，供前端 Leaflet 地图使用）
        self.lio_latlon_pub = self.create_publisher(Path, '/trajectory/lio_latlon', 10)
        self.fused_latlon_pub = self.create_publisher(Path, '/trajectory/fused_latlon', 10)

        # 订阅 GPS 经纬度，限频转发给前端（避免高频数据冲击）
        self.declare_parameter('gps_publish_interval', 5.0)  # 秒
        self.declare_parameter('gps_min_displacement', 1e-6)  # 最小经纬度变化（约0.1m）
        gps_interval = self.get_parameter('gps_publish_interval').value
        self.gps_min_displacement = self.get_parameter('gps_min_displacement').value

        self.gps_fix_sub = self.create_subscription(
            NavSatFix, '/fix_filtered', self._gps_fix_callback, 10)
        self.gps_latlon_pub = self.create_publisher(
            NavSatFix, '/gps/current_latlon', 10)

        # 经纬度限频状态
        self._last_gps_publish_time = 0.0
        self._last_published_lat = None
        self._last_published_lon = None

        # 定时发布
        period = 1.0 / publish_rate
        self.pub_timer = self.create_timer(period, self._publish_trajectories)

        self.get_logger().info(
            '轨迹发布节点已启动 (速率: %.1f Hz, 最大点数: %d)'
            % (publish_rate, max_points))

    def _lio_callback(self, msg: Odometry):
        t = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.store.add_lio(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            t,
        )

    def _fused_callback(self, msg: Odometry):
        t = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.store.add_fused(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            t,
        )

    def _gps_fix_callback(self, msg: NavSatFix):
        """限频 + 去重后转发 GPS 经纬度给前端"""
        now = self.get_clock().now().nanoseconds * 1e-9
        gps_interval = self.get_parameter('gps_publish_interval').value

        # 限频：未到间隔则跳过
        if now - self._last_gps_publish_time < gps_interval:
            return

        # 去重：位置未变则跳过
        if (self._last_published_lat is not None and
                abs(msg.latitude - self._last_published_lat) < self.gps_min_displacement and
                abs(msg.longitude - self._last_published_lon) < self.gps_min_displacement):
            return

        self._last_gps_publish_time = now
        self._last_published_lat = msg.latitude
        self._last_published_lon = msg.longitude

        out = NavSatFix()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.latitude = msg.latitude
        out.longitude = msg.longitude
        out.altitude = msg.altitude
        out.status = msg.status
        out.position_covariance = msg.position_covariance
        out.position_covariance_type = msg.position_covariance_type
        self.gps_latlon_pub.publish(out)

    def _points_to_latlon_path(self, points, frame_id='wgs84'):
        """将 UTM (x,y) 点列表转为经纬度 (lon=x, lat=y) 的 Path"""
        path = Path()
        path.header.frame_id = frame_id
        for (x, y, _t) in points:
            lon, lat = self._utm_to_wgs84.transform(x, y)
            pose = PoseStamped()
            pose.pose.position.x = lon   # Leaflet [lat, lng]
            pose.pose.position.y = lat
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _publish_trajectories(self):
        # UTM 坐标路径（Canvas 或 rosbag 分析用）
        lio_path = self.store.get_lio_path(self.frame_id)
        if lio_path.poses:
            lio_path.header.stamp = self.get_clock().now().to_msg()
            self.lio_path_pub.publish(lio_path)

        fused_path = self.store.get_fused_path(self.frame_id)
        if fused_path.poses:
            fused_path.header.stamp = self.get_clock().now().to_msg()
            self.fused_path_pub.publish(fused_path)

        # 经纬度路径（Leaflet 地图用）
        lio_ll = self.store.get_lio_raw()
        if lio_ll:
            ll_path = self._points_to_latlon_path(lio_ll)
            ll_path.header.stamp = self.get_clock().now().to_msg()
            self.lio_latlon_pub.publish(ll_path)

        fused_ll = self.store.get_fused_raw()
        if fused_ll:
            ll_path = self._points_to_latlon_path(fused_ll)
            ll_path.header.stamp = self.get_clock().now().to_msg()
            self.fused_latlon_pub.publish(ll_path)


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = TrajectoryBroadcaster()
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
