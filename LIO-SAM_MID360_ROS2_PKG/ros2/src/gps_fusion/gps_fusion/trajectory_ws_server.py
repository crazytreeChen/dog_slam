#!/usr/bin/env python3
"""
WebSocket 轨迹数据推送节点

订阅 /trajectory/lio_latlon 和 /trajectory/fused_latlon (nav_msgs/Path)，
将经纬度轨迹通过 WebSocket 推送到远程电脑的前端页面。

协议 (JSON):
  Client → Server:
    {"type": "mode", "mode": "incremental"}     # 切换到增量推送
    {"type": "mode", "mode": "full_periodic"}   # 切换到全量定时推送
    {"type": "get_full"}                         # 立即请求一次全量数据

  Server → Client:
    {"type": "full", "lio": [[lon,lat],...], "fused": [[lon,lat],...], "stats": {...}}
    {"type": "incremental", "lio": [[lon,lat]], "fused": [[lon,lat]], "stats": {...}}
    {"type": "info", "message": "..."}

用法:
  ros2 run gps_fusion trajectory_ws_server.py
  ros2 run gps_fusion trajectory_ws_server.py --ros-args -p ws_port:=8765

依赖: tornado（与 rosbridge_server 共用，已在机器人上安装）
"""

import json
import threading
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from nav_msgs.msg import Path
from sensor_msgs.msg import NavSatFix

try:
    import tornado.ioloop
    import tornado.web
    import tornado.websocket
    import tornado.httpserver
    import tornado.netutil
    HAS_TORNADO = True
except ImportError:
    HAS_TORNADO = False


WELCOME_MSG_TMPL = (
    '{{"type":"info","message":"GPS轨迹 WebSocket 已连接 | '
    '数据源: {gps_source} | 模式: incremental | '
    '发送 get_full 获取全量 | gps_source 切换数据源"}}'
)

def _welcome_msg(gps_source):
    return WELCOME_MSG_TMPL.format(gps_source=gps_source)


class TrajectoryBuffer:
    """线程安全的轨迹经纬度缓冲"""

    def __init__(self, max_points=20000):
        self._lock = threading.Lock()
        self._lio = deque(maxlen=max_points)
        self._fused = deque(maxlen=max_points)
        # 用于增量推送的游标
        self._lio_cursor = 0
        self._fused_cursor = 0

    def replace_lio(self, points):
        """全量替换 LIO 轨迹（points: [(lon, lat, t), ...]）"""
        with self._lock:
            self._lio = deque(points, maxlen=self._lio.maxlen)
            self._lio_cursor = 0

    def replace_fused(self, points):
        """全量替换融合轨迹"""
        with self._lock:
            self._fused = deque(points, maxlen=self._fused.maxlen)
            self._fused_cursor = 0

    def add_fused(self, lon, lat, t):
        """追加单个融合轨迹点（不重置游标，增量推送可用）"""
        with self._lock:
            self._fused.append((lon, lat, t))

    def drain_lio_incremental(self):
        """取出 LIO 新增点，返回 [(lon, lat), ...]"""
        with self._lock:
            pts = list(self._lio)
            new_pts = pts[self._lio_cursor:]
            self._lio_cursor = len(pts)
            return [(lon, lat) for (lon, lat, _t) in new_pts]

    def drain_fused_incremental(self):
        """取出融合新增点"""
        with self._lock:
            pts = list(self._fused)
            new_pts = pts[self._fused_cursor:]
            self._fused_cursor = len(pts)
            return [(lon, lat) for (lon, lat, _t) in new_pts]

    def get_full(self):
        """获取全量轨迹"""
        with self._lock:
            lio = [(lon, lat) for (lon, lat, _t) in self._lio]
            fused = [(lon, lat) for (lon, lat, _t) in self._fused]
            return lio, fused

    def get_stats(self):
        with self._lock:
            lio_len = len(self._lio)
            fused_len = len(self._fused)
            lio_last = self._lio[-1] if lio_len > 0 else None
            fused_last = self._fused[-1] if fused_len > 0 else None
        return {
            "lio_count": lio_len,
            "fused_count": fused_len,
            "lio_last": list(lio_last[:2]) if lio_last else None,
            "fused_last": list(fused_last[:2]) if fused_last else None,
        }

    def has_new_lio(self):
        with self._lock:
            return self._lio_cursor < len(self._lio)

    def has_new_fused(self):
        with self._lock:
            return self._fused_cursor < len(self._fused)


class _WsHandler(tornado.websocket.WebSocketHandler):
    """单个 WebSocket 客户端连接处理器（由 tornado 管理生命周期）"""

    def initialize(self, server_node):
        """server_node: TrajectoryWsServer 实例引用"""
        self._node = server_node

    def check_origin(self, origin):
        """允许所有来源（跨域）"""
        return True

    def open(self):
        remote = self.request.remote_ip
        self._node.get_logger().info('客户端连接: %s' % remote)

        with self._node._clients_lock:
            self._node._ws_clients.add(self)
            self._node._client_modes[self] = 'incremental'

        # 发送欢迎消息
        self.write_message(_welcome_msg(self._node._gps_source))
        # 发送当前全量数据
        lio, fused = self._node._buffer.get_full()
        stats = self._node._buffer.get_stats()
        stats['gps_source'] = self._node._gps_source
        self.write_message(json.dumps({
            "type": "full",
            "lio": lio,
            "fused": fused,
            "stats": stats,
            "timestamp": time.time(),
        }, ensure_ascii=False))

    def on_message(self, raw):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        mtype = msg.get('type', '')
        if mtype == 'mode':
            mode = msg.get('mode', 'incremental')
            if mode in ('incremental', 'full_periodic'):
                with self._node._clients_lock:
                    self._node._client_modes[self] = mode
                self._node.get_logger().info(
                    '客户端 %s 切换模式 → %s' %
                    (self.request.remote_ip, mode))
                self.write_message(json.dumps({
                    "type": "info",
                    "message": "模式已切换: %s" % mode,
                }))
        elif mtype == 'gps_source':
            source = msg.get('source', '/fix')
            if source in ('/fix', '/gps/fix'):
                self._node._set_gps_source(source)
                self.write_message(json.dumps({
                    "type": "info",
                    "message": "GPS数据源已切换: %s" % source,
                    "gps_source": source,
                }))
        elif mtype == 'rtk_source':
            source = msg.get('source', 'auto')
            if source in ('auto', 'real', 'test'):
                self._node._set_rtk_source(source)
                self.write_message(json.dumps({
                    "type": "info",
                    "message": "RTK数据源已切换: %s" % source,
                    "rtk_source": source,
                }))
        elif mtype == 'ping':
            # 心跳响应
            self.write_message(json.dumps({
                "type": "pong",
                "timestamp": time.time(),
            }))
        elif mtype == 'get_full':
            lio, fused = self._node._buffer.get_full()
            stats = self._node._buffer.get_stats()
            stats['gps_source'] = self._node._gps_source
            stats['rtk_source'] = self._node._rtk_source
            self.write_message(json.dumps({
                "type": "full",
                "lio": lio,
                "fused": fused,
                "stats": stats,
                "timestamp": time.time(),
            }, ensure_ascii=False))

    def on_close(self):
        remote = self.request.remote_ip
        with self._node._clients_lock:
            self._node._ws_clients.discard(self)
            self._node._client_modes.pop(self, None)
        self._node.get_logger().info('客户端断开: %s' % remote)


class TrajectoryWsServer(Node):
    """ROS2 节点：订阅经纬度轨迹 + 启动 WebSocket 服务（基于 tornado）"""

    def __init__(self):
        super().__init__('trajectory_ws_server')

        if not HAS_TORNADO:
            self.get_logger().fatal(
                '缺少 tornado 库，请安装: pip3 install tornado')
            raise RuntimeError('tornado not available')

        self.declare_parameter('ws_port', 8765)
        self.declare_parameter('ws_host', '0.0.0.0')
        self.declare_parameter('full_push_interval', 5.0)  # 全量推送间隔（秒）
        self.declare_parameter('gps_source', '/fix')        # 当前GPS数据源
        self.declare_parameter('rtk_source', 'auto')         # RTK源: auto/real/test

        self._ws_port = self.get_parameter('ws_port').value
        self._ws_host = self.get_parameter('ws_host').value
        self._full_push_interval = self.get_parameter('full_push_interval').value
        self._gps_source = self.get_parameter('gps_source').value
        self._rtk_source = self.get_parameter('rtk_source').value

        self._buffer = TrajectoryBuffer()
        self._ws_clients = set()          # 已连接的 WebSocket 客户端 (_WsHandler 实例)
        self._client_modes = {}           # _WsHandler → "incremental" | "full_periodic"
        self._clients_lock = threading.Lock()

        # 远程参数客户端（用于切换 gps_preprocessor 的数据源）
        self._gps_param_client = self.create_client(
            SetParameters, '/gps_preprocessor/set_parameters')

        # 订阅经纬度轨迹话题
        self._lio_sub = self.create_subscription(
            Path, '/trajectory/lio_latlon', self._lio_callback, 10)
        self._fused_sub = self.create_subscription(
            Path, '/trajectory/fused_latlon', self._fused_callback, 10)

        # 直接订阅 GPS 经纬度（绕过 EKF/navsat 链路，测试用）
        self._gps_raw_sub = self.create_subscription(
            NavSatFix, '/fix_filtered', self._gps_raw_callback, 10)

        # 全量定时推送定时器（ROS2 timer）
        self._full_timer = self.create_timer(
            self._full_push_interval, self._full_push_tick)

        # 增量推送检查定时器（~10Hz，低开销）
        self._inc_timer = self.create_timer(0.1, self._inc_push_tick)

        self.get_logger().info(
            'WebSocket 轨迹服务已就绪 ws://%s:%d (模式: 增量推送/全量定时)' %
            (self._ws_host, self._ws_port))

        # 在后台线程启动 tornado IOLoop（与 rosbridge_server 相同的模型）
        self._ioloop = None
        self._ws_app = None
        self._ws_server = None
        self._loop_thread = threading.Thread(
            target=self._run_tornado_loop, daemon=True)
        self._loop_thread.start()

    # ----- ROS2 回调 -----

    def _lio_callback(self, msg: Path):
        pts = []
        for pose in msg.poses:
            pts.append((pose.pose.position.x, pose.pose.position.y,
                        rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9))
        if pts:
            self._buffer.replace_lio(pts)

    def _fused_callback(self, msg: Path):
        pts = []
        for pose in msg.poses:
            pts.append((pose.pose.position.x, pose.pose.position.y,
                        rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9))
        if pts:
            self._buffer.replace_fused(pts)

    def _gps_raw_callback(self, msg: NavSatFix):
        """直接消费 GPS 经纬度，绕过 EKF/navsat 链路"""
        t = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self._buffer.add_fused(msg.longitude, msg.latitude, t)

    def _inc_push_tick(self):
        """增量推送检查：有新点就推"""
        if not self._ws_clients or not self._ioloop:
            return

        lio_new = self._buffer.drain_lio_incremental()
        fused_new = self._buffer.drain_fused_incremental()

        if not lio_new and not fused_new:
            return

        with self._clients_lock:
            target = [c for c in self._ws_clients
                      if self._client_modes.get(c, 'incremental') == 'incremental']

        if target:
            stats = self._buffer.get_stats()
            stats['gps_source'] = self._gps_source
            stats['rtk_source'] = self._rtk_source
            payload = json.dumps({
                "type": "incremental",
                "lio": lio_new,
                "fused": fused_new,
                "stats": stats,
            }, ensure_ascii=False)
            # 线程安全地投递到 tornado IOLoop
            self._ioloop.add_callback(self._broadcast, payload, target)

    def _full_push_tick(self):
        """全量定时推送"""
        if not self._ws_clients:
            return

        with self._clients_lock:
            target = [c for c in self._ws_clients
                      if self._client_modes.get(c, 'incremental') == 'full_periodic']

        if target:
            self._send_full(target)

    def _send_full(self, clients=None):
        """发送全量数据给指定客户端（None = 所有）"""
        if not self._ioloop:
            return
        lio, fused = self._buffer.get_full()
        stats = self._buffer.get_stats()
        stats['gps_source'] = self._gps_source
        stats['rtk_source'] = self._rtk_source
        payload = json.dumps({
            "type": "full",
            "lio": lio,
            "fused": fused,
            "stats": stats,
            "timestamp": time.time(),
        }, ensure_ascii=False)

        if clients is None:
            with self._clients_lock:
                clients = list(self._ws_clients)

        if clients:
            self._ioloop.add_callback(self._broadcast, payload, clients)

    # ----- Tornado WebSocket 服务 -----

    def _run_tornado_loop(self):
        """后台线程：运行 tornado IOLoop（对齐 rosbridge_server 模型）"""
        self._ioloop = tornado.ioloop.IOLoop()
        self._ioloop.make_current()

        # 构建 tornado Application
        self._ws_app = tornado.web.Application([
            (r'/', _WsHandler, {'server_node': self}),
        ])
        sockets = tornado.netutil.bind_sockets(
            self._ws_port, address=self._ws_host, reuse_port=True)
        self._ws_server = tornado.httpserver.HTTPServer(self._ws_app)
        self._ws_server.add_sockets(sockets)
        self.get_logger().info(
            'WebSocket 服务器已启动 (tornado): ws://%s:%d' %
            (self._ws_host, self._ws_port))

        try:
            self._ioloop.start()
        except Exception:
            pass

    async def _broadcast(self, message, clients):
        closed = []
        for handler in clients:
            try:
                await handler.write_message(message)
            except tornado.websocket.WebSocketClosedError:
                closed.append(handler)
            except Exception as e:
                self.get_logger().debug('发送失败: %s' % e)
                closed.append(handler)

        if closed:
            with self._clients_lock:
                for h in closed:
                    self._ws_clients.discard(h)
                    self._client_modes.pop(h, None)

    def shutdown(self):
        """优雅关闭 WebSocket 服务器"""
        if self._ioloop:
            self._ioloop.add_callback(self._ioloop.stop)
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=3.0)

    def _set_gps_source(self, source):
        """远程设置 gps_preprocessor 节点的 gps_source 参数"""
        if source == self._gps_source:
            return
        self._gps_source = source
        if not self._gps_param_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                'GPS参数服务不可用，仅更新本地记录: %s' % source)
            return
        req = SetParameters.Request()
        req.parameters = [Parameter('gps_source', value=source).to_parameter_msg()]
        future = self._gps_param_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_gps_source_result(f, source))

    def _on_gps_source_result(self, future, source):
        """处理 GPS 数据源远程切换结果"""
        try:
            result = future.result()
            if result is None:
                self.get_logger().warn(
                    'GPS数据源远程切换: %s (结果: future返回None，但参数可能已生效)' % source)
            elif hasattr(result, 'results') and all(
                    r.successful for r in result.results):
                self.get_logger().info(
                    'GPS数据源远程切换: %s (结果: 成功)' % source)
            else:
                self.get_logger().warn(
                    'GPS数据源远程切换: %s (结果: 失败, results=%s)' %
                    (source, getattr(result, 'results', 'N/A')))
        except Exception as e:
            self.get_logger().error(
                'GPS数据源远程切换异常: %s (err=%s)' % (source, e))

    def _set_rtk_source(self, source):
        """远程设置 gps_preprocessor 节点的 rtk_source 参数 (auto/real/test)"""
        if source == self._rtk_source:
            return
        self._rtk_source = source
        if not self._gps_param_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                'GPS参数服务不可用，仅更新本地RTK源记录: %s' % source)
            return
        req = SetParameters.Request()
        req.parameters = [Parameter('rtk_source', value=source).to_parameter_msg()]
        future = self._gps_param_client.call_async(req)
        future.add_done_callback(
            lambda f: self._on_rtk_source_result(f, source))

    def _on_rtk_source_result(self, future, source):
        """处理 RTK 数据源远程切换结果"""
        try:
            result = future.result()
            if result is None:
                self.get_logger().warn(
                    'RTK数据源远程切换: %s (future返回None)' % source)
            elif hasattr(result, 'results') and all(
                    r.successful for r in result.results):
                self.get_logger().info(
                    'RTK数据源远程切换: %s (成功)' % source)
            else:
                self.get_logger().warn(
                    'RTK数据源远程切换: %s (失败, results=%s)' %
                    (source, getattr(result, 'results', 'N/A')))
        except Exception as e:
            self.get_logger().error(
                'RTK数据源远程切换异常: %s (err=%s)' % (source, e))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrajectoryWsServer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            try:
                node.shutdown()
            except Exception:
                pass
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
