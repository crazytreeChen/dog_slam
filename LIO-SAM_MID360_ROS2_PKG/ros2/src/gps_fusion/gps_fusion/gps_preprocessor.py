#!/usr/bin/env python3
"""
GPS/RTK数据预处理节点 - gps_fusion 包独立版本

处理GPS无定位状态和NaN值，将经纬度转换为UTM坐标，确保只有有效的GPS数据进入EKF滤波器。
RTK高精度模式：RTK_FIX(4)/RTK_FLOAT(5) 时自动降低 covariance 门槛。
双数据源兼容：同时订阅 /fix (实际) 和 /gps/fix (测试)，通过 gps_source 参数切换。
额外支持 robots_dog_msgs/UniRtkPvh 原始 RTK 输入：
  - 真实硬件: 默认 /rtk_pvh（可通过 rtk_topic 参数配置）
  - 测试模拟: 始终订阅 /test/rtk_pvh（与真实硬件自动共存）

用法:
  ros2 run gps_fusion gps_preprocessor.py --ros-args -p utm_zone:=50
  ros2 run gps_fusion gps_preprocessor.py --ros-args -p gps_source:=/gps/fix
  ros2 run gps_fusion gps_preprocessor.py --ros-args -p rtk_topic:=/rtk_pvh
  ros2 param set /gps_preprocessor gps_source /gps/fix  # 运行时切换
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import math
from pyproj import Transformer


# 尝试导入 RTK 自定义消息；机器人上若已编译 robots_dog_msgs 则可用，
# 开发机未安装时也不影响 /fix 链路的常规功能。
try:
    from robots_dog_msgs.msg import UniRtkPvh
    _HAS_UNI_RTK_PVH = True
except ImportError:
    UniRtkPvh = None
    _HAS_UNI_RTK_PVH = False


# RTK 解状态常量（NavSatStatus.service 未明确定义时，通过 covariance 推断）
RTK_FIX_INDICATOR = 4
RTK_FLOAT_INDICATOR = 5
STATUS_GBAS_FIX = 2   # DGPS/地基增强（亚米级）


class GPSPreprocessor(Node):
    def __init__(self):
        super().__init__('gps_preprocessor')

        # 参数
        self.declare_parameter('min_satellites', 4)
        self.declare_parameter('max_hdop', 2.0)
        self.declare_parameter('min_accuracy', 0.1)         # 普通GPS精度门槛 (m)
        self.declare_parameter('rtk_min_accuracy', 0.02)    # RTK精度门槛 (m)
        self.declare_parameter('dgps_min_accuracy', 30.0)    # DGPS精度门槛 (m)，RTK模块上报的std可能偏保守
        self.declare_parameter('status_threshold', 0)       # 0=FIX, -1=NO_FIX
        self.declare_parameter('utm_zone', 50)
        self.declare_parameter('gps_source', '/fix')  # 双源切换: /fix (实际) /gps/fix (测试)
        self.declare_parameter('rtk_topic', '/rtk_pvh')     # RTK 原始数据话题，空字符串=不启用
        self.declare_parameter('indoor_threshold', 20)
        self.declare_parameter('outdoor_recover_threshold', 10)

        self.min_satellites = self.get_parameter('min_satellites').value
        self.max_hdop = self.get_parameter('max_hdop').value
        self.min_accuracy = self.get_parameter('min_accuracy').value
        self.rtk_min_accuracy = self.get_parameter('rtk_min_accuracy').value
        self.dgps_min_accuracy = self.get_parameter('dgps_min_accuracy').value
        self.status_threshold = self.get_parameter('status_threshold').value
        self.utm_zone = self.get_parameter('utm_zone').value
        self.gps_source = self.get_parameter('gps_source').value  # 当前活跃GPS源
        self.rtk_topic = self.get_parameter('rtk_topic').value
        self._indoor_threshold = self.get_parameter('indoor_threshold').value
        self._outdoor_recover_threshold = self.get_parameter('outdoor_recover_threshold').value

        # 创建投影器（使用新版 pyproj Transformer API）
        self.transformer = Transformer.from_crs(
            'epsg:4326',  # WGS84
            f'epsg:326{self.utm_zone:02d}' if self.utm_zone > 0 else 'epsg:4326',
            always_xy=True
        )

        # 双源订阅：同时订阅 /fix 和 /gps/fix，只处理活跃源的数据
        self._sub_fix = self.create_subscription(
            NavSatFix, '/fix', self._make_source_callback('/fix'), 10)
        self._sub_gps_fix = self.create_subscription(
            NavSatFix, '/gps/fix', self._make_source_callback('/gps/fix'), 10)

        # RTK 订阅：解析 /rtk_pvh 类型的 UniRtkPvh，转成 NavSatFix 后注入处理链路
        # 同时订阅真实硬件 (/rtk_pvh) 和测试模拟 (/test/rtk_pvh) 两条链路
        # 通过 rtk_source 参数控制活跃源: "real" / "test" / "auto"（默认auto=两条都收）
        self.declare_parameter('rtk_source', 'auto')
        self._rtk_subs = []
        self._rtk_source = self.get_parameter('rtk_source').value
        if _HAS_UNI_RTK_PVH:
            # 真实 GPS 硬件链路
            if self.rtk_topic:
                self._rtk_subs.append(
                    self.create_subscription(
                        UniRtkPvh, self.rtk_topic,
                        self._make_rtk_callback('real'), 10))
            # 测试/模拟链路
            self._rtk_subs.append(
                self.create_subscription(
                    UniRtkPvh, '/test/rtk_pvh',
                    self._make_rtk_callback('test'), 10))
            self.get_logger().info(
                f'RTK输入: 真实={self.rtk_topic or "(未启用)"}, 测试=/test/rtk_pvh, '
                f'当前源={self._rtk_source}')
        elif not _HAS_UNI_RTK_PVH:
            self.get_logger().warn(
                'robots_dog_msgs 未安装，RTK 输入将不可用。')

        # 参数变化回调（运行时切换数据源）
        self.add_on_set_parameters_callback(self._on_param_change)

        # 发布处理后的GPS数据（供 navsat_transform_node 消费）
        self.gps_pub = self.create_publisher(
            NavSatFix,
            '/fix_filtered',
            10
        )

        # 发布UTM坐标（原始绝对坐标）
        self.utm_pub = self.create_publisher(
            Odometry,
            '/fix_utm',
            10
        )

        # 发布相对UTM坐标（以第一条GPS为原点）
        self.odom_pub = self.create_publisher(
            Odometry,
            '/fix_odom',
            10
        )

        # 发布GPS可用状态
        self.gps_status_pub = self.create_publisher(
            Bool,
            '/gps/status',
            10
        )

        # 状态变量
        self.last_valid_gps = None
        self.gps_available = False
        self.gps_quality_counter = 0
        self.utm_origin = None
        self.rtk_mode = False  # 是否处于RTK模式
        self.rtk_active = False  # 当前是否正在使用 RTK 输入

        self._indoor_mode = False
        self._indoor_consecutive = 0
        self._outdoor_consecutive = 0

        # RTK 诊断状态（用于限频日志）
        self._rtk_reject_count = 0
        self._rtk_reject_reason = None     # 上次拒绝原因
        self._rtk_last_diag_time = 0.0     # 上次输出诊断的时间戳

        self.get_logger().info('GPS预处理节点已启动 (gps_fusion 独立包)')
        self.get_logger().info(
            '参数: min_sat=%d, max_hdop=%.1f, min_accuracy=%.2fm, dgps_accuracy=%.2fm, rtk_accuracy=%.3fm'
            % (self.min_satellites, self.max_hdop, self.min_accuracy, self.dgps_min_accuracy, self.rtk_min_accuracy)
        )
        self.get_logger().info(f'UTM区域: {self.utm_zone}')
        self.get_logger().info(f'GPS数据源: {self.gps_source} (双源兼容: /fix /gps/fix)')
        if self.rtk_topic and _HAS_UNI_RTK_PVH:
            self.get_logger().info(f'RTK数据源: {self.rtk_topic} (robots_dog_msgs/UniRtkPvh)')

    def _make_source_callback(self, source_name):
        """创建带源过滤的回调：只有活跃源的数据才进入处理"""
        def callback(msg):
            if source_name == self.gps_source:
                self.gps_callback(msg)
            # 非活跃源静默丢弃
        return callback

    def _make_rtk_callback(self, rtk_name):
        """创建带 RTK 源过滤的回调：只有当前活跃 RTK 源或 auto 模式才进入处理

        rtk_name: "real" (真实硬件 /rtk_pvh) 或 "test" (模拟 /test/rtk_pvh)
        """
        def callback(msg):
            # auto 模式：两条链路都接受；否则只接受匹配的链路
            if self._rtk_source not in ('auto', rtk_name):
                return
            self._rtk_callback_impl(msg, rtk_name)
        return callback

    def _rtk_callback_impl(self, msg, rtk_name='unknown'):
        """接收 robots_dog_msgs/UniRtkPvh，解析后转成 NavSatFix 注入 GPS 处理链路。

        RTK 有效性判定：
        - heading.sol_status ∈ {0, 2} 且 bestnav.p_sol_status ∈ {0, 2} (已解出/DGPS)
        - heading.heading_type ∈ {16, 17, 34, 50} (16=单点, 17=DGPS, 34=浮点解, 50=整数解)
          以 heading_type 作为统一质量指标，替代 pos_type
        - 未通过验证时输出 WARN 级诊断日志（限频），帮助判断 RTK 收敛状态

        室内/室外自动检测：
        - pos_type=0 且 svs_num=0 → 无卫星信号，累积后进入室内模式
        - 室内模式下 suppress 日志噪音，停止 GPS 数据输出
        - 信号恢复后自动退出室内模式
        """
        bestnav = msg.bestnav

        if bestnav.pos_type == 0 and bestnav.svs_num == 0:
            self._indoor_consecutive += 1
            self._outdoor_consecutive = 0
            if not self._indoor_mode and self._indoor_consecutive == self._indoor_threshold:
                self._indoor_mode = True
                self.gps_available = False
                self.get_logger().warn(
                    f'进入室内模式（连续{self._indoor_consecutive}帧无卫星信号），停止GPS输出')
            return
        else:
            self._outdoor_consecutive += 1
            self._indoor_consecutive = 0
            if self._indoor_mode and self._outdoor_consecutive >= self._outdoor_recover_threshold:
                self._indoor_mode = False
                self.get_logger().info('退出室内模式，GPS信号恢复')

        if self._indoor_mode:
            return
        navsat = self._rtk_to_navsat_fix(msg)
        if navsat is None:
            return

        if not self.rtk_active:
            self.rtk_active = True
            source_label = '/rtk_pvh' if rtk_name == 'real' else '/test/rtk_pvh'
            self.get_logger().info(f'RTK 输入已接入: {source_label}')

        # RTK 输入优先级高于普通 GPS，直接注入处理
        self.gps_callback(navsat)

    def _rtk_diag_log(self, reason, detail):
        """限频输出 RTK 诊断日志（每秒最多一次，且只在原因变化或每60次时输出）"""
        now = self.get_clock().now().nanoseconds * 1e-9
        self._rtk_reject_count += 1
        changed = (reason != self._rtk_reject_reason)
        if changed:
            self._rtk_reject_reason = reason
            self._rtk_reject_count = 1
        # 每秒最多输出一次，或每 60 次累计输出状态
        if changed or (now - self._rtk_last_diag_time) >= 1.0 or self._rtk_reject_count % 60 == 0:
            self._rtk_last_diag_time = now
            self.get_logger().warn(
                f'RTK 数据被过滤 [{self._rtk_reject_count}次]: {reason} — {detail}'
            )

    def _rtk_to_navsat_fix(self, msg):
        """将 UniRtkPvh 中的 bestnav + heading 转换为 NavSatFix。

        字段映射基于 robots_dog_msgs 自定义消息，与 auto_initial_pose_calibrator 保持一致。

        RTK 收敛判定：
        - sol_status: 0=已解出, 2=DGPS/差分  → 有效状态
        - p_sol_status: 0=已解出, 2=DGPS/差分 → 有效状态
        - heading_type: 16=单点, 17=DGPS, 34=浮点解, 50=整数解 → 统一质量指标，替代 pos_type
        """
        bestnav = msg.bestnav
        heading = msg.heading

        # 解状态必须有效：0=已解出, 2=DGPS/差分解
        sol_valid_h = heading.sol_status in (0, 2)
        sol_valid_b = bestnav.p_sol_status in (0, 2)
        if not sol_valid_h or not sol_valid_b:
            self._rtk_diag_log(
                'solution_status_invalid',
                f'heading.sol_status={heading.sol_status}'
                f'(期望0或2), bestnav.p_sol_status={bestnav.p_sol_status}(期望0或2)'
            )
            return None

        # 以 heading_type 作为统一质量指标（替代 pos_type）
        # 16=单点, 17=DGPS, 34=浮点解, 50=整数解；其余视为未收敛
        VALID_HEADING_TYPES = (16, 17, 34, 50)
        if heading.heading_type not in VALID_HEADING_TYPES:
            self._rtk_diag_log(
                'position_not_converged',
                f'heading_type={heading.heading_type}(期望16/17/34/50), '
                f'pos_type={bestnav.pos_type}, '
                f'可见星={bestnav.svs_num}, 解算星={bestnav.soln_svs_num}'
            )
            return None

        if math.isnan(bestnav.latitude_deg) or math.isnan(bestnav.longitude_deg):
            self.get_logger().debug('RTK 经纬度包含 NaN')
            return None

        navsat = NavSatFix()
        # 优先使用 bestnav 自带时间戳，否则退到 RTK 消息头
        navsat.header = bestnav.header if bestnav.header.stamp.sec else msg.header
        navsat.header.frame_id = 'gps'
        navsat.latitude = bestnav.latitude_deg
        navsat.longitude = bestnav.longitude_deg
        navsat.altitude = bestnav.altitude_m if not math.isnan(bestnav.altitude_m) else 0.0

        # heading_type → NavSatFix status（统一质量指标）
        if heading.heading_type == 50:
            navsat.status.status = RTK_FIX_INDICATOR
        elif heading.heading_type == 34:
            navsat.status.status = RTK_FLOAT_INDICATOR
        else:  # 17=DGPS, 16=单点
            navsat.status.status = STATUS_GBAS_FIX

        navsat.position_covariance_type = 1  # COVARIANCE_TYPE_KNOWN
        lat_var = bestnav.lat_std ** 2 if not math.isnan(bestnav.lat_std) else 1.0
        lon_var = bestnav.lon_std ** 2 if not math.isnan(bestnav.lon_std) else 1.0
        hgt_var = bestnav.hgt_std ** 2 if not math.isnan(bestnav.hgt_std) else 4.0
        navsat.position_covariance = [
            lat_var, 0.0, 0.0,
            0.0, lon_var, 0.0,
            0.0, 0.0, hgt_var,
        ]
        return navsat

    def _on_param_change(self, params):
        """运行时参数变化回调（支持远程切换数据源和RTK源）"""
        for p in params:
            if p.name == 'gps_source':
                old = self.gps_source
                self.gps_source = p.value
                self.get_logger().info(
                    f'GPS数据源切换: {old} → {self.gps_source}')
            elif p.name == 'rtk_source':
                old = self._rtk_source
                self._rtk_source = p.value
                self.get_logger().info(
                    f'RTK数据源切换: {old} → {self._rtk_source}')
        return SetParametersResult(successful=True)

    def _detect_rtk_mode(self, gps_msg):
        """检测RTK模式：RTK_FIX=4, RTK_FLOAT=5"""
        if gps_msg.status.status >= RTK_FIX_INDICATOR:
            return True
        if gps_msg.position_covariance_type > 0:
            h_var = math.sqrt(gps_msg.position_covariance[0] + gps_msg.position_covariance[4])
            if h_var < 0.1:
                return True
        return False

    def _get_rtk_status_str(self, gps_msg):
        """返回RTK状态描述字符串"""
        status = gps_msg.status.status
        if status >= RTK_FIX_INDICATOR:
            return 'RTK_FIX' if status >= RTK_FIX_INDICATOR and (
                status < RTK_FLOAT_INDICATOR or status == RTK_FIX_INDICATOR
            ) else 'RTK_FLOAT'
        if gps_msg.position_covariance_type > 0:
            h_var = math.sqrt(gps_msg.position_covariance[0] + gps_msg.position_covariance[4])
            if h_var < 0.02:
                return 'RTK_FIX(推断)'
            elif h_var < 0.1:
                return 'RTK_FLOAT(推断)'
        return 'GPS'

    def is_valid_gps_data(self, gps_msg):
        """检查GPS数据是否有效（RTK/DGPS自适应用门槛）"""

        if gps_msg.status.status < self.status_threshold:
            self.get_logger().debug(f'GPS状态无效: {gps_msg.status.status}')
            return False

        if math.isnan(gps_msg.latitude) or math.isnan(gps_msg.longitude):
            self.get_logger().debug('GPS经纬度包含NaN值')
            return False

        if not (-90 <= gps_msg.latitude <= 90) or not (-180 <= gps_msg.longitude <= 180):
            self.get_logger().debug(f'GPS经纬度超出范围: lat={gps_msg.latitude}, lon={gps_msg.longitude}')
            return False

        if gps_msg.position_covariance_type > 0:
            h_accuracy = math.sqrt(gps_msg.position_covariance[0] + gps_msg.position_covariance[4])
            is_rtk = self._detect_rtk_mode(gps_msg)
            # 三级精度门槛：RTK(2cm) < DGPS(6m) < 普通GPS(0.1m)
            if is_rtk:
                threshold = self.rtk_min_accuracy
            elif gps_msg.status.status == STATUS_GBAS_FIX:
                threshold = self.dgps_min_accuracy
            else:
                threshold = self.min_accuracy
            if h_accuracy > threshold:
                self.get_logger().warn(
                    'GPS精度过低: 水平误差=%.3fm > 门槛=%.3fm (status=%d)'
                    % (h_accuracy, threshold, gps_msg.status.status)
                )
                return False

        return True

    def calculate_hdop(self, gps_msg):
        """计算水平精度因子（HDOP）"""
        if gps_msg.position_covariance_type > 0:
            h_accuracy = math.sqrt(gps_msg.position_covariance[0] + gps_msg.position_covariance[4])
            return h_accuracy
        return float('inf')

    def gps_callback(self, msg):
        """处理原始GPS数据"""

        self.get_logger().debug(f'纬度: {msg.latitude}, 经度: {msg.longitude}')

        is_valid = self.is_valid_gps_data(msg)

        if is_valid:
            hdop = self.calculate_hdop(msg)
            is_rtk = self._detect_rtk_mode(msg)

            if is_rtk and not self.rtk_mode:
                self.get_logger().info(
                    f'进入RTK模式: {self._get_rtk_status_str(msg)}, h_accuracy={hdop:.3f}m'
                )
                self.rtk_mode = True
            elif not is_rtk and self.rtk_mode:
                self.get_logger().info(f'退出RTK模式, h_accuracy={hdop:.3f}m')
                self.rtk_mode = False

            self.gps_quality_counter += 1
            if self.gps_quality_counter > 5:
                self.gps_available = True

            self.last_valid_gps = msg

            # 发布处理后的GPS数据
            filtered_msg = NavSatFix()
            filtered_msg.header = msg.header
            filtered_msg.status = msg.status
            filtered_msg.latitude = msg.latitude
            filtered_msg.longitude = msg.longitude
            filtered_msg.altitude = msg.altitude
            filtered_msg.position_covariance = msg.position_covariance
            filtered_msg.position_covariance_type = msg.position_covariance_type
            filtered_msg.header.frame_id = 'gps'

            self.gps_pub.publish(filtered_msg)

            # 转换为UTM坐标并发布
            try:
                x, y = self.transformer.transform(msg.longitude, msg.latitude)
                z = msg.altitude if not math.isnan(msg.altitude) else 0.0

                # NaN/Inf 防护：UTM 转换结果异常时直接丢弃，避免污染下游
                if math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y):
                    self.get_logger().warn(
                        f'UTM转换产生NaN/Inf: lat={msg.latitude:.8f}, lon={msg.longitude:.8f}, '
                        f'utm_x={x}, utm_y={y} — 已丢弃'
                    )
                    return

                # 发布原始绝对UTM坐标
                odom = Odometry()
                odom.header = msg.header
                odom.header.frame_id = 'utm'
                odom.pose.pose.position.x = x
                odom.pose.pose.position.y = y
                odom.pose.pose.position.z = z
                self.utm_pub.publish(odom)
                self.get_logger().debug(f'发布UTM坐标: x={x:.2f}, y={y:.2f}')

                if self.utm_origin is None:
                    self.utm_origin = (x, y, z)
                    self.get_logger().info(f'设定GPS原点: UTM({x:.2f}, {y:.2f}, {z:.2f})')

                # 发布相对UTM坐标
                rel_odom = Odometry()
                rel_odom.header = msg.header
                rel_odom.header.frame_id = 'odom'
                rel_odom.child_frame_id = 'base_link'
                rel_odom.pose.pose.position.x = x - self.utm_origin[0]
                rel_odom.pose.pose.position.y = y - self.utm_origin[1]
                rel_odom.pose.pose.position.z = z - self.utm_origin[2]
                rel_odom.pose.pose.orientation.w = 1.0
                var_xy = 0.01 if self.rtk_mode else 1.0
                rel_odom.pose.covariance[0] = var_xy
                rel_odom.pose.covariance[7] = var_xy
                rel_odom.pose.covariance[14] = 1.0
                self.odom_pub.publish(rel_odom)
            except Exception as e:
                self.get_logger().error(f'UTM转换失败: {e}')

            self.get_logger().debug(
                '发布有效GPS: lat=%.6f, lon=%.6f, HDOP=%.1f, RTK=%s'
                % (msg.latitude, msg.longitude, hdop, is_rtk)
            )

        else:
            self.gps_quality_counter = max(0, self.gps_quality_counter - 1)
            if self.gps_quality_counter == 0:
                self.gps_available = False
                self.rtk_mode = False

            self.get_logger().debug('GPS数据无效，已过滤')

        # 发布GPS状态
        status_msg = Bool()
        status_msg.data = self.gps_available
        self.gps_status_pub.publish(status_msg)

        # 定期输出状态信息（每10秒）
        now_ns = self.get_clock().now().nanoseconds
        if now_ns % 10_000_000_000 < 100_000_000:
            self.get_logger().info(
                'GPS状态: 可用=%s, 质量计数=%d, RTK模式=%s'
                % (self.gps_available, self.gps_quality_counter, self.rtk_mode)
            )


def main(args=None):
    rclpy.init(args=args)

    preprocessor = None
    try:
        preprocessor = GPSPreprocessor()
        rclpy.spin(preprocessor)
    except KeyboardInterrupt:
        pass
    finally:
        if preprocessor is not None:
            try:
                preprocessor.destroy_node()
            except KeyboardInterrupt:
                pass
        try:
            rclpy.shutdown()
        except (RuntimeError, KeyboardInterrupt):
            pass  # 已被 spin 内部 shutdown


if __name__ == '__main__':
    main()
