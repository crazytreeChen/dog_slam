"""运动状态跟踪与漂移检测 —— 纯类，无 ROS 硬依赖。

两个核心场景：
  1. 运动→静止 边沿检测：机器人持续移动后停下，稳定后触发重算
  2. 静止漂移检测：机器人静止但 odom pose 在漂移（LIO 积累误差）

状态机：
  UNKNOWN → MOVING → SETTLING → STATIONARY
                                    ↓ (odom drift detected)
                                  DRIFTING
"""
import math
from enum import Enum


class MotionState(Enum):
    UNKNOWN = 0      # 初始状态，数据不足
    MOVING = 1       # 正在运动
    SETTLING = 2     # 刚停止，等待稳定
    STATIONARY = 3   # 稳定静止，可安全计算
    DRIFTING = 4     # 静止中但 odom 在漂移


class MotionTracker:
    """运动状态跟踪器：从 odom 数据判断运动/静止/漂移。"""

    def __init__(self, logger,
                 speed_threshold=0.05,
                 angular_threshold=0.05,
                 settle_time=1.0,
                 drift_threshold=0.15,
                 stop_cooldown=10.0):
        """
        Args:
            logger: 日志器
            speed_threshold:    线速度阈值 (m/s)，超过判定为运动
            angular_threshold:  角速度阈值 (rad/s)，超过判定为运动
            settle_time:        停止后等待稳定时间 (秒)
            drift_threshold:    静止时 odom 漂移触发阈值 (m)
            stop_cooldown:      停止事件触发后冷却时间 (秒)，避免频繁触发
        """
        self._logger = logger
        self._speed_threshold = speed_threshold
        self._angular_threshold = angular_threshold
        self._settle_time = settle_time
        self._drift_threshold = drift_threshold
        self._stop_cooldown = stop_cooldown

        # 状态
        self._state = MotionState.UNKNOWN
        self._stop_start_time = None       # SETTLING 开始时间
        self._stationary_odom_ref = None   # 进入 STATIONARY 时的 odom (x, y, yaw)
        self._stationary_start_time = None # 进入 STATIONARY 的时间
        self._just_stopped = False         # "刚停稳"事件标志（读取后需手动清除）
        self._is_drifting = False          # 当前是否在漂移
        self._drift_distance = 0.0         # 漂移距离
        self._last_stop_trigger_time = 0.0 # 上次停止事件触发时间（冷却用）

        # 统计
        self._update_count = 0

    @property
    def state(self):
        """当前运动状态。"""
        return self._state

    @property
    def just_stopped(self):
        """是否刚从运动变为静止（读取后需调用 clear_just_stopped 清除）。"""
        return self._just_stopped

    @property
    def is_drifting(self):
        """静止中是否检测到 odom 漂移。"""
        return self._is_drifting

    @property
    def drift_distance(self):
        """当前漂移距离 (m)。"""
        return self._drift_distance

    @property
    def stationary_duration(self):
        """静止持续时间 (秒)，非静止状态返回 0。
        注意: 返回的是进入 STATIONARY 时记录的 now_sec 值，
        调用方需自行计算 elapsed = current_time - tracker.stationary_duration。
        """
        if self._stationary_start_time is not None and self._state in (
                MotionState.STATIONARY, MotionState.DRIFTING):
            return self._stationary_start_time
        return 0.0

    def clear_just_stopped(self):
        """清除 just_stopped 事件标志。"""
        self._just_stopped = False

    def update(self, odom_msg, now_sec):
        """每帧 odom 回调时更新状态。

        Args:
            odom_msg: nav_msgs/Odometry 消息（需有 pose.pose.position/orientation
                      和 twist.twist.linear/angular）
            now_sec:  当前时间戳 (秒，float)
        """
        self._update_count += 1

        # 提取 twist（瞬时速度）
        tw = odom_msg.twist.twist
        speed = math.sqrt(tw.linear.x ** 2 + tw.linear.y ** 2)
        angular_speed = abs(tw.angular.z)

        # 提取 pose（位姿）
        pos = odom_msg.pose.pose.position
        yaw = self._quat_to_yaw(odom_msg.pose.pose.orientation)

        is_moving = (speed > self._speed_threshold or
                     angular_speed > self._angular_threshold)

        if is_moving:
            # ── 运动中 ──
            prev_state = self._state
            self._state = MotionState.MOVING
            self._stop_start_time = None
            self._stationary_odom_ref = None
            self._stationary_start_time = None
            self._is_drifting = False
            self._drift_distance = 0.0

            # 从非运动状态首次进入运动 → 日志
            if prev_state != MotionState.MOVING and prev_state != MotionState.UNKNOWN:
                self._logger.debug(
                    f'[运动跟踪] 进入运动状态 v={speed:.3f}m/s w={math.degrees(angular_speed):.1f}°/s')

        else:
            # ── 静止中 ──
            if self._state == MotionState.MOVING or self._state == MotionState.UNKNOWN:
                # 刚从运动变为静止 → 进入 SETTLING
                self._state = MotionState.SETTLING
                self._stop_start_time = now_sec
                self._logger.info(
                    f'[运动跟踪] 检测到运动停止，进入稳定等待 ({self._settle_time:.1f}s)...')

            elif self._state == MotionState.SETTLING:
                # 等待稳定
                elapsed = now_sec - self._stop_start_time
                if elapsed >= self._settle_time:
                    self._state = MotionState.STATIONARY
                    self._stationary_odom_ref = (pos.x, pos.y, yaw)
                    self._stationary_start_time = now_sec
                    self._is_drifting = False
                    self._drift_distance = 0.0

                    # 冷却检查: 避免频繁触发 just_stopped
                    since_last_trigger = now_sec - self._last_stop_trigger_time
                    if since_last_trigger > self._stop_cooldown:
                        self._just_stopped = True
                        self._last_stop_trigger_time = now_sec
                        self._logger.info(
                            f'[运动跟踪] ✓ 已稳定静止，触发"停止事件"'
                            f' (冷却剩余={max(0, self._stop_cooldown - since_last_trigger):.0f}s)')
                    else:
                        self._logger.info(
                            f'[运动跟踪] 已稳定静止（冷却中 {since_last_trigger:.0f}s < '
                            f'{self._stop_cooldown:.0f}s，不触发事件）')

            elif self._state in (MotionState.STATIONARY, MotionState.DRIFTING):
                # 持续静止 → 检查漂移
                if self._stationary_odom_ref is not None:
                    ref_x, ref_y, ref_yaw = self._stationary_odom_ref
                    dx = pos.x - ref_x
                    dy = pos.y - ref_y
                    self._drift_distance = math.sqrt(dx * dx + dy * dy)

                    if self._drift_distance > self._drift_threshold:
                        if not self._is_drifting:
                            self._logger.warn(
                                f'[运动跟踪] ⚠ 静止漂移检测: '
                                f'Δ={self._drift_distance:.3f}m > {self._drift_threshold:.2f}m 阈值')
                        self._is_drifting = True
                        self._state = MotionState.DRIFTING
                    else:
                        self._is_drifting = False
                        self._state = MotionState.STATIONARY

    @staticmethod
    def _quat_to_yaw(q):
        """四元数转 yaw 角。"""
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def get_status_text(self):
        """返回可读的状态文本（用于日志/调试）。"""
        txt = f'state={self._state.name}'
        if self._stationary_odom_ref is not None:
            txt += f' drift={self._drift_distance:.3f}m'
        if self._just_stopped:
            txt += ' [JUST_STOPPED]'
        if self._is_drifting:
            txt += ' [DRIFTING]'
        return txt
