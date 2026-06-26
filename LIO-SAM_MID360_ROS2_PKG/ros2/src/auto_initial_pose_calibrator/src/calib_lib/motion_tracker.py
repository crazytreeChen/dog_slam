"""运动状态跟踪 —— 纯类，无 ROS 硬依赖。

核心功能：检测"运动→静止"边沿，稳定后触发重算事件。
漂移检测（odom→map 变换）在主节点被动定时器中实现（需要 last_calibrated_pose）。

状态机：
  UNKNOWN → MOVING → SETTLING → STATIONARY
"""
import math
from enum import Enum


class MotionState(Enum):
    UNKNOWN = 0      # 初始状态，数据不足
    MOVING = 1       # 正在运动
    SETTLING = 2     # 刚停止，等待稳定
    STATIONARY = 3   # 稳定静止，可安全计算


class MotionTracker:
    """运动状态跟踪器：从 odom twist 判断运动/静止。"""

    def __init__(self, logger,
                 speed_threshold=0.05,
                 angular_threshold=0.05,
                 settle_time=1.0,
                 stop_cooldown=10.0):
        """
        Args:
            logger: 日志器
            speed_threshold:    线速度阈值 (m/s)，超过判定为运动
            angular_threshold:  角速度阈值 (rad/s)，超过判定为运动
            settle_time:        停止后等待稳定时间 (秒)
            stop_cooldown:      停止事件触发后冷却时间 (秒)，避免频繁触发
        """
        self._logger = logger
        self._speed_threshold = speed_threshold
        self._angular_threshold = angular_threshold
        self._settle_time = settle_time
        self._stop_cooldown = stop_cooldown

        # 状态
        self._state = MotionState.UNKNOWN
        self._stop_start_time = None       # SETTLING 开始时间
        self._stationary_start_time = None # 进入 STATIONARY 的时间
        self._just_stopped = False         # "刚停稳"事件标志（读取后需手动清除）
        self._last_stop_trigger_time = 0.0 # 上次停止事件触发时间（冷却用）

    @property
    def state(self):
        """当前运动状态。"""
        return self._state

    @property
    def just_stopped(self):
        """是否刚从运动变为静止（读取后需调用 clear_just_stopped 清除）。"""
        return self._just_stopped

    @property
    def is_stationary(self):
        """当前是否处于静止状态。"""
        return self._state == MotionState.STATIONARY

    def clear_just_stopped(self):
        """清除 just_stopped 事件标志。"""
        self._just_stopped = False

    def update(self, odom_msg, now_sec):
        """每帧 odom 回调时更新状态。

        仅根据 twist（瞬时速度）判断运动/静止，不做漂移检测。
        漂移检测需要 odom→map 变换，在主节点被动定时器中实现。

        Args:
            odom_msg: nav_msgs/Odometry 消息
            now_sec:  当前时间戳 (秒，float)
        """
        # 提取 twist（瞬时速度）
        tw = odom_msg.twist.twist
        speed = math.sqrt(tw.linear.x ** 2 + tw.linear.y ** 2)
        angular_speed = abs(tw.angular.z)

        is_moving = (speed > self._speed_threshold or
                     angular_speed > self._angular_threshold)

        if is_moving:
            # ── 运动中 ──
            prev_state = self._state
            self._state = MotionState.MOVING
            self._stop_start_time = None
            self._stationary_start_time = None

            if prev_state not in (MotionState.MOVING, MotionState.UNKNOWN):
                self._logger.debug(
                    f'[运动跟踪] 进入运动状态 v={speed:.3f}m/s w={math.degrees(angular_speed):.1f}°/s')

        else:
            # ── 静止中 ──
            if self._state in (MotionState.MOVING, MotionState.UNKNOWN):
                self._state = MotionState.SETTLING
                self._stop_start_time = now_sec
                self._logger.debug(
                    f'[运动跟踪] 运动停止，等待稳定 ({self._settle_time:.1f}s)')

            elif self._state == MotionState.SETTLING:
                elapsed = now_sec - self._stop_start_time
                if elapsed >= self._settle_time:
                    self._state = MotionState.STATIONARY
                    self._stationary_start_time = now_sec

                    since_last_trigger = now_sec - self._last_stop_trigger_time
                    if since_last_trigger > self._stop_cooldown:
                        self._just_stopped = True
                        self._last_stop_trigger_time = now_sec
                        self._logger.info(
                            f'[运动跟踪] ✓ 已稳定静止，触发"停止事件"')
                    else:
                        self._logger.debug(
                            f'[运动跟踪] 已稳定静止（冷却中，不触发事件）')

            # STATIONARY: 保持不变，等待外部检查

    @staticmethod
    def _quat_to_yaw(q):
        """四元数转 yaw 角。"""
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def get_status_text(self):
        """返回可读的状态文本（用于日志/调试）。"""
        txt = f'state={self._state.name}'
        if self._just_stopped:
            txt += ' [JUST_STOPPED]'
        return txt
