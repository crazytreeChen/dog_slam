"""主动运动选择与避障（control）—— 有状态类。

封装原 AutoInitialPoseCalibrator 中与主动运动相关的逻辑：
  - _do_active_motion_selection: 信息熵步长 + 8 方向信息增益选优
  - _check_local_direction_safety: scan 扇区避障
  - _get_map_signature: 似然场特征签名
  - _do_control_loop_and_avoidance: P 控制器 + 实时避障

设计：所有方法通过 node 引用读写主节点状态（self.current_scan,
self.current_odom, self.indoor_phase, self.target_odom_pose 等），
保证状态机同步的行为完全不变。
"""
import math

import numpy as np
from geometry_msgs.msg import Twist

from .scan_utils import norm_angle, quat_to_yaw
from . import IndoorPhase


class MotionController:
    """主动运动控制器：方向选择 + 避障 + odom 跟踪。"""

    def __init__(self, logger):
        """
        logger: 日志器
        """
        self._logger = logger

    # ================================================================
    #  运动方向选择（信息增益驱动）
    # ================================================================
    def select_motion(self, node):
        """信息增益驱动的主动运动方向选择。

        node: 主节点引用（读取 candidates, current_scan, current_odom 等；
              写入 target_odom_pose, motion_start_odom, motion_start_time, indoor_phase）。
        与原 _do_active_motion_selection 行为完全一致。
        """
        if not node.candidates:
            self._logger.error('候选 Pose 为空，无法进行主动运动规划，重置。')
            node._reset_indoor()
            return

        if node.current_scan is None:
            self._logger.warn('激光扫描未就位，等待数据...')
            return

        # 动态步长计算（根据候选 Pose 分布的信息熵）
        probs = np.array([c[0] for c in node.candidates])
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        motion_dist = node.base_motion_distance
        if entropy < 1.5:
            motion_dist = node.base_motion_distance * 0.5
            self._logger.info(f'[运动选择] 信息熵偏低 ({entropy:.2f})，采用短步长探索: {motion_dist:.2f}m')
        else:
            self._logger.info(f'[运动选择] 信息熵高 ({entropy:.2f})，采用常规步长探索: {motion_dist:.2f}m')

        # 定义 8 个粗粒度朝向
        test_angles_deg = [0, 45, 90, 135, 180, -135, -90, -45]
        best_dir = None
        max_ig = -1.0

        for deg in test_angles_deg:
            rad = math.radians(deg)

            # A. 局部安全检测
            is_safe = self.check_direction_safety(node, rad, motion_dist)
            if not is_safe:
                continue

            # B. 信息增益估算
            delta_rot = math.radians(45.0) if deg != 180 else 0.0

            signatures = []
            for prob, x, y, yaw in node.candidates:
                pred_x = x + motion_dist * math.cos(yaw + rad)
                pred_y = y + motion_dist * math.sin(yaw + rad)
                pred_yaw = norm_angle(yaw + delta_rot)

                sig = self._get_map_signature(node, pred_x, pred_y, pred_yaw)
                signatures.append(sig)

            sig_matrix = np.array(signatures)
            variance_sum = np.sum(np.var(sig_matrix, axis=0))

            self._logger.debug(f'方向 {deg}°: 局部安全=True, 信息增益 (方差和)={variance_sum:.4f}')

            if variance_sum > max_ig:
                max_ig = variance_sum
                best_dir = (rad, delta_rot, motion_dist)

        if best_dir is not None:
            rad, delta_rot, dist = best_dir
            self._logger.info(
                f'[运动决策] 最优安全探索动作: 相对朝向={math.degrees(rad):.1f}°, '
                f'旋转={math.degrees(delta_rot):.1f}°, 距离={dist:.2f}m, IG={max_ig:.4f}'
            )

            if node.current_odom is not None:
                curr_pos = node.current_odom.pose.pose.position
                curr_yaw = quat_to_yaw(node.current_odom.pose.pose.orientation)

                target_x = curr_pos.x + dist * math.cos(curr_yaw + rad)
                target_y = curr_pos.y + dist * math.sin(curr_yaw + rad)
                target_yaw = norm_angle(curr_yaw + delta_rot)

                node.target_odom_pose = (target_x, target_y, target_yaw)
                node.motion_start_odom = node.current_odom
                node.motion_start_time = node.get_clock().now()
                node.indoor_phase = IndoorPhase.MOVING
            else:
                self._logger.warn('里程计信号中断，无法启动移动。')
        else:
            self._logger.warn('[运动决策] 未能找到安全的平移方向。强制尝试就地旋转 45° 以获取环境信息。')
            if node.current_odom is not None:
                curr_pos = node.current_odom.pose.pose.position
                curr_yaw = quat_to_yaw(node.current_odom.pose.pose.orientation)
                node.target_odom_pose = (curr_pos.x, curr_pos.y, norm_angle(curr_yaw + math.radians(45.0)))
                node.motion_start_odom = node.current_odom
                node.motion_start_time = node.get_clock().now()
                node.indoor_phase = IndoorPhase.MOVING
            else:
                node._reset_indoor()

    def check_direction_safety(self, node, local_angle, dist):
        """基于当前 /scan，检查局部坐标系方向是否安全。

        与原 _check_local_direction_safety 行为一致。
        """
        if node.current_scan is None:
            return False

        ranges = node.current_scan.ranges
        num_beams = len(ranges)

        angle_sector = math.radians(22.5)
        for i in range(num_beams):
            r = ranges[i]
            if not (node.current_scan.range_min < r < node.current_scan.range_max):
                continue

            beam_angle = node.current_scan.angle_min + i * node.current_scan.angle_increment
            angle_diff = abs(norm_angle(beam_angle - local_angle))

            if angle_diff <= angle_sector:
                if r < (dist + node.min_safe_distance):
                    return False
        return True

    def _get_map_signature(self, node, x, y, yaw):
        """以指定位姿为中心，以 2 个不同距离、8 个径向方向读取似然场数值。

        与原 _get_map_signature 行为一致。
        """
        sig = []
        dists = [node.ig_sample_dist_1, node.ig_sample_dist_2]
        angles = [0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0]

        for d in dists:
            for ang_deg in angles:
                rad = math.radians(ang_deg)
                px = x + d * math.cos(yaw + rad)
                py = y + d * math.sin(yaw + rad)

                if node.map_info is not None and node.likelihood_field is not None:
                    col = int((px - node.map_info.origin.position.x) / node.map_info.resolution)
                    row = int(node.map_info.height - 1 - (py - node.map_info.origin.position.y) / node.map_info.resolution)

                    if 0 <= row < node.map_info.height and 0 <= col < node.map_info.width:
                        sig.append(float(node.likelihood_field[row, col]))
                    else:
                        sig.append(float(node.likelihood_max_dist))
                else:
                    sig.append(float(node.likelihood_max_dist))
        return sig

    # ================================================================
    #  控制器环路与实时激光避障
    # ================================================================
    def control_loop(self, node):
        """P 控制器速度输出 + 实时避障。

        node: 主节点引用（读写 target_odom_pose, current_odom, current_scan,
              indoor_phase, scan_buffer, submap_ready, cmd_vel_pub）。
        与原 _do_control_loop_and_avoidance 行为完全一致。
        """
        if node.target_odom_pose is None or node.current_odom is None:
            node.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
            return

        # 1. 超时限制（防卡死，12秒）
        time_elapsed = (node.get_clock().now() - node.motion_start_time).nanoseconds / 1e9
        if time_elapsed > 12.0:
            self._logger.warn('[主动控制] 移动超时限制，停止并开始下一步匹配。')
            node.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
            node.scan_buffer.clear()
            node.submap_ready = False
            return

        # 2. 闭环误差计算 (局部坐标系)
        tx, ty, tyaw = node.target_odom_pose
        curr_pos = node.current_odom.pose.pose.position
        curr_yaw = quat_to_yaw(node.current_odom.pose.pose.orientation)

        dx = tx - curr_pos.x
        dy = ty - curr_pos.y
        dyaw = norm_angle(tyaw - curr_yaw)

        err_x = dx * math.cos(curr_yaw) + dy * math.sin(curr_yaw)
        err_y = -dx * math.sin(curr_yaw) + dy * math.cos(curr_yaw)

        dist_err = math.sqrt(err_x * err_x + err_y * err_y)

        # 3. 终点判定
        if dist_err < 0.05 and abs(dyaw) < math.radians(5.0):
            self._logger.info('[主动控制] 已精准抵达目标运动位姿。')
            node.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
            node.scan_buffer.clear()
            node.submap_ready = False
            return

        # 4. 实时避障检测
        move_dir = math.atan2(err_y, err_x)
        if node.current_scan is not None:
            ranges = node.current_scan.ranges
            num_beams = len(ranges)

            avoid_sector = math.radians(30.0)
            for i in range(num_beams):
                r = ranges[i]
                if not (node.current_scan.range_min < r < node.current_scan.range_max):
                    continue
                beam_angle = node.current_scan.angle_min + i * node.current_scan.angle_increment
                diff = abs(norm_angle(beam_angle - move_dir))

                if diff <= avoid_sector:
                    if r < node.min_safe_distance:
                        self._logger.warn(f'[避障停机] 前方检测到障碍物距离过近 ({r:.2f}m)！紧急触发主动停止并进行子图构建。')
                        node.cmd_vel_pub.publish(Twist())
                        node.indoor_phase = IndoorPhase.COLLECTING_SUBMAP2
                        node.scan_buffer.clear()
                        node.submap_ready = False
                        return

        # 5. P 控制器速度输出计算
        vx = node.kp_linear * err_x
        vy = node.kp_linear * err_y
        wz = node.kp_angular * dyaw

        vx = float(np.clip(vx, -node.max_linear_vel, node.max_linear_vel))
        vy = float(np.clip(vy, -node.max_linear_vel, node.max_linear_vel))
        wz = float(np.clip(wz, -node.max_angular_vel, node.max_angular_vel))

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.angular.z = wz
        node.cmd_vel_pub.publish(cmd)
