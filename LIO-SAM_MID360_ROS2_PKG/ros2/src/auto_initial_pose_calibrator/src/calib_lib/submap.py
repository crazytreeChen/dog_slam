"""子图构建（SubmapBuilder）—— 有状态类。

利用缓存的 scan_buffer，使用 ICP 帧间匹配合成为高精度激光帧。
与原 AutoInitialPoseCalibrator._build_submap 行为一致。

依赖: scan_utils (scan_to_points + norm_angle), icp (icp_match), temporal (temporal_consistency_filter)
ROS 依赖: sensor_msgs.LaserScan (输入输出类型), 发布可视化 scan
"""
import math

import numpy as np
from sensor_msgs.msg import LaserScan

from .scan_utils import scan_to_points, norm_angle
from .icp import icp_match
from .temporal import temporal_consistency_filter


class SubmapBuilder:
    """子图构建器：将多帧 scan 拼合为一个高质量合成 LaserScan。

    构造时注入依赖，运行时通过 build() 方法执行构建。
    """

    def __init__(self, logger, outlier_cfg=None):
        """
        logger:       日志器（rclpy logger 或 logging.Logger）
        outlier_cfg:  ScanFilterConfig 实例（用于 scan_to_points 的 outlier 过滤）
        """
        self._logger = logger
        self._outlier_cfg = outlier_cfg

    def build(self, scan_buffer, use_icp, temporal_merge_enabled,
              temporal_merge_min_frames, temporal_merge_radius,
              submap_scan_pub=None):
        """利用缓存的 scan_buffer 合成高精度激光帧。

        参数:
          scan_buffer:    list of (scan_msg, odom_x, odom_y, odom_yaw) 或
                          list of (scan_msg, unused)
          use_icp:        bool, 是否使用 ICP 帧间匹配（否则用 odom）
          temporal_merge_*: 时序过滤参数
          submap_scan_pub: 可选 ROS publisher，发布可视化 composite scan

        返回:
          LaserScan 或 None（buffer 为空时）
        与原 _build_submap 行为完全一致。
        """
        if not scan_buffer:
            return None

        # 提取扫描消息列表
        scans = [item[0] for item in scan_buffer]
        num_beams = len(scans[0].ranges)

        transforms = self._compute_frame_transforms(
            scan_buffer, scans, use_icp
        )

        # ────── 合并点云到参考帧 ──────
        merged_ranges = self._merge_frames(
            scan_buffer, scans, transforms, num_beams,
            temporal_merge_enabled, temporal_merge_min_frames, temporal_merge_radius
        )

        # 封装成合成 LaserScan
        composite_scan = LaserScan()
        composite_scan.header.stamp = scans[0].header.stamp
        composite_scan.header.frame_id = scans[0].header.frame_id
        composite_scan.angle_min = scans[0].angle_min
        composite_scan.angle_max = scans[0].angle_max
        composite_scan.angle_increment = scans[0].angle_increment
        composite_scan.range_min = scans[0].range_min
        composite_scan.range_max = scans[0].range_max
        composite_scan.ranges = merged_ranges.tolist()

        self._logger.info(
            f'[子图构建] {len(scan_buffer)} 帧合并完成 '
            f'(ICP: {use_icp}, 时序过滤: {temporal_merge_enabled})'
        )

        # 发布可视化
        if submap_scan_pub is not None:
            submap_scan_pub.publish(composite_scan)

        return composite_scan

    def _compute_frame_transforms(self, scan_buffer, scans, use_icp):
        """计算每帧到第 0 帧的累积变换。"""
        if use_icp:
            return self._compute_icp_transforms(scans)
        else:
            return self._compute_odom_transforms(scan_buffer)

    def _compute_icp_transforms(self, scans):
        """ICP 帧间匹配计算累积变换（与原 _build_submap ICP 路径一致）。"""
        self._logger.info(f'[子图构建] 使用 ICP 帧间匹配拼接 {len(scans)} 帧...')
        transforms = [(0.0, 0.0, 0.0)]
        for i in range(1, len(scans)):
            pts_prev = scan_to_points(scans[i - 1], self._outlier_cfg, self._logger,
                                     apply_outlier_filter=True)
            pts_curr = scan_to_points(scans[i], self._outlier_cfg, self._logger,
                                     apply_outlier_filter=True)
            if len(pts_prev) < 3 or len(pts_curr) < 3:
                transforms.append(transforms[-1])
                continue
            dx, dy, dyaw = icp_match(pts_prev, pts_curr)
            prev_dx, prev_dy, prev_dyaw = transforms[-1]
            cum_dx = prev_dx + dx * math.cos(prev_dyaw) - dy * math.sin(prev_dyaw)
            cum_dy = prev_dy + dx * math.sin(prev_dyaw) + dy * math.cos(prev_dyaw)
            cum_dyaw = norm_angle(prev_dyaw + dyaw)
            transforms.append((cum_dx, cum_dy, cum_dyaw))

        final_dx, final_dy, final_dyaw = transforms[-1]
        final_dist = math.hypot(final_dx, final_dy)
        self._logger.info(
            f'[子图构建] ICP 变换计算完成: {len(transforms)} 帧, '
            f'累积位移=({final_dx:.2f},{final_dy:.2f})m '
            f'总行程={final_dist:.2f}m 航向Δ={math.degrees(final_dyaw):.1f}deg'
        )
        return transforms

    def _compute_odom_transforms(self, scan_buffer):
        """odom 帧间变换（备用路径，与原 _build_submap odom 路径一致）。"""
        transforms = []
        for scan, sx, sy, syaw in scan_buffer:
            if transforms:
                prev_rx, prev_ry, prev_ryaw = scan_buffer[len(transforms)][1:]
                dx = sx - prev_rx
                dy = sy - prev_ry
                rel_x = dx * math.cos(prev_ryaw) + dy * math.sin(prev_ryaw)
                rel_y = -dx * math.sin(prev_ryaw) + dy * math.cos(prev_ryaw)
                rel_yaw = norm_angle(syaw - prev_ryaw)
                prev_cum_dx, prev_cum_dy, prev_cum_dyaw = transforms[-1]
                cum_dx = prev_cum_dx + rel_x * math.cos(prev_cum_dyaw) - rel_y * math.sin(prev_cum_dyaw)
                cum_dy = prev_cum_dy + rel_x * math.sin(prev_cum_dyaw) + rel_y * math.cos(prev_cum_dyaw)
                cum_dyaw = norm_angle(prev_cum_dyaw + rel_yaw)
                transforms.append((cum_dx, cum_dy, cum_dyaw))
            else:
                transforms.append((0.0, 0.0, 0.0))
        return transforms

    def _merge_frames(self, scan_buffer, scans, transforms, num_beams,
                      temporal_merge_enabled, temporal_merge_min_frames,
                      temporal_merge_radius):
        """将所有帧投影到参考帧并合并为 ranges 数组。

        含时序过滤分支和传统合并分支，与原 _build_submap 一致。
        """
        # ────── 多帧时序一致性合并（带动态物体过滤）──────
        if temporal_merge_enabled and len(scans) >= temporal_merge_min_frames:
            self._logger.info(
                f'[子图构建] 启用多帧时序一致性过滤 '
                f'(最少帧数={temporal_merge_min_frames}, 半径={temporal_merge_radius}m)...'
            )
            all_projected = []
            all_frame_ids = []

            for i, (scan, _) in enumerate(scan_buffer):
                rel_x, rel_y, rel_yaw = transforms[i]
                for j in range(num_beams):
                    r = scan.ranges[j]
                    if not (scans[0].range_min < r < scans[0].range_max):
                        continue
                    beam_angle = scans[0].angle_min + j * scans[0].angle_increment
                    lx = r * math.cos(beam_angle)
                    ly = r * math.sin(beam_angle)
                    px = rel_x + lx * math.cos(rel_yaw) - ly * math.sin(rel_yaw)
                    py = rel_y + lx * math.sin(rel_yaw) + ly * math.cos(rel_yaw)
                    all_projected.append((px, py))
                    all_frame_ids.append(i)

            if len(all_projected) < 10:
                self._logger.warn('[子图构建] 投影后有效点过少，回退到传统合并')
                return self._traditional_merge(
                    scan_buffer, scans, transforms, num_beams
                )

            proj_pts = np.array(all_projected)
            frame_ids = np.array(all_frame_ids)

            keep_mask = temporal_consistency_filter(
                proj_pts, frame_ids,
                temporal_merge_min_frames, temporal_merge_radius,
                self._logger
            )
            static_pts = proj_pts[keep_mask]

            self._logger.info(
                f'[子图构建] 时序过滤后保留 {len(static_pts)}/{len(proj_pts)} 个静态点 '
                f'({100*len(static_pts)/max(1,len(proj_pts)):.1f}%)'
            )

            merged_ranges = np.full(num_beams, scans[0].range_max, dtype=np.float32)
            for (px, py) in static_pts:
                r_proj = math.sqrt(px * px + py * py)
                theta_proj = math.atan2(py, px)
                bin_idx = int(round((theta_proj - scans[0].angle_min) / scans[0].angle_increment))
                if 0 <= bin_idx < num_beams:
                    if r_proj < merged_ranges[bin_idx]:
                        merged_ranges[bin_idx] = r_proj

            filled_bins = np.sum(merged_ranges < scans[0].range_max)
            self._logger.info(
                f'[子图构建] 合成 Scan: {filled_bins}/{num_beams} 个角度 bin 被填充 '
                f'({100*filled_bins/num_beams:.1f}%)'
            )
            return merged_ranges

        else:
            return self._traditional_merge(
                scan_buffer, scans, transforms, num_beams
            )

    def _traditional_merge(self, scan_buffer, scans, transforms, num_beams):
        """传统合并（无时序过滤），与原 _build_submap else 分支一致。"""
        merged_ranges = np.full(num_beams, scans[0].range_max, dtype=np.float32)

        for i, (scan, _) in enumerate(scan_buffer):
            rel_x, rel_y, rel_yaw = transforms[i]
            for j in range(num_beams):
                r = scan.ranges[j]
                if not (scans[0].range_min < r < scans[0].range_max):
                    continue
                beam_angle = scans[0].angle_min + j * scans[0].angle_increment
                lx = r * math.cos(beam_angle)
                ly = r * math.sin(beam_angle)
                px = rel_x + lx * math.cos(rel_yaw) - ly * math.sin(rel_yaw)
                py = rel_y + lx * math.sin(rel_yaw) + ly * math.cos(rel_yaw)
                r_proj = math.sqrt(px * px + py * py)
                theta_proj = math.atan2(py, px)
                bin_idx = int(round((theta_proj - scans[0].angle_min) / scans[0].angle_increment))
                if 0 <= bin_idx < num_beams:
                    if r_proj < merged_ranges[bin_idx]:
                        merged_ranges[bin_idx] = r_proj

        return merged_ranges
