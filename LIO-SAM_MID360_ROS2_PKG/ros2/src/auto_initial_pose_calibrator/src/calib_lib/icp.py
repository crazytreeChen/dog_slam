"""帧间 ICP（Iterative Closest Point）匹配 —— 纯函数。

仅依赖 numpy + math + scan_utils，无 ROS 依赖。
与原 AutoInitialPoseCalibrator._icp_match / _icp_align_scans 行为一致。
"""
import math

import numpy as np

from .scan_utils import scan_to_points


def icp_match(points1, points2, max_iter=15, tol_trans=0.01, tol_rot_deg=0.1):
    """简化的点对点 ICP 匹配。

    输入：points1 (N,2) 参考帧点阵, points2 (M,2) 源帧点阵
    输出：(dx, dy, dyaw) 从 points2 到 points1 的变换 (points2 @ R + t -> points1)

    与原 _icp_match 完全一致：
      - 降采样到约 144 点
      - 暴力最近邻对应 + SVD 求解
      - 反射矩阵处理
    """
    if len(points1) < 3 or len(points2) < 3:
        return 0.0, 0.0, 0.0

    # 降采样
    step = max(1, len(points2) // 144)  # 降采样到约144点
    p2 = points2[::step]
    # 用角度有序特性做对应：直接按角度bin对应（两帧角度范围相同）
    # 更鲁棒做法：对 points1 每个点找 points2 最近邻（小数据集直接用暴力）
    p1 = points1

    T = np.eye(3)
    for i in range(max_iter):
        # 将 p2 用当前 T 变换到 p1 系
        p2_h = np.column_stack((p2, np.ones(len(p2))))
        p2_transformed = (T @ p2_h.T).T[:, :2]

        # 找对应点：对 p2_transformed 每个点，在 p1 中找最近邻
        # 暴力搜索（点数量少，可接受）
        dists = np.sum((p1[np.newaxis, :, :] - p2_transformed[:, np.newaxis, :]) ** 2, axis=2)
        idx = np.argmin(dists, axis=1)
        matched_p1 = p1[idx]

        # SVD 求解变换：p2 -> matched_p1
        centroid_p1 = np.mean(matched_p1, axis=0)
        centroid_p2 = np.mean(p2, axis=0)

        p1_centered = matched_p1 - centroid_p1
        p2_centered = p2 - centroid_p2

        H = p2_centered.T @ p1_centered
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            break
        R = Vt.T @ U.T
        # 处理反射矩阵
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_p1 - R @ centroid_p2

        T_new = np.eye(3)
        T_new[:2, :2] = R
        T_new[:2, 2] = t

        delta = np.linalg.norm(T_new - T)
        T = T_new
        if delta < tol_trans:
            break

    dx = T[0, 2]
    dy = T[1, 2]
    dyaw = math.atan2(T[1, 0], T[0, 0])
    return dx, dy, dyaw


def icp_align_scans(scan_list, outlier_cfg=None, logger=None,
                    apply_outlier_filter=False):
    """将多帧 LaserScan 对齐到第一帧参考系，返回累积变换列表。

    输入：scan_list = [scan_msg1, scan_msg2, ...]
    输出：transforms = [(0,0,0), (dx1,dy1,dyaw1), ...]
          每个变换将第 i 帧的点投影到第 0 帧参考系。

    与原 _icp_align_scans 一致（原代码 _scan_to_points 默认不开启 outlier 过滤）。
    """
    if not scan_list:
        return []
    transforms = [(0.0, 0.0, 0.0)]  # 第一帧为单位变换
    prev_points = scan_to_points(scan_list[0], outlier_cfg, logger,
                                 apply_outlier_filter=apply_outlier_filter)
    cum_T = np.eye(3)  # 累积变换：从当前帧到参考帧

    for i in range(1, len(scan_list)):
        curr_points = scan_to_points(scan_list[i], outlier_cfg, logger,
                                     apply_outlier_filter=apply_outlier_filter)
        if len(prev_points) < 3 or len(curr_points) < 3:
            transforms.append(transforms[-1])  # 跳过，使用单位变换
            prev_points = curr_points
            continue

        dx, dy, dyaw = icp_match(prev_points, curr_points)
        # 从 prev 到 curr 的变换
        R = np.array([[math.cos(dyaw), -math.sin(dyaw)],
                      [math.sin(dyaw), math.cos(dyaw)]])
        T_step = np.eye(3)
        T_step[:2, :2] = R
        T_step[:2, 2] = [dx, dy]

        cum_T = cum_T @ np.linalg.inv(T_step)  # 累积：curr -> prev -> ... -> ref
        transforms.append((cum_T[0, 2], cum_T[1, 2], math.atan2(cum_T[1, 0], cum_T[0, 0])))
        prev_points = curr_points

    return transforms
