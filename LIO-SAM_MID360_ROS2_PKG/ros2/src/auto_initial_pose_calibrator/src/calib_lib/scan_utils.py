"""激光扫描 ↔ 点云转换、离群点/FRF 过滤、角度规整等纯函数工具。

所有函数对 ROS 无硬依赖（仅 numpy + 可选 scipy），
scan 对象只需具备 ranges/angle_min/angle_increment/range_min/range_max 属性即可，
因此可用 mock 对象在 Windows 上单元测试。
"""
import math

import numpy as np


# ──────────────────────────────────────────────────────────────
#  角度规整（严格保持原实现，含 while 循环 —— 行为不变铁律）
# ──────────────────────────────────────────────────────────────
def norm_angle(a):
    """将角度规整到 [-pi, pi]。保留原 while 实现，不改为 atan2。"""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def quat_to_yaw(q):
    """四元数 → 偏航角 (z 轴旋转)。"""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


# ──────────────────────────────────────────────────────────────
#  扫描预处理过滤器配置（依赖注入，避免散落的 self.* 参数）
# ──────────────────────────────────────────────────────────────
class ScanFilterConfig:
    """离群点过滤参数（从原节点 self.scan_outlier_* 迁移）。"""

    def __init__(self, enabled=True, radius=0.10, min_neighbors=3):
        self.enabled = enabled
        self.radius = radius
        self.min_neighbors = min_neighbors


# ──────────────────────────────────────────────────────────────
#  扫描 → 点云
# ──────────────────────────────────────────────────────────────
def scan_to_points(scan, outlier_cfg=None, logger=None,
                   apply_outlier_filter=False, apply_frf_filter=False):
    """将 LaserScan 转换为 (N, 2) 点阵 (x, y in scan frame)。

    apply_outlier_filter: 是否应用半径离群点过滤去除动态障碍物/杂点
    apply_frf_filter:     是否应用 FRF 角度bin间隙过滤去除混合像素/遮挡伪影

    与原 AutoInitialPoseCalibrator._scan_to_points 行为一致：
      - FRF 过滤优先于范围裁剪，在原始 ranges 上操作
      - 仅当 self.scan_outlier_filter 为真且点数>10 时过滤
    """
    if scan is None:
        return np.empty((0, 2))
    ranges = np.array(scan.ranges, dtype=np.float64)
    angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment

    # FRF 过滤 (优先于范围裁剪, 在原始 ranges 上操作)
    if apply_frf_filter:
        keep_mask = frf_filter_frame(ranges, scan.angle_min, scan.angle_increment)
        ranges = np.where(keep_mask, ranges, 0.0)

    valid = (ranges > scan.range_min) & (ranges < scan.range_max)
    if not np.any(valid):
        return np.empty((0, 2))
    r_valid = ranges[valid]
    a_valid = angles[valid]
    points = np.column_stack((r_valid * np.cos(a_valid), r_valid * np.sin(a_valid)))

    if (apply_outlier_filter and outlier_cfg is not None
            and outlier_cfg.enabled and len(points) > 10):
        points = filter_scan_outliers(points, outlier_cfg, logger)
    return points


def filter_scan_outliers(points, outlier_cfg, logger=None):
    """半径离群点过滤：去除邻域内邻居不足的孤立点（动态障碍物、杂点）。

    输入: points (N,2)
    输出: 过滤后的 (M,2)
    与原 _filter_scan_outliers 行为一致（含 scipy fallback）。
    """
    if len(points) < outlier_cfg.min_neighbors + 1:
        return points  # 点数太少，不过滤

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # 查询每个点在 radius 内的邻居数
        counts = tree.query_ball_point(points, outlier_cfg.radius, return_length=True)
        mask = np.array(counts) >= outlier_cfg.min_neighbors
        n_removed = np.sum(~mask)
        if n_removed > 0 and logger is not None:
            logger.debug(f'[离群点过滤] 移除 {n_removed}/{len(points)} 个离群点 '
                         f'(半径={outlier_cfg.radius}m, 最少邻居={outlier_cfg.min_neighbors})')
        return points[mask]
    except ImportError:
        # scipy 不可用，回退到简单距离阈值过滤
        if logger is not None:
            logger.warn('[离群点过滤] scipy 未安装，使用简化过滤')
        # 简化版：去除距离超过中位数3倍标准差的点
        dists = np.sqrt(np.sum(points ** 2, axis=1))
        median_dist = np.median(dists)
        med_abs_dev = np.median(np.abs(dists - median_dist))
        if med_abs_dev > 0:
            mask = np.abs(dists - median_dist) < 3.0 * med_abs_dev * 1.4826
            return points[mask]
        return points


def frf_filter_frame(ranges, angle_min, angle_inc, bin_deg=2.0, gap_thresh=0.3):
    """FRF (Fast Ray Filter): 逐角度bin过滤混合像素/遮挡伪影。

    在每个 bin_deg° bin 内按距离排序, 剔除间隙>gap_thresh 后的异常束。
    返回: (N,) bool 数组, True=保留。
    与原 _frf_filter_frame 行为一致。
    """
    bin_size = np.radians(bin_deg)
    valid = (ranges > 0.15) & (ranges < 50.0)
    if not np.any(valid):
        return valid
    angles = angle_min + np.arange(len(ranges)) * angle_inc
    bins = np.round(angles / bin_size).astype(int)
    keep = np.ones(len(ranges), dtype=bool)
    for b in np.unique(bins[valid]):
        idx = np.where((bins == b) & valid)[0]
        if len(idx) < 2:
            continue
        sorted_idx = idx[np.argsort(ranges[idx])]
        gaps = np.diff(ranges[sorted_idx]) > gap_thresh
        if np.any(gaps):
            keep[sorted_idx[int(np.argmax(gaps)) + 1:]] = False
    return valid & keep
