"""评分引擎 —— 纯函数。

包含：似然场构建、点云/墙壁/光线投射评分、综合置信度、两级局部搜索。
仅依赖 numpy + math + 可选 cv2，无 ROS 依赖（map_info 用 duck-typed 对象）。
与原 AutoInitialPoseCalibrator 对应方法行为一致。

注意：本模块为纯重构，已知历史问题（如 _score_points 的重复 ri 赋值死代码、
_local_search_two_stage 无随机种子导致不可复现）原样保留，已记入 UNFINISHED_TASKS.md。
"""
import math

import numpy as np


class MapContext:
    """地图上下文（封装原节点 self.map_data / map_info / likelihood_field）。

    map_info 只需具备以下属性（duck-typing，ROS OccupancyGrid.info 兼容）：
      resolution, origin.position.x, origin.position.y, height, width
    """

    def __init__(self, map_data=None, map_info=None, likelihood_field=None,
                 likelihood_max_dist=2.0):
        self.map_data = map_data          # (H, W) int8 numpy, ROS 占用值 (-1/0/100)
        self.map_info = map_info
        self.likelihood_field = likelihood_field  # (H, W) float32 距离场
        self.likelihood_max_dist = likelihood_max_dist

    def is_valid(self):
        return self.map_data is not None and self.map_info is not None


def build_likelihood(map_ctx, logger=None):
    """构建似然场（距离变换 + 未知区惩罚）。

    与原 _build_likelihood 一致：写入 map_ctx.likelihood_field。
    依赖可选的 cv2；失败时记录错误。
    """
    if map_ctx.map_data is None:
        return
    try:
        import cv2
        obs = (map_ctx.map_data == 100).astype(np.uint8)
        max_px = map_ctx.likelihood_max_dist / map_ctx.map_info.resolution
        dist = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        map_ctx.likelihood_field = np.clip(dist, 0, max_px).astype(np.float32) \
                                   * map_ctx.map_info.resolution
        # ── 未知区域惩罚: 阻止优化器把扫描藏进灰色区域 ──
        map_ctx.likelihood_field[map_ctx.map_data == -1] = map_ctx.likelihood_max_dist
    except Exception as e:
        if logger is not None:
            logger.error(f'似然场构建失败: {e}')


def find_free_space(map_ctx):
    """统计自由空间像素索引（与原 _find_free_space 一致）。"""
    if map_ctx.map_data is None:
        return np.empty((0, 2), dtype=np.int64)
    free = (map_ctx.map_data == 0)
    rows, cols = np.where(free)
    return np.stack([rows, cols], axis=1)


def score_points(points_odom, cx, cy, yaw, map_ctx):
    """向量化点云评分: O(N) with numpy。

    返回: (score, hit_rate, n_valid)
    与原 _score_points 完全一致（含重复 ri 赋值死代码，原样保留）。
    """
    if map_ctx.likelihood_field is None or map_ctx.map_info is None:
        return -1e9, 0, 0
    res = map_ctx.map_info.resolution
    ox = map_ctx.map_info.origin.position.x
    oy = map_ctx.map_info.origin.position.y
    H, W = map_ctx.map_info.height, map_ctx.map_info.width

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    # 变换到地图坐标系
    if points_odom.ndim == 2:
        mx = c_y * points_odom[:, 0] - s_y * points_odom[:, 1] + cx
        my = s_y * points_odom[:, 0] + c_y * points_odom[:, 1] + cy
    else:
        mx = np.array([c_y * points_odom[0] - s_y * points_odom[1] + cx])
        my = np.array([s_y * points_odom[0] + c_y * points_odom[1] + cy])

    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    # ROS栅格: ri 越大 → y 越小, 需要翻转
    # ROS栅格: row 0 = y=0 (底部), 直接除res即可
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    # ROS栅格: ri 越大 → y 越大, 无需翻转 (与离线NPZ一致)

    valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    nv = int(np.sum(valid))
    if nv < max(len(points_odom) * 0.10, 5):
        return -1e9, 0, nv

    dists = map_ctx.likelihood_field[ri[valid], ci[valid]]
    # Gaussian kernel 评分
    lf_score = float(np.mean(np.exp(-dists ** 2 / 0.045)))
    n_hit = int(np.sum(dists < 0.15))
    hit_rate = n_hit / nv
    return lf_score + hit_rate * 0.5, hit_rate, nv


def score_pose_wallhit(points_odom, cx, cy, yaw, map_ctx):
    """墙壁命中评分: 只统计真正落在墙壁像素(100)上的点。

    比似然场更严格 — 不给"靠近墙壁"的点分数, 只在确实命中时计分。
    用于障碍物遮挡场景下区分相似走廊。
    返回: (score, n_wall, n_valid) 或 (-1e9,0,0) 表示无效。
    与原 _score_pose_wallhit 一致。
    """
    if map_ctx.map_data is None or map_ctx.map_info is None:
        return -1e9, 0, 0
    res = map_ctx.map_info.resolution
    ox = map_ctx.map_info.origin.position.x
    oy = map_ctx.map_info.origin.position.y
    H, W = map_ctx.map_info.height, map_ctx.map_info.width

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y * points_odom[:, 0] - s_y * points_odom[:, 1] + cx
    my = s_y * points_odom[:, 0] + c_y * points_odom[:, 1] + cy
    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    v = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    nv = int(np.sum(v))
    if nv < max(len(points_odom) * 0.1, 3):
        return -1e9, 0, 0
    cells = map_ctx.map_data[ri[v], ci[v]]
    valid_c = (cells != -1)
    n_v = int(np.sum(valid_c))
    if n_v < 5:
        return -1e9, 0, 0
    n_wall = int(np.sum(cells[valid_c] == 100))
    n_free = int(np.sum(cells[valid_c] == 0))
    # 扫描不能大部分落在空地上
    if n_free / max(n_v, 1) > 0.60:
        return -1e9, 0, 0
    hit_rate = n_wall / max(n_wall + n_free, 1)
    coverage = n_v / len(points_odom)
    return hit_rate * coverage * 2.0, n_wall, n_v


def score_pose_raycast(points_odom, cx, cy, yaw, map_ctx, range_tol=1.0, max_beams=200):
    """光线投射评分: 只对"实测距离 ≈ 地图预期距离"的束评分。

    自动过滤被障碍物/家具遮挡的异常束。
    原理: 对每个扫描点沿射线方向步进, 检查是否在预期距离处碰到墙壁。
    返回: (score, n_valid_beams, n_evaluated)。
    与原 _score_pose_raycast 一致（Python 逐束逐像素循环）。
    """
    if map_ctx.map_data is None or map_ctx.map_info is None:
        return -1e9, 0, 0
    res = map_ctx.map_info.resolution
    ox = map_ctx.map_info.origin.position.x
    oy = map_ctx.map_info.origin.position.y
    H, W = map_ctx.map_info.height, map_ctx.map_info.width

    n_pts = len(points_odom)
    beam_step = max(1, n_pts // max_beams)
    n_valid = 0
    total_score = 0.0

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    for i in range(0, n_pts, beam_step):
        px, py = points_odom[i, 0], points_odom[i, 1]
        dist_measured = math.sqrt(px * px + py * py)
        if dist_measured < 0.1:
            continue

        # 扫描点在 map 中的位置
        mx = c_y * px - s_y * py + cx
        my = s_y * px + c_y * py + cy
        ray_angle = math.atan2(my - cy, mx - cx)

        # 沿射线方向步进 (半像素步进)
        dx_r = math.cos(ray_angle) * res * 0.5
        dy_r = math.sin(ray_angle) * res * 0.5
        rx, ry = cx, cy
        hit_wall = False
        dist_expected = 50.0

        for _ in range(int(50.0 / (res * 0.5))):
            col = int((rx - ox) / res)
            row = int((ry - oy) / res)
            if col < 0 or col >= W or row < 0 or row >= H:
                break
            cell = map_ctx.map_data[row, col]
            if cell == 100:
                dist_expected = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
                hit_wall = True
                break
            elif cell == -1:
                dist_expected = -1
                break
            rx += dx_r
            ry += dy_r

        if hit_wall and abs(dist_measured - dist_expected) < range_tol:
            n_valid += 1
            total_score += 1.0

    if n_valid < 5:
        return -1e9, 0, n_pts // max(1, beam_step)

    n_evaluated = n_pts // max(1, beam_step)
    return total_score / max(n_evaluated, 1), n_valid, n_evaluated


def compute_wall_coverage(points_odom, cx, cy, yaw, map_ctx):
    """计算有效区域内的墙壁覆盖率 (排除灰色区域)。

    返回: (wall_ratio, coverage, free_ratio)
    与原 _compute_wall_coverage 一致。
    """
    if map_ctx.map_data is None:
        return 0.0, 0.0, 0.0
    res = map_ctx.map_info.resolution
    ox = map_ctx.map_info.origin.position.x
    oy = map_ctx.map_info.origin.position.y
    H, W = map_ctx.map_info.height, map_ctx.map_info.width

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y * points_odom[:, 0] - s_y * points_odom[:, 1] + cx
    my = s_y * points_odom[:, 0] + c_y * points_odom[:, 1] + cy
    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    cells = map_ctx.map_data[ri[valid], ci[valid]]
    valid_cell = (cells != -1)
    n_valid = int(np.sum(valid_cell))
    n_total = len(points_odom)
    if n_valid < 5:
        return 0.0, float(n_valid) / n_total, 0.0
    w = int(np.sum(cells[valid_cell] == 100))
    f = int(np.sum(cells[valid_cell] == 0))
    return float(w) / (w + f), float(n_valid) / n_total, float(f) / (w + f)


# ──────────────────────────────────────────────────────────────
#  综合置信度计算
# ──────────────────────────────────────────────────────────────
def compute_confidence(wall_ratio, lf_score, coverage,
                       sigma_pos, sigma_yaw_deg, buf_frames,
                       best_prob=0.0, second_prob=0.0):
    """综合置信度计算 (0.0 ~ 1.0)。

    加权方案:
      - wall_ratio:  30% — 墙壁命中率，最重要的约束强度指标
      - lf_score:    25% — 似然场匹配质量
      - coverage:    15% — 有效区域覆盖率
      - sigma_penalty: 20% — 不确定性惩罚 (位置+角度，越小越高)
      - buffer_bonus: 10% — 数据量奖励

    额外惩罚:
      - 多峰惩罚: 主/次候选概率比 < 2.0 → ×0.8
      - 过拟合检测: lf_score > 2.5 且 coverage < 0.5 → ×0.7 (罕见高分布空洞模式)

    与原 _compute_confidence 完全一致。
    """
    import numpy as np

    # 1. wall_ratio 归一化 (0% → 0.0,  60% → 1.0, clipping)
    w = np.clip(wall_ratio / 0.60, 0.0, 1.0)

    # 2. lf_score 归一化 (0.0 → 0.0,  1.5 → 1.0)
    l = np.clip(lf_score / 1.5, 0.0, 1.0)

    # 3. coverage 归一化 (30% → 0.0,  90% → 1.0)
    c = np.clip((coverage - 0.30) / 0.60, 0.0, 1.0)

    # 4. sigma 惩罚 (位置: 0.3m → 1.0,  3.0m → 0.0)
    s = max(1.0 - max(sigma_pos - 0.3, 0.0) / 2.7, 0.0)
    #    角度: 5° → 1.0,  45° → 0.0
    sy = max(1.0 - max(sigma_yaw_deg - 5.0, 0.0) / 40.0, 0.0)
    sigma_score = 0.6 * s + 0.4 * sy

    # 5. buffer 奖励 (10帧 → 0.0,  60帧 → 1.0)
    b = np.clip((buf_frames - 10) / 50.0, 0.0, 1.0)

    # 加权综合
    confidence = 0.30 * w + 0.25 * l + 0.15 * c + 0.20 * sigma_score + 0.10 * b

    # 多峰惩罚: 主/次候选概率比 < 2.0 → 存在歧义
    if best_prob > 0 and second_prob > 0 and (best_prob / max(second_prob, 1e-9)) < 2.0:
        confidence *= 0.8

    # 过拟合检测: 似然高但覆盖率低 → 可能匹配到了局部特征
    if lf_score > 2.5 and coverage < 0.5:
        confidence *= 0.7

    return float(np.clip(confidence, 0.0, 1.0))


# ──────────────────────────────────────────────────────────────
#  两级局部搜索
# ──────────────────────────────────────────────────────────────
def local_search_two_stage(points_odom, cx_pred, cy_pred, yaw_pred, map_ctx,
                           radius=3.0, pos_step=0.2, angle_range=15, angle_step=2):
    """两级粗→细局部搜索: 大幅减少计算量同时保持精度。

    Stage1: 大步长遍历全半径 → Stage2: 围绕粗搜最优做小窗口精修。
    返回: (best_pose, best_score)
        best_pose: (x, y, yaw) in map frame

    与原 _local_search_two_stage 一致（含无种子 rng，已记入 UNFINISHED_TASKS）。
    """
    # 降采样
    n_pts = min(len(points_odom), 800)
    rng = np.random.default_rng()
    if len(points_odom) > n_pts:
        idx = rng.choice(len(points_odom), size=n_pts, replace=False)
        pts_ds = points_odom[idx]
    else:
        pts_ds = points_odom

    # Stage 1: 粗搜索 (大步长, 定位大致区域)
    coarse_ps = max(pos_step * 2.0, 0.5)
    coarse_as = max(angle_step * 2, 5)
    best_sc = -1e9
    best_pose = (cx_pred, cy_pred, yaw_pred)
    for dx in np.arange(-radius, radius + 1e-5, coarse_ps):
        for dy in np.arange(-radius, radius + 1e-5, coarse_ps):
            for da in np.arange(-int(angle_range), int(angle_range) + 1, int(coarse_as)):
                ax, ay = cx_pred + dx, cy_pred + dy
                ayaw = yaw_pred + math.radians(da)
                sc, _, _ = score_points(pts_ds, ax, ay, ayaw, map_ctx)
                if sc > best_sc:
                    best_sc = sc
                    best_pose = (ax, ay, ayaw)

    # Stage 2: 细搜索 (围绕粗搜最优, 小窗口精修)
    fine_rad = pos_step * 1.5
    cx_r, cy_r, yaw_r = best_pose
    for dx in np.arange(-fine_rad, fine_rad + 1e-5, pos_step):
        for dy in np.arange(-fine_rad, fine_rad + 1e-5, pos_step):
            for da in np.arange(-int(angle_step * 2), int(angle_step * 2) + 1, int(angle_step)):
                ax, ay = cx_r + dx, cy_r + dy
                ayaw = yaw_r + math.radians(da)
                sc, _, _ = score_points(pts_ds, ax, ay, ayaw, map_ctx)
                if sc > best_sc:
                    best_sc = sc
                    best_pose = (ax, ay, ayaw)
    return best_pose, best_sc


def score_scan(scan, x, y, yaw, map_ctx, scan_to_points_fn, max_beams):
    """Beam model 评分 (兼容旧接口, 内部调用 score_points)。

    scan_to_points_fn: 由调用方注入（通常是 scan_utils.scan_to_points，
                       或主节点封装了配置的版本）。
    与原 _score_scan 一致。
    """
    if map_ctx.likelihood_field is None or map_ctx.map_info is None or scan is None:
        return -1e9

    pts = scan_to_points_fn(scan)
    if len(pts) < 10:
        return -1e9
    # 降采样
    ds = max(1, len(pts) // max_beams)
    pts_ds = pts[::ds]
    sc, _, _ = score_points(pts_ds, x, y, yaw, map_ctx)
    return sc
