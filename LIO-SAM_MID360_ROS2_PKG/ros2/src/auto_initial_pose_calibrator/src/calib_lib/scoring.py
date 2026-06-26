"""评分引擎 —— 纯函数。

包含：似然场构建、点云/墙壁/光线投射评分、综合置信度、两级局部搜索、
双模板光线投射全局匹配 (global_match2)。
仅依赖 numpy + math + 可选 cv2，无 ROS 依赖（map_info 用 duck-typed 对象）。
与原 AutoInitialPoseCalibrator 对应方法行为一致。

注意：本模块为纯重构，已知历史问题（如 _score_points 的重复 ri 赋值死代码、
_local_search_two_stage 无随机种子导致不可复现）原样保留，已记入 UNFINISHED_TASKS.md。
"""
import math
import time

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
        self.dist_field = None            # (H, W) float64, 原始距离场 (用于距离场匹配)
        # ── 双模板匹配地图 (global_match2 核心) ──
        self.dt_likelihood_map = None     # (H, W) float32, 似然图 (奖励命中墙壁)
        self.dt_ray_penalty_map = None    # (H, W) float32, 射线惩罚图 (墙壁=1.0, 未知=2.0)
        self.dt_valid_center_mask = None  # (H, W) bool, 合法机器狗中心区域

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


# ──────────────────────────────────────────────────────────────
#  距离场匹配: 原始距离场构建 + 距离场评分 (来自 global_match.py)
# ──────────────────────────────────────────────────────────────
def build_distance_field(map_ctx, logger=None):
    """构建原始距离场（不裁剪），用于距离场全局匹配。

    与 build_likelihood 的区别:
      - 二值化阈值: map_data > 50 (而非 == 100), 更宽松
      - 不裁剪距离上限, 保留完整距离信息
      - 存储为 float64 像素距离

    要求 cv2 可用。
    """
    if map_ctx.map_data is None:
        return
    try:
        import cv2
        # 二值化: > 50 视为障碍物 (ROS 栅格图中 > 50 为 likely occupied)
        binary = np.zeros_like(map_ctx.map_data, dtype=np.uint8)
        binary[map_ctx.map_data > 50] = 255
        # 距离变换: 离墙壁越近值越小
        map_ctx.dist_field = cv2.distanceTransform(
            255 - binary, cv2.DIST_L2, 5).astype(np.float64)
    except Exception as e:
        if logger is not None:
            logger.error(f'距离场构建失败: {e}')


def score_distance_field(points_odom, cx, cy, yaw, map_ctx):
    """距离场评分: 平均像素距离（越小越贴合墙壁）。

    原理与 global_match.py 一致:
      - 将点云变换到地图坐标系
      - 转换到像素坐标
      - 查询距离场中的值（像素距离）
      - 返回所有合法点的平均距离

    返回: (score, n_valid)
      score: float, 平均像素距离 (越小越好, float('inf') 表示无效)
      n_valid: int, 有效点数
    """
    if map_ctx.dist_field is None or map_ctx.map_info is None:
        return float('inf'), 0

    res = map_ctx.map_info.resolution
    ox = map_ctx.map_info.origin.position.x
    oy = map_ctx.map_info.origin.position.y
    H, W = map_ctx.map_info.height, map_ctx.map_info.width

    # 防御: dist_field 尺寸必须与 map_info 一致 (地图热重载后旧场可能未清除)
    if map_ctx.dist_field.shape[0] != H or map_ctx.dist_field.shape[1] != W:
        return float('inf'), 0

    # 变换到地图坐标系
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    if points_odom.ndim == 2:
        mx = c_y * points_odom[:, 0] - s_y * points_odom[:, 1] + cx
        my = s_y * points_odom[:, 0] + c_y * points_odom[:, 1] + cy
    else:
        mx = np.array([c_y * points_odom[0] - s_y * points_odom[1] + cx])
        my = np.array([s_y * points_odom[0] + c_y * points_odom[1] + cy])

    # 转换到像素坐标
    pix_x = ((mx - ox) / res + 0.5).astype(np.int32)
    pix_y = ((my - oy) / res + 0.5).astype(np.int32)

    valid = (pix_x >= 0) & (pix_x < W) & (pix_y >= 0) & (pix_y < H)
    n_valid = int(np.sum(valid))

    # 超过一半点在地图外 → 无效
    if n_valid < len(points_odom) * 0.5:
        return float('inf'), n_valid

    score = np.sum(map_ctx.dist_field[pix_y[valid], pix_x[valid]]) / n_valid
    return float(score), n_valid


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


# ──────────────────────────────────────────────────────────────
#  双模板光线投射全局匹配 (来自 global_match2.py)
#  核心: 命中模板 (likelihood) + 射线惩罚模板 (ray penalty)
#  使用 cv2.matchTemplate 做全图卷积加速
# ──────────────────────────────────────────────────────────────
def build_dual_template_maps(map_ctx, logger=None):
    """构建双模板匹配所需的似然地图和射线惩罚地图。

    写入 map_ctx 的新字段:
      - dt_likelihood_map: (H, W) float32, 用于奖励扫描点命中墙壁
      - dt_ray_penalty_map: (H, W) float32, 用于惩罚射线穿过墙壁/未知区域
      - dt_valid_center_mask: (H, W) bool, 标记机器狗中心可放置的合法区域

    依赖 cv2 可用。
    """
    if map_ctx.map_data is None:
        return
    try:
        import cv2
        map_data = map_ctx.map_data
        res = map_ctx.map_info.resolution

        # 墙壁掩码 (ROS occupancy: 51-100 视为潜在墙壁)
        walls_mask = (map_data > 50) & (map_data <= 100)
        binary_walls = np.zeros_like(map_data, dtype=np.uint8)
        binary_walls[walls_mask] = 255

        # 似然地图: 距离墙壁越近越高
        dist_map = cv2.distanceTransform(255 - binary_walls, cv2.DIST_L2, 5)
        max_dist_px = 10.0  # 最大奖励距离 (像素)
        likelihood_map = np.clip(max_dist_px - dist_map, 0, max_dist_px) / max_dist_px
        map_ctx.dt_likelihood_map = likelihood_map.astype(np.float32)

        # 射线惩罚地图: 墙壁=1.0, 未知区域=2.0, 空地=0.0
        ray_penalty_map = np.zeros_like(map_data, dtype=np.float32)
        ray_penalty_map[walls_mask] = 1.0
        unknown_mask = (map_data < 0) | (map_data > 100)
        ray_penalty_map[unknown_mask] = 2.0
        map_ctx.dt_ray_penalty_map = ray_penalty_map

        # 合法机器狗中心区域 (已知空地)
        map_ctx.dt_valid_center_mask = (map_data >= 0) & (map_data < 50)

        # 自由空间积分图 (用于面积比惩罚, O(1) 区域查询)
        free_space = (map_data >= 0) & (map_data <= 50)
        map_ctx.dt_free_space_integral = cv2.integral(free_space.astype(np.uint8))

        if logger is not None:
            logger.info(
                f'[双模板] 地图构建完成: 似然图 {likelihood_map.shape}, '
                f'惩罚图 {ray_penalty_map.shape}, '
                f'合法中心区域 {np.sum(map_ctx.dt_valid_center_mask)} 像素')

    except Exception as e:
        if logger is not None:
            logger.error(f'[双模板] 地图构建失败: {e}')


def dual_template_global_match(scan_pts, map_ctx,
                               coarse_angle_step_deg=2.0,
                               fine_angle_step_deg=0.5,
                               penalty_weight=3.0,
                               scan_max_points=500,
                               free_space_penalty_weight=0.0,
                               logger=None):
    """双模板光线投射全局匹配 — 核心算法。

    原理:
      - 命中模板: 扫描点投影到似然地图上获得正奖励
      - 射线模板: 从机器狗原点到扫描点的连线, 穿过墙壁/未知区则惩罚
      - 使用 cv2.matchTemplate 做全图卷积, 一次性计算所有 (x,y) 位置得分
      - 粗搜 (coarse_angle_step_deg) → 精搜 (fine_angle_step_deg) 两阶段

    参数:
      scan_pts:   (N, 2) numpy array, 雷达扫描点在 odom 坐标系下的坐标
      map_ctx:    MapContext 对象, 需已调用 build_dual_template_maps
      coarse_angle_step_deg: 粗搜角度步长 (度)
      fine_angle_step_deg:   精修角度步长 (度)
      penalty_weight:        射线穿透惩罚权重 (越大越排斥穿墙)
      scan_max_points:       扫描点最大采样数 (提高速度)
      free_space_penalty_weight: 面积比惩罚权重 (越大越排斥面积不匹配, 0=禁用, 防对称走廊误匹配)
      logger:                日志器

    返回: (best_pose, best_score)
      best_pose: (x, y, yaw) in map frame (世界坐标)
      best_score: float, 最终得分 (越高越贴合)
    """
    try:
        import cv2
    except ImportError:
        if logger is not None:
            logger.error('[双模板] cv2 不可用')
        return None, -float('inf')

    t0 = time.time()

    if map_ctx.dt_likelihood_map is None or map_ctx.dt_ray_penalty_map is None:
        if logger is not None:
            logger.error('[双模板] 双模板地图未构建')
        return None, -float('inf')

    likelihood_map = map_ctx.dt_likelihood_map
    ray_penalty_map = map_ctx.dt_ray_penalty_map
    valid_center_mask = map_ctx.dt_valid_center_mask

    res = map_ctx.map_info.resolution
    ox = map_ctx.map_info.origin.position.x
    oy = map_ctx.map_info.origin.position.y
    H, W = map_ctx.map_info.height, map_ctx.map_info.width

    # 降采样扫描点
    pts = scan_pts.copy()
    if len(pts) > scan_max_points:
        rng = np.random.default_rng(42)  # 种子固定使结果可复现
        indices = rng.choice(len(pts), scan_max_points, replace=False)
        pts = pts[indices]

    if logger is not None:
        logger.info(f'[双模板] 启动, {len(pts)} 个扫描点')

    best_score = -float('inf')
    best_pose = (0.0, 0.0, 0.0)
    pad = 0.5  # 模板边距 (米)

    def _build_templates(rotated_pts):
        """为给定旋转后的点云构建命中/射线模板。"""
        min_x, max_x = min(0.0, np.min(rotated_pts[:, 0])), max(0.0, np.max(rotated_pts[:, 0]))
        min_y, max_y = min(0.0, np.min(rotated_pts[:, 1])), max(0.0, np.max(rotated_pts[:, 1]))

        tpw = int((max_x - min_x + 2 * pad) / res)
        tph = int((max_y - min_y + 2 * pad) / res)

        if tpw <= 0 or tph <= 0 or tpw > W or tph > H:
            return None

        tpl_hit = np.zeros((tph, tpw), dtype=np.float32)
        tpl_ray = np.zeros((tph, tpw), dtype=np.float32)
        tpl_ox = min_x - pad
        tpl_oy = min_y - pad

        robot_x_tpl = int(-tpl_ox / res)
        robot_y_tpl = int(-tpl_oy / res)

        pix_x = ((rotated_pts[:, 0] - tpl_ox) / res).astype(int)
        pix_y = ((rotated_pts[:, 1] - tpl_oy) / res).astype(int)
        valid = (pix_x >= 0) & (pix_x < tpw) & (pix_y >= 0) & (pix_y < tph)
        vx, vy = pix_x[valid], pix_y[valid]

        if len(vx) == 0:
            return None

        # 绘制射线模板 (机器狗 → 各扫描点)
        for px, py in zip(vx, vy):
            cv2.line(tpl_ray, (robot_x_tpl, robot_y_tpl), (px, py), 1.0, 1)
        # 擦除终点 (扫描点命中墙壁是合法结果)
        tpl_ray[vy, vx] = 0.0

        # 绘制命中模板
        tpl_hit[vy, vx] = 1.0

        return (tpl_hit, tpl_ray, tpl_ox, tpl_oy, robot_x_tpl, robot_y_tpl, tpw, tph)

    def _score_one_angle(cos_t, sin_t):
        """对一个旋转角度评分, 返回 (best_local_x, best_local_y, score)。"""
        rotated = np.column_stack([
            pts[:, 0] * cos_t - pts[:, 1] * sin_t,
            pts[:, 0] * sin_t + pts[:, 1] * cos_t,
        ])

        tpl_result = _build_templates(rotated)
        if tpl_result is None:
            return None

        tpl_hit, tpl_ray, tpl_ox, tpl_oy, rxt, ryt, tpw, tph = tpl_result

        # 全图卷积
        res_hit = cv2.matchTemplate(likelihood_map, tpl_hit, cv2.TM_CCORR)
        res_ray = cv2.matchTemplate(ray_penalty_map, tpl_ray, cv2.TM_CCORR)
        res_final = res_hit - (res_ray * penalty_weight)

        rh, rw = res_final.shape
        ys, ye = ryt, ryt + rh
        xs, xe = rxt, rxt + rw

        if ye >= H or xe >= W:
            return None

        # 过滤机器狗中心不在合法空地的位置
        valid_slice = valid_center_mask[ys:ye, xs:xe]
        res_final[~valid_slice] = -float('inf')

        _, max_val, _, max_loc = cv2.minMaxLoc(res_final)
        if max_val <= -1e8:
            return None

        match_x, match_y = max_loc
        rx = match_x * res + ox - tpl_ox
        ry = match_y * res + oy - tpl_oy

        # ── 面积比惩罚: 扫描点围合区域 vs 地图对应位置自由空间面积 ──
        if free_space_penalty_weight > 0 and map_ctx.dt_free_space_integral is not None:
            area_penalty = _compute_area_ratio_penalty(
                rotated, rx, ry, map_ctx, free_space_penalty_weight, res, ox, oy, H, W)
            max_val -= area_penalty

        return (rx, ry, max_val)

    # ── Phase 1: 粗搜索 ──
    coarse_angles = np.arange(-np.pi, np.pi, np.deg2rad(coarse_angle_step_deg))
    if logger is not None:
        logger.info(f'[双模板] 粗搜: {len(coarse_angles)} 个角度 (步长 {coarse_angle_step_deg}°)')

    for theta in coarse_angles:
        result = _score_one_angle(math.cos(theta), math.sin(theta))
        if result is not None and result[2] > best_score:
            best_score = result[2]
            best_pose = (result[0], result[1], theta)

    if logger is not None:
        t1 = time.time()
        logger.info(f'[双模板] 粗搜完成 ({t1 - t0:.2f}s), best=({best_pose[0]:.2f},{best_pose[1]:.2f},'
                    f'{math.degrees(best_pose[2]):.1f}deg) score={best_score:.1f}')

    # ── Phase 2: 精细局部角度搜索 ──
    fine_angles = np.arange(
        best_pose[2] - np.deg2rad(coarse_angle_step_deg),
        best_pose[2] + np.deg2rad(coarse_angle_step_deg) + 1e-5,
        np.deg2rad(fine_angle_step_deg))

    for theta in fine_angles:
        result = _score_one_angle(math.cos(theta), math.sin(theta))
        if result is not None and result[2] > best_score:
            best_score = result[2]
            best_pose = (result[0], result[1], theta)

    elapsed = time.time() - t0
    if logger is not None:
        logger.info(
            f'[双模板] 全部完成 ({elapsed:.2f}s): '
            f'({best_pose[0]:.3f},{best_pose[1]:.3f},{math.degrees(best_pose[2]):.2f}deg) '
            f'score={best_score:.1f}')

    return best_pose, best_score


def _compute_area_ratio_penalty(rotated_pts, rx, ry, map_ctx, penalty_weight, res, ox, oy, H, W):
    """计算候选位姿处扫描包围盒面积与地图自由空间面积的比例惩罚。

    原理: 将旋转后的扫描点投影到候选位置 (rx, ry)，取包围盒，
    用积分图 O(1) 查询该区域内的自由空间面积，与扫描包围盒面积比较。
    面积比越接近 1.0 → 惩罚越轻；差距越大 → 惩罚越重。

    返回: float, 惩罚值 (非负, 0 表示无惩罚)
    """
    if map_ctx.dt_free_space_integral is None:
        return 0.0

    # 扫描点投影到地图坐标系
    proj = rotated_pts + np.array([rx, ry])

    # 包围盒 (地图像素坐标)
    min_px = int((proj[:, 0].min() - ox) / res)
    max_px = int((proj[:, 0].max() - ox) / res) + 1
    min_py = int((proj[:, 1].min() - oy) / res)
    max_py = int((proj[:, 1].max() - oy) / res) + 1

    # 裁剪到地图边界
    min_px = max(0, min(min_px, W - 2))
    max_px = max(min_px + 1, min(max_px, W - 1))
    min_py = max(0, min(min_py, H - 2))
    max_py = max(min_py + 1, min(max_py, H - 1))

    bbox_pixels = float((max_px - min_px) * (max_py - min_py))
    if bbox_pixels <= 0:
        return penalty_weight * len(rotated_pts)

    # 积分图查询包围盒内自由空间像素数
    integral = map_ctx.dt_free_space_integral
    A = float(integral[min_py, min_px])
    B = float(integral[min_py, max_px])
    C = float(integral[max_py, min_px])
    D = float(integral[max_py, max_px])
    free_pixels = D - B - C + A

    if free_pixels <= 0:
        return penalty_weight * len(rotated_pts)  # 无自由空间 → 最大惩罚

    # 面积比: 地图自由空间 vs 扫描包围盒面积
    scan_bbox_area = (proj[:, 0].max() - proj[:, 0].min()) * (proj[:, 1].max() - proj[:, 1].min())
    map_free_area = free_pixels * res * res

    if scan_bbox_area <= 0:
        return 0.0

    area_ratio = min(scan_bbox_area, map_free_area) / max(scan_bbox_area, map_free_area, 0.1)
    # 惩罚 = (1 - area_ratio) * weight * N, 与 hit_score(~N) 同一量级
    return (1.0 - area_ratio) * penalty_weight * len(rotated_pts)


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
