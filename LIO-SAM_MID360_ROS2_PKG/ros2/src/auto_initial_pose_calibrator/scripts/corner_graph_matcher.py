#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
墙角特征图匹配定位

核心思路:
  之前的点级评分(似然场/光线投射)无法区分形状相似的走廊/房间。
  墙角(L型拐角)是空间中最稳定的拓扑特征 — 位置、角度、相对布局构成唯一"指纹"。

算法流程:
  1. 扫描 → 提取墙角 (多边形简化 + 拐角检测)
  2. 地图 → 提取墙角 (边缘追踪 + 拐角检测)
  3. 对每个地图候选区域:
     构建墙角关系图 (角度指纹 + 距离指纹)
     与扫描墙角图匹配
     用 Umeyama 算法求最优刚体变换 (旋转+平移)
     计算匹配残差 → 评分

  4. Top-K 候选 → 似然场验证 → 输出

用法:
  python3 corner_graph_matcher.py
  python3 corner_graph_matcher.py --data scan_viz/debug_match_data.npz
"""

import os
import sys
import math
import time
import argparse
import itertools
import numpy as np

try:
    import cv2
except ImportError:
    print("需要 opencv-python"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("需要 matplotlib"); sys.exit(1)


def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()


# ============================================================
# 数据加载 (同前)
# ============================================================
def load_data(npz_path):
    if not os.path.exists(npz_path):
        print(f"[错误] 文件不存在: {npz_path}"); sys.exit(1)
    data = np.load(npz_path, allow_pickle=True)
    info = {
        'resolution': float(data['map_resolution']),
        'width': int(data['map_width']),
        'height': int(data['map_height']),
        'origin_x': float(data['map_origin_x']),
        'origin_y': float(data['map_origin_y']),
    }
    tf_gt = data['tf_odom_to_map']
    frame_tfs = data['frame_tfs']
    angle_min = float(data['frame_angle_min'])
    angle_inc = float(data['frame_angle_increment'])
    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in data:
        frame_ranges.append(np.array(data[f'frame_ranges_{i}'], dtype=np.float64))
        i += 1

    print(f"{'='*60}")
    print(f"  Frames={len(frame_ranges)}, Beams/Frame={len(frame_ranges[0])}")
    print(f"  Map: {info['width']}x{info['height']} @ {info['resolution']:.3f}m/pix")
    print(f"  TF(ref): ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f} deg)")
    print(f"{'='*60}")
    return data['map_data'], info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc


# ============================================================
# 1. 玻璃过滤 (逐帧FRF + 合并)
# ============================================================
def merge_and_filter(frame_ranges, frame_tfs, angle_min, angle_inc, bin_deg=2.0):
    """逐帧 FRF 过滤 + 合并"""
    total_raw = 0
    n_removed = 0
    all_pts = []
    all_fids = []
    bin_size = np.radians(bin_deg)

    for fi, (ranges, tf) in enumerate(zip(frame_ranges, frame_tfs)):
        valid = (ranges > 0.1) & (ranges < 50.0)
        total_raw += int(np.sum(valid))
        if not np.any(valid):
            continue

        angles_arr = angle_min + np.arange(len(ranges)) * angle_inc
        bins = np.round(angles_arr / bin_size).astype(int)
        keep = np.ones(len(ranges), dtype=bool)

        for b in np.unique(bins[valid]):
            idx = np.where((bins == b) & valid)[0]
            if len(idx) < 2:
                continue
            sorted_idx = idx[np.argsort(ranges[idx])]
            sorted_r = ranges[sorted_idx]
            gaps = np.diff(sorted_r) > 0.3
            if not np.any(gaps):
                continue
            first_gap = int(np.argmax(gaps))
            keep[sorted_idx[first_gap+1:]] = False

        n_removed += int(np.sum(valid & ~keep))
        final_valid = valid & keep

        if not np.any(final_valid):
            continue
        x_l = ranges[final_valid] * np.cos(angles_arr[final_valid])
        y_l = ranges[final_valid] * np.sin(angles_arr[final_valid])
        tx, ty, yaw = tf
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.column_stack([c*x_l - s*y_l + tx, s*x_l + c*y_l + ty])
        all_pts.append(pts)
        all_fids.append(np.full(len(pts), fi, dtype=np.int32))

    if not all_pts:
        return np.empty((0, 2)), {'total': 0, 'removed': 0}

    merged = np.vstack(all_pts)
    fids = np.concatenate(all_fids)
    stats = {'total': total_raw, 'removed': n_removed, 'final': len(merged)}
    print(f"  [FRF] {total_raw} -> {len(merged)} pts (removed {n_removed} ghosts, {100*n_removed/max(1,total_raw):.1f}%)")
    return merged, stats


# ============================================================
# 2. 墙角提取 (扫描侧)
# ============================================================
def extract_scan_corners(pts, min_corner_angle_deg=30, epsilon_m=0.15):
    """
    从扫描点云提取墙角。

    方法:
      1. 将点转为极坐标，按角度排序
      2. 连接相邻点形成多边形
      3. 用 Douglas-Peucker 简化多边形
      4. 简化后的顶点 = 候选墙角
      5. 过滤角度变化 < min_corner_angle_deg 的顶点（太直，不是角）

    返回: corners [(x, y, angle, wall_dir_1, wall_dir_2), ...]
    """
    if len(pts) < 10:
        return []

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    rel = pts - np.array([cx, cy])

    # 按角度排序形成有序边界
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(angles)
    ordered_pts = rel[order]

    # Douglas-Peucker 简化 (OpenCV)
    contour = ordered_pts.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(contour, epsilon_m, closed=True)
    simp_pts = simplified.reshape(-1, 2)

    if len(simp_pts) < 3:
        return []

    # 检查每个顶点的转向角
    corners = []
    n = len(simp_pts)
    for i in range(n):
        p_prev = simp_pts[(i - 1) % n]
        p_curr = simp_pts[i]
        p_next = simp_pts[(i + 1) % n]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        if l1 < 0.01 or l2 < 0.01:
            continue

        d1 = v1 / l1  # wall direction 1
        d2 = v2 / l2  # wall direction 2

        # 内角 = 两条边之间的夹角
        cos_a = np.clip(np.dot(d1, d2), -1, 1)
        interior_angle = math.degrees(np.arccos(cos_a))

        # 只保留有明显拐角的顶点
        turning = 180 - interior_angle
        if abs(turning) > min_corner_angle_deg:
            wall_dir_1 = math.atan2(-d1[1], -d1[0])  # 指向墙的方向
            wall_dir_2 = math.atan2(d2[1], d2[0])
            corners.append({
                'x': p_curr[0] + cx,
                'y': p_curr[1] + cy,
                'angle': interior_angle,
                'wall_dirs': (wall_dir_1, wall_dir_2),
                'turning': turning,
            })

    return corners


# ============================================================
# 3. 墙角提取 (地图侧)
# ============================================================
def extract_map_corners(map_data, info, region_center_m, region_radius_m=10.0,
                        min_corner_angle_deg=30):
    """
    从地图中提取指定区域的墙角。

    方法:
      1. 提取区域内的墙壁像素
      2. 边缘检测找到墙的边缘线
      3. HoughLinesP 或轮廓简化 → 线段
      4. 线段交点 = 墙角
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = map_data.shape
    cx_px = int((region_center_m[0] - ox) / res)
    cy_px = int(H - 1 - (region_center_m[1] - oy) / res)
    r_px = int(region_radius_m / res)

    r1 = max(0, cy_px - r_px)
    r2 = min(H, cy_px + r_px)
    c1 = max(0, cx_px - r_px)
    c2 = min(W, cx_px + r_px)

    if r2 - r1 < 20 or c2 - c1 < 20:
        return []

    roi = map_data[r1:r2, c1:c2]
    wall = (roi == 100).astype(np.uint8) * 255

    if np.sum(wall) < 100:
        return []

    # 边缘检测
    edges = cv2.Canny(wall, 50, 150)

    # HoughLinesP 提取线段
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                             minLineLength=15, maxLineGap=10)
    if lines is None or len(lines) < 2:
        return []

    # 线段交点 = 墙角
    map_corners = []
    lines_arr = lines.reshape(-1, 4)  # x1,y1,x2,y2

    for i in range(len(lines_arr)):
        for j in range(i+1, len(lines_arr)):
            x1, y1, x2, y2 = lines_arr[i].astype(float)
            x3, y3, x4, y4 = lines_arr[j].astype(float)

            # 检查是否相交或端点相邻
            d13 = math.sqrt((x1-x3)**2 + (y1-y3)**2) * res
            d14 = math.sqrt((x1-x4)**2 + (y1-y4)**2) * res
            d23 = math.sqrt((x2-x3)**2 + (y2-y3)**2) * res
            d24 = math.sqrt((x2-x4)**2 + (y2-y4)**2) * res

            # 取最近的端点对
            min_d = min(d13, d14, d23, d24)
            if min_d > 2.0:  # 端点间距 > 2m，不算墙角
                continue

            # 计算交点（两线段延长线交点）
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            ix = x1 + t*(x2-x1)
            iy = y1 + t*(y2-y1)

            # 交点在 ROI 内
            if ix < 0 or ix >= c2-c1 or iy < 0 or iy >= r2-r1:
                continue

            # 两条线的方向
            v1 = np.array([x2-x1, y2-y1], dtype=float)
            v2 = np.array([x4-x3, y4-y3], dtype=float)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            v1 /= n1
            v2 /= n2

            cos_a = np.clip(np.dot(v1, v2), -1, 1)
            angle = math.degrees(np.arccos(abs(cos_a)))  # 0-90 deg

            if angle < min_corner_angle_deg:
                continue

            # 转换到世界坐标
            wx = ix * res + c1 * res + ox
            wy = (H - 1 - (iy + r1)) * res + oy

            wall_dir_1 = math.atan2(v1[1], v1[0])
            wall_dir_2 = math.atan2(v2[1], v2[0])

            map_corners.append({
                'x': wx,
                'y': wy,
                'angle': angle,
                'wall_dirs': (wall_dir_1, wall_dir_2),
            })

    return map_corners


# ============================================================
# 3b. 全图墙角预计算 (加速版)
# ============================================================
def extract_all_map_corners_precomputed(map_data, info, min_corner_angle_deg=25):
    """一次性提取全地图所有墙角"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = map_data.shape

    wall = (map_data == 100).astype(np.uint8) * 255
    edges = cv2.Canny(wall, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                             minLineLength=30, maxLineGap=15)
    if lines is None or len(lines) < 2:
        return []
    # 限制线段数防止 O(n²) 爆炸
    if len(lines) > 200:
        arr = lines.reshape(-1, 4)
        lengths = np.sqrt((arr[:,2]-arr[:,0])**2+(arr[:,3]-arr[:,1])**2)
        top = np.argsort(lengths)[-200:]
        arr = arr[top]
    else:
        arr = lines.reshape(-1, 4)
    print(f"  HoughLinesP: {len(arr)} segments (trimmed)")
    all_corners = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            x1,y1,x2,y2 = arr[i].astype(float)
            x3,y3,x4,y4 = arr[j].astype(float)
            d13 = math.sqrt((x1-x3)**2+(y1-y3)**2)
            d24 = math.sqrt((x2-x4)**2+(y2-y4)**2)
            d14 = math.sqrt((x1-x4)**2+(y1-y4)**2)
            d23 = math.sqrt((x2-x3)**2+(y2-y3)**2)
            if min(d13,d14,d23,d24) > 40:
                continue
            denom = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
            if abs(denom) < 1e-6: continue
            t = ((x1-x3)*(y3-y4)-(y1-y3)*(x3-x4))/denom
            ix, iy = x1+t*(x2-x1), y1+t*(y2-y1)
            if ix < 0 or ix >= W or iy < 0 or iy >= H: continue
            v1 = np.array([x2-x1,y2-y1],float)
            v2 = np.array([x4-x3,y4-y3],float)
            n1,n2 = np.linalg.norm(v1),np.linalg.norm(v2)
            if n1<1e-6 or n2<1e-6: continue
            cos_a = np.clip(np.dot(v1/n1,v2/n2),-1,1)
            angle = math.degrees(np.arccos(abs(cos_a)))
            if angle < min_corner_angle_deg: continue
            all_corners.append({
                'x': ix*res+ox, 'y': (H-1-iy)*res+oy,
                'angle': angle,
                'wall_dirs': (math.atan2(v1[1],v1[0]), math.atan2(v2[1],v2[0])),
            })
    if len(all_corners) > 200:
        all_corners.sort(key=lambda c: c['angle'], reverse=True)
        all_corners = all_corners[:200]
    return all_corners


def query_nearby_corners(all_corners, cx, cy, radius_m=10.0):
    nearby = [c for c in all_corners
              if math.sqrt((c['x']-cx)**2+(c['y']-cy)**2) < radius_m]
    if len(nearby) > 30:
        nearby.sort(key=lambda c: (c['x']-cx)**2+(c['y']-cy)**2)
        nearby = nearby[:30]
    return nearby


# ============================================================
# 4. 墙角图匹配 (Umeyama 算法)
# ============================================================
def build_corner_fingerprint(corners):
    """
    对墙角集构建拓扑指纹: 每对角之间的距离和角度。
    返回: (pts_array, pairwise_matrix)
    """
    if len(corners) < 2:
        return np.array([]), np.array([])

    pts = np.array([[c['x'], c['y']] for c in corners])
    n = len(pts)

    # 成对距离矩阵
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.linalg.norm(pts[i] - pts[j])

    return pts, dists


def match_corner_sets(scan_corners, map_corners, max_residual=1.0):
    """
    匹配扫描墙角集与地图墙角集。

    用成对距离矩阵做兼容性筛选 + Umeyama 求最优刚体变换。

    当扫描有 N_s 个角，地图有 N_m 个角:
      1. 用距离矩阵找兼容对 (两角之间的距离一致)
      2. 对每个兼容的角对子集，用 Umeyama 求 (R, t, s)
      3. 用残差评分

    为避免组合爆炸，限制匹配子集大小 ≤ 4 个角对。
    """
    if len(scan_corners) < 2 or len(map_corners) < 2:
        return None, 0

    s_pts, s_dists = build_corner_fingerprint(scan_corners)
    m_pts, m_dists = build_corner_fingerprint(map_corners)

    if len(s_pts) == 0 or len(m_pts) == 0:
        return None, 0

    n_s = len(s_pts)
    n_m = len(m_pts)
    # 限制地图角数量，防止组合爆炸
    if n_m > 50:
        # 只保留距离中心最近的50个
        m_center = np.mean(m_pts, axis=0)
        dists = np.sqrt(np.sum((m_pts - m_center)**2, axis=1))
        top_idx = np.argsort(dists)[:50]
        m_pts = m_pts[top_idx]
        map_corners = [map_corners[i] for i in top_idx]
        n_m = len(m_pts)
    
    n_s = len(s_pts)
    n_m = len(m_pts)

    # 为每对 (scan角i, map角j) 检查兼容性
    # 兼容条件: 与各自其他角的距离比例一致 (允许10%误差)
    compat = {}  # (i,j) -> set of compatible (i',j')
    tol = 0.15  # 距离容差比例

    for i in range(n_s):
        for j in range(n_m):
            matches = [(i, j)]
            for i2 in range(n_s):
                if i2 == i:
                    continue
                for j2 in range(n_m):
                    if j2 == j:
                        continue
                    # 距离比一致性
                    sd = s_dists[i, i2]
                    md = m_dists[j, j2]
                    if sd < 0.1 or md < 0.1:
                        continue
                    ratio = abs(sd - md) / max(sd, md)
                    if ratio < tol:
                        matches.append((i2, j2))
            compat[(i, j)] = matches

    # 找最大的兼容匹配集
    best_result = None
    best_score = -1e9

    # 遍历所有 (i,j) 种子，尝试扩展到最多 min(n_s, n_m, 4) 对
    max_pairs = min(n_s, n_m, 5)

    for (seed_i, seed_j), matches in compat.items():
        if len(matches) < 2:
            continue

        # 从 matches 中选择最多 max_pairs 个不重复的 (scan角, map角) 对
        seen_s = set()
        seen_m = set()
        selected = []
        for mi, mj in matches:
            if mi not in seen_s and mj not in seen_m:
                selected.append((mi, mj))
                seen_s.add(mi)
                seen_m.add(mj)
                if len(selected) >= max_pairs:
                    break

        if len(selected) < 2:
            continue

        # Umeyama: 求最优刚体变换 s_pts[scan_idx] → m_pts[map_idx]
        src = np.array([s_pts[si] for si, _ in selected])
        dst = np.array([m_pts[mj] for _, mj in selected])

        result = umeyama(src, dst)
        if result is None:
            continue

        R, t, scale = result

        # 计算残差
        transformed = (R @ src.T).T * scale + t
        residuals = np.linalg.norm(transformed - dst, axis=1)
        mean_res = np.mean(residuals)

        if mean_res > max_residual:
            continue

        # 评分: 匹配角数多 + 残差小
        score = len(selected) * 2.0 - mean_res * 3.0

        if score > best_score:
            best_score = score
            # 从 R 提取旋转角
            yaw = math.atan2(R[1, 0], R[0, 0])
            # t 是 scan 坐标系原点在 map 坐标系中的位置
            # 但我们需要的是 odom→map 变换
            # scan 坐标是以 scan 质心为中心的, 所以:
            #   scan_global = R * (scan_local) * scale + t
            #   即 scan_local → map 坐标
            best_result = {
                'R': R, 't': t, 'scale': scale,
                'yaw': yaw,
                'residual': mean_res,
                'n_pairs': len(selected),
                'score': score,
                'pairs': selected,
                'transformed': transformed,
            }

    return best_result, best_score


def umeyama(src, dst):
    """Umeyama 算法: 求最优刚体变换 src → dst (允许均匀缩放)"""
    assert src.shape == dst.shape
    n, dim = src.shape
    if n < 2:
        return None

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    # 协方差
    H = (src_c.T @ dst_c) / n

    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None

    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.eye(dim)
    sign_mat[-1, -1] = np.sign(d)

    R = Vt.T @ sign_mat @ U.T

    # 缩放
    var_src = np.sum(src_c**2) / n
    scale = np.trace(np.diag(S) @ sign_mat) / max(var_src, 1e-10)
    scale = max(0.5, min(2.0, scale))  # 限制缩放范围

    t = mu_dst - scale * R @ mu_src

    return R, t, scale


# ============================================================
# 5. 全地图搜索
# ============================================================
def global_corner_search(pts_filtered, map_data, info, tf_gt=None,
                          grid_step=2.0, angle_step=30.0):
    """
    全地图墙角特征匹配。

    对地图中每个自由空间位置:
      1. 以该位置为中心提取地图墙角 (10m 范围)
      2. 与扫描墙角做图匹配 (Umeyama)
      3. 评分: 匹配角数 × 2 - 残差 × 3
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    map_w_m = info['width'] * res
    map_h_m = info['height'] * res
    H = info['height']
    W = info['width']

    # 提取扫描墙角
    scan_corners_raw = extract_scan_corners(pts_filtered, min_corner_angle_deg=15, epsilon_m=0.15)
    # 只保留距离中心最远的 top-N 角（最远离中心的墙段最长/最稳定）
    if len(scan_corners_raw) > 20:
        cx = np.mean([c['x'] for c in scan_corners_raw])
        cy = np.mean([c['y'] for c in scan_corners_raw])
        scan_corners_raw.sort(key=lambda c: (c['x']-cx)**2+(c['y']-cy)**2, reverse=True)
        scan_corners = scan_corners_raw[:20]
        print(f"\n  扫描墙角: {len(scan_corners_raw)} -> top-{len(scan_corners)}")
    else:
        scan_corners = scan_corners_raw
    print(f"\n  扫描墙角: {len(scan_corners)} 个")
    for i, c in enumerate(scan_corners[:10]):
        print(f"    #{i}: ({c['x']:.2f}, {c['y']:.2f}) "
              f"angle={c['angle']:.0f} turn={c['turning']:.0f}")

    if len(scan_corners) < 2:
        print("  [错误] 扫描墙角太少，无法匹配")
        return [], []


    # --- Precompute all map corners (one-time) ---
    print(f"\n  [预计算] 提取全图墙角...")
    t_pre = time.time()
    all_map_corners = extract_all_map_corners_precomputed(
        map_data, info, min_corner_angle_deg=25)
    print(f"  [预计算] {len(all_map_corners)} 个全图墙角, 耗时 {time.time()-t_pre:.1f}s")
    if len(all_map_corners) < 2:
        print("  [错误] 地图墙角太少")
        return [], []

    # 搜索网格
    xs = np.arange(ox + 5, ox + map_w_m - 5, grid_step)
    ys = np.arange(oy + 5, oy + map_h_m - 5, grid_step)
    n_total = 0
    for ax in xs:
        for ay in ys:
            col = int((ax - ox) / res)
            row = int(H - 1 - (ay - oy) / res)
            if 0 <= col < W and 0 <= row < H and map_data[row, col] == 0:
                n_total += 1

    print(f"\n  搜索网格: {len(xs)}x{len(ys)} = {len(xs)*len(ys)} 位置 ({n_total} 自由空间)")

    all_results = []
    t0 = time.time()
    count = 0

    for ax in xs:
        for ay in ys:
            col = int((ax - ox) / res)
            row = int(H - 1 - (ay - oy) / res)
            if col < 0 or col >= W or row < 0 or row >= H:
                continue
            if map_data[row, col] != 0:
                continue
            count += 1

            # 从预计算结果中查询附近墙角
            nearby = query_nearby_corners(all_map_corners, ax, ay, radius_m=10.0)
            if len(nearby) < 2:
                continue

            result, score = match_corner_sets(scan_corners, nearby, max_residual=2.0)
            if result is not None:
                all_results.append({
                    'center': (ax, ay),
                    'result': result,
                    'score': score,
                    'n_map_corners': len(nearby),
                })

        if count % 50 == 0:
            print(f"  进度: {count}/{n_total} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"\n  搜索完成: {elapsed:.1f}s, {len(all_results)} 个有效匹配")

    all_results.sort(key=lambda x: x['score'], reverse=True)
    nms = []
    for r in all_results:
        cx, cy = r['center']
        dup = any(math.sqrt((cx-x['center'][0])**2+(cy-x['center'][1])**2) < 2.0 for x in nms)
        if not dup:
            nms.append(r)
            if len(nms) >= 10:
                break

    print(f"  Top-{len(nms)}:")
    for i, r in enumerate(nms[:5]):
        cx, cy = r['center']; rr = r['result']
        err = ""
        if tf_gt is not None: err = f" 距GT={math.sqrt((cx-tf_gt[0])**2+(cy-tf_gt[1])**2):.1f}m"
        print(f"    #{i}: ({cx:.1f},{cy:.1f}) yaw={math.degrees(rr['yaw']):.0f}deg "
              f"pair={rr['n_pairs']} res={rr['residual']:.2f}m s={r['score']:.1f} "
              f"scale={rr['scale']:.3f}{err}")

    # GT诊断
    if tf_gt is not None:
        gt_nearby = query_nearby_corners(all_map_corners, tf_gt[0], tf_gt[1], 10.0)
        gt_res, gt_sc = match_corner_sets(scan_corners, gt_nearby, max_residual=2.0)
        if gt_res:
            print(f"  [GT] {len(gt_nearby)} near corners, match: pair={gt_res['n_pairs']} "
                  f"res={gt_res['residual']:.2f}m score={gt_sc:.1f} "
                  f"scale={gt_res['scale']:.3f}")
        else:
            print(f"  [GT] {len(gt_nearby)} near corners, NO MATCH")

    return nms, scan_corners
def refine_with_likelihood(pts_filtered, candidates, map_data, info, tf_gt=None):
    """对 Top-K 候选做似然场 + 面积匹配精细验证"""
    if not candidates:
        return None, None, None, -1e9, []

    # 构建似然场
    obs = (map_data == 100).astype(np.uint8)
    free = ((1 - obs) * 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    lf = np.clip(dist_px * info['resolution'], 0, 15.0)

    cx_pts, cy_pts = pts_filtered[:, 0].mean(), pts_filtered[:, 1].mean()
    pts_c = pts_filtered.copy()
    pts_c[:, 0] -= cx_pts
    pts_c[:, 1] -= cy_pts
    ds = max(1, len(pts_c) // 600)
    pts_ds = pts_c[::ds]

    # ─── 预设：扫描凸包特征 ───
    hull_pts = cv2.convexHull(pts_c.astype(np.float32).reshape(-1, 1, 2))
    hull_pts = hull_pts.reshape(-1, 2)
    scan_hull_area = cv2.contourArea(hull_pts.astype(np.float32))
    # 凸包长宽比 (PCA)
    hull_c = hull_pts - hull_pts.mean(axis=0)
    cov = np.cov(hull_c.T)
    eigvals, _ = np.linalg.eigh(cov)
    scan_aspect = math.sqrt(max(eigvals) / max(min(eigvals), 1e-6))  # ≥1.0
    print(f"  [预设] 扫描凸包面积={scan_hull_area:.1f}m² 长宽比={scan_aspect:.2f}")

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    all_refined = []
    n_filtered_by_area = 0

    for cand in candidates:
        result = cand['result']
        # Umeyama: map = scale * R * odom + t
        # 扫描机器人 odom 位置 = scan 质心
        R = result['R']; t = result['t']; s = result['scale']
        robot_map = s * (R @ np.array([cx_pts, cy_pts])) + t
        cx_base, cy_base = float(robot_map[0]), float(robot_map[1])
        yaw_base = result['yaw']
        # 首个候选打印诊断
        if cand == candidates[0]:
            print(f"  [DIAG] centroid_odom=({cx_pts:.2f},{cy_pts:.2f}) "
                  f"R=[[{R[0,0]:.3f},{R[0,1]:.3f}],[{R[1,0]:.3f},{R[1,1]:.3f}]] "
                  f"t=({t[0]:.2f},{t[1]:.2f}) scale={s:.3f} "
                  f"-> robot_map=({cx_base:.2f},{cy_base:.2f})")
        best_sc = -1e9
        best_pose = None
        best_area_score = 0

        for dx in np.arange(-2.0, 2.1, 0.5):
            for dy in np.arange(-2.0, 2.1, 0.5):
                fx, fy = cx_base + dx, cy_base + dy
                col = int((fx - ox) / res)
                row = int(H - 1 - (fy - oy) / res)
                if col < 0 or col >= W or row < 0 or row >= H:
                    continue
                if map_data[row, col] != 0:  # 必须出发于自由空间
                    continue

                for da in np.arange(-20, 21, 5):
                    fyaw = yaw_base + math.radians(da)
                    c_y, s_y = math.cos(fyaw), math.sin(fyaw)

                    # ─── 面积+长宽比匹配 ───
                    hx = c_y*hull_pts[:,0] - s_y*hull_pts[:,1] + fx
                    hy = s_y*hull_pts[:,0] + c_y*hull_pts[:,1] + fy
                    hcol = ((hx-ox)/res+0.5).astype(np.int32)
                    hrow = ((hy-oy)/res+0.5).astype(np.int32)
                    vh = (hcol>=0)&(hcol<W)&(hrow>=0)&(hrow<H)
                    if np.sum(vh) < len(hull_pts)*0.5: continue
                    mask = np.zeros((H,W),np.uint8)
                    poly = np.column_stack([hcol[vh],hrow[vh]]).astype(np.int32).reshape(-1,1,2)
                    cv2.fillPoly(mask,[poly],255)
                    hp = int(np.sum(mask>0))
                    if hp<10: continue
                    fp = int(np.sum((mask>0)&(map_data==0)))
                    up = int(np.sum((mask>0)&(map_data==-1)))
                    wp = int(np.sum((mask>0)&(map_data==100)))
                    if wp/hp>0.20 or (fp+up)/hp<0.60: n_filtered_by_area+=1; continue
                    # 面积比: 扫描凸包 vs 投影内自由空间
                    free_in_hull = (fp+up)*res*res
                    area_ratio = min(scan_hull_area, free_in_hull)/max(scan_hull_area, free_in_hull, 0.1)
                    # 长宽比: hull 投影 PCA
                    rr,cc = np.where(mask>0)
                    aspect_match = 0.5
                    if len(rr)>5:
                        pts = np.column_stack([cc,rr]).astype(np.float64)
                        c_ = pts-pts.mean(0); cr = np.cov(c_.T)
                        ev,_ = np.linalg.eigh(cr)
                        proj_asp = math.sqrt(max(ev)/max(min(ev),1e-6))
                        aspect_match = max(0.0, 1.0 - abs(proj_asp-scan_aspect)/max(scan_aspect,1.0))
                    area_free = fp/hp + up/hp*0.3  # hull 内有效空间占比
                    area_combined = area_free*0.3 + area_ratio*0.35 + aspect_match*0.35

                    # 似然场
                    mx = c_y*pts_ds[:,0]-s_y*pts_ds[:,1]+fx
                    my = s_y*pts_ds[:,0]+c_y*pts_ds[:,1]+fy
                    ca = ((mx-ox)/res+0.5).astype(np.int32)
                    ra = ((my-oy)/res+0.5).astype(np.int32)
                    vv = (ca>=0)&(ca<W)&(ra>=0)&(ra<H)
                    nv = int(np.sum(vv))
                    if nv<len(pts_ds)*0.1: continue
                    lf_sc = float(np.mean(np.exp(-lf[ra[vv],ca[vv]]**2/0.18)) +
                                  np.sum(lf[ra[vv],ca[vv]]<0.15)/max(nv,1))
                    sc = lf_sc + area_combined*3.0
                    if sc > best_sc:
                        best_sc = sc
                        best_pose = (fx, fy, fyaw)
                        best_area_score = area_combined

        if best_pose:
            all_refined.append({
                'x': best_pose[0],
                'y': best_pose[1],
                'yaw': best_pose[2],
                'lf_score': best_sc - best_area_score * 3.0,  # 还原纯LF分数
                'area_score': best_area_score,
                'corner_score': cand['score'],
                'n_pairs': cand['result']['n_pairs'],
                'total_score': cand['score'] + best_sc,  # corner + lf + area
            })

    all_refined.sort(key=lambda x: x['total_score'], reverse=True)

    if not all_refined:
        return None, None, None, -1e9, []

    print(f"  面积过滤: 淘汰 {n_filtered_by_area} 个无效位姿 (墙占比>20% or 无效<60%)")
    print(f"  似然场精搜 Top-5:")
    for i, r in enumerate(all_refined[:5]):
        err = ""
        if tf_gt is not None:
            err = f" 距GT={math.sqrt((r['x']-tf_gt[0])**2+(r['y']-tf_gt[1])**2):.1f}m"
        print(f"    #{i}: ({r['x']:.2f}, {r['y']:.2f}) yaw={math.degrees(r['yaw']):.1f}deg "
              f"corner={r['corner_score']:.1f} lf={r['lf_score']:.3f} "
              f"area={r['area_score']:.2f} total={r['total_score']:.2f}{err}")

    best = all_refined[0]
    return best['x'], best['y'], best['yaw'], best['total_score'], all_refined


# ============================================================
# 7. 可视化
# ============================================================
def create_visualization(map_data, info, tf_gt, pts_filtered, scan_corners,
                         candidates, best_x, best_y, best_yaw, best_score,
                         all_refined, stats, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W*res, oy, oy + H*res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1, 1, 1]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.75, 0.75, 0.75]

    cx_pts, cy_pts = pts_filtered[:, 0].mean(), pts_filtered[:, 1].mean()

    # (a) 扫描墙角 + 边界
    ax = axes[0, 0]
    ax.set_aspect('equal')
    # 画扫描点
    ax.scatter(pts_filtered[::5, 0], pts_filtered[::5, 1], s=0.5, c='gray', alpha=0.3)
    # 画墙角
    for i, c in enumerate(scan_corners):
        ax.plot(c['x'], c['y'], 'rs', markersize=8)
        ax.annotate(f"#{i}", (c['x'], c['y']), fontsize=7, color='red',
                    xytext=(3, 3), textcoords='offset points')
        # 画墙方向
        for wd in c['wall_dirs']:
            ax.annotate('', xy=(c['x'] + 1.5*math.cos(wd), c['y'] + 1.5*math.sin(wd)),
                        xytext=(c['x'], c['y']),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax.plot(cx_pts, cy_pts, 'b+', markersize=10, mew=2)
    ax.set_title(f"Scan Corners ({len(scan_corners)} found)")
    ax.grid(True, alpha=0.2)

    # (b) 全地图 + Top 候选
    ax = axes[0, 1]
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    for i, r in enumerate(all_refined[:5]):
        color = ['red', 'orange', 'green', 'cyan', 'magenta'][i]
        ax.plot(r['x'], r['y'], 'o', color=color, markersize=10-i)
        ax.annotate(f"#{i}", (r['x'], r['y']), fontsize=8, color=color, fontweight='bold')
        arrow = 2.0
        ax.annotate('', xy=(r['x']+arrow*math.cos(r['yaw']), r['y']+arrow*math.sin(r['yaw'])),
                    xytext=(r['x'], r['y']),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=12, label='GT')
    ax.legend(fontsize=8)
    ax.set_title("Map + Top-5 Candidates (arrows=yaw)")
    ax.grid(True, alpha=0.2)

    # (c) 最佳匹配叠加
    ax = axes[1, 0]
    if best_x is not None:
        zoom = 15
        ax.set_xlim(best_x - zoom, best_x + zoom)
        ax.set_ylim(best_y - zoom, best_y + zoom)
        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.set_aspect('equal')

        c_b, s_b = math.cos(best_yaw), math.sin(best_yaw)
        aligned = np.column_stack([
            c_b*pts_filtered[:, 0] - s_b*pts_filtered[:, 1] + best_x,
            s_b*pts_filtered[:, 0] + c_b*pts_filtered[:, 1] + best_y
        ])
        dists_a = np.sqrt((aligned[:, 0]-best_x)**2 + (aligned[:, 1]-best_y)**2)
        step = max(1, len(aligned)//2000)
        ax.scatter(aligned[::step, 0], aligned[::step, 1],
                   c=dists_a[::step], s=3, cmap='RdYlBu_r', alpha=0.7)
        ax.plot(best_x, best_y, 'r+', markersize=15, mew=3)
        ax.annotate('', xy=(best_x+2.5*math.cos(best_yaw), best_y+2.5*math.sin(best_yaw)),
                    xytext=(best_x, best_y),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax.set_title(f"Best Match: ({best_x:.2f}, {best_y:.2f}) yaw={math.degrees(best_yaw):.0f}deg\n"
                     f"Score={best_score:.2f}")
    ax.grid(True, alpha=0.2)

    # (d) 诊断
    ax = axes[1, 1]
    diag = "=== Corner Graph Match ===\n\n"
    diag += f"FRF: {stats.get('total',0)} -> {stats.get('final',0)} pts\n"
    diag += f"Scan corners: {len(scan_corners)}\n\n"
    diag += "Top-5:\n"
    for i, r in enumerate(all_refined[:5]):
        diag += (f"  #{i}: ({r['x']:.1f},{r['y']:.1f}) "
                 f"@{math.degrees(r['yaw']):.0f}deg "
                 f"pairs={r['n_pairs']} "
                 f"total={r['total_score']:.2f}\n")
    if tf_gt is not None:
        diag += f"\nGT(ref): ({tf_gt[0]:.1f}, {tf_gt[1]:.1f}) @ {math.degrees(tf_gt[2]):.1f}deg\n"
    diag += "\nCheck: scan points (colored)\n"
    diag += "should align with black walls\n"
    diag += "in the zoomed overlay (c)."
    ax.text(0.03, 0.97, diag, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[Output] PNG: {output_path}")
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Wall corner graph matching')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--grid-step', type=float, default=2.0)
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    output_dir = args.output or os.path.dirname(npz_path)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = \
        load_data(npz_path)

    # 2. Filter
    pts_filtered, stats = merge_and_filter(frame_ranges, frame_tfs, angle_min, angle_inc)

    # 3. Global corner search
    print("\n" + "="*60)
    print("Wall Corner Graph Matching")
    print("="*60)
    candidates, scan_corners = global_corner_search(
        pts_filtered, map_data, info, tf_gt=tf_gt, grid_step=args.grid_step)

    # 4. Likelihood refinement
    print("\n" + "="*60)
    print("Likelihood Refinement")
    print("="*60)
    best_x, best_y, best_yaw, best_score, all_refined = \
        refine_with_likelihood(pts_filtered, candidates, map_data, info, tf_gt=tf_gt)

    # 5. Results
    print("\n" + "="*60)
    print("Final Result")
    print("="*60)
    if best_x is not None:
        print(f"  Position: ({best_x:.3f}, {best_y:.3f})")
        print(f"  Yaw: {math.degrees(best_yaw):.1f} deg")
        print(f"  Score: {best_score:.4f}")
        if tf_gt is not None:
            print(f"  GT(ref): ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}) "
                  f"yaw={math.degrees(tf_gt[2]):.1f} deg")
    else:
        print("  [Failed] No match found")

    # 6. PNG
    output_path = os.path.join(output_dir, 'corner_match_result.png')
    create_visualization(
        map_data, info, tf_gt, pts_filtered, scan_corners,
        candidates, best_x, best_y, best_yaw, best_score,
        all_refined, stats, output_path)


if __name__ == '__main__':
    main()
