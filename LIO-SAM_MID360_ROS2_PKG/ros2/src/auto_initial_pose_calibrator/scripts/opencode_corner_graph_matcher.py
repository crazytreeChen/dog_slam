#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opencode_corner_graph_matcher.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
墙角特征图匹配 定位算法

核心思路:
  L型墙角是空间中最稳定的拓扑特征。位置、角度、成对距离构成唯一"指纹"。

算法流程:
  1. 多帧雷达 → FRF过滤 → 合并点云
  2. 扫描侧: Douglas-Peucker简化 + 转向角检测 → 提取墙角
  3. 地图侧: HoughLinesP线段检测 + 交点提取 → 全图墙角预计算
  4. 全地图自由空间网格搜索:
     - 每个位置查询附近地图墙角
     - 距离矩阵兼容性筛选 + Umeyama求最优刚体变换
     - 匹配残差评分
  5. Top-K候选 → 似然场精细验证 + 面积匹配
  6. 输出最佳位姿 + 可视化

与前两种方法的区别:
  - area_segment: 基于区域面积+形状匹配 → 适合有明显房间结构的场景
  - contour_hu: 基于Hu矩轮廓相似度 → 适合任意形状，但易受180°镜像影响
  - corner_graph: 基于墙角拓扑图匹配 → 适合多墙角的室内环境，抗旋转

用法:
  python opencode_corner_graph_matcher.py
  python opencode_corner_graph_matcher.py --data path/to/debug_match_data.npz
  python opencode_corner_graph_matcher.py --grid-step 1.5
"""

import os
import sys
import math
import time
import argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] Need opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("[ERROR] Need matplotlib: pip install matplotlib")
    sys.exit(1)


if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()


# ============================================================
# 1. Data Loading
# ============================================================
def load_data(npz_path):
    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}")
        sys.exit(1)
    d = np.load(npz_path, allow_pickle=True)

    if 'map_resolution' in d:
        map_data = d['map_data']
        info = {
            'resolution': float(d['map_resolution']),
            'width': int(d['map_width']),
            'height': int(d['map_height']),
            'origin_x': float(d['map_origin_x']),
            'origin_y': float(d['map_origin_y']),
        }
    elif 'map_info' in d:
        map_data = d['map_data']
        mi = d['map_info'].item()
        info = {
            'resolution': float(mi['resolution']),
            'width': int(mi['width']),
            'height': int(mi['height']),
            'origin_x': float(mi['origin_x']),
            'origin_y': float(mi['origin_y']),
        }
    else:
        print("[ERROR] Unknown NPZ format"); sys.exit(1)

    tf_gt = d['tf_odom_to_map']
    frame_tfs = d['frame_tfs']
    angle_min = float(d.get('frame_angle_min', -math.pi))
    angle_inc = float(d.get('frame_angle_increment', 2*math.pi/len(d.get('frame_ranges_0', [0]*360))))

    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in d:
        frame_ranges.append(np.array(d[f'frame_ranges_{i}'], dtype=np.float64))
        i += 1

    print(f"{'='*60}")
    print(f"  Frames={len(frame_ranges)}, Map={info['width']}x{info['height']}")
    print(f"  GT(ref): ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f}°)")
    print(f"{'='*60}")
    return map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc


# ============================================================
# 2. FRF Filter + Merge (逐帧半径过滤)
# ============================================================
def frf_filter_per_frame(ranges, angle_min, angle_inc, bin_deg=2.0, gap_thresh=0.3):
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
        sorted_r = ranges[sorted_idx]
        gaps = np.diff(sorted_r) > gap_thresh
        if np.any(gaps):
            first_gap = int(np.argmax(gaps))
            keep[sorted_idx[first_gap+1:]] = False
    return valid & keep


def merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc,
                bin_deg=2.0, gap_thresh=0.3):
    total_raw, total_kept = 0, 0
    all_pts = []
    for fi, (ranges, tf) in enumerate(zip(frame_ranges, frame_tfs)):
        keep_mask = frf_filter_per_frame(ranges, angle_min, angle_inc, bin_deg, gap_thresh)
        total_raw += int(np.sum(ranges > 0.15))
        kept = int(np.sum(keep_mask))
        total_kept += kept
        if kept < 10:
            continue
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        lx = ranges[keep_mask] * np.cos(angles[keep_mask])
        ly = ranges[keep_mask] * np.sin(angles[keep_mask])
        tx, ty, yaw = tf
        c, s = math.cos(yaw), math.sin(yaw)
        all_pts.append(np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty]))
    if not all_pts:
        return np.empty((0, 2)), (0, 0), {}
    merged = np.vstack(all_pts)
    cx, cy = merged[:, 0].mean(), merged[:, 1].mean()
    stats = {'raw': total_raw, 'kept': total_kept, 'merged': len(merged)}
    print(f"  [Merge] raw={total_raw} → FRF={total_kept} → merged={len(merged)} pts")
    return merged, (cx, cy), stats


# ============================================================
# 3. Scan Corner Extraction (Douglas-Peucker + Turning Angle)
# ============================================================
def extract_scan_corners(points, min_turning_deg=20, epsilon_m=0.15):
    """
    从扫描点云提取墙角:
      1. 转极坐标 → 按角度排序 → 形成有序多边形
      2. Douglas-Peucker 简化
      3. 简化顶点中, 转向角 > min_turning_deg 的保留为墙角
    """
    if len(points) < 10:
        return []

    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    rel = points - np.array([cx, cy])

    # 按角度排序
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(angles)
    ordered = rel[order]

    # Douglas-Peucker 简化
    contour = ordered.reshape(-1, 1, 2).astype(np.float32)
    simplified = cv2.approxPolyDP(contour, epsilon_m, closed=True)
    simp_pts = simplified.reshape(-1, 2)

    if len(simp_pts) < 3:
        return []

    # 检测转向角
    corners = []
    n = len(simp_pts)
    for i in range(n):
        p_prev = simp_pts[(i - 1) % n]
        p_curr = simp_pts[i]
        p_next = simp_pts[(i + 1) % n]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 0.01 or l2 < 0.01:
            continue

        d1 = v1 / l1
        d2 = v2 / l2
        cos_a = np.clip(np.dot(d1, d2), -1, 1)
        interior_angle = math.degrees(np.arccos(cos_a))
        turning = 180 - interior_angle

        if abs(turning) > min_turning_deg:
            wall_dir_1 = math.atan2(-d1[1], -d1[0])
            wall_dir_2 = math.atan2(d2[1], d2[0])
            corners.append({
                'x': p_curr[0] + cx,
                'y': p_curr[1] + cy,
                'angle': interior_angle,
                'turning': turning,
                'wall_dirs': (wall_dir_1, wall_dir_2),
            })

    # 去重: 空间距离 < 0.5m 的墙角合并
    unique = []
    for c in corners:
        is_dup = any(math.sqrt((c['x']-u['x'])**2 + (c['y']-u['y'])**2) < 0.5
                     for u in unique)
        if not is_dup:
            unique.append(c)

    return unique


# ============================================================
# 4. Map Corner Extraction (HoughLinesP + Intersections)
# ============================================================
def extract_all_map_corners(map_data, info, min_corner_angle_deg=25):
    """预计算全地图所有墙角 (HoughLinesP + 线段交点)"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = map_data.shape

    wall = (map_data == 100).astype(np.uint8) * 255
    edges = cv2.Canny(wall, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                             minLineLength=30, maxLineGap=15)
    if lines is None or len(lines) < 2:
        return []

    arr = lines.reshape(-1, 4)
    # 限制线段数防止 O(n²) 爆炸
    if len(arr) > 200:
        lengths = np.sqrt((arr[:,2]-arr[:,0])**2 + (arr[:,3]-arr[:,1])**2)
        arr = arr[np.argsort(lengths)[-200:]]

    all_corners = []
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            x1, y1, x2, y2 = arr[i].astype(float)
            x3, y3, x4, y4 = arr[j].astype(float)

            # 检查端点距离 (至少有一对端点足够近)
            d13 = np.hypot(x1-x3, y1-y3)
            d14 = np.hypot(x1-x4, y1-y4)
            d23 = np.hypot(x2-x3, y2-y3)
            d24 = np.hypot(x2-x4, y2-y4)
            if min(d13, d14, d23, d24) > 40:  # pixel units
                continue

            # 求两线段交点
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-6:
                continue
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            ix, iy = x1 + t*(x2-x1), y1 + t*(y2-y1)

            if ix < 0 or ix >= W or iy < 0 or iy >= H:
                continue

            # 两线夹角
            v1 = np.array([x2-x1, y2-y1], float)
            v2 = np.array([x4-x3, y4-y3], float)
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue

            cos_a = np.clip(np.dot(v1/n1, v2/n2), -1, 1)
            angle = math.degrees(np.arccos(abs(cos_a)))
            if angle < min_corner_angle_deg:
                continue

            wx = ix*res + ox
            wy = (H-1-iy)*res + oy

            all_corners.append({
                'x': wx, 'y': wy, 'angle': angle,
                'wall_dirs': (math.atan2(v1[1],v1[0]), math.atan2(v2[1],v2[0])),
            })

    # 去重 + 限制总数
    if len(all_corners) > 300:
        all_corners.sort(key=lambda c: c['angle'], reverse=True)
        all_corners = all_corners[:300]

    print(f"  [Map Corners] HoughLinesP → {len(all_corners)} corners")
    return all_corners


def query_nearby_corners(all_corners, cx, cy, radius_m=10.0):
    """查询指定位置附近的地图墙角"""
    nearby = [c for c in all_corners
              if math.sqrt((c['x']-cx)**2 + (c['y']-cy)**2) < radius_m]
    if len(nearby) > 40:
        nearby.sort(key=lambda c: (c['x']-cx)**2 + (c['y']-cy)**2)
        nearby = nearby[:40]
    return nearby


# ============================================================
# 5. Corner Graph Matching (Umeyama)
# ============================================================
def umeyama(src, dst):
    """Umeyama算法: 求 src→dst 最优刚体变换 (允许均匀缩放)"""
    assert src.shape == dst.shape
    n, dim = src.shape
    if n < 2:
        return None

    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    H = (src_c.T @ dst_c) / n
    try:
        U, S, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None

    d = np.linalg.det(Vt.T @ U.T)
    sign_mat = np.eye(dim)
    sign_mat[-1, -1] = np.sign(d)
    R = Vt.T @ sign_mat @ U.T

    var_src = np.sum(src_c**2) / n
    scale = np.trace(np.diag(S) @ sign_mat) / max(var_src, 1e-10)
    scale = max(0.3, min(3.0, scale))
    t = mu_d - scale * R @ mu_s
    return R, t, scale


def match_corner_sets(scan_corners, map_corners, max_residual=2.0):
    """匹配扫描墙角集与地图墙角集 (距离矩阵 + Umeyama)"""
    if len(scan_corners) < 2 or len(map_corners) < 2:
        return None

    s_pts = np.array([[c['x'], c['y']] for c in scan_corners])
    m_pts = np.array([[c['x'], c['y']] for c in map_corners])
    n_s, n_m = len(s_pts), len(m_pts)

    # 成对距离矩阵
    s_dists = np.zeros((n_s, n_s))
    m_dists = np.zeros((n_m, n_m))
    for i in range(n_s):
        for j in range(n_s):
            s_dists[i,j] = np.linalg.norm(s_pts[i]-s_pts[j])
    for i in range(n_m):
        for j in range(n_m):
            m_dists[i,j] = np.linalg.norm(m_pts[i]-m_pts[j])

    # 兼容性检查 + 最佳匹配
    best_result = None
    best_score = -1e9
    max_pairs = min(n_s, n_m, 5)
    tol = 0.15

    for seed_i in range(n_s):
        for seed_j in range(n_m):
            # 找兼容的角对
            selected = [(seed_i, seed_j)]
            seen_s = {seed_i}
            seen_m = {seed_j}

            for i2 in range(n_s):
                if i2 in seen_s:
                    continue
                best_j2 = -1
                best_ratio = 1e9
                for j2 in range(n_m):
                    if j2 in seen_m:
                        continue
                    sd, md = s_dists[seed_i, i2], m_dists[seed_j, j2]
                    if sd < 0.1 or md < 0.1:
                        continue
                    ratio = abs(sd - md) / max(sd, md)
                    if ratio < tol and ratio < best_ratio:
                        best_ratio = ratio
                        best_j2 = j2
                if best_j2 >= 0 and len(selected) < max_pairs:
                    selected.append((i2, best_j2))
                    seen_s.add(i2)
                    seen_m.add(best_j2)

            if len(selected) < 2:
                continue

            src = np.array([s_pts[si] for si, _ in selected])
            dst = np.array([m_pts[mj] for _, mj in selected])
            result = umeyama(src, dst)
            if result is None:
                continue

            R, t, scale = result
            transformed = (R @ src.T).T * scale + t
            residuals = np.linalg.norm(transformed - dst, axis=1)
            mean_res = np.mean(residuals)

            if mean_res > max_residual:
                continue

            score = len(selected)*3.0 - mean_res*4.0
            if score > best_score:
                best_score = score
                yaw = math.atan2(R[1,0], R[0,0])
                best_result = {
                    'R': R, 't': t, 'scale': scale, 'yaw': yaw,
                    'residual': mean_res, 'n_pairs': len(selected),
                    'score': score,
                }

    return best_result


# ============================================================
# 6. Global Corner Search
# ============================================================
def global_corner_search(scan_corners, map_data, info, all_map_corners,
                         grid_step=2.0):
    """全地图网格搜索墙角匹配"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    mw_m = info['width']*res
    mh_m = info['height']*res
    H, W = info['height'], info['width']

    xs = np.arange(ox + 3, ox + mw_m - 3, grid_step)
    ys = np.arange(oy + 3, oy + mh_m - 3, grid_step)

    # 统计自由空间位置
    n_free = 0
    for ax in xs:
        for ay in ys:
            col = int((ax-ox)/res)
            row = int(H-1-(ay-oy)/res)
            if 0<=col<W and 0<=row<H and map_data[row,col]==0:
                n_free += 1

    print(f"\n  [Corner Search] Grid: {len(xs)}x{len(ys)} → {n_free} free-space positions")
    t0 = time.time()

    all_results = []
    count = 0
    for ax in xs:
        for ay in ys:
            col = int((ax-ox)/res)
            row = int(H-1-(ay-oy)/res)
            if col<0 or col>=W or row<0 or row>=H:
                continue
            if map_data[row,col] != 0:
                continue
            count += 1

            nearby = query_nearby_corners(all_map_corners, ax, ay, radius_m=10.0)
            if len(nearby) < 2:
                continue

            result = match_corner_sets(scan_corners, nearby)
            if result is not None:
                all_results.append({
                    'center': (ax, ay),
                    'result': result,
                    'score': result['score'],
                    'n_nearby': len(nearby),
                })

            if count % 100 == 0:
                print(f"    {count}/{n_free} ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Search done: {elapsed:.1f}s, {len(all_results)} matches")

    # 排序 + NMS
    all_results.sort(key=lambda x: x['score'], reverse=True)
    nms = []
    for r in all_results:
        cx, cy = r['center']
        dup = any(math.sqrt((cx-c['center'][0])**2+(cy-c['center'][1])**2) < 2.0
                  for c in nms)
        if not dup:
            nms.append(r)
            if len(nms) >= 15:
                break

    print(f"  Top-{min(5, len(nms))}:")
    for i, r in enumerate(nms[:5]):
        res_r = r['result']
        print(f"    #{i}: ({r['center'][0]:.1f},{r['center'][1]:.1f}) "
              f"pairs={res_r['n_pairs']} res={res_r['residual']:.2f}m "
              f"score={r['score']:.1f} scale={res_r['scale']:.2f}")

    return nms


# ============================================================
# 7. Likelihood Field + Area Refinement
# ============================================================
def build_likelihood_field(map_data, info, max_dist=3.0):
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(1-obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_px = max_dist/info['resolution']
    return np.clip(dist_px, 0, max_px).astype(np.float32) * info['resolution']


def validate_pose(points, cx, cy, yaw, map_data, info):
    """后验验证: 检查扫描点是否落在有效地图区域"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y * points[:, 0] - s_y * points[:, 1] + cx
    my = s_y * points[:, 0] + c_y * points[:, 1] + cy

    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)

    n_total = len(points)
    n_in_map = int(np.sum(valid))
    n_outside = n_total - n_in_map

    if n_in_map < n_total * 0.3:
        return False, {'free_pct': 0, 'occupied_pct': 0, 'unknown_pct': 1.0, 'n_in_map': n_in_map, 'n_total': n_total}

    cells = map_data[ri[valid], ci[valid]]
    n_free = int(np.sum(cells == 0))
    n_occupied = int(np.sum(cells == 100))
    n_unknown = int(np.sum(cells == -1)) + n_outside

    free_pct = n_free / n_total
    occupied_pct = n_occupied / n_total
    unknown_pct = n_unknown / n_total

    report = {'free_pct': free_pct, 'occupied_pct': occupied_pct, 'unknown_pct': unknown_pct,
              'n_in_map': n_in_map, 'n_total': n_total}

    if unknown_pct > 0.25 or occupied_pct > 0.25 or free_pct < 0.35:
        return False, report
    return True, report


def validate_and_filter(refined_list, points, map_data, info):
    """对精搜结果做后验验证, 过滤无效候选"""
    valid_list = []
    for r in refined_list:
        is_valid, report = validate_pose(points, r['x'], r['y'], r['yaw'], map_data, info)
        r['validation'] = report
        r['is_valid'] = is_valid
        if is_valid:
            valid_list.append(r)
        else:
            print(f"    [REJECTED] ({r['x']:.1f},{r['y']:.1f},{math.degrees(r['yaw']):.0f}°): "
                  f"free={report['free_pct']:.1%} occupied={report['occupied_pct']:.1%} unknown={report['unknown_pct']:.1%}")
    return valid_list


def score_at_pose(points_c, cx, cy, yaw, lf, info):
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y*points_c[:,0]-s_y*points_c[:,1]+cx
    my = s_y*points_c[:,0]+c_y*points_c[:,1]+cy
    ci = ((mx-ox)/res+0.5).astype(np.int32)
    ri = ((my-oy)/res+0.5).astype(np.int32)
    valid = (ci>=0)&(ci<W)&(ri>=0)&(ri<H)
    nv = int(np.sum(valid))
    if nv < len(points_c)*0.15:
        return -1e9, 0
    dists = lf[ri[valid], ci[valid]]
    hit = np.sum(dists<0.15)/nv
    return float(np.mean(np.exp(-dists**2/0.045))) + hit*0.5, hit


def refine_with_likelihood(candidates, points_c, scan_cx, scan_cy, lf, map_data, info):
    """对 Top-K 候选做似然场 + 面积精搜索"""
    if not candidates:
        return []

    H, W = info['height'], info['width']
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']

    pts_c = points_c.copy()
    pts_c[:,0] -= scan_cx
    pts_c[:,1] -= scan_cy
    ds = max(1, len(pts_c)//600)
    pts_ds = pts_c[::ds]

    # 扫描凸包 (用于面积过滤)
    hull_pts = cv2.convexHull(pts_c.astype(np.float32).reshape(-1,1,2)).reshape(-1,2)
    scan_hull_area = cv2.contourArea(hull_pts.astype(np.float32))

    refined = []
    for cand in candidates:
        result = cand['result']
        R, t, s = result['R'], result['t'], result['scale']

        # 扫描中心在 map 系中的位置
        robot_map = s*(R @ np.array([scan_cx, scan_cy])) + t
        cx_base, cy_base = float(robot_map[0]), float(robot_map[1])
        yaw_base = result['yaw']

        best_sc = -1e9
        best_pose = (cx_base, cy_base, yaw_base)

        for dx in np.arange(-2.0, 2.1, 0.5):
            for dy in np.arange(-2.0, 2.1, 0.5):
                for da_deg in np.arange(-20, 21, 5):
                    fx, fy = cx_base+dx, cy_base+dy
                    col, row = int((fx-ox)/res), int(H-1-(fy-oy)/res)
                    if col<0 or col>=W or row<0 or row>=H or map_data[row,col]!=0:
                        continue

                    fyaw = yaw_base + math.radians(da_deg)
                    sc, _ = score_at_pose(pts_ds, fx, fy, fyaw, lf, info)
                    if sc > best_sc:
                        best_sc = sc
                        best_pose = (fx, fy, fyaw)

        refined.append({
            'x': best_pose[0], 'y': best_pose[1], 'yaw': best_pose[2],
            'lf_score': best_sc,
            'n_pairs': result['n_pairs'],
            'corner_score': cand['score'],
            'total_score': cand['score']*3 + best_sc*10,
        })

    refined.sort(key=lambda x: x['total_score'], reverse=True)

    print(f"\n  [Refine] Top candidates:")
    for i, r in enumerate(refined[:5]):
        print(f"    #{i}: ({r['x']:.2f},{r['y']:.2f}) yaw={math.degrees(r['yaw']):.0f}° "
              f"pairs={r['n_pairs']} corner={r['corner_score']:.1f} lf={r['lf_score']:.3f}")
    return refined


# ============================================================
# 8. Visualization
# ============================================================
def create_visualization(map_data, info, tf_gt, points, scan_corners,
                         candidates, refined_results, stats, output_path):
    fig = plt.figure(figsize=(26, 18))
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox+W*res, oy, oy+H*res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data==0] = [1,1,1]
    map_bg[map_data==100] = [0.1,0.1,0.1]
    map_bg[map_data==-1] = [0.7,0.7,0.7]

    scan_cx, scan_cy = points[:,0].mean(), points[:,1].mean()

    # (a) Scan corners
    ax = fig.add_subplot(2, 3, 1)
    ax.set_aspect('equal')
    step = max(1, len(points)//2000)
    ax.scatter(points[::step,0], points[::step,1], s=0.5, c='gray', alpha=0.3)
    for ci, c in enumerate(scan_corners):
        ax.plot(c['x'], c['y'], 'rs', markersize=6)
        ax.annotate(f"#{ci}", (c['x'],c['y']), fontsize=6, color='red',
                    xytext=(3,3), textcoords='offset points')
        for wd in c['wall_dirs']:
            ax.arrow(c['x'], c['y'], 1.0*math.cos(wd), 1.0*math.sin(wd),
                    head_width=0.2, head_length=0.15, fc='red', ec='darkred', alpha=0.6)
    ax.plot(scan_cx, scan_cy, 'b+', markersize=10, mew=2)
    ax.set_title(f"(a) Scan Corners ({len(scan_corners)} found)")
    ax.grid(True, alpha=0.2)

    # (b) Map + candidates
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    colors = ['red','orange','green','cyan','magenta']
    for i, r in enumerate(refined_results[:5]):
        ax.plot(r['x'], r['y'], 'o', color=colors[i], markersize=10-i)
        ax.arrow(r['x'], r['y'], 2.0*math.cos(r['yaw']), 2.0*math.sin(r['yaw']),
                head_width=0.4, head_length=0.3, fc=colors[i], ec=colors[i])
        ax.annotate(f"#{i}", (r['x'],r['y']), fontsize=7, color=colors[i], fontweight='bold')
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=14, label='GT')
    ax.legend(fontsize=8)
    ax.set_title("(b) Top-5 Corner Matches")
    ax.grid(True, alpha=0.1)

    # (c) Corner search grid candidates
    ax = fig.add_subplot(2, 3, 3)
    ax.set_aspect('equal')
    ranks = range(min(5, len(refined_results)))
    scores = [r['corner_score'] for r in refined_results[:5]]
    y_labels = [f"#{i}\npairs={r['n_pairs']}\n{math.degrees(r['yaw']):.0f}°"
                for i, r in enumerate(refined_results[:5])]
    ax.barh(list(ranks), scores, color=['red','orange','green','cyan','magenta'][:len(ranks)])
    ax.set_yticks(list(ranks))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Corner Match Score")
    ax.set_title("(c) Corner Graph Scores")

    # (d) Best overlay
    ax = fig.add_subplot(2, 3, 4)
    if refined_results:
        best = refined_results[0]
        bx, by, byaw = best['x'], best['y'], best['yaw']
        zoom = 12
        ax.set_xlim(bx-zoom, bx+zoom)
        ax.set_ylim(by-zoom, by+zoom)
        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.set_aspect('equal')
        c_b, s_b = math.cos(byaw), math.sin(byaw)
        aligned = np.column_stack([
            c_b*points[:,0]-s_b*points[:,1]+bx,
            s_b*points[:,0]+c_b*points[:,1]+by])
        step_v = max(1, len(aligned)//2000)
        ax.scatter(aligned[::step_v,0], aligned[::step_v,1], s=2, c='lime', alpha=0.6)
        ax.plot(bx, by, 'r+', markersize=14, mew=3)
        ax.arrow(bx, by, 2.0*math.cos(byaw), 2.0*math.sin(byaw),
                head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=2.5, zorder=10)
        ax.set_title(f"(d) Best: ({bx:.2f},{by:.2f}) {math.degrees(byaw):.0f}°")
    ax.grid(True, alpha=0.12)

    # (e) GT comparison
    ax = fig.add_subplot(2, 3, 5)
    if refined_results and tf_gt is not None:
        best = refined_results[0]
        bx, by, byaw = best['x'], best['y'], best['yaw']
        zoom = 12
        ax.set_xlim(tf_gt[0]-zoom, tf_gt[0]+zoom)
        ax.set_ylim(tf_gt[1]-zoom, tf_gt[1]+zoom)
        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.set_aspect('equal')
        c_g, s_g = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        gt_aligned = np.column_stack([
            c_g*points[:,0]-s_g*points[:,1]+tf_gt[0],
            s_g*points[:,0]+c_g*points[:,1]+tf_gt[1]])
        step_v = max(1, len(gt_aligned)//2000)
        ax.scatter(gt_aligned[::step_v,0], gt_aligned[::step_v,1], s=2, c='cyan', alpha=0.5, label='GT')
        c_b, s_b = math.cos(byaw), math.sin(byaw)
        est_aligned = np.column_stack([
            c_b*points[:,0]-s_b*points[:,1]+bx,
            s_b*points[:,0]+c_b*points[:,1]+by])
        ax.scatter(est_aligned[::step_v,0], est_aligned[::step_v,1], s=2, c='lime', alpha=0.5, label='Est')
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=12, label='GT pos')
        ax.plot(bx, by, 'r+', markersize=12, mew=2, label='Est pos')
        err = math.sqrt((bx-tf_gt[0])**2+(by-tf_gt[1])**2)
        yaw_err = math.degrees(abs(math.atan2(math.sin(byaw-tf_gt[2]), math.cos(byaw-tf_gt[2]))))
        ax.set_title(f"(e) GT vs Est: dist={err:.2f}m, yaw={yaw_err:.1f}°")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)

    # (f) Report
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    rep = ["=== Corner Graph Match ===", ""]
    rep.append(f"FRF: {stats.get('raw',0)} → {stats.get('merged',0)} pts")
    rep.append(f"Scan corners: {len(scan_corners)}")
    rep.append(f"Map corners (global): listed")
    rep.append(f"Corner candidates: {len(candidates)}")
    rep.append("")
    if refined_results:
        best = refined_results[0]
        rep.append(f"Best: ({best['x']:.3f}, {best['y']:.3f}, {math.degrees(best['yaw']):.1f}°)")
        rep.append(f"  Pairs={best['n_pairs']}, Corner={best['corner_score']:.1f}, LF={best['lf_score']:.3f}")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2+(best['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                   math.cos(best['yaw']-tf_gt[2]))))
            rep.append(f"  GT: ({tf_gt[0]:.3f},{tf_gt[1]:.3f},{math.degrees(tf_gt[2]):.1f}°)")
            rep.append(f"  Error: dist={err:.3f}m, yaw={yaw_err:.1f}°")
    rep.append("")
    rep.append("Method: Corner graph + Umeyama")
    rep.append("  → Douglas-Peucker scan corners")
    rep.append("  → HoughLinesP map corners")
    rep.append("  → distance matrix compatibility")
    rep.append("  → Umeyama rigid transform")
    rep.append("  → likelihood field refinement")
    ax.text(0.05, 0.95, "\n".join(rep), transform=ax.transAxes,
            fontfamily='monospace', fontsize=8.5, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.55))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  [Output] PNG: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Corner Graph Matcher')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--grid-step', type=float, default=2.0, help='Corner search grid step (m)')
    parser.add_argument('--min-turning', type=float, default=20, help='Min turning angle for corner (deg)')
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    if args.output and args.output.endswith('.png'):
        output_path = args.output
    else:
        output_dir = args.output or os.path.dirname(npz_path)
        if not os.path.isdir(output_dir):
            output_dir = os.path.dirname(npz_path)
        output_path = os.path.join(output_dir, 'corner_graph_match_result.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 1. Load
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(npz_path)

    # 2. Merge
    print("\n" + "="*60)
    print("Step 1: FRF Filter + Merge")
    print("="*60)
    points, (scan_cx, scan_cy), stats = merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc)

    # 3. Extract scan corners
    print("\n" + "="*60)
    print("Step 2: Extract Scan Corners")
    print("="*60)
    scan_corners = extract_scan_corners(points, min_turning_deg=args.min_turning)
    print(f"  Found {len(scan_corners)} scan corners:")
    for i, c in enumerate(scan_corners[:10]):
        print(f"    #{i}: ({c['x']:.2f},{c['y']:.2f}) angle={c['angle']:.0f}° turn={c['turning']:.0f}°")
    if len(scan_corners) < 2:
        print("[ERROR] Too few scan corners"); sys.exit(1)

    # 4. Precompute map corners
    print("\n" + "="*60)
    print("Step 3: Precompute Map Corners")
    print("="*60)
    all_map_corners = extract_all_map_corners(map_data, info)
    if len(all_map_corners) < 2:
        print("[ERROR] Too few map corners"); sys.exit(1)

    # 5. Global corner search
    print("\n" + "="*60)
    print("Step 4: Global Corner Graph Search")
    print("="*60)
    candidates = global_corner_search(scan_corners, map_data, info, all_map_corners,
                                      grid_step=args.grid_step)
    if not candidates:
        print("[ERROR] No corner matches found"); sys.exit(1)

    # 6. Likelihood refinement
    print("\n" + "="*60)
    print("Step 5: Likelihood Field Refinement")
    print("="*60)
    lf = build_likelihood_field(map_data, info)
    refined = refine_with_likelihood(candidates, points, scan_cx, scan_cy, lf, map_data, info)

    # 6. Validate
    print("\n" + "="*60)
    print("Step 6: Validate & Filter Results")
    print("="*60)
    valid_refined = validate_and_filter(refined, points, map_data, info)
    if not valid_refined:
        print("  [WARN] All refined candidates failed validation! Showing best invalid result.")
        valid_refined = refined

    # 7. Results
    print("\n" + "="*60)
    print("Final Result")
    print("="*60)
    if valid_refined:
        best = valid_refined[0]
        print(f"  Position: ({best['x']:.3f}, {best['y']:.3f})")
        print(f"  Yaw: {math.degrees(best['yaw']):.1f}°")
        print(f"  Corner pairs: {best['n_pairs']}")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2 + (best['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                   math.cos(best['yaw']-tf_gt[2]))))
            print(f"  GT: ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}, {math.degrees(tf_gt[2]):.1f}°)")
            print(f"  Error: dist={err:.3f}m, yaw={yaw_err:.1f}°")
    else:
        print("  [Failed] No valid match")

    # 8. Visualize
    create_visualization(map_data, info, tf_gt, points, scan_corners,
                         candidates, valid_refined, stats, output_path)


if __name__ == '__main__':
    main()
