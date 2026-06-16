#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opencode_contour_hu_matcher.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
全地图滑窗 + Hu矩轮廓形状匹配 定位算法

核心思路:
  1. 多帧雷达数据合并 → FRF过滤 → 生成闭合多边形轮廓
  2. 在全地图上以固定步长滑窗，每个窗口提取地图墙壁轮廓
  3. cv2.matchShapes (Hu矩) 比较扫描轮廓与地图轮廓的相似度
  4. NMS 去冗余 → Top-K 候选
  5. 似然场精细搜索 (位置+角度)
  6. 光线投射180°消歧 + ICP 精调

与 area_segment_matcher 的区别:
  - area_segment: 先分割地图为连通域，再用面积+形状预筛选
  - contour_hu: 全地图逐点滑窗，直接用Hu矩比较 → 更全局，但更慢

用法:
  python opencode_contour_hu_matcher.py
  python opencode_contour_hu_matcher.py --data path/to/debug_match_data.npz
  python opencode_contour_hu_matcher.py --step 1.0 --n-keep 8
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

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed, KDTree fallback will be used")


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
    print(f"  GT: ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f}°)")
    print(f"{'='*60}")
    return map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc


# ============================================================
# 2. FRF Filter + Merge
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
                bin_deg=2.0, gap_thresh=0.3, outlier_radius=0.3, min_neighbors=3):
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
        pts = np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty])
        all_pts.append(pts)

    if not all_pts:
        return np.empty((0, 2)), (0, 0)

    merged = np.vstack(all_pts)
    if HAS_SCIPY and len(merged) > min_neighbors + 1:
        tree = cKDTree(merged)
        counts = tree.query_ball_point(merged, outlier_radius, return_length=True)
        mask = np.array(counts) >= min_neighbors
        merged = merged[mask]

    cx, cy = merged[:, 0].mean(), merged[:, 1].mean()
    print(f"  [Merge] raw={total_raw} → FRF={total_kept} → final={len(merged)} pts")
    return merged, (cx, cy)


# ============================================================
# 3. Scan Contour Creation (Closed Polygon)
# ============================================================
def create_scan_contour(points, img_size=250, window_size_m=22.0):
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    centered = points.copy()
    centered[:, 0] -= cx
    centered[:, 1] -= cy

    mpp = window_size_m / img_size
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    half = img_size // 2

    angles = np.arctan2(centered[:, 1], centered[:, 0])
    order = np.argsort(angles)
    ordered = centered[order]

    pts_px = []
    for p in ordered:
        px = int(p[0] / mpp + half)
        py = int(half - p[1] / mpp)
        if 0 <= px < img_size and 0 <= py < img_size:
            pts_px.append([px, py])

    if len(pts_px) < 3:
        return None, img, centered

    pts_arr = np.array(pts_px, dtype=np.int32)
    cv2.polylines(img, [pts_arr], isClosed=True, color=255, thickness=1)
    cv2.fillPoly(img, [pts_arr], 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, img, centered

    main_contour = max(contours, key=cv2.contourArea)
    area_px = cv2.contourArea(main_contour)
    hull = cv2.convexHull(main_contour)
    solidity = area_px / max(cv2.contourArea(hull), 1.0)

    print(f"  Scan contour: area={area_px:.0f}px², solidity={solidity:.2f}")
    return main_contour, img, centered


# ============================================================
# 4. Extract Map Contour at Position (Flood Fill Method)
# ============================================================
def extract_map_contour_at(x, y, map_data, info, window_size_m=22.0):
    """
    在地图位置(x,y)提取房间轮廓。
    使用泛洪填充从中心点填充自由空间，提取其外轮廓。
    """
    res = info['resolution']
    half_w_px = max(int(window_size_m / 2 / res), 10)

    cx_px = int((x - info['origin_x']) / res)
    cy_px = int(info['height'] - 1 - (y - info['origin_y']) / res)

    r1 = max(0, cy_px - half_w_px)
    r2 = min(info['height'], cy_px + half_w_px)
    c1 = max(0, cx_px - half_w_px)
    c2 = min(info['width'], cx_px + half_w_px)

    if r2 - r1 < 15 or c2 - c1 < 15:
        return None

    roi = map_data[r1:r2, c1:c2]
    h_roi, w_roi = roi.shape
    cy_r, cx_r = h_roi // 2, w_roi // 2

    # 找自由空间种子点
    if roi[cy_r, cx_r] != 0:
        found = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = cy_r + dy, cx_r + dx
                if 0 <= ny < h_roi and 0 <= nx < w_roi and roi[ny, nx] == 0:
                    cy_r, cx_r = ny, nx
                    found = True
                    break
            if found:
                break
        if not found:
            return None

    free_binary = (roi == 0).astype(np.uint8) * 255
    mask = np.zeros((h_roi + 2, w_roi + 2), dtype=np.uint8)
    cv2.floodFill(free_binary, mask, (cx_r, cy_r), 128)

    room_mask = (free_binary == 128).astype(np.uint8) * 255
    contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 20:
        return None
    return main_contour


# ============================================================
# 5. Sliding Window Hu Moment Search
# ============================================================
def sliding_window_hu_search(scan_contour, map_data, info, window_size_m=22.0,
                             step_m=1.0, n_keep=8, nms_radius=3.5):
    """全地图滑窗 Hu 矩形状匹配"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    mw_m = info['width'] * res
    mh_m = info['height'] * res
    margin = window_size_m / 2

    xs = np.arange(ox + margin, ox + mw_m - margin, step_m)
    ys = np.arange(oy + margin, oy + mh_m - margin, step_m)
    n_total = len(xs) * len(ys)

    scan_area = cv2.contourArea(scan_contour)

    print(f"\n  [Hu Search] {len(xs)}x{len(ys)} = {n_total} positions...")
    t0 = time.time()

    all_evals = []
    count = 0

    for x_val in xs:
        for y_val in ys:
            count += 1
            map_contour = extract_map_contour_at(x_val, y_val, map_data, info, window_size_m)
            if map_contour is None:
                continue

            # 面积预筛选: 地图轮廓面积不能差太多
            map_area = cv2.contourArea(map_contour)
            area_ratio = min(scan_area, map_area) / max(scan_area, map_area, 1)
            if area_ratio < 0.15:
                continue

            # Hu 矩匹配
            hu_dist = cv2.matchShapes(scan_contour, map_contour, cv2.CONTOURS_MATCH_I2, 0)
            score = 1.0 - min(hu_dist, 1.0)
            all_evals.append((score, hu_dist, x_val, y_val, map_contour))

            if count % 200 == 0:
                print(f"    Progress: {count}/{n_total} ({time.time()-t0:.1f}s)")

    if not all_evals:
        print("  [WARN] No valid matches found!")
        return []

    # 排序 + NMS 去冗余
    all_evals.sort(key=lambda x: x[0], reverse=True)
    candidates = []
    for ev in all_evals:
        score, hu_dist, x, y, _ = ev
        is_dup = any(math.sqrt((x-c['x'])**2 + (y-c['y'])**2) < nms_radius
                     for c in candidates)
        if not is_dup:
            candidates.append({'x': x, 'y': y, 'score': score, 'hu_dist': hu_dist})
            if len(candidates) >= n_keep * 2:
                break

    elapsed = time.time() - t0
    print(f"  Hu search done: {elapsed:.1f}s, {len(candidates)} candidates after NMS")
    for i, c in enumerate(candidates[:n_keep]):
        print(f"    #{i}: ({c['x']:.1f},{c['y']:.1f}) score={c['score']:.3f} hu_dist={c['hu_dist']:.3f}")

    return candidates[:n_keep]


# ============================================================
# 6. Likelihood Field + Fine Search + 180° Disambiguation
# ============================================================
def build_likelihood_field(map_data, info, max_dist=3.0):
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_px = max_dist / info['resolution']
    return np.clip(dist_px, 0, max_px).astype(np.float32) * info['resolution']


def score_at_pose(points_c, cx, cy, yaw, lf, info):
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y * points_c[:, 0] - s_y * points_c[:, 1] + cx
    my = s_y * points_c[:, 0] + c_y * points_c[:, 1] + cy

    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)

    nv = int(np.sum(valid))
    if nv < len(points_c) * 0.15:
        return -1e9, 0

    dists = lf[ri[valid], ci[valid]]
    hit_rate = np.sum(dists < 0.15) / nv
    lf_sc = float(np.mean(np.exp(-dists**2 / 0.045)))
    return lf_sc + hit_rate * 0.5, hit_rate


def ray_cast_score(points_c, cx, cy, yaw, map_data, info, n_rays=36):
    res, ox, oy = info['resolution'], info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    ray_angles = np.linspace(0, 2*math.pi, n_rays, endpoint=False)
    errors = []

    for ray_ang in ray_angles:
        ang_diff = np.abs(np.arctan2(
            np.sin(np.arctan2(points_c[:,1], points_c[:,0]) - ray_ang),
            np.cos(np.arctan2(points_c[:,1], points_c[:,0]) - ray_ang)))
        near = ang_diff < math.radians(12)
        if not np.any(near):
            continue
        actual_range = np.min(np.sqrt(points_c[near, 0]**2 + points_c[near, 1]**2))

        map_ang = ray_ang + yaw
        dx_r, dy_r = math.cos(map_ang)*res, math.sin(map_ang)*res
        px, py = cx, cy
        sim_range = 30.0
        for _ in range(int(30.0/res)):
            col, row = int((px-ox)/res), int(H-1-(py-oy)/res)
            if col<0 or col>=W or row<0 or row>=H:
                break
            if map_data[row, col] == 100:
                sim_range = math.sqrt((px-cx)**2+(py-cy)**2)
                break
            px += dx_r; py += dy_r
        errors.append(abs(actual_range - sim_range))
    return np.mean(errors) if errors else 99.0


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


def fine_search_around_candidates(candidates, points_c, lf, map_data, info,
                                  pos_radius=1.5, pos_step=0.2, angle_step_deg=4.0):
    """对每个 Hu 候选做局部精细搜索 + 180°消歧"""
    print(f"\n  [Fine Search] Refining {len(candidates)} candidates...")
    t0 = time.time()

    ds = max(1, len(points_c)//800)
    pts_ds = points_c[::ds]

    refined = []
    for rank, cand in enumerate(candidates):
        hx, hy = cand['x'], cand['y']
        best_score = -1e9
        best_pose = (hx, hy, 0.0)

        angle_step_int = int(angle_step_deg)
        for fyaw_deg in range(0, 360, angle_step_int):
            fyaw = math.radians(fyaw_deg)
            for dx in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
                for dy in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
                    ax, ay = hx+dx, hy+dy
                    col = int((ax-info['origin_x'])/info['resolution'])
                    row = int(info['height']-1-(ay-info['origin_y'])/info['resolution'])
                    if col<0 or col>=info['width'] or row<0 or row>=info['height']:
                        continue
                    if map_data[row, col] != 0:
                        continue
                    sc, _ = score_at_pose(pts_ds, ax, ay, fyaw, lf, info)
                    if sc > best_score:
                        best_score = sc
                        best_pose = (ax, ay, fyaw)

        # 180° 消歧 (增强版: 似然场 + 光线投射 + 自由空间覆盖率)
        bx, by, byaw = best_pose
        byaw_alt = byaw + math.pi if byaw < 0 else byaw - math.pi
        sc1, _ = score_at_pose(pts_ds, bx, by, byaw, lf, info)
        sc2, _ = score_at_pose(pts_ds, bx, by, byaw_alt, lf, info)
        mae1 = ray_cast_score(pts_ds, bx, by, byaw, map_data, info)
        mae2 = ray_cast_score(pts_ds, bx, by, byaw_alt, map_data, info)
        # 自由空间覆盖率 (关键新增: 180°镜像可能在灰色区域)
        _, v1 = validate_pose(points_c, bx, by, byaw, map_data, info)
        _, v2 = validate_pose(points_c, bx, by, byaw_alt, map_data, info)
        free_bonus1 = v1['free_pct'] * 2.0
        free_bonus2 = v2['free_pct'] * 2.0

        if (sc2 + free_bonus2) - mae2*0.03 > (sc1 + free_bonus1) - mae1*0.03:
            byaw, best_score = byaw_alt, sc2

        refined.append({
            'x': bx, 'y': by, 'yaw': byaw,
            'lf_score': best_score,
            'hu_dist': cand['hu_dist'],
            'total_score': best_score - cand['hu_dist'] * 5,
        })

    refined.sort(key=lambda x: x['total_score'], reverse=True)
    elapsed = time.time() - t0
    print(f"  Fine search done: {elapsed:.1f}s")
    for i, r in enumerate(refined[:5]):
        print(f"    #{i}: ({r['x']:.2f},{r['y']:.2f}) yaw={math.degrees(r['yaw']):.0f}°, "
              f"lf={r['lf_score']:.3f}, hu={r['hu_dist']:.3f}, total={r['total_score']:.2f}")
    return refined


# ============================================================
# 7. ICP Refinement
# ============================================================
def icp_refine(src_pts, tgt_tree, max_iter=30, tol=1e-4, outlier_ratio=2.0):
    src = src_pts.copy()
    R_total, t_total = np.eye(2), np.zeros(2)
    for it in range(max_iter):
        dists, idx = tgt_tree.query(src)
        med = np.median(dists)
        mask = dists < max(0.2, med * outlier_ratio)
        if np.sum(mask) < 10:
            break
        s_pts, m_pts = src[mask], tgt_tree.data[idx[mask]]
        cs, cm = s_pts.mean(0), m_pts.mean(0)
        H = (s_pts-cs).T @ (m_pts-cm)
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            break
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        t = cm - R @ cs
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        if np.linalg.norm(t) < tol:
            break
    return R_total, t_total


# ============================================================
# 8. Visualization
# ============================================================
def create_visualization(map_data, info, tf_gt, points, scan_contour, scan_img, pts_c,
                         candidates, refined_results, output_path):
    fig = plt.figure(figsize=(26, 18))
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox+W*res, oy, oy+H*res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1, 1, 1]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.7, 0.7, 0.7]

    # (a) Scan contour
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(scan_img, cmap='gray_r', origin='upper')
    ax.set_title(f"(a) Scan Contour\nArea={cv2.contourArea(scan_contour):.0f}px²")
    ax.axis('off')

    # (b) Hu candidates on map
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    for i, c in enumerate(candidates[:8]):
        color = plt.cm.viridis(i/8)
        ax.plot(c['x'], c['y'], 'o', color=color, markersize=8-i*0.5)
        ax.annotate(f"#{i}", (c['x'], c['y']), fontsize=6, color=color)
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=15, label='GT')
    ax.set_title(f"(b) Hu Shape Candidates ({len(candidates)} after NMS)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # (c) Refined scores
    ax = fig.add_subplot(2, 3, 3)
    ranks = range(min(5, len(refined_results)))
    scores = [r['total_score'] for r in refined_results[:5]]
    y_labels = [f"#{i}\n{math.degrees(r['yaw']):.0f}°\nhu={r['hu_dist']:.3f}"
                for i, r in enumerate(refined_results[:5])]
    ax.barh(list(ranks), scores, color=['red','orange','green','cyan','magenta'][:len(ranks)])
    ax.set_yticks(list(ranks))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Total Score")
    ax.set_title("(c) Refined Candidate Scores")

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
        ax.plot(bx, by, 'r+', markersize=15, mew=3)
        ax.arrow(bx, by, 2.0*math.cos(byaw), 2.0*math.sin(byaw),
                 head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=2.5, zorder=10)
        ax.set_title(f"(d) Best Match\n({bx:.2f},{by:.2f}) yaw={math.degrees(byaw):.0f}°")
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
        # GT scan overlay
        c_g, s_g = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        gt_aligned = np.column_stack([
            c_g*points[:,0]-s_g*points[:,1]+tf_gt[0],
            s_g*points[:,0]+c_g*points[:,1]+tf_gt[1]])
        step_v = max(1, len(gt_aligned)//2000)
        ax.scatter(gt_aligned[::step_v,0], gt_aligned[::step_v,1], s=2, c='cyan', alpha=0.5, label='GT')
        # Our result
        c_b, s_b = math.cos(byaw), math.sin(byaw)
        est_aligned = np.column_stack([
            c_b*points[:,0]-s_b*points[:,1]+bx,
            s_b*points[:,0]+c_b*points[:,1]+by])
        ax.scatter(est_aligned[::step_v,0], est_aligned[::step_v,1], s=2, c='lime', alpha=0.5, label='Est')
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=12, label='GT pos')
        ax.plot(bx, by, 'r+', markersize=12, mew=2, label='Est pos')
        err = math.sqrt((bx-tf_gt[0])**2 + (by-tf_gt[1])**2)
        yaw_err = math.degrees(abs(math.atan2(math.sin(byaw-tf_gt[2]), math.cos(byaw-tf_gt[2]))))
        ax.set_title(f"(e) GT vs Est\nErr: dist={err:.2f}m, yaw={yaw_err:.1f}°")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)

    # (f) Report
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    rep = ["=== Contour Hu Moment Match ===", ""]
    rep.append(f"Scan: {len(points)} pts, contour area={cv2.contourArea(scan_contour):.0f}px²")
    rep.append(f"Hu search: {len(candidates)} candidates")
    rep.append("")
    if refined_results:
        best = refined_results[0]
        rep.append(f"Best: ({best['x']:.3f}, {best['y']:.3f}, {math.degrees(best['yaw']):.1f}°)")
        rep.append(f"  LF={best['lf_score']:.3f}, Hu={best['hu_dist']:.3f}")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2 + (best['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                   math.cos(best['yaw']-tf_gt[2]))))
            rep.append(f"  vs GT: dist={err:.3f}m, yaw={yaw_err:.1f}°")
            rep.append(f"  GT: ({tf_gt[0]:.3f},{tf_gt[1]:.3f},{math.degrees(tf_gt[2]):.1f}°)")
    rep.append("")
    rep.append("Method: cv2.matchShapes (Hu moments)")
    rep.append("  → flood-fill map contours")
    rep.append("  → slide window + shape matching")
    rep.append("  → likelihood field fine search")
    rep.append("  → ray casting 180° disambiguation")
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
    parser = argparse.ArgumentParser(description='Contour Hu Moment Shape Matcher')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--step', type=float, default=1.0, help='Sliding window step size (m)')
    parser.add_argument('--n-keep', type=int, default=8, help='Candidates after NMS')
    parser.add_argument('--window', type=float, default=22.0, help='Window size (m)')
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    if args.output and args.output.endswith('.png'):
        output_path = args.output
    else:
        output_dir = args.output or os.path.dirname(npz_path)
        if not os.path.isdir(output_dir):
            output_dir = os.path.dirname(npz_path)
        output_path = os.path.join(output_dir, 'contour_hu_match_result.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 1. Load
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(npz_path)

    # 2. Merge
    print("\n" + "="*60)
    print("Step 1: Merge Scans")
    print("="*60)
    points, (scan_cx, scan_cy) = merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc)

    # 3. Create scan contour
    print("\n" + "="*60)
    print("Step 2: Create Scan Contour")
    print("="*60)
    scan_contour, scan_img, pts_c = create_scan_contour(points, window_size_m=args.window)
    if scan_contour is None:
        print("[ERROR] Failed to create scan contour"); sys.exit(1)

    # 4. Hu moment sliding window search
    print("\n" + "="*60)
    print("Step 3: Hu Moment Sliding Window Search")
    print("="*60)
    candidates = sliding_window_hu_search(scan_contour, map_data, info,
                                          window_size_m=args.window,
                                          step_m=args.step, n_keep=args.n_keep)
    if not candidates:
        print("[ERROR] No candidates found"); sys.exit(1)

    # 5. Fine search + 180° disambiguation
    print("\n" + "="*60)
    print("Step 4: Fine Search + 180° Disambiguation")
    print("="*60)
    lf = build_likelihood_field(map_data, info)
    refined = fine_search_around_candidates(candidates, pts_c, lf, map_data, info)

    # 6. Validate results
    print("\n" + "="*60)
    print("Step 6: Validate & Filter Results")
    print("="*60)
    valid_refined = validate_and_filter(refined, points, map_data, info)
    if not valid_refined:
        print("  [WARN] All refined candidates failed validation! Showing best invalid result.")
        valid_refined = refined

    # 7. ICP refinement on best
    if valid_refined and HAS_SCIPY:
        print("\n" + "="*60)
        print("Step 7: ICP Refinement")
        print("="*60)
        best = valid_refined[0]
        bx, by, byaw = best['x'], best['y'], best['yaw']
        c_b, s_b = math.cos(byaw), math.sin(byaw)
        pts_in_map = np.column_stack([
            c_b*points[:,0]-s_b*points[:,1]+bx,
            s_b*points[:,0]+c_b*points[:,1]+by])

        wall_ys, wall_xs = np.where(map_data == 100)
        map_walls = np.column_stack([
            wall_xs*info['resolution']+info['origin_x'],
            (info['height']-1-wall_ys)*info['resolution']+info['origin_y']])
        map_tree = cKDTree(map_walls[::max(1, len(map_walls)//5000)])

        R_icp, t_icp = icp_refine(pts_in_map, map_tree)
        final_R = R_icp @ np.array([[c_b, -s_b], [s_b, c_b]])
        final_t = R_icp @ np.array([bx, by]) + t_icp
        final_yaw = math.atan2(final_R[1,0], final_R[0,0])

        # 重新计算扫描质心
        sc_map = final_R @ np.array([scan_cx, scan_cy]) + final_t
        refined[0]['x'] = sc_map[0]
        refined[0]['y'] = sc_map[1]
        refined[0]['yaw'] = final_yaw
        print(f"  ICP refined: ({sc_map[0]:.3f},{sc_map[1]:.3f}) yaw={math.degrees(final_yaw):.1f}°")

    # 8. Results
    print("\n" + "="*60)
    print("Final Result")
    print("="*60)
    if valid_refined:
        best = valid_refined[0]
        print(f"  Position: ({best['x']:.3f}, {best['y']:.3f})")
        print(f"  Yaw: {math.degrees(best['yaw']):.1f}°")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2 + (best['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                   math.cos(best['yaw']-tf_gt[2]))))
            print(f"  GT: ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}, {math.degrees(tf_gt[2]):.1f}°)")
            print(f"  Error: dist={err:.3f}m, yaw={yaw_err:.1f}°")

    # 9. Visualize
    create_visualization(map_data, info, tf_gt, points, scan_contour, scan_img, pts_c,
                         candidates, valid_refined, output_path)


if __name__ == '__main__':
    main()
