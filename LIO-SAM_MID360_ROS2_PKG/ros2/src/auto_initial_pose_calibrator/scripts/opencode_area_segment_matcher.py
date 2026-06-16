#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opencode_area_segment_matcher.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
地图连通域分割 + 面积/形状匹配 定位算法

核心思路:
  1. 将栅格地图的自由空间(白色区域)按黑线(墙壁)切分为独立连通域
  2. 每个连通域 = 一个房间/走廊，提取其轮廓、面积、长宽比等几何特征
  3. 扫描点云合成闭合轮廓，同样提取几何特征
  4. 用面积比 + 长宽比 + 轮廓形状(Hu矩) 三特征联合评分
  5. Top-K 连通域内做似然场精细搜索 + 180°消歧

与 contour_hu_matcher 的区别:
  - contour_hu: 全地图滑窗，每窗提取地图轮廓，与扫描轮廓做 Hu 矩匹配
  - area_segment: 先分割地图为独立区域，用面积+形状预筛选，再在区域内精搜
    → 优势: 大幅减少搜索空间, 利用拓扑连通性避免误匹配

用法:
  python opencode_area_segment_matcher.py
  python opencode_area_segment_matcher.py --data path/to/debug_match_data.npz
  python opencode_area_segment_matcher.py --data data.npz --output result.png --min-area 30
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
    from matplotlib.patches import Rectangle as MplRect, FancyArrowPatch
except ImportError:
    print("[ERROR] Need matplotlib: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed, using fallback nearest-neighbor")


# ============================================================
# 0. 字体设置
# ============================================================
# Windows GBK encoding fix
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()


# ============================================================
# 1. 数据加载 (兼容两种 NPZ 格式)
# ============================================================
def load_data(npz_path):
    """加载 debug_match_data.npz, 自动检测格式"""
    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}")
        sys.exit(1)

    d = np.load(npz_path, allow_pickle=True)

    # 检测格式
    if 'map_resolution' in d:
        # 格式A: 标量字段
        map_data = d['map_data']
        info = {
            'resolution': float(d['map_resolution']),
            'width': int(d['map_width']),
            'height': int(d['map_height']),
            'origin_x': float(d['map_origin_x']),
            'origin_y': float(d['map_origin_y']),
        }
    elif 'map_info' in d:
        # 格式B: 字典字段
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

    n_frames = len(frame_ranges)
    n_beams = len(frame_ranges[0]) if frame_ranges else 0
    print(f"{'='*60}")
    print(f"  Data: {n_frames} frames x {n_beams} beams")
    print(f"  Map: {info['width']}x{info['height']} @ {info['resolution']:.3f}m/pix")
    print(f"  Origin: ({info['origin_x']:.2f}, {info['origin_y']:.2f})")
    print(f"  GT(ref): ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f}°)")
    print(f"  Scan FOV: [{math.degrees(angle_min):.1f}°, {math.degrees(angle_min + n_beams*angle_inc):.1f}°]")
    print(f"{'='*60}")

    return map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc


# ============================================================
# 2. FRF过滤 + 合并 (逐帧半径过滤去玻璃/动态点)
# ============================================================
def frf_filter_per_frame(ranges, angle_min, angle_inc, bin_deg=2.0, gap_thresh=0.3):
    """
    逐帧半径过滤 (Frustum Range Filter):
    在每个角度bin内按距离排序, 找第一个大跳跃(glass穿透), 丢弃跳跃后的点
    """
    bin_size = np.radians(bin_deg)
    valid = (ranges > 0.15) & (ranges < 50.0)
    if not np.any(valid):
        return valid  # 全无效

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
    """FRF过滤 + 帧间变换投影 + 半径离群点过滤"""
    total_raw = 0
    total_kept = 0
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

        # 变换到 odom 坐标系
        tx, ty, yaw = tf
        c, s = math.cos(yaw), math.sin(yaw)
        pts = np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty])
        all_pts.append(pts)

    if not all_pts:
        return np.empty((0, 2)), (0, 0), {'raw': total_raw, 'kept': 0, 'merged': 0, 'outlier_removed': 0}

    merged = np.vstack(all_pts)
    n_before_outlier = len(merged)

    # 半径离群点过滤
    if HAS_SCIPY and len(merged) > min_neighbors + 1:
        tree = cKDTree(merged)
        counts = tree.query_ball_point(merged, outlier_radius, return_length=True)
        mask = np.array(counts) >= min_neighbors
        merged = merged[mask]

    cx, cy = merged[:, 0].mean(), merged[:, 1].mean()
    stats = {
        'raw': total_raw, 'kept': total_kept,
        'merged': n_before_outlier, 'outlier_removed': n_before_outlier - len(merged),
        'final': len(merged)
    }

    print(f"  [Filter] raw={total_raw} → FRF kept={total_kept} → merged={n_before_outlier} → outlier filtered={len(merged)}")
    return merged, (cx, cy), stats


# ============================================================
# 3. 扫描轮廓生成
# ============================================================
def create_scan_contour(points, img_size=300, window_size_m=25.0):
    """将点云渲染为闭合多边形轮廓, 返回轮廓+几何特征"""
    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    centered = points.copy()
    centered[:, 0] -= cx
    centered[:, 1] -= cy

    mpp = window_size_m / img_size  # meters per pixel
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    half = img_size // 2

    # 按角度排序后连接
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
        return None, img, centered, None

    pts_arr = np.array(pts_px, dtype=np.int32)
    cv2.polylines(img, [pts_arr], isClosed=True, color=255, thickness=1)
    cv2.fillPoly(img, [pts_arr], 255)

    # 形态学闭操作填补小缝隙
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, img, centered, None

    contour = max(contours, key=cv2.contourArea)

    # 提取几何特征
    features = extract_region_features(contour, img_size, window_size_m)
    return contour, img, centered, features


def extract_region_features(contour, img_size, window_size_m):
    """从轮廓提取几何特征: 面积(m²), 长宽比, Hu矩, 凸包面积比"""
    area_px = cv2.contourArea(contour)
    mpp = window_size_m / img_size
    area_m2 = area_px * mpp * mpp

    # 边界框长宽比
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / max(min(w, h), 1)

    # 凸包面积比 (solidity)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area_px / max(hull_area, 1.0)

    # Hu 矩 (前4个)
    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    # 对数变换
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    hu_normalized = hu_log[:4]

    # 圆形度
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * math.pi * area_px / max(perimeter**2, 1e-6)

    return {
        'area_m2': area_m2,
        'area_px': area_px,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'circularity': circularity,
        'hu': hu_normalized,
        'bounding_rect': (x, y, w, h),
    }


# ============================================================
# 4. 地图连通域分割
# ============================================================
def segment_map_regions(map_data, info, min_area_m2=20.0, dilation_kernel=2):
    """
    将地图自由空间分割为独立连通域。

    步骤:
      1. 提取自由空间 (map_data == 0)
      2. 形态学膨胀 (连接被墙壁厚度隔开的相邻房间)
      3. 连通域标记
      4. 对每个连通域提取轮廓 + 几何特征
    """
    res = info['resolution']
    free = (map_data == 0).astype(np.uint8)

    # 膨胀以跨越墙壁厚度
    if dilation_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
        free = cv2.dilate(free, kernel)

    # 连通域标记
    n_labels, labels = cv2.connectedComponents(free, connectivity=8)

    regions = []
    for label_id in range(1, n_labels):
        mask = (labels == label_id).astype(np.uint8) * 255
        area_px = cv2.countNonZero(mask)
        area_m2 = area_px * res * res

        if area_m2 < min_area_m2:
            continue

        # 找到连通域的重心作为参考点
        ys, xs = np.where(mask > 0)
        center_px = (xs.mean(), ys.mean())
        center_m = (center_px[0] * res + info['origin_x'],
                    (info['height'] - 1 - center_px[1]) * res + info['origin_y'])

        # 提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # 取最大的轮廓 (可能因为膨胀有多个)
        main_contour = max(contours, key=cv2.contourArea)

        # 几何特征
        contour_area_px = cv2.contourArea(main_contour)
        contour_area_m2 = contour_area_px * res * res

        # 边界框
        bx, by, bw, bh = cv2.boundingRect(main_contour)
        bbox_m = {
            'x': bx * res + info['origin_x'],
            'y': (info['height'] - 1 - (by + bh)) * res + info['origin_y'],
            'w': bw * res, 'h': bh * res,
        }

        # 凸包
        hull = cv2.convexHull(main_contour)
        hull_area_px = cv2.contourArea(hull)
        solidity = contour_area_px / max(hull_area_px, 1.0)

        # Hu 矩
        moments = cv2.moments(main_contour)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        # 长宽比
        aspect = max(bw, bh) / max(min(bw, bh), 1)

        regions.append({
            'id': label_id,
            'mask': mask,
            'contour': main_contour,
            'area_m2': contour_area_m2,
            'area_px': contour_area_px,
            'center': center_m,
            'bbox': bbox_m,
            'aspect_ratio': aspect,
            'solidity': solidity,
            'hu': hu_log[:4],
            'n_pixels': area_px,
        })

    print(f"\n  [Segmentation] {n_labels-1} connected components → {len(regions)} regions (>={min_area_m2}m²)")
    for i, r in enumerate(regions[:15]):
        print(f"    Region #{i}: area={r['area_m2']:.1f}m², aspect={r['aspect_ratio']:.1f}, "
              f"center=({r['center'][0]:.1f},{r['center'][1]:.1f}), solidity={r['solidity']:.2f}")
    if len(regions) > 15:
        print(f"    ... and {len(regions)-15} more")

    return regions


# ============================================================
# 5. 区域匹配评分
# ============================================================
def match_scan_to_regions(scan_features, regions, weights=None):
    """
    用多特征联合评分匹配扫描轮廓与地图区域。

    特征:
      1. 面积比: scan_area / region_area (理想=1.0)
      2. 长宽比匹配
      3. 实心率(solidity)匹配
      4. Hu矩距离 (cv2.matchShapes)
    """
    if weights is None:
        weights = {'area': 0.35, 'aspect': 0.15, 'solidity': 0.10, 'hu': 0.40}

    if scan_features is None or not regions:
        return []

    sf = scan_features
    results = []

    for region in regions:
        rf = region

        # 1. 面积比 (0~1, 1=完全匹配)
        area_ratio = min(sf['area_px'], rf['area_px']) / max(sf['area_px'], rf['area_px'], 1)
        area_score = area_ratio  # 越接近1越好

        # 2. 长宽比匹配
        aspect_ratio = min(sf['aspect_ratio'], rf['aspect_ratio']) / max(sf['aspect_ratio'], rf['aspect_ratio'], 1)
        aspect_score = aspect_ratio

        # 3. 实心率匹配
        solidity_diff = abs(sf['solidity'] - rf['solidity'])
        solidity_score = max(0, 1.0 - solidity_diff * 2)

        # 4. Hu矩形状匹配
        hu_dist = cv2.matchShapes(sf.get('contour', rf['contour']),
                                   rf['contour'],
                                   cv2.CONTOURS_MATCH_I2, 0)
        hu_score = max(0, 1.0 - hu_dist * 2)  # 距离越小越好

        # 加权总分
        total = (weights['area'] * area_score +
                 weights['aspect'] * aspect_score +
                 weights['solidity'] * solidity_score +
                 weights['hu'] * hu_score)

        results.append({
            'region': region,
            'total_score': total,
            'area_score': area_score,
            'aspect_score': aspect_score,
            'solidity_score': solidity_score,
            'hu_score': hu_score,
            'hu_dist': hu_dist,
        })

    results.sort(key=lambda x: x['total_score'], reverse=True)
    return results


# ============================================================
# 6. 似然场 + 区域内精细搜索
# ============================================================
def build_likelihood_field(map_data, info, max_dist=3.0):
    """构建障碍物距离场"""
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_px = max_dist / info['resolution']
    return np.clip(dist_px, 0, max_px).astype(np.float32) * info['resolution']


def score_at_pose(points_c, cx, cy, yaw, lf, info):
    """在指定位姿下评分 (似然场 + 命中率)"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y * points_c[:, 0] - s_y * points_c[:, 1] + cx
    my = s_y * points_c[:, 0] + c_y * points_c[:, 1] + cy

    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)

    n_valid = int(np.sum(valid))
    if n_valid < len(points_c) * 0.15:
        return -1e9, 0, 0

    dists = lf[ri[valid], ci[valid]]
    hit_rate = np.sum(dists < 0.15) / n_valid
    lf_score = float(np.mean(np.exp(-dists**2 / 0.045)))
    return lf_score + hit_rate * 0.5, hit_rate, n_valid


def ray_cast_score(points_c, cx, cy, yaw, map_data, info, n_rays=36, max_range=30.0):
    """光线投射评分: 期望测距 (模拟激光) vs 地图墙壁距离"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    ray_angles = np.linspace(0, 2*math.pi, n_rays, endpoint=False)
    errors = []

    for ray_ang in ray_angles:
        # 找扫描中该方向最近的点
        ang_diff = np.abs(np.arctan2(
            np.sin(np.arctan2(points_c[:,1], points_c[:,0]) - ray_ang),
            np.cos(np.arctan2(points_c[:,1], points_c[:,0]) - ray_ang)
        ))
        near = ang_diff < math.radians(15)
        if not np.any(near):
            continue
        actual_range = np.min(np.sqrt(points_c[near, 0]**2 + points_c[near, 1]**2))

        # 射线追踪
        map_ang = ray_ang + yaw
        dx_r = math.cos(map_ang) * res
        dy_r = math.sin(map_ang) * res
        px, py = cx, cy
        sim_range = max_range
        for _ in range(int(max_range / res)):
            col = int((px - ox) / res)
            row = int(H - 1 - (py - oy) / res)
            if col < 0 or col >= W or row < 0 or row >= H:
                break
            if map_data[row, col] == 100:
                sim_range = math.sqrt((px-cx)**2 + (py-cy)**2)
                break
            px += dx_r
            py += dy_r
        errors.append(abs(actual_range - sim_range))

    return np.mean(errors) if errors else 99.0


def validate_pose(points, cx, cy, yaw, map_data, info):
    """
    后验验证: 检查扫描点变换到map系后是否落在合理位置。
    
    返回: (is_valid, report_dict)
      report_dict: {free_pct, occupied_pct, unknown_pct, wall_crossing, n_points}
    
    判定规则:
      1. 未知区域占比 > 25% → 无效 (扫描在灰色/地图外)
      2. 占用区域占比 > 25% → 无效 (扫描穿过大量墙壁)
      3. 自由空间占比 < 35% → 无效 (扫描不在有效区域内)
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    # 变换到map系
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

    # 统计各类型占比
    cells = map_data[ri[valid], ci[valid]]
    n_free = int(np.sum(cells == 0))
    n_occupied = int(np.sum(cells == 100))
    n_unknown = int(np.sum(cells == -1)) + n_outside

    free_pct = n_free / n_total
    occupied_pct = n_occupied / n_total
    unknown_pct = n_unknown / n_total

    report = {
        'free_pct': free_pct,
        'occupied_pct': occupied_pct,
        'unknown_pct': unknown_pct,
        'n_in_map': n_in_map,
        'n_total': n_total,
    }

    # 判定
    if unknown_pct > 0.25:
        return False, report
    if occupied_pct > 0.25:
        return False, report
    if free_pct < 0.35:
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


def fine_search_in_regions(match_results, points_c, cx_scan, cy_scan, lf, map_data, info,
                           scan_features=None, top_k=3, pos_step=0.3, angle_step_deg=5.0):
    """在最佳匹配区域内做精细位置+角度搜索"""
    print(f"\n  [Fine Search] Refining top {top_k} regions...")
    t0 = time.time()

    ds = max(1, len(points_c) // 800)
    pts_ds = points_c[::ds]

    refined = []
    for rank, mr in enumerate(match_results[:top_k]):
        region = mr['region']
        bbox = region['bbox']

        # 在区域边界框内搜索
        margin = 1.0
        xs = np.arange(bbox['x'] + margin, bbox['x'] + bbox['w'] - margin, pos_step)
        ys = np.arange(bbox['y'] + margin, bbox['y'] + bbox['h'] - margin, pos_step)

        if len(xs) == 0 or len(ys) == 0:
            continue

        best_lf = -1e9
        best_pose = (bbox['x'] + bbox['w']/2, bbox['y'] + bbox['h']/2, 0.0)
        n_eval = 0

        angle_step_int = int(angle_step_deg)
        for fyaw_deg in range(0, 360, angle_step_int):
            fyaw = math.radians(fyaw_deg)
            for ax in xs:
                for ay in ys:
                    # 检查是否在自由空间
                    col = int((ax - info['origin_x']) / info['resolution'])
                    row = int(info['height'] - 1 - (ay - info['origin_y']) / info['resolution'])
                    if col < 0 or col >= info['width'] or row < 0 or row >= info['height']:
                        continue
                    if map_data[row, col] != 0:
                        continue

                    sc, _, _ = score_at_pose(pts_ds, ax, ay, fyaw, lf, info)
                    if sc > best_lf:
                        best_lf = sc
                        best_pose = (ax, ay, fyaw)
                    n_eval += 1

        # 180° 消歧
        bx, by, byaw = best_pose
        byaw_alt = byaw + math.pi if byaw < 0 else byaw - math.pi
        sc1, _, _ = score_at_pose(pts_ds, bx, by, byaw, lf, info)
        sc2, _, _ = score_at_pose(pts_ds, bx, by, byaw_alt, lf, info)
        mae1 = ray_cast_score(pts_ds, bx, by, byaw, map_data, info)
        mae2 = ray_cast_score(pts_ds, bx, by, byaw_alt, map_data, info)

        if sc2 - mae2 * 0.03 > sc1 - mae1 * 0.03:
            byaw = byaw_alt
            best_lf = sc2

        refined.append({
            'x': bx, 'y': by, 'yaw': byaw,
            'lf_score': best_lf,
            'region_id': region['id'],
            'region_area': region['area_m2'],
            'region_center': region['center'],
            'shape_score': mr['total_score'],
            'total_score': mr['total_score'] * 10 + best_lf,
            'n_evals': n_eval,
        })

        print(f"    Region #{rank} (id={region['id']}): area={region['area_m2']:.1f}m², "
              f"shape={mr['total_score']:.3f}, "
              f"best=({bx:.2f},{by:.2f},{math.degrees(byaw):.0f}°), lf={best_lf:.3f}, "
              f"evals={n_eval}")

    refined.sort(key=lambda x: x['total_score'], reverse=True)
    elapsed = time.time() - t0
    print(f"  Fine search done in {elapsed:.1f}s, {len(refined)} refined candidates")
    return refined


# ============================================================
# 7. 可视化
# ============================================================
def create_visualization(map_data, info, tf_gt, points, scan_features, regions,
                         match_results, refined_results, output_path):
    fig = plt.figure(figsize=(26, 18))

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W*res, oy, oy + H*res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1, 1, 1]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.7, 0.7, 0.7]

    cx_pts = points[:, 0].mean()
    cy_pts = points[:, 1].mean()

    # ── (a) 扫描轮廓 ──
    ax = fig.add_subplot(2, 3, 1)
    ax.set_title(f"(a) Scan Contour\nArea={scan_features['area_m2']:.1f}m², Aspect={scan_features['aspect_ratio']:.1f}, Solid={scan_features['solidity']:.2f}")
    step = max(1, len(points)//2000)
    ax.scatter(points[::step, 0], points[::step, 1], s=0.5, c='green', alpha=0.4)
    ax.scatter([cx_pts], [cy_pts], c='red', marker='+', s=100)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # ── (b) 地图连通域分割 ──
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(regions))))
    for i, r in enumerate(regions[:20]):
        color = colors[i % 20]
        contour_px = r['contour'].reshape(-1, 2)
        # 转换轮廓像素坐标 → 世界坐标
        cx_w = contour_px[:, 0] * res + ox
        cy_w = (H - 1 - contour_px[:, 1]) * res + oy
        ax.fill(cx_w, cy_w, alpha=0.15, color=color, linewidth=1, edgecolor=color)
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=15, label='GT')
    ax.set_title(f"(b) Map Regions ({len(regions)} connected components)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # ── (c) Top 候选区域 ──
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    top5 = match_results[:5]
    for i, mr in enumerate(top5):
        r = mr['region']
        color = ['red', 'orange', 'green', 'cyan', 'magenta'][i]
        contour_px = r['contour'].reshape(-1, 2)
        cx_w = contour_px[:, 0] * res + ox
        cy_w = (H - 1 - contour_px[:, 1]) * res + oy
        ax.fill(cx_w, cy_w, alpha=0.2, color=color, linewidth=2, edgecolor=color)
        ax.annotate(f"#{i}\nscore={mr['total_score']:.2f}\narea={r['area_m2']:.0f}m²",
                    xy=r['center'], fontsize=6, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=15, label='GT')
    ax.set_title("(c) Top-5 Shape Matches")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # ── (d) 精细搜索结果 ──
    ax = fig.add_subplot(2, 3, 4)
    ax.set_aspect('equal')
    ranks = range(min(5, len(refined_results)))
    scores = [r['total_score'] for r in refined_results[:5]]
    y_labels = [f"#{i}\nreg={r['region_id']}\n({r['x']:.1f},{r['y']:.1f})\n{math.degrees(r['yaw']):.0f}°"
                for i, r in enumerate(refined_results[:5])]
    bars = ax.barh(list(ranks), scores, color='steelblue')
    ax.set_yticks(list(ranks))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Total Score (shape+lf)")
    ax.set_title("(d) Refined Candidates Rank")
    for bar, sc in zip(bars, scores):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2, f"{sc:.2f}", va='center', fontsize=8)

    # ── (e) 最佳匹配叠加 ──
    ax = fig.add_subplot(2, 3, 5)
    if refined_results:
        best = refined_results[0]
        bx, by, byaw = best['x'], best['y'], best['yaw']
        zoom = 12
        ax.set_xlim(bx - zoom, bx + zoom)
        ax.set_ylim(by - zoom, by + zoom)
        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.set_aspect('equal')

        c_b, s_b = math.cos(byaw), math.sin(byaw)
        aligned = np.column_stack([
            c_b*points[:, 0] - s_b*points[:, 1] + bx,
            s_b*points[:, 0] + c_b*points[:, 1] + by
        ])
        step_v = max(1, len(aligned)//2000)
        ax.scatter(aligned[::step_v, 0], aligned[::step_v, 1], s=2, c='lime', alpha=0.6, label='Aligned Scan')
        ax.plot(bx, by, 'r+', markersize=15, mew=3)
        ax.arrow(bx, by, 2.0*math.cos(byaw), 2.0*math.sin(byaw),
                 head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=2.5, zorder=10)
        ax.set_title(f"(e) Best Match Overlay\n({bx:.2f},{by:.2f}) yaw={math.degrees(byaw):.0f}°")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.12)

    # ── (f) 诊断报告 ──
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    rep = ["=== Area Segment Match Report ===", ""]
    rep.append(f"Scan: {len(points)} pts, area={scan_features['area_m2']:.1f}m²")
    rep.append(f"Map: {W}x{H}, {len(regions)} regions")
    rep.append("")
    rep.append("Top-5 Matches:")
    for i, mr in enumerate(match_results[:5]):
        r = mr['region']
        rep.append(f"  #{i}: area={r['area_m2']:.1f}m², shape={mr['total_score']:.3f}")

    rep.append("")
    if refined_results:
        best = refined_results[0]
        rep.append(f"Best: ({best['x']:.3f}, {best['y']:.3f}, {math.degrees(best['yaw']):.1f}°)")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2 + (best['y']-tf_gt[1])**2)
            yaw_diff = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                    math.cos(best['yaw']-tf_gt[2]))))
            rep.append(f"  vs GT: dist_err={err:.3f}m, yaw_err={yaw_diff:.1f}°")
            rep.append(f"  GT: ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}, {math.degrees(tf_gt[2]):.1f}°)")

    rep.append("")
    rep.append("Method: Area segmentation + shape match")
    rep.append("  → segment free-space regions")
    rep.append("  → match area/aspect/solidity/Hu")
    rep.append("  → fine search within best regions")

    ax.text(0.05, 0.95, "\n".join(rep), transform=ax.transAxes,
            fontfamily='monospace', fontsize=8.5, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.55))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  [Output] PNG: {output_path}")


# ============================================================
# 8. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Area Segmentation Shape Matcher')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--min-area', type=float, default=20.0,
                        help='Minimum region area (m²) for segmentation')
    parser.add_argument('--top-k', type=int, default=8,
                        help='Number of shape candidates to refine')
    parser.add_argument('--pos-step', type=float, default=0.3,
                        help='Fine search position step (m)')
    parser.add_argument('--angle-step', type=float, default=5.0,
                        help='Fine search angle step (deg)')
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    output_dir = args.output if args.output else os.path.dirname(npz_path)
    if not os.path.isdir(output_dir) and not output_dir.endswith('.png'):
        output_dir = os.path.dirname(npz_path)
    if output_dir.endswith('.png'):
        output_path = output_dir
        output_dir = os.path.dirname(output_dir)
    else:
        output_path = os.path.join(output_dir, 'area_segment_match_result.png')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(npz_path)

    # 2. Filter & Merge
    print("\n" + "="*60)
    print("Step 1: FRF Filter & Merge Scans")
    print("="*60)
    points, (scan_cx, scan_cy), stats = merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc)

    # 3. Create scan contour
    print("\n" + "="*60)
    print("Step 2: Create Scan Contour")
    print("="*60)
    scan_contour, scan_img, pts_centered, scan_features = create_scan_contour(points)
    if scan_features is None:
        print("[ERROR] Failed to create scan contour"); sys.exit(1)
    scan_features['contour'] = scan_contour
    print(f"  Scan features: area={scan_features['area_m2']:.1f}m², "
          f"aspect={scan_features['aspect_ratio']:.1f}, solidity={scan_features['solidity']:.2f}")

    # 4. Segment map
    print("\n" + "="*60)
    print("Step 3: Segment Map Regions")
    print("="*60)
    regions = segment_map_regions(map_data, info, min_area_m2=args.min_area, dilation_kernel=0)

    if not regions:
        print("[ERROR] No regions found"); sys.exit(1)

    # 5. Match scan to regions
    print("\n" + "="*60)
    print("Step 4: Match Scan to Regions")
    print("="*60)
    match_results = match_scan_to_regions(scan_features, regions)
    for i, mr in enumerate(match_results[:5]):
        r = mr['region']
        print(f"  #{i}: region={r['id']}, area={r['area_m2']:.1f}m², "
              f"total={mr['total_score']:.3f} (area={mr['area_score']:.3f}, "
              f"aspect={mr['aspect_score']:.3f}, hu={mr['hu_score']:.3f})")

    # 6. Fine search
    print("\n" + "="*60)
    print("Step 5: Fine Likelihood Search")
    print("="*60)
    lf = build_likelihood_field(map_data, info)
    refined = fine_search_in_regions(
        match_results, pts_centered, scan_cx, scan_cy, lf, map_data, info,
        scan_features=scan_features, top_k=args.top_k,
        pos_step=args.pos_step, angle_step_deg=args.angle_step)

    print("\n" + "="*60)
    print("Step 6: Validate & Filter Results")
    print("="*60)
    valid_refined = validate_and_filter(refined, points, map_data, info)
    if not valid_refined:
        # 回退到未过滤的结果，但标记
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
        print(f"  Region: id={best['region_id']}, area={best['region_area']:.1f}m²")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2 + (best['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                   math.cos(best['yaw']-tf_gt[2]))))
            print(f"  GT: ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}, {math.degrees(tf_gt[2]):.1f}°)")
            print(f"  Error: dist={err:.3f}m, yaw={yaw_err:.1f}°")
    else:
        print("  [Failed] No valid match")

    # 8. Visualize
    create_visualization(map_data, info, tf_gt, points, scan_features, regions,
                         match_results, valid_refined, output_path)


if __name__ == '__main__':
    main()
