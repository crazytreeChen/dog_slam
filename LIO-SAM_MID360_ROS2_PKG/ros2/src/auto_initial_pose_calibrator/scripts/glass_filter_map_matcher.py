#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地图区域分割 + 玻璃过滤 + 形状匹配 + 似然场精搜 + 180°消歧

核心思路变化:
  旧: 全地图逐点滑窗 × 多旋转角 → O(位置数 × 旋转数) 评估
  新: 地图分割为房间/走廊 → 轮廓形状预筛选 → Top-K 区域内精搜

流程:
  ① 加载数据 + 玻璃穿透过滤
  ② 地图分割: 自由空间连通域 → 每个区域 = 一个房间/走廊
  ③ 每个区域提取墙面轮廓 + 凸包 + 面积 + 朝向
  ④ 扫描形状 vs 区域形状: 多旋转角 cv2.matchShapes + 面积比 + 凸包相似度
  ⑤ Top-K 区域内: 似然场精细搜索 (5°步长)
  ⑥ 180°消歧: 光线投射模拟
  ⑦ 输出: PNG + 坐标

用法:
  python3 glass_filter_map_matcher.py
  python3 glass_filter_map_matcher.py --data path/to/debug_match_data.npz
  python3 glass_filter_map_matcher.py --min-region-area 50 --top-k 5
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
    print("错误: 需要 opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
except ImportError:
    print("错误: 需要 matplotlib: pip install matplotlib")
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ============================================================
# CJK 字体
# ============================================================
def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun', 'WenQuanYi Micro Hei']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()


# ============================================================
# 1. 数据加载
# ============================================================
def load_data(npz_path):
    if not os.path.exists(npz_path):
        print(f"[错误] 文件不存在: {npz_path}")
        sys.exit(1)
    data = np.load(npz_path, allow_pickle=True)

    map_data = data['map_data']
    resolution = float(data['map_resolution'])
    map_w = int(data['map_width'])
    map_h = int(data['map_height'])
    origin_x = float(data['map_origin_x'])
    origin_y = float(data['map_origin_y'])
    tf_odom_to_map = data['tf_odom_to_map']
    frame_tfs = data['frame_tfs']
    angle_min = float(data['frame_angle_min'])
    angle_inc = float(data['frame_angle_increment'])

    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in data:
        frame_ranges.append(np.array(data[f'frame_ranges_{i}'], dtype=np.float64))
        i += 1

    info = {'resolution': resolution, 'width': map_w, 'height': map_h,
            'origin_x': origin_x, 'origin_y': origin_y}

    print(f"{'='*60}")
    print(f"  数据加载完成")
    print(f"  帧数: {len(frame_ranges)}, 光束/帧: {len(frame_ranges[0])}")
    print(f"  地图: {map_w}x{map_h} @ {resolution:.3f}m/pix")
    print(f"  TF 真值: ({tf_odom_to_map[0]:.2f}, {tf_odom_to_map[1]:.2f}, "
          f"{math.degrees(tf_odom_to_map[2]):.1f}°)")
    print(f"{'='*60}")

    return (map_data, info, tf_odom_to_map, frame_tfs,
            frame_ranges, angle_min, angle_inc)


# ============================================================
# 2. 玻璃穿透过滤
# ============================================================
def merge_and_filter_glass(frame_ranges, frame_tfs, angle_min, angle_inc,
                           mad_k=3.5, min_frames=3, first_return_bin_deg=2.0):
    """
    三阶段过滤（先合并，再过滤）:
      阶段0: 合并所有帧到 odom 系
      阶段1: 首次返回过滤 — 合并后的点云按角度bin只保留最近读数
      阶段2: 统计距离过滤 — 剔除剩余的异常远点
      阶段3: 多帧空间一致性过滤 — 只保留多帧反复击中的静态点
    """
    stats = {}

    # ─── 阶段0: 先合并所有帧到 odom 系 ───
    print(f"\n[阶段 0] 合并 {len(frame_ranges)} 帧到 odom 系...")
    all_pts_list = []
    all_frame_ids = []
    total_raw = 0

    for i, (ranges, tf) in enumerate(zip(frame_ranges, frame_tfs)):
        valid = (ranges > 0.1) & (ranges < 50.0)
        total_raw += np.sum(valid)
        if not np.any(valid):
            continue
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        x_l = ranges[valid] * np.cos(angles[valid])
        y_l = ranges[valid] * np.sin(angles[valid])
        tx, ty, yaw = tf
        c, s = np.cos(yaw), np.sin(yaw)
        pts = np.column_stack([c*x_l - s*y_l + tx, s*x_l + c*y_l + ty])
        all_pts_list.append(pts)
        all_frame_ids.append(np.full(len(pts), i, dtype=np.int32))

    if not all_pts_list:
        return np.empty((0, 2)), stats

    merged_pts = np.vstack(all_pts_list)
    merged_fids = np.concatenate(all_frame_ids)
    stats['total_raw'] = total_raw
    print(f"  合并完成: {len(merged_pts)} 点 (来自 {len(all_pts_list)} 帧)")

    # ─── 阶段1: 首次返回过滤（逐帧执行，保留最近距离簇） ───
    # 每帧是单一视点，同一方向不可能有两面墙。如果>1簇，最近的是真实墙壁。
    # 注意：必须在合并前逐帧过滤，否则不同帧因机器人移动而看到同一墙壁不同距离
    print(f"\n[阶段 1] 首次返回过滤 ({first_return_bin_deg}° bin, 逐帧执行)...")
    n_removed_fr = 0
    
    for i, (pts, ranges, tf) in enumerate(zip(all_pts_list, frame_ranges, frame_tfs)):
        if len(pts) < 3:
            continue
        # 计算每帧中每个点相对帧原点的角度和距离
        tx, ty, yaw = tf
        rel_x = pts[:, 0] - tx
        rel_y = pts[:, 1] - ty
        rel_angles = np.arctan2(rel_y, rel_x)
        rel_dists = np.sqrt(rel_x**2 + rel_y**2)
        
        bin_size = np.radians(first_return_bin_deg)
        bins = np.round(rel_angles / bin_size).astype(int)
        
        keep = np.ones(len(pts), dtype=bool)
        for b in np.unique(bins):
            bin_idx = np.where(bins == b)[0]
            if len(bin_idx) < 2:
                continue
            bin_dists = rel_dists[bin_idx]
            # 找最近的一簇：排序距离，在0.3m内连续的点为一簇
            sorted_order = np.argsort(bin_dists)
            sorted_dists = bin_dists[sorted_order]
            gaps = np.diff(sorted_dists) > 0.3  # >0.3m空隙 = 不同簇
            if not np.any(gaps):
                continue  # 只有一簇 → 全部保留
            # 保留最近簇（从开头到第一个gap之前）
            first_gap = np.argmax(gaps)
            cluster_end = first_gap  # sorted_order[0:cluster_end] 是最近簇
            keep[bin_idx[sorted_order[cluster_end:]]] = False
        
        n_removed_fr += np.sum(~keep)
        all_pts_list[i] = pts[keep]
        all_frame_ids[i] = all_frame_ids[i][keep]
    
    # 重新合并
    merged_pts = np.vstack(all_pts_list)
    merged_fids = np.concatenate(all_frame_ids)
    
    stats['removed_by_first_return'] = n_removed_fr
    print(f"  首次返回移除: {n_removed_fr}/{total_raw} "
          f"({100*n_removed_fr/max(1,total_raw):.1f}%)")

    # ─── 阶段2: 统计距离过滤 ───
    print(f"\n[阶段 2] 统计距离过滤 (MAD×{mad_k})...")
    cx_total = merged_pts[:, 0].mean()
    cy_total = merged_pts[:, 1].mean()
    dists_all = np.sqrt((merged_pts[:, 0] - cx_total)**2 + (merged_pts[:, 1] - cy_total)**2)
    median_r = np.median(dists_all)
    mad = np.median(np.abs(dists_all - median_r))
    threshold = min(median_r + mad_k * 1.4826 * mad, median_r * 3.0)
    print(f"  距离中位数={median_r:.2f}m, 阈值={threshold:.2f}m")

    range_keep = dists_all < threshold
    n_removed_range = int(np.sum(~range_keep))
    stats['removed_by_range'] = n_removed_range
    print(f"  距离过滤移除: {n_removed_range}")

    filtered_pts = merged_pts[range_keep]
    filtered_fids = merged_fids[range_keep]

    # ─── 阶段3: 多帧空间一致性过滤 ───
    print(f"\n[阶段 3] 多帧空间一致性过滤 (最少 {min_frames} 帧)...")
    if HAS_SCIPY and len(filtered_pts) > 10:
        tree = cKDTree(filtered_pts)
        nb = tree.query_ball_point(filtered_pts, 0.15)
        counts = np.array([len(np.unique(filtered_fids[n])) if n else 1 for n in nb])
        keep = counts >= min_frames
    else:
        keep = np.ones(len(filtered_pts), dtype=bool)

    n_rem = int(np.sum(~keep))
    stats['removed_by_consistency'] = n_rem
    stats['final_points'] = int(np.sum(keep))
    print(f"  一致性过滤移除: {n_rem}, 最终: {stats['final_points']} 点")

    return filtered_pts[keep], stats


# ============================================================
# 3. 地图分割 — 自由空间连通域 = 房间/走廊
# ============================================================
def segment_map_regions(map_data, info, min_area_pixels=50):
    """
    将地图分割为独立的自由空间区域（房间、走廊等）。

    方法:
      1. 自由空间 = map_data == 0
      2. 用形态学闭运算填补小缝隙
      3. floodFill / connectedComponents 找连通域
      4. 每个连通域 = 一个区域，提取其墙面边界轮廓

    返回: regions 列表, 每个元素 = {
        'id': int,
        'mask': binary mask (H,W),
        'contour': 墙面边界轮廓 (cv2 contour),
        'convex_hull': 凸包,
        'center_px': (cx, cy) 像素坐标,
        'center_m': (x, y) 世界坐标,
        'area_m2': 面积(平方米),
        'aspect_ratio': 长宽比,
        'orientation_rad': 主方向角,
        'bbox': (x, y, w, h) 外接矩形,
    }
    """
    H, W = map_data.shape
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']

    # 自由空间二值图
    free = (map_data == 0).astype(np.uint8) * 255

    # 形态学闭运算：填补小缝隙（让门洞处的自由空间连通起来）
    # 但如果想让"门"分隔不同房间，则不要闭运算，或者用很小的核
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    free_closed = cv2.morphologyEx(free, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # connectedComponents: label 每个连通域
    num_labels, labels = cv2.connectedComponents(free_closed)

    print(f"\n[地图分割] 发现 {num_labels - 1} 个自由空间连通域 (不含背景)")

    regions = []
    colors_for_viz = plt.cm.tab20(np.linspace(0, 1, 20))

    for label_id in range(1, num_labels):  # 0 = 背景
        mask = (labels == label_id).astype(np.uint8)
        area_px = np.sum(mask)

        # 过滤太小的区域（噪声）
        if area_px < min_area_pixels:
            continue

        # 提取该区域的墙面边界轮廓
        # 墙壁 = 区域边缘接触到的 occupied 像素
        # 用区域 mask 膨胀后减去原 mask 得到边界带
        mask_dilated = cv2.dilate(mask, kernel_small, iterations=2)
        boundary_band = mask_dilated - mask
        wall_near = cv2.bitwise_and(
            (map_data == 100).astype(np.uint8),
            boundary_band.astype(np.uint8))

        # 提取轮廓
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        main_contour = max(contours, key=cv2.contourArea)

        # 凸包
        hull = cv2.convexHull(main_contour)

        # 中心（像素坐标 → 世界坐标）
        M = cv2.moments(main_contour)
        if M['m00'] == 0:
            continue
        cx_px = int(M['m10'] / M['m00'])
        cy_px = int(M['m01'] / M['m00'])
        cx_m = cx_px * res + ox
        cy_m = (H - 1 - cy_px) * res + oy

        # 面积
        area_m2 = area_px * res * res

        # 外接矩形 → 长宽比
        x_br, y_br, w_br, h_br = cv2.boundingRect(main_contour)
        aspect = max(w_br, h_br) / max(min(w_br, h_br), 1)

        # 主方向角 (PCA)
        pts = main_contour.reshape(-1, 2).astype(np.float64)
        if len(pts) > 5:
            mean = np.mean(pts, axis=0)
            pts_c = pts - mean
            cov = np.cov(pts_c.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            orientation = math.atan2(eigvecs[1, -1], eigvecs[0, -1])
        else:
            orientation = 0.0

        region = {
            'id': len(regions),
            'mask': mask,
            'contour': main_contour,
            'convex_hull': hull,
            'center_px': (cx_px, cy_px),
            'center_m': (cx_m, cy_m),
            'area_m2': area_m2,
            'aspect_ratio': aspect,
            'orientation_rad': orientation,
            'bbox': (x_br, y_br, w_br, h_br),
            'color': colors_for_viz[len(regions) % 20],
        }
        regions.append(region)

    # 按面积降序排列
    regions.sort(key=lambda r: r['area_m2'], reverse=True)

    print(f"  有效区域: {len(regions)} 个")
    for r in regions:
        print(f"    区域#{r['id']}: 中心=({r['center_m'][0]:.1f}, {r['center_m'][1]:.1f}), "
              f"面积={r['area_m2']:.1f}m², 长宽比={r['aspect_ratio']:.1f}, "
              f"朝向={math.degrees(r['orientation_rad']):.0f}°")

    return regions


# ============================================================
# 4. 扫描形状提取
# ============================================================
def extract_scan_shape(pts_odom, resolution, phys_size_m=20.0):
    """
    从过滤后的点云提取:
      - 二值图像 (用于 matchShapes)
      - 凸包 (用于快速面积/形状比较)
      - 外轮廓 (用于精细比较)

    返回: (scan_img, scan_contour, scan_hull, scan_area, img_size, meters_per_px)
    """
    cx, cy = np.mean(pts_odom[:, 0]), np.mean(pts_odom[:, 1])
    pts_c = pts_odom.copy()
    pts_c[:, 0] -= cx
    pts_c[:, 1] -= cy

    img_size = int(np.ceil(phys_size_m / resolution))
    if img_size % 2 == 1:
        img_size += 1
    half = img_size // 2
    m_per_px = resolution

    img = np.zeros((img_size, img_size), dtype=np.uint8)
    px = np.clip((pts_c[:, 0] / m_per_px + half).astype(int), 0, img_size - 1)
    py = np.clip((half - pts_c[:, 1] / m_per_px).astype(int), 0, img_size - 1)
    img[py, px] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 外轮廓
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scan_contour = max(contours, key=cv2.contourArea) if contours else None

    # 凸包
    scan_hull = cv2.convexHull(scan_contour) if scan_contour is not None else None

    scan_area = cv2.contourArea(scan_contour) if scan_contour is not None else 0

    return img, scan_contour, scan_hull, scan_area, img_size, m_per_px, cx, cy


# ============================================================
# 5. Stage 1: 区域形状预筛选
# ============================================================
def stage1_region_shape_match(scan_contour, scan_hull, scan_area,
                              scan_img, scan_size, scan_m_per_px,
                              scan_cx_odom, scan_cy_odom,
                              regions, map_data, info,
                              n_rotations=12, top_k=5, tf_gt=None):
    """
    区域级形状匹配 — 比全图滑窗快得多。

    对每个区域:
      1. 提取区域的墙面轮廓
      2. 将扫描形状以多个旋转角放入区域中心
      3. 用 cv2.matchShapes 比较轮廓相似度
      4. 加分项: 面积比匹配、凸包 IoU

    返回: top_candidates [(score, region, best_rotation_deg), ...]
    """
    res = info['resolution']
    H = info['height']
    ox, oy = info['origin_x'], info['origin_y']
    angle_step = 360.0 / n_rotations

    print(f"\n[Stage 1] 区域形状匹配: {len(regions)} 区域 × {n_rotations} 旋转角")
    print(f"  扫描轮廓面积: {scan_area:.0f} px²")

    results = []
    t0 = time.time()

    for region in regions:
        r_contour = region['contour']
        r_hull = region['convex_hull']
        r_area = cv2.contourArea(r_contour)
        r_hull_area = cv2.contourArea(r_hull)

        best_score = -1e9
        best_rot = 0

        # 多旋转角比较
        for rot_deg in np.arange(0, 360, angle_step):
            # 旋转扫描凸包
            rot_rad = math.radians(rot_deg)
            c_r, s_r = np.cos(rot_rad), np.sin(rot_rad)

            # 旋转扫描轮廓点
            scan_pts = scan_contour.reshape(-1, 2).astype(np.float64)
            scan_center = np.mean(scan_pts, axis=0)
            scan_centered = scan_pts - scan_center
            rotated = np.column_stack([
                c_r * scan_centered[:, 0] - s_r * scan_centered[:, 1],
                s_r * scan_centered[:, 0] + c_r * scan_centered[:, 1]
            ]) + scan_center
            rotated_contour = rotated.reshape(-1, 1, 2).astype(np.int32)

            # ── 指标 1: Hu 矩距离 (matchShapes) ──
            hu_dist = cv2.matchShapes(rotated_contour, r_contour,
                                       cv2.CONTOURS_MATCH_I2, 0)
            hu_score = -hu_dist  # 越小越好 → 取负

            # ── 指标 2: 面积比 ──
            area_ratio = min(scan_area, r_area) / max(scan_area, r_area, 1.0)
            area_score = area_ratio  # [0, 1]

            # ── 指标 3: 凸包面积比 ──
            s_hull_area = cv2.contourArea(scan_hull) if scan_hull is not None else scan_area
            hull_ratio = min(s_hull_area, r_hull_area) / max(s_hull_area, r_hull_area, 1.0)
            hull_score = hull_ratio

            # ── 指标 4: 尺度匹配（扫描范围应小于等于区域尺寸）──
            # 扫描范围不应该超过区域尺寸
            r_bbox = region['bbox']
            r_max_dim = max(r_bbox[2], r_bbox[3]) * res  # 区域最大维度(m)
            scan_max_dim = math.sqrt(scan_area) * scan_m_per_px * 1.5  # 扫描大致尺寸
            scale_ratio = min(scan_max_dim, r_max_dim) / max(scan_max_dim, r_max_dim, 0.1)
            scale_score = scale_ratio

            # 综合评分
            combined = (hu_score * 3.0 +     # Hu 矩距离 (主导)
                        area_score * 1.0 +    # 面积匹配
                        hull_score * 0.5 +    # 凸包匹配
                        scale_score * 0.5)    # 尺度匹配

            if combined > best_score:
                best_score = combined
                best_rot = rot_deg

        results.append((best_score, region, best_rot))

    results.sort(key=lambda x: x[0], reverse=True)
    elapsed = time.time() - t0

    print(f"\n  Stage 1 完成，耗时 {elapsed:.2f}s")

    # GT 诊断
    if tf_gt is not None:
        gt_region = None
        gt_min_dist = float('inf')
        for region in regions:
            d = math.sqrt((region['center_m'][0] - tf_gt[0])**2 +
                          (region['center_m'][1] - tf_gt[1])**2)
            if d < gt_min_dist:
                gt_min_dist = d
                gt_region = region
        if gt_region is not None:
            gt_rank = next((i for i, (s, r, _) in enumerate(results)
                           if r['id'] == gt_region['id']), -1)
            print(f"  [诊断] GT 所在区域: #{gt_region['id']} "
                  f"(中心距离GT={gt_min_dist:.1f}m), 排名: #{gt_rank}")

    top = results[:top_k]
    print(f"  Top-{top_k} 区域:")
    for i, (score, region, rot) in enumerate(top):
        err_str = ""
        if tf_gt is not None:
            d = math.sqrt((region['center_m'][0] - tf_gt[0])**2 +
                          (region['center_m'][1] - tf_gt[1])**2)
            err_str = f"  距GT={d:.1f}m"
            if d < 3.0:
                err_str += " ← 近似"
        print(f"    #{i}: score={score:.3f}, 区域#{region['id']}, "
              f"中心=({region['center_m'][0]:.1f}, {region['center_m'][1]:.1f}), "
              f"面积={region['area_m2']:.0f}m², 旋转={rot}°{err_str}")

    return top


# ============================================================
# 6. Stage 2: 区域内似然场精搜
# ============================================================
def build_likelihood_field(map_data, info, max_dist_m=15.0):
    obs = (map_data == 100).astype(np.uint8)
    free = ((1 - obs) * 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.clip(dist_px * info['resolution'], 0, max_dist_m)


def score_at_pose(pts_c, cx, cy, yaw, lf, info, sigma=0.3):
    """似然场评分"""
    c, s = np.cos(yaw), np.sin(yaw)
    mx = c * pts_c[:, 0] - s * pts_c[:, 1] + cx
    my = s * pts_c[:, 0] + c * pts_c[:, 1] + cy

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    col = ((mx - ox) / res + 0.5).astype(np.int32)
    row = ((my - oy) / res + 0.5).astype(np.int32)
    valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    n_valid = int(np.sum(valid))

    if n_valid < len(pts_c) * 0.05:
        return -1e9, 0.0

    dists = lf[row[valid], col[valid]]
    scores = np.exp(-dists**2 / (2 * sigma**2))
    hit_rate = float(np.sum(dists < 0.15) / n_valid)
    return float(np.mean(scores) + hit_rate * 0.5), hit_rate


def build_ray_distance_map(map_data, info):
    """
    预计算地图中每个自由空间像素沿各方向光线击中墙壁的距离。
    返回: wall_dist_map (H,W) float32 — 每个像素到最近墙壁的距离(米)
    用于光线投射时快速跳步。
    """
    obs = (map_data == 100).astype(np.uint8)
    free = ((1 - obs) * 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist_px * info['resolution']


def ray_cast_score_fast(pts_c, cx, cy, yaw, wall_dist_map, info,
                        n_rays=72, max_range=30.0):
    """
    快速光线投射评分 — 用 wall_dist_map 自适应跳步。

    对每条光线:
      1. 从机器人位置出发，沿方向步进
      2. 每步查 wall_dist_map，若距离 < 安全阈值则已到墙壁
      3. 否则跳过 wall_dist_map 距离（安全跳步，不会跳过墙壁）
      4. 将光线击中距离与扫描实际距离比较

    返回: (score, mae)  score越大越好, mae越小越好
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    c_y, s_y = np.cos(yaw), np.sin(yaw)
    ray_angles = np.linspace(0, 2*math.pi, n_rays, endpoint=False)
    errors = []
    safe_thresh = res * 2  # 距墙壁 2 像素以内视为击中

    for ra in ray_angles:
        # 找该方向最近的扫描点距离
        pts_ang = np.arctan2(pts_c[:, 1], pts_c[:, 0])
        diff = np.abs(np.arctan2(np.sin(pts_ang - ra), np.cos(pts_ang - ra)))
        mask = diff < math.radians(8.0)
        if not np.any(mask):
            continue
        expected = float(np.min(np.sqrt(pts_c[mask, 0]**2 + pts_c[mask, 1]**2)))

        # 光线在世界坐标系中的方向
        map_ang = ra + yaw
        dx = math.cos(map_ang)
        dy = math.sin(map_ang)

        # 自适应跳步
        px, py = cx, cy
        traveled = 0.0
        actual = max_range

        while traveled < max_range:
            col = int((px - ox) / res)
            row = int(H - 1 - (py - oy) / res)

            if col < 0 or col >= W or row < 0 or row >= H:
                break

            pixel_val = wall_dist_map[row, col]

            if pixel_val < safe_thresh:
                actual = traveled
                break

            # 自适应跳步: 跳过 wall_dist_map 的安全距离
            step = max(pixel_val * 0.8, res)  # 跳 80% 的安全距离
            px += dx * step
            py += dy * step
            traveled += step

        errors.append(abs(expected - actual))

    if not errors:
        return -1e9, float('inf')

    mae = float(np.mean(errors))
    # 评分: 匹配度越高(MAE越低)越好
    # 用指数衰减: MAE=0 → score=1, MAE=1m → score=0.37, MAE=3m → score=0.05
    score = float(np.exp(-mae / 1.5))
    return score, mae


def stage2_region_refine(pts_odom, top_regions, lf, map_data, info,
                         angle_step_deg=5.0, tf_gt=None,
                         use_ray_cast_primary=True):
    """
    区域内精搜 — 使用光线投射作为主评分函数。

    光线投射 vs 似然场:
      似然场: "扫描点在墙壁附近吗?" → 走廊中任何朝向都高分
      光线投射: "从机器人到墙壁的距离匹配扫描距离吗?" → 物理一致性检验

    返回: (best_cx, best_cy, best_yaw, best_score, all_candidates)
    """
    cx_pts, cy_pts = pts_odom[:, 0].mean(), pts_odom[:, 1].mean()
    pts_c = pts_odom.copy()
    pts_c[:, 0] -= cx_pts
    pts_c[:, 1] -= cy_pts

    ds = max(1, len(pts_c) // 600)
    pts_ds = pts_c[::ds]

    res = info['resolution']
    H, W = info['height'], info['width']
    ox, oy = info['origin_x'], info['origin_y']

    n_angles = int(360.0 / angle_step_deg)
    scan_radius = math.sqrt(np.max(np.sum(pts_c**2, axis=1)))

    # 预计算距离图（光线跳步用）
    if use_ray_cast_primary:
        wall_dist_map = build_ray_distance_map(map_data, info)
        print(f"\n[Stage 2] 光线投射精搜: {len(pts_ds)} 点, {n_angles} 角度/区域")
    else:
        print(f"\n[Stage 2] 似然场精搜: {len(pts_ds)} 点, {n_angles} 角度/区域")

    all_candidates = []
    t0 = time.time()

    # ─── 第一阶段: 似然场粗筛 ───
    print(f"\n[Stage 2] 似然场精搜: {len(pts_ds)} 点, {n_angles} 角度/区域")
    t0 = time.time()

    for rank, (init_score, region, init_rot) in enumerate(top_regions):
        bx, by, bw, bh = region['bbox']
        x1 = bx * res + ox - scan_radius
        y1 = (H - 1 - (by + bh)) * res + oy - scan_radius
        x2 = (bx + bw) * res + ox + scan_radius
        y2 = (H - 1 - by) * res + oy + scan_radius

        grid_step = max(res * 5, 0.5)
        xs = np.arange(x1, x2, grid_step)
        ys = np.arange(y1, y2, grid_step)
        n_eval = 0
        best_for_region = (-1e9, None, None, None)

        for ax in xs:
            for ay in ys:
                col = int((ax - ox) / res)
                row = int(H - 1 - (ay - oy) / res)
                if col < 0 or col >= W or row < 0 or row >= H:
                    continue
                if map_data[row, col] != 0:
                    continue

                for a_deg in range(0, 360, int(angle_step_deg)):
                    ayaw = math.radians(a_deg)
                    cx_map = ax + cx_pts*math.cos(ayaw) - cy_pts*math.sin(ayaw)
                    cy_map = ay + cx_pts*math.sin(ayaw) + cy_pts*math.cos(ayaw)
                    s, _ = score_at_pose(pts_ds, cx_map, cy_map, ayaw, lf, info)
                    n_eval += 1
                    if s > best_for_region[0]:
                        best_for_region = (s, ax, ay, ayaw)
                    all_candidates.append((s, ax, ay, ayaw, rank))

        sc, bx_b, by_b, byaw_b = best_for_region
        if bx_b is not None:
            print(f"    区域#{region['id']}: LF最佳=({bx_b:.2f}, {by_b:.2f}), "
                  f"yaw={math.degrees(byaw_b):.1f}°, score={sc:.4f}")

    all_candidates.sort(key=lambda x: x[0], reverse=True)
    # NMS: 取空间多样性
    nms_lf = []
    for sc, x, y, yaw, rank in all_candidates:
        is_dup = False
        for _, cx2, cy2, _, _ in nms_lf:
            if math.sqrt((x-cx2)**2 + (y-cy2)**2) < 1.5:
                is_dup = True
                break
        if not is_dup:
            nms_lf.append((sc, x, y, yaw, rank))
            if len(nms_lf) >= 20:
                break
    elapsed1 = time.time() - t0
    print(f"  Phase1 完成，耗时 {elapsed1:.1f}s, Top-{len(nms_lf)} 进入光线投射")

    # ─── 第二阶段: 光线投射精排 (暂禁用，太慢) ───
    # TODO: 用 C 扩展或 numpy 向量化加速后启用
    if False and use_ray_cast_primary and len(nms_lf) > 0:
        print(f"\n[Stage 2-Phase2] 光线投射精排 (Top-{len(nms_lf)})...")
        wall_dist_map = build_ray_distance_map(map_data, info)
        rc_results = []
        t1 = time.time()

        for lf_sc, x, y, yaw, rank in nms_lf:
            # 细化搜索: 在 (x,y,yaw) 周围小范围精搜
            fine_step = res * 2
            fine_range = 1.0  # ±1m
            best_rc = (-1e9, x, y, yaw)

            for dx in np.arange(-fine_range, fine_range + 1e-5, fine_step):
                for dy in np.arange(-fine_range, fine_range + 1e-5, fine_step):
                    fx, fy = x + dx, y + dy
                    col = int((fx - ox) / res)
                    row = int(H - 1 - (fy - oy) / res)
                    if col < 0 or col >= W or row < 0 or row >= H:
                        continue
                    if map_data[row, col] != 0:
                        continue

                    # 在 yaw ± 15° 范围内精搜
                    for d_yaw in np.arange(-15, 15.1, 5):
                        fyaw = yaw + math.radians(d_yaw)
                        s, mae = ray_cast_score_fast(
                            pts_ds, fx, fy, fyaw, wall_dist_map, info,
                            n_rays=36, max_range=20.0)
                        if s > best_rc[0]:
                            best_rc = (s, fx, fy, fyaw)

            rc_results.append(best_rc)
            print(f"    ({x:.1f},{y:.1f}) yaw={math.degrees(yaw):.0f}° "
                  f"→ RC: score={best_rc[0]:.4f} "
                  f"({best_rc[1]:.2f},{best_rc[2]:.2f}) "
                  f"yaw={math.degrees(best_rc[3]):.1f}°")

        elapsed2 = time.time() - t1
        print(f"  Phase2 完成，耗时 {elapsed2:.1f}s")

        # 用光线投射分数重新排序
        rc_results.sort(key=lambda x: x[0], reverse=True)
        # 合并到 all_candidates 格式
        all_candidates = [(sc, x, y, yaw, 0) for sc, x, y, yaw in rc_results]

    elapsed = time.time() - t0
    print(f"  Stage 2 总耗时: {elapsed:.1f}s")

    if not all_candidates:
        return None, None, None, -1e9, []

    all_candidates.sort(key=lambda x: x[0], reverse=True)

    # NMS: 空间去重
    nms = []
    for sc, x, y, yaw, rank in all_candidates:
        is_dup = False
        for _, cx2, cy2, _, _ in nms:
            if math.sqrt((x-cx2)**2 + (y-cy2)**2) < 1.0:
                is_dup = True
                break
        if not is_dup:
            nms.append((sc, x, y, yaw, rank))
            if len(nms) >= 10:
                break

    best = nms[0]
    best_sc, best_x, best_y, best_yaw = best[0], best[1], best[2], best[3]

    # 转为扫描质心在 map 中的位置
    full_cx = best_x + cx_pts * math.cos(best_yaw) - cy_pts * math.sin(best_yaw)
    full_cy = best_y + cx_pts * math.sin(best_yaw) + cy_pts * math.cos(best_yaw)

    # GT 评分
    if tf_gt is not None:
        gt_cx = tf_gt[0] + cx_pts * math.cos(tf_gt[2]) - cy_pts * math.sin(tf_gt[2])
        gt_cy = tf_gt[1] + cx_pts * math.sin(tf_gt[2]) + cy_pts * math.cos(tf_gt[2])
        gt_sc, gt_hr = score_at_pose(pts_ds, gt_cx, gt_cy, tf_gt[2], lf, info)
        print(f"  [诊断] GT 似然场分数: {gt_sc:.4f} (命中率={gt_hr:.2f})")

    return full_cx, full_cy, best_yaw, best_sc, nms


# ============================================================
# 7. 180° 消歧
# ============================================================
def extract_wall_segments(pts_odom, angular_bin_deg=1.0, min_wall_length=0.5):
    """
    从扫描点云提取墙壁线段。
    方法: 按角度bin分组，在每个方向找"连续距离变化小"的区间 = 一段墙。
    返回: [(center_x, center_y, length, angle_rad), ...]  在 odom 系中
    """
    if len(pts_odom) < 5:
        return []

    cx, cy = pts_odom[:, 0].mean(), pts_odom[:, 1].mean()
    rel_x = pts_odom[:, 0] - cx
    rel_y = pts_odom[:, 1] - cy
    angles = np.arctan2(rel_y, rel_x)
    dists = np.sqrt(rel_x**2 + rel_y**2)

    bin_size = np.radians(angular_bin_deg)
    bins = np.round(angles / bin_size).astype(int)
    
    segments = []
    for b in np.unique(bins):
        mask = bins == b
        if np.sum(mask) < 3:
            continue
        b_dists = dists[mask]
        d_range = b_dists.max() - b_dists.min()
        # 距离跨度 > 0.5m 且点数足够 = 一段墙
        if d_range > min_wall_length and np.sum(mask) >= 5:
            # 取中点作为墙的中心，长度 = 距离跨度
            mid_dist = (b_dists.min() + b_dists.max()) / 2
            b_angle = b * bin_size
            # 在 odom 系中的墙段中心
            seg_cx = mid_dist * np.cos(b_angle)
            seg_cy = mid_dist * np.sin(b_angle)
            # 墙的朝向 = 垂直于视线方向
            wall_angle = b_angle + np.pi / 2
            segments.append((seg_cx, seg_cy, d_range, wall_angle))

    return segments


def score_wall_segment_match(segments_odom, cx_hypoth, cy_hypoth, yaw_hypoth,
                             map_data, info, match_radius_m=0.5):
    """
    将扫描提取的墙段投影到假设位姿下，检查与地图墙壁的重叠率。
    返回: (score, matched_segments, total_segments)
    """
    if not segments_odom:
        return 0.0, 0, 0

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    
    c, s = np.cos(yaw_hypoth), np.sin(yaw_hypoth)
    matched = 0
    
    for seg_cx, seg_cy, seg_len, seg_angle in segments_odom:
        # 变换到地图坐标系
        map_x = c*seg_cx - s*seg_cy + cx_hypoth
        map_y = s*seg_cx + c*seg_cy + cy_hypoth
        map_angle = seg_angle + yaw_hypoth
        
        # 沿墙段采样点
        n_samples = max(3, int(seg_len / 0.2))
        hits = 0
        for t in np.linspace(-0.5, 0.5, n_samples):
            sample_x = map_x + t * seg_len * math.cos(map_angle)
            sample_y = map_y + t * seg_len * math.sin(map_angle)
            col = int((sample_x - ox) / res)
            row = int(H - 1 - (sample_y - oy) / res)
            if 0 <= col < W and 0 <= row < H:
                # 检查附近(5像素)内是否有墙壁
                r1, r2 = max(0,row-5), min(H,row+6)
                c1, c2 = max(0,col-5), min(W,col+6)
                if np.any(map_data[r1:r2, c1:c2] == 100):
                    hits += 1
        if hits / n_samples >= 0.5:  # 超过一半采样点命中墙壁
            matched += 1

    score = matched / max(len(segments_odom), 1)
    return score, matched, len(segments_odom)


def global_ray_cast_search(pts_filtered, map_data, info, tf_gt=None,
                          coarse_step=2.0, angle_step_coarse=30.0,
                          n_rays=36):
    """
    全地图光线投射 + 墙段匹配 分阶段搜索。
    """
    cx_pts, cy_pts = pts_filtered[:, 0].mean(), pts_filtered[:, 1].mean()
    pts_c = pts_filtered.copy()
    pts_c[:, 0] -= cx_pts
    pts_c[:, 1] -= cy_pts
    ds = max(1, len(pts_c) // 500)
    pts_ds = pts_c[::ds]

    # 提取墙段
    segs = extract_wall_segments(pts_filtered)
    print(f"  提取墙段: {len(segs)} 段")

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    map_w_m = info['width'] * res
    map_h_m = info['height'] * res

    wall_dist = build_ray_distance_map(map_data, info)

    # ─── Phase 1: 粗搜 ───
    print(f"\n  Phase 1: 粗网格={coarse_step}m, 角度步长={angle_step_coarse}°, rays={n_rays}")
    xs = np.arange(ox + 1.0, ox + map_w_m - 1.0, coarse_step)
    ys = np.arange(oy + 1.0, oy + map_h_m - 1.0, coarse_step)
    total1 = len(xs)*len(ys)*int(360.0/angle_step_coarse)
    print(f"  网格 {len(xs)}x{len(ys)} x {int(360.0/angle_step_coarse)} 角度 = {total1} 评估")

    all_phase1 = []
    t0 = time.time()
    count = 0
    for ax in xs:
        for ay in ys:
            for a_deg in np.arange(0, 360, angle_step_coarse):
                ayaw = math.radians(a_deg)
                s_rc, mae = ray_cast_score_fast(
                    pts_ds, ax, ay, ayaw, wall_dist, info, n_rays=n_rays, max_range=20.0)
                s_wall, wm, wt = score_wall_segment_match(
                    segs, ax, ay, ayaw, map_data, info)
                # 组合: 光线投射主导 + 墙段加权
                s = s_rc * 0.6 + s_wall * 2.0  # 墙段匹配高权重
                count += 1
                if s > -1e8:
                    all_phase1.append((s, ax, ay, ayaw))

    all_phase1.sort(key=lambda x: x[0], reverse=True)
    elapsed1 = time.time()-t0
    print(f"  Phase1 耗时 {elapsed1:.1f}s, Top-5:")
    for i, (sc, x, y, yaw) in enumerate(all_phase1[:5]):
        err = ""
        if tf_gt is not None:
            err = f" 距GT={math.sqrt((x-tf_gt[0])**2 + (y-tf_gt[1])**2):.1f}m"
        print(f"    #{i}: RC={sc:.4f} ({x:.1f},{y:.1f})@{math.degrees(yaw):.0f}°{err}")

    # 保留 Top-20 候选 (NMS)
    nms1 = []
    for sc, x, y, yaw in all_phase1:
        dup = any(math.sqrt((x-cx2)**2 + (y-cy2)**2) < 3.0 for _, cx2, cy2, _ in nms1)
        if not dup:
            nms1.append((sc, x, y, yaw))
            if len(nms1) >= 20:
                break

    # ─── Phase 2: 精搜 ───
    print(f"\n  Phase 2: 精网格=0.5m, 角度步长=5°, Top-20 候选")
    all_phase2 = []
    t1 = time.time()
    for _, hx, hy, hyaw in nms1:
        for dx in np.arange(-1.0, 1.01, 0.5):
            for dy in np.arange(-1.0, 1.01, 0.5):
                fx, fy = hx+dx, hy+dy
                for da in np.arange(-15, 16, 5):
                    fyaw = hyaw + math.radians(da)
                    s_rc, _ = ray_cast_score_fast(
                        pts_ds, fx, fy, fyaw, wall_dist, info, n_rays=n_rays, max_range=20.0)
                    s_wall, _, _ = score_wall_segment_match(
                        segs, fx, fy, fyaw, map_data, info)
                    s = s_rc * 0.6 + s_wall * 2.0
                    if s > -1e8:
                        all_phase2.append((s, fx, fy, fyaw))

    all_phase2.sort(key=lambda x: x[0], reverse=True)
    elapsed2 = time.time()-t1
    print(f"  Phase2 耗时 {elapsed2:.1f}s, Top-5:")
    for i, (sc, x, y, yaw) in enumerate(all_phase2[:5]):
        err = ""
        if tf_gt is not None:
            err = f" 距GT={math.sqrt((x-tf_gt[0])**2 + (y-tf_gt[1])**2):.1f}m"
        print(f"    #{i}: RC={sc:.4f} ({x:.1f},{y:.1f})@{math.degrees(yaw):.0f}°{err}")

    # NMS Top-10
    final = []
    for sc, x, y, yaw in all_phase2:
        dup = any(math.sqrt((x-cx2)**2 + (y-cy2)**2) < 1.0 for _, cx2, cy2, _ in final)
        if not dup:
            final.append((sc, x, y, yaw))
            if len(final) >= 10:
                break

    # GT 诊断
    if tf_gt is not None:
        gt_s, gt_mae = ray_cast_score_fast(
            pts_ds, tf_gt[0], tf_gt[1], tf_gt[2], wall_dist, info, n_rays=n_rays, max_range=20.0)
        print(f"  [GT诊断] RC={gt_s:.4f} MAE={gt_mae:.2f}m")

    return final, wall_dist


def disambiguate_180(nms_candidates, pts_odom, lf, map_data, info):
    """180°消歧: 比较 yaw vs yaw+180°"""
    cx_pts, cy_pts = pts_odom[:, 0].mean(), pts_odom[:, 1].mean()
    pts_c = pts_odom.copy()
    pts_c[:, 0] -= cx_pts
    pts_c[:, 1] -= cy_pts
    ds = max(1, len(pts_c) // 800)
    pts_ds = pts_c[::ds]

    print(f"\n[180° 消歧] {len(nms_candidates)} 个候选...")

    refined = []
    for sc, x, y, yaw, rank in nms_candidates:
        yaw_b = yaw + math.pi

        cx_a = x + cx_pts * math.cos(yaw) - cy_pts * math.sin(yaw)
        cy_a = y + cx_pts * math.sin(yaw) + cy_pts * math.cos(yaw)
        cx_b = x + cx_pts * math.cos(yaw_b) - cy_pts * math.sin(yaw_b)
        cy_b = y + cx_pts * math.sin(yaw_b) + cy_pts * math.cos(yaw_b)

        lf_a, _ = score_at_pose(pts_ds, cx_a, cy_a, yaw, lf, info)
        lf_b, _ = score_at_pose(pts_ds, cx_b, cy_b, yaw_b, lf, info)
        _, rc_a = ray_cast_score(pts_ds, cx_a, cy_a, yaw, map_data, info)
        _, rc_b = ray_cast_score(pts_ds, cx_b, cy_b, yaw_b, map_data, info)

        comb_a = lf_a - rc_a * 0.05
        comb_b = lf_b - rc_b * 0.05

        if comb_a >= comb_b:
            refined.append((comb_a, x, y, yaw, lf_a, rc_a))
            chosen_yaw = yaw
        else:
            refined.append((comb_b, x, y, yaw_b, lf_b, rc_b))
            chosen_yaw = yaw_b

        print(f"  ({x:.1f},{y:.1f}): "
              f"yaw={math.degrees(yaw):.0f}°(LF={lf_a:.3f} RC={rc_a:.1f}m) vs "
              f"yaw={math.degrees(yaw_b):.0f}°(LF={lf_b:.3f} RC={rc_b:.1f}m) "
              f"→ 选择 {math.degrees(chosen_yaw):.0f}°")

    refined.sort(key=lambda x: x[0], reverse=True)
    return refined


# ============================================================
# 8. 可视化
# ============================================================
def create_visualization(pts_raw, pts_filtered, map_data, info, tf_gt,
                         scan_img, regions, candidates, wall_dist,
                         best_cx, best_cy, best_yaw, best_score,
                         disambiguated, stats, output_path):
    """4 子图 PNG — 侧重物理对齐可验证性"""

    fig, axes = plt.subplots(2, 2, figsize=(22, 18))

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W*res, oy, oy + H*res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1.0, 1.0, 1.0]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.75, 0.75, 0.75]

    cx_pts, cy_pts = pts_filtered[:, 0].mean(), pts_filtered[:, 1].mean()

    # ─── (a) 全地图 + Top-5 候选标注 ───
    ax = axes[0, 0]
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    for i, (sc, x, y, yaw, _, _) in enumerate(disambiguated[:5]):
        color = ['red', 'orange', 'green', 'cyan', 'magenta'][i]
        ax.plot(x, y, 'o', color=color, markersize=10 - i, mew=2)
        ax.annotate(f'#{i}', (x, y), fontsize=8, color=color, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')
        arrow = 1.8
        ax.annotate('', xy=(x+arrow*math.cos(yaw), y+arrow*math.sin(yaw)),
                    xytext=(x, y), arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=12, label='GT标注(参考)')
        ax.annotate('GT', (tf_gt[0], tf_gt[1]), fontsize=8, color='blue',
                    xytext=(5, 5), textcoords='offset points')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title("全地图 + Top-5 候选 (箭头=朝向)", fontsize=12)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # ─── (b) 最佳候选放大 + 扫描点叠加 ───
    ax = axes[0, 1]
    zoom = 15  # 放大窗口大小 (m)
    best = disambiguated[0] if disambiguated else (None, best_cx, best_cy, best_yaw, best_score, 0)
    if best[1] is not None:
        bx, by, b_yaw = best[1], best[2], best[3]

        # 放大窗口
        x1, x2 = bx - zoom, bx + zoom
        y1, y2 = by - zoom, by + zoom
        ax.set_xlim(x1, x2); ax.set_ylim(y1, y2)

        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.set_aspect('equal')

        # 扫描点叠加（变换到候选位姿）
        c_b, s_b = math.cos(b_yaw), math.sin(b_yaw)
        aligned = np.column_stack([
            c_b*pts_filtered[:,0] - s_b*pts_filtered[:,1] + bx,
            s_b*pts_filtered[:,0] + c_b*pts_filtered[:,1] + by
        ])
        # 按距离着色：近=红, 远=蓝
        dists = np.sqrt(np.sum((aligned - np.array([bx, by]))**2, axis=1))
        step = max(1, len(aligned) // 2000)
        ax.scatter(aligned[::step, 0], aligned[::step, 1],
                   c=dists[::step], s=3, cmap='RdYlBu_r', alpha=0.7)

        # 机器人位置 + 朝向
        ax.plot(bx, by, 'r+', markersize=15, mew=3)
        ax.annotate('', xy=(bx+2.5*math.cos(b_yaw), by+2.5*math.sin(b_yaw)),
                    xytext=(bx, by),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax.plot(bx, by, 'ro', markersize=8, fillstyle='none')

        ax.set_title(f"最佳候选 ({bx:.2f}, {by:.2f}) @ {math.degrees(b_yaw):.0f}°\n"
                     f"扫描点叠加 (红=近, 蓝=远), 红箭头=朝向", fontsize=10)
    ax.grid(True, alpha=0.2)

    # ─── (c) 过滤前后扫描形状对比 ───
    ax = axes[1, 0]
    raw_img, _, _, _, _ = rasterize_simple(pts_raw, res)
    filt_img, _, _, _, _ = rasterize_simple(pts_filtered, res)
    h = max(raw_img.shape[0], filt_img.shape[0])
    w = raw_img.shape[1] + filt_img.shape[1] + 10
    combined = np.zeros((h, w), dtype=np.uint8)
    combined[:raw_img.shape[0], :raw_img.shape[1]] = raw_img
    off = raw_img.shape[1] + 10
    combined[:filt_img.shape[0], off:off+filt_img.shape[1]] = filt_img
    ax.imshow(combined, cmap='gray_r')
    ax.axvline(x=raw_img.shape[1] + 4, color='red', linewidth=2)
    ax.text(raw_img.shape[1]//2, 12, f"原始 {len(pts_raw)} 点", ha='center', fontsize=9, color='yellow')
    ax.text(off+filt_img.shape[1]//2, 12, f"3阶段过滤 {stats['final_points']} 点", ha='center', fontsize=9, color='lime')
    # 标注过滤统计
    ax.text(raw_img.shape[1]//2, h-8,
            f"首次返回 -{stats.get('removed_by_first_return',0)}, 距离 -{stats.get('removed_by_range',0)}, 一致性 -{stats.get('removed_by_consistency',0)}",
            ha='center', fontsize=7, color='white')
    ax.set_title("扫描形状 (左=原始 / 右=过滤后)", fontsize=11)
    ax.axis('off')

    # ─── (d) 诊断信息 ───
    ax = axes[1, 1]
    diag = "=== 全局光线投射定位 诊断 ===\n\n"
    diag += f"过滤: {stats.get('total_raw',0)} → {stats.get('final_points',0)} 点"
    diag += f" ({100*stats.get('final_points',0)/max(1,stats.get('total_raw',1)):.0f}%)\n\n"
    diag += f"搜索: 全地图光线投射\n\n"

    diag += "Top-5 候选位姿:\n"
    for i, (sc, x, y, yaw, _, _) in enumerate(disambiguated[:5]):
        diag += f"  #{i}: ({x:.1f}, {y:.1f}) @ {math.degrees(yaw):.0f}°  RC={sc:.3f}\n"

    if tf_gt is not None:
        diag += f"\n参考标注 (GT, 可能有误差):\n"
        diag += f"  ({tf_gt[0]:.1f}, {tf_gt[1]:.1f}) @ {math.degrees(tf_gt[2]):.1f}°\n"

    diag += f"\n使用方法:\n"
    diag += f"  1. 查看右上放大图确认对齐\n"
    diag += f"  2. 绿色扫描点应对齐黑色墙壁\n"
    diag += f"  3. 红色箭头是机器人朝向\n"
    diag += f"  4. 选最佳候选作为起点"

    ax.text(0.03, 0.97, diag, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[输出] PNG: {output_path}")
    plt.close(fig)


def rasterize_simple(pts, resolution, padding=1.0):
    if len(pts) == 0:
        return np.zeros((10, 10), dtype=np.uint8), 0, 0, 1, 1
    x_min, x_max = pts[:, 0].min() - padding, pts[:, 0].max() + padding
    y_min, y_max = pts[:, 1].min() - padding, pts[:, 1].max() + padding
    w = max(int(np.ceil((x_max - x_min) / resolution)), 2)
    h = max(int(np.ceil((y_max - y_min) / resolution)), 2)
    img = np.zeros((h, w), dtype=np.uint8)
    cols = np.clip(((pts[:, 0] - x_min) / resolution).astype(int), 0, w-1)
    rows = np.clip((h - 1 - (pts[:, 1] - y_min) / resolution).astype(int), 0, h-1)
    img[rows, cols] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(img, kernel, iterations=1)
    return img, x_min, y_min, x_max, y_max


# ============================================================
# 9. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='地图区域分割 + 玻璃过滤 + 形状匹配 + 似然场精搜')
    parser.add_argument('--data', type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--min-frames', type=int, default=3)
    parser.add_argument('--mad-k', type=float, default=3.5)
    parser.add_argument('--no-show-gt', action='store_true')
    parser.add_argument('--min-region-area', type=int, default=50,
                        help='最小区域面积(像素数), 默认50')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Stage 1 保留区域数, 默认5')
    parser.add_argument('--n-rotations', type=int, default=12,
                        help='Stage 1 旋转角数, 默认12 (30°步长)')
    parser.add_argument('--fine-angle-step', type=float, default=5.0,
                        help='Stage 2 精搜角度步长, 默认5°')
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    output_dir = args.output or os.path.dirname(npz_path)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. 加载 ──
    (map_data, info, tf_gt, frame_tfs,
     frame_ranges, angle_min, angle_inc) = load_data(npz_path)
    if args.no_show_gt:
        tf_gt = None

    # ── 2. 玻璃过滤 ──
    pts_filtered, stats = merge_and_filter_glass(
        frame_ranges, frame_tfs, angle_min, angle_inc,
        mad_k=args.mad_k, min_frames=args.min_frames)
    if len(pts_filtered) == 0:
        print("[错误] 过滤后无有效点")
        sys.exit(1)

    # 原始 (对比)
    pts_raw_list = []
    for ranges, tf in zip(frame_ranges, frame_tfs):
        valid = (ranges > 0.1) & (ranges < 30.0)
        if not np.any(valid):
            continue
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        x_l = ranges[valid] * np.cos(angles[valid])
        y_l = ranges[valid] * np.sin(angles[valid])
        tx, ty, yaw = tf
        c, s = np.cos(yaw), np.sin(yaw)
        pts_raw_list.append(np.column_stack([c*x_l - s*y_l + tx,
                                              s*x_l + c*y_l + ty]))
    pts_raw = np.vstack(pts_raw_list) if pts_raw_list else np.empty((0, 2))

    # ── 3. 全地图光线投射搜索 ──
    print("\n" + "="*60)
    print("全局光线投射搜索")
    print("="*60)
    candidates, wall_dist = global_ray_cast_search(
        pts_filtered, map_data, info, tf_gt=tf_gt,
        coarse_step=2.0, angle_step_coarse=30.0, n_rays=36)

    if not candidates:
        print("[错误] 无候选")
        sys.exit(1)

    # ── 6. 180° 消歧 ──
    print("\n" + "="*60)
    print("180° 消歧")
    print("="*60)
    # 将光线投射候选转为消歧格式
    cx_pts, cy_pts = pts_filtered[:, 0].mean(), pts_filtered[:, 1].mean()
    disambiguated = []
    for rc_sc, x, y, yaw in candidates:
        yaw_b = yaw + math.pi
        s_a, _ = ray_cast_score_fast(
            pts_filtered - np.array([cx_pts, cy_pts]), x, y, yaw, wall_dist, info, 36, 20)
        s_b, _ = ray_cast_score_fast(
            pts_filtered - np.array([cx_pts, cy_pts]), x, y, yaw_b, wall_dist, info, 36, 20)
        if s_a >= s_b:
            disambiguated.append((s_a, x, y, yaw, s_a, 0))
        else:
            disambiguated.append((s_b, x, y, yaw_b, s_b, 0))
        print(f"  ({x:.1f},{y:.1f}) yaw={math.degrees(yaw):.0f}°(RC={s_a:.3f}) "
              f"vs yaw={math.degrees(yaw_b):.0f}°(RC={s_b:.3f})"
              f" → {math.degrees(yaw if s_a>=s_b else yaw_b):.0f}°")

    disambiguated.sort(key=lambda x: x[0], reverse=True)
    best = disambiguated[0]
    _, best_x, best_y, best_yaw, best_score, _ = best
    best_cx = best_x + cx_pts*math.cos(best_yaw) - cy_pts*math.sin(best_yaw)
    best_cy = best_y + cx_pts*math.sin(best_yaw) + cy_pts*math.cos(best_yaw)
    nms = [(sc, x, y, yaw, 0) for sc, x, y, yaw, _, _ in disambiguated]

    # 扫描形状 & 区域 (可视化用)
    scan_img, scan_contour, scan_hull, scan_area, scan_size, scan_mpx, scan_cx, scan_cy = \
        extract_scan_shape(pts_filtered, info['resolution'])
    regions = segment_map_regions(map_data, info, min_area_pixels=args.min_region_area)
    top_regions = []  # 区域方法未使用

    # ── 8. 终端结果 ──
    print("\n" + "="*60)
    print("最终结果 — 请目视验证PNG选择正确候选")
    print("="*60)
    if best_cx is not None:
        print(f"\n  最佳: ({best_cx:.3f}, {best_cy:.3f}), "
              f"yaw={math.degrees(best_yaw):.1f}°, score={best_score:.4f}")
        if tf_gt is not None:
            err = math.sqrt((best_cx - tf_gt[0])**2 + (best_cy - tf_gt[1])**2)
            yaw_diff = best_yaw - tf_gt[2]
            err_yaw = abs(math.degrees(math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))))
            print(f"  参考标注: ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}), "
                  f"yaw={math.degrees(tf_gt[2]):.1f}°  (仅供参考!)")

    print(f"\n  Top-5 候选 (请在PNG中目视选择正确的):")
    print(f"  {'#':<4} {'x(m)':<10} {'y(m)':<10} {'yaw(°)':<10} {'Score':<10}")
    print(f"  {'-'*44}")
    for i, (sc, x, y, yaw, _, _) in enumerate(disambiguated[:5]):
        marker = " ← 自动选择" if i == 0 else ""
        print(f"  {i:<4} {x:<10.2f} {y:<10.2f} {math.degrees(yaw):<10.1f} {sc:<10.4f}{marker}")
    print(f"\n  用法: ros2 topic pub /initialpose geometry_msgs/PoseWithCovarianceStamped '<pose>'")
    print(f"  或: 在 rviz2 中用 2D Pose Estimate 点击对应的位置和朝向")

    # ── 9. PNG ──
    output_path = os.path.join(output_dir, 'glass_filter_match_result.png')
    create_visualization(
        pts_raw, pts_filtered, map_data, info, tf_gt,
        scan_img, regions, candidates, wall_dist,
        best_cx, best_cy, best_yaw, best_score,
        disambiguated, stats, output_path)


if __name__ == '__main__':
    main()
