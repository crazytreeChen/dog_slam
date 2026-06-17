#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opencode_likelihood_search.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
纯似然场全局搜索 — 基准定位方案

核心思路:
  跳过所有形状/特征提取，直接在似然场(距离场)上做全局搜索。
  不受Hu矩、区域分割、墙角匹配等中间特征偏差影响。
  
  粗搜: 全图自由空间网格(2m步长, 15°步长) → Top-K
  精搜: Top-K周围(±1.5m, 3°步长) → 最佳
  
  这是最直接、最无偏的匹配方法——只要地图正确，似然场就是最优评分函数。

用法:
  python opencode_likelihood_search.py
  python opencode_likelihood_search.py --data debug_match_data.npz --coarse-step 1.5
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
    print("[ERROR] Need opencv-python"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("[ERROR] Need matplotlib"); sys.exit(1)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


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
        info = {'resolution': float(d['map_resolution']), 'width': int(d['map_width']),
                'height': int(d['map_height']),
                'origin_x': float(d['map_origin_x']), 'origin_y': float(d['map_origin_y'])}
    elif 'map_info' in d:
        map_data = d['map_data']
        mi = d['map_info'].item()
        info = {'resolution': float(mi['resolution']), 'width': int(mi['width']),
                'height': int(mi['height']),
                'origin_x': float(mi['origin_x']), 'origin_y': float(mi['origin_y'])}
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

    print(f"Data: {len(frame_ranges)} frames x {len(frame_ranges[0])} beams")
    print(f"Map: {info['width']}x{info['height']} @ {info['resolution']:.3f}m/pix")
    print(f"GT(ref): ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f}deg)")
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
        if len(idx) < 2: continue
        sorted_idx = idx[np.argsort(ranges[idx])]
        sorted_r = ranges[sorted_idx]
        gaps = np.diff(sorted_r) > gap_thresh
        if np.any(gaps):
            keep[sorted_idx[int(np.argmax(gaps))+1:]] = False
    return valid & keep


def merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc):
    total_raw, total_kept = 0, 0
    all_pts = []
    for fi, (ranges, tf) in enumerate(zip(frame_ranges, frame_tfs)):
        keep_mask = frf_filter_per_frame(ranges, angle_min, angle_inc)
        total_raw += int(np.sum(ranges > 0.15))
        kept = int(np.sum(keep_mask))
        total_kept += kept
        if kept < 10: continue
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        lx = ranges[keep_mask] * np.cos(angles[keep_mask])
        ly = ranges[keep_mask] * np.sin(angles[keep_mask])
        tx, ty, yaw = tf
        c, s = math.cos(yaw), math.sin(yaw)
        all_pts.append(np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty]))
    if not all_pts:
        return np.empty((0, 2)), (0, 0)
    merged = np.vstack(all_pts)
    cx, cy = merged[:, 0].mean(), merged[:, 1].mean()
    print(f"  Merge: raw={total_raw} -> FRF={total_kept} -> merged={len(merged)} pts")
    return merged, (cx, cy)


# ============================================================
# 3. Likelihood Field
# ============================================================
def build_likelihood_field(map_data, info, max_dist=3.0):
    """构建障碍物距离场, 未知区域设为max_dist作为惩罚"""
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_px = max_dist / info['resolution']
    lf = np.clip(dist_px, 0, max_px).astype(np.float32) * info['resolution']
    # 惩罚未知区域: 设为大距离值, 阻止优化器把扫描藏进灰色区域
    lf[map_data == -1] = max_dist
    return lf


def score_at_pose(points_c, cx, cy, yaw, lf, info):
    """似然场评分: 指数衰减 + 命中率bonus"""
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
    if nv < max(len(points_c) * 0.10, 1):
        return -1e9, 0, 0

    dists = lf[ri[valid], ci[valid]]
    n_hit = int(np.sum(dists < 0.15))
    hit_rate = n_hit / nv
    lf_score = float(np.mean(np.exp(-dists**2 / 0.045)))
    return lf_score + hit_rate * 0.5, hit_rate, nv


# ============================================================
# 4. Global Likelihood Search
# ============================================================
def global_likelihood_search(points_c, lf, map_data, info, 
                              coarse_step=2.0, angle_step_deg=15.0, 
                              top_k=10):
    """
    纯似然场全局搜索 (不限制质心必须在自由空间)。
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    mw_m = info['width'] * res
    mh_m = info['height'] * res
    H, W = info['height'], info['width']

    # 随机降采样 (保留约1000点, 避免系统采样偏差)
    ds = max(1, len(points_c) // 1000)
    if ds > 1:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points_c), size=min(len(points_c)//ds, 2000), replace=False)
        pts_ds = points_c[indices]
    else:
        pts_ds = points_c

    xs = np.arange(ox + 2, ox + mw_m - 2, coarse_step)
    ys = np.arange(oy + 2, oy + mh_m - 2, coarse_step)
    n_angles = int(360.0 / angle_step_deg)
    
    print(f"\n  [Global Search] Grid: {len(xs)}x{len(ys)}={len(xs)*len(ys)} positions x {n_angles} angles"
          f" = {len(xs)*len(ys)*n_angles} total evals")
    
    t0 = time.time()
    all_scores = []
    count = 0

    for ax in xs:
        for ay in ys:
            count += 1
            
            for adeg in range(n_angles):
                ayaw = math.radians(adeg * angle_step_deg)
                sc, hit, nv = score_at_pose(pts_ds, ax, ay, ayaw, lf, info)
                if sc > -1e8:
                    all_scores.append((sc, ax, ay, ayaw))
            
            if count % 200 == 0:
                elapsed = time.time() - t0
                n_found = len(all_scores)
                print(f"    {count}/{len(xs)*len(ys)} positions, {n_found} scores ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Coarse search done: {elapsed:.1f}s, {len(all_scores)} valid scores")

    if not all_scores:
        return []

    # 排序 + NMS
    all_scores.sort(key=lambda x: x[0], reverse=True)
    nms = []
    for sc, ax, ay, ayaw in all_scores:
        is_dup = any(math.sqrt((ax-cx)**2 + (ay-cy)**2) < 1.5 
                     and abs(math.atan2(math.sin(ayaw-cyaw), math.cos(ayaw-cyaw))) < math.radians(30)
                     for _, cx, cy, cyaw in nms)
        if not is_dup:
            nms.append((sc, ax, ay, ayaw))
            if len(nms) >= top_k:
                break

    print(f"  Top-{len(nms)} after NMS:")
    for i, (sc, ax, ay, ayaw) in enumerate(nms):
        print(f"    #{i}: ({ax:.1f},{ay:.1f},{math.degrees(ayaw):.0f}deg) score={sc:.3f}")
    
    return nms


def fine_search_refine(candidates, points_c, lf, map_data, info,
                       pos_radius=1.5, pos_step=0.3, angle_step_deg=3.0):
    """精搜: 在粗搜候选周围做高分辨率搜索"""
    print(f"\n  [Fine Search] Refining {len(candidates)} candidates...")
    t0 = time.time()
    angle_step_int = int(angle_step_deg)

    ds = max(1, len(points_c) // 1000)
    pts_ds = points_c[::ds]

    refined = []
    for rank, (sc, hx, hy, hyaw) in enumerate(candidates):
        best_score = -1e9
        best_pose = (hx, hy, hyaw)
        n_eval = 0

        for dx in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
            for dy in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
                ax, ay = hx + dx, hy + dy
                # 不检查质心是否在自由空间，改为后验验证
                for da_deg in range(-20, 21, angle_step_int):
                    ayaw = hyaw + math.radians(da_deg)
                    sc_new, _, _ = score_at_pose(pts_ds, ax, ay, ayaw, lf, info)
                    if sc_new > best_score:
                        best_score = sc_new
                        best_pose = (ax, ay, ayaw)
                    n_eval += 1

        # 180° 消歧: 比较两个相反方向
        bx, by, byaw = best_pose
        byaw_alt = byaw + math.pi if byaw < 0 else byaw - math.pi
        sc1, _, _ = score_at_pose(pts_ds, bx, by, byaw, lf, info)
        sc2, _, _ = score_at_pose(pts_ds, bx, by, byaw_alt, lf, info)
        # 检查自由空间覆盖率来辅助消歧
        _, v1 = validate_pose(points_c, bx, by, byaw, map_data, info)
        _, v2 = validate_pose(points_c, bx, by, byaw_alt, map_data, info)
        
        # 综合评分: 似然场 + 自由空间bonus
        comb1 = sc1 + v1['free_pct'] * 0.5
        comb2 = sc2 + v2['free_pct'] * 0.5
        
        if comb2 > comb1:
            byaw = byaw_alt
            best_score = sc2
        
        refined.append({
            'x': best_pose[0], 'y': best_pose[1], 'yaw': best_pose[2],
            'lf_score': best_score,
            'coarse_score': sc,
            'total_score': best_score,
            'n_evals': n_eval,
        })

        if rank < 5:
            bx, by, byaw = best_pose
            print(f"    #{rank}: ({bx:.2f},{by:.2f},{math.degrees(byaw):.0f}deg) "
                  f"lf={best_score:.3f} evals={n_eval}")

    refined.sort(key=lambda x: x['total_score'], reverse=True)
    elapsed = time.time() - t0
    print(f"  Fine search done: {elapsed:.1f}s")
    return refined


# ============================================================
# 5. Validation
# ============================================================
def validate_pose(points, cx, cy, yaw, map_data, info):
    """后验验证"""
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
        return False, {'free_pct': 0, 'occupied_pct': 0, 'unknown_pct': 1.0}

    cells = map_data[ri[valid], ci[valid]]
    n_free = int(np.sum(cells == 0))
    n_occupied = int(np.sum(cells == 100))
    n_unknown = int(np.sum(cells == -1)) + n_outside

    report = {
        'free_pct': n_free / n_total,
        'occupied_pct': n_occupied / n_total,
        'unknown_pct': n_unknown / n_total,
    }

    if report['unknown_pct'] > 0.55 or report['occupied_pct'] > 0.30 or report['free_pct'] < 0.20:
        return False, report
    return True, report


def validate_and_filter(refined, points, map_data, info):
    valid_list = []
    for r in refined:
        is_valid, report = validate_pose(points, r['x'], r['y'], r['yaw'], map_data, info)
        r['validation'] = report
        r['is_valid'] = is_valid
        if is_valid:
            valid_list.append(r)
        else:
            print(f"    [REJECTED] ({r['x']:.1f},{r['y']:.1f},{math.degrees(r['yaw']):.0f}deg): "
                  f"free={report['free_pct']:.1%} occupied={report['occupied_pct']:.1%} unknown={report['unknown_pct']:.1%}")
    return valid_list


# ============================================================
# 6. Visualization
# ============================================================
# ============================================================
# 6. ICP Refinement
# ============================================================
def icp_refine_scan(points_odom, cx, cy, yaw, map_data, info, 
                     max_iter=40, outlier_ratio=2.5, tol=1e-4):
    """
    ICP精调: 将odom系扫描点变换到map系后, 与地图墙壁对齐。
    
    Returns:
        (refined_cx, refined_cy, refined_yaw, n_iterations, final_residual)
    """
    if not HAS_SCIPY:
        print("  [ICP] scipy not available, skipping ICP")
        return cx, cy, yaw
    
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    
    # 提取地图墙壁点 (降采样)
    wall_ys, wall_xs = np.where(map_data == 100)
    map_walls = np.column_stack([
        wall_xs * res + ox,
        (H - 1 - wall_ys) * res + oy
    ])
    # 降采样到 ~5000 点
    step = max(1, len(map_walls) // 5000)
    map_walls_ds = map_walls[::step]
    
    # 初始变换: odom → map
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    R_init = np.array([[c_y, -s_y], [s_y, c_y]])
    t_init = np.array([cx, cy])
    
    # 扫描点变换到map系
    src = (R_init @ points_odom.T).T + t_init
    target_tree = cKDTree(map_walls_ds)
    
    R_total = np.eye(2)
    t_total = np.zeros(2)
    
    for it in range(max_iter):
        # 最近邻搜索
        dists, idx = target_tree.query(src)
        med = np.median(dists)
        mask = dists < max(0.15, med * outlier_ratio)
        
        n_inliers = int(np.sum(mask))
        if n_inliers < 20:
            break
        
        s_pts = src[mask]
        m_pts = target_tree.data[idx[mask]]
        
        cs, cm = s_pts.mean(0), m_pts.mean(0)
        H_mat = (s_pts - cs).T @ (m_pts - cm)
        
        try:
            U, S, Vt = np.linalg.svd(H_mat)
        except np.linalg.LinAlgError:
            break
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        
        t = cm - R @ cs
        
        # 更新
        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t
        
        if np.linalg.norm(t) < tol and np.linalg.norm(R - np.eye(2)) < tol:
            break
    
    # 最终变换: 新的 odom→map
    final_R = R_total @ R_init
    final_t = R_total @ t_init + t_total
    final_yaw = math.atan2(final_R[1, 0], final_R[0, 0])
    
    print(f"  [ICP] Refined: ({cx:.2f},{cy:.2f})→({final_t[0]:.2f},{final_t[1]:.2f}), "
          f"yaw {math.degrees(yaw):.1f}°→{math.degrees(final_yaw):.1f}° "
          f"(Δ={math.sqrt((final_t[0]-cx)**2+(final_t[1]-cy)**2):.3f}m, "
          f"Δyaw={math.degrees(abs(math.atan2(math.sin(final_yaw-yaw),math.cos(final_yaw-yaw)))):.2f}°)")
    
    return float(final_t[0]), float(final_t[1]), float(final_yaw)


def create_visualization(map_data, info, tf_gt, points, candidates, refined_results, output_path):
    fig = plt.figure(figsize=(24, 14))
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W*res, oy, oy + H*res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1, 1, 1]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.7, 0.7, 0.7]

    # (a) Global search candidates
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    colors = plt.cm.jet(np.linspace(0, 1, len(candidates)))
    for i, (sc, cx, cy, cyaw) in enumerate(candidates):
        c = colors[i]
        ax.plot(cx, cy, 'o', color=c, markersize=max(10-i, 4))
        ax.arrow(cx, cy, 1.5*math.cos(cyaw), 1.5*math.sin(cyaw),
                head_width=0.3, head_length=0.2, color=c, alpha=0.7)
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=14, label='GT')
    ax.set_title("(a) Global Search Top-K Candidates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # (b) Refined scores
    ax = fig.add_subplot(2, 3, 2)
    ranks = range(min(5, len(refined_results)))
    scores = [r['total_score'] for r in refined_results[:5]]
    y_labels = [f"#{i}\n{math.degrees(r['yaw']):.0f}deg" for i, r in enumerate(refined_results[:5])]
    ax.barh(list(ranks), scores, color=['red','orange','green','cyan','magenta'][:len(ranks)])
    ax.set_yticks(list(ranks))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Likelihood Score")
    ax.set_title("(b) Refined Scores")

    # (c) Best overlay
    ax = fig.add_subplot(2, 3, 3)
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
        ax.scatter(aligned[::step_v,0], aligned[::step_v,1], s=2, c='lime', alpha=0.6, label='Est')
        ax.plot(bx, by, 'r+', markersize=14, mew=3)
        ax.arrow(bx, by, 2.0*math.cos(byaw), 2.0*math.sin(byaw),
                head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=2.5, zorder=10)
        
        valid_info = best.get('validation', {})
        status = "VALID" if best.get('is_valid', False) else "INVALID"
        ax.set_title(f"(c) Best: ({bx:.2f},{by:.2f}) {math.degrees(byaw):.0f}deg [{status}]\n"
                    f"free={valid_info.get('free_pct',0):.0%} occ={valid_info.get('occupied_pct',0):.0%} unk={valid_info.get('unknown_pct',0):.0%}")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)

    # (d) GT comparison
    ax = fig.add_subplot(2, 3, 4)
    if refined_results and tf_gt is not None:
        best = refined_results[0]
        bx, by, byaw = best['x'], best['y'], best['yaw']
        zoom = 12
        ax.set_xlim(tf_gt[0]-zoom, tf_gt[0]+zoom)
        ax.set_ylim(tf_gt[1]-zoom, tf_gt[1]+zoom)
        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.set_aspect('equal')
        c_g, s_g = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        gt_al = np.column_stack([c_g*points[:,0]-s_g*points[:,1]+tf_gt[0],
                                  s_g*points[:,0]+c_g*points[:,1]+tf_gt[1]])
        step_v = max(1, len(gt_al)//2000)
        ax.scatter(gt_al[::step_v,0], gt_al[::step_v,1], s=2, c='cyan', alpha=0.5, label='GT')
        c_b, s_b = math.cos(byaw), math.sin(byaw)
        est_al = np.column_stack([c_b*points[:,0]-s_b*points[:,1]+bx,
                                   s_b*points[:,0]+c_b*points[:,1]+by])
        ax.scatter(est_al[::step_v,0], est_al[::step_v,1], s=2, c='lime', alpha=0.5, label='Est')
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=12, label='GT pos')
        ax.plot(bx, by, 'r+', markersize=12, mew=2, label='Est pos')
        err = math.sqrt((bx-tf_gt[0])**2+(by-tf_gt[1])**2)
        yaw_err = math.degrees(abs(math.atan2(math.sin(byaw-tf_gt[2]), math.cos(byaw-tf_gt[2]))))
        ax.set_title(f"(d) GT vs Est: dist={err:.2f}m, yaw={yaw_err:.1f}deg")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)

    # (e) Full map with result
    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    if refined_results:
        best = refined_results[0]
        ax.plot(best['x'], best['y'], 'rX', markersize=14, mew=3)
        ax.arrow(best['x'], best['y'], 3.0*math.cos(best['yaw']), 3.0*math.sin(best['yaw']),
                head_width=0.6, head_length=0.4, fc='red', ec='darkred', lw=3, zorder=10)
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=14)
    ax.set_title("(e) Full Map Result")
    ax.grid(True, alpha=0.1)

    # (f) Report
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    rep = ["=== Likelihood Field Global Search ===", ""]
    rep.append(f"Search: {len(candidates)} coarse candidates")
    rep.append(f"Scoring: Gaussian kernel on distance field")
    rep.append("")
    if refined_results:
        best = refined_results[0]
        rep.append(f"Best: ({best['x']:.3f}, {best['y']:.3f}, {math.degrees(best['yaw']):.1f}deg)")
        rep.append(f"  LF score: {best['lf_score']:.4f}")
        v = best.get('validation', {})
        rep.append(f"  Free: {v.get('free_pct',0):.1%} | Occ: {v.get('occupied_pct',0):.1%} | Unk: {v.get('unknown_pct',0):.1%}")
        rep.append(f"  Valid: {best.get('is_valid', 'N/A')}")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2+(best['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                   math.cos(best['yaw']-tf_gt[2]))))
            rep.append(f"  vs GT: dist={err:.3f}m, yaw={yaw_err:.1f}deg")
    rep.append("")
    rep.append("Method: Direct likelihood field optimization")
    rep.append("  -> No shape/feature pre-filtering")
    rep.append("  -> Global brute-force on distance field")
    rep.append("  -> Free-space mask filters gray areas")
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
    parser = argparse.ArgumentParser(description='Likelihood Field Global Search')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--coarse-step', type=float, default=2.0, help='Coarse search position step (m)')
    parser.add_argument('--angle-step', type=float, default=15.0, help='Coarse search angle step (deg)')
    parser.add_argument('--top-k', type=int, default=8, help='Top-K candidates to refine')
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    if args.output and args.output.endswith('.png'):
        output_path = args.output
    else:
        output_dir = args.output or os.path.dirname(npz_path)
        if not os.path.isdir(output_dir):
            output_dir = os.path.dirname(npz_path)
        output_path = os.path.join(output_dir, 'likelihood_search_result.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # 1. Load
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(npz_path)

    # 2. Merge
    print("\n" + "="*60)
    print("Step 1: Merge Scans")
    print("="*60)
    points, (scan_cx, scan_cy) = merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc)

    # 居中
    pts_centered = points.copy()
    pts_centered[:, 0] -= scan_cx
    pts_centered[:, 1] -= scan_cy

    # 3. Build likelihood field
    print("\n" + "="*60)
    print("Step 2: Build Likelihood Field")
    print("="*60)
    lf = build_likelihood_field(map_data, info)

    # 4. Global search
    print("\n" + "="*60)
    print("Step 3: Global Likelihood Search")
    print("="*60)
    candidates = global_likelihood_search(
        pts_centered, lf, map_data, info,
        coarse_step=args.coarse_step, angle_step_deg=args.angle_step,
        top_k=args.top_k)

    if not candidates:
        print("[ERROR] No candidates found"); sys.exit(1)

    # 5. Fine search
    print("\n" + "="*60)
    print("Step 4: Fine Search Refinement")
    print("="*60)
    refined = fine_search_refine(candidates, pts_centered, lf, map_data, info,
                                  pos_radius=2.5, pos_step=0.2, angle_step_deg=2.0)

    # 6. Validate
    print("\n" + "="*60)
    print("Step 6: Validate Results")
    print("="*60)
    valid_refined = validate_and_filter(refined, points, map_data, info)
    if not valid_refined:
        print("  [WARN] All refined candidates failed validation!")
        valid_refined = refined

    # 7. Results
    print("\n" + "="*60)
    print("Final Result")
    print("="*60)
    if valid_refined:
        best = valid_refined[0]
        print(f"  Position: ({best['x']:.3f}, {best['y']:.3f})")
        print(f"  Yaw: {math.degrees(best['yaw']):.1f}deg")
        print(f"  Score: {best['lf_score']:.4f}")
        v = best.get('validation', {})
        print(f"  Free/occ/unk: {v.get('free_pct',0):.1%}/{v.get('occupied_pct',0):.1%}/{v.get('unknown_pct',0):.1%}")
        if tf_gt is not None:
            err = math.sqrt((best['x']-tf_gt[0])**2+(best['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(best['yaw']-tf_gt[2]),
                                                   math.cos(best['yaw']-tf_gt[2]))))
            print(f"  GT: ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}, {math.degrees(tf_gt[2]):.1f}deg)")
            print(f"  Error: dist={err:.3f}m, yaw={yaw_err:.1f}deg")

    # 8. Visualize
    create_visualization(map_data, info, tf_gt, points, candidates, valid_refined, output_path)


if __name__ == '__main__':
    main()
