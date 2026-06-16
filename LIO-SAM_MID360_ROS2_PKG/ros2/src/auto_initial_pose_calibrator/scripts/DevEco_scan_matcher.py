#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DevEco_scan_matcher.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
离线雷达扫描数据与地图匹配 — 似然场粗搜 + 光线投射180°消歧 + ICP精调

流程:
  Step 1: 加载 NPZ 数据
  Step 2: 多帧扫描合并 + FRF 过滤
  Step 3: 构建似然场
  Step 4: 全局粗搜 (coarse grid + NMS) → Top-K
  Step 5: 精搜 (fine grid around Top-K) → Top-N
  Step 6: 光线投射 180° 消歧 → 唯一候选
  Step 7: ICP 精调
  Step 8: 输出结果 + 生成标记图片

用法:
  python DevEco_scan_matcher.py
  python DevEco_scan_matcher.py --data path/to/debug_match_data.npz
  python DevEco_scan_matcher.py --coarse-step 1.5 --top-k 10 --ray-lambda 0.3
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
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_px = max_dist / info['resolution']
    lf = np.clip(dist_px, 0, max_px).astype(np.float32) * info['resolution']
    lf[map_data == -1] = max_dist
    return lf


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
    if nv < len(points_c) * 0.10:
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
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    mw_m = info['width'] * res
    mh_m = info['height'] * res

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
                       pos_radius=2.5, pos_step=0.2, angle_step_deg=2.0,
                       angle_range_deg=20):
    print(f"\n  [Fine Search] Refining {len(candidates)} candidates...")
    t0 = time.time()
    angle_step_int = int(angle_step_deg)

    ds = max(1, len(points_c) // 1000)
    pts_ds = points_c[::ds]

    refined = []
    for rank, (sc, hx, hy, hyaw) in enumerate(candidates):
        best_score = -1e9
        best_pose = (hx, hy, hyaw)

        for dx in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
            for dy in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
                ax, ay = hx + dx, hy + dy
                for da_deg in range(-angle_range_deg, angle_range_deg + 1, angle_step_int):
                    ayaw = hyaw + math.radians(da_deg)
                    sc_new, _, _ = score_at_pose(pts_ds, ax, ay, ayaw, lf, info)
                    if sc_new > best_score:
                        best_score = sc_new
                        best_pose = (ax, ay, ayaw)

        refined.append({
            'x': best_pose[0], 'y': best_pose[1], 'yaw': best_pose[2],
            'lf_score': best_score,
            'coarse_score': sc,
            'total_score': best_score,
        })

        if rank < 5:
            bx, by, byaw = best_pose
            print(f"    #{rank}: ({bx:.2f},{by:.2f},{math.degrees(byaw):.0f}deg) "
                  f"lf={best_score:.3f}")

    refined.sort(key=lambda x: x['total_score'], reverse=True)
    elapsed = time.time() - t0
    print(f"  Fine search done: {elapsed:.1f}s")
    return refined


# ============================================================
# 5. Validation
# ============================================================
def validate_pose(points, cx, cy, yaw, map_data, info):
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
# 6. Ray-Cast 180° Disambiguation  (核心改进)
# ============================================================
def ray_cast_single(cx, cy, yaw, ranges, angles, map_data, info,
                    n_rays=120, max_range=30.0):
    """
    光线投射评分: 从候选位姿发射射线, 在栅格地图上步进得到理论测距,
    计算与实际scan测距的MAE。

    使用原始ranges数组(保留空旷射线信息), 而非合并后的点云。

    返回: (mae, valid_ratio)
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    step = max(1, len(ranges) // n_rays)
    sample_idx = np.arange(0, len(ranges), step)

    total_error = 0.0
    n_valid = 0

    for i in sample_idx:
        r_actual = ranges[i]
        if not (0.15 < r_actual < max_range):
            continue

        beam_angle = yaw + angles[i]
        dx = math.cos(beam_angle)
        dy = math.sin(beam_angle)

        px = (cx - ox) / res
        py = (cy - oy) / res

        max_steps = int(r_actual / res) + 5
        r_theoretical = r_actual

        for s in range(1, max_steps + 1):
            col = int(px + s * dx / res + 0.5)
            row = int(H - 1 - (py + s * dy / res) + 0.5)

            if col < 0 or col >= W or row < 0 or row >= H:
                break

            cell = map_data[row, col]
            if cell == 100:
                r_theoretical = s * res
                break

        total_error += abs(r_theoretical - r_actual)
        n_valid += 1

    mae = total_error / max(n_valid, 1)
    valid_ratio = n_valid / max(len(sample_idx), 1)
    return mae, valid_ratio


def ray_cast_disambiguate(candidates, frame_ranges, angle_min, angle_inc,
                          map_data, info, lambda_weight=0.3, n_rays=120):
    """
    对 Top-N 候选进行光线投射 180° 消歧。

    使用第一帧原始ranges(保留空旷射线信息)。
    对每个候选同时评估 yaw 和 yaw+180°,
    综合分 = 似然场分 - lambda * MAE。
    """
    ranges = frame_ranges[0]
    angles = angle_min + np.arange(len(ranges)) * angle_inc

    print(f"\n  [Ray-Cast Disambiguate] {len(candidates)} 候选, n_rays={n_rays}, lambda={lambda_weight}")

    results = []
    for rank, cand in enumerate(candidates):
        if isinstance(cand, dict):
            score_lf = cand['lf_score']
            cx, cy, yaw = cand['x'], cand['y'], cand['yaw']
        else:
            score_lf, cx, cy, yaw = cand[0], cand[1], cand[2], cand[3]

        mae1, vr1 = ray_cast_single(cx, cy, yaw, ranges, angles, map_data, info, n_rays)
        combined1 = score_lf - lambda_weight * mae1

        yaw_180 = yaw + math.pi
        yaw_180 = math.atan2(math.sin(yaw_180), math.cos(yaw_180))

        mae2, vr2 = ray_cast_single(cx, cy, yaw_180, ranges, angles, map_data, info, n_rays)
        combined2 = score_lf - lambda_weight * mae2

        if combined1 >= combined2:
            chosen_yaw = yaw
            chosen_mae = mae1
            chosen_combined = combined1
            direction = 'original'
        else:
            chosen_yaw = yaw_180
            chosen_mae = mae2
            chosen_combined = combined2
            direction = 'flipped'

        results.append({
            'x': cx, 'y': cy,
            'yaw': chosen_yaw,
            'lf_score': score_lf,
            'mae': chosen_mae,
            'combined_score': chosen_combined,
            'direction': direction,
            'mae_original': mae1,
            'mae_flipped': mae2,
            'combined_original': combined1,
            'combined_flipped': combined2,
        })

        if rank < 5:
            print(f"    #{rank}: ({cx:.2f},{cy:.2f}) "
                  f"yaw_orig={math.degrees(yaw):.1f}deg mae={mae1:.3f} "
                  f"yaw_flip={math.degrees(yaw_180):.1f}deg mae={mae2:.3f} "
                  f"-> {direction} (combined={chosen_combined:.3f})")

    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results


# ============================================================
# 7. ICP Refinement
# ============================================================
def icp_refine_scan(points_odom, cx, cy, yaw, map_data, info,
                     max_iter=40, outlier_ratio=2.5, tol=1e-4):
    if not HAS_SCIPY:
        print("  [ICP] scipy not available, skipping ICP")
        return cx, cy, yaw

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    wall_ys, wall_xs = np.where(map_data == 100)
    map_walls = np.column_stack([
        wall_xs * res + ox,
        (H - 1 - wall_ys) * res + oy
    ])
    step = max(1, len(map_walls) // 5000)
    map_walls_ds = map_walls[::step]

    c_y, s_y = math.cos(yaw), math.sin(yaw)
    R_init = np.array([[c_y, -s_y], [s_y, c_y]])
    t_init = np.array([cx, cy])

    src = (R_init @ points_odom.T).T + t_init
    target_tree = cKDTree(map_walls_ds)

    R_total = np.eye(2)
    t_total = np.zeros(2)

    for it in range(max_iter):
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

        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        if np.linalg.norm(t) < tol and np.linalg.norm(R - np.eye(2)) < tol:
            break

    final_R = R_total @ R_init
    final_t = R_total @ t_init + t_total
    final_yaw = math.atan2(final_R[1, 0], final_R[0, 0])

    print(f"  [ICP] Refined: ({cx:.2f},{cy:.2f})->({final_t[0]:.2f},{final_t[1]:.2f}), "
          f"yaw {math.degrees(yaw):.1f}deg->{math.degrees(final_yaw):.1f}deg "
          f"(delta={math.sqrt((final_t[0]-cx)**2+(final_t[1]-cy)**2):.3f}m, "
          f"dyaw={math.degrees(abs(math.atan2(math.sin(final_yaw-yaw),math.cos(final_yaw-yaw)))):.2f}deg)")

    return float(final_t[0]), float(final_t[1]), float(final_yaw)


# ============================================================
# 8. Visualization (6-panel)
# ============================================================
def create_visualization(map_data, info, tf_gt, points, scan_cx, scan_cy,
                         frame_tfs, coarse_candidates, refined_results,
                         disambig_results, final_result,
                         output_path):
    fig = plt.figure(figsize=(28, 18))
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W*res, oy, oy + H*res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1, 1, 1]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.7, 0.7, 0.7]

    # (a) Merged scan shape in odom frame
    ax = fig.add_subplot(2, 3, 1)
    ax.set_aspect('equal')
    if len(points) > 0:
        step_v = max(1, len(points) // 3000)
        ax.scatter(points[::step_v, 0], points[::step_v, 1], s=1, c='blue', alpha=0.4)
        ax.plot(scan_cx, scan_cy, 'r+', markersize=14, mew=3, label='Centroid')
        for fi, tf in enumerate(frame_tfs[:10]):
            tx, ty, yaw = tf
            ax.plot(tx, ty, 'g.', markersize=4)
            if fi == 0:
                ax.annotate('Trajectory', (tx, ty), fontsize=6, color='green')
    ax.set_title("(a) Merged Scan (odom frame) + FRF filter")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.1)

    # (b) Global search candidates
    ax = fig.add_subplot(2, 3, 2)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    colors = plt.cm.jet(np.linspace(0, 1, max(len(coarse_candidates), 1)))
    for i, (sc, cx, cy, cyaw) in enumerate(coarse_candidates[:10]):
        c = colors[min(i, len(colors)-1)]
        ax.plot(cx, cy, 'o', color=c, markersize=max(10-i, 4))
        ax.arrow(cx, cy, 1.5*math.cos(cyaw), 1.5*math.sin(cyaw),
                head_width=0.3, head_length=0.2, color=c, alpha=0.7)
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=14, label='GT')
    ax.set_title("(b) Global Search Top-K Candidates")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # (c) Ray-cast disambiguation comparison
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    if disambig_results:
        best = disambig_results[0]
        bx, by, byaw = best['x'], best['y'], best['yaw']
        zoom = 12
        ax.set_xlim(bx-zoom, bx+zoom)
        ax.set_ylim(by-zoom, by+zoom)

        c_b, s_b = math.cos(byaw), math.sin(byaw)
        aligned = np.column_stack([
            c_b*points[:,0]-s_b*points[:,1]+bx,
            s_b*points[:,0]+c_b*points[:,1]+by])
        step_v = max(1, len(aligned)//2000)
        ax.scatter(aligned[::step_v,0], aligned[::step_v,1], s=2, c='lime', alpha=0.6, label='Best')

        yaw_180 = byaw + math.pi
        yaw_180 = math.atan2(math.sin(yaw_180), math.cos(yaw_180))
        c_f, s_f = math.cos(yaw_180), math.sin(yaw_180)
        flipped = np.column_stack([
            c_f*points[:,0]-s_f*points[:,1]+bx,
            s_f*points[:,0]+c_f*points[:,1]+by])
        ax.scatter(flipped[::step_v,0], flipped[::step_v,1], s=2, c='red', alpha=0.3, label='180deg mirror')

        ax.plot(bx, by, 'r+', markersize=14, mew=3)
        ax.arrow(bx, by, 2.0*math.cos(byaw), 2.0*math.sin(byaw),
                head_width=0.4, head_length=0.3, fc='lime', ec='darkgreen', lw=2.5, zorder=10, label='Best yaw')
        ax.arrow(bx, by, 2.0*math.cos(yaw_180), 2.0*math.sin(yaw_180),
                head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=1.5, alpha=0.5, zorder=9, label='Mirror yaw')
        ax.set_title(f"(c) Ray-Cast Disambiguation\n"
                    f"MAE_orig={best['mae_original']:.3f} MAE_flip={best['mae_flipped']:.3f} -> {best['direction']}")
        ax.legend(fontsize=6, loc='lower right')
    ax.grid(True, alpha=0.12)

    # (d) Best result overlay
    ax = fig.add_subplot(2, 3, 4)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    if final_result:
        fr = final_result
        fx, fy, fyaw = fr['x'], fr['y'], fr['yaw']
        zoom = 12
        ax.set_xlim(fx-zoom, fx+zoom)
        ax.set_ylim(fy-zoom, fy+zoom)

        c_f, s_f = math.cos(fyaw), math.sin(fyaw)
        aligned = np.column_stack([
            c_f*points[:,0]-s_f*points[:,1]+fx,
            s_f*points[:,0]+c_f*points[:,1]+fy])
        step_v = max(1, len(aligned)//2000)
        ax.scatter(aligned[::step_v,0], aligned[::step_v,1], s=2, c='lime', alpha=0.6, label='Aligned scan')
        ax.plot(fx, fy, 'r+', markersize=14, mew=3)
        ax.arrow(fx, fy, 2.0*math.cos(fyaw), 2.0*math.sin(fyaw),
                head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=2.5, zorder=10)

        status = "VALID" if fr.get('is_valid', True) else "INVALID"
        icp_tag = " +ICP" if fr.get('icp_applied', False) else ""
        ax.set_title(f"(d) Final Result{icp_tag}: ({fx:.2f},{fy:.2f}) {math.degrees(fyaw):.1f}deg [{status}]")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)

    # (e) GT comparison
    ax = fig.add_subplot(2, 3, 5)
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    if final_result and tf_gt is not None:
        fr = final_result
        fx, fy, fyaw = fr['x'], fr['y'], fr['yaw']
        zoom = 12
        ax.set_xlim(tf_gt[0]-zoom, tf_gt[0]+zoom)
        ax.set_ylim(tf_gt[1]-zoom, tf_gt[1]+zoom)

        c_g, s_g = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        gt_al = np.column_stack([c_g*points[:,0]-s_g*points[:,1]+tf_gt[0],
                                  s_g*points[:,0]+c_g*points[:,1]+tf_gt[1]])
        step_v = max(1, len(gt_al)//2000)
        ax.scatter(gt_al[::step_v,0], gt_al[::step_v,1], s=2, c='cyan', alpha=0.5, label='GT aligned')

        c_f, s_f = math.cos(fyaw), math.sin(fyaw)
        est_al = np.column_stack([c_f*points[:,0]-s_f*points[:,1]+fx,
                                   s_f*points[:,0]+c_f*points[:,1]+fy])
        ax.scatter(est_al[::step_v,0], est_al[::step_v,1], s=2, c='lime', alpha=0.5, label='Est aligned')

        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=12, label='GT pos')
        ax.plot(fx, fy, 'r+', markersize=12, mew=2, label='Est pos')

        err = math.sqrt((fx-tf_gt[0])**2+(fy-tf_gt[1])**2)
        yaw_err = math.degrees(abs(math.atan2(math.sin(fyaw-tf_gt[2]), math.cos(fyaw-tf_gt[2]))))
        ax.set_title(f"(e) GT vs Est: dist={err:.2f}m, yaw={yaw_err:.1f}deg")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)

    # (f) Report panel
    ax = fig.add_subplot(2, 3, 6)
    ax.axis('off')
    rep = ["=== DevEco Scan Matcher Report ===", ""]

    if disambig_results:
        best_dis = disambig_results[0]
        rep.append(f"[Disambiguation]")
        rep.append(f"  Best: mae_orig={best_dis['mae_original']:.3f} mae_flip={best_dis['mae_flipped']:.3f}")
        rep.append(f"  Direction: {best_dis['direction']}")
        rep.append(f"  LF_score: {best_dis['lf_score']:.4f}")
        rep.append(f"  Combined: {best_dis['combined_score']:.4f}")
        rep.append("")

    if final_result:
        fr = final_result
        rep.append(f"[Final Result]")
        rep.append(f"  Position: ({fr['x']:.3f}, {fr['y']:.3f})")
        rep.append(f"  Yaw: {math.degrees(fr['yaw']):.1f}deg")
        rep.append(f"  LF score: {fr.get('lf_score', 0):.4f}")
        rep.append(f"  MAE: {fr.get('mae', 0):.3f}")
        if fr.get('icp_applied', False):
            rep.append(f"  ICP: applied")
        v = fr.get('validation', {})
        if v:
            rep.append(f"  Free: {v.get('free_pct',0):.1%} | Occ: {v.get('occupied_pct',0):.1%} | Unk: {v.get('unknown_pct',0):.1%}")
        if tf_gt is not None:
            err = math.sqrt((fr['x']-tf_gt[0])**2+(fr['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(fr['yaw']-tf_gt[2]),
                                                    math.cos(fr['yaw']-tf_gt[2]))))
            rep.append(f"  vs GT: dist={err:.3f}m, yaw={yaw_err:.1f}deg")
            if yaw_err < 10:
                rep.append(f"  >>> 180deg DISAMBIGUATION SUCCESS <<<")
            elif yaw_err > 170:
                rep.append(f"  >>> 180deg DISAMBIGUATION FAILED <<<")
    rep.append("")
    rep.append("Method: Likelihood Field + Ray-Cast Disambiguation + ICP")

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
    parser = argparse.ArgumentParser(description='DevEco Scan Matcher: Likelihood + Ray-Cast Disambiguation + ICP')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--coarse-step', type=float, default=1.5, help='Coarse search position step (m)')
    parser.add_argument('--angle-step', type=float, default=12.0, help='Coarse search angle step (deg)')
    parser.add_argument('--top-k', type=int, default=10, help='Top-K candidates after coarse search')
    parser.add_argument('--fine-radius', type=float, default=2.5, help='Fine search radius (m)')
    parser.add_argument('--fine-pos-step', type=float, default=0.2, help='Fine search position step (m)')
    parser.add_argument('--fine-angle-step', type=float, default=2.0, help='Fine search angle step (deg)')
    parser.add_argument('--ray-lambda', type=float, default=0.3, help='Ray-cast MAE weight for combined score')
    parser.add_argument('--ray-n-rays', type=int, default=120, help='Number of rays for ray-casting')
    parser.add_argument('--no-icp', action='store_true', help='Skip ICP refinement')
    parser.add_argument('--no-ray-cast', action='store_true', help='Skip ray-cast disambiguation')
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    if args.output and args.output.endswith('.png'):
        output_path = args.output
    else:
        output_dir = args.output or os.path.dirname(npz_path)
        if not os.path.isdir(output_dir):
            output_dir = os.path.dirname(npz_path)
        output_path = os.path.join(output_dir, 'DevEco_scan_matcher_result.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    # ── Step 1: Load ──
    print("\n" + "="*60)
    print("Step 1: Load Data")
    print("="*60)
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(npz_path)

    # ── Step 2: Merge ──
    print("\n" + "="*60)
    print("Step 2: Merge Scans + FRF Filter")
    print("="*60)
    points, (scan_cx, scan_cy) = merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc)

    pts_centered = points.copy()
    pts_centered[:, 0] -= scan_cx
    pts_centered[:, 1] -= scan_cy

    # ── Step 3: Build likelihood field ──
    print("\n" + "="*60)
    print("Step 3: Build Likelihood Field")
    print("="*60)
    lf = build_likelihood_field(map_data, info)

    # ── Step 4: Global search ──
    print("\n" + "="*60)
    print("Step 4: Global Likelihood Search (Coarse)")
    print("="*60)
    coarse_candidates = global_likelihood_search(
        pts_centered, lf, map_data, info,
        coarse_step=args.coarse_step, angle_step_deg=args.angle_step,
        top_k=args.top_k)

    if not coarse_candidates:
        print("[ERROR] No candidates found"); sys.exit(1)

    # ── Step 5: Fine search ──
    print("\n" + "="*60)
    print("Step 5: Fine Search Refinement")
    print("="*60)
    refined = fine_search_refine(coarse_candidates, pts_centered, lf, map_data, info,
                                  pos_radius=args.fine_radius, pos_step=args.fine_pos_step,
                                  angle_step_deg=args.fine_angle_step)

    # Validate
    print("\n  Validating fine search results...")
    valid_refined = validate_and_filter(refined, points, map_data, info)
    if not valid_refined:
        print("  [WARN] All refined candidates failed validation, using unfiltered results")
        valid_refined = refined

    # ── Step 6: Ray-cast 180° disambiguation ──
    disambig_results = None
    if not args.no_ray_cast:
        print("\n" + "="*60)
        print("Step 6: Ray-Cast 180 Degree Disambiguation")
        print("="*60)
        disambig_results = ray_cast_disambiguate(
            valid_refined, frame_ranges, angle_min, angle_inc,
            map_data, info, lambda_weight=args.ray_lambda,
            n_rays=args.ray_n_rays)

    # ── Step 7: ICP refinement ──
    final_result = None
    if disambig_results:
        best_dis = disambig_results[0]
        final_result = {
            'x': best_dis['x'], 'y': best_dis['y'], 'yaw': best_dis['yaw'],
            'lf_score': best_dis['lf_score'],
            'mae': best_dis['mae'],
            'combined_score': best_dis['combined_score'],
            'direction': best_dis['direction'],
            'is_valid': True,
            'icp_applied': False,
        }
    elif valid_refined:
        best_ref = valid_refined[0]
        final_result = {
            'x': best_ref['x'], 'y': best_ref['y'], 'yaw': best_ref['yaw'],
            'lf_score': best_ref['lf_score'],
            'mae': 0.0,
            'is_valid': best_ref.get('is_valid', True),
            'icp_applied': False,
        }

    if final_result and not args.no_icp:
        print("\n" + "="*60)
        print("Step 7: ICP Refinement")
        print("="*60)
        icp_x, icp_y, icp_yaw = icp_refine_scan(
            pts_centered, final_result['x'], final_result['y'], final_result['yaw'],
            map_data, info)
        final_result['x'] = icp_x
        final_result['y'] = icp_y
        final_result['yaw'] = icp_yaw
        final_result['icp_applied'] = True

    # ── Step 8: Report ──
    print("\n" + "="*60)
    print("Step 8: Final Result")
    print("="*60)
    if final_result:
        fr = final_result
        print(f"  Position: ({fr['x']:.3f}, {fr['y']:.3f})")
        print(f"  Yaw: {math.degrees(fr['yaw']):.1f}deg")
        print(f"  LF score: {fr.get('lf_score', 0):.4f}")
        print(f"  Ray-cast MAE: {fr.get('mae', 0):.3f}")
        if fr.get('icp_applied', False):
            print(f"  ICP: applied")
        if tf_gt is not None:
            err = math.sqrt((fr['x']-tf_gt[0])**2+(fr['y']-tf_gt[1])**2)
            yaw_err = math.degrees(abs(math.atan2(math.sin(fr['yaw']-tf_gt[2]),
                                                    math.cos(fr['yaw']-tf_gt[2]))))
            print(f"  GT: ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}, {math.degrees(tf_gt[2]):.1f}deg)")
            print(f"  Error: dist={err:.3f}m, yaw={yaw_err:.1f}deg")
            if yaw_err < 10:
                print(f"  >>> 180deg DISAMBIGUATION SUCCESS <<<")
            elif yaw_err > 170:
                print(f"  >>> 180deg DISAMBIGUATION FAILED <<<")

    # ── Step 8b: Visualization ──
    print("\n" + "="*60)
    print("Generating visualization...")
    print("="*60)
    create_visualization(map_data, info, tf_gt, points, scan_cx, scan_cy,
                         frame_tfs, coarse_candidates, valid_refined,
                         disambig_results, final_result, output_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
