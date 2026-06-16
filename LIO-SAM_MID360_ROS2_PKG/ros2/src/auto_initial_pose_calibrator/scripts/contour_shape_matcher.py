#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contour Shape Matcher:
1. Merge all lidar scans from NPZ in odom frame.
2. Centroid-align the merged scan and generate a closed contour using OpenCV.
3. Slide a cropping window matching the scan size across the entire map.
4. Crop local map regions, extract obstacle contours, and calculate shape similarity using cv2.matchShapes (Hu moments).
5. Extract top candidates (lowest shape distance), then run detailed local grid search (Likelihood Field) & 180° disambiguation (Ray Casting).
6. Perform ICP refinement and generate a beautiful 6-panel diagnostic visualization.
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import cv2
from scipy.spatial import cKDTree

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("Error: matplotlib required. Run: pip install matplotlib")
    sys.exit(1)

# Set up CJK font for Chinese logs in matplotlib if available
def setup_font():
    candidates = ['SimHei', 'Microsoft YaHei', 'SimSun']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
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
    map_data = d['map_data']
    res = float(d['map_resolution'])
    mw, mh = int(d['map_width']), int(d['map_height'])
    ox, oy = float(d['map_origin_x']), float(d['map_origin_y'])
    tf_gt = d['tf_odom_to_map']
    frame_tfs = d['frame_tfs']
    angle_min = float(d['frame_angle_min'])
    angle_inc = float(d['frame_angle_increment'])

    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in d:
        frame_ranges.append(d[f'frame_ranges_{i}'])
        i += 1

    info = {
        'resolution': res, 'width': mw, 'height': mh,
        'origin_x': ox, 'origin_y': oy,
    }
    return map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc


def filter_glass_by_consistency(frame_ranges, frame_tfs, angle_min, angle_inc,
                                grid_res=0.15, min_frames=3, max_range=15.0):
    """Filter glass penetration points using multi-frame spatial consensus"""
    frame_count = {}
    per_frame_grids = []
    n_beams = len(frame_ranges[0])
    angles = angle_min + np.arange(n_beams) * angle_inc
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    for ranges, ft in zip(frame_ranges, frame_tfs):
        r = np.array(ranges, dtype=np.float64)
        valid = (r > 0.15) & (r < max_range)
        if not np.any(valid):
            continue
        pts = np.column_stack([r[valid] * cos_a[valid], r[valid] * sin_a[valid]])
        c, s = math.cos(ft[2]), math.sin(ft[2])
        R = np.array([[c, -s], [s, c]])
        odom_pts = (R @ pts.T).T + [ft[0], ft[1]]

        gx = np.round(odom_pts[:, 0] / grid_res).astype(int)
        gy = np.round(odom_pts[:, 1] / grid_res).astype(int)
        seen = set()
        for x, y in zip(gx, gy):
            k = (int(x), int(y))
            if k not in seen:
                seen.add(k)
                frame_count[k] = frame_count.get(k, 0) + 1
        per_frame_grids.append((odom_pts, gx, gy))

    valid_cells = {k for k, v in frame_count.items() if v >= min_frames}
    print(f"  [Glass Filter] Grid vote: kept {len(valid_cells)} cells out of {len(frame_count)}")

    filtered_pts = []
    for odom_pts, gx, gy in per_frame_grids:
        keep = np.array([((int(x), int(y)) in valid_cells) for x, y in zip(gx, gy)])
        filtered_pts.append(odom_pts[keep])

    return np.vstack(filtered_pts) if filtered_pts else np.empty((0, 2))


def filter_outliers(points, radius=0.3, min_neighbors=3):
    """Filter isolated outlier/noise points using KDTree radius query"""
    if len(points) < min_neighbors + 1:
        return points
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, radius, return_length=True)
    mask = np.array(counts) >= min_neighbors
    return points[mask]


def merge_and_filter_ranges(frame_ranges, frame_tfs, angle_min, angle_inc, max_range=15.0,
                            grid_res=0.15, min_frames=3, outlier_radius=0.3, min_neighbors=3):
    """Merge scan ranges while filtering glass penetration and outlier noise"""
    # 1. Glass consistency filter
    merged_clean = filter_glass_by_consistency(
        frame_ranges, frame_tfs, angle_min, angle_inc,
        grid_res=grid_res, min_frames=min_frames, max_range=max_range
    )
    if len(merged_clean) == 0:
        raise ValueError("No valid points after glass filtering.")

    # 2. Statistical radius outlier filter
    filtered = filter_outliers(merged_clean, radius=outlier_radius, min_neighbors=min_neighbors)
    if len(filtered) == 0:
        print("[WARNING] All points removed by outlier filter, falling back to glass-filtered points.")
        filtered = merged_clean

    cx, cy = filtered[:, 0].mean(), filtered[:, 1].mean()
    print(f"  [Merge & Filter] Scans merged: raw points filtered from {len(merged_clean)} down to {len(filtered)} clean points")
    return filtered, cx, cy


def points_to_scan_contour(points, cx, cy, img_size=250, window_size_m=22.0):
    """Render centroid-aligned scan points to a closed polygon and extract contour"""
    pts_centered = points.copy()
    pts_centered[:, 0] -= cx
    pts_centered[:, 1] -= cy

    meters_per_px = window_size_m / img_size
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    half = img_size // 2

    # Sort by polar angle to connect outer boundary points in order
    angles = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
    sorted_idx = np.argsort(angles)

    pts_px = []
    for idx in sorted_idx:
        px = int(pts_centered[idx, 0] / meters_per_px + half)
        py = int(half - pts_centered[idx, 1] / meters_per_px)
        if 0 <= px < img_size and 0 <= py < img_size:
            pts_px.append([px, py])

    if len(pts_px) < 3:
        return None, img, pts_centered

    pts_arr = np.array(pts_px, dtype=np.int32)
    cv2.polylines(img, [pts_arr], isClosed=True, color=255, thickness=1)
    cv2.fillPoly(img, [pts_arr], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, img, pts_centered
    
    return max(contours, key=cv2.contourArea), img, pts_centered


# ============================================================
# 3. Crop Local Map Regions & Extract Map Contours
# ============================================================
def extract_map_contour_at(x, y, map_data, info, window_size_m=22.0):
    """Crop local map at (x, y), perform flood fill from center to extract room free space contour"""
    res = info['resolution']
    half_w_px = max(int(window_size_m / 2 / res), 10)

    cx_px = int((x - info['origin_x']) / res)
    cy_px = int(info['height'] - 1 - (y - info['origin_y']) / res)

    r1 = max(0, cy_px - half_w_px)
    r2 = min(info['height'], cy_px + half_w_px)
    c1 = max(0, cx_px - half_w_px)
    c2 = min(info['width'], cx_px + half_w_px)

    if r2 - r1 < 15 or c2 - c1 < 15:
        return None, None

    roi = map_data[r1:r2, c1:c2]
    
    # We want to find the connected component of free space (value 0) starting from the center of ROI.
    # Center pixel coords in ROI:
    h_roi, w_roi = roi.shape
    cy_roi = h_roi // 2
    cx_roi = w_roi // 2

    # Check if the seed point is actually free space (0)
    if roi[cy_roi, cx_roi] != 0:
        # If the center itself is not free space, search in a small 3x3 neighborhood for a free pixel
        found = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = cy_roi + dy, cx_roi + dx
                if 0 <= ny < h_roi and 0 <= nx < w_roi and roi[ny, nx] == 0:
                    cy_roi, cx_roi = ny, nx
                    found = True
                    break
            if found:
                break
        if not found:
            return None, None # Seed point is not in free space

    # Create a binary image of the ROI: free space (value 0) = 255, occupied/unknown = 0
    free_binary = (roi == 0).astype(np.uint8) * 255

    # Flood fill from the seed point on free_binary
    # cv2.floodFill modifies the image in-place, filling with a new color (e.g. 128)
    # It also requires a mask that is 2 pixels wider and taller than the image
    mask = np.zeros((h_roi + 2, w_roi + 2), dtype=np.uint8)
    cv2.floodFill(free_binary, mask, (cx_roi, cy_roi), 128)
    
    # Extract the filled region
    room_mask = (free_binary == 128).astype(np.uint8) * 255

    # Find the external contour of the filled room
    contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, room_mask

    main_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main_contour) < 20:
        return None, room_mask

    return main_contour, room_mask


# ============================================================
# 4. Shape Similarity Heatmap Search (Sliding Window)
# ============================================================
def sliding_window_shape_match(scan_contour, map_data, info, window_size_m=22.0,
                               step_m=1.0, max_dist_threshold=0.6, n_keep=8):
    """Slide window over map, crop regions, calculate shape distance using Hu moments"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    mw_m = info['width'] * res
    mh_m = info['height'] * res

    # Define sliding grid bounds
    margin = window_size_m / 2
    xs = np.arange(ox + margin, ox + mw_m - margin, step_m)
    ys = np.arange(oy + margin, oy + mh_m - margin, step_m)

    score_grid = np.full((len(ys), len(xs)), np.nan)
    candidates = []

    print(f"\n[Sliding Window] Starting sliding shape search on {len(xs)}x{len(ys)}={len(xs)*len(ys)} grid positions...")
    t0 = time.time()
    
    for j, y_val in enumerate(ys):
        for i, x_val in enumerate(xs):
            map_contour, _ = extract_map_contour_at(x_val, y_val, map_data, info, window_size_m)
            if map_contour is None:
                continue
            
            # cv2.matchShapes computes shape distance. Lower is more similar.
            # CONTOURS_MATCH_I2 uses Hu moments log difference.
            dist_score = cv2.matchShapes(scan_contour, map_contour, cv2.CONTOURS_MATCH_I2, 0.0)
            score_grid[j, i] = dist_score

            if dist_score <= max_dist_threshold:
                candidates.append((dist_score, x_val, y_val))

    if not candidates:
        print(f"  [WARNING] No candidates met threshold <= {max_dist_threshold}. Falling back to top distinct shape matches.")
        all_evals = []
        for j, y_val in enumerate(ys):
            for i, x_val in enumerate(xs):
                sc = score_grid[j, i]
                if not np.isnan(sc):
                    all_evals.append((sc, x_val, y_val))
        all_evals.sort(key=lambda c: c[0])
        candidates = all_evals

    # Apply Non-Maximum Suppression (NMS) to get spatially distinct candidates
    candidates.sort(key=lambda c: c[0])  # Sort by shape distance (lowest first)
    distinct_candidates = []
    for dist, x, y in candidates:
        is_new = True
        for _, cx, cy in distinct_candidates:
            if math.sqrt((x - cx)**2 + (y - cy)**2) < 3.5:
                is_new = False
                break
        if is_new:
            distinct_candidates.append((dist, x, y))
            if len(distinct_candidates) >= n_keep:
                break

    elapsed = time.time() - t0
    print(f"  Shape search completed in {elapsed:.1f}s. Kept {len(distinct_candidates)} distinct candidates.")
    for rank, (dist, x, y) in enumerate(distinct_candidates):
        print(f"    Candidate #{rank}: Shape Dist={dist:.4f}, Pos=({x:.2f}, {y:.2f})")
    
    return distinct_candidates, score_grid, xs, ys


# ============================================================
# 5. Detail Refinement & 180° Yaw Disambiguation
# ============================================================
def build_likelihood_field(map_data, info):
    obs = (map_data == 100).astype(np.uint8)
    free = ((1 - obs) * 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist_px * info['resolution']


def score_at_pose(pts_centered, cx, cy, yaw, dist_m, info, sigma=0.3):
    c, s = np.cos(yaw), np.sin(yaw)
    mx = c * pts_centered[:, 0] - s * pts_centered[:, 1] + cx
    my = s * pts_centered[:, 0] + c * pts_centered[:, 1] + cy

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    col = ((mx - ox) / res + 0.5).astype(np.int32)
    row = ((my - oy) / res + 0.5).astype(np.int32)
    valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    
    n_valid = np.sum(valid)
    if n_valid < len(pts_centered) * 0.05:
        return -1e9, 0.0

    dists = dist_m[row[valid], col[valid]]
    scores = np.exp(-dists**2 / (2 * sigma**2))
    hit_rate = np.sum(dists < 0.15) / n_valid
    return np.mean(scores) + hit_rate * 0.5, hit_rate


def ray_cast_score(pts_centered, cx, cy, yaw, map_data, info, max_range=30.0, n_rays=72):
    """Compute ray casting error to resolve 180° yaw ambiguity (lower MAE is correct direction)"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    ray_angles = np.linspace(0, 2 * math.pi, n_rays, endpoint=False)
    mae_list = []

    for ray_ang in ray_angles:
        pts_ang = np.arctan2(pts_centered[:, 1], pts_centered[:, 0])
        ang_diff = np.abs(np.arctan2(np.sin(pts_ang - ray_ang), np.cos(pts_ang - ray_ang)))
        near_mask = ang_diff < math.radians(8.0)
        if not np.any(near_mask):
            continue

        actual_range = np.min(np.sqrt(pts_centered[near_mask, 0]**2 + pts_centered[near_mask, 1]**2))

        # Ray trace on grid map
        map_ray_ang = ray_ang + yaw
        dx = math.cos(map_ray_ang) * res
        dy = math.sin(map_ray_ang) * res
        
        px, py = cx, cy
        sim_range = max_range
        for step in range(int(max_range / res)):
            col = int((px - ox) / res)
            row = int(H - 1 - (py - oy) / res)
            if col < 0 or col >= W or row < 0 or row >= H:
                break
            if map_data[row, col] == 100:
                sim_range = step * res
                break
            elif map_data[row, col] == -1:
                break # Unknown space
            px += dx
            py += dy
        
        mae_list.append(abs(actual_range - sim_range))
    
    return np.mean(mae_list) if mae_list else 99.0


def fine_search_candidates(candidates, pts_centered, scan_cx, scan_cy, dist_m, map_data, info,
                           pos_radius=1.8, pos_step=0.2, angle_step_deg=4.0):
    """For each shape candidate, do local grid search over positions & angles, and resolve 180° yaw"""
    print(f"\n[Detail Fine Search] Refining position and heading around {len(candidates)} shape candidates...")
    t0 = time.time()
    
    ds = max(1, len(pts_centered) // 1000)
    pts_ds = pts_centered[::ds]
    
    refined_results = []
    
    for rank, (shape_dist, hx, hy) in enumerate(candidates):
        # We place scan centroid at (hx, hy)
        # Local search ranges
        xs = np.arange(hx - pos_radius, hx + pos_radius + 1e-5, pos_step)
        ys = np.arange(hy - pos_radius, hy + pos_radius + 1e-5, pos_step)
        yaws = np.arange(-math.pi, math.pi, math.radians(angle_step_deg))

        best_score = -1e9
        best_pose = (hx, hy, 0.0)

        # Local grid search
        for yaw in yaws:
            for ax in xs:
                for ay in ys:
                    # Place robot at (ax, ay) in map frame
                    rc, rs = math.cos(yaw), math.sin(yaw)
                    r_cx = ax + scan_cx * rc - scan_cy * rs
                    r_cy = ay + scan_cx * rs + scan_cy * rc
                    s, _ = score_at_pose(pts_ds, r_cx, r_cy, yaw, dist_m, info)
                    if s > best_score:
                        best_score = s
                        best_pose = (ax, ay, yaw)

        # 180° Disambiguation on local best
        bx, by, byaw = best_pose
        byaw_alt = byaw + math.pi if byaw < 0 else byaw - math.pi
        
        rc1, rs1 = math.cos(byaw), math.sin(byaw)
        rc2, rs2 = math.cos(byaw_alt), math.sin(byaw_alt)
        
        # Centroid mapped position for both
        cx1 = bx + scan_cx * rc1 - scan_cy * rs1
        cy1 = by + scan_cx * rs1 + scan_cy * rc1
        cx2 = bx + scan_cx * rc2 - scan_cy * rs2
        cy2 = by + scan_cx * rs2 + scan_cy * rc2
        
        lf_score1, _ = score_at_pose(pts_ds, cx1, cy1, byaw, dist_m, info)
        lf_score2, _ = score_at_pose(pts_ds, cx2, cy2, byaw_alt, dist_m, info)

        mae1 = ray_cast_score(pts_ds, cx1, cy1, byaw, map_data, info)
        mae2 = ray_cast_score(pts_ds, cx2, cy2, byaw_alt, map_data, info)

        # Selection criteria: combined likelihood and ray casting error
        comb_score1 = lf_score1 - mae1 * 0.04
        comb_score2 = lf_score2 - mae2 * 0.04

        if comb_score1 >= comb_score2:
            refined_pose = (bx, by, byaw, lf_score1, mae1, comb_score1)
        else:
            refined_pose = (bx, by, byaw_alt, lf_score2, mae2, comb_score2)

        refined_results.append(refined_pose)
        print(f"    Candidate #{rank} refined: Centroid=({refined_pose[0]:.2f}, {refined_pose[1]:.2f}), "
              f"Yaw={math.degrees(refined_pose[2]):.1f}°, LF={refined_pose[3]:.3f}, RayMAE={refined_pose[4]:.2f}m")

    refined_results.sort(key=lambda r: -r[5]) # Sort by combined score (highest first)
    elapsed = time.time() - t0
    print(f"  Detail refinement done in {elapsed:.1f}s.")
    return refined_results


# ============================================================
# 6. ICP Alignment Refinement
# ============================================================
def icp_refine(source_pts, target_tree, max_iter=40, tol=1e-5):
    src = source_pts.copy()
    R_total = np.eye(2)
    t_total = np.zeros(2)

    for it in range(max_iter):
        dists, idx = target_tree.query(src)
        med = np.median(dists)
        mask = dists < max(0.2, med * 2.0)

        if np.sum(mask) < 10:
            break

        s_pts = src[mask]
        m_pts = target_tree.data[idx[mask]]
        cs, cm = s_pts.mean(0), m_pts.mean(0)
        H = (s_pts - cs).T @ (m_pts - cm)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cm - R @ cs

        src = (R @ src.T).T + t
        R_total = R @ R_total
        t_total = R @ t_total + t

        if np.linalg.norm(t) < tol:
            break

    return R_total, t_total


# ============================================================
# 7. Visualization & Main Script Logic
# ============================================================
def generate_panels(map_data, info, scan_pts, scan_contour, scan_img, pts_c,
                    score_grid, xs_grid, ys_grid, candidates, best_refined, refined_results, final_aligned,
                    tf_gt, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(26, 17))
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W * res, oy, oy + H * res]

    map_disp = np.zeros((H, W, 3), dtype=np.float32)
    map_disp[map_data == 0] = [1.0, 1.0, 1.0]
    map_disp[map_data == 100] = [0.1, 0.1, 0.1]
    map_disp[map_data == -1] = [0.7, 0.7, 0.7]

    # Panel (a): Scan Shape & Contour Extraction
    ax = axes[0, 0]
    ax.imshow(scan_img, cmap='gray_r', origin='upper')
    ax.set_title("(a) Merged Scan Contour (Centroid-aligned)\nOpenCV Contour Area: " + 
                 f"{cv2.contourArea(scan_contour):.0f} px²", fontsize=11)
    ax.axis('off')

    # Panel (b): Grid Map and Sample Local Map Bounding Box
    ax = axes[0, 1]
    ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    # Draw bounding box for the best candidate
    bx, by, _, _, _, _ = best_refined
    box_size_m = 22.0
    rect = plt.Rectangle((bx - box_size_m/2, by - box_size_m/2), box_size_m, box_size_m,
                         linewidth=2, edgecolor='red', facecolor='none', label='Best Shape Window')
    ax.add_patch(rect)
    for rank, (_, cx, cy) in enumerate(candidates[:5]):
        ax.plot(cx, cy, 'bo', ms=6)
        ax.text(cx + 0.3, cy + 0.3, f"#{rank}", color='blue', fontsize=8, weight='bold')
    ax.set_title("(b) Map Overview & Top Shape Windows\nBlue points represent candidate centroids", fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.15)

    # Panel (c): Shape Distance Heatmap
    ax = axes[0, 2]
    # Smaller shape distance is better (more similar)
    valid_mask = ~np.isnan(score_grid)
    if np.any(valid_mask):
        vmin = np.nanmin(score_grid[valid_mask])
        vmax = np.nanpercentile(score_grid[valid_mask], 92)
        im = ax.imshow(score_grid, extent=[xs_grid[0], xs_grid[-1], ys_grid[-1], ys_grid[0]],
                       aspect='auto', cmap='viridis_r', origin='upper', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Shape Distance (lower is better)')
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', ms=15, label='GT Pose')
    ax.plot(bx, by, 'rX', ms=12, label='Shape Match')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("(c) cv2.matchShapes Distance Heatmap\nSliding grid search results over map", fontsize=11)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")

    # Panel (d): Top Refined Candidates & Ray Casting Score Zoom
    ax = axes[1, 0]
    ranks = range(min(5, len(refined_details := refined_results[:5])))
    lf_scores = [r[3] for r in refined_details]
    y_labels = [f"#{i}\n{math.degrees(r[2]):.0f}°\nMAE:{r[4]:.1f}m" for i, r in enumerate(refined_details)]
    bars = ax.barh(ranks, lf_scores, color='steelblue', alpha=0.8)
    ax.set_yticks(ranks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Likelihood Field Score (higher is better)")
    ax.set_title("(d) Refined Candidates Rank\nHeading and Ray MAE Comparison", fontsize=11)
    for bar in bars:
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.2f}", va='center', ha='left', fontsize=8)

    # Panel (e): Final Zoomed Aligned Scan Overlay
    ax = axes[1, 1]
    ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    
    # Transform points to final estimated pose
    bx, by, byaw, _, _, _ = best_refined
    c_b, s_b = math.cos(byaw), math.sin(byaw)
    tx = bx - (c_b * scan_cx - s_b * scan_cy)
    ty = by - (s_b * scan_cx + c_b * scan_cy)
    final_aligned_pts = np.column_stack([
        c_b * pts_c[:, 0] - s_b * pts_c[:, 1] + tx,
        s_b * pts_c[:, 0] + c_b * pts_c[:, 1] + ty
    ])
    
    step = max(1, len(final_aligned_pts) // 3500)
    ax.scatter(final_aligned_pts[::step, 0], final_aligned_pts[::step, 1],
               s=1.5, c='lime', alpha=0.65, label='Aligned Scan (Est)')
    if tf_gt is not None:
        c_gt, s_gt = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        tx_gt = tf_gt[0] - (c_gt * scan_cx - s_gt * scan_cy)
        ty_gt = tf_gt[1] - (s_gt * scan_cx + c_gt * scan_cy)
        gt_aligned_pts = np.column_stack([
            c_gt * pts_c[:, 0] - s_gt * pts_c[:, 1] + tx_gt,
            s_gt * pts_c[:, 0] + c_gt * pts_c[:, 1] + ty_gt
        ])
        ax.scatter(gt_aligned_pts[::step, 0], gt_aligned_pts[::step, 1],
                   s=0.6, c='cyan', alpha=0.3, label='GT Scan')

    ax.plot(bx, by, 'rX', ms=16, mew=3)
    ax.arrow(bx, by, 2.0 * math.cos(byaw), 2.0 * math.sin(byaw),
             head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=2.5, zorder=10)
    ax.set_xlim(bx - 11.0, bx + 11.0)
    ax.set_ylim(by - 11.0, by + 11.0)
    ax.set_title("(e) Final Zoomed Alignment Overlay\nRed arrow represents estimated robot heading", fontsize=11)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.12)

    # Panel (f): Diagnostic Report Text
    ax = axes[1, 2]
    ax.axis('off')
    
    rep = []
    rep.append("=== Shape Match Localization Report ===")
    rep.append(f"Scan Cloud: {len(scan_pts)} pts")
    rep.append(f"Contour Area: {cv2.contourArea(scan_contour):.0f} px²")
    rep.append(f"Grid Map size: {info['width']}x{info['height']} @ {res*1000:.0f}mm/px")
    rep.append("")
    rep.append(f"Best Candidate Match:")
    rep.append(f"  Centroid: ({bx:.2f}, {by:.2f})")
    rep.append(f"  Heading: {math.degrees(byaw):.1f}°")
    rep.append(f"  Likelihood Score: {best_refined[3]:.3f}")
    rep.append(f"  Ray Cast MAE: {best_refined[4]:.2f}m")
    
    if tf_gt is not None:
        err = math.sqrt((bx - tf_gt[0])**2 + (by - tf_gt[1])**2)
        yaw_diff = byaw - tf_gt[2]
        err_yaw = abs(math.degrees(math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))))
        rep.append("")
        rep.append(f"GT Validation:")
        rep.append(f"  GT Centroid: ({tf_gt[0]:.2f}, {tf_gt[1]:.2f})")
        rep.append(f"  GT Yaw: {math.degrees(tf_gt[2]):.1f}°")
        rep.append(f"  Translational Error: {err:.3f}m")
        rep.append(f"  Heading Error: {err_yaw:.1f}°")
        if err < 1.5 and err_yaw < 12.0:
            rep.append("  ==> SUCCESS: Registration Converged! ✔")
        else:
            rep.append("  ==> FAILED: Misaligned ✘")

    report_text = "\n".join(rep)
    ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9.5, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.55))
    ax.set_title("(f) Localization Diagnostic Report", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Done] Visualization saved: {output_path}")


# ============================================================
# Main Entry Point
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV Contour Bounding Box Sliding Similarity Matcher')
    parser.add_argument('--data', type=str,
                        default='D:/01-Code/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/src/auto_initial_pose_calibrator/scan_viz/debug_match_data.npz',
                        help='Path to debug_match_data.npz')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output visualizer image')
    parser.add_argument('--min-frames', type=int, default=3,
                        help='Min frames for grid consensus (glass filter)')
    parser.add_argument('--outlier-radius', type=float, default=0.3,
                        help='Radius for outlier filter (m)')
    parser.add_argument('--min-neighbors', type=int, default=3,
                        help='Min neighbors inside radius for outlier filter')
    args = parser.parse_args()

    # 1. Load debug npz data
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(args.data)
    
    # 2. Merge & Filter Lidar Scan Points (removing glass and noise)
    scan_pts, scan_cx, scan_cy = merge_and_filter_ranges(
        frame_ranges, frame_tfs, angle_min, angle_inc, max_range=15.0,
        grid_res=0.15, min_frames=args.min_frames,
        outlier_radius=args.outlier_radius, min_neighbors=args.min_neighbors
    )
    
    # 3. Create Scan Shape Contour using OpenCV
    window_m = 22.0
    scan_contour, scan_img, pts_c = points_to_scan_contour(scan_pts, scan_cx, scan_cy, img_size=250, window_size_m=window_m)
    if scan_contour is None:
        print("[ERROR] Failed to extract scan shape contour.")
        sys.exit(1)
    
    # 4. Sliding Window Shape Similarity Matching
    candidates, score_grid, xs_grid, ys_grid = sliding_window_shape_match(
        scan_contour, map_data, info, window_size_m=window_m, step_m=1.0, max_dist_threshold=0.8, n_keep=8
    )

    # 5. Detail Refinement & 180° Ambiguity Solving
    dist_m = build_likelihood_field(map_data, info)
    refined_results = fine_search_candidates(candidates, pts_c, scan_cx, scan_cy, dist_m, map_data, info)

    # 6. ICP Fine-tuning on Best Candidate
    best_refined = refined_results[0]
    bx, by, byaw, lf_score, mae, comb_score = best_refined

    # Construct initial transform matrices
    c_b, s_b = math.cos(byaw), math.sin(byaw)
    R_b = np.array([[c_b, -s_b], [s_b, c_b]])
    t_b = np.array([bx - (c_b * scan_cx - s_b * scan_cy), by - (s_b * scan_cx + c_b * scan_cy)])
    scan_in_map = (R_b @ scan_pts.T).T + t_b

    # Map wall KDTree for ICP
    wall_ys, wall_xs = np.where(map_data == 100)
    map_walls_full = np.column_stack([wall_xs * info['resolution'] + info['origin_x'], 
                                     (info['height'] - 1 - wall_ys) * info['resolution'] + info['origin_y']])
    map_tree = cKDTree(map_walls_full[::max(1, len(map_walls_full) // 5000)])

    # ICP refinement
    R_icp, t_icp = icp_refine(scan_in_map, map_tree)
    final_R = R_icp @ R_b
    final_t = R_icp @ t_b + t_icp
    final_yaw = math.atan2(final_R[1, 0], final_R[0, 0])
    
    # Re-calculate robot centroid in map coordinate
    final_robot_x = final_t[0] + (final_R[0, 0] * scan_cx + final_R[0, 1] * scan_cy)
    final_robot_y = final_t[1] + (final_R[1, 0] * scan_cx + final_R[1, 1] * scan_cy)
    
    # Recalculate scoring parameters post-ICP
    final_lf_score, _ = score_at_pose(pts_c, final_robot_x, final_robot_y, final_yaw, dist_m, info)
    final_mae = ray_cast_score(pts_c, final_robot_x, final_robot_y, final_yaw, map_data, info)
    
    best_refined_updated = (final_robot_x, final_robot_y, final_yaw, final_lf_score, final_mae, final_lf_score - final_mae * 0.04)

    # 7. Visualization
    output_path = args.output or os.path.join(os.path.dirname(args.data), 'shape_contour_match_result.png')
    generate_panels(map_data, info, scan_pts, scan_contour, scan_img, pts_c,
                    score_grid, xs_grid, ys_grid, candidates, best_refined_updated, refined_results, scan_in_map,
                    tf_gt, output_path)
