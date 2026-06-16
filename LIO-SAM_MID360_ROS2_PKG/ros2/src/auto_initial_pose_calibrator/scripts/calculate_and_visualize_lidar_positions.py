#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline script: merge lidar scans -> build scan shape -> full-map search to locate on grid map

Key difference from previous versions:
  - Does NOT use tf_odom_to_map as search center (it's the unknown we want to solve)
  - Searches the ENTIRE map for the scan shape alignment
  - Uses multi-resolution strategy for efficiency:
      Phase 1: Very coarse grid (2m step, 15 deg) on heavily downsampled points
      Phase 2: Medium grid (0.5m, 5 deg) around top candidates
      Phase 3: Fine grid (0.05m, 0.5 deg) around the single best

Output: 4-panel figure showing scan shape, map, alignment overlay, trajectory
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import cv2

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("Error: matplotlib required. pip install matplotlib")
    sys.exit(1)

# CJK font setup
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
# Data Loading
# ============================================================

def load_data(npz_path):
    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}")
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
        frame_ranges.append(data[f'frame_ranges_{i}'])
        i += 1

    info = {
        'resolution': resolution, 'width': map_w, 'height': map_h,
        'origin_x': origin_x, 'origin_y': origin_y,
    }

    print(f"  Frames: {len(frame_ranges)},  Map: {map_w}x{map_h} @ {resolution:.3f}m/pix")
    print(f"  Map origin: ({origin_x:.2f}, {origin_y:.2f})")
    print(f"  Map world range: X=[{origin_x:.1f}, {origin_x + map_w*resolution:.1f}], "
          f"Y=[{origin_y:.1f}, {origin_y + map_h*resolution:.1f}]")
    print(f"  TF ground truth (for comparison only): "
          f"({tf_odom_to_map[0]:.2f}, {tf_odom_to_map[1]:.2f}, {math.degrees(tf_odom_to_map[2]):.1f} deg)")

    return map_data, info, tf_odom_to_map, frame_tfs, frame_ranges, angle_min, angle_inc


# ============================================================
# Merge Scans
# ============================================================

def merge_ranges_to_odom(frame_ranges, frame_tfs, angle_min, angle_inc):
    print("Merging all frame scans into odom frame...")
    all_pts = []
    for ranges, tf in zip(frame_ranges, frame_tfs):
        ranges = np.array(ranges, dtype=np.float64)
        valid = (ranges > 0.1) & (ranges < 30.0)
        if not np.any(valid):
            continue
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        x_l = ranges[valid] * np.cos(angles[valid])
        y_l = ranges[valid] * np.sin(angles[valid])
        tx, ty, yaw = tf
        c, s = np.cos(yaw), np.sin(yaw)
        all_pts.append(np.column_stack([
            c * x_l - s * y_l + tx,
            s * x_l + c * y_l + ty
        ]))
    if not all_pts:
        raise ValueError("No valid laser points")
    merged = np.vstack(all_pts)
    print(f"  Merged: {len(merged)} points")
    # Center the points around their centroid for rotation-invariant search
    cx, cy = merged[:, 0].mean(), merged[:, 1].mean()
    print(f"  Centroid (odom): ({cx:.2f}, {cy:.2f})")
    return merged, cx, cy


# ============================================================
# Rasterize for display
# ============================================================

def rasterize_points(pts, resolution, padding_m=1.0):
    x_min = pts[:, 0].min() - padding_m
    x_max = pts[:, 0].max() + padding_m
    y_min = pts[:, 1].min() - padding_m
    y_max = pts[:, 1].max() + padding_m

    w = int(np.ceil((x_max - x_min) / resolution))
    h = int(np.ceil((y_max - y_min) / resolution))
    img = np.zeros((h, w), dtype=np.uint8)

    cols = np.clip(((pts[:, 0] - x_min) / resolution).astype(int), 0, w - 1)
    rows = np.clip((h - 1 - (pts[:, 1] - y_min) / resolution).astype(int), 0, h - 1)
    img[rows, cols] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.dilate(img, kernel, iterations=1)
    return img, x_min, y_min, x_max, y_max


# ============================================================
# Distance transform
# ============================================================

def build_dist_transform(map_data, info):
    obs = (map_data == 100).astype(np.uint8)
    free = ((1 - obs) * 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return dist_px * info['resolution']


# ============================================================
# Score function (vectorized)
# ============================================================

def score_transform(pts_centered, cx_map, cy_map, yaw, dist_m, info, map_data=None):
    """
    Score alignment: place scan centroid at (cx_map, cy_map) in map frame,
    rotate scan by yaw. Evaluate Gaussian likelihood on distance-to-wall.

    pts_centered: points centered around (0,0) = pts_odom - centroid
    """
    c, s = np.cos(yaw), np.sin(yaw)
    mx = c * pts_centered[:, 0] - s * pts_centered[:, 1] + cx_map
    my = s * pts_centered[:, 0] + c * pts_centered[:, 1] + cy_map

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    col = ((mx - ox) / res + 0.5).astype(np.int32)
    row = ((my - oy) / res + 0.5).astype(np.int32)

    valid = (col >= 0) & (col < W) & (row >= 0) & (row < H)

    n_valid = np.sum(valid)
    if n_valid < len(pts_centered) * 0.05:  # relaxed: unknown-heavy maps
        return -1e9

    dists = dist_m[row[valid], col[valid]]

    # Gaussian likelihood, sigma=0.3m
    sigma = 0.3
    scores = np.exp(-dists**2 / (2 * sigma**2))
    # Bonus for high hit rate (points close to walls)
    hit_rate = np.sum(dists < 0.15) / n_valid
    return np.mean(scores) + hit_rate * 0.5



# ============================================================
# Multi-resolution full-map search
# ============================================================

def full_map_search(pts_odom, scan_cx, scan_cy, dist_m, info, frame_tfs, map_data,
                    frame_ranges, angle_min, angle_inc, tf_gt=None):
    """
    3-phase search over the entire map:
      Phase 1: Very coarse (2m, 15 deg) - scan entire map
      Phase 2: Medium (0.5m, 5 deg) - around top-5 candidates
      Phase 3: Fine (0.05m, 0.5 deg) - around single best
    """
    # Center points around centroid for rotation
    pts_c = pts_odom.copy()
    pts_c[:, 0] -= scan_cx
    pts_c[:, 1] -= scan_cy

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    map_w_m = info['width'] * res
    map_h_m = info['height'] * res

    # Use heavily downsampled points for phase 1
    ds1 = max(1, len(pts_c) // 200)
    pts1 = pts_c[::ds1]
    print(f"\n=== Phase 1: Coarse full-map search ===")
    print(f"  Points: {len(pts_c)} -> {len(pts1)} (downsampled)")

    # Search grid over entire map
    margin = 2.0  # keep away from edges
    step_xy = 2.0
    step_yaw = math.radians(15.0)

    xs = np.arange(ox + margin, ox + map_w_m - margin, step_xy)
    ys = np.arange(oy + margin, oy + map_h_m - margin, step_xy)
    yaws = np.arange(0, 2 * math.pi, step_yaw)

    total = len(xs) * len(ys) * len(yaws)
    print(f"  Grid: X={len(xs)}, Y={len(ys)}, Yaw={len(yaws)}, total={total}")

    t0 = time.time()
    results = []
    count = 0
    for yaw in yaws:
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        # Pre-rotate all points for this yaw
        rx = c_y * pts1[:, 0] - s_y * pts1[:, 1]
        ry = s_y * pts1[:, 0] + c_y * pts1[:, 1]

        for x in xs:
            mx = rx + x
            for y in ys:
                my_arr = ry + y
                # Inline scoring for speed
                col = ((mx - ox) / res + 0.5).astype(np.int32)
                row_arr = ((my_arr - oy) / res + 0.5).astype(np.int32)
                v = (col >= 0) & (col < info['width']) & (row_arr >= 0) & (row_arr < info['height'])
                nv = np.sum(v)
                if nv < len(pts1) * 0.05:
                    count += 1
                    continue
                d = dist_m[row_arr[v], col[v]]
                sc = np.mean(np.exp(-d**2 / 0.18)) + np.sum(d < 0.15) / nv * 0.5
                results.append((sc, x, y, yaw))
                count += 1

        if count % 50000 == 0 and count > 0:
            elapsed = time.time() - t0
            print(f"  Progress: {count}/{total} ({elapsed:.1f}s)")

    elapsed1 = time.time() - t0
    print(f"  Phase 1 done in {elapsed1:.1f}s, evaluated {len(results)} positions")

    if not results:
        raise ValueError("No valid candidate found in Phase 1!")

    # Extract top candidates using NMS
    results.sort(key=lambda r: r[0], reverse=True)
    candidates = []
    for sc, x, y, yaw in results:
        is_new = True
        for _, cx2, cy2, cyaw2 in candidates:
            if math.sqrt((x-cx2)**2 + (y-cy2)**2) < 3.0:
                is_new = False
                break
        if is_new:
            candidates.append((sc, x, y, yaw))
            if len(candidates) >= 5:
                break

    print(f"  Top candidates (NMS):")
    for i, (sc, x, y, yaw) in enumerate(candidates):
        print(f"    #{i}: score={sc:.4f}, pos=({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f} deg")

    # Diagnostic: evaluate GT centroid position with Phase 1 points
    if tf_gt is not None:
        c_gt, s_gt = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        gt_cx = c_gt * scan_cx - s_gt * scan_cy + tf_gt[0]
        gt_cy = s_gt * scan_cx + c_gt * scan_cy + tf_gt[1]
        gt_sc = score_transform(pts1, gt_cx, gt_cy, tf_gt[2], dist_m, info, map_data)
        print(f"\n  [DIAG] GT centroid ({gt_cx:.2f}, {gt_cy:.2f}), yaw={math.degrees(tf_gt[2]):.1f} deg")
        print(f"  [DIAG] GT score (Phase1 pts): {gt_sc:.4f}")
        # Also score with all points
        gt_sc_full = score_transform(pts_c, gt_cx, gt_cy, tf_gt[2], dist_m, info, map_data)
        print(f"  [DIAG] GT score (all pts): {gt_sc_full:.4f}")
        # Score the Phase 1 winner with all points too
        best_sc_full = score_transform(pts_c, candidates[0][1], candidates[0][2], candidates[0][3], dist_m, info, map_data)
        print(f"  [DIAG] Phase1 #0 score (all pts): {best_sc_full:.4f}")
        print()

    # === Phase 2: Medium search around each candidate ===
    print(f"\n=== Phase 2: Medium search around top candidates ===")
    ds2 = max(1, len(pts_c) // 600)
    pts2 = pts_c[::ds2]
    print(f"  Points: {len(pts2)}")

    phase2_results = []
    for _, cx0, cy0, yaw0 in candidates:
        step_xy2 = 0.5
        step_yaw2 = math.radians(5.0)
        range_xy2 = 3.0
        range_yaw2 = math.radians(20.0)

        xs2 = np.arange(cx0 - range_xy2, cx0 + range_xy2 + 1e-5, step_xy2)
        ys2 = np.arange(cy0 - range_xy2, cy0 + range_xy2 + 1e-5, step_xy2)
        yaws2 = np.arange(yaw0 - range_yaw2, yaw0 + range_yaw2 + 1e-5, step_yaw2)

        for yaw in yaws2:
            for x in xs2:
                for y in ys2:
                    sc = score_transform(pts2, x, y, yaw, dist_m, info, map_data)
                    if sc > -1e8:
                        phase2_results.append((sc, x, y, yaw))

    phase2_results.sort(key=lambda r: r[0], reverse=True)

    # Collect top-N distinct Phase 2 candidates (NMS again to get diverse results)
    phase2_top = []
    for sc, x, y, yaw in phase2_results:
        is_new = True
        for _, cx2, cy2, _ in phase2_top:
            if math.sqrt((x-cx2)**2 + (y-cy2)**2) < 2.0:
                is_new = False
                break
        if is_new:
            phase2_top.append((sc, x, y, yaw))
            if len(phase2_top) >= 8:
                break

    print(f"  Phase 2 top candidates (NMS):")
    for i, (sc, x, y, yaw) in enumerate(phase2_top):
        print(f"    #{i}: score={sc:.4f}, pos=({x:.2f}, {y:.2f}), yaw={math.degrees(yaw):.1f} deg")

    # === Phase 3: Fine search ===
    print(f"\n=== Phase 3: Fine search ===")
    ds3 = max(1, len(pts_c) // 2000)
    pts3 = pts_c[::ds3]
    print(f"  Points: {len(pts3)}")

    bx, by, byaw = phase2_top[0][1], phase2_top[0][2], phase2_top[0][3]
    step_xy3 = 0.05
    step_yaw3 = math.radians(0.5)
    range_xy3 = 0.6
    range_yaw3 = math.radians(6.0)

    xs3 = np.arange(bx - range_xy3, bx + range_xy3 + 1e-5, step_xy3)
    ys3 = np.arange(by - range_xy3, by + range_xy3 + 1e-5, step_xy3)
    yaws3 = np.arange(byaw - range_yaw3, byaw + range_yaw3 + 1e-5, step_yaw3)

    best_s3, best_p3 = -1e9, (bx, by, byaw)
    for yaw in yaws3:
        for x in xs3:
            for y in ys3:
                sc = score_transform(pts3, x, y, yaw, dist_m, info, map_data)
                if sc > best_s3:
                    best_s3 = sc
                    best_p3 = (x, y, yaw)

    print(f"  Phase 3 best: score={best_s3:.4f}, pos=({best_p3[0]:.2f}, {best_p3[1]:.2f}), "
          f"yaw={math.degrees(best_p3[2]):.1f} deg")

    # Convert: scan centroid placed at (best_x, best_y) with yaw
    # odom->map: map_pt = R(yaw) * odom_pt + t
    #   t = (best_x, best_y) - R(yaw) * (scan_cx, scan_cy)
    best_x, best_y, best_yaw = best_p3
    c_b, s_b = math.cos(best_yaw), math.sin(best_yaw)
    t_x = best_x - (c_b * scan_cx - s_b * scan_cy)
    t_y = best_y - (s_b * scan_cx + c_b * scan_cy)

    print(f"\n[RESULT] Estimated odom->map transform:")
    print(f"  t = ({t_x:.3f}, {t_y:.3f}), yaw = {math.degrees(best_yaw):.2f} deg")

    return t_x, t_y, best_yaw


# ============================================================
# Helpers
# ============================================================

def transform_points(pts, tx, ty, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([
        c * pts[:, 0] - s * pts[:, 1] + tx,
        s * pts[:, 0] + c * pts[:, 1] + ty
    ])

def transform_pose(pose, tx, ty, yaw):
    px, py, pyaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return c*px - s*py + tx, s*px + c*py + ty, pyaw + yaw


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="D:/01-Code/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/src/auto_initial_pose_calibrator/scan_viz/debug_match_data.npz")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # 1. Load
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(args.data)

    # 2. Merge scans
    merged_pts, scan_cx, scan_cy = merge_ranges_to_odom(frame_ranges, frame_tfs, angle_min, angle_inc)

    # 3. Build distance transform
    dist_m = build_dist_transform(map_data, info)
    print(f"  Distance transform: max={dist_m.max():.2f}m")

    # 4. Rasterize scan for display
    res = info['resolution']
    scan_img, sx_min, sy_min, sx_max, sy_max = rasterize_points(merged_pts, res)

    # 4.5. Diagnostic: score the GT position to understand the baseline
    gt_yaw = tf_gt[2]
    c_gt, s_gt = np.cos(gt_yaw), np.sin(gt_yaw)
    # GT centroid placement in map frame
    gt_cx = c_gt * scan_cx - s_gt * scan_cy + tf_gt[0]
    gt_cy = s_gt * scan_cx + c_gt * scan_cy + tf_gt[1]
    gt_score = score_transform(
        merged_pts - np.array([scan_cx, scan_cy]),
        gt_cx, gt_cy, gt_yaw, dist_m, info)
    print(f"\n[DIAG] GT centroid placement: ({gt_cx:.2f}, {gt_cy:.2f}), yaw={math.degrees(gt_yaw):.1f} deg")
    print(f"[DIAG] GT score (full points): {gt_score:.4f}")

    # 5. Full-map search (no reliance on TF ground truth)
    t_x, t_y, best_yaw = full_map_search(merged_pts, scan_cx, scan_cy, dist_m, info,
                                          frame_tfs, map_data, frame_ranges, angle_min, angle_inc,
                                          tf_gt=tf_gt)

    # Compare with ground truth (for evaluation only)
    pos_err = math.sqrt((t_x - tf_gt[0])**2 + (t_y - tf_gt[1])**2)
    yaw_diff = best_yaw - tf_gt[2]
    yaw_err = abs(math.degrees(math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))))
    print(f"\n[EVAL] vs TF ground truth:")
    print(f"  GT:    t=({tf_gt[0]:.2f}, {tf_gt[1]:.2f}), yaw={math.degrees(tf_gt[2]):.1f} deg")
    print(f"  Found: t=({t_x:.2f}, {t_y:.2f}), yaw={math.degrees(best_yaw):.1f} deg")
    print(f"  Error: pos={pos_err:.3f}m, yaw={yaw_err:.2f} deg")

    # 6. Transform everything to map frame
    aligned_pts = transform_points(merged_pts, t_x, t_y, best_yaw)
    lidar_poses = np.array([transform_pose(ft, t_x, t_y, best_yaw) for ft in frame_tfs])
    gt_aligned = transform_points(merged_pts, *tf_gt)

    # ─── 7. Visualization ───
    H, W = map_data.shape
    ox, oy = info['origin_x'], info['origin_y']
    extent = [ox, ox + W * res, oy, oy + H * res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1.0, 1.0, 1.0]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.75, 0.75, 0.75]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # (a) Merged scan shape
    ax = axes[0, 0]
    scan_ext = [sx_min, sx_max, sy_min, sy_max]
    ax.imshow(scan_img, cmap='gray_r', extent=scan_ext, origin='upper')
    for i, ft in enumerate(frame_tfs):
        ax.plot(ft[0], ft[1], 'r.', ms=4)
        if i % 5 == 0:
            ax.text(ft[0]+0.1, ft[1]+0.1, str(i), fontsize=6, color='red')
    ax.set_aspect('equal')
    ax.set_title("Merged Scan Shape (odom frame)\nThis is the shape to match on map", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.2)

    # (b) Map
    ax = axes[0, 1]
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.set_aspect('equal')
    ax.set_title("Occupancy Grid Map\n(full map to search)", fontsize=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.2)

    # (c) Overlay
    ax = axes[1, 0]
    ax.set_aspect('equal')
    ax.imshow(map_bg, origin='lower', extent=extent)
    step = max(1, len(aligned_pts) // 4000)
    ax.scatter(aligned_pts[::step, 0], aligned_pts[::step, 1],
               s=1.5, c='lime', alpha=0.6, label='Aligned scan (found)')
    ax.scatter(gt_aligned[::step, 0], gt_aligned[::step, 1],
               s=0.8, c='cyan', alpha=0.3, label='Aligned scan (GT TF)')
    ax.set_title(f"Scan Shape Overlaid on Map\n"
                 f"Found t=({t_x:.2f},{t_y:.2f}), yaw={math.degrees(best_yaw):.1f} deg  |  "
                 f"err={pos_err:.2f}m/{yaw_err:.1f} deg", fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.15)
    pad = 5.0
    cv_x, cv_y = aligned_pts[:, 0].mean(), aligned_pts[:, 1].mean()
    hw = max((aligned_pts[:, 0].max() - aligned_pts[:, 0].min()) / 2, 5) + pad
    hh = max((aligned_pts[:, 1].max() - aligned_pts[:, 1].min()) / 2, 5) + pad
    ax.set_xlim(cv_x - hw, cv_x + hw)
    ax.set_ylim(cv_y - hh, cv_y + hh)

    # (d) Trajectory
    ax = axes[1, 1]
    ax.set_aspect('equal')
    ax.imshow(map_bg, origin='lower', extent=extent)
    ax.scatter(aligned_pts[::step, 0], aligned_pts[::step, 1],
               s=0.5, c='lime', alpha=0.25)
    ax.plot(lidar_poses[:, 0], lidar_poses[:, 1],
            c='blue', lw=2, ls='-', marker='o', ms=4, label='Lidar trajectory')
    for idx in range(len(lidar_poses)):
        xm, ym, yawm = lidar_poses[idx]
        al = 0.5
        ax.arrow(xm, ym, al*math.cos(yawm), al*math.sin(yawm),
                 head_width=0.15, head_length=0.12, fc='red', ec='darkred', zorder=10)
        ax.text(xm+0.15, ym+0.15, str(idx), fontsize=7, color='navy', zorder=11,
                bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'))
    ax.plot(lidar_poses[0, 0], lidar_poses[0, 1], 'go', ms=9, label='Start')
    ax.plot(lidar_poses[-1, 0], lidar_poses[-1, 1], 'rs', ms=9, label='End')
    ax.set_title(f"Lidar Trajectory on Map\n"
                 f"Estimated odom->map: t=({t_x:.2f},{t_y:.2f}), yaw={math.degrees(best_yaw):.1f} deg",
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.15)
    pad2 = 8.0
    ax.set_xlim(lidar_poses[:, 0].min() - pad2, lidar_poses[:, 0].max() + pad2)
    ax.set_ylim(lidar_poses[:, 1].min() - pad2, lidar_poses[:, 1].max() + pad2)

    plt.tight_layout()
    output_path = args.output or os.path.join(os.path.dirname(args.data), "calculated_lidar_trajectory.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Done] Saved: {output_path}")


if __name__ == "__main__":
    main()
