#!/usr/bin/env python3
"""
诊断脚本：对比 GT 位姿和搜索结果在地图上的真实叠加效果
"""
import os, sys, math, argparse
import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
        default='D:/01-Code/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/src/auto_initial_pose_calibrator/scan_viz/debug_match_data.npz')
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    map_data = data['map_data']
    res = float(data['map_resolution'])
    mw, mh = int(data['map_width']), int(data['map_height'])
    ox, oy = float(data['map_origin_x']), float(data['map_origin_y'])
    tf_gt = data['tf_odom_to_map']
    frame_tfs = data['frame_tfs']
    angle_min = float(data['frame_angle_min'])
    angle_inc = float(data['frame_angle_increment'])

    print(f"Map: {mw}x{mh} @ {res}m, origin=({ox},{oy})")
    print(f"Map shape: {map_data.shape}, dtype={map_data.dtype}")
    print(f"GT TF: ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f} deg)")
    
    # 统计地图内容
    n_free = np.sum(map_data == 0)
    n_wall = np.sum(map_data == 100)
    n_unknown = np.sum(map_data == -1)
    print(f"Map cells: free={n_free}, wall={n_wall}, unknown={n_unknown}, total={map_data.size}")

    # 合并所有帧的点云到 odom 系
    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in data:
        frame_ranges.append(data[f'frame_ranges_{i}'])
        i += 1

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
    merged = np.vstack(all_pts)
    print(f"Merged: {len(merged)} points")

    # 用 GT 变换到 map 系
    c_gt, s_gt = np.cos(tf_gt[2]), np.sin(tf_gt[2])
    gt_mx = c_gt * merged[:, 0] - s_gt * merged[:, 1] + tf_gt[0]
    gt_my = s_gt * merged[:, 0] + c_gt * merged[:, 1] + tf_gt[1]

    # 检查 GT 变换后点云与地图墙壁的匹配
    # 方式1: 直接查 map_data
    gt_cols = ((gt_mx - ox) / res + 0.5).astype(np.int32)
    gt_rows = ((gt_my - oy) / res + 0.5).astype(np.int32)
    in_bounds = (gt_cols >= 0) & (gt_cols < mw) & (gt_rows >= 0) & (gt_rows < mh)
    
    print(f"\n--- GT 变换后点云分析 (row = (y-oy)/res) ---")
    print(f"  Points in map bounds: {np.sum(in_bounds)}/{len(merged)}")
    vals = map_data[gt_rows[in_bounds], gt_cols[in_bounds]]
    print(f"  On free cells (0): {np.sum(vals == 0)}")
    print(f"  On wall cells (100): {np.sum(vals == 100)}")
    print(f"  On unknown cells (-1): {np.sum(vals == -1)}")
    
    # 方式2: 翻转 row = H-1-(y-oy)/res
    gt_rows_flip = (mh - 1 - (gt_my - oy) / res).astype(np.int32)
    in_bounds_flip = (gt_cols >= 0) & (gt_cols < mw) & (gt_rows_flip >= 0) & (gt_rows_flip < mh)
    
    print(f"\n--- GT 变换后点云分析 (row = H-1-(y-oy)/res, 翻转) ---")
    print(f"  Points in map bounds: {np.sum(in_bounds_flip)}/{len(merged)}")
    vals_flip = map_data[gt_rows_flip[in_bounds_flip], gt_cols[in_bounds_flip]]
    print(f"  On free cells (0): {np.sum(vals_flip == 0)}")
    print(f"  On wall cells (100): {np.sum(vals_flip == 100)}")
    print(f"  On unknown cells (-1): {np.sum(vals_flip == -1)}")

    # 构建距离场
    obs = (map_data == 100).astype(np.uint8)
    free = ((1 - obs) * 255).astype(np.uint8)
    dist_px = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist_m = dist_px * res

    # 用两种 row 索引方式评分
    for label, rows_arr, ib in [
        ("row=(y-oy)/res", gt_rows, in_bounds),
        ("row=H-1-(y-oy)/res", gt_rows_flip, in_bounds_flip)
    ]:
        d = dist_m[rows_arr[ib], gt_cols[ib]]
        sigma = 0.3
        scores = np.exp(-d**2 / (2 * sigma**2))
        hit_rate = np.sum(d < 0.15) / np.sum(ib)
        total_score = np.mean(scores) + hit_rate * 0.5
        print(f"\n  [{label}] dist_mean={np.mean(d):.3f}m, dist_median={np.median(d):.3f}m, "
              f"hit_rate(d<0.15m)={hit_rate:.3f}, score={total_score:.4f}")

    # 可视化
    map_bg = np.zeros((mh, mw, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1.0, 1.0, 1.0]
    map_bg[map_data == 100] = [0.1, 0.1, 0.1]
    map_bg[map_data == -1] = [0.75, 0.75, 0.75]
    extent = [ox, ox + mw * res, oy, oy + mh * res]

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    # (a) 地图 + GT 变换后的点云 (origin='lower')
    ax = axes[0, 0]
    ax.imshow(map_bg, origin='lower', extent=extent)
    step = max(1, len(gt_mx) // 3000)
    ax.scatter(gt_mx[::step], gt_my[::step], s=1, c='lime', alpha=0.6)
    ax.set_title(f"GT overlay (origin='lower')\nGT=({tf_gt[0]:.1f},{tf_gt[1]:.1f},{math.degrees(tf_gt[2]):.1f}deg)")
    ax.set_aspect('equal')
    # Zoom to GT area
    cx, cy = gt_mx.mean(), gt_my.mean()
    ax.set_xlim(cx - 15, cx + 15)
    ax.set_ylim(cy - 15, cy + 15)
    ax.grid(True, alpha=0.2)

    # (b) 地图 + GT 变换后的点云 (origin='upper')  
    ax = axes[0, 1]
    ax.imshow(map_bg, origin='upper', extent=[ox, ox + mw * res, oy + mh * res, oy])
    ax.scatter(gt_mx[::step], gt_my[::step], s=1, c='lime', alpha=0.6)
    ax.set_title(f"GT overlay (origin='upper')\nsame data, different display")
    ax.set_aspect('equal')
    ax.set_xlim(cx - 15, cx + 15)
    ax.set_ylim(cy + 15, cy - 15)  # flipped for upper
    ax.grid(True, alpha=0.2)

    # (c) 地图墙壁像素 vs GT 点云像素 (两种 row 方式)
    ax = axes[1, 0]
    wall_rows, wall_cols = np.where(map_data == 100)
    ax.scatter(wall_cols[::5], wall_rows[::5], s=0.3, c='black', alpha=0.3, label='Walls (pixel)')
    ax.scatter(gt_cols[in_bounds][::step], gt_rows[in_bounds][::step], s=1, c='red', alpha=0.5, label='GT row=(y-oy)/res')
    ax.scatter(gt_cols[in_bounds_flip][::step], gt_rows_flip[in_bounds_flip][::step], s=1, c='blue', alpha=0.5, label='GT row=H-1-(y-oy)/res')
    ax.set_title("Pixel-space: walls vs GT scan (两种row)")
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)

    # (d) 距离场热力图 + GT 点云
    ax = axes[1, 1]
    ax.imshow(dist_m, origin='lower', cmap='hot', vmin=0, vmax=3.0,
              extent=extent)
    ax.scatter(gt_mx[::step], gt_my[::step], s=1, c='cyan', alpha=0.6)
    ax.set_title("Distance field (origin='lower') + GT scan")
    ax.set_aspect('equal')
    ax.set_xlim(cx - 15, cx + 15)
    ax.set_ylim(cy - 15, cy + 15)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(args.data), 'gt_diagnosis.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[Done] Saved: {out}")

if __name__ == '__main__':
    main()
