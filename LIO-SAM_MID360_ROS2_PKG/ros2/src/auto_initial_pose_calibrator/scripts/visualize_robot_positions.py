#!/usr/bin/env python3
"""
v6: 扫描栅格化 + 白色区域重叠最大化

核心思路：
1. 合并30帧 → First-Return过滤 → 生成扫描占据栅格
2. 扫描栅格有：占据(墙壁) + 空闲(free) + 未知
3. 地图栅格有：占据(wall=100) + 空闲(free=0) + 未知(-1)
4. 对每个候选位姿，将扫描栅格变换到map系
5. 评分 = 扫描free ∩ 地图free 的面积 / 扫描free 总面积
   即：扫描的空闲区域有多少落在地图的空闲区域内
"""

import os, sys, math, argparse
import numpy as np
import cv2
from scipy.ndimage import shift as ndshift

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("需要 matplotlib"); sys.exit(1)


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    map_data = d['map_data']
    res = float(d['map_resolution'])
    mw, mh = int(d['map_width']), int(d['map_height'])
    ox, oy = float(d['map_origin_x']), float(d['map_origin_y'])
    tf_init = d['tf_odom_to_map']
    frame_tfs = d['frame_tfs']
    angle_min = float(d['frame_angle_min'])
    angle_inc = float(d['frame_angle_increment'])
    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in d:
        frame_ranges.append(d[f'frame_ranges_{i}']); i += 1
    return map_data, {'resolution': res, 'width': mw, 'height': mh,
                      'origin_x': ox, 'origin_y': oy}, tf_init, frame_tfs, frame_ranges, angle_min, angle_inc


def merge_all_frames(frame_ranges, frame_tfs, angle_min, angle_inc):
    all_pts = []
    for ranges, ft in zip(frame_ranges, frame_tfs):
        r = np.array(ranges, dtype=np.float64)
        valid = (r > 0.1) & (r < 30.0)
        a = angle_min + np.arange(len(r)) * angle_inc
        pts = np.column_stack([r[valid] * np.cos(a[valid]), r[valid] * np.sin(a[valid])])
        c, s = math.cos(ft[2]), math.sin(ft[2])
        R = np.array([[c, -s], [s, c]])
        all_pts.append((R @ pts.T).T + [ft[0], ft[1]])
    return np.vstack(all_pts)


def first_return_filter(merged_pts, n_bins=108, min_wall_dist=2.0, max_wall_dist=15.0):
    ranges = np.sqrt(merged_pts[:, 0]**2 + merged_pts[:, 1]**2)
    angles = np.arctan2(merged_pts[:, 1], merged_pts[:, 0])
    bin_edges = np.linspace(-math.pi, math.pi, n_bins + 1)
    fr_pts = []
    for b in range(n_bins):
        mask = (angles >= bin_edges[b]) & (angles < bin_edges[b + 1])
        if not np.any(mask): continue
        br, bp = ranges[mask], merged_pts[mask]
        wm = (br >= min_wall_dist) & (br <= max_wall_dist)
        if not np.any(wm): continue
        wr, wp = br[wm], bp[wm]
        idx = np.argsort(wr)[:min(2, len(wr))]
        fr_pts.append(wp[idx])
    return np.vstack(fr_pts) if fr_pts else np.empty((0, 2))


def create_scan_grid(pts, scan_res=0.1):
    """从点云创建扫描占据栅格（相对于点云中心）"""
    center = pts.mean(axis=0)
    relative = pts - center

    # 像素坐标
    gx = ((relative[:, 0] - relative[:, 0].min()) / scan_res).astype(int)
    gy = ((relative[:, 1] - relative[:, 1].min()) / scan_res).astype(int)
    gw, gh = gx.max() + 1, gy.max() + 1

    # 占据栅格: 1=占据(墙壁), 0=未知
    occ = np.zeros((gh, gw), dtype=np.uint8)
    valid = (gx >= 0) & (gx < gw) & (gy >= 0) & (gy < gh)
    occ[gy[valid], gx[valid]] = 1

    # 膨胀占据区域（模拟墙壁宽度）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    occ_wall = cv2.dilate(occ, kernel, iterations=2)

    # FloodFill从中心填充空闲区域
    free = np.zeros_like(occ_wall)
    # 中心点
    cx, cy = gw // 2, gh // 2
    # FloodFill: 从中心出发，填充未被占据的区域
    flood = occ_wall.copy()
    flood_mask = np.zeros((gh + 2, gw + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (cx, cy), 128)
    free = (flood == 128).astype(np.uint8)  # 空闲区域

    return occ_wall, free, center, scan_res


def score_free_overlap(free_scan, map_data, dx, dy, scan_res, map_info):
    """计算扫描free区域与地图free区域的重叠率"""
    res = map_info['resolution']
    mw, mh = map_info['width'], map_info['height']
    ox, oy = map_info['origin_x'], map_info['origin_y']

    # 扫描free区域的像素坐标
    sh, sw = free_scan.shape
    scan_ys, scan_xs = np.where(free_scan > 0)
    if len(scan_xs) == 0:
        return 0, 0, 0

    # 扫描像素 → 世界坐标（相对于扫描中心 + 偏移）
    # 扫描中心在scan_res坐标系下的位置
    center_x = sw / 2 * scan_res
    center_y = sh / 2 * scan_res

    world_x = scan_xs * scan_res - center_x + dx
    world_y = scan_ys * scan_res - center_y + dy

    # 世界坐标 → 地图像素坐标
    map_cols = ((world_x - ox) / res).astype(int)
    map_rows = ((mh - 1 - (world_y - oy) / res)).astype(int)

    # 边界检查
    in_bounds = (map_cols >= 0) & (map_cols < mw) & (map_rows >= 0) & (map_rows < mh)
    n_total = np.sum(in_bounds)

    if n_total == 0:
        return 0, 0, 0

    # 检查这些位置在地图中是否是free
    map_cols_b = map_cols[in_bounds]
    map_rows_b = map_rows[in_bounds]
    map_vals = map_data[map_rows_b, map_cols_b]
    n_free_overlap = np.sum(map_vals == 0)  # 地图free区域

    # 也检查是否撞墙
    n_wall_hit = np.sum(map_vals == 100)

    overlap_rate = n_free_overlap / n_total if n_total > 0 else 0
    return overlap_rate, n_free_overlap, n_wall_hit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default='D:/01-Code/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/src/auto_initial_pose_calibrator/scan_viz/debug_match_data.npz')
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    map_data, info, tf_init, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(args.data)
    res = info['resolution']
    mw, mh = info['width'], info['height']
    ox, oy = info['origin_x'], info['origin_y']

    # 1. 合并 + First-Return
    print('=== 合并 + First-Return ===')
    merged = merge_all_frames(frame_ranges, frame_tfs, angle_min, angle_inc)
    fr_pts = first_return_filter(merged, n_bins=108, min_wall_dist=2.0, max_wall_dist=15.0)
    print(f'First-Return: {len(fr_pts)} 点')

    # 2. 创建扫描栅格
    print('\n=== 创建扫描栅格 ===')
    scan_occ, scan_free, scan_center, scan_res = create_scan_grid(fr_pts, scan_res=0.15)
    print(f'扫描栅格: {scan_free.shape}, free像素: {np.sum(scan_free)}')

    # 3. 搜索最佳位姿（tf附近）
    init_x, init_y, init_yaw = tf_init
    print(f'\n搜索: tf=({init_x:.2f},{init_y:.2f},{math.degrees(init_yaw):.1f}°)')

    best_score = -1
    best_x, best_y, best_yaw = init_x, init_y, init_yaw

    pos_r, pos_s = 5.0, 0.5  # 更大搜索范围
    yaw_r, yaw_s = math.radians(45), math.radians(5)

    xs = np.arange(init_x - pos_r, init_x + pos_r + pos_s, pos_s)
    ys = np.arange(init_y - pos_r, init_y + pos_r + pos_s, pos_s)
    yaws = np.arange(init_yaw - yaw_r, init_yaw + yaw_r + yaw_s, yaw_s)

    total = len(xs) * len(ys) * len(yaws)
    print(f'搜索: {len(xs)}x{len(ys)}x{len(yaws)}={total}')

    # 预旋转扫描free区域
    count = 0
    for yaw in yaws:
        # 旋转扫描free区域
        M = cv2.getRotationMatrix2D((scan_free.shape[1]//2, scan_free.shape[0]//2),
                                     math.degrees(yaw), 1.0)
        rotated_free = cv2.warpAffine(scan_free, M, (scan_free.shape[1], scan_free.shape[0]))

        for ax in xs:
            for ay in ys:
                score, n_free, n_wall = score_free_overlap(
                    rotated_free, map_data, ax, ay, scan_res, info)
                # 惩罚撞墙
                score -= n_wall * 0.001
                if score > best_score:
                    best_score = score
                    best_x, best_y, best_yaw = ax, ay, yaw
                count += 1
                if count % 5000 == 0:
                    print(f'  {count}/{total} ... best={best_score:.3f}')

    print(f'\n最佳: ({best_x:.2f},{best_y:.2f},{math.degrees(best_yaw):.1f}°) score={best_score:.3f}')

    # 4. 变换所有扫描点到map系
    rc, rs = math.cos(best_yaw), math.sin(best_yaw)
    R = np.array([[rc, -rs], [rs, rc]])
    all_map = (R @ merged.T).T + [best_x, best_y]
    fr_map = (R @ fr_pts.T).T + [best_x, best_y]

    # 最终评分
    M_final = cv2.getRotationMatrix2D((scan_free.shape[1]//2, scan_free.shape[0]//2),
                                       math.degrees(best_yaw), 1.0)
    final_free = cv2.warpAffine(scan_free, M_final, (scan_free.shape[1], scan_free.shape[0]))
    final_score, n_free, n_wall = score_free_overlap(final_free, map_data, best_x, best_y, scan_res, info)
    print(f'最终评分: free重叠={final_score:.3f}, free像素={n_free}, 撞墙={n_wall}')

    # 5. 画图
    map_disp = np.zeros((mh, mw, 3), dtype=np.float32)
    map_disp[map_data == 0] = [1,1,1]; map_disp[map_data == 100] = [0,0,0]; map_disp[map_data == -1] = [.7,.7,.7]
    extent = [ox, ox+mw*res, oy, oy+mh*res]

    fig, axes = plt.subplots(2, 2, figsize=(20, 18))

    # (a) 扫描栅格
    ax = axes[0,0]; ax.set_aspect('equal')
    scan_vis = np.zeros((*scan_free.shape, 3), dtype=np.float32)
    scan_vis[scan_occ > 0] = [0, 0, 0]  # 墙壁=黑
    scan_vis[scan_free > 0] = [1, 1, 1]  # 空闲=白
    scan_vis[(scan_occ == 0) & (scan_free == 0)] = [0.7, 0.7, 0.7]  # 未知=灰
    ax.imshow(scan_vis, origin='lower')
    ax.set_title(f'Scan Grid: {scan_free.shape}, free={np.sum(scan_free)}px')
    ax.grid(True, alpha=0.2)

    # (b) 重叠可视化
    ax = axes[0,1]; ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    # 画扫描free区域在map系的位置
    sh, sw = final_free.shape
    scan_ys, scan_xs = np.where(final_free > 0)
    world_x = scan_xs * scan_res - sw/2*scan_res + best_x
    world_y = scan_ys * scan_res - sh/2*scan_res + best_y
    ax.scatter(world_x, world_y, s=1, c='cyan', alpha=0.3, label=f'Scan free ({len(scan_xs)})')
    ax.plot(best_x, best_y, 'rX', markersize=16, markeredgewidth=3)
    ax.set_title(f'Free Overlap: {final_score:.3f}')
    ax.legend(); ax.grid(True, alpha=0.15)

    # (c) 全量扫描
    ax = axes[1,0]; ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    ax.scatter(all_map[:,0], all_map[:,1], s=0.5, c='cyan', alpha=0.2, edgecolors='none')
    ax.plot(best_x, best_y, 'rX', markersize=18, markeredgewidth=4, zorder=10)
    ax.set_title(f'Scan on Map (free overlap={final_score:.3f})')
    ax.grid(True, alpha=0.15)

    # (d) 局部放大
    ax = axes[1,1]; ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    ax.scatter(fr_map[:,0], fr_map[:,1], s=4, c='cyan', alpha=0.6, edgecolors='none')
    ax.plot(best_x, best_y, 'rX', markersize=18, markeredgewidth=4, zorder=10)
    hw = max(10, (fr_map[:,0].max()-fr_map[:,0].min())/2*1.3)
    hh = max(10, (fr_map[:,1].max()-fr_map[:,1].min())/2*1.3)
    ax.set_xlim(best_x-hw, best_x+hw); ax.set_ylim(best_y-hh, best_y+hh)
    ax.set_title('Zoomed')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    out = args.output or os.path.join(os.path.dirname(args.data), 'free_overlap_v6.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f'\n已保存: {out}')


if __name__ == '__main__':
    main()
