#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
室外 .npz 批量分析脚本（v5 - 直接使用预合并 scan_points_odom）
============================================================

核心思路（与 global_match2.py 一致）:
  - NPZ 中已有 scan_points_odom（采集阶段预合并的全帧扫描点云，~500pts）
  - 直接使用该合并点云做双模板全局匹配，无需逐帧重建
  - 每个 NPZ 一次匹配 = 1 张结果图 + 1 个定位点

室外参数:
  - min_wall_coverage_ratio: 0.10   （室外墙极稀疏）
  - scale_ref_pixels: 500000
  - scale_max: 2.5

用法:
  cd dog_slam
  python3 .../analyze_extend_outdoor.py
  python3 .../analyze_extend_outdoor.py --npz extend/debug_match_data_0.npz
"""

import os, sys, math, time, argparse, glob
import numpy as np
import cv2

# 自动定位路径（脚本深度嵌套在包内）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CALIB_DIR = os.path.join(_SCRIPT_DIR, '..', 'src')
# scripts/ -> auto_initial_pose_calibrator -> src -> ros2 -> LIO-SAM_MID360_ROS2_PKG -> repo_root
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, '..', '..', '..', '..', '..'))
_DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, 'extend')

if _CALIB_DIR not in sys.path:
    sys.path.insert(0, _CALIB_DIR)

from calib_lib.scoring import (
    MapContext, build_dual_template_maps, dual_template_global_match,
    compute_wall_coverage, build_likelihood, score_points,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    # 尝试使用系统中文黑体，回退到默认字体
    _cn_fonts = [f for f in fm.fontManager.ttflist
                 if 'Heiti' in f.name or 'PingFang' in f.name or 'Noto Sans CJK' in f.name]
    if _cn_fonts:
        plt.rcParams['font.family'] = _cn_fonts[0].name
except ImportError:
    print("[ERROR] matplotlib required: pip install matplotlib")
    sys.exit(1)


OUTDOOR_DT_PARAMS = {
    'coarse_angle_step_deg': 5.0,      # 粗搜步长（室外点位少, 放宽）
    'fine_angle_step_deg': 1.0,
    'penalty_weight': 3.0,
    'scan_max_points': 500,              # 多帧合并后点数远多于单帧
    'min_wall_coverage_ratio': 0.10,    # 室外墙极稀疏, 再次放宽
    'free_space_penalty_weight': 0.1,
    'dt_scale_ref_pixels': 500000,
    'dt_scale_max': 2.5,
}

class DuckMapInfo:
    class Origin:
        class Position:
            x = 0.0; y = 0.0
        position = Position()
    def __init__(self, resolution, width, height, origin_x, origin_y):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin = self.Origin()
        self.origin.position.x = origin_x
        self.origin.position.y = origin_y


def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    map_data = d['map_data']
    res = float(d['map_resolution'])
    W = int(d['map_width'])
    H = int(d['map_height'])
    ox = float(d['map_origin_x'])
    oy = float(d['map_origin_y'])
    tf_gt = (float(d['tf_odom_to_map'][0]), float(d['tf_odom_to_map'][1]), float(d['tf_odom_to_map'][2]))
    frame_tfs = d['frame_tfs']
    # 直接使用预合并的扫描点云（参考 global_match2.py，已累积多帧扫描）
    scan_points_odom = np.array(d['scan_points_odom'], dtype=np.float64)
    d.close()

    map_info = DuckMapInfo(res, W, H, ox, oy)
    map_ctx = MapContext(map_data=map_data, map_info=map_info)
    return map_ctx, frame_tfs, scan_points_odom, tf_gt


def build_local_scan_from_odom_points(scan_points_odom, frame_tfs, lidar_range=30.0):
    """将 UTM 坐标的 scan_points_odom 转换为局部坐标系下的密集扫描。

    scan_points_odom 是跨多帧积累的 UTM 绝对坐标点云（每帧 ~15-20 点）。
    直接居中后跨度达 22-78km，远超地图大小，无法匹配。
    
    正确做法：将每个扫描点归属到最近的 odom 帧，
    以该帧位置为原点转为局部坐标，合并所有帧的局部点。
    """
    if len(scan_points_odom) == 0 or len(frame_tfs) == 0:
        return np.zeros((0, 2))

    all_local = []
    for i in range(len(frame_tfs)):
        tx, ty = frame_tfs[i][0], frame_tfs[i][1]
        # 找该帧附近的扫描点（lidar 有效距离内）
        dists = np.sqrt((scan_points_odom[:, 0] - tx) ** 2 +
                         (scan_points_odom[:, 1] - ty) ** 2)
        nearby = dists < lidar_range
        if np.sum(nearby) < 3:
            continue
        # 转到以该帧位置为原点的局部坐标
        local_pts = scan_points_odom[nearby] - np.array([tx, ty])
        all_local.append(local_pts)

    if not all_local:
        return np.zeros((0, 2))

    merged = np.vstack(all_local)
    # 去重（可能有相邻帧共享的扫描点）
    if len(merged) > 1:
        _, unique_idx = np.unique(np.round(merged, 2), axis=0, return_index=True)
        merged = merged[np.sort(unique_idx)]
    return merged


def analyze_one(npz_path, output_dir, params):
    """分析单个 NPZ（scan_points_odom 按帧归属→局部坐标系→合并匹配）。"""
    label = os.path.basename(npz_path).replace('.npz', '')
    print(f"\n{'='*70}")
    print(f"[{label}] 室外扫描匹配分析")
    print(f"{'='*70}")

    t0 = time.time()
    map_ctx, frame_tfs, scan_points_odom, tf_gt = load_npz(npz_path)

    res = map_ctx.map_info.resolution
    W, H = map_ctx.map_info.width, map_ctx.map_info.height

    n_frames = len(frame_tfs)
    n_free = int(np.sum(map_ctx.map_data == 0))
    total_cells = map_ctx.map_data.size

    # 判断 GT 是否为 UTM 大坐标
    gt_is_utm = abs(tf_gt[0]) > 10000 or abs(tf_gt[1]) > 10000
    gt_valid = not gt_is_utm

    # 帧间位移统计
    inter_dists = []
    for j in range(1, n_frames):
        d = math.sqrt((frame_tfs[j][0]-frame_tfs[j-1][0])**2 + (frame_tfs[j][1]-frame_tfs[j-1][1])**2)
        inter_dists.append(d)
    mean_dist = np.mean(inter_dists) if inter_dists else 0
    odom_span = math.sqrt(
        (np.max(frame_tfs[:,0]) - np.min(frame_tfs[:,0]))**2 +
        (np.max(frame_tfs[:,1]) - np.min(frame_tfs[:,1]))**2
    )

    print(f"  NPZ: {npz_path}")
    print(f"  地图: {W}x{H} @ {res:.3f}m | 空地={100*n_free/total_cells:.1f}%")
    print(f"  帧数: {n_frames}, 帧间位移μ={mean_dist:.0f}m, odom跨度={odom_span:.0f}m, map={W*res:.0f}m×{H*res:.0f}m")
    if gt_valid:
        print(f"  GT: t=({tf_gt[0]:.1f},{tf_gt[1]:.1f}) yaw={math.degrees(tf_gt[2]):.1f}°")
    else:
        print(f"  GT: UTM大坐标 ({tf_gt[0]:.0f},{tf_gt[1]:.0f}) yaw={math.degrees(tf_gt[2]):.1f}° [不适用局部地图]")

    # === 将 scan_points_odom 按帧归属转局部坐标后合并 ===
    scan_local = build_local_scan_from_odom_points(scan_points_odom, frame_tfs)
    n_pts = len(scan_local)

    if n_pts > params['scan_max_points']:
        indices = np.random.choice(n_pts, params['scan_max_points'], replace=False)
        scan_local = scan_local[indices]
        n_pts = len(scan_local)

    span_x = scan_local[:, 0].max() - scan_local[:, 0].min() if n_pts > 0 else 0
    span_y = scan_local[:, 1].max() - scan_local[:, 1].min() if n_pts > 0 else 0
    print(f"  扫描: {n_pts}pts (原始{len(scan_points_odom)}pt→{n_frames}帧归属合并), span={span_x:.0f}m×{span_y:.0f}m")

    if n_pts < 10:
        print(f"  [FAIL] 扫描点太少({n_pts})")
        return None

    # 构建双模板地图
    t_build = time.time()
    build_dual_template_maps(map_ctx)
    build_time = time.time() - t_build

    # 双模板全局匹配
    t_match = time.time()
    best_pose, best_score = dual_template_global_match(
        scan_local, map_ctx,
        coarse_angle_step_deg=params['coarse_angle_step_deg'],
        fine_angle_step_deg=params['fine_angle_step_deg'],
        penalty_weight=params['penalty_weight'],
        scan_max_points=params['scan_max_points'],
        free_space_penalty_weight=params['free_space_penalty_weight'],
        dt_scale_ref_pixels=params['dt_scale_ref_pixels'],
        dt_scale_max=params['dt_scale_max'],
    )
    match_time = time.time() - t_match

    if best_pose is None or best_score <= -1e8:
        print(f"  [FAIL] 匹配失败 (score={best_score})")
        return None

    found_cx, found_cy, found_yaw = best_pose

    # 似然场评分
    build_likelihood(map_ctx)
    lf_score, _, _ = score_points(scan_local, found_cx, found_cy, found_yaw, map_ctx)
    wall_cov, coverage, _ = compute_wall_coverage(scan_local, found_cx, found_cy, found_yaw, map_ctx)

    # GT比较
    centroid_odom = (np.mean(frame_tfs[:, 0]), np.mean(frame_tfs[:, 1]))
    if gt_valid:
        cg, sg = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        gt_cx = cg * centroid_odom[0] - sg * centroid_odom[1] + tf_gt[0]
        gt_cy = sg * centroid_odom[0] + cg * centroid_odom[1] + tf_gt[1]
        gt_on_map = (0 <= gt_cx <= W*res and 0 <= gt_cy <= H*res)
        pos_err = math.sqrt((found_cx - gt_cx)**2 + (found_cy - gt_cy)**2)
        yaw_diff = found_yaw - tf_gt[2]
        yaw_err = abs(math.degrees(math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))))
    else:
        gt_cx, gt_cy, gt_on_map = 0.0, 0.0, False
        pos_err = -1.0
        yaw_err = -1.0

    print(f"  匹配: ({found_cx:.1f},{found_cy:.1f}) yaw={math.degrees(found_yaw):.1f}° "
          f"score={best_score:.2f}")
    if gt_valid:
        print(f"  GT:   ({gt_cx:.1f},{gt_cy:.1f}) yaw={math.degrees(tf_gt[2]):.1f}° | 在地图内={gt_on_map}")
        print(f"  误差: pos={pos_err:.2f}m yaw={yaw_err:.1f}° | "
              f"wall={100*wall_cov:.0f}% cov={100*coverage:.0f}% lf={lf_score:.3f}")
    else:
        print(f"  GT:   UTM坐标系, 无法比较")
        print(f"  质量: wall={100*wall_cov:.0f}% cov={100*coverage:.0f}% lf={lf_score:.3f}")

    result = {
        'file': label,
        'pos_err': pos_err, 'yaw_err': yaw_err,
        'score': best_score, 'wall_cov': wall_cov,
        'coverage': coverage, 'lf_score': lf_score,
        'time': time.time() - t0, 'gt_on_map': gt_on_map,
        'n_pts': n_pts, 'n_frames': n_frames,
        'match_x': found_cx, 'match_y': found_cy,
        'match_yaw_deg': math.degrees(found_yaw),
    }

    # 三张独立图片：地图 / 降噪点云 / 比对图
    map_path = os.path.join(output_dir, f"{label}_map.png")
    scan_path = os.path.join(output_dir, f"{label}_scan.png")
    match_path = os.path.join(output_dir, f"{label}_match.png")
    _visualize_3imgs(map_ctx, scan_local, found_cx, found_cy, found_yaw,
                     tf_gt, centroid_odom, label,
                     pos_err, yaw_err, wall_cov, best_score,
                     gt_cx, gt_cy, gt_on_map, gt_valid,
                     n_merged_frames=n_frames,
                     map_path=map_path, scan_path=scan_path, match_path=match_path)

    print(f"  耗时: build={build_time:.1f}s match={match_time:.1f}s total={result['time']:.1f}s")
    return result


def _visualize_3imgs(map_ctx, scan_local, cx, cy, yaw, tf_gt, centroid,
                     label, pos_err, yaw_err, wall_cov, score,
                     gt_cx, gt_cy, gt_on_map, gt_valid=True,
                     n_merged_frames=1,
                     map_path=None, scan_path=None, match_path=None):
    """生成三张独立图片：地图 / 降噪2D雷达点云 / 比对图。"""
    map_data = map_ctx.map_data
    H, W = map_data.shape
    res = map_ctx.map_info.resolution
    ox, oy = map_ctx.map_info.origin.position.x, map_ctx.map_info.origin.position.y
    extent = [ox, ox + W * res, oy, oy + H * res]

    map_bg = np.ones((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [0.95, 0.95, 0.95]   # 空地 - 浅灰
    map_bg[map_data == 100] = [0.15, 0.15, 0.15]  # 墙壁 - 深灰
    map_bg[map_data == -1] = [0.55, 0.55, 0.55]   # 未知 - 中灰

    c_f, s_f = math.cos(yaw), math.sin(yaw)
    aligned_x = c_f * scan_local[:,0] - s_f * scan_local[:,1] + cx
    aligned_y = s_f * scan_local[:,0] + c_f * scan_local[:,1] + cy

    if gt_valid and gt_on_map:
        cg, sg = math.cos(tf_gt[2]), math.sin(tf_gt[2])
        gt_x = cg * scan_local[:,0] - sg * scan_local[:,1] + gt_cx
        gt_y = sg * scan_local[:,0] + cg * scan_local[:,1] + gt_cy

    # 距离场
    binary_walls = ((map_data > 50) & (map_data <= 100)).astype(np.uint8) * 255
    dist_map_v = cv2.distanceTransform(255 - binary_walls, cv2.DIST_L2, 5)
    dist_clipped = np.clip(dist_map_v * res, 0, 3.0)

    zoom_r = max(30, (pos_err if pos_err > 0 else 30) * 3 + 15)
    step = max(1, len(aligned_x) // 500)
    scan_span_x = scan_local[:,0].max() - scan_local[:,0].min()
    scan_span_y = scan_local[:,1].max() - scan_local[:,1].min()

    # ===== 图1: 地图（带匹配点位标记） =====
    if map_path:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.plot(cx, cy, 'r+', ms=18, mew=4, label=f'Found ({cx:.1f},{cy:.1f})')
        if gt_valid and gt_on_map:
            ax.plot(gt_cx, gt_cy, 'yx', ms=14, mew=3, label=f'GT ({gt_cx:.1f},{gt_cy:.1f})')
        title = f"[{label}] 地图定位 | {n_merged_frames}帧合并"
        if pos_err >= 0:
            title += f" | err={pos_err:.1f}m/{yaw_err:.1f}°"
        title += f" | score={score:.0f} wall={100*wall_cov:.0f}%"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect('equal'); ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [VIS] 地图: {map_path}")

    # ===== 图2: 降噪2D雷达点云（局部坐标系，机器人原点居中） =====
    if scan_path:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(scan_local[:, 0], scan_local[:, 1], s=8, c='red',
                   edgecolors='darkred', linewidths=0.3, alpha=0.7, label=f'{len(scan_local)}pts')
        ax.plot(0, 0, 'b+', ms=20, mew=4, label='Robot Origin')
        # 标注跨度
        ax.set_title(f"[{label}] 降噪2D雷达点云 | {n_merged_frames}帧合并 "
                     f"| span={scan_span_x:.1f}m×{scan_span_y:.1f}m", fontsize=14)
        ax.set_xlabel("X local (m)"); ax.set_ylabel("Y local (m)")
        ax.set_aspect('equal')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        # 设置等距坐标轴范围
        max_span = max(scan_span_x, scan_span_y, 10) / 2 + 2
        ax.set_xlim(-max_span, max_span)
        ax.set_ylim(-max_span, max_span)
        plt.tight_layout()
        plt.savefig(scan_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [VIS] 点云: {scan_path}")

    # ===== 图3: 比对图（地图 + 雷达点云叠加 + 距离场） =====
    if match_path:
        fig, axes = plt.subplots(1, 2, figsize=(20, 9))

        # 左：全图叠加
        ax = axes[0]
        ax.imshow(map_bg, origin='lower', extent=extent)
        ax.scatter(aligned_x[::step], aligned_y[::step], s=0.5, c='lime', alpha=0.5, label='Matched Scan')
        ax.plot(cx, cy, 'r+', ms=16, mew=4, label=f'Found ({cx:.1f},{cy:.1f})')
        if gt_valid and gt_on_map:
            ax.scatter(gt_x[::step], gt_y[::step], s=0.3, c='cyan', alpha=0.3, label='GT Scan')
            ax.plot(gt_cx, gt_cy, 'yx', ms=12, mew=3, label=f'GT ({gt_cx:.1f},{gt_cy:.1f})')
        ax.set_title(f"[{label}] 全局比对 | yaw={math.degrees(yaw):.1f}° "
                     f"score={score:.0f} wall={100*wall_cov:.0f}%", fontsize=13)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_aspect('equal'); ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.1)

        # 右：局部放大 + 距离场
        ax = axes[1]
        ax.imshow(dist_clipped, origin='lower', cmap='hot', vmin=0, vmax=3.0, extent=extent)
        ax.scatter(aligned_x[::step], aligned_y[::step], s=1.5, c='cyan', alpha=0.6, edgecolors='none')
        ax.plot(cx, cy, 'r+', ms=14, mew=4)
        if gt_valid and gt_on_map:
            ax.plot(gt_cx, gt_cy, 'yx', ms=10, mew=3)
        ax.set_title(f"局部放大 + 距离场 | 范围={zoom_r:.0f}m (越暗越贴墙)", fontsize=13)
        ax.set_aspect('equal')
        ax.set_xlim(cx - zoom_r, cx + zoom_r); ax.set_ylim(cy - zoom_r, cy + zoom_r)
        ax.grid(True, alpha=0.15)

        plt.tight_layout()
        plt.savefig(match_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [VIS] 比对: {match_path}")


def print_summary(results):
    ok = [r for r in results if r is not None]
    if not ok: return
    print(f"\n{'='*120}")
    print(f"{'汇总结果':^120}")
    print(f"{'='*120}")
    hdr = (f"{'文件':<35} {'X(m)':>8} {'Y(m)':>8} {'Yaw':>7} {'score':>8} "
           f"{'wall':>6} {'cov':>6} {'lf':>7} {'pts':>5} {'帧':>5}")
    print(hdr)
    print("-" * 120)
    for r in ok:
        print(f"{r['file']:<35} {r['match_x']:>8.1f} {r['match_y']:>8.1f} "
              f"{r['match_yaw_deg']:>7.1f}° {r['score']:>8.1f} "
              f"{100*r['wall_cov']:>5.0f}% {100*r['coverage']:>5.0f}% "
              f"{r['lf_score']:>7.3f} {r['n_pts']:>5d} {r['n_frames']:>5d}")
    print("-" * 120)
    has_gt = any(r.get('pos_err', -1) >= 0 for r in ok)
    if has_gt:
        pts_with_gt = [r['pos_err'] for r in ok if r['pos_err'] >= 0]
        yaws_with_gt = [r['yaw_err'] for r in ok if r['yaw_err'] >= 0]
        print(f"{'统计':35} μpos={np.mean(pts_with_gt):.1f}m μyaw={np.mean(yaws_with_gt):.1f}° | {len(ok)} 个结果")
    else:
        print(f"{'统计':35} GT不可用(UTM坐标系) | {len(ok)} 个结果 | 仅输出匹配位置")
    print(f"{'='*120}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, default=None)
    parser.add_argument('--output', type=str, default=os.path.join(_DEFAULT_DATA_DIR, 'outdoor_results'))
    parser.add_argument('--data-dir', type=str, default=_DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if args.npz:
        npz_files = [os.path.abspath(args.npz)]
    else:
        pattern = os.path.join(args.data_dir, 'debug_match_data_*.npz')
        npz_files = sorted(glob.glob(pattern))
        if not npz_files:
            print(f"[ERROR] 未找到 .npz: {pattern}")
            sys.exit(1)

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"NPZ: {len(npz_files)}个 | 输出: {output_dir}")
    print(f"匹配参数: coarse={OUTDOOR_DT_PARAMS['coarse_angle_step_deg']}° "
          f"fine={OUTDOOR_DT_PARAMS['fine_angle_step_deg']}° "
          f"min_wall={OUTDOOR_DT_PARAMS['min_wall_coverage_ratio']} "
          f"penalty={OUTDOOR_DT_PARAMS['penalty_weight']}x "
          f"scan_max={OUTDOOR_DT_PARAMS['scan_max_points']}")

    results = []
    for npz_path in npz_files:
        try:
            r = analyze_one(npz_path, output_dir, OUTDOOR_DT_PARAMS)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {npz_path}: {e}")
            import traceback; traceback.print_exc()
            results.append(None)

    print_summary(results)


if __name__ == '__main__':
    main()
