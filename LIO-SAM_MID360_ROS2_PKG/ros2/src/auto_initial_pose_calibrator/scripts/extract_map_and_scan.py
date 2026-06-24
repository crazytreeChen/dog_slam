#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_map_and_scan.py
从 npz 中提取地图和雷达扫描图，分别保存为 2 张图片。

用法:
  python extract_map_and_scan.py
  python extract_map_and_scan.py --data ../scan_viz/debug_match_data_1.npz --frames 20
"""

import os, sys, math, argparse
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

def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from calib_lib.scan_utils import frf_filter_frame
from calib_lib.temporal import temporal_consistency_filter


def load_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    map_data = d['map_data']
    info = {'resolution': float(d['map_resolution']),
            'width': int(d['map_width']), 'height': int(d['map_height']),
            'origin_x': float(d['map_origin_x']), 'origin_y': float(d['map_origin_y'])}
    frame_tfs = d['frame_tfs']
    angle_min = float(d.get('frame_angle_min', -math.pi))
    angle_inc = float(d.get('frame_angle_increment', 2 * math.pi / 360))
    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in d:
        frame_ranges.append(np.array(d[f'frame_ranges_{i}'], dtype=np.float64))
        i += 1
    return map_data, info, frame_tfs, frame_ranges, angle_min, angle_inc


def merge_frames(frame_ranges, frame_tfs, angle_min, angle_inc,
                 temporal_min_frames=3, temporal_radius=0.15):
    all_pts, all_fids = [], []
    for i, ranges in enumerate(frame_ranges):
        tx, ty, yaw = frame_tfs[i]
        c, s = math.cos(yaw), math.sin(yaw)
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        keep = frf_filter_frame(ranges, angle_min, angle_inc)
        valid = keep & (ranges > 0.15) & (ranges < 50.0)
        if np.sum(valid) < 10:
            continue
        lx = ranges[valid] * np.cos(angles[valid])
        ly = ranges[valid] * np.sin(angles[valid])
        xy = np.column_stack([c * lx - s * ly + tx, s * lx + c * ly + ty])
        all_pts.append(xy)
        all_fids.append(np.full(len(xy), i))
    if not all_pts:
        return np.empty((0, 2))
    merged = np.vstack(all_pts)
    fids = np.concatenate(all_fids)
    if len(merged) > temporal_min_frames:
        mask = temporal_consistency_filter(merged, fids, temporal_min_frames, temporal_radius)
        merged = merged[mask]
    return merged


def plot_map(map_data, info, output_path):
    H, W = info['height'], info['width']
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    extent = [ox, ox + W * res, oy, oy + H * res]

    img = np.ones((H, W, 3), dtype=np.float32)
    img[map_data == 100] = [0.1, 0.1, 0.1]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, origin='lower', extent=extent)
    ax.set_title(f'Map ({W}x{H} @ {res*1000:.0f}mm/pix)', fontsize=14)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Map] {output_path}')


def plot_scan(scan_pts, output_path):
    cx, cy = scan_pts.mean(axis=0)
    span = max(np.ptp(scan_pts[:, 0]), np.ptp(scan_pts[:, 1])) * 0.6
    span = max(span, 5.0)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(scan_pts[:, 0], scan_pts[:, 1], s=2, c='cyan', alpha=0.7, edgecolors='none')
    ax.plot(cx, cy, 'r+', markersize=14, mew=2)
    ax.set_title(f'Scan ({len(scan_pts)} pts, FRF+temporal, 20 frames)', fontsize=14)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[Scan] {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(__file__),
                            '..', 'scan_viz', 'debug_match_data_1.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--frames', type=int, default=20)
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    output_dir = args.output or os.path.join(os.path.dirname(__file__), '..', 'scan_viz', 'extract')
    os.makedirs(output_dir, exist_ok=True)

    print("[1/3] 加载数据...")
    map_data, info, frame_tfs, frame_ranges, angle_min, angle_inc = load_npz(npz_path)
    if args.frames > 0:
        frame_ranges = frame_ranges[:args.frames]
        frame_tfs = frame_tfs[:args.frames]
    print(f"  {len(frame_ranges)} 帧, 地图 {info['width']}x{info['height']}")

    print("[2/3] 生成地图图片...")
    plot_map(map_data, info, os.path.join(output_dir, 'map.png'))

    print("[3/3] 合并扫描并生成图片...")
    scan_pts = merge_frames(frame_ranges, frame_tfs, angle_min, angle_inc)
    print(f"  合并后: {len(scan_pts)} 点")
    plot_scan(scan_pts, os.path.join(output_dir, 'scan.png'))

    print(f"\n[完成] 图片保存到: {output_dir}")


if __name__ == '__main__':
    main()
