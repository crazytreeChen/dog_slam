#!/usr/bin/env python3
"""
离线 Scan-to-Map 配准算法测试脚本（无需ROS）

用法:
  python3 test_match_algorithm.py                              # 使用默认路径
  python3 test_match_algorithm.py --data /tmp/scan_viz/debug_match_data.npz

功能:
  1. 加载 visualize_scan.py 导出的调试数据 (npz)
  2. 运行 Hu矩 + 似然场 两阶段配准算法
  3. 可视化每一步中间结果（扫描轮廓、地图轮廓、匹配分数热力图）
  4. 输出配准结果 vs TF真实值的对比
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
    print("错误: 需要 opencv-python: pip3 install opencv-python")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('TkAgg')  # 交互式窗口
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LogNorm
except ImportError:
    print("错误: 需要 matplotlib: pip3 install matplotlib")
    sys.exit(1)


# ============================================================
# 数据加载
# ============================================================

def load_debug_data(npz_path):
    """加载 visualize_scan.py 导出的调试数据"""
    data = np.load(npz_path, allow_pickle=True)

    scan_pts = data['scan_points_odom']
    map_data = data['map_data']
    resolution = float(data['map_resolution'])
    map_w = int(data['map_width'])
    map_h = int(data['map_height'])
    origin_x = float(data['map_origin_x'])
    origin_y = float(data['map_origin_y'])
    tf_odom_map = list(data['tf_odom_to_map'])

    info = {
        'resolution': resolution,
        'width': map_w,
        'height': map_h,
        'origin_x': origin_x,
        'origin_y': origin_y,
    }

    print(f'[数据] 加载成功:')
    print(f'  扫描点云: {len(scan_pts)} 点')
    print(f'  地图: {map_w}x{map_h} @ {resolution*1000:.1f}mm/pix')
    print(f'  地图原点: ({origin_x:.2f}, {origin_y:.2f})')
    print(f'  TF odom→map: ({tf_odom_map[0]:.2f}, {tf_odom_map[1]:.2f}, '
          f'{math.degrees(tf_odom_map[2]):.1f}°)')

    # 计算 TF 真实位姿（机器人在map中的位置）
    cx_odom = scan_pts[:, 0].mean()
    cy_odom = scan_pts[:, 1].mean()
    tf_x, tf_y, tf_yaw = tf_odom_map
    true_x = tf_x + cx_odom * math.cos(tf_yaw) - cy_odom * math.sin(tf_yaw)
    true_y = tf_y + cx_odom * math.sin(tf_yaw) + cy_odom * math.cos(tf_yaw)
    true_yaw = tf_yaw

    print(f'  扫描中心(odom): ({cx_odom:.2f}, {cy_odom:.2f})')
    print(f'  真实位姿(map):   ({true_x:.2f}, {true_y:.2f}, {math.degrees(true_yaw):.1f}°)')

    return scan_pts, map_data, info, tf_odom_map, (true_x, true_y, true_yaw)


# ============================================================
# 核心算法：与 visualize_scan.py 完全一致的实现
# ============================================================

def points_to_scan_contour(points, img_size=200, phys_size_m=20.0):
    """将扫描点渲染为封闭多边形，提取外轮廓用于Hu矩匹配"""
    meters_per_px = phys_size_m / img_size
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    half = img_size // 2

    angles = np.arctan2(points[:, 1], points[:, 0])
    sorted_idx = np.argsort(angles)

    pts_px = []
    for idx in sorted_idx:
        px = int(points[idx, 0] / meters_per_px + half)
        py = int(half - points[idx, 1] / meters_per_px)
        if 0 <= px < img_size and 0 <= py < img_size:
            pts_px.append([px, py])

    if len(pts_px) < 3:
        return None, img

    pts_arr = np.array(pts_px, dtype=np.int32)

    cv2.polylines(img, [pts_arr], isClosed=True, color=255, thickness=1)
    cv2.fillPoly(img, [pts_arr], 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, img

    return max(contours, key=cv2.contourArea), img


def extract_map_contour_at(x, y, map_data, info, phys_size_m=20.0):
    """在地图位置 (x,y) 周围提取房间墙面轮廓"""
    res = info['resolution']
    half_w_px = max(int(phys_size_m / 2 / res), 10)

    cx_px = int((x - info['origin_x']) / res)
    cy_px = int(info['height'] - 1 - (y - info['origin_y']) / res)

    r1 = max(0, cy_px - half_w_px)
    r2 = min(info['height'], cy_px + half_w_px)
    c1 = max(0, cx_px - half_w_px)
    c2 = min(info['width'], cx_px + half_w_px)

    if r2 - r1 < 10 or c2 - c1 < 10:
        return None, None

    roi = map_data[r1:r2, c1:c2]
    wall_binary = (roi == 100).astype(np.uint8) * 255

    if np.sum(wall_binary) < 50:
        return None, None

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_binary = cv2.dilate(wall_binary, kernel, iterations=2)
    wall_binary = cv2.erode(wall_binary, kernel, iterations=1)

    contours, _ = cv2.findContours(wall_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, wall_binary

    main_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(main_contour) < 20:
        return None, wall_binary

    return main_contour, wall_binary


def build_likelihood_field(map_data, info, max_dist_m=15.0):
    """构建似然场（距离变换）"""
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(obs, cv2.DIST_L2, 5)
    dist_m = dist_px * info['resolution']
    dist_m = np.clip(dist_m, 0, max_dist_m)
    return dist_m


def score_points_at_pose(centered_pts, x, y, yaw, likelihood_field, info):
    """似然场评分：扫描点在位姿(x,y,yaw)下对地图墙面的命中程度"""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    rx = cos_y * centered_pts[:, 0] - sin_y * centered_pts[:, 1] + x
    ry = sin_y * centered_pts[:, 0] + cos_y * centered_pts[:, 1] + y

    res = info['resolution']
    ox = info['origin_x']
    oy = info['origin_y']
    H, W = info['height'], info['width']

    valid = True
    scores = []

    for i in range(len(rx)):
        col = int((rx[i] - ox) / res)
        row = int(H - 1 - (ry[i] - oy) / res)
        if 0 <= col < W and 0 <= row < H:
            d = likelihood_field[row, col]
            if d < 0.3:  # 撞墙惩罚
                scores.append(-5.0)
            else:
                # 高斯似然：越近越好，sigma=0.5m
                scores.append(math.exp(-d * d / (2 * 0.25)))
        else:
            valid = False
            break

    if not valid or not scores:
        return -1e8

    return sum(scores) / len(scores)


# ============================================================
# 主测试流程
# ============================================================

def run_test(npz_path):
    """运行完整的离线测试"""
    print('=' * 60)
    print('离线 Scan-to-Map 配准算法测试')
    print('=' * 60)

    # ── 1. 加载数据 ──
    scan_pts, map_data, info, tf_odom_map, true_pose = load_debug_data(npz_path)
    true_x, true_y, true_yaw = true_pose

    # 居中归一化
    cx_odom = scan_pts[:, 0].mean()
    cy_odom = scan_pts[:, 1].mean()
    centered = scan_pts.copy()
    centered[:, 0] -= cx_odom
    centered[:, 1] -= cy_odom

    # 构建似然场
    lf = build_likelihood_field(map_data, info)

    # ── 2. 提取扫描轮廓并可视化 ──
    scan_extent = max(
        centered[:, 0].max() - centered[:, 0].min(),
        centered[:, 1].max() - centered[:, 1].min())
    window_m = max(scan_extent * 1.5, 15.0)

    scan_contour, scan_img = points_to_scan_contour(
        centered, img_size=200, phys_size_m=window_m)

    if scan_contour is None:
        print('[错误] 扫描轮廓提取失败')
        return

    scan_area = cv2.contourArea(scan_contour)
    print(f'\n[轮廓] 扫描轮廓: 窗口={window_m:.1f}m, 面积={scan_area:.0f}px²')

    # ── 3. 在真实位置提取地图轮廓（作为"金标准"对比）──
    true_contour, true_img = extract_map_contour_at(
        true_x, true_y, map_data, info, window_m)
    if true_contour is not None:
        true_dist = cv2.matchShapes(scan_contour, true_contour,
                                     cv2.CONTOURS_MATCH_I2, 0)
        true_area = cv2.contourArea(true_contour)
        print(f'[轮廓] 真实位置地图轮廓: 面积={true_area:.0f}px², '
              f'Hu距={true_dist:.4f}')
    else:
        print('[警告] 真实位置无有效地图轮廓！这就是问题所在')
        true_dist = float('inf')

    # ── 4. Phase 1: Hu矩粗搜 ──
    print(f'\n[Phase1] Hu矩粗搜...')
    coarse_step = 1.5
    n_keep = 5

    res = info['resolution']
    map_w_m = info['width'] * res
    map_h_m = info['height'] * res
    ox, oy = info['origin_x'], info['origin_y']

    xs = np.arange(ox + 1.5, ox + map_w_m - 1.5, coarse_step)
    ys = np.arange(oy + 1.5, oy + map_h_m - 1.5, coarse_step)
    n_pos = len(xs) * len(ys)

    # 记录所有位置的分数（用于绘制热力图）
    score_grid = np.full((len(ys), len(xs)), np.nan)

    top_candidates = []  # [(score, x, y), ...]

    t0 = time.time()
    count = 0
    for j, ay_val in enumerate(ys):
        for i, ax_val in enumerate(xs):
            count += 1
            map_c, _ = extract_map_contour_at(ax_val, ay_val, map_data, info, window_m)
            if map_c is None:
                continue

            dist = cv2.matchShapes(scan_contour, map_c, cv2.CONTOURS_MATCH_I2, 0)
            map_a = cv2.contourArea(map_c)
            area_ratio = min(map_a, scan_area) / max(map_a, scan_area, 1.0)
            penalty = (1.0 - area_ratio) * 2.0
            score = -(dist + penalty)

            score_grid[j, i] = score

            if len(top_candidates) < n_keep:
                top_candidates.append((score, ax_val, ay_val))
                top_candidates.sort(key=lambda x: x[0])
            elif score > top_candidates[0][0]:
                top_candidates[0] = (score, ax_val, ay_val)
                top_candidates.sort(key=lambda x: x[0])

        if count % 200 == 0:
            print(f'  {count}/{n_pos} ({time.time()-t0:.1f}s)')

    elapsed_hu = time.time() - t0
    top_candidates.sort(key=lambda x: x[0], reverse=True)

    print(f'\n[Hu结果] 耗时 {elapsed_hu:.1f}s, Top-K:')
    for rank, (s, x, y) in enumerate(top_candidates):
        dx = x - true_x
        dy = y - true_y
        err = math.sqrt(dx*dx + dy*dy)
        marker = ' <-- TRUE' if err < 2.0 else ''
        print(f'  #{rank}: score={s:.3f}, pos=({x:.2f},{y:.2f}), '
              f'距真实值={err:.2f}m{marker}')

    # ── 5. Phase 2: 似然场精搜 ──
    print(f'\n[Phase2] 似然场精搜 (在Top-{n_keep}候选周围)...')
    angle_step_deg = 10.0
    n_angles = int(360.0 / angle_step_deg)
    pos_radius = 2.0
    n_fine = 7

    best_score = -1e9
    best_pose = None
    all_results = []

    t1 = time.time()
    for rank, (hu_s, hx, hy) in enumerate(top_candidates):
        xs_fine = np.linspace(hx - pos_radius, hx + pos_radius, n_fine)
        ys_fine = np.linspace(hy - pos_radius, hy + pos_radius, n_fine)
        for ax_val in xs_fine:
            for ay_val in ys_fine:
                for adeg in range(n_angles):
                    ayaw = math.radians(adeg * angle_step_deg)
                    s = score_points_at_pose(
                        centered,
                        ax_val + cx_odom * math.cos(ayaw) - cy_odom * math.sin(ayaw),
                        ay_val + cx_odom * math.sin(ayaw) + cy_odom * math.cos(ayaw),
                        ayaw, lf, info)
                    if s > best_score:
                        best_score = s
                        best_pose = (ax_val, ay_val, ayaw)
                    all_results.append((s, ax_val, ayval, ayaw))

    elapsed_fine = time.time() - t1
    all_results.sort(key=lambda x: x[0], reverse=True)

    fx, fy, fyaw = best_pose
    full_fx = fx + cx_odom * math.cos(fyaw) - cy_odom * math.sin(fyaw)
    full_fy = fy + cx_odom * math.sin(fyaw) + cy_odom * math.cos(fyaw)

    err_x = full_fx - true_x
    err_y = full_fy - true_y
    err_d = math.sqrt(err_x**2 + err_y**2)
    err_yaw = abs(math.degrees(fyaw) - math.degrees(true_yaw))
    err_yaw = min(err_yaw, 360 - err_yaw)

    print(f'\n[最终结果] 最佳=({full_fx:.2f}, {full_fy:.2f}, {math.degrees(fyaw):.1f}°)')
    print(f'  真实值 =({true_x:.2f}, {true_y:.2f}, {math.degrees(true_yaw):.1f}°)')
    print(f'  偏差 Δ={err_d:.2f}m (dx={err_x:.2f}m, dy={err_y:.2f}m, dyaw={err_yaw:.1f}°)')
    print(f'  score={best_score:.2f}')
    print(f'  耗时: Hu={elapsed_hu:.1f}s + 似然={elapsed_fine:.1f}s')

    # ── 6. 综合可视化 ──
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) 扫描轮廓
    axes[0, 0].imshow(scan_img, cmap='gray')
    axes[0, 0].set_title(f'Scan Contour\n(window={window_m:.1f}m, area={scan_area:.0f}px²)')
    axes[0, 0].axis('off')

    # (b) 真实位置的地图轮廓
    if true_img is not None:
        axes[0, 1].imshow(true_img, cmap='gray')
        axes[0, 1].set_title(f'Map Contour at TRUE pos\n(Hu_dist={true_dist:.4f})')
    else:
        axes[0, 1].text(0.5, 0.5, 'No contour\nat true position!',
                        ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Map Contour at TRUE (EMPTY!)')
    axes[0, 1].axis('off')

    # (c) Hu矩分数热力图
    im = axes[0, 2].imshow(score_grid, extent=[xs[0], xs[-1], ys[-1], ys[0]],
                            aspect='auto', cmap='RdYlGn', origin='upper')
    axes[0, 2].plot(true_x, true_y, 'b*', markersize=20, label='TRUE position')
    for rank, (_, tx, ty) in enumerate(top_candidates[:3]):
        axes[0, 2].plot(tx, ty, 'rx', markersize=10, label=f'Top-{rank}' if rank < 3 else '')
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].set_title(f'Hu Score Heatmap (coarse step={coarse_step}m)')
    plt.colorbar(im, ax=axes[0, 2])

    # (d) 地图+TF真实位姿+匹配结果
    map_disp = np.zeros((info['height'], info['width'], 3), dtype=np.float32)
    map_disp[map_data == 0] = [1, 1, 1]
    map_disp[map_data == 100] = [0, 0, 0]
    map_disp[map_data == -1] = [0.6, 0.6, 0.6]
    axes[1, 0].imshow(map_disp, extent=[ox, ox + info['width'] * res,
                                         oy, oy + info['height'] * res],
                       origin='lower')
    axes[1, 0].plot(true_x, true_y, 'b+', markersize=15, mew=3,
                    label=f'TF ({true_x:.1f},{true_y:.1f})')
    axes[1, 0].plot(full_fx, full_fy, 'rx', markersize=15, mew=3,
                    label=f'Match ({full_fx:.1f},{full_fy:.1f})')
    # 画扫描点（用TF变换到map系）
    cos_t, sin_t = math.cos(true_yaw), math.sin(true_yaw)
    sx = cos_t * scan_pts[:, 0] - sin_t * scan_pts[:, 1] + tf_odom_map[0]
    sy = sin_t * scan_pts[:, 0] + cos_t * scan_pts[:, 1] + tf_odom_map[1]
    axes[1, 0].scatter(sx, sy, c='g', s=3, alpha=0.5, label='Scan(TF)')
    # 匹配结果的扫描点
    cos_m, sin_m = math.cos(fyaw), math.sin(fyaw)
    mx = cos_m * scan_pts[:, 0] - sin_m * scan_pts[:, 1] + tf_odom_map[0]
    my = sin_m * scan_pts[:, 0] + cos_m * scan_pts[:, 1] + tf_odom_map[1]
    axes[1, 0].scatter(mx, my, c='r', s=3, alpha=0.5, label='Scan(Match)')
    axes[1, 0].legend(fontsize=8, loc='best')
    axes[1, 0].set_title(f'Result: Δ={err_d:.2m}m, yaw_err={err_yaw:.1f}°')
    axes[1, 0].set_xlabel('X (m)')
    axes[1, 0].set_ylabel('Y (m)')

    # (e) Top-5精搜结果柱状图
    top5 = all_results[:5]
    labels = [f'{i}\n({s[1]:.1f},{s[2]:.1f}\n{math.degrees(s[3]):.0f}°)' for i, s in enumerate(top5)]
    scores_plot = [s[0] for s in top5]
    colors = ['green' if i == 0 else 'steelblue' for i in range(len(top5))]
    bars = axes[1, 1].bar(range(len(top5)), scores_plot, color=colors)
    axes[1, 1].set_xticks(range(len(top5)))
    axes[1, 1].set_xticklabels(labels, fontsize=8)
    axes[1, 1].set_ylabel('Likelihood Score')
    axes[1, 1].set_title('Top-5 Fine Search Results')
    for bar, sc in zip(bars, scores_plot):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{sc:.2f}', ha='center', va='bottom', fontsize=9)

    # (f) 诊断信息文本
    diag_text = f"""=== 诊断报告 ===

数据:
  扫描点数: {len(scan_pts)}
  地图尺寸: {info['width']}x{info['height']} @ {info['resolution']*1000:.1f}mm
  扫描范围: {scan_extent:.1f}m × {scan_extent:.1f}m
  窗口大小: {window_m:.1f}m

Hu矩分析:
  扫描轮廓面积: {scan_area:.0f} px²
  真实位置Hu距离: {true_dist:.4f}
  {'✓ 真实位置有有效轮廓' if true_contour is not None else '✗ 真实位置无轮廓 ← 主要问题'}

Phase1 (Hu矩):
  搜索网格: {len(xs)}×{len(ys)}={n_pos}
  Top-1 score: {top_candidates[0][0]:.3f}
  Top-1 位置: ({top_candidates[0][1]:.2f}, {top_candidates[0][2]:.2f})
  所有Top-K是否相同score? {len(set(s for s,_,_ in top_candidates)) == 1}

Phase2 (似然场):
  精搜范围: {pos_radius}m × {n_fine}² × {n_angles}角
  最终偏差: {err_d:.2f}m (dx={err_x:.2f}, dy={err_y:.2f}, dyaw={err_yaw:.1f}°)
"""
    axes[1, 2].text(0.05, 0.95, diag_text, transform=axes[1, 2].transAxes,
                    fontfamily='monospace', fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Diagnosis')

    plt.tight_layout()

    output_dir = os.path.dirname(npz_path)
    out_png = os.path.join(output_dir, 'test_match_result.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'\n[图片] 结果已保存: {out_png}')

    plt.show()


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='离线Scan-to-Map配准算法测试')
    parser.add_argument('--data', type=str,
                        default='/tmp/scan_viz/debug_match_data.npz',
                        help='debug_match_data.npz 文件路径')
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f'[错误] 数据文件不存在: {args.data}')
        print('请先运行 visualize_scan.py 采集数据')
        sys.exit(1)

    run_test(args.data)
