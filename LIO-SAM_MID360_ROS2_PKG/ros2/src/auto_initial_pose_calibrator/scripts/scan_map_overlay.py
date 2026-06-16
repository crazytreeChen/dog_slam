#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_map_overlay.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
离线扫描-地图叠加可视化

核心功能:
  1. 读取 debug_match_data.npz
  2. 合并所有帧激光点云 (odom系)
  3. 全局似然场搜索找到 odom->map 变换
  4. 将扫描点云投影到地图上，用红色标出
  5. 输出叠加可视化图片

用法:
  python scan_map_overlay.py
  python scan_map_overlay.py --data path/to/debug_match_data.npz
  python scan_map_overlay.py --output result.png
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
    print("[ERROR] Need opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("[ERROR] Need matplotlib: pip install matplotlib")
    sys.exit(1)

# 中文字体
def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()


# ============================================================
# 1. 数据加载
# ============================================================
def load_data(npz_path):
    """加载NPZ数据文件"""
    if not os.path.exists(npz_path):
        print(f"[ERROR] 文件不存在: {npz_path}")
        sys.exit(1)

    d = np.load(npz_path, allow_pickle=True)

    # 地图数据
    map_data = d['map_data']
    info = {
        'resolution': float(d['map_resolution']),
        'width': int(d['map_width']),
        'height': int(d['map_height']),
        'origin_x': float(d['map_origin_x']),
        'origin_y': float(d['map_origin_y']),
    }

    # GT参考 (仅用于对比，不参与搜索)
    tf_gt = d['tf_odom_to_map']

    # 每帧odom位姿
    frame_tfs = d['frame_tfs']

    # 激光参数
    angle_min = float(d['frame_angle_min'])
    angle_inc = float(d['frame_angle_increment'])

    # 读取所有帧ranges
    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in d:
        frame_ranges.append(np.array(d[f'frame_ranges_{i}'], dtype=np.float64))
        i += 1

    print(f"数据加载完成:")
    print(f"  帧数: {len(frame_ranges)}")
    print(f"  地图: {info['width']}x{info['height']} @ {info['resolution']:.3f}m/pixel")
    print(f"  地图原点: ({info['origin_x']:.2f}, {info['origin_y']:.2f})")
    print(f"  GT参考 (仅供对比): ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f}°)")

    return map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc


# ============================================================
# 2. 合并激光帧
# ============================================================
def merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc):
    """
    将所有帧激光数据合并到odom坐标系

    返回:
      merged_pts: (N, 2) 合并后的点云
      centroid: (cx, cy) 点云质心
    """
    all_pts = []
    total_raw = 0
    total_kept = 0

    for fi, (ranges, tf) in enumerate(zip(frame_ranges, frame_tfs)):
        # 基本有效性过滤
        valid = (ranges > 0.15) & (ranges < 50.0)
        total_raw += int(np.sum(valid))

        if not np.any(valid):
            continue

        # 极坐标转笛卡尔 (激光系)
        angles = angle_min + np.arange(len(ranges)) * angle_inc
        lx = ranges[valid] * np.cos(angles[valid])
        ly = ranges[valid] * np.sin(angles[valid])

        # 变换到odom系
        tx, ty, yaw = tf
        c, s = math.cos(yaw), math.sin(yaw)
        pts_odom = np.column_stack([
            c * lx - s * ly + tx,
            s * lx + c * ly + ty
        ])
        all_pts.append(pts_odom)
        total_kept += len(pts_odom)

    if not all_pts:
        raise ValueError("没有有效的激光点!")

    merged_pts = np.vstack(all_pts)
    cx, cy = merged_pts[:, 0].mean(), merged_pts[:, 1].mean()

    print(f"合并完成:")
    print(f"  原始点数: {total_raw}")
    print(f"  有效点数: {total_kept}")
    print(f"  质心位置: ({cx:.2f}, {cy:.2f})")

    return merged_pts, (cx, cy)


# ============================================================
# 3. 构建似然场
# ============================================================
def build_likelihood_field(map_data, info, max_dist=3.0):
    """
    构建似然场 (距离变换)

    障碍物距离=0, 距离越远值越大, 未知区域=max_dist
    """
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(1 - obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_px = max_dist / info['resolution']
    lf = np.clip(dist_px, 0, max_px).astype(np.float32) * info['resolution']
    # 未知区域设为最大距离 (惩罚)
    lf[map_data == -1] = max_dist
    return lf


# ============================================================
# 4. 评分函数
# ============================================================
def score_at_pose(points_c, cx, cy, yaw, lf, info):
    """
    在指定位姿评分

    参数:
      points_c: 点云 (已中心化，即减去质心)
      cx, cy: 评估位置 (map系)
      yaw: 评估航向角
      lf: 似然场
      info: 地图信息

    返回:
      score: 评分 (越高越好)
      hit_rate: 命中率
      n_valid: 有效点数
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    # 旋转+平移
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y * points_c[:, 0] - s_y * points_c[:, 1] + cx
    my = s_y * points_c[:, 0] + c_y * points_c[:, 1] + cy

    # 转像素坐标
    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)

    # 有效性检查
    valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    nv = int(np.sum(valid))

    if nv < len(points_c) * 0.10:
        return -1e9, 0, 0

    # 似然场评分
    dists = lf[ri[valid], ci[valid]]
    n_hit = int(np.sum(dists < 0.15))
    hit_rate = n_hit / nv
    lf_score = float(np.mean(np.exp(-dists**2 / 0.045)))

    return lf_score + hit_rate * 0.5, hit_rate, nv


# ============================================================
# 5. 全局搜索
# ============================================================
def global_search(points_c, lf, map_data, info,
                  coarse_step=2.0, angle_step_deg=15.0, top_k=10):
    """
    全局似然场搜索

    参数:
      points_c: 中心化的点云
      lf: 似然场
      map_data: 地图数据
      info: 地图信息
      coarse_step: 粗搜步长 (米)
      angle_step_deg: 角度步长 (度)
      top_k: 返回候选数

    返回:
      candidates: [(score, x, y, yaw), ...]
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    mw_m = info['width'] * res
    mh_m = info['height'] * res

    # 降采样加速
    ds = max(1, len(points_c) // 1000)
    if ds > 1:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(points_c), size=min(len(points_c) // ds, 2000), replace=False)
        pts_ds = points_c[indices]
    else:
        pts_ds = points_c

    # 搜索网格
    xs = np.arange(ox + 2, ox + mw_m - 2, coarse_step)
    ys = np.arange(oy + 2, oy + mh_m - 2, coarse_step)
    n_angles = int(360.0 / angle_step_deg)

    total_evals = len(xs) * len(ys) * n_angles
    print(f"\n全局搜索:")
    print(f"  网格: {len(xs)}x{len(ys)} = {len(xs) * len(ys)} 位置 x {n_angles} 角度")
    print(f"  总评估次数: {total_evals}")

    t0 = time.time()
    all_scores = []
    count = 0

    for ax in xs:
        for ay in ys:
            count += 1
            for adeg in range(n_angles):
                ayaw = math.radians(adeg * angle_step_deg)
                sc, _, _ = score_at_pose(pts_ds, ax, ay, ayaw, lf, info)
                if sc > -1e8:
                    all_scores.append((sc, ax, ay, ayaw))

            # 进度输出
            if count % 500 == 0:
                elapsed = time.time() - t0
                print(f"  进度: {count}/{len(xs) * len(ys)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  粗搜完成: {elapsed:.1f}s, {len(all_scores)} 有效评分")

    if not all_scores:
        return []

    # 排序 + NMS去冗余
    all_scores.sort(key=lambda x: x[0], reverse=True)
    nms = []
    for sc, ax, ay, ayaw in all_scores:
        is_dup = any(
            math.sqrt((ax - cx) ** 2 + (ay - cy) ** 2) < 1.5
            and abs(math.atan2(math.sin(ayaw - cyaw), math.cos(ayaw - cyaw))) < math.radians(30)
            for _, cx, cy, cyaw in nms
        )
        if not is_dup:
            nms.append((sc, ax, ay, ayaw))
            if len(nms) >= top_k:
                break

    print(f"  Top-{len(nms)} 候选 (NMS后):")
    for i, (sc, ax, ay, ayaw) in enumerate(nms):
        print(f"    #{i}: ({ax:.1f}, {ay:.1f}, {math.degrees(ayaw):.0f}°) score={sc:.3f}")

    return nms


# ============================================================
# 6. 精细搜索
# ============================================================
def fine_search(candidates, points_c, lf, info,
                pos_radius=1.5, pos_step=0.3, angle_range_deg=20, angle_step_deg=3):
    """
    在粗搜候选周围做精细搜索

    返回:
      best: (score, x, y, yaw)
    """
    print(f"\n精细搜索:")
    print(f"  候选数: {len(candidates)}")
    print(f"  搜索半径: ±{pos_radius}m, 步长: {pos_step}m")
    print(f"  角度范围: ±{angle_range_deg}°, 步长: {angle_step_deg}°")

    ds = max(1, len(points_c) // 1000)
    pts_ds = points_c[::ds]

    best_score = -1e9
    best_pose = None

    for rank, (_, hx, hy, hyaw) in enumerate(candidates[:3]):  # 只精搜Top-3
        for dx in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
            for dy in np.arange(-pos_radius, pos_radius + 1e-5, pos_step):
                for da_deg in range(-angle_range_deg, angle_range_deg + 1, angle_step_deg):
                    ax, ay = hx + dx, hy + dy
                    ayaw = hyaw + math.radians(da_deg)
                    sc, _, _ = score_at_pose(pts_ds, ax, ay, ayaw, lf, info)
                    if sc > best_score:
                        best_score = sc
                        best_pose = (ax, ay, ayaw)

    if best_pose is None:
        best_pose = candidates[0][1:]

    print(f"  最佳结果: ({best_pose[0]:.2f}, {best_pose[1]:.2f}, {math.degrees(best_pose[2]):.1f}°) score={best_score:.3f}")

    return best_score, best_pose[0], best_pose[1], best_pose[2]


# ============================================================
# 7. ICP精调
# ============================================================
def icp_refine(points_odom, cx, cy, yaw, map_data, info, max_iter=30):
    """
    ICP精调: 将扫描点与地图墙壁对齐

    返回:
      refined_cx, refined_cy, refined_yaw
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("  [ICP] scipy不可用，跳过ICP精调")
        return cx, cy, yaw

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    # 提取地图墙壁点
    wall_ys, wall_xs = np.where(map_data == 100)
    map_walls = np.column_stack([
        wall_xs * res + ox,
        (H - 1 - wall_ys) * res + oy
    ])
    # 降采样
    step = max(1, len(map_walls) // 5000)
    map_walls_ds = map_walls[::step]

    # 初始变换
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    R_init = np.array([[c_y, -s_y], [s_y, c_y]])
    t_init = np.array([cx, cy])

    # 扫描点变换到map系
    src = (R_init @ points_odom.T).T + t_init
    target_tree = cKDTree(map_walls_ds)

    R_total = np.eye(2)
    t_total = np.zeros(2)

    for it in range(max_iter):
        # 最近邻
        dists, idx = target_tree.query(src)
        med = np.median(dists)
        mask = dists < max(0.15, med * 2.5)

        if np.sum(mask) < 20:
            break

        s_pts = src[mask]
        m_pts = target_tree.data[idx[mask]]

        # SVD求解
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

        if np.linalg.norm(t) < 1e-5:
            break

    # 提取最终变换
    final_yaw = yaw + math.atan2(R_total[1, 0], R_total[0, 0])
    final_cx = t_total[0] + R_total[0, 0] * cx - R_total[0, 1] * cy
    final_cy = t_total[1] + R_total[1, 0] * cx - R_total[1, 1] * cy

    print(f"  ICP精调: ({cx:.2f},{cy:.2f},{math.degrees(yaw):.1f}°) → ({final_cx:.2f},{final_cy:.2f},{math.degrees(final_yaw):.1f}°)")

    return final_cx, final_cy, final_yaw


# ============================================================
# 8. 可视化
# ============================================================
def create_overlay_visualization(map_data, info, merged_pts, centroid,
                                 search_result, tf_gt, output_path):
    """
    创建叠加可视化图

    参数:
      map_data: 地图数据
      info: 地图信息
      merged_pts: 合并后的点云 (odom系)
      centroid: 点云质心 (cx, cy)
      search_result: (score, x, y, yaw) 搜索结果
      tf_gt: GT参考变换
      output_path: 输出路径
    """
    print(f"\n生成可视化...")

    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W * res, oy, oy + H * res]

    # 创建地图背景 (RGB)
    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1, 1, 1]        # 自由空间: 白色
    map_bg[map_data == 100] = [0.2, 0.2, 0.2]  # 障碍物: 深灰
    map_bg[map_data == -1] = [0.7, 0.7, 0.7]  # 未知: 浅灰

    # 解析搜索结果
    sc, est_x, est_y, est_yaw = search_result
    cx, cy = centroid

    # 计算odom->map变换
    # 我们需要的是: map_pt = R(yaw) * (odom_pt - centroid) + (est_x, est_y)
    # 即: t = (est_x, est_y) - R(yaw) * centroid
    c_y, s_y = math.cos(est_yaw), math.sin(est_yaw)
    t_x = est_x - (c_y * cx - s_y * cy)
    t_y = est_y - (s_y * cx + c_y * cy)

    # 变换点云到map系
    pts_map = np.column_stack([
        c_y * merged_pts[:, 0] - s_y * merged_pts[:, 1] + t_x,
        s_y * merged_pts[:, 0] + c_y * merged_pts[:, 1] + t_y
    ])

    # GT变换点云 (用于对比)
    gt_c, gt_s = math.cos(tf_gt[2]), math.sin(tf_gt[2])
    pts_gt = np.column_stack([
        gt_c * merged_pts[:, 0] - gt_s * merged_pts[:, 1] + tf_gt[0],
        gt_s * merged_pts[:, 0] + gt_c * merged_pts[:, 1] + tf_gt[1]
    ])

    # ========== 创建图形 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # (a) 原始地图 + odom系点云
    ax = axes[0, 0]
    ax.imshow(map_bg, origin='lower', extent=extent, aspect='equal')
    ax.scatter(merged_pts[::5, 0], merged_pts[::5, 1],
               s=0.5, c='blue', alpha=0.3, label='扫描点云 (odom系)')
    ax.plot(cx, cy, 'r+', markersize=15, mew=3, label=f'质心 ({cx:.1f},{cy:.1f})')
    ax.set_title('(a) 原始地图 + odom系点云\n(未对齐)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # (b) 搜索结果叠加
    ax = axes[0, 1]
    ax.imshow(map_bg, origin='lower', extent=extent, aspect='equal')
    # 红色扫描点
    ax.scatter(pts_map[::3, 0], pts_map[::3, 1],
               s=1, c='red', alpha=0.5, label='扫描点云 (搜索结果)')
    ax.plot(est_x, est_y, 'g+', markersize=15, mew=3)
    ax.arrow(est_x, est_y, 2.0 * c_y, 2.0 * s_y,
             head_width=0.3, head_length=0.2, fc='green', ec='darkgreen', lw=2)
    ax.set_title(f'(b) 搜索结果叠加\n'
                 f'位姿: ({est_x:.1f},{est_y:.1f},{math.degrees(est_yaw):.0f}°) score={sc:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # (c) GT叠加 (对比)
    ax = axes[1, 0]
    ax.imshow(map_bg, origin='lower', extent=extent, aspect='equal')
    ax.scatter(pts_gt[::3, 0], pts_gt[::3, 1],
               s=1, c='blue', alpha=0.5, label='扫描点云 (GT)')
    ax.plot(tf_gt[0], tf_gt[1], 'r+', markersize=15, mew=3)
    ax.arrow(tf_gt[0], tf_gt[1], 2.0 * gt_c, 2.0 * gt_s,
             head_width=0.3, head_length=0.2, fc='red', ec='darkred', lw=2)
    ax.set_title(f'(c) GT参考叠加\n'
                 f'位姿: ({tf_gt[0]:.1f},{tf_gt[1]:.1f},{math.degrees(tf_gt[2]):.0f}°)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    # (d) 搜索 vs GT 对比
    ax = axes[1, 1]
    ax.imshow(map_bg, origin='lower', extent=extent, aspect='equal')
    ax.scatter(pts_map[::3, 0], pts_map[::3, 1],
               s=0.8, c='red', alpha=0.4, label='搜索结果')
    ax.scatter(pts_gt[::3, 0], pts_gt[::3, 1],
               s=0.8, c='blue', alpha=0.4, label='GT参考')

    # 计算误差
    err_dist = math.sqrt((est_x - tf_gt[0]) ** 2 + (est_y - tf_gt[1]) ** 2)
    err_yaw = abs(math.atan2(math.sin(est_yaw - tf_gt[2]), math.cos(est_yaw - tf_gt[2])))

    ax.set_title(f'(d) 搜索 vs GT 对比\n'
                 f'位置误差: {err_dist:.2f}m, 角度误差: {math.degrees(err_yaw):.1f}°')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"可视化已保存: {output_path}")


# ============================================================
# 9. 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='扫描-地图叠加可视化')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'scan_viz', 'debug_match_data.npz'),
                        help='NPZ数据文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片路径')
    parser.add_argument('--coarse-step', type=float, default=2.0,
                        help='粗搜步长 (米)')
    parser.add_argument('--skip-icp', action='store_true',
                        help='跳过ICP精调')
    args = parser.parse_args()

    # 输出路径
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.dirname(os.path.abspath(args.data))
        output_path = os.path.join(output_dir, 'scan_map_overlay.png')

    # ========== 1. 加载数据 ==========
    print("=" * 60)
    print("步骤 1: 加载数据")
    print("=" * 60)
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(args.data)

    # ========== 2. 合并激光帧 ==========
    print("\n" + "=" * 60)
    print("步骤 2: 合并激光帧")
    print("=" * 60)
    merged_pts, centroid = merge_scans(frame_ranges, frame_tfs, angle_min, angle_inc)

    # ========== 3. 构建似然场 ==========
    print("\n" + "=" * 60)
    print("步骤 3: 构建似然场")
    print("=" * 60)
    lf = build_likelihood_field(map_data, info)
    print(f"似然场构建完成: shape={lf.shape}")

    # ========== 4. 中心化点云 ==========
    cx, cy = centroid
    pts_centered = merged_pts.copy()
    pts_centered[:, 0] -= cx
    pts_centered[:, 1] -= cy

    # ========== 5. 全局搜索 ==========
    print("\n" + "=" * 60)
    print("步骤 4: 全局搜索")
    print("=" * 60)
    candidates = global_search(pts_centered, lf, map_data, info,
                               coarse_step=args.coarse_step)

    if not candidates:
        print("[ERROR] 未找到有效候选!")
        sys.exit(1)

    # ========== 6. 精细搜索 ==========
    print("\n" + "=" * 60)
    print("步骤 5: 精细搜索")
    print("=" * 60)
    best_score, best_x, best_y, best_yaw = fine_search(candidates, pts_centered, lf, info)

    # ========== 7. ICP精调 ==========
    if not args.skip_icp:
        print("\n" + "=" * 60)
        print("步骤 6: ICP精调")
        print("=" * 60)
        best_x, best_y, best_yaw = icp_refine(
            pts_centered, best_x, best_y, best_yaw, map_data, info
        )

    # ========== 8. 结果汇总 ==========
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(f"搜索位姿: ({best_x:.3f}, {best_y:.3f}, {math.degrees(best_yaw):.1f}°)")
    print(f"GT参考:   ({tf_gt[0]:.3f}, {tf_gt[1]:.3f}, {math.degrees(tf_gt[2]):.1f}°)")

    err_dist = math.sqrt((best_x - tf_gt[0]) ** 2 + (best_y - tf_gt[1]) ** 2)
    err_yaw = abs(math.atan2(math.sin(best_yaw - tf_gt[2]), math.cos(best_yaw - tf_gt[2])))
    print(f"误差: 位置={err_dist:.3f}m, 角度={math.degrees(err_yaw):.1f}°")

    # ========== 9. 可视化 ==========
    print("\n" + "=" * 60)
    print("步骤 7: 生成可视化")
    print("=" * 60)
    create_overlay_visualization(
        map_data, info, merged_pts, centroid,
        (best_score, best_x, best_y, best_yaw),
        tf_gt, output_path
    )

    print("\n完成!")


if __name__ == '__main__':
    main()
