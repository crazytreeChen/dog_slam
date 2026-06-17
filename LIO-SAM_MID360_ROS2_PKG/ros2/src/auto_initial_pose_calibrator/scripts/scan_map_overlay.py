#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_map_overlay.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
离线扫描-地图叠加可视化 (首帧多候选并行递推)

问题背景:
  opencode_multistep_localizer.py 的逐帧 ICP 序列约束能破解 180° 镜像
  (角度误差 0.7°), 但首帧只取一个最优种子。当地图里有多个几何相似的
  房间时, 首帧全局搜索可能锁错房间, 后续帧只在错误房间里精修 → 位置偏 19m。

本脚本改进:
  1. 首帧全局搜索保留 Top-K 候选 (而非只取 1 个)
  2. 每个候选独立跑完整的逐帧递推 (ICP 帧间约束 + 局部搜索 + sigma 衰减)
  3. 用"全程平均似然 + 平均墙壁覆盖率"对各候选终态打分, 选最优
  4. 把最优候选的扫描点红色叠加到原始栅格地图

复用 opencode_multistep_localizer.py 的核心函数, 不修改原文件。

用法:
  python scan_map_overlay.py
  python scan_map_overlay.py --data scan_viz/debug_match_data.npz --topk 10
"""

import os
import sys
import math
import argparse
import importlib.util

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("[ERROR] Need matplotlib: pip install matplotlib")
    sys.exit(1)


def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()


def load_module(filename, alias):
    """动态导入同目录下的脚本模块"""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raycast_validate(mu, frame_ranges, frame_tfs, amin, ainc, map_data, info, hu, n_frames=12):
    """
    Ray-casting 正交验证器 (与似然场原理不同, 用于消歧几何相似房间)。
    从每帧真实传感器中心向各方向投射射线, 比对"地图撞墙距离"与"实际激光测距"。
    错误房间里空旷方向会撞墙、撞墙方向会射穿 → 误差暴增。
    返回: 平均射线误差 (米, 越小越可信)。
    """
    step = max(1, len(frame_ranges) // n_frames)
    errs = []
    for i in range(0, len(frame_ranges), step):
        ranges = np.asarray(frame_ranges[i], float)
        v = (ranges > 0.15) & (ranges < 50.0)
        if int(np.sum(v)) < 20:
            continue
        a = amin + np.arange(len(ranges)) * ainc
        lp = np.column_stack([ranges[v] * np.cos(a[v]), ranges[v] * np.sin(a[v])])
        t0 = frame_tfs[i]
        c, s = math.cos(mu[2]), math.sin(mu[2])
        sx = c * t0[0] - s * t0[1] + mu[0]
        sy = s * t0[0] + c * t0[1] + mu[1]
        syaw = t0[2] + mu[2]
        errs.append(hu.ray_cast_score(lp, sx, sy, syaw, map_data, info, n_rays=48))
    return float(np.mean(errs)) if errs else 99.0


def geometry_constraints(mu, frame_ranges, frame_tfs, amin, ainc, map_data, info,
                         n_frames=8, free_margin_cells=6):
    """
    几何自洽性三约束 (用于消歧几何相似房间):
      约束1 端点灰色率:   激光端点落在地图未知(-1)区的比例。
      约束2 墙穿空旷率:   沿射线空闲段穿过地图墙壁(100)的比例。
      约束3 传感器非自由率: 每帧传感器自身位置不在地图自由区(0)的帧比例。
                           机器人实体必须在可通行区,若传感器落在灰色/墙壁区说明位姿彻底错误。
    三者均越低越好。返回 (gray_end_pct, wall_in_free_pct, sensor_nonfree_pct)。
    """
    x, y, yaw = mu
    res, ox, oy = info['resolution'], info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    gray_end = tot_end = wall_in_free = tot_free = 0
    sensor_nonfree = sensor_tot = 0
    step = max(1, len(frame_ranges) // n_frames)
    for i in range(0, len(frame_ranges), step):
        r = np.asarray(frame_ranges[i], float)
        v = (r > 0.15) & (r < 50.0)
        if int(np.sum(v)) < 20:
            continue
        a = amin + np.arange(len(r)) * ainc
        lx = r[v] * np.cos(a[v]); ly = r[v] * np.sin(a[v])
        t0 = frame_tfs[i]
        c, s = math.cos(yaw), math.sin(yaw)
        sx = c * t0[0] - s * t0[1] + x
        sy = s * t0[0] + c * t0[1] + y
        syaw = t0[2] + yaw

        # 约束3: 传感器自身位置不得位于地图墙壁(100)上 — 机器人不可能站墙里
        #         灰色未知区允许 (data_4场景: 下房间地图未完整扫描时机器人可能站在灰色区)
        sensor_tot += 1
        s_col = int((sx - ox) / res); s_row = int(H - 1 - (sy - oy) / res)
        if not (0 <= s_row < H and 0 <= s_col < W):
            sensor_nonfree += 1  # 出图边界算非法
        elif map_data[s_row, s_col] == 100:
            sensor_nonfree += 1  # 在墙里算非法

        cc, ss = math.cos(syaw), math.sin(syaw)
        ex = cc * lx - ss * ly + sx
        ey = ss * lx + cc * ly + sy

        # 约束1: 端点灰色率 (向量化)
        e_col = ((ex - ox) / res + 0.5).astype(np.int32)
        e_row = (H - 1 - (ey - oy) / res + 0.5).astype(np.int32)
        e_v = (e_col >= 0) & (e_col < W) & (e_row >= 0) & (e_row < H)
        tot_end += int(np.sum(e_v))
        gray_end += int(np.sum(map_data[e_row[e_v], e_col[e_v]] == -1))

        # 约束2: 墙穿空旷率 (降采样端点 + 沿射线稀疏采样)
        sample_ends = np.where(e_v)[0][::4]  # 每4个端点取1个
        for j in sample_ends:
            dist = math.hypot(ex[j] - sx, ey[j] - sy)
            nstep = max(1, int(dist / res))
            # 沿射线每3格采一次，留free_margin_cells格不碰端点墙
            ks = np.arange(2, nstep - free_margin_cells, 3)
            if len(ks) == 0:
                continue
            t = ks / nstep
            px = sx + (ex[j] - sx) * t
            py = sy + (ey[j] - sy) * t
            col2 = ((px - ox) / res + 0.5).astype(np.int32)
            row2 = (H - 1 - (py - oy) / res + 0.5).astype(np.int32)
            v2 = (col2 >= 0) & (col2 < W) & (row2 >= 0) & (row2 < H)
            tot_free += int(np.sum(v2))
            wall_in_free += int(np.sum(map_data[row2[v2], col2[v2]] == 100))
    return (100.0 * gray_end / max(tot_end, 1),
            100.0 * wall_in_free / max(tot_free, 1),
            100.0 * sensor_nonfree / max(sensor_tot, 1))


def occlusion_prescreen(frame_ranges, near_thresh=3.0, near_ratio_max=0.5, median_min=3.0):
    """
    采集质量预检: 被近距遮挡物包围时, 扫描内容是遮挡物轮廓而非房间墙体, 应拒绝。
    返回 (是否通过, near_ratio, median_range)。
    """
    allr = []
    for r in frame_ranges:
        r = np.asarray(r, float)
        allr.append(r[(r > 0.15) & (r < 50.0)])
    allr = np.concatenate(allr) if allr else np.array([])
    if len(allr) == 0:
        return False, 1.0, 0.0
    near_ratio = float(np.mean(allr < near_thresh))
    median_range = float(np.median(allr))
    ok = (near_ratio <= near_ratio_max) and (median_range >= median_min)
    return ok, near_ratio, median_range


def load_multistep_module():
    return load_module('opencode_multistep_localizer.py', 'ms_localizer')


def project_pts(pts_odom, x, y, yaw):
    """odom 系点 → map 系"""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([
        c * pts_odom[:, 0] - s * pts_odom[:, 1] + x,
        s * pts_odom[:, 0] + c * pts_odom[:, 1] + y,
    ])


def eval_pose_full(pts, mu, map_data, lf, info, ms):
    """在给定位姿下计算 (墙壁覆盖率%, 似然评分) —— 用全量点"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    c_y, s_y = math.cos(mu[2]), math.sin(mu[2])
    mx = c_y * pts[:, 0] - s_y * pts[:, 1] + mu[0]
    my = s_y * pts[:, 0] + c_y * pts[:, 1] + mu[1]
    ci = ((mx - ox) / res + 0.5).astype(np.int32)
    ri = ((my - oy) / res + 0.5).astype(np.int32)
    v = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    cells = map_data[ri[v], ci[v]]
    valid = (cells != -1)
    w_v = int(np.sum(cells[valid] == 100))
    f_v = int(np.sum(cells[valid] == 0))
    wall_pct = 100.0 * w_v / max(w_v + f_v, 1)
    sc, _, _ = ms.score_pose(pts, mu[0], mu[1], mu[2], lf, info)
    return wall_pct, sc


def wall_in_free_fast(pts, mu, map_data, info, sample_step=4, free_margin_cells=6):
    """
    快速墙穿空旷率 (递推内用, 单帧 odom 点云已在 mu 系下投影):
      对降采样后的每个端点, 沿"传感器原点(mu位置)→端点"的射线检查空闲段是否穿墙。
      pts 为 odom 系单帧点 (已是相对传感器的局部坐标, 原点即传感器)。
    返回穿墙比例 (0~1)。
    """
    res, ox, oy = info['resolution'], info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    x, y, yaw = mu
    c, s = math.cos(yaw), math.sin(yaw)
    sub = pts[::sample_step]
    ex = c * sub[:, 0] - s * sub[:, 1] + x
    ey = s * sub[:, 0] + c * sub[:, 1] + y
    wall = tot = 0
    for j in range(len(ex)):
        dist = math.hypot(ex[j] - x, ey[j] - y)
        nstep = int(dist / res)
        for k in range(2, nstep - free_margin_cells, 3):  # 沿线每3格采一次
            px = x + (ex[j] - x) * k / nstep
            py = y + (ey[j] - y) * k / nstep
            col = int((px - ox) / res); rr = int(H - 1 - (py - oy) / res)
            if 0 <= rr < H and 0 <= col < W:
                tot += 1
                if map_data[rr, col] == 100:
                    wall += 1
    return wall / max(tot, 1)


def local_search_constrained(pts, cx, cy, yaw, lf, map_data, info, ms,
                             radius=3.0, pos_step=0.2, angle_range=15, angle_step=2,
                             wall_penalty=8.0):
    """
    约束感知局部搜索: 在似然评分基础上减去"墙穿空旷"惩罚。
    阻止位姿滑向相邻相似房间 (滑过去会让本该空旷的方向穿过隔壁墙)。
    """
    best_score = -1e9
    best_pose = (cx, cy, yaw)
    for dx in np.arange(-radius, radius + 1e-5, pos_step):
        for dy in np.arange(-radius, radius + 1e-5, pos_step):
            for da in range(-int(angle_range), int(angle_range) + 1, angle_step):
                ax, ay = cx + dx, cy + dy
                ayaw = yaw + math.radians(da)
                sc, _, _ = ms.score_pose(pts, ax, ay, ayaw, lf, info)
                if sc < -1e8:
                    continue
                # 墙穿惩罚 (粗采样, 仅在似然较优的位姿附近才精算以省时)
                if sc > best_score - 0.3:
                    wf = wall_in_free_fast(pts, (ax, ay, ayaw), map_data, info)
                    sc -= wall_penalty * wf
                if sc > best_score:
                    best_score = sc
                    best_pose = (ax, ay, ayaw)
    return best_pose, best_score


# ============================================================
# 单候选完整递推 (复刻 MultiStepLocalizer 的逐帧逻辑, 种子由外部给定)
# ============================================================
def recurse_from_seed(frames, seed_pose, ms, lf, map_data, info, decay=0.7, min_sigma=0.5,
                      constrained=True, verbose=False):
    """
    从指定首帧种子位姿出发, 逐帧 ICP + 局部搜索递推。

    constrained=True → 使用 local_search_constrained (墙穿惩罚),
                       防止轨迹滑入几何相似但错误的房间。数据4这类多房间场景必备。
    verbose → 逐帧打印 ICP/local_search/reject 详情。

    返回:
      final_pose (x,y,yaw), history[(i,x,y,yaw,sigma,wall%,score)],
      mean_score, mean_wall, unknown_pct(终态合并点落在未知区域比例)
    """
    mu = list(seed_pose)
    sigma = 5.0
    sigma_angle = 20.0
    history = []

    for i in range(len(frames)):
        pts, _ = frames[i]
        if len(pts) < 10:
            continue

        if i > 0:
            prev = frames[i - 1][0]
            R, t = np.eye(2), np.zeros(2)
            icp_used = False
            if len(pts) > 10 and len(prev) > 10:
                R, t = ms.icp_scan_to_scan(pts, prev)
                dyaw_icp = math.atan2(R[1, 0], R[0, 0])
                if abs(dyaw_icp) >= math.radians(30):  # 旋转过快 ICP 不可靠
                    R, t = np.eye(2), np.zeros(2)
                else:
                    icp_used = True
            dyaw_icp = math.atan2(R[1, 0], R[0, 0])

            c_m, s_m = math.cos(mu[2]), math.sin(mu[2])
            pred_x = mu[0] + c_m * t[0] - s_m * t[1]
            pred_y = mu[1] + s_m * t[0] + c_m * t[1]
            pred_yaw = mu[2] + dyaw_icp

            search_radius = min(sigma * 1.5, 3.0)
            angle_range = min(sigma_angle * 1.5, 20)
            if constrained:
                pose, _ = local_search_constrained(pts, pred_x, pred_y, pred_yaw, lf,
                                                   map_data, info, ms,
                                                   radius=search_radius, angle_range=int(angle_range))
            else:
                pose, _ = ms.local_search(pts, pred_x, pred_y, pred_yaw, lf, info,
                                          radius=search_radius, angle_range=int(angle_range))

            # ── ICP 失败保护 (复刻 MultiStepLocalizer.run() 第490-510行) ──
            # 搜索后计算当前帧的墙壁覆盖率, 防止 local_search 滑到错误房间
            c_y, s_y = math.cos(pose[2]), math.sin(pose[2])
            res_i = info['resolution']; ox_i = info['origin_x']; oy_i = info['origin_y']
            H_i, W_i = info['height'], info['width']
            mx = c_y * pts[:, 0] - s_y * pts[:, 1] + pose[0]
            my = s_y * pts[:, 0] + c_y * pts[:, 1] + pose[1]
            ci = ((mx - ox_i) / res_i + 0.5).astype(np.int32)
            ri = ((my - oy_i) / res_i + 0.5).astype(np.int32)
            mv = (ci >= 0) & (ci < W_i) & (ri >= 0) & (ri < H_i)
            cells = map_data[ri[mv], ci[mv]] if int(np.sum(mv)) > 10 else np.array([-1])
            valid_c = (cells != -1); n_vc = int(np.sum(valid_c))
            w_v = int(np.sum(cells[valid_c] == 100)) if n_vc > 0 else 0
            f_v = int(np.sum(cells[valid_c] == 0)) if n_vc > 0 else 0
            wall_pct_per_frame = w_v / max(w_v + f_v, 1)
            icp_jump = math.sqrt(t[0]**2 + t[1]**2) if icp_used else 0

            rejected = False
            # 条件1: 墙壁覆盖率 < 20% → 直接拒绝
            # 条件2: ICP跳变 > 2m 且覆盖率 < 35% → 拒绝 (保留上一帧位置, 只更新角度)
            if i > 3 and (wall_pct_per_frame < 0.20 or (icp_jump > 2.0 and wall_pct_per_frame < 0.35)):
                pose = (mu[0], mu[1], pose[2])
                rejected = True

            mu = list(pose)
            sigma = max(sigma * decay, min_sigma)
            sigma_angle = max(sigma_angle * decay, 2.0)

            if verbose:
                dx = mu[0] - pred_x; dy = mu[1] - pred_y
                corr = math.hypot(dx, dy)
                dya = math.degrees(abs(math.atan2(math.sin(mu[2] - pred_yaw),
                                                  math.cos(mu[2] - pred_yaw))))
                icp_tag = "[ICP]" if icp_used else "[pred]"
                mode_tag = "[cstr]" if constrained else "[free]"
                rej_tag = " [REJECT]" if rejected else ""
                print(f"  f{i:02d} {icp_tag}{mode_tag}: pred=({pred_x:.2f},{pred_y:.2f}) "
                      f"corr={corr:.3f}m,{dya:.1f}deg "
                      f"→ ({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.0f}°) "
                      f"jump={icp_jump:.1f}m wall={100*wall_pct_per_frame:.0f}%{rej_tag}")

        wp, sc = eval_pose_full(pts, mu, map_data, lf, info, ms)
        history.append((i, mu[0], mu[1], mu[2], sigma, wp, sc))

    mean_score = float(np.mean([h[6] for h in history])) if history else -1e9
    mean_wall = float(np.mean([h[5] for h in history])) if history else 0.0

    # 终态: 所有帧用各自最终位姿合并, 统计落在未知区域的比例 (越低越可信)
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    n_unk, n_tot = 0, 0
    for h in history:
        i = h[0]
        p = project_pts(frames[i][0], h[1], h[2], h[3])
        ci = ((p[:, 0] - ox) / res + 0.5).astype(np.int32)
        ri = ((p[:, 1] - oy) / res + 0.5).astype(np.int32)
        v = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
        cells = map_data[ri[v], ci[v]]
        n_unk += int(np.sum(cells == -1)) + int(np.sum(~v))
        n_tot += len(p)
    unknown_pct = 100.0 * n_unk / max(n_tot, 1)

    return tuple(mu), history, mean_score, mean_wall, unknown_pct


# ============================================================
# 可视化
# ============================================================
def create_overlay(map_data, info, frame_pts, best, all_cands, tf_gt, output_path):
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W * res, oy, oy + H * res]

    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data == 0] = [1, 1, 1]
    map_bg[map_data == 100] = [0.15, 0.15, 0.15]
    map_bg[map_data == -1] = [0.65, 0.65, 0.65]

    final_pose, history = best['final'], best['history']
    fx, fy, fyaw = final_pose

    red_pts = np.vstack([project_pts(frame_pts[h[0]][0], h[1], h[2], h[3]) for h in history])
    all_odom = np.vstack([fp[0] for fp in frame_pts if len(fp[0]) > 0])
    gt_pts = project_pts(all_odom, tf_gt[0], tf_gt[1], tf_gt[2])

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))

    # (a) 最优候选红色叠加
    ax = axes[0]
    ax.imshow(map_bg, origin='lower', extent=extent, aspect='equal')
    ax.scatter(red_pts[::2, 0], red_pts[::2, 1], s=1.2, c='red', alpha=0.55, label='匹配扫描 (红)')
    ax.plot(fx, fy, 'r*', markersize=16)
    ax.arrow(fx, fy, 2.5 * math.cos(fyaw), 2.5 * math.sin(fyaw),
             head_width=0.5, head_length=0.4, fc='red', ec='darkred', lw=2.5)
    ax.set_title(f'(a) 最优候选红色叠加\n位姿 ({fx:.2f}, {fy:.2f}, {math.degrees(fyaw):.1f}°)')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.12)

    # (b) GT 参考叠加
    ax = axes[1]
    ax.imshow(map_bg, origin='lower', extent=extent, aspect='equal')
    ax.scatter(gt_pts[::2, 0], gt_pts[::2, 1], s=1.2, c='blue', alpha=0.55, label='GT 扫描 (蓝)')
    ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=16)
    ax.arrow(tf_gt[0], tf_gt[1], 2.5 * math.cos(tf_gt[2]), 2.5 * math.sin(tf_gt[2]),
             head_width=0.5, head_length=0.4, fc='blue', ec='darkblue', lw=2.5)
    ax.set_title(f'(b) GT 参考叠加\n位姿 ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f}°)')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.12)

    # (c) 所有候选终态对比
    ax = axes[2]
    ax.imshow(map_bg, origin='lower', extent=extent, aspect='equal')
    ax.scatter(red_pts[::3, 0], red_pts[::3, 1], s=1, c='red', alpha=0.4, label='最优匹配 (红)')
    ax.scatter(gt_pts[::3, 0], gt_pts[::3, 1], s=1, c='blue', alpha=0.4, label='GT (蓝)')
    for k, c in enumerate(all_cands):
        cx, cy, _ = c['final']
        is_best = (c is best)
        ax.plot(cx, cy, 'o', color='lime' if is_best else 'orange',
                markersize=10 if is_best else 6,
                markeredgecolor='k', label='候选终态' if k == 0 else None)
        ax.annotate(f"{k}", (cx, cy), fontsize=8, color='k')
    ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=14)

    err_dist = math.hypot(fx - tf_gt[0], fy - tf_gt[1])
    err_yaw = abs(math.atan2(math.sin(fyaw - tf_gt[2]), math.cos(fyaw - tf_gt[2])))
    ax.set_title(f'(c) {len(all_cands)} 候选终态对比\n位置误差 {err_dist:.2f}m   角度误差 {math.degrees(err_yaw):.1f}°')
    ax.legend(fontsize=9, loc='upper right'); ax.grid(True, alpha=0.12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n可视化已保存: {output_path}")
    return err_dist, math.degrees(err_yaw)


def main():
    parser = argparse.ArgumentParser(description='扫描-地图叠加 (首帧多候选并行递推)')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--topk', type=int, default=10, help='首帧保留候选数')
    parser.add_argument('--decay', type=float, default=0.7)
    parser.add_argument('--force', action='store_true', help='预检失败仍强制定位')
    parser.add_argument('--constrained', action='store_true', default=True,
                        help='启用墙穿约束搜索 (默认, 防轨迹漂移到相似房间)')
    parser.add_argument('--no-constrained', dest='constrained', action='store_false',
                        help='禁用墙穿约束 (回退纯似然场搜索)')
    parser.add_argument('--force-wallhit', action='store_true',
                        help='始终强制启用 WallHit 独立搜索 (跳过 raycast 阈值检查)')
    parser.add_argument('--skip-wallhit', action='store_true',
                        help='跳过 WallHit 回退 (强制只用似然场首帧结果)')
    parser.add_argument('--verbose', action='store_true',
                        help='逐帧打印递推详情')
    args = parser.parse_args()

    npz_path = os.path.normpath(args.data)
    output_path = args.output or os.path.join(os.path.dirname(npz_path), 'scan_map_overlay.png')

    ms = load_multistep_module()
    hu = load_module('opencode_contour_hu_matcher.py', 'hu_matcher')  # 提供 ray_cast_score

    print("=" * 60)
    print("步骤 1: 加载数据 + 逐帧转 odom 点云 + 似然场")
    print("=" * 60)
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = ms.load_data(npz_path)
    frame_pts = [(ms.frame_to_odom_pts(r, tf, angle_min, angle_inc), tf)
                 for r, tf in zip(frame_ranges, frame_tfs)]
    lf = ms.build_likelihood_field(map_data, info)

    # ── 采集质量预检: 被近距遮挡物包围的扫描应拒绝 (如 data_5) ──
    ok, near_ratio, med_r = occlusion_prescreen(frame_ranges)
    print(f"\n[预检] 近距(<3m)点占比={near_ratio*100:.0f}%, 中位测距={med_r:.2f}m -> "
          f"{'通过' if ok else '拒绝(疑似被遮挡物包围, 扫描不含足够墙体特征)'}")
    if not ok:
        print("[预检] 该数据被判定为遮挡无效样本, 不输出定位结果。建议移至开阔处重新采集。")
        if not args.force:
            sys.exit(2)
        print("[预检] --force 指定, 继续尝试定位 (结果不可信)")

    # 首帧
    pts0 = frame_pts[0][0]

    print("\n" + "=" * 60)
    print(f"步骤 2: 首帧全局搜索 → 保留 Top-{args.topk} 候选")
    print("=" * 60)
    cand0 = ms.global_search_first_frame(pts0, lf, map_data, info, top_k=args.topk)
    if not cand0:
        print("[ERROR] 首帧无候选"); sys.exit(1)
    print(f"  得到 {len(cand0)} 个首帧候选")

    # ── WallHit fallback (借鉴 multistep): 似然场最强解射线投射有效率<30% ──
    #    说明似然场被几何相似但内容不同的房间吸引, 切换墙壁命中评分独立搜索。
    #    这是 data_4 (下房间扫描) 能找到正确房间的关键 — 上房间墙更完整,
    #    似然场会偏爱上房间, 但 wallhit 严格要求端点真的落在墙像素上。
    do_wallhit = False
    if args.skip_wallhit:
        print(f"  [WallHit] --skip-wallhit 指定, 跳过独立墙壁搜索")
    elif args.force_wallhit:
        print(f"  [WallHit] --force-wallhit 指定, 强制执行独立墙壁搜索")
        do_wallhit = True
    else:
        best_rc = ms.score_pose_raycast(pts0, cand0[0][1], cand0[0][2],
                                         math.radians(cand0[0][3]), map_data, info)
        rc_valid_rate = best_rc[2] / max(best_rc[1], 1) if best_rc[0] > -1e8 else 0
        print(f"  [诊断] 似然首选射线投射有效率={rc_valid_rate:.0%}")
        if rc_valid_rate < 0.30:
            print(f"  [WallHit fallback] 启动独立墙壁命中搜索...")
            do_wallhit = True
        else:
            print(f"  [WallHit] raycast有效={rc_valid_rate:.0%}≥30%, 信任似然场首帧结果")

    if do_wallhit:
        wh_candidates = []
        res_m = info['resolution']; ox_m = info['origin_x']; oy_m = info['origin_y']
        mw_m = info['width'] * res_m; mh_m = info['height'] * res_m
        for ax in np.arange(ox_m + 3, ox_m + mw_m - 3, 2.0):
            for ay in np.arange(oy_m + 3, oy_m + mh_m - 3, 2.0):
                for adeg in range(0, 360, 10):
                    wh, nw, _ = ms.score_pose_wallhit(pts0, ax, ay,
                                                      math.radians(adeg), map_data, info)
                    if wh > 0:
                        wh_candidates.append((wh, ax, ay, adeg))
        if wh_candidates:
            wh_candidates.sort(key=lambda x: x[0], reverse=True)
            wh_nms = []
            for wh, ax, ay, ad in wh_candidates:
                dup = any(math.sqrt((ax-wx)**2+(ay-wy)**2) < 2.0 for _, wx, wy, _ in wh_nms)
                if not dup:
                    wh_nms.append((wh, ax, ay, ad))
                if len(wh_nms) >= args.topk:
                    break
            cand0 = wh_nms
            print(f"  [WallHit] 替换为 {len(wh_nms)} 个墙命中候选, 最优"
                  f"({wh_nms[0][1]:.1f},{wh_nms[0][2]:.1f},{wh_nms[0][3]:.0f}°) "
                  f"命中率={wh_nms[0][0]:.2f}")
        else:
            print("  [WallHit] 无候选, 继续用似然场结果")

    print("\n" + "=" * 60)
    print(f"步骤 3: 每个候选独立跑完整递推 ({len(frame_pts)} 帧)")
    print("=" * 60)
    results = []
    for k, (sc0, hx, hy, had) in enumerate(cand0):
        # 候选种子精搜细化
        seed, _ = ms.local_search(pts0, hx, hy, math.radians(had), lf, info,
                                  radius=2.0, pos_step=0.3)
        final, hist, mscore, mwall, unk = recurse_from_seed(
            frame_pts, seed, ms, lf, map_data, info, decay=args.decay,
            constrained=args.constrained, verbose=args.verbose)
        # 几何自洽双约束 (主判据): 端点灰色率 + 墙穿空旷率
        gray_end, wall_free, sensor_nf = geometry_constraints(final, frame_ranges, frame_tfs,
                                                              angle_min, angle_inc, map_data, info)
        # Ray-cast 正交验证 (辅助)
        rc_err = raycast_validate(final, frame_ranges, frame_tfs, angle_min, angle_inc,
                                  map_data, info, hu)
        # Wallhit 评分 (multistep 关键消歧器): 严格要求端点真的落在墙像素上
        # 对所有帧合并点云算一次, 量化扫描和地图墙的匹配紧密度
        all_pts = np.vstack([fp[0] for fp in frame_pts if len(fp[0]) > 0])
        wh_score, wh_nwall, wh_nv = ms.score_pose_wallhit(all_pts, final[0], final[1],
                                                         final[2], map_data, info)
        results.append({'final': final, 'history': hist, 'mean_score': mscore,
                        'mean_wall': mwall, 'unknown_pct': unk, 'raycast': rc_err,
                        'gray_end': gray_end, 'wall_free': wall_free,
                        'sensor_nf': sensor_nf, 'wallhit': max(wh_score, 0.0)})
        print(f"  候选#{k}: 种子({hx:.1f},{hy:.1f},{had:.0f}°) → 终态"
              f"({final[0]:.1f},{final[1]:.1f},{math.degrees(final[2]):.0f}°) "
              f"| 墙命中={max(wh_score,0):.2f} 传感器非自由={sensor_nf:.0f}% "
              f"端点灰={gray_end:.0f}% 墙穿={wall_free:.1f}% 均分={mscore:.2f}")

    # 去重: 终态相近的候选合并 (保留 wallhit 最高的)
    dedup = []
    for r in sorted(results, key=lambda r: -r['wallhit']):
        if not any(math.hypot(r['final'][0]-d['final'][0], r['final'][1]-d['final'][1]) < 2.0
                   and abs(math.atan2(math.sin(r['final'][2]-d['final'][2]),
                                      math.cos(r['final'][2]-d['final'][2]))) < math.radians(20)
                   for d in dedup):
            dedup.append(r)

    # ── 消歧策略 ──
    #  传感器只排除"站墙里"的(放宽灰色区), 因为下房间地图可能未完整扫描
    #  排名: 多维综合评分
    #    似然场(mean_score) 在几何相似房间中区分度不足 → 降权
    #    墙壁命中(wallhit) 严格贴合, 可信度高
    #    墙穿空旷(wall_free) 关键区分信号: 正确位姿下射线穿过自由区, 错误位姿穿墙
    #    raycast 正交验证: 光线投射误差(米), 越小越可信, 原理独立于似然场
    #    gray_end 端点灰色率: 端点落在地图未知区说明位姿彻底错误
    SENSOR_WALL_MAX = 50.0  # 传感器在墙里的帧比例超此值 -> 硬淘汰
    for r in dedup:
        r['sensor_ok'] = r['sensor_nf'] <= SENSOR_WALL_MAX
        r['valid']     = r['sensor_ok']
        # 各指标典型范围:
        #   mean_score ~0.5-1.4, wallhit ~0-1.2, wall_free 0-20%
        #   raycast ~0-10m, gray_end 0-100%, sensor_nf 0-100%
        #
        # 多房间消歧靠 wall_free (穿墙率差异 5-15%), raycast (正交验证), gray_end (端点灰):
        #   这些指标在正确/错误房间之间差异显著 (正确: wall_free~2% raycast~2m gray_end~10%)
        #   而 mean_score 和 wallhit 在相似房间间区分不足
        r['combined'] = (r['mean_score'] * 0.3        # 似然场弱化 (几何相似房间区分不足)
                         + r['wallhit'] * 0.5         # 墙壁命中: 严格贴墙更可信
                         - r['wall_free'] * 0.20      # 墙穿空旷增强 (关键消歧)
                         - r['raycast'] * 0.25        # raycast 误差增强 (正交验证)
                         - r['gray_end'] * 0.015      # 端点灰色率增强
                         - r['sensor_nf'] * 0.03)     # 传感器非自由增强
    dedup.sort(key=lambda r: (r['valid'], r['combined']), reverse=True)
    results = dedup
    best = results[0]

    print("\n" + "=" * 60)
    print(f"步骤 4: 候选排名 (多维综合评分; 传感器在墙里≤{SENSOR_WALL_MAX}%)")
    print("=" * 60)
    for rank, r in enumerate(results):
        fx, fy, fyaw = r['final']
        trust = "可信" if r['valid'] else "淘汰(传感器在墙里)"
        tag = " ← 选中" if r is best else ""
        # 评分拆解: 更直观看出各维度贡献
        sc_lf  = r['mean_score'] * 0.3
        sc_wh  = r['wallhit'] * 0.5
        sc_wf  = -r['wall_free'] * 0.20
        sc_rc  = -r['raycast'] * 0.25
        sc_ge  = -r['gray_end'] * 0.015
        sc_sn  = -r['sensor_nf'] * 0.03
        print(f"  排名{rank}: 综合={r['combined']:.2f} "
              f"| 似然={sc_lf:+.2f} 墙命中={sc_wh:+.2f} 穿墙={sc_wf:+.2f} "
              f"raycast={sc_rc:+.2f} 灰色={sc_ge:+.2f} 传感={sc_sn:+.2f}")
        print(f"          raw: mean_score={r['mean_score']:.2f} wallhit={r['wallhit']:.2f} "
              f"wall_free={r['wall_free']:.1f}% raycast={r['raycast']:.2f}m "
              f"gray_end={r['gray_end']:.0f}% sensor_nf={r['sensor_nf']:.0f}%")
        print(f"          位姿 ({fx:.1f},{fy:.1f},{math.degrees(fyaw):.0f}°) [{trust}]{tag}")
    if not best['valid']:
        print("  [警告] 最优候选也站在墙里, 定位不可信")

    print("\n" + "=" * 60)
    print("步骤 5: 生成红色叠加可视化")
    print("=" * 60)
    err_d, err_y = create_overlay(map_data, info, frame_pts, best, results, tf_gt, output_path)

    fx, fy, fyaw = best['final']
    print("\n" + "=" * 60)
    print("最终结果 (不依赖GT, 纯几何自评分)")
    print("=" * 60)
    print(f"  匹配位姿: ({fx:.2f}, {fy:.2f}, {math.degrees(fyaw):.1f}°)")
    print(f"  传感器非自由={best['sensor_nf']:.0f}%  端点灰={best['gray_end']:.0f}%  墙穿空旷={best['wall_free']:.1f}%")
    print(f"  (仅供参考) AMCL存档GT: ({tf_gt[0]:.2f}, {tf_gt[1]:.2f}, {math.degrees(tf_gt[2]):.1f}°)")
    if best['valid']:
        print("  [OK] 几何约束通过: 传感器在自由区, 端点落已知区, 空旷区无墙穿")
    else:
        print("  [WARN] 几何约束未通过, 定位可信度低")


if __name__ == '__main__':
    main()
