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
import cv2

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
        s_col = int((sx - ox) / res); s_row = int((sy - oy) / res + 0.5)
        if not (0 <= s_row < H and 0 <= s_col < W):
            sensor_nonfree += 1  # 出图边界算非法
        elif map_data[s_row, s_col] == 100:
            sensor_nonfree += 1  # 在墙里算非法

        cc, ss = math.cos(syaw), math.sin(syaw)
        ex = cc * lx - ss * ly + sx
        ey = ss * lx + cc * ly + sy

        # 约束1: 端点灰色率 (向量化)
        e_col = ((ex - ox) / res + 0.5).astype(np.int32)
        e_row = ((ey - oy) / res + 0.5).astype(np.int32)
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
            row2 = ((py - oy) / res + 0.5).astype(np.int32)
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


def wall_in_free_fast(pts, mu, map_data, info, sample_step=4, free_margin_cells=6,
                      sensor_odom_x=0.0, sensor_odom_y=0.0):
    """
    快速墙穿空旷率 (递推内用, 单帧 odom 点云已在 mu 系下投影):
      对降采样后的每个端点, 沿"实际传感器原点→端点"的射线检查空闲段是否穿墙。
      pts 为 odom 系单帧点 (已含传感器在 odom 中的绝对位置),
      sensor_odom_* 为当前帧传感器在 odom 系中的位置 (来自 frame_tfs[i][:2])。
    返回穿墙比例 (0~1)。
    """
    res, ox, oy = info['resolution'], info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    x, y, yaw = mu
    c, s = math.cos(yaw), math.sin(yaw)
    # ── 射线原点: 实际传感器在 map 系中的位置 (非 odom 原点) ──
    origin_x = c * sensor_odom_x - s * sensor_odom_y + x
    origin_y = s * sensor_odom_x + c * sensor_odom_y + y
    sub = pts[::sample_step]
    ex = c * sub[:, 0] - s * sub[:, 1] + x
    ey = s * sub[:, 0] + c * sub[:, 1] + y
    wall = tot = 0
    for j in range(len(ex)):
        dist = math.hypot(ex[j] - origin_x, ey[j] - origin_y)
        nstep = int(dist / res)
        for k in range(2, nstep - free_margin_cells, 3):  # 沿线每3格采一次
            px = origin_x + (ex[j] - origin_x) * k / nstep
            py = origin_y + (ey[j] - origin_y) * k / nstep
            col = int((px - ox) / res + 0.5); row = int((py - oy) / res + 0.5)
            if 0 <= row < H and 0 <= col < W:
                tot += 1
                if map_data[row, col] == 100:
                    wall += 1
    return wall / max(tot, 1)


def local_search_constrained(pts, cx, cy, yaw, lf, map_data, info, ms,
                             radius=3.0, pos_step=0.2, angle_range=15, angle_step=2,
                             wall_penalty=8.0, sensor_odom_x=0.0, sensor_odom_y=0.0):
    """
    约束感知局部搜索: 两级粗→细搜索 + 墙穿惩罚。
    阻止位姿滑向相邻相似房间。
    """
    best_score = -1e9
    best_pose = (cx, cy, yaw)

    # ── 阶段1: 粗搜索 (大步长, 不用墙惩罚 — 用 wallhit 快速过滤) ──
    coarse_ps = max(pos_step * 2.0, 0.5)
    coarse_as = max(angle_step * 2, 5)
    coarse_rad = radius
    for dx in np.arange(-coarse_rad, coarse_rad + 1e-5, coarse_ps):
        for dy in np.arange(-coarse_rad, coarse_rad + 1e-5, coarse_ps):
            for da in range(-int(angle_range), int(angle_range) + 1, coarse_as):
                ax, ay = cx + dx, cy + dy
                ayaw = yaw + math.radians(da)
                sc, _, _ = ms.score_pose(pts, ax, ay, ayaw, lf, info)
                if sc < -1e8:
                    continue
                # 阶段1 只用 wallhit 快速补分 (不用耗时的 wall_in_free_fast)
                wh, _, _ = ms.score_pose_wallhit(pts, ax, ay, ayaw, map_data, info)
                sc += max(wh, 0) * 0.3
                if sc > best_score:
                    best_score = sc; best_pose = (ax, ay, ayaw)

    # ── 阶段2: 细搜索 (围绕粗搜最优, 小窗口 + 墙穿惩罚精修) ──
    fine_rad = pos_step * 1.5
    cx_r, cy_r, yaw_r = best_pose
    for dx in np.arange(-fine_rad, fine_rad + 1e-5, pos_step):
        for dy in np.arange(-fine_rad, fine_rad + 1e-5, pos_step):
            for da in range(-int(angle_step * 2), int(angle_step * 2) + 1, int(angle_step)):
                ax, ay = cx_r + dx, cy_r + dy
                ayaw = yaw_r + math.radians(da)
                sc, _, _ = ms.score_pose(pts, ax, ay, ayaw, lf, info)
                if sc < -1e8:
                    continue
                # 精细阶段使用墙穿惩罚 (传入传感器 odom 位置计算正确射线原点)
                if sc > best_score - 0.3:
                    wf = wall_in_free_fast(pts, (ax, ay, ayaw), map_data, info,
                                           sensor_odom_x=sensor_odom_x,
                                           sensor_odom_y=sensor_odom_y)
                    sc -= wall_penalty * wf
                if sc > best_score:
                    best_score = sc
                    best_pose = (ax, ay, ayaw)
    return best_pose, best_score


# ============================================================
# 房间分割: 形态学关门分离连通房间
# ============================================================
def find_rooms_freespace(map_data, info, gap_m=2.0, min_area_m2=20,
                         max_area_m2=800):
    """
    用形态学膨胀关闭走廊/门洞, 分割独立房间。
    返回房间列表, 按中心 x 排序。过滤超大房间 (>max_area_m2, 即走廊未关死)。
    若过滤后无剩余, 返回全图作为单房间回退。
    Return: [(cx, cy, bbox_x0, bbox_y0, bbox_x1, bbox_y1, area_m2), ...]
    """
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    W_img = info['width']; H_img = info['height']
    gap_px = int(gap_m / res)
    wall_mask = (map_data == 100).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px, gap_px))
    wall_dilated = cv2.dilate(wall_mask, kernel, iterations=1)
    room_mask = (wall_dilated == 0).astype(np.uint8)
    # 清除小噪点
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=4)

    rooms = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        area_m2 = a * res * res
        if area_m2 < min_area_m2:
            continue
        l = stats[i, cv2.CC_STAT_LEFT]; t = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        # 过滤超大房间 (走廊未关死, 包含全图连通区)
        if max_area_m2 is not None and area_m2 > max_area_m2:
            continue
        cx = ox + (l + w / 2) * res
        cy = oy + (t + h / 2) * res
        bx0 = ox + l * res;          by0 = oy + t * res
        bx1 = ox + (l + w) * res;    by1 = oy + (t + h) * res
        rooms.append((cx, cy, bx0, by0, bx1, by1, area_m2))

    if not rooms:
        # 回退: 全图作为单房间
        map_x0 = ox; map_y0 = oy
        map_x1 = ox + W_img * res; map_y1 = oy + H_img * res
        rooms.append((ox + W_img * res / 2, oy + H_img * res / 2,
                      map_x0, map_y0, map_x1, map_y1,
                      W_img * H_img * res * res))
    else:
        rooms.sort(key=lambda r: r[0])  # 按 x 排序
    return rooms


# ============================================================
# 方案A: 自由空间最大化评分 (data_4 开放空间定位)
# ============================================================
def score_pose_freespace(pts, x, y, yaw, map_data, info):
    """
    自由空间评分: 机器人位于开放空间时, 扫描点应绝大多数落在已知自由区(0)。
    返回 (combined_score, free_ratio, wall_ratio, n_free, n_wall, n_unknown, n_total)

    评分 = free_ratio * (n_free / n_total) - 2.0 * wall_ratio
    同时考虑 1) 自由占比(质量) 和 2) 绝对自由像素数(覆盖量),
    避免"所有点落未知区只剩1个自由点也拿满分"的退化情况。
    """
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    H, W = info['height'], info['width']
    c, s = math.cos(yaw), math.sin(yaw)
    mx = c * pts[:, 0] - s * pts[:, 1] + x
    my = s * pts[:, 0] + c * pts[:, 1] + y
    col = ((mx - ox) / res + 0.5).astype(np.int32)
    row = ((my - oy) / res + 0.5).astype(np.int32)
    v = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    cells = map_data[row[v], col[v]]
    n_total = len(pts)
    n_free = int(np.sum(cells == 0))
    n_wall = int(np.sum(cells == 100))
    n_unknown = int(np.sum(cells == -1))
    known = n_free + n_wall
    if known > 10:
        free_ratio = n_free / known
        wall_ratio = n_wall / known
    else:
        free_ratio = 0.0; wall_ratio = 1.0
    # 混合评分: 自由率(质量) * 绝对覆盖(数量) / 总点数 - 墙惩罚
    coverage_factor = n_free / max(n_total, 1)
    score = free_ratio * coverage_factor - 2.0 * wall_ratio * (n_wall / max(n_total, 1))
    return score, free_ratio, wall_ratio, n_free, n_wall, n_unknown, n_total


def _is_known_cell(mx, my, map_data, info):
    """返回地图坐标 (mx,my) 所在格是否在已知区域 (非灰色/unknown)。"""
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    H, W = info['height'], info['width']
    col = int((mx - ox) / res + 0.5)
    row = int((my - oy) / res + 0.5)
    if 0 <= col < W and 0 <= row < H:
        return map_data[row, col] != -1  # -1 = unknown/gray
    return False


def global_search_freespace(pts, map_data, info, top_k=10,
                            xy_step=2.0, angle_step_deg=15, search_bbox=None):
    """
    自由空间最大化全局搜索。
    遍历地图网格，对每个候选位姿计算扫描点落在已知自由区的比例。
    跳过灰色未知区(unknown)的候选位置, 大幅减少无效计算。
    可选 search_bbox=(x_min, y_min, x_max, y_max) 将搜索约束在指定矩形内。
    返回 Top-K (score, x, y, deg) 经 NMS 去重。
    """
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    mw = info['width'] * res; mh = info['height'] * res
    if search_bbox is not None:
        x_min, y_min, x_max, y_max = search_bbox
        x_start = max(ox + 3, x_min); x_end = min(ox + mw - 3, x_max)
        y_start = max(oy + 3, y_min); y_end = min(oy + mh - 3, y_max)
    else:
        x_start, x_end = ox + 3, ox + mw - 3
        y_start, y_end = oy + 3, oy + mh - 3
    # 预计算已知区域 mask (在搜索步长分辨率下)
    xs = np.arange(x_start, x_end, xy_step)
    ys = np.arange(y_start, y_end, xy_step)
    known_mask = np.zeros((len(ys), len(xs)), dtype=bool)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            known_mask[iy, ix] = _is_known_cell(x, y, map_data, info)
    n_known = int(known_mask.sum())
    n_total = len(xs) * len(ys)
    print(f"  [FreeSpace] search grid: {n_known}/{n_total} known cells "
          f"(跳过 {n_total-n_known} 个未知区, {100*(n_total-n_known)/max(n_total,1):.0f}%)")
    candidates = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if not known_mask[iy, ix]:
                continue  # 跳过灰色未知区
            for adeg in range(0, 360, angle_step_deg):
                sc, fr, wr, nf, nw, nu, nt = score_pose_freespace(
                    pts, x, y, math.radians(adeg), map_data, info)
                if sc > 0.05 and nf > 50:  # 至少50点落自由区, 且有区分度
                    candidates.append((sc, x, y, adeg, fr, wr, nf, nw))
    if not candidates:
        return []
    candidates.sort(key=lambda c: c[0], reverse=True)
    nms = []
    for item in candidates:
        sc, x, y, adeg = item[0], item[1], item[2], item[3]
        if any(math.hypot(x - nx, y - ny) < 3.0 for _, nx, ny, _ in nms):
            continue
        nms.append((sc, x, y, adeg))
        if len(nms) >= top_k:
            break
    return nms


def global_search_freespace_per_room(pts, map_data, info, top_k=3,
                                     xy_step=2.0, angle_step_deg=15,
                                     gap_m=2.0, min_area_m2=20):
    """
    逐房间自由空间搜索: 先用形态学关门分割房间, 再在每个房间内独立
    搜索 Top-K FreeSpace 候选。保证候选跨房间分布, 避免全挤在同一区域。
    返回 (rooms_info, per_room_candidates):
      rooms_info: [(cx, cy, bbox, area_m2), ...]
      per_room_candidates: 扁平化后的 (score, x, y, deg) 列表, 按 score 降序
    """
    rooms = find_rooms_freespace(map_data, info, gap_m=gap_m, min_area_m2=min_area_m2)
    if len(rooms) <= 1:
        # 单房间 (或未分割成功), 回退到全图搜索
        return rooms, global_search_freespace(pts, map_data, info, top_k=top_k,
                                              xy_step=xy_step, angle_step_deg=angle_step_deg)
    # 多房间: 每房间独立搜索并各取 Top-1, 不足 top_k 时用全图搜索补足
    all_cands = []
    seen_poses = set()
    for cx, cy, bx0, by0, bx1, by1, area_m2 in rooms:
        room_cands = global_search_freespace(
            pts, map_data, info, top_k=1,
            xy_step=xy_step, angle_step_deg=angle_step_deg,
            search_bbox=(bx0, by0, bx1, by1))
        for cand in room_cands:
            key = (round(cand[1], 1), round(cand[2], 1))
            if key not in seen_poses:
                seen_poses.add(key)
                all_cands.append(cand)
        if room_cands:
            print(f"  [RoomSearch] room({cx:.1f},{cy:.1f}) {area_m2:.0f}m2 "
                  f"→ best ({room_cands[0][1]:.1f},{room_cands[0][2]:.1f},{room_cands[0][3]:.0f}deg) "
                  f"fs={room_cands[0][0]:.3f}")
    if len(all_cands) < top_k:
        # 补足: 全图 NMS 后取额外候选
        full = global_search_freespace(pts, map_data, info, top_k=top_k,
                                       xy_step=xy_step, angle_step_deg=angle_step_deg)
        for cand in full:
            key = (round(cand[1], 1), round(cand[2], 1))
            if key not in seen_poses:
                seen_poses.add(key)
                all_cands.append(cand)
            if len(all_cands) >= top_k:
                break
    all_cands.sort(key=lambda c: c[0], reverse=True)
    return rooms, all_cands[:top_k]


# ============================================================
# 单候选完整递推 (复刻 MultiStepLocalizer 的逐帧逻辑, 种子由外部给定)
# ============================================================
def recurse_from_seed(frames, seed_pose, ms, lf, map_data, info, decay=0.7, min_sigma=0.5,
                      constrained=True, verbose=False, best_score_ref=None,
                      early_exit_gap=0.15, early_exit_frame=10):
    """
    从指定首帧种子位姿出发, 逐帧 ICP + 局部搜索递推。

    constrained=True → 使用 local_search_constrained (墙穿惩罚),
                       防止轨迹滑入几何相似但错误的房间。数据4这类多房间场景必备。
    verbose → 逐帧打印 ICP/local_search/reject 详情。

    返回:
      final_pose (x,y,yaw), history[(i,x,y,yaw,sigma,wall%,score)],
      mean_score, mean_wall, unknown_pct(终态合并点落在未知区域比例), early_exit (bool)
    """
    mu = list(seed_pose)
    sigma = 5.0
    sigma_angle = 20.0
    history = []
    early_exit = False

    for i in range(len(frames)):
        pts, tf = frames[i]
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

            # 自适应搜索粒度 (配合两级搜索, 粗搜大步长+细搜精修):
            #   sigma大 → 粗粒度, 让两级搜索的"粗搜阶段"覆盖更大空间
            #   sigma小 → 细粒度, 粗/细两级都收窄
            if sigma > 2.0:
                ps, a_s = 0.25, 4   # 粗搜: 0.5m/8° → 细搜: 0.25m/4°
            elif sigma > 1.0:
                ps, a_s = 0.2, 3    # 粗搜: 0.5m/6° → 细搜: 0.2m/3°
            else:
                ps, a_s = 0.15, 2   # 粗搜: 0.5m/4° → 细搜: 0.15m/2°

            if constrained:
                pose, _ = local_search_constrained(pts, pred_x, pred_y, pred_yaw, lf,
                                                   map_data, info, ms,
                                                   radius=search_radius, angle_range=int(angle_range),
                                                   pos_step=ps, angle_step=a_s,
                                                   sensor_odom_x=tf[0], sensor_odom_y=tf[1])
            else:
                pose, _ = ms.local_search(pts, pred_x, pred_y, pred_yaw, lf, info,
                                          radius=search_radius, angle_range=int(angle_range),
                                          pos_step=ps, angle_step=a_s)

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

        # 早期终止: 若当前候选明显劣于已知最优, 放弃后续帧
        if best_score_ref is not None and i >= early_exit_frame:
            running_mean = float(np.mean([h[6] for h in history]))
            if running_mean < best_score_ref - early_exit_gap:
                if verbose:
                    print(f"  [EARLY EXIT] f{i:02d} running_score={running_mean:.3f} < "
                          f"best={best_score_ref:.3f} - gap={early_exit_gap:.2f}")
                early_exit = True
                while i + 1 < len(frames):
                    i += 1
                    history.append((i, mu[0], mu[1], mu[2], sigma, wp, sc))
                break

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

    return tuple(mu), history, mean_score, mean_wall, unknown_pct, early_exit


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


# ═══════════════════════════════════════════════════════════════
# 去灰 + 面积/线条对比可视化
# ═══════════════════════════════════════════════════════════════

def _classify_scan_zones(pts_map, map_data, info):
    """将扫描点分类到 free/wall/unknown 三个区域，返回每类的 mask 和计数。"""
    res, ox, oy = info['resolution'], info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    ci = ((pts_map[:, 0] - ox) / res + 0.5).astype(np.int32)
    ri = ((pts_map[:, 1] - oy) / res + 0.5).astype(np.int32)
    valid = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
    zones = np.full(len(pts_map), -2, dtype=np.int8)  # -2=超出边界
    zones[valid] = map_data[ri[valid], ci[valid]]
    free_mask = zones == 0
    wall_mask = zones == 100
    unknown_mask = zones == -1
    oob_mask = zones == -2
    return free_mask, wall_mask, unknown_mask, oob_mask


def _extract_wall_contours(map_data, info, min_wall_area=200):
    """从地图 occupied 区域提取墙壁轮廓线。返回轮廓列表 (像素坐标)。"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    wall_bin = (map_data == 100).astype(np.uint8) * 255
    if wall_bin.sum() == 0:
        return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    wall_bin = cv2.morphologyEx(wall_bin, cv2.MORPH_CLOSE, k, iterations=1)
    contours, _ = cv2.findContours(wall_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in contours:
        if cv2.contourArea(c) < min_wall_area:
            continue
        # 像素 → 地图坐标 (origin='lower' 对应 numpy row 反转)
        pts = c.reshape(-1, 2).astype(np.float64)
        mx = ox + (pts[:, 0] + 0.5) * res
        my = oy + (H - pts[:, 1] - 0.5) * res
        kept.append(np.column_stack([mx, my]))
    return kept


def _extract_scan_boundary(pts_map, info, grid_cells=20):
    """将扫描端点栅格化后提取外轮廓，返回轮廓点 (地图坐标)。"""
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']

    if len(pts_map) < 10:
        return None

    # 创建比地图稍大的局部网格
    xs_min, xs_max = pts_map[:, 0].min() - 2.0, pts_map[:, 0].max() + 2.0
    ys_min, ys_max = pts_map[:, 1].min() - 2.0, pts_map[:, 1].max() + 2.0
    gx = int((xs_max - xs_min) / res) + 1
    gy = int((ys_max - ys_min) / res) + 1

    # 栅格化
    grid = np.zeros((gy, gx), dtype=np.uint8)
    ci = ((pts_map[:, 0] - xs_min) / res + 0.5).astype(np.int32)
    ri = (gy - 1 - (pts_map[:, 1] - ys_min) / res + 0.5).astype(np.int32)
    for cx, ry in zip(ci, ri):
        if 0 <= cx < gx and 0 <= ry < gy:
            grid[ry, cx] = 255

    if grid.sum() == 0:
        return None

    # 膨胀 + 找轮廓
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grid_cells, grid_cells))
    grid_d = cv2.dilate(grid, k, iterations=1)
    cs, _ = cv2.findContours(grid_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    pts = c.reshape(-1, 2).astype(np.float64)
    # 像素 → 地图坐标
    mx = xs_min + (pts[:, 0] + 0.5) * res
    my = ys_min + (gy - pts[:, 1] - 0.5) * res
    return np.column_stack([mx, my])


def create_no_gray_comparison(map_data, info, frame_pts, best, tf_gt, output_path):
    """
    去灰 + 面积/线条对比可视化 (3-panel):
      (a) 去灰地图 + 扫描按区域着色 (绿=自由区, 红=墙, 蓝=未知)
      (b) 面积统计: 扫描点落到各区域的比例 + 条形图
      (c) 边界线条对比: 地图墙体轮廓 (黑) vs 扫描外轮廓 (红虚线)
    """
    res = info['resolution']
    ox, oy = info['origin_x'], info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox + W * res, oy, oy + H * res]

    final_pose, history = best['final'], best['history']
    fx, fy, fyaw = final_pose

    # ── 汇总所有帧扫描点 (best 位姿) ──
    red_pts = np.vstack([project_pts(frame_pts[h[0]][0], h[1], h[2], h[3]) for h in history])
    # 降采样
    step = max(1, len(red_pts) // 8000)
    red_pts_sub = red_pts[::step]

    # ── 去灰地图 RGB ──
    map_nogray = np.full((H, W, 3), 1.0, dtype=np.float32)       # 白色底
    map_nogray[map_data == 0] = [0.92, 0.92, 0.92]               # 自由区 浅灰
    map_nogray[map_data == 100] = [0.15, 0.15, 0.15]             # 墙 深灰

    # ── 分类扫描点 ──
    free_m, wall_m, unk_m, oob_m = _classify_scan_zones(red_pts_sub, map_data, info)
    nb, nf, nw, nu = len(red_pts_sub), int(free_m.sum()), int(wall_m.sum()), int(unk_m.sum())
    noob = int(oob_m.sum())

    # ── 提取轮廓 ──
    wall_contours = _extract_wall_contours(map_data, info)
    scan_boundary = _extract_scan_boundary(red_pts_sub, info, grid_cells=15)

    # ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    # ────────────────────────────────────────

    # (a) 去灰地图 + 分类着色扫描
    ax = axes[0]
    ax.imshow(map_nogray, origin='lower', extent=extent, aspect='equal')
    # 按区域着色扫描点
    if nf > 0:
        ax.scatter(red_pts_sub[free_m, 0], red_pts_sub[free_m, 1],
                   s=0.8, c='#2ecc71', alpha=0.7, label=f'自由区 ({nf})')
    if nw > 0:
        ax.scatter(red_pts_sub[wall_m, 0], red_pts_sub[wall_m, 1],
                   s=0.8, c='#e74c3c', alpha=0.7, label=f'墙壁 ({nw})')
    if nu > 0:
        ax.scatter(red_pts_sub[unk_m, 0], red_pts_sub[unk_m, 1],
                   s=0.8, c='#3498db', alpha=0.7, label=f'未知 ({nu})')
    ax.plot(fx, fy, 'r*', markersize=16, zorder=6)
    ax.arrow(fx, fy, 2.5 * math.cos(fyaw), 2.5 * math.sin(fyaw),
             head_width=0.5, head_length=0.4, fc='red', ec='darkred', lw=2.5)
    ax.set_title(f'(a) 去灰地图 + 扫描分区着色\n'
                 f'位姿 ({fx:.1f}, {fy:.1f}, {math.degrees(fyaw):.0f}°)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right', markerscale=5)
    ax.grid(True, alpha=0.10)

    # (b) 面积统计
    ax = axes[1]
    zones = ['自由区', '墙壁', '未知区', '出界']
    counts = [nf, nw, nu, noob]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    bars = ax.bar(zones, counts, color=colors, edgecolor='white', linewidth=0.5, width=0.55)
    for bar, cnt in zip(bars, counts):
        pct = 100.0 * cnt / max(nb, 1)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylabel('扫描点数', fontsize=10)
    ax.set_title(f'(b) 面积统计 (总 {nb} 点)\n'
                 f'已知区覆盖率={(nf + nw) / max(nb, 1) * 100:.1f}%  未知率={nu / max(nb, 1) * 100:.1f}%',
                 fontsize=11)
    ax.grid(axis='y', alpha=0.15)

    # (c) 边界线条对比
    ax = axes[2]
    ax.imshow(map_nogray, origin='lower', extent=extent, aspect='equal')
    # 地图墙壁轮廓
    if wall_contours:
        for wc in wall_contours:
            ax.plot(wc[:, 0], wc[:, 1], color='black', lw=0.8, alpha=0.6, zorder=3)
        # 首个轮廓用于图例
        ax.plot([], [], color='black', lw=2.0, label=f'地图墙轮廓 ({len(wall_contours)}条)')
    # 扫描外轮廓
    if scan_boundary is not None and len(scan_boundary) > 0:
        ax.plot(scan_boundary[:, 0], scan_boundary[:, 1],
                color='#e74c3c', lw=2.0, ls='--', alpha=0.85, label='扫描外轮廓', zorder=4)
        # 填充扫描外轮廓区域
        ax.fill(scan_boundary[:, 0], scan_boundary[:, 1],
                color='red', alpha=0.06, zorder=2)
    ax.plot(fx, fy, 'r*', markersize=16, zorder=6)
    ax.arrow(fx, fy, 2.5 * math.cos(fyaw), 2.5 * math.sin(fyaw),
             head_width=0.5, head_length=0.4, fc='red', ec='darkred', lw=2.5)
    ax.set_title(f'(c) 边界线条对比\n'
                 f'map墙轮廓(黑) vs 扫描外轮廓(红虚线)', fontsize=11)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"去灰对比图已保存: {output_path}")


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
    parser.add_argument('--compare-viz', action='store_true',
                        help='额外生成去灰 + 面积/线条对比图')
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

    # ── 似然场失败时的回退策略 ──
    #   RayCast<30% → 似然场找到的位姿与地图无有效匹配。
    #   原因有二: (a) 多房间几何相似 → WallHit 墙壁命中搜索
    #            (b) 开放空间/地图覆盖不足 → FreeSpace 自由空间最大化搜索
    #   优先尝试 FreeSpace (data_4 场景), 若也失败再降级到 WallHit。
    do_wallhit = False
    do_freespace = False
    rc_valid_rate = 0.0

    if args.skip_wallhit:
        print(f"  [回退] --skip-wallhit 指定, 跳过替代搜索")
    elif args.force_wallhit:
        print(f"  [回退] --force-wallhit 指定, 强制执行墙壁搜索")
        do_wallhit = True
    else:
        best_rc = ms.score_pose_raycast(pts0, cand0[0][1], cand0[0][2],
                                         math.radians(cand0[0][3]), map_data, info)
        rc_valid_rate = best_rc[2] / max(best_rc[1], 1) if best_rc[0] > -1e8 else 0
        print(f"  [诊断] 似然首选射线投射有效率={rc_valid_rate:.0%}")
        if rc_valid_rate < 0.30:
            print(f"  [回退] 似然场不可信, 优先尝试自由空间最大化搜索...")
            do_freespace = True
        else:
            print(f"  [回退] raycast有效={rc_valid_rate:.0%}≥30%, 信任似然场首帧结果")

    # ── 方案A: 自由空间最大化 (data_4 开放空间场景) ──
    if do_freespace:
        # 房间感知搜索: 形态学关门分割房间, 逐房间分配候选, 保证跨区域覆盖
        rooms, room_cands = global_search_freespace_per_room(
            pts0, map_data, info, top_k=args.topk)
        fs_candidates = room_cands  # use room-scoped candidates directly
        if fs_candidates and fs_candidates[0][0] > 0.3:
            cand0 = fs_candidates
            best_fs = fs_candidates[0]
            if len(rooms) > 1:
                print(f"  [FreeSpace+Room] {len(rooms)}个房间 → 逐房间分发 {len(fs_candidates)} 个候选")
            print(f"  [FreeSpace] 替换为 {len(fs_candidates)} 个自由空间候选, 最优"
                  f"({best_fs[1]:.1f},{best_fs[2]:.1f},{best_fs[3]:.0f}°) "
                  f"自由率={best_fs[0]:.2f}")
        else:
            fs_score = fs_candidates[0][0] if fs_candidates else 0
            print(f"  [FreeSpace] 无高置信候选 (最高分={fs_score:.2f}), 降级到墙壁搜索...")
            do_wallhit = True

    # ── WallHit 墙壁命中回退 (多房间几何消歧) ──
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
    best_score_so_far = None
    for k, (sc0, hx, hy, had) in enumerate(cand0):
        print(f"\n  ▸ 候选 #{k} / {len(cand0)}  种子({hx:.1f}, {hy:.1f}, {had:.0f}°)")
        # 候选种子精搜细化
        seed, _ = ms.local_search(pts0, hx, hy, math.radians(had), lf, info,
                                  radius=2.0, pos_step=0.3)
        # 从第2个候选起, 传入已知最佳分用于早期终止
        ref = best_score_so_far if best_score_so_far is not None else None
        final, hist, mscore, mwall, unk, early_exit = recurse_from_seed(
            frame_pts, seed, ms, lf, map_data, info, decay=args.decay,
            constrained=args.constrained, verbose=args.verbose,
            best_score_ref=ref)
        if early_exit:
            print(f"    ⚠ 候选#{k} 跑分过低 → 早期终止 (均分={mscore:.2f})")

        # 更新最佳跑分 (用于后续候选早期终止判定)
        if best_score_so_far is None or mscore > best_score_so_far:
            best_score_so_far = mscore

        # 验证 (早期终止的候选跳过耗时的验证 + wallhit)
        if not early_exit:
            gray_end, wall_free, sensor_nf = geometry_constraints(final, frame_ranges, frame_tfs,
                                                                  angle_min, angle_inc, map_data, info)
            rc_err = raycast_validate(final, frame_ranges, frame_tfs, angle_min, angle_inc,
                                      map_data, info, hu)
            # wallhit 降采样: 每4个点取1个, 速度提升4x, 评分影响极小
            all_pts = np.vstack([fp[0] for fp in frame_pts if len(fp[0]) > 0])
            wh_score, wh_nwall, wh_nv = ms.score_pose_wallhit(
                all_pts[::4], final[0], final[1], final[2], map_data, info)
        else:
            gray_end = 50.0; wall_free = 50.0; sensor_nf = 50.0; rc_err = 1e9
            wh_score = -1e9  # 被淘汰候选无需墙命中评分
            wh_nwall = 0; wh_nv = 0
        results.append({'final': final, 'history': hist, 'mean_score': mscore,
                        'mean_wall': mwall, 'unknown_pct': unk, 'raycast': rc_err,
                        'gray_end': gray_end, 'wall_free': wall_free,
                        'sensor_nf': sensor_nf, 'wallhit': max(wh_score, 0.0),
                        'early_exit': early_exit})
        early_tag = " [EARLY EXIT]" if early_exit else ""
        print(f"  候选#{k}: 种子({hx:.1f},{hy:.1f},{had:.0f}°) → 终态"
              f"({final[0]:.1f},{final[1]:.1f},{math.degrees(final[2]):.0f}°) "
              f"| 墙命中={max(wh_score,0):.2f} 传感器非自由={sensor_nf:.0f}% "
              f"端点灰={gray_end:.0f}% 墙穿={wall_free:.1f}% 均分={mscore:.2f}{early_tag}")

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
        # 早期终止的候选降为无效 (跑分显著低于最优候选)
        r['early_ok'] = not r.get('early_exit', False)
        r['valid']    = r['sensor_ok'] and r['early_ok']
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
        trust = "可信" if r['valid'] else ("淘汰(早期终止)" if not r.get('early_ok', True) else "淘汰(传感器在墙里)")
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

    if args.compare_viz:
        cmp_out = output_path.replace('.png', '_compare.png')
        create_no_gray_comparison(map_data, info, frame_pts, best, tf_gt, cmp_out)

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
