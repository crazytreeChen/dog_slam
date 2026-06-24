#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
match_scan_to_map.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
图像级扫描-地图形状匹配：支持翻转、缩放、旋转、平移。

算法流程:
  1. 加载 map.png / scan.png → 灰度图
  2. 边缘提取 (Canny) → 距离变换 (Chamfer 匹配)
  3. 粗搜索: 多尺度 × 多角度 × 多翻转 → 找到 Top-K 候选
  4. 精搜索: 在粗搜最优附近细化尺度/角度/位置
  5. 可视化: 匹配结果叠加到地图

用法:
  python match_scan_to_map.py --map map.png --scan scan.png
  python match_scan_to_map.py --map map.png --scan scan.png --scale-range 0.3,3.0 --angle-step 10
  python match_scan_to_map.py --map map.png --scan scan.png --output result.png --show-flip
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
    print("[ERROR] Need opencv-python: pip install opencv-python"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] Need matplotlib: pip install matplotlib"); sys.exit(1)


# ============================================================
# 1. 图像加载与预处理
# ============================================================
def load_image(path, max_size=1200):
    """加载图像为灰度图，可选缩放以加速。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    # 若图太大则降采样加速
    h, w = img.shape
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"  [Resize] {w}x{h} → {new_w}x{new_h}")

    return img


def binarize_map(img, block_size=51, c_const=5):
    """
    将地图灰度图转为「边框+空白」的二值结构图。

    策略:
      1. 自适应阈值 → 分割出障碍物(黑)和自由空间(白)
      2. 形态学闭运算 → 填充障碍物内部细小空洞
      3. 仅保留障碍物边缘 (腐蚀-原图差) → 只留边框
      4. 与自由空间掩膜合并 → 保留空白区域信息

    返回:
      struct_img: 0=边框, 128=未知(灰), 255=空白(自由空间)
      border_mask: 纯边框二值图 (255=边框)
    """
    h, w = img.shape

    # ── 1. 自适应阈值: 分割障碍物 vs 自由空间 ──
    # block_size 必须为奇数
    bs = block_size if block_size % 2 == 1 else block_size + 1
    binary = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, bs, c_const)
    # binary: 255=障碍物, 0=自由空间

    # ── 2. 形态学去噪 ──
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 闭运算: 先膨胀再腐蚀，填充障碍物内部小空洞
    obstacles = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_med, iterations=1)
    # 开运算: 去除细小噪点
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # ── 3. 提取障碍物边框 (膨胀-腐蚀) ──
    dilated = cv2.dilate(obstacles, kernel_med, iterations=2)
    eroded = cv2.erode(obstacles, kernel_med, iterations=2)
    borders = dilated - eroded  # 边框带 (255=边框, 0=非边框)

    # ── 4. 识别自由空间 (原图中亮度较高的区域) ──
    # 地图中: 白色=自由空间(~255), 灰色=未知(~128), 黑色=障碍物(~0)
    _, free_space = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # ── 5. 构建结构化图像 ──
    struct_img = np.full((h, w), 128, dtype=np.uint8)  # 默认: 未知/灰色
    struct_img[borders > 0] = 0      # 边框=黑
    struct_img[free_space > 0] = 255  # 自由空间=白

    print(f"  [Binarize] obstacles={int(np.sum(obstacles>0))}px, "
          f"borders={int(np.sum(borders>0))}px, "
          f"free={int(np.sum(free_space>0))}px")

    return struct_img, borders


def extract_edges(img, low=50, high=150):
    """Canny 边缘提取，返回二值边缘图 (255=边缘, 0=背景)。"""
    # 先高斯模糊去噪
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
    edges = cv2.Canny(blurred, low, high)
    return edges


def clean_scan_edges(edges, dilate_iter=2, min_length=200):
    """
    清理扫描的边缘碎片: 点云投影图像经 Canny 后会产生大量离散短线段,
    通过膨胀连接 + 连通域过滤保留主要轮廓。

    参数:
      dilate_iter: 膨胀迭代次数 (连接近邻碎片)
      min_length: 最小轮廓线长度 (像素), 短于此值的丢弃

    返回: 过滤后的二值边缘图
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # 1. 膨胀连接近邻边缘碎片
    dilated = cv2.dilate(edges, kernel, iterations=dilate_iter)

    # 2. 找连通域, 只保留像素数最多的几个连通域
    #    点云轮廓应该形成一个大的连通结构
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8)

    if num_labels <= 1:
        return edges  # 没有有效连通域

    # 按面积排序 (跳过背景 label=0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_idx = np.argsort(areas)[::-1] + 1  # +1 回到原始 label

    # 保留面积 > min_length 的连通域
    keep_labels = set()
    for idx in sorted_idx:
        if areas[idx - 1] >= min_length:
            keep_labels.add(idx)

    if not keep_labels:
        # 至少保留最大的一个
        keep_labels.add(sorted_idx[0])

    mask = np.isin(labels, list(keep_labels))

    # 3. 用膨胀后的 mask 作用回原始边缘
    cleaned = edges.copy()
    cleaned[~mask] = 0

    print(f"  [CleanScan] {num_labels - 1} components → {len(keep_labels)} kept "
          f"(threshold={min_length}px, dilate={dilate_iter})")

    return cleaned


def extract_edges_from_binary(border_mask):
    """
    从二值化的边框掩膜直接构建边缘图（无需 Canny）。
    边框本身就是边缘，只需形态学细化确保单像素宽。
    """
    # 骨架化/细化: 多次腐蚀+重建
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skel = np.zeros(border_mask.shape, dtype=np.uint8)
    temp = border_mask.copy()
    while np.sum(temp) > 0:
        eroded = cv2.erode(temp, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, temp - opened)
        temp = eroded.copy()
    return skel


def build_distance_field(edges, max_dist=80):
    """从边缘图构建距离场（用于快速 Chamfer 匹配评分）。"""
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.clip(dist, 0, max_dist).astype(np.float32)


# ============================================================
# 2. 仿射变换矩阵构建
# ============================================================
def build_transform_matrix(scale, angle_deg, flip_h, flip_v,
                           src_h, src_w, dst_h, dst_w, tx=0, ty=0):
    """
    构建从模板(scan)到搜索图像(map)的透视变换矩阵。

    操作顺序: flip → scale → rotate → translate
    返回 2x3 仿射变换矩阵。
    """
    center = (src_w / 2.0, src_h / 2.0)

    # 旋转+缩放矩阵 (以模板中心为原点)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    # 翻转
    if flip_h:
        M[0, 0] *= -1   # x 方向镜像
        M[0, 1] *= -1
        M[0, 2] = src_w - 1 - M[0, 2]
    if flip_v:
        M[1, 0] *= -1   # y 方向镜像
        M[1, 1] *= -1
        M[1, 2] = src_h - 1 - M[1, 2]

    # 平移
    M[0, 2] += tx
    M[1, 2] += ty

    return M


def warp_template(template, M, dsize):
    """用仿射变换矩阵变换模板图像。"""
    return cv2.warpAffine(template, M, dsize,
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


# ============================================================
# 3. Chamfer 匹配评分
# ============================================================
def chamfer_score(scan_warped_edges, map_dist_field):
    """
    计算变换后 scan 边缘在 map 距离场上的 Chamfer 距离。
    返回值越低表示匹配越好。

    使用距离场直接采样，避免逐像素循环。
    """
    # 只统计 scan 中的边缘像素 (值 > 0)
    mask = scan_warped_edges > 0
    if np.sum(mask) < 20:
        return 1e9

    mean_dist = np.mean(map_dist_field[mask])
    n_edge_pixels = int(np.sum(mask))

    # 归一化：除以边缘点数，避免大面积模板天然占优
    # 降低大模板偏好, 加入轻微尺寸惩罚
    return mean_dist - 0.0001 * n_edge_pixels


def masked_ncc(scan_warped, map_region, mask):
    """在有效 mask 区域内计算归一化互相关 (NCC)。"""
    if np.sum(mask) < 50:
        return -1.0

    s = scan_warped[mask].astype(np.float64)
    m = map_region[mask].astype(np.float64)

    s_mean, m_mean = np.mean(s), np.mean(m)
    s_std, m_std = np.std(s), np.std(m)

    if s_std < 1e-6 or m_std < 1e-6:
        return 0.0

    return float(np.mean((s - s_mean) * (m - m_mean)) / (s_std * m_std))


# ============================================================
# 4. 粗搜索：多尺度 × 多角度 × 多翻转 × 卷积加速平移
# ============================================================
def coarse_search(scan_edges, map_edges, map_dist,
                  scales, angles_deg, flips,
                  step=16, top_k=5, use_ncc=True,
                  fast_mode=False):
    """
    粗搜索: 遍历所有 scale × angle × flip 组合, 用滑动窗口卷积一次性计算全图 Chamfer 得分。

    核心加速: 用 cv2.filter2D 替代逐像素循环, 时间复杂度从 O(N*M*W*H) 降到 O(N*W*H),
    其中 N=scale×angle×flip, W×H=地图尺寸。

    参数:
      step: 平移搜索步长 (像素), 在得分图上以 step 步长采样局部最优。
      top_k: 保留的最优候选数。
      fast_mode: True 时跳过 NCC 且扩大 step。

    返回: [(score, scale, angle, flip_h, flip_v, tx, ty), ...]
    """
    sh, sw = scan_edges.shape
    mh, mw = map_edges.shape

    total = len(scales) * len(angles_deg) * len(flips)
    candidates = []
    count = 0
    t_start = time.time()

    for scale in scales:
        for ang in angles_deg:
            for fh, fv in flips:
                count += 1
                if count % 5 == 0 or count == 1:
                    elapsed = time.time() - t_start
                    eta = elapsed / count * (total - count) if count > 0 else 0
                    print(f"  [Coarse] {count}/{total} | scale={scale:.2f} ang={ang}° "
                          f"flip=({'H' if fh else '-'}{'V' if fv else '-'}) | "
                          f"elapsed={elapsed:.0f}s ETA={eta:.0f}s")

                # 构建变换矩阵 (无平移)
                M0 = build_transform_matrix(
                    scale, ang, fh, fv, sh, sw, mh, mw, tx=0, ty=0)

                # 计算变换后模板的边界框
                corners = np.array([
                    [0, 0, 1], [sw, 0, 1], [0, sh, 1], [sw, sh, 1]
                ], dtype=np.float64).T
                warped_corners = (M0 @ corners).T
                min_x = max(0, int(np.floor(np.min(warped_corners[:, 0]))))
                min_y = max(0, int(np.floor(np.min(warped_corners[:, 1]))))
                max_x = min(mw - 1, int(np.ceil(np.max(warped_corners[:, 0]))))
                max_y = min(mh - 1, int(np.ceil(np.max(warped_corners[:, 1]))))

                bbox_w = max_x - min_x
                bbox_h = max_y - min_y

                if bbox_w < 15 or bbox_h < 15:
                    continue

                # 预 warp 模板 (局部坐标, 以 bbox 原点为参考)
                M_local = build_transform_matrix(
                    scale, ang, fh, fv, sh, sw, bbox_h, bbox_w, tx=-min_x, ty=-min_y)
                scan_warped = warp_template(scan_edges, M_local, (bbox_w, bbox_h))

                n_edge = np.sum(scan_warped > 0)
                if n_edge < 15:
                    continue

                # ── 卷积加速: 一次性计算所有位置的 Chamfer 得分 ──
                # 模板归一化: 除以边缘像素数得到均值
                template = scan_warped.astype(np.float32)
                template_mask = (template > 0).astype(np.float32)
                template_norm = template_mask / max(n_edge, 1)

                # 用 filter2D 计算每个位置的均值距离
                # score_map[x,y] = sum(template_norm * map_dist_roi) at offset (x,y)
                score_map = cv2.filter2D(
                    map_dist, cv2.CV_32F, template_norm,
                    anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)

                # 有效区域: 只取 bbox 完全在 map 内的位置
                valid_h = mh - bbox_h + 1
                valid_w = mw - bbox_w + 1
                if valid_h <= 0 or valid_w <= 0:
                    continue
                score_roi = score_map[:valid_h, :valid_w]

                # ── 以 step 步长采样局部最优 ──
                for ty in range(0, valid_h, step):
                    for tx in range(0, valid_w, step):
                        # 取 step×step 窗口内的最小值
                        y2 = min(ty + step, valid_h)
                        x2 = min(tx + step, valid_w)
                        local_patch = score_roi[ty:y2, tx:x2]
                        if local_patch.size == 0:
                            continue

                        min_val = float(np.min(local_patch))
                        min_idx = np.unravel_index(np.argmin(local_patch), local_patch.shape)
                        best_ty = ty + int(min_idx[0])
                        best_tx = tx + int(min_idx[1])

                        # NCC 辅助验证 (可选)
                        if use_ncc and not fast_mode:
                            roi_edges = map_edges[best_ty:best_ty + bbox_h,
                                                  best_tx:best_tx + bbox_w]
                            ncc = masked_ncc(scan_warped, roi_edges, template_mask > 0)
                            combined = min_val * 0.7 - ncc * 1.5
                        else:
                            combined = min_val

                        # 全局坐标
                        gx = best_tx + min_x
                        gy = best_ty + min_y

                        # 维持 top candidates
                        if len(candidates) < top_k * 3:
                            candidates.append((
                                combined, float(min_val), scale, ang, fh, fv, gx, gy))
                        else:
                            worst_idx = max(range(len(candidates)),
                                            key=lambda i: candidates[i][0])
                            if combined < candidates[worst_idx][0]:
                                candidates[worst_idx] = (
                                    combined, float(min_val), scale, ang, fh, fv, gx, gy)

    # 排序 + NMS 去重
    candidates.sort(key=lambda x: x[0])

    nms = []
    for c in candidates:
        _, _, _, ang_c, _, _, cx, cy = c
        is_dup = False
        for _, _, _, ang_n, _, _, nx, ny in nms:
            if math.hypot(cx - nx, cy - ny) < 50 and abs(ang_c - ang_n) < 20:
                is_dup = True
                break
        if not is_dup:
            nms.append(c)
            if len(nms) >= top_k:
                break

    print(f"  [Coarse] Done: {len(nms)} candidates after NMS, "
          f"total {time.time()-t_start:.1f}s")
    for i, (sc, ch, scale, ang, fh, fv, tx, ty) in enumerate(nms):
        print(f"    #{i}: score={sc:.3f} chamfer={ch:.2f} scale={scale:.2f} "
              f"ang={ang}° flip=({'H' if fh else '-'}{'V' if fv else '-'}) "
              f"pos=({tx:.0f},{ty:.0f})")

    return nms if nms else candidates[:top_k]


# ============================================================
# 5. 精搜索：在粗搜最优附近细化
# ============================================================
def fine_search(scan_edges, map_edges, map_dist,
                init_scale, init_ang, init_fh, init_fv, init_tx, init_ty,
                scale_range=0.15, scale_step=0.02,
                ang_range=15.0, ang_step=1.0,
                pos_range=50, pos_step=1,
                use_ncc=True, fast_mode=False):
    """
    在粗搜最优姿态附近卷积加速精细搜索。

    用 filter2D 一次性计算搜索窗口内所有位置的 Chamfer 得分,
    然后选全局最低点。O(combos × W×H) 而非 O(combos × positions × T²)。

    返回: (best_score, best_chamfer, best_scale, best_ang,
            best_fh, best_fv, best_tx, best_ty)
    """
    sh, sw = scan_edges.shape
    mh, mw = map_edges.shape

    # fast 模式用更大步长
    _scale_step = max(scale_step, 0.05) if fast_mode else scale_step
    _ang_step = max(ang_step, 3.0) if fast_mode else ang_step
    scales = np.arange(
        max(0.2, init_scale - scale_range),
        init_scale + scale_range + 1e-6,
        _scale_step)
    angles = np.arange(init_ang - ang_range, init_ang + ang_range + 1e-6, _ang_step)

    best = (1e9, 1e9, init_scale, init_ang, init_fh, init_fv, init_tx, init_ty)

    total = len(scales) * len(angles)
    count = 0
    t0 = time.time()

    for scale in scales:
        for ang in angles:
            count += 1
            if count % 30 == 0 or count == 1:
                print(f"  [Fine] {count}/{total}... elapsed={time.time()-t0:.1f}s")

            M0 = build_transform_matrix(
                scale, ang, init_fh, init_fv, sh, sw, mh, mw)

            corners = np.array([
                [0, 0, 1], [sw, 0, 1], [0, sh, 1], [sw, sh, 1]
            ], dtype=np.float64).T
            warped_corners = (M0 @ corners).T
            min_x = int(np.floor(np.min(warped_corners[:, 0])))
            min_y = int(np.floor(np.min(warped_corners[:, 1])))
            max_x = int(np.ceil(np.max(warped_corners[:, 0])))
            max_y = int(np.ceil(np.max(warped_corners[:, 1])))
            bbox_w = max_x - min_x
            bbox_h = max_y - min_y

            if bbox_w < 15 or bbox_h < 15:
                continue

            M_local = build_transform_matrix(
                scale, ang, init_fh, init_fv, sh, sw,
                bbox_h, bbox_w, tx=-min_x, ty=-min_y)
            scan_warped = warp_template(scan_edges, M_local, (bbox_w, bbox_h))

            n_edge = np.sum(scan_warped > 0)
            if n_edge < 15:
                continue

            # ── 卷积加速: filter2D 一次性算全图 ──
            template_mask = (scan_warped > 0).astype(np.float32)
            template_norm = template_mask / max(n_edge, 1)
            score_map = cv2.filter2D(
                map_dist, cv2.CV_32F, template_norm,
                anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)

            # ── 只在粗搜结果附近搜索 ──
            # 确定搜索窗口在 score_map 中的范围
            win_y1 = max(0, init_ty - pos_range - min_y)
            win_x1 = max(0, init_tx - pos_range - min_x)
            win_y2 = min(mh - bbox_h, init_ty + pos_range - min_y)
            win_x2 = min(mw - bbox_w, init_tx + pos_range - min_x)

            if win_y2 <= win_y1 or win_x2 <= win_x1:
                continue

            local_scores = score_map[win_y1:win_y2 + 1, win_x1:win_x2 + 1]
            if local_scores.size == 0:
                continue

            min_val = float(np.min(local_scores))
            min_idx = np.unravel_index(np.argmin(local_scores), local_scores.shape)
            best_ty = win_y1 + int(min_idx[0])
            best_tx = win_x1 + int(min_idx[1])

            # 全局坐标
            gx = best_tx + min_x
            gy = best_ty + min_y

            if use_ncc:
                roi_edges = map_edges[best_ty:best_ty + bbox_h,
                                      best_tx:best_tx + bbox_w]
                ncc = masked_ncc(scan_warped, roi_edges, template_mask > 0)
                combined = min_val * 0.7 - ncc * 1.5
            else:
                combined = min_val

            if combined < best[0]:
                best = (combined, min_val, scale, ang, init_fh, init_fv, gx, gy)

    print(f"  [Fine] Done: {time.time()-t0:.1f}s, best_chamfer={best[1]:.3f}, "
          f"scale={best[2]:.3f} ang={best[3]:.1f}° pos=({best[6]:.0f},{best[7]:.0f})")
    return best


# ============================================================
# 6. 最终精调：亚像素配准
# ============================================================
def final_refine(scan_edges, map_edges, map_dist,
                 scale, ang, fh, fv, tx, ty, iterations=5):
    """
    在最佳姿态处做亚像素级别精调。

    每次迭代尝试 ±1px 平移, ±0.01x 缩放, ±0.5° 旋转。
    """
    sh, sw = scan_edges.shape
    mh, mw = map_edges.shape

    cur_scale, cur_ang, cur_tx, cur_ty = scale, ang, tx, ty
    best_score = 1e9

    for it in range(iterations):
        improved = False

        # 搜索邻域
        for ds in [0, -0.005, 0.005, -0.01, 0.01]:
            for da in [0, -0.5, 0.5, -1.0, 1.0]:
                for dtx in [0, -1, 1, -2, 2]:
                    for dty in [0, -1, 1, -2, 2]:
                        if ds == 0 and da == 0 and dtx == 0 and dty == 0:
                            # 基准分单独算一次
                            if it == 0:
                                continue
                            else:
                                # 非首次迭代基准值已在上轮算过
                                M0 = build_transform_matrix(
                                    cur_scale, cur_ang, fh, fv, sh, sw, mh, mw)
                                corners = np.array([
                                    [0, 0, 1], [sw, 0, 1],
                                    [0, sh, 1], [sw, sh, 1]
                                ], dtype=np.float64).T
                                wc = (M0 @ corners).T
                                mnx = int(np.floor(np.min(wc[:, 0])))
                                mny = int(np.floor(np.min(wc[:, 1])))
                                mxx = int(np.ceil(np.max(wc[:, 0])))
                                mxy = int(np.ceil(np.max(wc[:, 1])))
                                bw = mxx - mnx
                                bh = mxy - mny
                                M_local = build_transform_matrix(
                                    cur_scale, cur_ang, fh, fv, sh, sw,
                                    bh, bw, tx=-mnx, ty=-mny)

                                scan_w = warp_template(scan_edges, M_local, (bw, bh))
                                if np.sum(scan_w > 0) < 20:
                                    continue
                                try:
                                    r_dist = map_dist[cur_ty - mny:cur_ty - mny + bh,
                                                      cur_tx - mnx:cur_tx - mnx + bw]
                                    ch = chamfer_score(scan_w, r_dist)
                                    best_score = ch
                                except Exception:
                                    pass
                                continue

                        new_scale = cur_scale + ds
                        new_ang = cur_ang + da
                        new_tx = cur_tx + dtx
                        new_ty = cur_ty + dty

                        if new_scale < 0.1 or new_scale > 5.0:
                            continue

                        M0 = build_transform_matrix(
                            new_scale, new_ang, fh, fv, sh, sw, mh, mw)
                        corners = np.array([
                            [0, 0, 1], [sw, 0, 1],
                            [0, sh, 1], [sw, sh, 1]
                        ], dtype=np.float64).T
                        wc = (M0 @ corners).T
                        mnx = int(np.floor(np.min(wc[:, 0])))
                        mny = int(np.floor(np.min(wc[:, 1])))
                        mxx = int(np.ceil(np.max(wc[:, 0])))
                        mxy = int(np.ceil(np.max(wc[:, 1])))
                        bw = mxx - mnx
                        bh = mxy - mny
                        if bw < 20 or bh < 20:
                            continue

                        M_local = build_transform_matrix(
                            new_scale, new_ang, fh, fv, sh, sw,
                            bh, bw, tx=-mnx, ty=-mny if mny >= 0 else -mny)

                        scan_w = warp_template(scan_edges, M_local, (bw, bh))
                        if np.sum(scan_w > 0) < 20:
                            continue

                        try:
                            r_dist = map_dist[new_ty - mny:new_ty - mny + bh,
                                              new_tx - mnx:new_tx - mnx + bw]
                            ch = chamfer_score(scan_w, r_dist)
                        except Exception:
                            continue

                        if ch < best_score:
                            best_score = ch
                            cur_scale, cur_ang = new_scale, new_ang
                            cur_tx, cur_ty = new_tx, new_ty
                            improved = True

        if not improved:
            break

    return cur_scale, cur_ang, cur_tx, cur_ty, best_score


# ============================================================
# 7. 可视化
# ============================================================
def visualize_result(map_img, map_edges, scan_img, scan_edges,
                     scale, ang, fh, fv, tx, ty,
                     output_path,
                     top_candidates=None):
    """生成匹配结果可视化。"""
    sh, sw = scan_edges.shape
    mh, mw = map_edges.shape
    center_x = tx + sw * scale * 0.5
    center_y = ty + sh * scale * 0.5

    fig, axes = plt.subplots(2, 3, figsize=(22, 15))

    # (a) 原始 map
    ax = axes[0, 0]
    ax.imshow(map_img, cmap='gray', origin='upper')
    ax.set_title(f'(a) Map ({mw}x{mh})')
    ax.axis('off')

    # (b) 原始 scan
    ax = axes[0, 1]
    ax.imshow(scan_img, cmap='gray', origin='upper')
    ax.set_title(f'(b) Scan ({sw}x{sh})')
    ax.axis('off')

    # (c) map 边缘 + 匹配框
    ax = axes[0, 2]
    ax.imshow(map_edges, cmap='gray', origin='upper')

    # 绘制匹配框 (4个角)
    M = build_transform_matrix(scale, ang, fh, fv, sh, sw, mh, mw, tx, ty)
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float64)
    warped = cv2.transform(corners.reshape(1, -1, 2), M).reshape(-1, 2)

    for i in range(4):
        j = (i + 1) % 4
        ax.plot([warped[i, 0], warped[j, 0]],
                [warped[i, 1], warped[j, 1]],
                'r-', linewidth=2.5, zorder=10)

    flip_str = f" flip=({'H' if fh else ''}{'V' if fv else ''})" if fh or fv else ""
    ax.set_title(f'(c) Match Found\n'
                 f'pos=({tx:.0f},{ty:.0f}) scale={scale:.2f} ang={ang:.1f}°{flip_str}')
    ax.axis('off')

    # (d) 地图边缘 + scan 边缘叠加
    ax = axes[1, 0]
    # 构建完整变换
    M_full = build_transform_matrix(scale, ang, fh, fv, sh, sw, mh, mw, tx, ty)

    # 创建叠加图
    overlay = np.zeros((mh, mw, 3), dtype=np.float32)
    overlay[:, :, 1] = map_edges.astype(np.float32) / 255.0  # 地图边缘=绿色

    # scan 边缘变换后在红色通道
    scan_warped = warp_template(scan_edges, M_full, (mw, mh))
    overlay[:, :, 2] = np.maximum(overlay[:, :, 2],
                                   scan_warped.astype(np.float32) / 255.0)

    # 重合区域 = 黄色 (绿+红)
    ax.imshow(overlay, origin='upper')
    ax.set_title(f'(d) Edge Overlay\nGreen=Map Red=Scan Yellow=Match')
    ax.axis('off')

    # (e) 放大匹配区域
    ax = axes[1, 1]
    zoom = 80
    x1 = max(0, int(center_x - zoom))
    y1 = max(0, int(center_y - zoom))
    x2 = min(mw, int(center_x + zoom))
    y2 = min(mh, int(center_y + zoom))

    ax.imshow(map_img[y1:y2, x1:x2], cmap='gray',
              extent=[x1, x2, y2, y1], origin='upper')

    # 在放大区域画 scan 轮廓
    ax.plot(warped[:, 0], warped[:, 1], 'r-', linewidth=2, zorder=10)
    ax.plot(center_x, center_y, 'r+', markersize=15, mew=3)
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.set_title(f'(e) Zoomed Match (±{zoom}px)')
    ax.axis('off')

    # (f) 报告文字
    ax = axes[1, 2]
    ax.axis('off')
    flip_names = []
    if fh:
        flip_names.append("水平翻转")
    if fv:
        flip_names.append("垂直翻转")
    flip_str = " + ".join(flip_names) if flip_names else "无"

    rep = [
        "=== 匹配报告 ===", "",
        f"Map 尺寸: {mw} x {mh} px",
        f"Scan 尺寸: {sw} x {sh} px",
        "",
        "变换参数:",
        f"  缩放: {scale:.3f}x",
        f"  旋转: {ang:.1f}°",
        f"  翻转: {flip_str}",
        f"  平移: ({tx:.0f}, {ty:.0f}) px",
        f"  中心: ({center_x:.0f}, {center_y:.0f}) px",
        "",
    ]
    if top_candidates:
        rep.append(f"候选数: {len(top_candidates)}")
        for i, c in enumerate(top_candidates[:3]):
            rep.append(f"  #{i}: chamfer={c[1]:.3f} "
                       f"s={c[2]:.2f} a={c[3]:.1f}° "
                       f"pos=({c[6]:.0f},{c[7]:.0f})")

    ax.text(0.05, 0.95, "\n".join(rep), transform=ax.transAxes,
            fontfamily='monospace', fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    ax.set_title('(f) Report')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[可视化] 已保存: {output_path}")


# ============================================================
# 8. Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='扫描-地图图像匹配 (翻转/缩放/旋转/平移)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python match_scan_to_map.py --map map.png --scan scan.png
  python match_scan_to_map.py -m map.png -s scan.png --scale-range 0.3,3.0 --angle-step 8
  python match_scan_to_map.py -m map.png -s scan.png --output result.png --show-flip
  python match_scan_to_map.py -m map.png -s scan.png --coarse-step 24 --no-ncc
        """)

    parser.add_argument('--map', '-m', type=str, required=True,
                        help='地图图片路径 (map.png)')
    parser.add_argument('--scan', '-s', type=str, required=True,
                        help='扫描图片路径 (scan.png)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出可视化图片路径 (默认: 同目录 match_result.png)')
    parser.add_argument('--scale-range', type=str, default='0.7,1.4',
                        help='缩放搜索范围 min,max (默认: 0.7,1.4)')
    parser.add_argument('--angle-step', type=int, default=15,
                        help='粗搜角度步长 (度, 默认15)')
    parser.add_argument('--coarse-step', type=int, default=24,
                        help='粗搜平移步长 (像素, 默认24)')
    parser.add_argument('--show-flip', action='store_true',
                        help='启用翻转搜索 (默认仅搜索无翻转, 用此选项开启)')
    parser.add_argument('--fast', action='store_true',
                        help='快速模式: scale=0.8-1.2 angle=30 coarse=32 clean_scan 跳过精调')
    parser.add_argument('--clean-scan', action='store_true',
                        help='清理扫描边缘: 连接碎片+过滤短轮廓 (点云投影图建议开启)')
    parser.add_argument('--scan-dilate', type=int, default=2,
                        help='扫描清理膨胀次数 (默认2)')
    parser.add_argument('--scan-minlen', type=int, default=200,
                        help='扫描清理最小轮廓长度 (像素, 默认200)')
    parser.add_argument('--no-ncc', action='store_true',
                        help='禁用 NCC 辅助验证 (仅用 Chamfer 距离)')
    parser.add_argument('--max-size', type=int, default=1200,
                        help='图像最大尺寸 (像素, 默认1200, 用于加速)')
    parser.add_argument('--canny-low', type=int, default=50,
                        help='Canny 低阈值 (默认50)')
    parser.add_argument('--canny-high', type=int, default=150,
                        help='Canny 高阈值 (默认150)')
    parser.add_argument('--binarize', action='store_true',
                        help='地图去灰: 仅保留障碍物边框+空白区域, 去除内部灰色噪点')
    parser.add_argument('--bin-block', type=int, default=51,
                        help='二值化 block_size (默认51, 越大越平滑)')
    parser.add_argument('--bin-c', type=int, default=5,
                        help='二值化 C 常数 (默认5, 越大越偏白)')
    args = parser.parse_args()

    # ── 解析参数 ──
    scale_min, scale_max = map(float, args.scale_range.split(','))
    if args.fast:
        # fast 模式: 缩窄 scale 范围、大步长、跳过 NCC/精调
        scale_min = max(scale_min, 0.8)
        scale_max = min(scale_max, 1.2)
    if args.show_flip:
        flips = [(False, False), (True, False), (False, True), (True, True)]
    else:
        flips = [(False, False)]

    # fast 模式覆盖
    angle_step = 30 if args.fast else args.angle_step
    coarse_step = 32 if args.fast else args.coarse_step
    use_ncc = False if (args.fast or args.no_ncc) else True
    clean_scan = True if args.fast else args.clean_scan
    do_refine = not args.fast  # fast 模式跳过最终精调

    # ── 输出路径 ──
    if args.output:
        output_path = args.output
    else:
        scan_dir = os.path.dirname(os.path.abspath(args.scan))
        output_path = os.path.join(scan_dir or '.', 'match_result.png')

    print("=" * 60)
    print("Step 1: 加载图像")
    print("=" * 60)
    map_img = load_image(args.map, args.max_size)
    scan_img = load_image(args.scan, args.max_size)
    print(f"  Map:  {map_img.shape[1]}x{map_img.shape[0]}")
    print(f"  Scan: {scan_img.shape[1]}x{scan_img.shape[0]}")

    # 确保 scan 不大于 map
    sh, sw = scan_img.shape
    mh, mw = map_img.shape
    if sh > mh * 0.9 or sw > mw * 0.9:
        # scan 太大，缩小
        scale_down = min(mh * 0.8 / sh, mw * 0.8 / sw)
        new_sh, new_sw = int(sh * scale_down), int(sw * scale_down)
        scan_img = cv2.resize(scan_img, (new_sw, new_sh))
        print(f"  [Resize] Scan too large → {new_sw}x{new_sh}")

    print("\n" + "=" * 60)
    print("Step 2: 边缘提取 + 距离变换")
    print("=" * 60)

    if args.binarize:
        # ── 地图去灰: 仅保留边框+空白 ──
        print("  [Mode] 地图二值化 (去灰保边)")
        map_struct, map_borders = binarize_map(map_img, args.bin_block, args.bin_c)
        map_edges = extract_edges_from_binary(map_borders)
        # 同时保留空白区域的边界作为辅助 (自由空间轮廓)
        _, free_space = cv2.threshold(map_img, 200, 255, cv2.THRESH_BINARY)
        free_edges = cv2.Canny(free_space, 50, 150)
        # 合并边框 + 自由空间轮廓
        map_edges = cv2.bitwise_or(map_edges, free_edges)

        # scan 用 Canny (scan 通常比较干净)
        scan_edges = extract_edges(scan_img, args.canny_low, args.canny_high)
    else:
        map_edges = extract_edges(map_img, args.canny_low, args.canny_high)
        scan_edges = extract_edges(scan_img, args.canny_low, args.canny_high)

    # ── 扫描边缘清理 (点云碎片 → 连续轮廓) ──
    if clean_scan:
        print(f"  [CleanScan] enabled, dilate={args.scan_dilate} minlen={args.scan_minlen}")
        scan_edges = clean_scan_edges(
            scan_edges, dilate_iter=args.scan_dilate, min_length=args.scan_minlen)

    map_dist = build_distance_field(map_edges, max_dist=80)
    print(f"  Map edges:  {int(np.sum(map_edges > 0))} pixels")
    print(f"  Scan edges: {int(np.sum(scan_edges > 0))} pixels")

    if np.sum(scan_edges > 0) < 20:
        print("[ERROR] Scan 边缘像素不足, 请检查图像质量或调整 Canny 阈值")
        sys.exit(1)

    # ── 确定搜索范围 ──
    scale_stride = 0.2
    scales = np.arange(scale_min, scale_max + 0.01, scale_stride)
    scales = np.round(scales, 2)
    scales = scales[(scales >= scale_min) & (scales <= scale_max)]

    angles_deg = list(range(0, 360, angle_step))

    total_combos = len(scales) * len(angles_deg) * len(flips)
    print(f"\n  Search space: scales={list(scales)} ({len(scales)}), "
          f"angles={angles_deg} ({len(angles_deg)}), "
          f"flips={len(flips)} → {total_combos} combos")
    print(f"  Coarse step: {coarse_step}px, NCC: {use_ncc}, "
          f"CleanScan: {clean_scan}, Refine: {do_refine}")

    # ── 粗搜索 ──
    print("\n" + "=" * 60)
    print("Step 3: 粗搜索 (多尺度 × 多角度 × 多翻转)")
    print("=" * 60)
    t0 = time.time()
    candidates = coarse_search(
        scan_edges, map_edges, map_dist,
        scales=scales,
        angles_deg=angles_deg,
        flips=flips,
        step=coarse_step,
        top_k=5,
        use_ncc=use_ncc,
        fast_mode=args.fast,
    )

    if not candidates:
        print("[ERROR] 未找到有效匹配!")
        print("  提示: 尝试 --show-flip 开启翻转搜索, 或调整 --canny-low/--canny-high")
        sys.exit(1)

    # ── 精搜索 ──
    print("\n" + "=" * 60)
    print("Step 4: 精搜索 (在最优候选附近细化)")
    print("=" * 60)
    best = candidates[0]
    _, _, init_scale, init_ang, init_fh, init_fv, init_tx, init_ty = best

    fine = fine_search(
        scan_edges, map_edges, map_dist,
        init_scale, init_ang, init_fh, init_fv, init_tx, init_ty,
        scale_range=0.15, scale_step=0.03,
        ang_range=angle_step, ang_step=1.5,
        pos_range=coarse_step, pos_step=1,
        use_ncc=use_ncc,
        fast_mode=args.fast,
    )

    # ── 最终精调 ──
    if do_refine:
        print("\n" + "=" * 60)
        print("Step 5: 亚像素精调")
        print("=" * 60)
        final_scale, final_ang, final_tx, final_ty, final_ch = final_refine(
            scan_edges, map_edges, map_dist,
            fine[2], fine[3], fine[4], fine[5], fine[6], fine[7],
            iterations=5)
    else:
        print("\n  [Skip] 亚像素精调 (fast mode)")
        final_scale, final_ang, final_tx, final_ty = (
            fine[2], fine[3], fine[6], fine[7])
        final_ch = fine[1]

    sh_f, sw_f = scan_edges.shape
    center_x = final_tx + sw_f * final_scale * 0.5
    center_y = final_ty + sh_f * final_scale * 0.5

    flip_names = []
    if init_fh:
        flip_names.append("水平翻转")
    if init_fv:
        flip_names.append("垂直翻转")
    flip_str = " + ".join(flip_names) if flip_names else "无"

    print(f"\n最终匹配结果:")
    print(f"  缩放: {final_scale:.3f}x")
    print(f"  旋转: {final_ang:.1f}°")
    print(f"  翻转: {flip_str}")
    print(f"  平移: ({final_tx:.0f}, {final_ty:.0f}) px")
    print(f"  中心: ({center_x:.0f}, {center_y:.0f}) px")
    print(f"  Chamfer 距离: {final_ch:.3f}")
    print(f"  总耗时: {time.time()-t0:.1f}s")

    # ── 可视化 ──
    print("\n" + "=" * 60)
    print("Step 6: 可视化")
    print("=" * 60)
    visualize_result(
        map_img, map_edges,
        scan_img, scan_edges,
        final_scale, final_ang, init_fh, init_fv, final_tx, final_ty,
        output_path,
        top_candidates=candidates,
    )

    print("\n[完成] 匹配成功!")


if __name__ == '__main__':
    main()
