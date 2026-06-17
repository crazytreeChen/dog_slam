#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opencode_multistep_localizer.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
多步递推定位 — 逐帧处理 + 帧间ICP约束 + 自适应不确定度缩小

核心思路:
  不再合并所有帧, 而是逐帧处理:
    帧₀: 全局搜索 → 初始位姿 + 不确定度 σ₀
    帧₁: ICP(帧₀,帧₁) → 相对位移 → 预测 → 局部搜索(±σ₀) → 更新 σ₁=σ₀×0.8
    帧₂: ICP(帧₁,帧₂) → 相对位移 → 预测 → 局部搜索(±σ₁) → 更新 σ₂=σ₁×0.8
    ...
  越走越准: σ 指数衰减, 搜索范围缩小, 位置精度提高

用法:
  python opencode_multistep_localizer.py
  python opencode_multistep_localizer.py --data debug_match_data.npz
"""

import os, sys, math, time, argparse
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] Need opencv-python"); sys.exit(1)

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[ERROR] Need scipy"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("[ERROR] Need matplotlib"); sys.exit(1)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def setup_font():
    for name in ['SimHei', 'Microsoft YaHei', 'SimSun']:
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.sans-serif'] = [name] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            return
setup_font()


# ============================================================
# 1. Data Loading
# ============================================================
def load_data(npz_path):
    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}"); sys.exit(1)
    d = np.load(npz_path, allow_pickle=True)
    if 'map_resolution' in d:
        map_data = d['map_data']
        info = {'resolution': float(d['map_resolution']), 'width': int(d['map_width']),
                'height': int(d['map_height']), 'origin_x': float(d['map_origin_x']),
                'origin_y': float(d['map_origin_y'])}
    elif 'map_info' in d:
        map_data = d['map_data']; mi = d['map_info'].item()
        info = {'resolution': float(mi['resolution']), 'width': int(mi['width']),
                'height': int(mi['height']), 'origin_x': float(mi['origin_x']),
                'origin_y': float(mi['origin_y'])}
    else:
        print("[ERROR] Unknown NPZ format"); sys.exit(1)
    tf_gt = d['tf_odom_to_map']; frame_tfs = d['frame_tfs']
    angle_min = float(d.get('frame_angle_min', -math.pi))
    angle_inc = float(d.get('frame_angle_increment', 2*math.pi/len(d.get('frame_ranges_0', [0]*360))))
    frame_ranges = []; i = 0
    while f'frame_ranges_{i}' in d:
        frame_ranges.append(np.array(d[f'frame_ranges_{i}'], dtype=np.float64)); i += 1
    print(f"Data: {len(frame_ranges)} frames x {len(frame_ranges[0])} beams, Map: {info['width']}x{info['height']}")
    return map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc


# ============================================================
# 2. Per-frame Processing
# ============================================================
def frf_filter_frame(ranges, angle_min, angle_inc, bin_deg=2.0, gap_thresh=0.3):
    bin_size = np.radians(bin_deg)
    valid = (ranges > 0.15) & (ranges < 50.0)
    if not np.any(valid): return valid
    angles = angle_min + np.arange(len(ranges)) * angle_inc
    bins = np.round(angles / bin_size).astype(int)
    keep = np.ones(len(ranges), dtype=bool)
    for b in np.unique(bins[valid]):
        idx = np.where((bins == b) & valid)[0]
        if len(idx) < 2: continue
        sorted_idx = idx[np.argsort(ranges[idx])]
        gaps = np.diff(ranges[sorted_idx]) > gap_thresh
        if np.any(gaps):
            keep[sorted_idx[int(np.argmax(gaps))+1:]] = False
    return valid & keep


def frame_to_odom_pts(ranges, tf, angle_min, angle_inc):
    """单帧雷达 → odom系点云"""
    keep = frf_filter_frame(ranges, angle_min, angle_inc)
    if np.sum(keep) < 10: return np.empty((0, 2))
    angles = angle_min + np.arange(len(ranges)) * angle_inc
    lx = ranges[keep] * np.cos(angles[keep]); ly = ranges[keep] * np.sin(angles[keep])
    tx, ty, yaw = tf; c, s = math.cos(yaw), math.sin(yaw)
    return np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty])


def icp_scan_to_scan(src_pts, tgt_pts, max_iter=30, outlier_ratio=3.0):
    """ICP: 点云 src → tgt 刚体变换 (R, t)"""
    if len(src_pts) < 10 or len(tgt_pts) < 10:
        return np.eye(2), np.zeros(2)
    tree = cKDTree(tgt_pts)
    src = src_pts.copy()
    R_total, t_total = np.eye(2), np.zeros(2)
    for it in range(max_iter):
        dists, idx = tree.query(src)
        med = np.median(dists)
        mask = dists < max(0.1, med * outlier_ratio)
        if np.sum(mask) < 5: break
        s, m = src[mask], tgt_pts[idx[mask]]
        cs, cm = s.mean(0), m.mean(0)
        H = (s-cs).T @ (m-cm)
        try:
            U, S, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            break
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0: Vt[-1]*=-1; R = Vt.T @ U.T
        t = cm - R @ cs
        src = (R @ src.T).T + t
        R_total = R @ R_total; t_total = R @ t_total + t
        if np.linalg.norm(t) < 1e-5: break
    return R_total, t_total


# ============================================================
# 3. Map Matching (Likelihood Field)
# ============================================================
def build_likelihood_field(map_data, info, max_dist=3.0):
    obs = (map_data == 100).astype(np.uint8)
    dist_px = cv2.distanceTransform(1-obs, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    max_px = max_dist / info['resolution']
    lf = np.clip(dist_px, 0, max_px).astype(np.float32) * info['resolution']
    lf[map_data == -1] = max_dist  # 惩罚未知区域
    return lf


def score_pose_wallhit(points_odom, cx, cy, yaw, map_data, info):
    """
    墙壁命中评分: 只统计真正落在墙壁像素上的点。
    比似然场更严格 — 不会给"靠近墙壁"的点分数, 只在确实命中时计分。
    用于障碍物遮挡场景下区分相似走廊。
    """
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    H, W = info['height'], info['width']
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y*points_odom[:,0] - s_y*points_odom[:,1] + cx
    my = s_y*points_odom[:,0] + c_y*points_odom[:,1] + cy
    ci = ((mx-ox)/res+0.5).astype(np.int32); ri = ((my-oy)/res+0.5).astype(np.int32)
    v = (ci>=0)&(ci<W)&(ri>=0)&(ri<H); nv = int(np.sum(v))
    if nv < max(len(points_odom)*0.1, 3): return -1e9, 0, 0
    cells = map_data[ri[v], ci[v]]
    valid = (cells != -1); n_v = int(np.sum(valid))
    if n_v < 5: return -1e9, 0, 0
    n_wall = int(np.sum(cells[valid] == 100))
    n_free = int(np.sum(cells[valid] == 0))
    # 墙壁命中率 = wall / (wall + free), 要求 free < 60% (扫描不能大部分在空地上)
    if n_free / max(n_v, 1) > 0.60: return -1e9, 0, 0
    hit_rate = n_wall / max(n_wall + n_free, 1)
    # 评分 = 命中率 × 覆盖率
    coverage = n_v / len(points_odom)
    return hit_rate * coverage * 2.0, n_wall, n_v


def score_pose(points_odom, cx, cy, yaw, lf, info):
    """似然场评分"""
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    H, W = info['height'], info['width']
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y*points_odom[:,0] - s_y*points_odom[:,1] + cx
    my = s_y*points_odom[:,0] + c_y*points_odom[:,1] + cy
    ci = ((mx-ox)/res+0.5).astype(np.int32); ri = ((my-oy)/res+0.5).astype(np.int32)
    v = (ci>=0)&(ci<W)&(ri>=0)&(ri<H); nv = int(np.sum(v))
    if nv < max(len(points_odom)*0.1, 1): return -1e9, 0, 0
    dists = lf[ri[v], ci[v]]
    sc = float(np.mean(np.exp(-dists**2/0.045)))
    hit = int(np.sum(dists < 0.15))
    return sc + hit/nv*0.5, hit, nv


def score_pose_raycast(points_odom, cx, cy, yaw, map_data, info, range_tol=1.0):
    """
    光线投射评分: 只对"实测距离 ≈ 地图预期距离"的束评分。
    自动过滤被障碍物/家具遮挡的异常束。

    原理:
      对每个扫描点, 从扫描原点沿该方向发射虚拟射线:
      - 射线在预期距离处碰到地图墙壁 → 有效束 (障碍物未遮挡)
      - 射线在预期距离前碰到地图墙壁 → 异常束 (有地图外障碍物)
      - 射线穿过预期位置没有墙壁 → 异常束 (扫描点在空地/门外)
    
    返回: (score, n_valid_beams, total_beams)
    """
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    H, W = info['height'], info['width']

    n_pts = len(points_odom)
    n_valid = 0
    total_score = 0.0

    # 扫描原点在 map 中的位置
    origin_x = cx
    origin_y = cy

    for i in range(0, n_pts, max(1, n_pts//200)):  # 最多评估200束
        px = points_odom[i, 0]
        py = points_odom[i, 1]
        # 扫描点在 map 中的位置
        c_y, s_y = math.cos(yaw), math.sin(yaw)
        mx = c_y*px - s_y*py + cx
        my = s_y*px + c_y*py + cy

        # 扫描点相对于扫描原点的距离和方向
        dist_measured = math.sqrt(px*px + py*py)
        if dist_measured < 0.1: continue
        ray_angle = math.atan2(my - origin_y, mx - origin_x)

        # 沿射线方向步进, 检查地图墙壁
        dx_r = math.cos(ray_angle) * res * 0.5  # 半像素步进
        dy_r = math.sin(ray_angle) * res * 0.5
        rx, ry = origin_x, origin_y
        hit_wall = False
        dist_expected = 50.0  # 默认超过最大范围

        for step in range(int(50.0 / (res*0.5))):
            col = int((rx - ox) / res)
            row = int((ry - oy) / res)
            if col < 0 or col >= W or row < 0 or row >= H: break
            cell = map_data[row, col]
            if cell == 100:  # 碰到墙壁
                dist_expected = math.sqrt((rx-origin_x)**2 + (ry-origin_y)**2)
                hit_wall = True
                break
            elif cell == -1:  # 进入未知区域 → 跳过此束
                dist_expected = -1
                break
            rx += dx_r; ry += dy_r

        if hit_wall and abs(dist_measured - dist_expected) < range_tol:
            # 实测距离与地图墙壁距离一致 → 有效束
            n_valid += 1
            total_score += 1.0
        elif hit_wall:
            # 碰到墙但距离不对 → 可能有障碍物遮挡 → 不评分
            pass

    if n_valid < 5: return -1e9, 0, n_pts

    # 评分 = 有效束比例 (越高越好)
    n_evaluated = n_pts // max(1, n_pts//200)
    score = total_score / max(n_evaluated, 1)
    return score, n_valid, n_evaluated


def global_search_first_frame(pts_odom, lf, map_data, info, step=1.5, angle_step=8, top_k=8):
    """首帧全局搜索 (带自动过滤: 排除灰色区域和穿墙候选)"""
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    mw = info['width']*res; mh = info['height']*res; H, W = info['height'], info['width']
    xs = np.arange(ox+2, ox+mw-2, step); ys = np.arange(oy+2, oy+mh-2, step)
    n_ang = int(360/angle_step)
    
    # 降采样
    n_pts = min(len(pts_odom), 1000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pts_odom), size=n_pts, replace=False) if len(pts_odom) > n_pts else np.arange(len(pts_odom))
    pts_ds = pts_odom[idx]
    
    results = []
    n_filtered = 0
    for ax in xs:
        for ay in ys:
            for adeg in range(n_ang):
                ayaw = math.radians(adeg*angle_step)
                sc, _, _ = score_pose(pts_ds, ax, ay, ayaw, lf, info)
                if sc < 0.3: continue
                # ── 快速过滤: 检查扫描是否在有效区域内 ──
                c_y, s_y = math.cos(ayaw), math.sin(ayaw)
                # 用少量点做快速检查
                mx = c_y*pts_ds[:100,0] - s_y*pts_ds[:100,1] + ax if len(pts_ds) >= 100 else \
                     c_y*pts_ds[:,0] - s_y*pts_ds[:,1] + ax
                my = s_y*pts_ds[:100,0] + c_y*pts_ds[:100,1] + ay if len(pts_ds) >= 100 else \
                     s_y*pts_ds[:,0] + c_y*pts_ds[:,1] + ay
                ci = ((mx-ox)/res+0.5).astype(np.int32)
                ri = ((my-oy)/res+0.5).astype(np.int32)
                v = (ci>=0)&(ci<W)&(ri>=0)&(ri<H)
                if int(np.sum(v)) < 20: n_filtered+=1; continue
                cells = map_data[ri[v], ci[v]]
                n_unk = int(np.sum(cells==-1))
                n_wall = int(np.sum(cells==100))
                n_valid = len(cells)
                # 过滤: 未知区域 > 50% 或 墙壁 > 60% → 无效
                if n_unk/n_valid > 0.50 or n_wall/n_valid > 0.60:
                    n_filtered += 1; continue
                results.append((sc, ax, ay, adeg*angle_step))
    results.sort(key=lambda x: x[0], reverse=True)
    
    if n_filtered > 0:
        print(f"  [Filter] Excluded {n_filtered} invalid candidates (gray/wall)")
    
    # NMS
    nms = []
    for sc, ax, ay, ad in results:
        dup = any(math.sqrt((ax-cx)**2+(ay-cy)**2)<1.5 and abs(ad-ca)<20 for _,cx,cy,ca in nms)
        if not dup:
            nms.append((sc, ax, ay, ad))
            if len(nms) >= top_k * 2: break  # 保留更多候选供光线投射精选

    # ── 光线投射精选: 对 Top-K 候选用 ray-cast 重新排序 ──
    if len(nms) >= 2:
        rc_scored = []
        for sc, ax, ay, ad in nms:
            ayaw = math.radians(ad)
            rc_sc, rc_valid, rc_total = score_pose_raycast(pts_ds, ax, ay, ayaw, map_data, info)
            if rc_sc > -1e8 and rc_valid >= 5:
                rc_scored.append((sc, rc_sc, rc_valid/rc_total, ax, ay, ad))
        if rc_scored:
            best_rc = rc_scored[0]
            # 光线投射有效束比例 > 30% 才启用 (否则障碍物太多, 回退似然场)
            if best_rc[2] > 0.30:
                rc_scored.sort(key=lambda x: x[0]*0.3 + x[1]*0.7, reverse=True)
                nms = [(s, ax, ay, ad) for s, _, _, ax, ay, ad in rc_scored[:top_k]]
                print(f"  [RayCast] Best: ({best_rc[3]:.1f},{best_rc[4]:.1f},{best_rc[5]:.0f}deg) "
                      f"rc_score={best_rc[1]:.3f} valid={best_rc[2]:.1%}")
            else:
                print(f"  [RayCast] valid={best_rc[2]:.1%}<30%, fallback to likelihood")
    else:
        nms = nms[:top_k]
    return nms


def local_search(pts_odom, cx_pred, cy_pred, yaw_pred, lf, info,
                 radius=3.0, pos_step=0.2, angle_range=15, angle_step=2):
    """局部搜索: 在预测位姿周围 ±radius 范围搜索"""
    best_sc = -1e9; best_pose = (cx_pred, cy_pred, yaw_pred)
    # 降采样
    n_pts = min(len(pts_odom), 800)
    rng = np.random.default_rng()
    idx = rng.choice(len(pts_odom), size=n_pts, replace=False) if len(pts_odom) > n_pts else np.arange(len(pts_odom))
    pts_ds = pts_odom[idx]
    
    for dx in np.arange(-radius, radius+1e-5, pos_step):
        for dy in np.arange(-radius, radius+1e-5, pos_step):
            for da in np.arange(-angle_range, angle_range+1, angle_step):
                ax, ay = cx_pred+dx, cy_pred+dy
                ayaw = yaw_pred + math.radians(da)
                sc, _, _ = score_pose(pts_ds, ax, ay, ayaw, lf, info)
                if sc > best_sc:
                    best_sc = sc; best_pose = (ax, ay, ayaw)
    return best_pose, best_sc


# ============================================================
# 4. Multi-Step Localizer
# ============================================================
class MultiStepLocalizer:
    def __init__(self, map_data, info, lf, frame_pts_list, uncertainty_decay=0.7, min_sigma=0.5,
                 prior_x=None, prior_y=None, prior_yaw=None, prior_radius=10.0):
        self.map_data = map_data
        self.info = info
        self.lf = lf
        self.frames = frame_pts_list
        self.n_frames = len(frame_pts_list)
        
        self.mu = None
        self.sigma = 5.0
        self.sigma_angle = 20.0
        self.decay = uncertainty_decay
        self.min_sigma = min_sigma
        self.prior_x = prior_x
        self.prior_y = prior_y
        self.prior_yaw = prior_yaw
        self.prior_radius = prior_radius
        
        self.history = []
    
    def run(self):
        """运行完整多步定位流程"""
        print(f"\n{'='*60}")
        print(f"Multi-Step Localizer: {self.n_frames} frames")
        print(f"{'='*60}")
        
        for i in range(self.n_frames):
            pts, tf = self.frames[i]
            if len(pts) < 10:
                print(f"  Frame {i}: too few points, skip")
                continue
            
            if i == 0:
                # 首帧: 全局搜索 (有先验则约束范围)
                if self.prior_x is not None:
                    print(f"\n--- Frame 0: Local Search (prior ±{self.prior_radius}m) ---")
                    best_sc = -1e9; best_pose = None
                    for dx in np.arange(-self.prior_radius, self.prior_radius+1, 1.5):
                        for dy in np.arange(-self.prior_radius, self.prior_radius+1, 1.5):
                            for adeg in range(0, 360, 8):
                                sc, _, _ = score_pose(pts, self.prior_x+dx, self.prior_y+dy, 
                                                      math.radians(adeg), self.lf, self.info)
                                if sc > best_sc: best_sc = sc; best_pose = (self.prior_x+dx, self.prior_y+dy, math.radians(adeg))
                    if best_pose:
                        self.mu = best_pose
                        print(f"  Init: ({self.mu[0]:.2f}, {self.mu[1]:.2f}, {math.degrees(self.mu[2]):.1f}deg)")
                else:
                    print(f"\n--- Frame 0: Global Search ---")
                    t0 = time.time()
                    candidates = global_search_first_frame(pts, self.lf, self.map_data, self.info)
                    if not candidates:
                        print("  [ERROR] No initial candidates found"); return False
                    
                    # ── 光线投射回退: 独立墙壁命中全局搜索 ──
                    best_rc_check = score_pose_raycast(pts, candidates[0][1], candidates[0][2], 
                                                       math.radians(candidates[0][3]), self.map_data, self.info)
                    rc_valid_rate = best_rc_check[2] / max(best_rc_check[1], 1) if best_rc_check[0] > -1e8 else 0
                    if rc_valid_rate < 0.30:
                        print(f"  [WallHit fallback] independent search...")
                        wh_candidates = []
                        res = self.info['resolution']; ox = self.info['origin_x']; oy = self.info['origin_y']
                        mw = self.info['width']*res; mh = self.info['height']*res
                        for ax in np.arange(ox+3, ox+mw-3, 2.0):
                            for ay in np.arange(oy+3, oy+mh-3, 2.0):
                                for adeg in range(0, 360, 10):
                                    wh, nw, _ = score_pose_wallhit(pts, ax, ay, math.radians(adeg), self.map_data, self.info)
                                    if wh > 0: wh_candidates.append((wh, ax, ay, adeg))
                        if wh_candidates:
                            wh_candidates.sort(key=lambda x: x[0], reverse=True)
                            # NMS
                            wh_nms = []
                            for wh, ax, ay, ad in wh_candidates:
                                dup = any(math.sqrt((ax-wx)**2+(ay-wy)**2)<2.0 for _, wx, wy, _ in wh_nms)
                                if not dup: wh_nms.append((wh, ax, ay, ad))
                                if len(wh_nms) >= 8: break
                            candidates = wh_nms
                            print(f"  [WallHit] Best: ({wh_nms[0][1]:.1f},{wh_nms[0][2]:.1f},{wh_nms[0][3]:.0f}deg) wall%={100*wh_nms[0][0]/2:.0f}%")
                    # ── end wallhit fallback ──
                    best_sc = -1e9; best_pose = None
                    for sc, hx, hy, had in candidates[:3]:
                        pose, lf_sc = local_search(pts, hx, hy, math.radians(had), self.lf, self.info, radius=2.0, pos_step=0.3)
                        if lf_sc > best_sc: best_sc = lf_sc; best_pose = pose
                    self.mu = best_pose
                    print(f"  Init: ({self.mu[0]:.2f}, {self.mu[1]:.2f}, {math.degrees(self.mu[2]):.1f}deg) sigma={self.sigma:.1f}m ({time.time()-t0:.1f}s)")
            else:
                # 后续帧: ICP + 局部搜索
                prev_pts = self.frames[i-1][0]
                
                # ICP 帧间匹配 (旋转>30°时跳过ICP, 只用位置预测)
                R_icp, t_icp = np.eye(2), np.zeros(2)
                icp_used = False
                if len(pts) > 10 and len(prev_pts) > 10:
                    R_icp, t_icp = icp_scan_to_scan(pts, prev_pts)
                    dyaw_icp = math.atan2(R_icp[1,0], R_icp[0,0])
                    # 检查: ICP 旋转不应该超过 30° (旋转太快时ICP不可靠)
                    if abs(dyaw_icp) < math.radians(30):
                        icp_used = True
                    else:
                        R_icp, t_icp = np.eye(2), np.zeros(2)
                
                dyaw_icp = math.atan2(R_icp[1,0], R_icp[0,0])
                
                # 预测: ICP 平移+旋转 提供最强帧间约束, 锁住正确房间
                c_m, s_m = math.cos(self.mu[2]), math.sin(self.mu[2])
                pred_x = self.mu[0] + c_m*t_icp[0] - s_m*t_icp[1]
                pred_y = self.mu[1] + s_m*t_icp[0] + c_m*t_icp[1]
                pred_yaw = self.mu[2] + dyaw_icp
                
                # 局部搜索 (范围 = sigma, ICP 平移约束后只需小范围)
                search_radius = min(self.sigma * 1.5, 3.0)
                angle_range = min(self.sigma_angle * 1.5, 20)
                
                pose, lf_sc = local_search(pts, pred_x, pred_y, pred_yaw,
                                           self.lf, self.info,
                                           radius=search_radius,
                                           angle_range=int(angle_range))
                
                # ── ICP 失败保护 ──
                # 计算当前帧的墙壁覆盖率
                c_y, s_y = math.cos(pose[2]), math.sin(pose[2])
                mx = c_y*pts[:,0] - s_y*pts[:,1] + pose[0]
                my = s_y*pts[:,0] + c_y*pts[:,1] + pose[1]
                ci = ((mx-self.info['origin_x'])/self.info['resolution']+0.5).astype(np.int32)
                ri = ((my-self.info['origin_y'])/self.info['resolution']+0.5).astype(np.int32)
                mv = (ci>=0)&(ci<self.info['width'])&(ri>=0)&(ri<self.info['height'])
                cells = self.map_data[ri[mv], ci[mv]] if int(np.sum(mv)) > 10 else np.array([-1])
                valid_c = (cells != -1); n_vc = int(np.sum(valid_c))
                w_v = int(np.sum(cells[valid_c]==100)) if n_vc > 0 else 0
                f_v = int(np.sum(cells[valid_c]==0)) if n_vc > 0 else 0
                wall_pct = w_v / max(w_v+f_v, 1)
                icp_jump = math.sqrt(t_icp[0]**2 + t_icp[1]**2) if icp_used else 0
                
                # 条件1: 墙壁覆盖率 < 20% → 直接拒绝
                # 条件2: ICP跳变 > 2m 且覆盖率 < 35% → 拒绝
                if i > 3 and (wall_pct < 0.20 or (icp_jump > 2.0 and wall_pct < 0.35)):
                    pose = (self.mu[0], self.mu[1], pose[2])
                    lf_sc = -1
                    print(f"  f{i:02d} [REJECT] wall={100*wall_pct:.0f}% jump={icp_jump:.1f}m, keeping prev")
                
                # 更新
                self.mu = pose
                self.sigma = max(self.sigma * self.decay, self.min_sigma)
                self.sigma_angle = max(self.sigma_angle * self.decay, 2.0)
                
                dx = self.mu[0] - pred_x; dy = self.mu[1] - pred_y
                corr = math.sqrt(dx**2 + dy**2)
                dya = math.degrees(abs(math.atan2(math.sin(self.mu[2]-pred_yaw), math.cos(self.mu[2]-pred_yaw))))
                icp_tag = "[ICP]" if icp_used else "[pred]"
                print(f"  f{i:02d} {icp_tag}: pred=({pred_x:.2f},{pred_y:.2f}) corr={corr:.3f}m,{dya:.1f}deg "
                      f"-> ({self.mu[0]:.2f},{self.mu[1]:.2f},{math.degrees(self.mu[2]):.0f}deg) "
                      f"sigma={self.sigma:.2f}m sc={lf_sc:.3f}")
            
            # 记录历史
            H, W = self.info['height'], self.info['width']
            res = self.info['resolution']; ox = self.info['origin_x']; oy = self.info['origin_y']
            c_y, s_y = math.cos(self.mu[2]), math.sin(self.mu[2])
            mx = c_y*pts[:,0]-s_y*pts[:,1]+self.mu[0]; my = s_y*pts[:,0]+c_y*pts[:,1]+self.mu[1]
            ci = ((mx-ox)/res+0.5).astype(np.int32); ri = ((my-oy)/res+0.5).astype(np.int32)
            v = (ci>=0)&(ci<W)&(ri>=0)&(ri<H)
            cells = self.map_data[ri[v], ci[v]]; nv = len(cells)
            valid = (cells!=-1); n_v = int(np.sum(valid))
            w_v = int(np.sum(cells[valid]==100)); f_v = int(np.sum(cells[valid]==0))
            wall_pct = 100*w_v/max(w_v+f_v, 1)
            
            sc, _, _ = score_pose(pts, self.mu[0], self.mu[1], self.mu[2], self.lf, self.info)
            self.history.append((i, self.mu[0], self.mu[1], self.mu[2], self.sigma, wall_pct, sc))
        
        return True
    
    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"Final: ({self.mu[0]:.3f}, {self.mu[1]:.3f}, {math.degrees(self.mu[2]):.1f}deg) "
              f"sigma={self.sigma:.3f}m")
        print(f"\nPer-step wall alignment improvement:")
        print(f"  {'Step':>5s} {'Wall%':>7s} {'Score':>7s} {'Sigma':>7s}")
        for i, x, y, yaw, sigma, wp, sc in self.history:
            print(f"  {i:5d} {wp:6.1f}% {sc:7.3f} {sigma:6.3f}m")


# ============================================================
# 5. Visualization
# ============================================================
def create_visualization(map_data, info, tf_gt, localizer, output_path):
    fig = plt.figure(figsize=(24, 14))
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    H, W = info['height'], info['width']
    extent = [ox, ox+W*res, oy, oy+H*res]
    
    map_bg = np.zeros((H, W, 3), dtype=np.float32)
    map_bg[map_data==0] = [1,1,1]; map_bg[map_data==100] = [0.1,0.1,0.1]; map_bg[map_data==-1] = [0.7,0.7,0.7]
    
    hist = localizer.history
    
    # (a) Trajectory on map
    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(map_bg, origin='lower', extent=extent); ax.set_aspect('equal')
    xs = [h[1] for h in hist]; ys = [h[2] for h in hist]
    colors = plt.cm.viridis(np.linspace(0, 1, len(hist)))
    for i in range(len(hist)):
        ax.plot(xs[i], ys[i], 'o', color=colors[i], markersize=8)
        if i > 0:
            ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]], '-', color=colors[i], alpha=0.5)
        if i == 0:
            ax.annotate('Start', (xs[i], ys[i]), fontsize=8, color='red')
        if i == len(hist)-1:
            ax.annotate('End', (xs[i], ys[i]), fontsize=8, color='green')
    if tf_gt is not None:
        ax.plot(tf_gt[0], tf_gt[1], 'b*', markersize=14, label='GT')
    ax.set_title("(a) Trajectory (color=step order)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.1)
    
    # (b) Uncertainty decay
    ax = fig.add_subplot(2, 3, 2)
    steps = [h[0] for h in hist]; sigmas = [h[4] for h in hist]
    ax.plot(steps, sigmas, 'b-o', linewidth=2, markersize=6)
    ax.fill_between(steps, 0, sigmas, alpha=0.2)
    ax.set_xlabel('Step'); ax.set_ylabel('Sigma (m)')
    ax.set_title('(b) Position Uncertainty Decay'); ax.grid(True, alpha=0.3)
    
    # (c) Wall coverage improvement
    ax = fig.add_subplot(2, 3, 3)
    wp = [h[5] for h in hist]
    ax.plot(steps, wp, 'g-o', linewidth=2, markersize=6)
    ax.set_xlabel('Step'); ax.set_ylabel('Wall Coverage (%)')
    ax.set_title('(c) Wall Alignment per Step'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # (d) Final frame aligned
    ax = fig.add_subplot(2, 3, 4)
    if hist:
        last = hist[-1]
        bx, by, byaw = last[1], last[2], last[3]
        zoom = 10
        ax.set_xlim(bx-zoom, bx+zoom); ax.set_ylim(by-zoom, by+zoom)
        ax.imshow(map_bg, origin='lower', extent=extent); ax.set_aspect('equal')
        # Show last frame
        pts = localizer.frames[-1][0]
        c_y, s_y = math.cos(byaw), math.sin(byaw)
        aligned = np.column_stack([c_y*pts[:,0]-s_y*pts[:,1]+bx, s_y*pts[:,0]+c_y*pts[:,1]+by])
        ax.scatter(aligned[::5,0], aligned[::5,1], s=1, c='lime', alpha=0.5, label='Last frame')
        ax.plot(bx, by, 'r+', markersize=14, mew=3)
        ax.arrow(bx, by, 2.0*math.cos(byaw), 2.0*math.sin(byaw), head_width=0.4, head_length=0.3, fc='red', ec='darkred', lw=2.5)
        ax.set_title(f"(d) Last Frame Aligned\n({bx:.2f},{by:.2f},{math.degrees(byaw):.0f}deg) wall={last[5]:.1f}%")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)
    
    # (e) Merged all frames aligned
    ax = fig.add_subplot(2, 3, 5)
    if hist:
        bx, by, byaw = hist[-1][1], hist[-1][2], hist[-1][3]
        zoom = 10
        ax.set_xlim(bx-zoom, bx+zoom); ax.set_ylim(by-zoom, by+zoom)
        ax.imshow(map_bg, origin='lower', extent=extent); ax.set_aspect('equal')
        # Merge all frames using their individual poses
        all_aligned = []
        for i, h in enumerate(hist):
            pts = localizer.frames[i][0]
            c_y, s_y = math.cos(h[3]), math.sin(h[3])
            aligned = np.column_stack([c_y*pts[:,0]-s_y*pts[:,1]+h[1], s_y*pts[:,0]+c_y*pts[:,1]+h[2]])
            all_aligned.append(aligned)
        merged = np.vstack(all_aligned)
        step = max(1, len(merged)//3000)
        ax.scatter(merged[::step,0], merged[::step,1], s=1, c='lime', alpha=0.4, label='All frames')
        ax.plot(bx, by, 'r+', markersize=14, mew=3)
        ax.set_title("(e) All Frames Merged")
        ax.legend(fontsize=7)
    ax.grid(True, alpha=0.12)
    
    # (f) Report
    ax = fig.add_subplot(2, 3, 6); ax.axis('off')
    rep = ["=== Multi-Step Localizer ===", ""]
    rep.append(f"Frames: {localizer.n_frames}")
    rep.append(f"Initial sigma: 5.0m -> Final: {localizer.sigma:.3f}m")
    rep.append("")
    rep.append("Wall coverage progression:")
    for i, x, y, yaw, sigma, wp, sc in hist:
        rep.append(f"  Step {i}: {wp:.1f}% wall, sigma={sigma:.3f}m")
    if tf_gt is not None and hist:
        last = hist[-1]
        err = math.sqrt((last[1]-tf_gt[0])**2+(last[2]-tf_gt[1])**2)
        rep.append(f"\nvs GT: dist={err:.2f}m")
    rep.append("\nMethod: Sequential ICP + adaptive local search")
    ax.text(0.05, 0.95, "\n".join(rep), transform=ax.transAxes,
            fontfamily='monospace', fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.55))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[Output] PNG: {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Multi-Step Sequential Localizer')
    parser.add_argument('--data', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scan_viz', 'debug_match_data.npz'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--decay', type=float, default=0.7, help='Uncertainty decay factor (0-1)')
    parser.add_argument('--prior-x', type=float, default=None, help='Prior X position constraint')
    parser.add_argument('--prior-y', type=float, default=None, help='Prior Y position constraint')
    parser.add_argument('--prior-yaw', type=float, default=None, help='Prior yaw constraint (deg)')
    parser.add_argument('--prior-radius', type=float, default=10.0, help='Search radius around prior (m)')
    args = parser.parse_args()
    
    npz_path = os.path.normpath(args.data)
    if args.output and args.output.endswith('.png'):
        output_path = args.output
    else:
        output_dir = args.output or os.path.dirname(npz_path)
        if not os.path.isdir(output_dir): output_dir = os.path.dirname(npz_path)
        output_path = os.path.join(output_dir, 'multistep_loc_result.png')
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Load
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(npz_path)
    
    # Process each frame individually
    print("\nProcessing frames...")
    frame_pts = []
    for i, (ranges, tf) in enumerate(zip(frame_ranges, frame_tfs)):
        pts = frame_to_odom_pts(ranges, tf, angle_min, angle_inc)
        frame_pts.append((pts, tf))
        if i % 10 == 0:
            print(f"  Frame {i}: {len(pts)} pts")
    print(f"  Done: {len(frame_pts)} frames processed")
    
    # Build likelihood field
    lf = build_likelihood_field(map_data, info)
    
    # Run multi-step localizer
    prior_x = args.prior_x; prior_y = args.prior_y
    prior_yaw = math.radians(args.prior_yaw) if args.prior_yaw is not None else None
    localizer = MultiStepLocalizer(map_data, info, lf, frame_pts, uncertainty_decay=args.decay,
                                   prior_x=prior_x, prior_y=prior_y, prior_yaw=prior_yaw,
                                   prior_radius=args.prior_radius)
    success = localizer.run()
    
    if success:
        localizer.print_summary()
    
    # Visualize
    create_visualization(map_data, info, tf_gt, localizer, output_path)


if __name__ == '__main__':
    main()
