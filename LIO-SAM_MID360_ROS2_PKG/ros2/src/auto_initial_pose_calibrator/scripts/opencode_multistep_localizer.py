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


def score_pose(points_odom, cx, cy, yaw, lf, info):
    """似然场评分"""
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    H, W = info['height'], info['width']
    c_y, s_y = math.cos(yaw), math.sin(yaw)
    mx = c_y*points_odom[:,0] - s_y*points_odom[:,1] + cx
    my = s_y*points_odom[:,0] + c_y*points_odom[:,1] + cy
    ci = ((mx-ox)/res+0.5).astype(np.int32); ri = ((my-oy)/res+0.5).astype(np.int32)
    v = (ci>=0)&(ci<W)&(ri>=0)&(ri<H); nv = int(np.sum(v))
    if nv < len(points_odom)*0.1: return -1e9, 0, 0
    dists = lf[ri[v], ci[v]]
    sc = float(np.mean(np.exp(-dists**2/0.045)))
    hit = int(np.sum(dists < 0.15))
    return sc + hit/nv*0.5, hit, nv


def global_search_first_frame(pts_odom, lf, map_data, info, step=1.5, angle_step=8, top_k=8):
    """首帧全局搜索"""
    res = info['resolution']; ox = info['origin_x']; oy = info['origin_y']
    mw = info['width']*res; mh = info['height']*res
    xs = np.arange(ox+2, ox+mw-2, step); ys = np.arange(oy+2, oy+mh-2, step)
    n_ang = int(360/angle_step)
    
    # 降采样
    n_pts = min(len(pts_odom), 1000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(pts_odom), size=n_pts, replace=False) if len(pts_odom) > n_pts else np.arange(len(pts_odom))
    pts_ds = pts_odom[idx]
    
    results = []
    for ax in xs:
        for ay in ys:
            for adeg in range(n_ang):
                sc, _, _ = score_pose(pts_ds, ax, ay, math.radians(adeg*angle_step), lf, info)
                if sc > 0.3: results.append((sc, ax, ay, adeg*angle_step))
    results.sort(key=lambda x: x[0], reverse=True)
    
    # NMS
    nms = []
    for sc, ax, ay, ad in results:
        dup = any(math.sqrt((ax-cx)**2+(ay-cy)**2)<1.5 and abs(ad-ca)<20 for _,cx,cy,ca in nms)
        if not dup:
            nms.append((sc, ax, ay, ad))
            if len(nms) >= top_k: break
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
    def __init__(self, map_data, info, lf, frame_pts_list, uncertainty_decay=0.7, min_sigma=0.5):
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
        
        self.history = []  # [(step, x, y, yaw, sigma, wall_pct, score), ...]
    
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
                # 首帧: 全局搜索
                print(f"\n--- Frame 0: Global Search ---")
                t0 = time.time()
                candidates = global_search_first_frame(pts, self.lf, self.map_data, self.info)
                
                if not candidates:
                    print("  [ERROR] No initial candidates found")
                    return False
                
                # 对 Top-3 做精细搜索
                best_sc = -1e9; best_pose = None
                for sc, hx, hy, had in candidates[:3]:
                    pose, lf_sc = local_search(pts, hx, hy, math.radians(had), 
                                               self.lf, self.info, radius=2.0, pos_step=0.3)
                    if lf_sc > best_sc:
                        best_sc = lf_sc; best_pose = pose
                
                self.mu = best_pose
                self.sigma = 5.0
                print(f"  Init: ({self.mu[0]:.2f}, {self.mu[1]:.2f}, {math.degrees(self.mu[2]):.1f}deg) "
                      f"sigma={self.sigma:.1f}m ({time.time()-t0:.1f}s)")
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
                
                # 预测
                c_m, s_m = math.cos(self.mu[2]), math.sin(self.mu[2])
                pred_x = self.mu[0] + c_m*t_icp[0] - s_m*t_icp[1]
                pred_y = self.mu[1] + s_m*t_icp[0] + c_m*t_icp[1]
                pred_yaw = self.mu[2] + dyaw_icp
                
                # 局部搜索 (范围 = sigma)
                search_radius = min(self.sigma * 1.5, 3.0)
                angle_range = min(self.sigma_angle * 1.5, 20)
                
                pose, lf_sc = local_search(pts, pred_x, pred_y, pred_yaw,
                                           self.lf, self.info,
                                           radius=search_radius,
                                           angle_range=int(angle_range))
                
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
    localizer = MultiStepLocalizer(map_data, info, lf, frame_pts, uncertainty_decay=args.decay)
    success = localizer.run()
    
    if success:
        localizer.print_summary()
    
    # Visualize
    create_visualization(map_data, info, tf_gt, localizer, output_path)


if __name__ == '__main__':
    main()
