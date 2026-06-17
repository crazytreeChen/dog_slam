"""Targeted search in the lower room (21.3, 22.2) + WallHit discrimination"""
import numpy as np, cv2, math, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(BASE, '..', 'scan_viz')
sys.path.insert(0, BASE)
import scan_map_overlay as mod
import opencode_multistep_localizer as msdl

d4 = np.load(os.path.join(VIZ, 'debug_match_data_4.npz'), allow_pickle=True)
m = d4['map_data']; res = float(d4['map_resolution'])
W = int(d4['map_width']); H = int(d4['map_height'])
ox = float(d4['map_origin_x']); oy = float(d4['map_origin_y'])
info = {'resolution': res, 'width': W, 'height': H, 'origin_x': ox, 'origin_y': oy}
tf_gt = d4['tf_odom_to_map']; frame_tfs = d4['frame_tfs']
angle_min = float(d4['frame_angle_min']); angle_inc = float(d4['frame_angle_increment'])

# Load pts0
frame_ranges = []
i = 0
while f'frame_ranges_{i}' in d4:
    frame_ranges.append(np.array(d4[f'frame_ranges_{i}'], dtype=np.float64))
    i += 1
r0 = frame_ranges[0]; tf0 = frame_tfs[0]
valid = (r0 > 0.15) & (r0 < 50.0)
angles0 = angle_min + np.arange(len(r0)) * angle_inc
lx = r0[valid]*np.cos(angles0[valid]); ly = r0[valid]*np.sin(angles0[valid])
tx, ty, yaw = tf0; c, s = math.cos(yaw), math.sin(yaw)
pts0 = np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty])

# ===== Search in lower room bbox =====
# Room: (21.3,22.2) bbox=[(15.7,19.2) 11.2m x 6.0m]
lx_min, lx_max = 15.0, 27.5
ly_min, ly_max = 18.0, 26.0
print(f"Lower room search: x=[{lx_min},{lx_max}] y=[{ly_min},{ly_max}]")

# Stage 1: FreeSpace score to find open areas
print("\n--- Stage 1: FreeSpace candidates ---")
fs_cands = []
for x in np.arange(lx_min, lx_max, 1.0):
    for y in np.arange(ly_min, ly_max, 1.0):
        for adeg in range(0, 360, 15):
            sc, fr, wr, nf, nw, nu, nt = mod.score_pose_freespace(
                pts0, x, y, math.radians(adeg), m, info)
            if nf > 80:
                fs_cands.append((sc, x, y, adeg, fr, wr, nf, nw))

fs_cands.sort(key=lambda c: c[0], reverse=True)
# NMS 1.5m
nms_fs = []
for c in fs_cands:
    if any(math.hypot(c[1]-nx, c[2]-ny) < 1.5 for _, nx, ny, *_ in nms_fs):
        continue
    nms_fs.append(c)
    if len(nms_fs) >= 20: break

print(f"  Top-{len(nms_fs)} FreeSpace candidates (NMS 1.5m):")
for i, (sc, x, y, ad, fr, wr, nf, nw) in enumerate(nms_fs[:10]):
    print(f"    #{i}: ({x:.1f},{y:.1f},{ad:>3d}deg) fs={sc:.4f} fr={fr:.1%} wr={wr:.1%} nf={nf} nw={nw}")

# Stage 2: WallHit score on Top-20 FreeSpace candidates
print("\n--- Stage 2: WallHit discrimination ---")
wh_cands = []
for sc_fs, x, y, ad, fr, wr, nf, nw in nms_fs:
    wh, nwall, nv = msdl.score_pose_wallhit(pts0, x, y, math.radians(ad), m, info)
    wh_cands.append({'x': x, 'y': y, 'deg': ad, 'fs': sc_fs, 'fr': fr, 'wr': wr,
                     'nf': nf, 'nw': nw, 'wh': max(wh, 0), 'nwall': nwall})

wh_cands.sort(key=lambda c: c['wh'], reverse=True)

print(f"  Top-10 by WallHit score:")
for i, c in enumerate(wh_cands[:10]):
    print(f"    #{i}: ({c['x']:.1f},{c['y']:.1f},{c['deg']:>3d}deg) "
          f"WH={c['wh']:.3f}(n={c['nwall']}) FS={c['fs']:.3f} fr={c['fr']:.1%} nf={c['nf']}")

# Also: combined score FS*0.3 + WH*0.7
combined = []
for c in wh_cands:
    comb = c['fs'] * 0.3 + c['wh'] * 0.7
    combined.append({**c, 'comb': comb})
combined.sort(key=lambda c: c['comb'], reverse=True)

print(f"\n--- Stage 3: Combined (FS*0.3 + WH*0.7) ---")
for i, c in enumerate(combined[:10]):
    err = math.hypot(c['x']-tf_gt[0], c['y']-tf_gt[1])
    print(f"    #{i}: ({c['x']:.1f},{c['y']:.1f},{c['deg']:>3d}deg) "
          f"comb={c['comb']:.4f} WH={c['wh']:.3f} FS={c['fs']:.3f} err_gt={err:.1f}m")

# Reference
print(f"\n=== Reference ===")
print(f"  data1-3 multistep: ~(29, 29)")
print(f"  data4 multistep baseline: (26.2, 15.3)")  
print(f"  GT (unreliable): ({tf_gt[0]:.1f}, {tf_gt[1]:.1f}, {math.degrees(tf_gt[2]):.0f}deg)")
print(f"  data4 GT (from npz): ({tf_gt[0]:.1f}, {tf_gt[1]:.1f})")
