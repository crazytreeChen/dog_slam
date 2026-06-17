"""Inspect map room layout + targeted search for data_4 lower room"""
import numpy as np, cv2, math, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(BASE, '..', 'scan_viz')
sys.path.insert(0, BASE)
import scan_map_overlay as mod

# Load data_4 (flat key format)
n4 = os.path.join(VIZ, 'debug_match_data_4.npz')
d4 = np.load(n4, allow_pickle=True)
m = d4['map_data']
res = float(d4['map_resolution']); W = int(d4['map_width']); H = int(d4['map_height'])
ox = float(d4['map_origin_x']); oy = float(d4['map_origin_y'])
info = {'resolution': res, 'width': W, 'height': H, 'origin_x': ox, 'origin_y': oy}
tf_gt = d4['tf_odom_to_map']
frame_tfs = d4['frame_tfs']
angle_min = float(d4['frame_angle_min']); angle_inc = float(d4['frame_angle_increment'])

# Load all frames
frame_ranges = []
i = 0
while f'frame_ranges_{i}' in d4:
    frame_ranges.append(np.array(d4[f'frame_ranges_{i}'], dtype=np.float64))
    i += 1
# Generate pts0 (odom frame: ranges -> 2D points)
r0 = frame_ranges[0]; tf0 = frame_tfs[0]
valid = (r0 > 0.15) & (r0 < 50.0)
angles0 = angle_min + np.arange(len(r0)) * angle_inc
lx = r0[valid] * np.cos(angles0[valid]); ly = r0[valid] * np.sin(angles0[valid])
tx, ty, yaw = tf0; c, s = math.cos(yaw), math.sin(yaw)
pts0 = np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty])
print(f"  pts0: {len(pts0)} points (filtered), {len(frame_ranges)} frames")

# ====== 1. Connected free-space regions ======
free_mask = (m == 0).astype(np.uint8)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(free_mask, connectivity=4)
print(f"\n=== Map: {W}x{H} ({W*res:.1f}m x {H*res:.1f}m), origin=({ox:.1f},{oy:.1f}) ===")
print(f"=== Free-space connected regions (>=100px) ===")
rooms = []
for i in range(1, num_labels):
    a = stats[i, cv2.CC_STAT_AREA]
    if a < 100: continue
    l, t, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    cx = ox + (l + w/2) * res; cy = oy + (t + h/2) * res
    print(f"  region#{i}: center=({cx:.1f},{cy:.1f}) bbox=[({ox+l*res:.1f},{oy+t*res:.1f}) {w*res:.1f}m x {h*res:.1f}m] area={a*res*res:.0f}m2")
    rooms.append({'cx': cx, 'cy': cy, 'area': a, 'l': l, 't': t, 'w': w, 'h': h})

# Find two largest regions (upper room and lower room)
areas_sorted = sorted(rooms, key=lambda r: r['area'], reverse=True)
print(f"\nTop 2 largest free regions:")
for i, r in enumerate(areas_sorted[:2]):
    lbl = 'UPPER' if r['cy'] > areas_sorted[1]['cy'] else 'LOWER'
    print(f"  #{i}: ({r['cx']:.1f},{r['cy']:.1f}) area={r['area']*res*res:.0f}m2 <- {lbl}")

upper = areas_sorted[0] if areas_sorted[0]['cy'] > areas_sorted[1]['cy'] else areas_sorted[1]
lower = areas_sorted[0] if areas_sorted[0]['cy'] < areas_sorted[1]['cy'] else areas_sorted[1]
print(f"  Upper room center: ({upper['cx']:.1f},{upper['cy']:.1f})")
print(f"  Lower room center: ({lower['cx']:.1f},{lower['cy']:.1f})")

# ====== 2. Targeted FreeSpace search in lower room (center +/- 8m) ======
cx_l, cy_l = lower['cx'], lower['cy']
margin = 8.0
print(f"\n=== FreeSpace search in lower room (center=({cx_l:.1f},{cy_l:.1f}) +/- {margin}m) ===")
cands = []
xmin = max(ox+2, cx_l - margin); xmax = min(ox+W*res-2, cx_l + margin)
ymin = max(oy+2, cy_l - margin); ymax = min(oy+H*res-2, cy_l + margin)
print(f"  Search bounds: x=[{xmin:.1f},{xmax:.1f}] y=[{ymin:.1f},{ymax:.1f}]")
for x in np.arange(xmin, xmax, 1.5):
    for y in np.arange(ymin, ymax, 1.5):
        for adeg in range(0, 360, 15):
            sc, fr, wr, nf, nw, nu, nt = mod.score_pose_freespace(
                pts0, x, y, math.radians(adeg), m, info)
            if sc > 0.05 and nf > 100:
                cands.append((sc, x, y, adeg, fr, wr, nf, nw))

cands.sort(key=lambda c: c[0], reverse=True)
nms = []
for c in cands:
    if any(math.hypot(c[1]-nx, c[2]-ny) < 2.5 for _, nx, ny, *_ in nms):
        continue
    nms.append(c)
    if len(nms) >= 10: break

print(f"\n  Top-{len(nms)} candidates (lower room, NMS 2.5m):")
for i, (sc, x, y, ad, fr, wr, nf, nw) in enumerate(nms):
    err = math.hypot(x - tf_gt[0], y - tf_gt[1])
    print(f"    #{i}: ({x:.1f},{y:.1f},{ad:>3d}deg) sc={sc:.4f} err_gt={err:.1f}m fr={fr:.1%} wr={wr:.1%} n_free={nf} n_wall={nw}")

# ====== 3. GT diagnostic ======
print(f"\n=== GT ({tf_gt[0]:.1f},{tf_gt[1]:.1f},{math.degrees(tf_gt[2]):.0f}deg) diagnostic ===")
sc_gt = mod.score_pose_freespace(pts0, tf_gt[0], tf_gt[1], tf_gt[2], m, info)
print(f"  score={sc_gt[0]:.4f} free_ratio={sc_gt[1]:.1%} wall_ratio={sc_gt[2]:.1%} "
      f"n_free={sc_gt[3]} n_wall={sc_gt[4]} n_unknown={sc_gt[5]}")

# ====== 4. Reference: data1-3 ======
print(f"\n=== Reference: data1-3 GT (likely upper room) ===")
for di in [1,2,3]:
    nd = os.path.join(VIZ, f'debug_match_data_{di}.npz')
    dr = np.load(nd, allow_pickle=True)
    gt_ref = dr['tf_odom_to_map']
    print(f"  Data{di}: ({gt_ref[0]:.1f},{gt_ref[1]:.1f},{math.degrees(gt_ref[2]):.0f}deg)")

# ====== 5. Baseline multistep result for data_4 ======
print(f"\n=== Baseline multistep result for data_4 ===")
print(f"  Init: (26.19, 15.30, -7deg) Final: (26.19, 15.30, -7deg)")
print(f"  Distance from lower room center: {math.hypot(26.19-cx_l, 15.30-cy_l):.1f}m")
print(f"  Distance from GT: {math.hypot(26.19-tf_gt[0], 15.30-tf_gt[1]):.1f}m")
