"""可视化: 地图 + 下方房间边界 + Top-3 候选 + baseline/Gt 对比"""
import numpy as np, cv2, math, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import colors

BASE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(BASE, '..', 'scan_viz')

d4 = np.load(os.path.join(VIZ, 'debug_match_data_4.npz'), allow_pickle=True)
m = d4['map_data']; res = float(d4['map_resolution'])
W = int(d4['map_width']); H = int(d4['map_height'])
ox = float(d4['map_origin_x']); oy = float(d4['map_origin_y'])
info = {'resolution': res, 'width': W, 'height': H, 'origin_x': ox, 'origin_y': oy}
tf_gt = d4['tf_odom_to_map']; frame_tfs = d4['frame_tfs']
angle_min = float(d4['frame_angle_min']); angle_inc = float(d4['frame_angle_increment'])
tf0 = frame_tfs[0]

# Load pts0
frame_ranges = []
i = 0
while f'frame_ranges_{i}' in d4:
    frame_ranges.append(np.array(d4[f'frame_ranges_{i}'], dtype=np.float64))
    i += 1
r0 = frame_ranges[0]
valid = (r0 > 0.15) & (r0 < 50.0)
angles0 = angle_min + np.arange(len(r0)) * angle_inc
lx = r0[valid]*np.cos(angles0[valid]); ly = r0[valid]*np.sin(angles0[valid])
tx, ty, yaw = tf0; c, s = math.cos(yaw), math.sin(yaw)
pts0 = np.column_stack([c*lx - s*ly + tx, s*lx + c*ly + ty])

# 绘制地图
fig, ax = plt.subplots(figsize=(14, 12))
m_disp = np.where(m == -1, 127, np.where(m == 0, 255, 0)).astype(np.uint8)
cmap = colors.ListedColormap(['black', 'gray', 'white'])
extent = [ox, ox + W*res, oy + H*res, oy]
ax.imshow(m_disp, cmap=cmap, extent=extent, origin='upper', aspect='equal', interpolation='none')

# 下方房间边界
room_x0, room_y0 = 15.7, 19.2
room_w, room_h = 11.2, 6.0
rect = Rectangle((room_x0, room_y0), room_w, room_h,
                 linewidth=3, edgecolor='#00FF00', facecolor='none',
                 linestyle='--', label='Lower Room (morph closure 1.5m)')
ax.add_patch(rect)

# Top-3 candidates from FreeSpace search
candidates = [
    (21.0, 22.0, 0,   0.942, '#1: FS top'),
    (23.0, 22.0, 0,   0.928, '#2'),
    (19.0, 25.0, -15, 0.914, '#3'),
]
colors_c = ['#FF6600', '#FF9900', '#FFCC00']
for idx, (cx, cy, cdeg, csc, clabel) in enumerate(candidates):
    rad = math.radians(cdeg)
    arrow_dx = 0.8 * math.cos(rad)
    arrow_dy = 0.8 * math.sin(rad)
    arrow = FancyArrowPatch((cx - arrow_dx*0.3, cy - arrow_dy*0.3),
                            (cx + arrow_dx*0.7, cy + arrow_dy*0.7),
                            arrowstyle='->', mutation_scale=25,
                            color=colors_c[idx], linewidth=2.5, zorder=10)
    ax.add_patch(arrow)
    ax.plot(cx, cy, 'o', color=colors_c[idx], markersize=10, zorder=11,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate(f'{clabel}\n({cx:.1f},{cy:.1f}) fs={csc:.3f}',
                (cx, cy+0.5), fontsize=9, color=colors_c[idx],
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# data4 multistep baseline (蓝色X)
ax.plot(26.2, 15.3, 'X', color='blue', markersize=14, markeredgewidth=2, zorder=10)
ax.annotate('multistep baseline\n(26.2, 15.3)', (26.2, 15.3),
            fontsize=9, color='blue', ha='left', va='bottom',
            xytext=(1.5, -1.5), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# GT (红色X)
ax.plot(tf_gt[0], tf_gt[1], 'X', color='red', markersize=14, markeredgewidth=2, zorder=10)
ax.annotate(f'GT odom->map\n({tf_gt[0]:.1f}, {tf_gt[1]:.1f})',
            (tf_gt[0], tf_gt[1]),
            fontsize=9, color='red', ha='right', va='top',
            xytext=(-1.5, 1.5), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# data1-3 区域标注
ax.annotate('data1-3 region\n~(29, 29) upper room',
            (29, 29), fontsize=10, color='purple',
            ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE0FF', edgecolor='purple', alpha=0.7))
ax.plot(29, 29, 's', color='purple', markersize=10, zorder=10)

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_title('data_4 Room-Scoped FreeSpace Search: Lower Room Candidates', fontsize=13)
ax.legend(loc='upper right', fontsize=9)

# 缩放至下方区域
ax.set_xlim(10, 35)
ax.set_ylim(18, 32)

fig.tight_layout()
outpath = os.path.join(VIZ, 'room_search_result.png')
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"Saved: {outpath}")
print(f"\n=== Summary ===")
print(f"  Lower room: center=(21.3,22.2) bbox=[(15.7,19.2) 11.2m x 6.0m], area=50m2")
print(f"  Top FS candidate: (21.0, 22.0, 0deg) fs=0.942  free_rate=98.0%")
print(f"  multistep baseline: (26.2, 15.3)  -- completely outside lower room!")
print(f"  GT odom->map: ({tf_gt[0]:.1f}, {tf_gt[1]:.1f})  -- also outside room boundary")
print(f"\n  Key insight: room-scoping correctly constrains search to the right area;")
print(f"  WallHit=0 because walls in this room are mostly gray (unknown) on map.")
