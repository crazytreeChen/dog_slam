"""Find rooms by closing wider gaps - try different thresholds"""
import numpy as np, cv2, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(BASE, '..', 'scan_viz')
d4 = np.load(os.path.join(VIZ, 'debug_match_data_4.npz'), allow_pickle=True)
m = d4['map_data']; res = float(d4['map_resolution'])
W = int(d4['map_width']); H = int(d4['map_height'])
ox = float(d4['map_origin_x']); oy = float(d4['map_origin_y'])

wall_mask = (m == 100).astype(np.uint8)

for gap_m in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    gap_px = int(gap_m / res)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_px, gap_px))
    wall_dilated = cv2.dilate(wall_mask, kernel, iterations=1)
    room_mask = (wall_dilated == 0).astype(np.uint8)
    room_mask = cv2.morphologyEx(room_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(room_mask, connectivity=4)

    # Count meaningful regions (>20m2)
    big_regions = []
    for i in range(1, num_labels):
        a = stats[i, cv2.CC_STAT_AREA]
        area_m2 = a * res * res
        if area_m2 < 20:
            continue
        l = stats[i, cv2.CC_STAT_LEFT]; t = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        cx = ox + (l + w/2)*res; cy = oy + (t + h/2)*res
        big_regions.append((area_m2, cx, cy, l, t, w, h))

    big_regions.sort(key=lambda r: r[1])  # sort by cx
    print(f"\ngap={gap_m:.1f}m ({gap_px}px): {len(big_regions)} big regions (>20m2)")
    for j, (area_m2, cx, cy, l, t, w, h) in enumerate(big_regions):
        bbox_str = f"[({ox+l*res:.1f},{oy+t*res:.1f}) {w*res:.1f}m x {h*res:.1f}m]"
        if cy > 23:
            zone = "UPPER"
        elif cy < 18:
            zone = "LOWER"
        else:
            zone = "MID"
        print(f"  #{j}: ({cx:.1f},{cy:.1f}) {area_m2:.0f}m2 {bbox_str} [{zone}]")
