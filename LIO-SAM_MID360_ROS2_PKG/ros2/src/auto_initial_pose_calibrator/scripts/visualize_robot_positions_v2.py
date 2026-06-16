#!/usr/bin/env python3
"""
Scan-to-Map Border Alignment v3 (Line-Segment Matching)

Core algorithm:
1. Merge all N frames into single dense point cloud in odom frame
2. First-Return Filter: per angle bin, keep only nearest range (real wall).
   Farther returns = glass ghost ── physics-based, no heuristics.
3. Extract BOUNDARY LINE SEGMENTS from merged scan
4. Extract wall segments from map occupancy
5. Convex-hull shape matching: rotate+translate to maximize wall overlap
6. Interior verification: check all points fall in free space
7. ICP fine-tuning on final pose

Key difference from v2: matches LINE SEGMENTS + SHAPE, not sparse point clouds.
A long line can only match ONE place on the map -> no false positives.
"""

import os, sys, math, argparse, time
import numpy as np
from scipy.spatial import cKDTree, ConvexHull

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
except ImportError:
    print("Need matplotlib"); sys.exit(1)


# ============================================================
# Data loading
# ============================================================
def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    map_data = d['map_data']
    res = float(d['map_resolution'])
    mw, mh = int(d['map_width']), int(d['map_height'])
    ox, oy = float(d['map_origin_x']), float(d['map_origin_y'])
    tf_init = d['tf_odom_to_map']
    frame_tfs = d['frame_tfs']
    angle_min = float(d['frame_angle_min'])
    angle_inc = float(d['frame_angle_increment'])
    frame_ranges = []
    i = 0
    while f'frame_ranges_{i}' in d:
        frame_ranges.append(d['frame_ranges_{}'.format(i)]); i += 1
    return map_data, {'resolution': res, 'width': mw, 'height': mh,
                      'origin_x': ox, 'origin_y': oy}, tf_init, frame_tfs, frame_ranges, angle_min, angle_inc


def pt_to_map_grid(pt, info):
    res, ox, oy = info['resolution'], info['origin_x'], info['origin_y']
    px = int((pt[0] - ox) / res)
    py = int((pt[1] - oy) / res)
    return px, py


# ============================================================
# First-Return Filter — keep nearest (+margin) per angle bin, remove far ghosts
# ============================================================
# Physics: at any angle from the sensor, the first surface (nearest range)
# = real wall. Points MUCH farther at the same angle = glass ghost.
#
# Applied in ODOM space after merging all frames:
#   30 frames overlapping in same area → same angle has ~33 points stacked.
#   Strategy: keep nearest + all within ghost_margin of nearest.
#   Only points > ghost_margin behind the wall are classified as glass ghost.
#
def filter_first_return(raw_odom_pts, n_angle_bins=360, ghost_margin=2.0):
    """
    Filter merged odom-frame point cloud: per azimuth angle bin (from origin),
    find the nearest point (= real wall surface).
    Keep: nearest + all points within ghost_margin of nearest.
    Remove: points > ghost_margin behind the wall (= glass ghost).

    Args:
        raw_odom_pts: (N,2) merged odom-frame point cloud
        n_angle_bins: angular bins (360 = 1deg, 720 = 0.5deg)
        ghost_margin: distance (m) behind nearest to classify as ghost

    Returns:
        filtered: (K,2) real-wall points
        glass:    (M,2) ghost points (far behind nearest)
        bin_keep_counts: dict {bin_id: n_kept}
        bin_rm_counts:   dict {bin_id: n_removed}
    """
    if n_angle_bins < 2:
        n_angle_bins = 360

    t0 = time.time()
    pts = np.asarray(raw_odom_pts)
    if len(pts) == 0:
        return pts, np.empty((0, 2)), {}, {}, n_angle_bins

    # Compute azimuth angle from origin (0,0) for each point
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)   # Euclidean distance from origin

    # Bin by angle
    bin_edges = np.linspace(-math.pi, math.pi, n_angle_bins + 1)
    bi = np.digitize(angles, bin_edges) - 1
    bi = np.clip(bi, 0, n_angle_bins - 1)

    # Step 1: find the nearest point in each bin (first return = real wall)
    order = np.lexsort((dist, bi))  # sort by bi, then dist
    sorted_bi = bi[order]
    bin_change = np.concatenate([[True], np.diff(sorted_bi) != 0])
    first_of_bin_at_order = np.where(bin_change)[0]
    first_of_bin_idx = order[first_of_bin_at_order]
    nearest_dist = dist[first_of_bin_idx]
    nearest_bin = bi[first_of_bin_idx]

    # Map: bin_id -> nearest distance
    bin_nearest = dict(zip(nearest_bin, nearest_dist))

    # Step 2: keep all points within ghost_margin of nearest in same bin
    keep_mask = np.zeros(len(pts), dtype=bool)
    for b, min_d in bin_nearest.items():
        mask_b = bi == b
        dist_in_bin = dist[mask_b]
        keep_in_bin = dist_in_bin <= min_d + ghost_margin
        keep_mask[mask_b] = keep_in_bin

    filtered = pts[keep_mask]
    glass_mask = ~keep_mask
    glass = pts[glass_mask] if glass_mask.sum() > 0 else np.empty((0, 2))

    # Per-bin statistics
    bin_keep_counts = {}
    bin_rm_counts = {}
    for b in range(n_angle_bins):
        mask_b = (bi == b)
        n_in = int(mask_b.sum())
        if n_in > 0:
            n_k = int(keep_mask[mask_b].sum())
            bin_keep_counts[b] = n_k
            bin_rm_counts[b] = n_in - n_k

    elapsed = time.time() - t0
    total = len(pts)
    ghost_ratio = len(glass) / max(1, total) * 100
    active_bins = sum(1 for b in range(n_angle_bins) if bin_keep_counts.get(b, 0) > 0)

    print(f'[First-Return Filter] {n_angle_bins} bins ({360/n_angle_bins:.1f}deg), '
          f'ghost_margin={ghost_margin}m')
    print(f'  Input: {len(pts)} merged odom pts → Keep: {len(filtered)} ({len(filtered)/max(1,total)*100:.1f}%), '
          f'Ghost: {len(glass)} ({ghost_ratio:.1f}%)')
    print(f'  Active bins: {active_bins}/{n_angle_bins}, '
          f'mean {len(pts)/max(1,active_bins):.1f} pts/bin [{elapsed:.2f}s]')

    return filtered, glass, bin_keep_counts, bin_rm_counts, n_angle_bins


# ============================================================
# NEW: Extract boundary LINE SEGMENTS from merged scan
# ============================================================
def extract_scan_boundary_lines(filtered_points, n_radial_bins=72,
                                 min_line_length=1.5,
                                 ransac_inlier_thresh=0.25,
                                 min_points_per_line=8):
    """
    Extract dominant line segments from the OUTER BOUNDARY of merged scan.

    Method:
      1. Compute centroid
      2. In each angular bin, take the FARTHEST point -> boundary sample
      3. Sequential RANSAC: fit line, remove inliers, repeat
      4. Each fitted line becomes a line segment (endpoint = projection of extreme inliers)

    Returns list of dicts: {'p1':(x,y), 'p2':(x,y), 'length', 'angle', 'n_inliers', 'inlier_pts'}
    """
    if len(filtered_points) < 30:
        print('[Boundary] Too few points, skipping')
        return []

    centroid = filtered_points.mean(0)
    vecs = filtered_points - centroid
    dists = np.linalg.norm(vecs, axis=1)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])

    # Radial boundary sampling: farthest point in each angular bin
    bin_edges = np.linspace(-math.pi, math.pi, n_radial_bins + 1)
    boundary_pts = []
    for i in range(n_radial_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if mask.sum() > 0:
            idx_in_bin = np.where(mask)[0]
            farthest_idx = idx_in_bin[np.argmax(dists[idx_in_bin])]
            boundary_pts.append(filtered_points[farthest_idx])

    boundary_pts = np.array(boundary_pts)
    print(f'[Boundary] Radial sampling: {len(boundary_pts)} boundary points '
          f'(from {len(filtered_points)} filtered)')

    if len(boundary_pts) < 6:
        print('[Boundary] Too few boundary points')
        return []

    # Sequential RANSAC line fitting
    remaining = boundary_pts.copy()
    segments = []
    max_iter = 200
    seg_idx = 0

    while len(remaining) >= min_points_per_line:
        best_line = None; best_inliers = 0; best_inlier_mask = None

        for _ in range(max_iter):
            # Pick 2 random points
            i1, i2 = np.random.choice(len(remaining), 2, replace=False)
            p1, p2 = remaining[i1], remaining[i2]

            if np.linalg.norm(p2 - p1) < min_line_length * 0.3:
                continue

            # Line params: ax + by + c = 0 (normalized)
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            length = math.hypot(dx, dy)
            a, b = -dy / length, dx / length  # normal vector
            c = -(a * p1[0] + b * p1[1])

            # Distances from all remaining points to line
            dists_line = np.abs(a * remaining[:, 0] + b * remaining[:, 1] + c)
            inlier_mask = dists_line < ransac_inlier_thresh
            n_inliers = inlier_mask.sum()

            if n_inliers > best_inliers:
                best_inliers = n_inliers
                best_inlier_mask = inlier_mask
                # Refit line using all inliers
                inlier_pts = remaining[inlier_mask]
                cx_i = inlier_pts.mean(0)
                if len(inlier_pts) >= 2:
                    _, _, Vt = np.linalg.svd(inlier_pts - cx_i)
                    direction = Vt[0]
                    best_line = {'direction': direction, 'center': cx_i,
                                 'normal': np.array([a, b]), 'c': c}

        if best_line is None or best_inliers < min_points_per_line:
            break

        inlier_pts = remaining[best_inlier_mask]
        direction = best_line['direction']
        center = best_line['center']

        # Project inliers onto line to get endpoints
        proj = center + (inlier_pts - center) @ direction.reshape(-1, 1) * direction
        t_vals = (proj - center) @ direction
        t_min, t_max = t_vals.min(), t_vals.max()
        ep1 = center + t_min * direction
        ep2 = center + t_max * direction
        seg_length = t_max - t_min

        if seg_length >= min_line_length:
            seg_angle = math.atan2(direction[1], direction[0])
            segments.append({
                'p1': ep1, 'p2': ep2,
                'length': seg_length, 'angle': seg_angle,
                'n_inliers': best_inliers,
                'inlier_pts': inlier_pts,
                'center': center, 'direction': direction
            })
            seg_idx += 1

        # Remove inliers for next iteration
        remaining = remaining[~best_inlier_mask]

    # Sort by length (longest first = most reliable)
    segments.sort(key=lambda s: s['length'], reverse=True)

    total_len = sum(s['length'] for s in segments)
    print(f'[Boundary Lines] Extracted {len(segments)} segments, total length={total_len:.1f}m')
    for i, s in enumerate(segments):
        print(f'  Seg#{i}: len={s["length"]:.2f}m ang={math.degrees(s["angle"]):.0f}deg '
              f'inliers={s["n_inliers"]} ({s["p1"][0]:.1f},{s["p1"][1]:.1f})->({s["p2"][0]:.1f},{s["p2"][1]:.1f})')

    return segments


# ============================================================
# NEW: Extract map wall segments (from occupancy grid)
# ============================================================
def extract_map_wall_segments(map_data, info, resample_step=3):
    """
    Sample map occupied cells as wall reference points.
    Also try to extract dominant line orientations for scoring.
    """
    res = info['resolution']; mw = info['width']; mh = info['height']
    ox, oy = info['origin_x'], info['origin_y']

    occ_ys, occ_xs = np.where(map_data == 100)
    if len(occ_xs) == 0:
        return np.empty((0, 2)), []

    map_wall_pts = np.column_stack([
        occ_xs[::resample_step] * res + ox,
        (mh - 1 - occ_ys[::resample_step]) * res + oy
    ])

    # Try to extract dominant wall orientations using gradient analysis
    # Create a local occupancy density map at lower resolution
    scale = 4  # downsample factor
    small_h, small_w = mh // scale, mw // scale
    density = np.zeros((small_h, small_w))
    for iy, ix in zip(occ_ys, occ_xs):
        sy, sx = iy // scale, ix // scale
        if 0 <= sy < small_h and 0 <= sx < small_w:
            density[sy, sx] = 1

    # Simple Hough-like: find dominant directions from occupied pixel positions
    # (Just use raw points for now - KDTree lookup is fast enough)
    print(f'[Map Walls] {len(map_wall_pts)} sampled points (from {len(occ_xs)} occupied)')
    return map_wall_pts, []


# ============================================================
# NEW: Direction histogram for rotation hypothesis generation
# ============================================================
def build_direction_histogram(angles_weights, n_bins=180):
    """
    Build a direction histogram from (angle, weight) pairs.
    Angles are in radians, histogram covers [-pi/2, pi/2] (undirected lines).
    Returns normalized histogram and bin centers.
    """
    # Normalize angles to [0, pi) for undirected lines
    norm_angles = np.array([a % math.pi for a, _ in angles_weights])
    weights = np.array([w for _, w in angles_weights])

    bin_edges = np.linspace(0, math.pi, n_bins + 1)
    hist = np.zeros(n_bins)
    for (ang, w) in zip(norm_angles, weights):
        bi = int(ang / math.pi * n_bins)
        bi = min(bi, n_bins - 1)
        hist[bi] += w

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Normalize
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist, bin_centers


def find_dominant_directions(segments_or_points, is_segments=True, n_top=4, n_bins=180):
    """
    Find dominant directions from line segments or points.
    Returns list of (angle_rad, strength) sorted by strength.
    """
    if is_segments:
        aw = [(s['angle'], s['length']) for s in segments_or_points]
    else:
        pts = segments_or_points
        # Use point-pair vectors to estimate local directions
        tree = cKDTree(pts)
        aw = []
        for p in pts:
            dists, idx = tree.query(p, k=6)
            neighbors = pts[idx[1:]]  # exclude self
            if len(neighbors) >= 2:
                vecs = neighbors - p
                dists_n = np.linalg.norm(vecs, axis=1)
                valid = dists_n > 0.1
                if valid.sum() >= 1:
                    for v in vecs[valid]:
                        ang = math.atan2(v[1], v[0])
                        w = min(np.linalg.norm(v), 2.0)
                        aw.append((ang, w))

    if not aw:
        return []

    hist, bins = build_direction_histogram(aw, n_bins=n_bins)

    # Find peaks (local maxima)
    peaks = []
    for i in range(1, n_bins - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.02:
            peaks.append((bins[i], hist[i]))

    peaks.sort(key=lambda x: -x[1])
    return peaks[:n_top]


def match_directions(scan_dirs, map_dirs, tolerance_deg=10):
    """
    Match scan directions to map directions.
    Returns list of candidate yaw offsets (in radians).
    Each candidate: scan_dir_angle - map_dir_angle (+ pi multiples).
    """
    candidates = []
    tol_rad = math.radians(tolerance_deg)

    for s_ang, s_w in scan_dirs:
        for m_ang, m_w in map_dirs:
            # Undirected: try both m_ang and m_ang + pi (same as m_ang for undirected)
            diff = s_ang - m_ang
            # Normalize to [-pi, pi]
            while diff > math.pi/2: diff -= math.pi/2
            while diff < -math.pi/2: diff += math.pi/2

            if abs(diff) < tol_rad or abs(abs(diff) - math.pi/2) < tol_rad:
                candidates.append({
                    'yaw': diff,
                    'scan_dir': s_ang,
                    'map_dir': m_ang,
                    'strength': s_w * m_w,
                })
                # Also add perpendicular (pi/2 offset)
                candidates.append({
                    'yaw': diff + math.pi/2,
                    'scan_dir': s_ang,
                    'map_dir': m_ang + math.pi/2,
                    'strength': s_w * m_w * 0.8,
                })

    # Deduplicate similar yaws
    if not candidates:
        return []
    candidates.sort(key=lambda x: -x['strength'])

    unique = []
    for c in candidates:
        is_dup = False
        for u in unique:
            yaw_diff = abs(c['yaw'] - u['yaw'])
            # normalize to [-pi, pi]
            while yaw_diff > math.pi: yaw_diff -= 2*math.pi
            while yaw_diff < -math.pi: yaw_diff += 2*math.pi
            if yaw_diff < math.radians(5):
                is_dup = True
                if c['strength'] > u['strength']:
                    u.update(c)
                break
        if not is_dup:
            unique.append(c)

    unique.sort(key=lambda x: -x['strength'])
    return unique


# ============================================================
# NEW: Convex-Hull Shape Matching (v3c: robust & simple)
# ============================================================
def compute_convex_hull_shape(points, n_boundary_samples=120):
    """
    Compute convex hull of points, return ordered boundary vertices + sampled edge points.
    
    Returns:
      hull_vertices: (N, 2) ordered vertices of convex hull
      edge_samples: (M, 2) points evenly sampled along hull edges
      centroid: (2,) center of hull
    """
    if len(points) < 4:
        return points, points, points.mean(0) if len(points) > 0 else np.zeros(2)

    try:
        hull = ConvexHull(points)
        hull_verts = points[hull.vertices]  # ordered CCW
    except Exception:
        # Fallback: angular sort around centroid
        c = points.mean(0)
        angles = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
        order = np.argsort(angles)
        hull_verts = points[order]

    # Close the loop
    closed = np.vstack([hull_verts, hull_verts[0:1]])

    # Sample points along edges proportionally to edge length
    edge_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total_len = edge_lengths.sum()
    n_samples_per_edge = np.maximum(2, (edge_lengths / total_len * n_boundary_samples).astype(int))

    edge_pts_list = []
    for i in range(len(closed) - 1):
        n_samp = max(2, int(n_boundary_samples * edge_lengths[i] / total_len))
        alphas = np.linspace(0, 1, n_samp)
        pts = np.outer(alphas, closed[i+1] - closed[i]) + closed[i]
        edge_pts_list.append(pts[:-1])  # exclude endpoint (next edge will include it)

    edge_samples = np.vstack(edge_pts_list) if edge_pts_list else hull_verts
    centroid = hull_verts.mean(0)

    return hull_verts, edge_samples, centroid


def shape_matching_search(filtered_points, scan_segments, map_wall_pts, map_tree,
                          map_data, info, tf_init=None,
                          xy_step_coarse=0.6, xy_step_fine=0.15,
                          search_radius=8.0, fine_yaw_range_deg=12):
    """
    Shape matching using CONVEX HULL boundary:
    
    Strategy (priority order):
     1. Search around TF init (high priority)
     2. Search at evenly-spaced yaws across map (global fallback)
     3. Fine-grained refinement of top candidates
    
    Scoring: 
      - Fraction of hull boundary samples within threshold of map walls
      - Mean distance of ALL filtered points to nearest FREE cell (interior check)
      - Penalize occupied-space overlap
    """
    res = info['resolution']; mw = info['width']; mh = info['height']
    ox, oy = info['origin_x'], info['origin_y']

    print(f'\n[Shape Match v3c] Using convex hull boundary matching')

    # Compute convex hull of filtered scan
    hull_vtx, edge_samples, centroid = compute_convex_hull_shape(filtered_points, n_boundary_samples=100)
    print(f'  Hull: {len(hull_vtx)} vertices, {len(edge_samples)} edge samples')

    # Also keep segments for additional scoring
    seg_inlier_pts = []
    if scan_segments:
        for s in scan_segments[:4]:
            seg_inlier_pts.append(s['inlier_pts'])
    if seg_inlier_pts:
        seg_pts_all = np.vstack(seg_inlier_pts)
    else:
        seg_pts_all = edge_samples

    # Map bounds
    occ_ys, occ_xs = np.where(map_data == 100)
    if len(occ_xs) == 0:
        return (ox + mw*res/2, oy + mh*res/2, 0.0), 0.0, {}
    map_min_x = float(occ_xs.min() * res + ox)
    map_max_x = float(occ_xs.max() * res + ox)
    map_min_y = float((mh - 1 - occ_ys.max()) * res + oy)
    map_max_y = float((mh - 1 - occ_ys.min()) * res + oy)

    def score_pose(tx, ty, yaw):
        """Comprehensive pose scoring."""
        rc, rs = math.cos(yaw), math.sin(yaw)
        R = np.array([[rc, -rs], [rs, rc]])

        # Transform edge samples
        edges_t = (R @ edge_samples.T).T + [tx, ty]
        dists_edge, _ = map_tree.query(edges_t)

        # Edge-wall proximity score
        hit_thresh = 0.35
        edge_hit = np.mean(dists_edge < hit_thresh)
        edge_mean_d = np.mean(dists_edge)

        # Segment-line score (if available)
        seg_score = 0
        if len(seg_pts_all) > 0:
            seg_t = (R @ seg_pts_all.T).T + [tx, ty]
            d_seg, _ = map_tree.query(seg_t)
            seg_hit = np.mean(d_seg < 0.40)
            seg_mean = np.mean(d_seg)
            seg_score = seg_hit * 30 - seg_mean * 2

        # Interior check: transform all filtered points, check occupancy
        all_t = (R @ filtered_points.T).T + [tx, ty]
        n_free = 0; n_occ = 0; n_checked = 0
        step_i = max(1, len(all_t) // 2000)
        for pt in all_t[::step_i]:
            gpx = int((pt[0] - ox) / res); gpy = int((pt[1] - oy) / res)
            if 0 <= gpx < mw and 0 <= gpy < mh:
                val = map_data[mh - 1 - gpy, gpx]
                if val == 0: n_free += 1
                elif val >= 80: n_occ += 1
                n_checked += 1

        pct_free = n_free / max(1, n_checked)
        pct_occ = n_occ / max(1, n_checked)

        # Combined score
        total = (
            edge_hit * 60.0          # primary: edge-wall overlap
            + seg_score              # secondary: segment alignment
            - edge_mean_d * 3.0      # penalize far-from-wall edges
            + pct_free * 15.0         # reward free-space interior
            - pct_occ * 50.0          # HEAVY penalty for occupied interior
        )
        return total, {
            'edge_hit': edge_hit, 'edge_d': edge_mean_d,
            'pct_free': pct_free, 'pct_occ': pct_occ,
        }

    # ── Search list: TF-init first, then global grid ──
    search_list = []

    # Priority 1: Dense search around TF init
    if tf_init is not None:
        tfx, tfy, tfyaw = tf_init
        print(f'\n  [Priority 1] TF-init region: ({tfx:.1f},{tfy:.1f},{math.degrees(tfyaw):.0f}deg)')
        xs_tf = np.arange(tfx - search_radius, tfx + search_radius + xy_step_coarse, xy_step_coarse)
        ys_tf = np.arange(tfy - search_radius, tfy + search_radius + xy_step_coarse, xy_step_coarse)
        yaws_tf = np.linspace(tfyaw - math.radians(25), tfyaw + math.radians(25), 15)
        for yaw in yaws_tf:
            for tx in xs_tf:
                for ty in ys_tf:
                    sc, dt = score_pose(tx, ty, yaw)
                    search_list.append((tx, ty, yaw, sc, dt, 'tf_region'))

    # Priority 2: Global coarse search (broader, coarser)
    print(f'  [Priority 2] Global grid')
    xs_g = np.arange(map_min_x + 2, map_max_x - 2, 2.0)
    ys_g = np.arange(map_min_y + 2, map_max_y - 2, 2.0)
    yaws_g = np.arange(-math.pi, math.pi, math.radians(15))
    for yaw in yaws_g:
        for tx in xs_g:
            for ty in ys_g:
                sc, dt = score_pose(tx, ty, yaw)
                search_list.append((tx, ty, yaw, sc, dt, 'global'))

    # Sort by score
    search_list.sort(key=lambda x: -x[3])

    print(f'\n  [Search Results Top-5]')
    for i, (cx, cy, cyaw, cs, cdet, src) in enumerate(search_list[:5]):
        print(f'    #{i}: ({cx:.1f},{cy:.1f},{math.degrees(cyaw):.0f}deg) '
              f'score={cs:.1f} edge_hit={cdet["edge_hit"]:.2f} '
              f'free={cdet["pct_free"]*100:.0f}% occ={cdet["pct_occ"]*100:.0f}% [{src}]')

    # ── Phase 4: Fine refinement around top-N ──
    top_n = min(3, len(search_list))
    best_score = -99999; best_pose = search_list[0][:3]; best_detail = search_list[0][4]

    print(f'\n  [Fine Refinement] Top-{top_n}')
    for ci in range(top_n):
        cx, cy, cyaw, cs, cdet, src = search_list[ci]
        xs_f = np.arange(cx - 2.5, cx + 2.5 + xy_step_fine, xy_step_fine)
        ys_f = np.arange(cy - 2.5, cy + 2.5 + xy_step_fine, xy_step_fine)
        yaws_f = np.linspace(cyaw - math.radians(fine_yaw_range_deg),
                              cyaw + math.radians(fine_yaw_range_deg), 9)

        for fyaw in yaws_f:
            for ftx in xs_f:
                for fty in ys_f:
                    fs, fd = score_pose(ftx, fty, fyaw)
                    if fs > best_score:
                        best_score = fs
                        best_pose = (ftx, fty, fyaw)
                        best_detail = fd

    print(f'\n  [Best Pose] ({best_pose[0]:.2f}, {best_pose[1]:.2f}, {math.degrees(best_pose[2]):.1f}deg)')
    print(f'  Score={best_score:.1f} edge_hit={best_detail["edge_hit"]:.2f} '
          f'free={best_detail["pct_free"]*100:.0f}% occ={best_detail["pct_occ"]*100:.0f}%')

    return best_pose, best_score, best_detail


# Keep old name for backward compat
line_segment_matching = shape_matching_search


# ============================================================
# Interior verification: after edge-match, verify interior points
# ============================================================
def verify_interior_points(pose, all_filtered_pts, map_data, info):
    """
    Given a candidate pose, transform all filtered points and check:
    1. What fraction falls in FREE space (good!)
    2. What fraction falls in OCCUPIED space (bad - wrong pose)
    3. What fraction falls in UNKNOWN (tolerable near edges)
    """
    tx, ty, yaw = pose
    rc, rs = math.cos(yaw), math.sin(yaw)
    R = np.array([[rc, -rs], [rs, rc]])

    transformed = (R @ all_filtered_pts.T).T + [tx, ty]

    mw, mh = info['width'], info['height']
    n_free = 0; n_occ = 0; n_unk = 0; n_out = 0

    # Sample to speed up (check every 3rd point)
    sample_step = max(1, len(transformed) // 3000)
    for pt in transformed[::sample_step]:
        px, py = pt_to_map_grid(pt, info)
        if 0 <= px < mw and 0 <= py < mh:
            val = map_data[mh - 1 - py, px]
            if val == 0:
                n_free += 1
            elif val == 100:
                n_occ += 1
            else:
                n_unk += 1
        else:
            n_out += 1

    total_checked = n_free + n_occ + n_unk
    pct_free = n_free / max(1, total_checked) * 100
    pct_occ = n_occ / max(1, total_checked) * 100

    # Good interior: mostly free space, minimal occupation
    interior_score = pct_free - pct_occ * 5  # heavy penalty for occupied

    print(f'[Interior Verify] Free={pct_free:.0f}% Occupied={pct_occ:.0f}% '
          f'Unknown={n_unk/max(1,total_checked)*100:.0f}% Out-of-bounds={n_out} '
          f'(score={interior_score:.0f})')

    return interior_score, pct_free, pct_occ, transformed


# ============================================================
# ICP Refinement (unchanged)
# ============================================================
def icp_refine(source, target_tree, max_iter=40, tol=1e-5,
               outlier_ratio=2.0, min_inliers=15):
    src = source.copy()
    R_total = np.eye(2); t_total = np.zeros(2); history = []

    for it in range(max_iter):
        dists, idx = target_tree.query(src)
        med = np.median(dists)
        thresh = max(0.25, med * outlier_ratio)
        mask = dists < thresh
        if np.sum(mask) < min_inliers:
            break
        s_pts = src[mask]; m_pts = target_tree.data[idx[mask]]
        cs, cm = s_pts.mean(0), m_pts.mean(0)
        H = (s_pts - cs).T @ (m_pts - cm)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1; R = Vt.T @ U.T
        t = cm - R @ cs
        src = (R @ src.T).T + t
        R_total = R @ R_total; t_total = R @ t_total + t
        history.append({'iter': it, 'inliers': np.sum(mask), 'dr': np.linalg.norm(t),
                        'mean_err': np.mean(dists[mask])})
        if np.linalg.norm(t) < tol:
            break
    return R_total, t_total, history


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Scan-to-Map Border Alignment v3 (Line Matching)')
    parser.add_argument('--data', type=str,
                        default='D:/01-Code/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/src/auto_initial_pose_calibrator/scan_viz/debug_match_data.npz')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--angle-bins', type=int, default=360,
                        help='Number of angle bins for first-return filter (360=1deg, 720=0.5deg)')
    parser.add_argument('--ghost-margin', type=float, default=2.0,
                        help='Distance (m) beyond nearest return to classify as glass ghost (default 2.0)')
    parser.add_argument('--max-range', type=float, default=12.0, help='Max valid lidar range (m)')
    parser.add_argument('--no-filter', action='store_true', help='Skip first-return filter (use all points)')
    args = parser.parse_args()

    t_total_start = time.time()
    print('=' * 60)
    print('  Scan-to-Map Border Alignment v3 (First-Return + Line Matching)')
    print('  Filter: First-Return (nearest per angle bin = real wall)')
    print('  Match:  Convex-Hull Shape + Line Overlap -> Interior Verify')
    print('=' * 60)

    map_data, info, tf_init, frame_tfs, frame_ranges, angle_min, angle_inc = load_data(args.data)
    res = info['resolution']; mw, mh = info['width'], info['height']
    ox, oy = info['origin_x'], info['origin_y']
    n_frames = len(frame_ranges)
    print(f'\nData: {n_frames} frames, map {mw}x{mh} @ {res}m/px, '
          f'origin=({ox:.1f},{oy:.1f})')
    print(f'TF init (REFERENCE ONLY!): ({tf_init[0]:.2f}, {tf_init[1]:.2f}, {math.degrees(tf_init[2]):.1f}deg)')
    print('WARNING: odom starts at (0,0,0), TF is runtime estimate, NOT ground truth!')

    # ── Step 1: Merge all frames into odom frame ──
    print('\n--- Step 1: Merge all frames ---')
    raw_all_list = []
    for ranges, ft in zip(frame_ranges, frame_tfs):
        r = np.array(ranges, dtype=np.float64)
        valid = (r > 0.15) & (r < args.max_range)
        a = np.arange(len(r)) * angle_inc + angle_min
        pts = np.column_stack([r[valid] * np.cos(a[valid]), r[valid] * np.sin(a[valid])])
        c, s = math.cos(ft[2]), math.sin(ft[2])
        R = np.array([[c, -s], [s, c]])
        raw_all_list.append((R @ pts.T).T + [ft[0], ft[1]])
    raw_all = np.vstack(raw_all_list)
    print(f'Raw merged cloud: {len(raw_all)} points')

    # ── Step 2: First-Return Glass Filter (on merged odom cloud) ──
    print('\n--- Step 2: First-Return Glass Filter (odom-space) ---')
    if not args.no_filter:
        filtered, glass, bin_keep_counts, bin_rm_counts, n_bins = filter_first_return(
            raw_all, n_angle_bins=args.angle_bins, ghost_margin=args.ghost_margin)
    else:
        print('[--no-filter] Skipped')
        filtered = raw_all; glass = np.empty((0, 2))
        bin_keep_counts = {}; bin_rm_counts = {}; n_bins = 0

    # ── Step 3: Extract BOUNDARY LINE SEGMENTS from merged scan ──
    print('\n--- Step 3: Extract Scan Boundary Lines ---')
    scan_segments = extract_scan_boundary_lines(
        filtered, n_radial_bins=72,
        min_line_length=1.5, ransac_inlier_thresh=0.25,
        min_points_per_line=6)

    # Also keep full filtered cloud for ICP + visualization
    scan_for_icp = filtered

    # ── Step 4: Map wall data ──
    print('\n--- Step 4: Prepare Map Data ---')
    map_wall_pts, _ = extract_map_wall_segments(map_data, info, resample_step=3)
    map_tree = cKDTree(map_wall_pts)

    # ── Step 5: Line-segment matching search ──
    print('\n--- Step 5: Line-Segment Pose Search ---')
    (search_x, search_y, search_yaw), search_score, search_detail = line_segment_matching(
        filtered, scan_segments, map_wall_pts, map_tree, map_data, info,
        tf_init=tf_init,
        xy_step_coarse=0.6, xy_step_fine=0.15,
        search_radius=8.0, fine_yaw_range_deg=12)

    # ── Step 6: Interior verification ──
    print('\n--- Step 6: Interior Verification ---')
    int_score, pct_free, pct_occ, transformed_filtered = verify_interior_points(
        (search_x, search_y, search_yaw), filtered, map_data, info)

    # If interior is bad (lots of occupied), try TF init as alternative
    if pct_occ > 15 and tf_init is not None:
        print('\n[Interior Bad] Trying TF-init as fallback...')
        int_tf, pct_free_tf, pct_occ_tf, _ = verify_interior_points(tf_init, filtered, map_data, info)
        if int_tf > int_score:
            print(f'  TF init is better! (int_score {int_tf:.0f}->{int_tf:.0f})')
            search_x, search_y, search_yaw = tf_init
            int_score = int_tf; pct_free = pct_free_tf; pct_occ = pct_occ_tf

    # ── Step 7: ICP refinement ──
    print('\n--- Step 7: ICP Refinement ---')
    rc, rs = math.cos(search_yaw), math.sin(search_yaw)
    R_s = np.array([[rc, -rs], [rs, rc]])
    t_s = np.array([search_x, search_y])

    # Use boundary line inlier points for ICP (more robust than random sampling)
    if scan_segments:
        icp_source = np.vstack([s['inlier_pts'] for s in scan_segments[:4]])  # top 4 segments
    else:
        icp_source = scan_for_icp

    scan_mapped = (R_s @ icp_source.T).T + t_s
    R_icp, t_icp, icp_hist = icp_refine(scan_mapped, map_tree)
    final_R = R_icp @ R_s
    final_t = R_icp @ t_s + t_icp
    final_yaw = math.atan2(final_R[1, 0], final_R[0, 0])

    # Final scoring
    final_transformed = (final_R @ filtered.T).T + final_t
    dfinal, _ = map_tree.query(final_transformed)
    final_score = np.mean(dfinal < 0.35)
    mean_err_final = np.mean(dfinal[dfinal < 0.5]) if np.any(dfinal < 0.5) else 999
    print(f'ICP: {len(icp_hist)} iters, score={final_score:.3f}, mean_err={mean_err_final:.3f}m')
    if icp_hist:
        last = icp_hist[-1]; print(f'  Last: {last["inliers"]}inliers dr={last["dr"]:.4f}m')

    # Transform all clouds for viz
    filtered_map = (final_R @ filtered.T).T + final_t
    glass_map = (final_R @ glass.T).T + final_t if len(glass) > 0 else np.empty((0, 2))
    raw_map = (final_R @ raw_all.T).T + final_t

    elapsed_total = time.time() - t_total_start
    print(f'\n{"="*55}')
    print(f'Final pose: ({final_t[0]:.2f}, {final_t[1]:.2f}, {math.degrees(final_yaw):.1f}deg)')
    print(f'Score: {final_score:.3f} | Interior: {pct_free:.0f}% free, {pct_occ:.0f}% occupied')
    if tf_init is not None:
        print(f'vs TF(ref): dx={final_t[0]-tf_init[0]:.2f}, dy={final_t[1]-tf_init[1]:.2f}, '
              f'dyaw={math.degrees(final_yaw-tf_init[2]):.1f}deg')
    print(f'Total time: {elapsed_total:.1f}s')
    print(f'{"="*55}')

    # ============================================================
    # Visualization: 2x3 panels
    # ============================================================
    print('\nGenerating visualization...')

    map_disp = np.zeros((mh, mw, 3), dtype=np.float32)
    map_disp[map_data == 0] = [1, 1, 1]
    map_disp[map_data == 100] = [0, 0, 0]
    map_disp[map_data == -1] = [0.75, 0.75, 0.75]
    extent = [ox, ox + mw * res, oy, oy + mh * res]

    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # ── (a) Merged scan + extracted boundary lines ──
    ax = axes[0, 0]; ax.set_aspect('equal')
    ax.scatter(filtered[:, 0], filtered[:, 1], s=0.3, c='cyan', alpha=0.25, edgecolors='none',
               label=f'Merged scan ({len(filtered)})')
    # Draw boundary line segments
    for i, seg in enumerate(scan_segments):
        color = plt.cm.Set1(i % 9)
        ax.plot([seg['p1'][0], seg['p2'][0]], [seg['p1'][1], seg['p2'][1]],
                '-', color=color, linewidth=3, alpha=0.9,
                label=f'Line#{i} ({seg["length"]:.1f}m)')
        # Mark inlier points
        ax.scatter(seg['inlier_pts'][:, 0], seg['inlier_pts'][:, 1],
                   s=8, c=[color], alpha=0.5, edgecolors='none')
    ax.set_title(f'(a) {n_frames}-Frame Merged Scan + {len(scan_segments)} Boundary Lines\n'
                 f'Cyan=merged points, Colored lines=RANSAC-fitted edges', fontsize=11)
    ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.15)

    # ── (b) Filter result (odometer frame) ──
    ax = axes[0, 1]; ax.set_aspect('equal')
    if len(glass) > 0:
        ax.scatter(glass[:, 0], glass[:, 1], s=0.5, c='red', alpha=0.25,
                   label=f'Removed (glass/noise): {len(glass)}')
    ax.scatter(filtered[:, 0], filtered[:, 1], s=0.5, c='#0066FF', alpha=0.4,
               label=f'Kept (clean): {len(filtered)}')
    kept_pct = len(filtered) / max(1, len(raw_all)) * 100
    ax.set_title(f'(b) Odom Space -- First-Return Filtered\n'
                 f'Kept {kept_pct:.0f}% ({len(filtered)}/{len(raw_all)}) — Blue=real wall, Red=ghost', fontsize=11)
    ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.15)

    # ── (c) First-Return Filter stats per angle bin ──
    ax = axes[0, 2]
    if bin_keep_counts:
        bins_all = sorted(bin_keep_counts.keys())
        keep_vals = [bin_keep_counts.get(b, 0) for b in bins_all]
        ghost_vals = [bin_rm_counts.get(b, 0) for b in bins_all]
        bar_width = 0.8
        x = np.arange(len(bins_all))
        ax.bar(x, keep_vals, bar_width, color='steelblue', alpha=0.85, label='Kept (nearest)')
        ax.bar(x, ghost_vals, bar_width, bottom=keep_vals, color='salmon', alpha=0.6, label='Ghost (farther)')
        # Highlight bins with high ghost ratio
        for bi in bins_all:
            nk = bin_keep_counts.get(bi, 0)
            ng = bin_rm_counts.get(bi, 0)
            if nk + ng > 0 and ng / (nk + ng) > 0.5:
                ax.axvspan(bi - 0.5, bi + 0.5, color='red', alpha=0.08)
        ax.set_xlabel(f'Angle bin ({360/n_bins:.1f}deg each, {n_bins} total)')
        ax.set_ylabel('Points (all frames)')
        ax.set_title(f'(c) First-Return Filter per Angle\n'
                     f'Blue=real wall (nearest), Red=ghost (farther)', fontsize=11)
        ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.2, axis='y')
    else:
        ax.text(0.5, 0.5, '--no-filter mode', ha='center', transform=ax.transAxes, fontsize=14)

    # ── (d) Full map: aligned result with matched lines ──
    ax = axes[1, 0]; ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    ax.scatter(filtered_map[:, 0], filtered_map[:, 1], s=0.3, c='cyan', alpha=0.25,
               edgecolors='none', label=f'Aligned scan ({len(filtered_map)})')
    # Draw aligned line segments
    for i, seg in enumerate(scan_segments):
        color = plt.cm.Set1(i % 9)
        # Transform endpoints
        p1t = final_R @ seg['p1'] + final_t
        p2t = final_R @ seg['p2'] + final_t
        ax.plot([p1t[0], p2t[0]], [p1t[1], p2t[1]],
                '-', color=color, linewidth=4, alpha=0.95,
                solid_capstyle='round', label=f'Line#{i} aligned')
    ax.plot(final_t[0], final_t[1], 'rX', markersize=16, markeredgewidth=4, zorder=10)
    ax.arrow(final_t[0], final_t[1],
             2.5 * math.cos(final_yaw), 2.5 * math.sin(final_yaw),
             head_width=0.5, head_length=0.3, fc='red', ec='darkred',
             linewidth=2.5, zorder=10)
    ax.set_title(f'(d) Full Map -- Line-Segment Aligned\n'
                 f'Pose: ({final_t[0]:.2f},{final_t[1]:.2f},{math.degrees(final_yaw):.0f}deg)  '
                 f'Score={final_score:.3f}', fontsize=11)
    ax.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.08)

    # ── (e) Zoomed view with lines ──
    ax = axes[1, 1]; ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    ax.scatter(filtered_map[:, 0], filtered_map[:, 1], s=1.2, c='cyan', alpha=0.45, edgecolors='none')
    for i, seg in enumerate(scan_segments):
        color = plt.cm.Set1(i % 9)
        p1t = final_R @ seg['p1'] + final_t
        p2t = final_R @ seg['p2'] + final_t
        ax.plot([p1t[0], p2t[0]], [p1t[1], p2t[1]],
                '-', color=color, linewidth=5, alpha=0.95, solid_capstyle='round')
    ax.plot(final_t[0], final_t[1], 'rX', markersize=18, markeredgewidth=4, zorder=10)
    ax.arrow(final_t[0], final_t[1],
             2.5 * math.cos(final_yaw), 2.5 * math.sin(final_yaw),
             head_width=0.5, head_length=0.3, fc='red', ec='darkred',
             linewidth=2.5, zorder=10)
    hw_z = max(8, (filtered_map[:, 0].max() - filtered_map[:, 0].min()) / 2 * 1.2)
    hh_z = max(8, (filtered_map[:, 1].max() - filtered_map[:, 1].min()) / 2 * 1.2)
    ax.set_xlim(final_t[0] - hw_z, final_t[0] + hw_z)
    ax.set_ylim(final_t[1] - hh_z, final_t[1] + hh_z)
    ax.set_title(f'(e) Zoomed -- Matched Line Segments on Map', fontsize=11)
    ax.grid(True, alpha=0.12)

    # ── (f) Before/After comparison ──
    ax = axes[1, 2]; ax.set_aspect('equal')
    ax.imshow(map_disp, origin='lower', extent=extent)
    sample_n = min(60000, len(raw_map))
    raw_sample = raw_map[np.random.choice(len(raw_map), sample_n, replace=False)] if len(raw_map) > sample_n else raw_map
    ax.scatter(raw_sample[:, 0], raw_sample[:, 1], s=0.3, c='#AAAAAA', alpha=0.12,
               edgecolors='none', label=f'Raw ({len(raw_map)})')
    ax.scatter(filtered_map[:, 0], filtered_map[:, 1], s=0.8, c='cyan', alpha=0.4,
               edgecolors='none', label=f'Filtered ({len(filtered_map)})')
    ax.plot(final_t[0], final_t[1], 'rX', markersize=14, markeredgewidth=3, zorder=10)
    ax.set_xlim(final_t[0] - hw_z, final_t[0] + hw_z)
    ax.set_ylim(final_t[1] - hh_z, final_t[1] + hh_z)
    ax.set_title(f'(f) Before/After: Gray=Raw, Cyan=Filtered\n'
                 f'Interior: {pct_free:.0f}% free / {pct_occ:.0f}% occupied', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.12)

    plt.suptitle(
        f'Border Alignment v3 (Line-Segment Matching)\n'
        f'{n_frames}-frame merge | {len(scan_segments)} boundary lines | '
        f'Pose: ({final_t[0]:.2f}, {final_t[1]:.2f}, {math.degrees(final_yaw):.1f}deg)  |  '
        f'Score={final_score:.3f}  |  '
        f'IntFree={pct_free:.0f}% IntOcc={pct_occ:.0f}%  |  '
        f'Time={elapsed_total:.1f}s',
        fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output or os.path.join(os.path.dirname(args.data), 'border_match_v3.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
