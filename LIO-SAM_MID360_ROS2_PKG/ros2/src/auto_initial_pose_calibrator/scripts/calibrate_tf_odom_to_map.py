#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_tf_odom_to_map.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
通过扫描-地图匹配反算 tf_odom_to_map，修正 GT 参考原点。

问题: 采集数据时未设置初始位姿，tf_odom_to_map = (0,0,0)，导致 GT 坐标无效。
方案: 用多候选递推匹配找到机器人在地图上的真实位姿，反算正确的 tf_odom_to_map。

原理:
  matched_pose = tf_odom_to_map ∘ frame_tfs[0]
  => tf_odom_to_map = matched_pose ∘ inverse(frame_tfs[0])

用法:
  python calibrate_tf_odom_to_map.py --data scan_viz/debug_match_data_1.npz
  python calibrate_tf_odom_to_map.py --data scan_viz/debug_match_data_1.npz --output scan_viz/fixed_1.npz
  python calibrate_tf_odom_to_map.py --all   # 处理所有 debug_match_data_*.npz
"""

import os, sys, math, argparse, shutil
import numpy as np

# 复用已有模块
import importlib.util

def load_module(filename, alias):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compose_tf(a, b):
    """复合两个变换: a ∘ b, 各为 (x, y, yaw)"""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (
        a[0] + c * b[0] - s * b[1],
        a[1] + s * b[0] + c * b[1],
        a[2] + b[2],
    )


def inverse_tf(t):
    """变换求逆: (x, y, yaw) → (-R^T @ t, -yaw)"""
    c, s = math.cos(t[2]), math.sin(t[2])
    return (
        -c * t[0] - s * t[1],
         s * t[0] - c * t[1],
        -t[2],
    )


def run_matching(npz_path, topk=10):
    """
    对单个 NPZ 跑多候选递推匹配，返回最优位姿 (x, y, yaw) in map frame。
    复用 scan_map_overlay.py 的核心逻辑。
    """
    ms = load_module('opencode_multistep_localizer.py', 'ms')
    overlay = load_module('scan_map_overlay.py', 'overlay')

    # 加载数据
    map_data, info, tf_gt, frame_tfs, frame_ranges, angle_min, angle_inc = ms.load_data(npz_path)
    frame_pts = [(ms.frame_to_odom_pts(r, tf, angle_min, angle_inc), tf)
                 for r, tf in zip(frame_ranges, frame_tfs)]
    lf = ms.build_likelihood_field(map_data, info)

    # 首帧全局搜索 → Top-K 候选
    pts0 = frame_pts[0][0]
    if len(pts0) < 10:
        print("[ERROR] 首帧点云为空"); return None

    cand0 = ms.global_search_first_frame(pts0, lf, map_data, info, top_k=topk)
    if not cand0:
        print("[ERROR] 首帧无候选"); return None

    # WallHit fallback
    best_rc = ms.score_pose_raycast(pts0, cand0[0][1], cand0[0][2],
                                     math.radians(cand0[0][3]), map_data, info)
    rc_valid_rate = best_rc[2] / max(best_rc[1], 1) if best_rc[0] > -1e8 else 0
    if rc_valid_rate < 0.30:
        print(f"  [WallHit fallback] 射线有效率={rc_valid_rate:.0%}, 启动墙命中搜索...")
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
                if len(wh_nms) >= topk:
                    break
            cand0 = wh_nms
            print(f"  [WallHit] {len(wh_nms)} 个候选")

    # 每个候选独立递推
    results = []
    for k, (sc0, hx, hy, had) in enumerate(cand0):
        seed, _ = ms.local_search(pts0, hx, hy, math.radians(had), lf, info,
                                   radius=2.0, pos_step=0.3)
        final, hist, mscore, mwall, unk = overlay.recurse_from_seed(
            frame_pts, seed, ms, lf, map_data, info, decay=0.7)
        # 几何约束
        gray_end, wall_free, sensor_nf = overlay.geometry_constraints(
            final, frame_ranges, frame_tfs, angle_min, angle_inc, map_data, info)
        # Wallhit
        all_pts = np.vstack([fp[0] for fp in frame_pts if len(fp[0]) > 0])
        wh_score, _, _ = ms.score_pose_wallhit(all_pts, final[0], final[1], final[2], map_data, info)
        results.append({'final': final, 'mean_score': mscore, 'mean_wall': mwall,
                        'wallhit': max(wh_score, 0.0), 'sensor_nf': sensor_nf,
                        'gray_end': gray_end, 'wall_free': wall_free})
        print(f"  候选#{k}: ({final[0]:.1f},{final[1]:.1f},{math.degrees(final[2]):.0f}°) "
              f"wallhit={max(wh_score,0):.2f} 均分={mscore:.2f}")

    # 去重 + 排名
    dedup = []
    for r in sorted(results, key=lambda r: -r['wallhit']):
        if not any(math.hypot(r['final'][0]-d['final'][0], r['final'][1]-d['final'][1]) < 2.0
                   and abs(math.atan2(math.sin(r['final'][2]-d['final'][2]),
                                      math.cos(r['final'][2]-d['final'][2]))) < math.radians(20)
                   for d in dedup):
            dedup.append(r)

    for r in dedup:
        r['combined'] = (r['mean_score'] * 1.0 + r['wallhit'] * 0.5
                         - r['wall_free'] * 0.05 - r['sensor_nf'] * 0.02)
    dedup.sort(key=lambda r: r['combined'], reverse=True)
    best = dedup[0]
    return best['final']


def calibrate_npz(npz_path, output_path=None, topk=10):
    """对单个 NPZ 文件执行校准并保存"""
    print("=" * 60)
    print(f"校准: {npz_path}")
    print("=" * 60)

    d = np.load(npz_path, allow_pickle=True)
    frame_tfs = d['frame_tfs']
    old_tf = d['tf_odom_to_map']
    print(f"  原始 tf_odom_to_map: ({old_tf[0]:.3f}, {old_tf[1]:.3f}, {math.degrees(old_tf[2]):.1f}°)")
    print(f"  frame_tfs[0]: ({frame_tfs[0][0]:.3f}, {frame_tfs[0][1]:.3f}, {math.degrees(frame_tfs[0][2]):.1f}°)")

    # 跑匹配
    matched_pose = run_matching(npz_path, topk=topk)
    if matched_pose is None:
        print("  [FAIL] 匹配失败，跳过"); return False

    print(f"\n  匹配位姿 (map系): ({matched_pose[0]:.3f}, {matched_pose[1]:.3f}, "
          f"{math.degrees(matched_pose[2]):.1f}°)")

    # 反算 tf_odom_to_map = matched_pose ∘ inverse(frame_tfs[0])
    inv_f0 = inverse_tf(frame_tfs[0])
    new_tf = compose_tf(matched_pose, inv_f0)

    print(f"  新 tf_odom_to_map: ({new_tf[0]:.3f}, {new_tf[1]:.3f}, {math.degrees(new_tf[2]):.1f}°)")

    # 保存修正后的 NPZ
    if output_path is None:
        base, ext = os.path.splitext(npz_path)
        output_path = base + '_calibrated' + ext

    # 复制原文件并替换 tf_odom_to_map
    data = dict(np.load(npz_path, allow_pickle=True))
    data['tf_odom_to_map'] = np.array(new_tf)
    # 保留原始值作为参考
    data['tf_odom_to_map_original'] = np.array(old_tf)
    data['calibration_method'] = 'multi_candidate_recursive_matching'

    np.savez(output_path, **data)
    print(f"  [OK] 已保存: {output_path}")

    # 验证: 用新 tf 重算首帧 GT 位姿
    gt_new = compose_tf(new_tf, frame_tfs[0])
    print(f"  验证 - 首帧 GT 位姿 (新): ({gt_new[0]:.3f}, {gt_new[1]:.3f}, {math.degrees(gt_new[2]):.1f}°)")
    print(f"  验证 - 匹配位姿:          ({matched_pose[0]:.3f}, {matched_pose[1]:.3f}, {math.degrees(matched_pose[2]):.1f}°)")
    err = math.hypot(gt_new[0]-matched_pose[0], gt_new[1]-matched_pose[1])
    print(f"  验证 - 偏差: {err:.4f}m (应接近 0)")
    return True


def main():
    parser = argparse.ArgumentParser(description='通过扫描匹配反算 tf_odom_to_map')
    parser.add_argument('--data', type=str, default=None, help='单个 NPZ 文件路径')
    parser.add_argument('--output', type=str, default=None, help='输出 NPZ 路径')
    parser.add_argument('--topk', type=int, default=10, help='首帧保留候选数')
    parser.add_argument('--all', action='store_true', help='处理所有 debug_match_data_*.npz')
    parser.add_argument('--overwrite', action='store_true', help='直接覆盖原文件 (不创建 _calibrated)')
    args = parser.parse_args()

    if args.all:
        scan_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scan_viz')
        npz_files = sorted([
            os.path.join(scan_dir, f) for f in os.listdir(scan_dir)
            if f.startswith('debug_match_data_') and f.endswith('.npz') and '_calibrated' not in f
        ])
        if not npz_files:
            print("[ERROR] 未找到 NPZ 文件"); sys.exit(1)
        print(f"找到 {len(npz_files)} 个文件待校准\n")
        ok = 0
        for npz in npz_files:
            out = npz if args.overwrite else None
            if calibrate_npz(npz, output_path=out, topk=args.topk):
                ok += 1
            print()
        print(f"完成: {ok}/{len(npz_files)} 成功")
    elif args.data:
        out = args.output
        if args.overwrite and not out:
            out = args.data
        calibrate_npz(args.data, output_path=out, topk=args.topk)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
