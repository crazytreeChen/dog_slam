"""扫描匹配编排（ScanMatcher）—— 有状态类。

封装原 AutoInitialPoseCalibrator 中与扫描匹配相关的核心逻辑：
  - do_hierarchical_matching: 网格全局搜索 + RayCast + 精修
  - do_multistep_matching:   逐帧递推匹配 (FRF + ICP + 两级局部搜索 + ICP保护)
  - do_passive_matching:     被动匹配 (多步递推 + 偏差二次验证 + 置信度门控)
  - do_passive_reverify:     二次验证 (更多帧 + 更精细搜索)
  - try_odom_fusion_verify:  里程计融合轻量验证
  - handle_quick_mode_result: 快速模式质量评估

设计：所有方法通过 node 引用读写主节点状态，保证行为完全不变。
"""
import math
import time

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion

from .scan_utils import scan_to_points, norm_angle, quat_to_yaw
from .icp import icp_match
from .scoring import (
    score_points, score_pose_raycast, score_pose_wallhit,
    compute_wall_coverage, compute_confidence, local_search_two_stage,
)


class ScanMatcher:
    """扫描匹配器：编排 hierarchical/multistep/passive 三路径匹配。"""

    def __init__(self, logger):
        """
        logger: 日志器
        """
        self._logger = logger

    def do_hierarchical(self, node):
        """全局网格搜索 + RayCast + 精修。

        node: 主节点引用。
        与原 _do_hierarchical_matching 行为完全一致。
        """
        if node.likelihood_field is None or node.map_data is None:
            self._logger.error('地图似然场尚未加载，重新等待...')
            node.indoor_phase = node.IndoorPhase.BOOT_DELAY
            return

        t0 = time.time()
        res = node.map_info.resolution
        ox = node.map_info.origin.position.x
        oy = node.map_info.origin.position.y
        mw = node.map_info.width * res
        mh = node.map_info.height * res

        submap_pts = node._scan_to_points(node.submap1)
        if len(submap_pts) < 10:
            self._logger.error('Submap1 点云为空')
            node.indoor_phase = node.IndoorPhase.BOOT_DELAY
            return

        ds = max(1, len(submap_pts) // 800)
        pts_ds = submap_pts[::ds]

        # ── Phase 1: 网格全局搜索 (1.5m步长, 10°角步长) ──
        coarse_step = 1.5
        angle_step = 10.0
        n_angles = int(360.0 / angle_step)
        xs = np.arange(ox + 2, ox + mw - 2, coarse_step)
        ys = np.arange(oy + 2, oy + mh - 2, coarse_step)

        self._logger.info(f'[全局网格搜索] {len(xs)}x{len(ys)}x{n_angles} = '
                          f'{len(xs)*len(ys)*n_angles} 次评估...')

        map_ctx = node._map_ctx  # MapContext 实例

        all_scores = []
        for ax in xs:
            for ay in ys:
                for adeg in range(n_angles):
                    sc, _, _ = node._score_points(pts_ds, ax, ay, math.radians(adeg * angle_step))
                    if sc > -1e8:
                        all_scores.append((sc, ax, ay, adeg * angle_step))

        all_scores.sort(key=lambda x: x[0], reverse=True)
        self._logger.info(f'  粗搜: {len(all_scores)} 个有效评分')

        # NMS 去冗余
        nms_candidates = []
        for sc, ax, ay, ad in all_scores:
            dup = any(math.sqrt((ax-cx)**2 + (ay-cy)**2) < 1.5
                      and abs(ad - ca) < 20 for _, cx, cy, ca in nms_candidates)
            if not dup:
                nms_candidates.append((sc, ax, ay, ad))
                if len(nms_candidates) >= node.top_n * 2:
                    break

        # ── RayCast 重排序 (v2) ──
        if len(nms_candidates) >= 2:
            rc_scored = []
            for sc, ax, ay, ad in nms_candidates:
                ayaw = math.radians(ad)
                rc_sc_val, rc_valid, rc_total = node._score_pose_raycast(pts_ds, ax, ay, ayaw)
                if rc_sc_val > -1e8 and rc_valid >= 5:
                    rc_scored.append((sc, rc_sc_val, rc_valid / rc_total, ax, ay, ad))
            if rc_scored:
                best_rc = rc_scored[0]
                if best_rc[2] > 0.30:
                    rc_scored.sort(key=lambda x: x[0]*0.3 + x[1]*0.7, reverse=True)
                    nms_candidates = [(s, ax, ay, ad)
                                      for s, _, _, ax, ay, ad in rc_scored[:node.top_n]]
                    self._logger.info(f'  [RayCast] Best valid={best_rc[2]:.1%}, '
                                    f'rc_score={best_rc[1]:.3f}, re-ranked top-{len(nms_candidates)}')
                else:
                    self._logger.info(f'  [RayCast] valid={best_rc[2]:.1%}<30%, keep likelihood order')

        # ── Phase 2: 精细局部搜索 ──
        self._logger.info(f'  精搜: 对 Top-{min(3, len(nms_candidates))} 候选做局部搜索...')
        fine_results = []
        for rank, (_, hx, hy, had) in enumerate(nms_candidates[:3]):
            hyaw = math.radians(had)
            for dx in np.arange(-2.5, 2.51, 0.3):
                for dy in np.arange(-2.5, 2.51, 0.3):
                    for da in range(-15, 16, 3):
                        ax, ay = hx + dx, hy + dy
                        ayaw = hyaw + math.radians(da)
                        sc, _, _ = node._score_points(pts_ds, ax, ay, ayaw)
                        fine_results.append((sc, ax, ay, ayaw))

        fine_results.sort(key=lambda x: x[0], reverse=True)

        unique_candidates = []
        for sc, x, y, yaw in fine_results:
            redundant = any(
                math.sqrt((x - u[1])**2 + (y - u[2])**2) < 0.3
                and abs(norm_angle(yaw - u[3])) < math.radians(20)
                for u in unique_candidates)
            if not redundant:
                unique_candidates.append((sc, x, y, yaw))
                if len(unique_candidates) >= node.top_n:
                    break

        scores_arr = np.array([u[0] for u in unique_candidates])
        max_sc = np.max(scores_arr)
        exp_scores = np.exp(scores_arr - max_sc)
        normalized_probs = exp_scores / np.sum(exp_scores)

        node.candidates = [(float(normalized_probs[i]), u[1], u[2], u[3])
                          for i, u in enumerate(unique_candidates)]

        elapsed = time.time() - t0
        self._logger.info(f'[网格匹配] 完成。耗时: {elapsed:.2f}s。Top-N 候选:')
        for i, (prob, x, y, yaw) in enumerate(node.candidates):
            wall_cov, cov, _ = node._compute_wall_coverage(submap_pts, x, y, yaw)
            self._logger.info(f'  #{i}: 概率={prob:.3f}, ({x:.2f},{y:.2f},{math.degrees(yaw):.1f}deg) '
                            f'wall={100*wall_cov:.0f}% cov={100*cov:.0f}%')

        node._publish_candidates()

        if node.quick_mode:
            node._handle_quick_mode_result(submap_pts)
        else:
            node.indoor_phase = node.IndoorPhase.SELECTING_ACTIVE_MOTION

    def handle_quick_mode_result(self, node, scan_pts):
        """快速模式: 评估匹配质量后发布。

        与原 _handle_quick_mode_result 行为完全一致（含两个几乎相同的分支）。
        """
        best_prob, best_x, best_y, best_yaw = node.candidates[0]
        second_prob = node.candidates[1][0] if len(node.candidates) > 1 else 0.0

        wall_cov, coverage, _ = node._compute_wall_coverage(scan_pts, best_x, best_y, best_yaw)
        real_lf, _, _ = node._score_points(scan_pts, best_x, best_y, best_yaw)
        node._last_match_quality = {'wall_cov': wall_cov, 'coverage': coverage, 'score': real_lf}

        # sigma 聚类估计
        SIGMA_CLUSTER_RADIUS = 3.0
        top_n = min(10, len(node.candidates))
        best_x_c, best_y_c = node.candidates[0][1], node.candidates[0][2]
        nearby_candidates = []
        for c in node.candidates[:top_n]:
            dx = c[1] - best_x_c
            dy = c[2] - best_y_c
            if dx * dx + dy * dy <= SIGMA_CLUSTER_RADIUS ** 2:
                nearby_candidates.append(c)
        if len(nearby_candidates) >= 3:
            xs = np.array([c[1] for c in nearby_candidates])
            ys = np.array([c[2] for c in nearby_candidates])
            yaws = np.array([c[3] for c in nearby_candidates])
            self._logger.info(
                f'[sigma聚类] {len(nearby_candidates)}/{top_n}候选在{SIGMA_CLUSTER_RADIUS}m内, '
                f'排除 {top_n - len(nearby_candidates)} 个远距离候选')
        else:
            xs = np.array([c[1] for c in node.candidates[:top_n]])
            ys = np.array([c[2] for c in node.candidates[:top_n]])
            yaws = np.array([c[3] for c in node.candidates[:top_n]])
            self._logger.warn(
                f'[sigma聚类] 候选分散: 仅{len(nearby_candidates)}/{top_n}候选在'
                f'{SIGMA_CLUSTER_RADIUS}m内, 退回全量计算')
        sigma_pos = float(np.std(xs) + np.std(ys)) / 2.0
        sigma_pos = min(max(sigma_pos, 0.5), 3.0)
        sigma_yaw_rad = float(np.std([norm_angle(yw) for yw in yaws]))
        sigma_yaw_deg = math.degrees(min(max(sigma_yaw_rad, math.radians(5)), math.radians(45)))
        buf_frames = len(node.scan_buffer) if hasattr(node, 'scan_buffer') else 30

        is_unique = best_prob > 0.5 and (
            second_prob == 0.0 or best_prob / max(second_prob, 1e-9) >= 1.5)

        if is_unique and wall_cov > 0.50:
            self._logger.info(f'[快速模式] 匹配唯一且可靠 (wall={100*wall_cov:.0f}%), 发布')
            node._publish_and_finish(best_x, best_y, best_yaw,
                                     sigma_pos=sigma_pos, sigma_yaw_deg=sigma_yaw_deg,
                                     buf_frames=buf_frames,
                                     best_prob=best_prob, second_prob=second_prob)
        else:
            self._logger.warn(
                f'[快速模式] 匹配质量: unique={is_unique} wall={100*wall_cov:.0f}% '
                f'cov={100*coverage:.0f}% lf={real_lf:.3f}')
            self._logger.warn('[快速模式] 尝试发布最佳结果 (将经置信度门控审核)')
            node._publish_and_finish(best_x, best_y, best_yaw,
                                     sigma_pos=sigma_pos, sigma_yaw_deg=sigma_yaw_deg,
                                     buf_frames=buf_frames,
                                     best_prob=best_prob, second_prob=second_prob)

    def do_multistep(self, node):
        """逐帧递推匹配 (v2 增强版)。

        与原 _do_multistep_matching 行为完全一致。
        """
        if node.likelihood_field is None or node.map_data is None:
            node.indoor_phase = node.IndoorPhase.BOOT_DELAY
            return

        scans = [item[0] for item in node.scan_buffer]
        n_frames = len(scans)
        if n_frames < 3:
            self._logger.warn('[多步递推] 帧数不足, 回退到合并匹配')
            node.indoor_phase = node.IndoorPhase.ROUGH_MATCHING
            return

        t0 = time.time()
        total_wall_hits = 0
        total_valid = 0
        res = node.map_info.resolution
        ox = node.map_info.origin.position.x
        oy = node.map_info.origin.position.y
        mw = node.map_info.width * res
        mh = node.map_info.height * res
        H, W = node.map_info.height, node.map_info.width

        mu = None
        sigma = 5.0
        sigma_angle = 20.0
        min_sigma = 0.5
        decay = 0.75
        step = max(1, n_frames // 20)

        self._logger.info(f'[多步递推 v2] 开始, {n_frames}帧, 步长={step}, FRF+RayCast+WallHit')

        for i in range(0, n_frames, step):
            frame_pts = node._scan_to_points(scans[i], apply_frf_filter=True)
            if len(frame_pts) < 10:
                continue

            if mu is None:
                # ══════ 首帧: 增强全局搜索 ══════
                mu, sigma, sigma_angle, total_valid = self._multistep_first_frame(
                    node, frame_pts, ox, oy, mw, mh, res, W, H, i
                )
                if mu is None:
                    return
                wall_cov, cov, _ = node._compute_wall_coverage(frame_pts, mu[0], mu[1], mu[2])
                self._logger.info(
                    f'  帧{i:02d}(首): 全局搜索 → '
                    f'({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.0f}deg) '
                    f'wall={100*wall_cov:.0f}% σ={sigma:.1f}m')
            else:
                # ══════ 后续帧: ICP + 两级局部搜索 ══════
                mu, sigma, sigma_angle = self._multistep_subsequent_frame(
                    node, scans, i, step, mu, sigma, sigma_angle, min_sigma, decay
                )
                total_wall_hits += int(wall_cov * 100)

            total_valid += 1

        elapsed = time.time() - t0
        avg_wall = total_wall_hits / max(total_valid, 1) / 100.0

        self._logger.info(
            f'[多步递推 v2] 完成 ({elapsed:.1f}s, {total_valid}帧): '
            f'({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.1f}deg) '
            f'avg_wall={100*avg_wall:.0f}% final_σ={sigma:.2f}m')

        node.candidates = [(1.0, mu[0], mu[1], mu[2])]
        node._last_match_quality = {'wall_cov': avg_wall, 'coverage': 0.5, 'score': 1.0}

        if node.quick_mode:
            node._publish_and_finish(mu[0], mu[1], mu[2],
                                     sigma_pos=sigma, sigma_yaw_deg=sigma_angle,
                                     buf_frames=total_valid,
                                     best_prob=1.0, second_prob=0.0)
        else:
            node._publish_candidates()
            node.indoor_phase = node.IndoorPhase.SELECTING_ACTIVE_MOTION

    def _multistep_first_frame(self, node, frame_pts, ox, oy, mw, mh, res, W, H, i):
        """首帧全局搜索（从 _do_multistep_matching 提取的子方法）。

        返回: (mu, sigma, sigma_angle, total_valid) 或 (None, ...) 表示失败。
        """
        sigma = 5.0
        sigma_angle = 20.0

        ds0 = max(1, len(frame_pts) // 800)
        pts_ds = frame_pts[::ds0]
        xs = np.arange(ox + 2, ox + mw - 2, 2.0)
        ys = np.arange(oy + 2, oy + mh - 2, 2.0)

        candidates_raw = []
        n_filtered = 0
        for ax in xs:
            for ay in ys:
                for adeg in range(0, 360, 15):
                    ayaw = math.radians(adeg)
                    sc, _, _ = node._score_points(pts_ds, ax, ay, ayaw)
                    if sc < 0.3:
                        continue
                    c_y, s_y = math.cos(ayaw), math.sin(ayaw)
                    fast_n = min(100, len(pts_ds))
                    mx = c_y * pts_ds[:fast_n, 0] - s_y * pts_ds[:fast_n, 1] + ax
                    my = s_y * pts_ds[:fast_n, 0] + c_y * pts_ds[:fast_n, 1] + ay
                    ci = ((mx - ox) / res + 0.5).astype(np.int32)
                    ri = ((my - oy) / res + 0.5).astype(np.int32)
                    v = (ci >= 0) & (ci < W) & (ri >= 0) & (ri < H)
                    if int(np.sum(v)) < 20:
                        n_filtered += 1
                        continue
                    cells = node.map_data[ri[v], ci[v]]
                    n_unk = int(np.sum(cells == -1))
                    n_wall = int(np.sum(cells == 100))
                    n_valid_c = len(cells)
                    if n_unk / n_valid_c > 0.50 or n_wall / n_valid_c > 0.60:
                        n_filtered += 1
                        continue
                    candidates_raw.append((sc, ax, ay, adeg))
        candidates_raw.sort(key=lambda x: x[0], reverse=True)
        self._logger.info(f'  首帧粗搜: {len(candidates_raw)}有效, 过滤{n_filtered}无效')

        if not candidates_raw:
            self._logger.error('[多步递推] 首帧全局搜索无有效候选')
            node.indoor_phase = node.IndoorPhase.ROUGH_MATCHING
            return None, sigma, sigma_angle, 0

        # NMS
        nms_candidates = []
        for sc, ax, ay, ad in candidates_raw:
            dup = any(math.sqrt((ax - cx)**2 + (ay - cy)**2) < 1.5 and abs(ad - ca) < 20
                      for _, cx, cy, ca in nms_candidates)
            if not dup:
                nms_candidates.append((sc, ax, ay, ad))
                if len(nms_candidates) >= 16:
                    break

        # RayCast 重排序
        raycast_picked = False
        if len(nms_candidates) >= 2:
            rc_scored = []
            for sc, ax, ay, ad in nms_candidates:
                ayaw = math.radians(ad)
                rc_sc_val, rc_valid, rc_total = node._score_pose_raycast(pts_ds, ax, ay, ayaw)
                if rc_sc_val > -1e8 and rc_valid >= 5:
                    rc_scored.append((sc, rc_sc_val, rc_valid / rc_total, ax, ay, ad))
            if rc_scored:
                best_rc = rc_scored[0]
                if best_rc[2] > 0.30:
                    rc_scored.sort(key=lambda x: x[0]*0.3 + x[1]*0.7, reverse=True)
                    nms_candidates = [(s, ax, ay, ad)
                                      for s, _, _, ax, ay, ad in rc_scored[:8]]
                    raycast_picked = True
                    self._logger.info(f'  [RayCast] Best valid={best_rc[2]:.1%}, weighted re-rank')
                else:
                    self._logger.info(f'  [RayCast] valid={best_rc[2]:.1%}<30%, fallback to likelihood')

        # WallHit 回退
        if not raycast_picked:
            self._logger.info(f'  [WallHit] fallback search...')
            wh_candidates = []
            for ax in np.arange(ox + 3, ox + mw - 3, 2.0):
                for ay in np.arange(oy + 3, oy + mh - 3, 2.0):
                    for adeg in range(0, 360, 10):
                        wh, _, _ = node._score_pose_wallhit(pts_ds, ax, ay, math.radians(adeg))
                        if wh > 0:
                            wh_candidates.append((wh, ax, ay, adeg))
            if wh_candidates:
                wh_candidates.sort(key=lambda x: x[0], reverse=True)
                wh_nms = []
                for wh, ax, ay, ad in wh_candidates:
                    dup = any(math.sqrt((ax - wx)**2 + (ay - wy)**2) < 2.0
                              for _, wx, wy, _ in wh_nms)
                    if not dup:
                        wh_nms.append((wh, ax, ay, ad))
                        if len(wh_nms) >= 8:
                            break
                nms_candidates = wh_nms
                self._logger.info(
                    f'  [WallHit] Best: ({wh_nms[0][1]:.1f},{wh_nms[0][2]:.1f}) '
                    f'wall%={100*wh_nms[0][0]/2:.0f}%')

        # 两级局部搜索精修 Top-3
        best_sc = -1e9
        best_pose = None
        for sc, hx, hy, had in nms_candidates[:3]:
            pose, lf_sc = node._local_search_two_stage(
                pts_ds, hx, hy, math.radians(had),
                radius=2.0, pos_step=0.3, angle_range=12, angle_step=2)
            if lf_sc > best_sc:
                best_sc = lf_sc
                best_pose = pose

        if best_pose is None:
            self._logger.error('[多步递推] 首帧两级精修失败')
            node.indoor_phase = node.IndoorPhase.ROUGH_MATCHING
            return None, sigma, sigma_angle, 0

        return best_pose, sigma, sigma_angle, 1

    def _multistep_subsequent_frame(self, node, scans, i, step,
                                    mu, sigma, sigma_angle, min_sigma, decay):
        """后续帧 ICP+局部搜索（从 _do_multistep_matching 提取的子方法）。"""
        frame_pts = node._scan_to_points(scans[i], apply_frf_filter=True)

        prev_pts = node._scan_to_points(scans[max(0, i - step)])
        dx_i, dy_i, dyaw_i = 0.0, 0.0, 0.0
        icp_used = False
        if len(prev_pts) > 10 and len(frame_pts) > 10:
            dx_i, dy_i, dyaw_i = node._icp_match(prev_pts, frame_pts)
            if abs(dyaw_i) > math.radians(30):
                dx_i = dy_i = dyaw_i = 0.0
            else:
                icp_used = True

        c_m, s_m = math.cos(mu[2]), math.sin(mu[2])
        pred_x = mu[0] + c_m * dx_i - s_m * dy_i
        pred_y = mu[1] + s_m * dx_i + c_m * dy_i
        pred_yaw = mu[2] + dyaw_i

        search_r = min(sigma * 1.5, 3.0)
        angle_r = min(sigma_angle * 1.5, 20)
        pose, lf_sc = node._local_search_two_stage(
            frame_pts, pred_x, pred_y, pred_yaw,
            radius=search_r, pos_step=0.2,
            angle_range=int(angle_r), angle_step=2)

        wall_cov, _, _ = node._compute_wall_coverage(frame_pts, pose[0], pose[1], pose[2])
        icp_jump = math.sqrt(dx_i**2 + dy_i**2) if icp_used else 0

        if i > 3 and (wall_cov < 0.20 or (icp_jump > 2.0 and wall_cov < 0.35)):
            pose = (mu[0], mu[1], pose[2])
            lf_sc = -1
            icp_tag = "[REJECT]"
            self._logger.warn(
                f'  帧{i:02d} {icp_tag}: wall={100*wall_cov:.0f}% jump={icp_jump:.1f}m, keeping prev')
        else:
            icp_tag = "[ICP]" if icp_used else "[pred]"

        mu = pose
        sigma = max(sigma * decay, min_sigma)
        sigma_angle = max(sigma_angle * decay, 2.0)

        if i % (step * 5) == 0 or i >= len(scans) - step:
            dx = mu[0] - pred_x
            dy = mu[1] - pred_y
            corr = math.sqrt(dx**2 + dy**2)
            dya = math.degrees(abs(math.atan2(math.sin(mu[2] - pred_yaw),
                                              math.cos(mu[2] - pred_yaw))))
            self._logger.info(
                f'  帧{i:02d} {icp_tag}: pred=({pred_x:.2f},{pred_y:.2f}) '
                f'corr={corr:.3f}m,{dya:.1f}deg → '
                f'({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.0f}deg) '
                f'wall={100*wall_cov:.0f}% σ={sigma:.2f}m')

        return mu, sigma, sigma_angle

    def do_passive(self, node):
        """被动匹配 (v2 增强版)。

        与原 _do_passive_matching 行为完全一致。
        """
        if node.likelihood_field is None or not node.passive_scan_buffer:
            node.indoor_phase = node.IndoorPhase.PASSIVE_COLLECTING
            return

        t0 = time.time()
        recent = node.passive_scan_buffer[-node.passive_frame_count:]
        scans = [item[0] for item in recent]
        if len(scans) < 5:
            node.indoor_phase = node.IndoorPhase.PASSIVE_COLLECTING
            return

        mu = node.passive_best_pose
        sigma = 2.0 if mu is not None else 5.0
        sigma_angle = 10.0 if mu is not None else 20.0
        min_sigma = 0.3
        decay = 0.8
        total_wall = 0
        total_n = 0
        method = '推算'
        res = node.map_info.resolution
        ox = node.map_info.origin.position.x
        oy = node.map_info.origin.position.y
        mw = node.map_info.width * res
        mh = node.map_info.height * res

        for i, scan in enumerate(scans):
            pts = node._scan_to_points(scan, apply_frf_filter=True)
            if len(pts) < 10:
                continue

            if mu is None:
                ds0 = max(1, len(pts) // 800)
                pts_ds = pts[::ds0]
                best_sc = -1e9
                best_pose = None
                for ax in np.arange(ox + 2, ox + mw - 2, 2.0):
                    for ay in np.arange(oy + 2, oy + mh - 2, 2.0):
                        for adeg in range(0, 360, 15):
                            sc, _, _ = node._score_points(pts_ds, ax, ay, math.radians(adeg))
                            if sc > best_sc:
                                best_sc = sc
                                best_pose = (ax, ay, math.radians(adeg))
                if best_pose is None:
                    continue
                mu = best_pose
                method = "全局"
            else:
                prev_pts = node._scan_to_points(scans[max(0, i - 1)])
                dx_i = dy_i = dyaw_i = 0.0
                icp_used = False
                if len(prev_pts) > 10:
                    dx_i, dy_i, dyaw_i = node._icp_match(prev_pts, pts)
                    if abs(dyaw_i) > math.radians(30):
                        dx_i = dy_i = dyaw_i = 0.0
                    else:
                        icp_used = True
                c_m, s_m = math.cos(mu[2]), math.sin(mu[2])
                pred_x = mu[0] + c_m * dx_i - s_m * dy_i
                pred_y = mu[1] + s_m * dx_i + c_m * dy_i
                pred_yaw = mu[2] + dyaw_i

                sr = min(sigma * 1.5, 2.0)
                ar = min(sigma_angle, 15)
                pose, _ = node._local_search_two_stage(
                    pts, pred_x, pred_y, pred_yaw,
                    radius=sr, pos_step=0.2, angle_range=int(ar), angle_step=2)

                wc, _, _ = node._compute_wall_coverage(pts, pose[0], pose[1], pose[2])
                icp_jump = math.sqrt(dx_i**2 + dy_i**2) if icp_used else 0
                if i > 3 and (wc < 0.20 or (icp_jump > 2.0 and wc < 0.35)):
                    pose = (mu[0], mu[1], pose[2])
                    method = "局部[REJECT]"
                else:
                    method = "局部"

                mu = pose
                sigma = max(sigma * decay, min_sigma)
                sigma_angle = max(sigma_angle * decay, 2.0)

            wc, _, _ = node._compute_wall_coverage(pts, mu[0], mu[1], mu[2])
            total_wall += wc
            total_n += 1

        avg_wall = total_wall / max(total_n, 1)
        node.passive_best_pose = mu
        elapsed = time.time() - t0
        ts = node.get_clock().now().nanoseconds / 1e9
        node.passive_pose_history.append((ts, mu[0], mu[1], mu[2], avg_wall))

        # 低覆盖率 debug 输出
        if avg_wall < 0.25:
            self._logger.info(
                f'[被动低覆盖] {elapsed:.1f}s wall={100*avg_wall:.0f}% '
                f'({mu[0]:.2f},{mu[1]:.2f},{math.degrees(mu[2]):.1f}deg) '
                f'buf={len(node.passive_scan_buffer)}帧 -- 继续发布')
        if avg_wall < 0.25 and hasattr(node, 'debug_auto_pose_pub'):
            msg = PoseWithCovarianceStamped()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.header.frame_id = node.map_frame
            msg.pose.pose.position = Point(x=mu[0], y=mu[1], z=0.0)
            qz_d = math.sin(mu[2] / 2.0)
            qw_d = math.cos(mu[2] / 2.0)
            msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz_d, w=qw_d)
            node.debug_auto_pose_pub.publish(msg)

        # 偏差检查 + 二次验证 + 置信度门控 + 发布
        self._passive_finalize(node, mu, avg_wall, total_n, method, scans, elapsed)

    def _passive_finalize(self, node, mu, avg_wall, total_n, method, scans, elapsed):
        """被动匹配的偏差检查、置信度门控和发布（从 _do_passive_matching 提取）。"""
        pose_to_publish = mu
        publish_wall = avg_wall
        recheck_passed = False

        if node.last_calibrated_pose is not None:
            last_x, last_y, last_yaw = node.last_calibrated_pose
            dev_dist = math.sqrt((mu[0] - last_x)**2 + (mu[1] - last_y)**2)
            dev_yaw = abs(norm_angle(mu[2] - last_yaw))
            if (dev_dist > node.passive_deviation_threshold or
                    math.degrees(dev_yaw) > node.passive_deviation_yaw_threshold):
                self._logger.warn(
                    f'[被动偏差] 与上次位姿偏差过大: Δd={dev_dist:.2f}m Δyaw={math.degrees(dev_yaw):.1f}°, '
                    f'触发二次验证...')
                mu2, wall2, elapsed2 = self.do_passive_reverify(node)
                if mu2 is not None:
                    dev2 = math.sqrt((mu[0] - mu2[0])**2 + (mu[1] - mu2[1])**2)
                    dev2_yaw = abs(norm_angle(mu[2] - mu2[2]))
                    recheck_passed = True
                    if wall2 > avg_wall:
                        pose_to_publish = mu2
                        publish_wall = wall2
                        self._logger.info(
                            f'[被动二次验证] 辅优于主: 主({mu[0]:.2f},{mu[1]:.2f},'
                            f'{math.degrees(mu[2]):.1f}° wall={100*avg_wall:.0f}%) ↔ '
                            f'辅({mu2[0]:.2f},{mu2[1]:.2f},{math.degrees(mu2[2]):.1f}° '
                            f'wall={100*wall2:.0f}%) Δd={dev2:.2f}m Δyaw={math.degrees(dev2_yaw):.1f}°')
                    else:
                        self._logger.info(
                            f'[被动二次验证] 主优于辅: 主({mu[0]:.2f},{mu[1]:.2f},'
                            f'{math.degrees(mu[2]):.1f}° wall={100*avg_wall:.0f}%) ↔ '
                            f'辅({mu2[0]:.2f},{mu2[1]:.2f},{math.degrees(mu2[2]):.1f}° '
                            f'wall={100*wall2:.0f}%) Δd={dev2:.2f}m Δyaw={math.degrees(dev2_yaw):.1f}°')
                else:
                    self._logger.warn('[被动二次验证] 二次匹配失败, 保留主结果')

        # 置信度评估
        pos_sigma = min(0.8 / max(publish_wall, 0.1), 2.0)
        yaw_sigma_deg = min(20.0 / max(publish_wall, 0.1), 30.0)

        last_scan_pts = (node._scan_to_points(scans[-1], apply_frf_filter=True)
                         if scans else None)
        est_lf = 0.5
        if last_scan_pts is not None and len(last_scan_pts) > 10:
            est_lf, _, _ = node._score_points(last_scan_pts, pose_to_publish[0],
                                              pose_to_publish[1], pose_to_publish[2])

        confidence = node._compute_confidence(
            wall_ratio=publish_wall, lf_score=est_lf, coverage=0.5,
            sigma_pos=pos_sigma, sigma_yaw_deg=yaw_sigma_deg,
            buf_frames=total_n, best_prob=1.0, second_prob=0.0)

        conf_pct = 100 * confidence
        th_up = node.confidence_threshold_update
        th_pub = node.confidence_threshold_publish

        if recheck_passed:
            th_pub = min(th_pub, node.confidence_threshold_amcl)
            self._logger.info(
                f'[被动二次验证] 两候选一致 → 发布阈值放宽至 {100*th_pub:.0f}%')

        # bootstrap 引导
        if node.last_calibrated_pose is None and confidence >= node.confidence_threshold_amcl:
            node.passive_bootstrap_failures += 1
            if node.passive_bootstrap_failures >= node.passive_bootstrap_threshold:
                th_pub = min(th_pub, node.confidence_threshold_amcl)
                self._logger.info(
                    f'[被动引导] 连续 {node.passive_bootstrap_failures} 次无历史位姿匹配, '
                    f'阈值放宽至 {100*th_pub:.0f}% 以打破死锁')
            else:
                self._logger.info(
                    f'[被动引导] 第 {node.passive_bootstrap_failures}/'
                    f'{node.passive_bootstrap_threshold} 次等待 (conf={conf_pct:.0f}%)')
        elif node.last_calibrated_pose is not None:
            node.passive_bootstrap_failures = 0

        # 置信度门控
        if confidence < th_pub:
            self._logger.warn(
                f'[被动门控✗] 拒绝发布: conf={conf_pct:.0f}% < {100*th_pub:.0f}% '
                f'(wall={100*publish_wall:.0f}% lf={est_lf:.2f} '
                f'σ=({pos_sigma:.2f}m,{yaw_sigma_deg:.1f}deg) buf={total_n}帧)')
            if node.debug_comparison_mode:
                node._debug_compare_all_references(
                    pose_to_publish[0], pose_to_publish[1], pose_to_publish[2],
                    source=f'被动{method}(拒绝)')
            node.indoor_phase = node.IndoorPhase.PASSIVE_COLLECTING
            return

        gate_level = '✓高' if confidence >= th_up else '○中'

        # 发布
        cov = [0.0] * 36
        cov[0] = pos_sigma ** 2
        cov[7] = pos_sigma ** 2
        cov[35] = math.radians(yaw_sigma_deg) ** 2

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = node.map_frame
        msg.pose.pose.position = Point(x=pose_to_publish[0], y=pose_to_publish[1], z=0.0)
        qz = math.sin(pose_to_publish[2] / 2.0)
        qw = math.cos(pose_to_publish[2] / 2.0)
        msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

        if node.auto_publish_initial_pose:
            msg.pose.covariance = cov
            node.initialpose_pub.publish(msg)
            node._publish_cooldown_until = (node.get_clock().now().nanoseconds * 1e-9
                                            + node.passive_publish_cooldown)
            node.passive_bootstrap_failures = 0
            set_reference = (confidence >= th_up) or (
                node.last_calibrated_pose is None and confidence >= th_pub)
            if set_reference:
                node.last_calibrated_pose = (pose_to_publish[0], pose_to_publish[1], pose_to_publish[2])
                if node.current_odom is not None:
                    curr_pos = node.current_odom.pose.pose.position
                    curr_yaw = quat_to_yaw(node.current_odom.pose.pose.orientation)
                    node.passive_last_odom = (curr_pos.x, curr_pos.y, curr_yaw)
                    node.passive_odom_accum_dist = 0.0
            status = '发布→/initialpose ' + gate_level + ('[二次验证]' if recheck_passed else '')
        else:
            if hasattr(node, 'debug_auto_pose_pub'):
                node.debug_auto_pose_pub.publish(msg)
            status = 'debug模式' + ('[二次验证]' if recheck_passed else '')
            node.passive_bootstrap_failures = 0

        self._logger.info(
            f'[被动{method}] {elapsed:.1f}s {total_n}帧: '
            f'({pose_to_publish[0]:.2f},{pose_to_publish[1]:.2f},'
            f'{math.degrees(pose_to_publish[2]):.1f}deg) '
            f'conf={conf_pct:.0f}% wall={100*publish_wall:.0f}% '
            f'σ={pos_sigma:.2f}m yσ={yaw_sigma_deg:.1f}deg '
            f'buf={len(node.passive_scan_buffer)}帧 [{status}]')

        if not node.auto_publish_initial_pose and node.debug_comparison_mode:
            node._debug_compare_all_references(
                pose_to_publish[0], pose_to_publish[1], pose_to_publish[2],
                source=f'被动{method}')
        node.indoor_phase = node.IndoorPhase.PASSIVE_COLLECTING

    def do_passive_reverify(self, node):
        """二次验证: 使用更多帧 + 更精细搜索重新匹配。

        返回 (pose, wall, elapsed) 或 (None, 0, 0)。
        与原 _do_passive_reverify 一致。
        """
        if not node.passive_scan_buffer or len(node.passive_scan_buffer) < 10:
            return None, 0.0, 0.0
        t0 = time.time()
        extra_frames = max(node.passive_frame_count * 2, 30)
        recent = node.passive_scan_buffer[-min(extra_frames, len(node.passive_scan_buffer)):]
        scans = [item[0] for item in recent]
        if len(scans) < 10:
            return None, 0.0, 0.0

        mu = node.passive_best_pose
        sigma = 1.5
        sigma_angle = 8.0
        min_sigma = 0.2
        decay = 0.7
        total_wall = 0
        total_n = 0
        res = node.map_info.resolution
        ox = node.map_info.origin.position.x
        oy = node.map_info.origin.position.y
        mw = node.map_info.width * res
        mh = node.map_info.height * res

        for i, scan in enumerate(scans):
            pts = node._scan_to_points(scan, apply_frf_filter=True)
            if len(pts) < 10:
                continue

            if mu is None:
                ds0 = max(1, len(pts) // 800)
                pts_ds = pts[::ds0]
                best_sc = -1e9
                best_pose = None
                for ax in np.arange(ox + 1, ox + mw - 1, 1.5):
                    for ay in np.arange(oy + 1, oy + mh - 1, 1.5):
                        for adeg in range(0, 360, 10):
                            sc, _, _ = node._score_points(pts_ds, ax, ay, math.radians(adeg))
                            if sc > best_sc:
                                best_sc = sc
                                best_pose = (ax, ay, math.radians(adeg))
                if best_pose is None:
                    continue
                mu = best_pose
            else:
                prev_pts = node._scan_to_points(scans[max(0, i - 1)])
                dx_i = dy_i = dyaw_i = 0.0
                icp_used = False
                if len(prev_pts) > 10:
                    dx_i, dy_i, dyaw_i = node._icp_match(prev_pts, pts)
                    if abs(dyaw_i) > math.radians(30):
                        dx_i = dy_i = dyaw_i = 0.0
                    else:
                        icp_used = True
                c_m, s_m = math.cos(mu[2]), math.sin(mu[2])
                pred_x = mu[0] + c_m * dx_i - s_m * dy_i
                pred_y = mu[1] + s_m * dx_i + c_m * dy_i
                pred_yaw = mu[2] + dyaw_i

                sr = min(sigma * 1.5, 1.5)
                ar = min(sigma_angle, 10)
                pose, _ = node._local_search_two_stage(
                    pts, pred_x, pred_y, pred_yaw,
                    radius=sr, pos_step=0.15, angle_range=int(ar), angle_step=1)
                wc, _, _ = node._compute_wall_coverage(pts, pose[0], pose[1], pose[2])
                icp_jump = math.sqrt(dx_i**2 + dy_i**2) if icp_used else 0
                if i > 3 and (wc < 0.15 or (icp_jump > 2.0 and wc < 0.30)):
                    pose = (mu[0], mu[1], pose[2])
                mu = pose
                sigma = max(sigma * decay, min_sigma)
                sigma_angle = max(sigma_angle * decay, 1.5)

            wc, _, _ = node._compute_wall_coverage(pts, mu[0], mu[1], mu[2])
            total_wall += wc
            total_n += 1

        avg_wall = total_wall / max(total_n, 1)
        elapsed = time.time() - t0
        return mu, avg_wall, elapsed

    def try_odom_fusion_verify(self, node):
        """里程计融合轻量验证。

        返回 (success, pose_or_None, reason)。
        与原 _try_odom_fusion_verify 一致。
        """
        if not node.passive_odom_fusion_enabled:
            return False, None, 'fusion disabled'
        if node.last_calibrated_pose is None:
            return False, None, 'no last calibrated pose'
        if node.passive_last_odom is None:
            return False, None, 'no passive_last_odom (first run)'
        if node.current_odom is None:
            return False, None, 'no current odom'
        if node.likelihood_field is None:
            return False, None, 'no likelihood field'
        if node.current_scan is None:
            return False, None, 'no current scan'

        now = node.get_clock().now().nanoseconds * 1e-9
        in_cooldown = now < node._publish_cooldown_until
        if in_cooldown:
            remaining = node._publish_cooldown_until - now
            odom_threshold = max(node.passive_odom_max_move * 2.0, 10.0)
            wall_threshold = max(node.passive_fusion_wall_threshold * 0.5, 0.10)
            skip_amcl = True
        else:
            odom_threshold = node.passive_odom_max_move
            wall_threshold = node.passive_fusion_wall_threshold
            skip_amcl = False

        last_x, last_y, last_yaw = node.passive_last_odom
        curr_pos = node.current_odom.pose.pose.position
        curr_yaw = quat_to_yaw(node.current_odom.pose.pose.orientation)
        dx_o = curr_pos.x - last_x
        dy_o = curr_pos.y - last_y
        dyaw_o = norm_angle(curr_yaw - last_yaw)

        move_dist = math.sqrt(dx_o**2 + dy_o**2)
        node.passive_odom_accum_dist += move_dist

        if node.passive_odom_accum_dist > odom_threshold:
            return False, None, f'odom accum {node.passive_odom_accum_dist:.1f}m > {odom_threshold:.1f}m'

        px, py, pyaw = node.last_calibrated_pose
        c_y, s_y = math.cos(pyaw), math.sin(pyaw)
        pred_x = px + dx_o * c_y - dy_o * s_y
        pred_y = py + dx_o * s_y + dy_o * c_y
        pred_yaw = norm_angle(pyaw + dyaw_o)
        pred_pose = (pred_x, pred_y, pred_yaw)

        pts = node._scan_to_points(node.current_scan, apply_frf_filter=True)
        if len(pts) < 10:
            return False, None, 'insufficient scan points'

        wall_ratio, valid_ratio, _ = node._compute_wall_coverage(pts, pred_x, pred_y, pred_yaw)
        if wall_ratio < wall_threshold:
            reason = f'wall {wall_ratio:.2f} < {wall_threshold:.2f}'
            if in_cooldown:
                reason += f' [冷却中 {remaining:.0f}s]'
            return False, None, reason

        lf_score, hit_rate, n_valid = node._score_points(pts, pred_x, pred_y, pred_yaw)
        if lf_score < node.passive_likelihood_threshold:
            reason = f'likelihood {lf_score:.3f} < {node.passive_likelihood_threshold:.2f}'
            if in_cooldown:
                reason += f' [冷却中 {remaining:.0f}s]'
            return False, None, reason

        if node.passive_amcl_cross_check and not skip_amcl and node.latest_amcl_pose is not None:
            ax, ay, _ = node.latest_amcl_pose
            dev = math.sqrt((pred_x - ax)**2 + (pred_y - ay)**2)
            if dev > node.passive_amcl_max_deviation:
                return False, None, f'AMCL deviation {dev:.2f}m > {node.passive_amcl_max_deviation:.1f}m'

        reason = f'ok wall={wall_ratio:.2f} lf={lf_score:.3f}'
        if in_cooldown:
            reason += f' [冷却中 {remaining:.0f}s, 放宽验证]'
        return True, pred_pose, reason
