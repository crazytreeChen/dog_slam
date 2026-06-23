"""多帧时序一致性过滤 —— 纯函数。

剔除只在少数帧出现的动态障碍物（人、动物等），仅保留多帧稳定点。
仅依赖 numpy + 可选 scipy，无 ROS 依赖。
与原 AutoInitialPoseCalibrator._temporal_consistency_filter 行为一致。
"""
import numpy as np


def temporal_consistency_filter(all_points, frame_ids, min_frames, radius, logger=None):
    """多帧时序一致性过滤：剔除只在少数帧出现的动态障碍物（人、动物等）。

    核心原理：
      - 静态墙壁/障碍物 → 多个不同帧的激光都会打到同一位置
      - 动态行人/动物 → 只在 1-2 帧出现在某位置，随后移走
      → 对每个点，检查其邻域内的点来自多少个不同帧，
        若少于 min_frames 帧则剔除。

    输入：
      all_points: (N, 2) numpy array, 所有帧投影到参考帧系后的点坐标
      frame_ids:  (N,)  numpy array, 每个点所属的源帧索引 (0 ~ n_frames-1)
      min_frames: int,    邻域内最少需要来自多少不同帧
      radius:     float,  邻域搜索半径 (m)
      logger:     可选日志器（rclpy logger 或 logging.Logger）
    输出：
      mask: (N,) boolean numpy array, True=保留, False=剔除
    """
    n_points = len(all_points)
    if n_points < min_frames:
        if logger is not None:
            logger.warn('[时序过滤] 总点数过少，跳过过滤')
        return np.ones(n_points, dtype=bool)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(all_points)
        # 对每个点查询半径内的邻居索引
        neighbors_list = tree.query_ball_point(all_points, radius)

        # 统计每个点的邻居来自多少不同帧
        distinct_count = np.zeros(n_points, dtype=np.int32)
        for i in range(n_points):
            if len(neighbors_list[i]) == 0:
                distinct_count[i] = 1  # 自邻域
            else:
                distinct_count[i] = len(np.unique(frame_ids[neighbors_list[i]]))

        mask = distinct_count >= min_frames
        n_kept = np.sum(mask)
        n_removed = n_points - n_kept

        if logger is not None:
            logger.info(
                f'[时序过滤] 总点={n_points}, 保留={n_kept}({100*n_kept/max(1,n_points):.1f}%), '
                f'剔除动态/噪声={n_removed}, '
                f'(最少帧数={min_frames}, 半径={radius}m)'
            )

            # 按帧统计过滤效果
            unique_frames = np.unique(frame_ids)
            if len(unique_frames) <= 10:
                for fid in unique_frames:
                    fmask = frame_ids == fid
                    f_kept = np.sum(mask[fmask])
                    f_total = np.sum(fmask)
                    logger.debug(
                        f'    帧[{fid}]: 保留 {f_kept}/{f_total} '
                        f'({100*f_kept/max(1,f_total):.0f}%)'
                    )

        return mask

    except ImportError:
        if logger is not None:
            logger.warn('[时序过滤] scipy 未安装，回退到纯空间离群点过滤')
        # 回退：简单的空间离群点过滤
        if len(all_points) < 10:
            return np.ones(n_points, dtype=bool)

        # 对每个点找最近邻，若最近邻距离超过中位数的3倍则剔除
        dists = []
        for i in range(n_points):
            dx = all_points[:, 0] - all_points[i, 0]
            dy = all_points[:, 1] - all_points[i, 1]
            d = np.sqrt(dx * dx + dy * dy)
            d[i] = np.inf
            dists.append(np.min(d))
        dists = np.array(dists)
        med = np.median(dists)
        if med > 0:
            mask = dists < 3.0 * med
            return mask
        return np.ones(n_points, dtype=bool)
