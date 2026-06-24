import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def visualize_match(map_data, origin_x, origin_y, resolution, scan_points, best_pose):
    """
    可视化匹配结果
    """
    cx, cy, theta, flip = best_pose
    
    points = scan_points.copy()
    if flip: points[:, 0] *= -1
    
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    rot_x = points[:, 0] * cos_t - points[:, 1] * sin_t
    rot_y = points[:, 0] * sin_t + points[:, 1] * cos_t
    
    trans_x, trans_y = rot_x + cx, rot_y + cy
    pix_x = ((trans_x - origin_x) / resolution).astype(int)
    pix_y = ((trans_y - origin_y) / resolution).astype(int)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(map_data, cmap='gray', origin='lower')
    plt.scatter(pix_x, pix_y, c='cyan', s=1.0, alpha=0.7, label='Aligned Scan')
    plt.plot((cx - origin_x)/resolution, (cy - origin_y)/resolution, 'rx', markersize=10, label='Robot Pose')
    plt.legend()
    plt.title(f"Result: X={cx:.2f}, Y={cy:.2f}, Theta={np.rad2deg(theta):.1f}°, Flip={flip}")
    plt.show()

def is_occluded(robot_pix, scan_pix, binary_map):
    """
    硬性遮挡检测：检查机器人与扫描点之间是否有墙壁
    """
    pts = np.linspace(robot_pix, scan_pix, num=10).astype(int)
    # 只要路径上有任何像素值为 255 (墙)，则判定为穿墙
    if np.any(binary_map[pts[:, 1], pts[:, 0]] == 255):
        return True
    return False

def load_and_match_robust(npz_path, prior_pose=None):
    data = np.load(npz_path)
    resolution = data['map_resolution'].item()
    width, height = data['map_width'].item(), data['map_height'].item()
    origin_x, origin_y = data['map_origin_x'].item(), data['map_origin_y'].item()
    map_data = data['map_data'].reshape((height, width))
    
    # 强制将墙壁二值化，作为遮挡物
    binary_map = np.zeros((height, width), dtype=np.uint8)
    binary_map[map_data > 50] = 255
    
    walls_mask = (map_data > 50) & (map_data <= 100)
    dist_map = cv2.distanceTransform((1 - walls_mask.astype(np.uint8)) * 255, cv2.DIST_L2, 5)
    
    sigma = 1.0 * resolution 
    likelihood_map = np.exp(-0.5 * (dist_map * resolution)**2 / (sigma**2))
    
    scan_points = data['scan_points_odom']
    scan_center = np.mean(scan_points, axis=0)
    scan_points -= scan_center

    print("启动强制遮挡抑制匹配引擎...")
    best_score = -float('inf') 
    best_pose = (0.0, 0.0, 0.0, False)

    search_range_x = (0, width) if prior_pose is None else (max(0, int((prior_pose[0]-5-origin_x)/resolution)), min(width, int((prior_pose[0]+5-origin_x)/resolution)))
    search_range_y = (0, height) if prior_pose is None else (max(0, int((prior_pose[1]-5-origin_y)/resolution)), min(height, int((prior_pose[1]+5-origin_y)/resolution)))

    for flip in [False, True]:
        current_points = scan_points.copy()
        if flip: current_points[:, 0] *= -1
        
        for theta in np.arange(-np.pi, np.pi, np.deg2rad(10)):
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rot_x = current_points[:, 0] * cos_t - current_points[:, 1] * sin_t
            rot_y = current_points[:, 0] * sin_t + current_points[:, 1] * cos_t
            
            for dx in range(search_range_x[0], search_range_x[1], 15):
                for dy in range(search_range_y[0], search_range_y[1], 15):
                    cx, cy = dx * resolution + origin_x, dy * resolution + origin_y
                    rx, ry = rot_x + cx, rot_y + cy
                    px = ((rx - origin_x) / resolution).astype(int)
                    py = ((ry - origin_y) / resolution).astype(int)
                    
                    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
                    
                    # 几何匹配得分
                    score = np.sum(likelihood_map[py[valid], px[valid]])
                    
                    # 【核心优化】：如果得分足够高，触发遮挡检测
                    if score > 50:
                        # 采样几个点检查是否穿墙
                        sample_indices = np.random.choice(np.where(valid)[0], size=min(10, np.sum(valid)), replace=False)
                        for idx in sample_indices:
                            if is_occluded((dx, dy), (px[idx], py[idx]), binary_map):
                                score -= 500  # 毁灭性惩罚
                                break
                    
                    if score > best_score:
                        best_score = score
                        best_pose = (cx, cy, theta, flip)

    print(f"匹配完成: X={best_pose[0]:.3f}, Y={best_pose[1]:.3f}, 翻转={best_pose[3]}")
    visualize_match(map_data, origin_x, origin_y, resolution, scan_points, best_pose)
    return best_pose, map_data, origin_x, origin_y, resolution, scan_points

if __name__ == "__main__":
    best_pose, map_data, origin_x, origin_y, resolution, scan_points = load_and_match_robust("../scan_viz/debug_match_data_1.npz")

