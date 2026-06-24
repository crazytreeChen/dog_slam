import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time

def load_and_match_robust(npz_path):
    print(f"正在读取数据: {npz_path}")
    data = np.load(npz_path)
    
    # 1. 提取地图数据
    resolution = data['map_resolution'].item()
    width = data['map_width'].item()
    height = data['map_height'].item()
    origin_x = data['map_origin_x'].item()
    origin_y = data['map_origin_y'].item()
    
    map_data = data['map_data'].reshape((height, width))
    
    # === 构建得分场 (Likelihood Map) ===
    # 用于奖励：雷达点打在墙上加分
    walls_mask = (map_data > 50) & (map_data <= 100)
    binary_walls = np.zeros_like(map_data, dtype=np.uint8)
    binary_walls[walls_mask] = 255
    
    dist_map = cv2.distanceTransform(255 - binary_walls, cv2.DIST_L2, 5)
    max_dist_px = 10.0 
    likelihood_map = np.clip(max_dist_px - dist_map, 0, max_dist_px) / max_dist_px
    likelihood_map = likelihood_map.astype(np.float32)
    
    # === 【核心核武器】：构建穿墙惩罚场 (Ray Penalty Map) ===
    # 用于惩罚：任何雷达射线如果划过这些区域，将遭受毁灭性扣分
    ray_penalty_map = np.zeros_like(map_data, dtype=np.float32)
    # 1. 实体墙壁 (不允许穿透)
    ray_penalty_map[walls_mask] = 1.0 
    # 2. 未知黑域 (不允许视线穿入虚空)
    unknown_mask = (map_data < 0) | (map_data > 100)
    ray_penalty_map[unknown_mask] = 2.0 
    
    # 机器狗中心必须在已知的空地内
    valid_robot_center_mask = (map_data >= 0) & (map_data < 50)
    
    # 2. 提取并下采样雷达点云 (使用 500 点以保证射线绘制极速完成)
    scan_points = data['scan_points_odom']
    if len(scan_points) > 500:
        indices = np.random.choice(len(scan_points), 500, replace=False)
        scan_points = scan_points[indices]

    print("启动全局双模板光线投射引擎 (Dual-Template Ray-Casting)...")
    
    best_score = -float('inf') 
    best_pose = (0.0, 0.0, 0.0)
    
    # 粗搜索：步长 2 度
    angle_step = np.deg2rad(2)
    angles_to_search = np.arange(-np.pi, np.pi, angle_step)
    
    start_time = time.time()
    
    # 惩罚权重：1个穿墙像素扣除的得分相当于抵消几个完美命中点
    # 设定为 3.0 意味着“宁可错杀绝不放过”，极度排斥穿墙！
    PENALTY_WEIGHT = 3.0 
    
    # 3. 粗匹配：旋转双模板匹配核心循环
    for theta in angles_to_search:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        rot_x = scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t
        rot_y = scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t
        
        # 强制将机器狗原点 (0,0) 包含在模板内
        pad = 0.5
        min_x, max_x = min(0.0, np.min(rot_x)), max(0.0, np.max(rot_x))
        min_y, max_y = min(0.0, np.min(rot_y)), max(0.0, np.max(rot_y))
        
        tpl_w_px = int((max_x - min_x + 2 * pad) / resolution)
        tpl_h_px = int((max_y - min_y + 2 * pad) / resolution)
        
        if tpl_w_px <= 0 or tpl_h_px <= 0 or tpl_w_px > width or tpl_h_px > height:
            continue
            
        template_hit = np.zeros((tpl_h_px, tpl_w_px), dtype=np.float32)
        template_ray = np.zeros((tpl_h_px, tpl_w_px), dtype=np.float32)
        
        tpl_origin_x = min_x - pad
        tpl_origin_y = min_y - pad
        
        robot_x_tpl = int(-tpl_origin_x / resolution)
        robot_y_tpl = int(-tpl_origin_y / resolution)
        
        pix_x = ((rot_x - tpl_origin_x) / resolution).astype(int)
        pix_y = ((rot_y - tpl_origin_y) / resolution).astype(int)
        
        valid = (pix_x >= 0) & (pix_x < tpl_w_px) & (pix_y >= 0) & (pix_y < tpl_h_px)
        vx, vy = pix_x[valid], pix_y[valid]
        
        # === 【核心防御网】：绘制光线空地掩码 ===
        # 把机器狗到雷达点之间的连线全部画出来
        for px, py in zip(vx, vy):
            cv2.line(template_ray, (robot_x_tpl, robot_y_tpl), (px, py), 1.0, 1)
        
        # 终点打在墙上是合法的，擦除终点，防止误扣分
        template_ray[vy, vx] = 0.0
        
        # 绘制命中模板
        template_hit[vy, vx] = 1.0 
        
        # 全局高速卷积：同时计算全图所有像素的命中得分与穿透扣分！
        res_hit = cv2.matchTemplate(likelihood_map, template_hit, cv2.TM_CCORR)
        res_ray = cv2.matchTemplate(ray_penalty_map, template_ray, cv2.TM_CCORR)
        
        # 最终得分 = 命中得分 - (穿墙惩罚 * 权重)
        # 只要穿了中间那道墙，分数直接变负数！
        res = res_hit - (res_ray * PENALTY_WEIGHT)
        
        res_h, res_w = res.shape
        y_start, y_end = robot_y_tpl, robot_y_tpl + res_h
        x_start, x_end = robot_x_tpl, robot_x_tpl + res_w
        
        # 过滤掉机器狗中心不在空地的非法位置
        if y_end <= height and x_end <= width:
            valid_slice = valid_robot_center_mask[y_start:y_end, x_start:x_end]
            res[~valid_slice] = -float('inf') 
        else:
            continue
            
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > best_score:
            best_score = max_val
            match_x, match_y = max_loc
            
            best_rx = match_x * resolution + origin_x - tpl_origin_x
            best_ry = match_y * resolution + origin_y - tpl_origin_y
            best_pose = (best_rx, best_ry, theta)

    print(f"粗匹配耗时: {time.time() - start_time:.2f} 秒。进入精细贴合阶段...")
    
    # 4. === 高精度局部微调 ===
    # 在粗匹配的安全位姿附近进行 0.5度的精细微调
    fine_best_score = -float('inf')
    fine_best_pose = best_pose
    
    fine_angles = np.arange(best_pose[2] - np.deg2rad(2.0), best_pose[2] + np.deg2rad(2.1), np.deg2rad(0.5))
    
    for theta in fine_angles:
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rot_x = scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t
        rot_y = scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t
        
        min_x, max_x = min(0.0, np.min(rot_x)), max(0.0, np.max(rot_x))
        min_y, max_y = min(0.0, np.min(rot_y)), max(0.0, np.max(rot_y))
        
        tpl_w_px = int((max_x - min_x + 2 * pad) / resolution)
        tpl_h_px = int((max_y - min_y + 2 * pad) / resolution)
        
        if tpl_w_px <= 0 or tpl_h_px <= 0: continue
            
        template_hit = np.zeros((tpl_h_px, tpl_w_px), dtype=np.float32)
        template_ray = np.zeros((tpl_h_px, tpl_w_px), dtype=np.float32)
        
        tpl_origin_x = min_x - pad
        tpl_origin_y = min_y - pad
        robot_x_tpl = int(-tpl_origin_x / resolution)
        robot_y_tpl = int(-tpl_origin_y / resolution)
        
        pix_x = ((rot_x - tpl_origin_x) / resolution).astype(int)
        pix_y = ((rot_y - tpl_origin_y) / resolution).astype(int)
        valid = (pix_x >= 0) & (pix_x < tpl_w_px) & (pix_y >= 0) & (pix_y < tpl_h_px)
        vx, vy = pix_x[valid], pix_y[valid]
        
        for px, py in zip(vx, vy):
            cv2.line(template_ray, (robot_x_tpl, robot_y_tpl), (px, py), 1.0, 1)
        template_ray[vy, vx] = 0.0
        template_hit[vy, vx] = 1.0 
        
        res_hit = cv2.matchTemplate(likelihood_map, template_hit, cv2.TM_CCORR)
        res_ray = cv2.matchTemplate(ray_penalty_map, template_ray, cv2.TM_CCORR)
        res = res_hit - (res_ray * PENALTY_WEIGHT)
        
        res_h, res_w = res.shape
        y_start, y_end = robot_y_tpl, robot_y_tpl + res_h
        x_start, x_end = robot_x_tpl, robot_x_tpl + res_w
        
        if y_end <= height and x_end <= width:
            valid_slice = valid_robot_center_mask[y_start:y_end, x_start:x_end]
            res[~valid_slice] = -float('inf') 
        else:
            continue
            
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val > fine_best_score:
            fine_best_score = max_val
            match_x, match_y = max_loc
            best_rx = match_x * resolution + origin_x - tpl_origin_x
            best_ry = match_y * resolution + origin_y - tpl_origin_y
            fine_best_pose = (best_rx, best_ry, theta)

    best_pose = fine_best_pose
    
    print(f"\n--- 匹配全部完成 ---")
    print(f"最终计算得出的大地坐标 (x, y, theta):")
    print(f"X: {best_pose[0]:.3f} m")
    print(f"Y: {best_pose[1]:.3f} m")
    print(f"Theta: {np.rad2deg(best_pose[2]):.2f} 度")
    print(f"终极得分 (越高代表贴合越完美): {fine_best_score:.1f}")
    
    return best_pose, map_data, origin_x, origin_y, resolution, scan_points

def visualize_result(map_data, origin_x, origin_y, resolution, scan_points, best_pose):
    print("正在生成防穿墙最终匹配结果图...")
    cx_world, cy_world, theta = best_pose
    
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    rotated_x = scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t
    rotated_y = scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t
    
    trans_x = rotated_x + cx_world
    trans_y = rotated_y + cy_world
    
    pix_x = ((trans_x - origin_x) / resolution).astype(int)
    pix_y = ((trans_y - origin_y) / resolution).astype(int)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(map_data, cmap='gray', origin='lower')
    
    plt.scatter(pix_x, pix_y, c='cyan', s=1.0, alpha=0.9, label='Aligned Scan (Dual-Template Ray-Cast)')
    
    dog_pix_x = int((cx_world - origin_x) / resolution)
    dog_pix_y = int((cy_world - origin_y) / resolution)
    plt.plot(dog_pix_x, dog_pix_y, 'r+', markersize=20, markeredgewidth=3, label='Estimated Pose')
    
    plt.title(f"Dual-Template Perfect Match\nX: {cx_world:.2f}m, Y: {cy_world:.2f}m, Theta: {np.rad2deg(theta):.1f}°")
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    file_path = "../scan_viz/debug_match_data_8.npz"
    best_pose, map_data, origin_x, origin_y, resolution, scan_points = load_and_match_robust(file_path)
    visualize_result(map_data, origin_x, origin_y, resolution, scan_points, best_pose)


