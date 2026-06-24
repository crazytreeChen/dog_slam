import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def load_and_match(npz_path):
    print(f"正在读取数据: {npz_path}")
    data = np.load(npz_path)
    
    # 1. 提取地图数据
    resolution = data['map_resolution'].item()
    width = data['map_width'].item()
    height = data['map_height'].item()
    origin_x = data['map_origin_x'].item()
    origin_y = data['map_origin_y'].item()
    
    # 将一维地图数据重塑为二维，并转换为 OpenCV 图像格式
    map_data = data['map_data'].reshape((height, width))
    # 假设 ROS 标准栅格图：大于 50 视为障碍物
    binary_map = np.zeros_like(map_data, dtype=np.uint8)
    binary_map[map_data > 50] = 255
    
    # 2. 生成距离场 (Distance Transform)
    # 距离墙壁越近，像素值越小（0）；越远值越大
    dist_map = cv2.distanceTransform(255 - binary_map, cv2.DIST_L2, 5)
    
    # 3. 提取预处理好的合并雷达点云
    scan_points = data['scan_points_odom']  # 形状应为 (N, 2)
    
    # 为了加速，我们可以对密集点云进行随机下采样
    if len(scan_points) > 500:
        indices = np.random.choice(len(scan_points), 500, replace=False)
        scan_points = scan_points[indices]

    print("开始全局扫描匹配计算...")
    
    # 4. 暴力搜索 (粗略搜索)
    # 设定搜索范围（如果有大致猜测可以缩小范围，这里演示全图旋转搜）
    best_score = float('inf')
    best_pose = (0.0, 0.0, 0.0)
    
    # 角度搜索步长 (例如每 5 度搜索一次)
    angle_step = np.deg2rad(5)
    angles_to_search = np.arange(-np.pi, np.pi, angle_step)
    
    # 仅遍历地图中非空白区域作为可能的平移起点 (减少计算量)
    # 这里我们简化处理，在真实场景中可以建立分辨率金字塔
    y_coords, x_coords = np.where(map_data == 0) # 0 表示 Free space
    
    # 为了演示计算，我们在随机的 free space 中抽取 1000 个可能的起点进行测试
    sample_indices = np.random.choice(len(x_coords), 1000, replace=False)
    
    for idx in sample_indices:
        cx_pix = x_coords[idx]
        cy_pix = y_coords[idx]
        
        # 将像素坐标转为世界坐标
        cx_world = cx_pix * resolution + origin_x
        cy_world = cy_pix * resolution + origin_y
        
        for theta in angles_to_search:
            # 构造旋转矩阵
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            # 将点云进行平移和旋转变换
            rotated_x = scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t
            rotated_y = scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t
            
            trans_x = rotated_x + cx_world
            trans_y = rotated_y + cy_world
            
            # 转换回像素坐标去查距离场表
            pix_x = ((trans_x - origin_x) / resolution).astype(int)
            pix_y = ((trans_y - origin_y) / resolution).astype(int)
            
            # 剔除越界的点
            valid_mask = (pix_x >= 0) & (pix_x < width) & (pix_y >= 0) & (pix_y < height)
            valid_x = pix_x[valid_mask]
            valid_y = pix_y[valid_mask]
            
            if len(valid_x) < len(scan_points) * 0.5:
                continue # 超过一半的点在地图外，直接丢弃
            
            # 计算得分：所有合法点在距离场中的值之和（越小越贴合墙壁）
            score = np.sum(dist_map[valid_y, valid_x]) / len(valid_x)
            
            if score < best_score:
                best_score = score
                best_pose = (cx_world, cy_world, theta)

    print(f"\n--- 匹配完成 ---")
    print(f"计算得出的全局位姿 (x, y, theta):")
    print(f"X: {best_pose[0]:.3f} m")
    print(f"Y: {best_pose[1]:.3f} m")
    print(f"Theta: {np.rad2deg(best_pose[2]):.2f} 度")
    print(f"匹配误差得分: {best_score:.4f}")
    
    return best_pose


def visualize_result(npz_path, best_pose):
    print("正在生成匹配结果可视化图像...")
    data = np.load(npz_path)
    resolution = data['map_resolution'].item()
    origin_x = data['map_origin_x'].item()
    origin_y = data['map_origin_y'].item()
    map_data = data['map_data'].reshape((data['map_height'].item(), data['map_width'].item()))
    scan_points = data['scan_points_odom']
    
    cx_world, cy_world, theta = best_pose
    
    # 按照计算出的位姿变换点云
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated_x = scan_points[:, 0] * cos_t - scan_points[:, 1] * sin_t
    rotated_y = scan_points[:, 0] * sin_t + scan_points[:, 1] * cos_t
    trans_x = rotated_x + cx_world
    trans_y = rotated_y + cy_world
    
    # 转换回像素坐标
    pix_x = ((trans_x - origin_x) / resolution).astype(int)
    pix_y = ((trans_y - origin_y) / resolution).astype(int)
    
    # 绘图
    plt.figure(figsize=(12, 10))
    # 显示地图 (注意坐标原点位置)
    plt.imshow(map_data, cmap='gray', origin='lower')
    # 叠加雷达点云
    plt.scatter(pix_x, pix_y, c='cyan', s=1, alpha=0.8, label='Aligned Scan')
    
    # 标记计算出的机器狗中心点
    dog_pix_x = int((cx_world - origin_x) / resolution)
    dog_pix_y = int((cy_world - origin_y) / resolution)
    plt.plot(dog_pix_x, dog_pix_y, 'r+', markersize=15, markeredgewidth=2, label='Estimated Pose')
    
    plt.title(f"Global Match Result\nX: {cx_world:.2f}m, Y: {cy_world:.2f}m, Theta: {np.rad2deg(theta):.1f}°")
    plt.legend()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    file_path = "../scan_viz/debug_match_data_3.npz"
    # 获取之前计算的位姿 (这里直接填入你刚算出的结果)
    calculated_pose = (28.600, 29.150, np.deg2rad(170.00)) 
    visualize_result(file_path, calculated_pose)

