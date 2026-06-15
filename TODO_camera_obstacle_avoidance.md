# 摄像头辅助避障 TODO

## 背景

- 设备：Livox MID360 激光雷达 + 前后摄像头（1920×1080, 25fps）
- 问题：拐角撞墙、越走越不稳
- 当前状态：摄像头仅用于录像，未参与导航避障

## 摄像头话题

| 摄像头 | 话题 | 分辨率 | 帧率 |
|--------|------|--------|------|
| 前摄像头 | `/front_camera/image_compressed` | 1920×1080 | 25fps |
| 后摄像头 | `/rear_camera/image_compressed` | 1920×1080 | 25fps |

## 相机内参（来自 camera.yaml）

```yaml
intrinsic_matrix:
  fx: 808.89
  fy: 796.39
  cx: 942.59
  cy: 482.99
depth_scale: 1.2
```

---

## 方案 1：光流碰撞检测（最简单，1-2天）

### 原理
光流场快速膨胀 → 有物体正在接近

### 实现
```python
# camera_obstacle_detector.py
# 1. 订阅 /front_camera/image_compressed
# 2. 计算 Farneback 光流
# 3. 计算径向膨胀率
# 4. 膨胀率 > 阈值 → 发布 /camera_obstacle (Bool)
```

### 关键参数
- `pyr_scale=0.5, levels=3, winsize=15`
- 膨胀率阈值：`2.0`（需调试）

### 优点
- 不需要深度图
- 计算量小，实时性好
- RK3588 可跑

### 缺点
- 只能检测"接近"，不能测距
- 光照变化时可能误触发

### 参考代码
```python
import cv2
import numpy as np

flow = cv2.calcOpticalFlowFarneback(
    prev_gray, gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

# 径向光流计算
y_coords, x_coords = np.mgrid[0:h, 0:w]
dx = x_coords - cx
dy = y_coords - cy
dist = np.sqrt(dx**2 + dy**2) + 1e-6
radial_flow = (flow[..., 0] * dx + flow[..., 1] * dy) / dist
expansion_ratio = np.mean(radial_flow[radial_flow > 0])
```

---

## 方案 2：相机转激光扫描（推荐，3-5天）

### 原理
将摄像头检测到的障碍物转换为 LaserScan 格式，融合到现有 costmap

### 实现
```python
# camera_to_scan_node.py
# 1. 订阅 /front_camera/image_compressed
# 2. 边缘检测 + 轮廓提取
# 3. 根据轮廓位置计算角度和估算深度
# 4. 发布 /camera/scan (LaserScan)
```

### Costmap 配置
```yaml
local_costmap:
  plugins: ["obstacle_layer_local", "camera_obstacle_layer", "inflation_layer_local"]
  
  camera_obstacle_layer:
    plugin: "nav2_costmap_2d::ObstacleLayer"
    enabled: true
    observation_sources: camera_scan
    camera_scan:
      topic: /camera/scan
      data_type: "LaserScan"
      marking: true
      clearing: true
      obstacle_max_range: 3.0
      obstacle_min_range: 0.1
```

### 优点
- 与现有 Nav2 架构无缝集成
- 补充激光雷达盲区（脚下、头顶）
- 可同时使用前后摄像头

### 缺点
- 深度估算不精确（基于轮廓大小）
- 需要标定相机外参

### 注意事项
- 需要将 `/camera/scan` 的 frame_id 设置为相机坐标系
- 需要添加 static_transform_publisher 发布相机到 base_link 的变换

---

## 方案 3：MiDaS 单目深度估计（1-2周）

### 原理
用预训练模型从 RGB 图生成深度图

### 依赖
```bash
pip install torch torchvision timm
```

### 实现
```python
# depth_estimation_node.py
# 1. 加载 MiDaS_small 模型
# 2. 订阅 /front_camera/image_compressed
# 3. MiDaS 推理生成深度图
# 4. 发布 /camera/depth_image (Image)
```

### 关键代码
```python
import torch

model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
model.eval()

midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

input_batch = midas_transforms(image)
with torch.no_grad():
    prediction = model(input_batch)
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.shape[:2],
        mode='bicubic',
        align_corners=False,
    ).squeeze()
```

### 优点
- 能生成相对深度图
- 可用于精确测距
- 模型成熟，效果好

### 缺点
- 需要 GPU 加速
- RK3588 可能较慢（需测试）
- 模型较大（~40MB）

### 优化方向
- 用 ONNX Runtime 加速
- 用 TFLite 量化模型
- 降低输入分辨率（640×480）

---

## 方案 4：YOLO 目标检测（可选，1-2周）

### 原理
用 YOLO 检测特定障碍物（人、家具等）

### 依赖
```bash
pip install ultralytics
```

### 实现
```python
# yolo_obstacle_detector.py
# 1. 加载 YOLOv8n 模型
# 2. 订阅 /front_camera/image_compressed
# 3. 检测障碍物并估算距离
# 4. 发布 /camera/obstacles (自定义消息)
```

### 优点
- 能识别特定障碍物类型
- 可针对不同障碍物设置不同避障策略

### 缺点
- 计算量大
- 需要训练或微调模型

---

## 实施优先级

| 优先级 | 方案 | 难度 | 时间 | 效果 |
|--------|------|------|------|------|
| ⭐⭐⭐ | 光流碰撞检测 | 低 | 1-2天 | 紧急减速 |
| ⭐⭐⭐ | 相机转激光扫描 | 中 | 3-5天 | 补充盲区 |
| ⭐⭐ | MiDaS 深度估计 | 中高 | 1-2周 | 精确测距 |
| ⭐ | YOLO 目标检测 | 高 | 1-2周 | 目标识别 |

---

## 测试计划

### 第一阶段：光流检测
1. 启动 `camera_obstacle_detector` 节点
2. 手动在前方放置障碍物
3. 检查 `/camera_obstacle` 话题是否正确发布
4. 调整膨胀率阈值

### 第二阶段：相机转激光
1. 启动 `camera_to_scan` 节点
2. 在 RViz 中查看 `/camera/scan` 话题
3. 验证障碍物位置是否正确
4. 融合到 costmap 并测试导航

### 第三阶段：深度估计
1. 测试 MiDaS 在 RK3588 上的推理速度
2. 如果速度可接受（>10fps），集成到系统
3. 验证深度图精度

---

## 相关文件

- 相机配置：`zsi_tools/d1max/3588/install/robot_camera/share/robot_camera/config/zsm.yaml`
- 相机启动：`zsi_tools/d1max/3588/install/robot_camera/share/robot_camera/launch/zsm.launch.py`
- 相机内参：`zsi_tools/d1max/orin/robot/perception/install/perception/config/perception/detection/camera.yaml`
- Nav2 参数：`nav2_dog_slam/config/nav2_params.yaml`

---

## 待办事项

- [ ] 实现光流碰撞检测节点
- [ ] 实现相机转激光扫描节点
- [ ] 测试 MiDaS 在 RK3588 上的性能
- [ ] 将摄像头障碍物融合到 costmap
- [ ] 添加相机到 base_link 的 TF 变换
- [ ] 测试前后摄像头同时工作
- [ ] 优化节点性能（CPU 占用）
