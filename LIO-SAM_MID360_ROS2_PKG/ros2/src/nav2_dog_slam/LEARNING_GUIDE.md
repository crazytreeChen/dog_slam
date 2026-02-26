# nav2_dog_slam 包学习指南

## 1. 包概述

**nav2_dog_slam** 是一个集成了导航和SLAM功能的ROS 2包，专为搭载MID360激光雷达的机器人设计。它提供了统一的接口来管理多种SLAM算法和Nav2导航栈，支持建图和导航两种模式。

**核心功能**：
- 集成多种SLAM算法（fast_lio、point_lio、lio_sam、super_lio）
- Nav2导航栈集成
- GPS数据融合与处理
- Web控制界面
- 建图与导航模式切换

## 2. 目录结构

```
nav2_dog_slam/
├── CMakeLists.txt          # CMake构建文件
├── package.xml            # 包描述和依赖
├── config/                # 配置文件目录
│   ├── gps_ekf.yaml       # GPS EKF滤波器配置
│   ├── navsat_transform.yaml  # GPS坐标转换配置
│   ├── nav2_params.yaml   # Nav2参数配置
│   └── navigate_to_pose_w_replanning_and_recovery.xml  # 行为树配置
├── docs/                  # 文档目录
├── launch/                # 启动文件目录
│   ├── gps_fusion.launch.py  # GPS融合启动文件
│   ├── lio_nav2_unified.launch.py  # 统一启动文件（核心）
│   ├── nav2_gps_fusion.launch.py  # Nav2与GPS融合启动文件
│   └── navigation_launch.py  # 导航启动文件
├── scripts/               # 脚本目录
│   ├── test_gps_fusion.py  # GPS融合测试脚本
│   └── verify_gps_config.py  # GPS配置验证脚本
├── src/                   # 源代码目录
│   ├── gps_preprocessor.py  # GPS数据预处理器
│   └── gps_simulator.py    # GPS数据模拟器
└── web/                   # Web控制界面
    ├── libs/              # Web依赖库
    ├── nav2_web_control.html  # Web控制界面
    └── run_web.sh         # Web服务启动脚本
```

## 3. 核心功能模块

### 3.1 统一SLAM启动器

**文件**：`launch/lio_nav2_unified.launch.py`

**功能**：根据配置自动选择和启动不同的SLAM算法，集成Nav2导航栈。

**主要特性**：
- 支持多种SLAM算法切换（通过 `SLAM_ALGORITHM` 参数）
- 建图模式与导航模式切换（通过 `MANUAL_BUILD_MAP` 参数）
- 自动配置传感器话题映射
- 集成Web控制界面

**使用示例**：
```bash
# 导航模式（默认）
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py

# 建图模式
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py MANUAL_BUILD_MAP:=true

# 指定SLAM算法
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py SLAM_ALGORITHM:=fast_lio
```

### 3.2 GPS融合系统

**文件**：
- `launch/gps_fusion.launch.py` - GPS融合启动
- `launch/nav2_gps_fusion.launch.py` - Nav2与GPS融合启动
- `src/gps_preprocessor.py` - GPS数据预处理
- `config/gps_ekf.yaml` - EKF滤波器配置

**功能**：处理GPS数据并与里程计数据融合，提高定位精度。

**主要特性**：
- GPS数据预处理（噪声过滤、坐标转换）
- 使用EKF滤波器融合GPS与里程计数据
- 支持GPS仿真模式

**使用示例**：
```bash
# 启动GPS融合
ros2 launch nav2_dog_slam gps_fusion.launch.py

# 启动Nav2与GPS融合
ros2 launch nav2_dog_slam nav2_gps_fusion.launch.py
```

### 3.3 Nav2导航集成

**文件**：`launch/navigation_launch.py`

**功能**：启动Nav2导航栈，包括规划器、控制器和行为树。

**主要特性**：
- 集成Nav2全套组件
- 支持多种导航参数配置
- 与SLAM算法无缝集成

### 3.4 Web控制界面

**文件**：
- `web/nav2_web_control.html` - Web控制界面
- `web/run_web.sh` - 启动脚本

**功能**：提供基于浏览器的机器人控制界面。

**主要特性**：
- 可视化机器人状态和地图
- 支持目标点设置
- 导航模式切换

**使用**：Web界面会在启动系统时自动打开。

## 4. 系统架构

### 4.1 数据流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 传感器数据  │ ──> │  SLAM算法   │ ──> │  里程计数据  │
└─────────────┘     └─────────────┘     └─────────────┘
                        ^                     │
                        │                     v
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  GPS数据    │ ──> │ GPS融合系统 │ <── │  Nav2导航   │
└─────────────┘     └─────────────┘     └─────────────┘
                        │                     │
                        v                     v
                  ┌─────────────┐     ┌─────────────┐
                  │  地图数据   │ <── │  Web界面    │
                  └─────────────┘     └─────────────┘
```

### 4.2 模式切换

**建图模式**：
- 启动SLAM算法进行地图构建
- 可选启动octomap_server或slam_toolbox
- 不启动Nav2导航栈

**导航模式**：
- 启动SLAM算法进行定位
- 启动Nav2导航栈
- 加载已构建的地图

## 5. 配置指南

### 5.1 主要配置文件

| 配置文件 | 功能 | 位置 |
|---------|------|------|
| nav2_params.yaml | Nav2导航参数 | config/ |
| gps_ekf.yaml | GPS EKF滤波器参数 | config/ |
| navsat_transform.yaml | GPS坐标转换参数 | config/ |

### 5.2 环境变量

| 环境变量 | 功能 | 默认值 |
|---------|------|-------|
| MANUAL_BUILD_MAP | 是否启动建图模式 | false |
| BUILD_TOOL | 建图工具选择 | octomap_server |
| SLAM_ALGORITHM | SLAM算法选择 | fast_lio |
| AUTO_BUILD_MAP | 是否启动自动建图 | false |
| NAVIGATION_MODE | 导航模式 | standalone |

## 6. 使用教程

### 6.1 基本使用流程

**步骤1：启动系统**
```bash
# 导航模式
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py

# 建图模式
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py MANUAL_BUILD_MAP:=true
```

**步骤2：使用Web界面**
- 系统启动后会自动打开Web控制界面
- 或手动访问 `http://localhost:8000/nav2_web_control.html`

**步骤3：设置导航目标**
- 在Web界面中点击地图设置目标点
- 或使用RViz2的 "2D Goal Pose" 工具

### 6.2 高级使用

**自定义SLAM算法**：
```bash
# 使用LIO-SAM算法
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py SLAM_ALGORITHM:=lio_sam

# 使用Point-LIO算法
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py SLAM_ALGORITHM:=point_lio
```

**使用GPS融合**：
```bash
# 启动带GPS融合的导航
ros2 launch nav2_dog_slam nav2_gps_fusion.launch.py
```

## 7. 故障排查

### 7.1 常见问题

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|--------|
| No such file or directory | 路径配置错误 | 检查global_config.py中的路径配置 |
| Timed out waiting for transform | 坐标变换缺失 | 确保base_footprint链接存在 |
| Server fastlio_mapping was unable to be reached | SLAM算法启动失败 | 检查雷达连接和配置 |
| Failed to load map yaml file | 地图文件不存在 | 确保地图文件路径正确 |

### 7.2 调试工具

**查看节点和话题**：
```bash
ros2 node list
ros2 topic list
```

**监听传感器数据**：
```bash
ros2 topic echo /livox/lidar  # 雷达数据
ros2 topic echo /Odometry      # 里程计数据
ros2 topic echo /gps/fix       # GPS数据
```

**查看TF树**：
```bash
ros2 run tf2_tools view_frames
```

## 8. 开发指南

### 8.1 添加新的SLAM算法

1. 在 `lio_nav2_unified.launch.py` 中添加新算法的启动配置
2. 在 `LIO_TOPIC_CONFIGS` 字典中添加算法的话题映射
3. 确保算法包已正确安装

### 8.2 自定义导航行为

1. 修改 `config/navigate_to_pose_w_replanning_and_recovery.xml` 行为树
2. 调整 `config/nav2_params.yaml` 中的导航参数

### 8.3 扩展Web界面

1. 修改 `web/nav2_web_control.html` 文件
2. 更新 `web/run_web.sh` 脚本

## 9. 依赖关系

**核心依赖**：
- nav2_bringup
- nav2_core
- nav2_lifecycle_manager
- livox_ros_driver2
- fast_lio (或其他SLAM算法)
- robot_localization (GPS融合)
- rosbridge_server (Web界面)

## 10. 总结

**nav2_dog_slam** 是一个功能强大的导航和SLAM集成包，为搭载MID360激光雷达的机器人提供了完整的解决方案。它的模块化设计使其易于扩展和定制，支持多种SLAM算法和导航模式，满足不同场景的需求。

**适用场景**：
- 室内外环境建图
- 自主导航
- 机器人定位
- GPS辅助导航

通过本指南，您应该能够快速上手使用 `nav2_dog_slam` 包，并根据需要进行定制和扩展。

## 11. 版本信息

- **ROS 2 版本**：Humble
- **支持的SLAM算法**：fast_lio、point_lio、lio_sam、super_lio
- **硬件支持**：Livox MID360激光雷达

## 12. 联系方式

如有问题或建议，请联系项目维护者。
