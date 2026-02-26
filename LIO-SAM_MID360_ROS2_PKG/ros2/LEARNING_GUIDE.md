# LIO-SAM_MID360_ROS2_PKG 项目学习引导

本指南介绍了 `LIO-SAM_MID360_ROS2_PKG` 项目中各ROS 2包的功能、结构和使用方法，帮助开发者快速上手。

## 项目结构概览

```
ros2/src/
├── FAST_LIO_ROS2_edit          # FAST-LIO算法的ROS 2实现
├── LIO-SAM_MID360_ROS2_DOG      # LIO-SAM算法针对MID360雷达的修改版本
├── SC_PGO_ROS2                  # Scan Context回环检测算法
├── Super-LIO                    # 增强版LIO算法
├── autorccar_interfaces         # 自动驾驶车接口定义
├── global_config                # 全局配置管理
├── lidar_localization_ros2      # 激光雷达定位算法
├── livox_gazebo_ros2_gpu_simulation  # Livox雷达Gazebo GPU仿真
├── livox_ros_driver2            # Livox雷达ROS 2驱动
├── m-explore                    # 自主探索算法
├── nav2_dog_slam                # Nav2导航与SLAM集成
├── ndt_omp_ros2                 # NDT定位算法（OpenMP加速）
└── point_lio_ros2               # Point-LIO算法的ROS 2实现
```

## 各包详细介绍

### 1. FAST_LIO_ROS2_edit
**功能**：基于FAST-LIO（Fast and Accurate LiDAR-Inertial Odometry）算法的ROS 2实现，用于激光雷达与IMU融合定位。

**特点**：
- 高性能激光雷达惯性里程计
- 支持MID360等Livox雷达
- 配置文件位于 `config/mid360.yaml`

**使用**：
```bash
ros2 launch fast_lio mapping.launch.py
```

### 2. LIO-SAM_MID360_ROS2_DOG
**功能**：LIO-SAM（LiDAR-Inertial Odometry with Scan Matching）算法针对MID360雷达的修改版本。

**特点**：
- 结合激光雷达、IMU和GPS数据
- 支持回环检测
- 机器人模型配置位于 `config/robot.urdf_tilt.xacro`

**使用**：
```bash
ros2 launch lio_sam lio_sam.launch.py
```

### 3. SC_PGO_ROS2
**功能**：Scan Context回环检测算法的ROS 2实现，用于SLAM系统的全局一致性优化。

**特点**：
- 高效的激光雷达场景识别
- 用于检测SLAM轨迹中的回环
- 提升地图全局一致性

### 4. Super-LIO
**功能**：增强版的LIO（LiDAR-Inertial Odometry）算法。

**特点**：
- 可能集成了多种优化策略
- 适用于复杂环境

### 5. autorccar_interfaces
**功能**：自动驾驶车的ROS 2接口定义包。

**特点**：
- 包含自定义消息类型
- 用于模块间通信

### 6. global_config
**功能**：全局配置管理包，提供统一的配置接口。

**特点**：
- 支持按主机名自动选择配置
- 包含雷达、导航等系统配置
- 自动更新Nav2和FAST-LIO参数文件

**使用**：配置文件位于 `global_config/__init__.py`，可根据主机名修改配置。

### 7. lidar_localization_ros2
**功能**：激光雷达定位算法的ROS 2实现。

**特点**：
- 基于激光雷达的定位系统
- 可能支持多种定位算法

### 8. livox_gazebo_ros2_gpu_simulation
**功能**：Livox雷达的Gazebo GPU仿真包。

**特点**：
- 支持GPU加速的雷达仿真
- 提供真实的点云数据模拟
- 适用于算法测试和开发

### 9. livox_ros_driver2
**功能**：Livox雷达的官方ROS 2驱动。

**特点**：
- 支持多种Livox雷达型号（如MID360）
- 提供点云数据采集和发布
- 配置文件位于 `config/` 目录

**使用**：
```bash
ros2 launch livox_ros_driver2 rviz_MID360_launch.py
```

### 10. m-explore
**功能**：自主探索算法包，用于机器人在未知环境中的导航。

**特点**：
- 基于前沿点的探索策略
- 支持多机器人协作探索
- 适用于建图模式

**使用**：
```bash
ros2 launch explore_lite explore.launch.py
```

### 11. nav2_dog_slam
**功能**：Nav2导航栈与SLAM算法的集成包，是项目的核心组件。

**特点**：
- 集成多种SLAM算法（fast_lio、point_lio、lio_sam、super_lio）
- 提供统一的导航接口
- 包含建图和导航两种模式
- Web控制界面支持

**使用**：
```bash
# 建图模式
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py MANUAL_BUILD_MAP:=true

# 导航模式
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py
```

### 12. ndt_omp_ros2
**功能**：NDT（正态分布变换）定位算法的ROS 2实现，使用OpenMP加速。

**特点**：
- 基于概率分布的点云配准
- OpenMP并行计算加速
- 适用于高精度定位

### 13. point_lio_ros2
**功能**：Point-LIO算法的ROS 2实现。

**特点**：
- 基于点云的激光雷达惯性里程计
- 可能针对特定场景优化

## 系统架构

### 数据流
1. **传感器数据采集**：`livox_ros_driver2` 采集MID360雷达数据
2. **状态估计**：各LIO算法包（如 `FAST_LIO_ROS2_edit`）处理传感器数据，输出位姿估计
3. **地图构建**：`nav2_dog_slam` 或 `SLAM_toolbox` 构建环境地图
4. **导航规划**：`nav2_dog_slam` 中的Nav2组件根据地图和目标点规划路径
5. **控制执行**：根据规划结果控制机器人运动

### 模块关系
- **底层驱动**：`livox_ros_driver2`、`livox_gazebo_ros2_gpu_simulation`
- **定位算法**：`FAST_LIO_ROS2_edit`、`LIO-SAM_MID360_ROS2_DOG`、`point_lio_ros2`、`Super-LIO`、`ndt_omp_ros2`
- **回环检测**：`SC_PGO_ROS2`
- **导航控制**：`nav2_dog_slam`、`m-explore`
- **配置管理**：`global_config`

## 快速开始

### 1. 环境搭建
```bash
# 安装依赖
rosdep install --from-paths /home/crazytree/ros_ws/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2 --ignore-src -r -y

# 编译项目
cd /home/crazytree/ros_ws/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2
colcon build --symlink-install

# 加载环境变量
source install/setup.bash
```

### 2. 运行示例

#### 2.1 启动完整系统（导航模式）
```bash
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py
```

#### 2.2 启动建图模式
```bash
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py MANUAL_BUILD_MAP:=true
```

#### 2.3 启动Gazebo仿真
```bash
# 首先确保安装了Gazebo
ros2 launch livox_gazebo_ros2_gpu_simulation livox_gazebo.launch.py
```

## 开发指南

### 1. 配置修改
- **全局配置**：修改 `global_config/global_config/__init__.py` 中的对应主机配置
- **算法参数**：修改各算法包下的 `config/` 目录中的参数文件
- **导航参数**：修改 `nav2_dog_slam/config/nav2_params.yaml`

### 2. 添加新传感器
1. 在机器人URDF文件中添加传感器描述
2. 编写或修改传感器驱动节点
3. 在SLAM算法中集成新传感器数据

### 3. 算法切换
通过环境变量或直接修改配置文件切换不同的SLAM算法：
```bash
# 通过环境变量切换
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py SLAM_ALGORITHM:=fast_lio

# 支持的算法：fast_lio, point_lio, lio_sam, super_lio
```

## 故障排查

### 常见错误及解决方案

1. **缺少 base_footprint 坐标系**
   - 错误：`Timed out waiting for transform from base_footprint to odom`
   - 解决：在URDF中添加base_footprint链接，或发布静态变换

2. **路径不存在错误**
   - 错误：`No such file or directory`
   - 解决：检查 `global_config.py` 中的路径配置，确保与实际路径匹配

3. **雷达驱动启动失败**
   - 错误：`Failed to connect to Livox device`
   - 解决：检查雷达连接，修改雷达配置文件中的设备信息

4. **导航规划失败**
   - 错误：`No valid path found`
   - 解决：检查地图质量，确保环境中有足够的可导航空间

### 调试工具

1. **查看节点和话题**
   ```bash
   ros2 node list
   ros2 topic list
   ```

2. **查看TF树**
   ```bash
   ros2 run tf2_tools view_frames
   ```

3. **监听传感器数据**
   ```bash
   ros2 topic echo /livox/lidar  # 雷达点云
   ros2 topic echo /Odometry      # 里程计数据
   ```

## 学习资源

### 核心算法文档
- [FAST-LIO](https://github.com/hku-mars/FAST_LIO)
- [LIO-SAM](https://github.com/TixiaoShan/LIO-SAM)
- [Livox ROS Driver 2](https://github.com/Livox-SDK/livox_ros_driver2)

### ROS 2相关
- [ROS 2官方文档](https://docs.ros.org/en/humble/)
- [Nav2文档](https://navigation.ros.org/)
- [Gazebo仿真](https://gazebosim.org/docs)

### 项目特定
- 本项目README.md文件
- 各包内的README和文档

## 贡献指南

1. **代码风格**：遵循ROS 2和Python/CMake标准风格
2. **提交规范**：使用清晰的提交信息，说明修改内容
3. **测试**：确保修改后系统能正常运行
4. **文档**：更新相关文档以反映修改

## 联系方式

如有问题或建议，请联系项目维护者。

---

**版本**：1.0
**最后更新**：2026-02-24
**适用ROS 2版本**：Humble
