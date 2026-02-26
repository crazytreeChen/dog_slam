# LIO-SAM_MID360_ROS2_PKG 系统文档

## 1. 项目概述

LIO-SAM_MID360_ROS2_PKG 是一个基于 ROS2 Humble 的激光雷达 SLAM 与导航系统，专为 Livox MID360 激光雷达优化，集成了多种先进的 SLAM 算法和导航功能。

### 1.1 主要特性

- **多 SLAM 算法支持**：FAST-LIO2、LIO-SAM、Point-LIO、Super-LIO
- **统一导航系统**：集成 Nav2 导航框架，支持自主探索和路径规划
- **实时建图**：支持在线建图和离线地图构建
- **稳定导航**：优化 base_footprint，实现高速稳定导航
- **Gazebo 仿真**：支持 Gazebo GPU 加速仿真

### 1.2 应用场景

- 室内外环境自主导航
- 机器人定位与建图
- 环境感知与监测
- 自动驾驶研究与开发

## 2. 系统要求

### 2.1 硬件要求

- **激光雷达**：Livox MID360
- **计算平台**：至少 8GB 内存，4 核心 CPU
- **存储**：至少 50GB 可用空间
- **网络**：支持有线/无线连接

### 2.2 软件要求

- **操作系统**：Ubuntu 22.04 LTS
- **ROS 版本**：ROS2 Humble
- **依赖库**：
  - Livox-SDK2
  - PCL (Point Cloud Library)
  - Boost
  - GTSAM (用于 LIO-SAM)
  - Nav2 相关包

## 3. 安装指南

### 3.1 系统准备

1. **安装 Ubuntu 22.04 LTS**
   - 从 [Ubuntu 官网](https://ubuntu.com/download/desktop) 下载镜像
   - 按照官方指南安装系统

2. **配置软件源**（推荐使用阿里源）
   ```bash
   # 添加 ROS2 阿里源
   sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.aliyun.com/ros2/ubuntu/ jammy main" > /etc/apt/sources.list.d/ros2.list'
   
   # 添加 Ubuntu 阿里源
   sudo sh -c 'echo "deb [arch=amd64] https://mirrors.aliyun.com/ubuntu/ jammy main" > /etc/apt/sources.list.d/aliyun.list'
   
   # 添加 GPG key
   sudo apt install curl gnupg2 lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   ```

### 3.2 ROS2 Humble 安装

按照 [ROS2 官方安装指南](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html) 安装 ROS2 Humble。

### 3.3 依赖库安装

1. **基础依赖**
   ```bash
   sudo apt-get update
   sudo apt-get install -y git cmake g++ libboost-all-dev libpcl-dev
   ```

2. **Livox-SDK2 安装**
   ```bash
   git clone https://github.com/Livox-SDK/Livox-SDK2.git
   cd Livox-SDK2
   mkdir build && cd build
   cmake ..
   make
   sudo make install
   ```

3. **ROS2 依赖包**
   ```bash
   # SLAM 相关依赖
   sudo apt install -y ros-humble-perception-pcl \
      	   ros-humble-pcl-msgs \
      	   ros-humble-vision-opencv \
      	   ros-humble-xacro \
    	   ros-humble-vision-msgs
   
   # GTSAM (LIO-SAM 依赖)
   sudo add-apt-repository ppa:borglab/gtsam-release-4.1
   sudo apt install -y libgtsam-dev libgtsam-unstable-dev
   
   # Super-LIO 依赖
   sudo apt install libgoogle-glog-dev libtbb-dev
   
   # Nav2 导航系统
   sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
   sudo apt install ros-humble-rosbridge-server 
   sudo apt-get update && sudo apt-get install -y \
       ros-humble-dwb-critics \
       ros-humble-nav2-dwb-controller \
       ros-humble-nav2-controller \
       ros-humble-nav2-amcl \
       ros-humble-nav2-planner \
       ros-humble-nav2-bt-navigator \
       ros-humble-nav2-lifecycle-manager \
       ros-humble-nav2-map-server \
       ros-humble-nav2-waypoint-follower \
       ros-humble-rosbridge-server
   
   # 点云转激光扫描
   sudo apt install ros-humble-pointcloud-to-laserscan
   
   # OctoMap 建图
   sudo apt install ros-humble-octomap ros-humble-octomap-msgs
   sudo apt install ros-humble-octomap-server
   ```

### 3.4 项目获取与构建

1. **克隆项目**
   ```bash
   mkdir -p ~/ros_ws/dog_slam
   cd ~/ros_ws/dog_slam
   git clone <项目仓库地址>
   ```

2. **构建项目**
   ```bash
   # 设置 DDS 实现
   export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
   
   # 进入项目目录
   cd LIO-SAM_MID360_ROS2_PKG
   
   # 首次构建
   rm -rf build/ install/ log/
   ./build_ros2.sh
   ```

## 4. 快速开始

### 4.1 统一启动方式

系统使用统一的启动文件，集成了所有 SLAM 算法和导航功能：

```bash
# 进入项目根目录
cd ~/ros_ws/dog_slam/LIO-SAM_MID360_ROS2_PKG

# 通过环境变量选择 SLAM 算法（默认使用 FAST-LIO2）
# export SLAM_ALGORITHM=fast_lio    # 使用 FAST-LIO2
# export SLAM_ALGORITHM=point_lio   # 使用 Point-LIO
# export SLAM_ALGORITHM=super_lio   # 使用 Super-LIO
# export SLAM_ALGORITHM=lio_sam     # 使用 LIO-SAM

# 启动统一 SLAM 导航系统
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py
```

### 4.2 环境变量配置

通过设置环境变量，可以自定义系统行为：

```bash
# 选择 SLAM 算法
export SLAM_ALGORITHM=fast_lio    # 默认值，使用 FAST-LIO2

# 建图模式
export MAP_BUILDING_MODE=true      # 启用建图模式

# 导航模式
export NAVIGATION_MODE=true        # 启用导航模式
```

### 4.3 Web 控制界面

系统启动后，可通过浏览器访问 Web 控制界面：

1. 打开浏览器，访问地址：`http://localhost:8083/nav2_web_control.html`
2. 界面功能：
   - 地图显示：实时显示构建的地图
   - 机器人位置：红色圆点表示机器人当前位置
   - 目标点设置：在地图上点击设置导航目标点
   - 位置源切换：可在 AMCL 定位和 LIO-SAM 里程计之间切换
   - 地图颜色配置：可自定义地图显示颜色
   - 缩放与拖拽：支持鼠标滚轮缩放和拖拽地图

## 5. 核心功能

### 5.1 SLAM 建图

系统支持多种 SLAM 算法，可根据需求选择最适合的算法：

- **FAST-LIO2**：基于迭代卡尔曼滤波的紧耦合激光-惯性里程计，计算效率高、实时性强
- **LIO-SAM**：基于因子图的松耦合激光-惯性里程计，回环检测能力强、全局优化精度高
- **Point-LIO**：基于点云的激光-惯性里程计，点云处理能力强、适应性好
- **Super-LIO**：高级激光-惯性里程计系统，鲁棒性强、精度高

### 5.2 导航功能

系统集成了完整的 Nav2 导航栈，包括：

- **全局路径规划**：规划从起点到目标点的全局路径
- **局部路径规划**：实时调整路径以避免障碍物
- **障碍物避障**：基于传感器数据实时检测和避开障碍物
- **动态重规划**：当环境发生变化时自动重新规划路径

### 5.3 地图管理

- **点云地图保存**：保存高精度点云地图
- **占用网格地图**：转换为 Nav2 可使用的 2D 占用网格地图
- **地图加载与使用**：加载已保存的地图进行导航

### 5.4 自主探索

系统支持使用 m-explore 包进行自主探索：

```bash
# 启动自主探索
ros2 launch m-explore explore.launch.py
```

## 6. 系统架构

### 6.1 项目结构

```
LIO-SAM_MID360_ROS2_PKG/
├── ros2/src/
│   ├── LIO-SAM_MID360_ROS2_DOG/      # LIO-SAM 实现
│   ├── FAST_LIO_ROS2_edit/           # FAST-LIO2 实现
│   ├── point_lio_ros2/               # Point-LIO 实现
│   ├── Super-LIO/                    # Super-LIO 实现
│   ├── SC_PGO_ROS2/                  # SC-PGO 姿态图优化
│   ├── nav2_dog_slam/                # 统一导航系统
│   ├── livox_ros_driver2/            # Livox 雷达驱动
│   ├── global_config/                # 全局配置管理
│   └── m-explore/                    # 自主探索
├── map_sample/                       # 地图示例
└── build_ros2.sh                     # 构建脚本
```

### 6.2 核心组件

1. **雷达驱动**：Livox MID360 激光雷达数据获取与处理
2. **SLAM 模块**：多种 SLAM 算法实现，负责定位与建图
3. **导航模块**：Nav2 导航栈，负责路径规划与避障
4. **地图管理**：点云地图与占用网格地图的生成与管理
5. **Web 控制**：基于浏览器的远程控制界面
6. **全局配置**：统一的配置管理系统，支持多主机配置

### 6.3 数据流程

1. **传感器数据获取**：从 Livox MID360 激光雷达和 IMU 获取原始数据
2. **数据预处理**：点云滤波、特征提取等
3. **SLAM 处理**：激光-惯性融合定位，地图构建
4. **导航规划**：基于地图和定位信息进行路径规划
5. **控制执行**：生成控制命令，驱动机器人运动
6. **地图更新**：根据新的传感器数据更新地图

## 7. 学习路径

### 7.1 阶段一：基础环境搭建（1-2周）

1. **操作系统安装**：Ubuntu 22.04 LTS
2. **ROS2 Humble 安装**：官方安装教程，环境变量配置
3. **依赖库安装**：Livox-SDK2，PCL，Boost，GTSAM 等

### 7.2 阶段二：系统架构学习（2-3周）

1. **项目结构分析**：目录结构，核心包和组件
2. **SLAM 算法基础**：激光 SLAM 原理，惯性导航原理
3. **ROS2 通信机制**：话题，服务，动作，节点生命周期管理

### 7.3 阶段三：核心功能开发（3-4周）

1. **雷达驱动使用**：Livox MID360 参数配置，点云数据处理
2. **SLAM 算法调优**：参数配置，传感器融合策略
3. **导航系统集成**：Nav2 配置，路径规划算法

### 7.4 阶段四：高级功能和优化（2-3周）

1. **地图管理**：点云地图保存和转换，多地图管理
2. **系统优化**：CPU/内存资源分配，ROS2 通信优化
3. **自主探索**：m-explore 使用，探索策略调优

### 7.5 阶段五：应用开发和部署（1-2周）

1. **Web 控制界面**：rosbridge_websocket 配置，浏览器控制界面使用
2. **系统部署**：硬件平台适配，启动脚本编写
3. **故障排除**：常见问题解决，调试工具使用

## 8. 启动路径

### 8.1 首次启动流程

1. **环境检查**：确认 Ubuntu 22.04 和 ROS2 Humble 已正确安装
2. **依赖安装**：按照 3.3 节安装所有依赖库
3. **项目构建**：按照 3.4 节构建项目
4. **配置修改**：根据实际硬件修改配置文件
5. **启动系统**：使用统一启动命令启动系统
6. **功能验证**：检查 SLAM 建图和导航功能是否正常

### 8.2 日常使用流程

1. **进入项目目录**：`cd ~/ros_ws/dog_slam/LIO-SAM_MID360_ROS2_PKG`
2. **选择 SLAM 算法**：设置 `SLAM_ALGORITHM` 环境变量
3. **启动系统**：`ros2 launch nav2_dog_slam lio_nav2_unified.launch.py`
4. **访问 Web 界面**：打开浏览器访问控制界面
5. **设置目标点**：在 Web 界面上点击地图设置导航目标
6. **监控运行状态**：观察系统运行状态和地图构建情况
7. **保存地图**：建图完成后保存地图

### 8.3 详细启动步骤（包含 Gazebo 仿真）

#### 8.3.1 启动 Gazebo 仿真环境

```bash
# 进入工作目录
cd /home/crazytree/ros_ws/dog_slam

# 启动 Gazebo 仿真环境
ros2 launch livox_gazebo_ros2_gpu_simulation my_robot.launch.py
```

#### 8.3.2 启动 TF 变换发布器

**打开新终端**：
```bash
# 启动 map 到 odom 的静态 TF 变换发布器
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
```

**打开新终端**：
```bash
# 启动 odom 到 base_footprint 的静态 TF 变换发布器
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_footprint
```

#### 8.3.3 启动地图服务器和生命周期管理器

**打开新终端**：
```bash
# 启动地图服务器
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=/home/crazytree/ros_ws/dog_slam/LIO-SAM_MID360_ROS2_PKG/map_sample/map.yaml -p use_sim_time:=true
```

**打开新终端**：
```bash
# 启动生命周期管理器激活地图服务器
ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -p node_names:=['map_server'] -p autostart:=true -p use_sim_time:=true
```

#### 8.3.4 启动使用仿真时间的 RViz

**打开新终端**：
```bash
# 启动使用仿真时间的 RViz
ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
```

**RViz 配置步骤**：
1. **设置 Fixed Frame**：
   - 在左侧"Displays"面板中，找到"Global Options"
   - 点击"Fixed Frame"旁边的下拉菜单
   - 选择"map"作为 Fixed Frame

2. **添加并配置 Map 显示插件**：
   - 点击左侧下方的"Add"按钮
   - 在弹出的对话框中，找到并选择"Map"
   - 点击"OK"按钮添加 Map 插件
   - 在左侧"Displays"面板中，找到刚刚添加的"Map"插件
   - 展开"Map"插件的配置选项
   - 确保"Topic"设置为"/map"
   - 确保"Color Scheme"设置为"map"
   - 调整"Alpha"值为合适的透明度（如 0.7）

#### 8.3.5 启动小车运动控制

**打开新终端**：
```bash
# 启动 planar_move 节点，用于控制小车运动
ros2 run planar_move planar_move
```

#### 8.3.6 启动导航功能

**打开新终端**：
```bash
# 启动 Nav2 导航系统
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true
```

### 8.4 地图保存与使用

#### 8.4.1 保存点云地图

```bash
# 保存点云地图
source install/setup.bash 
ros2 service call /lio_sam/save_map lio_sam/srv/SaveMap "{resolution: 0.05, destination: '/projects/LOAM/'}"
```

#### 8.4.2 转换为占用网格地图

```bash
# 当 SLAM 系统运行且地图构建完成后，保存为 PNG 格式
ros2 run nav2_map_server map_saver_cli -t /projected_map -f /path/to/map --fmt png
```

#### 8.4.3 使用保存的地图

修改 `global_config` 中的 `NAV2_DEFAULT_MAP_FILE` 路径，指向保存的地图文件，然后重启系统。

## 9. 核心 SLAM 算法分析

### 9.1 FAST-LIO2（推荐使用）

- **算法类型**：基于迭代卡尔曼滤波的紧耦合激光-惯性里程计
- **核心优势**：
  - 计算效率高，内存占用低
  - 实时性强，抗震性好
  - 适用于资源受限环境
- **适用场景**：
  - 实时建图和定位
  - 高频数据处理
  - 动态环境导航
- **配置文件**：`FAST_LIO_ROS2_edit/config/mid360.yaml`

### 9.2 LIO-SAM

- **算法类型**：基于因子图的松耦合激光-惯性里程计
- **核心优势**：
  - 回环检测能力强
  - 全局优化精度高
  - 地图一致性好
- **适用场景**：
  - 需要高精度全局地图
  - 存在重复场景的环境
  - 长距离导航
- **配置文件**：`lio_sam/config/liosam_params.yaml`

### 9.3 Point-LIO

- **算法类型**：基于点云的激光-惯性里程计
- **核心优势**：
  - 点云处理能力强
  - 适应性好
  - 鲁棒性高
- **适用场景**：
  - 复杂环境下的建图
  - 不规则场景定位
  - 多传感器融合
- **配置文件**：`point_lio_ros2/config/mid360.yaml`

### 9.4 Super-LIO

- **算法类型**：高级激光-惯性里程计系统
- **核心优势**：
  - 鲁棒性强
  - 精度高
  - 功能丰富
- **适用场景**：
  - 高精度导航任务
  - 复杂环境建图
  - 科研和开发
- **配置文件**：`Super-LIO/src/super_lio/config/livox_360.yaml`

## 10. 故障排除

### 10.1 常见问题

1. **雷达连接问题**
   - **症状**：无法获取点云数据
   - **解决方案**：
     - 检查雷达是否正确连接
     - 检查 Livox-SDK2 是否正确安装
     - 查看雷达驱动输出日志

2. **SLAM 系统无法启动**
   - **症状**：启动时出现错误或崩溃
   - **解决方案**：
     - 检查依赖项是否完整安装
     - 检查构建是否成功完成
     - 查看启动日志排查错误
     - 确保 ROS2 环境变量正确设置

3. **Web 界面无法访问**
   - **症状**：浏览器无法打开控制界面
   - **解决方案**：
     - 检查 rosbridge_websocket 是否正常启动
     - 检查网络连接和端口是否开放
     - 查看浏览器控制台错误信息
     - 确保防火墙未阻止端口访问

4. **导航精度问题**
   - **症状**：导航路径偏移或不稳定
   - **解决方案**：
     - 检查 AMCL 参数配置
     - 调整代价地图参数
     - 优化传感器数据质量
     - 确保地图质量良好

5. **TF 变换错误**
   - **症状**：出现 TF 相关错误，如 "Fixed Frame [map] does not exist"
   - **解决方案**：
     - 确认 base_footprint 坐标系设置正确
     - 检查传感器坐标系转换
     - 查看 TF 树是否完整
     - 确保时间同步正常
     - 重新启动静态 TF 变换发布器：
       ```bash
       # 启动 map 到 odom 的静态 TF 变换发布器
       ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
       
       # 启动 odom 到 base_footprint 的静态 TF 变换发布器
       ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 odom base_footprint
       ```

6. **地图显示问题**
   - **症状**：RViz 中显示 "No map received" 或地图为空
   - **解决方案**：
     - 确保地图服务器和生命周期管理器正常运行
     - 确保 TF 变换链完整：`map → odom → base_footprint → base_link`
     - 使用仿真时间启动 RViz：`ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true`
     - 确保 RViz 配置正确：
       - Fixed Frame 设置为 "map"
       - Map 插件的 Topic 设置为 "/map"
       - Color Scheme 设置为 "map"

7. **地图服务器未激活**
   - **症状**：地图服务器处于非激活状态，无法发布地图数据
   - **解决方案**：
     - 启动生命周期管理器激活地图服务器：
       ```bash
       ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -p node_names:=['map_server'] -p autostart:=true -p use_sim_time:=true
       ```

8. **时间同步问题**
   - **症状**：系统组件之间时间不同步，导致数据无法正确处理
   - **解决方案**：
     - 确保所有组件使用仿真时间：`-p use_sim_time:=true`
     - 检查 Gazebo 是否正常发布 `/clock` 话题
     - 重启系统组件，确保时间同步

### 10.2 调试工具

1. **查看 TF 树**
   ```bash
   ros2 run tf2_tools view_frames.py
   ```

2. **查看话题列表**
   ```bash
   ros2 topic list
   ```

3. **查看节点信息**
   ```bash
   ros2 node list
   ```

4. **查看话题数据**
   ```bash
   ros2 topic echo /topic_name
   ```

5. **查看服务列表**
   ```bash
   ros2 service list
   ```

## 11. 系统优化

### 11.1 硬件优化

- **CPU 资源分配**：
  - Livox 驱动 → A76 大核 2/3
  - FAST-LIO → A76 大核
  - Nav2 (planner/controller/BT) → A76
  - Costmap、AMCL、TF → 小核
  - RVIZ → 小核（不在主板上跑）

- **内存优化**：
  - 减少内存碎片造成的卡顿
  - TCP buffer/UDP buffer 优化
  - 可选使用 Hugepages（可能提高 10% 性能）

### 11.2 软件优化

- **DDS/ROS2 通信优化**：
  - 使用 CycloneDDS
  - 降低点云复制次数
  - 使用 loaned messages，降低内存 alloc

- **算法参数调优**：
  - 根据环境调整激光雷达参数
  - 优化 SLAM 算法参数
  - 调整导航系统参数

### 11.3 系统稳定性提升

- **TF 延迟/漂移监控**：
  - TF future/past extrapolation
  - 滤除不合规 TF
  - 每秒监控 TF 延迟

- **自动恢复策略**：
  - SLAM 崩溃自动重启
  - Nav2 崩溃自动恢复
  - Livox 数据延迟自动切换

## 12. 常见问题

### 12.1 安装问题

**Q: 安装 Livox-SDK2 时出现编译错误怎么办？**
A: 检查依赖库是否完整安装，确保使用最新版本的 Livox-SDK2，参考官方安装指南。

**Q: 构建项目时出现依赖错误怎么办？**
A: 检查所有依赖库是否正确安装，确保 ROS2 环境变量正确设置，尝试重新构建。

### 12.2 运行问题

**Q: 启动系统后没有点云数据怎么办？**
A: 检查雷达连接，查看雷达驱动日志，确认雷达配置文件正确。

**Q: SLAM 建图时地图漂移严重怎么办？**
A: 检查 IMU 校准，调整 SLAM 算法参数，确保传感器时间同步。

**Q: 导航时机器人碰撞障碍物怎么办？**
A: 调整代价地图参数，优化避障算法设置，确保传感器数据准确。

### 12.3 功能问题

**Q: 如何切换不同的 SLAM 算法？**
A: 通过设置 `SLAM_ALGORITHM` 环境变量来选择不同的算法。

**Q: 如何保存和加载地图？**
A: 使用服务调用保存点云地图，使用 map_saver_cli 保存占用网格地图，修改配置文件加载地图。

**Q: 如何进行自主探索？**
A: 启动系统后，运行 `ros2 launch m-explore explore.launch.py` 启动自主探索。

## 13. 总结

LIO-SAM_MID360_ROS2_PKG 是一个功能强大、架构清晰的 ROS2 SLAM 系统，为 Livox MID360 激光雷达提供了完整的 SLAM 和导航解决方案。通过本文档的指导，您可以快速搭建环境、启动系统、开发应用，并根据实际需求进行优化和扩展。

系统的统一启动文件设计使得不同 SLAM 算法的切换变得简单直观，同时 Nav2 导航框架的集成提供了完整的导航功能。无论是科研还是实际应用，这个系统都提供了一个理想的平台，可以根据具体需求进行定制和扩展。

随着技术的不断发展，系统也在持续更新和优化，未来将支持更多功能和硬件平台，为机器人自主导航领域提供更强大的工具和解决方案。