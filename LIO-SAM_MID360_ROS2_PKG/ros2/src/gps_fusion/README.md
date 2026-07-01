# GPS Fusion 独立测试包

GPS/RTK 融合管道的独立 ROS2 包，**不依赖 nav2_dog_slam**，可单独构建、测试和验证。

## 目录结构

```
gps_fusion/
├── package.xml
├── CMakeLists.txt
├── gps_fusion/                  # Python 模块
│   ├── __init__.py
│   ├── gps_preprocessor.py      # GPS预处理节点
│   └── gps_test_steps.py        # 分步测试脚本
├── config/
│   ├── gps_ekf.yaml             # EKF融合配置
│   └── navsat_transform.yaml    # 坐标转换配置
└── launch/
    ├── gps_fusion.launch.py     # 全链路启动（推荐）
    └── gps_preprocessor.launch.py # 仅预处理器启动
```

## 架构

```
                     gps_fusion 包
┌──────────────────────────────────────────────────┐
│                                                  │
│  /fix ──► gps_preprocessor ──► /fix_filtered    │
│                                   │              │
│                    ┌──────────────┘              │
│                    ▼                             │
│            navsat_transform_node                 │
│                    │                             │
│                    ▼                             │
│             /odometry/gps                        │
│                    │                             │
│  /lio_odom ────┬───▼                             │
│                ▼                                  │
│          ekf_filter_node                         │
│                │                                  │
│                ▼                                  │
│      /odometry/gps_fused                         │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 快速开始

### 构建

```bash
cd LIO-SAM_MID360_ROS2_PKG/ros2
colcon build --symlink-install --packages-select gps_fusion
source install/setup.bash
```

### 前提条件

- SLAM 系统已运行（LIO 发布里程计话题，如 `/Odometry` 或 `/lio/odom`）
- GPS 驱动已运行（发布 `/fix` 话题）

### 方式一：Launch 文件启动（推荐）

```bash
# 仅预处理器
ros2 launch gps_fusion gps_preprocessor.launch.py

# 全链路融合
ros2 launch gps_fusion gps_fusion.launch.py lio_odom_topic:=/lio/odom

# 仿真模式
ros2 launch gps_fusion gps_fusion.launch.py use_sim_time:=true
```

### 方式二：测试脚本分步验证

```bash
# 需要 source install/setup.bash 后执行

# Step 1: 仅预处理器
python3 gps_fusion/gps_test_steps.py --step 1

# Step 2: + 坐标转换
python3 gps_fusion/gps_test_steps.py --step 2 --slam super_lio

# Step 3: 全链路融合
python3 gps_fusion/gps_test_steps.py --step 3 --lio-odom /Odometry

# 交互式逐步
python3 gps_fusion/gps_test_steps.py --step all
```

## 各节点功能

### gps_preprocessor

- 订阅 `/fix` (sensor_msgs/NavSatFix)
- 数据质量过滤：状态检查、NaN过滤、精度门槛（GPS 1.0m / RTK 0.02m）
- RTK模式自动检测
- 发布 `/fix_filtered` (NavSatFix)、`/fix_utm` / `/fix_odom` (Odometry)、`/gps/status` (Bool)

### navsat_transform_node

- 来自 `robot_localization` 包
- 将 GPS 经纬度转换为 map 坐标系下的 odom
- 输出 `/odometry/gps`

### ekf_filter_node

- 来自 `robot_localization` 包
- 融合 LIO 里程计 + GPS 全局约束
- 输出 `/odometry/gps_fused`

## 关键话题

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/fix` | NavSatFix | 订阅 | 原始GPS（来自GPS驱动） |
| `/fix_filtered` | NavSatFix | 发布 | 预处理后GPS |
| `/fix_utm` | Odometry | 发布 | UTM绝对坐标 |
| `/odometry/gps` | Odometry | 发布 | navsat转换输出 |
| `/odometry/gps_fused` | Odometry | 发布 | EKF融合输出 |
| `/gps/status` | Bool | 发布 | GPS可用状态 |

## Launch 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lio_odom_topic` | `/Odometry` | LIO里程计话题 |
| `gps_topic` | `/fix` | 原始GPS话题 |
| `imu_topic` | `/livox/imu` | IMU话题 |
| `use_sim_time` | `false` | 仿真时间 |
| `utm_zone` | `50` | UTM区域编号 |
| `ns` | `` | 命名空间 |

## 验证清单

- [ ] `/fix_filtered` 有正常输出，经纬度无NaN
- [ ] `/gps/status` 输出 True（连续5帧有效GPS后）
- [ ] RTK模式下精度 < 0.02m
- [ ] `/odometry/gps` 位置与GPS实际位置吻合
- [ ] `/odometry/gps_fused` 能跟踪机器人运动
- [ ] 对比原始LIO odom，融合后漂移减小

## 后续计划

测试验证通过后，将 `gps_fusion` 包的核心逻辑合并回 `nav2_dog_slam`。
