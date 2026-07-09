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

## RTK 导航纠偏（rtk_nav_bridge.launch.py）

导航阶段用 GPS/RTK 纠正机器人在 `/map` 下的偏离位姿，独立于建图逻辑。
通过 `correction_mode` 参数切换两种纠偏模式：

| 模式 | 节点 | 说明 |
|------|------|------|
| `continuous`（默认） | `rtk_continuous_injector` | 首次 `/initialpose` 锚定 map 位姿，RTK 为绝对权威，LIO 仅辅助，连续平滑注入 `/<ns>/initialpose`，丝滑降级，不读 AMCL、不依赖 `map_gps_origin.yaml` |
| `threshold` | `rtk_pose_monitor` | 旧阈值跳变纠偏，深度耦合 AMCL，仅作回退保留 |

### 连续注入模式（推荐）

- **锚定**：首次收到绝对话题 `/initialpose`（室外默认初始点位）即记录地图位姿
  `(mx0,my0,maw0)`，多帧平均首个有效 RTK fix 反算 UTM 锚点 `(e0,n0)` 与
  `θ0 = maw0 + radians(h0)`，建立 `map↔经纬度↔yaw` 坐标体系。
- **连续注入**：按 `inject_rate`（默认 0.5Hz）将后续 RTK/GPS fix 推算为 map 位姿，
  叠加 initialpose 偏移发布；仅当 LIO 累计位移超 `reinject_motion_margin`（0.3m）
  或 yaw 超 `reinject_yaw_margin_deg`（5°）或定位质量显著改善时才重发，避免反复重置 AMCL。
- **LIO 融合**：订阅 `/Odometry` 积分相对位移，用于触发纠偏时机、RTK 丢失短时桥接、
  无航向时 yaw 平滑；`use_lio_deadreckon=false` 退回纯 RTK 模式。
- **丝滑降级**：逐帧按"当前帧有无可用航向"决定发布值；源切换（RTK↔GPS）只改位置协方差/
  阈值、**绝不改 yaw**；无航向时仅发位置、yaw 协方差放大交 AMCL 自收敛。

启动：

```bash
# 默认 continuous 模式
ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot
# 回退到旧阈值模式
ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot correction_mode:=threshold
```

关键参数见 `config/rtk_continuous_injector.yaml`。

## 后续计划

测试验证通过后，将 `gps_fusion` 包的核心逻辑合并回 `nav2_dog_slam`。
