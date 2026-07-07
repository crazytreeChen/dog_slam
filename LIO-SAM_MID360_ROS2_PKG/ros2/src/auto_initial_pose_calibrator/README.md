# auto_initial_pose_calibrator — 自动初始位姿校准器

## 概述

`auto_initial_pose_calibrator` 是 dog_slam 项目的核心节点之一，用于解决 **"绑架机器人"（Kidnapped Robot）问题**——机器人在已知地图上任意位置开机后，无需人工干预，自动确定自身在地图上的初始位姿。确定位姿后，节点将 `/initialpose` 发布给 AMCL，由 AMCL 接管后续的跟踪定位。

本节点支持 **室内主动模式**、**室内被动模式** 和 **室外 RTK/GPS 模式** 三种工作模式。节点默认使用 `rkbot` namespace。

---

## 功能特性

| 特性 | 描述 |
|------|------|
| **室内主动模式** | 机器人自动旋转 360° 采集雷达数据 → 多算法全局匹配 → 最优候选筛选 → 运动消歧 → 发布初始位姿 |
| **室内被动模式** | 不控制机器人运动，后台持续采集雷达帧并周期性进行全局匹配，可选自动发布 `/initialpose` |
| **室外 RTK 模式** | 利用 RTK GPS + 双天线航向角，通过标定的 LLA→Map 变换直接发布 `/initialpose` |
| **多算法融合匹配** | 三级匹配流水线：双模板光线投射 → 距离场全局匹配 → 似然场网格分层搜索，按优先级自动切换 |
| **快速模式** | 跳过旋转采集环节，利用开机静止数帧直接进行全局粒子匹配 |
| **运动消歧** | 多候选位姿时，主动移动到信息增益最大的方向，通过运动约束筛选出唯一正确位姿 |
| **多帧时序一致性过滤** | 利用帧间一致性剔除人、动物、动态障碍物，提高静态墙壁匹配精度 |
| **自适应协方差** | 根据墙壁覆盖率和匹配质量动态调整 `/initialpose` 的协方差矩阵，加速 AMCL 收敛 |
| **偏差监控** | 发布校准位姿后持续对比 AMCL 输出，偏差过大时告警 |
| **离线工具链** | `scripts/` 目录下提供多种离线匹配算法实验脚本 |

---

## 架构设计

### 包结构

```
auto_initial_pose_calibrator/
├── config/
│   └── auto_initial_pose_calibrator.yaml    # 全部配置参数
├── launch/
│   └── auto_initial_pose_calibration.launch.py  # 启动文件
├── scripts/                                 # 离线工具脚本
│   ├── global_match.py                      # 距离场全局匹配算法
│   ├── global_match2.py                     # 双模板光线投射匹配算法
│   ├── calculate_and_visualize_lidar_positions.py  # 离线多分辨率搜索
│   ├── opencode_likelihood_search.py        # 离线似然场全局搜索
│   ├── opencode_multistep_localizer.py      # 离线多步递推定位
│   └── ...                                  # 其他实验/工具脚本
├── src/
│   ├── auto_initial_pose_calibrator.py       # 主节点（~3383 行）
│   └── calib_lib/                           # 算法子包（拆分自主节点）
│       ├── __init__.py                      # IndoorPhase 枚举定义
│       ├── scan_utils.py                    # 扫描↔点云转换、离群点/FRF过滤
│       ├── icp.py                           # 帧间 ICP 匹配
│       ├── temporal.py                      # 多帧时序一致性过滤
│       ├── scoring.py                       # 似然场、距离场、双模板评分引擎
│       ├── submap.py                        # 子图构建
│       ├── matching.py                      # 扫描匹配编排（hierarchical/multistep/passive）
│       ├── control.py                       # 主动运动选择与避障
│       ├── motion_tracker.py                # 运动状态跟踪与漂移检测
│       └── rtk.py                           # RTK/GPS → Map 坐标变换
├── CMakeLists.txt
├── package.xml
├── CHANGELOG.md
└── UNFINISHED_TASKS.md
```

### `calib_lib` 模块职责

| 模块 | 职责 | 对外接口 |
|------|------|----------|
| `scan_utils` | LaserScan → 点云转换、半径离群点过滤、FRF 帧过滤、角度规整 | `scan_to_points()`, `filter_scan_outliers()`, `frf_filter_frame()`, `ScanFilterConfig` |
| `icp` | 帧间 ICP 匹配与 Scan 对齐 | `icp_match()`, `icp_align_scans()` |
| `temporal` | 多帧时序一致性过滤（动态物体剔除） | `temporal_consistency_filter()` |
| `scoring` | 似然场/距离场/双模板三种评分体系、墙壁覆盖率、置信度计算、两级局部搜索 | `build_likelihood()`, `score_points()`, `score_pose_raycast()`, `compute_wall_coverage()`, `compute_confidence()`, `local_search_two_stage()` |
| `submap` | 多帧 Scan 按 Odom 旋转拼接为合成 Scan | `SubmapBuilder` |
| `matching` | 编排 hierarchical / multistep / passive 三种匹配路径 | `ScanMatcher` |
| `control` | 主动运动选择（信息增益方向）、P 控制器、避障 | `MotionController` |
| `motion_tracker` | 运动状态跟踪、停止检测、odom→map 漂移检测 | `MotionTracker` |
| `rtk` | RTK 消息 → 地图坐标 + 协方差矩阵 | `rtk_to_map_coords()`, `build_pose_covariance()` |

> 所有 `calib_lib` 子模块不硬依赖 ROS（`rclpy`），可在 Windows 上直接 import 进行单元测试。

---

## 三种工作模式

### 1. 室内主动模式（Active Indoor）

**状态机流程**（10 个阶段）：

```
IDLE → BOOT_DELAY(2s静止) → ROTATING_360(360°旋转采集)
  → ACTIVE_MULTISTEP(多步递推匹配)
  → [快速模式: 发布→DONE]
  → COLLECTING_SUBMAP1(收集30帧合成子图)
  → ROUGH_MATCHING(全局匹配 Top-N 候选)
  → SELECTING_ACTIVE_MOTION(选择最优探索方向)
  → MOVING(执行移动 + 避障监控)
  → COLLECTING_SUBMAP2(收集第2个子图)
  → FILTERING(传播候选位姿 + 子图2重新评分)
  → [收敛: 发布→DONE | 未收敛: 重试←SELECTING_ACTIVE_MOTION]
```

**核心机制**：
- **旋转采集**：360° 全覆盖扫描，消除方向盲区
- **多算法匹配**：三算法级联，按优先级自动选择（详见 [匹配算法](#匹配算法架构)）
- **运动消歧**：多个候选位姿时，主动移动到信息增益最大的方向，利用运动前后的 odom 约束筛选唯一位姿
- **避障保护**：运动中使用雷达前方扇形区域实时监控，遇到障碍物立即停机

### 2. 室内被动模式（Passive Indoor）

```
PASSIVE_COLLECTING → 定时器(passive_interval秒) → PASSIVE_MATCHING → 循环
```

**核心机制**：
- **不控制机器人运动**：仅被动采集雷达数据
- **循环缓冲区**：最多保留 `passive_buffer_max` 帧，丢旧取新
- **ICP 帧间合并**：每次取最近 `passive_frame_count` 帧，用 ICP 拼接成子图
- **里程计融合验证**：有历史估计时，优先用里程计推算验证（跳过重匹配，节省计算）
- **AMCL 交叉验证**：计算结果与 AMCL 当前输出对比，偏差超阈值触发重匹配
- **二次验证**：匹配结果与上次位姿偏差超阈值时，使用更多帧 + 更精细搜索进行二次确认

### 3. 室外 RTK/GPS 模式

- 订阅 `/rtk_pvh`（`robots_dog_msgs/UniRtkPvh`）获取 RTK 定位 + 双天线航向
- 通过 `calibration_file` 中预标定的 LLA→Map 变换矩阵，将经纬度转换为地图坐标
- 直接构造 `PoseWithCovarianceStamped` 发布 `/initialpose`

---

## 匹配算法架构

本节点实现了三种全局匹配算法，按优先级从高到低排列：

### 算法 1: 双模板光线投射全局匹配（`global_match2.py`）

> **优先级最高，默认启用** (`use_dual_template_matching: true`)

**原理**：
- 从占据栅格地图构建两张模板图：
  - **命中模板（Hit Map）**：距离墙壁的倒数，越靠近墙壁得分越高
  - **射线惩罚模板（Ray Penalty Map）**：墙壁=1.0，未知=2.0，用于惩罚穿过墙壁的射线
- 对每个角度，同时渲染命中模板（扫描点位置）和射线模板（Bresenham 直线）
- 使用 **`cv2.matchTemplate`** 进行全图卷积——O(N) 复杂度，极快
- 得分 = 命中分 − 射线惩罚分 × `penalty_weight`
- **自由空间面积比惩罚**：利用积分图像 O(1) 区域查询，防止对称走廊误匹配
- **两阶段**：粗搜（3°步长，仅命中模板）→ 精搜（0.5°步长，双模板）在粗搜最佳角度附近

**自适应缩放**：当有效地图像素超过阈值时，自动对地图进行 2x/3x/4x 下采样并等比增大角度步长。

**室外/室内参数分离**：系统根据 RTK/GPS 信号自动选择参数组：
- 室内：墙壁覆盖率阈值 30%，缩放上限 1.5x
- 室外：墙壁覆盖率阈值 15%，缩放上限 2.5x

### 算法 2: 距离场全局匹配（`global_match.py`）

> 优先级第二 (`use_distance_field_matching: true`)

- 使用 `cv2.distanceTransform` 构建距离场（无高斯截断）
- 随机采样 N 个自由空间位置（默认 1000）
- 对每个位置 × 角度，计算变换后的扫描点到墙壁的平均像素距离
- 距离越小得分越好
- Top 候选经 NMS + 似然场两阶段局部精修

### 算法 3: 似然场网格分层搜索

> 原始算法，当上述两种算法未启用或失败时回退

- Phase 1：全图网格搜索（1.5m 步长，10° 角度步长）
- NMS 去重
- RayCast 重排序：对每个候选位姿进行射线追踪，比较测量距离 vs 预期墙壁命中距离
- Phase 2：局部精搜（±2.5m, 0.3m 步长, ±15°, 3° 步长）Top-3 候选
- 最终去重 + softmax 概率归一化

### 算法优先级决策流程

```
扫描匹配请求
  ├─ use_dual_template_matching=true ∧ cv2可用
  │   └─ 双模板光线投射全局匹配 → 完成
  ├─ use_distance_field_matching=true ∧ cv2可用
  │   └─ 距离场全局匹配 → 完成
  └─ 回退：似然场网格分层搜索 → 完成
```

---

## ROS2 接口

### 订阅（Subscriptions）

| 话题 | 类型 | QoS | 说明 |
|------|------|-----|------|
| `scan_topic`（默认 `scan`） | `sensor_msgs/LaserScan` | BestEffort | 激光雷达扫描数据 |
| `map_topic`（默认 `map`） | `nav_msgs/OccupancyGrid` | TransientLocal | 占据栅格地图 |
| `odom_topic`（默认 `lio/odom`） | `nav_msgs/Odometry` | BestEffort | LIO 里程计 |
| `amcl_pose_topic` | `geometry_msgs/PoseWithCovarianceStamped` | Volatile | AMCL 当前估计位姿 |
| `gps_topic`（默认 `/fix`） | `sensor_msgs/NavSatFix` | BestEffort | GPS 原始定位（室外模式） |
| `rtk_topic`（默认 `/rtk_pvh`） | `robots_dog_msgs/UniRtkPvh` | BestEffort | RTK 定位 + 双天线航向（室外模式） |
| `initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | Volatile | 监听外部发布的初始位姿 |

### 发布（Publishers）

| 话题 | 类型 | 说明 |
|------|------|------|
| `initialpose` | `geometry_msgs/PoseWithCovarianceStamped` | **核心输出**：计算出的初始位姿（给 AMCL） |
| `cmd_vel_topic`（默认 `/cmd_vel`） | `geometry_msgs/Twist` | 运动控制指令（主动模式） |
| `debug/candidates` | `geometry_msgs/PoseArray` | 候选位姿可视化 |
| `debug/submap_scan` | `sensor_msgs/LaserScan` | 合成子图 Scan 可视化 |
| `debug/auto_initial_pose` | `geometry_msgs/PoseWithCovarianceStamped` | 校准结果（调试用，不影响 AMCL） |
| `debug/pose_comparison` | `geometry_msgs/PoseArray` | 对比模式下的 pose 对比 |
| `debug/odom_est_pose` | `geometry_msgs/PoseStamped` | 里程计推算的估计位姿 |

### 服务（Services）

| 服务名 | 类型 | 说明 |
|--------|------|------|
| `start_auto_calibration` | `std_srvs/Trigger` | 触发自动校准（自动选择室内/室外模式） |
| `start_active_calibration` | `std_srvs/Trigger` | 启动室内主动校准 |
| `start_passive_calibration` | `std_srvs/Trigger` | 启动室内被动持续定位 |
| `stop_passive_calibration` | `std_srvs/Trigger` | 停止被动定位 |
| `auto_calibration_status` | `std_srvs/Trigger` | 查询当前状态与最佳估计位姿 |
| `reset_calibration` | `std_srvs/Trigger` | 重置校准状态到 IDLE |
| `toggle_auto_publish` | `std_srvs/Trigger` | 切换自动发布 `/initialpose` 开关 |
| `set_manual_ground_truth` | `std_srvs/SetBool` | 设置手动真值（调试对比模式） |

### 定时器（Timers）

| 回调 | 频率/间隔 | 说明 |
|------|-----------|------|
| `_indoor_loop` | 10 Hz | 室内模式主循环（状态机驱动） |
| `_outdoor_loop` | `publish_rate`（默认 2 Hz） | 室外模式 RTK 发布循环 |
| `_mode_timer` | 1 Hz | 自动检测 RTK/GPS 信号切换室内/室外模式 |
| `_passive_timer_cb` | `passive_interval`（默认 30s） | 被动模式定时匹配触发 |
| `_check_map_and_fallback` | 1.5s | 检查地图是否就绪 |
| `_check_deviation` | 2 Hz | AMCL 偏差监控 |

---

## 发布初始位姿的置信度门控

为防止发布错误的初始位姿导致 AMCL 发散，系统设置了三级置信度门控：

| 阈值 | 默认值 | 作用 |
|------|--------|------|
| `confidence_threshold_update` | 0.50 | 置信度高于此值，允许更新内部校准位姿 |
| `confidence_threshold_publish` | 0.40 | 置信度高于此值，允许发布 `/initialpose` |
| `confidence_threshold_amcl` | 0.35 | 低置信度时可辅助 AMCL（当前仅监控） |

**置信度计算因素**（来自 `compute_confidence()`）：
- **墙壁覆盖率**（wall_coverage）：扫描点落在地图墙壁区域的占比（越高越好）
- **有效区域覆盖率**（coverage）：扫描点落在地图已知区域的占比（越高越好）
- **似然场评分**：扫描点距离墙壁的平均距离（越小越好）
- **协方差自适应**：
  ```
  pos_sigma = 0.8 / max(wall_cov, 0.1) × max(1 - coverage, 0.3)
  yaw_sigma = 20.0 / max(wall_cov, 0.1) × (1 - coverage + 0.2)
  ```

---

## 多帧时序一致性过滤

室内环境中的动态物体（行人、动物等）可能导致误匹配。系统利用帧间一致性过滤动态障碍物：

1. 多帧扫描点在同一世界坐标下合并
2. 对每个点搜索邻域（半径 `temporal_merge_radius`，默认 0.15m）
3. 只有被 **至少 N 个不同帧**（`temporal_merge_min_frames`，默认 5）观察到的点才保留
4. 配合半径离群点过滤（`scan_outlier_filter`），进一步去除孤立噪点

---

## Move Stop & Deviation Detection（运动状态跟踪）

被动模式下增加了运动感知能力：

| 功能 | 参数 | 说明 |
|------|------|------|
| 运动检测 | `motion_speed_threshold: 0.05` | 线速度 > 阈值判定为运动中 |
| 停止检测 | `motion_settle_time: 1.0s` | 停止后等待稳定时间 |
| 停止冷却 | `motion_stop_cooldown: 10.0s` | 停止事件触发后的冷却避免频繁匹配 |
| Odom 漂移检测 | `passive_odom_deviation_threshold: 0.3m` | odom→map 推算偏差超阈值触发重匹配 |
| AMCL 交叉验证 | `passive_amcl_deviation_trigger: 1.0m` | 计算结果与 AMCL 偏差超阈值才更新位姿 |

---

## 依赖

### ROS2 包依赖

| 包名 | 用途 |
|------|------|
| `rclpy` | ROS2 Python 客户端库 |
| `sensor_msgs` | LaserScan, NavSatFix 消息 |
| `nav_msgs` | OccupancyGrid, Odometry 消息 |
| `geometry_msgs` | Pose, Twist, PoseArray 等几何消息 |
| `std_srvs` | Trigger 服务 |
| `global_config` | 跨机器/跨平台统一配置 |
| `robots_dog_msgs` | UniRtkPvh 自定义消息（室外 RTK） |

### Python 依赖

| 库 | 必要性 | 用途 |
|----|--------|------|
| `numpy` | 必需 | 数值计算、矩阵运算 |
| `opencv-python`（cv2） | 高度推荐 | 双模板匹配（`cv2.matchTemplate`）、距离场（`cv2.distanceTransform`） |
| `scipy` | 可选 | `scipy.ndimage` 用于距离场预处理（无 scipy 时有回退实现） |
| `yaml` | 必需 | 读取标定文件 |
| `transforms3d` | 可选 | 四元数/欧拉角转换（有回退实现） |

---

## 命名空间与话题映射

节点默认使用 `rkbot` namespace，所有相对路径话题自动拼接为 `/rkbot/<topic>`：

| 参数 | 默认值 | 实际话题 |
|------|--------|----------|
| `scan_topic` | `scan` | `/rkbot/scan` |
| `odom_topic` | `lio/odom` | `/rkbot/lio/odom` |
| `map_topic` | `map` | `/rkbot/map` |
| `amcl_pose_topic` | `amcl_pose` | `/rkbot/amcl_pose` |

| 全局话题（绝对路径） | 说明 |
|-----------------------|------|
| `/cmd_vel` | 运动控制指令 |
| `/rtk_pvh` | RTK 定位 + 双天线航向 |
| `/fix` | GPS 原始定位 |

可通过 launch 参数覆盖 namespace：`ns:=mybot`。

---

## 离线工具脚本

`scripts/` 目录提供丰富的离线实验工具，用于算法原型验证：

| 脚本 | 说明 |
|------|------|
| `global_match2.py` | 双模板光线投射全局匹配算法（核心匹配算法） |
| `global_match.py` | 距离场全局匹配算法 |
| `calculate_and_visualize_lidar_positions.py` | 全地图多分辨率网格搜索 + 可视化 |
| `opencode_likelihood_search.py` | 离线似然场全局搜索实验 |
| `opencode_multistep_localizer.py` | 离线多步递推定位实验 |
| `calibrate_tf_odom_to_map.py` | 标定 odom→map 变换 |
| `diagnose_gt.py` | GT 诊断工具 |
| `scan_map_overlay.py` | Scan 与地图叠加可视化 |

---

## 已知限制与未完成事项

> 详见 `UNFINISHED_TASKS.md`

1. **180° 航向角消歧**：在对称环境（如走廊）中，多帧融合点云旋转 180° 后与地图轮廓高度相似，当前评分函数难以区分正反方向
2. **离线 GT 诊断未跑通**：由于本地 Conda 环境问题，带有诊断输出的脚本尚未成功运行
3. **光线投射消歧**：计划引入 Ray-casting 模拟来彻底破除 180° 镜像模糊
4. **轨迹碰撞检查**：计划利用轨迹延伸方向进行墙体碰撞检查辅助消歧

---

## 版本历史

> 详见 `CHANGELOG.md`

| 版本 | 日期 | 关键改动 |
|------|------|----------|
| v2.0 | 2026-06-16 | 评分引擎重构、被动持续定位模式、偏差监控、自适应协方差、网格搜索替代粒子搜索、双模板匹配 |
| v1.0 | - | 初始版本：室内主动模式 + 室外 RTK 模式 |

---

## 相关文档

- [[nav2_dog_slam]] — 统一 launch 系统与 Nav2 集成（Obsidian）
- [[global_config]] — 配置单点真理
- `UNFINISHED_TASKS.md` — 未完成任务详情与解决方案思路
- `CHANGELOG.md` — 完整改动日志
