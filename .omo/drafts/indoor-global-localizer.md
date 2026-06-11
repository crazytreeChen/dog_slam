# 工作计划：室内全局定位器 — Submap匹配 + 主动运动策略

## TL;DR

> **目标**: 实现 indoor_global_localizer 节点，按用户设计的流程完成室内自动定位
>
> **核心流程**: 静止→构建Submap→多分辨率匹配→主动运动(安全+信息增益)→循环筛选→发布/initialpose
>
> **对比旧方案**: 新增 Submap 表示、多分辨率匹配、主动运动策略、安全避障、收敛循环

## 需求确认

### 你的流程
```
开机 → 静止2秒 → 累积20帧Scan → 构建Submap1
  → 多分辨率全局匹配 → Top N候选Pose
  → 主动运动策略(安全+信息增益) → 前进0.8m+转向
  → 构建Submap2
  → Odom传播候选 → 重新评分 → 筛选
  → 检查唯一收敛 → 否→循环 是→发布/initialpose → 导航
```

### 关键技术要求
- **避障**: 不固定前进1m。评估4方向安全+信息增益，选最优方向
- **收敛**: 最佳/次佳 > 2x 或 唯一候选 → 发布
- **循环**: 最多5次运动尝试，未收敛使用最佳候选

## 涉及文件

### 新建文件
| 文件 | 说明 |
|------|------|
| `src/indoor_global_localizer.py` | 核心节点 — Submap/似然场/主动运动/收敛检查 |
| `config/indoor_global_localizer.yaml` | 参数配置 |
| `launch/indoor_global_localization.launch.py` | Launch文件 |

### 修改文件
| 文件 | 变更 |
|------|------|
| `CMakeLists.txt` | 添加 indoor_global_localizer.py 安装规则 |

## 架构设计

### 类结构
```
Submap
  ├── build_from_scans(scans) → 累积20帧Scan → 降采样 → 2D点集
  ├── transform(x, y, yaw) → 变换到世界坐标系
  └── size → 点数

MultiResLikelihoodField
  ├── build(map_data, info) → 构建原生+4x降采样似然场
  ├── lookup(wx, wy, coarse) → 查似然场值
  └── ready → 是否就绪

IndoorGlobalLocalizer (Node)
  ├── Phase状态机: IDLE→STATIONARY→GLOBAL_MATCH→PLANNING_MOTION→EXECUTING→FILTERING→CONVERGED
  ├── 服务: /start_indoor_localization, /localization_status, /abort_localization
  ├── 发布: /cmd_vel, /initialpose
  └── 订阅: /map, /scan, /odom
```

### 状态机流转
```
IDLE ──[start服务]──→ STATIONARY ──[20帧满]──→ GLOBAL_MATCH
                                                   │
GLOBAL_MATCH ──[粗+精匹配完成]──→ PLANNING_MOTION  │
                                      │              │
PLANNING_MOTION ──[选方向]──→ EXECUTING_MOTION     │
                                  │                  │
EXECUTING_MOTION ──[0.8m到达]──→ FILTERING          │
                                    │                │
FILTERING ──[收敛?]──→ CONVERGED ──→ 发布/initialpose
          │
          └──[未收敛&未超次数]──→ PLANNING_MOTION (循环)
          └──[超次数]──→ 用最佳候选发布
```

### 多分辨率匹配算法
```
输入: Submap1 (N个2D点)
步骤1 — 粗分辨率:
  - 似然场4x降采样 (max-pool保守)
  - 自由空间采样1000粒子
  - 每粒子评分Submap1(使用粗似然场, 500个采样点)
  - 取top 50候选

步骤2 — 精分辨率:
  - 每个粗候选附近采样10粒子 (±1m, ±0.3rad)
  - 使用原生似然场评分(500个采样点)
  - 取top 10最终候选
```

### 信息增益计算
```
输入: 安全方向列表 + 候选Pose列表

对每个安全方向:
  对每个候选Pose:
    预测运动后的位置 = 候选 + 方向*motion_distance
    用Submap1快速评分(粗似然场, 100点) → 预测得分
  计算各候选预测得分的方差 = 信息增益

选方差最大的方向 = 该运动最能区分候选
```

### 安全检测
```
输入: 方向角

在 LaserScan 中查 direction ± 30°锥内:
  取最小有效 range = safety_clearance

若 min_range ≥ 0.5m → 安全
若 min_range < 0.5m → 堵塞
```

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| stationary_frames | 20 | 静止累积Scan帧数 |
| submap_downsample | 0.05m | Submap体素降采样 |
| coarse_res_factor | 4 | 粗分辨率降采样因子 |
| coarse_particles | 1000 | 粗匹配粒子数 |
| coarse_top_n | 50 | 粗匹配保留候选数 |
| fine_particles_per_candidate | 10 | 每个粗候选精匹配粒子数 |
| top_n_final | 10 | 最终保留候选数 |
| motion_distance | 0.8m | 每次运动距离 |
| safety_clearance | 0.5m | 安全距离阈值 |
| safety_cone_deg | 30° | 安全检测锥角 |
| max_motion_attempts | 5 | 最大运动尝试次数 |
| convergence_ratio | 2.0 | 收敛比例(最佳/次佳) |
| linear_speed | 0.3 | 线速度 m/s |
| angular_speed | 0.5 | 角速度 rad/s |

## 服务接口

```bash
# 开始室内定位
ros2 service call /start_indoor_localization std_srvs/Trigger

# 查询状态
ros2 service call /localization_status std_srvs/Trigger
# → "Phase: FILTERING, Candidates: 3, Motion attempts: 2/5"

# 中止
ros2 service call /abort_localization std_srvs/Trigger
```

## 使用方式

```bash
# 启动
ros2 launch nav2_dog_slam indoor_global_localization.launch.py

# 触发定位
ros2 service call /start_indoor_localization std_srvs/Trigger
# → 自动执行全部流程:
#   1. 等待20帧Scan → 构建Submap1
#   2. 多分辨率匹配 → 10候选
#   3. 选安全方向 → 发布cmd_vel运动
#   4. 运动中累积Scan → 构建Submap2
#   5. 筛选 → 收敛检查
#   6. 未收敛 → 重复3-5
#   7. 收敛 → 发布/initialpose
```
