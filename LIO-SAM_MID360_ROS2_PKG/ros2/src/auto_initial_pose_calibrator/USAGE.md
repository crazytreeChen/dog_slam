# auto_initial_pose_calibrator 使用说明

## 快速开始

### 1. 构建

该包已包含在 `LIO-SAM_MID360_ROS2_PKG/ros2/` 的 colcon workspace 中，随顶层 `build_ros2.sh` 一键构建：

```bash
cd LIO-SAM_MID360_ROS2_PKG
./build_ros2.sh
```

或单独构建：

```bash
cd LIO-SAM_MID360_ROS2_PKG/ros2
colcon build --symlink-install --packages-select auto_initial_pose_calibrator
source install/setup.bash
```

### 2. 启动

```bash
# 默认启动（namespace=rkbot）
ros2 launch auto_initial_pose_calibrator auto_initial_pose_calibration.launch.py

# 仿真模式
ros2 launch auto_initial_pose_calibrator auto_initial_pose_calibration.launch.py use_sim_time:=true

# 自定义 namespace
ros2 launch auto_initial_pose_calibrator auto_initial_pose_calibration.launch.py ns:=mybot
```

### 3. 触发校准

节点启动后处于 IDLE 状态，通过 ROS2 服务触发校准（服务名在 `rkbot` namespace 下）：

```bash
# 主动模式（室内，机器人会自主运动）
ros2 service call /rkbot/start_active_calibration std_srvs/srv/Trigger

# 被动模式（室内，不控制机器人）
ros2 service call /rkbot/start_passive_calibration std_srvs/srv/Trigger

# 自动模式（根据 GPS/RTK 信号自动选择室内/室外）
ros2 service call /rkbot/start_auto_calibration std_srvs/srv/Trigger
```

### 4. 查询状态

```bash
ros2 service call /rkbot/auto_calibration_status std_srvs/srv/Trigger
```

### 5. 查看校准结果

监听调试话题（在 `rkbot` namespace 下）：

```bash
# 候选位姿
ros2 topic echo /rkbot/debug/candidates

# 校准结果（含协方差）
ros2 topic echo /rkbot/debug/auto_initial_pose

# 合成子图 Scan
ros2 topic echo /rkbot/debug/submap_scan

# 位姿对比（调试对比模式）
ros2 topic echo /rkbot/debug/pose_comparison
```

---

## 工作模式详解

### 模式一：室内主动校准

适用场景：机器人静止在室内，需要自动确定位姿。机器人会自主旋转和移动。

#### 启动

```bash
ros2 service call /rkbot/start_active_calibration std_srvs/srv/Trigger
```

#### 执行流程

| 阶段 | 行为 | 耗时 |
|------|------|------|
| BOOT_DELAY | 静止等待数据稳定 | 2 秒 |
| ROTATING_360 | 原地旋转 360° 采集全方向雷达数据 | ~12 秒（角速度 0.5 rad/s） |
| ACTIVE_MULTISTEP | 多步递推匹配（快速模式下直接发布） | 数秒 |
| SUBMAP1 | 静止收集 30 帧合成第 1 个子图 | ~3 秒 |
| ROUGH_MATCHING | 全局匹配出 Top-N 候选位姿 | 数秒 |
| SELECTING | 选择信息增益最大的探索方向 | <1 秒 |
| MOVING | 向目标方向移动，实时避障 | 数秒 |
| SUBMAP2 | 收集第 2 个子图 | ~3 秒 |
| FILTERING | 利用运动约束筛选唯一位姿 | 数秒 |
| **DONE** | 发布 `/initialpose` 给 AMCL | — |

如果一次运动消歧后仍有多个候选，系统会重试（默认最多 5 轮）。

#### 快速模式（跳过运动探索）

配置 `publish_after_rotation: true` 后，旋转 360° + 多步匹配完成后直接发布初始位姿，跳过运动消歧环节。适用于环境特征丰富、匹配置信度高的场景。

#### 注意事项

- **机器人周围需要足够空间**：旋转 360° 和直线移动需要空地，半径建议 ≥ 1.5m
- **地图必须已加载**：确保 Nav2 map_server 已提供 `/map` 话题
- **LIO 里程计需正常运行**：节点依赖 odom 话题进行运动约束和数据融合
- **避障保护**：前方 0.5m 内有障碍物时自动停机，该轮探索结束

---

### 模式二：室内被动持续定位

适用场景：机器人被遥控或自主导航过程中，需要在后台持续维护定位。此模式不控制机器人运动。

#### 启动

```bash
ros2 service call /rkbot/start_passive_calibration std_srvs/srv/Trigger
```

#### 执行流程

```
持续循环:
  ├─ 被动采集雷达帧（循环缓冲区，最多60帧）
  ├─ 每 passive_interval 秒触发匹配
  │   ├─ 有历史位姿 + 里程计位移 ≤ passive_odom_max_move
  │   │   └─ 里程计融合轻量验证 → 通过则跳过重匹配
  │   ├─ 取最近 N 帧，用 ICP 拼接子图
  │   ├─ 有历史位姿 → 局部搜索 ±2m
  │   ├─ 无历史位姿 → 全局搜索
  │   └─ 与上次位姿偏差 > passive_deviation_threshold
  │       └─ 二次验证（更多帧 + 更精细搜索）
  └─ 结果输出到 /rkbot/debug/auto_initial_pose
```

#### 自动发布 /initialpose

默认不自动发布（仅输出到 debug 话题）。要启用自动发布：

```bash
ros2 service call /rkbot/toggle_auto_publish std_srvs/srv/Trigger
```

或在配置中将 `auto_publish_initial_pose` 设为 `true`。

#### 停止

```bash
ros2 service call /rkbot/stop_passive_calibration std_srvs/srv/Trigger
```

#### 关键配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `passive_interval` | 30.0 | 匹配间隔（秒），匹配计算较耗时，不建议低于 10s |
| `passive_frame_count` | 30 | 每次匹配使用的帧数，越多子图越密但匹配越慢 |
| `passive_buffer_max` | 60 | 缓冲区最多保留帧数 |
| `passive_odom_fusion_enabled` | true | 启用里程计融合验证，可显著减少重匹配次数 |
| `passive_odom_max_move` | 5.0 | 里程计累计位移上限（m），超则强制重匹配 |
| `passive_amcl_cross_check` | true | 启用 AMCL 交叉验证 |
| `passive_amcl_max_deviation` | 1.0 | AMCL 偏差触发重匹配的阈值（m） |
| `passive_odom_deviation_threshold` | 0.3 | odom→map 推算偏差触发重匹配的阈值（m） |
| `passive_publish_cooldown` | 60.0 | 发布 /initialpose 后的冷却时间（秒） |
| `motion_stop_cooldown` | 10.0 | 停止事件后的冷却时间（秒） |

---

### 模式三：室外 RTK 校准

适用场景：室外开阔环境，有 RTK GPS + 双天线航向信号。

#### 前置条件

1. **RTK/GPS 信号良好**：位置类型和航向类型需在 `valid_pos_types` / `valid_heading_types` 列表中（默认 `[34, 50]`）
2. **标定文件就绪**：`calibration_file` 指向的 YAML 文件包含 LLA→Map 变换矩阵

#### 启动

```bash
# 自动检测 RTK 信号，自动切换室外模式
ros2 service call /rkbot/start_auto_calibration std_srvs/srv/Trigger
```

系统根据 `mode_timer`（1 Hz）持续检测 RTK/GPS 信号，自动在室内/室外模式间切换。

#### 输出

- 按 `publish_rate`（默认 2 Hz）频率发布 `/rkbot/initialpose`
- 同时输出到 `/rkbot/debug/auto_initial_pose`

---

## 开机自动启动

在配置文件中设置 `auto_start: true`，节点启动后会自动等待以下三个条件：

1. 地图话题有数据（`/map`）
2. 雷达话题有数据（`scan`）
3. 里程计话题有数据（`odom`）

三个条件满足后，等待 `auto_start_delay`（默认 3 秒），自动触发校准。

```yaml
auto_start: true
auto_start_delay: 3.0
```

若同时设置 `quick_mode: true`，开机后跳过旋转环节，直接进入快速匹配：

```yaml
quick_mode: true
publish_after_rotation: true
```

---

## 调试与可视化

### 调试对比模式

启用后，节点会同时运行多种匹配算法并输出对比结果到 `debug/pose_comparison`：

```yaml
debug_comparison_mode: true
```

### 设置手动真值（Ground Truth）

用于算法评估，对比匹配结果与人工标注的真值：

```bash
ros2 service call /rkbot/set_manual_ground_truth std_srvs/srv/SetBool "{data: true}"
```

---

## 完整配置参数参考

以下为 `auto_initial_pose_calibrator.yaml` 中全部参数的说明。

### 话题

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rtk_topic` | `/rtk_pvh` | RTK 定位话题 |
| `rtk_topic_type` | `robots_dog_msgs.msg.UniRtkPvh` | RTK 消息类型 |
| `gps_topic` | `/fix` | GPS 话题 |
| `cmd_vel_topic` | `/cmd_vel` | 运动控制话题 |
| `map_file` | `""` | 地图文件路径（为空则从话题订阅） |
| `map_frame` | `map` | 地图坐标系 |

### 模式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `outdoor_mode` | `false` | 初始室外模式 |
| `indoor_mode` | `true` | 初始室内模式 |
| `auto_start` | `false` | 开机自动启动校准 |
| `auto_start_delay` | 3.0 | 自动启动等待延迟（秒） |
| `auto_publish_initial_pose` | `true` | 自动发布 /initialpose |
| `quick_mode` | `false` | 快速模式（跳过旋转） |
| `publish_after_rotation` | `true` | 旋转采集后直接发布（跳过运动探索） |

### 旋转采集

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rotation_enabled` | `true` | 启用 360° 旋转采集 |
| `rotation_total_deg` | 360.0 | 旋转总角度 |
| `rotation_angular_vel` | 0.5 | 旋转角速度 (rad/s) |

### 匹配算法选择

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_dual_template_matching` | `true` | 双模板光线投射匹配（推荐，精度最高） |
| `dt_multistep_enabled` | `true` | 多步递推首帧使用双模板 |
| `dt_passive_enabled` | `true` | 被动匹配首帧使用双模板 |
| `use_distance_field_matching` | `false` | 距离场匹配（备选） |
| `df_multistep_enabled` | `true` | 多步递推首帧使用距离场 |
| `df_passive_enabled` | `true` | 被动匹配首帧使用距离场 |

### 双模板匹配 — 室内参数 (`dt_indoor.*`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `angle_step_deg` | 3.0 | 粗搜角度步长 |
| `fine_angle_step_deg` | 0.5 | 精修角度步长 |
| `penalty_weight` | 3.0 | 射线穿透惩罚权重 |
| `scan_max_points` | 500 | 匹配用最大扫描点数 |
| `min_wall_coverage_ratio` | 0.30 | 最低墙壁覆盖率 |
| `free_space_penalty_weight` | 0.2 | 自由空间惩罚权重 |
| `scale_ref_pixels` | 300000 | 自适应缩放触发像素数 |
| `scale_max` | 1.5 | 自适应缩放倍数上限 |

### 双模板匹配 — 室外参数 (`dt_outdoor.*`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `angle_step_deg` | 3.0 | 粗搜角度步长 |
| `fine_angle_step_deg` | 0.5 | 精修角度步长 |
| `penalty_weight` | 3.0 | 射线穿透惩罚权重 |
| `scan_max_points` | 500 | 匹配用最大扫描点数 |
| `min_wall_coverage_ratio` | 0.15 | 最低墙壁覆盖率（室外墙少） |
| `free_space_penalty_weight` | 0.2 | 自由空间惩罚权重 |
| `scale_ref_pixels` | 500000 | 自适应缩放触发像素数 |
| `scale_max` | 2.5 | 自适应缩放倍数上限 |

### 似然场网格搜索

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grid_search_step` | 1.5 | 全局搜索步长 (m) |
| `grid_search_angle_step` | 10.0 | 全局搜索角度步长 (deg) |
| `fine_search_radius` | 2.5 | 局部精搜半径 (m) |
| `fine_search_pos_step` | 0.3 | 局部搜索位置步长 (m) |
| `fine_search_angle_step` | 3.0 | 局部搜索角度步长 (deg) |
| `likelihood_max_dist` | 2.0 | 似然场最大距离 (m) |
| `sigma_hit` | 0.2 | 高斯命中标准差 |
| `z_hit` | 0.8 | 命中权重 |
| `z_rand` | 0.1 | 随机噪声权重 |

### 置信度门控

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `confidence_threshold_update` | 0.50 | 更新内部校准位姿的最低置信度 |
| `confidence_threshold_publish` | 0.40 | 发布 /initialpose 的最低置信度 |
| `confidence_threshold_amcl` | 0.35 | 辅助 AMCL 的最低置信度 |
| `min_match_wall_coverage` | 0.50 | 最低墙壁覆盖率 |
| `min_match_coverage` | 0.15 | 最低有效区域覆盖率 |

### 运动控制与避障

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `motion_distance` | 1.0 | 探索移动距离 (m) |
| `max_linear_vel` | 0.3 | 最大线速度 (m/s) |
| `max_angular_vel` | 0.5 | 最大角速度 (rad/s) |
| `kp_linear` | 0.8 | 线速度 P 增益 |
| `kp_angular` | 1.0 | 角速度 P 增益 |
| `min_safe_distance` | 0.5 | 避障紧急停机距离 (m) |
| `collision_sector_angle_deg` | 45.0 | 前进避障监测扇区宽度 (deg) |
| `max_active_retry` | 5 | 最大探索移动轮数 |

### 扫描过滤

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `scan_outlier_filter` | `true` | 启用半径离群点过滤 |
| `scan_outlier_radius` | 0.3 | 邻域搜索半径 (m) |
| `scan_outlier_min_neighbors` | 3 | 最少邻居点数 |
| `temporal_merge_enabled` | `true` | 启用多帧时序一致性过滤 |
| `temporal_merge_min_frames` | 5 | 最少帧数判定为静态墙壁 |
| `temporal_merge_radius` | 0.15 | 邻域一致性搜索半径 (m) |

### 子图构建

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `submap_scan_count` | 30 | 合成单次子图的雷达帧数 |
| `submap_angle_resolution` | 0.5 | 合成 Scan 的角分辨率 (deg) |

### ICP

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `icp_max_iterations` | 15 | 最大迭代次数 |
| `icp_tolerance_trans` | 0.01 | 平移收敛阈值 (m) |
| `icp_tolerance_rot_deg` | 0.1 | 旋转收敛阈值 (deg) |
| `icp_downsample_step` | 5 | 降采样步长 |

### 偏差监控

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `deviation_monitor_enabled` | `false` | 启用偏差监控 |
| `deviation_check_interval` | 2.0 | 检查间隔 (秒) |
| `deviation_alert_threshold` | 0.5 | 偏差告警阈值 (m) |
| `deviation_alert_window` | 180.0 | 开机告警窗口 (秒) |

### 运动状态跟踪

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `motion_tracking_enabled` | `true` | 启用运动状态跟踪 |
| `motion_speed_threshold` | 0.05 | 线速度阈值 (m/s) |
| `motion_angular_threshold` | 0.05 | 角速度阈值 (rad/s) |
| `motion_settle_time` | 1.0 | 停止稳定等待 (秒) |
| `motion_stop_cooldown` | 10.0 | 停止事件冷却 (秒) |
| `passive_odom_deviation_threshold` | 0.3 | odom→map 偏差触发重匹配 (m) |
| `passive_amcl_deviation_trigger` | 1.0 | AMCL 交叉验证触发阈值 (m) |

### 被动模式 — 里程计融合

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `passive_odom_fusion_enabled` | `true` | 启用里程计融合验证 |
| `passive_odom_max_move` | 5.0 | 里程计累计位移上限 (m) |
| `passive_likelihood_threshold` | 0.25 | 似然场评分阈值 |
| `passive_fusion_wall_threshold` | 0.20 | 融合验证的墙壁覆盖率阈值 |
| `passive_bootstrap_threshold` | 2 | 连续失败 N 次后放宽门控 |
| `passive_publish_cooldown` | 60.0 | 发布 /initialpose 后冷却时间 (秒) |

### RTK

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `publish_rate` | 0.5 | RTK 发布频率 (Hz) |
| `min_soln_svs` | 4 | 最少卫星数 |
| `valid_pos_types` | `[34, 50]` | 有效的 RTK 位置类型 |
| `valid_heading_types` | `[34, 50]` | 有效的 RTK 航向类型 |
| `calibration_file` | `""` | LLA→Map 标定文件路径 |

---

## 常见问题

### Q: 启动后一直处于 IDLE 状态？

确认已调用服务触发校准：

```bash
ros2 service call /rkbot/start_active_calibration std_srvs/srv/Trigger
```

或配置 `auto_start: true` 实现开机自动启动。

### Q: 一直停在 BOOT_DELAY，不进入旋转？

检查三个必要条件：
- `/rkbot/map` 话题是否有数据（`ros2 topic hz /rkbot/map`）
- 雷达话题是否有数据（`ros2 topic hz /rkbot/scan`）
- 里程计话题是否有数据（`ros2 topic hz /rkbot/lio/odom`）

### Q: 匹配结果位置偏差很大？

1. 确认地图与实际环境一致（未发生环境变化）
2. 检查雷达数据是否正常（`ros2 topic echo /rkbot/scan`）
3. 尝试降低 `min_match_wall_coverage` 阈值
4. 启用 `debug_comparison_mode: true` 对比各算法结果

### Q: 室外模式无法获取 RTK 信号？

1. 确认 `calibration_file` 路径正确、文件存在
2. 检查 RTK 话题名匹配（`rtk_topic` 默认 `/rtk_pvh`）
3. 确认卫星数 ≥ `min_soln_svs`（默认 4）且位置类型在 `valid_pos_types` 中
4. 使用 `ros2 topic echo /rtk_pvh` 确认数据正常

### Q: 被动模式一直在匹配，CPU 占用高？

- 增大 `passive_interval`（如 60s）
- 减少 `passive_frame_count`（如 15）
- 启用 `passive_odom_fusion_enabled: true` 让里程计分担验证
- 降低 `passive_odom_deviation_threshold` 减少触发频率

### Q: 话题不通或找不到？

节点默认使用 `rkbot` namespace，所有话题在 `/rkbot/` 前缀下。确认：
- 地图话题：`ros2 topic hz /rkbot/map`
- 雷达话题：`ros2 topic hz /rkbot/scan`
- 里程计话题：`ros2 topic hz /rkbot/lio/odom`

如果实际话题名不同，修改 `auto_initial_pose_calibrator.yaml` 中对应的 topic 参数。

### Q: 如何评估匹配质量？

- 查看墙壁覆盖率：日志中的 `wall=XX% coverage=XX%`
- 查看置信度：`confidence=XX`
- 可视化候选位姿：`ros2 topic echo /rkbot/debug/candidates`
- 对比 AMCL 输出：被动模式有 `passive_amcl_cross_check` 功能

---

## 错误排查

### 启用详细日志

```yaml
log_level: "DEBUG"
log_dir: "/home/ztl/slam_data/logs"
```

日志会同时输出到终端和 `log_dir` 中的时间戳文件。

### 常见错误

| 错误日志 | 原因 | 解决方案 |
|----------|------|----------|
| `地图似然场尚未加载` | 未收到 /map 话题 | 确认 map_server 运行正常 |
| `没有候选位姿通过过滤` | 扫描数据与地图匹配度太低 | 检查地图/雷达数据，降低门控阈值 |
| `墙壁覆盖率=0%` | 似然场参数或地图分辨率有问题 | 检查 `likelihood_max_dist` 和地图分辨率 |
| `运动超过预期距离` | 避障导致移动距离不足 | 增大 `min_safe_distance` 或清理前方空间 |
| `ICP 不收敛` | 帧间位姿变化太大或环境缺乏特征 | 降低 `icp_tolerance_trans`，增加 `icp_max_iterations` |
