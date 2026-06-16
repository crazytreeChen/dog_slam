# auto_initial_pose_calibrator 改进日志

## 2026-06-16: v2.0 重大改进

### 评分引擎重构

| 改动 | 原因 | 效果 |
|------|------|------|
| `_build_likelihood`: 未知区域设 max_dist | 原: 灰色区域被当作"零距离", 优化器可用来藏不对齐的点 | 灰色区域被正确惩罚 |
| 新增 `_score_points`: 向量化点云评分 O(N) | 原 `_score_scan`: 逐束循环, 只采样120束, 评分方差大 | 全量点评分, 100× 更快, 统计稳定 |
| 新增 `_compute_wall_coverage`: 有效区域内墙壁覆盖率 | 原: 只有似然场分数, 无量化指标 | 输出 wall% 和 coverage%, 判断匹配质量 |
| `_do_hierarchical_matching`: 粒子搜索 → 网格搜索 | 原: 5000粒子随机撒在自由空间, 可能遗漏最优位置 | 1.5m网格 × 10°角度 遍历全图, 不漏解 |
| `_score_scan`: 改为调用 `_score_points` | 原: beam model log-likelihood, 逐束 Python 循环 | 向量化, 兼容旧接口 |

### 被动持续定位模式 (新增)

```
PASSIVE_COLLECTING → 定时器(60s) → PASSIVE_MATCHING → 循环
```

- **不控制机器人**: 仅被动采集雷达数据
- **循环缓冲区**: 最多保留300帧, 丢旧取新
- **ICP 帧间合并**: 每次取最近30帧, 用ICP拼接成子图
- **局部搜索**: 有历史估计时 ±2m 局部搜索; 无则全局搜索
- **结果输出**: 日志 + debug/auto_initial_pose 话题 (不发布 /initialpose)

### 偏差监控 (新增)

```
启动 → 发布校准位姿 → 每2s对比 AMCL vs 校准
                     → 开机3分钟内 > 0.5m → 告警
```

### 自适应协方差

```
pos_sigma = 0.8 / max(wall_cov, 0.1) × max(1 - coverage, 0.3)
yaw_sigma = 20.0 / max(wall_cov, 0.1) × (1 - coverage + 0.2)
```

墙壁覆盖率越高 → 协方差越小, AMCL 收敛越快。

### 服务拆分

| 旧 | 新 |
|----|----|
| `start_auto_calibration` (单一入口) | `start_active_calibration` + `start_passive_calibration` (独立) |
| 无 | `stop_passive_calibration` |
| `auto_calibration_status` | 增强: 显示被动模式状态 + 最佳估计位姿 |

### 配置变更 (auto_initial_pose_calibrator.yaml)

```yaml
# 新增
passive_mode_enabled: true
passive_interval: 60.0
passive_frame_count: 30
passive_buffer_max: 300
grid_search_step: 1.5
fine_search_radius: 2.5
deviation_monitor_enabled: true
deviation_alert_threshold: 0.5

# 变更
auto_publish_initial_pose: true → false  # 验证阶段不自动发布
rough_match_particles: 5000  # 保留但网格搜索不再使用
```

### 新增脚本

```
scripts/start_passive_calibration.sh    # 被动定位便捷脚本
scripts/start_active_calibration.sh     # 主动校准便捷脚本
scripts/opencode_likelihood_search.py   # 离线似然场全局搜索
scripts/opencode_multistep_localizer.py # 离线多步递推定位
scripts/opencode_area_segment_matcher.py
scripts/opencode_contour_hu_matcher.py
scripts/opencode_corner_graph_matcher.py
```

### 离线实验关键发现

1. 自由空间采样限制 → 遗漏85%的图位置; 正确位置质心在灰色区域
2. 似然场未知区域惩罚 → 阻止优化器把扫描藏进灰色
3. 多步递推 → 单帧76% → 多步90% 墙壁覆盖率
4. FRF gap=0.3 最优; 无 FRF 墙壁覆盖率反而下降
5. 系统降采样 `points[::10]` 引入偏差; 改用随机采样
