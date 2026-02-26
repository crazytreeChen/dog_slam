# CLAUDE.md — dog_slam 项目规范

## 沟通
- 所有回复使用**中文**，技术术语可中英对照

## 项目概述
基于 ROS2 Humble 的激光雷达 SLAM 与导航系统，专为四足机器人设计，使用 Livox MID360 激光雷达。
工作目录：`/home/crazytree/ros_ws/dog_slam/`

## 构建命令
```bash
cd ~/ros_ws/dog_slam/LIO-SAM_MID360_ROS2_PKG
bash build_ros2.sh
# 或单独构建某个包：
cd ros2 && colcon build --symlink-install --packages-select <package_name>
```

## 启动命令
```bash
cd ~/ros_ws/dog_slam/LIO-SAM_MID360_ROS2_PKG
export SLAM_ALGORITHM=super_lio   # 可选: fast_lio, point_lio, lio_sam, super_lio
ros2 launch nav2_dog_slam lio_nav2_unified.launch.py
```

## 关键文件
| 文件 | 说明 |
|------|------|
| `ros2/src/global_config/global_config/__init__.py` | 全局配置（按主机名自动切换） |
| `ros2/src/nav2_dog_slam/launch/lio_nav2_unified.launch.py` | 统一启动入口 |
| `ros2/src/nav2_dog_slam/config/nav2_params_sim.yaml` | Nav2 仿真参数 |
| `ros2/src/nav2_dog_slam/config/nav2_params.yaml` | Nav2 实机参数 |

## 代码规范
- Python 注释用中文
- 修改 global_config 时注意多主机兼容性，不要破坏 RK3588 实机配置
- 新增 SLAM 算法需在 `global_config/__init__.py` 的话题配置中注册
- 不要删除或修改 `RK3588` 主机的配置条目

## Git 规范
- commit 信息用**中文**
- 不自动 commit，需用户确认后再提交
- 不自动 push

## 注意事项
- 当前开发机 `CrazyTreeChen` 为仿真模式（`ONLINE_LIDAR=False`，`USE_SIM_TIME=True`）
- 实机部署在 `RK3588`，修改配置时注意区分
- `base_footprint` 由各 LIO 算法的里程计输出计算，不要改为固定 TF
