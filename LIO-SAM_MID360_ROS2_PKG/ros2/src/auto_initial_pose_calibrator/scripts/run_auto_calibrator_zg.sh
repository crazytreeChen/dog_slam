#!/bin/bash
# 中狗 (ZG) 自动初始位姿校准器启动脚本
# 用法: ./run_auto_calibrator_zg.sh

WORKSPACE_DIR="/home/ztl/cql"

# 加载 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/humble/setup.bash
fi
source $WORKSPACE_DIR/install/setup.bash

echo "===== 中狗 (ZG) 自动初始位姿校准器 ====="
echo "Workspace: $WORKSPACE_DIR"
echo "模式: 被动持续定位 (passive_mode_enabled=true)"
echo "命名空间: rkbot"
echo ""

ros2 launch auto_initial_pose_calibrator auto_initial_pose_calibration.launch.py ns:=rkbot
