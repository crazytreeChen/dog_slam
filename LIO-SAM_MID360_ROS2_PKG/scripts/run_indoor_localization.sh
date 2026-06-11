#!/bin/bash
echo "===== ROS2 室内 AMCL 人工初始定位启动脚本 ====="
WORKSPACE_DIR="/home/ztl/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2"

# 默认地图路径（可通过参数覆盖）
MAP_FILE="${1:-/home/ztl/slam_data/grid_map/map.yaml}"

# 检查是否已加载ROS2环境
if [ -z "$ROS_DISTRO" ]; then
    echo "加载ROS2环境..."
    source /opt/ros/humble/setup.bash
    if [ $? -ne 0 ]; then
        echo "错误: 无法加载ROS2环境"
        exit 1
    fi
fi

# 加载工作空间环境
echo "加载工作空间环境..."
source $WORKSPACE_DIR/install/setup.bash

echo "========================================="
echo "室内 AMCL 定位启动"
echo "地图文件: $MAP_FILE"
echo "========================================="
echo ""
echo "操作流程:"
echo "  1. 等待 map_server 和 AMCL 完成启动"
echo "  2. 在 RViz 中点击 '2D Pose Estimate' 设置初始位姿"
echo "  3. 或通过命令行发布初始位姿:"
echo "     ros2 run nav2_dog_slam set_initial_pose --x 2.0 --y 3.5 --yaw 1.57"
echo "  4. 观察 /particle_cloud 粒子云收敛情况"
echo "  5. 轻微移动机器人让激光与地图进一步匹配"
echo "========================================="

ros2 launch nav2_dog_slam indoor_amcl_localization.launch.py map:=$MAP_FILE
