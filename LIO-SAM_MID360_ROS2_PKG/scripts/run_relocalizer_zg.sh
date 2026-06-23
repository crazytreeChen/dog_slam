#!/bin/bash
# 中狗 (ZG) 重定位启动脚本
# 用法: ./run_relocalizer_zg.sh [pcd_map_path]

WORKSPACE_DIR="/home/ztl/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2"
PCD_MAP=${1:-"/home/ztl/cql/4f.pcd"}

# 加载 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/humble/setup.bash
fi
source $WORKSPACE_DIR/install/setup.bash

echo "===== 中狗 (ZG) 3D 重定位 ====="
echo "PCD 地图: $PCD_MAP"
echo "点云话题: /rkbot/lio/body/cloud"
echo "odom话题: /rkbot/lio/odom"
echo "坐标系:   map=rkbot/map  odom=rkbot/world  base=rkbot/imu"
echo ""

ros2 launch lidar_3d_relocalizer lidar_3d_relocalizer.launch.py \
    cloud_topic:=/rkbot/lio/body/cloud \
    odom_topic:=/rkbot/lio/odom \
    pcd_map_path:=$PCD_MAP \
    map_frame:=rkbot/map \
    odom_frame:=rkbot/world \
    base_frame:=rkbot/imu
