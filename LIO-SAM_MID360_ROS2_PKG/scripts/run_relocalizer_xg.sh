#!/bin/bash
# XG relocalizer launcher
# Usage: ./run_relocalizer_xg.sh [pcd_map_path]

WORKSPACE_DIR="/home/ztl/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2"
PCD_MAP=${1:-"/home/ztl/cql/4f.pcd"}

if [ -z "$ROS_DISTRO" ] && [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi

if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
else
    echo "ERROR: missing $WORKSPACE_DIR/install/setup.bash"
    exit 1
fi

echo "===== XG 3D relocalizer ====="
echo "PCD map: $PCD_MAP"
echo "cloud topic: /rkbot/lio/cloud_world"
echo "odom topic: /rkbot/lio/odom"
echo "frames: map=rkbot/map odom=rkbot/world base=rkbot/livox/imu"
echo ""

ros2 launch lidar_3d_relocalizer lidar_3d_relocalizer.launch.py \
    cloud_topic:=/rkbot/lio/cloud_world \
    odom_topic:=/rkbot/lio/odom \
    pcd_map_path:=$PCD_MAP \
    map_frame:=rkbot/map \
    odom_frame:=rkbot/world \
    base_frame:=rkbot/livox/imu
