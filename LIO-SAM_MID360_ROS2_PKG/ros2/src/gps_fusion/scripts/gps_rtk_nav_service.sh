#!/bin/bash
# ============================================================
# GPS RTK 导航纠偏 — systemd 服务启动脚本
# 部署路径: /home/ztl/cql/gps_fusion/scripts/
# 系统服务: gps-rtk-nav.service
# ============================================================
# 首次收到外部 /initialpose 时自动标定原点，无需预标定
# 位置+航向均直接从 /rtk_pvh 提取，不依赖 /fix
# ============================================================

set -e

export ROS_DOMAIN_ID=27

WORKSPACE_DIR="/home/ztl/cql"
GPS_FUSION_DIR="$WORKSPACE_DIR/gps_fusion"

if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
elif [ -f /opt/ros/humble/install/setup.bash ]; then
    source /opt/ros/humble/install/setup.bash
fi

if [ -f "$WORKSPACE_DIR/install/setup.bash" ]; then
    source "$WORKSPACE_DIR/install/setup.bash"
fi

pkill -f "rtk_pose_monitor" 2>/dev/null || true
sleep 1

echo "[gps-rtk] 启动纠偏服务 (WAIT_ORIGIN: 等待首次 /initialpose 标定) ..."
cd "$GPS_FUSION_DIR"
exec ros2 launch gps_fusion rtk_nav_bridge.launch.py \
    ns:=rkbot \
    rtk_topic:=/rtk_pvh \
    "$@"
