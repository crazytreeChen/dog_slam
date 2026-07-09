#!/bin/bash
# ============================================================
# GPS RTK 纠偏 全链路模拟测试 + Web 可视化
# 用法: ./simulate_test.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="/tmp/gps_fusion_sim_$$"
ORIGIN_FILE="$TEST_DIR/map_gps_origin.yaml"

cleanup() {
    echo ""
    echo "清理模拟进程..."
    kill $SIM_PID $SIM2_PID $MOCK_PID $LIO_PID $LAUNCH_PID 2>/dev/null || true
    pkill -f "rtk_simulator" 2>/dev/null || true
pkill -f "mock_tf_broadcaster" 2>/dev/null || true
    pkill -f "lio_simulator" 2>/dev/null || true
    echo "模拟结束。恢复服务: sudo systemctl restart gps-rtk-nav"
    exit 0
}
trap cleanup INT TERM

source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/cql/install/setup.bash 2>/dev/null || true
# fallback: 主 dog_slam workspace 提供 robots_dog_msgs
source ~/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/install/setup.bash 2>/dev/null || true
source ~/dog_slam/extend/robots_dog_msgs/install/setup.bash 2>/dev/null || true

mkdir -p "$TEST_DIR"

pkill -f "gps_preprocessor" 2>/dev/null || true
pkill -f "rtk_simulator" 2>/dev/null || true
pkill -f "rtk_pose_monitor" 2>/dev/null || true
pkill -f "mock_tf_broadcaster" 2>/dev/null || true
sleep 2

echo ""
echo "=========================================="
echo "  GPS RTK 纠偏 全链路模拟测试"
echo "=========================================="
echo ""

# ======== Phase 1: 生成模拟地图原点 ========
echo "[Phase 1] 写入模拟地图原点 (map_gps_origin.yaml) ..."
rm -f "$TEST_DIR/rtk_sim.log"

# 建图生成器 (map_origin_recorder) 已移除，直接写入合法原点供 threshold 纠偏模式读取
cat > "$ORIGIN_FILE" <<'EOF'
map_origin:
  latitude: 24.61000000
  longitude: 118.03000000
  altitude: 42.52
  heading_deg: 0.0
  utm_easting: 604684.19
  utm_northing: 2722418.28
  utm_zone: 50
note: "室内模拟预置原点 (生成器已移除，手动维护)"
EOF

if [ -f "$ORIGIN_FILE" ]; then
    echo "  [OK] 原点文件已写入: $ORIGIN_FILE"
else
    echo "  [FAIL] 原点文件写入失败"
    exit 1
fi

# ======== Phase 2: 启动全链路模拟 + Web ========
echo ""
echo "[Phase 2] 启动全链路模拟 + Web 可视化 ..."
echo ""
echo "  Web 页面: http://$(hostname -I | awk '{print $1}'):8084/map_viewer.html"
echo "  模拟场景: LIO+RTK 圆形巡逻 (30m), Mock AMCL 向东南漂移 0.3m/s"
echo "  颜色: 🔵 LIO轨迹  🟢 RTK轨迹  🟠 AMCL漂移(绿偏出去的部分)"
echo ""

# rtk_simulator: 模拟 RTK_FIX 圆形轨迹（模拟户外巡逻）
ros2 run gps_fusion rtk_simulator.py --ros-args \
    -p rate:=5.0 -p pos_type:=50 -p radius:=30.0 -p speed:=1.5 \
    -p topic:=/test/rtk_pvh > /dev/null 2>&1 &
SIM2_PID=$!

# mock_tf: 模拟 AMCL 向东南漂移 0.3m/s
ros2 run gps_fusion mock_tf_broadcaster.py --ros-args \
    -p init_x:=0.0 -p init_y:=0.0 -p init_yaw:=0.0 \
    -p drift_per_second:=0.3 -p drift_direction_deg:=135.0 \
    -p rate:=10.0 > /dev/null 2>&1 &
MOCK_PID=$!

ros2 run gps_fusion lio_simulator.py --ros-args \
    -p topic:=/Odometry -p rate:=10.0 \
    -p radius:=30.0 -p speed:=1.5 > /dev/null 2>&1 &
LIO_PID=$!

sleep 2

# 主启动: 纠偏 + Web
ros2 launch gps_fusion gps_fusion.launch.py \
    lio_odom_topic:=/Odometry \
    gps_source:=/fix \
    enable_web:=true \
    enable_correction:=true \
    enable_ekf:=false \
    rtk_min_accuracy:=10.0 \
    map_origin_file:="$ORIGIN_FILE" &
LAUNCH_PID=$!

# ======== Phase 3: 等待看效果 ========
echo ""
echo "=========================================="
echo "  模拟运行中..."
echo "  浏览器打开 http://$(hostname -I | awk '{print $1}'):8084/map_viewer.html"
echo "  等待 15 秒观察 drift > 2m 触发纠偏"
echo "  Ctrl+C 结束"
echo "=========================================="
echo ""

wait $LAUNCH_PID
