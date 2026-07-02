#!/bin/bash
# ============================================================
# GPS RTK 系统 topic 一键诊断脚本
# 用法: ./check_topics.sh
# ============================================================
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

check_topic() {
    local topic="$1"
    local desc="$2"
    local timeout="${3:-2}"
    if timeout "$timeout" ros2 topic echo "$topic" --once > /dev/null 2>&1; then
        echo -e "  ${GREEN}[OK]${NC} $desc ($topic)"
        return 0
    else
        echo -e "  ${RED}[MISS]${NC} $desc ($topic) — 无数据"
        return 1
    fi
}

check_topic_exists() {
    local topic="$1"
    local desc="$2"
    if ros2 topic list 2>/dev/null | grep -qF "$topic"; then
        echo -e "  ${GREEN}[OK]${NC} $desc ($topic)"
        return 0
    else
        echo -e "  ${YELLOW}[N/A]${NC} $desc ($topic) — 话题不存在"
        return 1
    fi
}

check_tf() {
    local from="$1" to="$2" desc="$3"
    if timeout 2 ros2 run tf2_ros tf2_echo "$from" "$to" 2>/dev/null | head -1 | grep -q 'Translation'; then
        echo -e "  ${GREEN}[OK]${NC} $desc ($from→$to)"
        return 0
    else
        echo -e "  ${YELLOW}[N/A]${NC} $desc ($from→$to) — TF 不可用"
        return 1
    fi
}

echo ""
echo -e "${CYAN}=========================================="
echo -e "  GPS RTK 系统 topic 诊断"
echo -e "==========================================${NC}"
echo ""

PASS=0
FAIL=0

echo -e "${CYAN}[GPS/RTK 数据链路]${NC}"
echo -e "  ${CYAN}[→]${NC} rtk_pose_monitor 直接从 /rtk_pvh 提取位置+航向（无需 gps_preprocessor）"
check_topic "/rtk_pvh" "RTK 原始数据（位置+航向来源）" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
check_topic "/test/rtk_pvh" "RTK 原始数据（模拟/测试）" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo -e "${CYAN}[纠偏链路]${NC}"
check_topic_exists "/initialpose" "/initialpose 纠偏话题" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
check_tf "map" "base_footprint" "AMCL 位姿 TF" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo -e "${CYAN}[状态与诊断]${NC}"
check_topic "/gps/status" "GPS 可用状态" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo -e "${CYAN}[EKF 融合（可选）]${NC}"
check_topic "/odometry/gps_fused" "EKF 融合里程计" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
check_topic "/odometry/gps" "navsat GPS 里程计" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo -e "${CYAN}[Web 可视化链路]${NC}"
check_topic "/trajectory/lio_latlon" "LIO 轨迹 Path" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
check_topic "/trajectory/fused_latlon" "融合轨迹 Path" && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo -e "${CYAN}[LIO 里程计]${NC}"
for odom in "/rkbot/lio/robo/odom" "/rkbot/lio/odom" "/lio/robo/odom" "/lio/odom" "/Odometry"; do
    check_topic "$odom" "LIO 里程计" 1 && PASS=$((PASS+1)) || true
done

echo ""
echo "------------------------------------------"
echo -e "  结果: ${GREEN}${PASS} OK${NC} / ${RED}${FAIL} MISS${NC}"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "  ${GREEN}全部通过，系统就绪！${NC}"
else
    echo -e "  ${YELLOW}某些话题不可用（正常：室内无 GPS、未启用的功能）${NC}"
fi
echo ""
