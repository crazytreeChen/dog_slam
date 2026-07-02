#!/bin/bash
# ============================================================
# GPS RTK 纠偏 — 部署检查 & 诊断脚本
# 用法: ./deploy_check.sh [check|calibrate|status|restart]
# ============================================================
set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

source /opt/ros/humble/setup.bash 2>/dev/null || true
source ~/cql/install/setup.bash 2>/dev/null || true
source ~/dog_slam/LIO-SAM_MID360_ROS2_PKG/ros2/install/setup.bash 2>/dev/null || true

WORKSPACE_DIR="/home/ztl/cql"
GPS_FUSION_DIR="$WORKSPACE_DIR/gps_fusion"

pass() { echo -e "  ${GREEN}[✓]${NC} $1"; }
fail() { echo -e "  ${RED}[✗]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[!]${NC} $1"; }
info() { echo -e "  ${CYAN}[→]${NC} $1"; }
header() { echo -e "\n${BOLD}${CYAN}═══ $1 ═══${NC}"; }

# ======== 全链路检查 ========
do_check() {
    echo -e "\n${BOLD}GPS RTK 纠偏 全链路诊断${NC}"
    echo "=========================================="

    # ---- 1. 代码部署 ----
    header "1. 代码部署"
    if [ -f "$GPS_FUSION_DIR/launch/rtk_nav_bridge.launch.py" ]; then
        pass "gps_fusion 包已部署"
    else
        fail "gps_fusion 包未找到: $GPS_FUSION_DIR"
        return
    fi
    ros2 pkg list 2>/dev/null | grep -q gps_fusion && pass "gps_fusion 已构建" || fail "gps_fusion 未构建，执行 colcon build"

    # ---- 2. 服务状态 ----
    header "2. 系统服务"
    if systemctl is-active --quiet gps-rtk-nav 2>/dev/null; then
        pass "gps-rtk-nav 服务运行中"
    else
        fail "gps-rtk-nav 服务未运行 → sudo systemctl start gps-rtk-nav"
    fi

    # ---- 3. 依赖检查 ----
    header "3. 依赖"
    ros2 pkg list 2>/dev/null | grep -q robots_dog_msgs && pass "robots_dog_msgs" || fail "robots_dog_msgs 缺失"
    python3 -c "import tornado" 2>/dev/null && pass "tornado" || fail "tornado 未安装: pip3 install tornado"
    python3 -c "import pyproj" 2>/dev/null && pass "pyproj" || fail "pyproj 未安装: pip3 install pyproj"

    # ---- 4. 端口 ----
    header "4. 端口"
    ss -tlnp 2>/dev/null | grep -q ':8084' && pass "HTTP 8084 (Web页面)" || fail "8084 未监听"
    ss -tlnp 2>/dev/null | grep -q ':8765' && pass "WebSocket 8765 (轨迹推送)" || fail "8765 未监听"

    # ---- 5. RTK 数据链路 ----
    header "5. RTK 数据链路"
    RTK_DATA=""
    if timeout 3 ros2 topic echo /rtk_pvh --once 2>/dev/null; then
        RTK_DATA=$(timeout 3 ros2 topic echo /rtk_pvh --once 2>/dev/null)
    fi

    if [ -n "$RTK_DATA" ]; then
        pass "RTK 原始数据 (/rtk_pvh)"

        POS_TYPE=$(echo "$RTK_DATA" | grep 'pos_type' | head -1 | awk '{print $2}')
        HDG_TYPE=$(echo "$RTK_DATA" | grep 'heading_type' | head -1 | awk '{print $2}')
        SOL_STATUS=$(echo "$RTK_DATA" | grep 'sol_status' | head -1 | awk '{print $2}')
        SVSNUM=$(echo "$RTK_DATA" | grep 'svs_num' | head -1 | awk '{print $2}')
        SOLN_SVS=$(echo "$RTK_DATA" | grep 'soln_svs_num' | head -1 | awk '{print $2}')
        LAT_STD=$(echo "$RTK_DATA" | grep 'lat_std' | head -1 | awk '{print $2}')
        LON_STD=$(echo "$RTK_DATA" | grep 'lon_std' | head -1 | awk '{print $2}')
        HDG_STD=$(echo "$RTK_DATA" | grep 'heading_std' | head -1 | awk '{print $2}')

        echo "  ┌─ 定位质量 ──────────────────"
        case "$POS_TYPE" in
            50) info "pos_type: $POS_TYPE (RTK_FIX — 厘米级)";;
            34) warn "pos_type: $POS_TYPE (RTK_FLOAT — 分米级)";;
            16) warn "pos_type: $POS_TYPE (DGPS — 亚米级)";;
            0|"") fail "pos_type: ${POS_TYPE:-无} (无卫星信号 → 室内)";;
            *) info "pos_type: $POS_TYPE";;
        esac

        echo "  ┌─ 精度指标 ──────────────────"
        if [ -n "$LAT_STD" ] && [ -n "$LON_STD" ]; then
            H_ACC=$(python3 -c "import math; print(f'{math.sqrt(${LAT_STD}**2 + ${LON_STD}**2):.3f}')" 2>/dev/null || echo "?")
            if [ "$H_ACC" != "?" ]; then
                if python3 -c "exit(0 if float('$H_ACC') < 0.1 else 1)" 2>/dev/null; then
                    pass "水平精度: ${H_ACC}m (RTK级)"
                elif python3 -c "exit(0 if float('$H_ACC') < 1.0 else 1)" 2>/dev/null; then
                    warn "水平精度: ${H_ACC}m (亚米级)"
                else
                    fail "水平精度: ${H_ACC}m (GPS级，偏差大)"
                fi
            fi
        fi
        [ -n "$SVSNUM" ] && info "卫星数: 可见${SVSNUM}颗 / 解算${SOLN_SVS:-?}颗"
        [ -n "$HDG_STD" ] && info "航向精度: ${HDG_STD}°"

        echo "  ┌─ 航向可用性 ────────────────"
        SOL_OK=0
        [ "$SOL_STATUS" = "0" ] || [ "$SOL_STATUS" = "2" ] && SOL_OK=1
        HDG_OK=0
        [ "$HDG_TYPE" = "50" ] || [ "$HDG_TYPE" = "34" ] || [ "$HDG_TYPE" = "16" ] && HDG_OK=1

        if [ $SOL_OK -eq 1 ] && [ $HDG_OK -eq 1 ]; then
            pass "RTK 航向可用（sol=$SOL_STATUS, hdg_type=$HDG_TYPE）→ 纠偏可使用真北航向"
        else
            warn "RTK 航向不可用（sol=$SOL_STATUS, hdg_type=$HDG_TYPE）→ 纠偏仅纠正位置，yaw 由 AMCL 维持"
        fi

        echo "  └─────────────────────────────"
    else
        fail "RTK 无数据 (/rtk_pvh) → 检查 RTK 天线/USB 连接"
    fi

    # ---- 6. 地图原点（自动标定） ----
    header "6. 地图原点（自动标定）"
    LOG_CALIB=$(journalctl -u gps-rtk-nav --no-pager -n 30 2>/dev/null | grep '自动标定完成' | tail -1)
    if [ -n "$LOG_CALIB" ]; then
        pass "已自动标定（首次 /initialpose 触发）"
        echo "  $LOG_CALIB"
    else
        info "等待首次 /initialpose 标定 — 发布初始位姿后自动完成"
    fi

    # ---- 7. 纠偏状态 ----
    header "7. 纠偏运行状态"
    LOG=$(journalctl -u gps-rtk-nav --no-pager -n 10 2>/dev/null | grep -E '状态|WAIT_ORIGIN|标定|纠偏' | tail -1)
    if [ -n "$LOG" ]; then
        echo "  $LOG"
        if echo "$LOG" | grep -q 'MONITORING'; then
            pass "纠偏监控中"
        elif echo "$LOG" | grep -q 'WAIT_ORIGIN'; then
            info "等待首次 /initialpose 自动标定"
        elif echo "$LOG" | grep -q 'GPS_LOST'; then
            warn "GPS_LOST 状态 → 等 RTK 恢复"
        fi
    else
        fail "无法获取纠偏日志 → journalctl -u gps-rtk-nav -f"
    fi

    # ---- 8. LIO 里程计 ----
    header "8. LIO 里程计"
    FOUND_ODOM=""
    for odom in "/lio/robo/odom" "/Odometry" "/lio/odom" "/rkbot/lio/robo/odom"; do
        if timeout 1 ros2 topic echo "$odom" --once 2>/dev/null | grep -q 'position'; then
            pass "LIO odom ($odom)"
            FOUND_ODOM="$odom"
            break
        fi
    done
    [ -z "$FOUND_ODOM" ] && fail "LIO 里程计无数据 → 检查 SLAM 是否运行"

    # ---- 总结 ----
    echo ""
    echo "=========================================="
    echo -e "  诊断完成。${GREEN}✓=正常${NC}  ${YELLOW}!=注意${NC}  ${RED}✗=需修复${NC}"
    echo "  下一步: 发布 /initialpose 自动标定原点"
    echo "         ./deploy_check.sh status      (持续监控)"
    echo "=========================================="
}

# ======== 标定原点（通过发布 /initialpose 触发自动标定） ========
do_calibrate() {
    echo -e "\n${BOLD}地图原点自动标定${NC}"
    echo "=========================================="

    echo ""
    echo "  前提检查:"
    echo "  1. 机器人已在地图中定位（可以停在任意已知位姿）"
    echo "  2. RTK 天线可见天空 (pos_type=50/34)"
    echo "  3. gps-rtk-nav 服务运行中"
    echo "  4. 标定后持续监控，偏差≥1m 自动纠偏"

    echo ""
    if timeout 2 ros2 topic echo /rtk_pvh --once 2>/dev/null | grep -q 'pos_type'; then
        PT=$(timeout 2 ros2 topic echo /rtk_pvh --once 2>/dev/null | grep 'pos_type' | head -1 | awk '{print $2}')
        if [ "$PT" -ge 34 ] 2>/dev/null; then
            pass "RTK 信号 OK (pos_type=$PT)"
        else
            warn "RTK 精度偏低 (pos_type=$PT)，建议等 FIX 后再标定"
        fi
    else
        fail "RTK 无信号！推到室外再试"
        return
    fi

    echo ""
    info "发送 /initialpose 触发自动标定（当前位姿为原点）..."
    echo "  坐标: (0, 0, 0), 朝向: 0°"
    echo ""
    echo "按回车发送..."
    read -r

    ros2 topic pub /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "
header:
  frame_id: map
pose:
  pose:
    position: {x: 0.0, y: 0.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
" -1

    sleep 2
    echo ""
    LOG_CALIB=$(journalctl -u gps-rtk-nav --no-pager -n 10 2>/dev/null | grep '自动标定完成' | tail -1)
    if [ -n "$LOG_CALIB" ]; then
        pass "标定成功！"
        echo "  $LOG_CALIB"
        echo ""
        pass "纠偏监控已自动开始 (threshold=1.0m)"
    else
        warn "未检测到标定日志"
        echo "  查看完整日志: journalctl -u gps-rtk-nav -f"
        echo ""
        echo "  如果看到 '尚未收到 RTK GPS 数据'，等待几秒后重试："
        echo "  ros2 topic pub /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \"header:{frame_id:map} pose:{pose:{position:{x:0,y:0,z:0} orientation:{x:0,y:0,z:0,w:1}}}\" -1"
    fi
}

# ======== 状态监控 ========
do_status() {
    echo -e "${BOLD}GPS RTK 纠偏 实时状态 (Ctrl+C 退出)${NC}"
    echo ""
    printf "%-12s %-12s %-20s %-20s %-8s %-10s %s\n" \
        "GPS(x)" "GPS(y)" "AMCL(x)" "AMCL(y)" "drift" "Q" "origin"
    echo "───────────────────────────────────────────────────────────────────────────"

    journalctl -u gps-rtk-nav -f --no-pager 2>/dev/null | \
        while read -r line; do
            if echo "$line" | grep -q '\[状态\]'; then
                echo "$line" | sed -E \
                    's/.*GPS=\(([^,]*),([^)]*)\).*AMCL=\(([^,]*),([^)]*)\).*drift=([^ ]*)m.*Q=([^ ]*).*origin=([^ ]*).*/'"$(printf '%-12s %-12s %-20s %-20s %-8s %-10s %s' '\1' '\2' '\3' '\4' '\5' '\6' '\7')/"
            elif echo "$line" | grep -q '纠偏:'; then
                echo -e "  ${GREEN}>>> 纠偏触发${NC} $(echo "$line" | grep -o 'drift=[^ ]*')"
            elif echo "$line" | grep -q '室内模式'; then
                echo -e "  ${YELLOW}>>> 进入室内模式${NC}"
            elif echo "$line" | grep -q '退出室内'; then
                echo -e "  ${GREEN}>>> 退出室内模式${NC}"
            fi
        done
}

# ======== 入口 ========
case "${1:-check}" in
    check)     do_check ;;
    calibrate) do_calibrate ;;
    status)    do_status ;;
    restart)   sudo systemctl restart gps-rtk-nav && sleep 3 && do_check ;;
    *)         echo "用法: $0 [check|calibrate|status|restart]" ;;
esac
