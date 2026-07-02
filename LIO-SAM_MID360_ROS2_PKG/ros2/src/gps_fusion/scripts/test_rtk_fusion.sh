#!/bin/bash
# ============================================================
# RTK-Nav 零侵入融合 集成测试脚本（室内模拟 + 室外真实两阶段）
# ============================================================
#
# ⚠️ 此脚本只能在 Ubuntu 22.04 + ROS2 Humble 环境运行
# ⚠️ 需要 robots_dog_msgs 包已编译安装
#
# 用法:
#   # Phase 1: 单元测试（macOS/Ubuntu 均可）
#   cd ros2/src/gps_fusion && ./scripts/test_rtk_fusion.sh unit
#
#   # Phase 2a: 建图记录室内模拟
#   ./scripts/test_rtk_fusion.sh sim-record
#
#   # Phase 2b: 导航纠偏室内模拟（需 Phase 2a 先生成 YAML）
#   ./scripts/test_rtk_fusion.sh sim-monitor
#
#   # Phase 2: 一键室内全链路模拟
#   ./scripts/test_rtk_fusion.sh sim-all
#
#   # Phase 3: 室外真实数据（建图阶段）
#   ./scripts/test_rtk_fusion.sh real-record
#
#   # Phase 3: 室外真实数据（导航阶段）
#   ./scripts/test_rtk_fusion.sh real-monitor
#
# 验证检查点:
#   - record: 检查生成的 map_gps_origin.yaml 内容是否合理
#   - monitor: 检查 /initialpose 是否在漂移超阈值时被发布
#   - 航向: 检查 heading_deg 是否在合理范围 [0, 360)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"          # gps_fusion 包根目录
CONFIG_DIR="$PKG_DIR/config"
WS_DIR="$PKG_DIR/../.."                     # ros2 工作空间

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS="${GREEN}[PASS]${NC}"
FAIL="${RED}[FAIL]${NC}"
WARN="${YELLOW}[WARN]${NC}"
STEP="${CYAN}[STEP]${NC}"

# 测试输出目录
TEST_OUTPUT_DIR="/tmp/gps_fusion_test_$$"
TOTAL_TESTS=0
PASSED_TESTS=0

# ======== 工具函数 ========

log_step() {
    echo -e "\n${STEP} $1"
}

log_pass() {
    echo -e "  ${PASS} $1"
    PASSED_TESTS=$((PASSED_TESTS + 1))
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

log_fail() {
    echo -e "  ${FAIL} $1"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

assert_file_exists() {
    local path="$1"
    local desc="$2"
    if [ -f "$path" ]; then
        log_pass "$desc ($path)"
    else
        log_fail "$desc — 文件不存在: $path"
    fi
}

assert_topic_active() {
    local topic="$1"
    local timeout="${2:-5}"
    local desc="$3"
    echo -e "  ... 检查话题 $topic ..."
    if timeout "$timeout" ros2 topic echo "$topic" --once &>/dev/null; then
        log_pass "$desc (话题 $topic 有数据)"
    else
        log_fail "$desc — 话题 $topic 超时无数据"
    fi
}

cleanup_procs() {
    echo -e "\n${YELLOW}清理进程...${NC}"
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    PIDS=()
}

trap cleanup_procs EXIT

wait_or_timeout() {
    local timeout="$1"
    local start=$(date +%s)
    while [ $(($(date +%s) - start)) -lt "$timeout" ]; do
        sleep 1
    done
}

# ======== Phase 1: 单元测试 (macOS/Ubuntu) ========

phase_unit() {
    log_step "Phase 1: Python 单元测试（纯函数，不依赖 ROS2）"

    cd "$PKG_DIR"
    if python3 -m pytest tests/test_gps_transform.py -v 2>&1; then
        log_pass "所有单元测试通过"
    else
        log_fail "单元测试失败"
    fi
}

# ======== Phase 2a: 建图记录室内模拟 ========

phase_sim_record() {
    log_step "Phase 2a: 建图原点记录（室内模拟）"

    local test_yaml="$TEST_OUTPUT_DIR/map_gps_origin.yaml"
    mkdir -p "$TEST_OUTPUT_DIR"
    echo "  输出文件: $test_yaml"

    # 1. 启动 rtk_simulator（RTK_FIX 模式，静止在圆心）
    log_step "启动 rtk_simulator (pos_type=50, 静止)"
    ros2 run gps_fusion rtk_simulator.py --ros-args \
        -p rate:=5.0 -p pos_type:=50 -p speed:=0.0 -p topic:=/test/rtk_pvh \
        > "$TEST_OUTPUT_DIR/rtk_sim.log" 2>&1 &
    PIDS+=($!)
    wait_or_timeout 2

    # 2. 启动 gps_preprocessor + map_origin_recorder
    log_step "启动 map_origin_record.launch.py"
    ros2 launch gps_fusion map_origin_record.launch.py \
        rtk_topic:=/test/rtk_pvh \
        sample_count:=5 \
        min_accuracy:=10.0 \
        rtk_min_accuracy:=10.0 \
        output_file:="$test_yaml" \
        > "$TEST_OUTPUT_DIR/record.log" 2>&1 &
    PIDS+=($!)
    wait_or_timeout 2

    # 验证话题
    assert_topic_active "/fix_filtered" 5 "GPS 预处理器输出 /fix_filtered"

    # 3. 等待自动记录完成（5帧采样 + buffering）
    echo -e "  ... 等待自动记录（约 15 秒）..."
    wait_or_timeout 15

    # 4. 手动触发也会记录（测试 service）
    log_step "手动触发 /gps_origin/record service"
    if ros2 service call /gps_origin/record std_srvs/srv/Trigger 2>/dev/null; then
        log_pass "手动记录 service 调用成功"
    else
        log_fail "手动记录 service 调用失败"
    fi

    wait_or_timeout 2

    # 5. 验证 YAML 文件
    log_step "验证 map_gps_origin.yaml"
    assert_file_exists "$test_yaml" "map_gps_origin.yaml 已生成"

    if [ -f "$test_yaml" ]; then
        # 检查关键字段
        echo "  文件内容:"
        cat "$test_yaml"
        echo ""

        local lat=$(grep 'latitude:' "$test_yaml" | head -1 | awk '{print $2}')
        local lon=$(grep 'longitude:' "$test_yaml" | head -1 | awk '{print $2}')
        local hdg=$(grep 'heading_deg:' "$test_yaml" | head -1 | awk '{print $2}')

        if [ -n "$lat" ] && [ -n "$lon" ]; then
            log_pass "经纬度: ($lat, $lon)"
        else
            log_fail "经纬度字段缺失"
        fi

        if [ -n "$hdg" ]; then
            # 检查航向在合理范围
            local in_range=$(python3 -c "h=$hdg; print('OK' if 0 <= h < 360 else 'BAD')")
            if [ "$in_range" = "OK" ]; then
                log_pass "航向 heading_deg=$hdg 在 [0, 360)"
            else
                log_fail "航向 heading_deg=$hdg 超出 [0, 360)"
            fi
        fi

        local source=$(grep 'source:' "$test_yaml" | head -1)
        if echo "$source" | grep -q 'manual\|auto'; then
            log_pass "记录来源: $source"
        fi
    fi

    cleanup_procs
}

# ======== Phase 2b: 导航纠偏室内模拟 ========

phase_sim_monitor() {
    log_step "Phase 2b: 导航纠偏（室内模拟 + Mock TF 漂移）"

    local test_yaml="$TEST_OUTPUT_DIR/map_gps_origin.yaml"

    if [ ! -f "$test_yaml" ]; then
        # 如果没有 YAML，先生成
        log_step "未找到 $test_yaml，先生成"
        phase_sim_record

        # 给自动记录完成的文件（第二次运行会报"已完成"但文件已生成）
        if [ ! -f "$test_yaml" ]; then
            log_fail "无法生成 map_gps_origin.yaml，无法继续纠偏测试"
            return 1
        fi
    fi

    # 1. 启动 rtk_simulator（模拟移动 GPS，圆形轨迹，RTK_FIX）
    log_step "启动 rtk_simulator (圆形轨迹，RTK_FIX, 2Hz)"
    ros2 run gps_fusion rtk_simulator.py --ros-args \
        -p rate:=2.0 -p pos_type:=50 -p radius:=10.0 -p speed:=0.5 \
        -p topic:=/test/rtk_pvh \
        > "$TEST_OUTPUT_DIR/rtk_sim_monitor.log" 2>&1 &
    PIDS+=($!)
    wait_or_timeout 2

    # 2. 启动 Mock TF（初始与 GPS 对齐，然后漂移）
    log_step "启动 Mock TF 广播器 (漂移 0.3 m/s，方向 135° = 东南)"
    ros2 run gps_fusion mock_tf_broadcaster.py --ros-args \
        -p init_x:=0.0 -p init_y:=0.0 -p init_yaw:=0.0 \
        -p drift_per_second:=0.3 -p drift_direction_deg:=135.0 \
        -p rate:=10.0 \
        > "$TEST_OUTPUT_DIR/mock_tf.log" 2>&1 &
    PIDS+=($!)
    wait_or_timeout 2

    # 3. 启动 gps_preprocessor + rtk_pose_monitor
    log_step "启动 rtk_nav_bridge.launch.py"
    ros2 launch gps_fusion rtk_nav_bridge.launch.py \
        rtk_topic:=/test/rtk_pvh \
        map_origin_file:="$test_yaml" \
        drift_threshold:=2.0 \
        min_correction_interval:=10.0 \
        monitor_rate:=2.0 \
        rtk_min_accuracy:=10.0 \
        use_rtk_heading:=true \
        > "$TEST_OUTPUT_DIR/monitor.log" 2>&1 &
    PIDS+=($!)
    wait_or_timeout 3

    # 验证话题
    assert_topic_active "/fix_filtered" 5 "GPS 预处理器输出 /fix_filtered"

    # 验证 TF 可用
    log_step "验证 TF: map → base_footprint"
    if timeout 3 ros2 run tf2_ros tf2_echo map base_footprint 2>/dev/null | head -5; then
        log_pass "TF map→base_footprint 可用"
    else
        log_fail "TF map→base_footprint 不可用"
    fi

    # 4. 等待漂移累积超过 2m 阈值
    # 漂移 0.3 m/s，达到 2m 需要约 7 秒
    log_step "等待漂移累积超过阈值 (约 10-15 秒)..."
    echo "  当前时间: $(date '+%H:%M:%S')"

    # 监控 /initialpose（预期在大约 12-15 秒后出现第一条纠偏消息）
    log_step "监控 /initialpose（等待纠偏触发）"

    CORRECTION_SEEN=0
    start_time=$(date +%s)
    while [ $(($(date +%s) - start_time)) -lt 25 ]; do
        if timeout 2 ros2 topic echo /initialpose --once 2>/dev/null | grep -q 'position'; then
            CORRECTION_SEEN=1
            echo ""
            log_pass "检测到 /initialpose 纠偏消息！"
            ros2 topic echo /initialpose --once 2>/dev/null
            break
        fi
        echo -n "."
        sleep 2
    done

    if [ $CORRECTION_SEEN -eq 0 ]; then
        log_fail "超时 25s 未检测到 /initialpose 纠偏"
        echo "  排查提示:"
        echo "    - 检查 rtk_pose_monitor 日志: cat $TEST_OUTPUT_DIR/monitor.log"
        echo "    - 检查 TF 是否正常: ros2 run tf2_ros tf2_echo map base_footprint"
        echo "    - 检查 /fix_filtered 是否有数据: ros2 topic echo /fix_filtered --once"
    fi

    cleanup_procs
}

# ======== Phase 2: 一键室内全链路 ========

phase_sim_all() {
    log_step "Phase 2: 室内全链路模拟测试"

    # 清理旧文件
    rm -rf "$TEST_OUTPUT_DIR"
    mkdir -p "$TEST_OUTPUT_DIR"

    echo ""
    echo "========================================"
    echo "  Step 2a: 建图原点记录"
    echo "========================================"
    phase_sim_record

    echo ""
    echo "========================================"
    echo "  Step 2b: 导航纠偏"
    echo "========================================"
    phase_sim_monitor

    echo ""
    echo "========================================"
    echo "  测试结果: $PASSED_TESTS/$TOTAL_TESTS 通过"
    echo "  日志目录: $TEST_OUTPUT_DIR"
    echo "========================================"
}

# ======== Phase 3: 室外真实数据 ========

phase_real_record() {
    log_step "Phase 3a: 室外真实建图原点记录"

    echo ""
    echo "  ⚠️  请确认以下前提条件:"
    echo "    1. 机器狗已放在地图原点 (0,0,0) 位置不动"
    echo "    2. RTK 天线已安装且可视天空"
    echo "    3. 建图 launch 已启动（slam_toolbox 或 octomap_server 运行中）"
    echo "    4. RTK 接收端 /rtk_pvh 正在输出数据"
    echo ""
    echo "  验证 RTK 状态:"
    echo "    ros2 topic echo /rtk_pvh --once | grep -E 'pos_type|heading_type|latitude'"
    echo ""
    echo "  启动记录（自动模式，RTK 收敛后自动采集）:"
    echo "    ros2 launch gps_fusion map_origin_record.launch.py \\"
    echo "        rtk_topic:=/rtk_pvh output_file:=/home/ztl/dog_slam/config/map_gps_origin.yaml"
    echo ""
    echo "  或手动触发（建图员确认 RTK 收敛后）:"
    echo "    启动后等待 RTK 收敛，然后:"
    echo "    ros2 service call /gps_origin/record std_srvs/srv/Trigger"
    echo ""
    echo "  验证检查点:"
    echo "    ✓ heading_deg 在 [0, 360) 范围"
    echo "    ✓ rtk_quality = RTK_FIX 或 RTK_FLOAT"
    echo "    ✓ 经纬度与手机 GPS 读取误差 < 5m"
    echo "    ✓ 输出文件存在且包含 map_origin 字段"
    echo ""
    echo "  ⚠️  RTK 航向方向验证（高优先级）:"
    echo "    1. 机器人朝正北 → 记录 RTK heading_deg (应为 0°)"
    echo "    2. 机器人顺时针转 90° 朝东 → 记录 RTK heading_deg (应为 90°)"
    echo "    3. 对比 LIO 的 ROS yaw (ros2 topic echo /lio/robo/odom --once)"
    echo "    4. 确认 RTK heading 是 真北顺时针 还是 东向逆时针"
}

phase_real_monitor() {
    log_step "Phase 3b: 室外真实导航纠偏"

    echo ""
    echo "  ⚠️  请确认以下前提条件:"
    echo "    1. nav2_dog_slam 导航 launch 已启动（AMCL 运行中，map→base_footprint TF 存在）"
    echo "    2. map_gps_origin.yaml 已通过建图阶段生成"
    echo "    3. RTK 接收端 /rtk_pvh 正在输出 GNSS 信号"
    echo "    4. 机器狗在户外 GNSS 可见区域内"
    echo ""
    echo "  验证 TF 可用:"
    echo "    ros2 run tf2_ros tf2_echo map base_footprint"
    echo ""
    echo "  启动监控:"
    echo "    ros2 launch gps_fusion rtk_nav_bridge.launch.py \\"
    echo "        rtk_topic:=/rtk_pvh \\"
    echo "        map_origin_file:=/home/ztl/dog_slam/config/map_gps_origin.yaml \\"
    echo "        drift_threshold:=2.0 \\"
    echo "        min_correction_interval:=15.0"
    echo ""
    echo "  监控验证检查点:"
    echo "    ✓ rtk_pose_monitor 日志显示 '地图原点已加载'"
    echo "    ✓ rtk_pose_monitor 日志显示 MONITORING 状态"
    echo "    ✓ ros2 topic echo /initialpose 在 AMCL 偏离 GPS >2m 时应有输出"
    echo "    ✓ GPS 信号中断后自动切换到 GPS_LOST 状态"
    echo "    ✓ GPS 恢复后自动回到 MONITORING 状态"
    echo ""
    echo "  主动测试纠偏:"
    echo "    # 方案1: 手动推动机器人偏离原位置 5m（AMCL 会漂移）"
    echo "    #   观察 /initialpose 是否被触发纠偏"
    echo "    # 方案2: 用 ros2 topic pub 伪造 /initialpose 打偏 AMCL"
    echo "    #   然后看 rtk_pose_monitor 是否纠正回来"
}

# ======== 帮助 ========

show_help() {
    echo "RTK-Nav 零侵入融合 测试脚本"
    echo ""
    echo "用法: $0 <phase>"
    echo ""
    echo "可用 Phase:"
    echo "  unit          Phase 1: Python 单元测试（macOS/Ubuntu，不需要 ROS2）"
    echo "  sim-record    Phase 2a: 建图记录 室内模拟测试"
    echo "  sim-monitor   Phase 2b: 导航纠偏 室内模拟测试（含 Mock TF 漂移）"
    echo "  sim-all       Phase 2: 一键室内全链路模拟"
    echo "  real-record   Phase 3a: 室外真实建图记录 操作说明"
    echo "  real-monitor  Phase 3b: 室外真实导航纠偏 操作说明"
    echo ""
    echo "环境要求:"
    echo "  unit:         Python 3 + pyproj（任何平台）"
    echo "  sim-*:        Ubuntu 22.04 + ROS2 Humble + robots_dog_msgs"
    echo "  real-*:       户外 + 机器狗 + RTK GNSS + nav2_dog_slam 运行中"
}

# ======== 主入口 ========

PHASE="$1"

if [ -z "$PHASE" ]; then
    show_help
    exit 0
fi

cd "$PKG_DIR"

case "$PHASE" in
    unit)
        phase_unit
        ;;
    sim-record)
        phase_sim_record
        ;;
    sim-monitor)
        phase_sim_monitor
        ;;
    sim-all)
        phase_sim_all
        ;;
    real-record)
        phase_real_record
        ;;
    real-monitor)
        phase_real_monitor
        ;;
    -h|--help|help)
        show_help
        ;;
    *)
        echo "未知 phase: $PHASE"
        show_help
        exit 1
        ;;
esac
