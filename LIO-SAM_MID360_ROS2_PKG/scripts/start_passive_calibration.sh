#!/bin/bash
# ============================================================
# 被动持续定位服务脚本
# 
# 用法:
#   ./start_passive_calibration.sh  [namespace]
#   ./start_passive_calibration.sh  rkbot   # 中狗
#
# 功能:
#   - 启动/停止/查询 被动持续定位服务
#   - 不控制机器人运动, 仅被动采集雷达数据 + 后台匹配
# ============================================================

NS=""
if [ -n "$1" ]; then
    NS="/$1"
fi

SERVICE="${NS}/start_passive_calibration"
STOP_SERVICE="${NS}/stop_passive_calibration"
STATUS_SERVICE="${NS}/auto_calibration_status"

echo "============================================"
echo "  被动持续定位服务"
echo "  Namespace: ${NS:-/} (默认)"
echo "============================================"

case "${2:-start}" in
    start)
        echo "[启动] 正在启动被动持续定位..."
        ros2 service call "$SERVICE" std_srvs/srv/Trigger
        echo ""
        echo "被动定位已启动, 每 N 秒自动匹配一次"
        echo "查看状态: $0 $1 status"
        echo "查看 debug 话题: ros2 topic echo ${NS}/debug/auto_initial_pose"
        ;;
    
    stop)
        echo "[停止] 正在停止被动持续定位..."
        ros2 service call "$STOP_SERVICE" std_srvs/srv/Trigger
        ;;
    
    status)
        echo "[状态] 查询当前状态..."
        ros2 service call "$STATUS_SERVICE" std_srvs/srv/Trigger
        ;;
    
    *)
        echo "用法: $0 [namespace] [start|stop|status]"
        echo ""
        echo "示例:"
        echo "  $0           start    # 默认 namespace, 启动被动定位"
        echo "  $0 rkbot     start    # 中狗 namespace, 启动被动定位"
        echo "  $0           stop     # 停止被动定位"
        echo "  $0           status   # 查询状态"
        exit 1
        ;;
esac
