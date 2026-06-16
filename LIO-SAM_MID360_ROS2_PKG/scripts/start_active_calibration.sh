#!/bin/bash
# ============================================================
# 主动校准服务脚本 (旋转360° + 8方向移动探索)
#
# 用法:
#   ./start_active_calibration.sh  [namespace]
# ============================================================

NS=""
if [ -n "$1" ]; then
    NS="/$1"
fi

SERVICE="${NS}/start_active_calibration"
STATUS_SERVICE="${NS}/auto_calibration_status"

echo "============================================"
echo "  主动校准服务 (旋转+移动探索)"
echo "  Namespace: ${NS:-/} (默认)"
echo "============================================"

case "${2:-start}" in
    start)
        echo "[启动] 正在启动主动校准..."
        echo "  - 静止等待 2 秒"
        echo "  - 旋转 360° 采集雷达数据"
        echo "  - 8 方向移动探索 + 避障"
        echo "  - 收敛后自动发布 /initialpose"
        echo ""
        ros2 service call "$SERVICE" std_srvs/srv/Trigger
        echo ""
        echo "查看状态: $0 $1 status"
        echo "查看候选位姿: ros2 topic echo ${NS}/debug/candidates"
        ;;
    
    status)
        ros2 service call "$STATUS_SERVICE" std_srvs/srv/Trigger
        ;;
    
    *)
        echo "用法: $0 [namespace] [start|status]"
        echo ""
        echo "示例:"
        echo "  $0           start    # 启动主动校准"
        echo "  $0 rkbot     start    # 中狗 namespace"
        echo "  $0           status   # 查询状态"
        exit 1
        ;;
esac
