#!/bin/bash
# ============================================================
# GPS融合 Web 可视化服务启动脚本
# 使用带 SO_REUSEADDR 的自定义 HTTP 服务器（解决端口残留问题）
# 前端 map_viewer.html 直连 WebSocket 8765 端口（无需 rosbridge）
# ============================================================
#
# 使用脚本自身所在目录作为 web 根目录（colcon install 会将 web/ 安装到
# share/gps_fusion/web/，此脚本也在同一目录下，无论 workspace 路径如何都正确）

WEB_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${WEB_PORT:-8084}"

if [ ! -f "$WEB_DIR/map_viewer.html" ] && [ ! -f "$WEB_DIR/map_tracking_view.html" ]; then
    echo "[web] 警告: $WEB_DIR 中未找到 map_viewer.html 或 map_tracking_view.html，HTTP 仍可启动用于浏览其他文件"
fi

# ============================================================
# 启动前清理：杀死占用端口的残留进程
# ============================================================
_cleanup_port() {
    local port=$1
    local pid
    pid=$(fuser "$port/tcp" 2>/dev/null | awk '{print $1}')
    if [ -n "$pid" ]; then
        echo "[web] 端口 $port 被 PID $pid 占用，正在清理..."
        kill -9 $pid 2>/dev/null || true
        sleep 0.5
    fi
}
_cleanup_port "$PORT"
_cleanup_port "${WS_TRAJECTORY_PORT:-8765}"

# 用自定义服务器（带 SO_REUSEADDR）替代 python3 -m http.server
# 避免重启时的 "Address already in use" 错误
exec python3 "$WEB_DIR/http_server.py" "$PORT" --dir "$WEB_DIR"
