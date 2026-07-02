#!/bin/bash
# ============================================================
# gps-rtk-nav 系统服务 一键安装/卸载/管理脚本
# 部署路径: /home/ztl/cql/gps_fusion/scripts/
# ============================================================

set -e

SERVICE_NAME="gps-rtk-nav"
SERVICE_FILE="gps-rtk-nav.service"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVICE_SRC="$SCRIPT_DIR/$SERVICE_FILE"
SERVICE_DST="/etc/systemd/system/$SERVICE_FILE"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}[OK]${NC} $1"; }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; }
info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }

require_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}需要 root 权限，请用 sudo 运行${NC}"
        exit 1
    fi
}

do_install() {
    require_root

    echo ""
    echo -e "${CYAN}===== 安装 gps-rtk-nav 系统服务 =====${NC}"
    echo ""

    if ! id -u ztl >/dev/null 2>&1; then
        fail "用户 ztl 不存在"
        exit 1
    fi

    if [ ! -f "$SERVICE_SRC" ]; then
        fail "服务文件不存在: $SERVICE_SRC"
        exit 1
    fi

    SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
    LAUNCH_SCRIPT="$SCRIPTS_DIR/scripts/gps_rtk_nav_service.sh"
    if [ ! -x "$LAUNCH_SCRIPT" ]; then
        info "设置执行权限: $LAUNCH_SCRIPT"
        chmod +x "$LAUNCH_SCRIPT"
    fi
    ok "启动脚本就绪"

    cp "$SERVICE_SRC" "$SERVICE_DST"
    ok "服务文件已安装: $SERVICE_DST"

    systemctl daemon-reload
    ok "systemd 配置已重载"

    systemctl enable "$SERVICE_NAME" 2>/dev/null && \
        ok "已设置开机自启" || \
        fail "设置开机自启失败"

    systemctl start "$SERVICE_NAME" 2>/dev/null && \
        ok "服务已启动" || \
        fail "服务启动失败"

    echo ""
    echo -e "${GREEN}===== 安装完成 =====${NC}"
    echo ""
    echo "  查看状态:  systemctl status $SERVICE_NAME"
    echo "  查看日志:  journalctl -u $SERVICE_NAME -f"
    echo "  停止服务:  systemctl stop $SERVICE_NAME"
    echo "  重启服务:  systemctl restart $SERVICE_NAME"
    echo ""
}

do_uninstall() {
    require_root

    echo ""
    echo -e "${YELLOW}===== 卸载 gps-rtk-nav 系统服务 =====${NC}"
    echo ""

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl stop "$SERVICE_NAME"
        ok "服务已停止"
    else
        info "服务未在运行"
    fi

    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        systemctl disable "$SERVICE_NAME"
        ok "已取消开机自启"
    else
        info "未设置开机自启"
    fi

    pkill -f "rtk_pose_monitor" 2>/dev/null && info "已清理 rtk_pose_monitor 残留进程" || true

    if [ -f "$SERVICE_DST" ]; then
        rm -f "$SERVICE_DST"
        ok "服务文件已删除: $SERVICE_DST"
    else
        info "服务文件不存在"
    fi

    systemctl daemon-reload
    ok "systemd 配置已重载"

    echo ""
    echo -e "${GREEN}===== 卸载完成 =====${NC}"
    echo ""
}

do_status() {
    echo ""
    echo -e "${CYAN}===== 服务状态: $SERVICE_NAME =====${NC}"
    echo ""

    if [ -f "$SERVICE_DST" ]; then
        ok "服务文件: $SERVICE_DST"
    else
        fail "服务文件未安装"
    fi

    if systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
        ok "运行状态: 运行中"
    else
        fail "运行状态: 未运行"
    fi

    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        ok "开机自启: 已启用"
    else
        info "开机自启: 未启用"
    fi

    echo ""
    echo "最近日志 (10行):"
    journalctl -u "$SERVICE_NAME" -n 10 --no-pager 2>/dev/null || echo "  无日志"
    echo ""
}

do_restart() {
    require_root
    echo ""
    systemctl restart "$SERVICE_NAME" && \
        ok "服务已重启" || \
        fail "重启失败"
    echo ""
}

show_menu() {
    echo ""
    echo "=========================================="
    echo "  GPS RTK 导航纠偏 - 系统服务管理"
    echo "=========================================="
    echo "  1) 安装并启动 (install + enable + start)"
    echo "  2) 卸载        (stop + disable + remove)"
    echo "  3) 查看状态    (status + recent logs)"
    echo "  4) 重启服务    (restart)"
    echo "  5) 停止服务    (stop)"
    echo "  0) 退出"
    echo "=========================================="
    echo -n "请选择 [1-5,0]: "
}

main() {
    if [ -n "$1" ]; then
        case "$1" in
            install)  do_install ;;
            uninstall) do_uninstall ;;
            status)   do_status ;;
            restart)  do_restart ;;
            stop)     require_root; systemctl stop "$SERVICE_NAME"; ok "已停止" ;;
            *)        echo "用法: $0 [install|uninstall|status|restart|stop]" ;;
        esac
        exit 0
    fi

    while true; do
        show_menu
        read -r choice
        case "$choice" in
            1) do_install ;;
            2) do_uninstall ;;
            3) do_status ;;
            4) do_restart ;;
            5) require_root; systemctl stop "$SERVICE_NAME" && ok "已停止" || fail "停止失败" ;;
            0) echo "退出"; exit 0 ;;
            *) echo -e "${RED}无效选项，请重新选择${NC}" ;;
        esac
    done
}

main "$@"
