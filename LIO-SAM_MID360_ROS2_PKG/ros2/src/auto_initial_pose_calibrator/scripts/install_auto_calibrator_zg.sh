#!/bin/bash
# 自动初始位姿校准器 (中狗) 服务安装脚本
# 用法: sudo bash install_auto_calibrator_zg.sh

set -e

echo "Installing Auto Calibrator Service (ZG)..."

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root or with sudo"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Make run script executable
chmod +x "$SCRIPT_DIR/run_auto_calibrator_zg.sh"

# Install the service
echo "Installing systemd service..."
cp "$SCRIPT_DIR/auto_calibrator.zg.service" /etc/systemd/system/auto_calibrator.service

# Reload systemd
systemctl daemon-reload

echo ""
echo "=== 安装完成 ==="
echo ""
echo "启动服务:   sudo systemctl start auto_calibrator"
echo "查看状态:   sudo systemctl status auto_calibrator"
echo "查看日志:   journalctl -u auto_calibrator -f"
echo "开机自启:   sudo systemctl enable auto_calibrator"
echo "停止服务:   sudo systemctl stop auto_calibrator"
echo "重启服务:   sudo systemctl restart auto_calibrator"
echo ""
