#!/bin/bash
# 中狗 (ZG) 自动初始位姿校准器启动脚本
# 用法: ./run_auto_calibrator_zg.sh

WORKSPACE_DIR="/home/ztl/cql"

# ────── CPU 亲和性绑定 ──────
# 将该进程及其子节点(校准器)限定在指定核心，避免与 LIO/Nav2 抢占大核时间片。
# RK3588 为 big.LITTLE，本机实测(cat /sys/devices/system/cpu/cpu*/cpu_capacity):
#   小核 A55 (capacity=530) -> cpu0,1,2,3
#   大核 A76 (capacity=1024) -> cpu4,5,6,7
# 校准器是后台重计算(Python+OpenCV 单线程全图匹配)，绑到小核即可，把大核让给 LIO/Nav2。
# 注: taskset 只限定"能在哪些核跑"，并非限制 CPU 占用百分比；
#     若要限百分比请用 cpulimit -p <pid> -l <百分比>。
CALIB_CPUS="0,1,2,3"

# 加载 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/humble/setup.bash
fi
source $WORKSPACE_DIR/install/setup.bash

echo "===== 中狗 (ZG) 自动初始位姿校准器 ====="
echo "Workspace: $WORKSPACE_DIR"
echo "命名空间: rkbot"
echo "CPU 亲和性绑定: cores=$CALIB_CPUS"
echo ""

taskset -c "$CALIB_CPUS" ros2 launch auto_initial_pose_calibrator auto_initial_pose_calibration.launch.py ns:=rkbot
