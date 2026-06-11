# Draft: 文件同步脚本

## 参考模板
用户提供的 G2-Robot 同步脚本使用 `rsync` 从本地同步到远程机器人主机。

## 项目上下文
- 项目: `dog_slam` — ROS2 Humble SLAM + Nav2 导航系统
- 目标: RK3588 机器人板卡 (IP: 10.44.10.x 等)
- 关键目录: `LIO-SAM_MID360_ROS2_PKG/` 为核心工作区
- 已有 `global_config/__init__.py` 按 hostname 自动配置路径
- 已有 `deploy_new_board.sh` 但这是全新部署脚本，不是快速同步脚本

## 待确认问题
1. 目标服务器 IP 和用户名
2. 同步目录范围（整个 repo 还是仅 LIO-SAM_MID360_ROS2_PKG？）
3. 排除规则
4. 是否需要多服务器配置
5. 是否需要集成 global_config 的 hostname 检测
