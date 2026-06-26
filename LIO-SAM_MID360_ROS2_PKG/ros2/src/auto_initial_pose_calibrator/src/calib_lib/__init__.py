"""auto_initial_pose_calibrator 内部算法子包。

本子包将原 3634 行单文件按职责拆分为分层模块：
  - scan_utils: 激光扫描 ↔ 点云转换、离群点/FRF 过滤、角度规整
  - icp: 帧间 ICP 匹配
  - temporal: 多帧时序一致性过滤（动态物体剔除）
  - scoring: 似然场、点云/墙壁/光线投射评分、综合置信度、两级局部搜索
  - submap: 子图构建（SubmapBuilder）
  - matching: 扫描匹配编排（ScanMatcher，含 hierarchical/multistep/passive 三路径）
  - control: 主动运动选择与避障（MotionController）
  - motion_tracker: 运动状态跟踪与漂移检测（MotionTracker）
  - rtk: RTK/GPS → map 变换

注意：子模块对 ROS 无硬依赖（scipy/cv2 为可选 import），
可在 Windows 上直接 import 与单元测试；rclpy 仅在主节点使用。
"""

from enum import Enum


class IndoorPhase(Enum):
    IDLE = 0
    BOOT_DELAY = 1
    ROTATING_360 = 2
    COLLECTING_SUBMAP1 = 3
    ROUGH_MATCHING = 4
    SELECTING_ACTIVE_MOTION = 5
    MOVING = 6
    COLLECTING_SUBMAP2 = 7
    FILTERING = 8
    DONE = 9
    PASSIVE_COLLECTING = 10
    PASSIVE_MATCHING = 11
    ACTIVE_MULTISTEP = 12
