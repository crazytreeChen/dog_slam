#!/usr/bin/env python3
"""
GPS融合独立启动文件 - gps_fusion 包

启动完整的 GPS 融合管道：
  gps_preprocessor → navsat_transform → ekf_filter + 轨迹可视化

Web 架构：
  - python3 http.server 提供静态页面（端口 8084）→ 浏览器打开 map_viewer.html
  - trajectory_ws_server WebSocket 直推轨迹（端口 8765）→ 前端直连，无需 rosbridge
  - trajectory_server 发布 nav_msgs/Path 到 ROS2 话题

用法:
  ros2 launch gps_fusion gps_fusion.launch.py
  ros2 launch gps_fusion gps_fusion.launch.py lio_odom_topic:=/lio/odom
  ros2 launch gps_fusion gps_fusion.launch.py web_port:=8085
  ros2 launch gps_fusion gps_fusion.launch.py enable_web:=false
  # 多机器人命名空间（ns 参数自动为 frame_id 加前缀）：
  ros2 launch gps_fusion gps_fusion.launch.py ns:=rkbot \\
      lio_odom_topic:=/rkbot/lio/odom \\
      imu_topic:=/front_lidar/imu
"""

import math
import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _load_map_origin(pkg_dir):
    """从 map_gps_origin.yaml 读取地图原点 datum。
    
    返回 [latitude, longitude, altitude, heading_deg] 或 None。
    """
    origin_path = os.path.join(pkg_dir, 'config', 'map_gps_origin.yaml')
    if not os.path.exists(origin_path):
        return None
    try:
        with open(origin_path, 'r') as f:
            data = yaml.safe_load(f)
        origin = data.get('map_origin', {})
        lat = origin.get('latitude')
        lon = origin.get('longitude')
        if lat is None or lon is None:
            return None
        alt = origin.get('altitude', 0.0)
        hdg = origin.get('heading_deg', 0.0)
        datum = [float(lat), float(lon), float(alt), float(hdg)]
        return datum
    except Exception:
        return None


def generate_launch_description():
    # 包路径
    pkg_dir = get_package_share_directory('gps_fusion')
    gps_ekf_config = os.path.join(pkg_dir, 'config', 'gps_ekf.yaml')
    navsat_config = os.path.join(pkg_dir, 'config', 'navsat_transform.yaml')
    web_script = os.path.join(pkg_dir, 'web', 'run_web.sh')
    
    # 尝试加载地图原点（预先标定的 datum）
    map_origin = _load_map_origin(pkg_dir)

    # ======== 启动参数 ========

    ns = LaunchConfiguration('ns', default='')

    # 命名空间感知的 frame_id（对齐 Super-LIO / nav2_dog_slam 命名规范）
    ns_map_frame = PythonExpression(["'map' if '", ns, "' == '' else str('", ns, "/map')"])
    ns_odom_frame = PythonExpression(["'odom' if '", ns, "' == '' else str('", ns, "/odom')"])
    ns_base_link_frame = PythonExpression(["'base_link' if '", ns, "' == '' else str('", ns, "/base_link')"])
    ns_base_footprint_frame = PythonExpression(["'base_footprint' if '", ns, "' == '' else str('", ns, "/base_footprint')"])
    ns_world_frame = PythonExpression(["'world' if '", ns, "' == '' else str('", ns, "/world')"])
    ns_imu_frame = PythonExpression(["'imu' if '", ns, "' == '' else str('", ns, "/imu')"])

    lio_odom_topic = LaunchConfiguration('lio_odom_topic', default='/Odometry')
    gps_topic = LaunchConfiguration('gps_topic', default='/fix')
    imu_topic = LaunchConfiguration('imu_topic', default='/livox/imu')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    utm_zone = LaunchConfiguration('utm_zone', default='50')
    fused_odom_topic = '/odometry/gps_fused'
    web_port = LaunchConfiguration('web_port', default='8084')
    ws_trajectory_port = LaunchConfiguration('ws_trajectory_port', default='8765')
    gps_source = LaunchConfiguration('gps_source', default='/fix')
    enable_web = LaunchConfiguration('enable_web', default='true')
    enable_correction = LaunchConfiguration('enable_correction', default='true')
    enable_ekf = LaunchConfiguration('enable_ekf', default='true')
    enable_recording = LaunchConfiguration('enable_recording', default='false')
    map_origin_file = LaunchConfiguration('map_origin_file', default='')
    rtk_min_accuracy = LaunchConfiguration('rtk_min_accuracy', default='0.02')
    dgps_min_accuracy = LaunchConfiguration('dgps_min_accuracy', default='30.0')

    # ======== 核心融合节点 ========

    # 1. GPS预处理器
    gps_preprocessor_node = Node(
        package='gps_fusion',
        executable='gps_preprocessor.py',
        name='gps_preprocessor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'utm_zone': utm_zone,
            'gps_source': gps_source,
            'min_satellites': 4,
            'max_hdop': 2.0,
            'min_accuracy': 1.0,
            'rtk_min_accuracy': rtk_min_accuracy,
            'dgps_min_accuracy': dgps_min_accuracy,
            'status_threshold': 0,
        }],
    )

    # 2. navsat_transform_node（lifecycle）
    # 如果 map_gps_origin.yaml 存在且有效，使用预标定的 datum（精度可控）
    # 否则回退到 navsat_transform.yaml 中的占位值 + 首次 GPS fix 自动检测
    #
    # ⚠️ robot_localization Humble 源码行为（navsat_transform.cpp）：
    #   - datum 参数格式为 [lat, lon, yaw_rad] （3 元素，不含 altitude）
    #   - 只有 wait_for_datum=true 时 datum 参数才会被声明和使用
    #   - altitude 在内部被硬编码为 0（使用 SetDatum 服务可设完整高度）
    #   - wait_for_datum=false 时 datum 参数被忽略，由首次 GPS fix 自动检测
    navsat_params_override = {
        'use_sim_time': use_sim_time,
        'gps_frame': 'gps',                # GPS天线帧不加命名空间前缀
        'world_frame': ns_map_frame,
        'base_link_frame': ns_base_footprint_frame,
    }
    if map_origin is not None:
        # datum 格式：[lat, lon, yaw_rad] ✓  (注意：不含 altitude)
        datum_yaw_rad = math.radians(map_origin[3])
        navsat_params_override['datum'] = [map_origin[0], map_origin[1], datum_yaw_rad]
        navsat_params_override['wait_for_datum'] = True   # 必须为 true 才能使用 datum
        print(f'[gps_fusion] 使用预标定地图原点作为 navsat datum: '
              f'lat={map_origin[0]:.8f}, lon={map_origin[1]:.8f}, '
              f'yaw={math.degrees(datum_yaw_rad):.4f}° '
              f'(高度由 GPS fix 提供，datum 基准高度=0)')
    else:
        print('[gps_fusion] map_gps_origin.yaml 不存在或无效，'
              'navsat_transform 将从首次 GPS fix 自动检测 datum')
    
    navsat_transform_node = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform_node',
        output='screen',
        condition=IfCondition(enable_ekf),
        parameters=[navsat_config, navsat_params_override],
        remappings=[
            ('/imu/data', imu_topic),
            ('/gps/fix', '/fix_filtered'),
            # navsat_transform 需要 odometry 输入才能输出 /gps/odom。
            # 使用 LIO odom 打破循环依赖：EKF 依赖 /gps/odom，/gps/odom 依赖 odom。
            # LIO odom 高频稳定，足够用于 GPS→UTM 坐标转换。
            ('/odometry/filtered', lio_odom_topic),
        ],
    )

    # 2.5 GPS天线静态变换：navsat_transform 通过 IMU 帧到 gps 帧的 tf 减除天线偏移。
    #     IMU 帧名由 ns 自动推导（ns=rkbot → rkbot/imu），gps 帧不加命名空间前缀。
    #     GPS 天线紧邻 IMU 安装时使用恒等变换即可。
    gps_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='gps_antenna_tf',
        output='screen',
        condition=IfCondition(enable_ekf),
        arguments=['0', '0', '0', '0', '0', '0',
                   ns_imu_frame, 'gps'],
    )

    # 2.6 map→odom 静态恒等变换（安全兜底）
    #     EKF 会发布动态 map→odom，但若 EKF 状态 NaN，动态 tf 会覆盖静态 tf。
    #     此处以 tf_static 发布，确保 navsat_transform 始终能从 tf2 读到合法变换，
    #     避免因读取 NaN 变换导致自身的 /gps/odom 输出被污染。
    #     注意：LIO launch 文件中也发布此变换，但 gps_fusion 独立启动时不依赖 LIO。
    map_odom_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_odom_tf_static',
        output='screen',
        condition=IfCondition(enable_ekf),
        arguments=['0', '0', '0', '0', '0', '0',
                   ns_map_frame, ns_odom_frame],
    )

    # 3. ekf_filter_node（lifecycle）
    ekf_filter_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        condition=IfCondition(enable_ekf),
        parameters=[gps_ekf_config, {
            'use_sim_time': use_sim_time,
            'odom_frame': ns_odom_frame,
            'base_link_frame': ns_base_link_frame,
            'world_frame': ns_map_frame,
            'map_frame': ns_map_frame,
        }],
        remappings=[
            ('/odometry/filtered', fused_odom_topic),
            ('/lio_odom', lio_odom_topic),
            ('/imu/data', imu_topic),
        ],
    )

    rtk_pose_monitor_node = Node(
        package='gps_fusion',
        executable='rtk_pose_monitor.py',
        name='rtk_pose_monitor',
        output='screen',
        condition=IfCondition(enable_correction),
        parameters=[{
            'use_sim_time': use_sim_time,
            'fix_topic': '/fix_filtered',
            'rtk_topic': '/rtk_pvh',
            'use_rtk_heading': True,
            'ns': ns,
            'drift_threshold': 2.0,
            'min_correction_interval': 15.0,
            'monitor_rate': 2.0,
            'map_origin_file': map_origin_file,
        }],
        prefix=['taskset -c 0,1,2,3'],
    )

    map_origin_recorder_node = Node(
        package='gps_fusion',
        executable='map_origin_recorder.py',
        name='map_origin_recorder',
        output='screen',
        condition=IfCondition(enable_recording),
        parameters=[{
            'use_sim_time': use_sim_time,
            'fix_topic': '/fix_filtered',
            'rtk_topic': '/rtk_pvh',
            'use_rtk_heading': True,
            'auto_record': True,
            'sample_count': 10,
            'min_accuracy': 5.0,
            'odom_topic': lio_odom_topic,
        }],
    )

    # ======== Web 可视化（对齐 nav2_dog_slam 模式） ========

    # 4. 轨迹数据发布节点（纯 ROS2，不内嵌 HTTP 服务）
    trajectory_broadcaster_node = Node(
        package='gps_fusion',
        executable='trajectory_server.py',
        name='trajectory_broadcaster',
        output='screen',
        condition=IfCondition(enable_web),
        parameters=[{
            'use_sim_time': use_sim_time,
            'max_points': 10000,
            'publish_rate': 5.0,
            'frame_id': ns_odom_frame,
        }],
        remappings=[
            # trajectory_server.py 硬编码订阅 /lio_odom，此处映射到实际 LIO 话题
            ('/lio_odom', lio_odom_topic),
        ],
    )

    # 4.5 WebSocket 轨迹推送节点（供远程电脑 direct 连接）
    ws_trajectory_node = Node(
        package='gps_fusion',
        executable='trajectory_ws_server.py',
        name='trajectory_ws_server',
        output='screen',
        condition=IfCondition(enable_web),
        parameters=[{
            'use_sim_time': use_sim_time,
            'ws_port': ws_trajectory_port,
            'ws_host': '0.0.0.0',
            'full_push_interval': 5.0,
            'gps_source': gps_source,
        }],
    )

    web_script_process = ExecuteProcess(
        cmd=['taskset', '-c', '0,1,2,3', 'bash', web_script],
        output='screen',
        shell=False,
        condition=IfCondition(enable_web),
        additional_env={'WEB_PORT': web_port},
    )

    # Web 组件延迟 3 秒启动（确保 ROS2 核心先就绪）
    web_actions = [
        trajectory_broadcaster_node,
        ws_trajectory_node,
        web_script_process,
    ]
    delayed_web = TimerAction(
        period=3.0,
        actions=web_actions,
        condition=IfCondition(enable_web),
    )

    # ======== 组装 LaunchDescription ========

    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='命名空间（例如 rkbot），frame_id 自动加前缀'),
        DeclareLaunchArgument('lio_odom_topic', default_value='/Odometry',
                              description='LIO里程计话题'),
        DeclareLaunchArgument('gps_topic', default_value='/fix',
                              description='原始GPS话题'),
        DeclareLaunchArgument('imu_topic', default_value='/livox/imu',
                              description='IMU话题'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='使用仿真时间'),
        DeclareLaunchArgument('utm_zone', default_value='50',
                              description='UTM区域编号'),
        DeclareLaunchArgument('web_port', default_value='8084',
                              description='Web静态服务端口'),
        DeclareLaunchArgument('ws_trajectory_port', default_value='8765',
                              description='轨迹WebSocket直连端口（远程电脑直连用）'),
        DeclareLaunchArgument('gps_source', default_value='/fix',
                              description='GPS数据源: /fix (实际RTK) /gps/fix (测试模拟)'),
        DeclareLaunchArgument('enable_web', default_value='true',
                              description='启用Web轨迹可视化'),
        DeclareLaunchArgument('enable_correction', default_value='true',
                              description='启用 RTK AMCL 漂移纠偏 (rtk_pose_monitor)'),
        DeclareLaunchArgument('enable_ekf', default_value='true',
                              description='启用 EKF+navsat GPS融合（导航时建议关闭，避免与AMCL冲突）'),
        DeclareLaunchArgument('enable_recording', default_value='false',
                              description='启用建图原点记录 (map_origin_recorder)。首次建图时设为true，记录完成后关闭'),
        DeclareLaunchArgument('rtk_min_accuracy', default_value='0.02',
                              description='RTK模式精度门槛（m），室内/真实GPS建议0.02，仿真测试建议10.0'),
        DeclareLaunchArgument('dgps_min_accuracy', default_value='30.0',
                              description='DGPS模式精度门槛（m）'),

        # 核心融合节点（立即启动，robot_localization Humble 中为普通节点，自动运行）
        gps_preprocessor_node,
        navsat_transform_node,
        gps_tf_node,
        map_odom_tf_node,
        ekf_filter_node,
        rtk_pose_monitor_node,
        map_origin_recorder_node,

        # Web 组件（延迟 3 秒启动，run_web.sh 已内置端口冲突清理）
        delayed_web,
    ])
