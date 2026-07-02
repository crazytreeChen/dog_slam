#!/usr/bin/env python3
"""
RTK 位姿监控与自动纠偏 launch 文件

启动 rtk_pose_monitor 节点：
  - 首次收到外部 /initialpose → 自动标定 RTK 原点（经纬度+朝向）
  - 之后持续对比 RTK 与当前位姿（tf map→base_footprint），超阈值时纠偏
  - 无需预标定文件，无需单独的 gps_preprocessor

用法:
    # 导航时启动（与 nav2_dog_slam lio_nav2_unified.launch.py 并行）
    ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot

    # 调整纠偏参数
    ros2 launch gps_fusion rtk_nav_bridge.launch.py \\
        ns:=rkbot drift_threshold:=1.5 min_correction_interval:=20.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('gps_fusion')
    monitor_config = os.path.join(pkg_dir, 'config', 'rtk_monitor.yaml')

    # ======== 启动参数 ========
    ns = LaunchConfiguration('ns', default='')
    rtk_topic = LaunchConfiguration('rtk_topic', default='/rtk_pvh')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    utm_zone = LaunchConfiguration('utm_zone', default='50')
    drift_threshold = LaunchConfiguration('drift_threshold', default='1.0')
    min_correction_interval = LaunchConfiguration('min_correction_interval', default='15.0')
    monitor_rate = LaunchConfiguration('monitor_rate', default='2.0')
    use_rtk_heading = LaunchConfiguration('use_rtk_heading', default='true')
    web_port = LaunchConfiguration('web_port', default='8084')
    ws_trajectory_port = LaunchConfiguration('ws_trajectory_port', default='8765')
    enable_web = LaunchConfiguration('enable_web', default='true')
    lio_odom_topic = LaunchConfiguration('lio_odom_topic', default='/Odometry')

    # 命名空间感知的 frame 名（参照 gps_fusion.launch.py 模式）
    ns_map_frame = PythonExpression(
        ["'map' if '", ns, "' == '' else str('", ns, "/map')"])
    ns_base_footprint_frame = PythonExpression(
        ["'base_footprint' if '", ns, "' == '' else str('", ns, "/base_footprint')"])
    ns_odom_frame = PythonExpression(
        ["'odom' if '", ns, "' == '' else str('", ns, "/odom')"])

    # ======== RTK 位姿监控节点（位置+航向均从 RTK 直接提取） ========
    rtk_pose_monitor_node = Node(
        package='gps_fusion',
        executable='rtk_pose_monitor.py',
        name='rtk_pose_monitor',
        output='screen',
        parameters=[
            monitor_config,
            {
                'use_sim_time': use_sim_time,
                'utm_zone': utm_zone,
                'rtk_topic': rtk_topic,
                'use_rtk_heading': use_rtk_heading,
                'ns': ns,
                'map_frame': ns_map_frame,
                'base_frame': ns_base_footprint_frame,
                'drift_threshold': drift_threshold,
                'min_correction_interval': min_correction_interval,
                'monitor_rate': monitor_rate,
            },
        ],
    )

    # ======== Web 可视化（轨迹推送 + 静态页面） ========

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
            ('/lio_odom', lio_odom_topic),
        ],
    )

    ws_trajectory_node = Node(
        package='gps_fusion',
        executable='trajectory_ws_server.py',
        name='trajectory_ws_server',
        output='screen',
        condition=IfCondition(enable_web),
        parameters=[{
            'ws_port': ws_trajectory_port,
            'ws_host': '0.0.0.0',
            'full_push_interval': 5.0,
            'gps_source': '/rtk_pvh',
        }],
    )

    web_script = os.path.join(pkg_dir, 'web', 'run_web.sh')
    web_script_process = ExecuteProcess(
        cmd=['bash', web_script],
        output='screen',
        shell=False,
        condition=IfCondition(enable_web),
        additional_env={'WEB_PORT': web_port},
    )

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

    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='命名空间（例如 rkbot），TF帧自动加前缀'),
        DeclareLaunchArgument('rtk_topic', default_value='/rtk_pvh',
                              description='RTK原始数据话题（位置+航向来源）'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='使用仿真时间'),
        DeclareLaunchArgument('utm_zone', default_value='50',
                              description='UTM区域编号'),
        DeclareLaunchArgument('drift_threshold', default_value='1.0',
                              description='漂移阈值（m），超过则纠偏'),
        DeclareLaunchArgument('min_correction_interval', default_value='15.0',
                              description='最小纠偏间隔（s）'),
        DeclareLaunchArgument('monitor_rate', default_value='2.0',
                              description='监控频率（Hz）'),
        DeclareLaunchArgument('use_rtk_heading', default_value='true',
                              description='是否使用RTK航向纠偏'),
        DeclareLaunchArgument('web_port', default_value='8084',
                              description='Web静态服务端口'),
        DeclareLaunchArgument('ws_trajectory_port', default_value='8765',
                              description='轨迹WebSocket直连端口'),
        DeclareLaunchArgument('enable_web', default_value='true',
                              description='启用Web轨迹可视化'),
        DeclareLaunchArgument('lio_odom_topic', default_value='/Odometry',
                              description='LIO里程计话题（轨迹采集用）'),

        rtk_pose_monitor_node,
        delayed_web,
    ])
