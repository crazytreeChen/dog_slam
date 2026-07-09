#!/usr/bin/env python3
"""
RTK/GPS 导航纠偏 launch 文件

提供两种纠偏模式（correction_mode 参数切换，默认 continuous）：
  - continuous : rtk_continuous_injector（推荐）
        用首次 /initialpose 锚定 map 位姿，RTK 为绝对权威，LIO 仅辅助，
        连续平滑注入 /initialpose，丝滑降级，不读 AMCL、不依赖建图原点文件。
  - threshold  : rtk_pose_monitor（旧回退）
        阈值跳变纠偏，深度耦合 AMCL，仅作回退保留。

用法:
    # 导航时启动（与 nav2_dog_slam lio_nav2_unified.launch.py 并行）
    ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot
    ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot correction_mode:=threshold
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
    injector_config = os.path.join(pkg_dir, 'config', 'rtk_continuous_injector.yaml')

    # ======== 启动参数 ========
    ns = LaunchConfiguration('ns', default='')
    rtk_topic = LaunchConfiguration('rtk_topic', default='/rtk_pvh')
    gps_topic = LaunchConfiguration('gps_topic', default='/fix')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    utm_zone = LaunchConfiguration('utm_zone', default='50')
    correction_mode = LaunchConfiguration('correction_mode', default='continuous')
    web_port = LaunchConfiguration('web_port', default='8084')
    ws_trajectory_port = LaunchConfiguration('ws_trajectory_port', default='8765')
    enable_web = LaunchConfiguration('enable_web', default='true')
    lio_odom_topic = LaunchConfiguration('lio_odom_topic', default='/Odometry')

    # 命名空间感知的 frame 名
    ns_map_frame = PythonExpression(
        ["'map' if '", ns, "' == '' else str('", ns, "/map')"])
    ns_base_footprint_frame = PythonExpression(
        ["'base_footprint' if '", ns, "' == '' else str('", ns, "/base_footprint')"])
    ns_odom_frame = PythonExpression(
        ["'odom' if '", ns, "' == '' else str('", ns, "/odom')"])

    # ======== 连续注入节点（continuous 模式，默认） ========
    rtk_continuous_injector_node = Node(
        package='gps_fusion',
        executable='rtk_continuous_injector.py',
        name='rtk_continuous_injector',
        namespace=ns,
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", correction_mode, "' == 'continuous'"])),
        parameters=[
            injector_config,
            {
                'use_sim_time': use_sim_time,
                'utm_zone': utm_zone,
                'rtk_topic': rtk_topic,
                'gps_topic': gps_topic,
                'lio_odom_topic': lio_odom_topic,
                'map_frame': ns_map_frame,
                'base_frame': ns_base_footprint_frame,
            },
        ],
    )

    # ======== 旧阈值纠偏节点（threshold 模式回退） ========
    rtk_pose_monitor_node = Node(
        package='gps_fusion',
        executable='rtk_pose_monitor.py',
        name='rtk_pose_monitor',
        output='screen',
        condition=IfCondition(
            PythonExpression(["'", correction_mode, "' == 'threshold'"])),
        parameters=[
            monitor_config,
            {
                'use_sim_time': use_sim_time,
                'utm_zone': utm_zone,
                'rtk_topic': rtk_topic,
                'gps_topic': gps_topic,
                'lio_odom_topic': lio_odom_topic,
                'ns': ns,
                'map_frame': ns_map_frame,
                'base_frame': ns_base_footprint_frame,
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
        DeclareLaunchArgument('gps_topic', default_value='/fix',
                              description='普通GPS fallback话题'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='使用仿真时间'),
        DeclareLaunchArgument('utm_zone', default_value='50',
                              description='UTM区域编号'),
        DeclareLaunchArgument('correction_mode', default_value='continuous',
                              description='纠偏模式: continuous(默认) | threshold(旧回退)'),
        DeclareLaunchArgument('web_port', default_value='8084',
                              description='Web静态服务端口'),
        DeclareLaunchArgument('ws_trajectory_port', default_value='8765',
                              description='轨迹WebSocket直连端口'),
        DeclareLaunchArgument('enable_web', default_value='true',
                              description='启用Web轨迹可视化'),
        DeclareLaunchArgument('lio_odom_topic', default_value='/Odometry',
                              description='LIO里程计话题（轨迹采集/死推算用）'),

        rtk_continuous_injector_node,
        rtk_pose_monitor_node,
        delayed_web,
    ])
