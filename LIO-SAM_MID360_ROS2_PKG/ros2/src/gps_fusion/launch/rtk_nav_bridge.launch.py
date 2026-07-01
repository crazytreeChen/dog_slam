#!/usr/bin/env python3
"""
导航阶段 RTK-AMCL 桥接 launch 文件

导航时并行运行此 launch，rtk_pose_monitor 持续监控 AMCL 位姿与 GPS 偏差，
超阈值时补发 /initialpose 纠偏。GPS 失效时自动退化为纯 AMCL。

零侵入：不修改 nav2_dog_slam 任何配置，仅通过 /initialpose 话题交互。

用法:
    # 导航时启动（与 nav2_dog_slam lio_nav2_unified.launch.py 并行）
    ros2 launch gps_fusion rtk_nav_bridge.launch.py

    # 命名空间模式（多机器人，TF帧自动加前缀）
    ros2 launch gps_fusion rtk_nav_bridge.launch.py ns:=rkbot \\
        gps_source:=/fix rtk_topic:=/rtk_pvh

    # 调整纠偏参数
    ros2 launch gps_fusion rtk_nav_bridge.launch.py \\
        drift_threshold:=3.0 min_correction_interval:=20.0
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('gps_fusion')
    monitor_config = os.path.join(pkg_dir, 'config', 'rtk_monitor.yaml')

    # ======== 启动参数 ========
    ns = LaunchConfiguration('ns', default='')
    gps_source = LaunchConfiguration('gps_source', default='/fix')
    rtk_topic = LaunchConfiguration('rtk_topic', default='/rtk_pvh')
    imu_topic = LaunchConfiguration('imu_topic', default='/livox/imu')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    utm_zone = LaunchConfiguration('utm_zone', default='50')
    map_origin_file = LaunchConfiguration('map_origin_file', default='')
    drift_threshold = LaunchConfiguration('drift_threshold', default='2.0')
    min_correction_interval = LaunchConfiguration('min_correction_interval', default='15.0')
    monitor_rate = LaunchConfiguration('monitor_rate', default='2.0')
    use_rtk_heading = LaunchConfiguration('use_rtk_heading', default='true')

    # 命名空间感知的 frame 名（参照 gps_fusion.launch.py 模式）
    ns_map_frame = PythonExpression(
        ["'map' if '", ns, "' == '' else str('", ns, "/map')"])
    ns_base_footprint_frame = PythonExpression(
        ["'base_footprint' if '", ns, "' == '' else str('", ns, "/base_footprint')"])

    # ======== GPS 预处理器（复用，提供 /fix_filtered） ========
    gps_preprocessor_node = Node(
        package='gps_fusion',
        executable='gps_preprocessor.py',
        name='gps_preprocessor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'utm_zone': utm_zone,
            'gps_source': gps_source,
            'rtk_topic': rtk_topic,
            'min_satellites': 4,
            'max_hdop': 2.0,
            'min_accuracy': 1.0,
            'rtk_min_accuracy': 0.02,
            'status_threshold': 0,
        }],
    )

    # ======== RTK 位姿监控节点 ========
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
                'fix_topic': '/fix_filtered',
                'rtk_topic': rtk_topic,
                'use_rtk_heading': use_rtk_heading,
                'map_origin_file': map_origin_file,
                'ns': ns,
                'map_frame': ns_map_frame,
                'base_frame': ns_base_footprint_frame,
                'drift_threshold': drift_threshold,
                'min_correction_interval': min_correction_interval,
                'monitor_rate': monitor_rate,
            },
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='命名空间（例如 rkbot），TF帧自动加前缀'),
        DeclareLaunchArgument('gps_source', default_value='/fix',
                              description='GPS数据源: /fix (实际RTK) /gps/fix (测试)'),
        DeclareLaunchArgument('rtk_topic', default_value='/rtk_pvh',
                              description='RTK原始数据话题（获取航向）'),
        DeclareLaunchArgument('imu_topic', default_value='/livox/imu',
                              description='IMU话题（预留）'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='使用仿真时间'),
        DeclareLaunchArgument('utm_zone', default_value='50',
                              description='UTM区域编号'),
        DeclareLaunchArgument('map_origin_file', default_value='',
                              description='地图原点YAML路径（空则用默认 config/map_gps_origin.yaml）'),
        DeclareLaunchArgument('drift_threshold', default_value='2.0',
                              description='AMCL漂移阈值（m），超过则纠偏'),
        DeclareLaunchArgument('min_correction_interval', default_value='15.0',
                              description='最小纠偏间隔（s），避免频繁打断AMCL收敛'),
        DeclareLaunchArgument('monitor_rate', default_value='2.0',
                              description='监控频率（Hz）'),
        DeclareLaunchArgument('use_rtk_heading', default_value='true',
                              description='是否使用RTK航向纠偏'),

        gps_preprocessor_node,
        rtk_pose_monitor_node,
    ])
