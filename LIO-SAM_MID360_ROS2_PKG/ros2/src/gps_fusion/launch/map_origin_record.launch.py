#!/usr/bin/env python3
"""
建图阶段地图原点 GPS 记录 launch 文件

启动机器人建图时并行运行此 launch，机器人在地图原点 (0,0,0) 启动后，
RTK 收敛即自动记录原点 GPS（经纬度+朝向）到 map_gps_origin.yaml。

用法:
    # 建图时启动（与建图 launch 并行）
    ros2 launch gps_fusion map_origin_record.launch.py

    # 命名空间模式（多机器人）
    ros2 launch gps_fusion map_origin_record.launch.py ns:=rkbot

    # 指定 RTK 话题和输出文件
    ros2 launch gps_fusion map_origin_record.launch.py \\
        rtk_topic:=/rtk_pvh output_file:=/tmp/map_origin.yaml

手动触发记录:
    ros2 service call /gps_origin/record std_srvs/srv/Trigger
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory('gps_fusion')

    # ======== 启动参数 ========
    ns = LaunchConfiguration('ns', default='')
    gps_source = LaunchConfiguration('gps_source', default='/fix')
    rtk_topic = LaunchConfiguration('rtk_topic', default='/rtk_pvh')
    imu_topic = LaunchConfiguration('imu_topic', default='/livox/imu')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    utm_zone = LaunchConfiguration('utm_zone', default='50')
    sample_count = LaunchConfiguration('sample_count', default='10')
    min_accuracy = LaunchConfiguration('min_accuracy', default='5.0')
    rtk_min_accuracy = LaunchConfiguration('rtk_min_accuracy', default='0.02')
    dgps_min_accuracy = LaunchConfiguration('dgps_min_accuracy', default='30.0')
    output_file = LaunchConfiguration('output_file', default='')
    use_rtk_heading = LaunchConfiguration('use_rtk_heading', default='true')
    require_origin_odom = LaunchConfiguration('require_origin_odom', default='true')

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
            'rtk_min_accuracy': rtk_min_accuracy,
            'dgps_min_accuracy': dgps_min_accuracy,
            'status_threshold': 0,
        }],
    )

    # ======== 地图原点记录节点 ========
    map_origin_recorder_node = Node(
        package='gps_fusion',
        executable='map_origin_recorder.py',
        name='map_origin_recorder',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'utm_zone': utm_zone,
            'fix_topic': '/fix_filtered',
            'rtk_topic': rtk_topic,
            'use_rtk_heading': use_rtk_heading,
            'sample_count': sample_count,
            'min_accuracy': min_accuracy,
            'output_file': output_file,
            'auto_record': True,
            'require_origin_odom': require_origin_odom,
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('ns', default_value='',
                              description='命名空间（建图阶段一般不需要）'),
        DeclareLaunchArgument('gps_source', default_value='/fix',
                              description='GPS数据源: /fix (实际RTK) /gps/fix (测试)'),
        DeclareLaunchArgument('rtk_topic', default_value='/rtk_pvh',
                              description='RTK原始数据话题（获取航向）'),
        DeclareLaunchArgument('imu_topic', default_value='/livox/imu',
                              description='IMU话题（预留，当前未使用）'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='使用仿真时间'),
        DeclareLaunchArgument('utm_zone', default_value='50',
                              description='UTM区域编号'),
        DeclareLaunchArgument('sample_count', default_value='10',
                              description='自动模式采集帧数'),
        DeclareLaunchArgument('min_accuracy', default_value='5.0',
                              description='最低水平精度门槛（m）'),
        DeclareLaunchArgument('rtk_min_accuracy', default_value='0.02',
                              description='RTK模式精度门槛（m），仿真测试建议设为10.0'),
        DeclareLaunchArgument('dgps_min_accuracy', default_value='30.0',
                              description='DGPS模式精度门槛（m）'),
        DeclareLaunchArgument('output_file', default_value='',
                              description='输出YAML路径（空则用默认 config/map_gps_origin.yaml）'),
        DeclareLaunchArgument('use_rtk_heading', default_value='true',
                              description='是否使用RTK航向作为地图朝向'),
        DeclareLaunchArgument('require_origin_odom', default_value='true',
                              description='是否要求 odom 在原点附近才记录（模拟测试建议设为 false）'),

        gps_preprocessor_node,
        map_origin_recorder_node,
    ])
