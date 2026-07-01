#!/usr/bin/env python3
"""
GPS预处理独立启动文件 - gps_fusion 包

仅启动 GPS 预处理器，用于测试 GPS 信号接收和处理质量。

用法:
  ros2 launch gps_fusion gps_preprocessor.launch.py
  ros2 launch gps_fusion gps_preprocessor.launch.py gps_topic:=/gps/fix
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # 启动参数
    gps_topic = LaunchConfiguration('gps_topic', default='/fix')
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    utm_zone = LaunchConfiguration('utm_zone', default='50')

    # GPS预处理器
    gps_preprocessor_node = Node(
        package='gps_fusion',
        executable='gps_preprocessor.py',
        name='gps_preprocessor',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'utm_zone': utm_zone,
            'min_satellites': 4,
            'max_hdop': 2.0,
            'min_accuracy': 1.0,
            'rtk_min_accuracy': 0.02,
            'status_threshold': 0,
        }],
        remappings=[
            ('/fix', gps_topic),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('gps_topic', default_value='/fix',
                              description='原始GPS话题'),
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='使用仿真时间'),
        DeclareLaunchArgument('utm_zone', default_value='50',
                              description='UTM区域编号'),
        gps_preprocessor_node,
    ])
