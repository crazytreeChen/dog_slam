#!/usr/bin/env python3
"""lidar_3d_relocalizer 启动文件 — 支持命名空间和 LIO 算法自适应"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params_file = os.path.join(
        get_package_share_directory('lidar_3d_relocalizer'),
        'config', 'relocalizer_params.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=params_file,
            description='lidar_3d_relocalizer 参数文件路径'
        ),
        DeclareLaunchArgument(
            'pcd_map_path',
            default_value='',
            description='PCD 地图路径（覆盖参数文件中的配置）'
        ),
        DeclareLaunchArgument(
            'cloud_topic',
            default_value='/lio/body/cloud',
            description='3D 点云话题（根据 LIO 算法设置）'
        ),
        DeclareLaunchArgument(
            'odom_topic',
            default_value='/lio/odom',
            description='里程计话题（根据 LIO 算法设置）'
        ),
        DeclareLaunchArgument(
            'map_frame',
            default_value='map',
            description='地图坐标系'
        ),
        DeclareLaunchArgument(
            'odom_frame',
            default_value='odom',
            description='里程计坐标系'
        ),
        DeclareLaunchArgument(
            'base_frame',
            default_value='base_footprint',
            description='机器人基座坐标系'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='False',
            description='是否使用仿真时间'
        ),

        Node(
            package='lidar_3d_relocalizer',
            executable='lidar_3d_relocalizer_node',
            name='lidar_3d_relocalizer_node',
            output='screen',
            parameters=[
                params_file,
                {'use_sim_time': LaunchConfiguration('use_sim_time')},
                {'pcd_map_path': LaunchConfiguration('pcd_map_path')},
                {'cloud_topic': LaunchConfiguration('cloud_topic')},
                {'odom_topic': LaunchConfiguration('odom_topic')},
                {'map_frame': LaunchConfiguration('map_frame')},
                {'odom_frame': LaunchConfiguration('odom_frame')},
                {'base_frame': LaunchConfiguration('base_frame')},
            ],
        ),
    ])
