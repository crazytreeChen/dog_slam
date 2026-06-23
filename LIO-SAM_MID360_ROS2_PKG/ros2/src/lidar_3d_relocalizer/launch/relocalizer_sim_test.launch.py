#!/usr/bin/env python3
"""
lidar_3d_relocalizer 仿真测试启动文件

启动顺序：
  1. Super-LIO (gazebo 模式) — 提供 lio/body/cloud + lio/odom
  2. lidar_3d_relocalizer — KISS-Matcher 全局匹配 + GICP 精配准 → map→odom TF

⚠️ 注意：
  - 需要先启动 Gazebo 仿真（livox_garden.launch.py）
  - 需要先建图生成 PCD 文件，或指定已有 PCD 路径
  - 本文件不发布 map→odom 静态 TF，由 relocalizer 动态提供

使用方法：
  ros2 launch lidar_3d_relocalizer relocalizer_sim_test.launch.py \
      pcd_map_path:=/path/to/your_map.pcd
"""

import os
import sys
import math
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def deg_to_rad(deg):
    return str(deg * math.pi / 180.0)


def generate_launch_description():
    # ── 包路径 ──
    pkg_relocalizer = get_package_share_directory('lidar_3d_relocalizer')
    pkg_super_lio   = get_package_share_directory('super_lio')

    # ── 尝试导入 global_config ──
    try:
        global_config_path = os.path.join(
            get_package_share_directory('global_config'),
            '../../src/global_config')
        sys.path.insert(0, global_config_path)
        from global_config import DEFAULT_USE_SIM_TIME, DEFAULT_NAMESPACE
    except ImportError:
        DEFAULT_USE_SIM_TIME = True
        DEFAULT_NAMESPACE = ""

    # ── Launch 参数 ──
    pcd_map_path_arg = DeclareLaunchArgument(
        'pcd_map_path',
        default_value='/home/ywj/slam_data/pcd/test.pcd',
        description='PCD 地图文件路径'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='True',
        description='使用仿真时间'
    )

    ns_arg = DeclareLaunchArgument(
        'ns',
        default_value=DEFAULT_NAMESPACE,
        description='命名空间'
    )

    super_lio_config = os.path.join(pkg_super_lio, 'config', 'gazebo_mid360.yaml')
    relocalizer_params = os.path.join(pkg_relocalizer, 'config', 'relocalizer_sim_params.yaml')

    ns = LaunchConfiguration('ns')

    # ── 命名空间 TF 帧名 ──
    ns_map_frame = PythonExpression(["'map' if '", ns, "' == '' else str('", ns, "/map')"])
    ns_odom_frame = PythonExpression(["'odom' if '", ns, "' == '' else str('", ns, "/odom')"])
    ns_base_frame = PythonExpression(["'base_footprint' if '", ns, "' == '' else str('", ns, "/base_footprint')"])
    ns_world_frame = PythonExpression(["'world' if '", ns, "' == '' else str('", ns, "/world')"])
    ns_imu_frame = PythonExpression(["'imu' if '", ns, "' == '' else str('", ns, "/imu')"])
    ns_livox_frame = PythonExpression(["'livox_frame' if '", ns, "' == '' else str('", ns, "/livox_frame')"])
    ns_base_link_frame = PythonExpression(["'base_link' if '", ns, "' == '' else str('", ns, "/base_link')"])

    # ================================================================
    # Super-LIO (gazebo 模式)
    # ================================================================
    super_lio_node = Node(
        package='super_lio',
        executable='super_lio_node',
        name='super_lio_node',
        output='screen',
        parameters=[
            super_lio_config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'lio.output.tf_base_footprint_frame': ns_base_frame},
            {'lio.output.world_frame': ns_world_frame},
            {'lio.output.imu_frame': ns_imu_frame},
        ],
        prefix=['taskset -c 7'],
        remappings=[
            ('/lio/odom',       'lio/odom'),
            ('/lio/imu/odom',   'lio/imu/odom'),
            ('/lio/robo/odom',  'lio/robo/odom'),
            ('/lio/path',       'lio/path'),
            ('/lio/cloud_world','lio/cloud_world'),
            ('/lio/body/cloud', 'lio/body/cloud'),
        ]
    )

    # ================================================================
    # 静态 TF（不包含 map→odom，由 relocalizer 提供）
    # ================================================================
    static_odom_to_world = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_odom_to_world',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
                   ns_odom_frame, ns_world_frame],
    )

    static_world_to_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_transform_world_to_imu',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
                   ns_world_frame, ns_imu_frame],
    )

    static_imu_to_livox_frame = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='imu_to_livox_frame_tf',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0',
                   ns_imu_frame, ns_livox_frame],
    )

    static_livox_to_base_link = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='livox_frame_to_base_link_tf',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=[
            '-0.1', '0', '-0.1', '0', deg_to_rad(-30), '0',
            'mid360_robot/livox_frame/lidar', ns_base_link_frame],
    )

    # ================================================================
    # lidar_3d_relocalizer
    # 🔑 话题映射为 Super-LIO gazebo 模式的话题
    # ================================================================
    relocalizer_node = Node(
        package='lidar_3d_relocalizer',
        executable='lidar_3d_relocalizer_node',
        name='lidar_3d_relocalizer_node',
        output='screen',
        parameters=[
            relocalizer_params,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'pcd_map_path': LaunchConfiguration('pcd_map_path')},
            # 覆盖话题名，匹配 Super-LIO gazebo 模式
            {'cloud_topic': '/lio/body/cloud'},
            {'odom_topic': '/lio/odom'},
        ],
    )

    return LaunchDescription([
        ns_arg,
        pcd_map_path_arg,
        use_sim_time_arg,
        super_lio_node,
        static_odom_to_world,
        static_world_to_imu,
        static_imu_to_livox_frame,
        static_livox_to_base_link,
        relocalizer_node,
    ])
