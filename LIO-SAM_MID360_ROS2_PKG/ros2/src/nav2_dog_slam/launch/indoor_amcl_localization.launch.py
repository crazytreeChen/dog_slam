import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# 导入全局配置
try:
    global_config_path = get_package_share_directory('global_config')
    if global_config_path not in sys.path:
        sys.path.insert(0, global_config_path)
    from global_config import (
        NAV2_DEFAULT_MAP_FILE,
        DEFAULT_USE_SIM_TIME_STRING,
        DEFAULT_NAMESPACE,
    )
except Exception as e:
    print(f"导入global_config失败: {e}")
    NAV2_DEFAULT_MAP_FILE = '/home/ztl/slam_data/grid_map/map.yaml'
    DEFAULT_USE_SIM_TIME_STRING = 'false'
    DEFAULT_NAMESPACE = ''


def generate_launch_description():
    # -------- 获取包路径 --------
    pkg_share = get_package_share_directory('nav2_dog_slam')

    # -------- 启动参数 --------
    map_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')

    declare_map = DeclareLaunchArgument(
        'map',
        default_value=NAV2_DEFAULT_MAP_FILE,
        description='Full path to map yaml file'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'amcl_indoor.yaml'),
        description='Full path to AMCL parameter yaml file'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value=DEFAULT_USE_SIM_TIME_STRING,
        description='Use simulation (Gazebo) clock'
    )

    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Launch RViz2 for manual 2D Pose Estimate'
    )

    # -------- TF: base_link -> laser --------
    base_to_laser = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=[
            '0.20', '0.0', '0.15',
            '0.0', '0.0', '0.0',
            'base_link', 'laser'
        ]
    )

    # -------- map_server --------
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'yaml_filename': map_file},
            {'frame_id': 'map'}
        ]
    )

    # -------- AMCL --------
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}],
        remappings=[
            ('scan', '/scan'),
            ('initialpose', '/initialpose')
        ]
    )

    # -------- lifecycle_manager (map_server + amcl) --------
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server', 'amcl']
        }]
    )

    # -------- RViz2 (可选，用于手动 2D Pose Estimate) --------
    rviz_cfg = os.path.join(pkg_share, 'config', 'localization.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(use_rviz)
    )

    # -------- 定时启动：确保节点按依赖顺序启动 --------
    ld = LaunchDescription()
    ld.add_action(declare_map)
    ld.add_action(declare_params_file)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_use_rviz)
    ld.add_action(base_to_laser)
    ld.add_action(map_server)
    ld.add_action(TimerAction(period=0.5, actions=[amcl]))
    ld.add_action(TimerAction(period=1.0, actions=[lifecycle_manager]))
    ld.add_action(TimerAction(period=1.5, actions=[rviz]))

    return ld
