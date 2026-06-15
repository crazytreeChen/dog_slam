import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# 导入全局配置
try:
    global_config_path = get_package_share_directory('global_config')
    if global_config_path not in sys.path:
        sys.path.insert(0, global_config_path)
    from global_config import (
        DEFAULT_USE_SIM_TIME_STRING,
        DEFAULT_NAMESPACE,
    )
except Exception as e:
    print(f"导入global_config失败: {e}")
    DEFAULT_USE_SIM_TIME_STRING = 'false'
    DEFAULT_NAMESPACE = ''


def generate_launch_description():
    # -------- 获取包路径 --------
    pkg_share = get_package_share_directory('auto_initial_pose_calibrator')

    # -------- 启动参数 --------
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    ns = LaunchConfiguration('ns')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_share, 'config', 'auto_initial_pose_calibrator.yaml'),
        description='Full path to auto calibration parameter yaml file'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value=DEFAULT_USE_SIM_TIME_STRING,
        description='Use simulation (Gazebo) clock'
    )

    declare_ns = DeclareLaunchArgument(
        'ns',
        default_value=DEFAULT_NAMESPACE,
        description='ROS namespace'
    )

    # -------- 自动校准节点 --------
    auto_calibrator = Node(
        package='auto_initial_pose_calibrator',
        executable='auto_initial_pose_calibrator.py',
        name='auto_initial_pose_calibrator',
        namespace=ns,
        output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time}]
    )

    ld = LaunchDescription()
    ld.add_action(declare_params_file)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_ns)
    ld.add_action(auto_calibrator)

    return ld
