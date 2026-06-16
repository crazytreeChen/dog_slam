import os
import sys
import yaml

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
    yaml_path = os.path.join(pkg_share, 'config', 'auto_initial_pose_calibrator.yaml')

    # -------- 从 yaml 加载机器人差异化配置 --------
    robot_configs = {}
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)
        params = raw['/**']['ros__parameters']
        robot_configs = params.get('robot_configs', {})

    # -------- 启动参数 --------
    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    ns = LaunchConfiguration('ns')

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=yaml_path,
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
        description='ROS namespace (rkbot=中狗ZG, 空=小狗)'
    )

    # -------- 根据 namespace 选择机器人差异化参数 --------
    # ns 为空 → 用 default 配置，节点在 namespace 下，topic 自动拼接
    # ns=rkbot → 用 rkbot 配置，节点无 namespace，直接使用绝对 topic
    effective_ns = DEFAULT_NAMESPACE.strip()
    config_key = effective_ns if effective_ns else 'default'
    if config_key not in robot_configs:
        print(f"[WARN] 未找到机器人配置 '{config_key}'，使用 default")
        config_key = 'default'

    robot_params = robot_configs.get(config_key, {})
    print(f"[INFO] 机器人配置: {config_key} → {robot_params}")

    # -------- 自动校准节点 --------
    auto_calibrator = Node(
        package='auto_initial_pose_calibrator',
        executable='auto_initial_pose_calibrator.py',
        name='auto_initial_pose_calibrator',
        namespace=ns if effective_ns != 'rkbot' else '',
        output='screen',
        parameters=[params_file, robot_params, {'use_sim_time': use_sim_time}]
    )

    ld = LaunchDescription()
    ld.add_action(declare_params_file)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_ns)
    ld.add_action(auto_calibrator)

    return ld
