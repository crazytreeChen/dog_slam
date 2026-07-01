#!/usr/bin/env python3
"""
GPS/RTK 分步验证脚本 - gps_fusion 独立包版本

不依赖 nav2_dog_slam，通过 ros2 launch/run 独立启动 GPS 融合管道各节点。

分 3 步递进验证：
  Step 1: 仅 gps_preprocessor → 验证GPS数据接收与预处理质量
  Step 2: + navsat_transform    → 验证GPS→UTM→odom坐标转换链路
  Step 3: + ekf_filter          → 验证全链路 GPS+LIO 融合输出

前提条件:
  主系统已启动（LIO 运行中，ttyACM0 有原始GPS数据流 /fix）

用法:
  # 单步运行
  python3 gps_test_steps.py --step 1
  python3 gps_test_steps.py --step 2 --slam super_lio
  python3 gps_test_steps.py --step 3 --lio-odom /Odometry

  # 逐步交互运行（1→2→3，每步停留观察）
  python3 gps_test_steps.py --step all

  # 使用 ros2 launch 启动（推荐，更稳定）
  ros2 launch gps_fusion gps_fusion.launch.py

各步验证命令:
  Step 1:
    ros2 topic echo /fix_filtered       # 预处理后的GPS
    ros2 topic echo /fix_utm            # UTM绝对坐标
    ros2 topic echo /gps/status         # GPS可用状态

  Step 2:
    ros2 topic echo /odometry/gps       # navsat输出 (map坐标系)
    ros2 topic echo /fix_filtered       # 预处理GPS

  Step 3:
    ros2 topic echo /odometry/gps_fused  # EKF融合输出
    ros2 topic echo /odometry/gps        # navsat转换
"""

import subprocess
import signal
import sys
import time
import os
import argparse

# ========== 路径配置 ==========
_PKG_SHARE_DIR = None


def _get_pkg_share_dir():
    """获取 gps_fusion 包的 share 目录"""
    global _PKG_SHARE_DIR
    if _PKG_SHARE_DIR is None:
        try:
            result = subprocess.run(
                ['ros2', 'pkg', 'prefix', 'gps_fusion'],
                capture_output=True, text=True, timeout=5
            )
            prefix = result.stdout.strip()
            if prefix:
                _PKG_SHARE_DIR = os.path.join(prefix, 'share', 'gps_fusion')
        except Exception:
            pass
    return _PKG_SHARE_DIR


def _get_config_path(filename):
    """获取配置文件路径"""
    share_dir = _get_pkg_share_dir()
    if share_dir:
        path = os.path.join(share_dir, 'config', filename)
        if os.path.exists(path):
            return path
    # 回退到源码目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'config', filename)


GPS_EKF_CONFIG = _get_config_path('gps_ekf.yaml')
NAVSAT_CONFIG = _get_config_path('navsat_transform.yaml')

# ========== LIO 里程计话题映射 ==========
LIO_ODOM_TOPICS = {
    'fast_lio': '/Odometry',
    'point_lio': '/Odometry',
    'lio_sam': '/lio_sam/mapping/odometry',
    'super_lio': '/lio/odom',
    'super_lio_zg': '/lio/odom',
    'super_lio_gazebo': '/lio/odom',
}

# ========== 进程管理 ==========
_procs = []


def _signal_handler(sig, frame):
    print("\n[测试脚本] 收到退出信号，正在清理...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def cleanup():
    """停止所有已启动的子进程"""
    for p in _procs:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
            except Exception:
                pass
    _procs.clear()


def launch_ros_node(name, package, executable, params=None, remaps=None,
                    params_file=None, sim_time=False):
    """
    通过 ros2 run 启动一个节点。
    返回 subprocess.Popen 对象。
    """
    cmd = ['ros2', 'run', package, executable]
    ros_args = ['-r', f'__node:={name}']

    if remaps:
        for src, dst in remaps.items():
            ros_args.extend(['-r', f'{src}:={dst}'])

    if params:
        for k, v in params.items():
            if isinstance(v, bool):
                ros_args.extend(['-p', f'{k}:={str(v).lower()}'])
            else:
                ros_args.extend(['-p', f'{k}:={v}'])

    if sim_time:
        ros_args.extend(['-p', 'use_sim_time:=true'])

    if ros_args:
        cmd.extend(['--ros-args'] + ros_args)

    if params_file:
        cmd.extend(['--params-file', params_file])

    print(f'  [启动] {" ".join(cmd)}')
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _procs.append(p)
    time.sleep(1.0)
    return p


def wait_for_node(name, timeout=10.0):
    """等待节点出现在 ros2 node list 中"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True, text=True, timeout=3
            )
            if name in result.stdout:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def launch_lifecycle_manager(node_names, sim_time=False):
    """
    启动 lifecycle_manager 并等待目标节点激活。
    """
    mgr_name = 'lc_mgr_gps_test'
    node_names_str = ','.join(node_names)
    launch_ros_node(
        name=mgr_name,
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        params={
            'autostart': True,
            'node_names': node_names_str,
        },
        sim_time=sim_time,
    )
    time.sleep(3.0)

    all_ok = True
    for name in node_names:
        if wait_for_node(name, timeout=5):
            print(f'  [就绪] {name} 已激活')
        else:
            print(f'  [警告] {name} 启动超时，可能未激活')
            all_ok = False
    return all_ok


# ========== 各 Step 实现 ==========

def step1_preprocessor(sim_time=False):
    """Step 1: 仅启动 gps_preprocessor"""
    print('\n' + '=' * 60)
    print('  Step 1: GPS 预处理节点')
    print('=' * 60)
    print('  包: gps_fusion')
    print('  输入: /fix (原始GPS)')
    print('  输出: /fix_filtered, /fix_utm, /fix_odom, /gps/status')
    print()
    print('  ▸ ros2 topic echo /fix_filtered     # 预处理GPS (NavSatFix)')
    print('  ▸ ros2 topic echo /fix_utm          # UTM绝对坐标 (Odometry)')
    print('  ▸ ros2 topic echo /gps/status       # GPS可用状态 (Bool)')
    print('  ▸ 检查点: RTK精度 <0.02m, 经纬度无NaN, status正常')

    launch_ros_node(
        name='gps_preprocessor',
        package='gps_fusion',
        executable='gps_preprocessor.py',
        params={
            'min_satellites': 4,
            'max_hdop': 2.0,
            'min_accuracy': 1.0,
            'rtk_min_accuracy': 0.02,
            'status_threshold': 0,
            'utm_zone': 50,
        },
        remaps={'/fix': '/fix'},
        sim_time=sim_time,
    )

    ok = wait_for_node('gps_preprocessor')
    _print_result(ok, 'Step 1 节点运行中')
    return ok


def step2_navsat(lio_odom_topic, sim_time=False):
    """Step 2: gps_preprocessor + navsat_transform"""
    print('\n' + '=' * 60)
    print('  Step 2: GPS预处理 + NavSat 坐标转换')
    print('=' * 60)
    print(f'  LIO odom: {lio_odom_topic}')
    print(f'  GPS→map 链路: /fix → /fix_filtered → navsat → /odometry/gps')
    print()
    print(f'  注意: 此步骤中 navsat 直接使用 LIO odom 作为机器人位姿')
    print(f'  (因 EKF 尚未启动，/odometry/filtered → {lio_odom_topic})')
    print()
    print('  ▸ ros2 topic echo /odometry/gps     # GPS转换后的map坐标')
    print('  ▸ 检查点: /odometry/gps 位置与 GPS 实际位置吻合')

    # 1) gps_preprocessor（来自 gps_fusion 包）
    launch_ros_node(
        name='gps_preprocessor',
        package='gps_fusion',
        executable='gps_preprocessor.py',
        params={
            'min_satellites': 4,
            'max_hdop': 2.0,
            'min_accuracy': 1.0,
            'rtk_min_accuracy': 0.02,
            'status_threshold': 0,
            'utm_zone': 50,
        },
        remaps={'/fix': '/fix'},
        sim_time=sim_time,
    )

    # 2) navsat_transform (lifecycle, 需要 --params-file)
    launch_ros_node(
        name='navsat_transform_node',
        package='robot_localization',
        executable='navsat_transform_node',
        remaps={
            '/imu/data': '/livox/imu',
            '/gps/fix': '/fix_filtered',
            '/odometry/filtered': lio_odom_topic,  # 先用 LIO odom
        },
        params_file=NAVSAT_CONFIG,
        sim_time=sim_time,
    )

    ok = launch_lifecycle_manager(['navsat_transform_node'], sim_time=sim_time)
    _print_result(ok, 'Step 2 节点运行中')
    return ok


def step3_full_fusion(lio_odom_topic, sim_time=False):
    """Step 3: 全链路 (gps_preprocessor + navsat + ekf_filter)"""
    print('\n' + '=' * 60)
    print('  Step 3: 全链路 GPS+LIO EKF 融合')
    print('=' * 60)
    print(f'  LIO odom: {lio_odom_topic}')
    print(f'  融合链路:')
    print(f'    /fix → gps_preprocessor → /fix_filtered')
    print(f'         → navsat_transform  → /odometry/gps')
    print(f'         → ekf_filter( + {lio_odom_topic}) → /odometry/gps_fused')
    print()
    print('  ▸ ros2 topic echo /odometry/gps_fused  # EKF融合输出')
    print('  ▸ ros2 topic echo /odometry/gps         # navsat转换')
    print('  ▸ 检查点1: /odometry/gps_fused 能跟踪机器人运动')
    print('  ▸ 检查点2: 对比原始LIO odom，融合后漂移是否减小')

    # 1) gps_preprocessor（来自 gps_fusion 包）
    launch_ros_node(
        name='gps_preprocessor',
        package='gps_fusion',
        executable='gps_preprocessor.py',
        params={
            'min_satellites': 4,
            'max_hdop': 2.0,
            'min_accuracy': 1.0,
            'rtk_min_accuracy': 0.02,
            'status_threshold': 0,
            'utm_zone': 50,
        },
        remaps={'/fix': '/fix'},
        sim_time=sim_time,
    )

    # 2) navsat_transform (全融合模式: /odometry/filtered → EKF输出)
    launch_ros_node(
        name='navsat_transform_node',
        package='robot_localization',
        executable='navsat_transform_node',
        remaps={
            '/imu/data': '/livox/imu',
            '/gps/fix': '/fix_filtered',
            '/odometry/filtered': '/odometry/gps_fused',
        },
        params_file=NAVSAT_CONFIG,
        sim_time=sim_time,
    )

    # 3) ekf_filter (融合 LIO odom + GPS + IMU)
    launch_ros_node(
        name='ekf_filter_node',
        package='robot_localization',
        executable='ekf_node',
        remaps={
            '/odometry/filtered': '/odometry/gps_fused',
            '/lio_odom': lio_odom_topic,
            '/imu/data': '/livox/imu',
        },
        params_file=GPS_EKF_CONFIG,
        sim_time=sim_time,
    )

    # 4) lifecycle_manager 激活 navsat + ekf
    ok = launch_lifecycle_manager(
        ['navsat_transform_node', 'ekf_filter_node'],
        sim_time=sim_time,
    )
    _print_result(ok, 'Step 3 全链路节点运行中')
    return ok


def step_all(lio_odom_topic, sim_time=False):
    """交互式逐步执行 1→2→3"""
    steps = [
        ('Step 1: GPS预处理', lambda: step1_preprocessor(sim_time)),
        ('Step 2: NavSat坐标转换', lambda: step2_navsat(lio_odom_topic, sim_time)),
        ('Step 3: 全链路融合', lambda: step3_full_fusion(lio_odom_topic, sim_time)),
    ]

    for i, (title, func) in enumerate(steps):
        print(f'\n{"#" * 60}')
        print(f'#  {title}')
        print(f'#  按 Enter 开始此步骤，输入 q 退出')
        print(f'{"#" * 60}')
        choice = input('> ').strip().lower()
        if choice == 'q':
            print('  退出测试')
            return

        func()

        if i < len(steps) - 1:
            print('\n  按 Enter 继续下一步，输入 q 退出')
            choice = input('> ').strip().lower()
            if choice == 'q':
                cleanup()
                print('  退出测试')
                return

        cleanup()  # 清理当前步骤进程，下一步重新启动

    print('\n  [完成] 三步测试全部执行完毕')


def _print_result(ok, msg):
    if ok:
        print(f'\n  [OK] {msg}，按 Ctrl+C 停止\n')
    else:
        print(f'\n  [FAIL] {msg}（部分节点启动异常）\n')


# ========== 入口 ==========

def resolve_odom_topic(args):
    """解析 LIO 里程计话题"""
    if args.lio_odom:
        return args.lio_odom
    if args.slam:
        return LIO_ODOM_TOPICS.get(args.slam, '/Odometry')

    # 默认值
    default = '/lio/odom'
    print(f'[提示] 未指定 --slam 或 --lio-odom，默认使用 {default}')
    print(f'       可通过 --slam super_lio / --lio-odom /Odometry 指定')
    return default


def main():
    parser = argparse.ArgumentParser(
        description='GPS/RTK 分步验证脚本 - gps_fusion 独立包版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 gps_test_steps.py --step 1
  python3 gps_test_steps.py --step 2 --slam super_lio
  python3 gps_test_steps.py --step 3 --lio-odom /Odometry
  python3 gps_test_steps.py --step all --slam fast_lio --sim-time

推荐: 使用 launch 文件启动
  ros2 launch gps_fusion gps_fusion.launch.py
        ''',
    )
    parser.add_argument(
        '--step', required=True,
        choices=['1', '2', '3', 'all'],
        help='测试步骤: 1(预处理) / 2(坐标转换) / 3(全融合) / all(逐步交互)',
    )
    parser.add_argument(
        '--slam',
        choices=list(LIO_ODOM_TOPICS.keys()),
        help='SLAM算法（自动推导 odom 话题）',
    )
    parser.add_argument(
        '--lio-odom',
        help='直接指定 LIO 里程计话题，例如 /Odometry, /lio/odom',
    )
    parser.add_argument(
        '--sim-time', action='store_true',
        help='使用仿真时间 (use_sim_time:=true)',
    )

    args = parser.parse_args()
    lio_odom_topic = resolve_odom_topic(args)

    print(f'GPS分步测试 [gps_fusion 独立包版] - Step: {args.step}')
    print(f'  LIO odom 话题: {lio_odom_topic}')
    print(f'  GPS EKF 配置:  {GPS_EKF_CONFIG}')
    print(f'  NavSat 配置:   {NAVSAT_CONFIG}')
    if args.sim_time:
        print(f'  仿真时间:      启用')

    try:
        if args.step == '1':
            step1_preprocessor(sim_time=args.sim_time)
        elif args.step == '2':
            step2_navsat(lio_odom_topic, sim_time=args.sim_time)
        elif args.step == '3':
            step3_full_fusion(lio_odom_topic, sim_time=args.sim_time)
        elif args.step == 'all':
            step_all(lio_odom_topic, sim_time=args.sim_time)

        # 单步模式：阻塞等待 Ctrl+C
        if args.step != 'all' and _procs:
            print('  按 Ctrl+C 停止当前步骤...', flush=True)
            while any(p.poll() is None for p in _procs):
                time.sleep(0.5)

    except KeyboardInterrupt:
        print('\n  用户中断')
    finally:
        cleanup()
        print('  测试脚本已退出')


if __name__ == '__main__':
    main()
