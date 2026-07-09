#!/usr/bin/env python3
"""
共享 GPS 坐标转换模块

提供 RTK ↔ UTM ↔ Map 坐标系的转换工具，供 rtk_pose_monitor 和
rtk_continuous_injector 复用，保证转换逻辑与 map_gps_origin.yaml 约定一致。

坐标系约定（与 map_gps_origin.yaml 约定一致）:
    RTK lat/lon → pyproj → UTM (E=东, N=北)
    地图原点: UTM (E₀, N₀) + 朝向 θ₀ (rad, θ₀ = rtk_heading₀ - amcl_yaw₀)
    dx = E - E₀,  dy = N - N₀
    map_x   =  dx * cos(θ₀) + dy * sin(θ₀)
    map_y   = -dx * sin(θ₀) + dy * cos(θ₀)
    map_yaw = θ₀ - θ_rtk_rad   (取反号：RTK heading 顺时针正，ROS yaw 逆时针正)

⚠️ 约定说明:
    map_yaw = θ₀ - θ_rtk 隐含假设 RTK heading_deg 的角度方向与 ROS yaw 一致。
    标准真北航向是 0°=北、顺时针正；ROS yaw 是 0°=东(X轴)、逆时针正。
    该约定已在生产环境验证，本模块作为统一实现。若实测发现方向相反，
    需同时修正本模块所有相关函数。
"""

import math
import os
import yaml
from pyproj import Transformer


def make_utm_transformer(utm_zone: int) -> Transformer:
    """创建 WGS84→UTM 转换器。

    Args:
        utm_zone: UTM 区域编号，正值=北半球(326xx)，负值=南半球(327xx)
    Returns:
        Transformer (always_xy=True, 输入/输出顺序为 lon, lat)
    """
    epsg = 32600 + abs(utm_zone) if utm_zone > 0 else 32700 + abs(utm_zone)
    return Transformer.from_crs('epsg:4326', f'epsg:{epsg}', always_xy=True)


def make_wgs_transformer(utm_zone: int) -> Transformer:
    """创建 UTM→WGS84 逆转换器（用于 UTM 坐标反算经纬度）。"""
    epsg = 32600 + abs(utm_zone) if utm_zone > 0 else 32700 + abs(utm_zone)
    return Transformer.from_crs(f'epsg:{epsg}', 'epsg:4326', always_xy=True)


def latlon_to_utm(transformer: Transformer, lon: float, lat: float):
    """经纬度 → UTM (easting, northing)。

    注意 pyproj always_xy=True，参数顺序为 (lon, lat)。
    """
    e, n = transformer.transform(lon, lat)
    return e, n


def utm_to_latlon(transformer: Transformer, e: float, n: float):
    """UTM → 经纬度 (lon, lat)。"""
    lon, lat = transformer.transform(e, n)
    return lon, lat


def rtk_to_map(transformer: Transformer, lon: float, lat: float,
               origin_utm: tuple, theta0_rad: float):
    """RTK 经纬度 → 地图坐标 (map_x, map_y)。

    转换链: lat/lon → UTM → 减原点 → 旋转 θ₀
    与 rtk_to_map_anchored / rtk_to_map 统一实现。

    Args:
        transformer: WGS84→UTM 转换器
        lon, lat: RTK 经纬度
        origin_utm: (easting0, northing0, altitude0) 地图原点 UTM
        theta0_rad: 地图原点朝向（弧度，0=正北）
    Returns:
        (map_x, map_y) 机器人地图坐标
    """
    e, n = transformer.transform(lon, lat)
    e0, n0, _ = origin_utm
    dx = e - e0
    dy = n - n0
    cos_t = math.cos(theta0_rad)
    sin_t = math.sin(theta0_rad)
    map_x = dx * cos_t + dy * sin_t
    map_y = -dx * sin_t + dy * cos_t
    return map_x, map_y


def rtk_heading_to_map_yaw(heading_deg: float, theta0_rad: float) -> float:
    """RTK 真北航向 → 地图 yaw (弧度)。

    map_yaw = θ₀ - θ_rtk (rad)

    RTK heading: 0°=北, 顺时针正 → ROS yaw: 0°=东, 逆时针正
    两者正方向相反，故取反号。

    Args:
        heading_deg: RTK 真北航向（度，0=正北，顺时针正）
        theta0_rad: 地图原点朝向（弧度，θ₀ = rtk_heading₀ - amcl_yaw₀）
    Returns:
        map_yaw (弧度，已归一化到 [-π, π])
    """
    yaw = theta0_rad - math.radians(heading_deg)
    while yaw > math.pi:
        yaw -= 2.0 * math.pi
    while yaw < -math.pi:
        yaw += 2.0 * math.pi
    return yaw


def detect_rtk_quality(msg) -> str:
    """NavSatFix → RTK 质量等级字符串。

    复用 gps_preprocessor.py:307-330 的判定逻辑。

    Returns:
        'RTK_FIX' | 'RTK_FLOAT' | 'DGPS' | 'GPS'
    """
    status = msg.status.status
    if status >= 4:
        if status == 4:
            return 'RTK_FIX'
        return 'RTK_FLOAT'
    if msg.position_covariance_type > 0:
        h_var = math.sqrt(msg.position_covariance[0] + msg.position_covariance[4])
        if h_var < 0.02:
            return 'RTK_FIX'
        if h_var < 0.1:
            return 'RTK_FLOAT'
    if status == 2:
        return 'DGPS'
    return 'GPS'


def compute_horizontal_accuracy(msg) -> float:
    """计算 NavSatFix 水平精度（米）。

    无协方差时返回 inf。
    """
    if msg.position_covariance_type > 0:
        return math.sqrt(msg.position_covariance[0] + msg.position_covariance[4])
    return float('inf')


def covariance_for_quality(quality: str,
                           cov_rtk_fix: float = 0.01,
                           cov_rtk_float: float = 0.1,
                           cov_dgps: float = 1.0,
                           cov_gps: float = 5.0) -> float:
    """根据 RTK 质量等级返回位置协方差（用于 /initialpose）。"""
    return {
        'RTK_FIX': cov_rtk_fix,
        'RTK_FLOAT': cov_rtk_float,
        'DGPS': cov_dgps,
        'GPS': cov_gps,
    }.get(quality, cov_gps)


def circular_mean_heading(headings_deg) -> float:
    """航向环形平均，处理 0°/360° 环绕。

    使用 atan2(sin_sum, cos_sum) 避免 359° 和 1° 平均得 180° 的错误。

    Args:
        headings_deg: 航向角度列表（度，0=正北）
    Returns:
        平均航向（度，范围 [-180, 180]）
    """
    if not headings_deg:
        return 0.0
    sin_sum = sum(math.sin(math.radians(h)) for h in headings_deg)
    cos_sum = sum(math.cos(math.radians(h)) for h in headings_deg)
    return math.degrees(math.atan2(sin_sum, cos_sum))


def load_map_origin(yaml_path: str):
    """加载 map_gps_origin.yaml → (origin_utm, heading_rad)。

    与 gps_fusion.launch.py:_load_map_origin() 兼容。

    Args:
        yaml_path: YAML 文件路径
    Returns:
        ((easting0, northing0, altitude0), heading_rad) 或 None
    """
    if not yaml_path or not os.path.exists(yaml_path):
        return None
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        origin = data.get('map_origin', {})
        lat = origin.get('latitude')
        lon = origin.get('longitude')
        if lat is None or lon is None:
            return None
        alt = origin.get('altitude', 0.0)
        hdg_deg = origin.get('heading_deg', 0.0)

        # 需要 UTM zone 才能转换，从文件读或默认 50
        utm_zone = origin.get('utm_zone', data.get('utm_zone', 50))
        transformer = make_utm_transformer(utm_zone)
        e, n = transformer.transform(lon, lat)
        return (e, n, alt), math.radians(hdg_deg)
    except Exception:
        return None


def write_map_origin_yaml(yaml_path: str, lat: float, lon: float,
                          altitude: float, heading_deg: float,
                          utm_zone: int, utm_e: float, utm_n: float,
                          source: str = 'auto', sample_count: int = 1,
                          rtk_quality: str = 'RTK_FIX',
                          extra_note: str = ''):
    """写入 map_gps_origin.yaml（手动格式化字符串保精度）。

    沿用 map_gps_origin.yaml 的格式约定，避免 yaml.dump 精度丢失。
    新增 record_info 字段，但保持 map_origin 结构不变，确保现有读取代码兼容。

    Args:
        yaml_path: 输出文件路径
        lat, lon: 地图原点经纬度
        altitude: 高度（米）
        heading_deg: 朝向（度，0=正北）
        utm_zone: UTM 区域
        utm_e, utm_n: UTM 坐标（用于校验）
        source: 'auto' | 'manual'
        sample_count: 采集样本数
        rtk_quality: RTK 质量等级
        extra_note: 附加说明
    """
    import datetime
    now_str = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    note = f'由 gps_transform 生成 (source={source}, samples={sample_count})'
    if extra_note:
        note += f' | {extra_note}'

    yaml_content = (
        f'map_origin:\n'
        f'  latitude: {lat:.12f}\n'
        f'  longitude: {lon:.12f}\n'
        f'  altitude: {altitude:.4f}\n'
        f'  heading_deg: {heading_deg:.8f}\n'
        f'  utm_easting: {utm_e:.6f}\n'
        f'  utm_northing: {utm_n:.6f}\n'
        f'  utm_zone: {utm_zone}\n'
        f'note: "{note}"\n'
        f'record_info:\n'
        f'  source: "{source}"\n'
        f'  sample_count: {sample_count}\n'
        f'  rtk_quality: "{rtk_quality}"\n'
        f'  recorded_at: "{now_str}"\n'
    )

    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)


def rtk_to_map_anchored(transformer, lon: float, lat: float, anchor: tuple,
                        heading_deg=None):
    """RTK fix → map 位姿（已叠加 initialpose 锚点偏移）。

    anchor = (e0, n0, mx0, my0, theta0_rad)
    返回 (map_x, map_y, map_yaw_rad)；无 heading_deg 时 map_yaw 返回 None。
    锚点处 (lon0,lat0,heading0) 应返回 (mx0, my0, maw0)，保证 θ0 内部自洽。
    """
    e0, n0, mx0, my0, theta0 = anchor
    bx, by = rtk_to_map(transformer, lon, lat, (e0, n0, 0.0), theta0)
    map_x = mx0 + bx
    map_y = my0 + by
    map_yaw = rtk_heading_to_map_yaw(heading_deg, theta0) if heading_deg is not None else None
    return map_x, map_y, map_yaw


def odom_pose_delta(prev, cur):
    """两帧 Odometry 位姿差（世界系位移 + 朝向差）。

    prev, cur: (x, y, yaw_rad)。返回 (dx, dy, dyaw)，dyaw 归一化到 [-π, π]。
    用于 LIO 累计位移驱动纠偏触发，与全局 frame 无关。
    """
    dx = cur[0] - prev[0]
    dy = cur[1] - prev[1]
    dyaw = cur[2] - prev[2]
    while dyaw > math.pi:
        dyaw -= 2.0 * math.pi
    while dyaw < -math.pi:
        dyaw += 2.0 * math.pi
    return dx, dy, dyaw


def decide_reinject(moved_dist: float, turned_rad: float,
                    motion_margin: float, yaw_margin_rad: float) -> bool:
    """平滑注入触发判定：LIO 累计位移或 yaw 超阈则重发 /initialpose。"""
    return moved_dist > motion_margin or turned_rad > yaw_margin_rad


def is_rtk_heading_valid(heading_type, sol_status, heading_std,
                         max_heading_std: float) -> bool:
    """RTK 航向可用性（独立于 pos_type）。

    heading_type∈{16,17,34,50} 且 sol_status∈{0,2} 且 std≤阈值 时有效。
    即 pos_type=50 无基线、或 pos_type=34/16 双天线基线收敛时仍可有航向。
    """
    if heading_type not in (16, 17, 34, 50):
        return False
    if sol_status not in (0, 2):
        return False
    if heading_std is not None and heading_std > max_heading_std:
        return False
    return True


def pos_type_to_quality(pos_type: int) -> str:
    """pos_type → RTK 质量等级（与 rtk_pose_monitor._rtk_callback 映射一致）。

    50→RTK_FIX, 34→RTK_FLOAT, 16/17→DGPS, 其它→GPS。
    """
    if pos_type == 50:
        return 'RTK_FIX'
    if pos_type == 34:
        return 'RTK_FLOAT'
    if pos_type in (16, 17):
        return 'DGPS'
    return 'GPS'


def select_anchor_theta0(maw0_rad: float, heading_deg=None):
    """锚点 θ0 选择：有 RTK 航向→maw0+rad(heading)；无→maw0（best-effort）。

    与 rtk_to_map_anchored 配合：锚点处 map_yaw = θ0 - rad(heading) = maw0。
    """
    if heading_deg is None:
        return maw0_rad
    return maw0_rad + math.radians(heading_deg)
