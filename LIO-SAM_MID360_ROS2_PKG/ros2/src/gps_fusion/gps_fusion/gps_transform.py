#!/usr/bin/env python3
"""
共享 GPS 坐标转换模块

提供 RTK ↔ UTM ↔ Map 坐标系的转换工具，供 map_origin_recorder 和
rtk_pose_monitor 复用，保证转换逻辑与现有 rtk_initial_pose.py / calibrate_map_origin.py
完全一致。

坐标系约定（与 rtk_initial_pose.py:263-286 保持一致）:
    RTK lat/lon → pyproj → UTM (E=东, N=北)
    地图原点: UTM (E₀, N₀) + 朝向 θ₀ (heading_deg, 0=正北)
    dx = E - E₀,  dy = N - N₀
    map_x   =  dx * cos(θ₀) + dy * sin(θ₀)
    map_y   = -dx * sin(θ₀) + dy * cos(θ₀)
    map_yaw = θ_rtk - θ₀

⚠️ 待验证确认点:
    map_yaw = θ_rtk - θ₀ 隐含假设 RTK heading_deg 的角度方向与 ROS yaw 一致。
    标准真北航向是 0°=北、顺时针正；ROS yaw 是 0°=东(X轴)、逆时针正。
    现有 rtk_initial_pose.py 和 calibrate_map_origin.py 都用此假设且生产验证通过，
    本模块保持一致。若实测发现方向相反，需同时修正所有三处。
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
    复用 rtk_initial_pose.py:263-273 的逻辑。

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

    复用 rtk_initial_pose.py:280 的逻辑: map_yaw = θ_rtk - θ₀

    ⚠️ 假设 RTK heading_deg 与 ROS yaw 方向一致，见模块文档警告。

    Args:
        heading_deg: RTK 真北航向（度，0=正北）
        theta0_rad: 地图原点朝向（弧度）
    Returns:
        map_yaw (弧度)
    """
    return math.radians(heading_deg) - theta0_rad


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

    与 gps_fusion.launch.py:_load_map_origin() 和
    rtk_initial_pose.py:_resolve_map_origin() 兼容。

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

    沿用 calibrate_map_origin.py:336-346 的格式，避免 yaml.dump 精度丢失。
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

    note = f'由 map_origin_recorder.py 生成 (source={source}, samples={sample_count})'
    if extra_note:
        note += f' | {extra_note}'

    yaml_content = (
        f'map_origin:\n'
        f'  latitude: {lat:.8f}\n'
        f'  longitude: {lon:.8f}\n'
        f'  altitude: {altitude:.1f}\n'
        f'  heading_deg: {heading_deg:.4f}\n'
        f'  utm_easting: {utm_e:.3f}\n'
        f'  utm_northing: {utm_n:.3f}\n'
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
