"""RTK → map 变换 —— 纯函数。

封装 _outdoor_loop 中的坐标变换计算。
原代码使用预校准的 calibration_tf (translation + rotation) 将 RTK LLA 变换到地图坐标。
无 ROS 消息依赖，仅接受标量参数。
"""
import math

from .scan_utils import norm_angle


def rtk_to_map_coords(bestnav_lon_deg, bestnav_lat_deg,
                      heading_deg,
                      calibration_tf):
    """将 RTK 位姿通过预校准 TF 变换到地图坐标系。

    与原 _outdoor_loop 行为完全一致：
      mx = lon * cos(yr) - lat * sin(yr) + tx
      my = lon * sin(yr) + lat * cos(yr) + ty
      myaw = heading_rad + yr

    参数:
      bestnav_lon_deg: RTK 经度 (度)
      bestnav_lat_deg: RTK 纬度 (度)
      heading_deg:     RTK 航向角 (度)
      calibration_tf:  dict with keys:
        translation.x, translation.y, rotation.yaw
    返回:
      (mx, my, myaw) 地图坐标系下的位姿
    """
    tx = calibration_tf['translation']['x']
    ty = calibration_tf['translation']['y']
    yr = calibration_tf['rotation']['yaw']
    cx = math.cos(yr)
    sx = math.sin(yr)

    mx = bestnav_lon_deg * cx - bestnav_lat_deg * sx + tx
    my = bestnav_lon_deg * sx + bestnav_lat_deg * cx + ty
    myaw = norm_angle(math.radians(heading_deg) + yr)

    return mx, my, myaw


def build_pose_covariance(lat_std, lon_std, heading_std):
    """构建 PoseWithCovariance 的 6x6 协方差矩阵（展平为 36 元素 list）。

    与原 _outdoor_loop 行为一致：仅填充对角线 (0,7,35)。
    """
    cov = [0.0] * 36
    cov[0] = lat_std ** 2
    cov[7] = lon_std ** 2
    cov[35] = math.radians(heading_std) ** 2
    return cov
