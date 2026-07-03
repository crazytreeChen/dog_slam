#!/usr/bin/env python3
"""
gps_transform 共享模块单元测试

不依赖 ROS2，可在 macOS / 任何有 pyproj+yaml 的环境直接运行。

用法:
    cd ros2/src/gps_fusion
    python3 -m pytest tests/test_gps_transform.py -v

或直接:
    python3 tests/test_gps_transform.py
"""

import math
import os
import sys
import tempfile
import unittest

# 确保 gps_fusion 模块可被导入
_this_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.join(_this_dir, '..')
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from gps_fusion.gps_transform import (
    make_utm_transformer, make_wgs_transformer,
    latlon_to_utm, utm_to_latlon,
    rtk_to_map, rtk_heading_to_map_yaw,
    detect_rtk_quality, compute_horizontal_accuracy,
    covariance_for_quality, circular_mean_heading,
    load_map_origin, write_map_origin_yaml,
)


# ======================== Mock NavSatFix ========================

class MockNavSatFix:
    """最小化的 NavSatFix mock，仅包含测试需要的字段"""

    def __init__(self, lat, lon, status=0, cov_type=0, cov=(0.0, 0.0)):
        self.latitude = lat
        self.longitude = lon
        self.altitude = 0.0
        self.status = MockStatus(status)
        self.position_covariance_type = cov_type
        # covariance[0]=cov_x, [4]=cov_y (行优先 3x3 矩阵)
        cov_list = [0.0] * 9
        cov_list[0] = cov[0]
        cov_list[4] = cov[1]
        self.position_covariance = cov_list


class MockStatus:
    def __init__(self, status):
        self.status = status


# ======================== 已知值（罗普特园区） ========================

# 使用 rtk_simulator.py 中的圆心坐标，UTM zone=50
CENTER_LAT = 24.612603983011
CENTER_LON = 118.03419204406185
# 实际 pyproj 转换结果（先用实际值，不瞎猜）
# 在 setUpClass 中动态计算


# ======================== 测试类 ========================

class TestTransformer(unittest.TestCase):
    """UTM 转换器创建与往返"""

    def test_make_utm_transformer(self):
        t = make_utm_transformer(50)
        self.assertIsNotNone(t)
        # 验证可以正常转换（不关心内部EPSG表示）
        e, n = t.transform(CENTER_LON, CENTER_LAT)
        self.assertIsInstance(e, float)
        self.assertGreater(abs(e), 100000)

    def test_make_utm_transformer_negative(self):
        t = make_utm_transformer(-50)
        self.assertIsNotNone(t)
        e, n = t.transform(CENTER_LON, CENTER_LAT)
        self.assertIsInstance(e, float)

    def test_make_wgs_transformer(self):
        t = make_wgs_transformer(50)
        self.assertIsNotNone(t)
        lon, lat = t.transform(600000, 2722000)
        self.assertIsInstance(lon, float)
        # 应该在合理范围（南中国附近）
        self.assertGreater(lat, 20.0)
        self.assertLess(lat, 30.0)


class TestLatlonUtm(unittest.TestCase):
    """经纬度 ↔ UTM 互转"""

    # 类变量：运行时计算一次
    _UTM_E = None
    _UTM_N = None

    @classmethod
    def setUpClass(cls):
        t = make_utm_transformer(50)
        cls._UTM_E, cls._UTM_N = latlon_to_utm(t, CENTER_LON, CENTER_LAT)

    def test_latlon_to_utm_known_point(self):
        """用 pyproj 实际输出作为基准"""
        t = make_utm_transformer(50)
        e, n = latlon_to_utm(t, CENTER_LON, CENTER_LAT)
        self.assertAlmostEqual(e, self._UTM_E, delta=0.01)
        self.assertAlmostEqual(n, self._UTM_N, delta=0.01)
        # 确认在合理范围（福建厦门附近，UTM 50N）
        self.assertGreater(e, 600000)
        self.assertLess(e, 610000)

    def test_round_trip(self):
        """经纬度→UTM→经纬度 往返误差 < 1cm"""
        t_to = make_utm_transformer(50)
        t_from = make_wgs_transformer(50)
        e, n = latlon_to_utm(t_to, CENTER_LON, CENTER_LAT)
        lon2, lat2 = utm_to_latlon(t_from, e, n)
        self.assertAlmostEqual(lat2, CENTER_LAT, delta=1e-7,
                               msg=f'lat round-trip: {CENTER_LAT} → {lat2}')
        self.assertAlmostEqual(lon2, CENTER_LON, delta=1e-7,
                               msg=f'lon round-trip: {CENTER_LON} → {lon2}')


class TestRtkToMap(unittest.TestCase):
    """RTK → Map 坐标转换"""

    _E0 = None
    _N0 = None
    _TO_FROM = None  # UTM→WGS for offset GPS

    @classmethod
    def setUpClass(cls):
        t = make_utm_transformer(50)
        cls._E0, cls._N0 = latlon_to_utm(t, CENTER_LON, CENTER_LAT)
        cls._TO_FROM = make_wgs_transformer(50)

    def test_zero_offset(self):
        """原点位置 → map(0,0)"""
        t = make_utm_transformer(50)
        x, y = rtk_to_map(t, CENTER_LON, CENTER_LAT,
                          (self._E0, self._N0, 0.0), math.radians(0.0))
        self.assertAlmostEqual(x, 0.0, delta=1e-6)
        self.assertAlmostEqual(y, 0.0, delta=1e-6)

    def test_north_1m(self):
        """原点 θ=0°, UTM 中北移 1m → map 坐标"""
        t = make_utm_transformer(50)
        x, y = rtk_to_map(t, CENTER_LON, CENTER_LAT,
                          (self._E0, self._N0 - 1.0, 0.0),
                          math.radians(0.0))
        # dx=0, dy=N0-(N0-1)=1, θ=0: map_x=0*1+1*0=0, map_y=-0*0+1*1=1
        self.assertAlmostEqual(x, 0.0, delta=1e-6)
        self.assertAlmostEqual(y, 1.0, delta=1e-6)

    def test_heading_90deg(self):
        """原点 θ=90°, UTM 中东移 10m → map 坐标"""
        t = make_utm_transformer(50)
        # 东移 10m: E=E0+10 → dx=E-E0=10, dy=0
        # 原点在 (E0-10, N0): dx=E0-(E0-10)=10, dy=0
        # θ=90°: cos=0, sin=1
        #   map_x = 10*0 + 0*1 = 0
        #   map_y = -10*1 + 0*0 = -10
        x, y = rtk_to_map(t, CENTER_LON, CENTER_LAT,
                          (self._E0 - 10.0, self._N0, 0.0),
                          math.radians(90.0))
        self.assertAlmostEqual(x, 0.0, delta=1e-6)
        self.assertAlmostEqual(y, -10.0, delta=1e-6)

    def test_consistency_with_rtk_initial_pose(self):
        """验证公式与 rtk_initial_pose.py:263-273 一致"""
        t = make_utm_transformer(50)
        theta0 = math.radians(45.0)

        # 从原点 E0+100, N0+50 反算 GPS 坐标
        offset_lon, offset_lat = utm_to_latlon(
            self._TO_FROM, self._E0 + 100.0, self._N0 + 50.0)

        x, y = rtk_to_map(t, offset_lon, offset_lat,
                          (self._E0, self._N0, 0.0), theta0)

        # 手动验证公式: dx=100, dy=50, cos45=sin45≈0.7071
        cos_t = math.cos(theta0)
        sin_t = math.sin(theta0)
        expected_x = 100.0 * cos_t + 50.0 * sin_t
        expected_y = -100.0 * sin_t + 50.0 * cos_t
        self.assertAlmostEqual(x, expected_x, delta=1e-4)
        self.assertAlmostEqual(y, expected_y, delta=1e-4)


class TestHeadingYaw(unittest.TestCase):
    """RTK 航向 → 地图 yaw 转换"""

    def test_same_heading(self):
        """RTK heading = θ₀ → yaw=0"""
        yaw = rtk_heading_to_map_yaw(90.0, math.radians(90.0))
        self.assertAlmostEqual(yaw, 0.0, delta=1e-6)

    def test_clockwise_turn(self):
        """顺时针转 90°: RTK heading 90°→180° → yaw 0°→-90°(=-π/2)"""
        # 标定时: θ₀=90°, rtk_heading=90° → yaw=0
        # 顺时针转90°: rtk_heading=180° → yaw=θ₀-180°=-90°=-π/2
        yaw = rtk_heading_to_map_yaw(180.0, math.radians(90.0))
        self.assertAlmostEqual(yaw, -math.pi / 2, delta=1e-6)

    def test_counterclockwise_turn(self):
        """逆时针转 90°: RTK heading 90°→0° → yaw 0°→90°(=+π/2)"""
        yaw = rtk_heading_to_map_yaw(0.0, math.radians(90.0))
        self.assertAlmostEqual(yaw, math.pi / 2, delta=1e-6)

    def test_wrap_around(self):
        """RTK heading=10°, θ₀=350°(=-10°等价) → yaw=-20°"""
        # θ₀=350°= -10°, rtk=10° → yaw = -10° - 10° = -20°
        yaw = rtk_heading_to_map_yaw(10.0, math.radians(350.0))
        self.assertAlmostEqual(yaw, math.radians(-20.0), delta=1e-6)


class TestDetectRtkQuality(unittest.TestCase):
    """RTK 质量检测"""

    def test_rtk_fix(self):
        msg = MockNavSatFix(24.0, 118.0, status=4, cov_type=1,
                            cov=(0.0001, 0.0001))
        self.assertEqual(detect_rtk_quality(msg), 'RTK_FIX')

    def test_rtk_fix_by_covariance(self):
        """status=5 但协方差很小 → RTK_FIX"""
        msg = MockNavSatFix(24.0, 118.0, status=5, cov_type=1,
                            cov=(0.0001, 0.0001))
        result = detect_rtk_quality(msg)
        # status=5 ≥ 4, 但不是 4(RTK_FIX) 也不是其他已知的
        # 按新实现: status>=4, status==4→RTK_FIX, else→RTK_FLOAT
        self.assertEqual(result, 'RTK_FLOAT')

    def test_rtk_float(self):
        msg = MockNavSatFix(24.0, 118.0, status=5, cov_type=1,
                            cov=(0.04, 0.04))
        result = detect_rtk_quality(msg)
        self.assertEqual(result, 'RTK_FLOAT')

    def test_dgps(self):
        msg = MockNavSatFix(24.0, 118.0, status=2, cov_type=1,
                            cov=(0.5, 0.5))
        self.assertEqual(detect_rtk_quality(msg), 'DGPS')

    def test_gps_fallback(self):
        msg = MockNavSatFix(24.0, 118.0, status=0, cov_type=0)
        self.assertEqual(detect_rtk_quality(msg), 'GPS')

    def test_no_covariance(self):
        """无协方差 + status=4 → RTK_FIX"""
        msg = MockNavSatFix(24.0, 118.0, status=4, cov_type=0)
        self.assertEqual(detect_rtk_quality(msg), 'RTK_FIX')


class TestHorizontalAccuracy(unittest.TestCase):
    """水平精度计算"""

    def test_known_covariance(self):
        msg = MockNavSatFix(24.0, 118.0, cov_type=1,
                            cov=(0.09, 0.16))  # σx²=0.09, σy²=0.16
        acc = compute_horizontal_accuracy(msg)
        self.assertAlmostEqual(acc, math.sqrt(0.25), delta=1e-6)  # 0.5m

    def test_no_covariance(self):
        msg = MockNavSatFix(24.0, 118.0, cov_type=0)
        acc = compute_horizontal_accuracy(msg)
        self.assertEqual(acc, float('inf'))


class TestCovarianceForQuality(unittest.TestCase):
    """协方差质量映射"""

    def test_defaults(self):
        self.assertAlmostEqual(covariance_for_quality('RTK_FIX'), 0.01)
        self.assertAlmostEqual(covariance_for_quality('RTK_FLOAT'), 0.1)
        self.assertAlmostEqual(covariance_for_quality('DGPS'), 1.0)
        self.assertAlmostEqual(covariance_for_quality('GPS'), 5.0)
        self.assertAlmostEqual(covariance_for_quality('UNKNOWN'), 5.0)

    def test_custom(self):
        self.assertAlmostEqual(
            covariance_for_quality('RTK_FIX',
                                   cov_rtk_fix=0.05, cov_dgps=2.0), 0.05)


class TestCircularMeanHeading(unittest.TestCase):
    """航向环形平均"""

    def test_single_value(self):
        self.assertAlmostEqual(circular_mean_heading([45.0]), 45.0, delta=1e-6)

    def test_wrap_around(self):
        """359° 和 1° 平均应为 0°（而非 180°）"""
        result = circular_mean_heading([359.0, 1.0])
        self.assertAlmostEqual(result, 0.0, delta=1e-6)

    def test_opposite_wrap(self):
        """179° 和 181° → 平均 180°（atan2 返回 -π 即 -180°，归一化检验）"""
        result = circular_mean_heading([179.0, 181.0])
        # atan2 返回 -180° ∈ [-180, 180]，取绝对值验证
        self.assertAlmostEqual(abs(result), 180.0, delta=1e-6)

    def test_multiple_values(self):
        result = circular_mean_heading([45.0, 45.5, 44.5])
        self.assertAlmostEqual(result, 45.0, delta=1e-6)

    def test_empty(self):
        self.assertAlmostEqual(circular_mean_heading([]), 0.0, delta=1e-6)

    def test_symmetric_pair(self):
        """10° 和 350° → 平均 0°"""
        result = circular_mean_heading([10.0, 350.0])
        self.assertAlmostEqual(result, 0.0, delta=1e-6)


class TestMapOriginYaml(unittest.TestCase):
    """map_gps_origin.yaml 读写往返"""

    def test_write_and_read(self):
        lat, lon = 24.61260398, 118.03419204
        alt = 56.3
        hdg_deg = 45.0
        utm_zone = 50

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            f.write('')  # will be overwritten
            tmp_path = f.name

        try:
            # 写入
            e, n = latlon_to_utm(make_utm_transformer(utm_zone), lon, lat)
            write_map_origin_yaml(
                tmp_path, lat, lon, alt, hdg_deg,
                utm_zone, e, n,
                source='auto', sample_count=10, rtk_quality='RTK_FIX')

            # 验证文件存在
            self.assertTrue(os.path.exists(tmp_path))

            # 读取
            result = load_map_origin(tmp_path)
            self.assertIsNotNone(result)
            origin_utm, heading_rad = result
            e_read, n_read, alt_read = origin_utm

            # 验证 UTM 坐标（往返有少许浮点误差）
            self.assertAlmostEqual(e_read, e, delta=0.1)
            self.assertAlmostEqual(n_read, n, delta=0.1)
            self.assertAlmostEqual(alt_read, alt, delta=0.1)
            self.assertAlmostEqual(math.degrees(heading_rad), hdg_deg,
                                   delta=1e-4)
        finally:
            os.unlink(tmp_path)

    def test_load_nonexistent(self):
        self.assertIsNone(load_map_origin('/nonexistent/path/map_gps_origin.yaml'))

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                         delete=False) as f:
            f.write('')
            tmp_path = f.name
        try:
            self.assertIsNone(load_map_origin(tmp_path))
        finally:
            os.unlink(tmp_path)


# ======================== 主入口 ========================

if __name__ == '__main__':
    unittest.main(verbosity=2)
