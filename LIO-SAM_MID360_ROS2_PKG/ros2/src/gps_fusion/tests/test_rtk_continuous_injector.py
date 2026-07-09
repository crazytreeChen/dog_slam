#!/usr/bin/env python3
"""
rtk_continuous_injector 核心逻辑单元测试。

设计：节点的坐标/降级/触发数学已下沉到 gps_fusion.gps_transform 的纯函数，
可在无 ROS2 的 macOS 环境直接验证（pyproj 必需）。节点级行为测试（rclpy）在
ROS2/Linux 环境下自动运行，macOS 上跳过。

用法:
    cd ros2/src/gps_fusion
    python3 -m pytest tests/test_rtk_continuous_injector.py -v
"""

import math
import os
import sys
import unittest

# 确保 gps_fusion 模块可被导入
_this_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.join(_this_dir, '..')
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from gps_fusion.gps_transform import (
    make_utm_transformer, make_wgs_transformer, latlon_to_utm, utm_to_latlon,
    rtk_to_map_anchored, select_anchor_theta0,
    odom_pose_delta, decide_reinject,
    is_rtk_heading_valid, pos_type_to_quality,
)

# ROS2 环境下才导入节点，macOS 跳过
try:
    import rclpy  # noqa: F401
    from gps_fusion.rtk_continuous_injector import (
        RtkContinuousInjector, WAIT_ANCHOR,
    )
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False


# 已知原点（厦门软件园附近，UTM zone=50），与 test_gps_transform.py 一致
CENTER_LAT = 24.612603983011
CENTER_LON = 118.03419204406185
UTM_ZONE = 50


class TestRtkToMapAnchored(unittest.TestCase):
    """rtk_to_map_anchored：锚点叠加 initialpose 偏移。"""

    def setUp(self):
        self.t = make_utm_transformer(UTM_ZONE)
        self.e0, self.n0 = latlon_to_utm(self.t, CENTER_LON, CENTER_LAT)

    def _anchor(self, mx0, my0, maw0_deg, h0_deg):
        theta0 = select_anchor_theta0(math.radians(maw0_deg), h0_deg)
        return (self.e0, self.n0, mx0, my0, theta0)

    def test_anchor_self_consistent_with_heading(self):
        """锚点处 (lon0,lat0,h0) → (mx0,my0,maw0)，θ0 内部自洽。"""
        anchor = self._anchor(3.0, 4.0, 30.0, 90.0)
        x, y, yaw = rtk_to_map_anchored(
            self.t, CENTER_LON, CENTER_LAT, anchor, 90.0)
        self.assertAlmostEqual(x, 3.0, delta=1e-6)
        self.assertAlmostEqual(y, 4.0, delta=1e-6)
        self.assertAlmostEqual(math.degrees(yaw), 30.0, delta=1e-6)

    def test_anchor_no_heading_returns_none_yaw(self):
        """无航向 → map_yaw 为 None，节点改发位置/沿用上次 yaw。"""
        anchor = self._anchor(3.0, 4.0, 30.0, None)
        x, y, yaw = rtk_to_map_anchored(
            self.t, CENTER_LON, CENTER_LAT, anchor)
        self.assertAlmostEqual(x, 3.0, delta=1e-6)
        self.assertAlmostEqual(y, 4.0, delta=1e-6)
        self.assertIsNone(yaw)

    def test_move_east_1m(self):
        """真北东移 1m（UTM E+1）相对锚点 → map 偏移正确。"""
        anchor = self._anchor(0.0, 0.0, 0.0, 0.0)
        lon1, lat1 = utm_to_latlon(
            make_wgs_transformer(UTM_ZONE), self.e0 + 1.0, self.n0)
        x, y, _ = rtk_to_map_anchored(self.t, lon1, lat1, anchor, 0.0)
        # θ0=0: map_x = dx*1 + dy*0 = 1, map_y = -dx*0 + dy*1 = 0
        self.assertAlmostEqual(x, 1.0, delta=1e-6)
        self.assertAlmostEqual(y, 0.0, delta=1e-6)


class TestSelectAnchorTheta0(unittest.TestCase):
    """锚点 θ0 选择：有航向→maw0+rad(h0)，无→maw0。"""

    def test_with_heading(self):
        theta0 = select_anchor_theta0(math.radians(30.0), 90.0)
        self.assertAlmostEqual(math.degrees(theta0), 120.0, delta=1e-9)

    def test_without_heading(self):
        theta0 = select_anchor_theta0(math.radians(30.0), None)
        self.assertAlmostEqual(math.degrees(theta0), 30.0, delta=1e-9)


class TestOdomPoseDelta(unittest.TestCase):
    """LIO 两帧位姿差：位移 + 朝向差（含环绕归一化）。"""

    def test_simple(self):
        dx, dy, dyaw = odom_pose_delta((0.0, 0.0, 0.0), (1.0, 2.0, 0.5))
        self.assertAlmostEqual(dx, 1.0)
        self.assertAlmostEqual(dy, 2.0)
        self.assertAlmostEqual(dyaw, 0.5)

    def test_yaw_wrap(self):
        # 从 3.0 到 -3.0 rad，未归一化差 -6.0，应归一化为 +0.283...
        dx, dy, dyaw = odom_pose_delta((0.0, 0.0, 3.0), (0.0, 0.0, -3.0))
        self.assertAlmostEqual(dyaw, 2.0 * math.pi - 6.0, delta=1e-9)

    def test_no_motion(self):
        d = odom_pose_delta((1.0, 1.0, 0.1), (1.0, 1.0, 0.1))
        self.assertAlmostEqual(d[0], 0.0)
        self.assertAlmostEqual(d[1], 0.0)
        self.assertAlmostEqual(d[2], 0.0)


class TestDecideReinject(unittest.TestCase):
    """平滑注入触发判定：严格大于阈值才重发。"""

    def test_no_motion(self):
        self.assertFalse(decide_reinject(0.0, 0.0, 0.3, math.radians(5.0)))

    def test_motion_exceed(self):
        self.assertTrue(decide_reinject(0.31, 0.0, 0.3, math.radians(5.0)))

    def test_motion_equal_boundary(self):
        # 等于阈值不算超（严格 >）
        self.assertFalse(decide_reinject(0.3, 0.0, 0.3, math.radians(5.0)))

    def test_turn_exceed(self):
        self.assertTrue(
            decide_reinject(0.0, math.radians(6.0), 0.3, math.radians(5.0)))


class TestIsRtkHeadingValid(unittest.TestCase):
    """航向可用性独立于 pos_type：heading_type/sol_status/std 组合判定。

    覆盖降级关键场景：
    - pos_type=50 无基线但 heading_type=50 → 有航向（双天线）
    - pos_type=34/16 且 heading_type 无效 → 无航向（同纯 GPS）
    - pos_type=34/16 但 heading_type=50 → 仍有航向
    """

    def test_valid_rtk_fix_with_baseline(self):
        self.assertTrue(is_rtk_heading_valid(50, 0, 1.0, 5.0))
        self.assertTrue(is_rtk_heading_valid(34, 2, 1.0, 5.0))

    def test_bad_heading_type(self):
        # pos_type=34 但 heading_type 不在 {16,17,34,50} → 无航向
        self.assertFalse(is_rtk_heading_valid(99, 0, 1.0, 5.0))
        self.assertFalse(is_rtk_heading_valid(0, 0, 1.0, 5.0))

    def test_bad_sol_status(self):
        self.assertFalse(is_rtk_heading_valid(50, 1, 1.0, 5.0))

    def test_heading_std_exceed(self):
        self.assertFalse(is_rtk_heading_valid(50, 0, 9.0, 5.0))

    def test_float_without_baseline_no_heading(self):
        # pos_type=34 浮点解但 heading_type=None/0 → 无航向
        self.assertFalse(is_rtk_heading_valid(0, 0, 0.0, 5.0))


class TestPosTypeToQuality(unittest.TestCase):
    """pos_type → RTK 质量等级映射（与 rtk_pose_monitor 一致）。"""

    def test_fix(self):
        self.assertEqual(pos_type_to_quality(50), 'RTK_FIX')

    def test_float(self):
        self.assertEqual(pos_type_to_quality(34), 'RTK_FLOAT')

    def test_dgps(self):
        self.assertEqual(pos_type_to_quality(16), 'DGPS')
        self.assertEqual(pos_type_to_quality(17), 'DGPS')

    def test_gps_fallback(self):
        self.assertEqual(pos_type_to_quality(1), 'GPS')
        self.assertEqual(pos_type_to_quality(0), 'GPS')


@unittest.skipUnless(HAS_RCLPY, 'rclpy 不可用（需 ROS2/Linux 环境）')
class TestNodeSmoke(unittest.TestCase):
    """节点级冒烟测试：仅在 ROS2/Linux 环境运行。"""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_init_wait_anchor(self):
        node = RtkContinuousInjector()
        try:
            self.assertEqual(node._state, WAIT_ANCHOR)
            self.assertIsNone(node._anchor)
            self.assertTrue(hasattr(node, '_inject_timer'))
        finally:
            node.destroy_node()


if __name__ == '__main__':
    unittest.main(verbosity=2)
