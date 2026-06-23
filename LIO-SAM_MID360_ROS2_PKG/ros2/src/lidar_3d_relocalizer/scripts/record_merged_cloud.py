#!/usr/bin/env python3
"""
录制连续 N 帧点云并合并保存为 PCD 文件。

用法:
  # 默认录制 10 帧 body 系点云
  python3 record_merged_cloud.py

  # 指定参数
  python3 record_merged_cloud.py --cloud_topic /rkbot/lio/body/cloud \
      --frame_count 20 --output /tmp/my_cloud.pcd

  # 变换到 odom 系后保存（与 relocalizer 内部行为一致）
  python3 record_merged_cloud.py --transform_to_odom --odom_frame rkbot/world

依赖: pip install open3d (或使用 pcl 保存)
"""

import sys
import argparse
import time
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False

from tf2_ros import Buffer, TransformListener, TransformException


class CloudRecorder(Node):
    """订阅点云，连续录制 N 帧后合并保存"""

    def __init__(self, cloud_topic: str, frame_count: int, output_path: str,
                 transform_to_odom: bool = False,
                 odom_frame: str = "odom"):
        super().__init__("cloud_recorder")

        self.frame_count = frame_count
        self.output_path = output_path
        self.transform_to_odom = transform_to_odom
        self.odom_frame = odom_frame

        self.clouds: deque = deque()
        self.done = False

        if transform_to_odom:
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        # 使用 BEST_EFFORT 以兼容传感器数据发布者（如 LIO）
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10)
        self.sub = self.create_subscription(
            PointCloud2, cloud_topic,
            self._cloud_cb, qos)

        self.get_logger().info(
            f"CloudRecorder: topic={cloud_topic}, "
            f"frames={frame_count}, output={output_path}, "
            f"transform_to_odom={transform_to_odom}")

    def _cloud_cb(self, msg: PointCloud2):
        if self.done:
            return

        if self.transform_to_odom:
            transformed = self._transform_cloud(msg)
            if transformed is not None:
                self.clouds.append(transformed)
            else:
                self.get_logger().warn(
                    f"[{len(self.clouds)+1}/{self.frame_count}] "
                    "TF transform failed, skip frame")
                return
        else:
            self.clouds.append(msg)

        cnt = len(self.clouds)
        self.get_logger().info(
            f"[{cnt}/{self.frame_count}] "
            f"pts={msg.width * msg.height}")

        if cnt >= self.frame_count:
            self._save_merged()

    def _transform_cloud(self, msg: PointCloud2):
        """将点云从 body 系变换到 odom 系"""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.odom_frame, msg.header.frame_id,
                msg.header.stamp,
                rclpy.duration.Duration(seconds=0.1))
        except TransformException as e:
            self.get_logger().debug(f"TF error: {e}")
            return None

        # 将变换转为 numpy 矩阵
        t = tf.transform.translation
        q = tf.transform.rotation
        from scipy.spatial.transform import Rotation  # noqa

        try:
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        except ImportError:
            # 回退到手动四元数旋转
            rot = self._quat_to_matrix(q.w, q.x, q.y, q.z)

        trans = np.array([t.x, t.y, t.z])

        # 提取 XYZ 点
        points = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        if points.shape[0] == 0:
            return msg

        # 旋转 + 平移
        transformed_pts = (rot @ points.T).T + trans

        # 构造新 PointCloud2 消息
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        data = transformed_pts.astype(np.float32).tobytes()
        new_msg = PointCloud2()
        new_msg.header.frame_id = self.odom_frame
        new_msg.header.stamp = msg.header.stamp
        new_msg.height = 1
        new_msg.width = points.shape[0]
        new_msg.fields = fields
        new_msg.point_step = 12
        new_msg.row_step = 12 * points.shape[0]
        new_msg.is_bigendian = False
        new_msg.is_dense = True
        new_msg.data = data
        return new_msg

    def _quat_to_matrix(self, w, x, y, z):
        """手动四元数→旋转矩阵 (fallback)"""
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array([
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ])

    def _save_merged(self):
        """合并所有帧并保存为 PCD"""
        self.done = True
        all_points = []

        for cloud in self.clouds:
            pts = point_cloud2.read_points_numpy(cloud, field_names=("x", "y", "z"))
            if pts.shape[0] > 0:
                all_points.append(pts)

        if not all_points:
            self.get_logger().error("No valid points collected")
            return

        merged = np.vstack(all_points)
        self.get_logger().info(
            f"Merged {len(self.clouds)} frames → {merged.shape[0]} points")

        # read_points_numpy 返回 structured array（带命名字段），
        # Open3D Vector3dVector 需要纯 (N,3) float64 数组，否则会段错误
        if merged.dtype.names:
            merged = np.column_stack(
                [merged[n] for n in merged.dtype.names]).astype(np.float64)
        elif merged.dtype != np.float64:
            merged = merged.astype(np.float64)

        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(merged)
            o3d.io.write_point_cloud(self.output_path, pcd)
        else:
            # 直接用 PCD ASCII 格式写入（不依赖 open3d）
            self._write_pcd_ascii(self.output_path, merged)

        self.get_logger().info(f"Saved to {self.output_path}")

        # 打印统计信息
        centroid = np.mean(merged, axis=0)
        bbox_min = np.min(merged, axis=0)
        bbox_max = np.max(merged, axis=0)
        self.get_logger().info(
            f"Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
        self.get_logger().info(
            f"BBox min: ({bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f})")
        self.get_logger().info(
            f"BBox max: ({bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f})")

    def _write_pcd_ascii(self, path: str, points: np.ndarray):
        """写入 PCD ASCII 格式 (不依赖 open3d)"""
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {points.shape[0]}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {points.shape[0]}
DATA ascii
"""
        with open(path, 'w') as f:
            f.write(header)
            for pt in points:
                f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="录制连续 N 帧点云并合并保存为 PCD")
    parser.add_argument("--cloud_topic", default="/rkbot/lio/body/cloud",
                        help="点云话题名 (default: /rkbot/lio/body/cloud)")
    parser.add_argument("--frame_count", type=int, default=10,
                        help="录制帧数 (default: 10)")
    parser.add_argument("--output", default="/tmp/merged_cloud.pcd",
                        help="输出 PCD 文件路径 (default: /tmp/merged_cloud.pcd)")
    parser.add_argument("--transform_to_odom", action="store_true",
                        help="是否变换到 odom 系（与 relocalizer 处理一致）")
    parser.add_argument("--odom_frame", default="rkbot/world",
                        help="odom 坐标系名 (default: rkbot/world)")

    args = parser.parse_args()

    print(f"Cloud topic: {args.cloud_topic}")
    print(f"Frame count: {args.frame_count}")
    print(f"Output:      {args.output}")
    print(f"To odom:     {args.transform_to_odom}")
    if args.transform_to_odom:
        print(f"Odom frame:  {args.odom_frame}")
    print("Waiting for cloud frames... (Ctrl+C to abort)\n")

    rclpy.init(args=sys.argv)
    node = CloudRecorder(
        cloud_topic=args.cloud_topic,
        frame_count=args.frame_count,
        output_path=args.output,
        transform_to_odom=args.transform_to_odom,
        odom_frame=args.odom_frame,
    )

    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\nAborted by user")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except RuntimeError:
            pass  # 忽略重复 shutdown 错误


if __name__ == "__main__":
    main()
