#!/usr/bin/env python3
"""
服务端点云降采样节点
用于 3D Web 可视化：将 LIO 世界点云降采样后发布，减少 WebSocket 传输压力

订阅:
  - {input_topic}  (默认 /lio/cloud_world): 世界坐标系累积点云

发布:
  - {output_topic} (默认 /web/map_cloud):  降采样后的点云

参数:
  - voxel_size:    降采样体素大小 (m), 默认 0.5
  - publish_rate:  发布频率 (Hz), 默认 1.0
  - input_topic:   输入话题
  - output_topic:  输出话题
  - max_points:    最大点数限制, 默认 500000
"""

import rclpy
import numpy as np
import struct

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class PointCloudDownsampler(Node):
    def __init__(self):
        super().__init__('pointcloud_downsampler')

        # === 参数 ===
        self.declare_parameter('voxel_size', 0.5)
        self.declare_parameter('publish_rate', 1.0)
        self.declare_parameter('input_topic', '/lio/cloud_world')
        self.declare_parameter('output_topic', '/web/map_cloud')
        self.declare_parameter('max_points', 500000)
        self.declare_parameter('use_intensity', True)

        self._voxel_size = self.get_parameter('voxel_size').value
        self._max_points = self.get_parameter('max_points').value
        self._use_intensity = self.get_parameter('use_intensity').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        publish_rate = self.get_parameter('publish_rate').value

        # === 订阅 & 发布 ===
        self._sub = self.create_subscription(
            PointCloud2, input_topic, self._cloud_cb, rclpy.qos.QoSProfile(
                depth=5,
                reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
                durability=rclpy.qos.DurabilityPolicy.VOLATILE
            )
        )
        self._pub = self.create_publisher(PointCloud2, output_topic, 10)
        self._timer = self.create_timer(1.0 / max(publish_rate, 0.1), self._publish_timer)

        self._latest_msg = None

        self.get_logger().info(
            f'[Downsampler] {input_topic} -> {output_topic} '
            f'(voxel={self._voxel_size}m, rate={publish_rate}Hz, max={self._max_points})'
        )

    def _cloud_cb(self, msg: PointCloud2):
        self._latest_msg = msg

    def _publish_timer(self):
        if self._latest_msg is None:
            return

        try:
            # 解析点云，提取 x,y,z 以及可选的 intensity
            if self._use_intensity:
                try:
                    gen = pc2.read_points(
                        self._latest_msg,
                        field_names=('x', 'y', 'z', 'intensity'),
                        skip_nans=True
                    )
                    pts_raw = np.array(list(gen))  # (N, 4)
                    pts_xyz = pts_raw[:, :3]
                    pts_intensity = pts_raw[:, 3]
                except Exception:
                    # fallback: 没有 intensity 字段
                    gen = pc2.read_points(
                        self._latest_msg,
                        field_names=('x', 'y', 'z'),
                        skip_nans=True
                    )
                    pts_raw = np.array(list(gen))
                    pts_xyz = pts_raw[:, :3]
                    pts_intensity = None
            else:
                gen = pc2.read_points(
                    self._latest_msg,
                    field_names=('x', 'y', 'z'),
                    skip_nans=True
                )
                pts_raw = np.array(list(gen))
                pts_xyz = pts_raw[:, :3]
                pts_intensity = None

            if len(pts_xyz) == 0:
                return

            # 体素降采样
            voxel_indices = np.floor(pts_xyz / self._voxel_size).astype(np.int64)
            _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
            unique_idx = np.sort(unique_idx)
            pts_down = pts_xyz[unique_idx]

            # 限制最大点数
            if len(pts_down) > self._max_points:
                rand_idx = np.random.choice(len(pts_down), self._max_points, replace=False)
                rand_idx = np.sort(rand_idx)
                pts_down = pts_down[rand_idx]
                unique_idx = unique_idx[rand_idx]

            # 构建输出 PointCloud2（带 intensity 则保留）
            header = self._latest_msg.header
            header.stamp = self.get_clock().now().to_msg()

            if pts_intensity is not None:
                intensities = pts_intensity[unique_idx[:len(pts_down)]]
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
                ]
                packed = []
                for i in range(len(pts_down)):
                    packed.append(struct.pack('ffff', pts_down[i][0], pts_down[i][1], pts_down[i][2],
                                              float(intensities[i])))
                data = b''.join(packed)
                point_step = 16
            else:
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                ]
                packed = []
                for i in range(len(pts_down)):
                    packed.append(struct.pack('fff', pts_down[i][0], pts_down[i][1], pts_down[i][2]))
                data = b''.join(packed)
                point_step = 12

            out_msg = PointCloud2(
                header=header,
                height=1,
                width=len(pts_down),
                fields=fields,
                is_bigendian=False,
                point_step=point_step,
                row_step=point_step * len(pts_down),
                data=data,
                is_dense=True
            )

            self._pub.publish(out_msg)
            self.get_logger().info(
                f'Published {len(pts_down):,} points '
                f'(from {len(pts_xyz):,}, voxel={self._voxel_size}m)',
                throttle_duration_sec=5.0
            )

        except Exception as e:
            self.get_logger().error(f'Error in downsampler: {e}', throttle_duration_sec=10.0)


def main():
    rclpy.init()
    node = PointCloudDownsampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
