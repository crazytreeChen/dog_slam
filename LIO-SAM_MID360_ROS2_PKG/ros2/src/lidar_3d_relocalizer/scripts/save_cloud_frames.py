#!/usr/bin/env python3
"""保存连续 N 帧点云为 PCD 文件"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
import struct, os, sys, time

def pointcloud2_to_array(msg):
    """从 PointCloud2 提取 xyz 点"""
    fmt = '<fff'
    n_points = msg.width * msg.height
    step = msg.point_step
    data = msg.data
    points = []
    for i in range(n_points):
        offset = i * step
        x, y, z = struct.unpack_from(fmt, data, offset)
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
            points.append((x, y, z))
    return np.array(points) if points else np.empty((0, 3))

class CloudSaver(Node):
    def __init__(self, num_frames=10, output_dir='/tmp/cloud_frames'):
        super().__init__('cloud_saver')
        self.num_frames = num_frames
        self.output_dir = output_dir
        self.frames = []
        self.start_time = None
        os.makedirs(output_dir, exist_ok=True)

        self.sub = self.create_subscription(
            PointCloud2, '/rkbot/lio/body/cloud', self.cb, 10)
        self.get_logger().info(f'等待点云数据... 目标: {num_frames} 帧')

    def cb(self, msg):
        if len(self.frames) >= self.num_frames:
            return
        if self.start_time is None:
            self.start_time = self.get_clock().now()
            self.get_logger().info('开始采集...')

        pts = pointcloud2_to_array(msg)
        if len(pts) == 0:
            return

        idx = len(self.frames)
        filename = os.path.join(self.output_dir, f'frame_{idx:04d}.pcd')
        with open(filename, 'w') as f:
            f.write(f'VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\n'
                    f'COUNT 1 1 1\nWIDTH {len(pts)}\nHEIGHT 1\n'
                    f'VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\nDATA ascii\n')
            for p in pts:
                f.write(f'{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n')

        self.frames.append(filename)
        self.get_logger().info(f'帧 {idx+1}/{self.num_frames}: {len(pts)} 点 → {filename}')

        if len(self.frames) >= self.num_frames:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            self.get_logger().info(f'采集完成! {self.num_frames} 帧, 耗时 {elapsed:.1f}s')
            self.get_logger().info(f'文件保存在: {self.output_dir}')
            rclpy.shutdown()

def main():
    rclpy.init()
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    outdir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/cloud_frames'
    node = CloudSaver(num, outdir)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
