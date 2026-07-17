#!/usr/bin/env python3
"""PCD 降采样工具：将大点云降采样到浏览器可加载的大小"""
import open3d as o3d
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("用法: python3 convert_pcd.py <输入.pcd> [输出.pcd] [分辨率]")
        print("示例: python3 convert_pcd.py map.pcd map_clean.pcd 0.2")
        print("      默认分辨率 0.2m，原始 500MB 降到约 50-80MB")
        sys.exit(1)

    input_pcd = sys.argv[1]
    output_pcd = sys.argv[2] if len(sys.argv) > 2 else input_pcd.replace('.pcd', '_clean.pcd')
    resolution = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

    print(f'加载: {input_pcd}')
    pcd = o3d.io.read_point_cloud(input_pcd)
    if len(pcd.points) == 0:
        print('错误: 没有读到点')
        sys.exit(1)

    print(f'原始点数: {len(pcd.points):,}')
    pcd = pcd.voxel_down_sample(resolution)
    print(f'降采样后 ({resolution}m): {len(pcd.points):,} 点')

    o3d.io.write_point_cloud(output_pcd, pcd, write_ascii=True)
    size = os.path.getsize(output_pcd) / 1024 / 1024
    print(f'已保存: {output_pcd} ({size:.1f}MB)')

if __name__ == '__main__':
    main()