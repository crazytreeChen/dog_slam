#!/usr/bin/env python3
"""
离线 PCD 查看服务 - 无需 ROS2 环境

在本地直接读取 PCD 文件，通过 HTTP 提供点云数据给浏览器 3D 查看器。
支持同时加载 TLS + GBL 两个图层，通过复选框切换可见性。

依赖:
    pip install open3d numpy

使用:
    # 单文件
    python3 serve_offline_pcd.py /path/to/map.pcd
    python3 serve_offline_pcd.py /path/to/map.pcd --port 8083 --voxel 0.1 --max-points 200000

    # TLS + GBL 双图层
    python3 serve_offline_pcd.py --tls-pcd /path/to/tls.pcd --gbl-pcd /path/to/gbl.pcd

然后浏览器打开: http://localhost:8083/3d_offline_viewer.html
"""

import http.server
import json
import os
import struct
import sys
import time
from urllib.parse import urlparse, parse_qs

import numpy as np


class OfflinePCDHandler(http.server.SimpleHTTPRequestHandler):
    """自定义 HTTP Handler：静态文件 + /points 二进制点云数据端点（支持 TLS/GBL 双图层）"""

    # 类级变量，由 serve_pcd() 设置
    pointcloud_binary: bytes = b""
    point_count: int = 0
    web_dir: str = ""
    stats: dict = {}

    # TLS 图层
    tls_binary: bytes = b""
    tls_count: int = 0
    tls_stats: dict = {}

    # GBL 图层
    gbl_binary: bytes = b""
    gbl_count: int = 0
    gbl_stats: dict = {}

    # 双图层模式标记
    dual_mode: bool = False

    def do_GET(self):
        parsed = urlparse(self.path)

        # /points - 返回降采样后的点云（raw float32 binary: x,y,z,r,g,b 循环）
        if parsed.path == "/points":
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(OfflinePCDHandler.pointcloud_binary)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Point-Count", str(OfflinePCDHandler.point_count))
            self.end_headers()
            self.wfile.write(OfflinePCDHandler.pointcloud_binary)
            return

        # /points/tls - TLS 图层
        if parsed.path == "/points/tls":
            self._serve_pointcloud(
                OfflinePCDHandler.tls_binary, OfflinePCDHandler.tls_count)
            return

        # /points/gbl - GBL 图层
        if parsed.path == "/points/gbl":
            self._serve_pointcloud(
                OfflinePCDHandler.gbl_binary, OfflinePCDHandler.gbl_count)
            return

        # /stats - 统计信息
        if parsed.path == "/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            result = {
                **OfflinePCDHandler.stats,
                "dual_mode": OfflinePCDHandler.dual_mode,
                "tls_available": bool(OfflinePCDHandler.tls_binary),
                "gbl_available": bool(OfflinePCDHandler.gbl_binary),
            }
            self.wfile.write(json.dumps(result).encode())
            return

        # /stats/tls - TLS 单独统计
        if parsed.path == "/stats/tls":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(OfflinePCDHandler.tls_stats).encode())
            return

        # /stats/gbl - GBL 单独统计
        if parsed.path == "/stats/gbl":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(OfflinePCDHandler.gbl_stats).encode())
            return

        # 静态文件
        return super().do_GET()

    def _serve_pointcloud(self, data: bytes, count: int):
        """发送点云二进制数据"""
        if not data:
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": "no data"}).encode())
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Point-Count", str(count))
        self.end_headers()
        self.wfile.write(data)


def read_pcd(filepath: str):
    """读取 PCD 文件，返回 (points_xyz, points_rgb) 或 (points_xyz, None)"""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(filepath)
        if not pcd.has_points():
            raise ValueError("PCD 文件中没有点")
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)  # 0-1
        else:
            colors = None
        return pts, colors
    except ImportError:
        # 回退：手动解析 ASCII PCD
        return _parse_pcd_manual(filepath)


def _parse_pcd_manual(filepath: str):
    """手动解析 PCD 文件（ASCII 格式）"""
    print("[*] open3d 不可用，使用手动 PCD 解析（仅支持 ASCII 格式）")
    with open(filepath, "rb") as f:
        header_lines = []
        for _ in range(50):
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                break

    header_text = "\n".join(header_lines)
    data_type = "ascii"
    fields = []
    width = 1

    for line in header_lines:
        if line.startswith("FIELDS"):
            fields = line.split()[1:]
        elif line.startswith("WIDTH"):
            width = int(line.split()[1])
        elif line.startswith("DATA"):
            data_type = line.split()[1]

    if data_type != "ascii":
        raise RuntimeError(f"手动解析仅支持 ASCII PCD，当前为 {data_type}。请安装 open3d: pip install open3d")

    has_xyz = all(f in fields for f in ["x", "y", "z"])
    has_rgb = "rgb" in fields

    if not has_xyz:
        # 尝试以纯坐标形式读取（三列数值）
        data = np.loadtxt(filepath, skiprows=len(header_lines), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        pts = data[:, :3].astype(np.float32)
        colors = data[:, 3:6].astype(np.float32) / 255.0 if data.shape[1] >= 6 else None
        return pts, colors

    # 解析结构化字段
    field_indices = {f: i for i, f in enumerate(fields)}
    xi, yi, zi = field_indices["x"], field_indices["y"], field_indices["z"]
    ri = field_indices.get("rgb", field_indices.get("rgba", None))

    data = np.loadtxt(filepath, skiprows=len(header_lines), dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, len(fields))

    pts = data[:, [xi, yi, zi]].astype(np.float32)

    if ri is not None:
        rgb_packed = data[:, ri].astype(np.uint32)
        r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_packed >> 8) & 0xFF).astype(np.float32) / 255.0
        b = (rgb_packed & 0xFF).astype(np.float32) / 255.0
        colors = np.stack([r, g, b], axis=1)
    else:
        colors = None

    return pts, colors


def voxel_downsample(pts, colors, voxel_size):
    """体素降采样（优先使用 open3d C++ 实现取重心，回退到均匀随机采样）"""
    if voxel_size <= 0 or len(pts) == 0:
        return pts, colors

    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_ds = pcd.voxel_down_sample(voxel_size)
        pts_ds = np.asarray(pcd_ds.points, dtype=np.float32)
        colors_ds = np.asarray(pcd_ds.colors, dtype=np.float32) if colors is not None else None
        return pts_ds, colors_ds
    except ImportError:
        pass

    # 回退：numpy 体素网格取重心（每 voxel_size 一格，格内所有点的重心）
    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)

    # 将 3D 体素索引编码为 1D key（假定每个维度范围不会太大）
    min_idx = voxel_idx.min(axis=0)
    voxel_idx_shifted = voxel_idx - min_idx
    max_shifted = voxel_idx_shifted.max(axis=0)
    str_z = int(max_shifted[0]) + 1
    str_y = int(max_shifted[1]) + 1
    key = (voxel_idx_shifted[:, 0].astype(np.int64) * str_y +
           voxel_idx_shifted[:, 1].astype(np.int64)) * str_z + voxel_idx_shifted[:, 2].astype(np.int64)

    # unique keys + 反向索引
    uniq_keys, inverse = np.unique(key, return_inverse=True)
    n_voxels = len(uniq_keys)

    if n_voxels >= len(pts):
        return pts, colors  # 已经够稀疏，无需降采样

    # 对每个体素累加 pts，取重心
    counts = np.bincount(inverse, minlength=n_voxels)
    sum_pts = np.zeros((n_voxels, 3), dtype=np.float64)
    for d in range(3):
        sum_pts[:, d] = np.bincount(inverse, weights=pts[:, d].astype(np.float64), minlength=n_voxels)
    pts_ds = (sum_pts / counts[:, None]).astype(np.float32)

    if colors is not None:
        sum_colors = np.zeros((n_voxels, 3), dtype=np.float64)
        for d in range(3):
            sum_colors[:, d] = np.bincount(inverse, weights=colors[:, d].astype(np.float64), minlength=n_voxels)
        colors_ds = (sum_colors / counts[:, None]).astype(np.float32)
    else:
        colors_ds = None

    return pts_ds, colors_ds


def build_buffer(pts, colors, color_scheme="jet"):
    """将点云构建为 binary buffer: 每点 6 个 float32 (x,y,z,r,g,b)"""
    n = pts.shape[0]
    buffer = np.zeros((n, 6), dtype=np.float32)
    buffer[:, :3] = pts
    if colors is not None:
        buffer[:, 3:6] = colors
    else:
        z = pts[:, 2]
        z_min, z_max = z.min(), z.max()
        if z_max - z_min < 1e-6:
            z_norm = np.zeros_like(z)
        else:
            z_norm = (z - z_min) / (z_max - z_min)

        if color_scheme == "gbl":
            # GBL: 白→浅蓝→深蓝渐变（高度越低越蓝）
            buffer[:, 3] = np.clip(1.0 - z_norm * 0.6, 0, 1)   # R: 1→0.4
            buffer[:, 4] = np.clip(1.0 - z_norm * 0.3, 0, 1)   # G: 1→0.7
            buffer[:, 5] = np.clip(0.4 + z_norm * 0.6, 0, 1)   # B: 0.4→1.0
        else:
            # Jet colormap（默认 TLS）
            buffer[:, 3] = np.clip(1.5 - np.abs(4 * z_norm - 2), 0, 1)
            buffer[:, 4] = np.clip(1.5 - np.abs(4 * z_norm - 1), 0, 1)
            buffer[:, 5] = np.clip(1.5 - np.abs(4 * z_norm - 3), 0, 1)

    return buffer, n


def process_pcd(filepath, voxel, max_points, color_scheme="jet", label=""):
    """读取、降采样、裁剪单个 PCD 文件，返回 (binary, count, stats)"""
    label_str = f" [{label}]" if label else ""
    print(f"[*] 读取 PCD{label_str}: {filepath}")
    t0 = time.time()
    pts, colors = read_pcd(filepath)
    print(f"    读取完成: {pts.shape[0]} 个点 ({time.time() - t0:.1f}s)")

    if voxel > 0:
        t0 = time.time()
        pts, colors = voxel_downsample(pts, colors, voxel)
        print(f"    降采样 ({voxel}m): {pts.shape[0]} 个点 ({time.time() - t0:.1f}s)")

    if pts.shape[0] > max_points:
        indices = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[indices]
        if colors is not None:
            colors = colors[indices]
        print(f"    随机裁剪到 {max_points} 个点")

    buffer, n = build_buffer(pts, colors, color_scheme)
    binary = buffer.tobytes()

    box_min = pts.min(axis=0)
    box_max = pts.max(axis=0)
    center = (box_min + box_max) / 2

    stats = {
        "points": int(n),
        "voxel_size": voxel,
        "bbox_min": box_min.tolist(),
        "bbox_max": box_max.tolist(),
        "center": center.tolist(),
        "size_mb": round(len(binary) / 1024 / 1024, 2),
    }

    print(f"    最终点数: {n}")
    print(f"    数据大小: {stats['size_mb']} MB")
    print(f"    包围盒: [{box_min[0]:.1f}, {box_min[1]:.1f}, {box_min[2]:.1f}] → "
          f"[{box_max[0]:.1f}, {box_max[1]:.1f}, {box_max[2]:.1f}]")

    return binary, n, stats


def serve_pcd(pcd_path=None, tls_pcd=None, gbl_pcd=None,
              port=8083, voxel=0.1, max_points=2000000):
    """主入口：读取 PCD、降采样、启动 HTTP 服务"""

    is_dual = bool(tls_pcd and gbl_pcd)
    OfflinePCDHandler.dual_mode = is_dual

    if is_dual:
        print("=" * 50)
        print("  双图层模式: TLS + GBL")
        print("=" * 50)
        # TLS 图层 (Jet 配色)
        binary_tls, count_tls, stats_tls = process_pcd(
            tls_pcd, voxel, max_points, color_scheme="jet", label="TLS")
        OfflinePCDHandler.tls_binary = binary_tls
        OfflinePCDHandler.tls_count = count_tls
        OfflinePCDHandler.tls_stats = stats_tls

        # GBL 图层 (蓝白配色)
        binary_gbl, count_gbl, stats_gbl = process_pcd(
            gbl_pcd, voxel, max_points, color_scheme="gbl", label="GBL")
        OfflinePCDHandler.gbl_binary = binary_gbl
        OfflinePCDHandler.gbl_count = count_gbl
        OfflinePCDHandler.gbl_stats = stats_gbl

        # /points 默认指向 TLS（兼容旧接口）
        OfflinePCDHandler.pointcloud_binary = binary_tls
        OfflinePCDHandler.point_count = count_tls
        OfflinePCDHandler.stats = stats_tls

        print(f"\n    TLS: {count_tls} 点 | GBL: {count_gbl} 点")
    else:
        # 单图层模式（向后兼容）
        if tls_pcd:
            filepath = tls_pcd
        elif gbl_pcd:
            filepath = gbl_pcd
        elif pcd_path:
            filepath = pcd_path
        else:
            print("错误: 未指定任何 PCD 文件")
            sys.exit(1)

        binary, count, stats = process_pcd(filepath, voxel, max_points)
        OfflinePCDHandler.pointcloud_binary = binary
        OfflinePCDHandler.point_count = count
        OfflinePCDHandler.stats = stats

        # 单文件时 TLS/GBL 指向同一份数据
        OfflinePCDHandler.tls_binary = binary
        OfflinePCDHandler.tls_count = count
        OfflinePCDHandler.tls_stats = stats
        OfflinePCDHandler.gbl_binary = binary
        OfflinePCDHandler.gbl_count = count
        OfflinePCDHandler.gbl_stats = stats

    # 设置 web 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    web_dir = os.path.normpath(os.path.join(script_dir, "..", "web"))
    if not os.path.isdir(web_dir):
        web_dir = os.path.join(script_dir, "web")  # fallback

    OfflinePCDHandler.web_dir = web_dir
    os.chdir(web_dir)

    # 启动 HTTP 服务
    handler = OfflinePCDHandler
    with http.server.ThreadingHTTPServer(("0.0.0.0", port), handler) as httpd:
        print(f"\n✅ 服务已启动: http://localhost:{port}/3d_offline_viewer.html")
        if is_dual:
            print(f"   TLS 端点: /points/tls  ({count_tls} 点)")
            print(f"   GBL 端点: /points/gbl  ({count_gbl} 点)")
        print(f"   按 Ctrl+C 停止")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] 关闭服务")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="离线 PCD 3D 查看服务（支持 TLS+GBL 双图层）")
    parser.add_argument("pcd", nargs="?", default=None,
                        help="PCD 文件路径（单文件模式，向后兼容）")
    parser.add_argument("--tls-pcd", type=str, default=None,
                        help="TLS 图层 PCD 文件路径")
    parser.add_argument("--gbl-pcd", type=str, default=None,
                        help="GBL 图层 PCD 文件路径")
    parser.add_argument("--port", type=int, default=8083, help="HTTP 端口 (默认 8083)")
    parser.add_argument("--voxel", type=float, default=0.1, help="降采样体素大小 (m, 默认 0.1)")
    parser.add_argument("--max-points", type=int, default=2000000, help="最大点数 (默认 2000k)")

    args = parser.parse_args()

    # 校验
    if not args.pcd and not args.tls_pcd and not args.gbl_pcd:
        print("错误: 请指定 PCD 文件路径，或使用 --tls-pcd / --gbl-pcd")
        sys.exit(1)

    if args.tls_pcd and not os.path.exists(args.tls_pcd):
        print(f"错误: TLS 文件不存在: {args.tls_pcd}")
        sys.exit(1)
    if args.gbl_pcd and not os.path.exists(args.gbl_pcd):
        print(f"错误: GBL 文件不存在: {args.gbl_pcd}")
        sys.exit(1)
    if args.pcd and not os.path.exists(args.pcd):
        print(f"错误: 文件不存在: {args.pcd}")
        sys.exit(1)

    serve_pcd(
        pcd_path=args.pcd,
        tls_pcd=args.tls_pcd,
        gbl_pcd=args.gbl_pcd,
        port=args.port,
        voxel=args.voxel,
        max_points=args.max_points,
    )
