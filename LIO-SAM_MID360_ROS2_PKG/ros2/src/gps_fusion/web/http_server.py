#!/usr/bin/env python3
"""
简易静态 HTTP 服务器，带 SO_REUSEADDR（解决端口残留导致 "Address already in use"）。
替代 python3 -m http.server，用于 GPS 融合 Web 可视化前端。
"""

import argparse
import http.server
import os
import socket
import sys


class ReuseAddrHTTPServer(http.server.HTTPServer):
    """在 bind 前设置 SO_REUSEADDR，允许立即复用 TIME_WAIT 状态的端口"""

    allow_reuse_address = True

    def server_bind(self):
        # 手动设置 SO_REUSEADDR 确保可靠复用
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 双栈支持（IPv4/IPv6）
        if hasattr(socket, 'SO_REUSEPORT'):
            try:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except (OSError, AttributeError):
                pass
        super().server_bind()


def main():
    parser = argparse.ArgumentParser(description='带端口复用的静态 HTTP 服务器')
    parser.add_argument('port', nargs='?', type=int, default=8084,
                        help='监听端口 (默认: 8084)')
    parser.add_argument('--dir', default=None,
                        help='Web 根目录 (默认: 当前目录)')
    args = parser.parse_args()

    if args.dir:
        os.chdir(args.dir)

    web_dir = os.getcwd()
    if not os.path.isfile(os.path.join(web_dir, 'map_viewer.html')):
        print(f'[web] 警告: {web_dir} 中未找到 map_viewer.html，继续启动...',
              file=sys.stderr)

    print(f'[web] 目录: {web_dir}')
    print(f'[web] 端口: {args.port}')

    server = ReuseAddrHTTPServer(
        ('', args.port),
        http.server.SimpleHTTPRequestHandler,
    )

    try:
        print(f'[web] HTTP 服务器已启动 http://0.0.0.0:{args.port}')
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[web] 服务器已停止')
        server.server_close()


if __name__ == '__main__':
    main()
