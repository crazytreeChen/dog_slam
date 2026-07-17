#!/bin/bash
# ============================================================
# Potree 地图转换脚本
# 将 PCD 点云文件转为 Potree Web 格式
#
# 用法:
#   ./convert_to_potree.sh <input.pcd> [output_dir]
#
# 示例:
#   # 转换建图结果
#   ./convert_to_potree.sh /home/ztl/dog_slam/maps/test.pcd
#
#   # 转换后数据在 web/potree_data/3dmap/
#   # 浏览器访问 http://<ip>:8083/3d_viewer_potree.html
#
# 依赖:
#   - PotreeConverter (https://github.com/potree/PotreeConverter)
#   - 或 Docker: mrayson/potreeconverter
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEB_DIR="$(dirname "$SCRIPT_DIR")/web"
POTREE_DATA_DIR="${WEB_DIR}/potree_data"

INPUT_PCD="${1:-}"
OUTPUT_NAME="${2:-3dmap}"

if [ -z "$INPUT_PCD" ]; then
  echo "用法: $0 <input.pcd> [output_name]"
  echo ""
  echo "示例:"
  echo "  $0 /home/ztl/dog_slam/maps/test.pcd"
  echo "  $0 /projects/LOAM/test.pcd mymap"
  exit 1
fi

if [ ! -f "$INPUT_PCD" ]; then
  echo "❌ 文件不存在: $INPUT_PCD"
  exit 1
fi

OUTPUT_DIR="${POTREE_DATA_DIR}/${OUTPUT_NAME}"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Potree 地图转换"
echo "============================================"
echo "  输入: $INPUT_PCD"
echo "  输出: $OUTPUT_DIR"
echo ""

# 方法1: 使用 Docker (推荐，无需本地编译)
if command -v docker &> /dev/null; then
  echo "🔧 使用 Docker PotreeConverter..."
  echo "   (首次运行会拉取镜像，约 500MB)"

  INPUT_ABS="$(realpath "$INPUT_PCD")"
  OUTPUT_ABS="$(realpath "$OUTPUT_DIR")"
  INPUT_DIR="$(dirname "$INPUT_ABS")"
  INPUT_NAME="$(basename "$INPUT_ABS")"

  docker run --rm \
    -v "${INPUT_DIR}:/input" \
    -v "${OUTPUT_ABS}:/output" \
    mrayson/potreeconverter:latest \
    PotreeConverter \
      "/input/${INPUT_NAME}" \
      -o /output \
      --overwrite \
      --output-format LAS \
      -p webviewer

  echo ""
  echo "✅ 转换完成！"
  echo "   输出目录: $OUTPUT_DIR"
  echo "   访问地址: http://<机器人IP>:8083/3d_viewer_potree.html"

# 方法2: 本地 PotreeConverter
elif command -v PotreeConverter &> /dev/null; then
  echo "🔧 使用本地 PotreeConverter..."

  PotreeConverter "$INPUT_PCD" \
    -o "$OUTPUT_DIR" \
    --overwrite \
    --output-format LAS \
    -p webviewer

  echo ""
  echo "✅ 转换完成！"
  echo "   输出目录: $OUTPUT_DIR"
  echo "   访问地址: http://<机器人IP>:8083/3d_viewer_potree.html"

else
  echo "❌ 未找到 PotreeConverter 或 Docker"
  echo ""
  echo "请安装以下之一:"
  echo ""
  echo "  方案1 (推荐): Docker"
  echo "    docker pull mrayson/potreeconverter:latest"
  echo ""
  echo "  方案2: 本地编译"
  echo "    git clone https://github.com/potree/PotreeConverter.git"
  echo "    cd PotreeConverter && mkdir build && cd build"
  echo "    cmake .. && make -j$(nproc)"
  echo "    sudo make install"
  echo ""
  echo "  方案3: 在另一台机器上用 CloudCompare 导出 Potree 格式"
  exit 1
fi

# 显示生成的文件
echo ""
echo "📁 生成的文件:"
ls -lh "$OUTPUT_DIR/" | head -20
echo ""
echo "💡 提示:"
echo "   - 确保 HTTP 服务器可访问 potree_data/ 目录"
echo "   - 3d_viewer_potree.html 会自动加载 potree_data/3dmap/"
echo "   - 如果输出名称不是 '3dmap'，在页面上手动输入路径加载"
