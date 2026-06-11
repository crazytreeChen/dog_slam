#!/bin/bash
# =============================================================================
# 一键同步 dog_slam 代码到机器人主机
# 使用方式: bash sync_to_robot.sh [--dry-run]
# =============================================================================

set -e

# =============================================================================
# 配置区 —— 请修改为你的机器人实际 IP、用户名和路径
# =============================================================================
ROBOT_IP="10.44.10.142"          # 机器人 IP 地址
ROBOT_USER="agi"                  # 机器人登录用户名
ROBOT_PATH="/data/rkrobot/dog_slam/LIO-SAM_MID360_ROS2_PKG"  # 机器人上的工程目录
ROBOT_PORT="22"                   # SSH 端口（默认 22）

# =============================================================================
# 排除规则 —— 这些目录/文件不会被同步
# =============================================================================
EXCLUDE_DIRS=(
    "build/"
    "install/"
    "log/"
    ".git/"
    "__pycache__/"
)
EXCLUDE_FILES=(
    "*.pyc"
    ".DS_Store"
)

# =============================================================================
# 颜色定义
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

# =============================================================================
# 参数解析
# =============================================================================
DRY_RUN=false
RSYNC_DRY_FLAG=""

for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            RSYNC_DRY_FLAG="-n"
            ;;
        --help|-h)
            echo "用法: bash sync_to_robot.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --dry-run    仅预览，不实际传输文件"
            echo "  --help, -h   显示此帮助信息"
            echo ""
            echo "配置: 编辑脚本顶部的 ROBOT_IP, ROBOT_USER, ROBOT_PATH 变量"
            echo ""
            echo "同步逻辑:"
            echo "  - 远程无工程目录 → 自动全量首次同步"
            echo "  - 远程已有工程   → 仅同步 git 未提交的修改文件（不含新增未跟踪文件）"
            echo "  - 自动排除构建产物 (build/, install/, log/ 等)"
            exit 0
            ;;
    esac
done

# =============================================================================
# 辅助函数
# =============================================================================
die() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# =============================================================================
# 环境检查
# =============================================================================
echo "======================================"
echo "🚀 dog_slam 代码同步到机器人主机"
echo "======================================"

if $DRY_RUN; then
    warn "【预览模式】不会实际传输文件"
fi
echo ""

# 检查必需命令
for cmd in git rsync ssh; do
    if ! command -v $cmd &>/dev/null; then
        die "未找到 '$cmd' 命令，请确认已安装 Git Bash (含 rsync 和 ssh)"
    fi
done

# 验证配置
if [[ -z "$ROBOT_IP" ]]; then
    die "ROBOT_IP 未配置，请编辑脚本顶部的配置变量"
fi
if [[ -z "$ROBOT_USER" ]]; then
    die "ROBOT_USER 未配置，请编辑脚本顶部的配置变量"
fi
if [[ -z "$ROBOT_PATH" ]]; then
    die "ROBOT_PATH 未配置，请编辑脚本顶部的配置变量"
fi

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 确认在 git 仓库中
if ! git rev-parse --git-dir &>/dev/null; then
    die "当前目录不是 git 仓库，请在 dog_slam 项目中运行此脚本"
fi

# =============================================================================
# 构建排除参数
# =============================================================================
build_rsync_excludes() {
    local excludes=""
    for dir in "${EXCLUDE_DIRS[@]}"; do
        excludes="$excludes --exclude='$dir'"
    done
    for file in "${EXCLUDE_FILES[@]}"; do
        excludes="$excludes --exclude='$file'"
    done
    echo "$excludes"
}

build_grep_filter() {
    # 构建用于过滤 git diff 输出的 grep 表达式
    # 匹配路径中包含排除目录或文件的，予以排除
    local patterns=""
    for dir in "${EXCLUDE_DIRS[@]}"; do
        # 去掉末尾 / 用于匹配
        local dname="${dir%/}"
        if [[ -z "$patterns" ]]; then
            patterns="(^|/)${dname}(/|$)"
        else
            patterns="$patterns|(^|/)${dname}(/|$)"
        fi
    done
    for file in "${EXCLUDE_FILES[@]}"; do
        # 将通配符 * 转换为 grep 正则
        local fname="${file//\*/.\*}"
        if [[ -z "$patterns" ]]; then
            patterns="${fname}$"
        else
            patterns="$patterns|${fname}$"
        fi
    done
    echo "$patterns"
}

# =============================================================================
# SSH 连接测试
# =============================================================================
info "测试 SSH 连接: ${ROBOT_USER}@${ROBOT_IP}:${ROBOT_PORT} ..."

SSH_CMD="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -p ${ROBOT_PORT}"

if ! $SSH_CMD "${ROBOT_USER}@${ROBOT_IP}" "echo ok" &>/dev/null; then
    die "SSH 连接失败，请检查网络、IP 和 SSH 密钥配置"
fi
success "SSH 连接正常"

# =============================================================================
# 检测远程目录是否存在
# =============================================================================
REMOTE_EXISTS=$($SSH_CMD "${ROBOT_USER}@${ROBOT_IP}" "test -d '${ROBOT_PATH}' && echo yes || echo no" 2>/dev/null)

RSYNC_RSH="ssh -p ${ROBOT_PORT} -o StrictHostKeyChecking=no"
RSYNC_BASE="rsync -avz ${RSYNC_DRY_FLAG} --progress -e '${RSYNC_RSH}'"

# =============================================================================
# 全量同步（首次部署）
# =============================================================================
do_full_sync() {
    echo ""
    echo "======================================"
    warn "远程目录不存在，执行首次全量同步..."
    echo "======================================"
    echo "目标: ${ROBOT_USER}@${ROBOT_IP}:${ROBOT_PATH}"
    echo ""

    # 确保远程父目录存在
    $SSH_CMD "${ROBOT_USER}@${ROBOT_IP}" "mkdir -p '${ROBOT_PATH}'"

    # 构建排除参数
    EXCLUDES=$(build_rsync_excludes)

    # 执行全量同步
    # 注意: 使用 eval 是因为 exclude 参数含引号需要正确展开
    eval ${RSYNC_BASE} ${EXCLUDES} ./ ${ROBOT_USER}@${ROBOT_IP}:${ROBOT_PATH}

    echo ""
    echo "======================================"
    success "首次全量同步完成！"
    echo "======================================"
}

# =============================================================================
# 增量同步（仅 git 变更文件）
# =============================================================================
do_incremental_sync() {
    echo ""
    echo "目标: ${ROBOT_USER}@${ROBOT_IP}:${ROBOT_PATH}"
    echo ""

    # 获取 git 变更文件（仅 Added 和 Modified，不含 Deleted 和 Renamed）
    info "检查本地 git 变更..."
    RAW_FILES=$(git diff --name-only --diff-filter=AM HEAD 2>/dev/null || true)

    if [[ -z "$RAW_FILES" ]]; then
        echo ""
        success "本地没有未提交的修改，无需同步"
        exit 0
    fi

    # 过滤排除目录和文件
    FILTER_PATTERN=$(build_grep_filter)
    FILTERED_FILES=$(echo "$RAW_FILES" | grep -vE "$FILTER_PATTERN" || true)

    if [[ -z "$FILTERED_FILES" ]]; then
        echo ""
        warn "所有变更文件都在排除目录中（build/、install/、log/ 等），无需同步"
        exit 0
    fi

    # 统计文件数
    FILE_COUNT=$(echo "$FILTERED_FILES" | wc -l)
    FILE_COUNT=$(echo "$FILE_COUNT" | tr -d '[:space:]')

    echo ""
    echo "======================================"
    info "发现 ${FILE_COUNT} 个变更文件，开始增量同步..."
    echo "======================================"
    echo ""

    # 显示将要同步的文件列表
    echo "--- 变更文件列表 ---"
    echo "$FILTERED_FILES" | while IFS= read -r f; do
        echo "  📄 $f"
    done
    echo ""

    # 创建临时文件列表供 rsync --files-from 使用
    TEMP_FILE=$(mktemp)
    # 确保退出时清理临时文件
    trap "rm -f '$TEMP_FILE'" EXIT

    echo "$FILTERED_FILES" > "$TEMP_FILE"

    # 执行增量同步
    eval ${RSYNC_BASE} --files-from='${TEMP_FILE}' ./ ${ROBOT_USER}@${ROBOT_IP}:${ROBOT_PATH}

    echo ""
    echo "======================================"
    success "增量同步完成！共同步 ${FILE_COUNT} 个文件"
    echo "======================================"
}

# =============================================================================
# 主逻辑
# =============================================================================
if [[ "$REMOTE_EXISTS" == "no" ]]; then
    do_full_sync
else
    do_incremental_sync
fi
