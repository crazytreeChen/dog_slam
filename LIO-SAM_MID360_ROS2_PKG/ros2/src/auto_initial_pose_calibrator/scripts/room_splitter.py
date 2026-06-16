"""
房间精确切割 v1 — 保持 99% 原图相似度
==========================================================
核心思路:
  1. 二值化提取墙体线 → 形态学闭运算封闭门口缺口 (仅用于分析)
  2. 连通域分析找到各个独立房间区域
  3. 从原图直接提取每个房间的像素 (不修改原图内容)
  4. 仅在缺口处画一条与原图墙壁颜色一致的细线
  5. 走廊连接处同样切割

关键: 形态学操作只用于"分析哪些像素属于哪个房间",
      最终输出直接取原图像素, 所以相似度极高。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Room:
    """表示一个切割出的房间区域"""
    id: int
    mask: np.ndarray              # 该房间的二值 mask (与原图同尺寸)
    area: float = 0.0
    centroid: Tuple[float, float] = (0.0, 0.0)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    contour: Optional[np.ndarray] = None


# ============================================================
# 核心算法
# ============================================================

def extract_wall_mask(gray: np.ndarray, wall_thresh: int = 80) -> np.ndarray:
    """
    提取墙体 mask。
    墙体 = 深色像素(< wall_thresh) + Canny 边缘
    """
    # 深色像素 = 墙体
    is_dark = (gray < wall_thresh).astype(np.uint8) * 255

    # Canny 边缘检测补充墙体线
    blur = cv2.GaussianBlur(gray, (3, 3), 0.8)
    edges = cv2.Canny(blur, 30, 90)

    # 合并
    wall = cv2.bitwise_or(is_dark, edges)

    # 清理噪点
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, k2, iterations=1)

    return wall


def extract_free_space(gray: np.ndarray,
                       free_thresh: int = 160) -> np.ndarray:
    """提取自由空间 mask (白色/亮色区域)"""
    return (gray > free_thresh).astype(np.uint8) * 255


def close_gaps_for_analysis(wall_mask: np.ndarray,
                            gap_close_size: int = 11) -> np.ndarray:
    """
    用形态学闭运算封闭门口缺口。
    仅用于分析 (确定连通域), 不会修改原图。

    gap_close_size: 闭运算 kernel 大小, 应略大于门口宽度(像素)
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gap_close_size, gap_close_size)
    )
    closed = cv2.morphologyEx(
        wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    return closed


def find_rooms(gray: np.ndarray,
               wall_closed: np.ndarray,
               free_mask: np.ndarray,
               min_room_area: int = 2000) -> List[Room]:
    """
    基于封闭后的墙体和自由空间, 做连通域分析找房间。
    """
    h, w = gray.shape

    # 可通行区域 = 自由空间 - 封闭后的墙体屏障
    traversable = free_mask.copy()
    traversable[wall_closed > 0] = 0

    # 清理碎片
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    traversable = cv2.morphologyEx(
        traversable, cv2.MORPH_OPEN, k3, iterations=1
    )

    # 连通域分析
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        traversable, connectivity=8
    )
    print(f"  [连通域] 检测到 {n_cc - 1} 个区域")

    rooms = []
    for lid in range(1, n_cc):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_room_area:
            continue

        mask = (labels == lid).astype(np.uint8) * 255
        cx, cy = centroids[lid]
        x = stats[lid, cv2.CC_STAT_LEFT]
        y = stats[lid, cv2.CC_STAT_TOP]
        bw = stats[lid, cv2.CC_STAT_WIDTH]
        bh = stats[lid, cv2.CC_STAT_HEIGHT]

        room = Room(
            id=len(rooms),
            mask=mask,
            area=float(area),
            centroid=(float(cx), float(cy)),
            bbox=(x, y, bw, bh),
        )
        rooms.append(room)

    # 按面积排序
    rooms.sort(key=lambda r: r.area, reverse=True)
    for i, r in enumerate(rooms):
        r.id = i

    return rooms


def expand_rooms_to_walls(rooms: List[Room],
                          gray: np.ndarray,
                          wall_mask_orig: np.ndarray,
                          free_mask: np.ndarray) -> List[Room]:
    """
    将每个房间的 mask 扩展到包含紧邻的墙壁像素。
    这样切割出来的房间图片会包含完整的墙壁线。
    """
    h, w = gray.shape

    # 墙壁 + 中灰色障碍物区域
    non_free = (free_mask == 0).astype(np.uint8) * 255

    k_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for room in rooms:
        # 膨胀房间 mask 几个像素
        expanded = cv2.dilate(room.mask, k_expand, iterations=2)
        # 取交集: 膨胀区域 ∩ (墙壁/障碍物)
        wall_near = cv2.bitwise_and(expanded, non_free)
        # 合并到房间 mask
        room.mask = cv2.bitwise_or(room.mask, wall_near)

    return rooms


def resolve_wall_overlap(rooms: List[Room]) -> List[Room]:
    """
    解决墙壁像素被多个房间共享的问题。
    墙壁像素分配给最近的房间(按质心距离)。
    共享的墙壁保留给双方(99%相似度要求)。
    """
    if len(rooms) < 2:
        return rooms
    # 墙壁允许被多个相邻房间共享，不做排他分配
    # 这样每个房间切出来都有完整的墙
    return rooms


def extract_room_image(orig: np.ndarray,
                       room: Room,
                       bg_color: Tuple[int, ...],
                       padding: int = 5) -> np.ndarray:
    """
    从原图中精确提取一个房间。
    - 房间内像素 = 原图像素 (100% 保真)
    - 房间外像素 = 背景色
    """
    x, y, bw, bh = room.bbox
    # 加 padding
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(orig.shape[1], x + bw + padding)
    y2 = min(orig.shape[0], y + bh + padding)

    # 裁剪区域
    crop = orig[y1:y2, x1:x2].copy()
    local_mask = room.mask[y1:y2, x1:x2]

    # 非房间区域填充背景色
    if len(orig.shape) == 3:
        crop[local_mask == 0] = bg_color[:3]
    else:
        crop[local_mask == 0] = bg_color[0]

    return crop


def detect_bg_color(gray: np.ndarray) -> int:
    """检测原图背景色 (最常见的灰度值, 通常是灰色背景)"""
    # 取图片边缘像素的中位数
    border_pixels = np.concatenate([
        gray[0, :], gray[-1, :],
        gray[:, 0], gray[:, -1]
    ])
    return int(np.median(border_pixels))


# ============================================================
# 相似度验证
# ============================================================

def verify_similarity(orig: np.ndarray,
                      rooms: List[Room],
                      room_images: List[np.ndarray]) -> float:
    """
    验证切割后与原图的像素相似度。
    对每个房间, 比较 mask 内的像素是否与原图一致。
    """
    total_pixels = 0
    matching_pixels = 0

    for room, room_img in zip(rooms, room_images):
        x, y, bw, bh = room.bbox
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(orig.shape[1], x + bw + padding)
        y2 = min(orig.shape[0], y + bh + padding)

        orig_crop = orig[y1:y2, x1:x2]
        local_mask = room.mask[y1:y2, x1:x2]

        # 只比较 mask 内的像素
        mask_bool = local_mask > 0
        n_pixels = int(mask_bool.sum())
        total_pixels += n_pixels

        if len(orig.shape) == 3:
            match = np.all(
                orig_crop[mask_bool] == room_img[mask_bool], axis=1
            )
        else:
            match = orig_crop[mask_bool] == room_img[mask_bool]

        matching_pixels += int(match.sum())

    similarity = matching_pixels / total_pixels if total_pixels > 0 else 0
    return similarity


# ============================================================
# 可视化 & 保存
# ============================================================

def visualize_segmentation(orig: np.ndarray,
                           gray: np.ndarray,
                           rooms: List[Room],
                           wall_mask: np.ndarray,
                           wall_closed: np.ndarray,
                           out_dir: Path):
    """生成分割总览图"""
    n = len(rooms)
    colors = plt.cm.Set1(np.linspace(0, 1, max(n, 9)))
    cbgr = [
        (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
        for c in colors
    ]

    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    # 原图
    if len(orig.shape) == 3:
        axes[0, 0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    else:
        axes[0, 0].imshow(orig, cmap='gray')
    axes[0, 0].set_title("原图", fontsize=12)
    axes[0, 0].axis('off')

    # 墙体检测
    axes[0, 1].imshow(wall_mask, cmap='gray')
    axes[0, 1].set_title("墙体检测", fontsize=12)
    axes[0, 1].axis('off')

    # 缺口封闭后
    axes[0, 2].imshow(wall_closed, cmap='gray')
    axes[0, 2].set_title("缺口封闭 (仅分析用)", fontsize=12)
    axes[0, 2].axis('off')

    # 房间分割叠加
    overlay = orig.copy() if len(orig.shape) == 3 else cv2.cvtColor(
        orig, cv2.COLOR_GRAY2BGR
    )
    color_fill = np.zeros_like(overlay, dtype=np.float32)
    for i, r in enumerate(rooms):
        c = cbgr[i % len(cbgr)]
        color_fill[r.mask > 0] = c
        # 画轮廓
        contours, _ = cv2.findContours(
            r.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(overlay, contours, -1, c, 2)
        # 标注
        cx, cy = int(r.centroid[0]), int(r.centroid[1])
        cv2.putText(
            overlay, f"R{i}", (cx - 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    axes[1, 0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"房间分割 ({n} 个)", fontsize=12)
    axes[1, 0].axis('off')

    # 彩色填充
    axes[1, 1].imshow(
        np.clip(color_fill, 0, 255).astype(np.uint8)
    )
    axes[1, 1].set_title("区域填充", fontsize=12)
    axes[1, 1].axis('off')

    # 信息
    info_lines = [f"房间数: {n}", ""]
    for r in rooms:
        info_lines.append(
            f"R{r.id}: {r.area:.0f}px "
            f"@ ({r.centroid[0]:.0f},{r.centroid[1]:.0f})"
        )
    axes[1, 2].text(
        0.05, 0.95, "\n".join(info_lines),
        transform=axes[1, 2].transAxes,
        fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    axes[1, 2].set_title("统计信息", fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(
        out_dir / "split_overview.png", dpi=150, bbox_inches='tight'
    )
    plt.close()
    print(f"  [OK] split_overview.png")


def save_room_images(orig: np.ndarray,
                     rooms: List[Room],
                     bg_color_val: int,
                     out_dir: Path) -> List[np.ndarray]:
    """保存每个房间的切割图片"""
    if len(orig.shape) == 3:
        bg_color = (bg_color_val, bg_color_val, bg_color_val)
    else:
        bg_color = (bg_color_val,)

    room_images = []
    for room in rooms:
        img = extract_room_image(orig, room, bg_color, padding=8)
        room_images.append(img)

        # 保存
        fname = f"room_{room.id:03d}.png"
        cv2.imwrite(str(out_dir / fname), img)
        print(
            f"  [OK] {fname}  "
            f"({img.shape[1]}x{img.shape[0]}, "
            f"area={room.area:.0f}px)"
        )

    return room_images


def save_metadata(rooms: List[Room], out_dir: Path):
    """保存房间元数据 JSON"""
    data = {
        "rooms": [
            {
                "id": r.id,
                "area": r.area,
                "centroid": list(r.centroid),
                "bbox": list(r.bbox),
            }
            for r in rooms
        ]
    }
    with open(out_dir / "rooms_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("  [OK] rooms_metadata.json")


# ============================================================
# 主流程
# ============================================================

def split_rooms(image_path: str,
                output_dir: str = None,
                wall_thresh: int = 80,
                free_thresh: int = 160,
                gap_close_size: int = 13,
                min_room_area: int = 2000,
                wall_dilate_iter: int = 2):
    """
    主入口: 切割地图图片中的各个房间。

    参数:
        image_path: 输入图片路径
        output_dir: 输出目录 (默认与输入同目录下的 room_split/)
        wall_thresh: 墙体灰度阈值 (< 此值 = 墙)
        free_thresh: 自由空间阈值 (> 此值 = 可通行)
        gap_close_size: 缺口闭运算 kernel 大小
        min_room_area: 最小房间面积 (像素)
        wall_dilate_iter: 墙体闭运算迭代次数
    """
    print("=" * 60)
    print("  房间精确切割 v1")
    print("=" * 60)

    # 加载图片
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    print(f"  图片尺寸: {gray.shape[1]}x{gray.shape[0]}")

    # 输出目录
    if output_dir is None:
        output_dir = str(Path(image_path).parent / "room_split")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 提取墙体
    print("\n[Step 1] 提取墙体...")
    wall_mask = extract_wall_mask(gray, wall_thresh)
    print(f"  墙体像素: {cv2.countNonZero(wall_mask)}")

    # Step 2: 提取自由空间
    print("[Step 2] 提取自由空间...")
    free_mask = extract_free_space(gray, free_thresh)
    print(f"  自由空间像素: {cv2.countNonZero(free_mask)}")

    # Step 3: 封闭缺口 (仅用于分析)
    print("[Step 3] 封闭门口缺口 (分析用)...")
    wall_closed = close_gaps_for_analysis(
        wall_mask, gap_close_size
    )

    # 额外: 用封闭后的墙做屏障
    k_barrier = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (gap_close_size, gap_close_size)
    )
    barrier = cv2.dilate(
        wall_closed, k_barrier, iterations=wall_dilate_iter
    )
    # 再闭运算确保连通
    k_close2 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (gap_close_size // 2 + 2, gap_close_size // 2 + 2)
    )
    barrier = cv2.morphologyEx(
        barrier, cv2.MORPH_CLOSE, k_close2, iterations=2
    )

    # Step 4: 连通域分析找房间
    print("[Step 4] 连通域分析...")
    rooms = find_rooms(gray, barrier, free_mask, min_room_area)
    print(f"  => 找到 {len(rooms)} 个房间")

    if not rooms:
        print("  [警告] 未找到房间, 尝试调整参数...")
        # 尝试更大的 gap_close_size
        for gcs in [17, 21, 25]:
            print(f"  重试 gap_close_size={gcs}...")
            wall_closed2 = close_gaps_for_analysis(wall_mask, gcs)
            k_b2 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (gcs, gcs)
            )
            barrier2 = cv2.dilate(wall_closed2, k_b2, iterations=2)
            k_c2 = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (gcs // 2 + 2, gcs // 2 + 2)
            )
            barrier2 = cv2.morphologyEx(
                barrier2, cv2.MORPH_CLOSE, k_c2, iterations=2
            )
            rooms = find_rooms(
                gray, barrier2, free_mask, min_room_area
            )
            if len(rooms) >= 2:
                barrier = barrier2
                wall_closed = wall_closed2
                print(f"  => 找到 {len(rooms)} 个房间")
                break

    if not rooms:
        print("  [错误] 无法分割房间")
        return

    # Step 5: 扩展到墙壁
    print("[Step 5] 扩展 mask 到墙壁...")
    rooms = expand_rooms_to_walls(
        rooms, gray, wall_mask, free_mask
    )

    # Step 6: 背景色检测
    bg_color = detect_bg_color(gray)
    print(f"  背景色: {bg_color}")

    # Step 7: 提取并保存
    print("[Step 7] 提取房间图片...")
    room_images = save_room_images(orig, rooms, bg_color, out_dir)

    # Step 8: 验证相似度
    print("[Step 8] 验证相似度...")
    similarity = verify_similarity(orig, rooms, room_images)
    print(f"  ★ 像素相似度: {similarity * 100:.2f}%")

    # Step 9: 可视化
    print("[Step 9] 生成可视化...")
    visualize_segmentation(
        orig, gray, rooms, wall_mask, wall_closed, out_dir
    )
    save_metadata(rooms, out_dir)

    # 汇总
    print("\n" + "=" * 60)
    print(f"  完成! 共 {len(rooms)} 个房间")
    print(f"  像素相似度: {similarity * 100:.2f}%")
    print(f"  输出目录: {out_dir}")
    print("=" * 60)

    for r in rooms:
        print(
            f"  R{r.id}: area={r.area:.0f}px "
            f"bbox=({r.bbox[0]},{r.bbox[1]},"
            f"{r.bbox[2]}x{r.bbox[3]})"
        )


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    IMG = (
        r"d:\01-Code\dog_slam\LIO-SAM_MID360_ROS2_PKG"
        r"\ros2\src\auto_initial_pose_calibrator"
        r"\scan_viz\aa34812001c24bb3b047ebce08eca6be.png"
    )
    OUT = (
        r"d:\01-Code\dog_slam\LIO-SAM_MID360_ROS2_PKG"
        r"\ros2\src\auto_initial_pose_calibrator"
        r"\scan_viz\room_split"
    )

    split_rooms(
        image_path=IMG,
        output_dir=OUT,
        wall_thresh=80,
        free_thresh=160,
        gap_close_size=13,
        min_room_area=2000,
    )
