"""
阅读顺序恢复工具模块
===================
功能:
1. detect_columns_by_projection - 基于投影直方图的稳健栏分割
2. assign_block_columns         - 将 block bbox 中心 x 映射到列区间
3. compute_page_median_gap      - 计算页面 blocks 的中位数垂直间距
4. sort_blocks_reading_order    - 按阅读顺序排序（含多栏、跨栏块、header/footer）
"""

from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore  # pragma: no cover


# ---------------------------------------------------------------------------
# 1. 投影直方图栏分割
# ---------------------------------------------------------------------------

def detect_columns_by_projection(
    blocks: List[Dict[str, Any]],
    page_width: float,
    smooth_sigma_ratio: float = 0.01,
    min_valley_depth_ratio: float = 0.3,
) -> List[float]:
    """
    基于水平投影直方图检测栏分割线。

    对页面宽度方向建立"水平投影直方图"（统计每个 x 区间的文本覆盖量），
    平滑后找谷值作为栏间分割线。

    Args:
        blocks:               block 列表（header/footer 和宽块会被自动跳过）
        page_width:           页面宽度（像素/点）
        smooth_sigma_ratio:   平滑 sigma 与直方图 bin 数之比，控制平滑程度
        min_valley_depth_ratio: 谷值需低于 peak * ratio 才视为有效分割

    Returns:
        column_boundaries: 栏边界 x 坐标列表，首元素为 0，末元素为 page_width。
                           例如 [0, x_split1, page_width] 表示两栏。
    """
    page_w = max(1.0, float(page_width))
    N = 200  # 直方图 bin 数
    _CROSS_COL_THRESHOLD = 0.7  # 跨栏块宽度阈值（相对页宽）

    if np is None:
        return _detect_columns_gap_fallback(blocks, page_w)

    hist = np.zeros(N, dtype=float)

    for b in blocks:
        if b.get("type") in ("header", "footer"):
            continue
        bbox = b.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1 = float(bbox[0])
        x2 = float(bbox[2])
        y1 = float(bbox[1])
        y2 = float(bbox[3])
        bw = x2 - x1
        # 跳过跨栏宽块（避免其填充栏间空隙影响检测）
        if bw > _CROSS_COL_THRESHOLD * page_w:
            continue
        bh = max(1.0, y2 - y1)
        bin1 = max(0, int(x1 / page_w * N))
        bin2 = min(N - 1, int(x2 / page_w * N))
        if bin2 >= bin1:
            hist[bin1 : bin2 + 1] += bh

    if hist.max() <= 0:
        return [0.0, page_w]

    sigma_bins = max(1.0, smooth_sigma_ratio * N)
    smoothed = _gaussian_smooth_1d(hist, sigma_bins)

    peak = float(smoothed.max())
    threshold = min_valley_depth_ratio * peak

    # 找局部最小值且低于阈值的 bin
    valleys: List[int] = []
    for i in range(1, N - 1):
        if smoothed[i] < threshold:
            if smoothed[i] <= smoothed[i - 1] and smoothed[i] <= smoothed[i + 1]:
                valleys.append(i)

    if not valleys:
        return [0.0, page_w]

    # 合并间距过近的谷（< 5% 页宽），每组取最深点
    min_gap_bins = max(1, int(0.05 * N))
    valley_groups: List[List[int]] = []
    current_group: List[int] = [valleys[0]]
    for v in valleys[1:]:
        if v - current_group[-1] <= min_gap_bins:
            current_group.append(v)
        else:
            valley_groups.append(current_group)
            current_group = [v]
    valley_groups.append(current_group)

    split_bins = [min(g, key=lambda i: smoothed[i]) for g in valley_groups]

    # 最多支持 3 个分割点（4 栏）
    if len(split_bins) > 3:
        split_bins = sorted(split_bins, key=lambda i: float(smoothed[i]))[:3]

    # 过滤会产生过窄列的分割点（最小列宽 = 10% 页宽）
    min_col_bins = max(1, int(0.10 * N))
    sorted_splits = sorted(split_bins)
    filtered_splits: List[int] = []
    prev_bin = 0
    for s in sorted_splits:
        if (s - prev_bin) >= min_col_bins:
            filtered_splits.append(s)
            prev_bin = s
    # 检查最后一列是否足够宽
    if filtered_splits and (N - filtered_splits[-1]) < min_col_bins:
        filtered_splits.pop()
    split_bins = filtered_splits

    if not split_bins:
        return [0.0, page_w]

    boundaries = [0.0]
    for bin_idx in sorted(split_bins):
        x = (bin_idx + 0.5) / N * page_w
        boundaries.append(float(x))
    boundaries.append(float(page_w))

    return boundaries


def _gaussian_smooth_1d(arr: "np.ndarray", sigma: float) -> "np.ndarray":
    """纯 numpy 实现的一维高斯平滑（边缘填充模式）。"""
    if sigma < 0.5:
        return arr.copy()
    radius = int(3.0 * sigma + 0.5)
    xs = np.arange(2 * radius + 1) - radius
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel = kernel / kernel.sum()
    padded = np.concatenate([
        np.full(radius, arr[0]),
        arr,
        np.full(radius, arr[-1]),
    ])
    result = np.convolve(padded, kernel, mode="valid")
    return result[: len(arr)]


def _detect_columns_gap_fallback(
    blocks: List[Dict[str, Any]], page_w: float
) -> List[float]:
    """numpy 不可用时的简单间距分割回退。"""
    centers: List[float] = []
    for b in blocks:
        if b.get("type") in ("header", "footer"):
            continue
        bbox = b.get("bbox")
        if bbox and len(bbox) >= 4:
            cx = (float(bbox[0]) + float(bbox[2])) / 2
            centers.append(cx)

    if len(centers) < 4:
        return [0.0, page_w]

    centers.sort()
    gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
    gap_vals = sorted(g[0] for g in gaps)
    median_gap = gap_vals[len(gap_vals) // 2]

    significant = sorted(
        [(g, i) for g, i in gaps if g > max(2.0 * median_gap, 0.1 * page_w)],
        key=lambda x: -x[0],
    )[:3]

    if not significant:
        return [0.0, page_w]

    boundaries = [0.0]
    for _, idx in sorted(significant, key=lambda x: x[1]):
        x = (centers[idx] + centers[idx + 1]) / 2
        boundaries.append(float(x))
    boundaries.append(float(page_w))
    return boundaries


# ---------------------------------------------------------------------------
# 2. Block → 列 ID 分配
# ---------------------------------------------------------------------------

def assign_block_columns(
    blocks: List[Dict[str, Any]],
    boundaries: List[float],
) -> Tuple[Dict[Any, int], int]:
    """
    根据 column boundaries 为每个 block 分配 column_id。

    header 分配到 column_id = -1，footer 分配到 column_id = n_cols。
    普通 block 根据中心 x 所在区间分配。

    Args:
        blocks:     block 列表
        boundaries: 栏边界列表，如 [0, x_split, page_width]

    Returns:
        (column_map, n_columns): {block_id: column_id}, 总栏数
    """
    n_cols = max(1, len(boundaries) - 1)
    column_map: Dict[Any, int] = {}

    for b in blocks:
        bid = b.get("id")
        btype = b.get("type", "")

        if btype == "header":
            column_map[bid] = -1
            continue
        if btype == "footer":
            column_map[bid] = n_cols
            continue

        bbox = b.get("bbox", [0, 0, 0, 0])
        cx = (float(bbox[0]) + float(bbox[2])) / 2

        col_id = 0
        for thresh in boundaries[1:-1]:
            if cx > thresh:
                col_id += 1
            else:
                break
        column_map[bid] = col_id

    return column_map, n_cols


# ---------------------------------------------------------------------------
# 3. 页面中位数垂直间距
# ---------------------------------------------------------------------------

def compute_page_median_gap(
    blocks: List[Dict[str, Any]],
    page_height: float,
) -> float:
    """
    计算页面内相邻 blocks 之间的中位数垂直间距（正值）。

    用于归一化 vertical_gap_to_median_ratio 特征。

    Args:
        blocks:      block 列表
        page_height: 页面高度

    Returns:
        median_gap (> 0)
    """
    page_h = max(1.0, float(page_height))

    body_blocks = sorted(
        [b for b in blocks if b.get("type") not in ("header", "footer")],
        key=lambda b: b.get("bbox", [0, 0, 0, 0])[1],
    )

    gaps: List[float] = []
    for i in range(len(body_blocks) - 1):
        bb1 = body_blocks[i].get("bbox", [0, 0, 0, 0])
        bb2 = body_blocks[i + 1].get("bbox", [0, 0, 0, 0])
        gap = max(0.0, float(bb2[1]) - float(bb1[3]))
        gaps.append(gap)

    if not gaps:
        return 0.01 * page_h

    gaps.sort()
    return max(1.0, gaps[len(gaps) // 2])


# ---------------------------------------------------------------------------
# 4. 阅读顺序排序（多栏 + 跨栏块）
# ---------------------------------------------------------------------------

def sort_blocks_reading_order(
    blocks: List[Dict[str, Any]],
    page: Dict[str, Any],
    cross_col_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """
    按阅读顺序排序 blocks。

    规则:
    1. header 永远最前，按 y 升序排列
    2. 正文：先左栏再右栏（多栏时），栏内按 y1 升序，y 相近时按 x1 升序
    3. 宽度超过 cross_col_threshold * page_width 的跨栏块按其 y1 位置插入
    4. footer 永远最后，按 y 升序排列

    Args:
        blocks:              block 列表
        page:                page 字典（需含 width）
        cross_col_threshold: 跨栏判断阈值（相对页宽，默认 0.7）

    Returns:
        按阅读顺序排列的 block 列表（原对象，不复制）
    """
    if not blocks:
        return []

    page_w = max(1.0, float(page.get("width", 1)))

    text_blocks = [b for b in blocks if b.get("type") not in ("header", "footer")]
    boundaries = detect_columns_by_projection(text_blocks, page_w)
    n_cols = max(1, len(boundaries) - 1)

    def _key(b: Dict[str, Any]) -> Tuple[float, float]:
        bb = b.get("bbox", [0, 0, 0, 0])
        return (float(bb[1]), float(bb[0]))

    headers: List[Dict[str, Any]] = []
    footers: List[Dict[str, Any]] = []
    cross_col: List[Dict[str, Any]] = []
    col_blocks: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(n_cols)}

    for b in blocks:
        btype = b.get("type", "")
        if btype == "header":
            headers.append(b)
            continue
        if btype == "footer":
            footers.append(b)
            continue

        bbox = b.get("bbox", [0, 0, 0, 0])
        bw = float(bbox[2]) - float(bbox[0])

        if n_cols > 1 and bw > cross_col_threshold * page_w:
            cross_col.append(b)
        else:
            cx = (float(bbox[0]) + float(bbox[2])) / 2
            col_id = 0
            for thresh in boundaries[1:-1]:
                if cx > thresh:
                    col_id += 1
                else:
                    break
            col_blocks[col_id].append(b)

    headers.sort(key=_key)
    footers.sort(key=_key)
    cross_col.sort(key=_key)
    for col_id in col_blocks:
        col_blocks[col_id].sort(key=_key)

    if n_cols <= 1 or not cross_col:
        body: List[Dict[str, Any]] = []
        for col_id in range(n_cols):
            body.extend(col_blocks[col_id])
        return headers + body + footers

    # 有跨栏块：按 y 插入
    body = []
    col_ptrs: Dict[int, int] = {col_id: 0 for col_id in range(n_cols)}

    def _flush_to_y(y_limit: float) -> None:
        for c in range(n_cols):
            col = col_blocks[c]
            ptr = col_ptrs[c]
            while ptr < len(col):
                bb = col[ptr].get("bbox", [0, 0, 0, 0])
                if float(bb[1]) < y_limit:
                    body.append(col[ptr])
                    ptr += 1
                else:
                    break
            col_ptrs[c] = ptr

    for cross_b in cross_col:
        y1 = float(cross_b.get("bbox", [0, 0, 0, 0])[1])
        _flush_to_y(y1)
        body.append(cross_b)

    _flush_to_y(float("inf"))

    return headers + body + footers
