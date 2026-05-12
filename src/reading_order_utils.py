import math
from typing import Any, Dict, List, Tuple


TEXT_TYPES = {"paragraph", "title", "list_item", "caption", "formula"}
IGNORED_TYPES = {"header", "footer", "page_number"}


def _bbox(block: Dict[str, Any]) -> List[float]:
    bbox = block.get("bbox", [0, 0, 0, 0])
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def _center_x(block: Dict[str, Any]) -> float:
    x1, _, x2, _ = _bbox(block)
    return 0.5 * (x1 + x2)


def _text_like_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    preferred = [b for b in blocks if str(b.get("type", "")).lower() in TEXT_TYPES]
    if preferred:
        return preferred
    return [b for b in blocks if str(b.get("type", "")).lower() not in IGNORED_TYPES]


def detect_columns_by_projection(
    blocks: List[Dict[str, Any]],
    page_width: float,
    bins: int = 96,
    min_gap_ratio: float = 0.04,
    low_occupancy_ratio: float = 0.05,
    max_columns: int = 4,
) -> List[float]:
    page_width = float(max(1.0, page_width))
    active = _text_like_blocks(blocks)
    if len(active) < 4:
        return [0.0, page_width]

    bins = max(32, int(bins))
    hist = [0.0] * bins
    for block in active:
        x1, y1, x2, y2 = _bbox(block)
        if x2 <= x1:
            continue
        start = max(0, min(bins - 1, int((x1 / page_width) * bins)))
        end = max(start + 1, min(bins, int(math.ceil((x2 / page_width) * bins))))
        weight = min(3.0, 0.5 + max(0.0, y2 - y1) / 80.0)
        for idx in range(start, end):
            hist[idx] += weight

    peak = max(hist) if hist else 0.0
    if peak <= 0:
        return [0.0, page_width]

    low_threshold = peak * max(0.01, float(low_occupancy_ratio))
    min_gap_bins = max(2, int(math.ceil(float(min_gap_ratio) * bins)))

    raw_gaps: List[Tuple[int, int]] = []
    gap_start = None
    for idx, val in enumerate(hist):
        if val <= low_threshold:
            if gap_start is None:
                gap_start = idx
        elif gap_start is not None:
            if idx - gap_start >= min_gap_bins:
                raw_gaps.append((gap_start, idx))
            gap_start = None
    if gap_start is not None and bins - gap_start >= min_gap_bins:
        raw_gaps.append((gap_start, bins))

    if not raw_gaps:
        return [0.0, page_width]

    cuts: List[float] = []
    for start, end in sorted(raw_gaps, key=lambda item: (item[1] - item[0]), reverse=True):
        cut = ((start + end) * 0.5 / bins) * page_width
        if cut <= 0.08 * page_width or cut >= 0.92 * page_width:
            continue
        if any(abs(cut - existing) < 0.08 * page_width for existing in cuts):
            continue
        cuts.append(cut)
        if len(cuts) >= max_columns - 1:
            break

    if not cuts:
        return [0.0, page_width]

    return [0.0] + sorted(cuts) + [page_width]


def assign_block_columns(blocks: List[Dict[str, Any]], boundaries: List[float]) -> Tuple[Dict[Any, int], int]:
    if not boundaries or len(boundaries) < 2:
        boundaries = [0.0, 1.0]

    interior = list(boundaries[1:-1])
    column_count = max(1, len(boundaries) - 1)
    mapping: Dict[Any, int] = {}
    for idx, block in enumerate(blocks):
        block_id = block.get("id", idx)
        if str(block.get("type", "")).lower() in IGNORED_TYPES:
            mapping[block_id] = 0
            continue
        col_id = 0
        cx = _center_x(block)
        for threshold in interior:
            if cx > threshold:
                col_id += 1
            else:
                break
        mapping[block_id] = col_id
    return mapping, column_count


def compute_page_median_gap(blocks: List[Dict[str, Any]], page_height: float) -> float:
    page_height = float(max(1.0, page_height))
    active = [b for b in blocks if str(b.get("type", "")).lower() not in IGNORED_TYPES]
    if len(active) < 2:
        return page_height * 0.01

    page_width = max((_bbox(b)[2] for b in active), default=page_height)
    boundaries = detect_columns_by_projection(active, page_width)
    column_map, column_count = assign_block_columns(active, boundaries)
    by_column: Dict[int, List[Dict[str, Any]]] = {col: [] for col in range(column_count)}
    for block in active:
        by_column.setdefault(column_map.get(block.get("id"), 0), []).append(block)

    gaps: List[float] = []
    for blocks_in_col in by_column.values():
        ordered = sorted(blocks_in_col, key=lambda item: (_bbox(item)[1], _bbox(item)[0]))
        for prev, cur in zip(ordered[:-1], ordered[1:]):
            prev_bb = _bbox(prev)
            cur_bb = _bbox(cur)
            gap = cur_bb[1] - prev_bb[3]
            if gap > 0:
                gaps.append(gap)

    if not gaps:
        heights = [max(1.0, _bbox(block)[3] - _bbox(block)[1]) for block in active]
        if not heights:
            return page_height * 0.01
        heights.sort()
        mid = len(heights) // 2
        median_h = heights[mid] if len(heights) % 2 == 1 else 0.5 * (heights[mid - 1] + heights[mid])
        return max(6.0, 0.6 * median_h)

    gaps.sort()
    mid = len(gaps) // 2
    return gaps[mid] if len(gaps) % 2 == 1 else 0.5 * (gaps[mid - 1] + gaps[mid])
