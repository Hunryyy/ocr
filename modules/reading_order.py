from typing import Any, Dict, List, Optional, Set

from reading_order import assign_block_columns, detect_columns_by_projection


HEADER_TYPES = {"header"}
FOOTER_TYPES = {"footer", "page_number"}
FLOAT_TYPES = {"figure", "table", "chart"}


def _bbox(block: Dict[str, Any]) -> List[float]:
    bbox = block.get("bbox", [0, 0, 0, 0])
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def _center_y(block: Dict[str, Any]) -> float:
    bbox = _bbox(block)
    return 0.5 * (bbox[1] + bbox[3])


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _sort_rowwise(blocks: List[Dict[str, Any]], page_h: float, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    row_tol = max(float(cfg.get("row_tol_px", 8) or 8), float(cfg.get("row_tol_ratio", 0.4) or 0.4) * 24.0)
    ordered = sorted(blocks, key=lambda block: (_bbox(block)[1], _bbox(block)[0]))
    rows: List[List[Dict[str, Any]]] = []
    row_centers: List[float] = []
    for block in ordered:
        cy = _center_y(block)
        if rows and abs(cy - row_centers[-1]) <= row_tol:
            rows[-1].append(block)
            row_centers[-1] = (row_centers[-1] * (len(rows[-1]) - 1) + cy) / len(rows[-1])
        else:
            rows.append([block])
            row_centers.append(cy)

    out: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda block: (_bbox(block)[0], _bbox(block)[1]))
        out.extend(row)
    return out


def _spans_multiple_columns(block: Dict[str, Any], boundaries: List[float], page_w: float, cfg: Dict[str, Any]) -> bool:
    bbox = _bbox(block)
    width_ratio = (bbox[2] - bbox[0]) / max(1.0, page_w)
    if width_ratio >= float(cfg.get("cross_span_ratio", 0.75) or 0.75):
        return True
    left = sum(1 for threshold in boundaries[1:-1] if bbox[0] >= threshold)
    right = sum(1 for threshold in boundaries[1:-1] if bbox[2] > threshold)
    return right > left


def _attach_captions(ordered: List[Dict[str, Any]], page_h: float) -> List[Dict[str, Any]]:
    if not ordered:
        return []

    captions = [block for block in ordered if str(block.get("type", "")).lower() == "caption"]
    if not captions:
        return ordered

    used: Set[Any] = set()
    by_id = {block.get("id"): block for block in ordered}
    caption_after: Dict[Any, List[Dict[str, Any]]] = {}
    for caption in captions:
        cap_bbox = _bbox(caption)
        best_target = None
        best_score = None
        for target in ordered:
            t_type = str(target.get("type", "")).lower()
            if t_type not in FLOAT_TYPES:
                continue
            tar_bbox = _bbox(target)
            dy = cap_bbox[1] - tar_bbox[3]
            if dy < -0.03 * page_h or dy > 0.16 * page_h:
                continue
            overlap = _overlap_1d(cap_bbox[0], cap_bbox[2], tar_bbox[0], tar_bbox[2])
            overlap_ratio = overlap / max(1.0, min(cap_bbox[2] - cap_bbox[0], tar_bbox[2] - tar_bbox[0]))
            dx = abs(0.5 * (cap_bbox[0] + cap_bbox[2]) - 0.5 * (tar_bbox[0] + tar_bbox[2]))
            score = dy + 0.3 * dx - 60.0 * overlap_ratio
            if best_score is None or score < best_score:
                best_target = target
                best_score = score
        if best_target is not None:
            caption_after.setdefault(best_target.get("id"), []).append(caption)

    out: List[Dict[str, Any]] = []
    for block in ordered:
        block_id = block.get("id")
        if block_id in used:
            continue
        if str(block.get("type", "")).lower() == "caption" and any(block is c for caps in caption_after.values() for c in caps):
            continue
        out.append(block)
        used.add(block_id)
        for caption in sorted(caption_after.get(block_id, []), key=lambda item: (_bbox(item)[1], _bbox(item)[0])):
            cap_id = caption.get("id")
            if cap_id in used:
                continue
            out.append(caption)
            used.add(cap_id)

    for block in ordered:
        block_id = block.get("id")
        if block_id not in used:
            out.append(block)
            used.add(block_id)
    return out


def xycut_graph_sort(elements: List[Dict[str, Any]], page: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    items = list(elements or [])
    if len(items) <= 1:
        return items

    cfg = cfg or {}
    page = page or {}
    page_w = float(max(1.0, page.get("width", max((_bbox(block)[2] for block in items), default=1.0))))
    page_h = float(max(1.0, page.get("height", max((_bbox(block)[3] for block in items), default=1.0))))

    headers = sorted(
        [block for block in items if str(block.get("type", "")).lower() in HEADER_TYPES],
        key=lambda block: (_bbox(block)[1], _bbox(block)[0]),
    )
    footers = sorted(
        [block for block in items if str(block.get("type", "")).lower() in FOOTER_TYPES],
        key=lambda block: (_bbox(block)[1], _bbox(block)[0]),
    )
    body = [block for block in items if block not in headers and block not in footers]
    if not body:
        return headers + footers

    boundaries = detect_columns_by_projection(
        body,
        page_w,
        bins=int(cfg.get("projection_bins", 96) or 96),
        min_gap_ratio=float(cfg.get("min_gap_ratio", 0.04) or 0.04),
        low_occupancy_ratio=float(cfg.get("low_occupancy_ratio", 0.05) or 0.05),
    )
    column_map, n_columns = assign_block_columns(body, boundaries)

    cross_blocks: List[Dict[str, Any]] = []
    regular_columns: Dict[int, List[Dict[str, Any]]] = {col: [] for col in range(max(1, n_columns))}
    for block in body:
        if _spans_multiple_columns(block, boundaries, page_w, cfg):
            cross_blocks.append(block)
        else:
            regular_columns.setdefault(column_map.get(block.get("id"), 0), []).append(block)

    for col_id, blocks_in_col in regular_columns.items():
        regular_columns[col_id] = _sort_rowwise(blocks_in_col, page_h, cfg)
    cross_blocks = sorted(cross_blocks, key=lambda block: (_bbox(block)[1], _bbox(block)[0]))

    if n_columns <= 1 and not cross_blocks:
        return headers + _attach_captions(_sort_rowwise(body, page_h, cfg), page_h) + footers

    merged: List[Dict[str, Any]] = []
    pointers = {col: 0 for col in regular_columns}

    def flush_until(y_limit: float) -> None:
        for col in range(max(1, n_columns)):
            column_blocks = regular_columns.get(col, [])
            ptr = pointers.get(col, 0)
            while ptr < len(column_blocks):
                if _bbox(column_blocks[ptr])[1] < y_limit:
                    merged.append(column_blocks[ptr])
                    ptr += 1
                else:
                    break
            pointers[col] = ptr

    for block in cross_blocks:
        flush_until(_bbox(block)[1])
        merged.append(block)

    flush_until(float("inf"))
    merged = _attach_captions(merged, page_h)

    used = {block.get("id") for block in merged}
    for block in body:
        if block.get("id") not in used:
            merged.append(block)

    return headers + merged + footers


def build_chain_order_edges(ordered_elements: List[Dict[str, Any]], default_score: float = 1.0) -> List[Dict[str, float]]:
    edges: List[Dict[str, float]] = []
    for left, right in zip(ordered_elements[:-1], ordered_elements[1:]):
        if "id" not in left or "id" not in right:
            continue
        edges.append({"u": left["id"], "v": right["id"], "score": float(default_score)})
    return edges
