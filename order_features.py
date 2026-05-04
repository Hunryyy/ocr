import math
from typing import Any, Dict, List


def _bbox(block: Dict[str, Any]) -> List[float]:
    bbox = block.get("bbox", [0, 0, 0, 0])
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]


def _center(bbox: List[float]) -> List[float]:
    return [0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])]


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _iou(a: List[float], b: List[float]) -> float:
    inter_w = _overlap_1d(a[0], a[2], b[0], b[2])
    inter_h = _overlap_1d(a[1], a[3], b[1], b[3])
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, area_a + area_b - inter)


def _same_physical_column_score(a: List[float], b: List[float], page_w: float) -> float:
    aw = max(1.0, a[2] - a[0])
    bw = max(1.0, b[2] - b[0])
    ax = 0.5 * (a[0] + a[2])
    bx = 0.5 * (b[0] + b[2])
    x_overlap = _overlap_1d(a[0], a[2], b[0], b[2]) / max(1.0, min(aw, bw))
    left_align = 1.0 - min(1.0, abs(a[0] - b[0]) / max(0.08 * page_w, min(aw, bw)))
    right_align = 1.0 - min(1.0, abs(a[2] - b[2]) / max(0.08 * page_w, min(aw, bw)))
    center_align = 1.0 - min(1.0, abs(ax - bx) / max(0.1 * page_w, 0.5 * (aw + bw)))
    return max(0.0, min(1.0, 0.4 * x_overlap + 0.25 * left_align + 0.2 * right_align + 0.15 * center_align))


def _reading_flow_score(u: Dict[str, Any], v: Dict[str, Any], a: List[float], b: List[float], page_w: float, page_h: float) -> float:
    ac = _center(a)
    bc = _center(b)
    dx = (bc[0] - ac[0]) / max(1.0, page_w)
    dy = (bc[1] - ac[1]) / max(1.0, page_h)
    same_col = _same_physical_column_score(a, b, page_w)
    same_row = 1.0 if abs(dy) <= 0.035 else 0.0

    if dy < -0.015 and same_row < 0.5:
        return 0.0

    vertical_flow = 0.0
    if dy >= 0:
        vertical_flow = max(0.0, 1.0 - min(1.0, abs(dx) / 0.18)) * max(0.0, 1.0 - min(1.0, dy / 0.55))

    row_flow = 0.0
    if same_row > 0.5 and dx > 0:
        row_flow = max(0.0, 1.0 - min(1.0, dx / 0.45))

    base = max(0.7 * vertical_flow * max(0.2, same_col), 0.55 * row_flow)

    u_type = str(u.get("type", "")).lower()
    v_type = str(v.get("type", "")).lower()
    if u_type in {"figure", "table", "chart"} and v_type == "caption" and 0 <= dy <= 0.12:
        base = max(base, 0.92)
    if u_type == "title" and v_type in {"paragraph", "list_item"} and dy >= 0:
        base = max(base, 0.88)
    if u_type in {"header", "footer", "page_number"} or v_type in {"header", "footer", "page_number"}:
        base *= 0.75

    return max(0.0, min(1.0, base))


def compute_advanced_pair_features(u: Dict[str, Any], v: Dict[str, Any], page: Dict[str, Any]) -> Dict[str, float]:
    page_w = float(max(1.0, page.get("width", 1)))
    page_h = float(max(1.0, page.get("height", 1)))
    bb_u = _bbox(u)
    bb_v = _bbox(v)
    cu = _center(bb_u)
    cv = _center(bb_v)
    dx = (cv[0] - cu[0]) / page_w
    dy = (cv[1] - cu[1]) / page_h
    norm = math.hypot(dx, dy)
    angle = math.atan2(dy, dx) if norm > 1e-6 else 0.0

    return {
        "relative_angle_sin": math.sin(angle),
        "relative_angle_cos": math.cos(angle),
        "bbox_iou": _iou(bb_u, bb_v),
        "center_l1_norm": min(1.0, 0.5 * (abs(dx) + abs(dy))),
        "same_physical_column": _same_physical_column_score(bb_u, bb_v, page_w),
        "reading_flow_score": _reading_flow_score(u, v, bb_u, bb_v, page_w, page_h),
    }
