from typing import Any, Dict, List, Tuple


TEXT_TYPES = {"paragraph", "title", "list_item", "caption", "header", "footer", "page_number"}


def _area(bbox: List[float]) -> float:
    return max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))


def _iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = _area([x1, y1, x2, y2])
    if inter <= 0:
        return 0.0
    union = _area(a) + _area(b) - inter
    return inter / max(1.0, union)


def _containment(inner: List[float], outer: List[float]) -> float:
    inter = _area([
        max(inner[0], outer[0]),
        max(inner[1], outer[1]),
        min(inner[2], outer[2]),
        min(inner[3], outer[3]),
    ])
    return inter / max(1.0, _area(inner))


def _priority(det: Dict[str, Any]) -> Tuple[float, float]:
    score = float(det.get("score", 0.0) or 0.0)
    area = _area(det.get("bbox", [0, 0, 0, 0]))
    label = str(det.get("label", "")).lower()
    label_bonus = 0.02 if label in {"table", "figure", "chart", "formula", "title"} else 0.0
    return score + label_bonus, area


def suppress_nested_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.92,
    containment_threshold: float = 0.94,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    if not detections:
        return [], {"suppressed_nested": 0}

    ordered = sorted(detections, key=_priority, reverse=True)
    kept: List[Dict[str, Any]] = []
    suppressed = 0
    for det in ordered:
        label = str(det.get("label", "")).lower()
        bbox = det.get("bbox", [0, 0, 0, 0])
        drop = False
        for kept_det in kept:
            kept_label = str(kept_det.get("label", "")).lower()
            if label != kept_label and not ({label, kept_label} <= TEXT_TYPES):
                continue
            kept_bbox = kept_det.get("bbox", [0, 0, 0, 0])
            overlap = _iou(bbox, kept_bbox)
            contain = _containment(bbox, kept_bbox)
            if overlap >= iou_threshold or contain >= containment_threshold:
                drop = True
                suppressed += 1
                break
        if not drop:
            kept.append(det)
    kept.sort(key=lambda item: (item.get("bbox", [0, 0, 0, 0])[1], item.get("bbox", [0, 0, 0, 0])[0]))
    return kept, {"suppressed_nested": suppressed}


def refine_title_paragraph_blocks(
    blocks: List[Dict[str, Any]],
    page: Dict[str, Any],
    title_boost_ratio: float = 1.45,
    paragraph_boost_ratio: float = 1.15,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    page_w = float(max(1.0, page.get("width", 1)))
    page_h = float(max(1.0, page.get("height", 1)))
    paragraph_to_title = 0
    title_to_paragraph = 0

    for block in blocks:
        label = str(block.get("type", "")).lower()
        if label not in {"title", "paragraph"}:
            continue

        text = str(block.get("text", "") or "").strip()
        bbox = block.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        rel_w = max(1.0, x2 - x1) / page_w
        rel_h = max(1.0, y2 - y1) / page_h
        text_len = len(text)
        line_count = max(1, text.count("\n") + 1) if text else 0
        has_sentence_punc = any(ch in text for ch in ("。", "，", "；", "：", ";", ",", "!", "！", "?", "？"))

        if label == "paragraph":
            if (
                2 <= text_len <= int(32 * title_boost_ratio)
                and line_count <= 2
                and not has_sentence_punc
                and rel_w <= min(0.78, 0.58 * title_boost_ratio)
                and 0.012 <= rel_h <= 0.12
            ):
                block["type"] = "title"
                paragraph_to_title += 1
        else:
            if (
                text_len >= int(70 / max(1.0, paragraph_boost_ratio))
                or (has_sentence_punc and rel_w >= 0.55)
                or line_count >= 3
                or rel_w >= min(0.95, 0.72 * paragraph_boost_ratio)
            ):
                block["type"] = "paragraph"
                title_to_paragraph += 1

    return blocks, {
        "paragraph_to_title": paragraph_to_title,
        "title_to_paragraph": title_to_paragraph,
    }
