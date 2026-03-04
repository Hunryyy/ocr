import argparse
import json
import os
import sys
import time
import math
import html
import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional deps
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from utils.formula import FormulaRecognizer, normalize_latex, preprocess_formula_image
except Exception:  # pragma: no cover
    FormulaRecognizer = None  # type: ignore[assignment,misc]
    normalize_latex = None  # type: ignore[assignment]
    preprocess_formula_image = None  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# Constants / type sets
# -----------------------------------------------------------------------------
DEFAULT_LAYOUT_CLASSES = [
    "paragraph", "title", "list_item", "caption", "table",
    "figure", "formula", "header", "footer", "chart", "unknown"
]

# Final supported block types (must be subset of label_map expectation)
SUPPORTED_BLOCK_TYPES = set(DEFAULT_LAYOUT_CLASSES)

TEXT_BLOCK_TYPES = {"paragraph", "title", "list_item", "caption", "header", "footer"}
NON_TEXT_BLOCK_TYPES = {"table", "figure", "chart", "formula"}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def safe_median(arr: List[float], default: float = 0.0) -> float:
    if not arr:
        return default
    if np is not None:
        return float(np.median(arr))
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    mid = n // 2
    if n % 2 == 1:
        return float(arr_sorted[mid])
    return float(0.5 * (arr_sorted[mid - 1] + arr_sorted[mid]))


def safe_argmax(seq: List[float]) -> int:
    if not seq:
        return 0
    if np is not None:
        return int(np.argmax(seq))
    return int(max(range(len(seq)), key=lambda i: seq[i]))


def safe_max(seq: List[float]) -> float:
    if not seq:
        return 0.0
    if np is not None:
        return float(np.max(seq))
    return float(max(seq))


def _now_ms() -> float:
    return time.time() * 1000.0


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ensure_dir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass
class ModelBundle:
    # LightGBM
    block_classifier: Any = None
    relation_scorer_order: Any = None
    relation_scorer_caption: Any = None

    # Schemas
    feature_schema_block: List[str] = field(default_factory=list)
    feature_schema_pair: List[str] = field(default_factory=list)
    schema_version_block: Optional[str] = None
    schema_version_pair: Optional[str] = None

    # Label map
    label_map: Dict[int, str] = field(default_factory=dict)

    # OCR
    ocr_engine: Any = None

    # Layout detector (onnx)
    layout_detector: Any = None
    layout_class_map: Dict[int, str] = field(default_factory=dict)
    layout_input_size: int = 1024
    layout_nms_threshold: float = 0.5
    layout_score_threshold: float = 0.3

    # Store cfg and warnings
    cfg: Dict[str, Any] = field(default_factory=dict)
    model_disabled_reason: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cfg(path: str) -> Dict[str, Any]:
    """
    Load YAML/JSON config and apply profile overlay.

    Output is a merged dict; we keep backward-compat with older keys.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    data = None
    if yaml is not None:
        try:
            data = yaml.safe_load(raw)
        except Exception:
            data = None
    if data is None:
        data = json.loads(raw)

    profile_name = data.get("profile", "balanced")
    profiles = data.get("profiles", {}) or {}
    active = profiles.get(profile_name, {}) or {}

    merged: Dict[str, Any] = {}

    def deep_merge(dst, src):
        for k, v in (src or {}).items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                deep_merge(dst[k], v)
            else:
                dst[k] = v

    deep_merge(merged, data)
    deep_merge(merged, active)
    merged["active_profile"] = profile_name

    # Ensure nested dicts exist
    merged.setdefault("io", {})
    merged.setdefault("pipeline", {})
    merged.setdefault("decode", {})
    merged.setdefault("schema", {})
    merged.setdefault("models", {})
    merged.setdefault("fallback_models", merged.get("fallback_models", {}))

    # Backward compat: if fallback models were placed under models, mirror them
    for k in ("layout_detector", "ocr", "table_refiner", "formula_ocr"):
        if k not in merged["fallback_models"] and k in merged["models"]:
            merged["fallback_models"][k] = merged["models"].get(k, {})

    return merged


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def _load_lgb_model(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    if lgb is None:
        return None
    try:
        return lgb.Booster(model_file=path)
    except Exception:
        return None


def _load_schema(path: Optional[str]) -> Tuple[List[str], Optional[str]]:
    if not path or not os.path.exists(path):
        return [], None
    obj = _load_json(path)
    version = None
    if isinstance(obj, dict):
        version = obj.get("schema_version")
        if "features" in obj and isinstance(obj["features"], list):
            feats_sorted = sorted(obj["features"], key=lambda x: x.get("index", 0))
            return [f.get("name", "") for f in feats_sorted], version
    if isinstance(obj, list):
        return [str(x) for x in obj], version
    return [], version


def _load_label_map(path: Optional[str]) -> Dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    obj = _load_json(path)
    if isinstance(obj, list):
        return {i: v for i, v in enumerate(obj)}
    if isinstance(obj, dict):
        if all(isinstance(k, str) and isinstance(v, int) for k, v in obj.items()):
            return {v: k for k, v in obj.items()}
        if all(isinstance(k, (int, str)) and isinstance(v, str) for k, v in obj.items()):
            return {int(k): v for k, v in obj.items()}
    return {}


def _maybe_load_paddle(cfg: Dict[str, Any]):
    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    enabled = cfg.get("pipeline", {}).get("enable_ocr", False) or ocr_cfg.get("enabled", False)
    if not enabled:
        return None
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception:
        return None

    lang = ocr_cfg.get("lang", "ch")
    det = ocr_cfg.get("det", True)
    rec = ocr_cfg.get("rec", True)
    try:
        return PaddleOCR(lang=lang, det=det, rec=rec, use_textline_orientation=True)
    except Exception:
        return None


def _load_layout_class_map(cfg_map: Optional[Dict]) -> Dict[int, str]:
    """
    Handle int->str, str->int, str->str (int keys as strings).
    """
    if not cfg_map:
        return {i: c for i, c in enumerate(DEFAULT_LAYOUT_CLASSES)}
    result: Dict[int, str] = {}
    for k, v in cfg_map.items():
        if isinstance(k, int) and isinstance(v, str):
            result[k] = v
        elif isinstance(k, str) and isinstance(v, int):
            result[v] = k
        elif isinstance(k, str) and isinstance(v, str):
            try:
                result[int(k)] = v
            except Exception:
                continue
    return result if result else {i: c for i, c in enumerate(DEFAULT_LAYOUT_CLASSES)}


def _load_onnx_model(path: Optional[str], debug_reasons: List[str]) -> Any:
    if not path:
        return None
    if not os.path.exists(path):
        debug_reasons.append(f"onnx_model_not_found:{path}")
        return None
    if ort is None:
        debug_reasons.append("onnxruntime_not_installed")
        return None
    try:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        debug_reasons.append(f"onnx_load_error:{str(e)[:120]}")
        return None


def load_artifacts(cfg: Dict[str, Any]) -> ModelBundle:
    """
    Load models + schemas. Prefer cfg['schema'] for schema paths, but keep
    backward compat with cfg['models'] if schemas are there.
    """
    bundle = ModelBundle(cfg=cfg)

    models_cfg = cfg.get("models", {}) or {}
    schema_cfg = cfg.get("schema", {}) or {}
    fallback_cfg = cfg.get("fallback_models", {}) or {}

    # ---- LGB models ----
    bundle.block_classifier = _load_lgb_model((models_cfg.get("block_classifier") or {}).get("path"))
    bundle.relation_scorer_order = _load_lgb_model((models_cfg.get("relation_scorer_order") or {}).get("path"))
    bundle.relation_scorer_caption = _load_lgb_model((models_cfg.get("relation_scorer_caption") or {}).get("path"))

    # ---- schemas ----
    # Preferred (new): cfg['schema']
    block_schema_path = schema_cfg.get("feature_schema_block_path") or ((models_cfg.get("feature_schema_block") or {}).get("path"))
    pair_schema_path = schema_cfg.get("feature_schema_pair_path") or ((models_cfg.get("feature_schema_pair") or {}).get("path"))
    label_map_path = schema_cfg.get("label_map_path") or ((models_cfg.get("label_map") or {}).get("path"))

    bundle.feature_schema_block, bundle.schema_version_block = _load_schema(block_schema_path)
    bundle.feature_schema_pair, bundle.schema_version_pair = _load_schema(pair_schema_path)
    bundle.label_map = _load_label_map(label_map_path)

    # ---- OCR ----
    bundle.ocr_engine = _maybe_load_paddle(cfg)

    # ---- layout detector ----
    layout_cfg = (fallback_cfg.get("layout_detector") or models_cfg.get("layout_detector") or {}) or {}
    if layout_cfg:
        bundle.layout_detector = _load_onnx_model(layout_cfg.get("path"), bundle.model_disabled_reason)
        bundle.layout_class_map = _load_layout_class_map(layout_cfg.get("class_map"))
        bundle.layout_input_size = int(layout_cfg.get("input_size", 1024) or 1024)
        bundle.layout_nms_threshold = float(layout_cfg.get("nms_threshold", 0.5) or 0.5)
        bundle.layout_score_threshold = float(layout_cfg.get("score_threshold", 0.3) or 0.3)
    else:
        bundle.layout_class_map = {i: c for i, c in enumerate(DEFAULT_LAYOUT_CLASSES)}

    # ---- schema checks (strict optional) ----
    strict_schema = bool(cfg.get("pipeline", {}).get("strict_schema_check", False))

    def check_schema(model, schema, name):
        if model is None or not schema:
            return
        try:
            nf = model.num_feature()
            if nf != len(schema):
                msg = f"{name}_schema_len_mismatch:{len(schema)}!=model:{nf}"
                bundle.model_disabled_reason.append(msg)
                if strict_schema:
                    if name == "block":
                        bundle.block_classifier = None
                    elif name == "pair":
                        bundle.relation_scorer_order = None
                        bundle.relation_scorer_caption = None
        except Exception:
            return

    check_schema(bundle.block_classifier, bundle.feature_schema_block, "block")
    check_schema(bundle.relation_scorer_order, bundle.feature_schema_pair, "pair")

    return bundle


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _center(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _overlap_1d(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))


def _bbox_attr(bbox: List[float]) -> str:
    return " ".join(str(int(v)) for v in bbox)


def _clamp_bbox(b: List[float], w: float, h: float, min_size: float = 1.0) -> List[float]:
    x1, y1, x2, y2 = b
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 - x1 < min_size:
        x2 = min(w, x1 + min_size)
    if y2 - y1 < min_size:
        y2 = min(h, y1 + min_size)
    return [x1, y1, x2, y2]


# -----------------------------------------------------------------------------
# NMS / IOU
# -----------------------------------------------------------------------------
def _iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _nms_python(detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
    if not detections:
        return []
    dets = sorted(detections, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best["bbox"], d["bbox"]) < iou_threshold]
    return keep


def _union_bbox(bboxes: List[List[float]]) -> List[float]:
    if not bboxes:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


# -----------------------------------------------------------------------------
# Text stats
# -----------------------------------------------------------------------------
def _text_stats(text: str) -> Dict[str, float]:
    if not text:
        return {
            "len": 0,
            "digit_ratio": 0.0,
            "upper_ratio": 0.0,
            "lower_ratio": 0.0,
            "punct_ratio": 0.0,
            "space_ratio": 0.0,
            "mean_word_len": 0.0,
            "is_alnum": 0.0,
            "ch_ratio": 0.0,
        }
    n = len(text)
    digits = sum(c.isdigit() for c in text)
    upp = sum(c.isupper() for c in text)
    low = sum(c.islower() for c in text)
    punct = sum(c in ".,;:!?\"'()[]{}，。；：！？、（）【】《》" for c in text)
    spaces = sum(c.isspace() for c in text)
    words = [w for w in text.split() if w]
    mean_word_len = sum(len(w) for w in words) / len(words) if words else 0.0
    is_alnum = sum(c.isalnum() for c in text) / n
    ch = sum(0x4E00 <= ord(c) <= 0x9FA5 for c in text)
    ch_ratio = ch / n
    return {
        "len": n,
        "digit_ratio": digits / n,
        "upper_ratio": upp / n,
        "lower_ratio": low / n,
        "punct_ratio": punct / n,
        "space_ratio": spaces / n,
        "mean_word_len": mean_word_len,
        "is_alnum": is_alnum,
        "ch_ratio": ch_ratio,
    }


# -----------------------------------------------------------------------------
# Feature extraction aligned to schema
# -----------------------------------------------------------------------------
def _height_percentiles(blocks: List[Dict[str, Any]]) -> Dict[int, float]:
    if not blocks:
        return {}
    heights = [(max(0.0, b["bbox"][3] - b["bbox"][1]), b["id"]) for b in blocks]
    heights_sorted = sorted(heights, key=lambda x: x[0])
    n = len(heights_sorted)
    rank_map: Dict[int, float] = {}
    for rank, (_, bid) in enumerate(heights_sorted):
        rank_map[bid] = rank / max(1, n - 1) if n > 1 else 0.5
    return rank_map



def _block_feature_dict(block: Dict[str, Any], page: Dict[str, Any], height_pct: float,
                        column_id: float = 0.0, column_count: float = 1.0,
                        is_first_in_column: float = 0.0, is_last_in_column: float = 0.0) -> Dict[str, float]:
    """
    提取 block 特征，与 train.py 的 BLOCK_SCHEMA (29维) 完全对齐
    """
    x1, y1, x2, y2 = block.get("bbox", [0, 0, 0, 0])
    w = max(1.0, page.get("width", 1))
    h = max(1.0, page.get("height", 1))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    txt = block.get("text") or ""
    ts = _text_stats(txt)
    source = (block.get("source") or "").lower()
    src_object = 1.0 if source == "object" else 0.0
    src_ocr = 1.0 if ("ocr" in source) else 0.0
    src_heur = 1.0 if source == "heuristic" or "heur" in source else 0.0
    y_top_region = 1.0 if y1 < 0.1 * h else 0.0
    y_bottom_region = 1.0 if y2 > 0.9 * h else 0.0
    level = 0.0
    if block.get("style") and block["style"].get("heading_level"):
        level = float(block["style"]["heading_level"])
    
    # 计算文本行数���启发式）
    text_line_count = max(1.0, float(txt.count("") + 1)) if txt.strip() else 0.0
    
    # 计算平均行高（归一化到页高）
    if text_line_count > 0 and bh > 0:
        avg_line_height_norm = (bh / text_line_count) / h
    else:
        avg_line_height_norm = 0.0
    
    # 特征字典，顺序必须与 train.py BLOCK_SCHEMA 一致
    feats = {
        # 位置与尺寸（8个）
        "rel_x1": x1 / w,
        "rel_y1": y1 / h,
        "rel_x2": x2 / w,
        "rel_y2": y2 / h,
        "rel_w": bw / w,
        "rel_h": bh / h,
        "area_ratio": _area([x1, y1, x2, y2]) / (w * h),
        "aspect": (bw / bh) if bh > 0 else 0.0,
        
        # 文本统计（8个）
        "text_len": ts["len"],
        "digit_ratio": ts["digit_ratio"],
        "upper_ratio": ts["upper_ratio"],
        "lower_ratio": ts["lower_ratio"],
        "punct_ratio": ts["punct_ratio"],
        "mean_word_len": ts["mean_word_len"],
        "is_alnum": ts["is_alnum"],
        "ch_ratio": ts["ch_ratio"],
        
        # 样式（1个）
        "heading_level": level,
        
        # 来源（3个）
        "src_object": src_object,
        "src_ocr": src_ocr,
        "src_heur": src_heur,
        
        # 位置区域（2个）
        "y_top_region": y_top_region,
        "y_bottom_region": y_bottom_region,
        
        # 高度百分位（1个）
        "height_percentile": height_pct,
        
        # ===== 新增特征（6个）与 train.py 对齐 =====
        "column_id": column_id,
        "column_count": column_count,
        "is_first_in_column": is_first_in_column,
        "is_last_in_column": is_last_in_column,
        "text_line_count": text_line_count,
        "avg_line_height_norm": avg_line_height_norm,
    }
    return feats



def _coarse_type_onehot(b: Dict[str, Any]) -> Dict[str, float]:
    # BUGFIX: original file had a syntax error; keep logic conservative.
    t = (b.get("type") or "").strip()
    if t == "text":
        t = "paragraph"
    if t in ("paragraph", "title", "list_item", "header", "footer", "formula", "unknown"):
        return {"text": 1.0, "table": 0.0, "figure": 0.0, "caption": 0.0, "other": 0.0}
    if t == "table":
        return {"text": 0.0, "table": 1.0, "figure": 0.0, "caption": 0.0, "other": 0.0}
    if t in ("figure", "chart"):
        return {"text": 0.0, "table": 0.0, "figure": 1.0, "caption": 0.0, "other": 0.0}
    if t == "caption":
        return {"text": 0.0, "table": 0.0, "figure": 0.0, "caption": 1.0, "other": 0.0}
    return {"text": 0.0, "table": 0.0, "figure": 0.0, "caption": 0.0, "other": 1.0}



def _pair_feature_dict(b1: Dict[str, Any], b2: Dict[str, Any], page: Dict[str, Any],
                       column_count: float = 1.0) -> Dict[str, float]:
    """
    提取 pair 特征，与 train.py 的 PAIR_SCHEMA (34维) 完全对齐
    """
    bb1 = b1.get("bbox", [0, 0, 0, 0])
    bb2 = b2.get("bbox", [0, 0, 0, 0])
    x11, y11, x12, y12 = bb1
    x21, y21, x22, y22 = bb2
    cx1, cy1 = _center(bb1)
    cx2, cy2 = _center(bb2)
    w = max(1.0, page.get("width", 1))
    h = max(1.0, page.get("height", 1))
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist = math.hypot(dx, dy)
    ovx = _overlap_1d(x11, x12, x21, x22)
    ovy = _overlap_1d(y11, y12, y21, y22)
    bw1, bh1 = max(1.0, x12 - x11), max(1.0, y12 - y11)
    bw2, bh2 = max(1.0, x22 - x21), max(1.0, y22 - y21)
    x_overlap_ratio = ovx / min(bw1, bw2)
    y_overlap_ratio = ovy / min(bh1, bh2)
    size_ratio_w = bw2 / bw1
    size_ratio_h = bh2 / bh1
    align_diff_left_norm = abs(x21 - x11) / w
    align_diff_right_norm = abs(x22 - x12) / w
    same_row = 1.0 if abs(cy1 - cy2) < 0.04 * h else 0.0
    same_col = 1.0 if abs(x11 - x21) < 0.08 * w else 0.0
    left_to_right = 1.0 if (same_row > 0.5 and x11 < x21) else 0.0
    is_above = 1.0 if y12 <= y21 else 0.0
    v_gap = max(0.0, y21 - y12)
    
    # 获取 column_id（从 block 的 meta 或默认 0）
    u_col_id = float(b1.get("_column_id", 0))
    v_col_id = float(b2.get("_column_id", 0))
    same_column_id = 1.0 if u_col_id == v_col_id else 0.0
    column_diff = max(-3.0, min(3.0, v_col_id - u_col_id))
    
    # 获取文本行数
    u_text = (b1.get("text") or "")
    v_text = (b2.get("text") or "")
    u_lines = max(1.0, float(u_text.count("") + 1)) if u_text.strip() else 1.0
    v_lines = max(1.0, float(v_text.count("") + 1)) if v_text.strip() else 1.0
    text_line_count_ratio = (v_lines + 1) / (u_lines + 1)
    
    # 特征字典，顺序必须与 train.py PAIR_SCHEMA 一致
    feats = {
        # 相对位置（3个）
        "dx_norm": dx / w,
        "dy_norm": dy / h,
        "center_dist_norm": dist / math.hypot(w, h),
        
        # 重叠（2个）
        "x_overlap": x_overlap_ratio,
        "y_overlap": y_overlap_ratio,
        
        # 尺寸比例（2个）
        "size_ratio_w": size_ratio_w,
        "size_ratio_h": size_ratio_h,
        
        # 布局关系（6个）
        "same_column": same_col,
        "is_above": is_above,
        "align_diff_left_norm": align_diff_left_norm,
        "align_diff_right_norm": align_diff_right_norm,
        "same_row": same_row,
        "left_to_right": left_to_right,
        
        # 间距（1个）
        "gap_y_norm": v_gap / h,
        
        # 类型特征（4个）
        "u_is_title": 1.0 if b1.get("type") == "title" else 0.0,
        "v_is_title": 1.0 if b2.get("type") == "title" else 0.0,
        "u_heading_level": float(b1.get("style", {}).get("heading_level", 0) if b1.get("style") else 0.0),
        "v_heading_level": float(b2.get("style", {}).get("heading_level", 0) if b2.get("style") else 0.0),
    }
    
    # 类型 one-hot（10个）
    u_one = _coarse_type_onehot(b1)
    v_one = _coarse_type_onehot(b2)
    feats.update({
        "u_text": u_one["text"],
        "u_table": u_one["table"],
        "u_figure": u_one["figure"],
        "u_caption": u_one["caption"],
        "u_other": u_one["other"],
        "v_text": v_one["text"],
        "v_table": v_one["table"],
        "v_figure": v_one["figure"],
        "v_caption": v_one["caption"],
        "v_other": v_one["other"],
    })
    
    # ===== 新增特征（6个）与 train.py 对齐 =====
    feats.update({
        "same_column_id": same_column_id,
        "column_diff": column_diff,
        "u_column_id": u_col_id,
        "v_column_id": v_col_id,
        "column_count": column_count,
        "text_line_count_ratio": text_line_count_ratio,
    })
    
    return feats



def _vectorize(feat_dict: Dict[str, float], schema: List[str], strict: bool = False, warn_missing: bool = True) -> List[float]:
    vec: List[float] = []
    missing: List[str] = []
    for name in schema:
        if name in feat_dict:
            val = feat_dict[name]
            if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                vec.append(0.0)
            else:
                vec.append(float(val))
        else:
            missing.append(name)
            vec.append(0.0)
    if strict and missing:
        raise ValueError(f"Missing {len(missing)} features in schema: {missing[:5]}...")
    if warn_missing and missing and len(missing) > len(schema) * 0.1:
        sys.stderr.write(f"[warn] {len(missing)}/{len(schema)} features missing in vectorization\n")
    return vec


# -----------------------------------------------------------------------------
# OCR cache (full-image OCR; ROI OCR has separate caching)
# -----------------------------------------------------------------------------
def _get_cache_dir(cfg: Dict[str, Any]) -> Optional[str]:
    cache_dir = (cfg.get("io", {}) or {}).get("cache_dir")
    if cache_dir is None:
        cache_dir = cfg.get("cache_dir")
    return cache_dir


def _compute_ocr_cache_key(image_path: str, ocr_cfg: Dict[str, Any], roi_bbox: Optional[List[float]] = None) -> str:
    key_parts = {
        "image_path": image_path,
        "lang": ocr_cfg.get("lang", "ch"),
        "det": bool(ocr_cfg.get("det", True)),
        "rec": bool(ocr_cfg.get("rec", True)),
        "use_gpu": bool(ocr_cfg.get("use_gpu", False)),
        "roi": [round(v, 2) for v in roi_bbox] if roi_bbox is not None else None,
    }
    return _sha256_short(json.dumps(key_parts, sort_keys=True, ensure_ascii=False))


def _get_ocr_cache_path(cache_dir: str, cache_key: str) -> str:
    return os.path.join(cache_dir, f"ocr_{cache_key}.json")


def _load_ocr_cache(cache_path: str) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    if not os.path.exists(cache_path):
        return None, False
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, False
        return None, True
    except Exception:
        return None, True


def _save_ocr_cache(cache_path: str, lines: List[Dict[str, Any]]) -> bool:
    try:
        _ensure_dir(os.path.dirname(cache_path))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(lines, f, ensure_ascii=False)
        return True
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Caption patterns (multi-lingual)
# -----------------------------------------------------------------------------
_CAPTION_PATTERN = re.compile(
    r'(?i)^[\s\[\(]*'
    r'(figure|fig|table|tab|图|表)'
    r'\.?\s*'
    r'(\d{1,4})'
    r'(?:\s*[\(\[]?\s*([a-zA-Z])\s*[\)\]]?)?'
)


def _extract_caption_info(text: str) -> Dict[str, Any]:
    if not text:
        return {"type": None, "number": None, "sub": None}
    m = _CAPTION_PATTERN.match(text.strip())
    if not m:
        return {"type": None, "number": None, "sub": None}
    keyword = (m.group(1) or "").lower()
    number = None
    try:
        number = int(m.group(2))
    except Exception:
        number = None
    sub = (m.group(3) or "").lower() or None

    if keyword in ("figure", "fig", "图"):
        cap_type = "figure"
    elif keyword in ("table", "tab", "表"):
        cap_type = "table"
    else:
        cap_type = None

    return {"type": cap_type, "number": number, "sub": sub}


def _caption_type_matches_target(caption_info: Dict[str, Any], target_type: str) -> bool:
    cap_type = caption_info.get("type")
    if cap_type is None:
        return True
    if cap_type == "figure" and target_type in ("figure", "chart"):
        return True
    if cap_type == "table" and target_type == "table":
        return True
    return False


# -----------------------------------------------------------------------------
# Layout detector helpers
# -----------------------------------------------------------------------------
def _letterbox_resize(img: Any, target_size: int):
    """
    Returns (padded_hwc_float01, scale, (pad_x, pad_y), orig_w, orig_h).
    """
    if Image is None or np is None:
        return None, 1.0, (0, 0), 0, 0
    orig_w, orig_h = img.size
    scale = min(target_size / max(1, orig_w), target_size / max(1, orig_h))
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    padded = np.zeros((target_size, target_size, 3), dtype=np.float32)
    pad_x, pad_y = (target_size - new_w) // 2, (target_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w, :] = np.array(resized, dtype=np.float32) / 255.0
    return padded, scale, (pad_x, pad_y), orig_w, orig_h


def _parse_yolo_output(output: Any, num_classes: int, score_threshold: float) -> List[Dict[str, Any]]:
    """
    Support YOLO-like outputs:
    - [1,N,4+1+num_cls] (cx,cy,w,h,obj,cls...)
    - [1,N,4+num_cls]   (cx,cy,w,h,cls...)
    - [1,N,6]           (cx,cy,w,h,score,cls_id)
    """
    if np is None:
        return []
    arr = np.array(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return []
    dets: List[Dict[str, Any]] = []
    for det in arr:
        det = det.tolist()
        if len(det) >= 5 + num_classes:
            cx, cy, w, h, obj = det[:5]
            cls_scores = det[5:5 + num_classes]
            cls_idx = int(np.argmax(cls_scores))
            score = float(obj * cls_scores[cls_idx])
        elif len(det) >= 4 + num_classes:
            cx, cy, w, h = det[:4]
            cls_scores = det[4:4 + num_classes]
            cls_idx = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_idx])
        elif len(det) >= 6:
            cx, cy, w, h, score, cls_idx = det[:6]
            score, cls_idx = float(score), int(cls_idx)
        else:
            continue
        if score >= score_threshold:
            dets.append({"bbox": [float(cx), float(cy), float(w), float(h)],
                         "label_idx": int(cls_idx), "score": float(score), "format": "cxcywh"})
    return dets


def _parse_fasterrcnn_outputs(outputs: Dict[str, Any], score_threshold: float) -> List[Dict[str, Any]]:
    if np is None:
        return []
    boxes, scores, labels = None, None, None
    for name, val in outputs.items():
        nl = name.lower()
        if "box" in nl:
            boxes = np.array(val)
        elif "score" in nl:
            scores = np.array(val)
        elif "label" in nl or "class" in nl:
            labels = np.array(val)
    if boxes is None:
        return []
    if boxes.ndim == 3:
        boxes = boxes[0]
    if scores is not None and scores.ndim == 2:
        scores = scores[0]
    if labels is not None and labels.ndim == 2:
        labels = labels[0]

    dets: List[Dict[str, Any]] = []
    for i in range(len(boxes)):
        s = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        if s < score_threshold:
            continue
        cls = int(labels[i]) if labels is not None and i < len(labels) else 0
        b = boxes[i].tolist()
        if len(b) < 4:
            continue
        dets.append({"bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                     "label_idx": cls, "score": s, "format": "xyxy"})
    return dets


def _convert_to_xyxy(det: Dict[str, Any], input_size: int) -> List[float]:
    fmt = det.get("format", "cxcywh")
    bbox = det.get("bbox", [0, 0, 0, 0])
    if fmt == "xyxy":
        x1, y1, x2, y2 = bbox
        return [x1, y1, x2, y2]
    cx, cy, w, h = bbox
    # detect normalized vs pixels (heuristic)
    if max(cx, cy, w, h) <= 2.0:
        cx, cy, w, h = cx * input_size, cy * input_size, w * input_size, h * input_size
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _map_to_original_coords(bbox_input_xyxy: List[float], scale: float, pad: Tuple[int, int],
                            orig_w: int, orig_h: int) -> List[float]:
    x1, y1, x2, y2 = bbox_input_xyxy
    pad_x, pad_y = pad
    x1 = (x1 - pad_x) / max(scale, 1e-6)
    y1 = (y1 - pad_y) / max(scale, 1e-6)
    x2 = (x2 - pad_x) / max(scale, 1e-6)
    y2 = (y2 - pad_y) / max(scale, 1e-6)
    return [
        _clip(x1, 0, orig_w),
        _clip(y1, 0, orig_h),
        _clip(x2, 0, orig_w),
        _clip(y2, 0, orig_h),
    ]


def _run_layout_detector(img_path: str, cfg: Dict[str, Any], models: ModelBundle, debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns list of {"bbox":[x1,y1,x2,y2], "label":str, "score":float}
    """
    t0 = _now_ms()
    if models.layout_detector is None:
        debug["layout_detector_status"] = "no_model"
        debug["layout_ms"] = 0.0
        return []
    if Image is None or np is None:
        debug["layout_detector_status"] = "missing_deps"
        debug["layout_ms"] = 0.0
        return []

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        debug["layout_detector_status"] = f"image_error:{str(e)[:80]}"
        debug["layout_ms"] = round(_now_ms() - t0, 2)
        return []

    input_size = int(models.layout_input_size or 1024)
    padded, scale, pad, orig_w, orig_h = _letterbox_resize(img, input_size)
    if padded is None:
        debug["layout_detector_status"] = "preprocess_failed"
        debug["layout_ms"] = round(_now_ms() - t0, 2)
        return []

    tensor = padded.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

    sess = models.layout_detector
    try:
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: tensor})
        output_names = [o.name for o in sess.get_outputs()]
    except Exception as e:
        debug["layout_detector_status"] = f"inference_error:{str(e)[:120]}"
        debug["layout_ms"] = round(_now_ms() - t0, 2)
        return []

    num_classes = max(1, len(models.layout_class_map))
    score_thr = float(models.layout_score_threshold or 0.3)

    dets: List[Dict[str, Any]] = []
    if len(outputs) == 1:
        dets = _parse_yolo_output(outputs[0], num_classes, score_thr)
    else:
        dets = _parse_fasterrcnn_outputs(dict(zip(output_names, outputs)), score_thr)

    results: List[Dict[str, Any]] = []
    for det in dets:
        bbox_in = _convert_to_xyxy(det, input_size)
        bbox = _map_to_original_coords(bbox_in, scale, pad, orig_w, orig_h)
        if bbox[2] - bbox[0] < 5 or bbox[3] - bbox[1] < 5:
            continue
        label = models.layout_class_map.get(int(det["label_idx"]), "unknown")
        if label == "text":
            label = "paragraph"
        if label not in SUPPORTED_BLOCK_TYPES:
            label = "unknown"
        results.append({"bbox": bbox, "label": label, "score": float(det["score"])})

    # NMS (class-agnostic)
    results = _nms_python(results, float(models.layout_nms_threshold or 0.5))

    debug["layout_detector_status"] = "ok"
    debug["layout_detections"] = len(results)
    debug["layout_ms"] = round(_now_ms() - t0, 2)
    return results


# -----------------------------------------------------------------------------
# OCR (full-image + ROI)
# -----------------------------------------------------------------------------
def _ocr_full_image(img_path: str, cfg: Dict[str, Any], models: ModelBundle, debug: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Full-image OCR with caching. Returns lines [{"bbox":[...],"text":...}]
    """
    if debug is None:
        debug = {}
    t0 = _now_ms()

    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    enabled = bool(cfg.get("pipeline", {}).get("enable_ocr", False) or ocr_cfg.get("enabled", False))
    # fine-grained switch
    if not bool(ocr_cfg.get("full_image_ocr_enabled", True)):
        debug["ocr_status"] = "full_image_disabled"
        debug["ocr_ms"] = 0.0
        return []

    if not enabled:
        debug["ocr_cache_hit"] = 0
        debug["ocr_cache_miss"] = 0
        debug["ocr_status"] = "disabled"
        debug["ocr_ms"] = 0.0
        return []

    cache_dir = _get_cache_dir(cfg)
    cache_path = None
    if cache_dir:
        cache_key = _compute_ocr_cache_key(img_path, ocr_cfg, roi_bbox=None)
        cache_path = _get_ocr_cache_path(cache_dir, cache_key)
        cached, corrupt = _load_ocr_cache(cache_path)
        if corrupt:
            debug["ocr_cache_corrupt"] = True
        if cached is not None and not corrupt:
            debug["ocr_cache_hit"] = 1
            debug["ocr_cache_miss"] = 0
            debug["ocr_status"] = "cache"
            debug["ocr_ms"] = round(_now_ms() - t0, 2)
            return cached

    debug["ocr_cache_hit"] = 0
    debug["ocr_cache_miss"] = 1

    if models.ocr_engine is None:
        debug["ocr_status"] = "no_engine"
        debug["ocr_ms"] = round(_now_ms() - t0, 2)
        return []

    try:
        result = models.ocr_engine.ocr(img_path, cls=False)
    except Exception as e:
        debug["ocr_status"] = f"error:{str(e)[:80]}"
        debug["ocr_ms"] = round(_now_ms() - t0, 2)
        return []

    lines: List[Dict[str, Any]] = []
    if result:
        for item in result:
            if not item:
                continue
            for line in item:
                if not line or len(line) < 2:
                    continue
                quad = line[0]
                txt_info = line[1]
                text = txt_info[0] if isinstance(txt_info, (list, tuple)) and txt_info else str(txt_info)
                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
                lines.append({"bbox": bbox, "text": text})

    if cache_path:
        _save_ocr_cache(cache_path, lines)

    debug["ocr_status"] = "ok"
    debug["ocr_ms"] = round(_now_ms() - t0, 2)
    return lines


def _should_do_roi_ocr(block: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    if (block.get("type") or "") not in TEXT_BLOCK_TYPES:
        return False
    bbox = block.get("bbox", [0, 0, 0, 0])
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    if min(w, h) < 8 or w * h < 64:
        return False
    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    force = bool(ocr_cfg.get("force", False))
    if force:
        return True
    text = (block.get("text") or "").strip()
    source = (block.get("source") or "").lower()
    if text and source not in ("heuristic", "layout_detector"):
        return False
    return True


def _ocr_roi(img_path: str, roi_bbox: List[float], cfg: Dict[str, Any], models: ModelBundle,
             page_size: Tuple[int, int], debug: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    ROI OCR with caching. If PaddleOCR supports ndarray input, use it; else
    fallback to temp file.
    """
    if debug is None:
        debug = {}
    if models.ocr_engine is None or Image is None:
        return []

    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    cache_dir = _get_cache_dir(cfg)
    cache_path = None
    if cache_dir:
        key = _compute_ocr_cache_key(img_path, ocr_cfg, roi_bbox=roi_bbox)
        cache_path = _get_ocr_cache_path(cache_dir, key)
        cached, corrupt = _load_ocr_cache(cache_path)
        if corrupt:
            debug["ocr_roi_cache_corrupt"] = True
        if cached is not None and not corrupt:
            debug["ocr_roi_cache_hit"] = debug.get("ocr_roi_cache_hit", 0) + 1
            return cached
        debug["ocr_roi_cache_miss"] = debug.get("ocr_roi_cache_miss", 0) + 1

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return []

    x1, y1, x2, y2 = [int(round(v)) for v in roi_bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(page_size[0], x2), min(page_size[1], y2)
    if x2 <= x1 or y2 <= y1:
        return []

    roi_img = img.crop((x1, y1, x2, y2))
    try:
        if np is not None:
            roi_arr = np.array(roi_img)
            result = models.ocr_engine.ocr(roi_arr, cls=False)
        else:
            # Last resort: temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                roi_img.save(tmp.name)
                tmp_path = tmp.name
            try:
                result = models.ocr_engine.ocr(tmp_path, cls=False)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    except Exception:
        return []

    lines: List[Dict[str, Any]] = []
    if result:
        for item in result:
            if not item:
                continue
            for line in item:
                if not line or len(line) < 2:
                    continue
                quad = line[0]
                txt_info = line[1]
                text = txt_info[0] if isinstance(txt_info, (list, tuple)) and txt_info else str(txt_info)
                xs = [p[0] for p in quad]
                ys = [p[1] for p in quad]
                bbox = [float(min(xs) + x1), float(min(ys) + y1), float(max(xs) + x1), float(max(ys) + y1)]
                lines.append({"bbox": bbox, "text": text})

    if cache_path:
        _save_ocr_cache(cache_path, lines)

    return lines


def _enrich_blocks_with_roi_ocr(blocks: List[Dict[str, Any]], img_path: str, page: Dict[str, Any],
                                cfg: Dict[str, Any], models: ModelBundle, debug: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    ROI OCR on text blocks, returning (blocks, all_lines).
    This is designed to be fast and conservative (skip very small/noisy blocks).
    """
    t0 = _now_ms()
    all_lines: List[Dict[str, Any]] = []
    if models.ocr_engine is None:
        debug["roi_ocr_status"] = "no_engine"
        debug["roi_ocr_ms"] = 0.0
        return blocks, all_lines

    page_size = (int(page.get("width", 1000)), int(page.get("height", 1400)))
    roi_cnt = 0
    for b in blocks:
        if not _should_do_roi_ocr(b, cfg):
            continue
        lines = _ocr_roi(img_path, b.get("bbox", [0, 0, 0, 0]), cfg, models, page_size, debug)
        if lines:
            roi_cnt += 1
            all_lines.extend(lines)
            b["text"] = "\n".join((ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip())
            b["source"] = "roi_ocr"

    debug["roi_ocr_status"] = "ok"
    debug["roi_ocr_count"] = roi_cnt
    debug["roi_ocr_ms"] = round(_now_ms() - t0, 2)
    return blocks, all_lines


# -----------------------------------------------------------------------------
# Line clustering -> blocks (used for OCR-only fallback)
# -----------------------------------------------------------------------------
def _detect_columns(lines: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    if not lines:
        return []
    centers = [(_center(ln["bbox"])[0], i) for i, ln in enumerate(lines)]
    centers.sort()
    xs = [c[0] for c in centers]
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    if not gaps:
        return [lines]
    max_gap, idx = max(gaps, key=lambda x: x[0])
    median_gap = safe_median([g[0] for g in gaps], default=max_gap)
    if max_gap > 1.5 * median_gap:
        thresh = (xs[idx] + xs[idx + 1]) / 2.0
        col1 = [lines[c[1]] for c in centers if c[0] <= thresh]
        col2 = [lines[c[1]] for c in centers if c[0] > thresh]
        return [col1, col2]
    return [lines]


def _merge_lines(lines: List[Dict[str, Any]], bid: int) -> Dict[str, Any]:
    xs, ys, texts = [], [], []
    for ln in lines:
        x1, y1, x2, y2 = ln["bbox"]
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        texts.append(ln.get("text", ""))
    bbox = [min(xs), min(ys), max(xs), max(ys)] if xs and ys else [0, 0, 0, 0]
    text = "\n".join(t for t in texts if t)
    return {
        "id": bid,
        "bbox": bbox,
        "type": "paragraph",
        "score": 1.0,
        "text": text,
        "style": None,
        "source": "ocr_full",
    }


def _group_lines_to_paragraphs(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not lines:
        return []
    cols = _detect_columns(lines)
    blocks: List[Dict[str, Any]] = []
    bid = 0
    for col in cols:
        col_sorted = sorted(col, key=lambda l: (l["bbox"][1], l["bbox"][0]))
        if col_sorted:
            heights = [ln["bbox"][3] - ln["bbox"][1] for ln in col_sorted]
            gaps = [max(0, col_sorted[i + 1]["bbox"][1] - col_sorted[i]["bbox"][3]) for i in range(len(col_sorted) - 1)]
            median_gap = safe_median(gaps, default=8.0)
        else:
            median_gap = 8.0

        para: List[Dict[str, Any]] = []
        for ln in col_sorted:
            if not para:
                para.append(ln)
                continue
            prev = para[-1]
            y_gap = ln["bbox"][1] - prev["bbox"][3]
            x_ov = _overlap_1d(prev["bbox"][0], prev["bbox"][2], ln["bbox"][0], ln["bbox"][2])
            min_w = min(prev["bbox"][2] - prev["bbox"][0], ln["bbox"][2] - ln["bbox"][0])
            if y_gap < 2.5 * median_gap and x_ov > 0.3 * max(1.0, min_w):
                para.append(ln)
            else:
                blocks.append(_merge_lines(para, bid))
                bid += 1
                para = [ln]
        if para:
            blocks.append(_merge_lines(para, bid))
            bid += 1
    return blocks


# -----------------------------------------------------------------------------
# IR building / normalization
# -----------------------------------------------------------------------------
def _normalize_blocks(ir: Dict[str, Any]) -> Dict[str, Any]:
    w = max(1, ir.get("page", {}).get("width", 1))
    h = max(1, ir.get("page", {}).get("height", 1))
    new_blocks: List[Dict[str, Any]] = []
    for idx, b in enumerate(ir.get("blocks", [])):
        b["id"] = idx
        b["bbox"] = _clamp_bbox(b.get("bbox", [0, 0, 0, 0]), w, h)
        # normalize type
        t = b.get("type", "paragraph") or "paragraph"
        if t == "text":
            t = "paragraph"
        if t not in SUPPORTED_BLOCK_TYPES:
            t = "unknown"
        b["type"] = t
        new_blocks.append(b)
    ir["blocks"] = new_blocks
    return ir


def build_ir_candidates(sample: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle, image_root: Optional[str],
                        debug: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build initial IR:
    - If preloaded IR/blocks exist: use them.
    - Else try layout detector (if enabled & available).
    - Else do full-image OCR -> group into paragraph blocks.
    - Else fall back to single full-page block.
    """
    image_path = sample.get("image", "unknown")
    prompt = sample.get("prompt", cfg.get("default_prompt", "")) or ""
    img_path = os.path.join(image_root, image_path) if image_root and not os.path.isabs(image_path) else image_path

    # page size
    w, h = 1000, 1400
    if Image is not None:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            pass

    ir = {
        "page": {"image": image_path, "width": int(w), "height": int(h)},
        "blocks": [],
        "tables": [],
        "relations": {"order_edges": [], "caption_links": [], "heading_parent": []},
        "debug": {"prompt": prompt},
    }

    # Preloaded IR/blocks compatibility
    preloaded = None
    if isinstance(sample.get("ir"), dict) and sample["ir"].get("blocks"):
        preloaded = sample["ir"]["blocks"]
        debug["blocks_source"] = "preloaded_ir"
    elif sample.get("blocks"):
        preloaded = sample.get("blocks")
        debug["blocks_source"] = "preloaded_blocks"

    if preloaded:
        ir["blocks"] = preloaded
        ir = _normalize_blocks(ir)
        return ir

    # Layout detector first (if model exists and config enabled)
    layout_cfg = (cfg.get("fallback_models", {}) or {}).get("layout_detector", {}) or {}
    layout_enabled = bool(layout_cfg.get("enabled", False))  # default false (config-consistent)
    layout_dets: List[Dict[str, Any]] = []
    if layout_enabled and models.layout_detector is not None:
        layout_dets = _run_layout_detector(img_path, cfg, models, debug)

    if layout_dets:
        blocks: List[Dict[str, Any]] = []
        for i, det in enumerate(layout_dets):
            b = {
                "id": i,
                "bbox": det["bbox"],
                "type": det["label"],
                "score": det["score"],
                "text": "",
                "style": None,
                "source": "layout_detector",
            }
            if det["label"] == "formula":
                b["latex"] = ""
            blocks.append(b)
        ir["blocks"] = blocks
        debug["blocks_source"] = "layout_detector"
        ir = _normalize_blocks(ir)
        return ir

    # OCR-only fallback (full image)
    lines = _ocr_full_image(img_path, cfg, models, debug)
    if lines:
        ir["blocks"] = _group_lines_to_paragraphs(lines)
        debug["blocks_source"] = "ocr_full_grouped"
    else:
        ir["blocks"] = [{
            "id": 0,
            "bbox": [0, 0, w, h],
            "type": "paragraph",
            "score": 1.0,
            "text": "",
            "style": None,
            "source": "heuristic",
        }]
        debug["blocks_source"] = "heuristic"

    ir = _normalize_blocks(ir)
    return ir


# -----------------------------------------------------------------------------
# Block type prediction (LightGBM; optional)
# -----------------------------------------------------------------------------
def predict_block_types(ir: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle) -> Dict[str, Any]:
    clf = models.block_classifier
    schema = models.feature_schema_block
    if clf is None or not schema:
        return ir

    page = ir.get("page", {})
    blocks = ir.get("blocks", [])

    # 计算栏信息
    _enrich_blocks_with_column_info(blocks, page)
    height_pct_map = _height_percentiles(blocks)

    feat_vecs: List[List[float]] = []
    feat_dicts: List[Dict[str, float]] = []
    for b in blocks:
        # 先计算栏信息
        col_id = float(b.get("_column_id", 0))
        col_count = float(b.get("_column_count", 1))
        is_first = float(b.get("_is_first_in_column", 0))
        is_last = float(b.get("_is_last_in_column", 0))
        feats = _block_feature_dict(b, page, height_pct_map.get(b["id"], 0.0),
                                    column_id=col_id, column_count=col_count,
                                    is_first_in_column=is_first, is_last_in_column=is_last)
        feat_dicts.append(feats)
        feat_vecs.append(_vectorize(feats, schema))

    if not feat_vecs:
        return ir

    pred = clf.predict(feat_vecs, raw_score=False)
    for i, b in enumerate(blocks):
        prob = pred[i]
        if isinstance(prob, (list, tuple)) or (np is not None and isinstance(prob, np.ndarray)):
            prob_list = list(prob)
            cls_idx = safe_argmax(prob_list)
            score = safe_max(prob_list)
        else:
            cls_idx = int(float(prob) > 0.5)
            score = float(prob)

        label = models.label_map.get(cls_idx, b.get("type", "paragraph"))
        if label == "text":
            label = "paragraph"
        if label not in SUPPORTED_BLOCK_TYPES:
            label = b.get("type", "paragraph")
        # fusion: if layout detector already gave a confident non-text type, keep it
        src = (b.get("source") or "").lower()
        if src == "layout_detector" and b.get("type") in NON_TEXT_BLOCK_TYPES and float(b.get("score", 0)) >= 0.6:
            continue

        b["type"] = label
        b["score"] = float(max(score, float(b.get("score", 0.0) or 0.0)))

    missing = []
    for name in schema:
        if all(name not in fd for fd in feat_dicts):
            missing.append(name)

    ir.setdefault("debug", {})["missing_block_features"] = missing[:20]
    ir["debug"]["schema_version_block"] = models.schema_version_block
    ir["debug"]["model_disabled_reason"] = models.model_disabled_reason[:10]
    ir["blocks"] = blocks
    return ir


# -----------------------------------------------------------------------------
# Candidate successors + order decode
# -----------------------------------------------------------------------------
def _candidate_successors(blocks: List[Dict[str, Any]], page: Dict[str, Any],
                          k: int = 8, max_blocks: int = 300) -> Dict[int, List[int]]:
    """
    Efficient candidate successor generation, optimized for large n.
    Strategy:
    - bucket by y to reduce search space
    - preference: same column downward, same row rightward
    - penalize cross-column jumps
    - special-case header/footer to reduce weird links
    """
    n = len(blocks)
    cand = {i: [] for i in range(n)}
    if n == 0:
        return cand

    h = float(max(1.0, page.get("height", 1)))
    w = float(max(1.0, page.get("width", 1)))

    if n > max_blocks:
        eff_k = max(3, min(int(k), int(900 / max(1, n))))
    else:
        eff_k = int(k)

    centers = [ _center(b.get("bbox", [0,0,0,0])) for b in blocks ]
    types = [ (b.get("type") or "paragraph") for b in blocks ]

    header_ids = {i for i,t in enumerate(types) if t == "header"}
    footer_ids = {i for i,t in enumerate(types) if t == "footer"}

    # detect rough column breaks by x gaps (no sklearn)
    x_centers_sorted = sorted([c[0] for c in centers])
    column_breaks: List[float] = []
    if len(x_centers_sorted) > 10:
        gaps = [x_centers_sorted[i+1] - x_centers_sorted[i] for i in range(len(x_centers_sorted)-1)]
        med_gap = safe_median(gaps, default=0.15*w)
        for i,g in enumerate(gaps):
            if g > max(0.15*w, 2.5*med_gap):
                column_breaks.append((x_centers_sorted[i] + x_centers_sorted[i+1]) * 0.5)
        column_breaks.sort()

    def col_id(cx: float) -> int:
        c = 0
        for br in column_breaks:
            if cx > br:
                c += 1
            else:
                break
        return c

    # bucket by y1
    bucket_h = max(16.0, h / 25.0)
    buckets: Dict[int, List[int]] = {}
    for idx, b in enumerate(blocks):
        y1 = float(b.get("bbox", [0,0,0,0])[1])
        bid = int(y1 // bucket_h)
        buckets.setdefault(bid, []).append(idx)

    for i, b in enumerate(blocks):
        if i in footer_ids:
            continue

        bb = b.get("bbox", [0,0,0,0])
        cx_i, cy_i = centers[i]
        col_i = col_id(cx_i)
        bid = int(float(bb[1]) // bucket_h)

        neighbor_idxs: List[int] = []
        for nb in range(bid - 1, bid + 6):
            neighbor_idxs.extend(buckets.get(nb, []))
        if not neighbor_idxs:
            continue

        seen = set()
        cands_scored: List[Tuple[float, int]] = []
        for j in neighbor_idxs:
            if j == i or j in seen:
                continue
            seen.add(j)

            if j in header_ids and i not in header_ids:
                continue

            bb2 = blocks[j].get("bbox", [0,0,0,0])
            cx_j, cy_j = centers[j]
            col_j = col_id(cx_j)

            same_row = abs(cy_i - cy_j) < 0.04*h
            if cy_j < cy_i - 0.02*h and not same_row:
                continue
            if same_row and cx_j <= cx_i:
                continue

            x_ov = _overlap_1d(bb[0], bb[2], bb2[0], bb2[2])
            min_w = max(1.0, min(bb[2]-bb[0], bb2[2]-bb2[0]))
            x_ovr = x_ov / min_w

            dist = math.hypot(cx_j - cx_i, cy_j - cy_i)
            score = dist

            if x_ovr > 0.4 and cy_j > cy_i:
                score *= 0.3
            if same_row and cx_j > cx_i:
                score *= 0.5
            if col_i != col_j:
                score *= 2.5

            # Encourage figure/table -> caption nearby
            if types[i] in ("figure", "table", "chart") and types[j] == "caption":
                if 0 <= (cy_j - cy_i) <= 0.12*h:
                    score *= 0.35

            cands_scored.append((score, j))

        cands_scored.sort(key=lambda x: x[0])
        cand[i] = [j for _, j in cands_scored[:eff_k]]

    return cand


def _would_cycle(u: int, v: int, out_map: Dict[int, Optional[int]]) -> bool:
    cur = v
    seen = set()
    while cur is not None:
        if cur == u:
            return True
        if cur in seen:
            break
        seen.add(cur)
        cur = out_map.get(cur)
    return False


def _build_sequence_from_chains(out_map: Dict[int, Optional[int]], blocks: List[Dict[str, Any]]) -> List[int]:
    all_nodes = list(out_map.keys())
    in_map = {v: k for k, v in out_map.items() if v is not None}
    heads = [n for n in all_nodes if n not in in_map]

    def pos_key(idx):
        b = blocks[idx]
        x1, y1, _, _ = b.get("bbox", [0, 0, 0, 0])
        return (y1, x1)

    heads.sort(key=pos_key)
    seq: List[int] = []
    for h in heads:
        cur = h
        while cur is not None:
            if cur in seq:
                break
            seq.append(cur)
            cur = out_map.get(cur)
    for n in all_nodes:
        if n not in seq:
            seq.append(n)
    return seq


def _beam_search_order(blocks, cand_succ, score_map, beam_width=3):
    n = len(blocks)
    if n == 0:
        return []
    if n == 1:
        return [0]

    initial_scores = []
    for i in range(n):
        y, x = blocks[i]["bbox"][1], blocks[i]["bbox"][0]
        initial_scores.append((-(y * 10000 + x), i))
    initial_scores.sort()

    beams = []
    for _, i in initial_scores[:beam_width]:
        beams.append({"seq": [i], "score": 0.0, "visited": {i}})

    max_iterations = n * beam_width * 2
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        if all(len(beam["visited"]) == n for beam in beams):
            break

        candidates = []
        for beam in beams:
            if len(beam["visited"]) == n:
                candidates.append(beam)
                continue

            last = beam["seq"][-1]
            succs = cand_succ.get(last, [])
            valid_succs = [j for j in succs if j not in beam["visited"]]
            if not valid_succs:
                remaining = [j for j in range(n) if j not in beam["visited"]]
                if remaining:
                    remaining.sort(key=lambda j: (blocks[j]["bbox"][1], blocks[j]["bbox"][0]))
                    valid_succs = remaining[:1]

            for j in valid_succs:
                edge_score = score_map.get((last, j), 0.5)
                candidates.append({
                    "seq": beam["seq"] + [j],
                    "score": beam["score"] + edge_score,
                    "visited": beam["visited"] | {j}
                })

        if not candidates:
            break
        candidates.sort(key=lambda x: -x["score"])
        beams = candidates[:beam_width]

    if beams:
        best = max(beams, key=lambda x: (len(x["seq"]), x["score"]))
        result = best["seq"]
        for i in range(n):
            if i not in result:
                result.append(i)
        return result

    return list(range(n))


# -----------------------------------------------------------------------------
# Caption matching
# -----------------------------------------------------------------------------
def _match_captions(captions: List[Dict[str, Any]], targets_all: List[Dict[str, Any]],
                    page: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle) -> List[Dict[str, Any]]:
    caption_links: List[Dict[str, Any]] = []
    if not captions or not targets_all:
        return caption_links

    h = float(max(1.0, page.get("height", 1)))
    w = float(max(1.0, page.get("width", 1)))

    cap_infos = [_extract_caption_info(c.get("text", "") or "") for c in captions]
    cap_ids = [c["id"] for c in captions]
    tar_ids = [t["id"] for t in targets_all]
    k_caption = int(cfg.get("decode", {}).get("k_caption", 6) or 6)

    # candidate pairs with constraints
    cand_pairs: List[Tuple[int, int]] = []
    for ci, c in enumerate(captions):
        c_center = _center(c["bbox"])
        c_info = cap_infos[ci]
        scored: List[Tuple[float, int]] = []
        for tj, t in enumerate(targets_all):
            t_type = t.get("type", "figure")
            if not _caption_type_matches_target(c_info, t_type):
                continue

            t_center = _center(t["bbox"])
            dy = c_center[1] - t_center[1]
            dx = abs(c_center[0] - t_center[0])
            dist = math.hypot(c_center[0] - t_center[0], c_center[1] - t_center[1])

            # Direction constraints: figure/chart captions should not be far above.
            if t_type in ("figure", "chart"):
                if c_center[1] < t["bbox"][1] - 0.05*h:
                    continue
                if dy < 0:
                    continue  # conservative: require below or same band
            elif t_type == "table":
                if abs(dy) > 0.25*h:
                    continue

            t_width = max(1.0, t["bbox"][2] - t["bbox"][0])
            if dx > 1.5*t_width and dx > 0.2*w:
                continue

            scored.append((dist, tj))

        scored.sort(key=lambda x: x[0])
        for _, tj in scored[:k_caption]:
            cand_pairs.append((ci, tj))

    if not cand_pairs:
        return caption_links

    feat_vecs: List[List[float]] = []
    for ci, tj in cand_pairs:
        feats = _pair_feature_dict(captions[ci], targets_all[tj], page, column_count=float(page.get("_column_count", 1)))
        feat_vecs.append(_vectorize(feats, models.feature_schema_pair, warn_missing=False))

    scores: List[float] = []
    use_model = bool(feat_vecs and models.relation_scorer_caption is not None and models.feature_schema_pair)
    if use_model:
        try:
            pred = models.relation_scorer_caption.predict(feat_vecs, raw_score=False)
            for p in pred:
                if isinstance(p, (list, tuple)) or (np is not None and isinstance(p, np.ndarray)):
                    scores.append(safe_max(list(p)))
                else:
                    scores.append(float(p))
        except Exception:
            use_model = False

    if not use_model:
        for (ci, tj) in cand_pairs:
            c = captions[ci]
            t = targets_all[tj]
            feats = _pair_feature_dict(c, t, page, column_count=float(page.get("_column_count", 1)))
            dist_norm = feats.get("center_dist_norm", 0.5)
            base = 1.0 / (1.0 + dist_norm * 3.0)

            c_info = cap_infos[ci]
            if c_info.get("type") and _caption_type_matches_target(c_info, t.get("type", "figure")):
                base *= 1.3

            dy = _center(c["bbox"])[1] - _center(t["bbox"])[1]
            if t.get("type") in ("figure", "chart") and 0 < dy < 0.1*h:
                base *= 1.4
            if t.get("type") == "table" and abs(dy) < 0.08*h:
                base *= 1.2

            x_ov = _overlap_1d(c["bbox"][0], c["bbox"][2], t["bbox"][0], t["bbox"][2])
            min_w = max(1.0, min(c["bbox"][2]-c["bbox"][0], t["bbox"][2]-t["bbox"][0]))
            if x_ov / min_w > 0.5:
                base *= 1.2

            scores.append(float(min(1.0, base)))

    m, n = len(captions), len(targets_all)
    large = 1.0
    cost = [[large for _ in range(n)] for _ in range(m)]
    for idx, (ci, tj) in enumerate(cand_pairs):
        cost[ci][tj] = 1.0 - float(scores[idx])

    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost)
        assigns = list(zip(row_ind, col_ind))
    else:
        used_t = set()
        assigns = []
        for ci in range(m):
            best_tj = None
            best_cost = large
            for tj in range(n):
                if tj in used_t:
                    continue
                if cost[ci][tj] < best_cost:
                    best_cost = cost[ci][tj]
                    best_tj = tj
            if best_tj is not None and best_cost < large:
                used_t.add(best_tj)
                assigns.append((ci, best_tj))

    cap_thr = float(cfg.get("decode", {}).get("caption_score_threshold", 0.25) or 0.25)
    for ci, tj in assigns:
        cst = cost[ci][tj]
        if cst >= large - 1e-6:
            continue
        s = 1.0 - cst
        if s < cap_thr:
            continue

        c = captions[ci]
        t = targets_all[tj]
        cap_y = _center(c["bbox"])[1]
        if t.get("type") in ("figure", "chart") and cap_y < t["bbox"][1] - 0.05*h:
            continue

        dist = math.hypot(_center(c["bbox"])[0] - _center(t["bbox"])[0], cap_y - _center(t["bbox"])[1])
        if dist / math.hypot(w, h) > 0.3:
            continue

        caption_links.append({
            "caption_id": cap_ids[ci],
            "target_id": tar_ids[tj],
            "score": float(s),
        })

    return caption_links
# -----------------------------------------------------------------------------
# Table structure extraction (heuristic grid; rowspan/colspan=1)
# -----------------------------------------------------------------------------
def _cluster_by_gaps(values: List[float], threshold: float) -> List[List[int]]:
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    groups: List[List[int]] = []
    cur: List[int] = [indexed[0][0]]
    for i in range(1, len(indexed)):
        if indexed[i][1] - indexed[i - 1][1] > threshold:
            groups.append(cur)
            cur = []
        cur.append(indexed[i][0])
    if cur:
        groups.append(cur)
    return groups


def _lines_in_bbox(lines: List[Dict[str, Any]], bbox: List[float]) -> List[Dict[str, Any]]:
    x1, y1, x2, y2 = bbox
    out = []
    for ln in lines:
        bb = ln.get("bbox", [0, 0, 0, 0])
        cx = 0.5 * (bb[0] + bb[2])
        cy = 0.5 * (bb[1] + bb[3])
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            out.append(ln)
    return out


def _extract_table_structure(table_block: Dict[str, Any], table_lines: List[Dict[str, Any]], page: Dict[str, Any]) -> Dict[str, Any]:
    table_bbox = table_block.get("bbox", [0, 0, 0, 0])
    table_id = int(table_block.get("id", 0))
    score = float(table_block.get("score", 1.0) or 1.0)

    if not table_lines:
        return {
            "id": table_id,
            "bbox": table_bbox,
            "type": "table",
            "score": score,
            "source": "table_heuristic",
            "rows": [[{"bbox": table_bbox, "text": "", "rowspan": 1, "colspan": 1}]],
        }

    # Row clustering by y-centers
    y_centers = [(ln["bbox"][1] + ln["bbox"][3]) / 2 for ln in table_lines]
    heights = [max(1.0, ln["bbox"][3] - ln["bbox"][1]) for ln in table_lines]
    row_thr = max(6.0, safe_median(heights, 20.0) * 0.7)
    row_groups = _cluster_by_gaps(y_centers, row_thr)
    row_groups.sort(key=lambda g: sum(y_centers[i] for i in g) / max(1, len(g)))

    rows: List[List[Dict[str, Any]]] = []
    for rg in row_groups:
        row_lines = [table_lines[i] for i in rg]
        if len(row_lines) == 1:
            ln = row_lines[0]
            rows.append([{
                "bbox": ln["bbox"],
                "text": (ln.get("text") or "").strip(),
                "rowspan": 1,
                "colspan": 1,
            }])
            continue

        # Column clustering within row by x-centers
        x_centers = [(ln["bbox"][0] + ln["bbox"][2]) / 2 for ln in row_lines]
        widths = [max(1.0, ln["bbox"][2] - ln["bbox"][0]) for ln in row_lines]
        col_thr = max(20.0, safe_median(widths, 50.0) * 0.5)
        col_groups = _cluster_by_gaps(x_centers, col_thr)
        col_groups.sort(key=lambda g: sum(x_centers[i] for i in g) / max(1, len(g)))

        row_cells: List[Dict[str, Any]] = []
        for cg in col_groups:
            cell_lines = [row_lines[i] for i in cg]
            cell_bbox = _union_bbox([ln["bbox"] for ln in cell_lines])
            cell_text = " ".join((ln.get("text") or "").strip() for ln in cell_lines if (ln.get("text") or "").strip())
            row_cells.append({
                "bbox": cell_bbox,
                "text": cell_text,
                "rowspan": 1,
                "colspan": 1
            })
        rows.append(row_cells)

    if not rows:
        rows = [[{"bbox": table_bbox, "text": "", "rowspan": 1, "colspan": 1}]]

    return {
        "id": table_id,
        "bbox": table_bbox,
        "type": "table",
        "score": score,
        "source": "table_heuristic",
        "rows": rows,
    }


def _extract_tables(blocks: List[Dict[str, Any]], ocr_lines: List[Dict[str, Any]], img_path: str,
                    page: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle, debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    t0 = _now_ms()
    page_size = (int(page.get("width", 1000)), int(page.get("height", 1400)))

    tables: List[Dict[str, Any]] = []
    for b in blocks:
        if b.get("type") != "table":
            continue
        table_bbox = b.get("bbox", [0, 0, 0, 0])

        # Prefer lines already collected by ROI OCR; if none, OCR the table ROI
        table_lines = _lines_in_bbox(ocr_lines, table_bbox)
        if not table_lines and models.ocr_engine is not None:
            table_lines = _ocr_roi(img_path, table_bbox, cfg, models, page_size, debug)

        tables.append(_extract_table_structure(b, table_lines, page))

    debug["tables_extracted"] = len(tables)
    debug["table_ms"] = round(_now_ms() - t0, 2)
    return tables


# -----------------------------------------------------------------------------
# Formula handling: multi-engine recognition with fallback
# -----------------------------------------------------------------------------
def _process_formula_blocks(
    blocks: List[Dict[str, Any]],
    img_path: Optional[str] = None,
    page: Optional[Dict[str, Any]] = None,
    models: Optional[Any] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Populate ``latex`` field for every formula block.

    When *img_path* and *models* are provided the function crops each formula
    ROI from the page image, pre-processes it, then runs :class:`FormulaRecognizer`
    (RapidLatexOCR → pix2tex → PaddleOCR fallback).  The recognised string is
    normalised with :func:`normalize_latex` before being stored.

    When no image / recognizer is available the field is left as an empty string
    so downstream rendering remains safe.
    """
    # Build a recognizer if the utils module is available
    recognizer = None
    if FormulaRecognizer is not None and models is not None:
        ocr_engine = getattr(models, "ocr_engine", None)
        recognizer = FormulaRecognizer(ocr_engine=ocr_engine, cfg=cfg or {})

    page = page or {}
    page_w = int(page.get("width", 0) or 0)
    page_h = int(page.get("height", 0) or 0)

    # Open the page image once (reused for all formula crops)
    pil_page = None
    if recognizer is not None and img_path and Image is not None:
        try:
            pil_page = Image.open(img_path).convert("RGB")
        except Exception:
            pil_page = None

    for b in blocks:
        if b.get("type") != "formula":
            continue
        # Keep existing non-empty latex (e.g. from upstream label)
        existing = b.get("latex")
        if existing:
            if normalize_latex is not None:
                b["latex"] = normalize_latex(str(existing))
            continue

        # Try to recognise from image ROI
        latex = ""
        if recognizer is not None and pil_page is not None and np is not None:
            try:
                bbox = b.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    if page_w > 0:
                        x2 = min(page_w, x2)
                    if page_h > 0:
                        y2 = min(page_h, y2)
                    if x2 > x1 and y2 > y1:
                        roi = pil_page.crop((x1, y1, x2, y2))
                        roi_arr = np.array(roi)
                        latex = recognizer.recognize(roi_arr)
            except Exception:
                latex = ""

        b["latex"] = latex if latex else ""

    return blocks


# -----------------------------------------------------------------------------
# Heading parent building (required; decode() depends on it)
# -----------------------------------------------------------------------------
def _build_heading_parent(ir: Dict[str, Any]) -> None:
    """
    Build heading_parent edges for title blocks using stack by heading_level.
    If heading_level missing, estimate from heuristics.
    Order of titles is based on order_edges; fallback to geometric order.
    """
    blocks = ir.get("blocks", [])
    rel = ir.setdefault("relations", {})
    rel.setdefault("heading_parent", [])

    if not blocks:
        rel["heading_parent"] = []
        return

    page = ir.get("page", {}) or {}
    page_h = float(max(1.0, page.get("height", 1)))

    id_to_block = {b["id"]: b for b in blocks}
    order_edges = rel.get("order_edges", []) or []

    # Build order sequence from edges (robust topo-like chain build)
    next_map = {}
    indeg = {b["id"]: 0 for b in blocks}
    for e in order_edges:
        u = e.get("u")
        v = e.get("v")
        if u in indeg and v in indeg:
            next_map[u] = v
            indeg[v] = indeg.get(v, 0) + 1

    heads = [bid for bid, d in indeg.items() if d == 0]
    heads.sort(key=lambda bid: (id_to_block[bid]["bbox"][1], id_to_block[bid]["bbox"][0]))

    order_seq: List[int] = []
    for h in heads:
        cur = h
        while cur is not None and cur not in order_seq:
            order_seq.append(cur)
            cur = next_map.get(cur)
    for bid in id_to_block:
        if bid not in order_seq:
            order_seq.append(bid)

    title_ids = [bid for bid in order_seq if id_to_block.get(bid, {}).get("type") == "title"]
    if not title_ids:
        rel["heading_parent"] = []
        return

    height_pct = _height_percentiles(blocks)

    def estimate_level(b: Dict[str, Any]) -> int:
        if b.get("style") and b["style"].get("heading_level"):
            try:
                return int(b["style"]["heading_level"])
            except Exception:
                pass
        bid = b["id"]
        bbox = b.get("bbox", [0, 0, 0, 0])
        txt = (b.get("text") or "").strip()
        block_h = max(1.0, bbox[3] - bbox[1])
        block_h_ratio = block_h / page_h
        y_pos = bbox[1] / page_h
        pct = float(height_pct.get(bid, 0.5))

        score = 0.0
        score += pct * 3.0
        if block_h_ratio > 0.05:
            score += 1.0
        elif block_h_ratio > 0.03:
            score += 0.5
        if y_pos < 0.15:
            score += 0.8
        elif y_pos < 0.3:
            score += 0.3
        if len(txt) < 20:
            score += 0.5
        elif len(txt) < 50:
            score += 0.2

        if score > 4.0:
            return 1
        if score > 3.0:
            return 2
        if score > 2.0:
            return 3
        if score > 1.2:
            return 4
        if score > 0.6:
            return 5
        return 6

    title_levels = {bid: estimate_level(id_to_block[bid]) for bid in title_ids}

    edges: List[Dict[str, Any]] = []
    stack: List[Tuple[int, int]] = []  # (id, level)
    for bid in title_ids:
        lvl = int(title_levels[bid])
        while stack and stack[-1][1] >= lvl:
            stack.pop()
        if stack and stack[-1][0] != bid:
            edges.append({"child_id": bid, "parent_id": stack[-1][0], "score": 1.0})
        stack.append((bid, lvl))

    rel["heading_parent"] = edges


# -----------------------------------------------------------------------------
# Relation prediction
# -----------------------------------------------------------------------------
def _add_heuristic_targets(ir: Dict[str, Any]) -> None:
    blocks = ir.get("blocks", [])
    page = ir.get("page", {})
    page_area = max(1.0, float(page.get("width", 1)) * float(page.get("height", 1)))
    has_target = any(b.get("type") in ("figure", "table", "chart") for b in blocks)
    if has_target or not blocks:
        return

    cand = None
    max_area = 0.0
    for b in blocks:
        txt_len = len((b.get("text") or "").strip())
        area = _area(b.get("bbox", [0, 0, 0, 0]))
        if txt_len <= 10 and area / page_area > 0.2 and area > max_area:
            cand = b
            max_area = area
    if cand is None:
        return

    new_id = len(blocks)
    blocks.append({
        "id": new_id,
        "bbox": cand["bbox"],
        "type": "figure",
        "score": 0.4,
        "text": "",
        "style": None,
        "source": "heuristic_figure",
    })
    ir["blocks"] = blocks


def predict_relations(ir: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle) -> Dict[str, Any]:
    """
    Predict reading order and caption links.
    - candidate edges from _candidate_successors
    - score edges using LGB or heuristic
    - greedy chain with constraints; optional beam search for small pages
    """
    blocks = ir.get("blocks", [])
    if not blocks:
        ir["relations"] = {"order_edges": [], "caption_links": [], "heading_parent": []}
        return ir

    page = ir.get("page", {}) or {}
    _add_heuristic_targets(ir)
    blocks = ir.get("blocks", [])
    n = len(blocks)

    types = [b.get("type", "paragraph") for b in blocks]
    header_idx = {i for i, t in enumerate(types) if t == "header"}
    footer_idx = {i for i, t in enumerate(types) if t == "footer"}
    caption_idx = {i for i, t in enumerate(types) if t == "caption"}
    target_idx = {i for i, t in enumerate(types) if t in ("figure", "table", "chart")}

    max_blocks_cfg = int(cfg.get("pipeline", {}).get("max_blocks", 300) or 300)
    k_order = int(cfg.get("decode", {}).get("k_order", 8) or 8)
    cand_succ = _candidate_successors(blocks, page, k=k_order, max_blocks=max_blocks_cfg)

    feat_vecs: List[List[float]] = []
    pair_indices: List[Tuple[int, int]] = []
    for i, succs in cand_succ.items():
        for j in succs:
            feats = _pair_feature_dict(blocks[i], blocks[j], page, column_count=float(page.get("_column_count", 1)))
            feat_vecs.append(_vectorize(feats, models.feature_schema_pair, warn_missing=False))
            pair_indices.append((i, j))

    score_map: Dict[Tuple[int, int], float] = {}
    use_model = bool(feat_vecs and models.relation_scorer_order is not None and models.feature_schema_pair)
    scores: List[float] = []

    if use_model:
        try:
            pred = models.relation_scorer_order.predict(feat_vecs, raw_score=False)
            for p in pred:
                if isinstance(p, (list, tuple)) or (np is not None and isinstance(p, np.ndarray)):
                    scores.append(float(safe_max(list(p))))
                else:
                    scores.append(float(p))
        except Exception:
            use_model = False

    if not use_model:
        for (i, j) in pair_indices:
            feats = _pair_feature_dict(blocks[i], blocks[j], page, column_count=float(page.get("_column_count", 1)))
            dist = float(feats.get("center_dist_norm", 0.5))
            base = 1.0 / (1.0 + dist * 2.0)
            if feats.get("same_column", 0) > 0.5 and feats.get("is_above", 0) > 0.5:
                base *= 1.5
            if feats.get("same_row", 0) > 0.5 and feats.get("left_to_right", 0) > 0.5:
                base *= 1.3
            scores.append(float(min(1.0, base)))

    for idx, (i, j) in enumerate(pair_indices):
        score_map[(i, j)] = float(scores[idx])

    # Optional beam search on small pages
    use_beam_search = bool(cfg.get("decode", {}).get("use_beam_search", False))
    beam_width = int(cfg.get("decode", {}).get("beam_width", 3) or 3)

    if use_beam_search and n <= 50:
        order_seq_idx = _beam_search_order(blocks, cand_succ, score_map, beam_width=beam_width)
    else:
        # constrained greedy chain
        def apply_constraints(i: int, j: int, s: float) -> float:
            if s <= 0:
                return 0.0
            ci = _center(blocks[i]["bbox"])
            cj = _center(blocks[j]["bbox"])
            page_h = float(max(1.0, page.get("height", 1)))

            # no body -> header
            if j in header_idx and i not in header_idx:
                return 0.0
            # no footer -> earlier body
            if i in footer_idx and j not in footer_idx:
                if cj[1] < ci[1]:
                    return 0.0

            score = s

            # Encourage target->caption nearby
            if i in target_idx and j in caption_idx:
                if 0 <= (cj[1] - ci[1]) <= 0.12 * page_h:
                    score *= 1.6

            # Discourage skipping a closer caption after a target
            if i in target_idx and j not in caption_idx and cand_succ.get(i):
                for cap in caption_idx:
                    if cap in cand_succ.get(i, []):
                        cap_c = _center(blocks[cap]["bbox"])
                        if cap_c[1] >= ci[1] and cap_c[1] <= cj[1]:
                            score *= 0.75
                            break

            return float(score)

        edges_scored: List[Tuple[int, int, float]] = []
        for (i, j) in pair_indices:
            s = apply_constraints(i, j, score_map.get((i, j), 0.0))
            if s > 0:
                edges_scored.append((i, j, s))
        edges_scored.sort(key=lambda x: -x[2])

        out_map: Dict[int, Optional[int]] = {i: None for i in range(n)}
        in_map: Dict[int, Optional[int]] = {i: None for i in range(n)}

        for i, j, s in edges_scored:
            if out_map[i] is not None or in_map[j] is not None:
                continue
            if _would_cycle(i, j, out_map):
                continue
            out_map[i] = j
            in_map[j] = i

        order_seq_idx = _build_sequence_from_chains(out_map, blocks)

        # enforce header first, footer last
        headers = [i for i in order_seq_idx if i in header_idx]
        footers = [i for i in order_seq_idx if i in footer_idx]
        body = [i for i in order_seq_idx if i not in header_idx and i not in footer_idx]

        def pos_key(i: int):
            bb = blocks[i].get("bbox", [0, 0, 0, 0])
            return (bb[1], bb[0])

        headers.sort(key=pos_key)
        footers.sort(key=pos_key)
        order_seq_idx = headers + body + footers

    order_edges: List[Dict[str, Any]] = []
    for a, b in zip(order_seq_idx[:-1], order_seq_idx[1:]):
        score = score_map.get((a, b), 1.0)
        order_edges.append({"u": blocks[a]["id"], "v": blocks[b]["id"], "score": float(score)})

    expected_edges = max(0, n - 1)
    if len(order_edges) != expected_edges:
        sorted_idx = sorted(range(n), key=lambda i: (blocks[i]["bbox"][1], blocks[i]["bbox"][0]))
        order_edges = [{"u": blocks[a]["id"], "v": blocks[b]["id"], "score": 1.0} for a, b in zip(sorted_idx[:-1], sorted_idx[1:])]

    captions = [b for b in blocks if b.get("type") == "caption"]
    targets_all = [b for b in blocks if b.get("type") in ("figure", "table", "chart")]
    caption_links = _match_captions(captions, targets_all, page, cfg, models)

    ir.setdefault("relations", {})
    ir["relations"]["order_edges"] = order_edges
    ir["relations"]["caption_links"] = caption_links
    ir["relations"].setdefault("heading_parent", [])
    ir["relations"]["order_is_chain"] = (len(order_edges) == expected_edges)
    return ir


# -----------------------------------------------------------------------------
# Decode: ensure heading parent + infer heading levels for titles
# -----------------------------------------------------------------------------
def decode(ir: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    _build_heading_parent(ir)

    blocks = ir.get("blocks", [])
    heading_edges = (ir.get("relations", {}) or {}).get("heading_parent", []) or []
    tree = {e["child_id"]: e["parent_id"] for e in heading_edges if "child_id" in e and "parent_id" in e}

    def depth(node: int, memo: Dict[int, int]) -> int:
        if node in memo:
            return memo[node]
        p = tree.get(node)
        if p is None or p == node:
            memo[node] = 1
        else:
            memo[node] = min(depth(p, memo) + 1, 6)
        return memo[node]

    memo: Dict[int, int] = {}
    height_pct = _height_percentiles(blocks)
    page_h = float(max(1.0, (ir.get("page", {}) or {}).get("height", 1)))

    for b in blocks:
        if b.get("type") != "title":
            continue

        level = None
        if b.get("style") and b["style"].get("heading_level"):
            level = b["style"]["heading_level"]
        elif b["id"] in tree:
            level = depth(b["id"], memo)
        else:
            pct = float(height_pct.get(b["id"], 0.5))
            bbox = b.get("bbox", [0, 0, 0, 0])
            block_h_ratio = (bbox[3] - bbox[1]) / page_h
            y_pos = bbox[1] / page_h
            if pct > 0.85 or (y_pos < 0.1 and block_h_ratio > 0.04):
                level = 1
            elif pct > 0.7 or (y_pos < 0.2 and block_h_ratio > 0.03):
                level = 2
            elif pct > 0.5:
                level = 3
            elif pct > 0.3:
                level = 4
            else:
                level = 5

        level = int(max(1, min(6, int(level or 2))))
        if b.get("style") is None:
            b["style"] = {}
        b["style"]["heading_level"] = level

    ir["blocks"] = blocks
    return ir


# -----------------------------------------------------------------------------
# Rendering HTML (debug artifact)
# -----------------------------------------------------------------------------

# ============================================================================
# HTML 渲染模块 (已按评测要求重写)
# ============================================================================

def _escape(txt: str) -> str:
    """HTML 转义"""
    return html.escape(txt or "")


def _bbox_attr(bbox: List[float]) -> str:
    """格式化 bbox 为空格分隔的整数字符串"""
    if not bbox or len(bbox) < 4:
        return "0 0 0 0"
    return " ".join(str(int(round(v))) for v in bbox[:4])


def _render_table_cell(cell: Dict[str, Any]) -> str:
    """渲染单个表格单元格"""
    bbox_str = _bbox_attr(cell.get("bbox", []))
    rowspan = int(cell.get("rowspan", 1) or 1)
    colspan = int(cell.get("colspan", 1) or 1)
    text = _escape(cell.get("text", "") or "")
    
    attrs = [f'data-bbox="{bbox_str}"']
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    
    return f'<td {" ".join(attrs)}>{text}</td>'


def _render_table_content(table_obj: Dict[str, Any]) -> str:
    """
    渲染表格内部结构 (thead + tbody)
    
    评测要求: <table><thead>...</thead><tbody>...</tbody></table>
    """
    rows = table_obj.get("rows", [])
    
    if not rows:
        # 空表格: 至少输出一个空行
        return "<table><thead><tr><td></td></tr></thead><tbody></tbody></table>"
    
    # 分离表头和表体
    # 策略: 第一行作为表头，其余作为表体
    thead_rows = rows[:1] if rows else []
    tbody_rows = rows[1:] if len(rows) > 1 else []
    
    parts = ["<table>"]
    
    # 渲染 thead
    parts.append("<thead>")
    for row in thead_rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(_render_table_cell(cell))
        parts.append("</tr>")
    if not thead_rows:
        parts.append("<tr><td></td></tr>")
    parts.append("</thead>")
    
    # 渲染 tbody
    parts.append("<tbody>")
    for row in tbody_rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(_render_table_cell(cell))
        parts.append("</tr>")
    parts.append("</tbody>")
    
    parts.append("</table>")
    return "".join(parts)


def _render_block(b: Dict[str, Any], caption_ref: Optional[int], 
                  table_obj: Optional[Dict[str, Any]], 
                  caption_links: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    渲染单个 block 为 HTML
    
    评测格式要求:
    - 标题: <h2 data-bbox="x1 y1 x2 y2">文本</h2>
    - 段落: <p data-bbox="x1 y1 x2 y2">文本</p>
    - 图像: <div class="image" data-bbox="x1 y1 x2 y2"></div>
    - 图表: <div class="chart" data-bbox="x1 y1 x2 y2"></div>
    - 表格: <div class="table" data-bbox="x1 y1 x2 y2"><table>...</table></div>
    - 公式: <div class="formula" data-bbox="x1 y1 x2 y2">LATEX</div>
    - 列表项: <div class="list_item" data-bbox="x1 y1 x2 y2">文本</div>
    - 标题说明: <div class="caption" data-bbox="x1 y1 x2 y2">文本</div>
    - 页眉: <div class="header" data-bbox="x1 y1 x2 y2">文本</div>
    - 页脚: <div class="footer" data-bbox="x1 y1 x2 y2">文本</div>
    """
    btype = (b.get("type") or "paragraph").lower().strip()
    bbox_str = _bbox_attr(b.get("bbox", []))
    text = _escape(b.get("text") or "")
    
    # 标题 -> 统一使用 h2
    if btype == "title":
        return f'<h2 data-bbox="{bbox_str}">{text}</h2>'
    
    # 段落
    if btype == "paragraph":
        return f'<p data-bbox="{bbox_str}">{text}</p>'
    
    # 列表项
    if btype == "list_item":
        return f'<div class="list_item" data-bbox="{bbox_str}">{text}</div>'
    
    # 标题说明 (caption)
    if btype == "caption":
        # 可选: 添加 data-ref 属性指向关联的 figure/table
        ref_attr = ""
        if caption_ref is not None:
            ref_attr = f' data-ref="{caption_ref}"'
        return f'<div class="caption" data-bbox="{bbox_str}"{ref_attr}>{text}</div>'
    
    # 图像 (figure -> image)
    if btype == "figure":
        return f'<div class="image" data-bbox="{bbox_str}"></div>'
    
    # 图表
    if btype == "chart":
        return f'<div class="chart" data-bbox="{bbox_str}"></div>'
    
    # 表格
    if btype == "table":
        table_content = _render_table_content(table_obj or {})
        return f'<div class="table" data-bbox="{bbox_str}">{table_content}</div>'
    
    # 公式 – LaTeX as element text content (no data-latex attribute)
    if btype == "formula":
        latex = b.get("latex") or ""
        latex = normalize_latex(latex) if normalize_latex is not None else latex.strip()
        latex_escaped = _escape(latex)
        return f'<div class="formula" data-bbox="{bbox_str}">{latex_escaped}</div>'
    
    # 页眉
    if btype == "header":
        return f'<div class="header" data-bbox="{bbox_str}">{text}</div>'
    
    # 页脚
    if btype == "footer":
        return f'<div class="footer" data-bbox="{bbox_str}">{text}</div>'
    
    # 未知类型 -> 作为段落处理
    # 注意: unknown 类型在评测中可能需要特殊处理
    if btype == "unknown":
        return f'<p data-bbox="{bbox_str}">{text}</p>'
    
    # 其他未识别类型 -> 使用 div class="类型"
    return f'<div class="{btype}" data-bbox="{bbox_str}">{text}</div>'


def render_html(ir: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    """
    将 IR 渲染为评测要求的 HTML 格式
    
    输出格式: <body>...</body>
    """
    blocks = ir.get("blocks", [])
    relations = ir.get("relations", {}) or {}
    order_edges = relations.get("order_edges", []) or []
    caption_links = relations.get("caption_links", []) or []
    tables = ir.get("tables", []) or []
    
    # 构建 ID 到对象的映射
    id_to_block = {b["id"]: b for b in blocks if "id" in b}
    id_to_table = {t["id"]: t for t in tables if "id" in t}
    
    # 构建 caption_id -> target_id 的映射
    cap_to_target = {}
    for link in caption_links:
        cap_id = link.get("caption_id")
        tar_id = link.get("target_id")
        if cap_id is not None and tar_id is not None:
            cap_to_target[cap_id] = tar_id
    
    if not blocks:
        return "<body></body>"
    
    # ========== 构建阅读顺序 ==========
    # 从 order_edges 构建顺序
    next_map: Dict[int, int] = {}
    indeg: Dict[int, int] = {b["id"]: 0 for b in blocks}
    
    for edge in order_edges:
        u = edge.get("u")
        v = edge.get("v")
        if u is not None and v is not None and u in indeg and v in indeg:
            next_map[u] = v
            indeg[v] = indeg.get(v, 0) + 1
    
    # 找到所有起始节点 (入度为 0)
    heads = [bid for bid, deg in indeg.items() if deg == 0]
    
    # 按位置排序起始节点
    def pos_key(bid: int) -> Tuple[float, float]:
        b = id_to_block.get(bid)
        if b:
            bbox = b.get("bbox", [0, 0, 0, 0])
            return (float(bbox[1]), float(bbox[0]))  # (y, x)
        return (0.0, 0.0)
    
    heads.sort(key=pos_key)
    
    # 遍历链表构建顺序
    ordered_ids: List[int] = []
    visited = set()
    
    for head in heads:
        cur = head
        while cur is not None and cur not in visited:
            if cur in id_to_block:
                ordered_ids.append(cur)
                visited.add(cur)
            cur = next_map.get(cur)
    
    # 添加未被遍历到的节点 (按位置排序)
    remaining = [bid for bid in id_to_block if bid not in visited]
    remaining.sort(key=pos_key)
    ordered_ids.extend(remaining)
    
    # ========== 渲染 HTML ==========
    html_parts: List[str] = []
    
    for bid in ordered_ids:
        block = id_to_block.get(bid)
        if not block:
            continue
        
        # 获取 caption 引用的 target
        caption_ref = cap_to_target.get(bid)
        
        # 获取表格对象
        table_obj = id_to_table.get(bid) if block.get("type") == "table" else None
        
        # 渲染 block
        html_str = _render_block(block, caption_ref, table_obj, caption_links)
        html_parts.append(html_str)
    
    return "<body>" + "".join(html_parts) + "</body>"


def validate_and_fix_html(html_str: str, cfg: Dict[str, Any], 
                          force_full_validate: bool = False,
                          debug: Optional[Dict[str, Any]] = None) -> str:
    """
    验证并修复 HTML 输出
    
    确保输出格式为: <body>...</body>
    """
    if debug is None:
        debug = {}
    
    if not html_str:
        return "<body></body>"
    
    html_str = html_str.strip()
    
    # 确保有 body 标签
    if not html_str.startswith("<body"):
        html_str = "<body>" + html_str
    if not html_str.endswith("</body>"):
        html_str = html_str + "</body>"
    
    # 检查是否为空 body
    if html_str in ("<body></body>", "<body> </body>", "<body/>"):
        debug["html_empty"] = True
    
    return html_str



# -----------------------------------------------------------------------------
# Pipeline: process_one -> outputs IR JSON in answer + optional answer_html
# -----------------------------------------------------------------------------
def process_one(sample: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle,
                image_root: Optional[str] = None, rng_seed: Optional[int] = None) -> Dict[str, Any]:
    """处理单个样本，返回包含 answer 的结果"""
    if rng_seed is not None:
        random.seed(rng_seed)

    image = sample.get("image", "unknown")
    prompt = sample.get("prompt", cfg.get("default_prompt", "")) or ""
    debug: Dict[str, Any] = {"image": image}

    t_total0 = _now_ms()

    # 1) Build IR candidates
    try:
        t0 = _now_ms()
        ir = build_ir_candidates(sample, cfg, models, image_root, debug)
        debug["build_ms"] = round(_now_ms() - t0, 2)
    except Exception as e:
        debug["build_error"] = str(e)[:200]
        ir = {
            "page": {"image": image, "width": 1000, "height": 1400},
            "blocks": [],
            "tables": [],
            "relations": {"order_edges": [], "caption_links": [], "heading_parent": []},
            "debug": {"prompt": prompt},
        }

    page = ir.get("page", {}) or {}
    img_path = os.path.join(image_root, image) if image_root and not os.path.isabs(image) else image

    # 2) ROI OCR on layout blocks (if present)
    ocr_lines: List[Dict[str, Any]] = []
    try:
        t1 = _now_ms()
        ir["blocks"], ocr_lines = _enrich_blocks_with_roi_ocr(ir.get("blocks", []), img_path, page, cfg, models, debug)
        debug["enrich_ms"] = round(_now_ms() - t1, 2)
    except Exception as e:
        debug["enrich_error"] = str(e)[:200]
        debug["enrich_ms"] = 0.0

    # 3) Block type prediction (LGB optional)
    try:
        t2 = _now_ms()
        ir = predict_block_types(ir, cfg, models)
        debug["block_ms"] = round(_now_ms() - t2, 2)
    except Exception as e:
        debug["block_error"] = str(e)[:200]
        debug["block_ms"] = 0.0

    # 4) Fallback trigger
    stats = _compute_page_stats(ir)
    triggered, reasons = _should_trigger_fallback(stats, cfg)
    debug["fallback_stats"] = {
        "text_chars": int(stats.get("text_chars", 0)),
        "num_blocks": int(stats.get("num_blocks", 0)),
        "largest_area_ratio": round(stats.get("largest_area_ratio", 0.0), 4),
    }
    debug["fallback_triggered"] = bool(triggered)
    debug["fallback_reasons"] = reasons if triggered else []

    if triggered and bool(cfg.get("pipeline", {}).get("use_visual_fallback", True)):
        try:
            ir, ocr_lines = _run_fallback(ir, cfg, models, image_root, reasons, debug)
        except Exception as e:
            debug["fallback_error"] = str(e)[:200]
            debug["fallback_ms"] = 0.0
            ir.setdefault("debug", {})["fallback_triggered"] = False
    else:
        ir.setdefault("debug", {})["fallback_triggered"] = False
        debug["fallback_ms"] = 0.0

    # 5) Table extraction (if not already in fallback)
    try:
        if not ir.get("tables"):
            ir["tables"] = _extract_tables(ir.get("blocks", []), ocr_lines, img_path, page, cfg, models, debug)
    except Exception as e:
        debug["table_error"] = str(e)[:200]
        ir["tables"] = []

    # 6) Formula blocks
    try:
        ir["blocks"] = _process_formula_blocks(ir.get("blocks", []), img_path=img_path, page=page, models=models, cfg=cfg)
    except Exception as e:
        debug["formula_error"] = str(e)[:200]

    # 7) Relations
    try:
        t3 = _now_ms()
        ir = predict_relations(ir, cfg, models)
        debug["rel_ms"] = round(_now_ms() - t3, 2)
    except Exception as e:
        debug["rel_error"] = str(e)[:200]
        debug["rel_ms"] = 0.0
        ir.setdefault("relations", {"order_edges": [], "caption_links": [], "heading_parent": []})

    # 8) Decode headings
    try:
        t4 = _now_ms()
        ir = decode(ir, cfg)
        debug["decode_ms"] = round(_now_ms() - t4, 2)
    except Exception as e:
        debug["decode_error"] = str(e)[:200]
        debug["decode_ms"] = 0.0

    # 9) Render HTML
    render_error = False
    html_str = ""
    try:
        t5 = _now_ms()
        html_str = render_html(ir, cfg)
        debug["render_ms"] = round(_now_ms() - t5, 2)
    except Exception as e:
        debug["render_error"] = str(e)[:200]
        debug["render_ms"] = 0.0
        render_error = True

    # 10) HTML validate
    try:
        t6 = _now_ms()
        html_str = validate_and_fix_html(html_str, cfg, force_full_validate=(render_error or triggered), debug=debug)
        debug["validate_ms"] = round(_now_ms() - t6, 2)
    except Exception as e:
        debug["validate_error"] = str(e)[:200]
        debug["validate_ms"] = 0.0
        html_str = "<body></body>"

    # 11) finalize debug + output format
    ir.setdefault("debug", {}).update(debug)
    ir["debug"]["blocks_count"] = len(ir.get("blocks", []))
    ir["debug"]["tables_count"] = len(ir.get("tables", []))
    ir["debug"]["total_ms"] = round(_now_ms() - t_total0, 2)

    return {
        "image": image,
        "prompt": prompt,
        "answer": html_str,
        "answer_html": html_str,
        "debug": ir.get("debug", {}),
    }


# -----------------------------------------------------------------------------
# Fallback trigger and execution
# -----------------------------------------------------------------------------
def _compute_page_stats(ir: Dict[str, Any]) -> Dict[str, float]:
    """计算页面统计信息用于 fallback 触发判断"""
    text_chars = 0
    num_blocks = len(ir.get("blocks", []))
    largest_area_ratio = 0.0
    page_w = float(max(1.0, (ir.get("page", {}) or {}).get("width", 1)))
    page_h = float(max(1.0, (ir.get("page", {}) or {}).get("height", 1)))
    page_area = page_w * page_h
    for b in ir.get("blocks", []):
        t = (b.get("text") or "")
        text_chars += len(t)
        area = _area(b.get("bbox", [0, 0, 0, 0]))
        largest_area_ratio = max(largest_area_ratio, area / max(1.0, page_area))
    return {
        "text_chars": float(text_chars),
        "num_blocks": float(num_blocks),
        "largest_area_ratio": float(largest_area_ratio),
        "page_area": float(page_area),
    }


def _should_trigger_fallback(stats: Dict[str, float], cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """判断是否需要触发 fallback"""
    trig = cfg.get("decode", {}).get("fallback_trigger", cfg.get("fallback_trigger", {})) or {}
    if not bool(trig.get("enabled", True)):
        return False, []
    reasons = []
    if stats["text_chars"] < float(trig.get("min_text_chars", 0) or 0):
        reasons.append("low_text_chars")
    if stats["num_blocks"] < float(trig.get("min_blocks", 0) or 0):
        reasons.append("low_blocks")
    if stats["largest_area_ratio"] > float(trig.get("max_image_area_ratio", 1.0) or 1.0):
        reasons.append("large_block")
    return (len(reasons) > 0, reasons)


def _run_fallback(ir: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle, image_root: Optional[str],
                  reasons: List[str], debug: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """执行 fallback 逻辑"""
    t0 = _now_ms()
    ir.setdefault("debug", {})
    ir["debug"]["fallback_triggered"] = True
    ir["debug"]["fallback_reason"] = "|".join(reasons)

    image_path = (ir.get("page", {}) or {}).get("image", "")
    img_path = os.path.join(image_root, image_path) if image_root and image_path and not os.path.isabs(image_path) else image_path

    # 1) layout detector rebuild
    layout_dets = _run_layout_detector(img_path, cfg, models, debug) if models.layout_detector is not None else []
    if layout_dets:
        new_blocks: List[Dict[str, Any]] = []
        for det in layout_dets:
            b = {
                "id": 0,
                "bbox": det["bbox"],
                "type": det["label"],
                "score": det["score"],
                "text": "",
                "style": None,
                "source": "layout_detector",
            }
            if det["label"] == "formula":
                b["latex"] = ""
            new_blocks.append(b)
        ir["blocks"] = new_blocks
        ir = _normalize_blocks(ir)

    # 2) ROI OCR
    blocks = ir.get("blocks", [])
    ocr_lines: List[Dict[str, Any]] = []
    blocks, ocr_lines = _enrich_blocks_with_roi_ocr(blocks, img_path, ir.get("page", {}), cfg, models, debug)
    ir["blocks"] = blocks

    # 3) tables + formulas
    ir["tables"] = _extract_tables(ir.get("blocks", []), ocr_lines, img_path, ir.get("page", {}), cfg, models, debug)
    ir["blocks"] = _process_formula_blocks(ir.get("blocks", []), img_path=img_path, page=ir.get("page", {}), models=models, cfg=cfg)

    debug["fallback_ms"] = round(_now_ms() - t0, 2)
    return ir, ocr_lines


# -----------------------------------------------------------------------------
# Batch processing
# -----------------------------------------------------------------------------
def _process_batch_sequential(samples: List[Dict[str, Any]], cfg: Dict[str, Any],
                              models: ModelBundle, image_root: Optional[str],
                              base_seed: int = 42) -> List[Dict[str, Any]]:
    """顺序处理批量样本"""
    out: List[Dict[str, Any]] = []
    for idx, s in enumerate(samples):
        try:
            out.append(process_one(s, cfg, models, image_root=image_root, rng_seed=base_seed + idx))
        except Exception as e:
            sys.stderr.write(f"Error processing {s.get('image')}: {e}\n")
            out.append({
                "image": s.get("image"),
                "prompt": s.get("prompt", cfg.get("default_prompt", "")),
                "answer": "<body></body>",
                "answer_html": "<body></body>",
                "debug": {"error": str(e)},
            })
    return out


# Worker 初始化变量
_worker_cfg = None
_worker_models = None


def _worker_init(cfg_dict: Dict[str, Any]) -> None:
    """Worker 进程初始化"""
    global _worker_cfg, _worker_models
    _worker_cfg = cfg_dict
    _worker_models = load_artifacts(cfg_dict)


def _worker_process_one(args: Tuple[int, Dict[str, Any], Optional[str], int]) -> Tuple[int, Dict[str, Any]]:
    """Worker 处理单个样本"""
    idx, sample, image_root, seed = args
    global _worker_cfg, _worker_models
    try:
        res = process_one(sample, _worker_cfg, _worker_models, image_root=image_root, rng_seed=seed)
    except Exception as e:
        res = {
            "image": sample.get("image"),
            "prompt": sample.get("prompt", (_worker_cfg or {}).get("default_prompt", "")),
            "answer": "<body></body>",
            "answer_html": "<body></body>",
            "debug": {"error": str(e), "worker_error": True},
        }
    return idx, res


def _process_batch_parallel(samples: List[Dict[str, Any]], cfg: Dict[str, Any],
                            image_root: Optional[str], num_workers: int,
                            base_seed: int = 42) -> List[Dict[str, Any]]:
    """并行处理批量样本"""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    tasks = [(idx, s, image_root, base_seed + idx) for idx, s in enumerate(samples)]
    results: List[Optional[Dict[str, Any]]] = [None] * len(samples)

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init, initargs=(cfg,)) as ex:
        futs = {ex.submit(_worker_process_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                i, res = fut.result()
                results[i] = res
            except Exception as e:
                sample = samples[idx]
                results[idx] = {
                    "image": sample.get("image"),
                    "prompt": sample.get("prompt", cfg.get("default_prompt", "")),
                    "answer": "<body></body>",
                    "answer_html": "<body></body>",
                    "debug": {"error": str(e), "parallel_error": True},
                }

    # 填充空结果
    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "image": samples[i].get("image"),
                "prompt": samples[i].get("prompt", cfg.get("default_prompt", "")),
                "answer": "<body></body>",
                "answer_html": "<body></body>",
                "debug": {"error": "missing_result"},
            }
    
    return results


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------
def write_submit_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """写入提交格式的 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            out = {"image": r.get("image"), "prompt": r.get("prompt"), "answer": r.get("answer")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def write_debug_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """写入调试信息的 JSONL 文件"""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def _get_block_meta_field(block: Dict[str, Any], field: str, default: float = 0.0) -> float:
    """安全获取 block["meta"][field]"""
    meta = block.get("meta")
    if meta is None:
        return default
    val = meta.get(field)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _detect_block_columns(blocks: List[Dict[str, Any]], page: Dict[str, Any]) -> Tuple[int, Dict[int, int]]:
    """
    检测页面栏布局，支持 1-4 栏
    
    Returns:
        (column_count, {block_id: column_id})
    """
    if not blocks:
        return 1, {}
    
    page_w = max(1.0, float(page.get("width", 1)))
    
    # 只用文本块检测，排除 header/footer
    text_blocks = [b for b in blocks 
                   if b.get("type") in ("paragraph", "list_item", "caption", "title")
                   and b.get("type") not in ("header", "footer")]
    
    if len(text_blocks) < 6:
        return 1, {b["id"]: 0 for b in blocks}
    
    # 提取中心 x 坐标
    centers = []
    for b in text_blocks:
        bbox = b.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        centers.append((b["id"], cx))
    
    if not centers:
        return 1, {b["id"]: 0 for b in blocks}
    
    # 按 x 排序
    centers.sort(key=lambda x: x[1])
    xs = [c[1] for c in centers]
    
    # 计算相邻间距
    gaps = []
    for i in range(len(xs) - 1):
        gaps.append((xs[i + 1] - xs[i], i))
    
    if not gaps:
        return 1, {b["id"]: 0 for b in blocks}
    
    # 找显著间距（大于中位数的 2 倍且大于页宽的 10%）
    gap_values = [g[0] for g in gaps]
    median_gap = safe_median(gap_values, default=0.1 * page_w)
    
    significant_gaps = []
    for gap_val, idx in gaps:
        if gap_val > max(2.0 * median_gap, 0.1 * page_w):
            significant_gaps.append((gap_val, idx))
    
    # 按间距大小排序，取最多 3 个（支持最多 4 栏）
    significant_gaps.sort(key=lambda x: -x[0])
    significant_gaps = significant_gaps[:3]
    
    if not significant_gaps:
        return 1, {b["id"]: 0 for b in blocks}
    
    # 计算分割阈值
    split_indices = sorted([idx for _, idx in significant_gaps])
    thresholds = [(xs[idx] + xs[idx + 1]) / 2 for idx in split_indices]
    
    n_columns = len(thresholds) + 1
    
    # 为所有 blocks 分配 column_id
    column_map = {}
    for b in blocks:
        # header/footer 强制 column_id=0
        if b.get("type") in ("header", "footer"):
            column_map[b["id"]] = 0
            continue
        
        bbox = b.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        
        col_id = 0
        for thresh in thresholds:
            if cx > thresh:
                col_id += 1
            else:
                break
        column_map[b["id"]] = col_id
    
    return n_columns, column_map


def _enrich_blocks_with_column_info(blocks: List[Dict[str, Any]], page: Dict[str, Any]) -> None:
    """为 blocks 添加栏信息到 meta 字段（原地修改）"""
    if not blocks:
        return
    
    n_columns, column_map = _detect_block_columns(blocks, page)
    
    # 按栏分组，并在栏内按 y 排序
    columns: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        col_id = column_map.get(b["id"], 0)
        if col_id not in columns:
            columns[col_id] = []
        columns[col_id].append(b)
    
    # 每栏内按 y 排序，确定 first/last
    for col_id, col_blocks in columns.items():
        col_blocks.sort(key=lambda x: (x.get("bbox", [0, 0, 0, 0])[1], x.get("bbox", [0, 0, 0, 0])[0]))
        
        for i, b in enumerate(col_blocks):
            if "meta" not in b or b["meta"] is None:
                b["meta"] = {}
            
            b["meta"]["column_id"] = col_id
            b["meta"]["column_count"] = n_columns
            b["meta"]["is_first_in_column"] = 1.0 if i == 0 else 0.0
            b["meta"]["is_last_in_column"] = 1.0 if i == len(col_blocks) - 1 else 0.0



def _block_feature_dict(block: Dict[str, Any], page: Dict[str, Any], height_pct: float,
                        column_id: float = 0.0, column_count: float = 1.0,
                        is_first_in_column: float = 0.0, is_last_in_column: float = 0.0) -> Dict[str, float]:
    """
    提取 block 特征，与 train.py 的 BLOCK_SCHEMA (29维) 完全对齐
    """
    x1, y1, x2, y2 = block.get("bbox", [0, 0, 0, 0])
    w = max(1.0, page.get("width", 1))
    h = max(1.0, page.get("height", 1))
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    txt = block.get("text") or ""
    ts = _text_stats(txt)
    source = (block.get("source") or "").lower()
    src_object = 1.0 if source == "object" else 0.0
    src_ocr = 1.0 if ("ocr" in source) else 0.0
    src_heur = 1.0 if source == "heuristic" or "heur" in source else 0.0
    y_top_region = 1.0 if y1 < 0.1 * h else 0.0
    y_bottom_region = 1.0 if y2 > 0.9 * h else 0.0
    level = 0.0
    if block.get("style") and block["style"].get("heading_level"):
        level = float(block["style"]["heading_level"])
    
    # 计算文本行数���启发式）
    text_line_count = max(1.0, float(txt.count("") + 1)) if txt.strip() else 0.0
    
    # 计算平均行高（归一化到页高）
    if text_line_count > 0 and bh > 0:
        avg_line_height_norm = (bh / text_line_count) / h
    else:
        avg_line_height_norm = 0.0
    
    # 特征字典，顺序必须与 train.py BLOCK_SCHEMA 一致
    feats = {
        # 位置与尺寸（8个）
        "rel_x1": x1 / w,
        "rel_y1": y1 / h,
        "rel_x2": x2 / w,
        "rel_y2": y2 / h,
        "rel_w": bw / w,
        "rel_h": bh / h,
        "area_ratio": _area([x1, y1, x2, y2]) / (w * h),
        "aspect": (bw / bh) if bh > 0 else 0.0,
        
        # 文本统计（8个）
        "text_len": ts["len"],
        "digit_ratio": ts["digit_ratio"],
        "upper_ratio": ts["upper_ratio"],
        "lower_ratio": ts["lower_ratio"],
        "punct_ratio": ts["punct_ratio"],
        "mean_word_len": ts["mean_word_len"],
        "is_alnum": ts["is_alnum"],
        "ch_ratio": ts["ch_ratio"],
        
        # 样式（1个）
        "heading_level": level,
        
        # 来源（3个）
        "src_object": src_object,
        "src_ocr": src_ocr,
        "src_heur": src_heur,
        
        # 位置区域（2个）
        "y_top_region": y_top_region,
        "y_bottom_region": y_bottom_region,
        
        # 高度百分位（1个）
        "height_percentile": height_pct,
        
        # ===== 新增特征（6个）与 train.py 对齐 =====
        "column_id": column_id,
        "column_count": column_count,
        "is_first_in_column": is_first_in_column,
        "is_last_in_column": is_last_in_column,
        "text_line_count": text_line_count,
        "avg_line_height_norm": avg_line_height_norm,
    }
    return feats




def _pair_feature_dict(b1: Dict[str, Any], b2: Dict[str, Any], page: Dict[str, Any],
                       column_count: float = 1.0) -> Dict[str, float]:
    """
    提取 pair 特征，与 train.py 的 PAIR_SCHEMA (34维) 完全对齐
    """
    bb1 = b1.get("bbox", [0, 0, 0, 0])
    bb2 = b2.get("bbox", [0, 0, 0, 0])
    x11, y11, x12, y12 = bb1
    x21, y21, x22, y22 = bb2
    cx1, cy1 = _center(bb1)
    cx2, cy2 = _center(bb2)
    w = max(1.0, page.get("width", 1))
    h = max(1.0, page.get("height", 1))
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist = math.hypot(dx, dy)
    ovx = _overlap_1d(x11, x12, x21, x22)
    ovy = _overlap_1d(y11, y12, y21, y22)
    bw1, bh1 = max(1.0, x12 - x11), max(1.0, y12 - y11)
    bw2, bh2 = max(1.0, x22 - x21), max(1.0, y22 - y21)
    x_overlap_ratio = ovx / min(bw1, bw2)
    y_overlap_ratio = ovy / min(bh1, bh2)
    size_ratio_w = bw2 / bw1
    size_ratio_h = bh2 / bh1
    align_diff_left_norm = abs(x21 - x11) / w
    align_diff_right_norm = abs(x22 - x12) / w
    same_row = 1.0 if abs(cy1 - cy2) < 0.04 * h else 0.0
    same_col = 1.0 if abs(x11 - x21) < 0.08 * w else 0.0
    left_to_right = 1.0 if (same_row > 0.5 and x11 < x21) else 0.0
    is_above = 1.0 if y12 <= y21 else 0.0
    v_gap = max(0.0, y21 - y12)
    
    # 获取 column_id（从 block 的 meta 或默认 0）
    u_col_id = float(b1.get("_column_id", 0))
    v_col_id = float(b2.get("_column_id", 0))
    same_column_id = 1.0 if u_col_id == v_col_id else 0.0
    column_diff = max(-3.0, min(3.0, v_col_id - u_col_id))
    
    # 获取文本行数
    u_text = (b1.get("text") or "")
    v_text = (b2.get("text") or "")
    u_lines = max(1.0, float(u_text.count("") + 1)) if u_text.strip() else 1.0
    v_lines = max(1.0, float(v_text.count("") + 1)) if v_text.strip() else 1.0
    text_line_count_ratio = (v_lines + 1) / (u_lines + 1)
    
    # 特征字典，顺序必须与 train.py PAIR_SCHEMA 一致
    feats = {
        # 相对位置（3个）
        "dx_norm": dx / w,
        "dy_norm": dy / h,
        "center_dist_norm": dist / math.hypot(w, h),
        
        # 重叠（2个）
        "x_overlap": x_overlap_ratio,
        "y_overlap": y_overlap_ratio,
        
        # 尺寸比例（2个）
        "size_ratio_w": size_ratio_w,
        "size_ratio_h": size_ratio_h,
        
        # 布局关系（6个）
        "same_column": same_col,
        "is_above": is_above,
        "align_diff_left_norm": align_diff_left_norm,
        "align_diff_right_norm": align_diff_right_norm,
        "same_row": same_row,
        "left_to_right": left_to_right,
        
        # 间距（1个）
        "gap_y_norm": v_gap / h,
        
        # 类型特征（4个）
        "u_is_title": 1.0 if b1.get("type") == "title" else 0.0,
        "v_is_title": 1.0 if b2.get("type") == "title" else 0.0,
        "u_heading_level": float(b1.get("style", {}).get("heading_level", 0) if b1.get("style") else 0.0),
        "v_heading_level": float(b2.get("style", {}).get("heading_level", 0) if b2.get("style") else 0.0),
    }
    
    # 类型 one-hot（10个）
    u_one = _coarse_type_onehot(b1)
    v_one = _coarse_type_onehot(b2)
    feats.update({
        "u_text": u_one["text"],
        "u_table": u_one["table"],
        "u_figure": u_one["figure"],
        "u_caption": u_one["caption"],
        "u_other": u_one["other"],
        "v_text": v_one["text"],
        "v_table": v_one["table"],
        "v_figure": v_one["figure"],
        "v_caption": v_one["caption"],
        "v_other": v_one["other"],
    })
    
    # ===== 新增特征（6个）与 train.py 对齐 =====
    feats.update({
        "same_column_id": same_column_id,
        "column_diff": column_diff,
        "u_column_id": u_col_id,
        "v_column_id": v_col_id,
        "column_count": column_count,
        "text_line_count_ratio": text_line_count_ratio,
    })
    
    return feats



def _detect_table_spans(rows: List[List[Dict[str, Any]]], page: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    检测表格的 rowspan 和 colspan
    
    策略:
    1. 构建单元格网格
    2. 检测跨列: 单元格宽度显著大于列平均宽度
    3. 检测跨行: 单元格高度显著大于行平均高度，或下方有空位
    """
    if not rows or len(rows) < 2:
        return rows
    
    # 计算每列的平均宽度和位置
    all_cells = []
    for row in rows:
        for cell in row:
            all_cells.append(cell)
    
    if not all_cells:
        return rows
    
    # 收集所有单元格的 x 边界
    x_bounds = set()
    for cell in all_cells:
        bbox = cell.get("bbox", [0, 0, 0, 0])
        x_bounds.add(round(bbox[0], 1))
        x_bounds.add(round(bbox[2], 1))
    
    x_bounds = sorted(x_bounds)
    
    # 构建列边界（合并相近的边界）
    col_bounds = []
    if x_bounds:
        col_bounds.append(x_bounds[0])
        for x in x_bounds[1:]:
            if x - col_bounds[-1] > 10:  # 间距阈值
                col_bounds.append(x)
    
    n_cols = max(1, len(col_bounds) - 1)
    
    # 计算每行的高度
    row_heights = []
    for row in rows:
        if row:
            h = max(cell.get("bbox", [0, 0, 0, 0])[3] - cell.get("bbox", [0, 0, 0, 0])[1] for cell in row)
            row_heights.append(h)
        else:
            row_heights.append(20)
    
    median_row_height = safe_median(row_heights, 20)
    
    # 为每个单元格计算 colspan
    for row in rows:
        for cell in row:
            bbox = cell.get("bbox", [0, 0, 0, 0])
            cell_x1, cell_x2 = bbox[0], bbox[2]
            cell_width = cell_x2 - cell_x1
            
            # 计算跨越的列数
            col_start = 0
            col_end = 0
            for i, bound in enumerate(col_bounds[:-1]):
                next_bound = col_bounds[i + 1]
                if cell_x1 <= bound + 5:
                    col_start = i
                    break
            
            for i in range(len(col_bounds) - 1, 0, -1):
                prev_bound = col_bounds[i - 1]
                bound = col_bounds[i]
                if cell_x2 >= bound - 5:
                    col_end = i
                    break
            
            colspan = max(1, col_end - col_start)
            cell["colspan"] = colspan
    
    # 简化的 rowspan 检测: 基于高度
    for ri, row in enumerate(rows):
        for cell in row:
            bbox = cell.get("bbox", [0, 0, 0, 0])
            cell_height = bbox[3] - bbox[1]
            
            # 如果单元格高度显著大于行高，可能跨行
            if cell_height > 1.5 * median_row_height:
                estimated_rows = min(len(rows) - ri, max(1, round(cell_height / median_row_height)))
                cell["rowspan"] = estimated_rows
            else:
                cell["rowspan"] = 1
    
    return rows


def _extract_table_structure(table_block: Dict[str, Any], table_lines: List[Dict[str, Any]], 
                             page: Dict[str, Any]) -> Dict[str, Any]:
    """
    提取表格结构（支持 rowspan/colspan 检测）
    """
    table_bbox = table_block.get("bbox", [0, 0, 0, 0])
    table_id = int(table_block.get("id", 0))
    score = float(table_block.get("score", 1.0) or 1.0)

    if not table_lines:
        return {
            "id": table_id,
            "bbox": table_bbox,
            "type": "table",
            "score": score,
            "source": "table_heuristic",
            "rows": [[{"bbox": table_bbox, "text": "", "rowspan": 1, "colspan": 1}]],
        }

    # Row clustering by y-centers
    y_centers = [(ln["bbox"][1] + ln["bbox"][3]) / 2 for ln in table_lines]
    heights = [max(1.0, ln["bbox"][3] - ln["bbox"][1]) for ln in table_lines]
    row_thr = max(6.0, safe_median(heights, 20.0) * 0.7)
    row_groups = _cluster_by_gaps(y_centers, row_thr)
    row_groups.sort(key=lambda g: sum(y_centers[i] for i in g) / max(1, len(g)))

    rows: List[List[Dict[str, Any]]] = []
    for rg in row_groups:
        row_lines = [table_lines[i] for i in rg]
        if len(row_lines) == 1:
            ln = row_lines[0]
            rows.append([{
                "bbox": ln["bbox"],
                "text": (ln.get("text") or "").strip(),
                "rowspan": 1,
                "colspan": 1,
            }])
            continue

        # Column clustering within row by x-centers
        x_centers = [(ln["bbox"][0] + ln["bbox"][2]) / 2 for ln in row_lines]
        widths = [max(1.0, ln["bbox"][2] - ln["bbox"][0]) for ln in row_lines]
        col_thr = max(20.0, safe_median(widths, 50.0) * 0.5)
        col_groups = _cluster_by_gaps(x_centers, col_thr)
        col_groups.sort(key=lambda g: sum(x_centers[i] for i in g) / max(1, len(g)))

        row_cells: List[Dict[str, Any]] = []
        for cg in col_groups:
            cell_lines = [row_lines[i] for i in cg]
            cell_bbox = _union_bbox([ln["bbox"] for ln in cell_lines])
            cell_text = " ".join((ln.get("text") or "").strip() for ln in cell_lines if (ln.get("text") or "").strip())
            row_cells.append({
                "bbox": cell_bbox,
                "text": cell_text,
                "rowspan": 1,
                "colspan": 1
            })
        rows.append(row_cells)

    if not rows:
        rows = [[{"bbox": table_bbox, "text": "", "rowspan": 1, "colspan": 1}]]
    
    # 检测 rowspan/colspan
    rows = _detect_table_spans(rows, page)

    return {
        "id": table_id,
        "bbox": table_bbox,
        "type": "table",
        "score": score,
        "source": "table_heuristic",
        "rows": rows,
    }



def _detect_columns_for_order(blocks: List[Dict[str, Any]], page: Dict[str, Any]) -> Tuple[int, Dict[int, int]]:
    """
    检测页面栏布局用于阅读顺序（支持 1-4 栏）
    
    Returns:
        (column_count, {block_index: column_id})
    """
    if not blocks:
        return 1, {}
    
    page_w = max(1.0, float(page.get("width", 1)))
    
    # 只用文本块检测，排除 header/footer
    text_indices = [i for i, b in enumerate(blocks) 
                    if b.get("type") in ("paragraph", "list_item", "caption", "title")
                    and b.get("type") not in ("header", "footer")]
    
    if len(text_indices) < 6:
        return 1, {i: 0 for i in range(len(blocks))}
    
    # 提取中心 x 坐标
    centers = []
    for i in text_indices:
        bbox = blocks[i].get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        centers.append((i, cx))
    
    # 按 x 排序
    centers.sort(key=lambda x: x[1])
    xs = [c[1] for c in centers]
    
    # 计算相邻间距
    gaps = []
    for j in range(len(xs) - 1):
        gaps.append((xs[j + 1] - xs[j], j))
    
    if not gaps:
        return 1, {i: 0 for i in range(len(blocks))}
    
    # 找显著间距
    gap_values = [g[0] for g in gaps]
    median_gap = safe_median(gap_values, default=0.1 * page_w)
    
    significant_gaps = []
    for gap_val, idx in gaps:
        if gap_val > max(2.0 * median_gap, 0.1 * page_w):
            significant_gaps.append((gap_val, idx))
    
    significant_gaps.sort(key=lambda x: -x[0])
    significant_gaps = significant_gaps[:3]
    
    if not significant_gaps:
        return 1, {i: 0 for i in range(len(blocks))}
    
    # 计算分割阈值
    split_indices = sorted([idx for _, idx in significant_gaps])
    thresholds = [(xs[idx] + xs[idx + 1]) / 2 for idx in split_indices]
    
    n_columns = len(thresholds) + 1
    
    # 为所有 blocks 分配 column_id
    column_map = {}
    for i, b in enumerate(blocks):
        if b.get("type") in ("header", "footer"):
            column_map[i] = -1 if b.get("type") == "header" else 999
            continue
        
        bbox = b.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        
        col_id = 0
        for thresh in thresholds:
            if cx > thresh:
                col_id += 1
            else:
                break
        column_map[i] = col_id
    
    return n_columns, column_map



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config")
    parser.add_argument("--input", type=str, required=True, help="Path to input test jsonl")
    parser.add_argument("--output", type=str, default="submit.jsonl", help="Path to output submit jsonl")
    parser.add_argument("--image-root", type=str, default=None, help="Root dir to prepend to relative image paths")
    parser.add_argument("--debug-output", type=str, default=None, help="Optional path to write debug jsonl")
    parser.add_argument("--parallel", type=int, default=None, help="Override pipeline.parallel_workers")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    num_workers = args.parallel if args.parallel is not None else int(cfg.get("pipeline", {}).get("parallel_workers", 0) or 0)

    samples: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            samples.append(json.loads(line))

    t0 = time.time()
    if num_workers > 0:
        sys.stderr.write(f"[info] parallel_workers={num_workers}\n")
        results = _process_batch_parallel(samples, cfg, args.image_root, num_workers=num_workers, base_seed=args.seed)
    else:
        models = load_artifacts(cfg)
        results = _process_batch_sequential(samples, cfg, models, args.image_root, base_seed=args.seed)

    total_time_s = round(time.time() - t0, 2)

    # Write submit
    rows = [{"image": r["image"], "prompt": r["prompt"], "answer": r["answer"]} for r in results]
    write_submit_jsonl(args.output, rows)

    # Debug output (optional): include answer_html + debug + summary
    if args.debug_output:
        dbg_rows = []
        stats = {
            "total_samples": len(results),
            "total_time_s": total_time_s,
            "parallel_workers": num_workers,
            "fallback_triggered": 0,
            "ocr_cache_hits": 0,
            "ocr_cache_misses": 0,
            "html_full_validate": 0,
            "errors": 0,
        }
        for r in results:
            d = r.get("debug", {}) or {}
            if d.get("fallback_triggered"):
                stats["fallback_triggered"] += 1
            stats["ocr_cache_hits"] += int(d.get("ocr_cache_hit", 0) or 0)
            stats["ocr_cache_misses"] += int(d.get("ocr_cache_miss", 0) or 0)
            if d.get("html_full_validate"):
                stats["html_full_validate"] += 1
            if d.get("error") or d.get("worker_error") or d.get("parallel_error"):
                stats["errors"] += 1
            dbg_rows.append({
                "image": r.get("image"),
                "prompt": r.get("prompt"),
                "answer_html": r.get("answer_html"),
                "debug": d,
            })
        dbg_rows.append({"_summary": stats})
        write_debug_jsonl(args.debug_output, dbg_rows)

    sys.stderr.write(f"[stats] total_time_s={total_time_s} samples={len(results)}\n")


if __name__ == "__main__":
    main()
