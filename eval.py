import argparse
import base64
import json
import os
import sys
import tempfile
import time
import math
import html
import hashlib
import io
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from html.parser import HTMLParser

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from src.order_features import compute_advanced_pair_features
except Exception:  # pragma: no cover
    def compute_advanced_pair_features(u: Dict[str, Any], v: Dict[str, Any], page: Dict[str, Any]) -> Dict[str, float]:  # type: ignore
        return {}

try:
    from src.layout_postprocess import suppress_nested_detections, refine_title_paragraph_blocks
except Exception:  # pragma: no cover
    def suppress_nested_detections(detections: List[Dict[str, Any]], iou_threshold: float = 0.92, containment_threshold: float = 0.94):  # type: ignore
        return detections, {"suppressed_nested": 0}

    def refine_title_paragraph_blocks(blocks: List[Dict[str, Any]], page: Dict[str, Any], title_boost_ratio: float = 1.35, paragraph_boost_ratio: float = 1.05):  # type: ignore
        return blocks, {"paragraph_to_title": 0, "title_to_paragraph": 0}

try:
    from table_transformer import TableTransformerParser
except Exception:  # pragma: no cover
    TableTransformerParser = None  # type: ignore

try:
    from formula_expert import ExpertFormulaRecognizer, sanitize_latex_expression, ensure_display_math_wrapped
except Exception:  # pragma: no cover
    ExpertFormulaRecognizer = None  # type: ignore

    def sanitize_latex_expression(x: str) -> str:  # type: ignore
        return (x or "").strip()

    def ensure_display_math_wrapped(x: str) -> str:  # type: ignore
        txt = (x or "").strip()
        if txt.startswith("$$") and txt.endswith("$$") and len(txt) >= 4:
            return txt
        if txt.startswith("$") and txt.endswith("$") and len(txt) >= 2:
            txt = txt[1:-1].strip()
        return f"$${txt}$$" if txt else "$$ $$"

try:
    from text_correction import text_correction
except Exception:  # pragma: no cover
    def text_correction(raw_text: str, cfg: Optional[Dict[str, Any]] = None) -> str:  # type: ignore
        return raw_text or ""

_FORMULA_MODULE_OK = False
try:
    from formula import FormulaRecognizer, normalize_latex
    _FORMULA_MODULE_OK = True
except Exception:  # pragma: no cover
    try:
        from utils.formula import FormulaRecognizer, normalize_latex  # type: ignore
        _FORMULA_MODULE_OK = True
    except Exception:  # pragma: no cover
        FormulaRecognizer = None

        def normalize_latex(x: str) -> str:  # type: ignore
            return (x or "").strip()

try:
    from src.reading_order_utils import (
        detect_columns_by_projection as _detect_cols_proj,
        assign_block_columns as _assign_block_cols,
        compute_page_median_gap as _compute_median_gap,
    )
    _HAS_READING_ORDER_UTILS = True
except Exception:  # pragma: no cover
    try:
        from utils.reading_order import (  # type: ignore
            detect_columns_by_projection as _detect_cols_proj,
            assign_block_columns as _assign_block_cols,
            compute_page_median_gap as _compute_median_gap,
        )
        _HAS_READING_ORDER_UTILS = True
    except Exception:
        _HAS_READING_ORDER_UTILS = False
        _detect_cols_proj = None  # type: ignore
        _assign_block_cols = None  # type: ignore
        _compute_median_gap = None  # type: ignore

try:
    from src.reading_order_pipeline import (
        xycut_graph_sort as _xycut_graph_sort,
        build_chain_order_edges as _build_chain_order_edges,
    )
    _HAS_HYBRID_READING_ORDER = True
except Exception:  # pragma: no cover
    _HAS_HYBRID_READING_ORDER = False

    def _xycut_graph_sort(elements: List[Dict[str, Any]], page: Optional[Dict[str, Any]] = None, cfg: Optional[Dict[str, Any]] = None):  # type: ignore
        return list(elements or [])

    def _build_chain_order_edges(ordered_elements: List[Dict[str, Any]], default_score: float = 1.0):  # type: ignore
        out = []
        ordered_elements = ordered_elements or []
        for a, b in zip(ordered_elements[:-1], ordered_elements[1:]):
            if isinstance(a, dict) and isinstance(b, dict) and "id" in a and "id" in b:
                out.append({"u": a["id"], "v": b["id"], "score": float(default_score)})
        return out


DEFAULT_LAYOUT_CLASSES = [
    "paragraph", "title", "list_item", "caption", "table",
    "figure", "formula", "header", "footer", "chart", "unknown", "page_number"
]

SUPPORTED_BLOCK_TYPES = set(DEFAULT_LAYOUT_CLASSES)

TEXT_BLOCK_TYPES = {"paragraph", "title", "list_item", "caption", "header", "footer", "page_number"}
NON_TEXT_BLOCK_TYPES = {"table", "figure", "chart", "formula"}

_SPACE_RE = re.compile(r"\s+")
_TEXT_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]")
_FORMULA_TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z0-9]+|[^\s]")
_ALNUM_OR_PUNC_RE = re.compile(r"[A-Za-z0-9_]+|[^\s]")

_PADDLE_FATAL_ERROR_MARKERS = (
    "convertpirattribute2runtimeattribute",
    "new_executor/instruction/onednn",
    "pir::arrayattribute",
)

_TESSERACT_AVAILABLE: Optional[bool] = None
_TESSERACT_DISABLED_REASON: str = ""


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


def _is_fatal_paddle_runtime_error(err: Exception) -> bool:
    msg = str(err or "").strip().lower()
    if not msg:
        return False
    if any(marker in msg for marker in _PADDLE_FATAL_ERROR_MARKERS):
        return True
    # Typical pattern in some Paddle/oneDNN runtime builds.
    if "unimplemented" in msg and "onednn" in msg:
        return True
    return False


def _has_tesseract_binary() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_DISABLED_REASON:
        return False
    if _TESSERACT_AVAILABLE is None:
        _TESSERACT_AVAILABLE = shutil.which("tesseract") is not None
    return bool(_TESSERACT_AVAILABLE)


def _disable_tesseract(reason: str) -> None:
    global _TESSERACT_AVAILABLE, _TESSERACT_DISABLED_REASON
    _TESSERACT_AVAILABLE = False
    _TESSERACT_DISABLED_REASON = (reason or "disabled")[:160]


def _sample_prompt(sample: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    if not isinstance(sample, dict):
        return str((cfg or {}).get("default_prompt", "") or "")
    val = sample.get("prompt")
    if isinstance(val, str) and val.strip():
        return val
    val2 = sample.get("prefix")
    if isinstance(val2, str) and val2.strip():
        return val2
    return str((cfg or {}).get("default_prompt", "") or "")


@dataclass
class ModelBundle:
    block_classifier: Any = None
    relation_scorer_order: Any = None
    relation_scorer_caption: Any = None

    feature_schema_block: List[str] = field(default_factory=list)
    feature_schema_pair: List[str] = field(default_factory=list)
    schema_version_block: Optional[str] = None
    schema_version_pair: Optional[str] = None

    label_map: Dict[int, str] = field(default_factory=dict)

    ocr_engine: Any = None

    layout_detector: Any = None
    layout_class_map: Dict[int, str] = field(default_factory=dict)
    layout_input_size: int = 1024
    layout_nms_threshold: float = 0.5
    layout_score_threshold: float = 0.3
    layout_nms_class_aware: bool = True
    layout_bbox_format: str = "auto"          # "auto" | "xywh" | "xyxy"
    layout_hf_correction: bool = False        # header/footer post-correction
    layout_hf_top_ratio: float = 0.08        # top region fraction
    layout_hf_bottom_ratio: float = 0.08     # bottom region fraction
    layout_hf_width_ratio: float = 0.5       # min width fraction to qualify
    layout_second_pass: bool = False          # second-pass inference
    layout_second_pass_min_dets: int = 2      # trigger if dets < this
    layout_second_pass_min_nontext: int = 0   # trigger if non-text dets <= this
    layout_second_pass_min_avg_score: float = 0.0  # trigger if avg_score < this
    layout_second_pass_input_size: int = 1280
    layout_second_pass_score_threshold: float = 0.15

    cfg: Dict[str, Any] = field(default_factory=dict)
    model_disabled_reason: List[str] = field(default_factory=list)

    formula_recognizer: Any = None
    openai_formula_cfg: Dict[str, Any] = field(default_factory=dict)
    table_transformer: Any = None


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
    merged["_config_dir"] = os.path.dirname(os.path.abspath(path))

    merged.setdefault("io", {})
    merged.setdefault("pipeline", {})
    merged.setdefault("decode", {})
    merged.setdefault("schema", {})
    merged.setdefault("models", {})
    merged.setdefault("fallback_models", merged.get("fallback_models", {}))

    nested_fallback = (merged.get("models", {}) or {}).get("fallback_models", {}) or {}
    for key, val in nested_fallback.items():
        if key not in merged["fallback_models"]:
            merged["fallback_models"][key] = val
            continue
        if isinstance(val, dict) and isinstance(merged["fallback_models"].get(key), dict):
            merged_cfg: Dict[str, Any] = {}
            deep_merge(merged_cfg, val)                         # base (nested)
            deep_merge(merged_cfg, merged["fallback_models"][key])  # overlay (profile/root)
            merged["fallback_models"][key] = merged_cfg

    for k in ("layout_detector", "ocr", "table_refiner", "formula_ocr"):
        if k not in merged["fallback_models"] and k in merged["models"]:
            merged["fallback_models"][k] = merged["models"].get(k, {})

    return merged


def _resolve_artifact_path(raw_path: Optional[str], cfg: Dict[str, Any], prefer_models_dir: bool = False) -> Optional[str]:
    """Resolve artifact path from config with backward-compatible search."""
    if not raw_path:
        return None
    p = str(raw_path).strip()
    if not p:
        return None
    if os.path.isabs(p):
        return p if os.path.exists(p) else None
    if os.path.exists(p):
        return os.path.abspath(p)

    cfg_dir = (cfg.get("_config_dir") or "").strip()
    io_cfg = cfg.get("io", {}) or {}
    output_dir = io_cfg.get("output_dir")
    search_roots: List[str] = []
    for root in (cfg_dir, output_dir, os.getcwd()):
        if not root:
            continue
        if os.path.isabs(str(root)):
            search_roots.append(str(root))
        elif cfg_dir:
            search_roots.append(os.path.normpath(os.path.join(cfg_dir, str(root))))
        else:
            search_roots.append(os.path.abspath(str(root)))

    extra_subdirs = ["", "models", "artifacts", "artifacts/models", "output", "output/models"]
    if prefer_models_dir:
        extra_subdirs = ["models", "", "artifacts/models", "output/models", "artifacts", "output"]

    tried: List[str] = []
    for root in search_roots:
        for sub in extra_subdirs:
            base = os.path.join(root, sub) if sub else root
            cand = os.path.normpath(os.path.join(base, p))
            if cand in tried:
                continue
            tried.append(cand)
            if os.path.exists(cand):
                return cand

    return None


def _resolve_logic_model_name(models_cfg: Dict[str, Any], key: str) -> Optional[str]:
    """Read model filename from legacy `models.logic_models.*` config."""
    logic_models = models_cfg.get("logic_models", {}) or {}
    val = logic_models.get(key)
    return str(val).strip() if isinstance(val, str) and val.strip() else None


def _resolve_schema_name(models_cfg: Dict[str, Any], key: str) -> Optional[str]:
    """Read schema filename from legacy `models.feature_schemas.*` config."""
    feat_schemas = models_cfg.get("feature_schemas", {}) or {}
    val = feat_schemas.get(key)
    return str(val).strip() if isinstance(val, str) and val.strip() else None


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

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    lang = ocr_cfg.get("lang", "ch")
    det = ocr_cfg.get("det", True)
    rec = ocr_cfg.get("rec", True)
    text_det_limit_side_len = int(ocr_cfg.get("text_det_limit_side_len", 1536) or 1536)
    text_rec_score_thresh = float(ocr_cfg.get("text_rec_score_thresh", 0.35) or 0.35)

    candidates = [
        {
            "lang": lang,
            "use_doc_orientation_classify": bool(ocr_cfg.get("use_doc_orientation_classify", False)),
            "use_doc_unwarping": bool(ocr_cfg.get("use_doc_unwarping", False)),
            "use_textline_orientation": bool(ocr_cfg.get("use_textline_orientation", False)),
            "text_det_limit_side_len": text_det_limit_side_len,
            "text_rec_score_thresh": text_rec_score_thresh,
        },
        {
            "lang": lang,
            "det": det,
            "rec": rec,
            "use_textline_orientation": bool(ocr_cfg.get("use_textline_orientation", True)),
        },
        {"lang": lang},
    ]

    for kwargs in candidates:
        try:
            return PaddleOCR(**kwargs)
        except Exception:
            continue
    return None


def _load_layout_class_map(cfg_map) -> Dict[int, str]:
    """
    Handle:
    - str path to a JSON file (e.g. "models/layout_class_map.json")
    - dict with int->str, str->int, or str->str (int keys as strings) mappings
    """
    if not cfg_map:
        return {i: c for i, c in enumerate(DEFAULT_LAYOUT_CLASSES)}
    if isinstance(cfg_map, str):
        if os.path.exists(cfg_map):
            try:
                with open(cfg_map, "r", encoding="utf-8") as _f:
                    cfg_map = json.load(_f)
            except Exception:
                return {i: c for i, c in enumerate(DEFAULT_LAYOUT_CLASSES)}
        else:
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
        available = ort.get_available_providers() if hasattr(ort, "get_available_providers") else []
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return ort.InferenceSession(path, providers=providers)
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

    def _get_model_raw(name: str, legacy_logic_key: Optional[str] = None) -> Optional[str]:
        entry = models_cfg.get(name)
        if isinstance(entry, dict):
            path_val = entry.get("path")
            if isinstance(path_val, str) and path_val.strip():
                return path_val.strip()
        elif isinstance(entry, str) and entry.strip():
            return entry.strip()
        logic_key = legacy_logic_key or name
        return _resolve_logic_model_name(models_cfg, logic_key)

    def _get_schema_raw(name: str, legacy_schema_key: Optional[str] = None) -> Optional[str]:
        val = schema_cfg.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
        schema_entry = models_cfg.get(name.replace("_path", ""))
        if isinstance(schema_entry, dict):
            path_val = schema_entry.get("path")
            if isinstance(path_val, str) and path_val.strip():
                return path_val.strip()
        elif isinstance(schema_entry, str) and schema_entry.strip():
            return schema_entry.strip()
        legacy_key = legacy_schema_key or name
        return _resolve_schema_name(models_cfg, legacy_key)

    block_model_path = _resolve_artifact_path(
        _get_model_raw("block_classifier", legacy_logic_key="block_classifier"),
        cfg,
        prefer_models_dir=True,
    )
    order_model_path = _resolve_artifact_path(
        _get_model_raw("relation_scorer_order", legacy_logic_key="relation_scorer"),
        cfg,
        prefer_models_dir=True,
    )
    caption_model_path = _resolve_artifact_path(
        _get_model_raw("relation_scorer_caption", legacy_logic_key="relation_scorer_caption"),
        cfg,
        prefer_models_dir=True,
    )

    bundle.block_classifier = _load_lgb_model(block_model_path)
    bundle.relation_scorer_order = _load_lgb_model(order_model_path)
    bundle.relation_scorer_caption = _load_lgb_model(caption_model_path)

    block_schema_path = _resolve_artifact_path(
        _get_schema_raw("feature_schema_block_path", legacy_schema_key="block"),
        cfg,
        prefer_models_dir=False,
    )
    pair_schema_path = _resolve_artifact_path(
        _get_schema_raw("feature_schema_pair_path", legacy_schema_key="pair"),
        cfg,
        prefer_models_dir=False,
    )
    label_map_path = _resolve_artifact_path(
        _get_schema_raw("label_map_path", legacy_schema_key="label_map"),
        cfg,
        prefer_models_dir=False,
    )

    bundle.feature_schema_block, bundle.schema_version_block = _load_schema(block_schema_path)
    bundle.feature_schema_pair, bundle.schema_version_pair = _load_schema(pair_schema_path)
    bundle.label_map = _load_label_map(label_map_path)

    bundle.ocr_engine = _maybe_load_paddle(cfg)

    formula_cfg_all = (fallback_cfg.get("formula_ocr") or {}) or {}
    formula_openai_cfg = (formula_cfg_all.get("openai_54") or {}) or {}
    formula_transformer_cfg = (formula_cfg_all.get("transformer") or {}) or {}

    if _FORMULA_MODULE_OK:
        base_formula = None
        try:
            base_formula = FormulaRecognizer(paddle_ocr=bundle.ocr_engine)
        except Exception:
            base_formula = None

        if ExpertFormulaRecognizer is not None:
            try:
                bundle.formula_recognizer = ExpertFormulaRecognizer(base_recognizer=base_formula, cfg=formula_transformer_cfg)
            except Exception:
                bundle.formula_recognizer = base_formula
        else:
            bundle.formula_recognizer = base_formula

    bundle.openai_formula_cfg = formula_openai_cfg

    layout_cfg = (fallback_cfg.get("layout_detector") or models_cfg.get("layout_detector") or {}) or {}
    if layout_cfg:
        layout_path = _resolve_artifact_path(layout_cfg.get("path"), cfg, prefer_models_dir=False)
        class_map_path = _resolve_artifact_path(layout_cfg.get("class_map"), cfg, prefer_models_dir=False)
        bundle.layout_detector = _load_onnx_model(layout_path, bundle.model_disabled_reason)
        bundle.layout_class_map = _load_layout_class_map(class_map_path or layout_cfg.get("class_map"))
        bundle.layout_input_size = int(layout_cfg.get("input_size", 1024) or 1024)
        bundle.layout_nms_threshold = float(layout_cfg.get("nms_threshold", 0.5) or 0.5)
        bundle.layout_score_threshold = float(layout_cfg.get("score_threshold", 0.3) or 0.3)
        bundle.layout_nms_class_aware = bool(layout_cfg.get("nms_class_aware", True))
        bundle.layout_bbox_format = str(layout_cfg.get("bbox_format", "auto") or "auto")
        bundle.layout_hf_correction = bool(layout_cfg.get("hf_correction", False))
        bundle.layout_hf_top_ratio = float(layout_cfg.get("hf_top_ratio", 0.08) or 0.08)
        bundle.layout_hf_bottom_ratio = float(layout_cfg.get("hf_bottom_ratio", 0.08) or 0.08)
        bundle.layout_hf_width_ratio = float(layout_cfg.get("hf_width_ratio", 0.5) or 0.5)
        bundle.layout_second_pass = bool(layout_cfg.get("second_pass", False))
        bundle.layout_second_pass_min_dets = int(layout_cfg.get("second_pass_min_dets", 2) or 2)
        bundle.layout_second_pass_min_nontext = int(layout_cfg.get("second_pass_min_nontext", 0) or 0)
        bundle.layout_second_pass_min_avg_score = float(layout_cfg.get("second_pass_min_avg_score", 0.0) or 0.0)
        bundle.layout_second_pass_input_size = int(layout_cfg.get("second_pass_input_size", 1280) or 1280)
        bundle.layout_second_pass_score_threshold = float(layout_cfg.get("second_pass_score_threshold", 0.15) or 0.15)
    else:
        bundle.layout_class_map = {i: c for i, c in enumerate(DEFAULT_LAYOUT_CLASSES)}

    table_cfg = (fallback_cfg.get("table_refiner") or {}) or {}
    table_transformer_cfg = (table_cfg.get("transformer") or {}) or {}
    if TableTransformerParser is not None and table_transformer_cfg:
        try:
            bundle.table_transformer = TableTransformerParser(table_transformer_cfg)
        except Exception as e:
            bundle.table_transformer = None
            bundle.model_disabled_reason.append(f"table_transformer_init_error:{str(e)[:120]}")

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


def _area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def _center(b):
    x1, y1, x2, y2 = b
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _overlap_1d(a1, a2, b1, b2):
    return max(0, min(a2, b2) - max(a1, b1))

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


CLASS_NMS_THRESHOLDS = {
    "table": 0.3,
    "figure": 0.3,
    "image": 0.3,
    "paragraph": 0.5,
    "formula": 0.4,
    "title": 0.4,
    "header": 0.4,
    "footer": 0.4,
}
DEFAULT_NMS_THRESHOLD = 0.45


def _nms_single_class(detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    if not detections:
        return []
    dets = sorted(detections, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best["bbox"], d["bbox"]) < iou_threshold]
    return keep


def _nms_python(detections: List[Dict[str, Any]], iou_threshold: float = None) -> List[Dict[str, Any]]:
    if not detections:
        return []
    if iou_threshold is None:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for det in detections:
            groups[det.get("type", det.get("label", "unknown"))].append(det)
        results = []
        for cls, dets in groups.items():
            threshold = CLASS_NMS_THRESHOLDS.get(cls, DEFAULT_NMS_THRESHOLD)
            results.extend(_nms_single_class(dets, threshold))
        return results
    else:
        return _nms_single_class(detections, iou_threshold)


def _union_bbox(bboxes: List[List[float]]) -> List[float]:
    if not bboxes:
        return [0, 0, 0, 0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def crop_roi(image: Any, bbox: List[float], pad: int = 0) -> Any:
    """Crop ROI from a PIL Image with optional padding and bbox clamping.

    Args:
        image: PIL Image object.
        bbox:  [x1, y1, x2, y2] in pixel coordinates.
        pad:   Extra pixels to add around the bbox (clamped to image bounds).

    Returns:
        Cropped PIL Image, or None if bbox is invalid.
    """
    if image is None or Image is None:
        return None
    w, h = image.size
    x1 = max(0, int(round(bbox[0])) - pad)
    y1 = max(0, int(round(bbox[1])) - pad)
    x2 = min(w, int(round(bbox[2])) + pad)
    y2 = min(h, int(round(bbox[3])) + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


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



def _coarse_type_onehot(b: Dict[str, Any]) -> Dict[str, float]:
    t = (b.get("type") or "").strip()
    if t == "text":
        t = "paragraph"
    if t in ("paragraph", "title", "list_item", "header", "footer", "page_number", "formula", "unknown"):
        return {"text": 1.0, "table": 0.0, "figure": 0.0, "caption": 0.0, "other": 0.0}
    if t == "table":
        return {"text": 0.0, "table": 1.0, "figure": 0.0, "caption": 0.0, "other": 0.0}
    if t in ("figure", "chart"):
        return {"text": 0.0, "table": 0.0, "figure": 1.0, "caption": 0.0, "other": 0.0}
    if t == "caption":
        return {"text": 0.0, "table": 0.0, "figure": 0.0, "caption": 1.0, "other": 0.0}
    return {"text": 0.0, "table": 0.0, "figure": 0.0, "caption": 0.0, "other": 1.0}


def _meta_float(block: Dict[str, Any], name: str, default: float = 0.0) -> float:
    meta = block.get("meta") or {}
    keys = [name]
    if not name.startswith("_"):
        keys.append(f"_{name}")
    for key in keys:
        if isinstance(meta, dict) and key in meta:
            try:
                return float(meta[key])
            except Exception:
                pass
        if key in block:
            try:
                return float(block[key])
            except Exception:
                pass
    return default


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


_TEXT_CORR_RUNTIME_CFG: Dict[str, Any] = {
    "enabled": False,
    "use_phrase_rules": True,
    "use_domain_terms": True,
    "use_char_lm": True,
    "normalize_punctuation": False,
}


def _configure_text_correction_runtime(cfg: Dict[str, Any]) -> None:
    """Configure OCR text correction behavior from runtime config."""
    global _TEXT_CORR_RUNTIME_CFG
    ocr_cfg = (((cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}) if isinstance(cfg, dict) else {})
    tcfg = (ocr_cfg.get("text_correction", {}) or {}) if isinstance(ocr_cfg, dict) else {}
    _TEXT_CORR_RUNTIME_CFG = {
        "enabled": bool(tcfg.get("enabled", False)),
        "use_phrase_rules": bool(tcfg.get("use_phrase_rules", True)),
        "use_domain_terms": bool(tcfg.get("use_domain_terms", True)),
        "use_char_lm": bool(tcfg.get("use_char_lm", True)),
        "normalize_punctuation": bool(tcfg.get("normalize_punctuation", False)),
    }


def postprocess_ocr_text(text: str, block_type: str = "paragraph") -> str:
    """OCR 文本后处理"""
    if not text:
        return text

    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()

    text = re.sub(r'([a-zA-Z0-9])([\u4e00-\u9fff])', r'\1 \2', text)
    text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z0-9])', r'\1 \2', text)

    text = text.replace('\u2014', '-')
    text = text.replace('\u2026', '...')
    text = text.replace('\u00a0', ' ')  # non-breaking space

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    if _TEXT_CORR_RUNTIME_CFG.get("enabled", False):
        text = text_correction(text, _TEXT_CORR_RUNTIME_CFG)

    return text


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


_CAPTION_PATTERN = re.compile(
    r'(?i)^[\s\[\(]*'
    r'(figure|fig|table|tab|图|表)'
    r'[\s.:：-]*'
    r'(S?\d+(?:[.\-]\d+)*)'
    r'(?:\s*[-–—]\s*(S?\d+(?:[.\-]\d+)*))?'
    r'(?:\s*[\(\[]\s*([a-zA-Z])\s*[\)\]])?'
    r'(?:\s*(?:和|and|&|,)\s*'
    r'(?:figure|fig|table|tab|图|表)?[\s.:：-]*'
    r'(S?\d+(?:[.\-]\d+)*)'
    r'(?:\s*[\(\[]\s*([a-zA-Z])\s*[\)\]])?)?'
)


def _extract_caption_info(text: str) -> Dict[str, Any]:
    if not text:
        return {"type": None, "number": None, "main_number": None, "range_end": None, "sub": None, "second_number": None, "second_sub": None}
    m = _CAPTION_PATTERN.match(text.strip())
    if not m:
        return {"type": None, "number": None, "main_number": None, "range_end": None, "sub": None, "second_number": None, "second_sub": None}
    keyword = (m.group(1) or "").lower()
    main_number = (m.group(2) or "").strip() or None
    range_end = (m.group(3) or "").strip() or None
    sub = (m.group(4) or "").lower() or None
    second_number = (m.group(5) or "").strip() or None
    second_sub = (m.group(6) or "").lower() or None
    number = None
    if main_number:
        main_digits = main_number.lower().lstrip("s")
        if main_digits.isdigit():
            try:
                number = int(main_digits)
            except Exception:
                number = None

    if keyword in ("figure", "fig", "图"):
        cap_type = "figure"
    elif keyword in ("table", "tab", "表"):
        cap_type = "table"
    else:
        cap_type = None

    return {
        "type": cap_type,
        "number": number,
        "main_number": main_number,
        "range_end": range_end,
        "sub": sub,
        "second_number": second_number,
        "second_sub": second_sub,
    }


def _normalize_caption_number(number: Any) -> Optional[str]:
    if number is None:
        return None
    value = str(number).strip().lower()
    if not value:
        return None
    return value.lstrip("s")


def _get_caption_target_numbers(caption_info: Dict[str, Any]) -> List[str]:
    numbers: List[str] = []
    main = _normalize_caption_number(caption_info.get("main_number"))
    if main:
        numbers.append(main)

    range_end = _normalize_caption_number(caption_info.get("range_end"))
    if main and range_end and main.isdigit() and range_end.isdigit():
        start = int(main)
        end = int(range_end)
        if start <= end and end - start <= 4:
            numbers = [str(num) for num in range(start, end + 1)]
        elif range_end not in numbers:
            numbers.append(range_end)

    second = _normalize_caption_number(caption_info.get("second_number"))
    if second and second not in numbers:
        numbers.append(second)
    return numbers


def _target_group_type(target_type: str) -> str:
    return "figure" if target_type in ("figure", "chart") else target_type


def _caption_rank_matches(rank: Optional[int], caption_numbers: List[str]) -> bool:
    if rank is None or not caption_numbers:
        return False
    rank_str = str(rank)
    for number in caption_numbers:
        if number == rank_str or number.startswith(rank_str + ".") or number.startswith(rank_str + "-"):
            return True
    return False


def _caption_type_matches_target(caption_info: Dict[str, Any], target_type: str) -> bool:
    cap_type = caption_info.get("type")
    if cap_type is None:
        return True
    if cap_type == "figure" and target_type in ("figure", "chart"):
        return True
    if cap_type == "table" and target_type == "table":
        return True
    return False


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


def _parse_yolo_output(output: Any, num_classes: int, score_threshold: float,
                       bbox_format: str = "auto") -> List[Dict[str, Any]]:
    """
    Parse YOLO-like ONNX outputs with broad shape compatibility:
      - (1, N, 4+1+C)  cx,cy,w,h, objectness, cls_scores...  (YOLOv5/v8 multi-class)
      - (1, N, 4+C)    cx,cy,w,h, cls_scores...              (anchor-free)
      - (1, N, 6/7)    cx,cy,w,h, score, cls_id[, extra]     (single-score compact)
      - (N, *)         same without batch dim
      - (1, 4+C, N)    transposed DocLayout-YOLO export       (automatically transposed)
    bbox_format: "auto" detects xywh vs xyxy heuristically; "xywh" / "xyxy" forces format.
    """
    if np is None:
        return []
    arr = np.array(output, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]  # (N, C) or (C, N) after squeeze

    if (arr.ndim == 2
            and arr.shape[0] < arr.shape[1]
            and arr.shape[0] >= (4 + num_classes)
            and arr.shape[1] > arr.shape[0] * 2):
        arr = arr.T

    if arr.ndim != 2:
        sys.stderr.write(
            f"[debug] _parse_yolo_output: unexpected array shape {arr.shape} after reshape; skipping\n"
        )
        return []

    dets: List[Dict[str, Any]] = []
    n_cols = arr.shape[1]

    for row in arr:
        det = row.tolist()
        cls_idx: int = 0
        score: float = 0.0

        if n_cols >= 5 + num_classes:
            cx, cy, w, h, obj = det[0], det[1], det[2], det[3], det[4]
            cls_scores = det[5:5 + num_classes]
            cls_idx = int(np.argmax(cls_scores))
            score = float(obj) * float(cls_scores[cls_idx])
            fmt = "xywh"
        elif n_cols >= 4 + num_classes:
            cx, cy, w, h = det[0], det[1], det[2], det[3]
            cls_scores = det[4:4 + num_classes]
            cls_idx = int(np.argmax(cls_scores))
            score = float(cls_scores[cls_idx])
            fmt = "xywh"
        elif n_cols >= 6:
            cx, cy, w, h, score, cls_idx = det[0], det[1], det[2], det[3], det[4], det[5]
            score, cls_idx = float(score), int(cls_idx)
            fmt = "xywh"  # will be overridden by bbox_format below
        else:
            sys.stderr.write(
                f"[debug] _parse_yolo_output: row too short ({n_cols} cols, need >=6); skipping row\n"
            )
            continue

        if score < score_threshold:
            continue

        resolved_fmt = bbox_format if bbox_format in ("xywh", "xyxy") else fmt
        if bbox_format == "auto" and n_cols < 4 + num_classes:
            if det[2] > det[0] and det[3] > det[1]:
                resolved_fmt = "xyxy"
            else:
                resolved_fmt = "xywh"

        dets.append({
            "bbox": [float(cx), float(cy), float(w), float(h)],
            "label_idx": cls_idx,
            "score": score,
            "format": resolved_fmt,
        })
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


def _postprocess_header_footer(results: List[Dict[str, Any]], orig_h: int, orig_w: int,
                               top_ratio: float, bottom_ratio: float,
                               width_ratio: float) -> List[Dict[str, Any]]:
    """
    Heuristic post-correction: re-label paragraph/caption blocks that sit in
    the top or bottom margin regions as header/footer respectively.
    """
    top_thresh = orig_h * top_ratio
    bot_thresh = orig_h * (1.0 - bottom_ratio)
    page_width = max(orig_w, 1)
    out = []
    for r in results:
        new_r = dict(r)
        label = r.get("label", "unknown")
        if label in ("paragraph", "caption"):
            x1, y1, x2, y2 = r["bbox"]
            block_w = x2 - x1
            if block_w / page_width >= width_ratio:
                if y2 <= top_thresh:
                    new_r["label"] = "header"
                elif y1 >= bot_thresh:
                    new_r["label"] = "footer"
        out.append(new_r)
    return out


def _run_layout_once(img: Any, sess: Any, input_size: int, num_classes: int,
                     score_thr: float, bbox_format: str,
                     orig_w: int, orig_h: int) -> List[Dict[str, Any]]:
    """
    Run one forward pass through the layout detector session.
    Returns raw results (bbox in original image coords, label mapped to SUPPORTED_BLOCK_TYPES).
    """
    padded, scale, pad, _ow, _oh = _letterbox_resize(img, input_size)
    if padded is None:
        return []
    tensor = padded.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    try:
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: tensor})
        output_names = [o.name for o in sess.get_outputs()]
    except Exception as e:
        sys.stderr.write(f"[warning] layout detector inference error: {e}\n")
        return []

    dets: List[Dict[str, Any]] = []
    if len(outputs) == 1:
        dets = _parse_yolo_output(outputs[0], num_classes, score_thr, bbox_format)
    else:
        dets = _parse_fasterrcnn_outputs(dict(zip(output_names, outputs)), score_thr)

    results: List[Dict[str, Any]] = []
    for det in dets:
        bbox_in = _convert_to_xyxy(det, input_size)
        bbox = _map_to_original_coords(bbox_in, scale, pad, orig_w, orig_h)
        if bbox[2] - bbox[0] < 5 or bbox[3] - bbox[1] < 5:
            continue
        results.append(det | {"bbox": bbox})
    return results


def _run_layout_detector(img_path: str, cfg: Dict[str, Any], models: ModelBundle, debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns list of {"bbox":[x1,y1,x2,y2], "label":str, "score":float}
    Falls back to empty list (caller should use OCR fallback) when unavailable.

    Supports:
    - class-aware NMS by default (nms_mode="class_aware") to avoid suppressing
      nearby different-class boxes (e.g. caption next to figure/table).
    - optional header/footer post-hoc refinement (hf_refine_enabled=True).
    - optional second-pass at higher resolution for difficult/sparse pages.
    """
    t0 = _now_ms()
    if models.layout_detector is None:
        sys.stderr.write("[warning] Layout detector not available; falling back to OCR-based block detection.\n")
        debug["layout_detector_status"] = "no_model"
        debug["layout_ms"] = 0.0
        return []
    if Image is None or np is None:
        sys.stderr.write("[warning] Layout detector dependencies (PIL/numpy) missing; falling back.\n")
        debug["layout_detector_status"] = "missing_deps"
        debug["layout_ms"] = 0.0
        return []

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        sys.stderr.write(f"[warning] Cannot open image for layout detection ({img_path}): {e}\n")
        debug["layout_detector_status"] = f"image_error:{str(e)[:80]}"
        debug["layout_ms"] = round(_now_ms() - t0, 2)
        return []

    orig_w, orig_h = img.size
    input_size = int(models.layout_input_size or 1024)
    num_classes = max(1, len(models.layout_class_map))
    score_thr = float(models.layout_score_threshold or 0.3)
    bbox_format = str(models.layout_bbox_format or "auto")
    sess = models.layout_detector

    raw_dets = _run_layout_once(img, sess, input_size, num_classes, score_thr,
                                bbox_format, orig_w, orig_h)
    if not raw_dets and models.layout_input_size == input_size:
        debug["layout_detector_status"] = "preprocess_failed"
        debug["layout_ms"] = round(_now_ms() - t0, 2)
        return []

    debug["layout_pass1_dets"] = len(raw_dets)

    second_pass_triggered = False
    if models.layout_second_pass:
        n_dets = len(raw_dets)
        n_nontext = sum(
            1 for d in raw_dets
            if models.layout_class_map.get(int(d.get("label_idx", -1)), "unknown")
            in NON_TEXT_BLOCK_TYPES
        )
        avg_score = (sum(float(d.get("score", 0)) for d in raw_dets) / max(n_dets, 1)) if n_dets else 0.0
        trigger = (
            n_dets < models.layout_second_pass_min_dets
            or n_nontext <= models.layout_second_pass_min_nontext
            or (models.layout_second_pass_min_avg_score > 0 and avg_score < models.layout_second_pass_min_avg_score)
        )
        if trigger:
            second_pass_triggered = True
            sp_size = int(models.layout_second_pass_input_size or 1280)
            sp_thr = float(models.layout_second_pass_score_threshold or 0.15)
            raw_dets2 = _run_layout_once(img, sess, sp_size, num_classes, sp_thr,
                                         bbox_format, orig_w, orig_h)
            if len(raw_dets2) > len(raw_dets):
                raw_dets = raw_dets2
            debug["layout_pass2_dets"] = len(raw_dets2)

    debug["layout_second_pass"] = second_pass_triggered

    results: List[Dict[str, Any]] = []
    for det in raw_dets:
        bbox = det["bbox"]
        label = models.layout_class_map.get(int(det["label_idx"]), "unknown")
        if label == "text":
            label = "paragraph"
        if label not in SUPPORTED_BLOCK_TYPES:
            label = "unknown"
        results.append({"bbox": bbox, "label": label, "score": float(det["score"])})

    if models.layout_nms_class_aware:
        results = _nms_python(results)
    else:
        results = _nms_python(results, float(models.layout_nms_threshold or 0.5))

    layout_pp_cfg = (((cfg.get("fallback_models", {}) or {}).get("layout_detector", {}) or {}).get("postprocess", {}) or {})
    if bool(layout_pp_cfg.get("nested_suppression", True)):
        nst_iou = float(layout_pp_cfg.get("nested_iou_threshold", 0.92) or 0.92)
        nst_cont = float(layout_pp_cfg.get("nested_containment_threshold", 0.94) or 0.94)
        results, nst_stats = suppress_nested_detections(results, iou_threshold=nst_iou, containment_threshold=nst_cont)
        debug["layout_suppressed_nested"] = int(nst_stats.get("suppressed_nested", 0))

    if models.layout_hf_correction:
        results = _postprocess_header_footer(
            results, orig_h, orig_w,
            models.layout_hf_top_ratio,
            models.layout_hf_bottom_ratio,
            models.layout_hf_width_ratio,
        )

    debug["layout_detector_status"] = "ok"
    debug["layout_nms_class_aware"] = models.layout_nms_class_aware
    debug["layout_detections"] = len(results)
    debug["layout_ms"] = round(_now_ms() - t0, 2)
    return results


_PADDLE_LAYOUT_ENGINE = None
_PADDLE_LAYOUT_ENGINE_KIND = "none"
_PADDLE_LAYOUT_DISABLED_REASON = ""


def _init_paddle_layout():
    """延迟初始化布局检测引擎（单例）。"""
    global _PADDLE_LAYOUT_ENGINE, _PADDLE_LAYOUT_ENGINE_KIND
    if _PADDLE_LAYOUT_ENGINE is not None:
        return _PADDLE_LAYOUT_ENGINE
    if _PADDLE_LAYOUT_DISABLED_REASON:
        _PADDLE_LAYOUT_ENGINE_KIND = "disabled"
        return None

    try:
        from paddleocr import LayoutDetection
        _PADDLE_LAYOUT_ENGINE = LayoutDetection(threshold=0.2)
        _PADDLE_LAYOUT_ENGINE_KIND = "layout_detection_v3"
        return _PADDLE_LAYOUT_ENGINE
    except Exception:
        pass

    try:
        from paddleocr import PPStructure
        _PADDLE_LAYOUT_ENGINE = PPStructure(
            table=False,
            ocr=False,
            show_log=False,
            layout=True,
        )
        _PADDLE_LAYOUT_ENGINE_KIND = "ppstructure_legacy"
        return _PADDLE_LAYOUT_ENGINE
    except Exception:
        pass

    try:
        from paddleocr import PaddleOCR
        _PADDLE_LAYOUT_ENGINE = PaddleOCR(show_log=False, layout=True)
        _PADDLE_LAYOUT_ENGINE_KIND = "paddleocr_layout_legacy"
        return _PADDLE_LAYOUT_ENGINE
    except Exception:
        pass

    _PADDLE_LAYOUT_ENGINE_KIND = "none"
    return None


_PADDLE_LAYOUT_LABEL_MAP = {
    "title": "title",
    "document_title": "title",
    "paragraph_title": "title",
    "text": "paragraph",
    "paragraph": "paragraph",
    "abstract": "paragraph",
    "references": "paragraph",
    "reference": "paragraph",
    "content": "paragraph",
    "footnote": "paragraph",
    "list": "list_item",
    "list_item": "list_item",
    "figure": "figure",
    "image": "figure",
    "table": "table",
    "chart": "chart",
    "formula": "formula",
    "equation": "formula",
    "formula_number": "formula",
    "figure_caption": "caption",
    "figure_title": "caption",
    "table_caption": "caption",
    "caption": "caption",
    "header": "header",
    "footer": "footer",
    "number": "page_number",
    "page_number": "page_number",
    "page number": "page_number",
    "page_no": "page_number",
    "pagenumber": "page_number",
    "unknown": "unknown",
}


def _map_paddle_layout_label(raw_label: str) -> str:
    key = str(raw_label or "unknown").strip().lower().replace("-", "_").replace(" ", "_")
    mapped = _PADDLE_LAYOUT_LABEL_MAP.get(key)
    if mapped:
        return mapped
    if "caption" in key or key.endswith("_title"):
        return "caption"
    if "formula" in key or "equation" in key:
        return "formula"
    if "table" in key:
        return "table"
    if "chart" in key:
        return "chart"
    if "image" in key or "figure" in key:
        return "figure"
    if "header" in key:
        return "header"
    if "footer" in key:
        return "footer"
    if "page" in key and "number" in key:
        return "page_number"
    if "title" in key:
        return "title"
    if "text" in key or "paragraph" in key:
        return "paragraph"
    return "unknown"


def _bbox_from_paddle_coordinate(coord: Any) -> Optional[List[int]]:
    if coord is None:
        return None
    vals: List[float] = []
    if isinstance(coord, (list, tuple)):
        for v in coord:
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    vals.extend([float(v[0]), float(v[1])])
                except Exception:
                    continue
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    continue

    if len(vals) >= 8 and len(vals) % 2 == 0:
        xs = vals[0::2]
        ys = vals[1::2]
        if xs and ys:
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            if x2 > x1 and y2 > y1:
                return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    if len(vals) >= 4:
        x1, y1, x2, y2 = vals[:4]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        if x2 > x1 and y2 > y1:
            return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

    return None


def _run_paddle_layout(img_path: str, debug: Dict[str, Any], cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """使用 Paddle 系布局检测，返回 {bbox,label,score} 列表。"""
    global _PADDLE_LAYOUT_ENGINE, _PADDLE_LAYOUT_ENGINE_KIND, _PADDLE_LAYOUT_DISABLED_REASON
    t0 = _now_ms()
    engine = _init_paddle_layout()
    if engine is None:
        if _PADDLE_LAYOUT_DISABLED_REASON:
            debug["paddle_layout_status"] = f"disabled:{_PADDLE_LAYOUT_DISABLED_REASON[:80]}"
        else:
            debug["paddle_layout_status"] = "unavailable"
        debug["paddle_layout_backend"] = _PADDLE_LAYOUT_ENGINE_KIND
        debug["paddle_layout_ms"] = 0.0
        return []

    dets: List[Dict[str, Any]] = []
    try:
        if hasattr(engine, "predict"):
            result = engine.predict(input=img_path)
        else:
            result = engine(img_path)

        chunks = list(result) if result is not None else []
        for chunk in chunks:
            payload = getattr(chunk, "json", chunk)
            if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
                payload = payload.get("res", {})

            items: List[Dict[str, Any]] = []
            if isinstance(payload, dict) and isinstance(payload.get("boxes"), list):
                items = [it for it in payload.get("boxes", []) if isinstance(it, dict)]
            elif isinstance(payload, list):
                items = [it for it in payload if isinstance(it, dict)]
            elif isinstance(payload, dict):
                items = [payload]

            for item in items:
                bbox = _bbox_from_paddle_coordinate(item.get("coordinate") or item.get("bbox") or item.get("region"))
                if not bbox:
                    continue
                raw_label = item.get("label") or item.get("type") or item.get("cls_name") or "unknown"
                label = _map_paddle_layout_label(raw_label)
                score = float(item.get("score", item.get("conf", 1.0)) or 1.0)
                dets.append({"bbox": bbox, "label": label, "score": score})

        if dets:
            dets = _nms_python(dets)

            layout_pp_cfg = (((cfg or {}).get("fallback_models", {}) or {}).get("layout_detector", {}) or {})
            layout_pp_cfg = (layout_pp_cfg.get("postprocess", {}) or {}) if isinstance(layout_pp_cfg, dict) else {}
            if bool(layout_pp_cfg.get("nested_suppression", True)):
                nst_iou = float(layout_pp_cfg.get("nested_iou_threshold", 0.92) or 0.92)
                nst_cont = float(layout_pp_cfg.get("nested_containment_threshold", 0.94) or 0.94)
                dets, nst_stats = suppress_nested_detections(dets, iou_threshold=nst_iou, containment_threshold=nst_cont)
                debug["paddle_layout_suppressed_nested"] = int(nst_stats.get("suppressed_nested", 0))

            if Image is not None:
                try:
                    with Image.open(img_path) as im:
                        orig_w, orig_h = im.size
                    dets = _postprocess_header_footer(dets, orig_h, orig_w, 0.08, 0.08, 0.5)
                except Exception:
                    pass

        debug["paddle_layout_status"] = "ok"
        debug["paddle_layout_backend"] = _PADDLE_LAYOUT_ENGINE_KIND
        debug["paddle_layout_detections"] = len(dets)
        debug["paddle_layout_ms"] = round(_now_ms() - t0, 2)
        return dets
    except Exception as e:
        if _is_fatal_paddle_runtime_error(e):
            _PADDLE_LAYOUT_ENGINE = None
            _PADDLE_LAYOUT_ENGINE_KIND = "disabled"
            _PADDLE_LAYOUT_DISABLED_REASON = str(e)[:160]
            debug["paddle_layout_disabled"] = True
        debug["paddle_layout_status"] = f"error:{str(e)[:120]}"
        debug["paddle_layout_backend"] = _PADDLE_LAYOUT_ENGINE_KIND
        debug["paddle_layout_ms"] = round(_now_ms() - t0, 2)
        return []


def _parse_tesseract_tsv(tsv_text: str, x_offset: int = 0, y_offset: int = 0) -> List[Dict[str, Any]]:
    """Parse tesseract TSV output into OCR line dicts."""
    lines: List[Dict[str, Any]] = []
    if not tsv_text:
        return lines

    rows = tsv_text.splitlines()
    if not rows:
        return lines

    for row in rows[1:]:
        parts = row.split("\t")
        if len(parts) < 12:
            continue
        try:
            conf = float(parts[10])
        except Exception:
            conf = -1.0

        text = (parts[11] or "").strip()
        if not text:
            continue

        try:
            left = int(float(parts[6])) + x_offset
            top = int(float(parts[7])) + y_offset
            width = int(float(parts[8]))
            height = int(float(parts[9]))
        except Exception:
            continue

        if width <= 0 or height <= 0:
            continue

        bbox = [float(left), float(top), float(left + width), float(top + height)]
        lines.append({
            "bbox": bbox,
            "text": postprocess_ocr_text(text),
            "score": 0.0 if conf < 0 else conf / 100.0,
        })

    return lines


def _ocr_tesseract(img_path: str, lang: str = "chi_sim+eng", psm: int = 6,
                   x_offset: int = 0, y_offset: int = 0) -> List[Dict[str, Any]]:
    """Fallback OCR by tesseract CLI when PaddleOCR is unavailable."""
    if not _has_tesseract_binary():
        return []
    cmd = [
        "tesseract",
        img_path,
        "stdout",
        "--oem", "1",
        "--psm", str(psm),
        "-l", lang,
        "tsv",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        _disable_tesseract("binary_not_found")
        return []
    except Exception:
        return []

    if proc.returncode != 0:
        err_lower = (proc.stderr or "").lower()
        if "error opening data file" in err_lower or "failed loading language" in err_lower:
            _disable_tesseract(f"lang_data_missing:{lang}")
        elif "not found" in err_lower and "tesseract" in err_lower:
            _disable_tesseract("binary_not_found_runtime")
        return []

    return _parse_tesseract_tsv(proc.stdout, x_offset=x_offset, y_offset=y_offset)


def _call_paddle_ocr(ocr_engine: Any, inp: Any):
    """Call PaddleOCR across API variants (prefer 3.x predict)."""
    last_error = None

    if hasattr(ocr_engine, "predict"):
        try:
            return ocr_engine.predict(inp)
        except Exception as e:
            last_error = e

    for kwargs in ({"cls": False}, {}):
        try:
            return ocr_engine.ocr(inp, **kwargs)
        except TypeError as e:
            last_error = e
            if kwargs and "cls" in str(e):
                continue
            raise
        except Exception as e:
            last_error = e
            if kwargs and "cls" in str(e):
                continue

    if last_error is not None:
        raise last_error
    raise RuntimeError("PaddleOCR call failed")


def _extract_paddle_lines(result: Any, x_offset: int = 0, y_offset: int = 0) -> List[Dict[str, Any]]:
    """Parse PaddleOCR old/new outputs into unified line list."""
    lines: List[Dict[str, Any]] = []
    if not result:
        return lines

    def _append_line(quad, text, score=None):
        if quad is None:
            return
        xs: List[float] = []
        ys: List[float] = []
        try:
            if isinstance(quad, (list, tuple)) and len(quad) >= 4 and not isinstance(quad[0], (list, tuple)):
                x1, y1, x2, y2 = [float(v) for v in quad[:4]]
                xs = [x1, x2]
                ys = [y1, y2]
            else:
                for p in quad:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        xs.append(float(p[0]))
                        ys.append(float(p[1]))
        except Exception:
            return
        if not xs or not ys:
            return
        bbox = [float(min(xs) + x_offset), float(min(ys) + y_offset), float(max(xs) + x_offset), float(max(ys) + y_offset)]
        payload = {"bbox": bbox, "text": postprocess_ocr_text(str(text or ""))}
        if score is not None:
            try:
                payload["score"] = float(score)
            except Exception:
                pass
        if payload["text"].strip():
            lines.append(payload)

    def _to_payload(item: Any) -> Any:
        if item is None:
            return None
        payload = item
        # Paddle 3.x result wrapper often exposes `.json` property.
        maybe_json = getattr(item, "json", None)
        if maybe_json is not None:
            try:
                payload = maybe_json() if callable(maybe_json) else maybe_json
            except Exception:
                payload = item
        if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
            return payload.get("res", {})
        return payload

    def _parse_item(item):
        if item is None:
            return
        item = _to_payload(item)

        if isinstance(item, dict) and ("rec_texts" in item or "dt_polys" in item or "rec_polys" in item):
            texts = item.get("rec_texts") or []
            polys = item.get("rec_polys") or item.get("dt_polys") or []
            scores = item.get("rec_scores") or []
            n = min(len(texts), len(polys))
            for i in range(n):
                score = scores[i] if i < len(scores) else None
                _append_line(polys[i], texts[i], score)
            return

        if isinstance(item, dict) and isinstance(item.get("ocr_res"), list):
            _parse_item(item.get("ocr_res"))
            return

        if isinstance(item, list):
            for ln in item:
                if not ln or len(ln) < 2:
                    continue
                quad = ln[0]
                txt_info = ln[1]
                text = txt_info[0] if isinstance(txt_info, (list, tuple)) and txt_info else str(txt_info)
                score = txt_info[1] if isinstance(txt_info, (list, tuple)) and len(txt_info) > 1 else None
                _append_line(quad, text, score)
            return

        if isinstance(item, dict) and all(k in item for k in ("bbox", "text")):
            _append_line(item.get("bbox"), item.get("text"), item.get("score"))
            return

        if isinstance(item, Iterable) and not isinstance(item, (str, bytes, dict)):
            for sub in item:
                _parse_item(sub)
            return

    _parse_item(result)

    return lines


def _filter_ocr_lines(lines: List[Dict[str, Any]], min_score: float = 0.0, min_text_len: int = 1) -> List[Dict[str, Any]]:
    """Filter low-confidence / empty OCR lines to improve precision."""
    out: List[Dict[str, Any]] = []
    for ln in lines:
        text = (ln.get("text") or "").strip()
        if len(text) < max(1, int(min_text_len)):
            continue
        score = float(ln.get("score", 1.0) or 0.0)
        if score < float(min_score):
            continue
        out.append(ln)
    return out


def _ocr_full_image(img_path: str, cfg: Dict[str, Any], models: ModelBundle, debug: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Full-image OCR with caching. Returns lines [{"bbox":[...],"text":...}]
    """
    if debug is None:
        debug = {}
    t0 = _now_ms()

    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    enabled = bool(cfg.get("pipeline", {}).get("enable_ocr", False) or ocr_cfg.get("enabled", False))
    line_score_thr = float(ocr_cfg.get("line_score_threshold", 0.0) or 0.0)
    line_min_text_len = int(ocr_cfg.get("line_min_text_len", 1) or 1)
    full_image_max_side = int(ocr_cfg.get("full_image_max_side", 1600) or 0)
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
        tesseract_enabled = bool(ocr_cfg.get("tesseract_fallback", True)) and _has_tesseract_binary()
        if tesseract_enabled:
            tess_lang = str(ocr_cfg.get("tesseract_lang", "chi_sim+eng") or "chi_sim+eng")
            tess_psm = int(ocr_cfg.get("tesseract_psm", 6) or 6)
            lines = _ocr_tesseract(img_path, lang=tess_lang, psm=tess_psm)
            lines = _filter_ocr_lines(lines, min_score=0.0, min_text_len=line_min_text_len)
            if cache_path:
                _save_ocr_cache(cache_path, lines)
            if lines:
                debug["ocr_status"] = "tesseract"
                debug["ocr_ms"] = round(_now_ms() - t0, 2)
                return lines
        debug["ocr_status"] = "no_engine"
        debug["ocr_ms"] = round(_now_ms() - t0, 2)
        return []

    try:
        scale_back = 1.0
        ocr_input: Any = img_path
        if full_image_max_side > 0 and Image is not None and np is not None:
            with Image.open(img_path) as _im:
                rgb = _im.convert("RGB")
                w, h = rgb.size
                m = max(w, h)
                if m > full_image_max_side:
                    ratio = float(full_image_max_side) / float(m)
                    new_w = max(32, int(round(w * ratio)))
                    new_h = max(32, int(round(h * ratio)))
                    resized = rgb.resize((new_w, new_h), Image.BILINEAR)
                    ocr_input = np.array(resized)
                    scale_back = 1.0 / ratio
        result = _call_paddle_ocr(models.ocr_engine, ocr_input)
        lines = _extract_paddle_lines(result)
        if abs(scale_back - 1.0) > 1e-6:
            for ln in lines:
                b = ln.get("bbox", [0, 0, 0, 0])
                if len(b) >= 4:
                    ln["bbox"] = [
                        int(round(float(b[0]) * scale_back)),
                        int(round(float(b[1]) * scale_back)),
                        int(round(float(b[2]) * scale_back)),
                        int(round(float(b[3]) * scale_back)),
                    ]
        lines = _filter_ocr_lines(lines, min_score=line_score_thr, min_text_len=line_min_text_len)
    except Exception as e:
        if _is_fatal_paddle_runtime_error(e):
            models.ocr_engine = None
            debug["ocr_engine_disabled"] = str(e)[:160]
        tesseract_enabled = bool(ocr_cfg.get("tesseract_fallback", True)) and _has_tesseract_binary()
        if tesseract_enabled:
            tess_lang = str(ocr_cfg.get("tesseract_lang", "chi_sim+eng") or "chi_sim+eng")
            tess_psm = int(ocr_cfg.get("tesseract_psm", 6) or 6)
            lines = _ocr_tesseract(img_path, lang=tess_lang, psm=tess_psm)
            lines = _filter_ocr_lines(lines, min_score=0.0, min_text_len=line_min_text_len)
            if lines:
                if cache_path:
                    _save_ocr_cache(cache_path, lines)
                debug["ocr_status"] = "paddle_error_tesseract"
                debug["ocr_error"] = str(e)[:120]
                debug["ocr_ms"] = round(_now_ms() - t0, 2)
                return lines
        debug["ocr_status"] = f"error:{str(e)[:80]}"
        debug["ocr_ms"] = round(_now_ms() - t0, 2)
        return []

    if cache_path:
        _save_ocr_cache(cache_path, lines)

    debug["ocr_status"] = "ok"
    debug["ocr_ms"] = round(_now_ms() - t0, 2)
    return lines


def _should_do_roi_ocr(block: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    block_type = (block.get("type") or "")
    if block_type not in TEXT_BLOCK_TYPES:
        return False

    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    roi_types = ocr_cfg.get("roi_types") or ["paragraph", "title", "list_item"]
    roi_type_set = {str(t).strip().lower() for t in roi_types}
    if block_type not in roi_type_set:
        return False

    bbox = block.get("bbox", [0, 0, 0, 0])
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    if min(w, h) < 8 or w * h < 64:
        return False

    force = bool(ocr_cfg.get("force", False))
    if force:
        return True

    text = (block.get("text") or "").strip()
    source = (block.get("source") or "").lower()
    if text and source not in ("heuristic", "layout_detector"):
        return False

    return True


def _ocr_roi(img_path: str, roi_bbox: List[float], cfg: Dict[str, Any], models: ModelBundle,
             page_size: Tuple[int, int], debug: Optional[Dict[str, Any]] = None,
             page_image: Any = None) -> List[Dict[str, Any]]:
    """
    ROI OCR with caching. If PaddleOCR supports ndarray input, use it; else
    fallback to temp file.
    """
    if debug is None:
        debug = {}

    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    line_score_thr = float(ocr_cfg.get("line_score_threshold", 0.0) or 0.0)
    line_min_text_len = int(ocr_cfg.get("line_min_text_len", 1) or 1)
    tesseract_enabled = bool(ocr_cfg.get("tesseract_fallback", True)) and _has_tesseract_binary()

    if models.ocr_engine is None:
        if Image is None or not tesseract_enabled:
            return []
        if page_image is not None:
            img = page_image
        else:
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
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            roi_img.save(tmp_path)
        try:
            tess_lang = str(ocr_cfg.get("tesseract_lang", "chi_sim+eng") or "chi_sim+eng")
            tess_psm = int(ocr_cfg.get("tesseract_psm", 6) or 6)
            lines = _ocr_tesseract(tmp_path, lang=tess_lang, psm=tess_psm, x_offset=x1, y_offset=y1)
            return _filter_ocr_lines(lines, min_score=0.0, min_text_len=line_min_text_len)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    if Image is None:
        return []
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

    if page_image is not None:
        img = page_image
    else:
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
            result = _call_paddle_ocr(models.ocr_engine, roi_arr)
        else:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                roi_img.save(tmp.name)
                tmp_path = tmp.name
            try:
                result = _call_paddle_ocr(models.ocr_engine, tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        lines = _extract_paddle_lines(result, x_offset=x1, y_offset=y1)
        lines = _filter_ocr_lines(lines, min_score=line_score_thr, min_text_len=line_min_text_len)
    except Exception as e:
        if _is_fatal_paddle_runtime_error(e):
            models.ocr_engine = None
            debug["ocr_engine_disabled"] = str(e)[:160]
        if tesseract_enabled:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                roi_img.save(tmp_path)
            try:
                tess_lang = str(ocr_cfg.get("tesseract_lang", "chi_sim+eng") or "chi_sim+eng")
                tess_psm = int(ocr_cfg.get("tesseract_psm", 6) or 6)
                lines = _ocr_tesseract(tmp_path, lang=tess_lang, psm=tess_psm, x_offset=x1, y_offset=y1)
                lines = _filter_ocr_lines(lines, min_score=0.0, min_text_len=line_min_text_len)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        else:
            return []

    if cache_path:
        _save_ocr_cache(cache_path, lines)

    return lines


def _batch_ocr_pending_blocks(blocks: List[Dict[str, Any]], pending_indices: List[int], image: Any,
                              ocr_engine: Any, line_score_thr: float, line_min_text_len: int,
                              batch_size: int = 12) -> Dict[int, List[Dict[str, Any]]]:
    """
    Batch OCR for pending text blocks and return parsed lines per block id.

    Raises:
        Exception: Bubble up fatal Paddle runtime errors so caller can disable engine.
    """
    if ocr_engine is None or image is None or np is None or not pending_indices:
        return {}

    out: Dict[int, List[Dict[str, Any]]] = {}
    bs = max(1, int(batch_size))

    for start in range(0, len(pending_indices), bs):
        chunk = pending_indices[start:start + bs]
        rois: List[Any] = []
        metas: List[Tuple[int, int, int]] = []  # (block_idx, x_offset, y_offset)

        for bid in chunk:
            if bid < 0 or bid >= len(blocks):
                continue
            bbox = blocks[bid].get("bbox", [0, 0, 0, 0])
            roi = crop_roi(image, bbox)
            if roi is None:
                continue
            x1 = max(0, int(round(bbox[0])))
            y1 = max(0, int(round(bbox[1])))
            rois.append(np.array(roi))
            metas.append((bid, x1, y1))

        if not rois:
            continue

        batch_results: Optional[List[Any]] = None
        try:
            batch_raw = _call_paddle_ocr(ocr_engine, rois)
            if isinstance(batch_raw, list) and len(batch_raw) == len(rois):
                batch_results = batch_raw
        except Exception as e:
            if _is_fatal_paddle_runtime_error(e):
                raise

        if batch_results is None:
            batch_results = []
            for roi_arr in rois:
                try:
                    batch_results.append(_call_paddle_ocr(ocr_engine, roi_arr))
                except Exception as e:
                    if _is_fatal_paddle_runtime_error(e):
                        raise
                    batch_results.append(None)

        for raw, (bid, x_off, y_off) in zip(batch_results, metas):
            lines = _extract_paddle_lines(raw, x_offset=x_off, y_offset=y_off)
            lines = _filter_ocr_lines(lines, min_score=line_score_thr, min_text_len=line_min_text_len)
            if lines:
                out[bid] = lines

    return out


def _enrich_blocks_with_roi_ocr(blocks: List[Dict[str, Any]], img_path: str, page: Dict[str, Any],
                                cfg: Dict[str, Any], models: ModelBundle, debug: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    ROI OCR on text blocks, returning (blocks, all_lines).

    Optimization:
    - For layout-heavy pages, run one full-image OCR and reuse lines per block.
    - Only fallback to per-ROI OCR for uncovered blocks.
    """
    t0 = _now_ms()
    all_lines: List[Dict[str, Any]] = []
    ocr_cfg = (cfg.get("fallback_models", {}) or {}).get("ocr", {}) or {}
    has_tesseract_fallback = bool(ocr_cfg.get("tesseract_fallback", True)) and _has_tesseract_binary()
    line_score_thr = float(ocr_cfg.get("line_score_threshold", 0.0) or 0.0)
    line_min_text_len = int(ocr_cfg.get("line_min_text_len", 1) or 1)

    if models.ocr_engine is None and not has_tesseract_fallback:
        debug["roi_ocr_status"] = "no_engine"
        debug["roi_ocr_ms"] = 0.0
        return blocks, all_lines

    # In degraded environments (no OCR engine + heuristic single full-page block),
    # ROI OCR almost never helps but can be very expensive.
    if (
        len(blocks) == 1
        and models.ocr_engine is None
        and (blocks[0].get("source") or "") == "heuristic"
        and not (blocks[0].get("text") or "").strip()
    ):
        debug["roi_ocr_status"] = "skipped_heuristic_single_block"
        debug["roi_ocr_ms"] = 0.0
        return blocks, all_lines

    page_size = (int(page.get("width", 1000)), int(page.get("height", 1400)))

    reuse_full = bool(ocr_cfg.get("reuse_full_image_for_layout", True))
    reuse_min_blocks = int(ocr_cfg.get("reuse_min_blocks", 8) or 8)
    candidate_blocks = [b for b in blocks if _should_do_roi_ocr(b, cfg)]
    reused_blocks = 0
    if reuse_full and len(candidate_blocks) >= reuse_min_blocks:
        full_lines = _ocr_full_image(img_path, cfg, models, debug=None)
        full_lines = _filter_ocr_lines(full_lines, min_score=line_score_thr, min_text_len=line_min_text_len)
        if full_lines:
            all_lines.extend(full_lines)
            for b in candidate_blocks:
                bbox = b.get("bbox", [0, 0, 0, 0])
                lines = _lines_in_bbox(full_lines, bbox)
                if not lines:
                    continue
                lines.sort(key=lambda ln: (ln.get("bbox", [0, 0, 0, 0])[1], ln.get("bbox", [0, 0, 0, 0])[0]))
                txt = "\n".join((ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip())
                if not txt.strip():
                    continue
                b["text"] = txt
                b["source"] = "roi_ocr_reuse"
                reused_blocks += 1

    pending_indices = [
        i for i, b in enumerate(blocks)
        if _should_do_roi_ocr(b, cfg) and not (b.get("text") or "").strip()
    ]

    pil_img: Any = None
    if pending_indices and models.ocr_engine is not None and Image is not None:
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            pil_img = None

    roi_cnt = 0
    if pending_indices and models.ocr_engine is not None and pil_img is not None and np is not None:
        batch_size = int(ocr_cfg.get("batch_size", 12) or 12)
        try:
            batch_lines_map = _batch_ocr_pending_blocks(
                blocks=blocks,
                pending_indices=pending_indices,
                image=pil_img,
                ocr_engine=models.ocr_engine,
                line_score_thr=line_score_thr,
                line_min_text_len=line_min_text_len,
                batch_size=batch_size,
            )
            for bid, lines in batch_lines_map.items():
                if not lines:
                    continue
                roi_cnt += 1
                all_lines.extend(lines)
                blocks[bid]["text"] = "\n".join(
                    (ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip()
                )
                blocks[bid]["source"] = "roi_ocr_batch"
        except Exception as e:
            if _is_fatal_paddle_runtime_error(e):
                models.ocr_engine = None
                debug["ocr_engine_disabled"] = str(e)[:160]
            debug["roi_ocr_batch_error"] = str(e)[:120]

    for bid in pending_indices:
        b = blocks[bid]
        if (b.get("text") or "").strip():
            continue
        bbox = b.get("bbox", [0, 0, 0, 0])
        lines = _ocr_roi(img_path, bbox, cfg, models, page_size, debug, page_image=pil_img)
        if lines:
            roi_cnt += 1
            all_lines.extend(lines)
            b["text"] = "\n".join((ln.get("text") or "").strip() for ln in lines if (ln.get("text") or "").strip())
            b["source"] = "roi_ocr"

    try:
        if pil_img is not None:
            pil_img.close()
    except Exception:
        pass

    debug["roi_ocr_status"] = "ok"
    debug["roi_ocr_reused_blocks"] = int(reused_blocks)
    debug["roi_ocr_count"] = roi_cnt
    debug["roi_ocr_ms"] = round(_now_ms() - t0, 2)
    return blocks, all_lines


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


def _normalize_blocks(ir: Dict[str, Any]) -> Dict[str, Any]:
    w = max(1, ir.get("page", {}).get("width", 1))
    h = max(1, ir.get("page", {}).get("height", 1))
    new_blocks: List[Dict[str, Any]] = []
    for idx, b in enumerate(ir.get("blocks", [])):
        b["id"] = idx
        b["bbox"] = _clamp_bbox(b.get("bbox", [0, 0, 0, 0]), w, h)
        t = b.get("type", "paragraph") or "paragraph"
        t = str(t).strip().lower().replace("-", "_").replace(" ", "_")
        if t == "text":
            t = "paragraph"
        if t in {"page_no", "page_num", "pagenumber"}:
            t = "page_number"
        if t not in SUPPORTED_BLOCK_TYPES:
            t = "unknown"
        b["type"] = t
        new_blocks.append(b)
    ir["blocks"] = new_blocks
    return ir


def _looks_like_page_number_text(text: str) -> bool:
    """Heuristic check for page-number-like strings."""
    if not text:
        return False

    raw = str(text).strip()
    if not raw or len(raw) > 24:
        return False

    compact = _SPACE_RE.sub("", raw)
    if not compact:
        return False

    stripped = compact.strip("-—_·•.()（）[]【】<>《》")
    lowered = stripped.lower()

    if re.fullmatch(r"\d{1,4}", lowered):
        return True
    if re.fullmatch(r"(page|p)\.?\d{1,4}", lowered):
        return True
    if re.fullmatch(r"第\d{1,4}页", stripped):
        return True
    if re.fullmatch(r"[ivxlcdm]{1,7}", lowered):
        return True

    return False


def _promote_page_number_blocks(ir: Dict[str, Any]) -> Dict[str, Any]:
    """
    Promote footer-like short text blocks near the bottom margin to page_number.

    This improves class recall for the competition category without changing
    HTML format constraints.
    """
    blocks = ir.get("blocks", []) or []
    if not blocks:
        return ir

    page = ir.get("page", {}) or {}
    page_w = float(max(1.0, page.get("width", 1)))
    page_h = float(max(1.0, page.get("height", 1)))

    promoted = 0
    for block in blocks:
        block_type = str(block.get("type", "")).strip().lower()
        if block_type in {"table", "figure", "chart", "formula", "title", "caption"}:
            continue

        bbox = block.get("bbox", [0, 0, 0, 0])
        if len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        block_w = max(1.0, x2 - x1)
        block_h = max(1.0, y2 - y1)
        y_mid = (y1 + y2) * 0.5 / page_h

        if y_mid < 0.78:
            continue
        if block_w / page_w > 0.35:
            continue
        if block_h / page_h > 0.08:
            continue

        txt = (block.get("text") or "").strip()
        if not _looks_like_page_number_text(txt):
            continue

        block["type"] = "page_number"
        promoted += 1

    if promoted > 0:
        ir.setdefault("debug", {})["page_number_promoted"] = promoted

    return ir


def _apply_dataset_type_priors(ir: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply conservative type priors learned from train/eval label statistics.

    Main target is title/paragraph confusion:
    - Titles are usually shorter, narrower, and upper/middle-page.
    - Long wide sentence-like text should remain paragraph.
    """
    blocks = ir.get("blocks", []) or []
    page = ir.get("page", {}) or {}
    page_w = float(max(1.0, page.get("width", 1)))
    page_h = float(max(1.0, page.get("height", 1)))

    p2t = 0
    t2p = 0
    for b in blocks:
        btype = str(b.get("type", "")).strip().lower()
        if btype not in {"title", "paragraph"}:
            continue

        txt = (b.get("text") or "").strip()
        if not txt:
            continue

        x1, y1, x2, y2 = [float(v) for v in (b.get("bbox", [0, 0, 0, 0])[:4] or [0, 0, 0, 0])]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        rel_w = bw / page_w
        rel_h = bh / page_h
        y_mid = ((y1 + y2) * 0.5) / page_h
        txt_len = len(txt)
        has_sentence_punc = any(ch in txt for ch in ("。", "；", ";", "！", "!", "？", "?", "，", ","))
        line_cnt = txt.count("\n") + 1

        if btype == "paragraph":
            # Short heading-like text -> title.
            if (
                4 <= txt_len <= 36
                and line_cnt <= 2
                and not has_sentence_punc
                and rel_w <= 0.72
                and 0.015 <= rel_h <= 0.13
                and y_mid <= 0.82
            ):
                b["type"] = "title"
                p2t += 1
        else:  # title
            # Long or sentence-like wide text -> paragraph.
            if (
                (txt_len >= 90 or has_sentence_punc)
                and rel_w >= 0.62
                and rel_h <= 0.11
            ):
                b["type"] = "paragraph"
                t2p += 1

    if p2t or t2p:
        ir.setdefault("debug", {})["dataset_prior_paragraph_to_title"] = int(p2t)
        ir.setdefault("debug", {})["dataset_prior_title_to_paragraph"] = int(t2p)
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
    prompt = sample.get("prompt", sample.get("prefix", cfg.get("default_prompt", ""))) or ""
    img_path = os.path.join(image_root, image_path) if image_root and not os.path.isabs(image_path) else image_path

    w, h = 1000, 1400
    has_sample_page = False
    sample_page = sample.get("page", {}) if isinstance(sample, dict) else {}
    if isinstance(sample_page, dict):
        sw = _as_int(sample_page.get("width", 0), 0)
        sh = _as_int(sample_page.get("height", 0), 0)
        if sw > 0 and sh > 0:
            w, h = sw, sh
            has_sample_page = True
    if Image is not None:
        if not has_sample_page:
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

    layout_cfg = (cfg.get("fallback_models", {}) or {}).get("layout_detector", {}) or {}
    layout_enabled = bool(layout_cfg.get("enabled", False))  # default false (config-consistent)
    layout_dets: List[Dict[str, Any]] = []
    if layout_enabled and models.layout_detector is not None:
        layout_dets = _run_layout_detector(img_path, cfg, models, debug)

    if not layout_dets and models.layout_detector is None:
        layout_dets = _run_paddle_layout(img_path, debug, cfg=cfg)

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

    lines = _ocr_full_image(img_path, cfg, models, debug)
    if lines:
        ir["blocks"] = _group_lines_to_paragraphs(lines)
        debug["blocks_source"] = "ocr_full_grouped"
    else:
        sys.stderr.write(f"[warning] No layout detection or OCR output for {image_path}; using full-page paragraph fallback.\n")
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


def predict_block_types(ir: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle) -> Dict[str, Any]:
    clf = models.block_classifier
    schema = models.feature_schema_block
    if clf is None or not schema:
        return ir

    page = ir.get("page", {})
    blocks = ir.get("blocks", [])

    _enrich_blocks_with_column_info(blocks, page)
    height_pct_map = _height_percentiles(blocks)

    feat_vecs: List[List[float]] = []
    feat_dicts: List[Dict[str, float]] = []
    for b in blocks:
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
    footer_ids = {i for i,t in enumerate(types) if t in ("footer", "page_number")}

    block_columns: Dict[int, int] = {}
    if any("_column_id" in b for b in blocks):
        for idx, block in enumerate(blocks):
            try:
                block_columns[idx] = int(round(float(block.get("_column_id", 0))))
            except Exception:
                block_columns[idx] = 0
    else:
        x_centers_sorted = sorted([c[0] for c in centers])
        column_breaks: List[float] = []
        if len(x_centers_sorted) > 10:
            gaps = [x_centers_sorted[i + 1] - x_centers_sorted[i] for i in range(len(x_centers_sorted) - 1)]
            med_gap = safe_median(gaps, default=0.15 * w)
            for i, gap in enumerate(gaps):
                if gap > max(0.15 * w, 2.5 * med_gap):
                    column_breaks.append((x_centers_sorted[i] + x_centers_sorted[i + 1]) * 0.5)
            column_breaks.sort()

        def _fallback_col_id(cx: float) -> int:
            col = 0
            for threshold in column_breaks:
                if cx > threshold:
                    col += 1
                else:
                    break
            return col

        for idx, (cx, _) in enumerate(centers):
            block_columns[idx] = _fallback_col_id(cx)

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
        col_i = block_columns.get(i, 0)
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
            col_j = block_columns.get(j, 0)

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
        initial_scores.append((y * 10000 + x, i))
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
    target_rank_map: Dict[Tuple[str, Any], int] = {}
    grouped_targets: Dict[str, List[Dict[str, Any]]] = {"figure": [], "table": []}
    for target in targets_all:
        grouped_targets.setdefault(_target_group_type(target.get("type", "figure")), []).append(target)
    for group_type, targets in grouped_targets.items():
        for rank, target in enumerate(sorted(targets, key=lambda item: (item["bbox"][1], item["bbox"][0])), 1):
            target_rank_map[(group_type, target["id"])] = rank

    cand_pairs: List[Tuple[int, int]] = []
    pair_priors: Dict[Tuple[int, int], float] = {}
    for ci, c in enumerate(captions):
        c_center = _center(c["bbox"])
        c_info = cap_infos[ci]
        cap_numbers = _get_caption_target_numbers(c_info)
        scored: List[Tuple[float, int]] = []
        for tj, t in enumerate(targets_all):
            t_type = t.get("type", "figure")
            if not _caption_type_matches_target(c_info, t_type):
                continue

            t_center = _center(t["bbox"])
            dy = c_center[1] - t_center[1]
            dx = abs(c_center[0] - t_center[0])
            dist = math.hypot(c_center[0] - t_center[0], c_center[1] - t_center[1])

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

            prior = 1.0
            rank = target_rank_map.get((_target_group_type(t_type), t["id"]))
            if _caption_rank_matches(rank, cap_numbers):
                prior *= 1.35
            elif cap_numbers and rank is not None and cap_numbers[0].isdigit():
                prior *= max(0.7, 1.0 - 0.08 * abs(int(cap_numbers[0]) - rank))

            x_ov = _overlap_1d(c["bbox"][0], c["bbox"][2], t["bbox"][0], t["bbox"][2])
            min_w = max(1.0, min(c["bbox"][2] - c["bbox"][0], t["bbox"][2] - t["bbox"][0]))
            if x_ov / min_w > 0.5:
                prior *= 1.12
            if t_type in ("figure", "chart") and 0 <= dy <= 0.1 * h:
                prior *= 1.18
            if t_type == "table" and abs(dy) <= 0.08 * h:
                prior *= 1.08

            pair_priors[(ci, tj)] = prior
            scored.append((dist / max(0.5, prior), tj))

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
            for idx, p in enumerate(pred):
                ci, tj = cand_pairs[idx]
                prior = pair_priors.get((ci, tj), 1.0)
                if isinstance(p, (list, tuple)) or (np is not None and isinstance(p, np.ndarray)):
                    base_score = safe_max(list(p))
                else:
                    base_score = float(p)
                scores.append(float(min(1.0, base_score * prior)))
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

            base *= pair_priors.get((ci, tj), 1.0)
            scores.append(float(min(1.0, base)))

    m, n = len(captions), len(targets_all)
    large = 1.0
    cost = [[large for _ in range(n)] for _ in range(m)]
    score_by_pair: Dict[Tuple[int, int], float] = {}
    for idx, (ci, tj) in enumerate(cand_pairs):
        score_value = float(scores[idx])
        cost[ci][tj] = 1.0 - score_value
        score_by_pair[(ci, tj)] = score_value

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
    used_pairs: Set[Tuple[Any, Any]] = set()
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
        used_pairs.add((cap_ids[ci], tar_ids[tj]))

    for ci, c_info in enumerate(cap_infos):
        cap_numbers = _get_caption_target_numbers(c_info)
        if len(cap_numbers) <= 1:
            continue
        extras: List[Tuple[float, int]] = []
        for tj, target in enumerate(targets_all):
            if (cap_ids[ci], tar_ids[tj]) in used_pairs:
                continue
            if not _caption_type_matches_target(c_info, target.get("type", "figure")):
                continue
            rank = target_rank_map.get((_target_group_type(target.get("type", "figure")), target["id"]))
            if not _caption_rank_matches(rank, cap_numbers):
                continue
            score_value = score_by_pair.get((ci, tj), 0.0)
            if score_value < cap_thr * 0.85:
                continue
            extras.append((score_value, tj))

        extras.sort(key=lambda item: -item[0])
        for score_value, tj in extras[:2]:
            caption_links.append({
                "caption_id": cap_ids[ci],
                "target_id": tar_ids[tj],
                "score": float(score_value),
            })
            used_pairs.add((cap_ids[ci], tar_ids[tj]))

    return caption_links
_TABLE_STRUCTURE_ENGINE = None
_TABLE_STRUCTURE_ENGINE_STATUS = "none"


def _init_table_structure_engine() -> Any:
    """Lazy-init PaddleOCR table structure engine (SLANet family)."""
    global _TABLE_STRUCTURE_ENGINE, _TABLE_STRUCTURE_ENGINE_STATUS
    if _TABLE_STRUCTURE_ENGINE is not None:
        return _TABLE_STRUCTURE_ENGINE
    if _TABLE_STRUCTURE_ENGINE_STATUS.startswith("error"):
        return None

    try:
        from paddleocr import TableStructureRecognition
        _TABLE_STRUCTURE_ENGINE = TableStructureRecognition()
        _TABLE_STRUCTURE_ENGINE_STATUS = "ok"
    except Exception as e:
        _TABLE_STRUCTURE_ENGINE = None
        _TABLE_STRUCTURE_ENGINE_STATUS = f"error:{str(e)[:120]}"
    return _TABLE_STRUCTURE_ENGINE


def _parse_table_html_tokens(tokens: List[Any]) -> List[List[Dict[str, Any]]]:
    """Parse SLANet-like token sequence into row/cell skeleton with spans."""
    if not tokens:
        return []

    html_txt = "".join(str(t) for t in tokens)
    row_chunks = re.findall(r"<tr[^>]*>(.*?)</tr>", html_txt, flags=re.IGNORECASE | re.DOTALL)
    rows: List[List[Dict[str, Any]]] = []
    for chunk in row_chunks:
        cell_matches = re.findall(r"<td([^>]*)>(.*?)</td>", chunk, flags=re.IGNORECASE | re.DOTALL)
        cur_row: List[Dict[str, Any]] = []
        for attrs, _ in cell_matches:
            rowspan = 1
            colspan = 1
            m = re.search(r"rowspan\s*=\s*\"(\d+)\"", attrs, flags=re.IGNORECASE)
            if m:
                rowspan = max(1, int(m.group(1)))
            m = re.search(r"colspan\s*=\s*\"(\d+)\"", attrs, flags=re.IGNORECASE)
            if m:
                colspan = max(1, int(m.group(1)))
            cur_row.append({"rowspan": rowspan, "colspan": colspan})
        if cur_row:
            rows.append(cur_row)
    return rows


def _extract_table_structure_model(table_block: Dict[str, Any], table_lines: List[Dict[str, Any]],
                                  page_image: Any, cfg: Dict[str, Any],
                                  debug: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use Paddle table structure model on ROI; fallback returns None."""
    engine = _init_table_structure_engine()
    if engine is None or Image is None or np is None:
        debug.setdefault("table_model_status", _TABLE_STRUCTURE_ENGINE_STATUS)
        return None

    bbox = table_block.get("bbox", [0, 0, 0, 0])
    if len(bbox) < 4:
        return None

    x1, y1, x2, y2 = [int(round(v)) for v in bbox[:4]]
    if x2 <= x1 or y2 <= y1:
        return None

    table_cfg = (cfg.get("fallback_models", {}) or {}).get("table_refiner", {}) or {}
    max_side = int(table_cfg.get("max_side", 1400) or 1400)

    try:
        roi = page_image.crop((x1, y1, x2, y2))
        scale = 1.0
        if max_side > 0:
            rw, rh = roi.size
            m = max(rw, rh)
            if m > max_side:
                scale = float(max_side) / float(m)
                roi = roi.resize((max(1, int(rw * scale)), max(1, int(rh * scale))), Image.LANCZOS)

        result = engine.predict(input=np.array(roi))
        chunks = list(result) if result is not None else []
        if not chunks:
            debug["table_model_status"] = "empty"
            return None

        payload = getattr(chunks[0], "json", chunks[0])
        if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
            payload = payload["res"]
        if not isinstance(payload, dict):
            debug["table_model_status"] = "invalid_payload"
            return None

        tokens = payload.get("structure") or []
        raw_cells = payload.get("bbox") or []
        rows = _parse_table_html_tokens(tokens)
        if not rows:
            debug["table_model_status"] = "parse_rows_failed"
            return None

        cell_bboxes: List[List[int]] = []
        for c in raw_cells:
            bb = _bbox_from_paddle_coordinate(c)
            if not bb:
                continue
            if scale != 1.0:
                inv = 1.0 / scale
                bb = [int(round(bb[0] * inv)), int(round(bb[1] * inv)), int(round(bb[2] * inv)), int(round(bb[3] * inv))]
            bb = [bb[0] + x1, bb[1] + y1, bb[2] + x1, bb[3] + y1]
            cell_bboxes.append(bb)

        cell_idx = 0
        filled_rows: List[List[Dict[str, Any]]] = []
        for row in rows:
            out_row: List[Dict[str, Any]] = []
            for cell in row:
                cb = cell_bboxes[cell_idx] if cell_idx < len(cell_bboxes) else [x1, y1, x2, y2]
                cell_idx += 1
                clines = _lines_in_bbox(table_lines, cb)
                clines.sort(key=lambda ln: (ln.get("bbox", [0, 0, 0, 0])[1], ln.get("bbox", [0, 0, 0, 0])[0]))
                ctext = " ".join((ln.get("text") or "").strip() for ln in clines if (ln.get("text") or "").strip())
                out_row.append({
                    "bbox": cb,
                    "text": ctext,
                    "rowspan": int(cell.get("rowspan", 1) or 1),
                    "colspan": int(cell.get("colspan", 1) or 1),
                })
            filled_rows.append(out_row)

        debug["table_model_status"] = "ok"
        debug["table_model_cells"] = len(cell_bboxes)
        return {
            "id": int(table_block.get("id", 0)),
            "bbox": table_block.get("bbox", [x1, y1, x2, y2]),
            "type": "table",
            "score": float(table_block.get("score", 1.0) or 1.0),
            "source": "table_structure_model",
            "rows": filled_rows,
        }
    except Exception as e:
        debug["table_model_status"] = f"error:{str(e)[:120]}"
        return None


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


def _extract_tables(blocks: List[Dict[str, Any]], ocr_lines: List[Dict[str, Any]], img_path: str,
                    page: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle, debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    t0 = _now_ms()
    page_size = (int(page.get("width", 1000)), int(page.get("height", 1400)))
    page_area = float(max(1.0, page_size[0] * page_size[1]))

    table_cfg = (cfg.get("fallback_models", {}) or {}).get("table_refiner", {}) or {}
    use_table_model = bool(table_cfg.get("enabled", True))
    use_transformer = bool((table_cfg.get("transformer") or {}).get("enabled", False))
    min_area_ratio = float(table_cfg.get("min_area_ratio", 0.005) or 0.005)

    page_image = None
    if (use_table_model or use_transformer) and Image is not None:
        try:
            page_image = Image.open(img_path).convert("RGB")
        except Exception:
            page_image = None

    tables: List[Dict[str, Any]] = []
    table_model_used = 0
    table_transformer_used = 0
    table_transformer_status = "disabled"
    for b in blocks:
        if b.get("type") != "table":
            continue

        table_bbox = b.get("bbox", [0, 0, 0, 0])

        table_lines = _lines_in_bbox(ocr_lines, table_bbox)
        if not table_lines and models.ocr_engine is not None:
            table_lines = _ocr_roi(img_path, table_bbox, cfg, models, page_size, debug, page_image=page_image)

        table_obj = None
        area_ratio = _area(table_bbox) / page_area if len(table_bbox) >= 4 else 0.0

        if use_transformer and page_image is not None and area_ratio >= min_area_ratio:
            try:
                parser = getattr(models, "table_transformer", None)
                if parser is not None and (hasattr(parser, "forward") or hasattr(parser, "predict")):
                    x1, y1, x2, y2 = [int(round(v)) for v in table_bbox[:4]]
                    if x2 > x1 and y2 > y1:
                        roi = page_image.crop((x1, y1, x2, y2))
                        if hasattr(parser, "forward"):
                            parsed = parser.forward(
                                roi,
                                ocr_lines=table_lines,
                                roi_offset=(x1, y1),
                                fallback_bbox=table_bbox,
                            )
                        else:
                            parsed = parser.predict(roi)
                        table_transformer_status = getattr(parser, "status", "unknown")
                        if parsed is not None:
                            rows = list(getattr(parsed, "rows", []) or [])
                            html_table = str(getattr(parsed, "html", "") or "")
                            table_obj = {
                                "id": int(b.get("id", 0)),
                                "bbox": table_bbox,
                                "type": "table",
                                "score": float(b.get("score", 1.0) or 1.0),
                                "source": "table_transformer",
                                "rows": rows if rows else [[{"bbox": table_bbox, "text": "", "rowspan": 1, "colspan": 1}]],
                                "html": html_table,
                            }
                            table_transformer_used += 1
            except Exception as e:
                table_transformer_status = f"error:{str(e)[:120]}"

        if use_table_model and page_image is not None and area_ratio >= min_area_ratio:
            if table_obj is None:
                table_obj = _extract_table_structure_model(b, table_lines, page_image, cfg, debug)
                if table_obj is not None:
                    table_model_used += 1

        if table_obj is None:
            table_obj = _extract_table_structure(b, table_lines, page)

        tables.append(table_obj)

    try:
        if page_image is not None:
            page_image.close()
    except Exception:
        pass

    debug["tables_extracted"] = len(tables)
    debug["table_model_used"] = int(table_model_used)
    debug["table_transformer_used"] = int(table_transformer_used)
    debug["table_transformer_status"] = table_transformer_status
    debug["table_ms"] = round(_now_ms() - t0, 2)
    return tables


def _extract_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: List[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    out.append(txt)
        return "\n".join(out).strip()
    return ""


def _openai_formula_from_roi(roi: Any, openai_cfg: Dict[str, Any], debug: Optional[Dict[str, Any]] = None) -> str:
    """Optional OpenAI vision formula OCR (model openai-5.4 by default)."""
    if not openai_cfg or not bool(openai_cfg.get("enabled", False)):
        return ""
    if requests is None or Image is None:
        return ""

    api_key_env = str(openai_cfg.get("api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY")
    api_key = (openai_cfg.get("api_key") or os.getenv(api_key_env) or "").strip()
    if not api_key:
        if debug is not None:
            debug.setdefault("openai_formula_status", "no_api_key")
        return ""

    model_name = str(openai_cfg.get("model", "openai-5.4") or "openai-5.4")
    base_url = str(openai_cfg.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")
    timeout_s = float(openai_cfg.get("timeout_s", 20) or 20)
    max_side = int(openai_cfg.get("max_side", 1024) or 1024)

    try:
        roi_img = roi
        if not isinstance(roi_img, Image.Image):
            roi_img = Image.fromarray(roi_img)
        roi_img = roi_img.convert("RGB")

        if max_side > 0:
            rw, rh = roi_img.size
            side = max(rw, rh)
            if side > max_side:
                ratio = float(max_side) / float(side)
                roi_img = roi_img.resize((max(1, int(rw * ratio)), max(1, int(rh * ratio))), Image.LANCZOS)

        buf = io.BytesIO()
        roi_img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        payload = {
            "model": model_name,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a formula OCR engine. Return only LaTeX for the formula in the image.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Recognize the formula and output LaTeX only. No explanation."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ],
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=timeout_s)
        if resp.status_code >= 400:
            if debug is not None:
                debug["openai_formula_status"] = f"http_{resp.status_code}"
            return ""

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            if debug is not None:
                debug["openai_formula_status"] = "empty_choices"
            return ""

        msg = choices[0].get("message") or {}
        text = _extract_message_text(msg.get("content"))
        text = text.strip().strip("`")
        text = re.sub(r"^latex\s*", "", text, flags=re.IGNORECASE)
        if debug is not None:
            debug["openai_formula_status"] = "ok"
        return sanitize_latex_expression(text)
    except Exception as e:
        if debug is not None:
            debug["openai_formula_status"] = f"error:{str(e)[:120]}"
        return ""


def _process_formula_blocks(
    blocks: List[Dict[str, Any]],
    img_path: Optional[str] = None,
    recognizer: Optional[Any] = None,
    openai_formula_cfg: Optional[Dict[str, Any]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    处理所有 formula 类型 block：
    - 若 latex 未设置，尝试用 FormulaRecognizer 识别（需要 img_path）
    - 若识别失败，回退使用 OpenAI-5.4（可选）
    - 若仍失败，回退使用块内 OCR 文本
    - 规范化 latex 字符串
    """
    page_img = None
    try:
        if img_path and Image is not None:
            page_img = Image.open(img_path).convert("RGB")
    except Exception:
        page_img = None

    openai_calls = 0
    openai_enabled = bool((openai_formula_cfg or {}).get("enabled", False))
    max_openai_calls = int((openai_formula_cfg or {}).get("max_calls_per_page", 2) or 2)
    if not openai_enabled and debug is not None:
        debug.setdefault("openai_formula_status", "disabled_by_policy")
    if debug is not None and recognizer is not None and hasattr(recognizer, "status"):
        try:
            debug["formula_transformer_status"] = str(getattr(recognizer, "status"))
        except Exception:
            pass

    for b in blocks:
        if b.get("type") != "formula":
            continue

        raw = (b.get("latex") or "").strip()
        roi = None
        if page_img is not None:
            try:
                bbox = b.get("bbox", [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    if x2 > x1 and y2 > y1:
                        roi = page_img.crop((x1, y1, x2, y2))
            except Exception:
                roi = None

        if raw == "" and roi is not None and recognizer is not None:
            try:
                raw = (recognizer.recognize(roi) or "").strip()
            except Exception:
                raw = ""

        if raw == "" and roi is not None and openai_enabled and openai_calls < max_openai_calls:
            raw = _openai_formula_from_roi(roi, openai_formula_cfg or {}, debug=debug)
            if raw:
                openai_calls += 1

        if raw == "":
            raw = (b.get("text") or "").strip()

        b["latex"] = sanitize_latex_expression(raw)

    if debug is not None:
        debug["openai_formula_calls"] = int(openai_calls)

    try:
        if page_img is not None:
            page_img.close()
    except Exception:
        pass

    return blocks

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
    _enrich_blocks_with_column_info(blocks, page)

    ro_hybrid_cfg = (cfg.get("decode", {}) or {}).get("reading_order_hybrid", {}) or {}
    use_hybrid_ro = bool(ro_hybrid_cfg.get("enabled", True))
    if use_hybrid_ro and _HAS_HYBRID_READING_ORDER and n > 1:
        try:
            ordered_blocks = _xycut_graph_sort(blocks, page=page, cfg=ro_hybrid_cfg)
            if isinstance(ordered_blocks, list) and len(ordered_blocks) == n:
                default_edge_score = float(ro_hybrid_cfg.get("default_edge_score", 0.95) or 0.95)
                order_edges = _build_chain_order_edges(ordered_blocks, default_score=default_edge_score)
                captions = [b for b in blocks if b.get("type") == "caption"]
                targets_all = [b for b in blocks if b.get("type") in ("figure", "table", "chart")]
                caption_links = _match_captions(captions, targets_all, page, cfg, models)

                ir.setdefault("relations", {})
                ir["relations"]["order_edges"] = order_edges
                ir["relations"]["caption_links"] = caption_links
                ir["relations"].setdefault("heading_parent", [])
                ir["relations"]["order_is_chain"] = (len(order_edges) == max(0, n - 1))
                ir["relations"]["order_method"] = "xycut_graph_hybrid"
                return ir
        except Exception:
            pass

    types = [b.get("type", "paragraph") for b in blocks]
    header_idx = {i for i, t in enumerate(types) if t == "header"}
    footer_idx = {i for i, t in enumerate(types) if t in ("footer", "page_number")}
    caption_idx = {i for i, t in enumerate(types) if t == "caption"}
    target_idx = {i for i, t in enumerate(types) if t in ("figure", "table", "chart")}

    max_blocks_cfg = int(cfg.get("pipeline", {}).get("max_blocks", 300) or 300)
    k_order = int(cfg.get("decode", {}).get("k_order", 8) or 8)
    cand_succ = _candidate_successors(blocks, page, k=k_order, max_blocks=max_blocks_cfg)

    _page_median_gap = float(page.get("_median_gap", 0))
    if _page_median_gap <= 0:
        page_h = max(1.0, float(page.get("height", 1)))
        if _HAS_READING_ORDER_UTILS:
            _page_median_gap = _compute_median_gap(blocks, page_h)
        else:
            _page_median_gap = page_h * 0.01
        page["_median_gap"] = _page_median_gap

    _col_count = float(page.get("_column_count", 1))

    feat_vecs: List[List[float]] = []
    pair_indices: List[Tuple[int, int]] = []
    for i, succs in cand_succ.items():
        for j in succs:
            feats = _pair_feature_dict(blocks[i], blocks[j], page, column_count=_col_count,
                                       median_gap=_page_median_gap)
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
            feats = _pair_feature_dict(blocks[i], blocks[j], page, column_count=_col_count,
                                       median_gap=_page_median_gap)
            dist = float(feats.get("center_dist_norm", 0.5))
            base = 1.0 / (1.0 + dist * 2.0)
            if feats.get("same_column_id", 0) > 0.5 and feats.get("is_above", 0) > 0.5:
                base *= 1.5
            elif feats.get("same_column", 0) > 0.5 and feats.get("is_above", 0) > 0.5:
                base *= 1.3
            if feats.get("same_row", 0) > 0.5 and feats.get("left_to_right", 0) > 0.5:
                base *= 1.3
            scores.append(float(min(1.0, base)))

    for idx, (i, j) in enumerate(pair_indices):
        score_map[(i, j)] = float(scores[idx])

    use_beam_search = bool(cfg.get("decode", {}).get("use_beam_search", False))
    beam_width = int(cfg.get("decode", {}).get("beam_width", 3) or 3)

    if use_beam_search and n <= 50:
        order_seq_idx = _beam_search_order(blocks, cand_succ, score_map, beam_width=beam_width)
    else:
        def apply_constraints(i: int, j: int, s: float) -> float:
            if s <= 0:
                return 0.0
            ci = _center(blocks[i]["bbox"])
            cj = _center(blocks[j]["bbox"])
            page_h = float(max(1.0, page.get("height", 1)))

            if j in header_idx and i not in header_idx:
                return 0.0
            if i in footer_idx and j not in footer_idx:
                if cj[1] < ci[1]:
                    return 0.0

            score = s

            if i in target_idx and j in caption_idx:
                if 0 <= (cj[1] - ci[1]) <= 0.12 * page_h:
                    score *= 1.6

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

        headers = [i for i in order_seq_idx if i in header_idx]
        footers = [i for i in order_seq_idx if i in footer_idx]
        body = [i for i in order_seq_idx if i not in header_idx and i not in footer_idx]

        def pos_key(i: int):
            bb = blocks[i].get("bbox", [0, 0, 0, 0])
            return (bb[1], bb[0])

        headers.sort(key=pos_key)
        footers.sort(key=pos_key)

        n_cols_order, col_idx_map = _detect_columns_for_order(blocks, page)
        if n_cols_order > 1:
            page_w = max(1.0, float(page.get("width", 1)))
            cross_col_thresh = 0.7
            cross_col_body: List[int] = []
            regular_body: Dict[int, List[int]] = {}
            for idx in body:
                bb = blocks[idx].get("bbox", [0, 0, 0, 0])
                bw = float(bb[2]) - float(bb[0])
                col_id = col_idx_map.get(idx, 0)
                if bw > cross_col_thresh * page_w:
                    cross_col_body.append(idx)
                else:
                    regular_body.setdefault(col_id, []).append(idx)
            cross_col_body.sort(key=pos_key)
            for col_id in regular_body:
                regular_body[col_id].sort(key=pos_key)
            body = []
            col_lists: Dict[int, List[int]] = {c: list(regular_body.get(c, [])) for c in range(n_cols_order)}
            col_ptrs: Dict[int, int] = {c: 0 for c in range(n_cols_order)}

            def _flush_body_to_y(y_limit: float) -> None:
                for c in range(n_cols_order):
                    ptr = col_ptrs[c]
                    col = col_lists[c]
                    while ptr < len(col):
                        bb = blocks[col[ptr]].get("bbox", [0, 0, 0, 0])
                        if float(bb[1]) < y_limit:
                            body.append(col[ptr])
                            ptr += 1
                        else:
                            break
                    col_ptrs[c] = ptr

            for cross_idx in cross_col_body:
                y1 = float(blocks[cross_idx].get("bbox", [0, 0, 0, 0])[1])
                _flush_body_to_y(y1)
                body.append(cross_idx)
            _flush_body_to_y(float("inf"))

        order_seq_idx = headers + body + footers

    order_edges: List[Dict[str, Any]] = []
    for a, b in zip(order_seq_idx[:-1], order_seq_idx[1:]):
        score = score_map.get((a, b), 1.0)
        order_edges.append({"u": blocks[a]["id"], "v": blocks[b]["id"], "score": float(score)})

    expected_edges = max(0, n - 1)
    if len(order_edges) != expected_edges:
        sorted_idx = sorted(
            range(n),
            key=lambda i: (
                0 if blocks[i].get("type") == "header" else 2 if blocks[i].get("type") in ("footer", "page_number") else 1,
                blocks[i].get("bbox", [0, 0, 0, 0])[1],
                blocks[i].get("bbox", [0, 0, 0, 0])[0],
            ),
        )
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
    rowspan = int(cell.get("rowspan", 1) or 1)
    colspan = int(cell.get("colspan", 1) or 1)
    text = _escape(cell.get("text", "") or "")
    
    attrs = []
    if rowspan > 1:
        attrs.append(f'rowspan="{rowspan}"')
    if colspan > 1:
        attrs.append(f'colspan="{colspan}"')
    
    if attrs:
        return f'<td {" ".join(attrs)}>{text}</td>'
    return f'<td>{text}</td>'


def _render_table_content(table_obj: Dict[str, Any]) -> str:
    """
    渲染表格内部结构

    评测要求: <table><tr><td>文本</td></tr></table>（不要 thead/tbody）
    """
    html_table = (table_obj.get("html") or "").strip()
    if html_table:
        if "<table" not in html_table.lower():
            html_table = f"<table>{html_table}</table>"
        return html_table

    rows = table_obj.get("rows", [])
    if not rows:
        return "<table><tr><td></td></tr></table>"

    parts = ["<table>"]
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(_render_table_cell(cell))
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


def _format_formula_text(latex: str) -> str:
    """Render formula content as display-style `$$...$$` text."""
    txt = sanitize_latex_expression((latex or "").strip())
    return ensure_display_math_wrapped(txt)


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
    - 公式: <div class="formula" data-bbox="x1 y1 x2 y2">latex</div>
    - 列表项: <div class="list_item" data-bbox="x1 y1 x2 y2">文本</div>
    - 标题说明: <div class="caption" data-bbox="x1 y1 x2 y2">文本</div>
    - 页眉: <div class="header" data-bbox="x1 y1 x2 y2">文本</div>
    - 页脚: <div class="footer" data-bbox="x1 y1 x2 y2">文本</div>
    - 页码: <div class="page_number" data-bbox="x1 y1 x2 y2">文本</div>
    """
    btype = (b.get("type") or "paragraph").lower().strip()
    bbox_str = _bbox_attr(b.get("bbox", []))
    text = _escape(b.get("text") or "")
    
    if btype == "title":
        return f'<h2 data-bbox="{bbox_str}">{text}</h2>'
    
    if btype == "paragraph":
        return f'<p data-bbox="{bbox_str}">{text}</p>'
    
    if btype == "list_item":
        return f'<div class="list_item" data-bbox="{bbox_str}">{text}</div>'
    
    if btype == "caption":
        return f'<div class="caption" data-bbox="{bbox_str}">{text}</div>'
    
    if btype == "figure":
        return f'<div class="image" data-bbox="{bbox_str}"></div>'
    
    if btype == "chart":
        return f'<div class="chart" data-bbox="{bbox_str}"></div>'
    
    if btype == "table":
        table_content = _render_table_content(table_obj or {})
        return f'<div class="table" data-bbox="{bbox_str}">{table_content}</div>'
    
    if btype == "formula":
        latex_text = b.get("latex") or b.get("text") or ""
        return f'<div class="formula" data-bbox="{bbox_str}">{_escape(_format_formula_text(latex_text))}</div>'
    
    if btype == "header":
        return f'<div class="header" data-bbox="{bbox_str}">{text}</div>'
    
    if btype == "footer":
        return f'<div class="footer" data-bbox="{bbox_str}">{text}</div>'

    if btype == "page_number":
        return f'<div class="page_number" data-bbox="{bbox_str}">{text}</div>'
    
    if btype == "unknown":
        return f'<p data-bbox="{bbox_str}">{text}</p>'
    
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
    
    id_to_block = {b["id"]: b for b in blocks if "id" in b}
    id_to_table = {t["id"]: t for t in tables if "id" in t}
    
    cap_to_target = {}
    for link in caption_links:
        cap_id = link.get("caption_id")
        tar_id = link.get("target_id")
        if cap_id is not None and tar_id is not None:
            cap_to_target[cap_id] = tar_id
    
    if not blocks:
        return "<body></body>"
    
    next_map: Dict[int, int] = {}
    indeg: Dict[int, int] = {b["id"]: 0 for b in blocks}
    
    for edge in order_edges:
        u = edge.get("u")
        v = edge.get("v")
        if u is not None and v is not None and u in indeg and v in indeg:
            next_map[u] = v
            indeg[v] = indeg.get(v, 0) + 1
    
    heads = [bid for bid, deg in indeg.items() if deg == 0]
    
    def pos_key(bid: int) -> Tuple[float, float]:
        b = id_to_block.get(bid)
        if b:
            bbox = b.get("bbox", [0, 0, 0, 0])
            return (float(bbox[1]), float(bbox[0]))  # (y, x)
        return (0.0, 0.0)
    
    heads.sort(key=pos_key)
    
    ordered_ids: List[int] = []
    visited = set()
    
    for head in heads:
        cur = head
        while cur is not None and cur not in visited:
            if cur in id_to_block:
                ordered_ids.append(cur)
                visited.add(cur)
            cur = next_map.get(cur)
    
    remaining = [bid for bid in id_to_block if bid not in visited]
    remaining.sort(key=pos_key)
    ordered_ids.extend(remaining)
    
    html_parts: List[str] = []
    
    for bid in ordered_ids:
        block = id_to_block.get(bid)
        if not block:
            continue
        
        caption_ref = cap_to_target.get(bid)
        
        table_obj = id_to_table.get(bid) if block.get("type") == "table" else None
        
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
    
    if not html_str.startswith("<body"):
        html_str = "<body>" + html_str
    if not html_str.endswith("</body>"):
        html_str = html_str + "</body>"
    
    if html_str in ("<body></body>", "<body> </body>", "<body/>"):
        debug["html_empty"] = True
    
    return html_str



def process_one(sample: Dict[str, Any], cfg: Dict[str, Any], models: ModelBundle,
                image_root: Optional[str] = None, rng_seed: Optional[int] = None) -> Dict[str, Any]:
    """处理单个样本，返回包含 answer 的结果"""
    if rng_seed is not None:
        random.seed(rng_seed)

    image = sample.get("image", "unknown")
    prompt = _sample_prompt(sample, cfg)
    debug: Dict[str, Any] = {"image": image}
    _configure_text_correction_runtime(cfg)
    debug["text_correction_enabled"] = bool(_TEXT_CORR_RUNTIME_CFG.get("enabled", False))

    t_total0 = _now_ms()

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

    ocr_lines: List[Dict[str, Any]] = []
    try:
        t1 = _now_ms()
        ir["blocks"], ocr_lines = _enrich_blocks_with_roi_ocr(ir.get("blocks", []), img_path, page, cfg, models, debug)
        debug["enrich_ms"] = round(_now_ms() - t1, 2)
    except Exception as e:
        debug["enrich_error"] = str(e)[:200]
        debug["enrich_ms"] = 0.0

    try:
        t2 = _now_ms()
        ir = predict_block_types(ir, cfg, models)
        debug["block_ms"] = round(_now_ms() - t2, 2)
    except Exception as e:
        debug["block_error"] = str(e)[:200]
        debug["block_ms"] = 0.0

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

    try:
        if not ir.get("tables"):
            ir["tables"] = _extract_tables(ir.get("blocks", []), ocr_lines, img_path, page, cfg, models, debug)
    except Exception as e:
        debug["table_error"] = str(e)[:200]
        ir["tables"] = []

    try:
        ir["blocks"] = _process_formula_blocks(
            ir.get("blocks", []),
            img_path=img_path,
            recognizer=getattr(models, "formula_recognizer", None),
            openai_formula_cfg=getattr(models, "openai_formula_cfg", {}) or {},
            debug=debug,
        )
    except Exception as e:
        debug["formula_error"] = str(e)[:200]

    try:
        layout_pp_cfg = (((cfg.get("fallback_models", {}) or {}).get("layout_detector", {}) or {}).get("postprocess", {}) or {})
        if bool(layout_pp_cfg.get("title_paragraph_refine", True)):
            ttp_title = float(layout_pp_cfg.get("title_boost_ratio", 1.35) or 1.35)
            ttp_para = float(layout_pp_cfg.get("paragraph_boost_ratio", 1.05) or 1.05)
            ir["blocks"], refine_stats = refine_title_paragraph_blocks(
                ir.get("blocks", []),
                ir.get("page", {}) or {},
                title_boost_ratio=ttp_title,
                paragraph_boost_ratio=ttp_para,
            )
            debug["title_paragraph_refine"] = True
            debug["paragraph_to_title"] = int(refine_stats.get("paragraph_to_title", 0))
            debug["title_to_paragraph"] = int(refine_stats.get("title_to_paragraph", 0))
        else:
            debug["title_paragraph_refine"] = False
    except Exception as e:
        debug["title_paragraph_refine_error"] = str(e)[:200]

    try:
        ir = _apply_dataset_type_priors(ir)
    except Exception as e:
        debug["dataset_prior_error"] = str(e)[:200]

    try:
        ir = _promote_page_number_blocks(ir)
    except Exception as e:
        debug["page_number_error"] = str(e)[:200]

    try:
        t3 = _now_ms()
        ir = predict_relations(ir, cfg, models)
        debug["rel_ms"] = round(_now_ms() - t3, 2)
    except Exception as e:
        debug["rel_error"] = str(e)[:200]
        debug["rel_ms"] = 0.0
        ir.setdefault("relations", {"order_edges": [], "caption_links": [], "heading_parent": []})

    try:
        t4 = _now_ms()
        ir = decode(ir, cfg)
        debug["decode_ms"] = round(_now_ms() - t4, 2)
    except Exception as e:
        debug["decode_error"] = str(e)[:200]
        debug["decode_ms"] = 0.0

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

    try:
        t6 = _now_ms()
        html_str = validate_and_fix_html(html_str, cfg, force_full_validate=(render_error or triggered), debug=debug)
        debug["validate_ms"] = round(_now_ms() - t6, 2)
    except Exception as e:
        debug["validate_error"] = str(e)[:200]
        debug["validate_ms"] = 0.0
        html_str = "<body></body>"

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

    blocks0 = ir.get("blocks", []) or []
    if (
        len(blocks0) <= 1
        and "low_blocks" in reasons
        and models.layout_detector is None
        and _PADDLE_LAYOUT_DISABLED_REASON
        and models.ocr_engine is None
    ):
        debug["fallback_skipped"] = "no_visual_engine_available"
        debug["fallback_ms"] = round(_now_ms() - t0, 2)
        return ir, []

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

    blocks = ir.get("blocks", [])
    ocr_lines: List[Dict[str, Any]] = []
    blocks, ocr_lines = _enrich_blocks_with_roi_ocr(blocks, img_path, ir.get("page", {}), cfg, models, debug)
    ir["blocks"] = blocks

    ir["tables"] = _extract_tables(ir.get("blocks", []), ocr_lines, img_path, ir.get("page", {}), cfg, models, debug)
    ir["blocks"] = _process_formula_blocks(
        ir.get("blocks", []),
        img_path=img_path,
        recognizer=getattr(models, "formula_recognizer", None),
        openai_formula_cfg=getattr(models, "openai_formula_cfg", {}) or {},
        debug=debug,
    )

    debug["fallback_ms"] = round(_now_ms() - t0, 2)
    return ir, ocr_lines


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
                "prompt": _sample_prompt(s, cfg),
                "answer": "<body></body>",
                "answer_html": "<body></body>",
                "debug": {"error": str(e)},
            })
    return out


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
            "prompt": _sample_prompt(sample, _worker_cfg or {}),
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
                    "prompt": _sample_prompt(sample, cfg),
                    "answer": "<body></body>",
                    "answer_html": "<body></body>",
                    "debug": {"error": str(e), "parallel_error": True},
                }

    for i, r in enumerate(results):
        if r is None:
            results[i] = {
                "image": samples[i].get("image"),
                "prompt": _sample_prompt(samples[i], cfg),
                "answer": "<body></body>",
                "answer_html": "<body></body>",
                "debug": {"error": "missing_result"},
            }
    
    return results


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


EVAL_MAP_CLASSES = ("title", "paragraph", "figure", "table", "formula", "chart")
EVAL_TEXT_TYPES = {"title", "paragraph", "list_item", "caption", "header", "footer", "page_number"}
EVAL_FORMULA_TYPES = {"formula"}
EVAL_TABLE_TYPES = {"table"}

_EVAL_TYPE_ALIAS = {
    "image": "figure",
    "figure": "figure",
    "fig": "figure",
    "text": "paragraph",
    "heading": "title",
    "list-item": "list_item",
    "page-number": "page_number",
    "equation": "formula",
}

_DIV_CLASS_TO_TYPE = {
    "image": "figure",
    "figure": "figure",
    "chart": "chart",
    "table": "table",
    "formula": "formula",
    "header": "header",
    "footer": "footer",
    "caption": "caption",
    "list_item": "list_item",
    "list-item": "list_item",
    "page_number": "page_number",
    "page-number": "page_number",
    "paragraph": "paragraph",
    "title": "title",
}


def _normalize_eval_type(t: str) -> str:
    t_norm = (t or "").strip().lower()
    if not t_norm:
        return "unknown"
    return _EVAL_TYPE_ALIAS.get(t_norm, t_norm)


def _normalize_space(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "")).strip()


def _parse_bbox_attr_value(raw: str) -> Optional[List[float]]:
    if not raw:
        return None
    parts = re.split(r"[,\s]+", raw.strip())
    if len(parts) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(parts[i]) for i in range(4)]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    return [x1, y1, x2, y2]


def _tag_to_eval_type(tag: str, attrs: Dict[str, str]) -> Optional[str]:
    tag = (tag or "").lower()
    if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        return "title"
    if tag == "p":
        return "paragraph"
    if tag == "li":
        return "list_item"
    if tag == "figcaption":
        return "caption"
    if tag == "header":
        return "header"
    if tag == "footer":
        return "footer"
    if tag == "table" and attrs.get("data-bbox"):
        return "table"
    if tag == "figure" and attrs.get("data-bbox"):
        return "figure"
    if tag == "div":
        class_tokens = [c.strip().lower() for c in (attrs.get("class") or "").split() if c.strip()]
        for cls in class_tokens:
            if cls in _DIV_CLASS_TO_TYPE:
                return _normalize_eval_type(_DIV_CLASS_TO_TYPE[cls])
    return None


class _EvalHTMLParser(HTMLParser):
    """Parse block-level HTML elements while preserving appearance order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks: List[Dict[str, Any]] = []
        self._stack: List[Dict[str, Any]] = []  # {tag, block_idx}
        self._table_cell_depth = 0
        self._in_ignored = False

    def _current_block_index(self) -> Optional[int]:
        for item in reversed(self._stack):
            bi = item.get("block_idx")
            if bi is not None:
                return int(bi)
        return None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag_l = (tag or "").lower()
        attrs_dict = {str(k).lower(): (v if v is not None else "") for k, v in attrs}

        if tag_l in {"script", "style"}:
            self._in_ignored = True
            self._stack.append({"tag": tag_l, "block_idx": None})
            return

        block_type = _tag_to_eval_type(tag_l, attrs_dict)
        bbox = _parse_bbox_attr_value(attrs_dict.get("data-bbox", ""))
        block_idx = None

        if block_type is not None and bbox is not None:
            score = 1.0
            try:
                score = float(attrs_dict.get("data-score", "1.0") or 1.0)
            except Exception:
                score = 1.0
            block: Dict[str, Any] = {
                "type": _normalize_eval_type(block_type),
                "bbox": bbox,
                "score": score,
                "text_parts": [],
                "table_tokens": [],
            }
            if block["type"] == "formula":
                data_latex = attrs_dict.get("data-latex", "")
                if data_latex:
                    block["text_parts"].append(data_latex)
            self.blocks.append(block)
            block_idx = len(self.blocks) - 1

        self._stack.append({"tag": tag_l, "block_idx": block_idx})

        cur_idx = self._current_block_index()
        if cur_idx is None:
            return
        cur = self.blocks[cur_idx]
        if cur.get("type") != "table":
            return

        if tag_l in {"table", "thead", "tbody", "tfoot", "tr"}:
            cur["table_tokens"].append(f"<{tag_l}>")
        elif tag_l in {"td", "th"}:
            rs = max(1, _as_int(attrs_dict.get("rowspan", "1"), 1))
            cs = max(1, _as_int(attrs_dict.get("colspan", "1"), 1))
            cur["table_tokens"].append(f"<{tag_l}:{rs}:{cs}>")
            self._table_cell_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag_l = (tag or "").lower()

        cur_idx = self._current_block_index()
        if cur_idx is not None:
            cur = self.blocks[cur_idx]
            if cur.get("type") == "table":
                if tag_l in {"table", "thead", "tbody", "tfoot", "tr"}:
                    cur["table_tokens"].append(f"</{tag_l}>")
                elif tag_l in {"td", "th"}:
                    cur["table_tokens"].append(f"</{tag_l}>")
                    self._table_cell_depth = max(0, self._table_cell_depth - 1)

        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i].get("tag") == tag_l:
                tail = self._stack[i:]
                del self._stack[i:]
                if any(item.get("tag") in {"script", "style"} for item in tail):
                    self._in_ignored = False
                return

    def handle_data(self, data: str) -> None:
        if self._in_ignored:
            return
        txt = _normalize_space(data)
        if not txt:
            return
        cur_idx = self._current_block_index()
        if cur_idx is None:
            return
        cur = self.blocks[cur_idx]
        cur["text_parts"].append(txt)
        if cur.get("type") == "table" and self._table_cell_depth > 0:
            cur["table_tokens"].append(f"T:{txt}")


def _parse_html_blocks_for_eval(html_str: str) -> List[Dict[str, Any]]:
    parser = _EvalHTMLParser()
    try:
        parser.feed(html_str or "")
        parser.close()
    except Exception:
        return []

    blocks: List[Dict[str, Any]] = []
    for idx, b in enumerate(parser.blocks):
        bbox = b.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        text = _normalize_space(" ".join(b.get("text_parts", [])))
        blocks.append({
            "index": idx,
            "type": _normalize_eval_type(str(b.get("type", "unknown"))),
            "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            "text": text,
            "score": float(b.get("score", 1.0) or 1.0),
            "table_tokens": list(b.get("table_tokens", [])),
        })
    return blocks


def _extract_html_from_record(rec: Dict[str, Any]) -> str:
    if not isinstance(rec, dict):
        return "<body></body>"
    for key in ("answer", "answer_html", "html", "gt_html", "label", "target", "suffix"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict):
            for k2 in ("html", "answer", "answer_html", "suffix"):
                v2 = val.get(k2)
                if isinstance(v2, str) and v2.strip():
                    return v2.strip()
    return "<body></body>"


def _read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _ngrams(tokens: List[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _tokenize_text(text: str) -> List[str]:
    text = _normalize_space(html.unescape(text or ""))
    if not text:
        return []
    tokens = _TEXT_TOKEN_RE.findall(text)
    return [t for t in tokens if t and not t.isspace()]


def _strip_formula_delimiters(text: str) -> str:
    txt = _normalize_space(text or "")
    if txt.startswith("$$") and txt.endswith("$$") and len(txt) >= 4:
        return txt[2:-2].strip()
    if txt.startswith("$") and txt.endswith("$") and len(txt) >= 2:
        return txt[1:-1].strip()
    return txt


def _tokenize_formula(text: str) -> List[str]:
    txt = normalize_latex(_strip_formula_delimiters(text or ""))
    if not txt:
        return []
    tokens = _FORMULA_TOKEN_RE.findall(txt)
    return [t for t in tokens if t and not t.isspace()]


def _corpus_bleu(refs: List[List[List[str]]], hyps: List[List[str]], max_order: int = 4) -> float:
    if not refs and not hyps:
        return 1.0
    if not hyps:
        return 0.0

    matches_by_order = [0.0] * max_order
    possible_by_order = [0.0] * max_order
    ref_len = 0
    hyp_len = 0

    for ref_list, hyp in zip(refs, hyps):
        hyp_len += len(hyp)
        cand_len = len(hyp)
        ref_lengths = [len(r) for r in ref_list] or [0]
        ref_len += min(ref_lengths, key=lambda rl: (abs(rl - cand_len), rl))

        for n in range(1, max_order + 1):
            hyp_ngrams = _ngrams(hyp, n)
            possible_by_order[n - 1] += max(0, len(hyp) - n + 1)

            max_ref_counts: Counter = Counter()
            for ref in ref_list:
                ref_ng = _ngrams(ref, n)
                for ng, cnt in ref_ng.items():
                    if cnt > max_ref_counts[ng]:
                        max_ref_counts[ng] = cnt

            for ng, cnt in hyp_ngrams.items():
                matches_by_order[n - 1] += min(cnt, max_ref_counts.get(ng, 0))

    precisions: List[float] = []
    effective_orders = 0
    for i in range(max_order):
        matched = matches_by_order[i]
        possible = possible_by_order[i]
        if possible <= 0:
            continue
        else:
            precisions.append((matched + 1.0) / (possible + 1.0))
            effective_orders += 1

    if hyp_len == 0:
        return 0.0 if ref_len > 0 else 1.0
    if effective_orders <= 0:
        return 1.0 if ref_len == hyp_len else 0.0

    log_prec_sum = 0.0
    for p in precisions:
        log_prec_sum += (1.0 / effective_orders) * math.log(max(p, 1e-12))
    geo_mean = math.exp(log_prec_sum)

    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - float(ref_len) / max(1, hyp_len))
    return float(_clip(geo_mean * bp, 0.0, 1.0))


def _levenshtein_distance(tokens_a: List[str], tokens_b: List[str]) -> int:
    if tokens_a == tokens_b:
        return 0
    if not tokens_a:
        return len(tokens_b)
    if not tokens_b:
        return len(tokens_a)
    if len(tokens_a) < len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a

    prev = list(range(len(tokens_b) + 1))
    for i, ta in enumerate(tokens_a, start=1):
        cur = [i] + [0] * len(tokens_b)
        for j, tb in enumerate(tokens_b, start=1):
            cost = 0 if ta == tb else 1
            cur[j] = min(
                prev[j] + 1,      # delete
                cur[j - 1] + 1,   # insert
                prev[j - 1] + cost,  # replace
            )
        prev = cur
    return int(prev[-1])


def _table_similarity(pred_block: Dict[str, Any], gt_block: Dict[str, Any]) -> float:
    pa = list(pred_block.get("table_tokens", []))
    ga = list(gt_block.get("table_tokens", []))
    if not pa:
        txt = _normalize_space(pred_block.get("text", ""))
        pa = _ALNUM_OR_PUNC_RE.findall(txt)
    if not ga:
        txt = _normalize_space(gt_block.get("text", ""))
        ga = _ALNUM_OR_PUNC_RE.findall(txt)

    if not pa and not ga:
        return 1.0
    dist = _levenshtein_distance(pa, ga)
    denom = max(len(pa), len(ga), 1)
    return float(_clip(1.0 - dist / denom, 0.0, 1.0))


def _match_by_iou(
    pred_blocks: List[Dict[str, Any]],
    gt_blocks: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    class_aware: bool = False,
) -> Tuple[List[Tuple[int, int, float]], Set[int], Set[int]]:
    if not pred_blocks or not gt_blocks:
        return [], set(), set()

    m, n = len(pred_blocks), len(gt_blocks)
    iou_mat = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        pb = pred_blocks[i]
        for j in range(n):
            gb = gt_blocks[j]
            iou = _iou(pb.get("bbox", [0, 0, 0, 0]), gb.get("bbox", [0, 0, 0, 0]))
            if class_aware and pb.get("type") != gb.get("type"):
                iou = 0.0
            iou_mat[i][j] = float(iou)

    pair_candidates: List[Tuple[int, int]] = []
    if linear_sum_assignment is not None:
        big = 1e6
        cost = [[big for _ in range(n)] for _ in range(m)]
        for i in range(m):
            for j in range(n):
                iou = iou_mat[i][j]
                if iou > 0:
                    cost[i][j] = 1.0 - iou
        row_ind, col_ind = linear_sum_assignment(cost)
        pair_candidates = [(int(i), int(j)) for i, j in zip(row_ind, col_ind)]
    else:
        scored_pairs: List[Tuple[float, int, int]] = []
        for i in range(m):
            for j in range(n):
                iou = iou_mat[i][j]
                if iou > 0:
                    scored_pairs.append((iou, i, j))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        used_i, used_j = set(), set()
        for _, i, j in scored_pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            pair_candidates.append((i, j))

    matches: List[Tuple[int, int, float]] = []
    used_pred: Set[int] = set()
    used_gt: Set[int] = set()
    for i, j in pair_candidates:
        iou = float(iou_mat[i][j])
        if iou < iou_threshold:
            continue
        if class_aware and pred_blocks[i].get("type") != gt_blocks[j].get("type"):
            continue
        matches.append((i, j, iou))
        used_pred.add(i)
        used_gt.add(j)

    return matches, used_pred, used_gt


def _average_precision(tp_flags: List[int], fp_flags: List[int], gt_count: int) -> float:
    if gt_count <= 0:
        return 0.0
    if not tp_flags:
        return 0.0

    cum_tp: List[float] = []
    cum_fp: List[float] = []
    tp_sum = 0.0
    fp_sum = 0.0
    for tp, fp in zip(tp_flags, fp_flags):
        tp_sum += float(tp)
        fp_sum += float(fp)
        cum_tp.append(tp_sum)
        cum_fp.append(fp_sum)

    rec = [tp / gt_count for tp in cum_tp]
    prec = [cum_tp[i] / max(1e-12, (cum_tp[i] + cum_fp[i])) for i in range(len(cum_tp))]

    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] > mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return float(_clip(ap, 0.0, 1.0))


def _compute_map50(samples: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Optional[float]]]:
    per_class_ap: Dict[str, Optional[float]] = {}
    valid_aps: List[float] = []

    for cls in EVAL_MAP_CLASSES:
        gt_count = 0
        pred_entries: List[Tuple[float, int, Dict[str, Any]]] = []
        gt_pool: Dict[int, Dict[str, Any]] = {}

        for sid, sample in enumerate(samples):
            gt_cls = [b for b in sample["gt_blocks"] if b.get("type") == cls]
            pred_cls = [b for b in sample["pred_blocks"] if b.get("type") == cls]
            gt_count += len(gt_cls)
            gt_pool[sid] = {"items": gt_cls, "matched": [False] * len(gt_cls)}

            for pb in pred_cls:
                score = float(pb.get("score", 1.0) or 1.0)
                pred_entries.append((score, sid, pb))

        if gt_count <= 0:
            per_class_ap[cls] = None
            continue

        pred_entries.sort(key=lambda x: x[0], reverse=True)
        tp_flags: List[int] = []
        fp_flags: List[int] = []

        for _, sid, pb in pred_entries:
            gt_info = gt_pool[sid]
            best_iou = 0.0
            best_j = -1
            for j, gb in enumerate(gt_info["items"]):
                if gt_info["matched"][j]:
                    continue
                iou = _iou(pb.get("bbox", [0, 0, 0, 0]), gb.get("bbox", [0, 0, 0, 0]))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= 0.5 and best_j >= 0:
                gt_info["matched"][best_j] = True
                tp_flags.append(1)
                fp_flags.append(0)
            else:
                tp_flags.append(0)
                fp_flags.append(1)

        ap = _average_precision(tp_flags, fp_flags, gt_count)
        per_class_ap[cls] = ap
        valid_aps.append(ap)

    map50 = float(sum(valid_aps) / len(valid_aps)) if valid_aps else 0.0
    return map50, per_class_ap


def _compute_hungarian_f1(samples: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    tp = 0.0
    fp = 0.0
    fn = 0.0

    for sample in samples:
        pred_blocks = sample["pred_blocks"]
        gt_blocks = sample["gt_blocks"]
        matches, used_pred, used_gt = _match_by_iou(pred_blocks, gt_blocks, iou_threshold=0.5, class_aware=False)

        for pi, gi, _ in matches:
            if pred_blocks[pi].get("type") == gt_blocks[gi].get("type"):
                tp += 1.0
            else:
                fp += 1.0
                fn += 1.0

        fp += float(len(pred_blocks) - len(used_pred))
        fn += float(len(gt_blocks) - len(used_gt))

    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return float(f1), {"precision": float(precision), "recall": float(recall)}


def _compute_text_bleu(samples: List[Dict[str, Any]]) -> float:
    refs: List[List[List[str]]] = []
    hyps: List[List[str]] = []
    total_pred = 0
    total_gt = 0

    for sample in samples:
        pred_blocks = [b for b in sample["pred_blocks"] if b.get("type") in EVAL_TEXT_TYPES]
        gt_blocks = [b for b in sample["gt_blocks"] if b.get("type") in EVAL_TEXT_TYPES]
        total_pred += len(pred_blocks)
        total_gt += len(gt_blocks)
        matches, _, _ = _match_by_iou(pred_blocks, gt_blocks, iou_threshold=0.5, class_aware=False)
        for pi, gi, _ in matches:
            hyp = _tokenize_text(pred_blocks[pi].get("text", ""))
            ref = _tokenize_text(gt_blocks[gi].get("text", ""))
            hyps.append(hyp)
            refs.append([ref])

    if not hyps:
        if total_pred == 0 and total_gt == 0:
            return 1.0
        return 0.0
    return _corpus_bleu(refs, hyps, max_order=4)


def _compute_formula_bleu(samples: List[Dict[str, Any]]) -> float:
    refs: List[List[List[str]]] = []
    hyps: List[List[str]] = []
    total_pred = 0
    total_gt = 0

    for sample in samples:
        pred_blocks = [b for b in sample["pred_blocks"] if b.get("type") in EVAL_FORMULA_TYPES]
        gt_blocks = [b for b in sample["gt_blocks"] if b.get("type") in EVAL_FORMULA_TYPES]
        total_pred += len(pred_blocks)
        total_gt += len(gt_blocks)
        matches, _, _ = _match_by_iou(pred_blocks, gt_blocks, iou_threshold=0.5, class_aware=True)
        for pi, gi, _ in matches:
            hyp = _tokenize_formula(pred_blocks[pi].get("text", ""))
            ref = _tokenize_formula(gt_blocks[gi].get("text", ""))
            hyps.append(hyp)
            refs.append([ref])

    if not hyps:
        if total_pred == 0 and total_gt == 0:
            return 1.0
        return 0.0
    return _corpus_bleu(refs, hyps, max_order=4)


def _compute_teds(samples: List[Dict[str, Any]]) -> float:
    total_gt = 0
    total_pred = 0
    score_sum = 0.0

    for sample in samples:
        pred_blocks = [b for b in sample["pred_blocks"] if b.get("type") in EVAL_TABLE_TYPES]
        gt_blocks = [b for b in sample["gt_blocks"] if b.get("type") in EVAL_TABLE_TYPES]
        total_pred += len(pred_blocks)
        total_gt += len(gt_blocks)

        matches, _, _ = _match_by_iou(pred_blocks, gt_blocks, iou_threshold=0.5, class_aware=True)
        matched_by_gt = {gi: pi for pi, gi, _ in matches}
        for gi in range(len(gt_blocks)):
            pi = matched_by_gt.get(gi)
            if pi is None:
                continue
            score_sum += _table_similarity(pred_blocks[pi], gt_blocks[gi])

    if total_gt == 0:
        return 1.0 if total_pred == 0 else 0.0
    return float(_clip(score_sum / total_gt, 0.0, 1.0))


def _kendall_similarity(order_values: List[int]) -> float:
    n = len(order_values)
    if n <= 1:
        return 1.0
    total_pairs = n * (n - 1) // 2
    if total_pairs <= 0:
        return 1.0
    discordant = 0
    for i in range(n):
        vi = order_values[i]
        for j in range(i + 1, n):
            if vi > order_values[j]:
                discordant += 1
    return float(_clip(1.0 - discordant / total_pairs, 0.0, 1.0))


def _compute_ktds(samples: List[Dict[str, Any]]) -> float:
    weighted_sum = 0.0
    weight_total = 0.0

    for sample in samples:
        pred_blocks = sample["pred_blocks"]
        gt_blocks = sample["gt_blocks"]
        matches, _, _ = _match_by_iou(pred_blocks, gt_blocks, iou_threshold=0.5, class_aware=False)

        if not matches:
            if not pred_blocks and not gt_blocks:
                sim = 1.0
            else:
                sim = 0.0
            weighted_sum += sim
            weight_total += 1.0
            continue

        pairs = sorted([(pi, gi) for pi, gi, _ in matches], key=lambda x: x[1])
        pred_order = [pi for pi, _ in pairs]
        pair_count = len(pred_order) * (len(pred_order) - 1) // 2
        sim = _kendall_similarity(pred_order)
        w = float(pair_count if pair_count > 0 else 1)
        weighted_sum += sim * w
        weight_total += w

    return float(_clip(weighted_sum / max(weight_total, 1e-12), 0.0, 1.0))


def evaluate_prediction_records(pred_records: List[Dict[str, Any]], gt_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    gt_by_image: Dict[str, Dict[str, Any]] = {}
    pred_by_image: Dict[str, Dict[str, Any]] = {}
    for row in gt_records:
        img = str(row.get("image", "") or "")
        if img:
            gt_by_image[img] = row
    for row in pred_records:
        img = str(row.get("image", "") or "")
        if img:
            pred_by_image[img] = row

    if not gt_by_image:
        raise ValueError("GT records are empty or missing `image` keys.")

    samples: List[Dict[str, Any]] = []
    missing_pred_images: List[str] = []
    for img, gt_row in gt_by_image.items():
        pred_row = pred_by_image.get(img)
        if pred_row is None:
            missing_pred_images.append(img)
        pred_html = _extract_html_from_record(pred_row or {})
        gt_html = _extract_html_from_record(gt_row)
        samples.append({
            "image": img,
            "pred_blocks": _parse_html_blocks_for_eval(pred_html),
            "gt_blocks": _parse_html_blocks_for_eval(gt_html),
        })

    map50, per_class_ap = _compute_map50(samples)
    f1, f1_pr = _compute_hungarian_f1(samples)
    text_bleu = _compute_text_bleu(samples)
    formula_bleu = _compute_formula_bleu(samples)
    teds = _compute_teds(samples)
    ktds = _compute_ktds(samples)

    final_score = (
        0.3 * map50
        + 0.1 * f1
        + 0.2 * text_bleu
        + 0.15 * formula_bleu
        + 0.15 * teds
        + 0.1 * ktds
    )

    extra_pred_images = [img for img in pred_by_image.keys() if img not in gt_by_image]

    return {
        "num_samples_gt": len(gt_by_image),
        "num_samples_pred": len(pred_by_image),
        "num_missing_predictions": len(missing_pred_images),
        "num_extra_predictions": len(extra_pred_images),
        "missing_prediction_images": missing_pred_images[:100],
        "extra_prediction_images": extra_pred_images[:100],
        "mAP": float(map50),
        "mAP_per_class": {k: (None if v is None else float(v)) for k, v in per_class_ap.items()},
        "F1": float(f1),
        "F1_precision": float(f1_pr["precision"]),
        "F1_recall": float(f1_pr["recall"]),
        "Text_BLEU": float(text_bleu),
        "Formula_BLEU": float(formula_bleu),
        "TEDS": float(teds),
        "KTDS": float(ktds),
        "Score": float(_clip(final_score, 0.0, 1.0)),
        "score_formula": "0.3*mAP + 0.1*F1 + 0.2*Text_BLEU + 0.15*Formula_BLEU + 0.15*TEDS + 0.1*KTDS",
    }


def evaluate_prediction_jsonl(pred_jsonl: str, gt_jsonl: str) -> Dict[str, Any]:
    pred_records = _read_jsonl_records(pred_jsonl)
    gt_records = _read_jsonl_records(gt_jsonl)
    return evaluate_prediction_records(pred_records, gt_records)


def _detect_block_columns(blocks: List[Dict[str, Any]], page: Dict[str, Any]) -> Tuple[int, Dict[int, int]]:
    """
    检测页面栏布局，支持 1-4 栏。

    优先使用基于投影直方图的稳健栏分割（utils.reading_order），
    回退到间距分割。

    Returns:
        (column_count, {block_id: column_id})
    """
    if not blocks:
        return 1, {}

    page_w = max(1.0, float(page.get("width", 1)))

    if _HAS_READING_ORDER_UTILS:
        boundaries = _detect_cols_proj(blocks, page_w)
        column_map, n_columns = _assign_block_cols(blocks, boundaries)
        for b in blocks:
            bid = b["id"]
            if b.get("type") in ("header", "footer", "page_number"):
                column_map[bid] = 0
        return n_columns, column_map

    text_blocks = [b for b in blocks
                   if b.get("type") in ("paragraph", "list_item", "caption", "title")
                   and b.get("type") not in ("header", "footer", "page_number")]

    if len(text_blocks) < 6:
        return 1, {b["id"]: 0 for b in blocks}

    centers = []
    for b in text_blocks:
        bbox = b.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        centers.append((b["id"], cx))

    if not centers:
        return 1, {b["id"]: 0 for b in blocks}

    centers.sort(key=lambda x: x[1])
    xs = [c[1] for c in centers]

    gaps = []
    for i in range(len(xs) - 1):
        gaps.append((xs[i + 1] - xs[i], i))

    if not gaps:
        return 1, {b["id"]: 0 for b in blocks}

    gap_values = [g[0] for g in gaps]
    median_gap = safe_median(gap_values, default=0.1 * page_w)

    significant_gaps = []
    for gap_val, idx in gaps:
        if gap_val > max(2.0 * median_gap, 0.1 * page_w):
            significant_gaps.append((gap_val, idx))

    significant_gaps.sort(key=lambda x: -x[0])
    significant_gaps = significant_gaps[:3]

    if not significant_gaps:
        return 1, {b["id"]: 0 for b in blocks}

    split_indices = sorted([idx for _, idx in significant_gaps])
    thresholds = [(xs[idx] + xs[idx + 1]) / 2 for idx in split_indices]

    n_columns = len(thresholds) + 1

    column_map = {}
    for b in blocks:
        if b.get("type") in ("header", "footer", "page_number"):
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
    """为 blocks 添加栏信息到 meta 字段和顶层属性（原地修改）。

    设置 meta 字段：column_id, column_count, is_first_in_column, is_last_in_column
    同时设置顶层属性：_column_id, _column_count, _is_first_in_column, _is_last_in_column
    （顶层属性供 _pair_feature_dict 等函数直接读取）
    """
    if not blocks:
        return

    n_columns, column_map = _detect_block_columns(blocks, page)

    page_w = max(1.0, float(page.get("width", 1)))
    page_h = max(1.0, float(page.get("height", 1)))
    if _HAS_READING_ORDER_UTILS:
        median_gap = _compute_median_gap(blocks, page_h)
    else:
        median_gap = page_h * 0.01

    columns: Dict[int, List[Dict[str, Any]]] = {}
    for b in blocks:
        col_id = column_map.get(b["id"], 0)
        if col_id not in columns:
            columns[col_id] = []
        columns[col_id].append(b)

    centers_y = {
        b["id"]: 0.5 * (b.get("bbox", [0, 0, 0, 0])[1] + b.get("bbox", [0, 0, 0, 0])[3])
        for b in blocks
    }
    density_window = 0.1 * page_h
    for col_id, col_blocks in columns.items():
        col_blocks.sort(key=lambda x: (x.get("bbox", [0, 0, 0, 0])[1], x.get("bbox", [0, 0, 0, 0])[0]))
        col_y_min = min(b.get("bbox", [0, 0, 0, 0])[1] for b in col_blocks)
        col_y_max = max(b.get("bbox", [0, 0, 0, 0])[3] for b in col_blocks)
        col_x_coords = [coord for block in col_blocks for coord in (block.get("bbox", [0, 0, 0, 0])[0], block.get("bbox", [0, 0, 0, 0])[2])]
        col_left = min(col_x_coords) if col_x_coords else 0.0
        col_right = max(col_x_coords) if col_x_coords else page_w
        col_height = max(1.0, col_y_max - col_y_min)

        for i, b in enumerate(col_blocks):
            if "meta" not in b or b["meta"] is None:
                b["meta"] = {}

            is_first = 1.0 if i == 0 else 0.0
            is_last = 1.0 if i == len(col_blocks) - 1 else 0.0

            b["meta"]["column_id"] = col_id
            b["meta"]["column_count"] = n_columns
            b["meta"]["is_first_in_column"] = is_first
            b["meta"]["is_last_in_column"] = is_last
            b["meta"]["_median_gap"] = float(median_gap)
            cy = centers_y.get(b["id"], 0.5 * page_h)
            bbox = b.get("bbox", [0, 0, 0, 0])
            b["meta"]["rel_y_in_column"] = (cy - col_y_min) / col_height
            b["meta"]["dist_to_left_margin"] = (bbox[0] - col_left) / page_w
            b["meta"]["dist_to_right_margin"] = (col_right - bbox[2]) / page_w
            nearby = sum(1 for other_y in centers_y.values() if abs(other_y - cy) < density_window)
            b["meta"]["vertical_density"] = nearby / max(1, len(blocks))

            b["_column_id"] = col_id
            b["_column_count"] = n_columns
            b["_is_first_in_column"] = is_first
            b["_is_last_in_column"] = is_last
            b["_median_gap"] = float(median_gap)

    page["_column_count"] = n_columns
    page["_median_gap"] = float(median_gap)



def _block_feature_dict(block: Dict[str, Any], page: Dict[str, Any], height_pct: float,
                        column_id: float = 0.0, column_count: float = 1.0,
                        is_first_in_column: float = 0.0, is_last_in_column: float = 0.0) -> Dict[str, float]:
    """
    提取 block 特征，与 train.py 的 BLOCK_SCHEMA (33维) 完全对齐
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

    text_line_count = _meta_float(block, "text_line_count", -1.0)
    if text_line_count < 0:
        text_line_count = max(1.0, float(txt.count("\n") + 1)) if txt.strip() else 0.0

    avg_line_height_px = _meta_float(block, "avg_line_height_px", -1.0)
    if avg_line_height_px > 0:
        avg_line_height_norm = avg_line_height_px / h
    elif text_line_count > 0 and bh > 0:
        avg_line_height_norm = (bh / text_line_count) / h
    else:
        avg_line_height_norm = 0.0

    feats = {
        "rel_x1": x1 / w,
        "rel_y1": y1 / h,
        "rel_x2": x2 / w,
        "rel_y2": y2 / h,
        "rel_w": bw / w,
        "rel_h": bh / h,
        "area_ratio": _area([x1, y1, x2, y2]) / (w * h),
        "aspect": (bw / bh) if bh > 0 else 0.0,
        
        "text_len": ts["len"],
        "digit_ratio": ts["digit_ratio"],
        "upper_ratio": ts["upper_ratio"],
        "lower_ratio": ts["lower_ratio"],
        "punct_ratio": ts["punct_ratio"],
        "mean_word_len": ts["mean_word_len"],
        "is_alnum": ts["is_alnum"],
        "ch_ratio": ts["ch_ratio"],
        
        "heading_level": level,
        
        "src_object": src_object,
        "src_ocr": src_ocr,
        "src_heur": src_heur,
        
        "y_top_region": y_top_region,
        "y_bottom_region": y_bottom_region,
        
        "height_percentile": height_pct,
        
        "column_id": column_id,
        "column_count": column_count,
        "is_first_in_column": is_first_in_column,
        "is_last_in_column": is_last_in_column,
        "text_line_count": text_line_count,
        "avg_line_height_norm": avg_line_height_norm,
        "rel_y_in_column": _meta_float(block, "rel_y_in_column", y1 / h),
        "dist_to_left_margin": _meta_float(block, "dist_to_left_margin", x1 / w),
        "dist_to_right_margin": _meta_float(block, "dist_to_right_margin", (w - x2) / w),
        "vertical_density": _meta_float(block, "vertical_density", 0.1),
    }
    return feats




def _pair_feature_dict(b1: Dict[str, Any], b2: Dict[str, Any], page: Dict[str, Any],
                       column_count: float = 1.0,
                       median_gap: float = None) -> Dict[str, float]:
    """
    提取 pair 特征，与 train.py 的 PAIR_SCHEMA (51维) 完全对齐
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

    u_col_id = float(b1.get("_column_id", 0))
    v_col_id = float(b2.get("_column_id", 0))
    same_column_id = 1.0 if abs(u_col_id - v_col_id) < 0.5 else 0.0
    column_diff = max(-3.0, min(3.0, v_col_id - u_col_id))

    u_text = (b1.get("text") or "")
    v_text = (b2.get("text") or "")
    u_lines = _meta_float(b1, "text_line_count", -1.0)
    if u_lines < 0:
        u_lines = max(1.0, float(u_text.count("\n") + 1)) if u_text.strip() else 1.0
    v_lines = _meta_float(b2, "text_line_count", -1.0)
    if v_lines < 0:
        v_lines = max(1.0, float(v_text.count("\n") + 1)) if v_text.strip() else 1.0
    text_line_count_ratio = (v_lines + 1) / (u_lines + 1)

    feats = {
        "dx_norm": dx / w,
        "dy_norm": dy / h,
        "center_dist_norm": dist / math.hypot(w, h),

        "x_overlap": x_overlap_ratio,
        "y_overlap": y_overlap_ratio,

        "size_ratio_w": size_ratio_w,
        "size_ratio_h": size_ratio_h,

        "same_column": same_col,
        "is_above": is_above,
        "align_diff_left_norm": align_diff_left_norm,
        "align_diff_right_norm": align_diff_right_norm,
        "same_row": same_row,
        "left_to_right": left_to_right,

        "gap_y_norm": v_gap / h,

        "u_is_title": 1.0 if b1.get("type") == "title" else 0.0,
        "v_is_title": 1.0 if b2.get("type") == "title" else 0.0,
        "u_heading_level": float(b1.get("style", {}).get("heading_level", 0) if b1.get("style") else 0.0),
        "v_heading_level": float(b2.get("style", {}).get("heading_level", 0) if b2.get("style") else 0.0),
    }

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

    feats.update({
        "same_column_id": same_column_id,
        "column_diff": column_diff,
        "u_column_id": u_col_id,
        "v_column_id": v_col_id,
        "column_count": column_count,
        "text_line_count_ratio": text_line_count_ratio,
    })

    _median_gap = float(median_gap) if median_gap is not None else float(b1.get("_median_gap", h * 0.01))
    _median_gap = max(1.0, _median_gap)
    u_type = b1.get("type", "")
    v_type = b2.get("type", "")
    feats.update({
        "vertical_gap_to_median_ratio": v_gap / _median_gap,
        "horizontal_gap_norm": abs(cx2 - cx1) / w,
        "column_distance": abs(v_col_id - u_col_id),
        "indent_diff_norm": (x21 - x11) / w,
        "width_ratio": bw2 / (bw1 + 1.0),
        "u_is_cross_column": 1.0 if bw1 > 0.7 * w else 0.0,
        "header_footer_penalty": 1.0 if (u_type in ("header", "footer", "page_number") or v_type in ("header", "footer", "page_number")) else 0.0,
    })

    va_overlap = max(0.0, min(x12, x22) - max(x11, x21))
    vertical_alignment_score = va_overlap / min(bw1, bw2) if min(bw1, bw2) > 0 else 0.0

    dy_norm = dy / h
    dx_norm = dx / w
    if dy_norm > 0 and abs(dx_norm) < 0.05:
        reading_momentum = 1.0
    elif dy_norm > 0 and abs(dx_norm) < 0.15:
        reading_momentum = 0.85
    elif dy_norm > 0 and dx_norm > 0:
        reading_momentum = 0.65
    elif abs(dy_norm) < 0.03 and dx_norm > 0:
        reading_momentum = 0.5
    elif dy_norm < -0.02:
        reading_momentum = -0.3
    else:
        reading_momentum = 0.0

    u_text_len = len((b1.get("text") or "").strip())
    v_text_len = len((b2.get("text") or "").strip())
    text_continuity = 0.0
    if u_text_len > 0 and v_text_len > 0:
        ratio = min(u_text_len, v_text_len) / max(u_text_len, v_text_len)
        text_continuity = ratio * 0.6
    if u_type == v_type and vertical_alignment_score > 0.5:
        text_continuity += 0.3
    if (v_gap / h) > 0.15:
        text_continuity *= 0.5
    context_features = max(0.0, min(1.0, text_continuity))
    
    type_transitions = {
        ("title", "paragraph"): 0.95,
        ("title", "list_item"): 0.90,
        ("title", "table"): 0.80,
        ("title", "figure"): 0.75,
        ("title", "formula"): 0.70,
        ("title", "title"): 0.40,
        ("paragraph", "paragraph"): 0.85,
        ("paragraph", "list_item"): 0.80,
        ("paragraph", "title"): 0.45,
        ("paragraph", "table"): 0.55,
        ("paragraph", "figure"): 0.55,
        ("paragraph", "formula"): 0.65,
        ("list_item", "list_item"): 0.90,
        ("list_item", "paragraph"): 0.70,
        ("list_item", "title"): 0.40,
        ("figure", "caption"): 0.92,
        ("table", "caption"): 0.92,
        ("chart", "caption"): 0.92,
        ("caption", "paragraph"): 0.70,
        ("caption", "title"): 0.50,
        ("caption", "figure"): 0.35,
        ("caption", "table"): 0.35,
        ("formula", "paragraph"): 0.70,
        ("formula", "formula"): 0.55,
        ("header", "title"): 0.60,
        ("header", "paragraph"): 0.55,
        ("paragraph", "footer"): 0.30,
        ("footer", "footer"): 0.20,
    }
    type_transition_prob = type_transitions.get((u_type, v_type), 0.40)

    adv = compute_advanced_pair_features(b1, b2, page)
    feats.update({
        "relative_angle_sin": float(adv.get("relative_angle_sin", 0.0)),
        "relative_angle_cos": float(adv.get("relative_angle_cos", 0.0)),
        "bbox_iou": float(adv.get("bbox_iou", 0.0)),
        "center_l1_norm": float(adv.get("center_l1_norm", 0.0)),
        "same_physical_column": float(adv.get("same_physical_column", 0.0)),
        "reading_flow_score": float(adv.get("reading_flow_score", 0.0)),
        "vertical_alignment_score": vertical_alignment_score,
        "reading_momentum": reading_momentum,
        "type_transition_prob": type_transition_prob,
        "context_features": context_features,
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
    
    all_cells = []
    for row in rows:
        for cell in row:
            all_cells.append(cell)
    
    if not all_cells:
        return rows
    
    x_bounds = set()
    for cell in all_cells:
        bbox = cell.get("bbox", [0, 0, 0, 0])
        x_bounds.add(round(bbox[0], 1))
        x_bounds.add(round(bbox[2], 1))
    
    x_bounds = sorted(x_bounds)
    
    col_bounds = []
    if x_bounds:
        col_bounds.append(x_bounds[0])
        for x in x_bounds[1:]:
            if x - col_bounds[-1] > 10:  # Gap threshold
                col_bounds.append(x)
    
    n_cols = max(1, len(col_bounds) - 1)
    
    row_heights = []
    for row in rows:
        if row:
            h = max(cell.get("bbox", [0, 0, 0, 0])[3] - cell.get("bbox", [0, 0, 0, 0])[1] for cell in row)
            row_heights.append(h)
        else:
            row_heights.append(20)
    
    median_row_height = safe_median(row_heights, 20)
    
    for row in rows:
        for cell in row:
            bbox = cell.get("bbox", [0, 0, 0, 0])
            cell_x1, cell_x2 = bbox[0], bbox[2]
            cell_width = cell_x2 - cell_x1
            
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
    
    for ri, row in enumerate(rows):
        for cell in row:
            bbox = cell.get("bbox", [0, 0, 0, 0])
            cell_height = bbox[3] - bbox[1]
            
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
    检测页面栏布局用于阅读顺序（支持 1-4 栏）。

    优先使用基于投影直方图的稳健栏分割，回退到间距分割。

    Returns:
        (column_count, {block_index: column_id})
    """
    if not blocks:
        return 1, {}

    page_w = max(1.0, float(page.get("width", 1)))

    if _HAS_READING_ORDER_UTILS:
        boundaries = _detect_cols_proj(blocks, page_w)
        n_columns = max(1, len(boundaries) - 1)
        column_map: Dict[int, int] = {}
        for i, b in enumerate(blocks):
            if b.get("type") == "header":
                column_map[i] = -1
                continue
            if b.get("type") in ("footer", "page_number"):
                column_map[i] = n_columns
                continue
            bbox = b.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            col_id = 0
            for thresh in boundaries[1:-1]:
                if cx > thresh:
                    col_id += 1
                else:
                    break
            column_map[i] = col_id
        return n_columns, column_map

    text_indices = [i for i, b in enumerate(blocks)
                    if b.get("type") in ("paragraph", "list_item", "caption", "title")
                    and b.get("type") not in ("header", "footer", "page_number")]

    if len(text_indices) < 6:
        return 1, {i: 0 for i in range(len(blocks))}

    centers = []
    for i in text_indices:
        bbox = blocks[i].get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2
        centers.append((i, cx))

    centers.sort(key=lambda x: x[1])
    xs = [c[1] for c in centers]

    gaps = []
    for j in range(len(xs) - 1):
        gaps.append((xs[j + 1] - xs[j], j))

    if not gaps:
        return 1, {i: 0 for i in range(len(blocks))}

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

    split_indices = sorted([idx for _, idx in significant_gaps])
    thresholds = [(xs[idx] + xs[idx + 1]) / 2 for idx in split_indices]

    n_columns = len(thresholds) + 1

    column_map = {}
    for i, b in enumerate(blocks):
        if b.get("type") in ("header", "footer", "page_number"):
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
    parser.add_argument("--gt", type=str, default=None, help="Optional GT jsonl path for local metric evaluation")
    parser.add_argument("--metrics-output", type=str, default=None, help="Optional path to save computed metrics as JSON")
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

    rows = [{"image": r["image"], "prompt": r["prompt"], "answer": r["answer"]} for r in results]
    write_submit_jsonl(args.output, rows)

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

    if args.gt:
        try:
            gt_records = _read_jsonl_records(args.gt)
            metrics = evaluate_prediction_records(rows, gt_records)
            sys.stderr.write(
                "[metrics] "
                f"mAP={metrics['mAP']:.4f} "
                f"F1={metrics['F1']:.4f} "
                f"TextBLEU={metrics['Text_BLEU']:.4f} "
                f"FormulaBLEU={metrics['Formula_BLEU']:.4f} "
                f"TEDS={metrics['TEDS']:.4f} "
                f"KTDS={metrics['KTDS']:.4f} "
                f"Score={metrics['Score']:.4f}\n"
            )
            if args.metrics_output:
                out_dir = os.path.dirname(os.path.abspath(args.metrics_output))
                _ensure_dir(out_dir)
                with open(args.metrics_output, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            sys.stderr.write(f"[warn] metric_eval_failed:{str(e)[:300]}\n")
    elif args.metrics_output:
        sys.stderr.write("[warn] --metrics-output is set but --gt is missing; metrics were skipped.\n")

    sys.stderr.write(f"[stats] total_time_s={total_time_s} samples={len(results)}\n")


if __name__ == "__main__":
    main()
