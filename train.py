#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Layout Analysis - Training Script
==========================================
训练 Block 分类器、Reading Order 关系模型、Caption 匹配模型

Schema Version: 2.0
- 新增 Block 特征：column_id, column_count, is_first/last_in_column, text_line_count, avg_line_height_norm
- 新增 Pair 特征：same_column_id, column_diff, u/v_column_id, text_line_count_ratio
- 改进负样本策略：跨栏负样本、同栏跳跃负样本、困难近邻负样本
"""

import argparse
import json
import os
import re
import random
import time
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("⚠️ LightGBM 未安装，训练功能不可用")

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    import os as _os
    import sys as _sys
    _utils_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _utils_dir not in _sys.path:
        _sys.path.insert(0, _utils_dir)
    from utils.reading_order import (
        detect_columns_by_projection,
        assign_block_columns,
        compute_page_median_gap,
    )
    _HAS_READING_ORDER_UTILS = True
except Exception:
    _HAS_READING_ORDER_UTILS = False
    detect_columns_by_projection = None  # type: ignore
    assign_block_columns = None  # type: ignore
    compute_page_median_gap = None  # type: ignore


# =============================================================================
# 常量与 Schema 定义
# =============================================================================

SCHEMA_VERSION = "2.1"  # PR4: projection-based column detection, 7 new pair features

LABEL_MAP = [
    "title", "paragraph", "list_item", "caption", "table", "figure",
    "formula", "header", "footer", "chart", "unknown"
]

# Block 特征 Schema（23 -> 29 个特征）
BLOCK_SCHEMA = [
    # 位置与尺寸（8个）
    ("rel_x1", "float32"),
    ("rel_y1", "float32"),
    ("rel_x2", "float32"),
    ("rel_y2", "float32"),
    ("rel_w", "float32"),
    ("rel_h", "float32"),
    ("area_ratio", "float32"),
    ("aspect", "float32"),
    
    # 文本统计（8个）
    ("text_len", "float32"),
    ("digit_ratio", "float32"),
    ("upper_ratio", "float32"),
    ("lower_ratio", "float32"),
    ("punct_ratio", "float32"),
    ("mean_word_len", "float32"),
    ("is_alnum", "float32"),
    ("ch_ratio", "float32"),
    
    # 样式（1个）
    ("heading_level", "float32"),
    
    # 来源（3个）
    ("src_object", "float32"),
    ("src_ocr", "float32"),
    ("src_heur", "float32"),
    
    # 位置区域（2个）
    ("y_top_region", "float32"),
    ("y_bottom_region", "float32"),
    
    # 高度百分位（1个）
    ("height_percentile", "float32"),
    
    # ===== 新增特征（6个） =====
    ("column_id", "float32"),           # 所属栏 ID（0,1,2...）
    ("column_count", "float32"),        # 页面总栏数
    ("is_first_in_column", "float32"),  # 是否为栏内第一个块
    ("is_last_in_column", "float32"),   # 是否为栏内最后一个块
    ("text_line_count", "float32"),     # 文本行数
    ("avg_line_height_norm", "float32"), # 平均行高（归一化到页高）
]

# Pair 特征 Schema（28 -> 34 个特征）
PAIR_SCHEMA = [
    # 相对位置（3个）
    ("dx_norm", "float32"),
    ("dy_norm", "float32"),
    ("center_dist_norm", "float32"),
    
    # 重叠（2个）
    ("x_overlap", "float32"),
    ("y_overlap", "float32"),
    
    # 尺寸比例（2个）
    ("size_ratio_w", "float32"),
    ("size_ratio_h", "float32"),
    
    # 布局关系（6个）
    ("same_column", "float32"),         # 几何启发式同栏
    ("is_above", "float32"),
    ("align_diff_left_norm", "float32"),
    ("align_diff_right_norm", "float32"),
    ("same_row", "float32"),
    ("left_to_right", "float32"),
    
    # 间距（1个）
    ("gap_y_norm", "float32"),
    
    # 类型特征（4个）
    ("u_is_title", "float32"),
    ("v_is_title", "float32"),
    ("u_heading_level", "float32"),
    ("v_heading_level", "float32"),
    
    # 类型 one-hot（10个）
    ("u_text", "float32"),
    ("u_table", "float32"),
    ("u_figure", "float32"),
    ("u_caption", "float32"),
    ("u_other", "float32"),
    ("v_text", "float32"),
    ("v_table", "float32"),
    ("v_figure", "float32"),
    ("v_caption", "float32"),
    ("v_other", "float32"),
    
    # ===== 新增特征（6个） =====
    ("same_column_id", "float32"),      # 基于 column_id 的同栏判断
    ("column_diff", "float32"),         # v_col - u_col
    ("u_column_id", "float32"),         # u 的栏 ID
    ("v_column_id", "float32"),         # v 的栏 ID
    ("column_count", "float32"),        # 页面栏数
    ("text_line_count_ratio", "float32"), # v_lines / u_lines

    # ===== PR4 新增特征（7个） =====
    ("vertical_gap_to_median_ratio", "float32"),  # (v.y1 - u.y2) / median_gap
    ("horizontal_gap_norm", "float32"),           # abs(v.cx - u.cx) / page_width
    ("column_distance", "float32"),               # abs(u.col - v.col)
    ("indent_diff_norm", "float32"),              # (v.x1 - u.x1) / page_width
    ("width_ratio", "float32"),                   # v.w / (u.w + 1)
    ("u_is_cross_column", "float32"),             # u 是否跨栏（宽 > 0.7 * page_w）
    ("header_footer_penalty", "float32"),         # 是否 header/footer
]

# 特征维度常量
BLOCK_FEAT_DIM = len(BLOCK_SCHEMA)  # 29
PAIR_FEAT_DIM = len(PAIR_SCHEMA)    # 41


# =============================================================================
# 数据统计容器
# =============================================================================

@dataclass
class SkipStats:
    """跳过样本的统计"""
    json_parse_error: int = 0
    no_page: int = 0
    no_blocks: int = 0
    invalid_page_size: int = 0
    missing_width_height: int = 0
    block_feat_error: int = 0
    pair_feat_error: int = 0
    other_error: int = 0
    
    def total(self) -> int:
        return (self.json_parse_error + self.no_page + self.no_blocks +
                self.invalid_page_size + self.missing_width_height +
                self.block_feat_error + self.pair_feat_error + self.other_error)
    
    def summary(self) -> Dict[str, int]:
        return {
            "json_parse_error": self.json_parse_error,
            "no_page": self.no_page,
            "no_blocks": self.no_blocks,
            "invalid_page_size": self.invalid_page_size,
            "missing_width_height": self.missing_width_height,
            "block_feat_error": self.block_feat_error,
            "pair_feat_error": self.pair_feat_error,
            "other_error": self.other_error,
            "total": self.total()
        }


@dataclass
class TrainingMetrics:
    """训练指标"""
    best_iteration: int = 0
    best_score: float = 0.0
    train_samples: int = 0
    val_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importance: List[Tuple[int, float]] = field(default_factory=list)


# =============================================================================
# 辅助函数
# =============================================================================

def is_chinese(c: str) -> bool:
    return '\u4e00' <= c <= '\u9fff'


def re_split(t: str) -> List[str]:
    return [x for x in re.split(r"\s+", t) if x]


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def coarse_type_onehot(t: str) -> List[float]:
    """
    将 block type 转换为粗粒度 one-hot，与 eval.py _coarse_type_onehot 对齐
    
    顺序: [text, table, figure, caption, other]
    """
    t = (t or "").strip().lower()
    if t == "text":
        t = "paragraph"
    
    if t in ("paragraph", "title", "list_item", "header", "footer", "formula", "unknown"):
        return [1.0, 0.0, 0.0, 0.0, 0.0]
    if t == "table":
        return [0.0, 1.0, 0.0, 0.0, 0.0]
    if t in ("figure", "chart"):
        return [0.0, 0.0, 1.0, 0.0, 0.0]
    if t == "caption":
        return [0.0, 0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 0.0, 1.0]


def overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0, min(a2, b2) - max(a1, b1))


def get_meta_field(block: Dict, field: str, default: float = 0.0) -> float:
    """
    安全获取 block["meta"][field]，缺失返回 default
    支持从 block 直接获取（如 block["_column_id"]）
    """
    # 先尝试从 meta 获取
    meta = block.get("meta")
    if meta is not None:
        val = meta.get(field)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    
    # 再尝试从 block 直接获取（带下划线前缀）
    direct_key = f"_{field}" if not field.startswith("_") else field
    val = block.get(direct_key)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    
    # 最后尝试不带下划线
    val = block.get(field)
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    
    return default


def save_schema(schema: List[Tuple[str, str]], path: str, version: str = SCHEMA_VERSION):
    """保存特征 schema 到 JSON"""
    feats = []
    for idx, (name, dtype) in enumerate(schema):
        feats.append({
            "name": name,
            "dtype": dtype,
            "shape": [1],
            "index": idx
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": version,
            "features": feats
        }, f, indent=2, ensure_ascii=False)


# =============================================================================
# 特征提取函数
# =============================================================================

def extract_block_feats(
    b: Dict[str, Any],
    page: Dict[str, Any],
    height_percentile: float = 0.0,
    column_id: float = None,
    column_count: float = None,
    is_first_in_column: float = None,
    is_last_in_column: float = None
) -> List[float]:
    """
    提取单个 block 的特征向量，与 eval.py 完全对齐
    
    Args:
        b: block 字典
        page: page 字典（含 width, height）
        height_percentile: 该 block 在页面中的高度百分位
        column_id: 栏 ID（��选，优先于 meta）
        column_count: 总栏数（可选）
        is_first_in_column: 是否栏首（可选）
        is_last_in_column: 是否栏尾（可选）
    
    Returns:
        特征向量（长度 = BLOCK_FEAT_DIM）
    
    Raises:
        ValueError: bbox 格式无效
    """
    bbox = b.get("bbox")
    if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        raise ValueError(f"Invalid bbox format: {bbox}")
    
    x1, y1, x2, y2 = bbox[:4]
    bw, bh = x2 - x1, y2 - y1
    
    page_w = max(1, page.get("width", 1))
    page_h = max(1, page.get("height", 1))
    
    # 位置与尺寸特征
    area_ratio = (bw * bh) / (page_w * page_h)
    aspect = safe_div(bw, bh, default=0.0)
    
    # 文本统计（与 eval.py _text_stats 对齐）
    txt = b.get("text", "") or ""
    n = len(txt)
    
    if n > 0:
        digit_ratio = sum(c.isdigit() for c in txt) / n
        upper_ratio = sum(c.isupper() for c in txt) / n
        lower_ratio = sum(c.islower() for c in txt) / n
        punct_ratio = sum(c in ".,;:!?\"'()[]{}，。；：！？、（）【】《》" for c in txt) / n
        is_alnum = sum(c.isalnum() for c in txt) / n
        ch_ratio = sum(0x4E00 <= ord(c) <= 0x9FA5 for c in txt) / n
    else:
        digit_ratio = upper_ratio = lower_ratio = punct_ratio = is_alnum = ch_ratio = 0.0
    
    words = [w for w in txt.split() if w]
    mean_word_len = sum(len(w) for w in words) / len(words) if words else 0.0
    
    # 来源
    src = (b.get("source") or "object").lower()
    src_object = 1.0 if src == "object" else 0.0
    src_ocr = 1.0 if "ocr" in src else 0.0
    src_heur = 1.0 if "heur" in src else 0.0
    
    # 样式
    heading_level = 0.0
    style = b.get("style")
    if style and style.get("heading_level"):
        try:
            heading_level = float(style["heading_level"])
        except (ValueError, TypeError):
            heading_level = 0.0
    
    # 位置区域
    y_top_region = 1.0 if y1 < 0.1 * page_h else 0.0
    y_bottom_region = 1.0 if y2 > 0.9 * page_h else 0.0
    
    # ===== 栏信息：优先使用参数，否则从 meta 读取 =====
    _column_id = column_id if column_id is not None else get_meta_field(b, "column_id", 0.0)
    _column_count = column_count if column_count is not None else get_meta_field(b, "column_count", 1.0)
    _is_first = is_first_in_column if is_first_in_column is not None else get_meta_field(b, "is_first_in_column", 0.0)
    _is_last = is_last_in_column if is_last_in_column is not None else get_meta_field(b, "is_last_in_column", 0.0)
    
    # 文本行数：优先从 meta 读取，否则启发式估计
    text_line_count = get_meta_field(b, "text_line_count", -1.0)
    if text_line_count < 0:
        text_line_count = get_meta_field(b, "block_text_line_count", -1.0)
    if text_line_count < 0:
        # 启发式：按换行符计数
        text_line_count = max(1.0, float(txt.count("\n") + 1)) if txt.strip() else 0.0
    
    # 平均行高（归一化）
    avg_line_height_px = get_meta_field(b, "avg_line_height_px", -1.0)
    if avg_line_height_px < 0:
        avg_line_height_px = get_meta_field(b, "avg_line_height", -1.0)
    
    if avg_line_height_px > 0:
        avg_line_height_norm = avg_line_height_px / page_h
    else:
        # 启发式：用 bbox 高度 / 行数
        if text_line_count > 0 and bh > 0:
            avg_line_height_norm = (bh / text_line_count) / page_h
        else:
            avg_line_height_norm = 0.0
    
    # 构建特征向量（顺序必须与 BLOCK_SCHEMA 一致）
    feats = [
        # 位置与尺寸（8个）
        x1 / page_w,
        y1 / page_h,
        x2 / page_w,
        y2 / page_h,
        bw / page_w,
        bh / page_h,
        area_ratio,
        aspect,
        
        # 文本统计（8个）
        float(n),
        digit_ratio,
        upper_ratio,
        lower_ratio,
        punct_ratio,
        mean_word_len,
        is_alnum,
        ch_ratio,
        
        # 样式（1个）
        heading_level,
        
        # 来源（3个）
        src_object,
        src_ocr,
        src_heur,
        
        # 位置区域（2个）
        y_top_region,
        y_bottom_region,
        
        # 高度百分位（1个）
        height_percentile,
        
        # 新增特征（6个）
        float(_column_id),
        float(_column_count),
        float(_is_first),
        float(_is_last),
        float(text_line_count),
        float(avg_line_height_norm),
    ]
    
    assert len(feats) == BLOCK_FEAT_DIM, f"Block 特征维度错误: {len(feats)} != {BLOCK_FEAT_DIM}"
    return feats


def extract_pair_feats(
    u: Dict[str, Any],
    v: Dict[str, Any],
    page: Dict[str, Any],
    column_count: float = None,
    median_gap: float = None,
) -> List[float]:
    """
    提取 block pair (u, v) 的特征向量，与 eval.py 完全对齐
    
    Args:
        u: 源 block
        v: 目标 block
        page: page 字典
        column_count: 总栏数（可选）
        median_gap: 页面中位数垂直间距（可选，用于 vertical_gap_to_median_ratio）
    
    Returns:
        特征向量（长度 = PAIR_FEAT_DIM）
    """
    import math
    
    # 处理空输入
    if u is None or v is None:
        return [0.0] * PAIR_FEAT_DIM
    
    u_bbox = u.get("bbox")
    v_bbox = v.get("bbox")
    if not u_bbox or not v_bbox or len(u_bbox) < 4 or len(v_bbox) < 4:
        return [0.0] * PAIR_FEAT_DIM
    
    ux1, uy1, ux2, uy2 = u_bbox[:4]
    vx1, vy1, vx2, vy2 = v_bbox[:4]
    
    # 中心点
    uc_x, uc_y = (ux1 + ux2) / 2, (uy1 + uy2) / 2
    vc_x, vc_y = (vx1 + vx2) / 2, (vy1 + vy2) / 2
    
    page_w = max(1, page.get("width", 1))
    page_h = max(1, page.get("height", 1))
    
    # 相对位置（与 eval.py 对齐）
    dx = vc_x - uc_x
    dy = vc_y - uc_y
    dist = math.hypot(dx, dy)
    dx_norm = dx / page_w
    dy_norm = dy / page_h
    center_dist_norm = dist / math.hypot(page_w, page_h)
    
    # 重叠
    uw, uh = max(1.0, ux2 - ux1), max(1.0, uy2 - uy1)
    vw, vh = max(1.0, vx2 - vx1), max(1.0, vy2 - vy1)
    ovx = overlap_1d(ux1, ux2, vx1, vx2)
    ovy = overlap_1d(uy1, uy2, vy1, vy2)
    x_overlap = ovx / min(uw, vw)
    y_overlap = ovy / min(uh, vh)
    
    # 尺寸比例
    size_ratio_w = vw / uw
    size_ratio_h = vh / uh
    
    # 布局关系（几何启发式）
    same_col = 1.0 if abs(ux1 - vx1) < 0.08 * page_w else 0.0
    is_above = 1.0 if uy2 <= vy1 else 0.0
    align_diff_left_norm = abs(ux1 - vx1) / page_w
    align_diff_right_norm = abs(ux2 - vx2) / page_w
    same_row = 1.0 if abs(uc_y - vc_y) < 0.04 * page_h else 0.0
    left_to_right = 1.0 if (same_row > 0.5 and ux1 < vx1) else 0.0
    
    # 间距
    gap_y = max(0.0, vy1 - uy2)
    gap_y_norm = gap_y / page_h
    
    # 类型特征
    u_type = u.get("type", "unknown")
    v_type = v.get("type", "unknown")
    u_is_title = 1.0 if u_type == "title" else 0.0
    v_is_title = 1.0 if v_type == "title" else 0.0
    
    u_hl = 0.0
    if u.get("style") and u["style"].get("heading_level"):
        try:
            u_hl = float(u["style"]["heading_level"])
        except (ValueError, TypeError):
            u_hl = 0.0
    
    v_hl = 0.0
    if v.get("style") and v["style"].get("heading_level"):
        try:
            v_hl = float(v["style"]["heading_level"])
        except (ValueError, TypeError):
            v_hl = 0.0
    
    # 类型 one-hot
    u_oh = coarse_type_onehot(u_type)
    v_oh = coarse_type_onehot(v_type)
    
    # ===== 新增特征（与 eval.py 对齐）=====
    u_col_id = get_meta_field(u, "column_id", 0.0)
    v_col_id = get_meta_field(v, "column_id", 0.0)
    same_column_id = 1.0 if abs(u_col_id - v_col_id) < 0.5 else 0.0
    column_diff = max(-3.0, min(3.0, v_col_id - u_col_id))
    
    _column_count = column_count if column_count is not None else get_meta_field(u, "column_count", 1.0)
    
    # 文本行数（多种来源兼容）
    u_lines = get_meta_field(u, "text_line_count", -1.0)
    if u_lines < 0:
        u_lines = get_meta_field(u, "block_text_line_count", 1.0)
    if u_lines < 0:
        u_text = (u.get("text") or "")
        u_lines = max(1.0, float(u_text.count("\n") + 1)) if u_text.strip() else 1.0
    
    v_lines = get_meta_field(v, "text_line_count", -1.0)
    if v_lines < 0:
        v_lines = get_meta_field(v, "block_text_line_count", 1.0)
    if v_lines < 0:
        v_text = (v.get("text") or "")
        v_lines = max(1.0, float(v_text.count("\n") + 1)) if v_text.strip() else 1.0
    
    text_line_count_ratio = (v_lines + 1) / (u_lines + 1)

    # ===== PR4 新增特征（7个）=====
    _median_gap = float(median_gap) if median_gap is not None else get_meta_field(u, "_median_gap", page_h * 0.01)
    _median_gap = max(1.0, _median_gap)
    vertical_gap_to_median_ratio = gap_y / _median_gap

    horizontal_gap_norm = abs(vc_x - uc_x) / page_w

    column_distance = abs(v_col_id - u_col_id)

    indent_diff_norm = (vx1 - ux1) / page_w

    width_ratio = vw / (uw + 1.0)

    u_is_cross_column = 1.0 if (uw > 0.7 * page_w) else 0.0

    u_hf = u_type in ("header", "footer")
    v_hf = v_type in ("header", "footer")
    header_footer_penalty = 1.0 if (u_hf or v_hf) else 0.0

    # 构建特征向量（顺序必须与 PAIR_SCHEMA 一致）
    feats = [
        # 相对位置（3个）
        dx_norm,
        dy_norm,
        center_dist_norm,
        
        # 重叠（2个）
        x_overlap,
        y_overlap,
        
        # 尺寸比例（2个）
        size_ratio_w,
        size_ratio_h,
        
        # 布局关系（6个）
        same_col,
        is_above,
        align_diff_left_norm,
        align_diff_right_norm,
        same_row,
        left_to_right,
        
        # 间距（1个）
        gap_y_norm,
        
        # 类型特征（4个）
        u_is_title,
        v_is_title,
        u_hl,
        v_hl,
        
        # 类型 one-hot（10个）
        u_oh[0], u_oh[1], u_oh[2], u_oh[3], u_oh[4],
        v_oh[0], v_oh[1], v_oh[2], v_oh[3], v_oh[4],
        
        # 新增特征（6个）
        same_column_id,
        column_diff,
        u_col_id,
        v_col_id,
        float(_column_count),
        text_line_count_ratio,

        # PR4 新增特征（7个）
        vertical_gap_to_median_ratio,
        horizontal_gap_norm,
        column_distance,
        indent_diff_norm,
        width_ratio,
        u_is_cross_column,
        header_footer_penalty,
    ]
    
    assert len(feats) == PAIR_FEAT_DIM, f"Pair 特征维度错误: {len(feats)} != {PAIR_FEAT_DIM}"
    return feats

# =============================================================================
# 辅助数据结构
# =============================================================================

def build_id_map(blocks: List[Dict]) -> Dict[Any, Dict]:
    """构建 block id -> block 的映射"""
    return {b["id"]: b for b in blocks if b is not None and "id" in b}


def rebuild_sequence_from_edges(order_edges: List[Dict]) -> List[Any]:
    """从 order_edges 重建阅读顺序序列"""
    if not order_edges:
        return []

    nxt = {e["u"]: e["v"] for e in order_edges if "u" in e and "v" in e}
    if not nxt:
        return []

    heads = set(nxt.keys()) - set(nxt.values())
    if not heads:
        # 可能有环，取任意起点
        heads = {list(nxt.keys())[0]}

    head = list(heads)[0]
    seq, seen, cur = [], set(), head

    while cur is not None and cur not in seen:
        seq.append(cur)
        seen.add(cur)
        cur = nxt.get(cur)

    # 添加最后一个节点
    if cur is not None and cur not in seen:
        seq.append(cur)

    return seq


def compute_height_percentiles(blocks: List[Dict]) -> Dict[Any, float]:
    """计算每个 block 的高度百分位"""
    if not blocks:
        return {}

    heights = []
    for b in blocks:
        bbox = b.get("bbox")
        if bbox and len(bbox) >= 4:
            heights.append(bbox[3] - bbox[1])
        else:
            heights.append(0)

    hs = np.array(heights)
    if len(hs) == 0 or hs.max() == hs.min():
        return {b["id"]: 0.5 for b in blocks if "id" in b}

    ranks = np.argsort(np.argsort(hs))
    percentiles = ranks / max(len(hs) - 1, 1)

    return {b["id"]: float(percentiles[i]) for i, b in enumerate(blocks) if "id" in b}


def detect_columns(blocks: List[Dict], page: Dict) -> Dict[Any, int]:
    """
    检测页面栏布局，为每个 block 分配 column_id。

    优先使用基于投影直方图的稳健栏分割（来自 utils.reading_order），
    回退到简单间距分割。

    返回 {block_id: column_id}
    """
    if not blocks:
        return {}

    page_w = max(1, page.get("width", 1))

    if _HAS_READING_ORDER_UTILS:
        boundaries = detect_columns_by_projection(blocks, page_w)
        column_map, _ = assign_block_columns(blocks, boundaries)
        return column_map

    # 回退：简单间距分割
    centers = []
    for b in blocks:
        bbox = b.get("bbox")
        if bbox and len(bbox) >= 4:
            cx = (bbox[0] + bbox[2]) / 2
            centers.append((b["id"], cx))

    if not centers:
        return {}

    centers.sort(key=lambda x: x[1])
    xs = [c[1] for c in centers]

    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    if not gaps:
        return {bid: 0 for bid, _ in centers}

    max_gap, max_idx = max(gaps, key=lambda x: x[0])
    median_gap = np.median([g[0] for g in gaps]) if gaps else max_gap

    column_map = {}
    if max_gap > 1.5 * median_gap and max_gap > 0.15 * page_w:
        threshold = (xs[max_idx] + xs[max_idx + 1]) / 2
        for bid, cx in centers:
            column_map[bid] = 0 if cx <= threshold else 1
    else:
        for bid, _ in centers:
            column_map[bid] = 0

    return column_map


def enrich_blocks_with_column_info(blocks: List[Dict], page: Dict) -> None:
    """
    为 blocks 添加栏信息到 meta 字段（原地修改）

    添加：column_id, column_count, is_first_in_column, is_last_in_column
    同时添加文本行数估计和中位数垂直间距
    """
    if not blocks:
        return

    column_map = detect_columns(blocks, page)
    column_count = len(set(column_map.values())) if column_map else 1
    page_h = max(1, page.get("height", 1))

    # 计算页面中位数垂直间距（用于 PR4 新特征）
    if _HAS_READING_ORDER_UTILS:
        median_gap = compute_page_median_gap(blocks, page_h)
    else:
        median_gap = page_h * 0.01

    # 按栏分组，并在栏内按 y 排序
    columns: Dict[int, List[Dict]] = defaultdict(list)
    for b in blocks:
        bid = b.get("id")
        col_id = column_map.get(bid, 0)
        columns[col_id].append(b)

    # 每栏内按 y 排序，确定 first/last
    for col_id, col_blocks in columns.items():
        col_blocks.sort(key=lambda x: (x.get("bbox", [0, 0, 0, 0])[1], x.get("bbox", [0, 0, 0, 0])[0]))

        for i, b in enumerate(col_blocks):
            if "meta" not in b or b["meta"] is None:
                b["meta"] = {}

            b["meta"]["column_id"] = float(col_id)
            b["meta"]["column_count"] = float(column_count)
            b["meta"]["is_first_in_column"] = 1.0 if i == 0 else 0.0
            b["meta"]["is_last_in_column"] = 1.0 if i == len(col_blocks) - 1 else 0.0
            b["meta"]["_median_gap"] = float(median_gap)
            
            # 估计文本行数（如果尚未设置）
            if "text_line_count" not in b["meta"] and "block_text_line_count" not in b["meta"]:
                txt = b.get("text", "") or ""
                bbox = b.get("bbox", [0, 0, 0, 0])
                bh = bbox[3] - bbox[1]
                
                if txt.strip():
                    line_count = max(1, txt.count("\n") + 1)
                else:
                    line_count = 1
                
                b["meta"]["text_line_count"] = float(line_count)
                
                # 估计平均行高
                if line_count > 0 and bh > 0:
                    b["meta"]["avg_line_height_px"] = bh / line_count


# =============================================================================
# 负样本生成策略
# =============================================================================

@dataclass
class NegativeSample:
    """负样本"""
    u_id: Any
    v_id: Any
    weight: float
    neg_type: str  # hard_near, same_column_skip, cross_column, reverse, random


def knn_negatives(
    blocks: List[Dict],
    positives: Set[Tuple],
    k: int = 6,
    max_neg_ratio: int = 4
) -> List[NegativeSample]:
    """
    生成 KNN 困难负样本

    分类：
    - hard_near: 物理距离近但不相邻（权重高）
    - random: 随机远距离负样本（权重低）
    """
    negs = []

    if len(blocks) < 2:
        return negs

    max_negs = max_neg_ratio * max(len(positives), 1)

    # 构建 KDTree
    centers = []
    ids = []
    for b in blocks:
        bbox = b.get("bbox")
        if bbox and len(bbox) >= 4:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append([cx, cy])
            ids.append(b["id"])

    if len(centers) < 2:
        return negs

    centers_arr = np.array(centers)

    if cKDTree is not None:
        tree = cKDTree(centers_arr)

        for i, b in enumerate(blocks):
            if b.get("id") not in ids:
                continue

            idx = ids.index(b["id"])
            query_k = min(k * 2 + 1, len(centers))
            dists, idxs = tree.query(centers_arr[idx], k=query_k)

            for rank, (j, dist) in enumerate(zip(idxs[1:], dists[1:])):
                if j >= len(ids):
                    continue

                u_id, v_id = ids[idx], ids[j]

                # 跳过正样本
                if (u_id, v_id) in positives or (v_id, u_id) in positives:
                    continue

                # 区分困难程度
                if rank < k // 2:
                    # 最近邻：困难负样本
                    negs.append(NegativeSample(u_id, v_id, 2.0, "hard_near"))
                else:
                    negs.append(NegativeSample(u_id, v_id, 1.0, "random"))

                if len(negs) >= max_negs:
                    return negs
    else:
        # 降级：暴力搜索
        for i, b1 in enumerate(blocks):
            for j, b2 in enumerate(blocks):
                if i >= j:
                    continue

                u_id, v_id = b1.get("id"), b2.get("id")
                if u_id is None or v_id is None:
                    continue

                if (u_id, v_id) not in positives and (v_id, u_id) not in positives:
                    negs.append(NegativeSample(u_id, v_id, 1.0, "random"))

                    if len(negs) >= max_negs:
                        return negs

    return negs


def same_column_skip_negatives(
    blocks: List[Dict],
    positives: Set[Tuple],
    id2b: Dict,
    max_samples: int = 100
) -> List[NegativeSample]:
    """
    生成同栏跳跃负样本：同栏内但中间有其他块的 (u, v)

    这类负样本帮助模型学习"不要跳读"
    """
    negs = []

    # 按栏分组
    columns: Dict[int, List[Dict]] = defaultdict(list)
    for b in blocks:
        col_id = int(get_meta_field(b, "column_id", 0))
        columns[col_id].append(b)

    for col_id, col_blocks in columns.items():
        # 按 y 排序
        col_blocks.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])

        # 生成跳跃负样本（隔 2-4 个块）
        for i in range(len(col_blocks)):
            for step in [2, 3, 4]:
                j = i + step
                if j >= len(col_blocks):
                    break

                u_id = col_blocks[i].get("id")
                v_id = col_blocks[j].get("id")

                if u_id is None or v_id is None:
                    continue

                # 确保不是正样本
                if (u_id, v_id) in positives:
                    continue

                # 权重随跳跃步数递减
                weight = 1.8 - 0.2 * step
                negs.append(NegativeSample(u_id, v_id, weight, "same_column_skip"))

                if len(negs) >= max_samples:
                    return negs

    return negs


def cross_column_negatives(
    blocks: List[Dict],
    positives: Set[Tuple],
    page: Dict,
    max_samples: int = 50
) -> List[NegativeSample]:
    """
    生成跨栏负样本：不同栏的块对

    注意：不是所有跨栏都是负样本，只选择"明显不合理"的跨栏
    - 左栏中部 -> 右栏中部（应该不跨栏）
    - 右栏 -> 左栏（逆向跨栏）
    """
    negs = []
    page_h = max(1, page.get("height", 1))

    # 按栏分组
    columns: Dict[int, List[Dict]] = defaultdict(list)
    for b in blocks:
        col_id = int(get_meta_field(b, "column_id", 0))
        columns[col_id].append(b)

    if len(columns) < 2:
        return negs

    col_ids = sorted(columns.keys())

    for i, col_i in enumerate(col_ids):
        for j, col_j in enumerate(col_ids):
            if i == j:
                continue

            for u in columns[col_i]:
                u_id = u.get("id")
                u_bbox = u.get("bbox", [0, 0, 0, 0])
                u_y_ratio = (u_bbox[1] + u_bbox[3]) / 2 / page_h

                for v in columns[col_j]:
                    v_id = v.get("id")
                    v_bbox = v.get("bbox", [0, 0, 0, 0])
                    v_y_ratio = (v_bbox[1] + v_bbox[3]) / 2 / page_h

                    if u_id is None or v_id is None:
                        continue

                    if (u_id, v_id) in positives:
                        continue

                    # 逆向跨栏（右->左）：高权重负样本
                    if col_i > col_j:
                        negs.append(NegativeSample(u_id, v_id, 2.5, "cross_column"))

                    # 正向跨栏但 y 位置不合理（不是"末尾->顶部"）
                    elif not (u_y_ratio > 0.7 and v_y_ratio < 0.3):
                        negs.append(NegativeSample(u_id, v_id, 1.5, "cross_column"))

                    if len(negs) >= max_samples:
                        return negs

    return negs


def reverse_negatives(
    positives: Set[Tuple],
    id2b: Dict,
    weight: float = 2.0
) -> List[NegativeSample]:
    """
    生成反向负样本：对每个正样本 (u, v)，添加 (v, u) 作为负样本
    """
    negs = []

    for (u_id, v_id) in positives:
        # 确保反向边不是正样本
        if (v_id, u_id) not in positives:
            negs.append(NegativeSample(v_id, u_id, weight, "reverse"))

    return negs


def weak_positive_samples(
    blocks: List[Dict],
    positives: Set[Tuple],
    id2b: Dict,
    weight: float = 0.5
) -> List[Tuple[Any, Any, float]]:
    """
    生成弱正样本：同栏内相邻的正文块（类型为 paragraph/list_item）

    返回 [(u_id, v_id, weight), ...]
    """
    weak_pos = []

    # 只对正文类型生成弱正样本
    text_types = {"paragraph", "list_item"}

    # 按栏分组
    columns: Dict[int, List[Dict]] = defaultdict(list)
    for b in blocks:
        if b.get("type") in text_types:
            col_id = int(get_meta_field(b, "column_id", 0))
            columns[col_id].append(b)

    for col_id, col_blocks in columns.items():
        # 按 y 排序
        col_blocks.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])

        for i in range(len(col_blocks) - 1):
            u_id = col_blocks[i].get("id")
            v_id = col_blocks[i + 1].get("id")

            if u_id is None or v_id is None:
                continue

            # 如果已经是正样本，跳过
            if (u_id, v_id) in positives:
                continue

            weak_pos.append((u_id, v_id, weight))

    return weak_pos


# =============================================================================
# Caption 负样本
# =============================================================================

def caption_neg_cost(cap: Dict, tar: Dict, page: Dict) -> float:
    """计算 caption-target 的负样本代价（越小越困难）"""
    cx1, cy1, cx2, cy2 = cap.get("bbox", [0, 0, 0, 0])
    tx1, ty1, tx2, ty2 = tar.get("bbox", [0, 0, 0, 0])

    page_w = max(1, page.get("width", 1))
    page_h = max(1, page.get("height", 1))

    c_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
    t_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)

    dy = (c_center[1] - t_center[1]) / page_h
    dx = abs(c_center[0] - t_center[0]) / page_w

    ovx = overlap_1d(cx1, cx2, tx1, tx2) / max(1, min(cx2 - cx1, tx2 - tx1))

    return abs(dy) * 2 + dx - ovx * 2


def caption_negatives(
    caps: List[Dict],
    tars: List[Dict],
    pos_cap: Set[Tuple],
    page: Dict,
    k: int = 5
) -> List[Tuple[Dict, Dict]]:
    """生成 caption 负样本"""
    negs = []

    for cap in caps:
        # 按距离排序，取最近的 k 个
        cand = sorted(tars, key=lambda t: caption_neg_cost(cap, t, page))[:k]

        for tar in cand:
            cap_id = cap.get("id")
            tar_id = tar.get("id")

            if cap_id is None or tar_id is None:
                continue

            if (cap_id, tar_id) in pos_cap:
                continue

            negs.append((cap, tar))

    return negs


# =============================================================================
# 数据加载
# =============================================================================

def load_data(
    path: str,
    max_samples: Optional[int] = None,
    jump_weight: float = 0.3,
    rev_neg_weight: float = 2.0,
    weak_pos_weight: float = 0.5,
    cross_col_neg_weight: float = 1.5,
    enrich_column_info: bool = True
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           SkipStats]:
    """
    加载训练数据并提取特征

    Args:
        path: JSONL 文件路径
        max_samples: 最大样本数
        jump_weight: 跳步正样本权重
        rev_neg_weight: 反向负样本权重
        weak_pos_weight: 弱正样本权重
        cross_col_neg_weight: 跨栏负样本权重
        enrich_column_info: 是否自动补充栏信息

    Returns:
        (Xb, yb, Xo, yo, wo, Xc, yc, wc, skip_stats)
    """
    Xb, yb = [], []
    Xo, yo, wo = [], [], []
    Xc, yc, wc = [], [], []

    cnt = 0
    skip_stats = SkipStats()

    if not os.path.exists(path):
        print(f"❌ 错误: 找不到文件 {path}")
        return (np.zeros((0, BLOCK_FEAT_DIM)), np.array([]),
                np.zeros((0, PAIR_FEAT_DIM)), np.array([]), np.array([]),
                np.zeros((0, PAIR_FEAT_DIM)), np.array([]), np.array([]),
                skip_stats)

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # 解析 JSON
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                skip_stats.json_parse_error += 1
                if skip_stats.json_parse_error <= 5:
                    print(f"⚠️ 第 {line_num} 行 JSON 解析失败: {e}")
                continue

            try:
                ir = obj.get("ir", {})
                blocks = ir.get("blocks", [])
                page = ir.get("page", {})
                rel = ir.get("relations", {})

                # 数据验证
                if not page or not isinstance(page, dict):
                    skip_stats.no_page += 1
                    continue

                if not blocks or len(blocks) == 0:
                    skip_stats.no_blocks += 1
                    continue

                if "width" not in page or "height" not in page:
                    skip_stats.missing_width_height += 1
                    continue

                if page["width"] <= 0 or page["height"] <= 0:
                    skip_stats.invalid_page_size += 1
                    continue

                # 自动补充栏信息（如果缺失）
                if enrich_column_info:
                    enrich_blocks_with_column_info(blocks, page)

                id2b = build_id_map(blocks)
                hp = compute_height_percentiles(blocks)

                # ===== 提取 Block 特征 =====
                # 预先计算栏信息（如果 enrich_column_info 已经做过，这里从 meta 读取）
                column_info_map = {}
                column_count = 1.0
                for b in blocks:
                    if b is None or "id" not in b:
                        continue
                    bid = b["id"]
                    col_id = get_meta_field(b, "column_id", 0.0)
                    col_cnt = get_meta_field(b, "column_count", 1.0)
                    is_first = get_meta_field(b, "is_first_in_column", 0.0)
                    is_last = get_meta_field(b, "is_last_in_column", 0.0)
                    column_info_map[bid] = (col_id, col_cnt, is_first, is_last)
                    column_count = max(column_count, col_cnt)
                
                for b in blocks:
                    if b is None or "id" not in b:
                        continue

                    try:
                        bid = b["id"]
                        col_info = column_info_map.get(bid, (0.0, 1.0, 0.0, 0.0))
                        feats = extract_block_feats(
                            b, page, 
                            height_percentile=hp.get(bid, 0.0),
                            column_id=col_info[0],
                            column_count=col_info[1],
                            is_first_in_column=col_info[2],
                            is_last_in_column=col_info[3]
                        )

                        if len(feats) != BLOCK_FEAT_DIM:
                            raise ValueError(f"Block 特征维度错误: {len(feats)} != {BLOCK_FEAT_DIM}")

                        Xb.append(feats)

                        b_type = b.get("type", "unknown")
                        label_idx = LABEL_MAP.index(b_type) if b_type in LABEL_MAP else LABEL_MAP.index("unknown")
                        yb.append(label_idx)

                    except (ValueError, KeyError, TypeError) as e:
                        skip_stats.block_feat_error += 1
                        continue

                # ===== 提取 Order 关系特征 =====
                pos_order: Set[Tuple] = set()

                # 1. 直接相邻的正样本
                for e in rel.get("order_edges", []):
                    u_obj = id2b.get(e.get("u"))
                    v_obj = id2b.get(e.get("v"))

                    if u_obj and v_obj:
                        try:
                            feats = extract_pair_feats(u_obj, v_obj, page, column_count=column_count)

                            if len(feats) != PAIR_FEAT_DIM:
                                raise ValueError(f"Pair 特征维度错误: {len(feats)} != {PAIR_FEAT_DIM}")

                            Xo.append(feats)
                            yo.append(1)
                            wo.append(1.0)
                            pos_order.add((e["u"], e["v"]))

                        except Exception:
                            skip_stats.pair_feat_error += 1
                            continue

                # 2. 跳步正样本
                seq = rebuild_sequence_from_edges(rel.get("order_edges", []))
                if seq and len(seq) > 2:
                    for step in [2, 3]:
                        weight = jump_weight / step

                        for i in range(len(seq) - step):
                            u_id, v_id = seq[i], seq[i + step]

                            if (u_id, v_id) in pos_order:
                                continue

                            u_obj = id2b.get(u_id)
                            v_obj = id2b.get(v_id)

                            if u_obj and v_obj:
                                try:
                                    feats = extract_pair_feats(u_obj, v_obj, page, column_count=column_count)
                                    if len(feats) == PAIR_FEAT_DIM:
                                        Xo.append(feats)
                                        yo.append(1)
                                        wo.append(weight)
                                        pos_order.add((u_id, v_id))
                                except Exception:
                                    pass

                # 3. 弱正样本（同栏相邻正文块）
                weak_pos = weak_positive_samples(blocks, pos_order, id2b, weak_pos_weight)
                for u_id, v_id, weight in weak_pos:
                    u_obj = id2b.get(u_id)
                    v_obj = id2b.get(v_id)

                    if u_obj and v_obj:
                        try:
                            feats = extract_pair_feats(u_obj, v_obj, page, column_count=column_count)
                            if len(feats) == PAIR_FEAT_DIM:
                                Xo.append(feats)
                                yo.append(1)
                                wo.append(weight)
                        except Exception:
                            pass

                # 4. 负样本：KNN 困难负样本
                knn_negs = knn_negatives(blocks, pos_order, k=6, max_neg_ratio=3)
                for neg in knn_negs:
                    u_obj = id2b.get(neg.u_id)
                    v_obj = id2b.get(neg.v_id)

                    if u_obj and v_obj:
                        try:
                            feats = extract_pair_feats(u_obj, v_obj, page, column_count=column_count)
                            if len(feats) == PAIR_FEAT_DIM:
                                Xo.append(feats)
                                yo.append(0)
                                wo.append(neg.weight)
                        except Exception:
                            pass

                # 5. 负样本：同栏跳跃负样本
                skip_negs = same_column_skip_negatives(blocks, pos_order, id2b, max_samples=50)
                for neg in skip_negs:
                    u_obj = id2b.get(neg.u_id)
                    v_obj = id2b.get(neg.v_id)

                    if u_obj and v_obj:
                        try:
                            feats = extract_pair_feats(u_obj, v_obj, page, column_count=column_count)
                            if len(feats) == PAIR_FEAT_DIM:
                                Xo.append(feats)
                                yo.append(0)
                                wo.append(neg.weight)
                        except Exception:
                            pass

                # 6. 负样本：跨栏负样本
                cross_negs = cross_column_negatives(blocks, pos_order, page, max_samples=30)
                for neg in cross_negs:
                    u_obj = id2b.get(neg.u_id)
                    v_obj = id2b.get(neg.v_id)

                    if u_obj and v_obj:
                        try:
                            feats = extract_pair_feats(u_obj, v_obj, page, column_count=column_count)
                            if len(feats) == PAIR_FEAT_DIM:
                                Xo.append(feats)
                                yo.append(0)
                                wo.append(neg.weight * cross_col_neg_weight)
                        except Exception:
                            pass

                # 7. 负样本：反向负样本
                rev_negs = reverse_negatives(pos_order, id2b, weight=rev_neg_weight)
                for neg in rev_negs:
                    u_obj = id2b.get(neg.u_id)
                    v_obj = id2b.get(neg.v_id)

                    if u_obj and v_obj:
                        try:
                            feats = extract_pair_feats(u_obj, v_obj, page, column_count=column_count)
                            if len(feats) == PAIR_FEAT_DIM:
                                Xo.append(feats)
                                yo.append(0)
                                wo.append(neg.weight)
                        except Exception:
                            pass

                # ===== 提取 Caption 关系特征 =====
                pos_cap: Set[Tuple] = set()

                for c in rel.get("caption_links", []):
                    cid, tid = c.get("caption_id"), c.get("target_id")
                    c_obj, t_obj = id2b.get(cid), id2b.get(tid)

                    if c_obj and t_obj:
                        try:
                            feats = extract_pair_feats(c_obj, t_obj, page)
                            if len(feats) == PAIR_FEAT_DIM:
                                Xc.append(feats)
                                yc.append(1)
                                wc.append(1.0)
                                pos_cap.add((cid, tid))
                        except Exception:
                            pass

                # Caption 负样本
                caps = [b for b in blocks if b.get("type") == "caption"]
                tars = [b for b in blocks if b.get("type") in ("figure", "table", "chart")]

                if caps and tars:
                    neg_pairs = caption_negatives(caps, tars, pos_cap, page, k=5)

                    for c_obj, t_obj in neg_pairs:
                        try:
                            feats = extract_pair_feats(c_obj, t_obj, page)
                            if len(feats) == PAIR_FEAT_DIM:
                                Xc.append(feats)
                                yc.append(0)
                                wc.append(1.0)
                        except Exception:
                            pass

                cnt += 1

                if max_samples and cnt >= max_samples:
                    break

            except Exception as err:
                skip_stats.other_error += 1
                if skip_stats.other_error <= 5:
                    print(f"⚠️ 跳过第 {line_num} 行: {err}")
                continue

    # 输出统计
    print(f"\n📊 数据加载完成:")
    print(f"   有效样本: {cnt}")
    print(f"   跳过样本: {skip_stats.total()}")
    if skip_stats.total() > 0:
        print(f"   跳过原因: {skip_stats.summary()}")
    print(f"   Block 特征: {len(Xb)} (维度={BLOCK_FEAT_DIM})")
    print(f"   Order 特征: {len(Xo)} (正样本={sum(yo)}, 负样本={len(yo)-sum(yo) if yo else 0})")
    print(f"   Caption 特征: {len(Xc)} (正样本={sum(yc)}, 负样本={len(yc)-sum(yc) if yc else 0})")

    return (
        np.array(Xb, dtype=np.float32) if Xb else np.zeros((0, BLOCK_FEAT_DIM), dtype=np.float32),
        np.array(yb, dtype=np.int32) if yb else np.array([], dtype=np.int32),
        np.array(Xo, dtype=np.float32) if Xo else np.zeros((0, PAIR_FEAT_DIM), dtype=np.float32),
        np.array(yo, dtype=np.int32) if yo else np.array([], dtype=np.int32),
        np.array(wo, dtype=np.float32) if wo else np.array([], dtype=np.float32),
        np.array(Xc, dtype=np.float32) if Xc else np.zeros((0, PAIR_FEAT_DIM), dtype=np.float32),
        np.array(yc, dtype=np.int32) if yc else np.array([], dtype=np.int32),
        np.array(wc, dtype=np.float32) if wc else np.array([], dtype=np.float32),
        skip_stats
    )

# =============================================================================
# 类别权重计算
# =============================================================================

def compute_class_weight(
    labels: List[int],
    gamma: float = 1.5,
    smooth: float = 0.1,
    max_weight: float = 10.0,
    min_weight: float = 0.1
) -> Dict[int, float]:
    """
    计算类别权重，支持 Focal 调制和平滑

    Args:
        labels: 标签列表
        gamma: Focal 因子（越大则罕见类权重越高）
        smooth: 平滑因子
        max_weight: 最大权重限制
        min_weight: 最小权重限制

    Returns:
        {class_id: weight}
    """
    if not labels:
        return {}

    cnt = Counter(labels)
    total = sum(cnt.values())
    n_classes = len(cnt)

    if n_classes == 0:
        return {}

    # 确保所有 LABEL_MAP 中的类都有权重
    for idx in range(len(LABEL_MAP)):
        if idx not in cnt:
            cnt[idx] = 0

    weights = {}
    for cls, count in cnt.items():
        if count == 0:
            # 未出现的类给予最大权重
            weights[cls] = max_weight
            continue
            
        # 基础权重：逆频率
        base_weight = total / (n_classes * max(count, 1))

        # Focal 调制：罕见类获得更高权重
        freq = count / total
        focal_factor = (1 - freq) ** gamma

        # 组合权重
        weight = base_weight * focal_factor

        # 平滑处理
        weight = weight * (1 - smooth) + smooth

        # 限制范围
        weight = max(min_weight, min(weight, max_weight))

        weights[cls] = weight

    # 归一化，使平均权重约为 1
    if weights:
        avg_weight = sum(weights.values()) / len(weights)
        if avg_weight > 0:
            weights = {cls: w / avg_weight for cls, w in weights.items()}

    return weights


# =============================================================================
# 指标计算
# =============================================================================

def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    label_names: List[str]
) -> Dict[str, Any]:
    """
    计算多分类指标

    Returns:
        {
            "multi_logloss": float,
            "accuracy": float,
            "macro_f1": float,
            "per_class": {class_name: {"precision", "recall", "f1", "support"}}
        }
    """
    from collections import defaultdict

    n_samples = len(y_true)
    n_classes = y_pred_proba.shape[1] if len(y_pred_proba.shape) > 1 else len(label_names)

    # 预测类别
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Accuracy
    accuracy = np.mean(y_pred == y_true)

    # Multi-class log loss
    eps = 1e-15
    y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)
    log_loss = 0.0
    for i in range(n_samples):
        log_loss -= np.log(y_pred_proba_clipped[i, y_true[i]])
    log_loss /= n_samples

    # Per-class metrics
    per_class = {}
    f1_scores = []

    for cls_idx in range(min(n_classes, len(label_names))):
        cls_name = label_names[cls_idx]

        # True positives, false positives, false negatives
        tp = np.sum((y_pred == cls_idx) & (y_true == cls_idx))
        fp = np.sum((y_pred == cls_idx) & (y_true != cls_idx))
        fn = np.sum((y_pred != cls_idx) & (y_true == cls_idx))
        support = np.sum(y_true == cls_idx)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support)
        }

        if support > 0:
            f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return {
        "multi_logloss": round(log_loss, 6),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class
    }


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    计算二分类指标

    Returns:
        {
            "binary_logloss": float,
            "auc": float,
            "precision": float,
            "recall": float,
            "f1": float,
            "threshold": float
        }
    """
    n_samples = len(y_true)

    # 处理概率
    if len(y_pred_proba.shape) > 1:
        y_proba = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
    else:
        y_proba = y_pred_proba

    # Binary log loss
    eps = 1e-15
    y_proba_clipped = np.clip(y_proba, eps, 1 - eps)
    log_loss = -np.mean(
        y_true * np.log(y_proba_clipped) + (1 - y_true) * np.log(1 - y_proba_clipped)
    )

    # AUC (简化计算)
    try:
        # 按预测概率排序
        sorted_indices = np.argsort(y_proba)[::-1]
        y_true_sorted = y_true[sorted_indices]

        # 计算 AUC
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos > 0 and n_neg > 0:
            # 累积正样本数
            cum_pos = np.cumsum(y_true_sorted)
            # 每个负样本之前的正样本数之和
            auc = np.sum(cum_pos[y_true_sorted == 0]) / (n_pos * n_neg)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    # Precision, Recall, F1
    y_pred = (y_proba >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "binary_logloss": round(log_loss, 6),
        "auc": round(auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": threshold
    }


# =============================================================================
# 训练函数
# =============================================================================

def train_lgb_classifier(
    X: np.ndarray,
    y: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    params: Dict[str, Any],
    out_path: str,
    num_class: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
    sample_weight_val: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None
) -> Optional[TrainingMetrics]:
    """
    训练 LightGBM 分类器

    Args:
        X, y: 训练数据
        Xv, yv: 验证数据
        params: LightGBM 参数
        out_path: 模型保存路径
        num_class: 类别数（多分类时需要）
        sample_weight: 训练样本权重
        sample_weight_val: 验证样本权重
        feature_names: 特征名称列表

    Returns:
        TrainingMetrics 或 None（失败时）
    """
    if lgb is None:
        print(f"❌ LightGBM 未安装，无法训练 {out_path}")
        return None

    if len(X) == 0 or len(y) == 0:
        print(f"⚠️ 训练数据为空，跳过模型 {out_path}")
        return None

    # 验证集为空时分割训练集
    if len(Xv) == 0 or len(yv) == 0:
        print(f"⚠️ 验证数据为空，使用训练数据的 10% 作为验证集")
        split_idx = int(len(X) * 0.9)
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        Xv, yv = X[val_idx], y[val_idx]
        X, y = X[train_idx], y[train_idx]

        if sample_weight is not None:
            sample_weight_val = sample_weight[val_idx]
            sample_weight = sample_weight[train_idx]

    # 创建数据集
    ds = lgb.Dataset(
        X, label=y, weight=sample_weight,
        feature_name=feature_names if feature_names else 'auto'
    )
    dv = lgb.Dataset(
        Xv, label=yv, weight=sample_weight_val,
        reference=ds,
        feature_name=feature_names if feature_names else 'auto'
    )

    # 配置参数
    p = params.copy()
    n_estimators = p.pop("n_estimators", 500)

    if num_class and num_class > 2:
        p["objective"] = "multiclass"
        p["num_class"] = num_class
        p["metric"] = "multi_logloss"
    else:
        p["objective"] = "binary"
        p["metric"] = ["binary_logloss", "auc"]

    # 回调
    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100)
    ]

    print(f"\n🚀 开始训练: {os.path.basename(out_path)}")
    print(f"   训练样本: {len(X)}, 验证样本: {len(Xv)}")

    try:
        model = lgb.train(
            p, ds,
            valid_sets=[dv],
            valid_names=["valid"],
            num_boost_round=n_estimators,
            callbacks=callbacks
        )

        # 保存模型
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        model.save_model(out_path, num_iteration=model.best_iteration)

        # 计算指标
        y_pred_proba = model.predict(Xv, num_iteration=model.best_iteration)

        if num_class and num_class > 2:
            metrics = compute_multiclass_metrics(yv, y_pred_proba, LABEL_MAP)
        else:
            metrics = compute_binary_metrics(yv, y_pred_proba)

        # 特征重要性
        importance = model.feature_importance(importance_type='gain')
        top_features = sorted(enumerate(importance), key=lambda x: -x[1])[:10]

        # 输出结果
        print(f"✅ 模型已保存: {out_path}")
        print(f"   Best iteration: {model.best_iteration}")
        print(f"   验证指标: {json.dumps(metrics, ensure_ascii=False)}")

        if feature_names:
            print(f"   Top 10 特征:")
            for idx, imp in top_features[:10]:
                if idx < len(feature_names):
                    print(f"      {feature_names[idx]}: {imp:.2f}")

        return TrainingMetrics(
            best_iteration=model.best_iteration,
            best_score=model.best_score.get("valid", {}).get(
                "multi_logloss" if num_class and num_class > 2 else "binary_logloss", 0.0
            ),
            train_samples=len(X),
            val_samples=len(Xv),
            metrics=metrics,
            feature_importance=top_features
        )

    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# 缓存管理
# =============================================================================

def cache_path(cache_dir: str, split_name: str) -> str:
    """生成缓存文件路径"""
    return os.path.join(cache_dir, f"{split_name}_v{SCHEMA_VERSION}.npz")


def load_or_extract(
    split_path: str,
    split_name: str,
    args: argparse.Namespace
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           Optional[SkipStats]]:
    """
    加载缓存或提取特征
    """
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cpath = cache_path(args.cache_dir, split_name)

        if os.path.exists(cpath) and not args.force_rebuild:
            print(f"📦 从缓存加载: {cpath}")
            data = np.load(cpath, allow_pickle=True)
            return (
                data["Xb"], data["yb"],
                data["Xo"], data["yo"], data["wo"],
                data["Xc"], data["yc"], data["wc"],
                None
            )

    print(f"📖 从文件提取特征: {split_path}")
    result = load_data(
        split_path,
        max_samples=args.max_samples,
        jump_weight=args.jump_weight,
        rev_neg_weight=args.rev_neg_weight,
        weak_pos_weight=args.weak_pos_weight,
        cross_col_neg_weight=args.cross_col_neg_weight,
        enrich_column_info=True
    )

    Xb, yb, Xo, yo, wo, Xc, yc, wc, skip_stats = result

    if args.cache_dir:
        cpath = cache_path(args.cache_dir, split_name)
        print(f"💾 保存缓存: {cpath}")
        np.savez_compressed(
            cpath,
            Xb=Xb, yb=yb,
            Xo=Xo, yo=yo, wo=wo,
            Xc=Xc, yc=yc, wc=wc
        )

    return Xb, yb, Xo, yo, wo, Xc, yc, wc, skip_stats


# =============================================================================
# 训练摘要
# =============================================================================

def save_training_summary(
    out_dir: str,
    args: argparse.Namespace,
    block_metrics: Optional[TrainingMetrics],
    order_metrics: Optional[TrainingMetrics],
    caption_metrics: Optional[TrainingMetrics],
    train_skip_stats: Optional[SkipStats],
    val_skip_stats: Optional[SkipStats]
) -> None:
    """
    保存训练摘要到 JSON 文件
    """
    summary = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": {
            "train": args.train,
            "val": args.val,
            "seed": args.seed,
            "n_estimators": args.n_estimators,
            "num_leaves": args.num_leaves,
            "learning_rate": args.learning_rate,
            "max_depth": args.max_depth,
            "jump_weight": args.jump_weight,
            "rev_neg_weight": args.rev_neg_weight,
            "weak_pos_weight": args.weak_pos_weight,
            "cross_col_neg_weight": args.cross_col_neg_weight,
        },
        "feature_dims": {
            "block": BLOCK_FEAT_DIM,
            "pair": PAIR_FEAT_DIM
        },
        "label_map": LABEL_MAP,
        "models": {}
    }

    # Block 分类器指标
    if block_metrics:
        summary["models"]["block_classifier"] = {
            "best_iteration": block_metrics.best_iteration,
            "train_samples": block_metrics.train_samples,
            "val_samples": block_metrics.val_samples,
            "metrics": block_metrics.metrics,
            "top_features": [
                {"index": idx, "importance": imp}
                for idx, imp in block_metrics.feature_importance[:10]
            ]
        }

    # Order 模型指标
    if order_metrics:
        summary["models"]["relation_scorer_order"] = {
            "best_iteration": order_metrics.best_iteration,
            "train_samples": order_metrics.train_samples,
            "val_samples": order_metrics.val_samples,
            "metrics": order_metrics.metrics,
            "top_features": [
                {"index": idx, "importance": imp}
                for idx, imp in order_metrics.feature_importance[:10]
            ]
        }

    # Caption 模型指标
    if caption_metrics:
        summary["models"]["relation_scorer_caption"] = {
            "best_iteration": caption_metrics.best_iteration,
            "train_samples": caption_metrics.train_samples,
            "val_samples": caption_metrics.val_samples,
            "metrics": caption_metrics.metrics,
            "top_features": [
                {"index": idx, "importance": imp}
                for idx, imp in caption_metrics.feature_importance[:10]
            ]
        }

    # 数据统计
    summary["data_stats"] = {
        "train_skip": train_skip_stats.summary() if train_skip_stats else None,
        "val_skip": val_skip_stats.summary() if val_skip_stats else None
    }

    # 保存
    summary_path = os.path.join(out_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📝 训练摘要已保存: {summary_path}")


def print_final_summary(
    block_metrics: Optional[TrainingMetrics],
    order_metrics: Optional[TrainingMetrics],
    caption_metrics: Optional[TrainingMetrics]
) -> None:
    """
    打印最终训练摘要
    """
    print("\n" + "=" * 60)
    print("训练完成摘要")
    print("=" * 60)

    if block_metrics:
        m = block_metrics.metrics
        print(f"\n📦 Block 分类器:")
        print(f"   Multi-LogLoss: {m.get('multi_logloss', 'N/A')}")
        print(f"   Accuracy: {m.get('accuracy', 'N/A')}")
        print(f"   Macro-F1: {m.get('macro_f1', 'N/A')}")

        # 打印重要类别的 F1
        per_class = m.get("per_class", {})
        important_classes = ["title", "paragraph", "table", "figure", "caption"]
        for cls in important_classes:
            if cls in per_class:
                print(f"   {cls}: F1={per_class[cls]['f1']}, Support={per_class[cls]['support']}")

    if order_metrics:
        m = order_metrics.metrics
        print(f"\n🔗 Order 关系模型:")
        print(f"   Binary-LogLoss: {m.get('binary_logloss', 'N/A')}")
        print(f"   AUC: {m.get('auc', 'N/A')}")
        print(f"   Precision: {m.get('precision', 'N/A')}")
        print(f"   Recall: {m.get('recall', 'N/A')}")
        print(f"   F1: {m.get('f1', 'N/A')}")

    if caption_metrics:
        m = caption_metrics.metrics
        print(f"\n🏷️ Caption 匹配模型:")
        print(f"   Binary-LogLoss: {m.get('binary_logloss', 'N/A')}")
        print(f"   AUC: {m.get('auc', 'N/A')}")
        print(f"   Precision: {m.get('precision', 'N/A')}")
        print(f"   Recall: {m.get('recall', 'N/A')}")
        print(f"   F1: {m.get('f1', 'N/A')}")

    print("\n" + "=" * 60)


# =============================================================================
# 主函数
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Document Layout Analysis - Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据路径
    ap.add_argument("--train", required=True, help="训练数据 JSONL 路径")
    ap.add_argument("--val", required=True, help="验证数据 JSONL 路径")
    ap.add_argument("--out-dir", default="artifacts", help="输出目录")
    ap.add_argument("--cache-dir", default=None, help="特征缓存目录")
    ap.add_argument("--force-rebuild", action="store_true", help="强制重新提取特征")

    # 随机种子
    ap.add_argument("--seed", type=int, default=42, help="随机种子")

    # 数据采样
    ap.add_argument("--max-samples", type=int, default=None, help="最大样本数")

    # LightGBM 参数
    ap.add_argument("--n-jobs", type=int, default=8, help="并行线程数")
    ap.add_argument("--num-leaves", type=int, default=96, help="叶子节点数")
    ap.add_argument("--max-depth", type=int, default=-1, help="最大深度")
    ap.add_argument("--learning-rate", type=float, default=0.05, help="学习率")
    ap.add_argument("--n-estimators", type=int, default=500, help="迭代次数")
    ap.add_argument("--feature-fraction", type=float, default=0.9, help="特征采样比例")
    ap.add_argument("--bagging-fraction", type=float, default=0.8, help="样本采样比例")
    ap.add_argument("--bagging-freq", type=int, default=1, help="采样频率")
    ap.add_argument("--lambda-l1", type=float, default=0.0, help="L1 正则")
    ap.add_argument("--lambda-l2", type=float, default=1.0, help="L2 正则")
    ap.add_argument("--min-data-in-leaf", type=int, default=20, help="叶子最小样本数")
    ap.add_argument("--max-bin", type=int, default=255, help="最大分箱数")

    # 样本权重参数
    ap.add_argument("--jump-weight", type=float, default=0.3, help="跳步正样本权重")
    ap.add_argument("--rev-neg-weight", type=float, default=2.0, help="反向负样本权重")
    ap.add_argument("--weak-pos-weight", type=float, default=0.5, help="弱正样本权重")
    ap.add_argument("--cross-col-neg-weight", type=float, default=1.5, help="跨栏负样本权重系数")

    # 类别权重参数
    ap.add_argument("--class-weight-gamma", type=float, default=1.5, help="类别权重 Focal gamma")
    ap.add_argument("--class-weight-smooth", type=float, default=0.1, help="类别权重平滑系数")

    args = ap.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    Path(os.path.join(args.out_dir, "models")).mkdir(parents=True, exist_ok=True)

    # 打印配置
    print("=" * 60)
    print("Document Layout Analysis - Training")
    print("=" * 60)
    print(f"Schema Version: {SCHEMA_VERSION}")
    print(f"Block 特征维度: {BLOCK_FEAT_DIM}")
    print(f"Pair 特征维度: {PAIR_FEAT_DIM}")
    print(f"随机种子: {args.seed}")
    print(f"输出目录: {args.out_dir}")
    print("=" * 60)

    # 加载数据
    print("\n" + "-" * 40)
    print("加载训练数据...")
    print("-" * 40)
    (Xb_tr, yb_tr, Xo_tr, yo_tr, wo_tr,
     Xc_tr, yc_tr, wc_tr, train_skip_stats) = load_or_extract(args.train, "train", args)

    print("\n" + "-" * 40)
    print("加载验证数据...")
    print("-" * 40)
    (Xb_va, yb_va, Xo_va, yo_va, wo_va,
     Xc_va, yc_va, wc_va, val_skip_stats) = load_or_extract(args.val, "val", args)

    # 验证特征维度
    if len(Xb_tr) > 0 and Xb_tr.shape[1] != BLOCK_FEAT_DIM:
        print(f"❌ Block 特征维度不匹配: {Xb_tr.shape[1]} != {BLOCK_FEAT_DIM}")
        print("   可能是缓存版本不匹配，请使用 --force-rebuild 重新提取特征")
        sys.exit(1)

    if len(Xo_tr) > 0 and Xo_tr.shape[1] != PAIR_FEAT_DIM:
        print(f"❌ Pair 特征维度不匹配: {Xo_tr.shape[1]} != {PAIR_FEAT_DIM}")
        print("   可能是缓存版本不匹配，请使用 --force-rebuild 重新提取特征")
        sys.exit(1)

    # LightGBM 基础参数
    base_params = {
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "n_jobs": args.n_jobs,
        "verbose": -1,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "min_data_in_leaf": args.min_data_in_leaf,
        "max_bin": args.max_bin,
        "seed": args.seed
    }

    # 获取特征名称
    block_feature_names = [name for name, _ in BLOCK_SCHEMA]
    pair_feature_names = [name for name, _ in PAIR_SCHEMA]

    # ===== 训练 Block 分类器 =====
    print("\n" + "=" * 60)
    print("训练 Block 分类器")
    print("=" * 60)

    # 计算类别权重
    cw = compute_class_weight(
        yb_tr.tolist(),
        gamma=args.class_weight_gamma,
        smooth=args.class_weight_smooth
    )

    if cw:
        print(f"类别权重: {json.dumps({LABEL_MAP[k]: round(v, 2) for k, v in cw.items()}, ensure_ascii=False)}")
        sw_tr = np.array([cw.get(c, 1.0) for c in yb_tr], dtype=np.float32)
        sw_va = np.array([cw.get(c, 1.0) for c in yb_va], dtype=np.float32)
    else:
        sw_tr, sw_va = None, None

    bc_params = base_params.copy()
    block_metrics = train_lgb_classifier(
        Xb_tr, yb_tr, Xb_va, yb_va, bc_params,
        os.path.join(args.out_dir, "models/block_classifier.json"),
        num_class=len(LABEL_MAP),
        sample_weight=sw_tr,
        sample_weight_val=sw_va,
        feature_names=block_feature_names
    )

    # ===== 训练 Order 关系模型 =====
    print("\n" + "=" * 60)
    print("训练 Order 关系模型")
    print("=" * 60)

    ro_params = base_params.copy()
    order_metrics = train_lgb_classifier(
        Xo_tr, yo_tr, Xo_va, yo_va, ro_params,
        os.path.join(args.out_dir, "models/relation_scorer_order.json"),
        sample_weight=wo_tr,
        sample_weight_val=wo_va,
        feature_names=pair_feature_names
    )

    # ===== 训练 Caption 匹配模型 =====
    print("\n" + "=" * 60)
    print("训练 Caption 匹配模型")
    print("=" * 60)

    caption_metrics = None
    if Xc_tr is not None and len(Xc_tr) > 0:
        rc_params = base_params.copy()
        caption_metrics = train_lgb_classifier(
            Xc_tr, yc_tr, Xc_va, yc_va, rc_params,
            os.path.join(args.out_dir, "models/relation_scorer_caption.json"),
            sample_weight=wc_tr,
            sample_weight_val=wc_va,
            feature_names=pair_feature_names
        )
    else:
        print("⚠️ 数据集中没有 Caption 样本，跳过 relation_scorer_caption 训练")

    # ===== 保存 Schema 和 Label Map =====
    print("\n" + "-" * 40)
    print("保存 Schema 和 Label Map...")
    print("-" * 40)

    label_map_path = os.path.join(args.out_dir, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, indent=2, ensure_ascii=False)
    print(f"✅ 保存: {label_map_path}")

    block_schema_path = os.path.join(args.out_dir, "feature_schema_block.json")
    save_schema(BLOCK_SCHEMA, block_schema_path, version=SCHEMA_VERSION)
    print(f"✅ 保存: {block_schema_path}")

    pair_schema_path = os.path.join(args.out_dir, "feature_schema_pair.json")
    save_schema(PAIR_SCHEMA, pair_schema_path, version=SCHEMA_VERSION)
    print(f"✅ 保存: {pair_schema_path}")

    # ===== 保存训练摘要 =====
    save_training_summary(
        args.out_dir, args,
        block_metrics, order_metrics, caption_metrics,
        train_skip_stats, val_skip_stats
    )

    # ===== 打印最终摘要 =====
    print_final_summary(block_metrics, order_metrics, caption_metrics)

    print("\n🎉 训练完成!")
    print(f"   输出目录: {args.out_dir}")
    print(f"   Schema 版本: {SCHEMA_VERSION}")


if __name__ == "__main__":
    main()
