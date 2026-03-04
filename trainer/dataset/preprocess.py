import argparse, json, random, os, re, math, hashlib, html, logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import html5lib
from lxml import etree
from PIL import Image

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    from scipy.spatial import cKDTree
    KDTree_OK = True
except Exception:
    KDTree_OK = False

# PaddleOCR 可选依赖
PADDLE_OCR_OK = False
_PADDLE_OCR_ENGINE = None

def _init_paddle_ocr(lang: str = "ch", use_gpu: bool = False, use_v4: bool = True):
    """
    延迟初始化 PaddleOCR（单例模式）
    
    Args:
        lang: 语言 (ch/en)
        use_gpu: 是否使用GPU
        use_v4: 是否使用PP-OCRv4（推荐）
    
    Returns:
        PaddleOCR实例或None
    """
    global PADDLE_OCR_OK, _PADDLE_OCR_ENGINE
    if _PADDLE_OCR_ENGINE is not None:
        return _PADDLE_OCR_ENGINE
    try:
        from paddleocr import PaddleOCR
        
        # PP-OCRv4 配置（更高精度）
        ocr_params = {
            "lang": lang,
            "use_angle_cls": True,  # v4支持角度分类
            "use_gpu": use_gpu,
            "show_log": False,
            "det": True,
            "rec": True,
        }
        
        if use_v4:
            # PP-OCRv4 特定参数
            ocr_params.update({
                "det_model_dir": None,  # 使用默认v4模型
                "rec_model_dir": None,
                "cls_model_dir": None,
                "use_space_char": True,  # 支持空格识别
                "drop_score": 0.5,  # 置信度阈值
            })
        
        _PADDLE_OCR_ENGINE = PaddleOCR(**ocr_params)
        PADDLE_OCR_OK = True
        logging.info(f"PaddleOCR initialized: lang={lang}, gpu={use_gpu}, v4={use_v4}")
        return _PADDLE_OCR_ENGINE
    except Exception as e:
        logging.warning(f"PaddleOCR init failed: {e}")
        PADDLE_OCR_OK = False
        return None


LABEL_MAP = [
    "title", "paragraph", "list_item", "caption", "table", "figure",
    "formula", "header", "footer", "chart", "unknown"
]

# 与 eval.py / train.py 对齐的常量
SCHEMA_VERSION = "2.0"
BLOCK_FEAT_DIM = 29
PAIR_FEAT_DIM = 34

# 文本类型与非文本类型
TEXT_BLOCK_TYPES = {"paragraph", "title", "list_item", "caption", "header", "footer"}
NON_TEXT_BLOCK_TYPES = {"table", "figure", "chart", "formula"}
TAG_WHITELIST = {f"h{i}" for i in range(1, 7)} | {
    "p", "li", "figcaption", "figure", "table", "span", "header", "footer",
    "div", "img", "br", "thead", "tbody", "tr", "th", "td",
}

# Caption 关键词正则（多语言、多格式支持）
_CAPTION_PATTERN = re.compile(
    r'(?i)^\s*[\[\(]*'                           # 开头可选括号
    r'(figure|fig|table|tab|图|表)'              # 关键词
    r'\.?\s*'                                     # 可选点号
    r'(S?\d+(?:\.\d+)?)'                          # 主编号：S1, 2, 2.1
    r'(?:\s*[-–—]\s*(S?\d+(?:\.\d+)?))?'          # 可选范围：-3
    r'(?:\s*[\(\[]?\s*([a-zA-Z])\s*[\)\]]?)?'     # 可选子图：(a)
    r'(?:\s*(?:和|and|&|,)\s*'                    # 可选多目标连接词
    r'(?:figure|fig|table|tab|图|表)?\.?\s*'
    r'(S?\d+(?:\.\d+)?)'                          # 第二个编号
    r'(?:\s*[\(\[]?\s*([a-zA-Z])\s*[\)\]]?)?)?'   # 第二个子图
)


# ---------- Utils ----------
def stable_hash(s: str, seed: int = 42) -> int:
    h = hashlib.sha256((str(seed) + s).encode()).hexdigest()
    return int(h[:8], 16)


def wrap_body(html_str: str) -> str:
    html_str = html_str.strip()
    if "<body" not in html_str.lower():
        return f"<body>{html_str}</body>"
    return html_str


def parse_html(html_str: str) -> Optional[etree.Element]:
    try:
        doc = html5lib.parse(html_str, namespaceHTMLElements=False)
        return doc.find(".//body")
    except Exception as e:
        logging.warning(f"HTML parse failed: {e}")
        return None


def bbox_from_attr(el) -> Optional[List[int]]:
    bbox_str = el.get("data-bbox")
    if not bbox_str:
        return None
    parts = bbox_str.strip().split()
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, parts)
        if x2 <= x1 or y2 <= y1:
            return None
        return [int(x1), int(y1), int(x2), int(y2)]
    except Exception:
        return None


def clamp_bbox(bbox, w, h) -> Optional[List[int]]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


CLASS_TO_TYPE = {
    "image": "image",
    "figure": "image",
    "chart": "chart",
    "formula": "formula",
    "equation": "formula",
    "header": "header",
    "footer": "footer",
    "page_number": "page_number",
    "page-number": "page_number",
    "list_item": "list_item",
    "list-item": "list_item",
    "caption": "caption",
    "table": "table",
    "paragraph": "paragraph",
    "text": "paragraph",
    "title": "title",
    "heading": "title",
    "seal": "seal",
    "handwriting": "handwriting",
    "code": "code",
    "toc": "toc",
    "reference": "reference",
    "abstract": "abstract",
    "footnote": "footnote",
    "watermark": "watermark",
}


def detect_block_type(el) -> Tuple[str, Optional[int]]:
    tag = el.tag.lower()
    if tag in [f"h{i}" for i in range(1, 7)]:
        return "title", int(tag[1])
    if tag == "p":
        return "paragraph", None
    if tag == "li":
        return "list_item", None
    if tag == "figcaption":
        return "caption", None
    if tag == "figure":
        return "figure", None
    if tag == "table":
        return "table", None
    if tag == "header":
        return "header", None
    if tag == "footer":
        return "footer", None
    if tag == "div":
        classes = el.get("class", "").split()
        for cls in classes:
            mapped = CLASS_TO_TYPE.get(cls, None)
            if mapped is not None:
                return mapped, None
        return "paragraph", None
    cls_list = el.get("class", "").split()
    if "formula" in cls_list or "equation" in cls_list:
        return "formula", None
    return "unknown", None

# ---------- 公式LaTeX处理增强（BLEU指标优化） ----------
def normalize_latex(latex: str) -> str:
    """
    规范化LaTeX公式字符串
    
    1. 移除多余空格
    2. 统一命令格式
    3. 移除注释
    """
    if not latex:
        return ""
    
    # 移除注释
    latex = re.sub(r'%.*$', '', latex, flags=re.MULTILINE)
    
    # 统一空格
    latex = re.sub(r'\s+', ' ', latex)
    latex = latex.strip()
    
    # 统一常见命令
    replacements = [
        (r'\\left\s*\(', r'\left('),
        (r'\\right\s*\)', r'\right)'),
        (r'\\left\s*\[', r'\left['),
        (r'\\right\s*\]', r'\right]'),
        (r'\\frac\s*{', r'\frac{'),
        (r'\\sum\s*_', r'\sum_'),
        (r'\\int\s*_', r'\int_'),
        (r'\\prod\s*_', r'\prod_'),
    ]
    
    for pattern, replacement in replacements:
        latex = re.sub(pattern, replacement, latex)
    
    return latex


def extract_latex_from_element(el) -> str:
    """从HTML元素提取LaTeX"""
    latex = ""
    
    # 优先从data-latex属性获取
    if el.get("data-latex"):
        latex = el.get("data-latex")
    
    # 尝试从MathML转换
    mathml = el.find(".//math")
    if mathml is not None and not latex:
        try:
            # 简单的MathML到LaTeX转换
            latex = _mathml_to_latex_simple(mathml)
        except Exception:
            pass
    
    # 尝试从文本内容提取
    if not latex:
        text = "".join(el.itertext()).strip()
        # 检查是否是LaTeX格式
        if text.startswith("$") or text.startswith("\\"):
            latex = text.strip("$").strip()
    
    return normalize_latex(latex)


def _mathml_to_latex_simple(mathml_elem) -> str:
    """简单的MathML到LaTeX转换"""
    # 基础实现，实际项目中可用专业库
    result = []
    
    for elem in mathml_elem.iter():
        tag = elem.tag.lower() if isinstance(elem.tag, str) else ""
        text = (elem.text or "").strip()
        
        if tag.endswith("mi"):  # 标识符
            result.append(text)
        elif tag.endswith("mn"):  # 数字
            result.append(text)
        elif tag.endswith("mo"):  # 运算符
            result.append(f" {text} ")
        elif tag.endswith("mfrac"):  # 分数
            result.append("\\frac")
        elif tag.endswith("msup"):  # 上标
            result.append("^")
        elif tag.endswith("msub"):  # 下标
            result.append("_")
        elif tag.endswith("msqrt"):  # 平方根
            result.append("\\sqrt")
    
    return "".join(result)




def safe_median(arr: List[float], default: float = 0.0) -> float:
    if not arr:
        return default
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    mid = n // 2
    if n % 2 == 1:
        return float(arr_sorted[mid])
    return float(0.5 * (arr_sorted[mid - 1] + arr_sorted[mid]))

# ---------- OCR ROI 处理 ----------
def ocr_roi(image: Image.Image, bbox: List[int], ocr_engine, 
             padding_ratio: float = 0.03, min_area: int = 64,
             use_cls: bool = True) -> Tuple[str, float]:
    """
    对指定 bbox 区域进行 OCR（PP-OCRv4优化版）

    Args:
        image: PIL Image 对象
        bbox: [x1, y1, x2, y2]
        ocr_engine: PaddleOCR 实例
        padding_ratio: bbox 扩展比例
        min_area: 最小ROI面积
        use_cls: 是否使用角度分类

    Returns:
        (ocr_text, avg_confidence)
    """
    if ocr_engine is None:
        return "", 0.0

    x1, y1, x2, y2 = bbox
    img_w, img_h = image.size

    # ROI 尺寸检查
    roi_w, roi_h = x2 - x1, y2 - y1
    if roi_w < 8 or roi_h < 8 or roi_w * roi_h < min_area:
        return "", 0.0

    # 自适应padding（小区域用更大padding）
    adaptive_ratio = padding_ratio * (1 + 100 / max(roi_w * roi_h, 1))
    adaptive_ratio = min(adaptive_ratio, 0.15)
    
    pad_x = int(roi_w * adaptive_ratio)
    pad_y = int(roi_h * adaptive_ratio)
    x1_p = max(0, x1 - pad_x)
    y1_p = max(0, y1 - pad_y)
    x2_p = min(img_w, x2 + pad_x)
    y2_p = min(img_h, y2 + pad_y)

    # 裁剪 ROI
    try:
        roi_img = image.crop((x1_p, y1_p, x2_p, y2_p))
        roi_array = np.array(roi_img)
        
        # 确保RGB格式
        if len(roi_array.shape) == 2:
            roi_array = np.stack([roi_array] * 3, axis=-1)
        elif roi_array.shape[2] == 4:
            roi_array = roi_array[:, :, :3]
    except Exception:
        return "", 0.0

    # OCR识别
    try:
        result = ocr_engine.ocr(roi_array, cls=use_cls)
    except Exception:
        return "", 0.0

    if not result or not result[0]:
        return "", 0.0

    # 按y坐标排序文本行（保持阅读顺序）
    lines_with_pos = []
    for line in result[0]:
        if len(line) >= 2:
            txt = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
            conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) > 1 else 0.9
            # 获取文本行的y坐标
            if isinstance(line[0], (list, tuple)) and len(line[0]) >= 4:
                y_center = (line[0][0][1] + line[0][2][1]) / 2
            else:
                y_center = 0
            lines_with_pos.append((y_center, txt, float(conf)))

    if not lines_with_pos:
        return "", 0.0

    # 按y坐标排序
    lines_with_pos.sort(key=lambda x: x[0])
    
    texts = [l[1] for l in lines_with_pos]
    confs = [l[2] for l in lines_with_pos]

    # 智能拼接（检测是否需要换行）
    if len(texts) > 1:
        # 检查是否为多行文本（y坐标差异大）
        y_coords = [l[0] for l in lines_with_pos]
        y_diff = max(y_coords) - min(y_coords) if y_coords else 0
        if y_diff > roi_h * 0.3:
            ocr_text = "\n".join(texts)  # 多行保留换行
        else:
            ocr_text = " ".join(texts)  # 单行用空格
    else:
        ocr_text = texts[0] if texts else ""

    avg_conf = sum(confs) / len(confs) if confs else 0.0

    return ocr_text, avg_conf


def should_replace_text(original: str, ocr_text: str, ocr_conf: float,
                        min_len_thresh: int = 10, conf_thresh: float = 0.7) -> bool:
    """
    判断是否应该用 OCR 结果替换原文本。

    策略：
    1. 原文本为空或过短时替换
    2. OCR 结果明显更长且置信度高时替换
    3. 其他情况保守不替换
    """
    original = (original or "").strip()
    ocr_text = (ocr_text or "").strip()

    if not ocr_text:
        return False

    # 原文本为空或过短
    if len(original) < min_len_thresh:
        return ocr_conf >= conf_thresh

    # OCR 结果明显更长且置信度高
    if len(ocr_text) > len(original) * 1.5 and ocr_conf >= conf_thresh:
        # 额外检查：不能是大量乱码/重复
        unique_chars = len(set(ocr_text))
        if unique_chars > len(ocr_text) * 0.3:  # 字符多样性检查
            return True

    return False


def augment_blocks_with_ocr(blocks: List[Dict[str, Any]], image: Image.Image,
                            ocr_engine, args, debug: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    对 blocks 进行 OCR 文本增强。

    Returns:
        增强后的 blocks 列表
    """
    if ocr_engine is None:
        return blocks

    ocr_stats = {
        "total_blocks": len(blocks),
        "ocr_attempted": 0,
        "ocr_replaced": 0,
        "ocr_skipped": 0,
        "avg_conf": 0.0,
        "confs": []
    }

    text_types = {"paragraph", "list_item", "caption", "title", "header", "footer"}

    for block in blocks:
        # 只对文本类型块做 OCR
        if block.get("type") not in text_types:
            continue

        # figure/table 等非文本块跳过
        if block.get("type") in ("figure", "table", "chart", "formula"):
            continue

        ocr_stats["ocr_attempted"] += 1

        original_text = block.get("text") or ""
        bbox = block.get("bbox")

        if not bbox:
            ocr_stats["ocr_skipped"] += 1
            continue

        # 执行 ROI OCR
        ocr_text, ocr_conf = ocr_roi(image, bbox, ocr_engine,
                                      padding_ratio=getattr(args, 'ocr_padding_ratio', 0.03))

        if ocr_conf > 0:
            ocr_stats["confs"].append(ocr_conf)

        # 判断是否替换
        min_len = getattr(args, 'ocr_min_text_len', 10)
        conf_thresh = getattr(args, 'ocr_conf_thresh', 0.7)

        if should_replace_text(original_text, ocr_text, ocr_conf, min_len, conf_thresh):
            # 保存原文本到 debug
            if "meta" not in block:
                block["meta"] = {}
            block["meta"]["original_text"] = original_text
            block["meta"]["ocr_conf"] = ocr_conf

            # 替换文本
            block["text"] = ocr_text
            block["source"] = "roi_ocr"
            ocr_stats["ocr_replaced"] += 1
        else:
            ocr_stats["ocr_skipped"] += 1
            # 即使不替换，也记录 OCR 信息供参考
            if ocr_text and "meta" not in block:
                block["meta"] = {}
            if ocr_text:
                block["meta"]["ocr_text_alt"] = ocr_text
                block["meta"]["ocr_conf"] = ocr_conf

    # 汇总统计
    if ocr_stats["confs"]:
        ocr_stats["avg_conf"] = sum(ocr_stats["confs"]) / len(ocr_stats["confs"])
    del ocr_stats["confs"]  # 不保存完整列表

    debug["ocr_stats"] = ocr_stats
    debug["ocr_used"] = ocr_stats["ocr_replaced"] > 0

    return blocks


# ---------- Extraction ----------
def extract_blocks_and_tables(body) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    blocks, tables = [], []
    bid = 0
    bbox_key_to_id = {}
    for el in body.iter():
        if el.tag is etree.Comment:
            continue
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        if tag not in TAG_WHITELIST:
            continue
        bbox = bbox_from_attr(el)
        if not bbox:
            continue
        btype, hlevel = detect_block_type(el)
        text = "".join(el.itertext()).strip()
        if btype == "table":
            tbl_block = {
                "id": bid, "bbox": bbox, "type": "table",
                "score": 1.0, "text": None, "style": None, "source": "object"
            }
            blocks.append(tbl_block)
            rows = []
            for tr in el.findall(".//tr"):
                row_cells = []
                for td in tr.findall(".//td") + tr.findall(".//th"):
                    cb = bbox_from_attr(td) or bbox
                    txt = "".join(td.itertext()).strip()
                    rowspan = int(td.get("rowspan", "1"))
                    colspan = int(td.get("colspan", "1"))
                    row_cells.append({
                        "bbox": cb, "text": txt, "rowspan": rowspan, "colspan": colspan
                    })
                if row_cells:
                    rows.append(row_cells)
            tables.append({
                "id": bid, "bbox": bbox, "type": "table",
                "score": 1.0, "rows": rows, "source": "object"
            })
            bbox_key_to_id[" ".join(map(str, bbox))] = bid
            bid += 1
        elif btype == "figure":
            blocks.append({
                "id": bid, "bbox": bbox, "type": btype,
                "score": 1.0, "text": None, "style": None, "source": "object"
            })
            bbox_key_to_id[" ".join(map(str, bbox))] = bid
            bid += 1
        elif btype == "formula":
            latex = el.get("data-latex", "") or ""
            blocks.append({
                "id": bid, "bbox": bbox, "type": btype,
                "score": 1.0, "text": text, "style": None,
                "source": "object", "latex": latex
            })
            bbox_key_to_id[" ".join(map(str, bbox))] = bid
            bid += 1
        else:
            blocks.append({
                "id": bid, "bbox": bbox, "type": btype,
                "score": 1.0, "text": text,
                "style": {"heading_level": hlevel} if hlevel else None,
                "source": "object"
            })
            bbox_key_to_id[" ".join(map(str, bbox))] = bid
            bid += 1
    return blocks, tables, bbox_key_to_id


# ---------- Caption 解析增强 ----------
def parse_caption_info(text: str) -> Dict[str, Any]:
    """
    解析 caption 文本，提取类型、编号、子图等信息。

    支持格式：
    - Fig.2a / Fig. 2a / Fig 2(a) / Figure 2-a
    - Table S1 / Tab. S1
    - Fig 2.1（层级）
    - 图2（a） / 图 2(a) / 图2a
    - 表S1 / 表 S1
    - 图2和图3 / Fig. 2 and 3 / Fig 2-3

    Returns:
        {
            "type": "figure" | "table" | None,
            "main_number": "2" | "S1" | "2.1" | None,
            "sub": "a" | None,
            "range_end": "3" | None,  # 范围结束
            "second_number": "3" | None,  # 多目标引用
            "second_sub": "b" | None
        }
    """
    if not text:
        return {"type": None, "main_number": None, "sub": None,
                "range_end": None, "second_number": None, "second_sub": None}

    text_clean = text.strip()
    match = _CAPTION_PATTERN.match(text_clean)

    if not match:
        return {"type": None, "main_number": None, "sub": None,
                "range_end": None, "second_number": None, "second_sub": None}

    keyword = match.group(1).lower()
    main_num = match.group(2)
    range_end = match.group(3)
    sub = match.group(4).lower() if match.group(4) else None
    second_num = match.group(5)
    second_sub = match.group(6).lower() if match.group(6) else None

    # 归一化类型
    if keyword in ("figure", "fig", "图"):
        cap_type = "figure"
    elif keyword in ("table", "tab", "表"):
        cap_type = "table"
    else:
        cap_type = None

    return {
        "type": cap_type,
        "main_number": main_num,
        "sub": sub,
        "range_end": range_end,
        "second_number": second_num,
        "second_sub": second_sub
    }


def caption_number(text: str) -> Optional[str]:
    """提取 caption 主编号（兼容旧接口，返回 str 以支持 S1 等格式）"""
    info = parse_caption_info(text)
    return info.get("main_number")


def caption_type_matches_target(caption_info: Dict[str, Any], target_type: str) -> bool:
    """检查 caption 类型是否与 target 类型匹配"""
    cap_type = caption_info.get("type")
    if cap_type is None:
        return True  # 无法确定时不限制

    if cap_type == "figure" and target_type in ("figure", "chart"):
        return True
    if cap_type == "table" and target_type == "table":
        return True

    return False


def get_caption_target_numbers(caption_info: Dict[str, Any]) -> List[str]:
    """获取 caption 引用的所有目标编号（支持范围和多目标）"""
    numbers = []

    main = caption_info.get("main_number")
    if main:
        numbers.append(main)

    # 处理范围 (Fig 2-3)
    range_end = caption_info.get("range_end")
    if range_end and main:
        try:
            start = int(main)
            end = int(range_end)
            for i in range(start + 1, end + 1):
                numbers.append(str(i))
        except ValueError:
            pass

    # 处理多目标引用 (图2和图3)
    second = caption_info.get("second_number")
    if second:
        numbers.append(second)

    return numbers

# ---------- Order helpers ----------
def build_kdtree(blocks):
    if not KDTree_OK or len(blocks) == 0:
        return None, None
    centers = np.array([((b["bbox"][0] + b["bbox"][2]) / 2,
                         (b["bbox"][1] + b["bbox"][3]) / 2) for b in blocks])
    ids = [b["id"] for b in blocks]
    return cKDTree(centers), ids


def dom_order_blocks(body, blocks, bbox_key_to_id):
    tree, ids = build_kdtree(blocks)
    order = []
    seen = set()
    for el in body.iter():
        if el.tag is etree.Comment:
            continue
        tag = el.tag.lower() if isinstance(el.tag, str) else ""
        if tag not in TAG_WHITELIST:
            continue
        bbox = bbox_from_attr(el)
        if not bbox:
            continue
        key = " ".join(map(str, bbox))
        bid = bbox_key_to_id.get(key, None)
        if bid is None and tree is not None:
            q = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            dist, idx = tree.query(q, k=1)
            bid = ids[idx]
        elif bid is None and tree is None:
            bx1, by1 = bbox[0], bbox[1]
            bid = min(blocks, key=lambda b: abs(b["bbox"][0] - bx1) + abs(b["bbox"][1] - by1))["id"]
        if bid is not None and bid not in seen:
            order.append(bid)
            seen.add(bid)
    return order


def detect_columns(blocks, page_w, min_col_gap_ratio=0.10, min_count=3, max_columns=4):
    """
    检测页面栏数，支持 1~max_columns 栏。

    改进点：
    1. DBSCAN 支持多栏
    2. header/footer 不参与分栏检测
    3. 栏宽不等时用最近中心分配
    """
    # 只用文本块检测，排除 header/footer
    text_blocks = [b for b in blocks
                   if b["type"] in ("paragraph", "list_item", "caption", "title")
                   and b["type"] not in ("header", "footer")]

    if len(text_blocks) < min_count * 2:
        labels = {b["id"]: 0 for b in blocks}
        return 1, labels

    xs = np.array([(b["bbox"][0] + b["bbox"][2]) / 2 for b in text_blocks])

    # 尝试使用 DBSCAN
    try:
        from sklearn.cluster import DBSCAN
        eps = min_col_gap_ratio * page_w
        X = xs.reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=min_count).fit(X)
        cluster_labels = clustering.labels_

        unique_labels = set(cluster_labels) - {-1}
        n_columns = len(unique_labels)

        if n_columns < 1 or n_columns > max_columns:
            labels = {b["id"]: 0 for b in blocks}
            return 1, labels

        # 计算每个聚类的中心
        cluster_centers = {}
        for label in unique_labels:
            cluster_centers[label] = np.mean(xs[cluster_labels == label])

        # 按 x 坐标排序聚类，重新编号
        sorted_labels = sorted(cluster_centers.keys(), key=lambda l: cluster_centers[l])
        label_remap = {old: new for new, old in enumerate(sorted_labels)}
        center_list = [cluster_centers[old] for old in sorted_labels]

        # 为所有 blocks 分配列标签
        labels = {}
        for b in blocks:
            # header/footer 强制 column_id=0
            if b["type"] in ("header", "footer"):
                labels[b["id"]] = 0
                continue

            cx = (b["bbox"][0] + b["bbox"][2]) / 2
            min_dist = float('inf')
            best_label = 0
            for new_label, center in enumerate(center_list):
                dist = abs(cx - center)
                if dist < min_dist:
                    min_dist = dist
                    best_label = new_label
            labels[b["id"]] = best_label

        return n_columns, labels

    except ImportError:
        pass

    # 降级：两列检测
    x_min, x_max = xs.min(), xs.max()
    gap = (x_max - x_min) * min_col_gap_ratio
    med = np.median(xs)
    left = xs[xs <= med]
    right = xs[xs > med]

    if len(left) >= min_count and len(right) >= min_count and (right.mean() - left.mean()) > gap:
        centers = [left.mean(), right.mean()]
        n_columns = 2
    else:
        labels = {b["id"]: 0 for b in blocks}
        return 1, labels

    labels = {}
    for b in blocks:
        if b["type"] in ("header", "footer"):
            labels[b["id"]] = 0
            continue
        c = (b["bbox"][0] + b["bbox"][2]) / 2
        lbl = 0 if abs(c - centers[0]) < abs(c - centers[1]) else 1
        labels[b["id"]] = lbl

    return n_columns, labels


def compute_line_groups(blocks: List[Dict[str, Any]], page_h: float) -> Dict[int, int]:
    """
    按 y 坐标聚类计算行组 ID。

    阈值：max(0.012 * page_h, 0.6 * median_text_block_height)

    Returns:
        {block_id: line_group_id}
    """
    if not blocks:
        return {}

    # 计算中位块高度
    text_blocks = [b for b in blocks if b["type"] in ("paragraph", "list_item", "caption", "title")]
    if text_blocks:
        heights = [(b["bbox"][3] - b["bbox"][1]) for b in text_blocks]
        median_h = safe_median(heights, default=20.0)
    else:
        median_h = 20.0

    thresh = max(0.012 * page_h, 0.6 * median_h)

    # 按 y_center 排序
    sorted_blocks = sorted(blocks, key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2)

    line_groups = {}
    current_group = 0
    prev_y = None

    for b in sorted_blocks:
        y_center = (b["bbox"][1] + b["bbox"][3]) / 2

        if prev_y is None or abs(y_center - prev_y) > thresh:
            current_group += 1

        line_groups[b["id"]] = current_group
        prev_y = y_center

    return line_groups


def add_block_meta_features(blocks: List[Dict[str, Any]], page_w: float, page_h: float,
                            column_labels: Dict[int, int], line_groups: Dict[int, int],
                            tables: List[Dict[str, Any]]) -> None:
    """
    为 blocks 添加训练用的 meta 特征字段（与train.py BLOCK_SCHEMA对齐）
    
    添加字段:
    - column_id, column_count, is_first_in_column, is_last_in_column
    - text_line_count, avg_line_height_norm (归一化)
    - line_group_id, avg_char_width
    - table_stats (对于表格块)
    """
    table_id_set = {t["id"] for t in tables}
    table_stats = {t["id"]: {
        "rows": len(t.get("rows", [])),
        "cols": max((sum(c.get("colspan", 1) for c in r) for r in t.get("rows", [[]])), default=0),
        "num_cells": sum(len(r) for r in t.get("rows", []))
    } for t in tables}
    
    # 计算栏数
    column_count = len(set(column_labels.values())) if column_labels else 1
    
    # 按栏分组确定first/last
    columns_blocks: Dict[int, List[Dict]] = {}
    for b in blocks:
        col_id = column_labels.get(b["id"], 0)
        columns_blocks.setdefault(col_id, []).append(b)
    
    # 每栏内按y排序
    for col_id, col_blocks in columns_blocks.items():
        col_blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        for i, b in enumerate(col_blocks):
            if "meta" not in b:
                b["meta"] = {}
            b["meta"]["is_first_in_column"] = 1.0 if i == 0 else 0.0
            b["meta"]["is_last_in_column"] = 1.0 if i == len(col_blocks) - 1 else 0.0

    for b in blocks:
        if "meta" not in b:
            b["meta"] = {}

        bid = b["id"]
        bbox = b["bbox"]
        text = b.get("text") or ""

        # column信息
        b["meta"]["column_id"] = column_labels.get(bid, 0)
        b["meta"]["column_count"] = column_count

        # line_group_id
        b["meta"]["line_group_id"] = line_groups.get(bid, 0)

        # 文本统计
        bbox_w = max(1, bbox[2] - bbox[0])
        bbox_h = max(1, bbox[3] - bbox[1])
        text_len = len(text)

        # avg_char_width
        if text_len > 0:
            b["meta"]["avg_char_width"] = bbox_w / text_len
        else:
            b["meta"]["avg_char_width"] = -1

        # text_line_count（多种估算方法）
        line_count = 1
        if text:
            # 方法1: 换行符计数
            newline_count = text.count("\n") + 1
            # 方法2: 基于bbox高度和典型行高估算
            typical_line_height = 20  # 典型行高像素
            height_based_count = max(1, int(bbox_h / typical_line_height))
            # 取两者较大值
            line_count = max(newline_count, height_based_count)
        
        b["meta"]["text_line_count"] = line_count
        b["meta"]["block_text_line_count"] = line_count  # 兼容旧字段

        # avg_line_height (像素)
        b["meta"]["avg_line_height"] = bbox_h / line_count if line_count > 0 else bbox_h
        
        # avg_line_height_norm (归一化到页高，与train.py对齐)
        b["meta"]["avg_line_height_norm"] = (bbox_h / line_count) / page_h if line_count > 0 and page_h > 0 else 0
        b["meta"]["avg_line_height_px"] = bbox_h / line_count if line_count > 0 else bbox_h

        # table_stats
        if bid in table_id_set:
            b["meta"]["table_stats"] = table_stats.get(bid, {})

def build_order_edges_dom(blocks, body, page_w, page_h):
    """基于 DOM 顺序构建 order edges，结合多栏检测优化"""
    id2b = {b["id"]: b for b in blocks}
    order_dom = dom_order_blocks(body, blocks, {" ".join(map(str, b["bbox"])): b["id"] for b in blocks})
    k, labels = detect_columns(blocks, page_w)

    # 分离 header/title、正文、footer
    titles = [bid for bid in order_dom if id2b[bid]["type"] in ("title", "header")]
    footers = [bid for bid in order_dom if id2b[bid]["type"] == "footer"]
    mid = [bid for bid in order_dom if bid not in titles and bid not in footers]

    order = []

    # 1. header/title 按 y 坐标排序放在最前
    titles_sorted = sorted(titles, key=lambda bid: (id2b[bid]["bbox"][1], id2b[bid]["bbox"][0]))
    order.extend(titles_sorted)

    # 2. 正文按栏处理
    if k >= 2:
        grouped = {}
        for bid in mid:
            c = labels.get(bid, 0)
            grouped.setdefault(c, []).append(id2b[bid])

        # 按栏的 x 中心排序
        for c in sorted(grouped.keys(), key=lambda c: np.mean([b["bbox"][0] for b in grouped[c]])):
            col_blocks = grouped[c]
            # 栏内按 y 排序，同行按 x 排序
            ys = sorted(col_blocks, key=lambda b: b["bbox"][1])
            merged = []
            line = []
            if ys:
                line_y = ys[0]["bbox"][1]
                for b in ys:
                    if abs(b["bbox"][1] - line_y) < 0.02 * page_h:
                        line.append(b)
                    else:
                        merged.extend(sorted(line, key=lambda x: x["bbox"][0]))
                        line = [b]
                        line_y = b["bbox"][1]
                merged.extend(sorted(line, key=lambda x: x["bbox"][0]))
                order.extend([b["id"] for b in merged])
    else:
        # 单栏：按 y, x 排序
        mid_sorted = sorted([id2b[bid] for bid in mid], key=lambda b: (b["bbox"][1], b["bbox"][0]))
        order.extend([b["id"] for b in mid_sorted])

    # 3. footer 放在最后
    footers_sorted = sorted(footers, key=lambda bid: (id2b[bid]["bbox"][1], id2b[bid]["bbox"][0]))
    order.extend(footers_sorted)

    edges = topo_order_edges(order)
    return edges, order


def build_order_edges_geom(blocks, page_w, page_h):
    """
    纯几何方式构建 order edges（KTDS指标优化）
    
    支持:
    1. 多栏排版（最多4栏）
    2. 浮动图片绕排
    3. 表格/图片的caption关联
    """
    if not blocks:
        return [], []
    
    id2b = {b["id"]: b for b in blocks}

    # 分类块
    titles = [b for b in blocks if b["type"] in ("title", "header")]
    footers = [b for b in blocks if b["type"] == "footer"]
    captions = [b for b in blocks if b["type"] == "caption"]
    floats = [b for b in blocks if b["type"] in ("figure", "table", "chart")]
    mids = [b for b in blocks if b not in titles and b not in footers 
            and b not in captions and b not in floats]

    k, labels = detect_columns(mids + floats, page_w)

    order = []

    # 1. 页眉/标题按位置排序
    order.extend([b["id"] for b in sorted(titles, key=lambda x: (x["bbox"][1], x["bbox"][0]))])

    # 2. 处理多栏正文
    if k >= 2:
        grouped = {}
        for b in mids:
            c = labels.get(b["id"], 0)
            grouped.setdefault(c, []).append(b)
        
        # 浮动元素按栏分组
        float_grouped = {}
        for b in floats:
            c = labels.get(b["id"], 0)
            float_grouped.setdefault(c, []).append(b)

        # 按栏顺序处理
        col_order = sorted(grouped.keys(), 
                          key=lambda c: np.mean([b["bbox"][0] for b in grouped.get(c, [{"bbox":[0,0,0,0]}])]))
        
        for c in col_order:
            col_blocks = grouped.get(c, [])
            col_floats = float_grouped.get(c, [])
            
            # 合并该栏的所有元素
            all_in_col = col_blocks + col_floats
            
            # 按y坐标排序，同行按x排序
            sorted_blocks = sorted(all_in_col, key=lambda b: (b["bbox"][1], b["bbox"][0]))
            
            # 行合并逻辑
            merged = []
            line = []
            line_y = sorted_blocks[0]["bbox"][1] if sorted_blocks else 0
            line_thresh = 0.025 * page_h  # 行高阈值
            
            for b in sorted_blocks:
                if abs(b["bbox"][1] - line_y) < line_thresh:
                    line.append(b)
                else:
                    # 处理当前行
                    merged.extend(sorted(line, key=lambda x: x["bbox"][0]))
                    line = [b]
                    line_y = b["bbox"][1]
            
            merged.extend(sorted(line, key=lambda x: x["bbox"][0]))
            
            # 插入caption（紧跟其目标）
            final_order = []
            used_captions = set()
            
            for b in merged:
                final_order.append(b["id"])
                
                # 查找该块的caption
                if b["type"] in ("figure", "table", "chart"):
                    b_bbox = b["bbox"]
                    for cap in captions:
                        if cap["id"] in used_captions:
                            continue
                        cap_bbox = cap["bbox"]
                        # caption在目标下方或上方
                        vertical_dist = abs((cap_bbox[1] + cap_bbox[3])/2 - (b_bbox[1] + b_bbox[3])/2)
                        horizontal_overlap = min(cap_bbox[2], b_bbox[2]) - max(cap_bbox[0], b_bbox[0])
                        if vertical_dist < 0.1 * page_h and horizontal_overlap > 0:
                            final_order.append(cap["id"])
                            used_captions.add(cap["id"])
            
            order.extend(final_order)
        
        # 添加未使用的caption
        for cap in captions:
            if cap["id"] not in used_captions:
                order.append(cap["id"])
    else:
        # 单栏处理
        all_content = mids + floats
        sorted_content = sorted(all_content, key=lambda x: (x["bbox"][1], x["bbox"][0]))
        
        used_captions = set()
        for b in sorted_content:
            order.append(b["id"])
            
            if b["type"] in ("figure", "table", "chart"):
                b_bbox = b["bbox"]
                for cap in captions:
                    if cap["id"] in used_captions:
                        continue
                    cap_bbox = cap["bbox"]
                    vertical_dist = abs((cap_bbox[1] + cap_bbox[3])/2 - (b_bbox[1] + b_bbox[3])/2)
                    horizontal_overlap = min(cap_bbox[2], b_bbox[2]) - max(cap_bbox[0], b_bbox[0])
                    if vertical_dist < 0.1 * page_h and horizontal_overlap > 0:
                        order.append(cap["id"])
                        used_captions.add(cap["id"])
        
        for cap in captions:
            if cap["id"] not in used_captions:
                order.append(cap["id"])

    # 3. 页脚最后
    order.extend([b["id"] for b in sorted(footers, key=lambda x: (x["bbox"][1], x["bbox"][0]))])

    # 去重保序
    seen = set()
    unique_order = []
    for bid in order:
        if bid not in seen:
            seen.add(bid)
            unique_order.append(bid)
    
    # 确保所有块都在序列中
    for b in blocks:
        if b["id"] not in seen:
            unique_order.append(b["id"])

    edges = topo_order_edges(unique_order)
    return edges, unique_order


def topo_order_edges(order_ids):
    return [{"u": u, "v": v, "score": 1.0} for u, v in zip(order_ids[:-1], order_ids[1:])]


# ---------- Kendall proxy with guard ----------
def kendall_proxy(order_a, order_b, limit=200):
    """计算两个序列的 Kendall tau 距离代理"""
    if len(order_a) > limit or len(order_b) > limit:
        return 0.0
    pos_a = {bid: i for i, bid in enumerate(order_a)}
    common = [bid for bid in order_b if bid in pos_a]
    inv = 0
    total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            total += 1
            if pos_a[common[i]] > pos_a[common[j]]:
                inv += 1
    return inv / total if total > 0 else 0.0


def choose_order(order_dom, order_geom, thresh):
    score = kendall_proxy(order_dom, order_geom)
    return (order_geom if score > thresh else order_dom), score

# ---------- Caption 匹配改进 ----------
def caption_cost(caption, target, page_diag=None, all_targets=None,
                 caption_info=None, target_rank_map=None):
    """
    计算 caption 与 target 的匹配代价。

    改进点：
    1. 类型匹配约束
    2. 编号匹配
    3. 空间方向约束（figure 下方，table 上下均可）
    """
    cx1, cy1, cx2, cy2 = caption["bbox"]
    tx1, ty1, tx2, ty2 = target["bbox"]
    c_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
    t_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)

    dy = c_center[1] - t_center[1]
    dx = abs(c_center[0] - t_center[0])

    page_diag = page_diag or 1000.0
    dy_norm = dy / page_diag
    dx_norm = dx / page_diag

    dist = math.hypot(dx_norm, dy_norm)
    target_type = target.get("type", "figure")

    # === 类型匹配约束 ===
    type_penalty = 0
    if caption_info:
        if not caption_type_matches_target(caption_info, target_type):
            type_penalty = 50  # 类型不匹配，高惩罚

    # === 空间方向约束 ===
    if target_type in ("figure", "chart"):
        # figure/chart: caption 通常在下方
        if dy > 0:
            dir_term = -10  # 在下方，奖励
        elif dy < -0.05 * page_diag:
            dir_term = 30   # 在很上方，强惩罚
        else:
            dir_term = 10   # 略上方，轻惩罚
    else:  # table
        # table: 允许上下
        if abs(dy) < 0.08 * page_diag:
            dir_term = -5   # 距离近，奖励
        elif abs(dy) > 0.2 * page_diag:
            dir_term = 20   # 太远，惩罚
        else:
            dir_term = 5

    # === 水平对齐 ===
    ovx = max(0, min(cx2, tx2) - max(cx1, tx1)) / max(1, min(cx2 - cx1, tx2 - tx1))
    align_bonus = -ovx * 10

    # === 编号匹配 ===
    semantic_cost = 0
    if caption_info and target_rank_map:
        cap_numbers = get_caption_target_numbers(caption_info)
        target_id = target["id"]
        target_rank = target_rank_map.get((target_type, target_id))

        if cap_numbers and target_rank is not None:
            # 检查是否有匹配的编号
            matched = False
            for num_str in cap_numbers:
                try:
                    num = int(num_str) if num_str.isdigit() else num_str
                    if num == target_rank or str(num) == str(target_rank):
                        matched = True
                        break
                except ValueError:
                    if num_str == str(target_rank):
                        matched = True
                        break

            if matched:
                semantic_cost = -15  # 编号匹配，强奖励
            else:
                # 编号不匹配，根据差距惩罚
                try:
                    first_num = int(cap_numbers[0]) if cap_numbers[0].isdigit() else 0
                    semantic_cost = min(20, 5 * abs(first_num - target_rank))
                except (ValueError, TypeError):
                    semantic_cost = 5

    # === 距离惩罚 ===
    dist_penalty = dist * 40

    # === 垂直间距 ===
    vertical_gap = abs(dy_norm) * 20

    total_cost = dist_penalty + dir_term + align_bonus + semantic_cost + vertical_gap + type_penalty

    return total_cost


def normalize_scores(cost_matrix):
    c_min = cost_matrix.min()
    c_max = cost_matrix.max()
    denom = max(1e-6, c_max - c_min)
    norm = (cost_matrix - c_min) / denom
    return np.exp(-norm)


def greedy_match(cost):
    m, n = cost.shape
    used_r, used_c = set(), set()
    pairs = []
    for _ in range(min(m, n)):
        idx = np.unravel_index(np.argmin(cost), cost.shape)
        r, c = idx
        if r in used_r or c in used_c:
            cost[r, c] = np.inf
            continue
        pairs.append((r, c))
        used_r.add(r)
        used_c.add(c)
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return zip(*pairs) if pairs else ([], [])


def build_caption_links(blocks: List[Dict[str, Any]], cost_gating: float, page_diag: float,
                        allow_one_to_many: bool = False, max_links_per_caption: int = 2
                        ) -> Tuple[List[Dict[str, float]], List[float]]:
    """
    构建 caption-target 链接。

    改进点：
    1. 类型匹配过滤
    2. 编号匹配加权
    3. 空间约束
    4. 一对多支持（可选）
    5. 保守策略（宁可不连）
    """
    captions = [b for b in blocks if b["type"] == "caption"]
    targets = [b for b in blocks if b["type"] in ("figure", "table", "chart")]
    links, costs_dbg = [], []

    if not captions or not targets:
        return links, costs_dbg

    # 预计算 target 排名（按类型分组，按位置排序）
    target_rank_map = {}
    for t_type in ("figure", "table", "chart"):
        typed_targets = sorted(
            [t for t in targets if t["type"] == t_type],
            key=lambda t: (t["bbox"][1], t["bbox"][0])
        )
        for rank, t in enumerate(typed_targets, 1):
            target_rank_map[(t_type, t["id"])] = rank

    # 解析所有 caption 信息
    caption_infos = [parse_caption_info(c.get("text", "")) for c in captions]

    # 构建代价矩阵
    cost = np.full((len(captions), len(targets)), np.inf)

    for i, c in enumerate(captions):
        c_info = caption_infos[i]

        for j, t in enumerate(targets):
            # 类型不匹配直接跳过（设为 inf）
            if c_info.get("type") and not caption_type_matches_target(c_info, t["type"]):
                continue

            cost[i, j] = caption_cost(
                c, t, page_diag, targets,
                caption_info=c_info,
                target_rank_map=target_rank_map
            )

    # 匈牙利算法或贪心匹配
    if SCIPY_OK:
        # 处理 inf 值
        cost_for_assign = cost.copy()
        cost_for_assign[np.isinf(cost_for_assign)] = 1e9
        row_ind, col_ind = linear_sum_assignment(cost_for_assign)
    else:
        row_ind, col_ind = greedy_match(cost.copy())

    score_mat = normalize_scores(np.where(np.isinf(cost), 1e9, cost))

    # 构建链接
    used_captions = set()
    used_targets = set()

    for r, c in zip(row_ind, col_ind):
        if np.isinf(cost[r, c]):
            continue

        # 保守阈值
        if score_mat[r, c] < 0.20:
            continue
        if cost[r, c] > cost_gating:
            continue

        # 额外空间检查
        cap = captions[r]
        tar = targets[c]
        cap_y = (cap["bbox"][1] + cap["bbox"][3]) / 2
        tar_y = (tar["bbox"][1] + tar["bbox"][3]) / 2

        # figure 的 caption 不能在目标很上方
        if tar["type"] in ("figure", "chart"):
            if cap_y < tar["bbox"][1] - 0.05 * page_diag:
                continue

        links.append({
            "caption_id": captions[r]["id"],
            "target_id": targets[c]["id"],
            "score": float(score_mat[r, c])
        })
        costs_dbg.append(float(cost[r, c]))
        used_captions.add(r)
        used_targets.add(c)

    # 一对多处理（可选）
    if allow_one_to_many:
        for i, c in enumerate(captions):
            if i in used_captions:
                continue

            c_info = caption_infos[i]
            cap_numbers = get_caption_target_numbers(c_info)

            # 如果有范围或多目标引用
            if len(cap_numbers) > 1:
                matched_targets = []
                for j, t in enumerate(targets):
                    if j in used_targets:
                        continue
                    if np.isinf(cost[i, j]) or cost[i, j] > cost_gating:
                        continue

                    t_rank = target_rank_map.get((t["type"], t["id"]))
                    if t_rank and str(t_rank) in cap_numbers:
                        matched_targets.append((j, cost[i, j]))

                # 取最近的 max_links_per_caption 个
                matched_targets.sort(key=lambda x: x[1])
                for j, c_val in matched_targets[:max_links_per_caption]:
                    links.append({
                        "caption_id": captions[i]["id"],
                        "target_id": targets[j]["id"],
                        "score": float(score_mat[i, j])
                    })
                    costs_dbg.append(float(c_val))

    return links, costs_dbg

# ---------- Image heavy ----------
def detect_image_heavy(blocks, page_area, args):
    text_blocks = [b for b in blocks if b["type"] in ("title", "paragraph", "list_item", "caption")]
    num_text_blocks = len(text_blocks)
    text_chars = sum(len(b.get("text", "") or "") for b in text_blocks)
    text_area = sum((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) for b in text_blocks)
    largest_area = max(((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) for b in blocks), default=0)
    text_area_ratio = text_area / max(page_area, 1)
    largest_ratio = largest_area / max(page_area, 1)
    cond1 = num_text_blocks < args.min_text_blocks
    cond2 = text_area_ratio < args.text_area_ratio_thresh
    cond3 = largest_ratio > args.largest_block_ratio_thresh
    cond4 = text_chars < args.text_chars_thresh
    return (cond1 and cond3) or (cond2 and cond3) or cond4


def build_heading_parent(blocks, order_seq=None):
    """
    构建标题层级关系。

    改进点：
    1. 过滤 header/footer（不参与层级）
    2. 按 order_seq 顺序处理
    3. 使用栈法构建父子关系
    """
    # 只处理有 heading_level 的 title，排除 header/footer
    headings = [
        b for b in blocks
        if b["type"] == "title"
        and b.get("style")
        and b["style"].get("heading_level")
        and b["type"] not in ("header", "footer")
    ]

    if not headings:
        return []

    # 按 order_seq 排序
    if order_seq:
        id_pos = {bid: i for i, bid in enumerate(order_seq)}
        headings = sorted(headings, key=lambda b: id_pos.get(b["id"], b["id"]))
    else:
        headings = sorted(headings, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    parent_edges = []
    stack = []  # [(id, level)]

    for h in headings:
        lvl = h["style"]["heading_level"]

        # 弹出 level >= 当前的节点
        while stack and stack[-1][1] >= lvl:
            stack.pop()

        # 栈顶是父节点
        if stack:
            parent_id = stack[-1][0]
            if parent_id != h["id"]:  # 防止自引用
                parent_edges.append({
                    "child_id": h["id"],
                    "parent_id": parent_id,
                    "score": 1.0
                })

        stack.append((h["id"], lvl))

    return parent_edges


# ---------- Chain enforcement ----------
def enforce_chain(blocks, order_edges):
    id_set = {b["id"] for b in blocks}
    outd, ind = {}, {}
    for e in order_edges:
        outd[e["u"]] = outd.get(e["u"], 0) + 1
        ind[e["v"]] = ind.get(e["v"], 0) + 1
    if any(v > 1 for v in outd.values()) or any(v > 1 for v in ind.values()):
        return False
    heads = [i for i in id_set if ind.get(i, 0) == 0]
    if len(heads) != 1:
        return False
    head = heads[0]
    nxt = {e["u"]: e["v"] for e in order_edges}
    seen = set()
    cur = head
    while cur in nxt and cur not in seen:
        seen.add(cur)
        cur = nxt[cur]
    seen.add(cur)
    return len(seen) == len(id_set)


def enforce_chain_or_rebuild(blocks, order_edges, page_w, page_h):
    if enforce_chain(blocks, order_edges) and len(order_edges) == len(blocks) - 1:
        return order_edges
    edges, order = build_order_edges_geom(blocks, page_w, page_h)
    return edges


def rebuild_sequence_from_edges(order_edges):
    if not order_edges:
        return []
    nxt = {e["u"]: e["v"] for e in order_edges}
    heads = set(nxt.keys()) - set(nxt.values())
    if not heads:
        return []
    head = list(heads)[0]
    seq, seen, cur = [], set(), head
    while cur in nxt and cur not in seen:
        seq.append(cur)
        seen.add(cur)
        cur = nxt[cur]
    seq.append(cur)
    return seq

# ---------- Renderer ----------
def renderer(ir: Dict[str, Any]) -> str:
    """
    将 IR 渲染为 HTML 字符串（严格符合评测Schema）
    
    格式要求:
    - 标题: <h1>-<h6> 或 <h2> (默认)
    - 段落: <p>
    - 列表: <ul><li>...</li></ul>
    - 图片/图表: <figure> 或 <div class="figure">
    - 表格: <table> with <thead>/<tbody>/<tr>/<td>
    - 公式: <span class="formula" data-latex="...">
    - 页眉/页脚: <header>/<footer> 或 <div class="header/footer">
    """
    def bbox_str(b):
        return " ".join(str(int(round(v))) for v in b)

    blocks = ir.get("blocks", [])
    relations = ir.get("relations", {})
    order_edges = relations.get("order_edges", [])
    id2b = {b["id"]: b for b in blocks}

    if not blocks:
        return "<body></body>"

    order = None
    max_iterations = len(blocks) + 10

    # 尝试链式顺序
    try:
        if enforce_chain(blocks, order_edges) and len(order_edges) == len(blocks) - 1:
            nxt = {e["u"]: e["v"] for e in order_edges}
            heads = set(nxt.keys()) - set(nxt.values())

            if heads:
                head = list(heads)[0]
            elif order_edges:
                head = order_edges[0]["u"]
            else:
                head = blocks[0]["id"]

            seq = []
            seen = set()
            cur = head
            iterations = 0
            while cur in nxt and cur not in seen and iterations < max_iterations:
                seq.append(cur)
                seen.add(cur)
                cur = nxt[cur]
                iterations += 1

            if cur not in seen:
                seq.append(cur)
            order = seq
    except Exception:
        order = None

    # 拓扑排序
    if order is None:
        try:
            nxt = {}
            indeg = {b["id"]: 0 for b in blocks}
            for e in order_edges:
                nxt.setdefault(e["u"], []).append(e["v"])
                if e["v"] in indeg:
                    indeg[e["v"]] = indeg.get(e["v"], 0) + 1
                else:
                    indeg[e["v"]] = 1

            queue = [bid for bid, count in indeg.items() if count == 0]
            if not queue and blocks:
                queue = [blocks[0]["id"]]

            seq = []
            iterations = 0
            while queue and iterations < max_iterations:
                u = queue.pop(0)
                if u in id2b and u not in seq:
                    seq.append(u)
                    for v in nxt.get(u, []):
                        indeg[v] = indeg.get(v, 1) - 1
                        if indeg[v] == 0:
                            queue.append(v)
                iterations += 1

            for bid in id2b.keys():
                if bid not in seq:
                    seq.append(bid)
            order = seq
        except Exception:
            order = [b["id"] for b in blocks]

    if order is None:
        order = [b["id"] for b in blocks]

    # 生成 HTML
    html_parts = ["<body>"]
    for bid in order:
        b = id2b.get(bid)
        if b is None:
            continue
        bb = bbox_str(b["bbox"])
        t = b["type"]
        txt = html.escape(b.get("text", "") or "")

        if t == "title":
            hlevel = b.get("style", {}).get("heading_level") if b.get("style") else None
            tag = f"h{hlevel}" if hlevel else "h2"
            html_parts.append(f'<{tag} data-bbox="{bb}">{txt}</{tag}>')
        elif t == "paragraph":
            html_parts.append(f'<p data-bbox="{bb}">{txt}</p>')
        elif t == "list_item":
            html_parts.append(f'<ul><li data-bbox="{bb}">{txt}</li></ul>')
        elif t == "caption":
            ref = ""
            for lk in relations.get("caption_links", []):
                if lk["caption_id"] == b["id"]:
                    ref = f' data-ref="{lk["target_id"]}"'
                    break
            html_parts.append(f'<figcaption data-bbox="{bb}"{ref}>{txt}</figcaption>')
        elif t == "figure":
            html_parts.append(f'<figure data-bbox="{bb}"></figure>')
        elif t == "table":
            html_parts.append(f'<table data-bbox="{bb}">')
            tbl = next((tb for tb in ir.get("tables", []) if tb["id"] == b["id"]), None)
            rows = tbl.get("rows", []) if tbl else []
            for row in rows:
                html_parts.append("<tr>")
                for cell in row:
                    cbb = bbox_str(cell["bbox"])
                    rs = cell.get("rowspan", 1)
                    cs = cell.get("colspan", 1)
                    ctext = html.escape(cell.get("text", "") or "")
                    html_parts.append(f'<td data-bbox="{cbb}" rowspan="{rs}" colspan="{cs}">{ctext}</td>')
                html_parts.append("</tr>")
            html_parts.append("</table>")
        elif t == "formula":
            latex = html.escape(b.get("latex", "") or "")
            html_parts.append(f'<span class="formula" data-bbox="{bb}" data-latex="{latex}"></span>')
        else:
            tag = t if t != "unknown" else "p"
            html_parts.append(f'<{tag} data-bbox="{bb}">{txt}</{tag}>')

    html_parts.append("</body>")
    return "".join(html_parts)

# ---------- OCR / detectors (stubs) ----------
def run_ocr(image_path: str, regions=None):
    return []


def run_layout_det(image_path: str):
    return []


def run_table_det(image_path: str):
    return []


# ---------- Table validation ----------
def validate_table_structure(tbl, bad_records, img_name):
    """
    验证并修复表格结构（TEDS指标优化）
    
    确保:
    1. rowspan/colspan 不越界
    2. 单元格不重叠
    3. 表格结构完整闭合
    """
    rows = tbl.get("rows", [])
    if not rows:
        return tbl
    
    # 计算最大列数
    max_cols = 0
    for r in rows:
        ccount = sum(c.get("colspan", 1) for c in r)
        max_cols = max(max_cols, ccount)
    
    if max_cols == 0:
        return tbl
    
    # 计算考虑rowspan后的实际行数
    row_heights = []
    for r in rows:
        rmax = max((c.get("rowspan", 1) for c in r), default=1)
        row_heights.append(rmax)
    
    max_over = max(row_heights) - 1 if row_heights else 0
    grid_rows = len(rows) + max_over
    
    # 构建占用网格
    grid = [[0] * max_cols for _ in range(grid_rows)]
    fixed = 0
    new_rows = []
    
    for ri, r in enumerate(rows):
        cur_row = []
        ci = 0
        
        for c in r:
            # 跳过已占用的列
            while ci < max_cols and grid[ri][ci] == 1:
                ci += 1
            
            if ci >= max_cols:
                fixed += 1
                continue
            
            rs = max(1, min(c.get("rowspan", 1), grid_rows - ri))
            cs = max(1, min(c.get("colspan", 1), max_cols - ci))
            
            # 检测冲突
            conflict = False
            for rr in range(ri, min(ri + rs, grid_rows)):
                for cc in range(ci, min(ci + cs, max_cols)):
                    if grid[rr][cc] == 1:
                        conflict = True
                        break
                if conflict:
                    break
            
            if conflict:
                # 冲突时收缩为1x1
                rs, cs = 1, 1
                while ci < max_cols and grid[ri][ci] == 1:
                    ci += 1
                if ci >= max_cols:
                    fixed += 1
                    continue
            
            # 标记占用
            for rr in range(ri, min(ri + rs, grid_rows)):
                for cc in range(ci, min(ci + cs, max_cols)):
                    grid[rr][cc] = 1
            
            # 更新单元格
            c["rowspan"], c["colspan"] = rs, cs
            
            # 确保bbox有效
            if "bbox" not in c or not c["bbox"]:
                c["bbox"] = tbl.get("bbox", [0, 0, 100, 100])
            
            cur_row.append(c)
            ci += cs
        
        if cur_row:
            new_rows.append(cur_row)
    
    if fixed > 0:
        bad_records.append({
            "image": img_name, 
            "reason": "table_span_fix", 
            "fixed": fixed,
            "original_rows": len(rows),
            "final_rows": len(new_rows)
        })
    
    tbl["rows"] = new_rows
    
    # 添加表格元信息
    tbl["meta"] = tbl.get("meta", {})
    tbl["meta"]["num_rows"] = len(new_rows)
    tbl["meta"]["num_cols"] = max_cols
    tbl["meta"]["total_cells"] = sum(len(r) for r in new_rows)
    
    return tbl


def validate_and_clamp_blocks(blocks, tables, page_w, page_h, bad_records, img_name):
    good_blocks = []
    for b in blocks:
        cb = clamp_bbox(b["bbox"], page_w, page_h)
        if not cb:
            bad_records.append({"image": img_name, "reason": "bbox_invalid", "bbox": b["bbox"]})
            continue
        b["bbox"] = cb
        good_blocks.append(b)
    good_tables = []
    for t in tables:
        cb = clamp_bbox(t["bbox"], page_w, page_h)
        if not cb:
            bad_records.append({"image": img_name, "reason": "table_bbox_invalid", "bbox": t["bbox"]})
            continue
        t["bbox"] = cb
        new_rows = []
        for row in t.get("rows", []):
            new_row = []
            for cell in row:
                cbb = clamp_bbox(cell["bbox"], page_w, page_h) if cell.get("bbox") else cb
                if not cbb:
                    cbb = cb
                cell["bbox"] = cbb
                new_row.append(cell)
            new_rows.append(new_row)
        t["rows"] = new_rows
        t = validate_table_structure(t, bad_records, img_name)
        good_tables.append(t)
    return good_blocks, good_tables


# ---------- Main processing ----------
def process_line(obj, args, image_root, ocr_engine=None):
    """处理单个样本"""
    html_raw = obj.get("suffix", "")
    html_wrapped = wrap_body(html_raw)
    body = parse_html(html_wrapped)
    if body is None:
        return None, {"image": obj.get("image"), "reason": "parse_fail"}

    blocks, tables, bbox_idx = extract_blocks_and_tables(body)
    if not blocks:
        return None, {"image": obj.get("image"), "reason": "no_blocks"}

    img_path = os.path.join(image_root, obj.get("image"))
    page_w = page_h = None
    debug = {}
    pil_image = None

    try:
        pil_image = Image.open(img_path)
        page_w, page_h = pil_image.size
    except Exception:
        debug["page_size_inferred"] = True
        page_w = max(b["bbox"][2] for b in blocks)
        page_h = max(b["bbox"][3] for b in blocks)

    bad_local = []
    blocks, tables = validate_and_clamp_blocks(blocks, tables, page_w, page_h, bad_local, obj.get("image"))
    if not blocks:
        if pil_image:
            pil_image.close()
        return None, {"image": obj.get("image"), "reason": "all_blocks_clamped_out"}

    page = {"image": obj.get("image"), "width": int(page_w), "height": int(page_h)}
    page_diag = math.hypot(page_w, page_h)
    page_area = page_w * page_h

    # 检测是否需要 OCR
    is_heavy = detect_image_heavy(blocks, page_area, args)
    debug["is_image_heavy"] = bool(is_heavy)

    # OCR 增强逻辑
    should_do_ocr = False
    if getattr(args, 'enable_ocr', False):
        if getattr(args, 'augment_ocr_text', False):
            should_do_ocr = True
        elif is_heavy:
            should_do_ocr = True

    if should_do_ocr and ocr_engine is not None and pil_image is not None:
        blocks = augment_blocks_with_ocr(blocks, pil_image, ocr_engine, args, debug)

    # 关闭图像
    if pil_image:
        pil_image.close()

    # 计算栏和行组
    k, column_labels = detect_columns(blocks, page_w)
    line_groups = compute_line_groups(blocks, page_h)

    # 添加 meta 特征
    add_block_meta_features(blocks, page_w, page_h, column_labels, line_groups, tables)

    debug["num_columns"] = k

    # 构建 order
    od_edges, od_seq = build_order_edges_dom(blocks, body, page_w, page_h)
    od_edges = enforce_chain_or_rebuild(blocks, od_edges, page_w, page_h)
    od_seq = rebuild_sequence_from_edges(od_edges) if len(od_edges) == len(blocks) - 1 else od_seq

    og_edges, og_seq = build_order_edges_geom(blocks, page_w, page_h)
    og_edges = enforce_chain_or_rebuild(blocks, og_edges, page_w, page_h)
    og_seq = rebuild_sequence_from_edges(og_edges) if len(og_edges) == len(blocks) - 1 else og_seq

    order_seq, disagree = choose_order(od_seq, og_seq, args.order_disagree_thresh)
    order_edges = topo_order_edges(order_seq)

    # 构建 caption links
    allow_one_to_many = getattr(args, 'caption_one_to_many', False)
    caption_links, cap_costs = build_caption_links(
        blocks,
        cost_gating=args.caption_cost_gating,
        page_diag=page_diag,
        allow_one_to_many=allow_one_to_many
    )

    # 构建 heading parent
    heading_parent = build_heading_parent(blocks, order_seq)

    debug["caption_costs"] = cap_costs
    debug["order_disagree"] = disagree

    ir = {
        "page": page,
        "blocks": blocks,
        "tables": tables,
        "relations": {
            "order_edges": order_edges,
            "caption_links": caption_links,
            "heading_parent": heading_parent
        },
        "debug": debug
    }

    # 如果 image_heavy 且启用了 OCR/detector，重新构建
    if is_heavy:
        if args.enable_ocr and not should_do_ocr:
            # 使用旧的 run_ocr 添加新块
            ocr_blocks = []
            max_id = max(b["id"] for b in blocks)
            for idx, ob in enumerate(run_ocr(img_path)):
                ob["id"] = max_id + idx + 1
                ob["type"] = "paragraph"
                ob["style"] = None
                ob["source"] = "ocr"
                ob["score"] = ob.get("conf", 1.0)
                ocr_blocks.append(ob)
            if ocr_blocks:
                blocks.extend(ocr_blocks)
                ir["debug"]["ocr_used"] = True

        if args.enable_layout_det:
            ir["debug"]["layout_det_used"] = True
        if args.enable_table_det:
            ir["debug"]["table_det_used"] = True

        # 重新构建 order
        og_edges, og_seq = build_order_edges_geom(blocks, page_w, page_h)
        og_edges = enforce_chain_or_rebuild(blocks, og_edges, page_w, page_h)
        order_seq = rebuild_sequence_from_edges(og_edges) if len(og_edges) == len(blocks) - 1 else og_seq
        order_edges = topo_order_edges(order_seq)

        caption_links, cap_costs = build_caption_links(
            blocks,
            cost_gating=args.caption_cost_gating,
            page_diag=page_diag,
            allow_one_to_many=allow_one_to_many
        )
        heading_parent = build_heading_parent(blocks, order_seq)

        ir["relations"]["order_edges"] = order_edges
        ir["relations"]["caption_links"] = caption_links
        ir["relations"]["heading_parent"] = heading_parent
        ir["debug"]["caption_costs"] = cap_costs
        ir["debug"]["order_disagree"] = None

    html_out = renderer(ir)
    try:
        _ = html5lib.parse(html_out, namespaceHTMLElements=False)
    except Exception:
        return None, {"image": obj.get("image"), "reason": "render_fail"}

    bad_merge = bad_local if bad_local else None
    return {
        "image": obj.get("image"),
        "prompt": obj.get("prefix", ""),
        "ir": ir,
        "html": html_out
    }, bad_merge

# ---------- Profile ----------
def profile_defaults(profile: str, args):
    if profile == "fast":
        args.enable_ocr = False
        args.enable_layout_det = False
        args.enable_table_det = False
        args.text_area_ratio_thresh = 0.05
        args.largest_block_ratio_thresh = 0.75
        args.text_chars_thresh = 80
    elif profile == "accurate":
        args.enable_ocr = True
        args.enable_layout_det = True
        args.enable_table_det = True
        args.text_area_ratio_thresh = 0.12
        args.largest_block_ratio_thresh = 0.5
        args.text_chars_thresh = 40
    else:  # balanced
        args.text_area_ratio_thresh = 0.08
        args.largest_block_ratio_thresh = 0.6
        args.text_chars_thresh = 60
    return args


def worker(line_data):
    """Worker 函数，用于多进程处理"""
    line, global_args = line_data
    
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        return None, {"image": "unknown", "reason": f"json_parse_error: {str(e)[:100]}"}
    
    image_name = obj.get("image", "unknown")
    
    # 初始化 OCR（在 worker 内部，避免序列化问题）
    ocr_engine = None
    if getattr(global_args, 'enable_ocr', False) and (
        getattr(global_args, 'augment_ocr_text', False) or True
    ):
        ocr_engine = _init_paddle_ocr(
            lang=getattr(global_args, 'ocr_lang', 'ch'),
            use_gpu=getattr(global_args, 'ocr_use_gpu', False)
        )
    
    try:
        return process_line(obj, global_args, global_args.image_root, ocr_engine=ocr_engine)
    except Exception as e:
        import traceback
        error_msg = f"process_error: {str(e)[:200]}"
        return None, {"image": image_name, "reason": error_msg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-train", required=True)
    ap.add_argument("--output-val", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    
    # OCR 相关参数 (PP-OCRv4)
    ap.add_argument("--enable-ocr", action="store_true", help="启用 OCR 能力（总开关）")
    ap.add_argument("--augment-ocr-text", action="store_true", 
                    help="强制对所有页面进行 ROI OCR 文本增强")
    ap.add_argument("--ocr-lang", default="ch", help="OCR 语言（ch/en）")
    ap.add_argument("--ocr-use-gpu", action="store_true", help="OCR 使用 GPU")
    ap.add_argument("--ocr-use-v4", action="store_true", default=True,
                    help="使用 PP-OCRv4 模型（推荐）")
    ap.add_argument("--ocr-min-text-len", type=int, default=10, 
                    help="原文本长度低于此阈值时用 OCR 替换")
    ap.add_argument("--ocr-conf-thresh", type=float, default=0.7, 
                    help="OCR 置信度阈值")
    ap.add_argument("--ocr-padding-ratio", type=float, default=0.03, 
                    help="ROI 裁剪时的 padding 比例")
    ap.add_argument("--ocr-max-pages", type=int, default=None,
                    help="最多对多少页进行 OCR（吞吐保护）")
    ap.add_argument("--ocr-sample-rate", type=float, default=1.0,
                    help="OCR 采样率（0~1）")
    
    # Layout/Table 检测
    ap.add_argument("--enable-layout-det", action="store_true")
    ap.add_argument("--enable-table-det", action="store_true")
    
    # Profile
    ap.add_argument("--profile", choices=["fast", "balanced", "accurate"], default="balanced")
    
    # 阈值参数
    ap.add_argument("--min-text-blocks", type=int, default=5)
    ap.add_argument("--text-area-ratio-thresh", type=float, default=None)
    ap.add_argument("--largest-block-ratio-thresh", type=float, default=None)
    ap.add_argument("--text-chars-thresh", type=int, default=None)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--caption-cost-gating", type=float, default=250.0)
    ap.add_argument("--order-disagree-thresh", type=float, default=0.35)
    
    # Caption 一对多
    ap.add_argument("--caption-one-to-many", action="store_true",
                    help="允许 caption 链接多个 target（范围引用）")
    
    args = ap.parse_args()
    random.seed(args.seed)
    args = profile_defaults(args.profile, args)
    
    # 确保阈值参数有值
    if args.text_area_ratio_thresh is None:
        args.text_area_ratio_thresh = 0.08
    if args.largest_block_ratio_thresh is None:
        args.largest_block_ratio_thresh = 0.6
    if args.text_chars_thresh is None:
        args.text_chars_thresh = 60
    
    Path("data").mkdir(exist_ok=True)
    bad_records = []
    samples = []

    lines = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line)
            if args.max_samples and len(lines) >= args.max_samples:
                break

    # OCR 吞吐保护
    ocr_page_count = 0
    ocr_max = args.ocr_max_pages or len(lines)

    if args.num_workers and args.num_workers > 0:
        from multiprocessing import Pool
        with Pool(args.num_workers) as pool:
            for rec, bad in pool.imap_unordered(worker, [(line, args) for line in lines]):
                if rec:
                    samples.append(rec)
                if bad:
                    if isinstance(bad, list):
                        bad_records.extend(bad)
                    else:
                        bad_records.append(bad)
    else:
        # 单进程模式，可以共享 OCR engine
        ocr_engine = None
        if args.enable_ocr:
            ocr_engine = _init_paddle_ocr(
                lang=args.ocr_lang,
                use_gpu=args.ocr_use_gpu
            )
        
        for line in lines:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad_records.append({"image": "unknown", "reason": f"json_parse_error: {str(e)[:100]}"})
                continue
            
            image_name = obj.get("image", "unknown")
            
            # OCR 采样和限额控制
            current_ocr_engine = None
            if ocr_engine and ocr_page_count < ocr_max:
                if random.random() < args.ocr_sample_rate:
                    current_ocr_engine = ocr_engine
                    ocr_page_count += 1
            
            try:
                rec, bad = process_line(obj, args, args.image_root, ocr_engine=current_ocr_engine)
                if rec:
                    samples.append(rec)
                if bad:
                    if isinstance(bad, list):
                        bad_records.extend(bad)
                    else:
                        bad_records.append(bad)
            except Exception as e:
                bad_records.append({"image": image_name, "reason": f"process_error: {str(e)[:200]}"})

    n = len(samples)
    if args.train_ratio is None:
        if n > 100_000:
            ratio = 0.95
        elif n < 10_000:
            ratio = 0.8
        else:
            ratio = 0.9
    else:
        ratio = args.train_ratio
    
    train, val = [], []
    for s in samples:
        h = stable_hash(s["image"], args.seed)
        if (h % 1000) / 1000.0 < ratio:
            train.append(s)
        else:
            val.append(s)
    
    with open(args.output_train, "w", encoding="utf-8") as ft:
        for r in train:
            ft.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    with open(args.output_val, "w", encoding="utf-8") as fv:
        for r in val:
            fv.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    with open("data/bad_samples.jsonl", "w", encoding="utf-8") as fb:
        for b in bad_records:
            fb.write(json.dumps(b, ensure_ascii=False) + "\n")
    
    # 打印统计
    print(f"Total samples: {n}")
    print(f"Train: {len(train)}, Val: {len(val)}")
    print(f"Bad records: {len(bad_records)}")
    if PADDLE_OCR_OK:
        print(f"OCR pages processed: {ocr_page_count}")


if __name__ == "__main__":
    main()
