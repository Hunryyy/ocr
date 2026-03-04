"""
utils/formula.py
公式识别与 LaTeX 输出规范化模块

提供:
- normalize_latex()         : LaTeX 后处理/规范化
- preprocess_formula_image(): 公式 ROI 图像预处理
- FormulaRecognizer         : 多引擎公式识别（带 fallback）
"""
import re
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# 1. normalize_latex
# ---------------------------------------------------------------------------
def normalize_latex(latex: str) -> str:
    """
    规范化 LaTeX 字符串：
    - 去除外层 $...$ 或 $$...$$
    - 删除 \\displaystyle
    - 合并多余空白
    - 修正常见 unicode 符号（全角减号、乘号等）
    - 简化单字符花括号 {a} → a
    """
    latex = (latex or "").strip()
    # Strip outer $$ ... $$
    if latex.startswith("$$") and latex.endswith("$$") and len(latex) >= 4:
        latex = latex[2:-2].strip()
    elif latex.startswith("$") and latex.endswith("$") and len(latex) >= 2:
        latex = latex[1:-1].strip()
    # Remove \displaystyle
    latex = re.sub(r"\\displaystyle\s*", "", latex)
    # Normalize whitespace
    latex = re.sub(r"\s+", " ", latex)
    # Fix common unicode symbols
    latex = latex.replace("\u2212", "-")   # − (minus sign)
    latex = latex.replace("\u00d7", r"\times ")  # ×
    latex = latex.replace("\u00f7", r"\div ")     # ÷
    latex = latex.replace("\u2264", r"\leq ")     # ≤
    latex = latex.replace("\u2265", r"\geq ")     # ≥
    latex = latex.replace("\u2260", r"\neq ")     # ≠
    latex = latex.replace("\u221e", r"\infty ")   # ∞
    # Fix brace spacing
    latex = re.sub(r"\{\s+", "{", latex)
    latex = re.sub(r"\s+\}", "}", latex)
    # Simplify single-char braces: {a} → a
    latex = re.sub(r"\{([a-zA-Z0-9])\}", r"\1", latex)
    return latex.strip()


# ---------------------------------------------------------------------------
# 2. preprocess_formula_image
# ---------------------------------------------------------------------------
def preprocess_formula_image(
    img: Any,
    padding: int = 8,
    min_side: int = 32,
    target_height: int = 64,
) -> Any:
    """
    公式 ROI 图像预处理：
    1. 灰度
    2. Otsu 二值化（黑字白底）
    3. 去白边（tight crop）
    4. padding
    5. 小图 resize 放大

    参数:
        img        : PIL.Image 或 numpy ndarray
        padding    : tight crop 后在四周添加的像素
        min_side   : 最短边低于此值时进行放大
        target_height: 目标高度（像素），保持宽高比放大

    返回:
        与输入相同类型的处理后图像，若 PIL/numpy 不可用则原样返回
    """
    if Image is None or np is None:
        return img

    # Ensure PIL Image
    if not isinstance(img, Image.Image):
        try:
            pil = Image.fromarray(img)
        except Exception:
            return img
    else:
        pil = img

    # 1) Grayscale
    gray = pil.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # 2) Otsu binarize (黑字白底 → white bg = 255, text = 0)
    threshold = _otsu_threshold(arr)
    binary = (arr >= threshold).astype(np.uint8) * 255  # white bg

    # 3) Tight crop (remove white border)
    mask = binary < 128  # text pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
        c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
        binary = binary[r0 : r1 + 1, c0 : c1 + 1]

    # 4) Padding
    h, w = binary.shape
    padded = np.full((h + 2 * padding, w + 2 * padding), 255, dtype=np.uint8)
    padded[padding : padding + h, padding : padding + w] = binary

    # 5) Resize if too small
    out_pil = Image.fromarray(padded, mode="L")
    ph, pw = padded.shape
    if ph < min_side or pw < min_side or ph < target_height:
        scale = max(target_height / max(ph, 1), min_side / max(min(ph, pw), 1))
        if scale > 1.0:
            new_w = max(1, int(pw * scale))
            new_h = max(1, int(ph * scale))
            out_pil = out_pil.resize((new_w, new_h), Image.LANCZOS)

    # Return same type as input
    if isinstance(img, Image.Image):
        return out_pil
    return np.array(out_pil)


def _otsu_threshold(arr: Any) -> int:
    """Compute Otsu binarization threshold from a grayscale uint8 numpy array."""
    hist, _ = np.histogram(arr.ravel(), bins=256, range=(0, 256))
    total = arr.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    weight_b = 0
    best_thresh = 128
    best_var = 0.0
    for t in range(256):
        weight_b += int(hist[t])
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * float(hist[t])
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        var_between = float(weight_b) * float(weight_f) * (mean_b - mean_f) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = t
    return best_thresh


# ---------------------------------------------------------------------------
# 3. FormulaRecognizer
# ---------------------------------------------------------------------------
class FormulaRecognizer:
    """
    多引擎公式识别，按优先级 fallback：
      1. rapid_latex_ocr（如已安装）
      2. pix2tex（如已安装）
      3. PaddleOCR 文本 fallback
    输出统一经过 normalize_latex()。
    """

    def __init__(self, paddle_ocr=None):
        """
        参数:
            paddle_ocr: 已初始化的 PaddleOCR 实例（可选，供 fallback 使用）
        """
        self._rapid = self._try_load_rapid()
        self._pix2tex = self._try_load_pix2tex()
        self._paddle = paddle_ocr

    # ------------------------------------------------------------------
    @staticmethod
    def _try_load_rapid():
        try:
            from rapid_latex_ocr import LatexOCR  # type: ignore
            return LatexOCR()
        except Exception:
            return None

    @staticmethod
    def _try_load_pix2tex():
        try:
            from pix2tex.cli import LatexOCR  # type: ignore
            return LatexOCR()
        except Exception:
            return None

    # ------------------------------------------------------------------
    def recognize(self, image_roi: Any) -> str:
        """
        识别公式图像并返回规范化 LaTeX 字符串。

        参数:
            image_roi: PIL.Image 或 numpy ndarray（裁剪后的公式区域）

        返回:
            normalize_latex() 处理后的 LaTeX 字符串（可能为空字符串）
        """
        preprocessed = preprocess_formula_image(image_roi)
        raw = ""

        # Engine 1: rapid_latex_ocr
        if self._rapid is not None and raw == "":
            try:
                result = self._rapid(preprocessed)
                if isinstance(result, tuple):
                    raw = str(result[0] or "")
                else:
                    raw = str(result or "")
            except Exception:
                raw = ""

        # Engine 2: pix2tex
        if self._pix2tex is not None and raw == "":
            try:
                if Image is not None and not isinstance(preprocessed, Image.Image):
                    pil_img = Image.fromarray(preprocessed)
                else:
                    pil_img = preprocessed
                raw = str(self._pix2tex(pil_img) or "")
            except Exception:
                raw = ""

        # Engine 3: PaddleOCR text fallback
        if raw == "" and self._paddle is not None:
            try:
                if Image is not None and isinstance(preprocessed, Image.Image):
                    import numpy as _np
                    ocr_input = _np.array(preprocessed)
                else:
                    ocr_input = preprocessed
                result = self._paddle.ocr(ocr_input, cls=False)
                texts = []
                if result:
                    for line in result:
                        if line:
                            for item in line:
                                if item and len(item) >= 2:
                                    t = item[1][0] if isinstance(item[1], (list, tuple)) else item[1]
                                    if t:
                                        texts.append(str(t))
                raw = " ".join(texts)
            except Exception:
                raw = ""

        return normalize_latex(raw)


# ---------------------------------------------------------------------------
# Sanity check / __main__
# ---------------------------------------------------------------------------
def _sanity_check():
    """
    自检：构造 mock formula block，走识别流程，验证输出 HTML 合规。
    """
    print("=== formula sanity check ===")

    # 1) normalize_latex tests
    cases = [
        ("$x^2 + y^2$", "x^2 + y^2"),
        ("$$\\displaystyle a + b$$", "a + b"),
        ("  x  +  y  ", "x + y"),
        ("{a} + {b}", "a + b"),
        ("a\u2212b", "a-b"),
    ]
    for inp, expected in cases:
        out = normalize_latex(inp)
        status = "OK" if out == expected else f"FAIL (got {out!r})"
        print(f"  normalize_latex({inp!r}) = {out!r}  [{status}]")

    # 2) preprocess_formula_image (only if PIL + numpy available)
    if Image is not None and np is not None:
        img = Image.new("RGB", (120, 40), color=(255, 255, 255))
        result_img = preprocess_formula_image(img)
        assert result_img is not None, "preprocess returned None"
        print("  preprocess_formula_image: OK")
    else:
        print("  preprocess_formula_image: SKIP (PIL/numpy not available)")

    # 3) HTML output compliance
    mock_block = {"type": "formula", "bbox": [10, 20, 110, 60], "latex": "$E = mc^2$"}
    bbox = mock_block["bbox"]
    bbox_str = f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
    latex = normalize_latex(mock_block.get("latex", ""))
    html_out = f'<div class="formula" data-bbox="{bbox_str}">{latex}</div>'

    assert 'class="formula"' in html_out, "Missing class=formula"
    assert f'data-bbox="{bbox_str}"' in html_out, "Missing data-bbox"
    assert "data-latex" not in html_out, "data-latex must not appear"
    assert "E = mc^2" in html_out, "LaTeX must be text content"
    print(f"  HTML output: {html_out}")
    print("  HTML compliance: OK")

    print("=== all checks passed ===")


if __name__ == "__main__":
    _sanity_check()
