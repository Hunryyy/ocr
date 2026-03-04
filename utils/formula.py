"""
Formula recognition utilities.

Provides:
- preprocess_formula_image: image preprocessing for formula ROIs
- normalize_latex: LaTeX string normalization
- FormulaRecognizer: multi-engine recognizer with fallback chain
  (RapidLatexOCR -> pix2tex -> PaddleOCR)
"""
import re
from typing import Any, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None

# Optional high-quality formula OCR engines
try:
    from rapid_latex_ocr import LatexOCR as _RapidLatexOCR
except Exception:  # pragma: no cover
    _RapidLatexOCR = None

try:
    from pix2tex.cli import LatexOCR as _Pix2TexOCR
except Exception:  # pragma: no cover
    _Pix2TexOCR = None


def preprocess_formula_image(roi: "np.ndarray") -> "np.ndarray":
    """
    Preprocess a formula image ROI for better OCR accuracy.

    Steps:
    1. Convert to grayscale
    2. Otsu binarization
    3. Trim surrounding white border
    4. Add padding
    5. Upscale small formulas (shortest side < 40 px -> 2x)
    """
    if np is None or PILImage is None:
        return roi

    # Ensure we work on a copy
    img = roi.copy()

    # 1) Convert to grayscale
    if img.ndim == 3:
        pil = PILImage.fromarray(img)
        pil = pil.convert("L")
        img = np.array(pil)

    # 2) Otsu binarization
    try:
        import cv2  # type: ignore
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except ImportError:
        # Fallback: simple mean threshold
        thresh = int(img.mean())
        img = (img > thresh).astype(np.uint8) * 255

    # 3) Trim white border (rows/cols that are all white)
    mask = img < 255  # non-white pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        img = img[rmin : rmax + 1, cmin : cmax + 1]

    # 4) Add padding (8 px on each side)
    pad = 8
    h, w = img.shape[:2]
    padded = np.full((h + 2 * pad, w + 2 * pad), 255, dtype=np.uint8)
    padded[pad : pad + h, pad : pad + w] = img
    img = padded

    # 5) Upscale small formulas
    h, w = img.shape[:2]
    if min(h, w) < 40:
        pil = PILImage.fromarray(img)
        pil = pil.resize((w * 2, h * 2), PILImage.LANCZOS)
        img = np.array(pil)

    return img


def normalize_latex(latex: str) -> str:
    """
    Normalize a LaTeX string for consistent output and better BLEU scores.

    - Strip leading/trailing whitespace
    - Remove wrapping $ ... $ or $$ ... $$
    - Remove leading \\displaystyle
    - Collapse internal whitespace runs to single spaces
    """
    latex = latex.strip()

    # Remove wrapping $$ ... $$
    if latex.startswith("$$") and latex.endswith("$$") and len(latex) > 4:
        latex = latex[2:-2].strip()

    # Remove wrapping $ ... $
    if latex.startswith("$") and latex.endswith("$") and len(latex) > 2:
        latex = latex[1:-1].strip()

    # Remove leading \displaystyle
    latex = re.sub(r"^\\displaystyle\s*", "", latex)

    # Collapse whitespace
    latex = re.sub(r"\s+", " ", latex).strip()

    return latex


class FormulaRecognizer:
    """
    Multi-engine formula recognizer with fallback chain:
      1. RapidLatexOCR (if available)
      2. pix2tex (if available)
      3. PaddleOCR (plain-text fallback, low accuracy)

    Parameters
    ----------
    ocr_engine : optional PaddleOCR instance for the fallback engine.
    cfg : optional config dict; may contain ``formula_engine`` section.
    """

    def __init__(self, ocr_engine: Any = None, cfg: Optional[dict] = None):
        self._ocr_engine = ocr_engine  # PaddleOCR instance (fallback)
        cfg = cfg or {}
        fe_cfg = cfg.get("formula_engine", {}) or {}

        self._rapid = None
        self._pix2tex = None

        # Initialize RapidLatexOCR if available and not disabled
        if _RapidLatexOCR is not None and fe_cfg.get("rapid_latex_ocr", True):
            try:
                self._rapid = _RapidLatexOCR()
            except Exception:
                self._rapid = None

        # Initialize pix2tex if available and not disabled
        if _Pix2TexOCR is not None and fe_cfg.get("pix2tex", True):
            try:
                self._pix2tex = _Pix2TexOCR()
            except Exception:
                self._pix2tex = None

    def recognize(self, image_roi: "np.ndarray") -> str:
        """
        Recognize LaTeX from *image_roi* (numpy array, BGR or grayscale).

        Returns a normalized LaTeX string, or empty string on failure.
        """
        if np is None:
            return ""

        processed = preprocess_formula_image(image_roi)

        # -- Engine 1: RapidLatexOCR --
        if self._rapid is not None:
            try:
                result = self._rapid(processed)
                # API returns (latex_str, elapsed) or just latex_str
                if isinstance(result, (tuple, list)):
                    latex = str(result[0])
                else:
                    latex = str(result)
                latex = normalize_latex(latex)
                if latex:
                    return latex
            except Exception:
                pass

        # -- Engine 2: pix2tex --
        if self._pix2tex is not None and PILImage is not None:
            try:
                pil_img = PILImage.fromarray(processed)
                latex = self._pix2tex(pil_img)
                latex = normalize_latex(str(latex))
                if latex:
                    return latex
            except Exception:
                pass

        # -- Engine 3: PaddleOCR fallback --
        if self._ocr_engine is not None and PILImage is not None:
            try:
                pil_img = PILImage.fromarray(processed)
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                pil_img.save(tmp_path)
                try:
                    result = self._ocr_engine.ocr(tmp_path, cls=False)
                    texts = []
                    if result:
                        for page_result in result:
                            if not page_result:
                                continue
                            for line in page_result:
                                if isinstance(line, (list, tuple)) and len(line) >= 2:
                                    txt_info = line[1]
                                    if isinstance(txt_info, (list, tuple)):
                                        texts.append(str(txt_info[0]))
                                    else:
                                        texts.append(str(txt_info))
                    latex = normalize_latex(" ".join(texts))
                    return latex
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            except Exception:
                pass

        return ""
