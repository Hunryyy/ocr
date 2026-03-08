"""
Unit tests for DocLayout-YOLO integration: output parsing, class-aware NMS,
header/footer post-correction, and no-model fallback.

Run from the project root with:
    python -m unittest tests.test_layout_detector
"""

import sys
import os
import unittest

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from eval import (
    _parse_yolo_output,
    _nms_python,
    _postprocess_header_footer,
    _run_layout_detector,
    _looks_like_page_number_text,
    _promote_page_number_blocks,
    _beam_search_order,
    ModelBundle,
)


@unittest.skipUnless(_HAS_NUMPY, "numpy not installed")
class TestParseYoloOutput(unittest.TestCase):
    """Tests for _parse_yolo_output shape compatibility."""

    def _make_det(self, cx, cy, w, h, obj, cls_scores):
        """Helper to build a raw detection row with objectness."""
        return [cx, cy, w, h, obj] + list(cls_scores)

    def test_shape_1_N_5plusC(self):
        """(1, N, 5+C) with objectness * cls_score."""
        arr = np.zeros((1, 3, 7), dtype=np.float32)
        arr[0, 0] = [100, 100, 50, 50, 0.9, 0.8, 0.1]   # score=0.72 -> cls 0
        arr[0, 1] = [200, 200, 50, 50, 0.1, 0.1, 0.9]   # score=0.09 -> filtered
        arr[0, 2] = [300, 300, 50, 50, 0.8, 0.1, 0.9]   # score=0.72 -> cls 1
        dets = _parse_yolo_output(arr, 2, 0.25)
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0]["label_idx"], 0)
        self.assertEqual(dets[1]["label_idx"], 1)
        self.assertEqual(dets[0]["format"], "xywh")

    def test_shape_N_4plusC_anchor_free(self):
        """(N, 4+C) without objectness - anchor-free style."""
        arr = np.zeros((50, 6), dtype=np.float32)
        arr[0] = [500, 400, 100, 80, 0.9, 0.1]   # cls_scores=[0.9,0.1] -> idx=0
        arr[1] = [200, 300, 80, 60, 0.1, 0.85]   # cls_scores=[0.1,0.85] -> idx=1
        dets = _parse_yolo_output(arr, 2, 0.25)
        self.assertEqual(len(dets), 2)
        self.assertEqual(dets[0]["label_idx"], 0)
        self.assertEqual(dets[1]["label_idx"], 1)

    def test_shape_N_6_compact(self):
        """(N, 6) compact: x,y,w,h,score,cls_id."""
        arr = np.array([
            [100, 200, 50, 40, 0.9, 2],
            [300, 400, 60, 50, 0.1, 0],   # filtered
        ], dtype=np.float32)
        dets = _parse_yolo_output(arr, 5, 0.25)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["label_idx"], 2)
        self.assertAlmostEqual(dets[0]["score"], 0.9, places=4)

    def test_shape_transposed_1_C_N(self):
        """(1, C, N) transposed export - should auto-detect and transpose."""
        # Normal shape (1, 100, 7), then transpose to (1, 7, 100)
        arr_norm = np.zeros((1, 100, 7), dtype=np.float32)
        arr_norm[0, 0] = [100, 100, 50, 50, 0.9, 0.8, 0.1]
        arr_norm[0, 1] = [300, 300, 50, 50, 0.8, 0.1, 0.9]
        arr_transposed = arr_norm.transpose(0, 2, 1)  # (1, 7, 100)
        dets = _parse_yolo_output(arr_transposed, 2, 0.25)
        self.assertEqual(len(dets), 2)

    def test_bbox_format_xyxy_forced(self):
        """bbox_format='xyxy' forces xyxy interpretation."""
        # Use num_classes=2 so 6 columns don't match the (5+C) branch
        arr = np.array([[10, 20, 60, 80, 0.9, 0]], dtype=np.float32)
        dets = _parse_yolo_output(arr, 2, 0.25, bbox_format="xyxy")
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["format"], "xyxy")

    def test_score_threshold_filters(self):
        """Detections below score_threshold must be dropped."""
        arr = np.array([
            [100, 100, 50, 50, 0.9, 0.3, 0.1],  # score=0.27 -> filtered at 0.3
            [200, 200, 50, 50, 0.9, 0.5, 0.1],  # score=0.45 -> kept
        ], dtype=np.float32)
        dets = _parse_yolo_output(arr, 2, 0.3)
        self.assertEqual(len(dets), 1)

    def test_empty_output(self):
        """Empty array returns empty list."""
        arr = np.zeros((1, 0, 7), dtype=np.float32)
        dets = _parse_yolo_output(arr, 2, 0.25)
        self.assertEqual(dets, [])

    def test_wrong_ndim_returns_empty(self):
        """4-D array not handled - should return []."""
        arr = np.zeros((1, 2, 3, 4), dtype=np.float32)
        dets = _parse_yolo_output(arr, 2, 0.25)
        self.assertEqual(dets, [])


class TestNMSClassAware(unittest.TestCase):
    """Tests for class-aware NMS via _nms_python."""

    def test_same_class_suppressed(self):
        """Highly-overlapping same-class boxes -> only best kept."""
        dets = [
            {"bbox": [0, 0, 100, 100], "label": "figure", "score": 0.9},
            {"bbox": [5, 5, 95, 95],   "label": "figure", "score": 0.7},
        ]
        out = _nms_python(dets)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0]["score"], 0.9)

    def test_different_class_both_kept(self):
        """Overlapping boxes of different classes -> both kept."""
        dets = [
            {"bbox": [0, 0, 100, 100], "label": "figure",  "score": 0.9},
            {"bbox": [0, 0, 100, 100], "label": "caption", "score": 0.8},
        ]
        out = _nms_python(dets)
        self.assertEqual(len(out), 2)

    def test_class_agnostic_with_threshold(self):
        """Explicit iou_threshold -> class-agnostic NMS (old behaviour)."""
        dets = [
            {"bbox": [0, 0, 100, 100], "label": "figure",  "score": 0.9},
            {"bbox": [0, 0, 100, 100], "label": "caption", "score": 0.8},
        ]
        out = _nms_python(dets, iou_threshold=0.5)
        self.assertEqual(len(out), 1)  # agnostic: overlapping -> suppressed

    def test_empty_returns_empty(self):
        self.assertEqual(_nms_python([]), [])


class TestPostprocessHeaderFooter(unittest.TestCase):
    """Tests for header/footer position-based post-correction."""

    def test_top_block_becomes_header(self):
        """Wide block at top of page -> header."""
        blocks = [{"bbox": [0, 0, 800, 30], "label": "paragraph", "score": 0.9}]
        out = _postprocess_header_footer(blocks, 1000, 800, 0.08, 0.08, 0.5)
        self.assertEqual(out[0]["label"], "header")

    def test_bottom_block_becomes_footer(self):
        """Wide block at bottom of page -> footer."""
        blocks = [{"bbox": [0, 970, 800, 1000], "label": "paragraph", "score": 0.9}]
        out = _postprocess_header_footer(blocks, 1000, 800, 0.08, 0.08, 0.5)
        self.assertEqual(out[0]["label"], "footer")

    def test_middle_block_unchanged(self):
        """Block in page middle -> unchanged."""
        blocks = [{"bbox": [100, 400, 700, 450], "label": "paragraph", "score": 0.7}]
        out = _postprocess_header_footer(blocks, 1000, 800, 0.08, 0.08, 0.5)
        self.assertEqual(out[0]["label"], "paragraph")

    def test_narrow_top_block_unchanged(self):
        """Narrow block at top (< width_ratio) -> not promoted to header."""
        blocks = [{"bbox": [0, 0, 200, 30], "label": "paragraph", "score": 0.6}]
        out = _postprocess_header_footer(blocks, 1000, 800, 0.08, 0.08, 0.5)
        self.assertEqual(out[0]["label"], "paragraph")

    def test_caption_promoted(self):
        """caption is also eligible for correction."""
        blocks = [{"bbox": [0, 0, 800, 20], "label": "caption", "score": 0.8}]
        out = _postprocess_header_footer(blocks, 1000, 800, 0.08, 0.08, 0.5)
        self.assertEqual(out[0]["label"], "header")

    def test_table_not_promoted(self):
        """table label is not eligible."""
        blocks = [{"bbox": [0, 0, 800, 30], "label": "table", "score": 0.9}]
        out = _postprocess_header_footer(blocks, 1000, 800, 0.08, 0.08, 0.5)
        self.assertEqual(out[0]["label"], "table")

    def test_empty_returns_empty(self):
        self.assertEqual(_postprocess_header_footer([], 1000, 800, 0.08, 0.08, 0.5), [])


class TestRunLayoutDetectorNoModel(unittest.TestCase):
    """Tests for _run_layout_detector fallback when no model is loaded."""

    def test_no_model_returns_empty(self):
        bundle = ModelBundle()
        debug = {}
        result = _run_layout_detector("/nonexistent/image.png", {}, bundle, debug)
        self.assertEqual(result, [])
        self.assertEqual(debug.get("layout_detector_status"), "no_model")

    def test_missing_image_returns_empty(self):
        """With a model stub but missing image, should get a non-ok status."""
        bundle = ModelBundle()
        bundle.layout_detector = object()  # non-None stub
        debug = {}
        result = _run_layout_detector("/nonexistent/image.png", {}, bundle, debug)
        self.assertEqual(result, [])
        status = debug.get("layout_detector_status", "")
        # Could be "image_error" (PIL available) or "missing_deps" (PIL absent)
        self.assertNotEqual(status, "ok")


class TestPageNumberHeuristics(unittest.TestCase):
    def test_page_number_regex(self):
        self.assertTrue(_looks_like_page_number_text("-2-"))
        self.assertTrue(_looks_like_page_number_text("第12页"))
        self.assertTrue(_looks_like_page_number_text("Page 3"))
        self.assertFalse(_looks_like_page_number_text("这是正文段落，不是页码"))

    def test_promote_page_number_blocks(self):
        ir = {
            "page": {"width": 1000, "height": 1400},
            "blocks": [
                {"id": 0, "type": "footer", "bbox": [460, 1320, 540, 1360], "text": "-2-"},
                {"id": 1, "type": "paragraph", "bbox": [80, 200, 920, 320], "text": "正文内容"},
            ],
            "debug": {},
        }
        out = _promote_page_number_blocks(ir)
        self.assertEqual(out["blocks"][0]["type"], "page_number")
        self.assertEqual(out["blocks"][1]["type"], "paragraph")
        self.assertEqual(out["debug"].get("page_number_promoted"), 1)


class TestBeamSearchOrdering(unittest.TestCase):
    def test_starts_from_top_left(self):
        blocks = [
            {"id": 0, "bbox": [10, 20, 100, 40], "type": "paragraph"},
            {"id": 1, "bbox": [10, 900, 100, 940], "type": "footer"},
        ]
        order = _beam_search_order(blocks, {0: [], 1: []}, {}, beam_width=2)
        self.assertEqual(order[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
