#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/smoke_layout_detector.py
=================================
Minimal smoke test for the DocLayout-YOLO layout detector integration.

Usage
-----
    python scripts/smoke_layout_detector.py \
        --config ./trainer/config/config.yaml \
        --image  path/to/sample_page.png

If no ONNX path is configured in the config the script skips the model test
and exits 0, so it is safe to run in CI without a model file present.

The script exercises:
  - Config loading and ModelBundle construction
  - _run_layout_detector (including class-aware NMS and debug fields)
  - No-model fallback behaviour (layout_detector_status == "no_model")
"""

import argparse
import json
import os
import sys
import tempfile

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval import (  # noqa: E402
    load_cfg,
    load_artifacts,
    _run_layout_detector,
    ModelBundle,
    DEFAULT_LAYOUT_CLASSES,
)


def _banner(msg: str) -> None:
    print(f"[smoke] {msg}")


def _run_no_model_check() -> None:
    """Verify that missing model returns empty list and correct status."""
    bundle = ModelBundle()
    debug: dict = {}
    result = _run_layout_detector("/nonexistent/path.png", {}, bundle, debug)
    assert result == [], f"Expected [] for no-model, got {result}"
    assert debug.get("layout_detector_status") == "no_model", (
        f"Expected 'no_model', got {debug.get('layout_detector_status')}"
    )
    _banner("No-model fallback: OK")


def _run_model_check(cfg_path: str, image_path: str) -> None:
    cfg = load_cfg(cfg_path)
    bundle = load_artifacts(cfg)

    if bundle.layout_detector is None:
        _banner("No ONNX path configured or model file not found – skipping model test")
        return

    _banner("ONNX session loaded OK")

    try:
        from PIL import Image as _Image
        img = _Image.open(image_path)
        _banner(f"Image size: {img.size[0]} x {img.size[1]}")
    except Exception as e:
        _banner(f"Cannot open image: {e}")
        sys.exit(1)

    debug: dict = {}
    results = _run_layout_detector(image_path, cfg, bundle, debug)

    _banner(f"Pass-1 raw detections : {debug.get('layout_pass1_dets', 'n/a')}")
    _banner(f"After NMS              : {debug.get('layout_detections', len(results))}")
    _banner(f"layout_detector_status : {debug.get('layout_detector_status')}")
    _banner(f"layout_nms_class_aware : {debug.get('layout_nms_class_aware')}")
    if debug.get("layout_second_pass"):
        _banner(f"Second pass triggered  : pass2_dets={debug.get('layout_pass2_dets', 'n/a')}")

    _banner("Detections:")
    for i, det in enumerate(results):
        bbox = [round(v, 1) for v in det["bbox"]]
        print(f"  [{i}] {det['label']:<12} score={det['score']:.2f}  bbox={bbox}")

    assert debug.get("layout_detector_status") == "ok", (
        f"Expected status='ok', got {debug.get('layout_detector_status')}"
    )
    _banner("Model smoke test: PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test for layout detector")
    parser.add_argument("--config", default=None,
                        help="Path to config YAML/JSON (optional; uses defaults if omitted)")
    parser.add_argument("--image", default=None,
                        help="Path to a sample image (required if --config is given)")
    args = parser.parse_args()

    # Always run no-model check
    _run_no_model_check()

    if args.config:
        if not args.image:
            print("[smoke] --image is required when --config is provided", file=sys.stderr)
            return 1
        if not os.path.exists(args.image):
            print(f"[smoke] Image not found: {args.image}", file=sys.stderr)
            return 1
        _run_model_check(args.config, args.image)
    else:
        _banner("No --config provided; only no-model fallback was tested")

    return 0


if __name__ == "__main__":
    sys.exit(main())
