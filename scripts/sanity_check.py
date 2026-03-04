#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/sanity_check.py
=======================
Minimal end-to-end sanity check for the OCR inference pipeline.

Usage
-----
    python scripts/sanity_check.py \
        --config ./trainer/config/config.yaml \
        --images path/to/image1.png path/to/image2.jpg \
        --output /tmp/sanity_output.jsonl

The script:
1. Loads the model bundle (layout detector, OCR, LightGBM).
2. Runs ``process_one`` for each supplied image.
3. Prints per-stage timing and a summary.
4. Writes the results to a JSONL file so you can inspect them.

Exit code is 0 on success, 1 if any image produced an empty answer.
"""

import argparse
import json
import os
import sys
import time

# Make sure the repo root is importable regardless of working directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eval import load_cfg, load_artifacts, process_one, write_jsonl  # noqa: E402


def _fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end sanity check for OCR pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    parser.add_argument(
        "--images", nargs="+", required=True,
        help="One or more image file paths to process",
    )
    parser.add_argument(
        "--output", default="/tmp/sanity_check_output.jsonl",
        help="Destination JSONL file for results (default: /tmp/sanity_check_output.jsonl)",
    )
    parser.add_argument("--prompt", default="", help="Optional prompt string to attach to each sample")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config + models
    # ------------------------------------------------------------------
    print(f"[sanity] Loading config from: {args.config}")
    t_cfg = time.time()
    cfg = load_cfg(args.config)
    print(f"[sanity]   config loaded in {time.time() - t_cfg:.2f}s")

    print("[sanity] Loading model artifacts …")
    t_models = time.time()
    models = load_artifacts(cfg)
    print(f"[sanity]   artifacts loaded in {time.time() - t_models:.2f}s")

    # ------------------------------------------------------------------
    # Per-image inference
    # ------------------------------------------------------------------
    timing_keys = ["build_ms", "enrich_ms", "block_ms", "rel_ms", "decode_ms", "render_ms", "total_ms"]
    aggregated: dict = {k: 0.0 for k in timing_keys}

    results = []
    any_empty = False

    for img_path in args.images:
        img_name = os.path.basename(img_path)
        image_root = os.path.dirname(os.path.abspath(img_path))

        sample = {
            "image": img_name,
            "prompt": args.prompt,
        }

        print(f"\n[sanity] Processing: {img_path}")
        t0 = time.time()
        result = process_one(sample, cfg, models, image_root=image_root)
        elapsed = time.time() - t0

        dbg = result.get("debug", {}) or {}
        answer = result.get("answer", "")

        # Print per-stage timing
        print(f"  Total wall time : {elapsed:.3f}s")
        for key in timing_keys:
            val = dbg.get(key, 0.0)
            aggregated[key] += float(val)
            label = key.replace("_ms", "")
            print(f"    {label:12s}: {_fmt_ms(float(val))}")

        # Print block/table counts
        print(f"  blocks_count    : {dbg.get('blocks_count', '?')}")
        print(f"  tables_count    : {dbg.get('tables_count', '?')}")
        print(f"  blocks_source   : {dbg.get('blocks_source', '?')}")
        print(f"  fallback        : {dbg.get('fallback_triggered', False)}")

        # Check answer
        is_empty = not answer or answer.strip() in ("<body></body>", "<body> </body>")
        if is_empty:
            print(f"  ⚠  answer is EMPTY for {img_name}")
            any_empty = True
        else:
            preview = answer[:120].replace("\n", " ")
            print(f"  answer preview  : {preview}…")

        results.append(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n = max(1, len(args.images))
    print("\n[sanity] ===== Summary =====")
    print(f"  Images processed: {n}")
    for key in timing_keys:
        label = key.replace("_ms", "")
        avg = aggregated[key] / n
        print(f"  avg {label:12s}: {_fmt_ms(avg)}")

    # ------------------------------------------------------------------
    # Write JSONL
    # ------------------------------------------------------------------
    records = [
        {"image": r.get("image"), "prompt": r.get("prompt"), "answer": r.get("answer")}
        for r in results
    ]
    write_jsonl(records, args.output)
    print(f"\n[sanity] Results written to: {args.output}")
    print(f"[sanity] First 3 lines:")
    with open(args.output, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= 3:
                break
            print(f"  {line.rstrip()[:200]}")

    if any_empty:
        print("\n[sanity] ❌ One or more images produced an empty answer.")
        return 1

    print("\n[sanity] ✅ All images produced non-empty answers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
