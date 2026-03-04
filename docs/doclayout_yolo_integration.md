# DocLayout-YOLO ONNX Integration Guide

## Overview

This guide explains how to export [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
to ONNX and plug it into the layout detection pipeline in `eval.py`.

---

## 1. Export DocLayout-YOLO to ONNX

### Requirements
- DocLayout-YOLO trained weights (`.pt` file, **≤150 MB**)
- `ultralytics` package installed

### Recommended export command

```bash
yolo export \
  model=doclayout_yolo.pt \
  format=onnx \
  imgsz=1024 \
  opset=12 \
  dynamic=False
```

This produces a `doclayout_yolo.onnx` file with a **single output tensor**
of shape `[1, N, 4 + num_classes]` (YOLOv8-style), where each row is:

```
cx, cy, w, h, score_cls0, score_cls1, ..., score_clsN-1
```

Coordinates are in **pixel space** relative to the letterboxed input
(input_size × input_size).  The existing `_parse_yolo_output` function
handles this format directly.

> **Do NOT commit `.onnx` or `.pt` weight files to git.**
> Place them in a local `models/` directory and reference them in config.

---

## 2. Class map file

Copy and edit `models/layout_class_map.doclayout_yolo.json` to match your
model's class ordering:

```jsonc
{
  "0": "title",
  "1": "paragraph",
  "2": "figure",
  "3": "figure_caption",   // automatically normalized to "caption"
  "4": "table",
  "5": "table_caption",    // automatically normalized to "caption"
  "6": "header",
  "7": "footer",
  "8": "formula",
  "9": "list_item",
  "10": "chart"
}
```

### Automatic synonym normalization

The pipeline automatically maps these raw class names to canonical labels:

| Raw name        | Canonical label |
|-----------------|-----------------|
| `text`          | `paragraph`     |
| `equation`      | `formula`       |
| `math`          | `formula`       |
| `image`         | `figure`        |
| `picture`       | `figure`        |
| `figure_caption`| `caption`       |
| `table_caption` | `caption`       |

---

## 3. Config (`trainer/config/config.yaml`)

```yaml
models:
  fallback_models:
    layout_detector:
      enabled: true
      type: "yolo"
      path: "models/doclayout_yolo.onnx"          # path to ONNX weights
      class_map: "models/layout_class_map.doclayout_yolo.json"  # or inline dict
      input_size: 1024                             # letterbox size
      score_threshold: 0.25                        # min detection confidence
      nms_mode: "class_aware"                      # class_aware | agnostic
      nms_threshold: 0.5                           # used only when nms_mode=agnostic
      hf_refine_enabled: true                      # header/footer heuristic
      hf_top_ratio: 0.08                           # top 8% of page = potential header
      hf_bottom_ratio: 0.08                        # bottom 8% = potential footer
      hf_width_ratio: 0.6                          # must span ≥60% of page width
      second_pass_enabled: false                   # set true for accuracy-first
      second_pass_input_size: 1280
      second_pass_score_threshold: 0.2
      second_pass_min_detections: 2
```

---

## 4. NMS behaviour

| `nms_mode`    | Behaviour |
|---------------|-----------|
| `class_aware` | *(default)* NMS is applied **per class** using per-class IoU thresholds (e.g. `table`/`figure` at 0.3, `paragraph` at 0.5). This prevents a figure from suppressing a nearby caption or formula. |
| `agnostic`    | All classes compete together — legacy behaviour. |

---

## 5. Header/footer post-hoc refinement

When `hf_refine_enabled: true`, after NMS a lightweight heuristic relabels
text-like blocks:

- **Header**: block's bottom edge is within the top `hf_top_ratio` of the page
  *and* its width is ≥ `hf_width_ratio` of the page width.
- **Footer**: block's top edge is within the bottom `hf_bottom_ratio` of the page
  *and* its width is ≥ `hf_width_ratio` of the page width.

Non-text blocks (`table`, `figure`, `chart`, `formula`) are never relabelled.

---

## 6. Second-pass strategy

When `second_pass_enabled: true`, the pipeline checks after the first pass
whether the page looks difficult (fewer than `second_pass_min_detections`
blocks, or average confidence below 0.35).  If so, it retries at
`second_pass_input_size` (e.g. 1280) with a lower score threshold
(`second_pass_score_threshold`), and keeps whichever pass produced more
detections.

This keeps average latency low while recovering accuracy on hard pages.

---

## 7. GPU acceleration

`onnxruntime` automatically uses CUDA when available.  No code changes needed;
the model is loaded with `CUDAExecutionProvider` as the first provider if
present in your `onnxruntime-gpu` installation.
