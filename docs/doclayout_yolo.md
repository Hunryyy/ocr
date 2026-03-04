# DocLayout-YOLO Integration Guide

This guide explains how to plug in a **DocLayout-YOLO** ONNX model as the
layout detector for the OCR pipeline, replacing the default PP-Structure
fallback with higher-accuracy document layout detection.

---

## 1. Obtain the ONNX weight

DocLayout-YOLO is published at <https://github.com/opendatalab/DocLayout-YOLO>.

### Option A – download a pre-exported ONNX (recommended)

Check the project releases or Hugging Face Hub for a ready-made `.onnx` file
(`doclayout_yolo_docstructbench_imgsz1024.onnx`, ≤ 150 MB).

### Option B – export from PyTorch yourself

```bash
pip install doclayout-yolo          # or clone the repo
python - <<'EOF'
from doclayout_yolo import YOLOv10
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")
model.export(format="onnx", imgsz=1024, simplify=True, opset=17)
EOF
```

Place the resulting `.onnx` file somewhere accessible, e.g. `./models/`.

---

## 2. Prepare the class map

The model outputs raw integer class indices.  You must tell the pipeline how
to map them to the supported label set.  A ready-made example is provided at
`trainer/config/layout_class_map.json`.

Typical DocLayout-YOLO class order (DocStructBench checkpoint):

| index | original name   | pipeline label |
|-------|-----------------|---------------|
| 0     | caption         | caption       |
| 1     | footnote        | footer        |
| 2     | formula         | formula       |
| 3     | list-item       | paragraph     |
| 4     | page-footer     | footer        |
| 5     | page-header     | header        |
| 6     | picture         | figure        |
| 7     | section-heading | title         |
| 8     | table           | table         |
| 9     | table-caption   | caption       |
| 10    | table-footnote  | caption       |
| 11    | text            | paragraph     |

Edit `trainer/config/layout_class_map.json` to match the exact class order of
your export, then reference it in the config.

---

## 3. Update `trainer/config/config.yaml`

Uncomment and fill in the `layout_detector` block:

```yaml
fallback_models:
  layout_detector:
    enabled: true
    path: "./models/doclayout_yolo_docstructbench_imgsz1024.onnx"
    class_map: "./trainer/config/layout_class_map.json"
    input_size: 1024          # match the export imgsz
    score_threshold: 0.25
    nms_threshold: 0.5
    nms_class_aware: true     # class-aware NMS (recommended)
    bbox_format: "auto"       # or "xywh" / "xyxy" to force
    hf_correction: false      # set true in accurate profile
    second_pass: false        # set true in accurate profile
```

### Profile recommendations

| Setting                  | fast     | balanced | accurate |
|--------------------------|----------|----------|----------|
| `input_size`             | 640      | 1024     | 1024     |
| `score_threshold`        | 0.30     | 0.25     | 0.20     |
| `nms_class_aware`        | true     | true     | true     |
| `hf_correction`          | false    | false    | true     |
| `second_pass`            | false    | false    | true     |
| `second_pass_input_size` | —        | —        | 1280     |
| `second_pass_score_thr`  | —        | —        | 0.15     |

Switch profiles with `profile: "accurate"` at the top of `config.yaml`.

---

## 4. Verify the setup

Run the smoke test:

```bash
python scripts/smoke_layout_detector.py \
    --config ./trainer/config/config.yaml \
    --image  path/to/sample_page.png
```

Expected output (model found):

```
[smoke] ONNX session loaded OK
[smoke] Image size: 2480 x 3508
[smoke] Pass-1 raw detections : 12
[smoke] After NMS              : 10
[smoke] layout_detector_status : ok
[smoke] layout_nms_class_aware : True
[smoke] Detections:
  [0] paragraph  score=0.87  bbox=[112, 210, 1380, 260]
  ...
```

If no ONNX path is configured the script prints `[smoke] No ONNX path
configured – skipping model test` and exits 0 (safe for CI).

---

## 5. How it works (implementation notes)

### Output shape compatibility (`_parse_yolo_output`)

The parser handles all common YOLO ONNX output layouts:

| Shape               | Format                                  |
|---------------------|-----------------------------------------|
| `(1, N, 4+1+C)`     | cx cy w h · objectness · class_scores  |
| `(1, N, 4+C)`       | cx cy w h · class_scores               |
| `(1, N, 6)`         | cx cy w h · score · class_id (compact) |
| `(N, *)`            | same without batch dim                  |
| `(1, 4+C, N)`       | transposed (auto-detected & fixed)      |

Set `bbox_format: "xyxy"` if your export uses corner coordinates instead of
centre + width/height.

### Class-aware NMS

When `nms_class_aware: true` (default), detections are grouped by class before
NMS.  This prevents large `figure`/`table` boxes from suppressing small
`caption`/`formula` boxes that don't overlap spatially with same-class boxes.
Per-class IoU thresholds are defined in `CLASS_NMS_THRESHOLDS` in `eval.py`.

### Header/footer post-correction

When `hf_correction: true`, any `paragraph` or `caption` block that
• spans ≥ 50 % of page width **and**
• sits in the top 8 % (header) or bottom 8 % (footer) of the page
is re-labelled `header` / `footer` respectively.  Thresholds are configurable
via `hf_top_ratio`, `hf_bottom_ratio`, `hf_width_ratio`.

### Second-pass inference

When `second_pass: true` and the first-pass result looks suspicious (too few
detections, no non-text blocks, or low average confidence), the page is
re-processed at a higher resolution and/or lower score threshold.  The
diagnostic fields `layout_second_pass`, `layout_pass1_dets`, and
`layout_pass2_dets` are written to the `debug` dict for every page.
