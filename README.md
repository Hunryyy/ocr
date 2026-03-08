# OCR

A modular multimodal OCR / document parsing repository for structured documents, page layout analysis, HTML rendering, and lightweight evaluation.

This project is **not positioned as a closed-box end-to-end VLM parser**. Instead, it is built as a **debuggable, locally reproducible OCR engineering stack** with explicit components for:

- layout detection
- OCR text recognition
- table HTML rendering
- formula-region handling hooks
- reading-order / relation decoding
- proxy scoring and sanity checks

The current repository has already been pushed with code, configs, docs, and datasets, while excluding machine-specific environment artifacts such as `.venv/`, `cache/`, and `tmp/`.

---

## Highlights

- **Modular pipeline** rather than monolithic black-box parsing
- **Layout-first parsing** with PaddleX / PaddleOCR based fallback chain
- **Table HTML rendering tests** and layout detector tests included
- **Proxy scorer** (`scripts/score_proxy.py`) for fast iteration before official evaluation
- **End-to-end sanity check** script for real image inference
- **Datasets included** in the repository for local debugging / reproduction

---

## Positioning

This README is written in a style inspired by repositories such as **MonkeyOCR** and **dots.ocr**, but the project scope is different.

### Where this repo sits

| Project | Main style | Core idea | Strength | Current repo relation |
|---|---|---|---|---|
| MonkeyOCR | end-to-end LMM / document parser | SRR-style document parsing with strong benchmark presentation | strong document parsing benchmarks and model packaging | a reference for presentation style and benchmark framing |
| dots.ocr | single VLM-style multilingual parser | benchmark-heavy universal parsing with strong leaderboard emphasis | multilingual parsing and broad benchmark coverage | a reference for README structure and comparison framing |
| **this repo** | modular OCR engineering stack | inspectable layout + OCR + decoding + rendering pipeline | local reproducibility, debugging, controllable evaluation, code-level iteration | **not an apples-to-apples SOTA claim** |

### Important note

This repository **does not claim SOTA against MonkeyOCR or dots.ocr**.

Those projects are benchmark-centric model releases; this repository is currently a **work-in-progress engineering pipeline** focused on:

1. getting the full parsing stack to run locally,
2. improving controllability and observability,
3. keeping evaluation scripts and unit tests inside the repo,
4. iterating toward stronger document parsing quality.

---

## Internal Evaluation Snapshot

Below are **real internal measurements currently available in this repo**.

### 1) Unit tests

```text
34 tests passed
```

Verified with:

```bash
python -m unittest tests.test_layout_detector tests.test_table_html
```

What these tests cover:
- DocLayout-YOLO output parsing
- class-aware NMS
- header/footer post-correction
- no-model fallback behavior
- page-number heuristics
- table cell rendering and HTML structure

### 2) End-to-end sanity check (single image)

Observed on one real sample image with the current config:

- **non-empty output produced**
- `blocks_source = layout_detector`
- `blocks_count = 12`
- `tables_count = 0`
- `total_time ≈ 39.02s wall time` in the one-sample check

This confirms that the pipeline is **not just code-complete** — it already runs inference end-to-end and emits HTML.

### 3) 5-image micro-benchmark (internal proxy evaluation)

A 5-image micro-benchmark was run on the current branch using:

```bash
python eval.py \
  --config trainer/config/config.yaml \
  --input /tmp/ocr_eval_5.jsonl \
  --image-root datasets/image/eval \
  --output /tmp/ocr_eval_5_submit.jsonl \
  --debug-output /tmp/ocr_eval_5_debug.jsonl

python scripts/score_proxy.py \
  --gt datasets/label/eval.jsonl \
  --debug /tmp/ocr_eval_5_debug.jsonl
```

Results:

| Metric | Value |
|---|---:|
| Samples | 5 |
| mAP | 0.2360 |
| F1 | 0.4000 |
| Text BLEU proxy (`B_text`) | 0.4766 |
| Formula BLEU proxy (`B_formula`) | 0.0000 |
| Table TEDS proxy (`T_table`) | 0.0000 |
| Reading-order proxy (`K_order`) | 0.6371 |
| **Weighted proxy score** | **0.2699** |

### 4) Runtime snapshot for the 5-image run

| Item | Value |
|---|---:|
| Total runtime | 67.92s |
| Average per image | 13.58s |
| First image (cold start) | 18.74s |
| Subsequent images | ~7.35s to ~7.83s |
| Fallback triggered | 0 / 5 |
| Block source | layout detector for all 5 |

Interpretation:
- the stack is already operational,
- cold-start overhead is noticeable,
- warm runs are materially faster,
- quality is still far from a polished production parser,
- current failure modes are mostly about **duplicate blocks / duplicate labels / weak structure normalization**, not “cannot run at all”.

---

## Current Qualitative Status

### What is already working

- local inference runs successfully
- layout detector path is active
- OCR enrichment runs successfully
- HTML is produced
- unit tests pass
- proxy scorer works
- repository is cloneable and inspectable

### What is still weak

- duplicate detections leak into final HTML
- headings / captions can be over-produced
- formula and table quality are not demonstrated by the current 5-image slice
- the current proxy score is **usable for iteration**, but not yet competitive with benchmark-oriented parsing systems

In short:

> **The pipeline has crossed the “runs end-to-end” threshold, but it has not yet crossed the “high-quality document parser” threshold.**

---

## Architecture Overview

```text
Image / Page
   │
   ├─► Layout detector
   │      └─► block candidates
   │
   ├─► OCR enrichment on selected ROIs
   │
   ├─► relation / reading-order decoding
   │
   ├─► optional table / formula processing
   │
   └─► HTML rendering + debug statistics
```

Main implementation entry points:

- `eval.py` — main inference / evaluation entry
- `scripts/sanity_check.py` — end-to-end sanity runner
- `scripts/score_proxy.py` — lightweight proxy scorer
- `scripts/smoke_layout_detector.py` — layout smoke test
- `tests/test_layout_detector.py` — layout detector unit tests
- `tests/test_table_html.py` — HTML/table rendering unit tests

---

## Repository Layout

```text
ocr/
├── datasets/                     # images + labels for local evaluation/debugging
├── docs/                         # integration notes
├── eval.py                       # main parsing / inference entry
├── eval.sh                       # helper shell entry
├── merge_lora.py                 # model merge utility
├── models/                       # lightweight config/model mapping files
├── README.md
├── requirements.txt
├── scripts/
│   ├── sanity_check.py
│   ├── score_proxy.py
│   └── smoke_layout_detector.py
├── tests/
│   ├── test_layout_detector.py
│   └── test_table_html.py
├── train.py
├── train.sh
├── trainer/
└── utils/
```

---

## Installation

### Recommended environment

- Python **3.12**
- Linux (validated in the current workspace)
- GPU recommended for practical runtime

### Setup

```bash
git clone https://github.com/Hunryyy/ocr.git
cd ocr
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Notes on weights

This repository does **not** ship local machine environments or downloaded runtime caches.

However, the current stack can still be made runnable after clone because:
- the code is included,
- the configs are included,
- the datasets are included,
- PaddleX / PaddleOCR can auto-download required official models on first run when network access is available.

Optional custom ONNX layout weights are **not bundled**.

---

## Quick Start

### 1. Run unit tests

```bash
python -m unittest tests.test_layout_detector tests.test_table_html
```

### 2. Run a sanity check on one image

```bash
python scripts/sanity_check.py \
  --config trainer/config/config.yaml \
  --images datasets/image/eval/f850e9fc-a4f8-4393-8f6c-ae6c21abdffc.jpg \
  --output /tmp/sanity_output.jsonl
```

### 3. Run the main parser

```bash
python eval.py \
  --config trainer/config/config.yaml \
  --input your_input.jsonl \
  --image-root datasets/image/eval \
  --output submit.jsonl \
  --debug-output debug.jsonl
```

### 4. Run the proxy scorer

```bash
python scripts/score_proxy.py \
  --gt datasets/label/eval.jsonl \
  --debug debug.jsonl
```

---

## Comparison Notes vs MonkeyOCR / dots.ocr

### Inspired by them

This README intentionally borrows the following good practices seen in strong OCR repositories:

- public benchmark snapshots
- explicit “what works / what does not” framing
- install + inference + evaluation sections
- model-positioning discussion instead of vague marketing

### What we do differently

Unlike MonkeyOCR or dots.ocr, this repository currently emphasizes:

- **pipeline transparency** over monolithic model packaging
- **local debugging** over leaderboard storytelling
- **proxy scoring** over benchmark-only presentation
- **component-level tests** over pure demo-driven release

### Honest comparison

If you want:
- **state-of-the-art benchmark chasing**, MonkeyOCR / dots.ocr are much closer to that world today.
- **hackable code paths, local iteration, explicit scripts, and inspectable failure modes**, this repository is a more natural engineering playground.

---

## Known Limitations

- current README metrics are **internal proxy metrics**, not official benchmark submissions
- current quality is still limited by duplicate block generation and incomplete structure cleanup
- table / formula quality are not yet well represented by the current 5-image slice
- current layout path relies on runtime-downloaded dependencies unless custom weights are provided manually
- the config file currently mixes “PaddleOCR v4” wording with runtime behavior that loads newer official Paddle models; this should be cleaned up in future iterations

---

## Roadmap

Short-term:
- reduce duplicate blocks in rendered HTML
- tighten class normalization for title / caption / image overlap
- stabilize formula and table parsing paths
- improve README benchmark reporting with larger reproducible slices

Mid-term:
- stronger structure-aware evaluation
- cleaner warm-start / runtime profile
- more realistic document subsets for internal reporting

Long-term:
- move from “pipeline runs” to “pipeline scores well and renders cleanly”

---

## Docs

- `docs/doclayout_yolo.md`
- `docs/doclayout_yolo_integration.md`

---

## Included vs Excluded in GitHub

### Included
- source code
- configs
- docs
- datasets
- tests
- proxy scoring / sanity scripts

### Excluded
- `.venv/`
- `cache/`
- `tmp/`
- machine-specific runtime artifacts
- local downloaded large model files

---

## Acknowledgement

This repository has been iterated as an engineering-first OCR project, with ongoing local improvements to layout parsing, HTML rendering, sanity-check coverage, and proxy scoring.

If you are looking for a strong benchmark-oriented parser, check the public work from **MonkeyOCR** and **dots.ocr**. If you are looking for a repository you can clone, inspect, debug, and keep improving step by step, this repo is aimed exactly at that workflow.
