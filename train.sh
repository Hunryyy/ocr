#!/usr/bin/env bash
set -euo pipefail

# train.sh

# Overridable variables
RAW_ANNOTATION="${RAW_ANNOTATION:-./datasets/label/train.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-./datasets/image/train}"
WORK_DIR="${WORK_DIR:-./output/train_workdir}"
DEPLOY_DIR="${DEPLOY_DIR:-./artifacts_v2}"
CONFIG_FILE="${CONFIG_FILE:-./trainer/config/config.yaml}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-42}"
PROFILE="${PROFILE:-accurate}"
DRY_RUN="${DRY_RUN:-0}"

# Environment activation (prefer local .venv, then conda, then system Python)
if [ -f "./.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "./.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
    CONDA_PATH=$(conda info --base)
    # shellcheck disable=SC1091
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate docparse || echo "⚠️ conda env 'docparse' not found, falling back to system Python"
else
    echo "⚠️ No .venv or conda detected, using system Python."
fi

PYTHON_BIN="$(command -v python || command -v python3)"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
mkdir -p "$WORK_DIR/models"
mkdir -p "$DEPLOY_DIR"

if [ ! -f "$RAW_ANNOTATION" ]; then
    echo "❌ Missing training annotation file: $RAW_ANNOTATION"
    exit 1
fi
if [ ! -d "$IMAGE_ROOT" ]; then
    echo "❌ Missing training image root: $IMAGE_ROOT"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Missing config file: $CONFIG_FILE"
    exit 1
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] $PYTHON_BIN ./trainer/dataset/preprocess.py --input $RAW_ANNOTATION --output-train $WORK_DIR/train.jsonl --output-val $WORK_DIR/val.jsonl --image-root $IMAGE_ROOT --train-ratio $TRAIN_RATIO --seed $SEED --enable-ocr --ocr-use-gpu --profile $PROFILE"
    echo "[DRY_RUN] $PYTHON_BIN train.py --train $WORK_DIR/train.jsonl --val $WORK_DIR/val.jsonl --out-dir $WORK_DIR --seed $SEED"
    echo "[DRY_RUN] $PYTHON_BIN merge_lora.py export --config $CONFIG_FILE --output $DEPLOY_DIR --models-dir $WORK_DIR/models --schema-dir $WORK_DIR"
    exit 0
fi

echo "==== [Step 1/3] Preprocess and feature extraction ===="
"$PYTHON_BIN" ./trainer/dataset/preprocess.py \
    --input "$RAW_ANNOTATION" \
    --output-train "$WORK_DIR/train.jsonl" \
    --output-val "$WORK_DIR/val.jsonl" \
    --image-root "$IMAGE_ROOT" \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED" \
    --enable-ocr \
    --ocr-use-gpu \
    --profile "$PROFILE"

echo "==== [Step 2/3] Train LightGBM models ===="
"$PYTHON_BIN" train.py \
    --train "$WORK_DIR/train.jsonl" \
    --val "$WORK_DIR/val.jsonl" \
    --out-dir "$WORK_DIR" \
    --seed "$SEED"

echo "==== [Step 3/3] Export deployment artifacts ===="
"$PYTHON_BIN" merge_lora.py export \
    --config "$CONFIG_FILE" \
    --output "$DEPLOY_DIR" \
    --models-dir "$WORK_DIR/models" \
    --schema-dir "$WORK_DIR"

echo "==== Training finished ===="
echo "Artifacts directory: $DEPLOY_DIR"
