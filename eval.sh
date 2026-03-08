#!/usr/bin/env bash
set -euo pipefail

# eval.sh

# Overridable variables
CONFIG_FILE="${CONFIG_FILE:-./trainer/config/config.yaml}"
INPUT_DATA="${INPUT_DATA:-./datasets/label/eval.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-./datasets/image/eval}"
RESULT_FILE="${RESULT_FILE:-./eval_results/results.jsonl}"
PARALLEL="${PARALLEL:-1}"
DEBUG_FILE="${DEBUG_FILE:-}"
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
mkdir -p "$(dirname "$RESULT_FILE")"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Missing config file: $CONFIG_FILE"
    exit 1
fi
if [ ! -f "$INPUT_DATA" ]; then
    echo "❌ Missing input jsonl: $INPUT_DATA"
    exit 1
fi
if [ ! -d "$IMAGE_ROOT" ]; then
    echo "❌ Missing image root: $IMAGE_ROOT"
    exit 1
fi

echo "==== Starting evaluation ===="
echo "CONFIG_FILE=$CONFIG_FILE"
echo "INPUT_DATA=$INPUT_DATA"
echo "IMAGE_ROOT=$IMAGE_ROOT"
echo "RESULT_FILE=$RESULT_FILE"
echo "PARALLEL=$PARALLEL"

CMD=(
    "$PYTHON_BIN" eval.py
    --config "$CONFIG_FILE"
    --input "$INPUT_DATA"
    --image-root "$IMAGE_ROOT"
    --output "$RESULT_FILE"
    --parallel "$PARALLEL"
)

if [ -n "$DEBUG_FILE" ]; then
    mkdir -p "$(dirname "$DEBUG_FILE")"
    CMD+=(--debug-output "$DEBUG_FILE")
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY_RUN] ${CMD[*]}"
    exit 0
fi

"${CMD[@]}"

echo "==== Evaluation finished ===="
if [ ! -s "$RESULT_FILE" ]; then
    echo "❌ Output file is still empty. Check the logs above."
    exit 1
else
    echo "✅ Results generated successfully. First 3 lines:"
    head -n 3 "$RESULT_FILE"
fi
