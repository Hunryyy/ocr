#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[ERROR] train.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="${DATA_DIR:-}"
TRAIN_LABEL="${TRAIN_LABEL:-./datasets/label/train.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-./datasets/image/train}"
WORK_DIR="${WORK_DIR:-./output/train_workdir}"
OUTPUT_PATH="${OUTPUT_PATH:-./artifacts_v2}"
CONFIG_FILE="${CONFIG_FILE:-./trainer/config/config_optimized.yaml}"
TRAIN_RATIO="${TRAIN_RATIO:-0.9}"
SEED="${SEED:-42}"
PROFILE="${PROFILE:-accurate}"
LOG_DIR="${LOG_DIR:-./logs}"

usage() {
  cat <<'USAGE'
Usage:
  bash train.sh [options]

Options:
  --data_dir DIR           Shortcut: DIR/train.jsonl and DIR/images
  --train-label FILE       Training label JSONL
  --image-root DIR         Training image root
  --work-dir DIR           Intermediate output dir
  --output-path DIR        Final artifact export dir
  --config FILE            Config path
  --train-ratio FLOAT      Train split ratio
  --seed INT               Random seed
  --profile NAME           preprocess profile
  --log-dir DIR            log directory
  --help                   show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --train-label) TRAIN_LABEL="$2"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "$DATA_DIR" ]]; then
  if [[ "$TRAIN_LABEL" == "./datasets/label/train.jsonl" ]]; then
    TRAIN_LABEL="${DATA_DIR}/train.jsonl"
  fi
  if [[ "$IMAGE_ROOT" == "./datasets/image/train" ]]; then
    IMAGE_ROOT="${DATA_DIR}/images"
  fi
fi

if [[ -f "./.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "./.venv/bin/activate"
fi
PYTHON_BIN="$(command -v python3 || command -v python)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[ERROR] python not found" >&2
  exit 1
fi

[[ -f "$TRAIN_LABEL" ]] || { echo "[ERROR] train jsonl missing: $TRAIN_LABEL" >&2; exit 1; }
[[ -d "$IMAGE_ROOT" ]] || { echo "[ERROR] image root missing: $IMAGE_ROOT" >&2; exit 1; }
[[ -f "$CONFIG_FILE" ]] || { echo "[ERROR] config missing: $CONFIG_FILE" >&2; exit 1; }

mkdir -p "$WORK_DIR/models" "$OUTPUT_PATH" "$LOG_DIR"

TRAIN_JSONL="${WORK_DIR}/train.jsonl"
VAL_JSONL="${WORK_DIR}/val.jsonl"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/train_${TS}.log"

{
  echo "[INFO] Step 0.5 clean dataset"
  "$PYTHON_BIN" ./trainer/dataset/clean_dataset.py \
    --input "$TRAIN_LABEL" \
    --output "${WORK_DIR}/train_cleaned.jsonl"

  echo "[INFO] Step1 preprocess"
  "$PYTHON_BIN" ./trainer/dataset/preprocess.py \
    --input "${WORK_DIR}/train_cleaned.jsonl" \
    --output-train "$TRAIN_JSONL" \
    --output-val "$VAL_JSONL" \
    --image-root "$IMAGE_ROOT" \
    --train-ratio "$TRAIN_RATIO" \
    --seed "$SEED" \
    --enable-ocr \
    --ocr-use-gpu \
    --profile "$PROFILE"

  echo "[INFO] Step2 train models"
  "$PYTHON_BIN" train.py \
    --train "$TRAIN_JSONL" \
    --val "$VAL_JSONL" \
    --out-dir "$WORK_DIR" \
    --seed "$SEED"

  echo "[INFO] Step3 export artifacts"
  "$PYTHON_BIN" ./trainer/merge/merge_lora.py export \
    --config "$CONFIG_FILE" \
    --output "$OUTPUT_PATH" \
    --models-dir "$WORK_DIR/models" \
    --schema-dir "$WORK_DIR"

  echo "[INFO] training pipeline done"
  echo "[INFO] artifacts=${OUTPUT_PATH}"
} 2>&1 | tee "$RUN_LOG"
