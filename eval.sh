#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "[ERROR] eval.sh failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_FILE="${CONFIG_FILE:-./trainer/config/config_optimized.yaml}"
INPUT_JSONL="${INPUT_JSONL:-./datasets/label/eval.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-./datasets/image/eval}"
OUTPUT_PATH="${OUTPUT_PATH:-./eval_results/submission.jsonl}"
DEBUG_OUTPUT="${DEBUG_OUTPUT:-}"
GT_FILE="${GT_FILE:-}"
METRICS_OUTPUT="${METRICS_OUTPUT:-}"
PARALLEL="${PARALLEL:-0}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-}"
LOG_DIR="${LOG_DIR:-./logs}"

usage() {
  cat <<'USAGE'
Usage:
  bash eval.sh [options]

Options:
  --config FILE          Config path
  --input FILE           Input eval JSONL
  --image-root DIR       Image root
  --output-path FILE     Output submission JSONL
  --data_dir DIR         Shortcut: DIR/eval.jsonl + DIR/images
  --parallel N           Process workers (0 means use config/default)
  --seed N               Random seed
  --debug-output FILE    Optional debug JSONL output
  --gt FILE              Optional GT JSONL for local metric report
  --metrics-output FILE  Optional metric JSON output
  --log-dir DIR          Log directory
  --help                 Show help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --input) INPUT_JSONL="$2"; shift 2 ;;
    --image-root) IMAGE_ROOT="$2"; shift 2 ;;
    --output-path) OUTPUT_PATH="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --parallel) PARALLEL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --debug-output) DEBUG_OUTPUT="$2"; shift 2 ;;
    --gt) GT_FILE="$2"; shift 2 ;;
    --metrics-output) METRICS_OUTPUT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "$DATA_DIR" ]]; then
  [[ "$INPUT_JSONL" == "./datasets/label/eval.jsonl" ]] && INPUT_JSONL="${DATA_DIR}/eval.jsonl"
  [[ "$IMAGE_ROOT" == "./datasets/image/eval" ]] && IMAGE_ROOT="${DATA_DIR}/images"
fi

if [[ -f "./.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "./.venv/bin/activate"
fi
PYTHON_BIN="$(command -v python3 || command -v python)"
[[ -n "$PYTHON_BIN" ]] || { echo "[ERROR] python not found" >&2; exit 1; }

[[ -f "$CONFIG_FILE" ]] || { echo "[ERROR] config missing: $CONFIG_FILE" >&2; exit 1; }
[[ -f "$INPUT_JSONL" ]] || { echo "[ERROR] input jsonl missing: $INPUT_JSONL" >&2; exit 1; }
[[ -d "$IMAGE_ROOT" ]] || { echo "[ERROR] image root missing: $IMAGE_ROOT" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT_PATH")" "$LOG_DIR"
[[ -n "$DEBUG_OUTPUT" ]] && mkdir -p "$(dirname "$DEBUG_OUTPUT")"
[[ -n "$METRICS_OUTPUT" ]] && mkdir -p "$(dirname "$METRICS_OUTPUT")"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/eval_${TS}.log"

CMD=(
  "$PYTHON_BIN" eval.py
  --config "$CONFIG_FILE"
  --input "$INPUT_JSONL"
  --image-root "$IMAGE_ROOT"
  --output "$OUTPUT_PATH"
  --parallel "$PARALLEL"
  --seed "$SEED"
)
[[ -n "$DEBUG_OUTPUT" ]] && CMD+=(--debug-output "$DEBUG_OUTPUT")
if [[ -n "$GT_FILE" ]]; then
  [[ -f "$GT_FILE" ]] || { echo "[ERROR] gt missing: $GT_FILE" >&2; exit 1; }
  CMD+=(--gt "$GT_FILE")
fi
[[ -n "$METRICS_OUTPUT" ]] && CMD+=(--metrics-output "$METRICS_OUTPUT")

{
  echo "[INFO] Running command: ${CMD[*]}"
  "${CMD[@]}"
} 2>&1 | tee "$RUN_LOG"

[[ -s "$OUTPUT_PATH" ]] || { echo "[ERROR] output not generated: $OUTPUT_PATH" >&2; exit 1; }
echo "[INFO] done. output=$OUTPUT_PATH"
echo "[INFO] log=$RUN_LOG"
head -n 3 "$OUTPUT_PATH"
