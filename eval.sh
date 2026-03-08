#!/bin/bash
# eval_debug.sh

# 变量配置
CONFIG_FILE="./trainer/config/config.yaml"
INPUT_DATA="./datasets/label/eval.jsonl"
IMAGE_ROOT="./datasets/image"
RESULT_FILE="./eval_results/results.jsonl"

# 环境激活（优先项目内 .venv，其次 conda）
if [ -f "./.venv/bin/activate" ]; then
    source "./.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
    CONDA_PATH=$(conda info --base)
    source "$CONDA_PATH/etc/profile.d/conda.sh"
    conda activate docparse
else
    echo "⚠️ 未检测到 .venv 或 conda，使用系统 Python。"
fi

mkdir -p "$(dirname "$RESULT_FILE")"

echo "==== 启动推理 ===="
python eval.py \
    --config "$CONFIG_FILE" \
    --input "$INPUT_DATA" \
    --image-root "$IMAGE_ROOT" \
    --output "$RESULT_FILE" \
    --parallel 1


echo "==== 运行结束 ===="
if [ ! -s "$RESULT_FILE" ]; then
    echo "❌ 警告：输出文件仍然为空！请向上滚动查看 Python 报错日志（可能是 PaddleOCR 或 ONNX 加载失败）。"
else
    echo "✅ 成功生成结果，前三行如下："
    head -n 3 "$RESULT_FILE"
fi
