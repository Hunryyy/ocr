#!/bin/bash
# eval_debug.sh

# 变量配置
CONFIG_FILE="./trainer/config/config.yaml"
INPUT_DATA="./data/clean_test.jsonl" 
IMAGE_ROOT="./datasets/image/train"  # 修正为 debug 确认成功的路径
RESULT_FILE="./eval_results/debug_results.jsonl"

# 环境激活
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate docparse

mkdir -p "$(dirname "$RESULT_FILE")"

echo "==== 启动单进程调试推理 ===="
# 1. 将 parallel 设为 0 或 1 (取决于 eval.py 实现)，强制单进程运行
# 2. 开启 --debug-output 
python eval.py \
    --config "$CONFIG_FILE" \
    --input "$INPUT_DATA" \
    --image-root "$IMAGE_ROOT" \
    --output "$RESULT_FILE" \
    --parallel 1 \
  

echo "==== 运行结束 ===="
if [ ! -s "$RESULT_FILE" ]; then
    echo "❌ 警告：输出文件仍然为空！请向上滚动查看 Python 报错日志（可能是 PaddleOCR 或 ONNX 加载失败）。"
else
    echo "✅ 成功生成结果，前三行如下："
    head -n 3 "$RESULT_FILE"
fi
