#!/bin/bash
# 使用 bash 运行以支持 source 命令

# --- 变量配置 ---
RAW_ANNOTATION="./datasets/label/train.jsonl"  # 对应 preprocess.py 的 --input
IMAGE_ROOT="./datasets/image/train"             # 对应 preprocess.py 的 --image-root
WORK_DIR="./output/train_workdir"
DEPLOY_DIR="./artifacts_v2"

# 1. 激活 Conda 环境 (更稳健的写法)
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate docparse || { echo "Conda env activate failed"; exit 1; }

export PYTHONPATH=$PYTHONPATH:.
mkdir -p "$WORK_DIR/models"
mkdir -p "$DEPLOY_DIR"

echo "==== [Step 1/3] 数据预处理与特征提取 ===="
# 对照 preprocess.py 实际参数名: --input, --output-train, --output-val, --image-root
python ./trainer/dataset/preprocess.py \
    --input "$RAW_ANNOTATION" \
    --output-train "$WORK_DIR/train.jsonl" \
    --output-val "$WORK_DIR/val.jsonl" \
    --image-root "$IMAGE_ROOT" \
    --train-ratio 0.9 \
    --seed 42

echo "==== [Step 2/3] 训练 LightGBM 模型 (Schema 2.0) ===="
# 对照 train.py 实际参数名: --train, --val, --out-dir
python train.py \
    --train "$WORK_DIR/train.jsonl" \
    --val "$WORK_DIR/val.jsonl" \
    --out-dir "$WORK_DIR" \
    --seed 42

echo "==== [Step 3/3] 导出部署 Artifacts 包 ===="
# merge_lora.py 参数保持不变
python merge_lora.py export \
    --config ./trainer/config/config.yaml \
    --output "$DEPLOY_DIR" \
    --models-dir "$WORK_DIR/models" \
    --schema-dir "$WORK_DIR"

echo "==== 训练完成！产物目录: $DEPLOY_DIR ===="
