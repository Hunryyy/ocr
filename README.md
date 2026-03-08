# OCR

一个面向多模态文档解析的 OCR 项目仓库，包含：
- 文档布局检测
- OCR 文本识别
- 表格 HTML 渲染
- 公式区域处理
- 代理评分与基础验收脚本

## 项目结构

```text
ocr/
├── datasets/               # 数据集（图像 + 标签）
├── docs/                   # 集成说明文档
├── eval.py                 # 主评测 / 推理入口
├── eval.sh                 # 评测脚本
├── merge_lora.py           # 模型合并相关脚本
├── models/                 # 轻量配置类模型文件（不含本地环境权重）
├── requirements.txt        # 基础依赖
├── scripts/
│   ├── sanity_check.py     # 端到端 sanity check
│   ├── score_proxy.py      # 代理评分脚本
│   └── smoke_layout_detector.py  # 布局检测烟测
├── tests/
│   ├── test_layout_detector.py
│   └── test_table_html.py
├── train.py                # 训练入口
├── train.sh                # 训练脚本
├── trainer/                # 训练与配置
└── utils/                  # 工具函数
```

## 当前特性

- 支持基于 PaddleOCR / PaddleX 的文档解析链路
- 支持布局检测增强与表格 HTML 输出
- 支持可选的 DocLayout-YOLO 集成（见 `docs/`）
- 支持代理评分脚本 `scripts/score_proxy.py`
- 支持基础单元测试与端到端 sanity check

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 说明：本仓库不包含本地虚拟环境、缓存、临时文件和本机下载的运行时权重。

### 2. 运行单元测试

```bash
python -m unittest tests.test_layout_detector tests.test_table_html
```

### 3. 运行端到端检查

```bash
python scripts/sanity_check.py \
  --config trainer/config/config.yaml \
  --images datasets/image/eval/your_sample.jpg \
  --output /tmp/sanity_output.jsonl
```

### 4. 运行主评测

```bash
python eval.py \
  --config trainer/config/config.yaml \
  --input your_input.jsonl \
  --image-root datasets/image/eval \
  --output submit.jsonl \
  --debug-output debug.jsonl
```

### 5. 运行代理评分

```bash
python scripts/score_proxy.py \
  --gt datasets/label/eval.jsonl \
  --debug debug.jsonl
```

## 文档

- `docs/doclayout_yolo.md`
- `docs/doclayout_yolo_integration.md`

## 上传说明

当前仓库默认保留：
- 源代码
- 配置文件
- 文档
- datasets

默认排除：
- `.venv/`
- `cache/`
- `tmp/`
- 本地下载权重
- 机器相关环境产物

## 备注

这是一个持续迭代中的 OCR 工程仓库。当前版本重点在于：
- 跑通多模态文档解析链路
- 强化布局检测与表格/公式处理
- 保留可复现的测试与评分脚本
