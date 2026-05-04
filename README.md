# 多模态文档解析系统（竞赛版）

默认配置：`trainer/config/config_optimized.yaml`。  
本仓库已经按竞赛格式整理训练、推理、导出与提交打包流程。

## 第一部分：算法、优势与创新点

### 1) 整体方案
系统采用“规则 + 学习 + 兜底”混合架构：
- 预处理：`trainer/dataset/preprocess.py`
- 训练：`train.py`（LightGBM block 分类器 + 关系评分器）
- 推理：`eval.py`（布局检测、OCR、关系推理、HTML 渲染）
- 导出：`trainer/merge/merge_lora.py`

### 2) 数据驱动设计（基于 `datasets` 全量统计）
- 训练集/验证集：`2000 / 500` 页。
- 页面块数量：均值约 `8`，`p95` 约 `20`。
- 类别分布长尾明显：`paragraph/title` 占绝大多数，`table/formula/chart` 较少。
- 图像分辨率分布离散，主流尺寸为 `1710x963`、`1440x1080`、`970x546`。

这些统计用于指导阈值、fallback 和类型校正策略，避免盲目调参。

### 3) 关键算法模块
- 布局检测：ONNX/Paddle 路径自动适配，后处理含 NMS、嵌套框抑制、页眉页脚修正。
- OCR：全图 OCR + ROI OCR，支持缓存；ROI 采用批量处理降低调用开销。
- 关系推理：利用 block/pair 特征学习阅读顺序与标题层级关系。
- 表格与公式：表格结构抽取（含 rowspan/colspan）与公式 LaTeX 标准化输出。

### 4) 工程亮点（运维友好）
- 运行时熔断：Paddle 致命错误后自动禁用高成本失败路径，避免每页重复报错。
- 退化模式降本：无可用视觉引擎时跳过无收益 fallback，缩短失败场景耗时。
- Prompt 对齐：输出 `prompt` 自动从 `prompt/prefix/default_prompt` 回退，保证提交字段稳定。
- 多进程预处理优化：worker 级 OCR 引擎复用，减少重复初始化开销。
- 训练权重修正：类别权重归一化仅基于真实出现类别，提升长尾训练稳定性。

### 5) 当前测试结果（本环境）
- `python3 test_suite.py --config trainer/config/config_optimized.yaml --loops 20`  
  11/11 通过，Pass Rate 100%，Wall Time 5.36s
- `python3 test_suite.py --config trainer/config/config_optimized.yaml --loops 100`  
  11/11 通过，Pass Rate 100%，Wall Time 5.52s
- 12 张小样本端到端：Score 0.1500，耗时 9.13s（相对历史基线 21.78s 下降 58.1%）

> 说明：本机存在 Paddle oneDNN 兼容问题，会触发降级路径；上述结果体现的是“稳定性与效率”而非最终线上上限。建议在竞赛一致环境复测全量指标。

---

## 第二部分：Windows 新手从零跑通训练 + 评估（超详细）

下面步骤假设你第一次接触命令行，全部在 **PowerShell** 执行。

### A. 准备环境

1. 安装 Python（建议 3.10 或 3.11）
- 打开浏览器访问 `https://www.python.org/downloads/windows/`
- 下载并安装 Python
- 安装时勾选 `Add python.exe to PATH`

2. 安装 Git（可选，但建议）
- 访问 `https://git-scm.com/download/win` 安装

3. 打开 PowerShell，检查是否安装成功
```powershell
python --version
pip --version
```

如果提示“不是内部或外部命令”，请重启电脑后重试。

### B. 获取代码与数据

1. 把项目放到一个英文路径目录，例如：
- `D:\docparse\vibe1`

2. 在 PowerShell 进入项目目录：
```powershell
cd D:\docparse\vibe1
```

3. 检查数据是否存在：
```powershell
dir .\datasets\label
dir .\datasets\image\train
dir .\datasets\image\eval
```

你应看到：
- `datasets\label\train.jsonl`
- `datasets\label\eval.jsonl`

### C. 创建虚拟环境并安装依赖

1. 创建虚拟环境：
```powershell
python -m venv .venv
```

2. 激活虚拟环境：
```powershell
.\.venv\Scripts\Activate.ps1
```

3. 升级 pip 并安装依赖：
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果网络慢，可使用镜像：
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### D. 训练模型（分步）

1. 运行完整训练脚本（推荐）：
```powershell
bash train.sh --data_dir .\datasets --work-dir .\output\train_workdir --output-path .\artifacts_v23
```

如果没有 `bash`，用 Python 分步执行：

2. 先预处理：
```powershell
python .\trainer\dataset\preprocess.py `
  --input .\datasets\label\train.jsonl `
  --output-train .\output\train_preprocessed.jsonl `
  --output-val .\output\val_preprocessed.jsonl `
  --image-root .\datasets\image\train `
  --profile accurate `
  --num-workers 0
```

3. 再训练：
```powershell
python .\train.py `
  --train .\output\train_preprocessed.jsonl `
  --val .\output\val_preprocessed.jsonl `
  --out-dir .\output\optimized_v23 `
  --seed 42
```

4. 导出 artifacts：
```powershell
python .\trainer\merge\merge_lora.py export `
  --config .\trainer\config\config_optimized.yaml `
  --output .\artifacts_v23 `
  --models-dir .\output\optimized_v23\models `
  --schema-dir .\output\optimized_v23
```

### E. 评估模型（eval）

```powershell
python .\eval.py `
  --config .\trainer\config\config_optimized.yaml `
  --input .\datasets\label\eval.jsonl `
  --image-root .\datasets\image\eval `
  --output .\eval_results\submission.jsonl `
  --gt .\datasets\label\eval.jsonl `
  --metrics-output .\eval_results\metrics.json
```

完成后重点查看：
- `.\eval_results\submission.jsonl`
- `.\eval_results\metrics.json`

### F. 一键全链路（预处理+训练+导出+评估）

```powershell
bash run_pipeline.sh
```

或：
```powershell
bash run_full_pipeline.sh
```

`run_full_pipeline.sh` 已简化为对 `run_pipeline.sh` 的统一入口，减少维护重复。

### G. 竞赛提交打包

```powershell
bash package_submission.sh
```

会生成：
- `.\submission_pkg\docparse-challenge\`
- `.\submission_pkg\docparse-challenge.zip`

该目录结构已对齐竞赛代码检查要求。

### H. 常见问题（Windows）

1. `Activate.ps1` 被策略阻止
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
然后重新激活虚拟环境。

2. `bash` 不可用
- 安装 Git Bash，或使用 WSL。
- 如果不用 bash，请按上面的“Python 分步执行”命令运行。

3. Paddle/OCR 报错
- CPU 环境先确保基础链路可跑，再切换 GPU。
- 若遇到底层 oneDNN 不兼容，系统会自动走降级路径以保证流程不中断。

4. 训练太慢
- 先用小样本验证链路是否正确：
```powershell
Get-Content .\datasets\label\train.jsonl -TotalCount 50 > .\tmp_train_50.jsonl
```
然后把预处理输入改为这个小文件快速检查。

---

