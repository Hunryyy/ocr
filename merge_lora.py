

"""
merge_lora.py - Export / Merge / Validate / Report 工具

功能：
1. export   - 导出 artifacts 包（从训练产物复制，不自创标准）
2. merge    - 合并 LoRA 权重到基础模型（预留给未来视觉模型）
3. validate - 校验 artifacts 完整性与可部署性
4. report   - 生成部署自检报告（含 dry-run 测试）

设计原则：
- export 只做"打包与校验"，不自创标准
- schema/label_map 必须从训练产物复制，确保与 A 完全一致
- 模型文件后缀统一策略：保持源文件后缀不变
- validate 严格检查 schema 与模型特征数一致性
- 支持多模态兜底模型（layout_detector/ocr/table_refiner/formula_ocr）

使用示例：
    python merge_lora.py export --config config.yaml --output artifacts/ --models-dir models/ --schema-dir train_output/
    python merge_lora.py validate --artifacts artifacts/
    python merge_lora.py report --artifacts artifacts/ --output report.json
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 可选依赖
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    lgb = None
    HAS_LGB = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    ort = None
    HAS_ORT = False


# ============================================================================
# 常量定义
# ============================================================================

# Schema 版本（仅作为参考，实际以训练产物为准）
DEFAULT_FEATURE_SCHEMA_VERSION = "2.0"

# 特征维度常量（与 train.py 对齐）
EXPECTED_BLOCK_FEAT_DIM = 29  # Block 特征维度
EXPECTED_PAIR_FEAT_DIM = 34   # Pair 特征维度

# IR Schema 版本（用于 manifest，描述 IR 结构）
IR_SCHEMA_VERSION = "v1"

# 支持的 Profiles
SUPPORTED_PROFILES = ["fast", "balanced", "accurate"]

# 固定类别集合（必须与 A 的 LABEL_MAP 完全一致，顺序也一致）
EXPECTED_LABEL_SET = [
    "title",
    "paragraph",
    "list_item",
    "caption",
    "table",
    "figure",
    "formula",
    "header",
    "footer",
    "chart",
    "unknown"
]

# 必需文件列表
REQUIRED_FILES = {
    "manifest": "manifest.json",
    "config": "config.default.yaml",
    "label_map": "label_map.json",
    "feature_schema_block": "feature_schema_block.json",
    "feature_schema_pair": "feature_schema_pair.json",
}

# 必需模型列表（主链路 LightGBM 模型）
# 注意：relation_scorer_caption 是可选的，训练数据中可能没有 caption 样本
REQUIRED_MODELS = [
    "block_classifier",
    "relation_scorer_order",
]

# 可选模型列表
OPTIONAL_MODELS = [
    "relation_scorer_caption",  # 训练集中可能无 caption 样本
]

# 可选兜底模型及其后端类型
OPTIONAL_FALLBACK_MODELS = {
    "layout_detector": {"backend": "onnx", "extensions": [".onnx"]},
    "ocr": {"backend": "paddleocr", "extensions": [".onnx", ".pdmodel"]},
    "table_refiner": {"backend": "onnx", "extensions": [".onnx"]},
    "formula_ocr": {"backend": "onnx", "extensions": [".onnx"]},
    
    
}

# 模型文件支持的后缀（按优先级排序）
SUPPORTED_MODEL_EXTENSIONS = [".json", ".txt"]
ONNX_EXTENSIONS = [".onnx"]
PADDLE_EXTENSIONS = [".onnx", ".pdmodel", ".pdiparams"]

# Hash 模式阈值：超过此大小使用 quick hash
QUICK_HASH_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10MB

# 调试开关
DEBUG_MODE = os.environ.get("MERGE_LORA_DEBUG", "0") == "1"

# ============================================================================
# 日志配置
# ============================================================================

class SafeFormatter(logging.Formatter):
    """安全的日志格式化器，避免敏感信息泄露"""
    def format(self, record):
        # 在非调试模式下截断过长的消息
        if not DEBUG_MODE and len(str(record.msg)) > 500:
            record.msg = str(record.msg)[:500] + "..."
        return super().format(record)

def setup_logging(verbose: bool = False) -> logging.Logger:
    """配置日志"""
    level = logging.DEBUG if (verbose or DEBUG_MODE) else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(SafeFormatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    
    logger = logging.getLogger("merge_lora")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger

logger = setup_logging()


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    path: str
    backend: str
    version: str
    file_hash: str = ""
    hash_mode: str = "full"  # full 或 quick
    num_features: Optional[int] = None
    num_classes: Optional[int] = None
    file_size_bytes: int = 0
    is_optional: bool = False
    is_fallback: bool = False
    is_bundled: bool = True
    enabled: bool = True


@dataclass
class SchemaInfo:
    """Schema 信息"""
    schema_version: str
    model_type: str
    feature_count: int
    feature_names: List[str]
    file_hash: str = ""


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class DryRunResult:
    """Dry-run 测试结果"""
    passed: bool
    models_load_time_ms: Dict[str, float] = field(default_factory=dict)
    schema_load_time_ms: Dict[str, float] = field(default_factory=dict)
    prediction_test_passed: Dict[str, bool] = field(default_factory=dict)
    fallback_models_status: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_time_ms: float = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class DeploymentReport:
    """部署自检报告"""
    timestamp: str
    artifacts_path: str
    ir_schema_version: str
    feature_schema_version: str
    profiles_supported: List[str]
    validation_passed: bool
    models: List[Dict[str, Any]]
    schemas: Dict[str, Dict[str, Any]]
    label_map_info: Dict[str, Any]
    fallback_models: Dict[str, Any]
    config_summary: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    dry_run_stats: Optional[Dict[str, Any]] = None


@dataclass
class ExportResult:
    """导出结果摘要"""
    success: bool
    artifacts_path: str
    files_copied: int
    models_copied: int
    fallback_models_copied: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# 工具函数
# ============================================================================

def compute_sha256(filepath: Union[str, Path], quick_mode: bool = False) -> Tuple[str, str]:
    """
    计算文件 SHA256 哈希

    Args:
        filepath: 文件路径
        quick_mode: 快速模式（只读取前后各 1MB，用于大模型文件）

    Returns:
        (hash_value, hash_mode) 元组
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return "", "none"

    sha256_hash = hashlib.sha256()
    file_size = filepath.stat().st_size

    # 自动决定是否使用 quick mode
    use_quick = quick_mode or file_size > QUICK_HASH_THRESHOLD_BYTES
    hash_mode = "quick" if use_quick else "full"

    try:
        with open(filepath, "rb") as f:
            if use_quick and file_size > 2 * 1024 * 1024:
                # 快速模式：读取前 1MB + 文件大小 + 后 1MB
                sha256_hash.update(f.read(1024 * 1024))
                sha256_hash.update(f"__size_{file_size}__".encode())
                f.seek(-1024 * 1024, 2)
                sha256_hash.update(f.read(1024 * 1024))
            else:
                # 完整模式
                hash_mode = "full"
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
    except (IOError, OSError) as e:
        logger.warning(f"计算哈希失败 {filepath}: {e}")
        return "", "error"

    return sha256_hash.hexdigest(), hash_mode


def compute_full_sha256(filepath: Union[str, Path]) -> str:
    """计算完整文件 SHA256（用于小文件）"""
    hash_value, _ = compute_sha256(filepath, quick_mode=False)
    return hash_value


def load_yaml_safe(filepath: Union[str, Path]) -> Dict[str, Any]:
    """安全加载 YAML 文件，兼容无 yaml 库的情况"""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    if HAS_YAML:
        return yaml.safe_load(content) or {}
    else:
        # 降级：尝试作为 JSON 解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("yaml 库未安装，且文件不是有效 JSON，返回空配置")
            return {}


def save_yaml_safe(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """安全保存 YAML 文件"""
    filepath = Path(filepath)
    if HAS_YAML:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    else:
        # 降级：保存为 JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.warning(f"yaml 库未安装，已将配置保存为 JSON 格式: {filepath}")


def load_json(filepath: Union[str, Path]) -> Any:
    """加载 JSON 文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2) -> None:
    """保存 JSON 文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def deep_merge(base: Dict, overlay: Dict) -> Dict:
    """深度合并两个字典，overlay 覆盖 base"""
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def resolve_config(config: Dict[str, Any], profile: Optional[str] = None) -> Dict[str, Any]:
    """
    解析配置：应用 profile overlay

    优先级（从低到高）：
    1. base defaults
    2. profiles[active] overlay
    3. root-level overrides
    """
    base_config = {k: v for k, v in config.items() if k not in ("profiles", "profile")}
    active_profile = profile or config.get("profile", "balanced")
    profiles = config.get("profiles", {})
    profile_overlay = profiles.get(active_profile, {})
    resolved = deep_merge(base_config, profile_overlay)
    resolved["_active_profile"] = active_profile
    return resolved


def find_model_file(models_dir: Path, model_name: str, 
                    extensions: Optional[List[str]] = None) -> Optional[Path]:
    """
    在目录中查找模型文件

    Args:
        models_dir: 模型目录
        model_name: 模型名称
        extensions: 支持的扩展名列表

    Returns:
        找到的模型文件路径，未找到返回 None
    """
    if extensions is None:
        extensions = SUPPORTED_MODEL_EXTENSIONS
    
    for ext in extensions:
        candidate = models_dir / f"{model_name}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_fallback_model_file(search_dir: Path, model_name: str, 
                             config_path: Optional[str] = None) -> Optional[Path]:
    """
    查找兜底模型文件
    
    Args:
        search_dir: 搜索目录
        model_name: 模型名称
        config_path: 配置中指定的路径
    
    Returns:
        找到的模型文件路径
    """
    model_info = OPTIONAL_FALLBACK_MODELS.get(model_name, {})
    extensions = model_info.get("extensions", ONNX_EXTENSIONS)
    
    # 优先使用配置中指定的路径
    if config_path:
        config_file = search_dir / config_path
        if config_file.exists():
            return config_file
        # 也检查相对于 search_dir 的子目录
        for subdir in ["", "fallback_models", "models"]:
            if subdir:
                candidate = search_dir / subdir / Path(config_path).name
            else:
                candidate = search_dir / Path(config_path).name
            if candidate.exists():
                return candidate
    
    # 按扩展名搜索
    return find_model_file(search_dir, model_name, extensions)


def load_schema_info(schema_path: Path) -> Optional[SchemaInfo]:
    """
    加载并解析 schema 文件

    Args:
        schema_path: schema 文件路径

    Returns:
        SchemaInfo 对象，加载失败返回 None
    """
    if not schema_path.exists():
        return None

    try:
        schema = load_json(schema_path)
        features = schema.get("features", [])
        feature_names = [f.get("name", "") for f in features]

        return SchemaInfo(
            schema_version=schema.get("schema_version", "unknown"),
            model_type=schema.get("model_type", "unknown"),
            feature_count=len(features),
            feature_names=feature_names,
            file_hash=compute_full_sha256(schema_path)
        )
    except Exception as e:
        logger.error(f"加载 schema 失败 {schema_path}: {e}")
        return None


def load_label_map(label_map_path: Path) -> Optional[List[str]]:
    """加载 label_map 文件"""
    if not label_map_path.exists():
        return None

    try:
        label_map = load_json(label_map_path)
        if isinstance(label_map, list):
            return label_map
        return None
    except Exception:
        return None


def compare_label_maps(actual: List[str], expected: List[str]) -> Dict[str, Any]:
    """
    比较两个 label map，返回差异详情
    
    Returns:
        包含 missing, extra, order_diff, is_match 的字典
    """
    actual_set = set(actual)
    expected_set = set(expected)
    
    missing = list(expected_set - actual_set)
    extra = list(actual_set - expected_set)
    order_diff = actual != expected and actual_set == expected_set
    
    return {
        "is_match": actual == expected,
        "missing": missing,
        "extra": extra,
        "order_diff": order_diff,
        "actual_count": len(actual),
        "expected_count": len(expected),
    }


def get_model_num_features(model_path: Path, backend: str = "lightgbm") -> Optional[int]:
    """
    获取模型的特征数量
    
    Args:
        model_path: 模型文件路径
        backend: 模型后端类型
    
    Returns:
        特征数量，失败返回 None
    """
    if backend == "lightgbm" or model_path.suffix in [".json", ".txt"]:
        if not HAS_LGB:
            return None
        try:
            booster = lgb.Booster(model_file=str(model_path))
            return booster.num_feature()
        except Exception:
            return None
    elif backend == "onnx" or model_path.suffix == ".onnx":
        if not HAS_ORT:
            return None
        try:
            session = ort.InferenceSession(str(model_path))
            inputs = session.get_inputs()
            if inputs:
                shape = inputs[0].shape
                if len(shape) >= 2:
                    return shape[-1] if isinstance(shape[-1], int) else None
            return None
        except Exception:
            return None
    return None


# ============================================================================
# Export 命令 - Artifacts 导出器
# ============================================================================

class ArtifactsExporter:
    """
    Artifacts 导出器

    设计原则：
    - 只做"打包与校验"，不自创标准
    - schema/label_map 必须从训练产物复制
    - 模型文件保持源文件后缀不变
    - config 中的路径与实际复制的文件名一致
    - 支持 fallback 模型的条件复制
    """

    def __init__(
        self,
        config_path: str,
        output_dir: str,
        models_dir: Optional[str] = None,
        schema_dir: Optional[str] = None
    ):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir) if models_dir else None
        self.schema_dir = Path(schema_dir) if schema_dir else None
        self.config = load_yaml_safe(self.config_path)

        # 记录实际复制的模型文件路径
        self.copied_model_paths: Dict[str, str] = {}
        self.copied_fallback_paths: Dict[str, str] = {}

        # 记录 schema 信息
        self.block_schema_info: Optional[SchemaInfo] = None
        self.pair_schema_info: Optional[SchemaInfo] = None

        # 导出结果
        self.result = ExportResult(
            success=False,
            artifacts_path=str(self.output_dir),
            files_copied=0,
            models_copied=0,
            fallback_models_copied=0
        )

    def export(self) -> ExportResult:
        """执行导出"""
        logger.info(f"{'='*60}")
        logger.info(f"开始导出 artifacts")
        logger.info(f"  输出目录: {self.output_dir}")
        logger.info(f"  模型目录: {self.models_dir}")
        logger.info(f"  Schema目录: {self.schema_dir}")
        logger.info(f"{'='*60}")

        # 前置检查
        if not self._validate_inputs():
            return self.result

        try:
            # 1. 创建目录结构
            self._create_directory_structure()

            # 2. 复制 schema 文件
            if not self._copy_schema_files():
                return self.result

            # 3. 复制 label_map
            if not self._copy_label_map():
                return self.result

            # 4. 复制主模型文件
            if not self._copy_models():
                return self.result

            # 5. 复制 fallback 模型
            self._copy_fallback_models()

            # 6. 导出并重写配置
            self._export_config()

            # 7. 生成 manifest
            self._generate_manifest()

            # 8. 验证导出结果
            if not self._verify_export():
                return self.result

            self.result.success = True
            self._print_summary()
            return self.result

        except Exception as e:
            error_msg = f"导出失败: {str(e)[:200]}"
            self.result.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return self.result

    def _validate_inputs(self) -> bool:
        """验证输入参数"""
        errors = []

        if not self.config_path.exists():
            errors.append(f"配置文件不存在: {self.config_path}")

        if not self.models_dir:
            errors.append("必须指定 --models-dir 参数")
        elif not self.models_dir.exists():
            errors.append(f"模型目录不存在: {self.models_dir}")

        if not self.schema_dir:
            errors.append("必须指定 --schema-dir 参数（训练产物目录）")
        elif not self.schema_dir.exists():
            errors.append(f"Schema 目录不存在: {self.schema_dir}")

        if errors:
            for err in errors:
                logger.error(f"❌ {err}")
                self.result.errors.append(err)
            return False

        return True

    def _create_directory_structure(self) -> None:
        """创建目录结构"""
        dirs = [
            self.output_dir,
            self.output_dir / "models",
            self.output_dir / "fallback_models",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        logger.info("[1/8] 目录结构创建完成")

    def _copy_schema_files(self) -> bool:
        """复制 schema 文件"""
        logger.info("[2/8] 复制 schema 文件...")
        schema_files = [
            ("feature_schema_block.json", "block"),
            ("feature_schema_pair.json", "pair"),
        ]

        for filename, schema_type in schema_files:
            src_path = self.schema_dir / filename
            dst_path = self.output_dir / filename

            if not src_path.exists():
                error_msg = f"训练产物中缺少 {filename}"
                self.result.errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
                return False

            shutil.copy(src_path, dst_path)
            self.result.files_copied += 1
            logger.info(f"  ✓ {filename}")

            # 加载并记录 schema 信息
            schema_info = load_schema_info(dst_path)
            if schema_type == "block":
                self.block_schema_info = schema_info
            else:
                self.pair_schema_info = schema_info

        return True

    def _copy_label_map(self) -> bool:
        """复制 label_map 文件"""
        logger.info("[3/8] 复制 label_map...")
        src_path = self.schema_dir / "label_map.json"
        dst_path = self.output_dir / "label_map.json"

        if not src_path.exists():
            error_msg = "训练产物中缺少 label_map.json"
            self.result.errors.append(error_msg)
            logger.error(f"❌ {error_msg}")
            return False

        shutil.copy(src_path, dst_path)
        self.result.files_copied += 1
        logger.info(f"  ✓ label_map.json")

        # 验证 label_map 内容
        label_map = load_label_map(dst_path)
        if label_map:
            diff = compare_label_maps(label_map, EXPECTED_LABEL_SET)
            if not diff["is_match"]:
                if diff["missing"]:
                    self.result.warnings.append(f"label_map 缺少: {diff['missing']}")
                if diff["extra"]:
                    self.result.warnings.append(f"label_map 多出: {diff['extra']}")
                if diff["order_diff"]:
                    self.result.warnings.append("label_map 顺序与预期不一致")
                logger.warning(f"⚠️ label_map 与预期不完全一致")

        return True

    def _copy_models(self) -> bool:
        """复制主模型文件"""
        logger.info("[4/8] 复制主模型文件...")
        models_out = self.output_dir / "models"

        # 复制必需模型
        for model_name in REQUIRED_MODELS:
            src_path = find_model_file(self.models_dir, model_name)

            if not src_path:
                error_msg = f"未找到必需模型文件: {model_name}"
                self.result.errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
                return False

            # 保持源文件后缀
            dst_filename = f"{model_name}{src_path.suffix}"
            dst_path = models_out / dst_filename
            shutil.copy(src_path, dst_path)

            # 记录实际路径
            self.copied_model_paths[model_name] = f"models/{dst_filename}"
            self.result.models_copied += 1
            logger.info(f"  ✓ {model_name} -> {dst_filename}")

        # 复制可选模型（如果存在）
        for model_name in OPTIONAL_MODELS:
            src_path = find_model_file(self.models_dir, model_name)

            if src_path:
                dst_filename = f"{model_name}{src_path.suffix}"
                dst_path = models_out / dst_filename
                shutil.copy(src_path, dst_path)
                self.copied_model_paths[model_name] = f"models/{dst_filename}"
                self.result.models_copied += 1
                logger.info(f"  ✓ {model_name} -> {dst_filename} (可选)")
            else:
                logger.info(f"  - {model_name}: 可选模型未找到，跳过")

        return True

    def _copy_fallback_models(self) -> None:
        """复制 fallback 模型（条件复制）"""
        logger.info("[5/8] 处理 fallback 模型...")
        
        resolved_config = resolve_config(self.config)
        fallback_config = resolved_config.get("fallback_models", {})
        models_config = resolved_config.get("models", {})
        
        # 合并 fallback 配置来源
        all_fallback_config = {**fallback_config}
        for name in OPTIONAL_FALLBACK_MODELS:
            if name in models_config:
                all_fallback_config[name] = models_config[name]
        
        fallback_out = self.output_dir / "fallback_models"
        
        for model_name, model_meta in OPTIONAL_FALLBACK_MODELS.items():
            model_cfg = all_fallback_config.get(model_name, {})
            enabled = model_cfg.get("enabled", False)
            config_path = model_cfg.get("path", "")
            
            if not enabled and not config_path:
                logger.info(f"  - {model_name}: 未启用，跳过")
                continue
            
            # 查找模型文件
            src_path = find_fallback_model_file(self.schema_dir, model_name, config_path)
            
            if src_path and src_path.exists():
                # 复制到 fallback_models 目录
                dst_filename = f"{model_name}{src_path.suffix}"
                dst_path = fallback_out / dst_filename
                shutil.copy(src_path, dst_path)
                
                self.copied_fallback_paths[model_name] = f"fallback_models/{dst_filename}"
                self.result.fallback_models_copied += 1
                logger.info(f"  ✓ {model_name} -> {dst_filename} (bundled)")
            else:
                # 外部路径或未找到
                self.copied_fallback_paths[model_name] = config_path or ""
                if enabled:
                    self.result.warnings.append(
                        f"fallback 模型 {model_name} 已启用但未找到可复制的文件"
                    )
                    logger.warning(f"  ⚠ {model_name}: 已启用但未 bundled")
                else:
                    logger.info(f"  - {model_name}: 外部路径，未复制")

    def _export_config(self) -> None:
        """导出并重写配置"""
        logger.info("[6/8] 导出配置文件...")
        config = self.config.copy()

        # 重写 models 路径
        if "models" not in config:
            config["models"] = {}
        
        models = config["models"].copy()
        for model_name, actual_path in self.copied_model_paths.items():
            if model_name in models and isinstance(models[model_name], dict):
                model_cfg = models[model_name].copy()
                model_cfg["path"] = actual_path
                models[model_name] = model_cfg
            else:
                models[model_name] = {"path": actual_path, "backend": "lightgbm"}
        config["models"] = models

        # 重写 fallback_models 路径
        if "fallback_models" not in config:
            config["fallback_models"] = {}
        
        fallback = config["fallback_models"].copy()
        for model_name, actual_path in self.copied_fallback_paths.items():
            if model_name in fallback and isinstance(fallback[model_name], dict):
                fb_cfg = fallback[model_name].copy()
                if actual_path.startswith("fallback_models/"):
                    fb_cfg["path"] = actual_path
                    fb_cfg["is_bundled"] = True
                else:
                    fb_cfg["is_bundled"] = False
                fallback[model_name] = fb_cfg
            else:
                is_bundled = actual_path.startswith("fallback_models/")
                fallback[model_name] = {
                    "path": actual_path,
                    "enabled": False,
                    "is_bundled": is_bundled,
                }
        config["fallback_models"] = fallback

        # 重写 schema 路径
        config["schema"] = {
            "label_map_path": "label_map.json",
            "feature_schema_block_path": "feature_schema_block.json",
            "feature_schema_pair_path": "feature_schema_pair.json",
        }

        # 保存
        config_out = self.output_dir / "config.default.yaml"
        save_yaml_safe(config, config_out)
        self.result.files_copied += 1
        logger.info(f"  ✓ config.default.yaml")


    def _generate_manifest(self) -> None:
        """生成 manifest"""
        logger.info("[7/8] 生成 manifest...")

        # 从 schema 文件读取信息
        block_feature_count = 0
        pair_feature_count = 0
        schema_version = "unknown"

        if self.block_schema_info:
            block_feature_count = self.block_schema_info.feature_count
            schema_version = self.block_schema_info.schema_version

        if self.pair_schema_info:
            pair_feature_count = self.pair_schema_info.feature_count

        # 加载 label_map
        label_map_path = self.output_dir / "label_map.json"
        label_map = load_label_map(label_map_path) or []

        manifest = {
            # 版本信息
            "ir_schema_version": IR_SCHEMA_VERSION,
            "feature_schema_version": schema_version,
            "created_at": datetime.now().isoformat(),
            "export_tool_version": "2.0.0",

            # Profile 支持
            "profiles_supported": SUPPORTED_PROFILES,

            # Label 信息
            "label_set": label_map,
            "label_count": len(label_map),
            "label_map_hash": compute_full_sha256(label_map_path),

            # Schema 信息
            "schemas": {
                "block": {
                    "path": "feature_schema_block.json",
                    "version": self.block_schema_info.schema_version if self.block_schema_info else "unknown",
                    "feature_count": block_feature_count,
                    "feature_names": self.block_schema_info.feature_names if self.block_schema_info else [],
                    "hash": self.block_schema_info.file_hash if self.block_schema_info else "",
                },
                "pair": {
                    "path": "feature_schema_pair.json",
                    "version": self.pair_schema_info.schema_version if self.pair_schema_info else "unknown",
                    "feature_count": pair_feature_count,
                    "feature_names": self.pair_schema_info.feature_names if self.pair_schema_info else [],
                    "hash": self.pair_schema_info.file_hash if self.pair_schema_info else "",
                },
            },

            # 特征数量
            "expected_block_features": block_feature_count,
            "expected_pair_features": pair_feature_count,

            # 主模型列表
            "models": self._collect_model_info(),

            # 兜底模型
            "fallback_models": self._collect_fallback_info(),

            # 配置信息
            "config_hash": compute_full_sha256(self.output_dir / "config.default.yaml"),
            "config_summary": self._get_config_summary(),
        }

        save_json(manifest, self.output_dir / "manifest.json")
        self.result.files_copied += 1
        logger.info(f"  ✓ manifest.json")

    def _collect_model_info(self) -> List[Dict[str, Any]]:
        """收集主模型信息"""
        models = []
        models_dir = self.output_dir / "models"
        resolved_config = resolve_config(self.config)
        models_config = resolved_config.get("models", {})

        # 收集必需模型
        for model_name in REQUIRED_MODELS:
            model_cfg = models_config.get(model_name, {})
            model_path = find_model_file(models_dir, model_name)

            if model_path:
                file_size = model_path.stat().st_size
                hash_value, hash_mode = compute_sha256(model_path)
                num_features = get_model_num_features(model_path, "lightgbm")

                model_info = {
                    "name": model_name,
                    "path": self.copied_model_paths.get(model_name, ""),
                    "backend": model_cfg.get("backend", "lightgbm"),
                    "version": model_cfg.get("version", "1.0.0"),
                    "hash": hash_value,
                    "hash_mode": hash_mode,
                    "file_size_bytes": file_size,
                    "num_features": num_features,
                    "is_required": True,
                    "is_optional": False,
                    "is_bundled": True,
                }
            else:
                model_info = {
                    "name": model_name,
                    "path": "",
                    "backend": "lightgbm",
                    "version": "unknown",
                    "hash": "",
                    "hash_mode": "none",
                    "file_size_bytes": 0,
                    "num_features": None,
                    "is_required": True,
                    "is_optional": False,
                    "is_bundled": False,
                }

            models.append(model_info)

        # 收集可选模型
        for model_name in OPTIONAL_MODELS:
            model_cfg = models_config.get(model_name, {})
            model_path = find_model_file(models_dir, model_name)

            if model_path:
                file_size = model_path.stat().st_size
                hash_value, hash_mode = compute_sha256(model_path)
                num_features = get_model_num_features(model_path, "lightgbm")

                model_info = {
                    "name": model_name,
                    "path": self.copied_model_paths.get(model_name, ""),
                    "backend": model_cfg.get("backend", "lightgbm"),
                    "version": model_cfg.get("version", "1.0.0"),
                    "hash": hash_value,
                    "hash_mode": hash_mode,
                    "file_size_bytes": file_size,
                    "num_features": num_features,
                    "is_required": False,
                    "is_optional": True,
                    "is_bundled": True,
                }
                models.append(model_info)

        return models

    def _collect_fallback_info(self) -> Dict[str, Any]:
        """收集兜底模型信息"""
        resolved_config = resolve_config(self.config)
        fallback_config = resolved_config.get("fallback_models", {})
        models_config = resolved_config.get("models", {})

        # 合并配置来源
        all_config = {**fallback_config}
        for name in OPTIONAL_FALLBACK_MODELS:
            if name in models_config:
                all_config[name] = models_config[name]

        fallback_info = {}
        fallback_dir = self.output_dir / "fallback_models"

        for model_name, model_meta in OPTIONAL_FALLBACK_MODELS.items():
            model_cfg = all_config.get(model_name, {})
            copied_path = self.copied_fallback_paths.get(model_name, "")
            is_bundled = copied_path.startswith("fallback_models/")

            # 获取文件信息
            file_hash = ""
            hash_mode = "none"
            file_size = 0

            if is_bundled:
                file_path = self.output_dir / copied_path
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    file_hash, hash_mode = compute_sha256(file_path)

            fallback_info[model_name] = {
                "enabled": model_cfg.get("enabled", False),
                "backend": model_meta.get("backend", "onnx"),
                "version": model_cfg.get("version", ""),
                "path": copied_path,
                "is_bundled": is_bundled,
                "hash": file_hash,
                "hash_mode": hash_mode,
                "file_size_bytes": file_size,
            }

        return fallback_info

    def _get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "default_profile": self.config.get("profile", "balanced"),
            "profiles_available": list(self.config.get("profiles", {}).keys()),
            "ir_schema_version": IR_SCHEMA_VERSION,
        }

    def _verify_export(self) -> bool:
        """验证导出结果"""
        logger.info("[8/8] 验证导出结果...")
        errors = []
        warnings = []

        # 1. 检查必需文件
        for name, filename in REQUIRED_FILES.items():
            filepath = self.output_dir / filename
            if not filepath.exists():
                errors.append(f"缺少文件: {filename}")
            elif filepath.stat().st_size == 0:
                errors.append(f"文件为空: {filename}")

        # 2. 检查模型文件
        models_dir = self.output_dir / "models"
        
        # 检查必需模型
        for model_name in REQUIRED_MODELS:
            model_path = find_model_file(models_dir, model_name)
            if not model_path:
                errors.append(f"缺少必需模型: {model_name}")
            elif HAS_LGB:
                try:
                    booster = lgb.Booster(model_file=str(model_path))
                    num_features = booster.num_feature()
                    logger.info(f"  ✓ {model_name}: {num_features} 特征")
                except Exception as e:
                    errors.append(f"{model_name}: 加载失败 - {str(e)[:100]}")
        
        # 检查可选模型（仅警告，不报错）
        for model_name in OPTIONAL_MODELS:
            model_path = find_model_file(models_dir, model_name)
            if model_path:
                if HAS_LGB:
                    try:
                        booster = lgb.Booster(model_file=str(model_path))
                        num_features = booster.num_feature()
                        logger.info(f"  ✓ {model_name}: {num_features} 特征 [可选]")
                    except Exception as e:
                        warnings.append(f"{model_name} (可选): 加载失败 - {str(e)[:100]}")
            else:
                logger.info(f"  - {model_name}: 可选模型未包含")

        # 3. 验证 schema 与模型特征数一致
        if HAS_LGB:
            if self.block_schema_info:
                block_model_path = find_model_file(models_dir, "block_classifier")
                if block_model_path:
                    try:
                        booster = lgb.Booster(model_file=str(block_model_path))
                        model_features = booster.num_feature()
                        schema_features = self.block_schema_info.feature_count
                        if model_features != schema_features:
                            errors.append(
                                f"block_classifier 特征数不一致: "
                                f"模型={model_features}, schema={schema_features}, "
                                f"schema_version={self.block_schema_info.schema_version}"
                            )
                        # 验证与预期维度匹配
                        if model_features != EXPECTED_BLOCK_FEAT_DIM:
                            warnings.append(
                                f"block_classifier 特征数 {model_features} 与预期 {EXPECTED_BLOCK_FEAT_DIM} 不匹配"
                            )
                    except Exception:
                        pass

            if self.pair_schema_info:
                # 必需模型
                pair_model_path = find_model_file(models_dir, "relation_scorer_order")
                if pair_model_path:
                    try:
                        booster = lgb.Booster(model_file=str(pair_model_path))
                        model_features = booster.num_feature()
                        schema_features = self.pair_schema_info.feature_count
                        if model_features != schema_features:
                            errors.append(
                                f"relation_scorer_order 特征数不一致: "
                                f"模型={model_features}, schema={schema_features}, "
                                f"schema_version={self.pair_schema_info.schema_version}"
                            )
                        if model_features != EXPECTED_PAIR_FEAT_DIM:
                            warnings.append(
                                f"relation_scorer_order 特征数 {model_features} 与预期 {EXPECTED_PAIR_FEAT_DIM} 不匹配"
                            )
                    except Exception:
                        pass
                
                # 可选模型 - 仅警告
                caption_model_path = find_model_file(models_dir, "relation_scorer_caption")
                if caption_model_path:
                    try:
                        booster = lgb.Booster(model_file=str(caption_model_path))
                        model_features = booster.num_feature()
                        schema_features = self.pair_schema_info.feature_count
                        if model_features != schema_features:
                            warnings.append(
                                f"relation_scorer_caption (可选) 特征数不一致: "
                                f"模型={model_features}, schema={schema_features}"
                            )
                    except Exception:
                        pass

        # 4. 验证 config 路径
        config_path = self.output_dir / "config.default.yaml"
        if config_path.exists():
            try:
                config = load_yaml_safe(config_path)
                models_config = config.get("models", {})
                for model_name in REQUIRED_MODELS:
                    if model_name in models_config:
                        cfg_path = models_config[model_name].get("path", "")
                        if cfg_path:
                            actual_path = self.output_dir / cfg_path
                            if not actual_path.exists():
                                errors.append(f"配置路径不存在: {cfg_path}")
            except Exception as e:
                warnings.append(f"配置解析警告: {str(e)[:100]}")

        # 记录结果
        self.result.warnings.extend(warnings)
        self.result.errors.extend(errors)

        if warnings:
            for w in warnings:
                logger.warning(f"  ⚠ {w}")

        if errors:
            for err in errors:
                logger.error(f"  ✗ {err}")
            return False

        logger.info("  ✓ 验证通过")
        return True

    def _print_summary(self) -> None:
        """打印导出摘要"""
        print("\n" + "=" * 60)
        print("✅ Artifacts 导出成功")
        print("=" * 60)
        print(f"  输出目录: {self.output_dir}")
        print(f"  文件数量: {self.result.files_copied}")
        print(f"  主模型数: {self.result.models_copied}")
        print(f"  兜底模型: {self.result.fallback_models_copied}")
        if self.result.warnings:
            print(f"  警告数量: {len(self.result.warnings)}")
        print("=" * 60)

        # 输出机器可解析的 JSON 摘要
        print("\n[EXPORT_RESULT]")
        print(json.dumps(self.result.to_dict(), ensure_ascii=False))


# ============================================================================
# Merge 命令 - LoRA 权重合并器
# ============================================================================

class LoRAMerger:
    """
    LoRA 权重合并器

    注意：
    - 当前主模型为 LightGBM，不支持 LoRA 合并
    - 此命令为未来视觉大模型（ONNX/PyTorch）预留
    - LightGBM 模型会直接复制，不做合并
    """

    def __init__(self, base_path: str, lora_path: str, output_path: str, alpha: float = 1.0):
        self.base_path = Path(base_path)
        self.lora_path = Path(lora_path)
        self.output_path = Path(output_path)
        self.alpha = alpha

    def merge(self) -> bool:
        """执行合并"""
        logger.info("开始合并 LoRA 权重")
        logger.info(f"  Base: {self.base_path}")
        logger.info(f"  LoRA: {self.lora_path}")
        logger.info(f"  Alpha: {self.alpha}")
        logger.info(f"  Output: {self.output_path}")

        if not self.base_path.exists():
            logger.error(f"基础模型不存在: {self.base_path}")
            return False

        if not self.lora_path.exists():
            logger.error(f"LoRA 权重不存在: {self.lora_path}")
            return False

        try:
            base_ext = self.base_path.suffix.lower()

            if base_ext == ".onnx":
                return self._merge_onnx()
            elif base_ext in [".json", ".txt"]:
                return self._merge_lightgbm()
            elif base_ext in [".pt", ".pth", ".bin"]:
                return self._merge_pytorch()
            else:
                logger.error(f"不支持的模型格式: {base_ext}")
                return False

        except Exception as e:
            logger.error(f"❌ 合并失败: {str(e)[:200]}")
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()
            return False

    def _merge_onnx(self) -> bool:
        """合并 ONNX 模型的 LoRA 权重"""
        try:
            import onnx
            from onnx import numpy_helper

            if not HAS_NUMPY:
                logger.error("需要安装 numpy")
                return False

            logger.info("加载 ONNX 模型...")
            base_model = onnx.load(str(self.base_path))

            logger.info("加载 LoRA 权重...")
            lora_weights = self._load_lora_weights()

            if not lora_weights:
                logger.warning("LoRA 权重为空，直接复制原模型")
                shutil.copy(self.base_path, self.output_path)
                return True

            logger.info(f"合并 {len(lora_weights)} 个权重层...")
            initializers = {init.name: init for init in base_model.graph.initializer}
            merged_count = 0

            for name, lora_data in lora_weights.items():
                if name in initializers:
                    init = initializers[name]
                    original = numpy_helper.to_array(init)

                    if isinstance(lora_data, dict) and "A" in lora_data and "B" in lora_data:
                        A = np.array(lora_data["A"])
                        B = np.array(lora_data["B"])
                        delta = self.alpha * (A @ B)
                    else:
                        delta = self.alpha * np.array(lora_data)

                    if delta.shape != original.shape:
                        delta = delta.reshape(original.shape)

                    merged = original + delta
                    new_init = numpy_helper.from_array(merged.astype(original.dtype), name)

                    for i, init in enumerate(base_model.graph.initializer):
                        if init.name == name:
                            base_model.graph.initializer[i].CopyFrom(new_init)
                            merged_count += 1
                            break

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            onnx.save(base_model, str(self.output_path))

            logger.info(f"✅ ONNX 模型合并成功: {self.output_path}")
            logger.info(f"   合并了 {merged_count} 个权重层")
            return True

        except ImportError:
            logger.error("需要安装 onnx: pip install onnx")
            return False

    def _merge_lightgbm(self) -> bool:
        """LightGBM 模型不支持 LoRA，直接复制"""
        logger.warning("LightGBM 模型不支持 LoRA 合并")
        logger.info("直接复制原模型...")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.base_path, self.output_path)

        logger.info(f"✅ 模型已复制: {self.output_path}")
        return True

    def _merge_pytorch(self) -> bool:
        """合并 PyTorch 模型的 LoRA 权重"""
        try:
            import torch

            logger.info("加载 PyTorch 模型...")
            base_state_dict = torch.load(str(self.base_path), map_location="cpu")

            logger.info("加载 LoRA 权重...")
            lora_state_dict = torch.load(str(self.lora_path), map_location="cpu")

            if not lora_state_dict:
                logger.warning("LoRA 权重为空，直接复制原模型")
                shutil.copy(self.base_path, self.output_path)
                return True

            logger.info("合并权重...")
            merged = base_state_dict.copy()
            merged_count = 0

            for name, lora_data in lora_state_dict.items():
                if ".lora_A" in name:
                    base_name = name.replace(".lora_A", ".weight")
                    lora_b_name = name.replace(".lora_A", ".lora_B")

                    if base_name in merged and lora_b_name in lora_state_dict:
                        A = lora_data
                        B = lora_state_dict[lora_b_name]

                        if len(A.shape) == 2 and len(B.shape) == 2:
                            delta = self.alpha * torch.mm(B, A)
                        else:
                            delta = self.alpha * torch.matmul(B, A)

                        merged[base_name] = merged[base_name] + delta
                        merged_count += 1

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(merged, str(self.output_path))

            logger.info(f"✅ PyTorch 模型合并成功: {self.output_path}")
            logger.info(f"   合并了 {merged_count} 个 LoRA 层")
            return True

        except ImportError:
            logger.error("需要安装 torch: pip install torch")
            return False

    def _load_lora_weights(self) -> Dict[str, Any]:
        """加载 LoRA 权重"""
        ext = self.lora_path.suffix.lower()

        if ext == ".json":
            return load_json(self.lora_path)
        elif ext == ".bin":
            try:
                import torch
                return torch.load(str(self.lora_path), map_location="cpu")
            except ImportError:
                logger.error("需要安装 torch 来加载 .bin 文件")
                return {}
        elif ext == ".npy":
            if not HAS_NUMPY:
                logger.error("需要安装 numpy 来加载 .npy 文件")
                return {}
            data = np.load(str(self.lora_path), allow_pickle=True)
            return data.item() if data.ndim == 0 else dict(data)
        else:
            raise ValueError(f"不支持的 LoRA 权重格式: {ext}")


# ============================================================================
# Validate 命令 - Artifacts 验证器
# ============================================================================

class ArtifactsValidator:
    """
    Artifacts 验证器

    验证内容：
    1. 目录结构完整性
    2. 必需文件存在性
    3. Manifest 格式与内容
    4. Label Map 与预期类别集合一致性（含差异报告）
    5. Feature Schema 格式与版本
    6. 模型文件可加载性
    7. 模型特征数与 Schema 一致性（详细错误信息）
    8. 文件哈希一致性（支持 quick/full 模式）
    9. Config 路径与 Manifest 一致性
    10. Fallback 模型验证
    """

    def __init__(self, artifacts_dir: str, strict: bool = True):
        self.artifacts_dir = Path(artifacts_dir)
        self.strict = strict
        self.result = ValidationResult(passed=True)
        self.manifest: Dict[str, Any] = {}
        self._model_features: Dict[str, int] = {}
        self._schema_features: Dict[str, int] = {}
        self._schema_versions: Dict[str, str] = {}

    def validate(self) -> ValidationResult:
        """执行验证"""
        logger.info(f"{'='*60}")
        logger.info(f"开始验证 artifacts: {self.artifacts_dir}")
        logger.info(f"严格模式: {self.strict}")
        logger.info(f"{'='*60}")

        self._check_directory_exists()
        if not self.result.passed:
            return self.result

        self._check_directory_structure()
        self._check_required_files()
        self._validate_manifest()
        self._validate_label_map()
        self._validate_feature_schema()
        self._validate_models()
        self._validate_model_schema_consistency()
        self._validate_fallback_models()
        self._validate_hashes()
        self._validate_config_paths()
        self._validate_config_manifest_consistency()
        self._print_result()

        return self.result

    def _add_error(self, msg: str) -> None:
        self.result.errors.append(msg)
        self.result.passed = False
        logger.error(f"❌ {msg}")

    def _add_warning(self, msg: str) -> None:
        self.result.warnings.append(msg)
        if self.strict:
            self.result.passed = False
        logger.warning(f"⚠️ {msg}")

    def _add_info(self, msg: str) -> None:
        self.result.info.append(msg)
        logger.info(f"✓ {msg}")

    def _check_directory_exists(self) -> None:
        if not self.artifacts_dir.exists():
            self._add_error(f"Artifacts 目录不存在: {self.artifacts_dir}")
        elif not self.artifacts_dir.is_dir():
            self._add_error(f"路径不是目录: {self.artifacts_dir}")

    def _check_directory_structure(self) -> None:
        required_dirs = ["models"]
        for d in required_dirs:
            dir_path = self.artifacts_dir / d
            if not dir_path.exists():
                self._add_error(f"缺少必需目录: {d}")
            else:
                self._add_info(f"目录存在: {d}")

    def _check_required_files(self) -> None:
        for name, filename in REQUIRED_FILES.items():
            filepath = self.artifacts_dir / filename
            if not filepath.exists():
                self._add_error(f"缺少必需文件: {filename}")
            elif filepath.stat().st_size == 0:
                self._add_error(f"文件为空: {filename}")
            else:
                self._add_info(f"文件存在: {filename}")

    def _validate_manifest(self) -> None:
        manifest_path = self.artifacts_dir / "manifest.json"
        if not manifest_path.exists():
            return

        try:
            self.manifest = load_json(manifest_path)

            ir_version = self.manifest.get("ir_schema_version")
            if ir_version != IR_SCHEMA_VERSION:
                self._add_warning(f"IR schema 版本: {ir_version} (预期 {IR_SCHEMA_VERSION})")
            else:
                self._add_info(f"IR schema 版本正确: {IR_SCHEMA_VERSION}")

            label_set = self.manifest.get("label_set", [])
            if label_set != EXPECTED_LABEL_SET:
                diff = compare_label_maps(label_set, EXPECTED_LABEL_SET)
                if diff["missing"]:
                    self._add_error(f"Manifest label_set 缺少: {diff['missing']}")
                if diff["extra"]:
                    self._add_warning(f"Manifest label_set 多出: {diff['extra']}")
                if diff["order_diff"]:
                    self._add_warning("Manifest label_set 顺序不一致")
            else:
                self._add_info("Label set 正确")

        except Exception as e:
            self._add_error(f"Manifest 解析失败: {str(e)[:100]}")

    def _validate_label_map(self) -> None:
        label_map_path = self.artifacts_dir / "label_map.json"
        if not label_map_path.exists():
            return

        try:
            label_map = load_json(label_map_path)

            if not isinstance(label_map, list):
                self._add_error("label_map 应为列表格式")
                return

            diff = compare_label_maps(label_map, EXPECTED_LABEL_SET)

            if diff["is_match"]:
                self._add_info(f"label_map 验证通过 ({len(label_map)} 类别)")
            else:
                if diff["missing"]:
                    self._add_error(f"label_map 缺少类别: {diff['missing']}")
                if diff["extra"]:
                    self._add_warning(f"label_map 多出类别: {diff['extra']}")
                if diff["order_diff"]:
                    self._add_warning(
                        f"label_map 顺序不一致 "
                        f"(实际: {label_map[:3]}..., 预期: {EXPECTED_LABEL_SET[:3]}...)"
                    )

        except Exception as e:
            self._add_error(f"label_map 验证失败: {str(e)[:100]}")

    def _validate_feature_schema(self) -> None:
        schema_files = [
            ("feature_schema_block.json", "block"),
            ("feature_schema_pair.json", "pair"),
        ]

        for filename, schema_type in schema_files:
            schema_path = self.artifacts_dir / filename
            if not schema_path.exists():
                continue

            schema_info = load_schema_info(schema_path)
            if not schema_info:
                self._add_error(f"{filename}: 无法加载")
                continue

            self._schema_features[schema_type] = schema_info.feature_count
            self._schema_versions[schema_type] = schema_info.schema_version

            if not schema_info.feature_names:
                self._add_error(f"{filename}: 缺少特征定义")
            else:
                self._add_info(
                    f"{filename}: {schema_info.feature_count} 特征, "
                    f"版本 {schema_info.schema_version}"
                )

    def _validate_models(self) -> None:
        models_dir = self.artifacts_dir / "models"
        if not models_dir.exists():
            return

        # 验证必需模型
        for model_name in REQUIRED_MODELS:
            model_path = find_model_file(models_dir, model_name)

            if not model_path:
                self._add_error(f"缺少必需模型: {model_name}")
                continue

            if HAS_LGB:
                try:
                    booster = lgb.Booster(model_file=str(model_path))
                    num_features = booster.num_feature()
                    self._model_features[model_name] = num_features
                    self._add_info(f"{model_name}: 可加载 ({num_features} 特征)")
                except Exception as e:
                    self._add_error(f"{model_name}: 加载失败 - {str(e)[:100]}")
            else:
                self._add_warning(f"{model_name}: lightgbm 未安装，跳过加载验证")

        # 验证可选模型（仅检查是否可加载，不报错）
        for model_name in OPTIONAL_MODELS:
            model_path = find_model_file(models_dir, model_name)

            if not model_path:
                self._add_info(f"{model_name}: 可选模型未找到")
                continue

            if HAS_LGB:
                try:
                    booster = lgb.Booster(model_file=str(model_path))
                    num_features = booster.num_feature()
                    self._model_features[model_name] = num_features
                    self._add_info(f"{model_name}: 可加载 ({num_features} 特征) [可选]")
                except Exception as e:
                    self._add_warning(f"{model_name}: 可选模型加载失败 - {str(e)[:100]}")

    def _validate_model_schema_consistency(self) -> None:
        """验证模型特征数与 Schema 一致性"""
        # block_classifier - 预期 29 维
        if "block_classifier" in self._model_features and "block" in self._schema_features:
            model_f = self._model_features["block_classifier"]
            schema_f = self._schema_features["block"]
            schema_v = self._schema_versions.get("block", "unknown")
            if model_f != schema_f:
                self._add_error(
                    f"block_classifier 特征数不一致: "
                    f"模型特征数={model_f}, schema特征数={schema_f}, "
                    f"schema版本={schema_v}, 预期维度={EXPECTED_BLOCK_FEAT_DIM}"
                )
            else:
                self._add_info(f"block_classifier 特征数一致: {model_f}")
                # 额外检查是否与预期维度匹配
                if model_f != EXPECTED_BLOCK_FEAT_DIM:
                    self._add_warning(
                        f"block_classifier 特征数 {model_f} 与预期维度 {EXPECTED_BLOCK_FEAT_DIM} 不匹配"
                    )

        # relation_scorer_order - 必需，预期 34 维
        if "relation_scorer_order" in self._model_features and "pair" in self._schema_features:
            model_f = self._model_features["relation_scorer_order"]
            schema_f = self._schema_features["pair"]
            schema_v = self._schema_versions.get("pair", "unknown")
            if model_f != schema_f:
                self._add_error(
                    f"relation_scorer_order 特征数不一致: "
                    f"模型特征数={model_f}, schema特征数={schema_f}, "
                    f"schema版本={schema_v}, 预期维度={EXPECTED_PAIR_FEAT_DIM}"
                )
            else:
                self._add_info(f"relation_scorer_order 特征数一致: {model_f}")
                if model_f != EXPECTED_PAIR_FEAT_DIM:
                    self._add_warning(
                        f"relation_scorer_order 特征数 {model_f} 与预期维度 {EXPECTED_PAIR_FEAT_DIM} 不匹配"
                    )

        # relation_scorer_caption - 可选
        if "relation_scorer_caption" in self._model_features and "pair" in self._schema_features:
            model_f = self._model_features["relation_scorer_caption"]
            schema_f = self._schema_features["pair"]
            schema_v = self._schema_versions.get("pair", "unknown")
            if model_f != schema_f:
                self._add_warning(
                    f"relation_scorer_caption 特征数不一致 (可选模型): "
                    f"模型特征数={model_f}, schema特征数={schema_f}"
                )
            else:
                self._add_info(f"relation_scorer_caption 特征数一致: {model_f} [可选]")

    def _validate_fallback_models(self) -> None:
        """验证 fallback 模型"""
        if not self.manifest:
            return

        fallback_models = self.manifest.get("fallback_models", {})
        fallback_dir = self.artifacts_dir / "fallback_models"

        for model_name, model_info in fallback_models.items():
            enabled = model_info.get("enabled", False)
            is_bundled = model_info.get("is_bundled", False)
            model_path = model_info.get("path", "")
            expected_hash = model_info.get("hash", "")
            hash_mode = model_info.get("hash_mode", "full")

            if enabled and is_bundled:
                # 必须存在且 hash 匹配
                if model_path:
                    full_path = self.artifacts_dir / model_path
                    if not full_path.exists():
                        self._add_error(f"fallback 模型 {model_name}: 已启用且 bundled 但文件不存在")
                    elif expected_hash:
                        # 验证 hash（考虑 hash_mode）
                        use_quick = hash_mode == "quick"
                        actual_hash, _ = compute_sha256(full_path, quick_mode=use_quick)
                        if actual_hash != expected_hash:
                            self._add_error(
                                f"fallback 模型 {model_name}: hash 不匹配 "
                                f"(expected={expected_hash[:16]}..., actual={actual_hash[:16]}...)"
                            )
                        else:
                            self._add_info(f"fallback 模型 {model_name}: 验证通过")
                    else:
                        self._add_info(f"fallback 模型 {model_name}: 存在 (无 hash 验证)")
            elif enabled and not is_bundled:
                self._add_warning(f"fallback 模型 {model_name}: 已启用但未 bundled")
            else:
                self._add_info(f"fallback 模型 {model_name}: 未启用")

    def _validate_hashes(self) -> None:
        if not self.manifest:
            return

        hash_checks = [
            ("label_map.json", self.manifest.get("label_map_hash"), "full"),
            ("config.default.yaml", self.manifest.get("config_hash"), "full"),
        ]

        schemas = self.manifest.get("schemas", {})
        if "block" in schemas:
            hash_checks.append(("feature_schema_block.json", schemas["block"].get("hash"), "full"))
        if "pair" in schemas:
            hash_checks.append(("feature_schema_pair.json", schemas["pair"].get("hash"), "full"))

        for filename, expected_hash, expected_mode in hash_checks:
            if not expected_hash:
                continue
            filepath = self.artifacts_dir / filename
            if filepath.exists():
                actual_hash = compute_full_sha256(filepath)
                if actual_hash != expected_hash:
                    self._add_warning(f"{filename}: 哈希不匹配")
                else:
                    self._add_info(f"{filename}: 哈希验证通过")

    def _validate_config_paths(self) -> None:
        """验证 config 中的路径"""
        config_path = self.artifacts_dir / "config.default.yaml"
        if not config_path.exists():
            return

        try:
            config = load_yaml_safe(config_path)
            models_config = config.get("models", {})

            for model_name in REQUIRED_MODELS:
                if model_name in models_config:
                    cfg_path = models_config[model_name].get("path", "")
                    if cfg_path:
                        actual_path = self.artifacts_dir / cfg_path
                        if not actual_path.exists():
                            self._add_error(f"配置路径无效: {cfg_path}")
                        else:
                            self._add_info(f"配置路径有效: {cfg_path}")

            # 验证 fallback 路径
            fallback_config = config.get("fallback_models", {})
            for model_name, model_cfg in fallback_config.items():
                if isinstance(model_cfg, dict):
                    fb_path = model_cfg.get("path", "")
                    is_bundled = model_cfg.get("is_bundled", False)
                    if fb_path and is_bundled:
                        actual_path = self.artifacts_dir / fb_path
                        if not actual_path.exists():
                            self._add_warning(f"fallback 配置路径无效: {fb_path}")

        except Exception as e:
            self._add_warning(f"配置路径验证失败: {str(e)[:100]}")

    def _validate_config_manifest_consistency(self) -> None:
        """验证 config 与 manifest 一致性"""
        if not self.manifest:
            return

        config_path = self.artifacts_dir / "config.default.yaml"
        if not config_path.exists():
            return

        try:
            config = load_yaml_safe(config_path)
            models_config = config.get("models", {})
            manifest_models = {m["name"]: m for m in self.manifest.get("models", [])}

            for model_name in REQUIRED_MODELS:
                config_model = models_config.get(model_name, {})
                manifest_model = manifest_models.get(model_name, {})

                config_path_val = config_model.get("path", "")
                manifest_path_val = manifest_model.get("path", "")

                if config_path_val and manifest_path_val:
                    if config_path_val != manifest_path_val:
                        self._add_warning(
                            f"{model_name}: config 路径({config_path_val}) "
                            f"与 manifest 路径({manifest_path_val}) 不一致"
                        )

        except Exception as e:
            self._add_warning(f"config-manifest 一致性验证失败: {str(e)[:100]}")

    def _print_result(self) -> None:
        print("\n" + "=" * 60)
        print("✅ 验证通过" if self.result.passed else "❌ 验证失败")
        print("=" * 60)
        print(f"  错误: {len(self.result.errors)}")
        print(f"  警告: {len(self.result.warnings)}")
        print(f"  通过: {len(self.result.info)}")

        if self.result.errors:
            print("\n错误详情:")
            for err in self.result.errors:
                print(f"  ✗ {err}")

        if self.result.warnings:
            print("\n警告详情:")
            for warn in self.result.warnings:
                print(f"  ⚠ {warn}")

        print("=" * 60)

        # 输出机器可解析的结果
        print("\n[VALIDATE_RESULT]")
        print(json.dumps(self.result.to_dict(), ensure_ascii=False))


# ============================================================================
# Report 命令 - 部署自检报告生成器
# ============================================================================

class DeploymentReporter:
    """部署自检报告生成器（含 dry-run 测试）"""

    def __init__(self, artifacts_dir: str, output_path: Optional[str] = None):
        self.artifacts_dir = Path(artifacts_dir)
        self.output_path = Path(output_path) if output_path else None

    def generate(self) -> DeploymentReport:
        logger.info("生成部署自检报告...")

        validator = ArtifactsValidator(str(self.artifacts_dir), strict=False)
        validation_result = validator.validate()

        manifest = self._load_manifest()
        config_summary = self._get_config_summary()
        dry_run_stats = self._run_dry_test()

        report = DeploymentReport(
            timestamp=datetime.now().isoformat(),
            artifacts_path=str(self.artifacts_dir.absolute()),
            ir_schema_version=manifest.get("ir_schema_version", "unknown"),
            feature_schema_version=manifest.get("feature_schema_version", "unknown"),
            profiles_supported=manifest.get("profiles_supported", []),
            validation_passed=validation_result.passed,
            models=manifest.get("models", []),
            schemas=manifest.get("schemas", {}),
            label_map_info={
                "count": manifest.get("label_count", 0),
                "hash": manifest.get("label_map_hash", ""),
            },
            fallback_models=manifest.get("fallback_models", {}),
            config_summary=config_summary,
            errors=validation_result.errors,
            warnings=validation_result.warnings,
            dry_run_stats=asdict(dry_run_stats) if dry_run_stats else None,
        )

        report_dict = asdict(report)

        if self.output_path:
            save_json(report_dict, self.output_path)
            logger.info(f"报告已保存: {self.output_path}")
        else:
            print("\n" + "=" * 60)
            print("部署自检报告")
            print("=" * 60)
            print(json.dumps(report_dict, indent=2, ensure_ascii=False))

        return report

    def _load_manifest(self) -> Dict[str, Any]:
        manifest_path = self.artifacts_dir / "manifest.json"
        return load_json(manifest_path) if manifest_path.exists() else {}

    def _get_config_summary(self) -> Dict[str, Any]:
        config_path = self.artifacts_dir / "config.default.yaml"
        if not config_path.exists():
            return {}

        config = load_yaml_safe(config_path)
        summary = {
            "default_profile": config.get("profile", "balanced"),
            "profiles": list(config.get("profiles", {}).keys()),
        }

        for profile_name in SUPPORTED_PROFILES:
            resolved = resolve_config(config, profile_name)
            decode = resolved.get("decode", {})
            fallback = decode.get("fallback_trigger", {})
            runtime = resolved.get("runtime", {})

            summary[f"{profile_name}_params"] = {
                "k_order": decode.get("k_order"),
                "k_caption": decode.get("k_caption"),
                "fallback_enabled": fallback.get("enabled", False),
                "use_fp16": runtime.get("use_fp16", True),
            }

        return summary

    def _run_dry_test(self) -> Optional[DryRunResult]:
        """执行 dry-run 测试"""
        result = DryRunResult(passed=True)
        start_total = time.time()
        models_dir = self.artifacts_dir / "models"
        fallback_dir = self.artifacts_dir / "fallback_models"

        # 加载 schema
        block_schema = load_schema_info(self.artifacts_dir / "feature_schema_block.json")
        pair_schema = load_schema_info(self.artifacts_dir / "feature_schema_pair.json")

        # 测试必需模型
        for model_name in REQUIRED_MODELS:
            model_path = find_model_file(models_dir, model_name)
            if not model_path:
                result.errors.append(f"{model_name}: 必需模型文件不存在")
                result.passed = False
                continue

            start = time.time()
            try:
                if not HAS_LGB:
                    result.warnings.append(f"{model_name}: lightgbm 未安装")
                    result.models_load_time_ms[model_name] = -1
                    continue

                booster = lgb.Booster(model_file=str(model_path))
                result.models_load_time_ms[model_name] = round((time.time() - start) * 1000, 2)

                # 确定特征数
                if "block" in model_name and block_schema:
                    expected_features = block_schema.feature_count
                elif pair_schema:
                    expected_features = pair_schema.feature_count
                else:
                    expected_features = booster.num_feature()

                # 测试预测
                if HAS_NUMPY:
                    dummy_input = np.zeros((1, expected_features), dtype=np.float32)
                    output = booster.predict(dummy_input)
                    result.prediction_test_passed[model_name] = True
                else:
                    result.warnings.append(f"{model_name}: numpy 未安装，跳过预测测试")
                    result.prediction_test_passed[model_name] = False

            except Exception as e:
                result.models_load_time_ms[model_name] = -1
                result.errors.append(f"{model_name}: {str(e)[:100]}")
                result.passed = False

        # 测试 fallback 模型
        manifest = self._load_manifest()
        fallback_models = manifest.get("fallback_models", {})

        for model_name, model_info in fallback_models.items():
            enabled = model_info.get("enabled", False)
            is_bundled = model_info.get("is_bundled", False)
            model_path_str = model_info.get("path", "")
            backend = model_info.get("backend", "onnx")

            status = {
                "enabled": enabled,
                "is_bundled": is_bundled,
                "load_time_ms": -1,
                "load_success": False,
                "error": None,
            }

            if enabled and is_bundled and model_path_str:
                model_path = self.artifacts_dir / model_path_str
                if model_path.exists():
                    start = time.time()
                    try:
                        if backend == "onnx" or model_path.suffix == ".onnx":
                            if HAS_ORT:
                                session = ort.InferenceSession(str(model_path))
                                status["load_time_ms"] = round((time.time() - start) * 1000, 2)
                                status["load_success"] = True
                            else:
                                status["error"] = "onnxruntime 未安装"
                                result.warnings.append(f"fallback {model_name}: onnxruntime 未安装")
                        elif backend == "lightgbm":
                            if HAS_LGB:
                                booster = lgb.Booster(model_file=str(model_path))
                                status["load_time_ms"] = round((time.time() - start) * 1000, 2)
                                status["load_success"] = True
                            else:
                                status["error"] = "lightgbm 未安装"
                        else:
                            status["error"] = f"未知后端: {backend}"
                    except Exception as e:
                        status["error"] = str(e)[:100]
                        result.warnings.append(f"fallback {model_name}: {status['error']}")
                else:
                    status["error"] = "文件不存在"

            result.fallback_models_status[model_name] = status

        # 测试加载 schema
        schema_files = ["label_map.json", "feature_schema_block.json", "feature_schema_pair.json"]
        for filename in schema_files:
            filepath = self.artifacts_dir / filename
            if filepath.exists():
                start = time.time()
                try:
                    load_json(filepath)
                    result.schema_load_time_ms[filename] = round((time.time() - start) * 1000, 2)
                except Exception as e:
                    result.schema_load_time_ms[filename] = -1
                    result.errors.append(f"{filename}: {str(e)[:100]}")
                    result.passed = False

        # 测试可选模型
        for model_name in OPTIONAL_MODELS:
            model_path = find_model_file(models_dir, model_name)
            if model_path:
                start = time.time()
                try:
                    if HAS_LGB:
                        booster = lgb.Booster(model_file=str(model_path))
                        result.models_load_time_ms[model_name] = round((time.time() - start) * 1000, 2)
                        
                        if pair_schema:
                            expected_features = pair_schema.feature_count
                        else:
                            expected_features = booster.num_feature()
                        
                        if HAS_NUMPY:
                            dummy_input = np.zeros((1, expected_features), dtype=np.float32)
                            output = booster.predict(dummy_input)
                            result.prediction_test_passed[model_name] = True
                except Exception as e:
                    result.models_load_time_ms[model_name] = -1
                    result.warnings.append(f"{model_name} (可选): {str(e)[:100]}")

        result.total_time_ms = round((time.time() - start_total) * 1000, 2)

        if result.errors:
            logger.warning(f"Dry-run 测试发现 {len(result.errors)} 个错误")
        else:
            logger.info(f"Dry-run 测试通过，总耗时 {result.total_time_ms}ms")

        return result


# ============================================================================
# CLI 主入口
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merge_lora.py",
        description="Export / Merge / Validate / Report 工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 导出 artifacts 包（必须指定训练产物目录）
  python merge_lora.py export -c config.yaml -o artifacts/ -m models/ -s train_output/

  # 验证 artifacts 完整性
  python merge_lora.py validate -a artifacts/

  # 严格模式验证
  python merge_lora.py validate -a artifacts/ --strict

  # 生成部署自检报告
  python merge_lora.py report -a artifacts/ -o report.json

环境变量:
  MERGE_LORA_DEBUG=1  启用调试模式，显示完整堆栈

注意事项:
  - export 必须指定 --schema-dir，从训练产物复制 schema
  - validate 会严格检查模型特征数与 schema 一致性
  - report 包含 dry-run 测试，验证模型可加载和预测
  - 支持 fallback 模型的条件复制和验证
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # export
    export_parser = subparsers.add_parser("export", help="导出 artifacts 包")
    export_parser.add_argument("--config", "-c", required=True, help="配置文件路径")
    export_parser.add_argument("--output", "-o", required=True, help="输出目录")
    export_parser.add_argument("--models-dir", "-m", required=True, help="模型文件目录")
    export_parser.add_argument("--schema-dir", "-s", required=True, help="Schema 文件目录（训练产物）")
    export_parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    # merge
    merge_parser = subparsers.add_parser("merge", help="合并 LoRA 权重")
    merge_parser.add_argument("--base", "-b", required=True, help="基础模型路径")
    merge_parser.add_argument("--lora", "-l", required=True, help="LoRA 权重路径")
    merge_parser.add_argument("--output", "-o", required=True, help="输出模型路径")
    merge_parser.add_argument("--alpha", "-a", type=float, default=1.0, help="LoRA 缩放系数")
    merge_parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    # validate
    validate_parser = subparsers.add_parser("validate", help="验证 artifacts 完整性")
    validate_parser.add_argument("--artifacts", "-a", required=True, help="Artifacts 目录")
    validate_parser.add_argument("--strict", action="store_true", help="严格模式")
    validate_parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    # report
    report_parser = subparsers.add_parser("report", help="生成部署自检报告")
    report_parser.add_argument("--artifacts", "-a", required=True, help="Artifacts 目录")
    report_parser.add_argument("--output", "-o", help="报告输出路径")
    report_parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # 设置日志级别
    verbose = getattr(args, "verbose", False)
    if verbose:
        setup_logging(verbose=True)

    success = True

    try:
        if args.command == "export":
            exporter = ArtifactsExporter(
                config_path=args.config,
                output_dir=args.output,
                models_dir=args.models_dir,
                schema_dir=args.schema_dir
            )
            result = exporter.export()
            success = result.success

        elif args.command == "merge":
            merger = LoRAMerger(
                base_path=args.base,
                lora_path=args.lora,
                output_path=args.output,
                alpha=args.alpha
            )
            success = merger.merge()

        elif args.command == "validate":
            validator = ArtifactsValidator(
                artifacts_dir=args.artifacts,
                strict=args.strict
            )
            result = validator.validate()
            success = result.passed

        elif args.command == "report":
            reporter = DeploymentReporter(
                artifacts_dir=args.artifacts,
                output_path=args.output
            )
            report = reporter.generate()
            success = report.validation_passed

    except KeyboardInterrupt:
        logger.info("操作已取消")
        sys.exit(130)
    except Exception as e:
        logger.error(f"执行失败: {str(e)[:200]}")
        if DEBUG_MODE:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
