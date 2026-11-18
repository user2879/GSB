"""校对配置 YAML 加载与严格校验。"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import yaml

from .errors import ProofreadError, E_CONFIG_INVALID

_REQUIRED = (
    "ngram_order",
    "similarity_threshold",
    "fixed_point_bits",
    "quant_levels",
    "aggregation_weights",
    "precision",
    "tolerance",
    "max_workers",
    "snapshot_interval",
    "decision_threshold",
)


@dataclass(frozen=True)
class AppConfig:
    ngram_order: int
    similarity_threshold: float
    fixed_point_bits: int
    quant_levels: int
    aggregation_weights: Dict[str, float]
    precision: str
    tolerance: float
    max_workers: int
    snapshot_interval: int
    decision_threshold: float

    def to_fingerprint_obj(self) -> Dict[str, Any]:
        return asdict(self)

    def with_updates(self, **changes: Any) -> "AppConfig":
        data = asdict(self)
        data.update(changes)
        return build_config(data)


def _require_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProofreadError(E_CONFIG_INVALID, f"配置字段 {name} 必须是数值，实际为 {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ProofreadError(E_CONFIG_INVALID, f"配置字段 {name} 必须是有限数")
    return number


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProofreadError(E_CONFIG_INVALID, f"配置字段 {name} 必须是正整数，实际为 {value!r}")
    return value


def build_config(raw: Dict[str, Any]) -> AppConfig:
    if not isinstance(raw, dict):
        raise ProofreadError(E_CONFIG_INVALID, "配置根节点必须是映射")
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise ProofreadError(E_CONFIG_INVALID, f"配置字段缺失: {', '.join(sorted(missing))}")

    ngram_order = _require_positive_int("ngram_order", raw["ngram_order"])
    fixed_point_bits = _require_positive_int("fixed_point_bits", raw["fixed_point_bits"])
    quant_levels = _require_positive_int("quant_levels", raw["quant_levels"])
    max_workers = _require_positive_int("max_workers", raw["max_workers"])
    snapshot_interval = _require_positive_int("snapshot_interval", raw["snapshot_interval"])

    similarity_threshold = _require_number("similarity_threshold", raw["similarity_threshold"])
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ProofreadError(E_CONFIG_INVALID, "similarity_threshold 必须落在 [0, 1]")

    tolerance = _require_number("tolerance", raw["tolerance"])
    if tolerance <= 0:
        raise ProofreadError(E_CONFIG_INVALID, "tolerance 必须为正数")
    decision_threshold = _require_number("decision_threshold", raw["decision_threshold"])

    precision = raw["precision"]
    if precision not in ("float32", "float64"):
        raise ProofreadError(E_CONFIG_INVALID, "precision 仅支持 float32 / float64")

    weights = raw["aggregation_weights"]
    if not isinstance(weights, dict) or not weights:
        raise ProofreadError(E_CONFIG_INVALID, "aggregation_weights 必须是非空映射")
    norm_weights: Dict[str, float] = {}
    weight_sum = 0.0
    for key, value in weights.items():
        number = _require_number(f"aggregation_weights.{key}", value)
        if number < 0:
            raise ProofreadError(E_CONFIG_INVALID, f"aggregation_weights.{key} 不能为负")
        norm_weights[key] = number
        weight_sum += number
    if weight_sum <= 0:
        raise ProofreadError(E_CONFIG_INVALID, "aggregation_weights 权重之和必须为正")

    return AppConfig(
        ngram_order=ngram_order,
        similarity_threshold=similarity_threshold,
        fixed_point_bits=fixed_point_bits,
        quant_levels=quant_levels,
        aggregation_weights=norm_weights,
        precision=precision,
        tolerance=tolerance,
        max_workers=max_workers,
        snapshot_interval=snapshot_interval,
        decision_threshold=decision_threshold,
    )


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ProofreadError(E_CONFIG_INVALID, f"配置文件不存在: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProofreadError(E_CONFIG_INVALID, f"YAML 解析失败: {exc}") from exc
    return build_config(raw)
