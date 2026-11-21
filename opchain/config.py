"""Execution configuration parsing and precision/quantization descriptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .errors import E_CONFIG_INVALID, BusinessError
from .util import sha256_json


@dataclass(frozen=True)
class PrecisionSpec:
    tier: str
    dtype: np.dtype
    decimal_places: int


PRECISION_SPECS: dict[str, PrecisionSpec] = {
    "fp64": PrecisionSpec("fp64", np.dtype("float64"), 9),
    "fp32": PrecisionSpec("fp32", np.dtype("float32"), 6),
    "fp16": PrecisionSpec("fp16", np.dtype("float16"), 3),
}

PRECISION_ALIASES = {
    "fp64": "fp64",
    "float64": "fp64",
    "f64": "fp64",
    "double": "fp64",
    "high": "fp64",
    "fp32": "fp32",
    "float32": "fp32",
    "f32": "fp32",
    "single": "fp32",
    "medium": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "f16": "fp16",
    "half": "fp16",
    "low": "fp16",
}


@dataclass(frozen=True)
class QuantizationSpec:
    tier: str
    scale: float | None


QUANTIZATION_SPECS: dict[str, QuantizationSpec] = {
    "none": QuantizationSpec("none", None),
    "q8": QuantizationSpec("q8", 16.0),
    "q12": QuantizationSpec("q12", 256.0),
    "q16": QuantizationSpec("q16", 4096.0),
    "int8": QuantizationSpec("int8", 32.0),
    "int16": QuantizationSpec("int16", 1024.0),
}

QUANTIZATION_ALIASES = {
    "none": "none",
    "off": "none",
    "disabled": "none",
    **{name: name for name in QUANTIZATION_SPECS if name != "none"},
}


@dataclass(frozen=True)
class ExecutionConfig:
    precision_tier: str = "fp64"
    quantization_tier: str = "none"
    thread_pool_size: int = 1
    recompute_budget: int = 1024
    snapshot_interval: int = 1
    tolerance: float = 1e-9
    fusion_strategy: str = "none"
    raw: dict[str, Any] | None = None

    @property
    def precision(self) -> PrecisionSpec:
        return PRECISION_SPECS[self.precision_tier]

    @property
    def quantization(self) -> QuantizationSpec:
        return QUANTIZATION_SPECS[self.quantization_tier]

    @property
    def fusion_enabled(self) -> bool:
        return self.fusion_strategy in {"auto", "fused", "enabled", "on"}

    def cache_material(self) -> dict[str, Any]:
        return {
            "precision_tier": self.precision_tier,
            "quantization_tier": self.quantization_tier,
            "fusion_strategy": self.fusion_strategy,
        }

    def config_hash(self) -> str:
        return sha256_json(self.to_json())

    def as_record(self) -> dict[str, Any]:
        return asdict(self) | {"raw": self.raw or {}}

    def to_json(self) -> dict[str, Any]:
        return {
            "precision_tier": self.precision_tier,
            "quantization_tier": self.quantization_tier,
            "thread_pool_size": self.thread_pool_size,
            "recompute_budget": self.recompute_budget,
            "snapshot_interval": self.snapshot_interval,
            "tolerance": self.tolerance,
            "fusion_strategy": self.fusion_strategy,
        }


def _positive_int(value: Any, field_name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise BusinessError(E_CONFIG_INVALID, f"{field_name} must be an integer", stage="config")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, f"{field_name} must be an integer", stage="config") from exc
    minimum = 0 if allow_zero else 1
    if parsed < minimum:
        raise BusinessError(E_CONFIG_INVALID, f"{field_name} must be >= {minimum}", stage="config")
    return parsed


def parse_config(raw: dict[str, Any] | ExecutionConfig | None = None) -> ExecutionConfig:
    if isinstance(raw, ExecutionConfig):
        return raw
    source = dict(raw or {})
    precision_raw = source.get("precision_tier", source.get("precision", "fp64"))
    precision_tier = PRECISION_ALIASES.get(str(precision_raw).lower())
    if precision_tier is None:
        raise BusinessError(E_CONFIG_INVALID, f"unsupported precision tier: {precision_raw!r}", stage="config")

    quantization_raw = source.get("quantization_tier", source.get("quantization", "none"))
    quantization_tier = QUANTIZATION_ALIASES.get(str(quantization_raw).lower())
    if quantization_tier is None:
        raise BusinessError(E_CONFIG_INVALID, f"unsupported quantization tier: {quantization_raw!r}", stage="config")

    fusion_raw: Any = source.get("fusion_strategy", None)
    if fusion_raw is None and "fusion" in source:
        fusion_raw = source["fusion"]
    if fusion_raw is None and "fusion_enabled" in source:
        fusion_raw = "auto" if source["fusion_enabled"] else "none"
    if isinstance(fusion_raw, bool):
        fusion_raw = "auto" if fusion_raw else "none"
    fusion_strategy = str(fusion_raw or "none").lower()
    if fusion_strategy in {"disabled", "off", "false"}:
        fusion_strategy = "none"
    if fusion_strategy in {"true", "on", "enabled"}:
        fusion_strategy = "auto"
    if fusion_strategy not in {"none", "auto", "fused"}:
        raise BusinessError(E_CONFIG_INVALID, f"unsupported fusion strategy: {fusion_raw!r}", stage="config")

    thread_pool_size = _positive_int(
        source.get("thread_pool_size", source.get("thread_pool_concurrency", source.get("threads", 1))),
        "thread_pool_size",
    )
    recompute_budget = _positive_int(
        source.get("recompute_budget", source.get("node_recompute_budget", 1024)),
        "recompute_budget",
        allow_zero=True,
    )
    snapshot_interval = _positive_int(
        source.get("snapshot_interval", source.get("snapshot_commit_interval", 1)),
        "snapshot_interval",
    )

    tolerance_raw = source.get("tolerance", 1e-9)
    try:
        tolerance = float(tolerance_raw)
    except (TypeError, ValueError) as exc:
        raise BusinessError(E_CONFIG_INVALID, "tolerance must be a number", stage="config") from exc
    if tolerance < 0 or not np.isfinite(tolerance):
        raise BusinessError(E_CONFIG_INVALID, "tolerance must be finite and non-negative", stage="config")

    return ExecutionConfig(
        precision_tier=precision_tier,
        quantization_tier=quantization_tier,
        thread_pool_size=thread_pool_size,
        recompute_budget=recompute_budget,
        snapshot_interval=snapshot_interval,
        tolerance=tolerance,
        fusion_strategy=fusion_strategy,
        raw=source,
    )
