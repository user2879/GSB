"""Configuration model and validation."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .models import Issue


SCHEME_ALIASES = {
    "rope": "rope",
    "rotary": "rope",
    "旋转位置编码": "rope",
    "rotary_position_encoding": "rope",
    "linear": "linear",
    "linear_interpolation": "linear",
    "线性内插": "linear",
    "ntk": "ntk",
    "ntk-aware": "ntk",
    "ntk_aware": "ntk",
    "NTK 感知缩放": "ntk",
}

SCHEME_DISPLAY = {
    "rope": "旋转位置编码",
    "linear": "线性内插",
    "ntk": "NTK 感知缩放",
}

PRECISION_ALIASES = {
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
    "single": "fp32",
    "fp64": "fp64",
    "float64": "fp64",
    "double": "fp64",
}

QUANTIZATION_ALIASES = {
    "fine": "fine",
    "高": "fine",
    "high": "fine",
    "balanced": "balanced",
    "中": "balanced",
    "medium": "balanced",
    "coarse": "coarse",
    "低": "coarse",
    "low": "coarse",
}

TOLERANCE_ALIASES = {
    "position_vector": "position_vector",
    "position": "position_vector",
    "pe": "position_vector",
    "dot_score": "dot_score",
    "score": "dot_score",
    "dot": "dot_score",
    "attention_weight": "attention_weight",
    "attention": "attention_weight",
    "weight": "attention_weight",
}


@dataclass(frozen=True)
class ToleranceConfig:
    position_vector: float
    dot_score: float
    attention_weight: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class WorkbenchConfig:
    scheme: str
    base_frequency: float
    train_context_length: int
    extrapolation_target_length: int
    dimension: int
    scale_factor: float
    precision_level: str
    precision_levels: tuple[str, ...]
    dimension_block_size: int
    fixed_decimal_bits: int
    fixed_decimal_widths: tuple[int, ...]
    quantization_level: str
    quantization_levels: tuple[str, ...]
    tolerances: ToleranceConfig
    tolerance_candidates: dict[str, tuple[float, ...]]
    thread_pool_size: int
    point_budget: int
    sample_quota: int
    snapshot_interval: int
    scheme_candidates: tuple[str, ...] = field(default=("rope", "linear", "ntk"))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["tolerances"] = self.tolerances.to_dict()
        out["tolerance_candidates"] = {
            key: list(values) for key, values in sorted(self.tolerance_candidates.items())
        }
        for key in (
            "precision_levels",
            "fixed_decimal_widths",
            "quantization_levels",
            "scheme_candidates",
        ):
            out[key] = list(out[key])
        return out

    def with_changes(self, patch: dict[str, Any]) -> "WorkbenchConfig":
        return replace(self, **patch)

    @property
    def scheme_display(self) -> str:
        return SCHEME_DISPLAY[self.scheme]

    @property
    def attention_scale(self) -> float:
        return 1.0 / math.sqrt(self.dimension)


CONFIG_FIELDS = {
    "scheme",
    "base_frequency",
    "train_context_length",
    "extrapolation_target_length",
    "dimension",
    "scale_factor",
    "precision_level",
    "precision_levels",
    "dimension_block_size",
    "fixed_decimal_bits",
    "fixed_decimal_widths",
    "quantization_level",
    "quantization_levels",
    "tolerances",
    "tolerance_candidates",
    "thread_pool_size",
    "point_budget",
    "sample_quota",
    "snapshot_interval",
    "scheme_candidates",
}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_finite(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value)) and float(value) > 0


def _normalize_enum(value: Any, aliases: dict[str, Any], allowed: Iterable[Any]) -> Any | None:
    if not isinstance(value, str):
        return None
    normalized = aliases.get(value, aliases.get(value.strip(), value))
    if normalized in allowed:
        return normalized
    return None


def _config_issue(code: str, field_name: str, message: str) -> Issue:
    return Issue(code=code, sample_id=None, field=field_name, message=message)


def validate_config_dict(raw: Any) -> tuple[WorkbenchConfig | None, list[Issue]]:
    """Validate a decoded configuration object.

    Validation is intentionally field-by-field so one report can list every
    configuration defect rather than stopping at the first one.
    """
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        return None, [_config_issue("E_CONFIG_OBJECT", "$", "configuration must be a JSON object")]

    unknown = sorted(set(raw) - CONFIG_FIELDS)
    for key in unknown:
        issues.append(_config_issue("E_CONFIG_UNKNOWN_FIELD", key, "unknown configuration field"))

    def issue(code: str, field_name: str, message: str) -> None:
        issues.append(_config_issue(code, field_name, message))

    scheme = _normalize_enum(raw.get("scheme"), SCHEME_ALIASES, SCHEME_DISPLAY)
    if scheme is None:
        issue("E_CONFIG_SCHEME", "scheme", "scheme must be rope, linear, or ntk")

    raw_schemes = raw.get("scheme_candidates", ["rope", "linear", "ntk"])
    scheme_candidates: list[str] = []
    if not isinstance(raw_schemes, list) or not raw_schemes:
        issue("E_CONFIG_SCHEME_CANDIDATES", "scheme_candidates", "must be a non-empty list")
    else:
        for index, item in enumerate(raw_schemes):
            normalized = _normalize_enum(item, SCHEME_ALIASES, SCHEME_DISPLAY)
            if normalized is None:
                issue(
                    "E_CONFIG_SCHEME_CANDIDATES",
                    f"scheme_candidates[{index}]",
                    "unsupported scheme candidate",
                )
            elif normalized not in scheme_candidates:
                scheme_candidates.append(normalized)

    base_frequency = raw.get("base_frequency")
    if not _positive_finite(base_frequency):
        issue("E_CONFIG_BASE_FREQUENCY", "base_frequency", "must be positive and finite")

    train_context_length = raw.get("train_context_length")
    if not _is_positive_int(train_context_length):
        issue("E_CONFIG_TRAIN_LENGTH", "train_context_length", "must be a positive integer")

    extrapolation_target_length = raw.get("extrapolation_target_length")
    if not _is_positive_int(extrapolation_target_length):
        issue("E_CONFIG_TARGET_LENGTH", "extrapolation_target_length", "must be a positive integer")
    elif _is_positive_int(train_context_length) and train_context_length >= extrapolation_target_length:
        issue(
            "E_CONFIG_LENGTH_ORDER",
            "train_context_length/extrapolation_target_length",
            "train_context_length must be smaller than extrapolation_target_length",
        )

    dimension = raw.get("dimension")
    if not _is_positive_int(dimension):
        issue("E_CONFIG_DIMENSION", "dimension", "must be a positive integer")
    elif dimension % 2 != 0:
        issue("E_CONFIG_DIMENSION_EVEN", "dimension", "dimension must be even")

    scale_factor = raw.get("scale_factor")
    if not _positive_finite(scale_factor):
        issue("E_CONFIG_SCALE_FACTOR", "scale_factor", "must be positive and finite")

    raw_precision_levels = raw.get("precision_levels", ["fp16", "fp32", "fp64"])
    precision_levels: list[str] = []
    if not isinstance(raw_precision_levels, list) or not raw_precision_levels:
        issue("E_CONFIG_PRECISION_LEVELS", "precision_levels", "must be a non-empty list")
    else:
        for index, item in enumerate(raw_precision_levels):
            normalized = _normalize_enum(item, PRECISION_ALIASES, {"fp16", "fp32", "fp64"})
            if normalized is None:
                issue(
                    "E_CONFIG_PRECISION_LEVELS",
                    f"precision_levels[{index}]",
                    "unsupported precision level",
                )
            elif normalized not in precision_levels:
                precision_levels.append(normalized)
    precision_level = _normalize_enum(
        raw.get("precision_level"), PRECISION_ALIASES, {"fp16", "fp32", "fp64"}
    )
    if precision_level is None:
        issue("E_CONFIG_PRECISION_LEVEL", "precision_level", "unsupported precision level")
    elif precision_levels and precision_level not in precision_levels:
        issue(
            "E_CONFIG_PRECISION_NOT_ALLOWED",
            "precision_level",
            "precision_level is not present in precision_levels",
        )

    raw_widths = raw.get("fixed_decimal_widths")
    widths: list[int] = []
    if not isinstance(raw_widths, list) or not raw_widths:
        issue("E_CONFIG_FIXED_WIDTHS", "fixed_decimal_widths", "must be a non-empty list")
    else:
        for index, item in enumerate(raw_widths):
            if not _is_positive_int(item):
                issue(
                    "E_CONFIG_FIXED_WIDTHS",
                    f"fixed_decimal_widths[{index}]",
                    "fractional widths must be positive integers",
                )
            elif item not in widths:
                widths.append(item)
        widths.sort()

    fixed_decimal_bits = raw.get("fixed_decimal_bits")
    if not _is_positive_int(fixed_decimal_bits):
        issue("E_CONFIG_FIXED_WIDTH", "fixed_decimal_bits", "must be a positive integer")
    elif widths and fixed_decimal_bits not in widths:
        issue(
            "E_CONFIG_FIXED_WIDTH_NOT_ALLOWED",
            "fixed_decimal_bits",
            "fixed_decimal_bits is not present in fixed_decimal_widths",
        )

    raw_quant_levels = raw.get("quantization_levels")
    quant_levels: list[str] = []
    if not isinstance(raw_quant_levels, list) or not raw_quant_levels:
        issue("E_CONFIG_QUANTIZATION_LEVELS", "quantization_levels", "must be a non-empty list")
    else:
        for index, item in enumerate(raw_quant_levels):
            normalized = _normalize_enum(item, QUANTIZATION_ALIASES, {"fine", "balanced", "coarse"})
            if normalized is None:
                issue(
                    "E_CONFIG_QUANTIZATION_LEVELS",
                    f"quantization_levels[{index}]",
                    "unsupported quantization level",
                )
            elif normalized not in quant_levels:
                quant_levels.append(normalized)
    quantization_level = _normalize_enum(
        raw.get("quantization_level"),
        QUANTIZATION_ALIASES,
        {"fine", "balanced", "coarse"},
    )
    if quantization_level is None:
        issue("E_CONFIG_QUANTIZATION_LEVEL", "quantization_level", "unsupported quantization level")
    elif quant_levels and quantization_level not in quant_levels:
        issue(
            "E_CONFIG_QUANTIZATION_NOT_ALLOWED",
            "quantization_level",
            "quantization_level is not present in quantization_levels",
        )

    raw_tolerances = raw.get("tolerances")
    tolerances: ToleranceConfig | None = None
    if not isinstance(raw_tolerances, dict):
        issue("E_CONFIG_TOLERANCES", "tolerances", "must be an object")
    else:
        canonical: dict[str, float] = {}
        for key, value in raw_tolerances.items():
            canonical_key = TOLERANCE_ALIASES.get(key)
            if canonical_key is None:
                issue("E_CONFIG_TOLERANCE_FIELD", f"tolerances.{key}", "unknown tolerance field")
            elif not _positive_finite(value):
                issue(
                    "E_CONFIG_TOLERANCE_VALUE",
                    f"tolerances.{canonical_key}",
                    "tolerance must be positive and finite",
                )
            elif canonical_key in canonical:
                issue(
                    "E_CONFIG_TOLERANCE_DUPLICATE",
                    f"tolerances.{canonical_key}",
                    "tolerance is specified more than once",
                )
            else:
                canonical[canonical_key] = float(value)
        for required in ("position_vector", "dot_score", "attention_weight"):
            if required not in canonical:
                issue(
                    "E_CONFIG_TOLERANCE_MISSING",
                    f"tolerances.{required}",
                    "required tolerance is missing",
                )
        if len(canonical) == 3:
            tolerances = ToleranceConfig(
                position_vector=canonical["position_vector"],
                dot_score=canonical["dot_score"],
                attention_weight=canonical["attention_weight"],
            )

    raw_tol_candidates = raw.get("tolerance_candidates", {})
    tolerance_candidates: dict[str, tuple[float, ...]] = {}
    if not isinstance(raw_tol_candidates, dict):
        issue("E_CONFIG_TOLERANCE_CANDIDATES", "tolerance_candidates", "must be an object")
    else:
        for key, values in raw_tol_candidates.items():
            canonical_key = TOLERANCE_ALIASES.get(key)
            if canonical_key is None:
                issue(
                    "E_CONFIG_TOLERANCE_CANDIDATES",
                    f"tolerance_candidates.{key}",
                    "unknown tolerance field",
                )
                continue
            if not isinstance(values, list) or not values:
                issue(
                    "E_CONFIG_TOLERANCE_CANDIDATES",
                    f"tolerance_candidates.{canonical_key}",
                    "must be a non-empty list",
                )
                continue
            valid_values: list[float] = []
            for index, value in enumerate(values):
                if not _positive_finite(value):
                    issue(
                        "E_CONFIG_TOLERANCE_CANDIDATES",
                        f"tolerance_candidates.{canonical_key}[{index}]",
                        "candidate tolerance must be positive and finite",
                    )
                elif float(value) not in valid_values:
                    valid_values.append(float(value))
            tolerance_candidates[canonical_key] = tuple(sorted(valid_values))

    dimension_block_size = raw.get("dimension_block_size")
    if not _is_positive_int(dimension_block_size):
        issue("E_CONFIG_BLOCK_SIZE", "dimension_block_size", "must be a positive integer")
    elif _is_positive_int(dimension) and dimension_block_size > dimension:
        issue(
            "E_CONFIG_BLOCK_SIZE",
            "dimension_block_size",
            "dimension_block_size must not exceed dimension",
        )

    thread_pool_size = raw.get("thread_pool_size")
    if not _is_positive_int(thread_pool_size):
        issue("E_CONFIG_THREAD_POOL_SIZE", "thread_pool_size", "must be a positive integer")

    point_budget = raw.get("point_budget")
    if not _is_positive_int(point_budget):
        issue("E_CONFIG_POINT_BUDGET", "point_budget", "must be a positive integer")

    sample_quota = raw.get("sample_quota")
    if not _is_positive_int(sample_quota):
        issue("E_CONFIG_SAMPLE_QUOTA", "sample_quota", "must be a positive integer")

    snapshot_interval = raw.get("snapshot_interval")
    if not _is_positive_int(snapshot_interval):
        issue("E_CONFIG_SNAPSHOT_INTERVAL", "snapshot_interval", "must be a positive integer")

    if issues:
        return None, issues

    assert tolerances is not None
    config = WorkbenchConfig(
        scheme=scheme,
        base_frequency=float(base_frequency),
        train_context_length=int(train_context_length),
        extrapolation_target_length=int(extrapolation_target_length),
        dimension=int(dimension),
        scale_factor=float(scale_factor),
        precision_level=str(precision_level),
        precision_levels=tuple(precision_levels),
        dimension_block_size=int(dimension_block_size),
        fixed_decimal_bits=int(fixed_decimal_bits),
        fixed_decimal_widths=tuple(widths),
        quantization_level=str(quantization_level),
        quantization_levels=tuple(quant_levels),
        tolerances=tolerances,
        tolerance_candidates=tolerance_candidates,
        thread_pool_size=int(thread_pool_size),
        point_budget=int(point_budget),
        sample_quota=int(sample_quota),
        snapshot_interval=int(snapshot_interval),
        scheme_candidates=tuple(scheme_candidates or ["rope", "linear", "ntk"]),
    )
    return config, []


def load_config(path: str | Path) -> tuple[WorkbenchConfig | None, list[Issue]]:
    import json

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, [
            _config_issue("E_CONFIG_FILE", str(config_path), "configuration file does not exist")
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, [
            _config_issue("E_CONFIG_JSON", str(config_path), f"cannot parse configuration: {exc}")
        ]
    return validate_config_dict(raw)


def apply_config_patch(config: WorkbenchConfig, patch: Any) -> tuple[WorkbenchConfig | None, list[Issue]]:
    if not isinstance(patch, dict):
        return None, [
            _config_issue("E_COMMAND_PATCH", "patch", "set command patch must be an object")
        ]
    current = config.to_dict()
    current["tolerances"] = dict(current["tolerances"])
    current["tolerance_candidates"] = {
        key: list(values) for key, values in current["tolerance_candidates"].items()
    }
    for key, value in patch.items():
        if key == "tolerances":
            if not isinstance(value, dict):
                current[key] = value
            else:
                merged = copy.deepcopy(current["tolerances"])
                merged.update(value)
                current[key] = merged
        else:
            current[key] = copy.deepcopy(value)
    return validate_config_dict(current)
