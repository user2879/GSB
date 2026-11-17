"""Strict configuration parsing with no silent scheme-parameter defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import ConfigError
from .util import hash_object, read_json


SCHEME_ORDER = ("RoPE", "LinearInterpolation", "NTK-aware")
SCHEME_RANK = {name: index for index, name in enumerate(SCHEME_ORDER)}
PRECISION_ALIASES = {
    "float64": "float64",
    "fp64": "float64",
    "double": "float64",
    "float32": "float32",
    "fp32": "float32",
    "single": "float32",
}

_SCHEME_ALIASES = {
    "rope": "RoPE",
    "rotary": "RoPE",
    "rotarypositionembedding": "RoPE",
    "旋转位置编码": "RoPE",
    "线性内插": "LinearInterpolation",
    "linearinterpolation": "LinearInterpolation",
    "linearinterpolationrope": "LinearInterpolation",
    "pi": "LinearInterpolation",
    "positioninterpolation": "LinearInterpolation",
    "ntkaware": "NTK-aware",
    "ntk感知缩放": "NTK-aware",
    "ntk": "NTK-aware",
    "ntkscaling": "NTK-aware",
}


def canonical_scheme(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("scheme must be a non-empty string", {"value": value})
    key = "".join(value.strip().lower().split()).replace("_", "").replace("-", "")
    try:
        return _SCHEME_ALIASES[key]
    except KeyError as exc:
        raise ConfigError(
            "unsupported position encoding scheme",
            {"value": value, "supported": list(SCHEME_ORDER)},
        ) from exc


@dataclass(frozen=True)
class Tolerances:
    position_vector: float
    dot_score: float
    normalized_attention_weight: float

    def as_dict(self) -> dict[str, float]:
        return {
            "position_vector": self.position_vector,
            "dot_score": self.dot_score,
            "normalized_attention_weight": self.normalized_attention_weight,
        }


@dataclass(frozen=True)
class DiagnosticConfig:
    schemes: tuple[str, ...]
    base: float
    train_context_length: int
    extrapolation_target_length: int
    dimension: int
    scaling_factor: float
    precision: str
    concurrency: int
    tolerances: Tolerances
    sensitivity_base_values: tuple[float, ...]
    sensitivity_scaling_factor_values: tuple[float, ...]
    raw: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schemes": list(self.schemes),
            "base": self.base,
            "train_context_length": self.train_context_length,
            "extrapolation_target_length": self.extrapolation_target_length,
            "dimension": self.dimension,
            "scaling_factor": self.scaling_factor,
            "precision": self.precision,
            "concurrency": self.concurrency,
            "tolerances": self.tolerances.as_dict(),
            "sensitivity": {
                "base_values": list(self.sensitivity_base_values),
                "scaling_factor_values": list(self.sensitivity_scaling_factor_values),
            },
        }


def _required(raw: dict[str, Any], names: Iterable[str], logical_name: str) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    raise ConfigError(
        f"missing required configuration field: {logical_name}",
        {"accepted_keys": list(names)},
    )


def _as_number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be numeric", {"value": value})
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{field} must be finite", {"value": value})
    if positive and number <= 0.0:
        raise ConfigError(f"{field} must be greater than zero", {"value": value})
    return number


def _as_int(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer", {"value": value})
    if positive and value <= 0:
        raise ConfigError(f"{field} must be greater than zero", {"value": value})
    return value


def _parse_schemes(raw: dict[str, Any]) -> tuple[str, ...]:
    value = _required(raw, ("schemes", "scheme"), "schemes")
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values:
        raise ConfigError("schemes must be a non-empty string or list", {"value": value})
    schemes = {canonical_scheme(item) for item in values}
    return tuple(sorted(schemes, key=SCHEME_RANK.__getitem__))


def _parse_tolerances(raw: dict[str, Any]) -> Tolerances:
    tolerance = _required(raw, ("tolerances", "tolerance"), "tolerances")
    if not isinstance(tolerance, dict):
        raise ConfigError("tolerances must be an object", {"value": tolerance})
    position = _required(
        tolerance,
        ("position_vector", "position", "encoding"),
        "tolerances.position_vector",
    )
    score = _required(tolerance, ("dot_score", "score"), "tolerances.dot_score")
    weight = _required(
        tolerance,
        ("normalized_attention_weight", "weight", "attention_weight"),
        "tolerances.normalized_attention_weight",
    )
    return Tolerances(
        position_vector=_as_number(position, "tolerances.position_vector", positive=True),
        dot_score=_as_number(score, "tolerances.dot_score", positive=True),
        normalized_attention_weight=_as_number(
            weight, "tolerances.normalized_attention_weight", positive=True
        ),
    )


def _parse_scan_values(
    sensitivity: dict[str, Any],
    *,
    absolute_keys: tuple[str, ...],
    multiplier_keys: tuple[str, ...],
    baseline: float,
    logical_name: str,
) -> tuple[float, ...]:
    present_absolute = next((key for key in absolute_keys if key in sensitivity), None)
    present_multiplier = next((key for key in multiplier_keys if key in sensitivity), None)
    if present_absolute is None and present_multiplier is None:
        raise ConfigError(
            f"missing sensitivity scan values: {logical_name}",
            {"accepted_keys": list(absolute_keys + multiplier_keys)},
        )
    if present_absolute is not None and present_multiplier is not None:
        raise ConfigError(
            f"ambiguous sensitivity scan values for {logical_name}",
            {"absolute_key": present_absolute, "multiplier_key": present_multiplier},
        )
    key = present_absolute or present_multiplier
    values = sensitivity[key]
    if not isinstance(values, list) or not values:
        raise ConfigError(f"sensitivity.{key} must be a non-empty list", {"value": values})
    parsed: list[float] = []
    for item in values:
        number = _as_number(item, f"sensitivity.{key}")
        parsed.append(number * baseline if present_multiplier else number)
    return tuple(sorted(set(parsed)))


def _parse_sensitivity(raw: dict[str, Any], base: float, scale: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    value = _required(raw, ("sensitivity", "scan"), "sensitivity")
    if not isinstance(value, dict):
        raise ConfigError("sensitivity must be an object", {"value": value})
    base_values = _parse_scan_values(
        value,
        absolute_keys=("base_values", "base"),
        multiplier_keys=("base_multipliers",),
        baseline=base,
        logical_name="base_values",
    )
    scale_values = _parse_scan_values(
        value,
        absolute_keys=("scaling_factor_values", "scale_values", "scale"),
        multiplier_keys=("scaling_factor_multipliers", "scale_multipliers"),
        baseline=scale,
        logical_name="scaling_factor_values",
    )
    return base_values, scale_values


def parse_config(raw: Any) -> DiagnosticConfig:
    if not isinstance(raw, dict):
        raise ConfigError("configuration root must be a JSON object", {"type": type(raw).__name__})
    schemes = _parse_schemes(raw)
    base = _as_number(_required(raw, ("base",), "base"), "base", positive=True)
    train = _as_int(
        _required(
            raw,
            ("train_context_length", "training_context_length", "context_length", "train_length"),
            "train_context_length",
        ),
        "train_context_length",
        positive=True,
    )
    target = _as_int(
        _required(
            raw,
            ("extrapolation_target_length", "target_length", "extrapolation_length"),
            "extrapolation_target_length",
        ),
        "extrapolation_target_length",
        positive=True,
    )
    if target <= train:
        raise ConfigError(
            "extrapolation_target_length must be greater than train_context_length",
            {"train_context_length": train, "extrapolation_target_length": target},
        )
    dimension = _as_int(
        _required(raw, ("dimension", "d_model", "dim", "hidden_size"), "dimension"),
        "dimension",
        positive=True,
    )
    if dimension % 2 != 0:
        raise ConfigError("dimension must be even", {"dimension": dimension})
    scaling_factor = _as_number(
        _required(raw, ("scaling_factor", "scale"), "scaling_factor"),
        "scaling_factor",
        positive=True,
    )
    precision_raw = _required(raw, ("precision",), "precision")
    if not isinstance(precision_raw, str):
        raise ConfigError("precision must be a string", {"value": precision_raw})
    precision = PRECISION_ALIASES.get(precision_raw.strip().lower())
    if precision is None:
        raise ConfigError(
            "unsupported precision tier",
            {"value": precision_raw, "supported": ["float64", "float32"]},
        )
    concurrency = _as_int(
        _required(raw, ("concurrency", "max_concurrency"), "concurrency"),
        "concurrency",
        positive=True,
    )
    tolerances = _parse_tolerances(raw)
    base_values, scale_values = _parse_sensitivity(raw, base, scaling_factor)
    return DiagnosticConfig(
        schemes=schemes,
        base=base,
        train_context_length=train,
        extrapolation_target_length=target,
        dimension=dimension,
        scaling_factor=scaling_factor,
        precision=precision,
        concurrency=concurrency,
        tolerances=tolerances,
        sensitivity_base_values=base_values,
        sensitivity_scaling_factor_values=scale_values,
        raw=raw,
    )


def load_config(path: str | Path) -> tuple[DiagnosticConfig, dict[str, Any], str]:
    raw = read_json(Path(path))
    config = parse_config(raw)
    return config, raw, hash_object(raw)
