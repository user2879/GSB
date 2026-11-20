"""Calibration configuration parsing and per-tensor resolution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import QuantCalError
from .models import (
    SUPPORTED_ROUNDING,
    SUPPORTED_SATURATION,
    SUPPORTED_SCHEMES,
)
from .util import fingerprint, is_int, load_json


@dataclass(frozen=True)
class ResolvedTensorConfig:
    quantile_lower_permille: int
    quantile_upper_permille: int
    quantization_bits: int
    fixed_fractional_bits: int
    rounding_mode: str
    saturation_strategy: str
    quantization_scheme: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantile_lower_permille": self.quantile_lower_permille,
            "quantile_upper_permille": self.quantile_upper_permille,
            "quantization_bits": self.quantization_bits,
            "fixed_fractional_bits": self.fixed_fractional_bits,
            "rounding_mode": self.rounding_mode,
            "saturation_strategy": self.saturation_strategy,
            "quantization_scheme": self.quantization_scheme,
        }


@dataclass
class QuantConfig:
    quantile_lower_permille: int
    quantile_upper_permille: int
    quantization_bits_candidates: tuple[int, ...]
    quantization_bits: int
    fixed_fractional_bits: int
    rounding_mode_candidates: tuple[str, ...]
    rounding_mode: str
    saturation_strategy: str
    max_concurrency: int
    batch_point_budget: int
    snapshot_commit_interval: int
    quantization_scheme: str = "symmetric"
    tensor_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def resolve(self, tensor_name: str) -> ResolvedTensorConfig:
        override = self.tensor_overrides.get(tensor_name, {})
        try:
            lower = int(override.get("quantile_lower_permille", self.quantile_lower_permille))
            upper = int(override.get("quantile_upper_permille", self.quantile_upper_permille))
            bits = int(override.get("quantization_bits", self.quantization_bits))
            fractional = int(override.get("fixed_fractional_bits", self.fixed_fractional_bits))
        except (TypeError, ValueError) as exc:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                "per-tensor integer override is malformed",
                tensor=tensor_name,
                stage="config",
            ) from exc
        rounding = override.get("rounding_mode", self.rounding_mode)
        saturation = override.get("saturation_strategy", self.saturation_strategy)
        scheme = override.get("quantization_scheme", self.quantization_scheme)
        _validate_permille(lower, upper, tensor_name)
        if bits not in self.quantization_bits_candidates or not (2 <= bits <= 32):
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                f"quantization bits {bits} is not in valid candidates {list(self.quantization_bits_candidates)}",
                tensor=tensor_name,
                stage="config",
            )
        if not (0 <= fractional <= 30):
            raise QuantCalError(
                "E_QUANT_RANGE_INVALID",
                f"fixed fractional bits {fractional} outside [0, 30]",
                tensor=tensor_name,
                stage="config",
            )
        if rounding not in SUPPORTED_ROUNDING or rounding not in self.rounding_mode_candidates:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"rounding mode {rounding!r} is not a valid active candidate",
                tensor=tensor_name,
                stage="config",
            )
        if saturation not in SUPPORTED_SATURATION:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"unsupported saturation strategy {saturation!r}",
                tensor=tensor_name,
                stage="config",
            )
        if scheme not in SUPPORTED_SCHEMES:
            raise QuantCalError(
                "E_FORMAT_INVALID",
                f"unsupported quantization scheme {scheme!r}",
                tensor=tensor_name,
                stage="config",
            )
        return ResolvedTensorConfig(
            quantile_lower_permille=lower,
            quantile_upper_permille=upper,
            quantization_bits=bits,
            fixed_fractional_bits=fractional,
            rounding_mode=str(rounding),
            saturation_strategy=str(saturation),
            quantization_scheme=str(scheme),
        )


def _validate_permille(lower: int, upper: int, tensor: str | None = None) -> None:
    if not (0 <= lower <= 1000 and 0 <= upper <= 1000) or lower >= upper:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            f"quantile permille range must satisfy 0 <= lower < upper <= 1000, got [{lower}, {upper}]",
            tensor=tensor,
            stage="config",
        )


def _as_int(value: Any, name: str) -> int:
    if not is_int(value):
        raise QuantCalError("E_FORMAT_INVALID", f"{name} must be an integer", stage="config")
    return int(value)


def load_config(path: str | Path) -> QuantConfig:
    document = load_json(path)
    return parse_config(document)


def parse_config(document: Any) -> QuantConfig:
    if not isinstance(document, dict):
        raise QuantCalError("E_FORMAT_INVALID", "config must be a JSON object", stage="config")
    lower = _as_int(document.get("quantile_lower_permille", 1), "quantile_lower_permille")
    upper = _as_int(document.get("quantile_upper_permille", 999), "quantile_upper_permille")
    _validate_permille(lower, upper)
    candidates_raw = document.get("quantization_bits_candidates", [8, 12])
    if not isinstance(candidates_raw, list) or not candidates_raw:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID", "quantization_bits_candidates must be a non-empty array", stage="config"
        )
    candidates = tuple(_as_int(value, "quantization candidate") for value in candidates_raw)
    if len(set(candidates)) != len(candidates) or any(not (2 <= value <= 32) for value in candidates):
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            f"invalid unique 2..32 quantization candidates: {list(candidates)}",
            stage="config",
        )
    bits = _as_int(document.get("quantization_bits", candidates[0]), "quantization_bits")
    if bits not in candidates:
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID",
            f"active quantization_bits {bits} is not one of {list(candidates)}",
            stage="config",
        )
    fractional = _as_int(document.get("fixed_fractional_bits", 16), "fixed_fractional_bits")
    if not (0 <= fractional <= 30):
        raise QuantCalError(
            "E_QUANT_RANGE_INVALID", "fixed_fractional_bits must be in [0, 30]", stage="config"
        )
    rounding_candidates_raw = document.get(
        "rounding_mode_candidates", ["nearest_even", "nearest_away"]
    )
    if not isinstance(rounding_candidates_raw, list) or not rounding_candidates_raw:
        raise QuantCalError(
            "E_FORMAT_INVALID", "rounding_mode_candidates must be a non-empty array", stage="config"
        )
    if any(value not in SUPPORTED_ROUNDING for value in rounding_candidates_raw):
        raise QuantCalError(
            "E_FORMAT_INVALID",
            f"unknown rounding mode candidate: {rounding_candidates_raw}",
            stage="config",
        )
    rounding_candidates = tuple(str(value) for value in rounding_candidates_raw)
    rounding = document.get("rounding_mode", rounding_candidates[0])
    if rounding not in rounding_candidates:
        raise QuantCalError(
            "E_FORMAT_INVALID",
            f"active rounding mode {rounding!r} is not a candidate",
            stage="config",
        )
    saturation = document.get("saturation_strategy", "saturate")
    if saturation not in SUPPORTED_SATURATION:
        raise QuantCalError(
            "E_FORMAT_INVALID", f"unsupported saturation strategy {saturation!r}", stage="config"
        )
    scheme = document.get("quantization_scheme", "symmetric")
    if scheme not in SUPPORTED_SCHEMES:
        raise QuantCalError(
            "E_FORMAT_INVALID", f"unsupported quantization scheme {scheme!r}", stage="config"
        )
    max_concurrency = _as_int(document.get("max_concurrency", 4), "max_concurrency")
    budget = _as_int(document.get("batch_point_budget", 1_000_000), "batch_point_budget")
    commit_interval = _as_int(
        document.get("snapshot_commit_interval", 1), "snapshot_commit_interval"
    )
    if max_concurrency < 1:
        raise QuantCalError("E_FORMAT_INVALID", "max_concurrency must be >= 1", stage="config")
    if budget < 0:
        raise QuantCalError("E_FORMAT_INVALID", "batch_point_budget must be >= 0", stage="config")
    if commit_interval < 1:
        raise QuantCalError(
            "E_FORMAT_INVALID", "snapshot_commit_interval must be >= 1", stage="config"
        )
    overrides_raw = document.get("tensor_overrides", {})
    if not isinstance(overrides_raw, dict):
        raise QuantCalError("E_FORMAT_INVALID", "tensor_overrides must be an object", stage="config")
    overrides: dict[str, dict[str, Any]] = {}
    for name, value in overrides_raw.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise QuantCalError(
                "E_FORMAT_INVALID", "tensor_overrides entries must be name -> object", stage="config"
            )
        overrides[name] = deepcopy(value)
    raw = deepcopy(document)
    config = QuantConfig(
        quantile_lower_permille=lower,
        quantile_upper_permille=upper,
        quantization_bits_candidates=candidates,
        quantization_bits=bits,
        fixed_fractional_bits=fractional,
        rounding_mode_candidates=rounding_candidates,
        rounding_mode=str(rounding),
        saturation_strategy=str(saturation),
        max_concurrency=max_concurrency,
        batch_point_budget=budget,
        snapshot_commit_interval=commit_interval,
        quantization_scheme=str(scheme),
        tensor_overrides=overrides,
        raw=raw,
    )
    config.fingerprint = fingerprint(
        {
            "raw": raw,
            "parser": "quantcal.config.v1",
            "quantile_method": "linear_type7",
        }
    )
    return config
