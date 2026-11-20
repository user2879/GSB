"""Shared intermediate representation for both exchange versions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORTED_SCHEMES = ("symmetric", "asymmetric")
SUPPORTED_ROUNDING = ("nearest_even", "nearest_away", "floor", "ceil", "truncate")
SUPPORTED_SATURATION = ("saturate", "saturate_symmetric")


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    distribution: str
    special_positions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    features: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class TensorParam:
    name: str
    scale: float
    zero_point: int
    scheme: str
    rounding: str
    precision_tag: str
    quantization_bits: int
    saturation_max: int
    quantile_lower: float | None = None
    quantile_upper: float | None = None
    source: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeSet:
    version: str
    quantization_bits: int
    tensors: list[TensorParam]
    metadata: dict[str, Any] = field(default_factory=dict)


def quant_bounds(scheme: str, bits: int) -> tuple[int, int]:
    if bits < 2 or bits > 32:
        raise ValueError(f"unsupported bit width: {bits}")
    if scheme == "symmetric":
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    if scheme == "asymmetric":
        return 0, (1 << bits) - 1
    raise ValueError(f"unsupported scheme: {scheme}")


def symmetric_saturation_max(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def scheme_saturation_max(scheme: str, bits: int) -> int:
    return quant_bounds(scheme, bits)[1]

