"""Shared immutable-ish data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class WeightPack:
    layer: str
    shape: tuple[int, ...]
    weights: np.ndarray
    content_version: str
    content_fingerprint: str

    @property
    def numel(self) -> int:
        return int(self.weights.size)


@dataclass
class CalibrationRecord:
    layer: str
    scale: float
    zero_point: int
    bit_width: int
    scheme: str
    rounding_mode: str
    precision_label: str
    per_channel: bool
    source_version: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "bit_width": self.bit_width,
            "scheme": self.scheme,
            "rounding_mode": self.rounding_mode,
            "precision_label": self.precision_label,
            "per_channel": self.per_channel,
            "source_version": self.source_version,
            "provenance": self.provenance,
        }


@dataclass
class FailureRecord:
    layer: str
    stage: str
    code: str
    message: str
    injection_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "layer": self.layer,
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
        }
        if self.injection_id:
            payload["injection_id"] = self.injection_id
        if self.detail:
            payload["detail"] = self.detail
        return payload


@dataclass
class GroupQuantParam:
    index: int
    start: int
    end: int
    q_low: float
    q_high: float
    observed_min: float
    observed_max: float
    raw_scale: float
    scale: float
    zero_point: int
    qmin: int
    qmax: int
    scheme: str
    interpolation: str
    tie_break: str
    fixed_fraction_bits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "q_low": self.q_low,
            "q_high": self.q_high,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
            "raw_scale": self.raw_scale,
            "scale": self.scale,
            "zero_point": self.zero_point,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "scheme": self.scheme,
            "interpolation": self.interpolation,
            "tie_break": self.tie_break,
            "fixed_fraction_bits": self.fixed_fraction_bits,
        }


@dataclass
class LayerQuantResult:
    layer: str
    shape: list[int]
    bit_width: int
    qmin: int
    qmax: int
    group_size: int
    per_channel: bool
    scheme: str
    interpolation: str
    tie_break: str
    quantile_lower: float
    quantile_upper: float
    fixed_fraction_bits: int
    groups: list[GroupQuantParam]
    content_fingerprint: str
    param_fingerprint: str
    exchange_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "shape": self.shape,
            "bit_width": self.bit_width,
            "qmin": self.qmin,
            "qmax": self.qmax,
            "group_size": self.group_size,
            "per_channel": self.per_channel,
            "scheme": self.scheme,
            "interpolation": self.interpolation,
            "tie_break": self.tie_break,
            "quantile_lower": self.quantile_lower,
            "quantile_upper": self.quantile_upper,
            "fixed_fraction_bits": self.fixed_fraction_bits,
            "groups": [group.to_dict() for group in self.groups],
            "content_fingerprint": self.content_fingerprint,
            "param_fingerprint": self.param_fingerprint,
            "exchange_version": self.exchange_version,
            "metadata": self.metadata,
        }


@dataclass
class QuantizationResult:
    layer: str
    shape: list[int]
    bit_width: int
    fixed_point_fraction_bits: int
    rounding_mode: str
    saturation: str
    full_precision: bool
    integers: list[int | None]
    dequantized: list[float]
    clip_directions: list[int]
    clipped_count: int
    clip_rate: float
    max_abs_error: float
    mse: float
    group_parameters: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def error_summary(self) -> dict[str, Any]:
        return {
            "bit_width": self.bit_width,
            "fixed_point_fraction_bits": self.fixed_point_fraction_bits,
            "rounding_mode": self.rounding_mode,
            "saturation": self.saturation,
            "full_precision": self.full_precision,
            "max_abs_error": self.max_abs_error,
            "mse": self.mse,
            "clip_rate": self.clip_rate,
            "clipped_count": self.clipped_count,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "layer": self.layer,
            "shape": self.shape,
            "bit_width": self.bit_width,
            "fixed_point_fraction_bits": self.fixed_point_fraction_bits,
            "rounding_mode": self.rounding_mode,
            "saturation": self.saturation,
            "full_precision": self.full_precision,
            "integers": self.integers,
            "dequantized": self.dequantized,
            "clip_directions": self.clip_directions,
            "clipped_count": self.clipped_count,
            "clip_rate": self.clip_rate,
            "max_abs_error": self.max_abs_error,
            "mse": self.mse,
            "group_parameters": self.group_parameters,
            "metadata": self.metadata,
        }
        return payload
