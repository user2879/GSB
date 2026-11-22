"""Immutable configuration models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    columns: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"group_id": self.group_id, "columns": list(self.columns)}


@dataclass(frozen=True)
class TensorConfig:
    tensor_id: str
    shape: tuple[int, ...]
    column_axis: int
    quantile_lower_permille: int
    quantile_upper_permille: int
    scheme: str
    bit_width_candidates: tuple[int, ...]
    active_bit_width: int
    fractional_bits: int
    saturation_strategy: str
    zero_scale_policy: str
    tolerance: float
    group_over_tolerance_limit: float
    groups: tuple[GroupSpec, ...]

    @property
    def column_count(self) -> int:
        return self.shape[self.column_axis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_id": self.tensor_id,
            "shape": list(self.shape),
            "column_axis": self.column_axis,
            "column_count": self.column_count,
            "quantile_lower_permille": self.quantile_lower_permille,
            "quantile_upper_permille": self.quantile_upper_permille,
            "scheme": self.scheme,
            "bit_width_candidates": list(self.bit_width_candidates),
            "active_bit_width": self.active_bit_width,
            "fractional_bits": self.fractional_bits,
            "saturation_strategy": self.saturation_strategy,
            "zero_scale_policy": self.zero_scale_policy,
            "tolerance": self.tolerance,
            "group_over_tolerance_limit": self.group_over_tolerance_limit,
            "groups": [group.to_dict() for group in self.groups],
        }


@dataclass(frozen=True)
class WorkbenchConfig:
    config_id: str
    interpolation: str
    tie_break: str
    scale_quantization: str
    allowed_bit_widths: tuple[int, ...]
    max_workers: int
    calibration_element_budget: int
    snapshot_commit_interval_batches: int
    tensors: tuple[TensorConfig, ...]
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "config_id": self.config_id,
            "interpolation": self.interpolation,
            "tie_break": self.tie_break,
            "scale_quantization": self.scale_quantization,
            "allowed_bit_widths": list(self.allowed_bit_widths),
            "max_workers": self.max_workers,
            "calibration_element_budget": self.calibration_element_budget,
            "snapshot_commit_interval_batches": self.snapshot_commit_interval_batches,
            "tensors": [tensor.to_dict() for tensor in self.tensors],
        }


@dataclass
class ConfigLoadResult:
    config: WorkbenchConfig | None
    failures: list[dict[str, Any]]
