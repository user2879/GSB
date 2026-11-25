"""Core immutable domain objects."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .errors import E_FIT_OVERFLOW, FitError

MISSING_MARKERS = frozenset({"NA", "NaN", "null", "NULL", "missing", "MISSING", ""})
ALLOWED_FIXED_POINT_BITS = (16, 24, 32, 40, 48, 56, 62)
ALLOWED_QUANTIZATION_SHIFTS = (6, 8, 10, 12, 14, 16, 18, 20)
MAX_HARMONIC_ORDER = 8
MAX_REGULARIZATION = 100.0
INTERPOLATION_DECIMALS = 12
SCALE_PROFILES = ("small", "standard", "large")
SCALE_PRESETS: dict[str, dict[str, int]] = {
    "small": {
        "global_fit_points": 640,
        "per_series_quota": 256,
        "batch_observation_points": 320,
    },
    "standard": {
        "global_fit_points": 4096,
        "per_series_quota": 768,
        "batch_observation_points": 1024,
    },
    "large": {
        "global_fit_points": 65536,
        "per_series_quota": 8192,
        "batch_observation_points": 16384,
    },
}


@dataclass(frozen=True)
class SeriesRecord:
    series_id: str
    timestamps: tuple[float, ...]
    values: tuple[float | None, ...]
    source: str = ""
    source_line: int = 0

    @property
    def length(self) -> int:
        return len(self.values)

    @property
    def observed_indices(self) -> tuple[int, ...]:
        return tuple(i for i, value in enumerate(self.values) if value is not None)

    @property
    def missing_indices(self) -> tuple[int, ...]:
        return tuple(i for i, value in enumerate(self.values) if value is None)

    def with_value(self, index: int, value: float | None) -> "SeriesRecord":
        values = list(self.values)
        values[index] = value
        return replace(self, values=tuple(values))

    def as_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "timestamps": list(self.timestamps),
            "values": [None if value is None else value for value in self.values],
        }


@dataclass(frozen=True)
class FitConfig:
    period: int = 12
    harmonic_order: int = 2
    regularization_strength: float = 0.01
    fixed_point_bits: int = 48
    quantization_shift: int = 16
    pivot_threshold: int = 1000
    tolerance_relative: float = 1.0e-6
    tolerance_absolute: float = 1.0e-6
    global_fit_points: int = 4096
    per_series_quota: int = 768
    batch_observation_points: int = 1024
    thread_pool_workers: int = 4
    scale_profile: str = "standard"
    version: int = 1
    comparison_bit_widths: tuple[int, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def quantization_scale(self) -> int:
        return 1 << self.quantization_shift

    @property
    def bit_tier(self) -> str:
        return f"bits{self.fixed_point_bits}-q{self.quantization_shift}"

    @property
    def column_count(self) -> int:
        return 2 + 2 * self.harmonic_order

    def with_version(self, version: int) -> "FitConfig":
        return replace(self, version=version)

    def with_changes(self, **changes: Any) -> "FitConfig":
        return replace(self, **changes)

    def computational_dict(self) -> dict[str, Any]:
        """Configuration that can change coefficients or error attribution.

        Budget, worker count and version are deliberately excluded.  A budget
        edit must not evict valid coefficient products.
        """

        return {
            "model": {
                "period": self.period,
                "harmonic_order": self.harmonic_order,
                "regularization_strength": self.regularization_strength,
            },
            "fixed_point": {
                "bits": self.fixed_point_bits,
                "quantization_shift": self.quantization_shift,
                "pivot_threshold": self.pivot_threshold,
                "rounding": "nearest_half_away_from_zero",
                "pivot_selection": "max_abs_then_lowest_row",
                "accumulation_order": "time_ascending_then_column_ascending",
            },
            "interpolation": {
                "rule": "inside_linear_nearest_edge",
                "decimals": INTERPOLATION_DECIMALS,
                "rounding": "nearest_half_away_from_zero",
                "distance_tie": "lower_index_left",
            },
            "reference": {
                "tolerance_relative": self.tolerance_relative,
                "tolerance_absolute": self.tolerance_absolute,
                "truth": "float64_observed_only_no_imputation",
            },
        }

    def full_dict(self) -> dict[str, Any]:
        data = self.computational_dict()
        data["budget"] = {
            "scale_profile": self.scale_profile,
            "global_fit_points": self.global_fit_points,
            "per_series_quota": self.per_series_quota,
            "batch_observation_points": self.batch_observation_points,
        }
        data["runtime"] = {"thread_pool_workers": self.thread_pool_workers}
        data["diagnostics"] = {
            "comparison_bit_widths": list(self.comparison_bit_widths)
        }
        data["version"] = self.version
        return data


@dataclass(frozen=True)
class CacheAction:
    stage: str
    key: str
    action: str
    reason: str
    resumed: bool = False

    def as_dict(self) -> dict[str, Any]:
        data = {
            "stage": self.stage,
            "key": self.key,
            "action": self.action,
            "reason": self.reason,
        }
        if self.resumed:
            data["resumed"] = True
        return data


def assert_width(value: int, bits: int, label: str) -> None:
    limit = (1 << (bits - 1)) - 1
    if abs(value) > limit:
        raise FitError(
            E_FIT_OVERFLOW,
            f"{label} {value} exceeds signed fixed-point width {bits}",
            value=value,
            bits=bits,
            label=label,
        )


def sorted_series(series: Iterable[SeriesRecord]) -> tuple[SeriesRecord, ...]:
    return tuple(sorted(series, key=lambda item: item.series_id))
