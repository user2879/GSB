"""Integer fixed-point aggregation and auditable error attribution."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from typing import Any, Iterable

from .canonical import (
    INT64_MAX,
    INT64_MIN,
    clamp_int64,
    sat_add,
    sat_mul,
    trunc_div,
)


SUPPORTED_AGGREGATES = ("count", "sum", "mean", "weighted_mean")


@dataclass(frozen=True)
class ScaleContext:
    scale: int

    @property
    def quantum(self) -> Decimal:
        return Decimal(1).scaleb(-self.scale)

    def quantize_raw_integer(self, raw: int, source_decimals: int) -> int:
        """Quantize ``raw * 10^-source_decimals`` toward zero."""

        raw = int(raw)
        if source_decimals <= self.scale:
            return clamp_int64(raw * (10 ** (self.scale - source_decimals)))
        divisor = 10 ** (source_decimals - self.scale)
        return clamp_int64(trunc_div(raw, divisor))

    def quantize_decimal(self, value: Any) -> int:
        """Quantize a JSON scalar toward zero, then saturate to int64."""

        try:
            decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
            scaled = decimal_value.scaleb(self.scale).to_integral_value(
                rounding=ROUND_DOWN
            )
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ValueError(f"cannot quantize value {value!r}") from exc
        return clamp_int64(int(scaled))

    def as_decimal(self, fixed: int) -> Decimal:
        return Decimal(int(fixed)).scaleb(-self.scale)


@dataclass(frozen=True)
class FixedEventValue:
    event_seq: int
    event_time_ms: int
    measure_fixed: int
    weight_fixed: int


@dataclass
class FixedStats:
    count: int = 0
    sum_fixed: int = 0
    mean_fixed: int = 0
    weight_sum_fixed: int = 0
    weighted_mean_fixed: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "count": self.count,
            "sum_fixed": self.sum_fixed,
            "mean_fixed": self.mean_fixed,
            "weight_sum_fixed": self.weight_sum_fixed,
            "weighted_mean_fixed": self.weighted_mean_fixed,
        }


def aggregate_fixed(values: Iterable[FixedEventValue]) -> FixedStats:
    """Aggregate in integer fixed-point space with int64 saturation."""

    rows = list(values)
    count = len(rows)
    if not rows:
        return FixedStats()

    sum_fixed = 0
    weight_sum = 0
    weighted_numerator = 0
    for row in rows:
        sum_fixed = sat_add(sum_fixed, row.measure_fixed)
        weight_sum = sat_add(weight_sum, row.weight_fixed)
        weighted_numerator = sat_add(
            weighted_numerator,
            sat_mul(row.measure_fixed, row.weight_fixed),
        )

    return FixedStats(
        count=count,
        sum_fixed=sum_fixed,
        mean_fixed=trunc_div(sum_fixed, count),
        weight_sum_fixed=weight_sum,
        weighted_mean_fixed=trunc_div(weighted_numerator, weight_sum),
    )


def _segment_payload(
    apportioned: dict[str, int], total: int, threshold_ratio: str = "1/2"
) -> tuple[str, str]:
    """Find the dominant error segment using exact integer comparisons."""

    order = ("quantization_truncation", "late_admission", "boundary_attribution")
    if total <= 0:
        return "none", "0/1"
    # Saturating addition can hide arithmetic overflow, but the tie rule is still
    # deterministic even in that saturated case.
    best = max(order, key=lambda name: (apportioned[name], -order.index(name)))
    best_value = apportioned[best]
    if best_value * 2 < total:
        return "none", f"{best_value}/{total}"
    return best, f"{best_value}/{total}"


def error_attribution(
    *,
    included: list[FixedEventValue],
    accepted_late: list[FixedEventValue],
    nominal_end_ms: int,
    scale: int,
) -> dict[str, Any]:
    """Build the three-part error envelope for one window.

    The arithmetic is entirely in fixed-point units. Each raw value can lose at
    most one quantum by truncation toward zero, so the quantization envelope is
    ``event_count``. Accepted late values contribute their absolute admitted
    magnitude. Events made visible by a derived end boundary contribute their
    absolute magnitude to the boundary segment.
    """

    count = len(included)
    quantization_fixed = clamp_int64(count)
    late_fixed = sat_add(*(abs(row.measure_fixed) for row in accepted_late))
    boundary_rows = [row for row in included if row.event_time_ms >= nominal_end_ms]
    boundary_fixed = sat_add(*(abs(row.measure_fixed) for row in boundary_rows))
    total_fixed = sat_add(quantization_fixed, late_fixed, boundary_fixed)
    mean_bound_fixed = (
        sat_add(trunc_div(total_fixed, count), 1 if count else 0)
        if count
        else 0
    )
    segments = {
        "quantization_truncation": quantization_fixed,
        "late_admission": late_fixed,
        "boundary_attribution": boundary_fixed,
    }
    dominant, share = _segment_payload(segments, total_fixed)
    return {
        "scale": scale,
        "quantum": str(ScaleContext(scale).quantum),
        "truncation_direction": "toward_zero",
        "overflow_rule": "saturate_int64",
        "single_value_error_bound_fixed": 1,
        "aggregation_error_bound_fixed": total_fixed,
        "mean_error_bound_fixed": mean_bound_fixed,
        "segments_fixed": segments,
        "dominant_segment": dominant,
        "dominant_share": share,
        "dominant_threshold_ratio": "1/2",
        "boundary_event_count": len(boundary_rows),
        "accepted_late_event_count": len(accepted_late),
    }


@dataclass
class WindowAccumulator:
    scale: int
    rows: list[FixedEventValue] = field(default_factory=list)

    def add(
        self,
        *,
        event_seq: int,
        event_time_ms: int,
        measure_raw: int,
        measure_decimals: int,
        weight_raw: int,
        weight_decimals: int,
    ) -> FixedEventValue:
        row = FixedEventValue(
            event_seq=event_seq,
            event_time_ms=event_time_ms,
            measure_fixed=ScaleContext(self.scale).quantize_raw_integer(
                measure_raw, measure_decimals
            ),
            weight_fixed=ScaleContext(self.scale).quantize_raw_integer(
                weight_raw, weight_decimals
            ),
        )
        self.rows.append(row)
        return row

    def stats(self) -> FixedStats:
        return aggregate_fixed(self.rows)


def describe_fixed_contract(scale: int) -> dict[str, Any]:
    return {
        "scale": scale,
        "quantum": str(ScaleContext(scale).quantum),
        "truncation_direction": "toward_zero",
        "quantization": "truncate_toward_zero_after_multiplying_by_10^scale",
        "accumulation": "integer_add_then_saturate_int64",
        "mean_division": "truncate_toward_zero",
        "int64_min": INT64_MIN,
        "int64_max": INT64_MAX,
    }
