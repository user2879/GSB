"""Fixed-point aggregation and explicit error bounds.

Rules:
* scale is decimal 1/1,000,000.
* Integer column values are exact after scaling.
* Float column values are truncated toward zero at the scale boundary.
* Sums saturate to signed 64-bit limits. Saturation is sticky and reported.
* Means truncate toward zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import canonical_json, digest_value

FIXED_SCALE = 1_000_000
FIXED_POINT_VERSION = "fixed-s64-scale1e6-trunc-toward-zero-v1"
SIGNED_64_MAX = (1 << 63) - 1
SIGNED_64_MIN = -(1 << 63)


def quantize_value(value: int | float, column_type: str) -> int:
    if column_type == "int":
        return int(value) * FIXED_SCALE
    if column_type == "float":
        scaled = float(value) * FIXED_SCALE
        if not math.isfinite(scaled):
            return SIGNED_64_MAX if scaled > 0 else SIGNED_64_MIN
        return int(scaled)
    raise ValueError(f"unsupported column type: {column_type}")


def trunc_div_toward_zero(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError("fixed-point division by zero")
    if numerator == 0:
        return 0
    sign = -1 if (numerator < 0) != (denominator < 0) else 1
    return sign * (abs(numerator) // abs(denominator))


def saturating_add(current: int, increment: int) -> tuple[int, bool]:
    if increment > 0 and current > SIGNED_64_MAX - increment:
        return SIGNED_64_MAX, True
    if increment < 0 and current < SIGNED_64_MIN - increment:
        return SIGNED_64_MIN, True
    return current + increment, False


def scaled_decimal(units: int | None) -> str | None:
    if units is None:
        return None
    sign = "-" if units < 0 else ""
    absolute = abs(units)
    whole, fraction = divmod(absolute, FIXED_SCALE)
    return f"{sign}{whole}.{fraction:06d}"


@dataclass
class MetricAccumulator:
    column: str
    column_type: str
    value_count: int = 0
    null_count: int = 0
    exact_int_values: int = 0
    quantized_float_values: int = 0
    sum_scaled: int = 0
    saturated: bool = False
    page_terms: list[dict[str, Any]] = field(default_factory=list)

    def add_page_values(
        self,
        *,
        access_seq: int,
        page_id: int,
        values: Iterable[int | float | None],
        page_rows: int,
        group_rows: int,
    ) -> None:
        term = {
            "access_seq": access_seq,
            "page_id": page_id,
            "page_rows": page_rows,
            "group_rows": group_rows,
            "exact_int_values": 0,
            "quantized_float_values": 0,
            "null_values": 0,
            "error_units": 0,
        }
        for value in values:
            if value is None:
                self.null_count += 1
                term["null_values"] += 1
                continue
            self.value_count += 1
            if self.column_type == "int":
                self.exact_int_values += 1
                term["exact_int_values"] += 1
            else:
                self.quantized_float_values += 1
                term["quantized_float_values"] += 1
                # Truncation toward zero is strictly less than one unit.
                term["error_units"] += 1
            self.sum_scaled, overflowed = saturating_add(
                self.sum_scaled,
                quantize_value(value, self.column_type),
            )
            self.saturated = self.saturated or overflowed
        self.page_terms.append(term)

    def payload(self) -> dict[str, Any]:
        if self.value_count == 0:
            mean_scaled: int | None = None
            mean_bound: int = 0
        else:
            mean_scaled = trunc_div_toward_zero(self.sum_scaled, self.value_count)
            if self.saturated:
                mean_bound = SIGNED_64_MAX
            else:
                # ceil(K/n) bounds the sum quantization error; one extra unit
                # bounds truncation of the mean itself.
                error_numerator = self.quantized_float_values
                mean_bound = (
                    (error_numerator + self.value_count - 1) // self.value_count
                ) + 1
        sum_bound = SIGNED_64_MAX if self.saturated else self.quantized_float_values
        return {
            "column": self.column,
            "column_type": self.column_type,
            "value_count": self.value_count,
            "null_count": self.null_count,
            "sum_scaled": self.sum_scaled,
            "sum_decimal": scaled_decimal(self.sum_scaled),
            "mean_scaled": mean_scaled,
            "mean_decimal": scaled_decimal(mean_scaled),
            "saturated": self.saturated,
            "quantization": {
                "scale": FIXED_SCALE,
                "fixed_point_version": FIXED_POINT_VERSION,
                "float_truncation": "toward_zero",
                "single_page_error_rule": "quantized_float_values * 1 unit",
                "aggregation_growth_rule": "sum(page_terms[].error_units)",
                "page_terms": [dict(term) for term in self.page_terms],
                "page_term_count": len(self.page_terms),
                "float_values": self.quantized_float_values,
                "int_values_exact": self.exact_int_values,
                "sum_error_bound_units": sum_bound,
                "mean_error_bound_units": mean_bound,
                "mean_bound_rule": "ceil(float_values/value_count)+1",
                "saturation_rule": "sticky_signed_int64",
            },
        }


@dataclass
class GroupAccumulator:
    group_key: int
    count_rows: int = 0
    metrics: dict[str, MetricAccumulator] = field(default_factory=dict)

    def add_page(
        self,
        *,
        access_seq: int,
        page_id: int,
        page_rows: int,
        group_rows: int,
        column_values: dict[str, Iterable[int | float | None]],
    ) -> None:
        self.count_rows += group_rows
        for column, values in column_values.items():
            if column not in self.metrics:
                raise KeyError(column)
            self.metrics[column].add_page_values(
                access_seq=access_seq,
                page_id=page_id,
                values=values,
                page_rows=page_rows,
                group_rows=group_rows,
            )

    def payload(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key,
            "count_rows": self.count_rows,
            "metrics": [self.metrics[name].payload() for name in sorted(self.metrics)],
        }


def result_view(groups: dict[int, GroupAccumulator], columns: list[dict[str, Any]]) -> dict[str, Any]:
    group_payloads = [groups[key].payload() for key in sorted(groups)]
    return {
        "result_format_version": 1,
        "fixed_point": {
            "scale": FIXED_SCALE,
            "fixed_point_version": FIXED_POINT_VERSION,
            "float_quantization": "truncate_toward_zero",
            "mean_rounding": "truncate_toward_zero",
            "sum_overflow": "saturate_signed_int64_sticky",
            "error_bound_unit": "scaled_unit_1e-6",
        },
        "columns": columns,
        "groups": group_payloads,
    }


def result_digest(view: dict[str, Any]) -> str:
    return digest_value(view)


def result_bytes(view: dict[str, Any]) -> bytes:
    return canonical_json(view).encode("utf-8")
