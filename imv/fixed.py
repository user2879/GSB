from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN, ROUND_FLOOR
from typing import Any

from .errors import fail
from .util import INT64_MAX, INT64_MIN, clamp_int64


TRUNCATION_MODES = {"toward_zero", "floor", "ceil"}
OVERFLOW_MODES = {"saturate_int64"}


@dataclass(frozen=True)
class FixedPointContext:
    scale_factor: int
    truncation: str = "toward_zero"
    overflow: str = "saturate_int64"

    def __post_init__(self) -> None:
        if not isinstance(self.scale_factor, int) or isinstance(self.scale_factor, bool) or self.scale_factor <= 0:
            fail("E_SCHEMA", "scale_factor must be a positive integer", location={"path": "$.session.scale_factor"})
        if self.truncation not in TRUNCATION_MODES:
            fail("E_SCHEMA", "unsupported truncation mode", location={"path": "$.session.truncation"})
        if self.overflow not in OVERFLOW_MODES:
            fail("E_SCHEMA", "unsupported overflow mode", location={"path": "$.session.overflow"})

    def quantize(self, value: int | float, *, location: dict[str, Any] | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            fail("E_DATA", "fixed-point input must be numeric", location=location or {})
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            fail("E_DATA", "fixed-point input must be finite", location=location or {})
        try:
            decimal_value = Decimal(str(value))
            scaled = decimal_value * Decimal(self.scale_factor)
            if self.truncation == "toward_zero":
                integer = int(scaled.to_integral_value(rounding=ROUND_DOWN))
            elif self.truncation == "floor":
                integer = int(scaled.to_integral_value(rounding=ROUND_FLOOR))
            elif self.truncation == "ceil":
                integer = int(scaled.to_integral_value(rounding=ROUND_CEILING))
            else:  # guarded by __post_init__, retained for defensive clarity
                integer = int(scaled)
        except (InvalidOperation, ValueError) as exc:
            fail("E_DATA", f"cannot convert numeric value to fixed point: {exc}", location=location or {})
        return clamp_int64(integer)


@dataclass
class AggregateAccumulator:
    spec: dict[str, Any]
    fixed: FixedPointContext
    count: int = 0
    total: int = 0
    minimum: int | None = None
    maximum: int | None = None
    saturated: bool = False
    seen: int = 0

    @property
    def alias(self) -> str:
        return self.spec["alias"]

    def add(self, row: dict[str, Any]) -> None:
        func = self.spec["func"]
        column = self.spec["column"]
        if func == "count":
            if column == "*" or row.get(column) is not None:
                self.count += self.fixed.scale_factor
                self.seen += 1
            return

        raw = row.get(column)
        if raw is None:
            return
        value = self.fixed.quantize(raw, location={"column": column, "row_id": row.get("__row_id__")})
        self.seen += 1
        if func == "sum":
            self.total += value
        elif func == "min":
            self.minimum = value if self.minimum is None else min(self.minimum, value)
        elif func == "max":
            self.maximum = value if self.maximum is None else max(self.maximum, value)
        else:
            fail("E_SCHEMA", f"unsupported aggregate function: {func}")

    def value(self) -> int | None:
        func = self.spec["func"]
        if func == "count":
            value = clamp_int64(self.count)
            self.saturated = self.saturated or value != self.count
            return value
        if func == "sum":
            value = clamp_int64(self.total)
            self.saturated = self.saturated or value != self.total
            return value
        if func == "min":
            return self.minimum
        if func == "max":
            return self.maximum
        fail("E_SCHEMA", f"unsupported aggregate function: {func}")


def aggregate_row(
    specs: list[dict[str, Any]],
    fixed: FixedPointContext,
    row: dict[str, Any],
) -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for spec in specs:
        acc = AggregateAccumulator(spec, fixed)
        acc.add(row)
        values[spec["alias"]] = acc.value()
    return values


def describe_fixed_point() -> dict[str, Any]:
    return {
        "representation": "signed_int64_fixed_point",
        "count": "cardinality * scale_factor",
        "sum": "exact sum(quantize(each non-null input)); saturate once at output",
        "min_max": "min/max(quantize(each non-null input))",
        "truncation": "toward_zero | floor | ceil, Decimal(str(value)) * scale_factor",
        "overflow": "saturate to [-9223372036854775808, 9223372036854775807]",
        "empty_sum": 0,
        "empty_min_max": None,
        "null_group_key": "excluded before grouping",
    }


INT64_BOUNDS = {"min": INT64_MIN, "max": INT64_MAX}
