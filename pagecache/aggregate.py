from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from .canonical import atomic_write_bytes, canonical_bytes
from .errors import EngineError, ErrorCode
from .model import ColumnKind, PageColumn, PageData


FIXED_SCALE = 1_000_000
FIXED_POINT_VERSION = "fixed-v1-half-even-trunc-toward-zero"
MEAN_ERROR_BOUND = Fraction(3, 2 * FIXED_SCALE)
VALUE_ERROR_BOUND = Fraction(1, 2 * FIXED_SCALE)


@dataclass(slots=True)
class GroupState:
    count: int = 0
    sum_scaled: int = 0

    def to_dict(self, key: int) -> dict[str, int]:
        return {"key": key, "count": self.count, "sum_scaled": self.sum_scaled}


@dataclass(slots=True)
class FixedPointAggregator:
    group_column: str = "group"
    value_column: str = "value"
    scale: int = FIXED_SCALE
    groups: dict[int, GroupState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scale != FIXED_SCALE:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "unsupported fixed-point scale",
                path="$aggregation.scale",
                details={"scale": self.scale, "supported": FIXED_SCALE},
            )

    def consume_page(self, page: PageData) -> None:
        try:
            group_column = page.column(self.group_column)
            value_column = page.column(self.value_column)
        except KeyError as exc:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "aggregation column is missing",
                path=f"$page[{page.page_id}].columns",
                page_id=page.page_id,
                details={"missing_column": str(exc.args[0])},
            ) from exc
        if group_column.kind is not ColumnKind.INT:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "group column must be an integer column",
                path=f"$page[{page.page_id}].columns[{self.group_column}].kind",
                page_id=page.page_id,
            )
        self.consume_rows(group_column, value_column)

    def consume_rows(self, group_column: PageColumn, value_column: PageColumn) -> None:
        if len(group_column.values) != len(value_column.values):
            raise EngineError(ErrorCode.E_FORMAT, "aggregation columns have different row counts", path="$aggregation.rows")
        for row_index, (group_value, value) in enumerate(zip(group_column.values, value_column.values, strict=True)):
            if group_value is None or value is None:
                continue
            if isinstance(group_value, bool) or not isinstance(group_value, int):
                raise EngineError(
                    ErrorCode.E_FORMAT,
                    "group value is not an integer",
                    path=f"$aggregation.group.values[{row_index}]",
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EngineError(
                    ErrorCode.E_FORMAT,
                    "value is not numeric",
                    path=f"$aggregation.value.values[{row_index}]",
                )
            quantized = quantize_value(value, self.scale)
            state = self.groups.get(group_value)
            if state is None:
                state = GroupState()
                self.groups[group_value] = state
            state.count += 1
            state.sum_scaled += quantized

    def state(self) -> dict[str, Any]:
        return {
            "fixed_point_version": FIXED_POINT_VERSION,
            "scale": self.scale,
            "group_column": self.group_column,
            "value_column": self.value_column,
            "groups": [self.groups[key].to_dict(key) for key in sorted(self.groups)],
        }

    def load_state(self, state: dict[str, Any]) -> None:
        try:
            version = state["fixed_point_version"]
            scale = int(state["scale"])
            rows = state["groups"]
        except (KeyError, TypeError, ValueError) as exc:
            raise EngineError(ErrorCode.E_REPLAY, "aggregation state is malformed", path="$snapshot.aggregation") from exc
        if version != FIXED_POINT_VERSION or scale != self.scale:
            raise EngineError(
                ErrorCode.E_REPLAY,
                "fixed-point semantics changed; aggregation cannot be replayed",
                path="$snapshot.aggregation.fixed_point_version",
                details={"snapshot_version": version, "current_version": FIXED_POINT_VERSION, "basis": "fixed_point_version_changed"},
            )
        parsed: dict[int, GroupState] = {}
        previous_key: int | None = None
        for index, row in enumerate(rows):
            try:
                key = int(row["key"])
                count = int(row["count"])
                sum_scaled = int(row["sum_scaled"])
            except (KeyError, TypeError, ValueError) as exc:
                raise EngineError(
                    ErrorCode.E_REPLAY,
                    "aggregation group is malformed",
                    path=f"$snapshot.aggregation.groups[{index}]",
                ) from exc
            if previous_key is not None and key <= previous_key:
                raise EngineError(
                    ErrorCode.E_REPLAY,
                    "aggregation groups are not in strict key order",
                    path=f"$snapshot.aggregation.groups[{index}].key",
                )
            if count <= 0:
                raise EngineError(
                    ErrorCode.E_REPLAY,
                    "aggregation group count must be positive",
                    path=f"$snapshot.aggregation.groups[{index}].count",
                )
            previous_key = key
            parsed[key] = GroupState(count=count, sum_scaled=sum_scaled)
        self.groups = parsed

    def output(self) -> dict[str, Any]:
        groups: list[dict[str, Any]] = []
        for key in sorted(self.groups):
            state = self.groups[key]
            mean_scaled = truncate_toward_zero(state.sum_scaled, state.count)
            groups.append(
                {
                    "key": key,
                    "count": state.count,
                    "sum_scaled": state.sum_scaled,
                    "mean_scaled": mean_scaled,
                    "mean": format_scaled(mean_scaled, self.scale),
                    "sum_abs_error_upper_bound": {
                        "numerator": state.count,
                        "denominator": 2 * self.scale,
                    },
                }
            )
        return {
            "canonical_version": 1,
            "fixed_point": {
                "version": FIXED_POINT_VERSION,
                "scale": self.scale,
                "value_quantization": "decimal_str_round_half_even",
                "mean_rounding": "truncate_toward_zero",
                "value_abs_error_upper_bound": {
                    "numerator": 1,
                    "denominator": 2 * self.scale,
                },
                "mean_abs_error_upper_bound": {
                    "numerator": 3,
                    "denominator": 2 * self.scale,
                },
            },
            "groups": groups,
        }

    def output_bytes(self) -> bytes:
        return canonical_bytes(self.output())

    def write_output(self, path: str | Path) -> bytes:
        data = self.output_bytes()
        atomic_write_bytes(path, data)
        return data

    def clone(self) -> "FixedPointAggregator":
        clone = FixedPointAggregator(self.group_column, self.value_column, self.scale)
        clone.groups = {key: GroupState(state.count, state.sum_scaled) for key, state in self.groups.items()}
        return clone


def quantize_value(value: int | float, scale: int = FIXED_SCALE) -> int:
    if isinstance(value, bool):
        raise EngineError(ErrorCode.E_FORMAT, "boolean is not a numeric aggregation value", path="$aggregation.value")
    if isinstance(value, int):
        return value * scale
    if not isinstance(value, float):
        raise EngineError(ErrorCode.E_FORMAT, "unsupported aggregation value type", path="$aggregation.value")
    with localcontext() as context:
        context.prec = 50
        decimal_value = Decimal(str(value)) * Decimal(scale)
        quantized = decimal_value.to_integral_value(rounding=ROUND_HALF_EVEN)
    return int(quantized)


def truncate_toward_zero(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def format_scaled(value: int, scale: int = FIXED_SCALE) -> str:
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    whole, fraction = divmod(magnitude, scale)
    return f"{sign}{whole}.{fraction:0{len(str(scale)) - 1}d}"


def quantized_mean_error(mean_scaled: int, count: int, scale: int, true_sum: Fraction) -> Fraction:
    reconstructed = Fraction(mean_scaled, scale)
    return abs(reconstructed - true_sum / count)


def quantized_sum_error(sum_scaled: int, scale: int, true_sum: Fraction) -> Fraction:
    return abs(Fraction(sum_scaled, scale) - true_sum)
