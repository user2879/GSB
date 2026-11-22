"""Per-column quantile calibration and fixed-point scale/zero-point derivation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .config import CalibrationConfig
from .hashing import fingerprint_rows
from .keys import calibration_keys
from .table import FeatureRow, FeatureTable


@dataclass(frozen=True)
class QuantileResult:
    value: float
    position: float
    lower_index: int
    upper_index: int
    lower_value: float
    upper_value: float
    neighbour_tie: bool

    def as_record(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "position": self.position,
            "lower_index": self.lower_index,
            "upper_index": self.upper_index,
            "lower_value": self.lower_value,
            "upper_value": self.upper_value,
            "neighbour_tie": self.neighbour_tie,
        }


def round_half_away_from_zero(value: float) -> int:
    """Deterministic rounding independent of Python's bankers-rounding mode."""

    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def linear_quantile(sorted_values: list[float], permille: int, tie_break: str) -> QuantileResult:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if permille < 0 or permille > 1000:
        raise ValueError("permille must be within [0, 1000]")
    if tie_break != "lower":
        raise ValueError("only the fixed lower tie-break policy is supported")
    if len(sorted_values) == 1:
        return QuantileResult(
            value=sorted_values[0],
            position=0.0,
            lower_index=0,
            upper_index=0,
            lower_value=sorted_values[0],
            upper_value=sorted_values[0],
            neighbour_tie=False,
        )
    position = (len(sorted_values) - 1) * (permille / 1000.0)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index or lower_value == upper_value:
        value = lower_value
    else:
        fraction = position - lower_index
        value = lower_value + (upper_value - lower_value) * fraction
    if value == 0.0:
        value = 0.0
    return QuantileResult(
        value=value,
        position=position,
        lower_index=lower_index,
        upper_index=upper_index,
        lower_value=lower_value,
        upper_value=upper_value,
        neighbour_tie=(lower_index != upper_index and lower_value == upper_value),
    )


def _fixed_scale(scale_raw: float, fractional_bits: int) -> tuple[int, float]:
    factor = 1 << fractional_bits
    units = max(1, round_half_away_from_zero(scale_raw * factor))
    return units, units / factor


def _derive_parameters(
    *,
    low: float,
    high: float,
    scheme: str,
    bit_width: int,
    fractional_bits: int,
) -> dict[str, Any]:
    if bit_width < 2:
        raise ValueError("bit width must be at least 2")
    if scheme == "symmetric":
        qmin = -(1 << (bit_width - 1))
        qmax = (1 << (bit_width - 1)) - 1
        bound = max(abs(low), abs(high))
        if bound == 0.0:
            scale_raw = 1.0 / (1 << fractional_bits)
            degenerate = True
        else:
            scale_raw = bound / float(qmax)
            degenerate = False
        scale_units, scale = _fixed_scale(scale_raw, fractional_bits)
        zero_point_unclipped = 0
        zero_point = 0
        zero_point_clipped = False
    elif scheme == "asymmetric":
        qmin = 0
        qmax = (1 << bit_width) - 1
        span = high - low
        if span <= 0.0:
            scale_raw = 1.0 / (1 << fractional_bits)
            degenerate = True
        else:
            scale_raw = span / float(qmax - qmin)
            degenerate = False
        scale_units, scale = _fixed_scale(scale_raw, fractional_bits)
        zero_point_unclipped = round_half_away_from_zero(qmin - low / scale)
        zero_point = min(qmax, max(qmin, zero_point_unclipped))
        zero_point_clipped = zero_point != zero_point_unclipped
    else:
        raise ValueError(f"unsupported scheme {scheme!r}")

    return {
        "scheme": scheme,
        "qmin": qmin,
        "qmax": qmax,
        "scale_raw": scale_raw,
        "scale_factor_fixed_units": scale_units,
        "scale_factor": scale,
        "zero_point_unclipped": zero_point_unclipped,
        "zero_point": zero_point,
        "zero_point_clipped": zero_point_clipped,
        "degenerate_range": degenerate,
    }


def _calibrate_values(
    values: list[float],
    *,
    scheme: str,
    lower_permille: int,
    upper_permille: int,
    bit_width: int,
    fractional_bits: int,
    tie_break: str,
) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot calibrate an all-missing column")
    ordered = sorted(float(value) for value in values)
    low_q = linear_quantile(ordered, lower_permille, tie_break)
    high_q = linear_quantile(ordered, upper_permille, tie_break)
    parameters = _derive_parameters(
        low=low_q.value,
        high=high_q.value,
        scheme=scheme,
        bit_width=bit_width,
        fractional_bits=fractional_bits,
    )
    duplicate_count = sum(
        1 for index in range(1, len(ordered)) if ordered[index] == ordered[index - 1]
    )
    return {
        "nonmissing_count": len(ordered),
        "missing_count": None,
        "observed_min": ordered[0],
        "observed_max": ordered[-1],
        "duplicate_value_count": duplicate_count,
        "quantile_low": low_q.as_record(),
        "quantile_high": high_q.as_record(),
        **parameters,
    }


def _column_content_fingerprint(rows: Iterable[FeatureRow], column: str) -> str:
    return fingerprint_rows(
        (
            (row.row_id, row.group_id, {column: row.values.get(column)})
            for row in rows
        ),
        [column],
        include_group=False,
    )


def calibrate_group_columns(
    config: CalibrationConfig,
    table: FeatureTable,
    column_groups: dict[str, tuple[str, ...]],
    group_id: str,
    bit_width: int,
) -> dict[str, Any]:
    """Build the immutable quantile-calibration stage artifact for one group/width."""

    columns = tuple(sorted(column_groups[group_id]))
    group_rows = sorted(table.rows_for_group(group_id), key=lambda row: row.row_id)
    if not group_rows:
        raise ValueError(f"group {group_id!r} has no rows")

    column_records: dict[str, Any] = {}
    reference_records: dict[str, Any] = {}
    for column in columns:
        values = [row.values[column] for row in group_rows if row.values.get(column) is not None]
        assert all(value is not None for value in values)
        typed_values = [float(value) for value in values if value is not None]
        if not typed_values:
            raise ValueError(f"group {group_id!r} column {column!r} has no non-missing values")
        record = _calibrate_values(
            typed_values,
            scheme=config.scheme,
            lower_permille=config.quantile_lower_permille,
            upper_permille=config.quantile_upper_permille,
            bit_width=bit_width,
            fractional_bits=config.quantization.fractional_bits,
            tie_break=config.tie_break,
        )
        record["missing_count"] = len(group_rows) - len(typed_values)
        record["column_content_fingerprint"] = _column_content_fingerprint(group_rows, column)
        column_records[column] = record
        reference_records[column] = _calibrate_values(
            typed_values,
            scheme=config.scheme,
            lower_permille=0,
            upper_permille=1000,
            bit_width=bit_width,
            fractional_bits=config.quantization.fractional_bits,
            tie_break=config.tie_break,
        )
    keys = calibration_keys(config, table, group_id, columns, bit_width)
    return {
        "stage": "quantile_calibration",
        "task_sort_key": f"{group_id}|quantile_calibration|{bit_width:03d}",
        "group_id": group_id,
        "columns": list(columns),
        "row_ids": [row.row_id for row in group_rows],
        "bit_width": bit_width,
        "fractional_bits": config.quantization.fractional_bits,
        "active_bit_width": bit_width == config.active_bit_width,
        "qmin": column_records[columns[0]]["qmin"],
        "qmax": column_records[columns[0]]["qmax"],
        "interpolation": "linear_type7",
        "tie_break": "lower_order_statistic",
        "tie_policy": "equal neighbouring values resolve to the lower order statistic",
        "quantile_permille": [config.quantile_lower_permille, config.quantile_upper_permille],
        "scheme": config.scheme,
        "columns_calibration": column_records,
        "reference_full_range": reference_records,
        "content_fingerprint": keys["content_fingerprint"],
        "parameter_fingerprint": keys["parameter_fingerprint"],
        "cache_key": keys["cache_key"],
    }
