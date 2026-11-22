"""Configuration loading and strict validation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

from .errors import E_CONFIG_INVALID, Failure, failure_sort_key
from .hashing import sha256_value
from .version import SCHEMA_VERSION


ALLOWED_BIT_WIDTHS = {2, 3, 4, 5, 6, 8, 10, 12, 16}
SUPPORTED_SCHEMES = {"symmetric", "asymmetric"}
SUPPORTED_CLIPPING = {"saturate"}
SUPPORTED_INTERPOLATION = {"linear"}
SUPPORTED_TIE_BREAK = {"lower"}
SUPPORTED_TOLERANCE_COMPARISON = {"either", "both"}


@dataclass(frozen=True)
class QuantizationConfig:
    bit_width_candidates: tuple[int, ...]
    fractional_bits: int
    gear: int

    @property
    def active_bit_width(self) -> int:
        return self.bit_width_candidates[self.gear]


@dataclass(frozen=True)
class ToleranceConfig:
    absolute: float
    relative: float
    comparison: str
    outlier_ratio_limit: float
    tie_epsilon: float
    zero_relative_branch: str


@dataclass(frozen=True)
class CalibrationConfig:
    raw: dict[str, Any]
    config_hash: str
    table_name: str
    tensor_dimensions: tuple[int | None, ...]
    column_axis: int
    row_id_field: str
    group_id_field: str
    missing_value_markers: tuple[Any, ...]
    column_groups: dict[str, tuple[str, ...]]
    quantile_lower_permille: int
    quantile_upper_permille: int
    scheme: str
    quantization: QuantizationConfig
    clipping_strategy: str
    tolerance: ToleranceConfig
    thread_pool_size: int
    element_budget_per_batch: int
    interpolation: str
    tie_break: str

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.column_groups))

    @property
    def all_columns(self) -> tuple[str, ...]:
        return tuple(sorted(column for cols in self.column_groups.values() for column in cols))

    @property
    def active_bit_width(self) -> int:
        return self.quantization.active_bit_width

    @property
    def bit_widths(self) -> tuple[int, ...]:
        return tuple(self.quantization.bit_width_candidates)

    def calibration_parameters(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "quantile_permille": [self.quantile_lower_permille, self.quantile_upper_permille],
            "interpolation": self.interpolation,
            "tie_break": self.tie_break,
            "fractional_bits": self.quantization.fractional_bits,
        }

    def parameter_fingerprint(self) -> str:
        return sha256_value(
            {
                "table_name": self.table_name,
                "column_axis": self.column_axis,
                "calibration": self.calibration_parameters(),
                "quantization": {
                    "candidates": list(self.quantization.bit_width_candidates),
                    "fractional_bits": self.quantization.fractional_bits,
                    "gear": self.quantization.gear,
                },
                "clipping_strategy": self.clipping_strategy,
                "tolerance": {
                    "absolute": self.tolerance.absolute,
                    "relative": self.tolerance.relative,
                    "comparison": self.tolerance.comparison,
                    "outlier_ratio_limit": self.tolerance.outlier_ratio_limit,
                    "tie_epsilon": self.tolerance.tie_epsilon,
                    "zero_relative_branch": self.tolerance.zero_relative_branch,
                },
                "schema_version": SCHEMA_VERSION,
            }
        )


@dataclass
class ConfigLoadResult:
    config: CalibrationConfig | None
    failures: list[Failure]
    warnings: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return self.config is not None and not self.failures


def _failure(message: str, field: str, *, tag: str = "config") -> Failure:
    return Failure(
        code=E_CONFIG_INVALID,
        message=message,
        sort_key=failure_sort_key(source="config", code=E_CONFIG_INVALID, field=field, message=message),
        details={"field": field, "tag": tag},
    )


def _as_int(value: Any, name: str, failures: list[Failure], *, minimum: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        failures.append(_failure(f"{name} must be an integer", name))
        return None
    if minimum is not None and value < minimum:
        failures.append(_failure(f"{name} must be >= {minimum}", name))
        return None
    return value


def _as_float(
    value: Any,
    name: str,
    failures: list[Failure],
    *,
    minimum: float | None = None,
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        failures.append(_failure(f"{name} must be a finite number", name))
        return None
    result = float(value)
    if not math.isfinite(result):
        failures.append(_failure(f"{name} must be finite", name))
        return None
    if minimum is not None and result < minimum:
        failures.append(_failure(f"{name} must be >= {minimum}", name))
        return None
    return result


def _load_json_object(path: str) -> tuple[dict[str, Any] | None, Failure | None]:
    if not os.path.exists(path):
        message = f"configuration file does not exist: {path}"
        return None, Failure(
            code=E_CONFIG_INVALID,
            message=message,
            sort_key=failure_sort_key(source="config", code=E_CONFIG_INVALID, field="path", message=message),
            details={"field": "path", "tag": "config", "path": path},
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        message = f"cannot read configuration JSON: {exc}"
        return None, Failure(
            code=E_CONFIG_INVALID,
            message=message,
            sort_key=failure_sort_key(source="config", code=E_CONFIG_INVALID, field="json", message=message),
            details={"field": "json", "tag": "config"},
        )
    if not isinstance(value, dict):
        message = "configuration root must be a JSON object"
        return None, Failure(
            code=E_CONFIG_INVALID,
            message=message,
            sort_key=failure_sort_key(source="config", code=E_CONFIG_INVALID, field="$", message=message),
            details={"field": "$", "tag": "config"},
        )
    return value, None


def load_config(path: str | os.PathLike[str]) -> ConfigLoadResult:
    failures: list[Failure] = []
    raw, initial_failure = _load_json_object(os.fspath(path))
    if initial_failure is not None:
        return ConfigLoadResult(None, [initial_failure], [])
    assert raw is not None

    schema_version = raw.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        failures.append(_failure(f"schema_version must be {SCHEMA_VERSION}", "schema_version"))

    table_name = raw.get("table_name")
    if not isinstance(table_name, str) or not table_name.strip():
        failures.append(_failure("table_name must be a non-empty string", "table_name"))
        table_name = ""

    tensor_dimensions_raw = raw.get("tensor_dimensions")
    tensor_dimensions: tuple[int | None, ...] = ()
    if not isinstance(tensor_dimensions_raw, list) or not tensor_dimensions_raw:
        failures.append(_failure("tensor_dimensions must be a non-empty array", "tensor_dimensions"))
    else:
        parsed_dimensions: list[int | None] = []
        for index, dimension in enumerate(tensor_dimensions_raw):
            field = f"tensor_dimensions[{index}]"
            if dimension is None:
                parsed_dimensions.append(None)
            elif isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
                failures.append(_failure(f"{field} must be a positive integer or null", field))
            else:
                parsed_dimensions.append(dimension)
        tensor_dimensions = tuple(parsed_dimensions)

    column_axis = _as_int(raw.get("column_axis"), "column_axis", failures, minimum=0)
    if column_axis is not None and tensor_dimensions and column_axis >= len(tensor_dimensions):
        failures.append(
            _failure(
                "column_axis is outside tensor_dimensions",
                "column_axis",
            )
        )

    row_id_field = raw.get("row_id_field", "row_id")
    group_id_field = raw.get("group_id_field", "group_id")
    if not isinstance(row_id_field, str) or not row_id_field:
        failures.append(_failure("row_id_field must be a non-empty string", "row_id_field"))
        row_id_field = "row_id"
    if not isinstance(group_id_field, str) or not group_id_field:
        failures.append(_failure("group_id_field must be a non-empty string", "group_id_field"))
        group_id_field = "group_id"
    if row_id_field == group_id_field:
        failures.append(_failure("row_id_field and group_id_field must differ", "row_id_field"))

    missing_raw = raw.get("missing_value_markers", [None, "__MISSING__"])
    missing_markers: list[Any] = []
    if not isinstance(missing_raw, list) or not missing_raw:
        failures.append(_failure("missing_value_markers must be a non-empty array", "missing_value_markers"))
    else:
        for index, marker in enumerate(missing_raw):
            if isinstance(marker, (dict, list)):
                failures.append(
                    _failure(
                        f"missing_value_markers[{index}] must be a scalar",
                        f"missing_value_markers[{index}]",
                    )
                )
            else:
                missing_markers.append(marker)

    groups_raw = raw.get("column_groups")
    column_groups: dict[str, tuple[str, ...]] = {}
    seen_columns: dict[str, str] = {}
    if not isinstance(groups_raw, dict) or not groups_raw:
        failures.append(_failure("column_groups must be a non-empty object", "column_groups"))
    else:
        for group_id in sorted(groups_raw, key=str):
            group_field = f"column_groups.{group_id}"
            if not isinstance(group_id, str) or not group_id:
                failures.append(_failure("column group identifiers must be non-empty strings", group_field))
                continue
            columns_raw = groups_raw[group_id]
            if not isinstance(columns_raw, list) or not columns_raw:
                failures.append(_failure("each column group must contain at least one column", group_field))
                continue
            columns: list[str] = []
            local_seen: set[str] = set()
            for index, column in enumerate(columns_raw):
                if not isinstance(column, str) or not column:
                    failures.append(
                        _failure(
                            f"{group_field}[{index}] must be a non-empty string",
                            f"{group_field}[{index}]",
                        )
                    )
                    continue
                if column in local_seen:
                    failures.append(_failure(f"duplicate column {column!r} in group", group_field))
                    continue
                local_seen.add(column)
                if column in seen_columns:
                    failures.append(
                        _failure(
                            f"column {column!r} appears in multiple groups",
                            group_field,
                        )
                    )
                    continue
                seen_columns[column] = group_id
                columns.append(column)
            if columns:
                column_groups[group_id] = tuple(sorted(columns))

    quantile_raw = raw.get("quantile_permille")
    lower: int | None = None
    upper: int | None = None
    if not isinstance(quantile_raw, dict):
        failures.append(_failure("quantile_permille must be an object", "quantile_permille"))
    else:
        lower = _as_int(quantile_raw.get("lower"), "quantile_permille.lower", failures, minimum=0)
        upper = _as_int(quantile_raw.get("upper"), "quantile_permille.upper", failures, minimum=0)
        if lower is not None and lower > 1000:
            failures.append(_failure("quantile_permille.lower must be <= 1000", "quantile_permille.lower"))
        if upper is not None and upper > 1000:
            failures.append(_failure("quantile_permille.upper must be <= 1000", "quantile_permille.upper"))
        if lower is not None and upper is not None and lower >= upper:
            failures.append(
                _failure(
                    "quantile_permille.lower must be strictly less than upper",
                    "quantile_permille",
                )
            )

    scheme = raw.get("scheme")
    if scheme not in SUPPORTED_SCHEMES:
        failures.append(_failure(f"scheme must be one of {sorted(SUPPORTED_SCHEMES)}", "scheme"))
        scheme = "asymmetric"

    quantization_raw = raw.get("quantization")
    bit_widths: list[int] = []
    fractional_bits: int | None = None
    gear: int | None = None
    if not isinstance(quantization_raw, dict):
        failures.append(_failure("quantization must be an object", "quantization"))
    else:
        widths_raw = quantization_raw.get("bit_width_candidates")
        if not isinstance(widths_raw, list) or not widths_raw:
            failures.append(
                _failure("quantization.bit_width_candidates must be a non-empty array", "quantization.bit_width_candidates")
            )
        else:
            for index, width in enumerate(widths_raw):
                field = f"quantization.bit_width_candidates[{index}]"
                if isinstance(width, bool) or not isinstance(width, int):
                    failures.append(_failure(f"{field} must be an integer", field))
                    continue
                if width not in ALLOWED_BIT_WIDTHS:
                    failures.append(
                        _failure(
                            f"{field} must be one of {sorted(ALLOWED_BIT_WIDTHS)}",
                            field,
                        )
                    )
                    continue
                if width in bit_widths:
                    failures.append(_failure(f"{field} must not duplicate another width", field))
                    continue
                bit_widths.append(width)
        fractional_bits = _as_int(
            quantization_raw.get("fractional_bits"),
            "quantization.fractional_bits",
            failures,
            minimum=0,
        )
        if fractional_bits is not None and bit_widths and fractional_bits >= min(bit_widths):
            failures.append(
                _failure(
                    "quantization.fractional_bits must be less than every candidate bit width",
                    "quantization.fractional_bits",
                )
            )
        gear = _as_int(quantization_raw.get("gear"), "quantization.gear", failures, minimum=0)
        if gear is not None and bit_widths and gear >= len(bit_widths):
            failures.append(
                _failure(
                    "quantization.gear must index an entry in bit_width_candidates",
                    "quantization.gear",
                )
            )

    clipping_strategy = raw.get("clipping_strategy", "saturate")
    if clipping_strategy not in SUPPORTED_CLIPPING:
        failures.append(
            _failure(
                f"clipping_strategy must be one of {sorted(SUPPORTED_CLIPPING)}",
                "clipping_strategy",
            )
        )
        clipping_strategy = "saturate"

    tolerance_raw = raw.get("tolerance")
    absolute: float | None = None
    relative: float | None = None
    comparison = "either"
    outlier_ratio_limit: float | None = None
    tie_epsilon: float | None = None
    zero_branch = "absolute_only"
    if not isinstance(tolerance_raw, dict):
        failures.append(_failure("tolerance must be an object", "tolerance"))
    else:
        absolute = _as_float(tolerance_raw.get("absolute"), "tolerance.absolute", failures, minimum=0.0)
        relative = _as_float(tolerance_raw.get("relative"), "tolerance.relative", failures, minimum=0.0)
        comparison = tolerance_raw.get("comparison", "either")
        if comparison not in SUPPORTED_TOLERANCE_COMPARISON:
            failures.append(
                _failure(
                    f"tolerance.comparison must be one of {sorted(SUPPORTED_TOLERANCE_COMPARISON)}",
                    "tolerance.comparison",
                )
            )
        outlier_ratio_limit = _as_float(
            tolerance_raw.get("outlier_ratio_limit"),
            "tolerance.outlier_ratio_limit",
            failures,
            minimum=0.0,
        )
        if outlier_ratio_limit is not None and outlier_ratio_limit > 1.0:
            failures.append(
                _failure(
                    "tolerance.outlier_ratio_limit must be <= 1",
                    "tolerance.outlier_ratio_limit",
                )
            )
        tie_epsilon = _as_float(
            tolerance_raw.get("tie_epsilon", 1e-9),
            "tolerance.tie_epsilon",
            failures,
            minimum=0.0,
        )
        zero_branch = tolerance_raw.get("zero_relative_branch", "absolute_only")
        if zero_branch != "absolute_only":
            failures.append(
                _failure(
                    "tolerance.zero_relative_branch must be 'absolute_only'",
                    "tolerance.zero_relative_branch",
                )
            )

    thread_pool_size = _as_int(raw.get("thread_pool_size"), "thread_pool_size", failures, minimum=1)
    if thread_pool_size is not None and thread_pool_size > 64:
        failures.append(_failure("thread_pool_size must be <= 64", "thread_pool_size"))

    element_budget = _as_int(
        raw.get("element_budget_per_batch"),
        "element_budget_per_batch",
        failures,
        minimum=1,
    )

    interpolation = raw.get("interpolation", "linear")
    if interpolation not in SUPPORTED_INTERPOLATION:
        failures.append(
            _failure(
                f"interpolation must be one of {sorted(SUPPORTED_INTERPOLATION)}",
                "interpolation",
            )
        )
        interpolation = "linear"
    tie_break = raw.get("tie_break", "lower")
    if tie_break not in SUPPORTED_TIE_BREAK:
        failures.append(
            _failure(f"tie_break must be one of {sorted(SUPPORTED_TIE_BREAK)}", "tie_break")
        )
        tie_break = "lower"

    if failures:
        return ConfigLoadResult(None, failures, [])

    assert column_axis is not None
    assert lower is not None and upper is not None
    assert fractional_bits is not None and gear is not None
    assert absolute is not None and relative is not None
    assert outlier_ratio_limit is not None and tie_epsilon is not None
    assert thread_pool_size is not None and element_budget is not None

    config = CalibrationConfig(
        raw=raw,
        config_hash=sha256_value(raw),
        table_name=table_name,
        tensor_dimensions=tensor_dimensions,
        column_axis=column_axis,
        row_id_field=row_id_field,
        group_id_field=group_id_field,
        missing_value_markers=tuple(missing_markers),
        column_groups=column_groups,
        quantile_lower_permille=lower,
        quantile_upper_permille=upper,
        scheme=scheme,
        quantization=QuantizationConfig(tuple(bit_widths), fractional_bits, gear),
        clipping_strategy=clipping_strategy,
        tolerance=ToleranceConfig(
            absolute=absolute,
            relative=relative,
            comparison=comparison,
            outlier_ratio_limit=outlier_ratio_limit,
            tie_epsilon=tie_epsilon,
            zero_relative_branch=zero_branch,
        ),
        thread_pool_size=thread_pool_size,
        element_budget_per_batch=element_budget,
        interpolation=interpolation,
        tie_break=tie_break,
    )
    return ConfigLoadResult(config, [], [])

