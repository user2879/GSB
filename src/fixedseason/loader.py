"""Loading and validation for series JSONL, YAML config and JSONL commands."""

from __future__ import annotations

import json
import math
import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import E_COMMAND_INVALID, E_CONFIG_INVALID, E_SERIES_INVALID, Issue
from .jsonutil import loads_strict
from .model import (
    ALLOWED_FIXED_POINT_BITS,
    ALLOWED_QUANTIZATION_SHIFTS,
    MAX_HARMONIC_ORDER,
    MAX_REGULARIZATION,
    MISSING_MARKERS,
    SCALE_PRESETS,
    SCALE_PROFILES,
    FitConfig,
    SeriesRecord,
)
from .yamlmini import YamlError, loads as yaml_loads


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_observation_value(value: Any) -> tuple[float | None, str | None]:
    """Return (value, error). Missing markers map to None."""

    if value is None:
        return None, None
    if isinstance(value, str):
        if value in MISSING_MARKERS:
            return None, None
        return None, f"observation string {value!r} is not a missing marker"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, f"observation must be a finite number or missing marker, got {type(value).__name__}"
    number = float(value)
    if not math.isfinite(number):
        return None, "observation must be finite"
    return number, None


def validate_series_object(
    obj: Any,
    *,
    source: str = "",
    line: int | None = None,
) -> tuple[SeriesRecord | None, list[Issue]]:
    issues: list[Issue] = []

    def fail(message: str, series_id: str | None = None) -> None:
        issues.append(
            Issue(
                code=E_SERIES_INVALID,
                message=message,
                source=source,
                line=line,
                series_id=series_id,
            )
        )

    if not isinstance(obj, dict):
        fail("series line must be a JSON object")
        return None, issues

    series_id = obj.get("series_id")
    if not isinstance(series_id, str) or not series_id.strip():
        fail("series_id must be a non-empty string", None)
        series_id = None
    timestamps = obj.get("timestamps")
    values = obj.get("values")
    if not isinstance(timestamps, list):
        fail("timestamps must be a list", series_id)
        timestamps = []
    if not isinstance(values, list):
        fail("values must be a list", series_id)
        values = []
    if len(timestamps) != len(values):
        fail("timestamps and values must have equal length", series_id)
    if len(timestamps) < 2:
        fail("a series needs at least two observations slots", series_id)

    parsed_timestamps: list[float] = []
    for index, raw in enumerate(timestamps):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            fail(f"timestamp[{index}] must be numeric", series_id)
            parsed_timestamps.append(float("nan"))
            continue
        number = float(raw)
        if not math.isfinite(number):
            fail(f"timestamp[{index}] must be finite", series_id)
            parsed_timestamps.append(float("nan"))
        else:
            parsed_timestamps.append(number)

    parsed_values: list[float | None] = []
    for index, raw in enumerate(values):
        value, error = parse_observation_value(raw)
        if error is not None:
            fail(f"values[{index}] {error}", series_id)
            parsed_values.append(None)
        else:
            parsed_values.append(value)

    if not issues:
        decimals = [Decimal(str(value)) for value in parsed_timestamps]
        differences = [
            decimals[index + 1] - decimals[index]
            for index in range(len(decimals) - 1)
        ]
        if any(diff <= 0 for diff in differences):
            fail("timestamps must be strictly increasing", series_id)
        elif differences and any(diff != differences[0] for diff in differences[1:]):
            fail("timestamps must have a constant positive interval", series_id)
        if not any(value is not None for value in parsed_values):
            fail("at least one observed value is required for deterministic imputation", series_id)

    if issues:
        return None, issues
    assert series_id is not None
    return (
        SeriesRecord(
            series_id=series_id,
            timestamps=tuple(parsed_timestamps),
            values=tuple(parsed_values),
            source=source,
            source_line=line or 0,
        ),
        [],
    )


def load_series_jsonl(path: str | Path) -> tuple[list[SeriesRecord], list[Issue]]:
    path = Path(path)
    series: list[SeriesRecord] = []
    issues: list[Issue] = []
    seen: set[str] = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [
            Issue(
                code=E_SERIES_INVALID,
                message=f"cannot read series file: {exc}",
                source=str(path),
            )
        ]

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = loads_strict(line)
        except (json.JSONDecodeError, ValueError) as exc:
            issues.append(
                Issue(
                    code=E_SERIES_INVALID,
                    message=f"invalid JSON: {exc}",
                    source=str(path),
                    line=line_number,
                )
            )
            continue
        record, record_issues = validate_series_object(
            obj, source=str(path), line=line_number
        )
        if record_issues:
            issues.extend(record_issues)
            continue
        assert record is not None
        if record.series_id in seen:
            issues.append(
                Issue(
                    code=E_SERIES_INVALID,
                    message=f"duplicate series_id {record.series_id!r}; first occurrence retained",
                    source=str(path),
                    line=line_number,
                    series_id=record.series_id,
                )
            )
            continue
        seen.add(record.series_id)
        series.append(record)
    return series, issues


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return repr(value)


def load_config(path: str | Path) -> tuple[FitConfig | None, list[Issue]]:
    path = Path(path)
    issues: list[Issue] = []

    def fail(message: str) -> None:
        issues.append(Issue(code=E_CONFIG_INVALID, message=message, source=str(path)))

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [
            Issue(
                code=E_CONFIG_INVALID,
                message=f"cannot read config file: {exc}",
                source=str(path),
            )
        ]

    try:
        data = yaml_loads(text)
    except YamlError as exc:
        return None, [
            Issue(
                code=E_CONFIG_INVALID,
                message=f"invalid YAML: {exc}",
                source=str(path),
                line=exc.line,
            )
        ]
    if not isinstance(data, dict):
        return None, [
            Issue(
                code=E_CONFIG_INVALID,
                message="config root must be a mapping",
                source=str(path),
            )
        ]

    defaults = FitConfig()

    def section(name: str) -> dict[str, Any]:
        value = data.get(name, {})
        if value is None:
            return {}
        if not isinstance(value, dict):
            fail(f"{name} must be a mapping")
            return {}
        return value

    model = section("model")
    fixed = section("fixed_point")
    tolerance = section("tolerance")
    budget = section("budget")
    runtime = section("runtime")
    diagnostics = section("diagnostics")

    def choose(nested: dict[str, Any], section_name: str, key: str, default: Any) -> Any:
        if key in nested:
            return nested[key]
        if key in data:
            return data[key]
        full_key = f"{section_name}_{key}" if section_name else key
        if full_key in data:
            return data[full_key]
        return default

    period = choose(model, "", "period", defaults.period)
    harmonic_order = choose(model, "", "harmonic_order", defaults.harmonic_order)
    regularization = choose(
        model, "", "regularization_strength", defaults.regularization_strength
    )
    bits = choose(fixed, "", "bits", choose(fixed, "", "fixed_point_bits", defaults.fixed_point_bits))
    quantization_shift = choose(
        fixed,
        "",
        "quantization_shift",
        choose(fixed, "", "quantization_tier_shift", defaults.quantization_shift),
    )
    pivot_threshold = choose(
        fixed, "", "pivot_threshold", defaults.pivot_threshold
    )
    tolerance_relative = choose(
        tolerance, "", "relative", defaults.tolerance_relative
    )
    tolerance_absolute = choose(
        tolerance, "", "absolute", defaults.tolerance_absolute
    )
    scale_profile = choose(
        budget, "", "scale_profile", data.get("scale_profile", defaults.scale_profile)
    )
    if not isinstance(scale_profile, str) or scale_profile not in SCALE_PROFILES:
        fail(f"scale_profile must be one of {list(SCALE_PROFILES)}, got {_fmt(scale_profile)}")
        scale_profile = defaults.scale_profile
    preset = SCALE_PRESETS[scale_profile]
    global_fit_points = choose(
        budget, "", "global_fit_points", preset["global_fit_points"]
    )
    per_series_quota = choose(
        budget, "", "per_series_quota", preset["per_series_quota"]
    )
    batch_observation_points = choose(
        budget, "", "batch_observation_points", preset["batch_observation_points"]
    )
    workers = choose(
        runtime,
        "",
        "thread_pool_workers",
        choose(runtime, "", "workers", defaults.thread_pool_workers),
    )
    version = data.get("version", defaults.version)
    comparison_widths = diagnostics.get(
        "comparison_bit_widths", data.get("comparison_bit_widths", [])
    )
    if comparison_widths is None:
        comparison_widths = []
    if not isinstance(comparison_widths, list):
        fail("comparison_bit_widths must be a list")
        comparison_widths = []

    if isinstance(period, bool) or not isinstance(period, int) or not (2 <= period <= 366):
        fail(f"period must be an integer in [2, 366], got {_fmt(period)}")
        period = defaults.period
    if (
        isinstance(harmonic_order, bool)
        or not isinstance(harmonic_order, int)
        or not (0 <= harmonic_order <= MAX_HARMONIC_ORDER)
    ):
        fail(
            f"harmonic_order must be an integer in [0, {MAX_HARMONIC_ORDER}], "
            f"got {_fmt(harmonic_order)}"
        )
        harmonic_order = defaults.harmonic_order
    if (
        isinstance(regularization, bool)
        or not _is_number(regularization)
        or not math.isfinite(float(regularization))
        or not (0.0 <= float(regularization) <= MAX_REGULARIZATION)
    ):
        fail(
            f"regularization_strength must be a finite number in [0, {MAX_REGULARIZATION}], "
            f"got {_fmt(regularization)}"
        )
        regularization = defaults.regularization_strength
    else:
        regularization = float(regularization)
    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in ALLOWED_FIXED_POINT_BITS:
        fail(
            f"fixed_point.bits must be one of {list(ALLOWED_FIXED_POINT_BITS)}, "
            f"got {_fmt(bits)}"
        )
        bits = defaults.fixed_point_bits
    if (
        isinstance(quantization_shift, bool)
        or not isinstance(quantization_shift, int)
        or quantization_shift not in ALLOWED_QUANTIZATION_SHIFTS
    ):
        fail(
            "quantization_shift must be one of "
            f"{list(ALLOWED_QUANTIZATION_SHIFTS)}, got {_fmt(quantization_shift)}"
        )
        quantization_shift = defaults.quantization_shift
    if quantization_shift > bits - 8:
        fail(
            f"quantization_shift {quantization_shift} leaves fewer than 8 integer bits "
            f"for fixed_point.bits {bits}"
        )
    if (
        isinstance(pivot_threshold, bool)
        or not isinstance(pivot_threshold, int)
        or pivot_threshold < 0
        or pivot_threshold > (1 << (bits - 1)) - 1
    ):
        fail(f"pivot_threshold must be a non-negative integer fitting in {bits} bits")
        pivot_threshold = defaults.pivot_threshold
    for name, value in (
        ("tolerance.relative", tolerance_relative),
        ("tolerance.absolute", tolerance_absolute),
    ):
        if (
            isinstance(value, bool)
            or not _is_number(value)
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            fail(f"{name} must be a finite non-negative number")
    tolerance_relative = float(tolerance_relative)
    tolerance_absolute = float(tolerance_absolute)
    if tolerance_relative == 0.0 and tolerance_absolute == 0.0:
        fail("relative and absolute tolerances cannot both be zero")
    for name, value in (
        ("global_fit_points", global_fit_points),
        ("per_series_quota", per_series_quota),
        ("batch_observation_points", batch_observation_points),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            fail(f"budget.{name} must be a positive integer")
    cpu_count = os.cpu_count() or 1
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not (1 <= workers <= cpu_count)
    ):
        fail(
            f"thread_pool_workers must be an integer in [1, {cpu_count}] "
            f"(detected logical cores), got {_fmt(workers)}"
        )
        workers = min(defaults.thread_pool_workers, cpu_count)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        fail(f"version must be an integer >= 1, got {_fmt(version)}")
        version = defaults.version
    parsed_widths: list[int] = []
    for value in comparison_widths:
        if isinstance(value, bool) or not isinstance(value, int) or value not in ALLOWED_FIXED_POINT_BITS:
            fail(f"comparison bit width {_fmt(value)} is not allowed")
        elif value not in parsed_widths:
            parsed_widths.append(value)

    if issues:
        return None, issues
    config = FitConfig(
        period=period,
        harmonic_order=harmonic_order,
        regularization_strength=regularization,
        fixed_point_bits=bits,
        quantization_shift=quantization_shift,
        pivot_threshold=pivot_threshold,
        tolerance_relative=tolerance_relative,
        tolerance_absolute=tolerance_absolute,
        global_fit_points=global_fit_points,
        per_series_quota=per_series_quota,
        batch_observation_points=batch_observation_points,
        thread_pool_workers=workers,
        scale_profile=scale_profile,
        version=version,
        comparison_bit_widths=tuple(parsed_widths),
        raw=data,
    )
    return config, []


def load_command_file(path: str | Path) -> tuple[list[tuple[int, dict[str, Any]]], list[Issue]]:
    """Parse command JSONL lines; semantic validation happens in commands.py."""

    path = Path(path)
    commands: list[tuple[int, dict[str, Any]]] = []
    issues: list[Issue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [
            Issue(
                code=E_COMMAND_INVALID,
                message=f"cannot read command file: {exc}",
                source=str(path),
            )
        ]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = loads_strict(line)
        except (json.JSONDecodeError, ValueError) as exc:
            issues.append(
                Issue(
                    code=E_COMMAND_INVALID,
                    message=f"invalid JSON: {exc}",
                    source=str(path),
                    line=line_number,
                )
            )
            continue
        if not isinstance(obj, dict):
            issues.append(
                Issue(
                    code=E_COMMAND_INVALID,
                    message="command line must be a JSON object",
                    source=str(path),
                    line=line_number,
                )
            )
            continue
        commands.append((line_number, obj))
    return commands, issues
