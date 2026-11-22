"""Configuration loading and field-level validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .errors import make_failure, sort_failures
from .fingerprint import sha256_json
from .models import ConfigLoadResult, GroupSpec, TensorConfig, WorkbenchConfig

SCHEMES = ("symmetric", "asymmetric")
INTERPOLATIONS = ("linear",)
TIE_BREAKS = (
    "stable_ascending_first_occurrence",
    "stable_ascending_lowest_value",
)
SCALE_QUANTIZATIONS = ("round_half_away_from_zero",)
SATURATION_STRATEGIES = ("saturate", "wrap", "reject")
ZERO_SCALE_POLICIES = ("clamp_to_min_scale", "reject")
SCHEME_ALIASES = {
    "sym": "symmetric",
    "affine": "asymmetric",
    "asymmetric_affine": "asymmetric",
}
SATURATION_ALIASES = {
    "clip": "saturate",
    "clamp": "saturate",
    "saturate_clip": "saturate",
    "error": "reject",
    "reject_overflow": "reject",
}
ZERO_SCALE_ALIASES = {
    "fallback_min_scale": "clamp_to_min_scale",
    "clamp": "clamp_to_min_scale",
}

_MISSING = object()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _get(source: dict[str, Any], *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if name in source:
            return source[name]
    return default


def _tensor_field(
    tensor_raw: dict[str, Any],
    global_raw: dict[str, Any],
    names: Iterable[str],
    default: Any = _MISSING,
) -> Any:
    names = tuple(names)
    value = _get(tensor_raw, *names, default=_MISSING)
    if value is not _MISSING:
        return value
    return _get(global_raw, *names, default=default)


def _add_error(
    failures: list[dict[str, Any]],
    field: str,
    message: str,
    *,
    tensor_id: str | None = None,
) -> None:
    failures.append(
        make_failure(
            "E_CONFIG_INVALID",
            field,
            message,
            tensor_id=tensor_id,
        )
    )


def _validate_positive_int(
    value: Any,
    field: str,
    failures: list[dict[str, Any]],
    *,
    tensor_id: str | None = None,
    maximum: int | None = None,
) -> int | None:
    if not _is_int(value) or value <= 0:
        _add_error(failures, field, "must be a positive integer", tensor_id=tensor_id)
        return None
    if maximum is not None and value > maximum:
        _add_error(
            failures,
            field,
            f"must be <= {maximum}",
            tensor_id=tensor_id,
        )
        return None
    return int(value)


def _normalize_shape(
    value: Any,
    field: str,
    failures: list[dict[str, Any]],
    tensor_id: str | None,
) -> tuple[int, ...] | None:
    if not isinstance(value, list) or not value:
        _add_error(
            failures,
            field,
            "must be a non-empty list of positive integers",
            tensor_id=tensor_id,
        )
        return None
    shape: list[int] = []
    for index, dimension in enumerate(value):
        if not _is_int(dimension) or dimension <= 0:
            _add_error(
                failures,
                f"{field}[{index}]",
                "shape dimension must be a positive integer",
                tensor_id=tensor_id,
            )
            return None
        shape.append(int(dimension))
    return tuple(shape)


def _normalize_groups(
    value: Any,
    *,
    field: str,
    column_count: int,
    failures: list[dict[str, Any]],
    tensor_id: str,
) -> tuple[GroupSpec, ...] | None:
    if value is _MISSING:
        return (GroupSpec("all", tuple(range(column_count))),)
    if not isinstance(value, list) or not value:
        _add_error(
            failures,
            field,
            "column_groups must be a non-empty list",
            tensor_id=tensor_id,
        )
        return None

    groups: list[GroupSpec] = []
    seen_ids: set[str] = set()
    assigned: dict[int, str] = {}
    valid = True
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, dict):
            _add_error(
                failures,
                item_field,
                "group must be an object",
                tensor_id=tensor_id,
            )
            valid = False
            continue
        group_id = _get(item, "group_id", "id", default=_MISSING)
        if not isinstance(group_id, str) or not group_id.strip():
            _add_error(
                failures,
                f"{item_field}.group_id",
                "group_id must be a non-empty string",
                tensor_id=tensor_id,
            )
            valid = False
            continue
        group_id = group_id.strip()
        if group_id in seen_ids:
            _add_error(
                failures,
                f"{item_field}.group_id",
                f"duplicate group_id {group_id!r}",
                tensor_id=tensor_id,
            )
            valid = False
            continue
        seen_ids.add(group_id)
        columns_raw = _get(item, "columns", default=_MISSING)
        if not isinstance(columns_raw, list) or not columns_raw:
            _add_error(
                failures,
                f"{item_field}.columns",
                "columns must be a non-empty list",
                tensor_id=tensor_id,
            )
            valid = False
            continue
        columns: list[int] = []
        for position, column in enumerate(columns_raw):
            if not _is_int(column) or not (0 <= column < column_count):
                _add_error(
                    failures,
                    f"{item_field}.columns[{position}]",
                    f"column must be an integer in [0, {column_count - 1}]",
                    tensor_id=tensor_id,
                )
                valid = False
                continue
            if column in assigned:
                _add_error(
                    failures,
                    f"{item_field}.columns[{position}]",
                    f"column {column} is already assigned to group {assigned[column]!r}",
                    tensor_id=tensor_id,
                )
                valid = False
                continue
            assigned[int(column)] = group_id
            columns.append(int(column))
        groups.append(GroupSpec(group_id, tuple(sorted(set(columns)))))
    if not valid:
        return None
    uncovered = [column for column in range(column_count) if column not in assigned]
    if uncovered:
        groups.append(GroupSpec("ungrouped", tuple(uncovered)))
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _parse_tensor(
    tensor_raw: dict[str, Any],
    global_raw: dict[str, Any],
    *,
    index: int,
    singleton: bool,
    allowed_bit_widths: tuple[int, ...],
    seen_tensor_ids: set[str],
    failures: list[dict[str, Any]],
) -> TensorConfig | None:
    prefix = "tensor" if singleton else f"tensors[{index}]"
    tensor_id_raw = _get(tensor_raw, "tensor_id", "id", default=_MISSING)
    if tensor_id_raw is _MISSING:
        tensor_id = "default" if singleton else f"tensor_{index}"
        tensor_id_field = f"{prefix}.tensor_id"
        if not singleton:
            _add_error(
                failures,
                tensor_id_field,
                "tensor_id is required for a tensors[] entry",
            )
            return None
    elif not isinstance(tensor_id_raw, str) or not tensor_id_raw.strip():
        _add_error(
            failures,
            f"{prefix}.tensor_id",
            "tensor_id must be a non-empty string",
        )
        return None
    else:
        tensor_id = tensor_id_raw.strip()
    if tensor_id in seen_tensor_ids:
        _add_error(
            failures,
            f"{prefix}.tensor_id",
            f"duplicate tensor_id {tensor_id!r}",
            tensor_id=tensor_id,
        )
        return None

    error_start = len(failures)
    shape = _normalize_shape(
        _tensor_field(tensor_raw, global_raw, ("shape",), default=_MISSING),
        f"{prefix}.shape",
        failures,
        tensor_id,
    )
    column_axis_raw = _tensor_field(
        tensor_raw, global_raw, ("column_axis", "axis"), default=_MISSING
    )
    column_axis: int | None = None
    if not _is_int(column_axis_raw):
        _add_error(
            failures,
            f"{prefix}.column_axis",
            "column_axis must be an integer",
            tensor_id=tensor_id,
        )
    elif shape is not None and not (0 <= column_axis_raw < len(shape)):
        _add_error(
            failures,
            f"{prefix}.column_axis",
            f"column_axis must be in [0, {len(shape) - 1}]",
            tensor_id=tensor_id,
        )
    else:
        column_axis = int(column_axis_raw)

    lower_raw = _tensor_field(
        tensor_raw,
        global_raw,
        ("quantile_lower_permille", "lower_permille"),
        default=_MISSING,
    )
    upper_raw = _tensor_field(
        tensor_raw,
        global_raw,
        ("quantile_upper_permille", "upper_permille"),
        default=_MISSING,
    )
    lower_ok = _is_int(lower_raw) and 0 <= lower_raw <= 1000
    upper_ok = _is_int(upper_raw) and 0 <= upper_raw <= 1000
    if not lower_ok:
        _add_error(
            failures,
            f"{prefix}.quantile_lower_permille",
            "must be an integer in [0, 1000]",
            tensor_id=tensor_id,
        )
    if not upper_ok:
        _add_error(
            failures,
            f"{prefix}.quantile_upper_permille",
            "must be an integer in [0, 1000]",
            tensor_id=tensor_id,
        )
    if lower_ok and upper_ok and not (lower_raw < upper_raw):
        _add_error(
            failures,
            f"{prefix}.quantile_lower_permille",
            "quantile_lower_permille must be less than quantile_upper_permille",
            tensor_id=tensor_id,
        )

    scheme_raw = _tensor_field(
        tensor_raw, global_raw, ("scheme", "calibration_scheme"), default=_MISSING
    )
    scheme = (
        SCHEME_ALIASES.get(scheme_raw, scheme_raw)
        if isinstance(scheme_raw, str)
        else scheme_raw
    )
    if scheme not in SCHEMES:
        _add_error(
            failures,
            f"{prefix}.scheme",
            f"scheme must be one of {list(SCHEMES)}",
            tensor_id=tensor_id,
        )

    candidates_raw = _tensor_field(
        tensor_raw,
        global_raw,
        ("bit_width_candidates", "bit_widths", "quantization_bit_widths"),
        default=_MISSING,
    )
    candidates: tuple[int, ...] | None = None
    if not isinstance(candidates_raw, list) or not candidates_raw:
        _add_error(
            failures,
            f"{prefix}.bit_width_candidates",
            "bit_width_candidates must be a non-empty list",
            tensor_id=tensor_id,
        )
    else:
        candidate_values: list[int] = []
        valid_candidates = True
        for position, width in enumerate(candidates_raw):
            if not _is_int(width) or width not in allowed_bit_widths:
                _add_error(
                    failures,
                    f"{prefix}.bit_width_candidates[{position}]",
                    f"bit width must be one of {list(allowed_bit_widths)}",
                    tensor_id=tensor_id,
                )
                valid_candidates = False
                continue
            candidate_values.append(int(width))
        if valid_candidates and len(set(candidate_values)) != len(candidate_values):
            _add_error(
                failures,
                f"{prefix}.bit_width_candidates",
                "bit width candidates must be unique",
                tensor_id=tensor_id,
            )
            valid_candidates = False
        if valid_candidates:
            candidates = tuple(candidate_values)

    active_raw = _tensor_field(
        tensor_raw,
        global_raw,
        ("active_bit_width", "bit_width", "quantization_tier"),
        default=_MISSING,
    )
    active_bit_width: int | None = None
    if active_raw is _MISSING and candidates:
        active_bit_width = candidates[-1]
    elif not _is_int(active_raw) or active_raw not in allowed_bit_widths:
        _add_error(
            failures,
            f"{prefix}.active_bit_width",
            f"active_bit_width must be one of {list(allowed_bit_widths)}",
            tensor_id=tensor_id,
        )
    else:
        active_bit_width = int(active_raw)
        if candidates is not None and active_bit_width not in candidates:
            _add_error(
                failures,
                f"{prefix}.active_bit_width",
                "active_bit_width must also appear in bit_width_candidates",
                tensor_id=tensor_id,
            )
            active_bit_width = None

    fractional_bits = _validate_positive_int(
        _tensor_field(
            tensor_raw,
            global_raw,
            ("fractional_bits", "fixed_fractional_bits"),
            default=_MISSING,
        ),
        f"{prefix}.fractional_bits",
        failures,
        tensor_id=tensor_id,
        maximum=1000,
    )

    saturation_strategy = _tensor_field(
        tensor_raw,
        global_raw,
        ("saturation_strategy", "saturation"),
        default=_MISSING,
    )
    if isinstance(saturation_strategy, str):
        saturation_strategy = SATURATION_ALIASES.get(
            saturation_strategy, saturation_strategy
        )
    if saturation_strategy not in SATURATION_STRATEGIES:
        _add_error(
            failures,
            f"{prefix}.saturation_strategy",
            f"must be one of {list(SATURATION_STRATEGIES)}",
            tensor_id=tensor_id,
        )
        saturation_strategy = str(saturation_strategy)

    zero_scale_policy = _tensor_field(
        tensor_raw,
        global_raw,
        ("zero_scale_policy", "zero_scale_rule"),
        default="clamp_to_min_scale",
    )
    if isinstance(zero_scale_policy, str):
        zero_scale_policy = ZERO_SCALE_ALIASES.get(
            zero_scale_policy, zero_scale_policy
        )
    if zero_scale_policy not in ZERO_SCALE_POLICIES:
        _add_error(
            failures,
            f"{prefix}.zero_scale_policy",
            f"must be one of {list(ZERO_SCALE_POLICIES)}",
            tensor_id=tensor_id,
        )

    tolerance_raw = _tensor_field(
        tensor_raw, global_raw, ("tolerance", "relative_tolerance"), default=_MISSING
    )
    tolerance: float | None = None
    if not _is_finite_number(tolerance_raw) or float(tolerance_raw) <= 0:
        _add_error(
            failures,
            f"{prefix}.tolerance",
            "tolerance must be a positive finite number",
            tensor_id=tensor_id,
        )
    else:
        tolerance = float(tolerance_raw)

    group_limit_raw = _tensor_field(
        tensor_raw,
        global_raw,
        ("group_over_tolerance_limit", "group_tolerance_limit"),
        default=1.0,
    )
    group_limit: float | None = None
    if (
        not _is_finite_number(group_limit_raw)
        or not (0.0 <= float(group_limit_raw) <= 1.0)
    ):
        _add_error(
            failures,
            f"{prefix}.group_over_tolerance_limit",
            "must be a finite number in [0, 1]",
            tensor_id=tensor_id,
        )
    else:
        group_limit = float(group_limit_raw)

    groups: tuple[GroupSpec, ...] | None = None
    if shape is not None and column_axis is not None:
        groups_raw = _tensor_field(
            tensor_raw, global_raw, ("column_groups", "groups"), default=_MISSING
        )
        groups = _normalize_groups(
            groups_raw,
            field=f"{prefix}.column_groups",
            column_count=shape[column_axis],
            failures=failures,
            tensor_id=tensor_id,
        )

    if len(failures) != error_start:
        return None
    assert shape is not None
    assert column_axis is not None
    assert candidates is not None
    assert active_bit_width is not None
    assert fractional_bits is not None
    assert tolerance is not None
    assert group_limit is not None
    assert groups is not None
    seen_tensor_ids.add(tensor_id)
    return TensorConfig(
        tensor_id=tensor_id,
        shape=shape,
        column_axis=column_axis,
        quantile_lower_permille=int(lower_raw),
        quantile_upper_permille=int(upper_raw),
        scheme=str(scheme),
        bit_width_candidates=candidates,
        active_bit_width=active_bit_width,
        fractional_bits=fractional_bits,
        saturation_strategy=str(saturation_strategy),
        zero_scale_policy=str(zero_scale_policy),
        tolerance=tolerance,
        group_over_tolerance_limit=group_limit,
        groups=groups,
    )


def load_config(path: str | Path) -> ConfigLoadResult:
    """Load a configuration file and validate tensor entries independently."""

    failures: list[dict[str, Any]] = []
    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        failures.append(
            make_failure(
                "E_CONFIG_INVALID",
                "config_path",
                f"configuration file not found: {config_path}",
            )
        )
        return ConfigLoadResult(None, sort_failures(failures))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        failures.append(
            make_failure(
                "E_CONFIG_INVALID",
                "config_json",
                f"cannot read configuration JSON: {exc}",
            )
        )
        return ConfigLoadResult(None, sort_failures(failures))

    if not isinstance(raw, dict):
        failures.append(
            make_failure(
                "E_CONFIG_INVALID",
                "root",
                "configuration root must be a JSON object",
            )
        )
        return ConfigLoadResult(None, sort_failures(failures))

    interpolation = raw.get("interpolation", "linear")
    if interpolation not in INTERPOLATIONS:
        _add_error(
            failures,
            "interpolation",
            f"interpolation must be one of {list(INTERPOLATIONS)}",
        )
        interpolation = "linear"
    tie_break = raw.get("tie_break", "stable_ascending_first_occurrence")
    if tie_break not in TIE_BREAKS:
        _add_error(
            failures,
            "tie_break",
            f"tie_break must be one of {list(TIE_BREAKS)}",
        )
        tie_break = "stable_ascending_first_occurrence"
    scale_quantization = raw.get("scale_quantization", "round_half_away_from_zero")
    if scale_quantization not in SCALE_QUANTIZATIONS:
        _add_error(
            failures,
            "scale_quantization",
            f"scale_quantization must be one of {list(SCALE_QUANTIZATIONS)}",
        )
        scale_quantization = "round_half_away_from_zero"

    allowed_raw = raw.get("allowed_bit_widths", [4, 8, 12])
    allowed_bit_widths: tuple[int, ...] = (4, 8, 12)
    if (
        not isinstance(allowed_raw, list)
        or not allowed_raw
        or any(not _is_int(width) or width <= 0 for width in allowed_raw)
        or len(set(allowed_raw)) != len(allowed_raw)
    ):
        _add_error(
            failures,
            "allowed_bit_widths",
            "must be a non-empty list of unique positive integers",
        )
    else:
        allowed_bit_widths = tuple(int(width) for width in allowed_raw)

    max_workers_raw = raw.get("max_workers", 1)
    max_workers = _validate_positive_int(
        max_workers_raw, "max_workers", failures
    )
    if max_workers is None:
        max_workers = 1

    budget_raw = raw.get("calibration_element_budget", 1_000_000_000)
    budget = _validate_positive_int(
        budget_raw, "calibration_element_budget", failures
    )
    if budget is None:
        budget = 1_000_000_000

    interval_raw = raw.get("snapshot_commit_interval_batches", 1)
    interval = _validate_positive_int(
        interval_raw, "snapshot_commit_interval_batches", failures
    )
    if interval is None:
        interval = 1

    config_id_raw = raw.get("config_id", "column-calibration-workbench")
    if not isinstance(config_id_raw, str) or not config_id_raw.strip():
        _add_error(failures, "config_id", "config_id must be a non-empty string")
        config_id = "column-calibration-workbench"
    else:
        config_id = config_id_raw.strip()

    if "tensors" in raw:
        tensor_entries_raw = raw["tensors"]
        if not isinstance(tensor_entries_raw, list) or not tensor_entries_raw:
            _add_error(
                failures,
                "tensors",
                "tensors must be a non-empty list",
            )
            tensor_entries_raw = []
        singleton = False
    else:
        tensor_entries_raw = [raw]
        singleton = True

    tensors: list[TensorConfig] = []
    seen_tensor_ids: set[str] = set()
    for index, tensor_raw in enumerate(tensor_entries_raw):
        if not isinstance(tensor_raw, dict):
            _add_error(
                failures,
                f"tensors[{index}]",
                "tensor entry must be an object",
            )
            continue
        tensor = _parse_tensor(
            tensor_raw,
            raw,
            index=index,
            singleton=singleton,
            allowed_bit_widths=allowed_bit_widths,
            seen_tensor_ids=seen_tensor_ids,
            failures=failures,
        )
        if tensor is not None:
            tensors.append(tensor)

    tensors_sorted = tuple(sorted(tensors, key=lambda item: item.tensor_id))
    normalized = {
        "schema_version": "1.0",
        "config_id": config_id,
        "interpolation": str(interpolation),
        "tie_break": str(tie_break),
        "scale_quantization": str(scale_quantization),
        "allowed_bit_widths": list(allowed_bit_widths),
        "max_workers": int(max_workers),
        "calibration_element_budget": int(budget),
        "snapshot_commit_interval_batches": int(interval),
        "tensors": [tensor.to_dict() for tensor in tensors_sorted],
    }
    config_hash = sha256_json(
        {
            "normalized_config": normalized,
            "validation_failures": sort_failures(failures),
        }
    )
    config = WorkbenchConfig(
        config_id=config_id,
        interpolation=str(interpolation),
        tie_break=str(tie_break),
        scale_quantization=str(scale_quantization),
        allowed_bit_widths=allowed_bit_widths,
        max_workers=int(max_workers),
        calibration_element_budget=int(budget),
        snapshot_commit_interval_batches=int(interval),
        tensors=tensors_sorted,
        config_hash=config_hash,
    )
    return ConfigLoadResult(config, sort_failures(failures))
