"""Configuration loading and strict validation."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from .errors import E_INPUT_INVALID, InputValidationError
from .fixedpoint import FixedParams, as_decimal
from .hashing import canonical_sha256


PENALTIES = {"ridge", "lasso"}
SPACING = {"manual", "linear", "geometric"}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_decimal_json(path: str | Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(
        text,
        parse_float=Decimal,
        parse_constant=_reject_json_constant,
    )


def _err(errors: list[dict[str, Any]], field: str, message: str) -> None:
    errors.append({"code": E_INPUT_INVALID, "field": field, "message": message})


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_alphas(raw_path: dict[str, Any], field_prefix: str, errors: list[dict[str, Any]]) -> list[Decimal]:
    alphas_raw = raw_path.get("alphas")
    if not isinstance(alphas_raw, list) or not alphas_raw:
        _err(errors, f"{field_prefix}.alphas", "must be a non-empty array")
        return []
    alphas: list[Decimal] = []
    for index, value in enumerate(alphas_raw):
        try:
            alpha = as_decimal(value, f"alphas[{index}]")
        except ValueError as exc:
            _err(errors, f"{field_prefix}.alphas[{index}]", str(exc))
            continue
        if alpha < 0:
            _err(errors, f"{field_prefix}.alphas[{index}]", "must be non-negative")
        alphas.append(alpha)

    if len(alphas) != len(alphas_raw):
        return []
    for index in range(1, len(alphas)):
        if alphas[index] <= alphas[index - 1]:
            _err(errors, f"{field_prefix}.alphas[{index}]", "regularization levels must be strictly increasing")

    min_alpha = Decimal("0")
    max_alpha = None
    if "min_alpha" in raw_path:
        try:
            min_alpha = as_decimal(raw_path["min_alpha"], "min_alpha")
        except ValueError as exc:
            _err(errors, f"{field_prefix}.min_alpha", str(exc))
    if "max_alpha" in raw_path:
        try:
            max_alpha = as_decimal(raw_path["max_alpha"], "max_alpha")
        except ValueError as exc:
            _err(errors, f"{field_prefix}.max_alpha", str(exc))
    else:
        _err(errors, f"{field_prefix}.max_alpha", "regularization upper bound is required")

    if min_alpha < 0:
        _err(errors, f"{field_prefix}.min_alpha", "must be non-negative")
    if max_alpha is not None:
        if max_alpha < min_alpha:
            _err(errors, f"{field_prefix}.max_alpha", "must be >= min_alpha")
        for index, alpha in enumerate(alphas):
            if not (min_alpha <= alpha <= max_alpha):
                _err(errors, f"{field_prefix}.alphas[{index}]", f"must lie in [{min_alpha}, {max_alpha}]")

    spacing = raw_path.get("spacing", "manual")
    if spacing not in SPACING:
        _err(errors, f"{field_prefix}.spacing", f"must be one of {sorted(SPACING)}")
    elif spacing == "linear" and "spacing_value" in raw_path:
        try:
            step = as_decimal(raw_path["spacing_value"], "spacing_value")
            if step <= 0:
                raise ValueError("must be positive")
            for i in range(1, len(alphas)):
                if alphas[i] - alphas[i - 1] != step:
                    _err(errors, f"{field_prefix}.alphas", "adjacent gaps must equal spacing_value for linear spacing")
                    break
        except ValueError as exc:
            _err(errors, f"{field_prefix}.spacing_value", str(exc))
    elif spacing == "geometric" and "spacing_value" in raw_path:
        try:
            ratio = as_decimal(raw_path["spacing_value"], "spacing_value")
            if ratio <= 1:
                raise ValueError("must be > 1")
            for i in range(1, len(alphas)):
                if alphas[i - 1] == 0 or alphas[i] / alphas[i - 1] != ratio:
                    _err(errors, f"{field_prefix}.alphas", "adjacent ratios must equal spacing_value for geometric spacing")
                    break
        except ValueError as exc:
            _err(errors, f"{field_prefix}.spacing_value", str(exc))
    return alphas


def validate_solver(raw: dict[str, Any], field_prefix: str, errors: list[dict[str, Any]]) -> dict[str, Any]:
    solver = {
        "coefficient_tolerance": Decimal("1e-8"),
        "max_iterations": 200,
        "pivot_abs_threshold": 1,
        "fallback_factor": Decimal("0.5"),
        "max_fallbacks": 4,
        "zero_half_step_policy": "zero",
        "convergence_boundary_policy": "converge",
        "attribution_tolerance": Decimal("1e-6"),
    }
    if not isinstance(raw, dict):
        _err(errors, field_prefix, "must be an object")
        return solver
    if "coefficient_tolerance" in raw:
        try:
            tol = as_decimal(raw["coefficient_tolerance"], "coefficient_tolerance")
            if tol <= 0:
                raise ValueError("must be positive")
            solver["coefficient_tolerance"] = tol
        except ValueError as exc:
            _err(errors, f"{field_prefix}.coefficient_tolerance", str(exc))
    for key, minimum in [("max_iterations", 1), ("max_fallbacks", 0)]:
        if key in raw:
            value = raw[key]
            if not _is_int(value) or value < minimum or value > 1_000_000:
                _err(errors, f"{field_prefix}.{key}", f"must be an integer in [{minimum}, 1000000]")
            else:
                solver[key] = value
    if "pivot_abs_threshold" in raw:
        value = raw["pivot_abs_threshold"]
        if not _is_int(value) or value < 0:
            _err(errors, f"{field_prefix}.pivot_abs_threshold", "must be a non-negative integer")
        else:
            solver["pivot_abs_threshold"] = value
    if "fallback_factor" in raw:
        try:
            factor = as_decimal(raw["fallback_factor"], "fallback_factor")
            if not (0 < factor < 1):
                raise ValueError("must be in (0, 1)")
            solver["fallback_factor"] = factor
        except ValueError as exc:
            _err(errors, f"{field_prefix}.fallback_factor", str(exc))
    for key, allowed in [
        ("zero_half_step_policy", {"zero", "nonzero"}),
        ("convergence_boundary_policy", {"converge", "continue"}),
    ]:
        if key in raw:
            value = raw[key]
            if value not in allowed:
                _err(errors, f"{field_prefix}.{key}", f"must be one of {sorted(allowed)}")
            else:
                solver[key] = value
    if "attribution_tolerance" in raw:
        try:
            tol = as_decimal(raw["attribution_tolerance"], "attribution_tolerance")
            if tol < 0:
                raise ValueError("must be non-negative")
            solver["attribution_tolerance"] = tol
        except ValueError as exc:
            _err(errors, f"{field_prefix}.attribution_tolerance", str(exc))
    return solver


def normalize_config(raw: Any) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        raise InputValidationError("configuration must be a JSON object", field="$", errors=[
            {"code": E_INPUT_INVALID, "field": "$", "message": "configuration must be a JSON object"}
        ])

    columns = raw.get("feature_columns")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) and c for c in columns):
        _err(errors, "feature_columns", "must be a non-empty array of non-empty strings")
        columns = []
    elif len(set(columns)) != len(columns):
        _err(errors, "feature_columns", "column names must be unique")

    fp_raw = raw.get("fixed_point", {})
    if not isinstance(fp_raw, dict):
        _err(errors, "fixed_point", "must be an object")
        fp_raw = {}
    accumulation_raw = raw.get("accumulation", {})
    if not isinstance(accumulation_raw, dict):
        _err(errors, "accumulation", "must be an object")
        accumulation_raw = {}
    decimal_places = fp_raw.get("decimal_places", 9)
    quant_level = fp_raw.get("quant_level", 1)
    margin = fp_raw.get("accum_rounding_margin", accumulation_raw.get("rounding_margin", 0))
    fixed_errors_before = len(errors)
    if not _is_int(decimal_places) or not (0 <= decimal_places <= 18):
        _err(errors, "fixed_point.decimal_places", "must be an integer in [0, 18]")
    if not _is_int(quant_level) or not (0 <= quant_level <= decimal_places if _is_int(decimal_places) else False):
        _err(errors, "fixed_point.quant_level", "must be an integer in [0, decimal_places]")
    if not _is_int(margin) or not (0 <= margin <= 3 * decimal_places if _is_int(decimal_places) else False):
        _err(errors, "fixed_point.accum_rounding_margin", "must be an integer in [0, 3*decimal_places]")
    try:
        fixed = FixedParams(decimal_places, quant_level, margin if margin is not None else 0)
    except ValueError as exc:
        if len(errors) == fixed_errors_before:
            _err(errors, "fixed_point", str(exc))
        fixed = FixedParams()

    corr_threshold = Decimal("0.90")
    collinear_threshold = Decimal("0.995")
    attribution_tolerance = Decimal("1e-6")
    for key, target in [
        ("correlation_threshold", "corr"),
        ("collinearity_threshold", "col"),
        ("attribution_tolerance", "attr"),
    ]:
        if key in raw:
            try:
                value = as_decimal(raw[key], key)
                if not (0 <= value <= 1) if key != "attribution_tolerance" else value < 0:
                    raise ValueError("invalid range")
                if target == "corr":
                    corr_threshold = value
                elif target == "col":
                    collinear_threshold = value
                else:
                    attribution_tolerance = value
            except ValueError:
                if key == "attribution_tolerance":
                    try:
                        attribution_tolerance = as_decimal(raw[key], key)
                        if attribution_tolerance < 0:
                            raise ValueError
                    except ValueError as exc:
                        _err(errors, key, str(exc))
                else:
                    _err(errors, key, "must be a decimal in [0, 1]")

    sched_raw = raw.get("scheduler", raw)
    if not isinstance(sched_raw, dict):
        sched_raw = raw
    worker_count = sched_raw.get("worker_count", raw.get("thread_pool_size", 1))
    mac_budget = sched_raw.get("mac_budget_per_batch", raw.get("mac_budget_per_batch", 10_000))
    column_quota = sched_raw.get("single_column_quota", raw.get("single_column_quota", 10_000))
    if not _is_int(worker_count) or not (1 <= worker_count <= 64):
        _err(errors, "scheduler.worker_count", "must be an integer in [1, 64]")
        worker_count = 1
    if not _is_int(mac_budget) or mac_budget <= 0:
        _err(errors, "scheduler.mac_budget_per_batch", "must be a positive integer")
        mac_budget = 10_000
    if not _is_int(column_quota) or column_quota <= 0:
        _err(errors, "scheduler.single_column_quota", "must be a positive integer")
        column_quota = 10_000

    accumulation = {
        "arithmetic": accumulation_raw.get("arithmetic", "ordered_python_integer"),
        "order": accumulation_raw.get("order", ["target_column_asc", "source_column_asc", "sample_id_asc"]),
        "rounding_margin": margin if _is_int(margin) else 0,
        "mac_point_definition": "one weighted scalar product contribution for one sample/element",
    }

    paths_raw = raw.get("paths")
    paths: dict[str, Any] = {}
    if paths_raw is not None:
        if not isinstance(paths_raw, dict) or not paths_raw:
            _err(errors, "paths", "must be a non-empty object when provided")
        else:
            for path_name, path_raw in paths_raw.items():
                prefix = f"paths.{path_name}"
                if not isinstance(path_raw, dict):
                    _err(errors, prefix, "must be an object")
                    continue
                penalty = path_raw.get("penalty")
                if penalty not in PENALTIES:
                    _err(errors, f"{prefix}.penalty", f"must be one of {sorted(PENALTIES)}")
                alphas = validate_alphas(path_raw, prefix, errors)
                solver = validate_solver(path_raw.get("solver", {}), f"{prefix}.solver", errors)
                paths[path_name] = {
                    "name": path_name,
                    "penalty": penalty if penalty in PENALTIES else "ridge",
                    "alphas": alphas,
                    "spacing": path_raw.get("spacing", "manual"),
                    "solver": solver,
                }
    else:
        penalty = raw.get("penalty")
        if penalty not in PENALTIES:
            _err(errors, "penalty", f"must be one of {sorted(PENALTIES)} or provide paths")
        alphas = validate_alphas(raw, "$", errors)
        solver = validate_solver(raw.get("solver", {}), "solver", errors)
        paths["default"] = {
            "name": "default",
            "penalty": penalty if penalty in PENALTIES else "ridge",
            "alphas": alphas,
            "spacing": raw.get("spacing", "manual"),
            "solver": solver,
        }

    if errors:
        raise InputValidationError("configuration validation failed", errors=errors)

    return {
        "schema_version": str(raw.get("schema_version", "1.0")),
        "columns": columns,
        "fixed": fixed,
        "accumulation": accumulation,
        "correlation_threshold": corr_threshold,
        "collinearity_threshold": collinear_threshold,
        "attribution_tolerance": attribution_tolerance,
        "scheduler": {
            "worker_count": worker_count,
            "mac_budget_per_batch": mac_budget,
            "single_column_quota": column_quota,
        },
        "paths": paths,
        "cache_dir": raw.get("cache_dir", "ledger/cache"),
        "ledger_path": raw.get("ledger_path", "ledger/reuse_ledger.json"),
        "lineage_path": raw.get("lineage_path", "ledger/lineage.json"),
        "snapshot_dir": raw.get("snapshot_dir", "snapshots"),
        "config_fingerprint": canonical_sha256({
            "columns": columns,
            "fixed": fixed.fingerprint_material(),
            "correlation_threshold": str(corr_threshold),
            "collinearity_threshold": str(collinear_threshold),
            "attribution_tolerance": str(attribution_tolerance),
            "paths": {
                name: {
                    "penalty": p["penalty"],
                    "alphas": [str(a) for a in p["alphas"]],
                    "solver": {k: (str(v) if isinstance(v, Decimal) else v) for k, v in p["solver"].items()},
                }
                for name, p in paths.items()
            },
        }),
    }


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        raw = load_decimal_json(path)
    except FileNotFoundError:
        raise InputValidationError(f"config not found: {path}", field="config")
    except (json.JSONDecodeError, ValueError) as exc:
        raise InputValidationError(f"config JSON invalid: {exc}", field="config")
    return normalize_config(raw)
