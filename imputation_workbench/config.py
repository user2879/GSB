"""Configuration loading, normalisation, and validation."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import E_CONFIG_INVALID, WorkbenchError, diagnostic
from .fixedpoint import ROUNDING_MODES
from .util import canonical_json, is_int, sha256_obj


CONTENT_COLUMN_KEYS = (
    "type",
    "imputer",
    "min_valid_samples",
    "fallback",
    "group_key",
    "noise_coefficient",
    "constant",
    "domain",
    "decimal_places",
    "quantization_level",
)

NUMERIC_METHODS = {"median", "mean_noise", "grouped_median", "constant"}
CATEGORY_METHODS = {"mode", "constant"}
ALL_METHODS = NUMERIC_METHODS | CATEGORY_METHODS


def _int_value(value: Any) -> int | None:
    if is_int(value):
        return int(value)
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def _normalise_columns(raw_columns: Any, errors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    columns: dict[str, dict[str, Any]] = {}
    if isinstance(raw_columns, dict):
        items: Iterable[tuple[Any, Any]] = raw_columns.items()
    elif isinstance(raw_columns, list):
        converted: list[tuple[Any, Any]] = []
        for index, item in enumerate(raw_columns):
            if not isinstance(item, dict) or "name" not in item:
                errors.append(
                    diagnostic(
                        E_CONFIG_INVALID,
                        f"columns[{index}] must be an object with a name",
                        field=f"columns[{index}]",
                    )
                )
                continue
            converted.append((item.get("name"), item))
        items = converted
    else:
        errors.append(diagnostic(E_CONFIG_INVALID, "columns must be a mapping or list", field="columns"))
        return columns

    for raw_name, raw_spec in items:
        if not isinstance(raw_name, str) or not raw_name:
            errors.append(diagnostic(E_CONFIG_INVALID, "column name must be a non-empty string", field="columns"))
            continue
        if not isinstance(raw_spec, dict):
            errors.append(
                diagnostic(E_CONFIG_INVALID, "column specification must be an object", field=f"columns.{raw_name}")
            )
            continue
        spec = deepcopy(raw_spec)
        spec.pop("name", None)
        raw_type = spec.get("type", spec.get("dtype"))
        if raw_type == "categorical":
            raw_type = "category"
        spec["type"] = raw_type
        spec["imputer"] = spec.get("imputer", spec.get("method"))
        if isinstance(spec.get("domain"), list):
            spec["domain"] = list(spec["domain"])
        columns[raw_name] = spec
    return columns


def normalise_config(raw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not isinstance(raw, dict):
        raise WorkbenchError(E_CONFIG_INVALID, "configuration root must be a mapping")

    config = deepcopy(raw)
    # Accept both names used in the task description and the shorter YAML spelling.
    if "quantization_level" not in config and "quantization_levels" in config:
        config["quantization_level"] = config["quantization_levels"]
    if "quantization_level" not in config and "quantization_tier" in config:
        config["quantization_level"] = config["quantization_tier"]

    defaults: dict[str, Any] = {
        "format": 1,
        "seed": 0,
        "decimal_places": 2,
        "quantization_level": 1,
        "rounding": "half_even",
        "thread_pool_workers": 1,
        "batch_cell_budget": 1000,
        "snapshot_interval_batches": 1,
        "validation_fraction": 0.25,
        "model": {
            "learning_rate": 0.1,
            "epochs": 30,
            "batch_size": 4,
            "l2": 0.0,
        },
    }
    for key, value in defaults.items():
        if key not in config:
            config[key] = deepcopy(value)

    config["id_column"] = config.get("id_column", "row_id")
    config["target_column"] = config.get("target_column")
    config["columns"] = _normalise_columns(config.get("columns"), errors)

    for key in ("id_column", "target_column"):
        if not isinstance(config.get(key), str) or not config[key]:
            errors.append(diagnostic(E_CONFIG_INVALID, f"{key} must be a non-empty string", field=key))
    if not is_int(config.get("seed")) or config.get("seed") < 0:
        errors.append(diagnostic(E_CONFIG_INVALID, "seed must be a non-negative integer", field="seed"))

    decimal_places = _int_value(config.get("decimal_places"))
    if decimal_places is None or not (0 <= decimal_places <= 9):
        errors.append(
            diagnostic(E_CONFIG_INVALID, "decimal_places must be an integer from 0 through 9", field="decimal_places")
        )
    quantization = _int_value(config.get("quantization_level"))
    if quantization is None or not (1 <= quantization <= 1_000_000):
        errors.append(
            diagnostic(
                E_CONFIG_INVALID,
                "quantization_level must be an integer from 1 through 1000000",
                field="quantization_level",
            )
        )
    if config.get("rounding") not in ROUNDING_MODES:
        errors.append(
            diagnostic(
                E_CONFIG_INVALID,
                f"rounding must be one of {sorted(ROUNDING_MODES)}",
                field="rounding",
            )
        )
    workers = _int_value(config.get("thread_pool_workers"))
    if workers is None or not (1 <= workers <= 128):
        errors.append(
            diagnostic(
                E_CONFIG_INVALID,
                "thread_pool_workers must be a positive integer from 1 through 128",
                field="thread_pool_workers",
            )
        )
    budget = _int_value(config.get("batch_cell_budget"))
    if budget is None or budget <= 0:
        errors.append(
            diagnostic(E_CONFIG_INVALID, "batch_cell_budget must be a positive integer", field="batch_cell_budget")
        )
    interval = _int_value(config.get("snapshot_interval_batches"))
    if interval is None or interval <= 0:
        errors.append(
            diagnostic(
                E_CONFIG_INVALID,
                "snapshot_interval_batches must be a positive integer",
                field="snapshot_interval_batches",
            )
        )
    fraction = _as_number(config.get("validation_fraction"))
    if fraction is None or not (0.0 < fraction < 1.0):
        errors.append(
            diagnostic(
                E_CONFIG_INVALID,
                "validation_fraction must be a finite number strictly between 0 and 1",
                field="validation_fraction",
            )
        )

    model = config.get("model")
    if not isinstance(model, dict):
        errors.append(diagnostic(E_CONFIG_INVALID, "model must be an object", field="model"))
        model = {}
        config["model"] = model
    model_defaults = defaults["model"]
    for key, value in model_defaults.items():
        model.setdefault(key, value)
    for key in ("learning_rate", "l2"):
        number = _as_number(model.get(key))
        if number is None or number < 0:
            errors.append(
                diagnostic(E_CONFIG_INVALID, f"model.{key} must be a non-negative finite number", field=f"model.{key}")
            )
    epochs = _int_value(model.get("epochs"))
    if epochs is None or epochs <= 0:
        errors.append(diagnostic(E_CONFIG_INVALID, "model.epochs must be a positive integer", field="model.epochs"))
    batch_size = _int_value(model.get("batch_size"))
    if batch_size is None or batch_size <= 0:
        errors.append(
            diagnostic(E_CONFIG_INVALID, "model.batch_size must be a positive integer", field="model.batch_size")
        )

    for name, spec in config["columns"].items():
        prefix = f"columns.{name}"
        col_type = spec.get("type")
        if col_type not in {"numeric", "category"}:
            errors.append(diagnostic(E_CONFIG_INVALID, "column type must be numeric or category", field=f"{prefix}.type"))
            continue
        imputer = spec.get("imputer")
        if imputer not in ALL_METHODS:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    f"unsupported imputer {imputer!r}",
                    field=f"{prefix}.imputer",
                )
            )
            continue
        allowed = NUMERIC_METHODS if col_type == "numeric" else CATEGORY_METHODS
        if imputer not in allowed:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    f"imputer {imputer} is incompatible with {col_type} column",
                    field=f"{prefix}.imputer",
                )
            )
        min_samples = _int_value(spec.get("min_valid_samples"))
        if min_samples is None or min_samples <= 0:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    "min_valid_samples must be a positive integer",
                    field=f"{prefix}.min_valid_samples",
                )
            )
        elif min_samples > 1_000_000:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    "min_valid_samples exceeds the supported range (1..1000000)",
                    field=f"{prefix}.min_valid_samples",
                )
            )
        spec.setdefault("min_valid_samples", 1)
        places = _int_value(spec.get("decimal_places", config.get("decimal_places")))
        if places is None or not (0 <= places <= 9):
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    "column decimal_places must be an integer from 0 through 9",
                    field=f"{prefix}.decimal_places",
                )
            )
        spec["decimal_places"] = config.get("decimal_places") if places is None else places
        quantum = _int_value(spec.get("quantization_level", config.get("quantization_level")))
        if quantum is None or not (1 <= quantum <= 1_000_000):
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    "column quantization_level must be from 1 through 1000000",
                    field=f"{prefix}.quantization_level",
                )
            )
        spec["quantization_level"] = config.get("quantization_level") if quantum is None else quantum
        spec["rounding"] = config.get("rounding")

        if col_type == "category":
            domain = spec.get("domain")
            if not isinstance(domain, list) or not domain:
                errors.append(
                    diagnostic(E_CONFIG_INVALID, "category domain must be a non-empty list", field=f"{prefix}.domain")
                )
                spec["domain"] = []
            else:
                if any(not isinstance(value, str) or not value for value in domain):
                    errors.append(
                        diagnostic(
                            E_CONFIG_INVALID,
                            "category domain values must be non-empty strings",
                            field=f"{prefix}.domain",
                        )
                    )
                if len(set(domain)) != len(domain):
                    errors.append(
                        diagnostic(E_CONFIG_INVALID, "category domain values must be unique", field=f"{prefix}.domain")
                    )
                spec["domain"] = sorted(domain)
        if imputer == "constant":
            if col_type == "numeric":
                constant = _as_number(spec.get("constant"))
                if constant is None:
                    errors.append(
                        diagnostic(
                            E_CONFIG_INVALID,
                            "numeric constant imputer requires a finite constant",
                            field=f"{prefix}.constant",
                        )
                    )
            else:
                constant = spec.get("constant")
                if not isinstance(constant, str) or constant not in spec.get("domain", []):
                    errors.append(
                        diagnostic(
                            E_CONFIG_INVALID,
                            "category constant imputer requires a constant in the declared domain",
                            field=f"{prefix}.constant",
                        )
                    )
        if imputer == "grouped_median":
            group_key = spec.get("group_key")
            if not isinstance(group_key, str) or not group_key:
                errors.append(
                    diagnostic(
                        E_CONFIG_INVALID,
                        "grouped_median requires a group_key column name",
                        field=f"{prefix}.group_key",
                    )
                )
        if imputer == "mean_noise":
            coefficient = _as_number(spec.get("noise_coefficient"))
            if coefficient is None or coefficient < 0:
                errors.append(
                    diagnostic(
                        E_CONFIG_INVALID,
                        "mean_noise requires a non-negative finite noise_coefficient",
                        field=f"{prefix}.noise_coefficient",
                    )
                )
            spec.setdefault("noise_coefficient", 0.25)

        fallback = spec.get("fallback")
        if col_type == "numeric":
            parsed = _as_number(fallback)
            if parsed is None:
                errors.append(
                    diagnostic(
                        E_CONFIG_INVALID,
                        "numeric columns require a finite fallback value",
                        field=f"{prefix}.fallback",
                    )
                )
        else:
            if not isinstance(fallback, str) or fallback not in spec.get("domain", []):
                errors.append(
                    diagnostic(
                        E_CONFIG_INVALID,
                        "category columns require a fallback value in the declared domain",
                        field=f"{prefix}.fallback",
                    )
                )

    # Group-key compatibility can be checked without seeing data rows.
    for name, spec in config["columns"].items():
        if spec.get("imputer") != "grouped_median":
            continue
        key = spec.get("group_key")
        key_spec = config["columns"].get(key)
        if key is None:
            continue
        if key_spec is None:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    f"group key column {key!r} is not declared",
                    field=f"columns.{name}.group_key",
                )
            )
        elif key_spec.get("type") != "category":
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    f"group key column {key!r} must be categorical",
                    field=f"columns.{name}.group_key",
                )
            )

    target = config.get("target_column")
    if isinstance(target, str):
        target_spec = config["columns"].get(target)
        if target_spec is None:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    "target_column must be declared in columns",
                    field="target_column",
                )
            )
        elif target_spec.get("type") != "numeric":
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    "linear classification target must be a numeric column",
                    field="target_column",
                )
            )
    return config, errors


def load_config(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise WorkbenchError(E_CONFIG_INVALID, f"cannot load YAML configuration: {exc}") from exc
    config, errors = normalise_config(raw)
    return config, errors


def validate_config_against_schema(
    config: dict[str, Any], schema_columns: set[str]
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    configured = set(config.get("columns", {}))
    for name in sorted(configured):
        if name not in schema_columns:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    f"configured column {name!r} does not exist in the raw table",
                    field=f"columns.{name}",
                )
            )
    for key in ("id_column", "target_column"):
        value = config.get(key)
        if isinstance(value, str) and schema_columns and value not in schema_columns:
            errors.append(
                diagnostic(
                    E_CONFIG_INVALID,
                    f"{key} {value!r} does not exist in the raw table",
                    field=key,
                )
            )
    return errors


def config_hash(config: dict[str, Any]) -> str:
    return sha256_obj(config)


def executable_config_hash(config: dict[str, Any]) -> str:
    """Hash effective content while normalising execution-only knobs.

    Parallelism and snapshot cadence do not change the computed table.  Budget is
    also a scheduler knob and is excluded from the comparison hash.
    """

    material = deepcopy(config)
    for key in ("thread_pool_workers", "batch_cell_budget", "snapshot_interval_batches"):
        material.pop(key, None)
    return sha256_obj(material)


def config_debug(config: dict[str, Any]) -> str:
    return canonical_json(config)

