from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import (
    E_COLUMN_DESCRIPTION_MISSING,
    E_COLUMN_UNKNOWN,
    E_CONFIG_INVALID,
    E_DATA_INVALID,
    E_INJECTION_INVALID,
    E_NONFINITE_FEATURE,
    E_TYPE_MISMATCH,
    Issue,
)
from .util import as_float, as_int, deep_copy, hash_array, read_yaml_mapping, stable_hash


@dataclass
class Table:
    rows: int
    values: dict[str, list[Any]]
    column_specs: dict[str, dict[str, Any]]
    declared_columns: set[str]
    column_content_fingerprints: dict[str, str]
    column_description_fingerprints: dict[str, str]
    issues: list[Issue] = field(default_factory=list)
    injection_report: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available_columns(self) -> set[str]:
        return set(self.values)

    @property
    def content_fingerprint(self) -> str:
        return stable_hash(self.column_content_fingerprints)

    def as_object_row(self, index: int, columns: list[str] | None = None) -> dict[str, Any]:
        chosen = columns if columns is not None else list(self.values)
        return {column: self.values[column][index] for column in chosen}


@dataclass(frozen=True)
class MaterializedColumn:
    name: str
    kind: str
    values: tuple[Any, ...]
    spec: dict[str, Any]
    content_fingerprint: str
    description_fingerprint: str


def load_data_description(path: str | Path) -> tuple[dict[str, Any], list[Issue]]:
    raw, error = read_yaml_mapping(path)
    if error is not None or raw is None:
        return {}, [
            Issue(
                E_DATA_INVALID,
                error or "data description must be a YAML mapping",
                context={"path": str(path), "scope": "data"},
            )
        ]
    return normalize_data_description(raw)


def normalize_data_description(raw: dict[str, Any]) -> tuple[dict[str, Any], list[Issue]]:
    issues: list[Issue] = []
    if not isinstance(raw, dict):
        return {}, [Issue(E_DATA_INVALID, "data description must be a mapping")]
    result = deep_copy(raw)
    version = as_int(result.get("schema_version", 1))
    if version != 1:
        issues.append(
            Issue(E_DATA_INVALID, "data schema_version must be 1", context={"scope": "data"})
        )
    result["schema_version"] = 1
    seed = as_int(result.get("seed", 1))
    if seed is None:
        issues.append(Issue(E_DATA_INVALID, "data seed must be an integer", context={"scope": "data"}))
        seed = 1
    result["seed"] = seed

    size_levels = result.get("size_levels", {})
    if size_levels is None:
        size_levels = {}
    if not isinstance(size_levels, dict):
        issues.append(
            Issue(E_TYPE_MISMATCH, "size_levels must be a mapping", context={"scope": "data"})
        )
        size_levels = {}
    normalized_levels: dict[str, int] = {}
    for name, value in size_levels.items():
        parsed = as_int(value)
        if parsed is None or parsed <= 0:
            issues.append(
                Issue(
                    E_DATA_INVALID,
                    f"size level {name!r} must be positive",
                    context={"scope": "data", "size_level": str(name)},
                )
            )
        else:
            normalized_levels[str(name)] = parsed
    result["size_levels"] = normalized_levels

    rows = as_int(result.get("rows"))
    level = str(result.get("size_level", "")) if result.get("size_level") is not None else ""
    if rows is None:
        rows = normalized_levels.get(level)
    if rows is None or rows <= 0:
        issues.append(
            Issue(
                E_DATA_INVALID,
                "data description must provide positive rows or a valid size_level",
                context={"scope": "data", "size_level": level},
            )
        )
        rows = normalized_levels.get(level, 1)
    result["rows"] = max(1, rows)
    result["size_level"] = level

    columns = result.get("columns")
    if not isinstance(columns, list) or not columns:
        issues.append(
            Issue(E_DATA_INVALID, "columns must be a non-empty list", context={"scope": "data"})
        )
        columns = []
    result["columns"] = deep_copy(columns)
    return result, issues


def _rng(*parts: Any) -> np.random.Generator:
    digest = stable_hash(list(parts))
    entropy = int(digest[:32], 16)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _numeric_values(
    spec: dict[str, Any], rows: int, base_values: dict[str, np.ndarray], issues: list[Issue]
) -> np.ndarray:
    name = str(spec.get("name", "<unnamed>"))
    source = spec.get("source")
    if source is not None:
        if source not in base_values:
            issues.append(
                Issue(
                    E_COLUMN_UNKNOWN,
                    f"numeric column {name!r} references unknown source {source!r}",
                    context={"scope": "data", "column": name, "source": source},
                )
            )
            return np.zeros(rows, dtype=np.float64)
        values = np.asarray(base_values[str(source)], dtype=np.float64).copy()
        multiplier = as_float(spec.get("multiplier", 1.0))
        offset = as_float(spec.get("offset", 0.0))
        if multiplier is None or offset is None:
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    f"column {name!r} source multiplier/offset must be finite numbers",
                    context={"scope": "data", "column": name},
                )
            )
            multiplier = 1.0
            offset = 0.0
        return values * multiplier + offset

    distribution = spec.get("distribution", "normal")
    generator = _rng("numeric", spec.get("seed", 0), name, distribution)
    if distribution == "normal":
        mean = as_float(spec.get("mean", 0.0))
        std = as_float(spec.get("std", 1.0))
        if mean is None or std is None or std < 0:
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    f"column {name!r} normal mean/std must be finite and std non-negative",
                    context={"scope": "data", "column": name},
                )
            )
            mean, std = 0.0, 1.0
        return mean + std * generator.standard_normal(rows)
    if distribution == "uniform":
        low = as_float(spec.get("low", 0.0))
        high = as_float(spec.get("high", 1.0))
        if low is None or high is None or high < low:
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    f"column {name!r} uniform bounds are invalid",
                    context={"scope": "data", "column": name},
                )
            )
            low, high = 0.0, 1.0
        return low + (high - low) * generator.random(rows)
    if distribution in {"integer", "integers"}:
        low = as_int(spec.get("low", 0)) or 0
        high = as_int(spec.get("high", 10))
        if high is None or high <= low:
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    f"column {name!r} integer bounds are invalid",
                    context={"scope": "data", "column": name},
                )
            )
            low, high = 0, 1
        return generator.integers(low, high, size=rows, endpoint=False).astype(np.float64)
    if distribution == "mixture":
        components = spec.get("components")
        if not isinstance(components, list) or not components:
            issues.append(
                Issue(
                    E_DATA_INVALID,
                    f"column {name!r} mixture requires components",
                    context={"scope": "data", "column": name},
                )
            )
            return generator.standard_normal(rows)
        weights: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        for component in components:
            if not isinstance(component, dict):
                continue
            weight = as_float(component.get("weight", 1.0))
            mean = as_float(component.get("mean", 0.0))
            std = as_float(component.get("std", 1.0))
            if weight is None or weight < 0 or mean is None or std is None or std < 0:
                issues.append(
                    Issue(
                        E_DATA_INVALID,
                        f"column {name!r} has an invalid mixture component",
                        context={"scope": "data", "column": name},
                    )
                )
                continue
            weights.append(weight)
            means.append(mean)
            stds.append(std)
        if not weights or sum(weights) <= 0:
            issues.append(
                Issue(
                    E_DATA_INVALID,
                    f"column {name!r} mixture weights must have positive sum",
                    context={"scope": "data", "column": name},
                )
            )
            return generator.standard_normal(rows)
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        choices = generator.choice(len(weights), size=rows, p=probabilities)
        noise = generator.standard_normal(rows)
        mean_array = np.asarray(means, dtype=np.float64)
        std_array = np.asarray(stds, dtype=np.float64)
        return mean_array[choices] + std_array[choices] * noise
    issues.append(
        Issue(
            E_TYPE_MISMATCH,
            f"column {name!r} has unsupported distribution {distribution!r}",
            context={"scope": "data", "column": name},
        )
    )
    return generator.standard_normal(rows)


def _categorical_values(
    spec: dict[str, Any], rows: int, issues: list[Issue]
) -> list[Any]:
    name = str(spec.get("name", "<unnamed>"))
    categories = spec.get("categories")
    if not isinstance(categories, list) or not categories or not all(
        isinstance(item, (str, int, float, bool)) for item in categories
    ):
        issues.append(
            Issue(
                E_DATA_INVALID,
                f"categorical column {name!r} requires scalar categories",
                context={"scope": "data", "column": name},
            )
        )
        categories = ["__invalid__"]
    categories = list(dict.fromkeys(str(item) for item in categories))
    generator = _rng("categorical", spec.get("seed", 0), name, categories)
    weights = spec.get("weights")
    if weights is None:
        probabilities = np.full(len(categories), 1.0 / len(categories), dtype=np.float64)
    else:
        if not isinstance(weights, list) or len(weights) != len(categories):
            issues.append(
                Issue(
                    E_DATA_INVALID,
                    f"categorical column {name!r} weights must match categories",
                    context={"scope": "data", "column": name},
                )
            )
            weights = None
        parsed_weights = [as_float(item) for item in weights] if weights is not None else []
        if any(item is None or item < 0 for item in parsed_weights) or not any(parsed_weights):
            issues.append(
                Issue(
                    E_DATA_INVALID,
                    f"categorical column {name!r} weights must be finite, non-negative, and not all zero",
                    context={"scope": "data", "column": name},
                )
            )
            probabilities = np.full(len(categories), 1.0 / len(categories), dtype=np.float64)
        else:
            probabilities = np.asarray(parsed_weights, dtype=np.float64)
            probabilities /= probabilities.sum()
    indexes = generator.choice(len(categories), size=rows, p=probabilities)
    return [categories[int(index)] for index in indexes]


def _missing_mask(spec: dict[str, Any], rows: int, issues: list[Issue]) -> np.ndarray:
    name = str(spec.get("name", "<unnamed>"))
    missing_rate = as_float(spec.get("missing_rate", 0.0))
    if missing_rate is None or not (0.0 <= missing_rate <= 1.0):
        issues.append(
            Issue(
                E_DATA_INVALID,
                f"column {name!r} missing_rate must be in [0, 1]",
                context={"scope": "data", "column": name},
            )
        )
        missing_rate = 0.0
    generator = _rng("missing", spec.get("seed", 0), name)
    return generator.random(rows) < missing_rate


def generate_table(
    description: dict[str, Any],
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
    injections: list[dict[str, Any]] | None = None,
) -> Table:
    issues: list[Issue] = []
    overrides = overrides or {}
    injections = injections or []
    rows = int(description.get("rows", 1))
    raw_columns = description.get("columns", [])
    base_values: dict[str, np.ndarray] = {}
    values: dict[str, list[Any]] = {}
    specs: dict[str, dict[str, Any]] = {}
    declared: set[str] = set()
    removed: set[str] = set()
    injection_report: list[dict[str, Any]] = []

    for injection in injections:
        if not isinstance(injection, dict):
            issues.append(
                Issue(E_INJECTION_INVALID, "injection entry must be a mapping", context={"scope": "injection"})
            )
            continue
        kind = injection.get("type")
        if kind == "missing_column_description":
            column = injection.get("column")
            if isinstance(column, str):
                removed.add(column)
            else:
                issues.append(
                    Issue(
                        E_INJECTION_INVALID,
                        "missing_column_description requires column",
                        context={"scope": "injection", "injection": injection.get("id")},
                    )
                )
            injection_report.append(
                {"id": injection.get("id"), "type": kind, "column": column, "isolated": True}
            )
        elif kind not in {"nonfinite_feature", "invalid_hyperparameter", "slow_candidate"}:
            issues.append(
                Issue(
                    E_INJECTION_INVALID,
                    f"unsupported injection type {kind!r}",
                    context={"scope": "injection", "injection": injection.get("id")},
                )
            )

    names_seen: set[str] = set()
    for raw_spec in raw_columns:
        if not isinstance(raw_spec, dict):
            issues.append(
                Issue(E_DATA_INVALID, "column description must be a mapping", context={"scope": "data"})
            )
            continue
        spec = deep_copy(raw_spec)
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            issues.append(
                Issue(E_DATA_INVALID, "column name must be a non-empty string", context={"scope": "data"})
            )
            continue
        if name in names_seen:
            issues.append(
                Issue(
                    E_DATA_INVALID,
                    f"duplicate column name {name!r}",
                    context={"scope": "data", "column": name},
                )
            )
            continue
        names_seen.add(name)
        declared.add(name)
        if name in removed:
            continue
        kind = spec.get("type")
        if kind not in {"numeric", "categorical"}:
            issues.append(
                Issue(
                    E_TYPE_MISMATCH,
                    f"column {name!r} has unsupported type {kind!r}",
                    context={"scope": "data", "column": name},
                )
            )
            continue
        if kind == "numeric":
            array = _numeric_values(spec, rows, base_values, issues)
            base_values[name] = array
            mask = _missing_mask(spec, rows, issues)
            column_values: list[Any] = [None if bool(missing) else float(array[index]) for index, missing in enumerate(mask)]
        else:
            column_values = _categorical_values(spec, rows, issues)
            mask = _missing_mask(spec, rows, issues)
            column_values = [None if bool(missing) else value for value, missing in zip(column_values, mask)]

        override = overrides.get(name)
        if isinstance(override, dict):
            column_values = _apply_column_override(column_values, kind, override, issues)
            spec["override"] = deep_copy(override)
        if kind == "numeric":
            # Preserve base-value dependency semantics for redundant columns: a
            # replacement of a source column flows into later derived columns,
            # while each derived column keeps its own missing mask.
            base_values[name] = np.asarray(
                [
                    float(value) if value is not None else float(base_values[name][index])
                    for index, value in enumerate(column_values)
                ],
                dtype=np.float64,
            )
        specs[name] = spec
        values[name] = column_values

    for column in overrides:
        if column not in declared:
            issues.append(
                Issue(
                    E_COLUMN_UNKNOWN,
                    f"override references unknown column {column!r}",
                    context={"scope": "data", "column": column},
                )
            )

    # Injections are applied after deterministic generation and are included in
    # the content fingerprint, so they invalidate exactly their dependent paths.
    for injection in injections:
        if not isinstance(injection, dict) or injection.get("type") != "nonfinite_feature":
            continue
        column = injection.get("column")
        row = as_int(injection.get("row", 0))
        value = injection.get("value", "inf")
        if column not in values or row is None or not (0 <= row < rows):
            issues.append(
                Issue(
                    E_INJECTION_INVALID,
                    "nonfinite_feature has an unknown column or out-of-range row",
                    context={"scope": "injection", "injection": injection.get("id"), "column": column},
                )
            )
            injection_report.append(
                {"id": injection.get("id"), "type": "nonfinite_feature", "column": column, "isolated": False}
            )
            continue
        if isinstance(value, str):
            lowered = value.lower()
            injected = float("inf") if lowered in {"inf", "+inf", "infinity", "+infinity"} else (
                float("-inf") if lowered in {"-inf", "-infinity"} else float("nan")
            )
        else:
            injected = as_float(value)
            if injected is None:
                injected = float("nan")
        values[column][int(row)] = injected
        injection_report.append(
            {
                "id": injection.get("id"),
                "type": "nonfinite_feature",
                "column": column,
                "row": int(row),
                "isolated": True,
            }
        )

    content_fingerprints: dict[str, str] = {}
    description_fingerprints: dict[str, str] = {}
    for column, column_values in values.items():
        content_fingerprints[column] = stable_hash({"type": specs[column].get("type"), "values": column_values})
        description = {key: value for key, value in specs[column].items() if key != "override"}
        description_fingerprints[column] = stable_hash(description)

    for name, spec in specs.items():
        if spec.get("type") == "numeric":
            for index, item in enumerate(values[name]):
                if item is not None and not math.isfinite(float(item)):
                    issues.append(
                        Issue(
                            E_NONFINITE_FEATURE,
                            f"non-finite feature in column {name!r}",
                            context={"scope": "data", "column": name, "row": index, "value": item},
                            severity="warning",
                        )
                    )
                    break

    return Table(
        rows=rows,
        values=values,
        column_specs=specs,
        declared_columns=declared,
        column_content_fingerprints=content_fingerprints,
        column_description_fingerprints=description_fingerprints,
        issues=issues,
        injection_report=injection_report,
    )


def _apply_column_override(
    column_values: list[Any], kind: str, override: dict[str, Any], issues: list[Issue]
) -> list[Any]:
    mode = override.get("kind", "constant")
    result = list(column_values)
    if mode == "constant":
        value = override.get("value")
        if kind == "numeric":
            parsed = as_float(value)
            if parsed is None:
                issues.append(Issue(E_TYPE_MISMATCH, "numeric constant override must be finite"))
                return result
            return [None if item is None else parsed for item in result]
        return [None if item is None else str(value) for item in result]
    if mode == "offset" and kind == "numeric":
        offset = as_float(override.get("value"))
        if offset is None:
            issues.append(Issue(E_TYPE_MISMATCH, "numeric offset override must be finite"))
            return result
        return [None if item is None else float(item) + offset for item in result]
    if mode == "multiply" and kind == "numeric":
        multiplier = as_float(override.get("value"))
        if multiplier is None:
            issues.append(Issue(E_TYPE_MISMATCH, "numeric multiply override must be finite"))
            return result
        return [None if item is None else float(item) * multiplier for item in result]
    if mode == "values":
        supplied = override.get("values")
        if not isinstance(supplied, list) or len(supplied) != len(result):
            issues.append(Issue(E_TYPE_MISMATCH, "replacement values length does not match rows"))
            return result
        if kind == "numeric":
            parsed_values: list[Any] = []
            for value in supplied:
                if value is None:
                    parsed_values.append(None)
                    continue
                parsed = as_float(value)
                if parsed is None:
                    issues.append(
                        Issue(
                            E_TYPE_MISMATCH,
                            "numeric replacement values must be finite numbers or null",
                        )
                    )
                    return result
                parsed_values.append(parsed)
            return parsed_values
        return [None if value is None else str(value) for value in supplied]
    if mode == "permute":
        generator = _rng("override-permute", override.get("seed", 0))
        permutation = generator.permutation(len(result))
        return [result[int(index)] for index in permutation]
    issues.append(Issue(E_CONFIG_INVALID, f"unsupported column override mode {mode!r}"))
    return result


def encode_categorical(values: list[Any], categories: list[Any]) -> tuple[np.ndarray, int]:
    categories = [str(item) for item in categories]
    mapping = {category: float(index) for index, category in enumerate(categories)}
    encoded = np.empty(len(values), dtype=np.float64)
    unknown = 0
    for index, value in enumerate(values):
        if value is None:
            encoded[index] = np.nan
        elif str(value) in mapping:
            encoded[index] = mapping[str(value)]
        else:
            encoded[index] = np.nan
            unknown += 1
    return encoded, unknown


def impute_numeric(values: np.ndarray, fill_value: float) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result[np.isnan(result)] = float(fill_value)
    return result


def build_feature_matrix(
    table: Table,
    columns: list[str],
    *,
    fill_values: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """Encode columns and optionally impute with supplied per-column statistics.

    Missing numeric values intentionally remain NaN when ``fill_values`` is
    omitted.  The preprocessing stage always supplies candidate-training-only
    fill statistics before fitting transformations.
    """

    fill_values = fill_values or {}
    unknown_counts: dict[str, int] = {}
    arrays: list[np.ndarray] = []
    for column in columns:
        spec = table.column_specs[column]
        if spec.get("type") == "numeric":
            array = np.asarray(
                [np.nan if item is None else float(item) for item in table.values[column]],
                dtype=np.float64,
            )
        else:
            array, unknown = encode_categorical(table.values[column], spec.get("categories", []))
            if unknown:
                unknown_counts[column] = unknown
        if column in fill_values:
            array = impute_numeric(array, fill_values[column])
        arrays.append(array)
    if arrays:
        matrix = np.column_stack(arrays).astype(np.float64, copy=False)
    else:
        matrix = np.empty((table.rows, 0), dtype=np.float64)
    return matrix, list(columns), unknown_counts
