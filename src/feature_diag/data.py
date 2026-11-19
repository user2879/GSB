"""Deterministic synthetic table generation from a local column description."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ScenarioConfig
from .data_spec import CATEGORY_KINDS, NUMERIC_KINDS, ColumnSpec, DataSpec
from .fingerprint import array_fingerprint, fingerprint, stable_normal, stable_uniform


@dataclass
class GeneratedTable:
    size: int
    seed: int
    spec: DataSpec
    sample_ids: np.ndarray
    columns: dict[str, np.ndarray]
    encoded_columns: dict[str, np.ndarray]
    target: np.ndarray
    missing_counts: dict[str, int]
    category_maps: dict[str, dict[str, int]]
    fingerprint: str

    @property
    def target_name(self) -> str:
        return self.spec.target_column or "target"


def _missing(seed: int, spec: DataSpec, column: str, index: int, rate: float) -> bool:
    if rate <= 0.0:
        return False
    return stable_uniform(seed, spec.name, "missing", column, index) < min(1.0, rate)


def _round(value: float, decimals: Any) -> float:
    if decimals is None:
        return value
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        return value
    return round(value, decimals)


def _generate_numeric(column: ColumnSpec, size: int, seed: int, spec: DataSpec) -> np.ndarray:
    params = column.params
    missing_rate = float(params.get("missing_rate", 0.0))
    values = np.empty(size, dtype=np.float64)
    decimals = params.get("decimals")
    if column.kind == "int":
        minimum = int(params.get("min", 0))
        maximum = int(params.get("max", max(minimum, minimum + 100)))
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        span = maximum - minimum + 1
        for index in range(size):
            if _missing(seed, spec, column.name, index, missing_rate):
                values[index] = np.nan
                continue
            unit = stable_uniform(seed, spec.name, "value", column.name, index)
            values[index] = float(minimum + min(span - 1, int(unit * span)))
        return values

    distribution = params.get("distribution", "normal")
    mean = float(params.get("mean", 0.0))
    std = float(params.get("std", 1.0))
    minimum = params.get("min")
    maximum = params.get("max")
    for index in range(size):
        if _missing(seed, spec, column.name, index, missing_rate):
            values[index] = np.nan
            continue
        if distribution == "uniform":
            low = float(params.get("low", mean - std))
            high = float(params.get("high", mean + std))
            unit = stable_uniform(seed, spec.name, "value", column.name, index)
            value = low + (high - low) * unit
        else:
            value = mean + std * stable_normal(seed, spec.name, "value", column.name, index)
        if minimum is not None:
            value = max(float(minimum), value)
        if maximum is not None:
            value = min(float(maximum), value)
        values[index] = _round(value, decimals)
    return values


def _category_values(column: ColumnSpec) -> list[str]:
    params = column.params
    if column.kind == "category_high":
        cardinality = params.get("cardinality", 0)
        if isinstance(cardinality, bool) or not isinstance(cardinality, int) or cardinality <= 0:
            raise ValueError(f"category column {column.name} requires positive cardinality")
        prefix = str(params.get("prefix", column.name))
        return [f"{prefix}-{index:04d}" for index in range(cardinality)]
    raw_values = params.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"category column {column.name} requires non-empty values")
    return [str(value) for value in raw_values]


def _category_weights(column: ColumnSpec) -> list[float] | None:
    raw_weights = column.params.get("weights")
    if raw_weights is None:
        return None
    if not isinstance(raw_weights, list) or any(
        not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight < 0 for weight in raw_weights
    ):
        raise ValueError(f"category column {column.name} has invalid weights")
    weights = [float(weight) for weight in raw_weights]
    if len(weights) != len(_category_values(column)) or sum(weights) <= 0:
        raise ValueError(f"category column {column.name} weights do not match values")
    return weights


def _generate_category(column: ColumnSpec, size: int, seed: int, spec: DataSpec) -> np.ndarray:
    values = _category_values(column)
    weights = _category_weights(column)
    missing_rate = float(column.params.get("missing_rate", 0.0))
    output = np.empty(size, dtype=object)
    if weights is None:
        for index in range(size):
            if _missing(seed, spec, column.name, index, missing_rate):
                output[index] = None
                continue
            unit = stable_uniform(seed, spec.name, "value", column.name, index)
            output[index] = values[min(len(values) - 1, int(unit * len(values)))]
        return output

    total = sum(weights)
    thresholds: list[float] = []
    running = 0.0
    for weight in weights:
        running += weight
        thresholds.append(running / total)
    for index in range(size):
        if _missing(seed, spec, column.name, index, missing_rate):
            output[index] = None
            continue
        unit = stable_uniform(seed, spec.name, "value", column.name, index)
        position = 0
        while position < len(thresholds) - 1 and unit > thresholds[position]:
            position += 1
        output[index] = values[position]
    return output


def _numeric_fill(values: np.ndarray) -> tuple[np.ndarray, int]:
    missing_mask = ~np.isfinite(values)
    missing_count = int(np.count_nonzero(missing_mask))
    if missing_count == 0:
        return values.astype(np.float64, copy=True), 0
    observed = values[~missing_mask]
    fill = float(np.median(observed)) if observed.size else 0.0
    filled = values.astype(np.float64, copy=True)
    filled[missing_mask] = fill
    return filled, missing_count


def _category_fill(values: np.ndarray) -> tuple[np.ndarray, int]:
    missing_mask = np.asarray([value is None for value in values], dtype=bool)
    missing_count = int(np.count_nonzero(missing_mask))
    if missing_count == 0:
        return np.asarray([str(value) for value in values], dtype=object), 0
    observed = [str(value) for value in values[~missing_mask]]
    if observed:
        counts: dict[str, int] = {}
        for value in observed:
            counts[value] = counts.get(value, 0) + 1
        fill = sorted(counts, key=lambda item: (-counts[item], item))[0]
    else:
        fill = "<missing>"
    filled = np.asarray([fill if value is None else str(value) for value in values], dtype=object)
    return filled, missing_count


def _fit_category_map(values: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    categories = sorted({str(value) for value in values})
    mapping = {value: index for index, value in enumerate(categories)}
    encoded = np.asarray([mapping[str(value)] for value in values], dtype=np.int64)
    return mapping, encoded


def _target_value(
    column: ColumnSpec,
    index: int,
    seed: int,
    spec: DataSpec,
    columns: dict[str, np.ndarray],
    encoded: dict[str, np.ndarray],
) -> float:
    params = column.params
    formula = params.get("formula", [])
    value = float(params.get("intercept", 0.0))
    if isinstance(formula, list):
        for term in formula:
            if not isinstance(term, dict):
                continue
            source = term.get("column")
            weight = float(term.get("weight", 0.0))
            if source in encoded:
                source_value = float(encoded[source][index])
            elif source in columns:
                source_value = float(columns[source][index])
            else:
                source_value = 0.0
            transform = term.get("transform", "identity")
            if transform == "square":
                source_value *= source_value
            elif transform == "log1p":
                source_value = math.log1p(max(-0.999999, source_value))
            value += weight * source_value
    value += float(params.get("noise_std", 0.0)) * stable_normal(seed, spec.name, "target", column.name, index)
    return _round(value, params.get("decimals"))


def generate_table(spec: DataSpec, scenario: ScenarioConfig, size: int) -> GeneratedTable:
    seed = int(scenario.seed)
    sample_ids: np.ndarray | None = None
    columns: dict[str, np.ndarray] = {}
    missing_counts: dict[str, int] = {}

    for column in spec.columns:
        if column.kind == "id":
            prefix = str(column.params.get("prefix", "row"))
            sample_ids = np.asarray([f"{prefix}-{index:06d}" for index in range(size)], dtype=object)
            continue
        if column.kind in NUMERIC_KINDS:
            raw = _generate_numeric(column, size, seed, spec)
            columns[column.name], missing_counts[column.name] = _numeric_fill(raw)
            continue
        if column.kind in CATEGORY_KINDS:
            raw = _generate_category(column, size, seed, spec)
            columns[column.name], missing_counts[column.name] = _category_fill(raw)
            continue
        if column.kind in {"redundant", "target"}:
            continue

    if sample_ids is None:
        sample_ids = np.asarray([f"row-{index:06d}" for index in range(size)], dtype=object)

    for column in spec.columns:
        if column.kind != "redundant":
            continue
        source = str(column.params["source"])
        base = columns[source]
        noise_std = float(column.params.get("noise_std", 0.0))
        missing_rate = float(column.params.get("missing_rate", 0.0))
        values = np.empty(size, dtype=np.float64)
        for index in range(size):
            if _missing(seed, spec, column.name, index, missing_rate):
                values[index] = np.nan
                continue
            noise = noise_std * stable_normal(seed, spec.name, "redundant", column.name, index)
            values[index] = _round(float(base[index]) + noise, column.params.get("decimals"))
        columns[column.name], missing_counts[column.name] = _numeric_fill(values)

    category_maps: dict[str, dict[str, int]] = {}
    encoded_columns: dict[str, np.ndarray] = {}
    for column in spec.columns:
        if column.kind in CATEGORY_KINDS:
            mapping, encoded = _fit_category_map(columns[column.name])
            category_maps[column.name] = mapping
            encoded_columns[column.name] = encoded

    target_column = spec.column_map.get(scenario.target_column)
    if target_column is None:
        raise ValueError(f"target column not found: {scenario.target_column}")
    target_values = np.empty(size, dtype=np.float64)
    for index in range(size):
        target_values[index] = _target_value(
            target_column,
            index,
            seed,
            spec,
            columns,
            encoded_columns,
        )
    target, target_missing = _numeric_fill(target_values)
    missing_counts[target_column.name] = target_missing

    content = {
        "spec_hash": spec.content_hash,
        "scenario": scenario.scenario_id,
        "seed": seed,
        "size": size,
        "sample_ids_hash": array_fingerprint(sample_ids),
        "columns": {name: array_fingerprint(values) for name, values in sorted(columns.items())},
        "target": array_fingerprint(target),
        "category_maps": category_maps,
        "missing_counts": missing_counts,
    }
    table = GeneratedTable(
        size=size,
        seed=seed,
        spec=spec,
        sample_ids=sample_ids,
        columns=columns,
        encoded_columns=encoded_columns,
        target=target,
        missing_counts=missing_counts,
        category_maps=category_maps,
        fingerprint=fingerprint(content),
    )
    return table
