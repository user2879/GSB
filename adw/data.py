"""Deterministic synthetic tabular data generation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .canonical import StableRNG, canonical_json, sha256_text, stable_float
from .errors import ConfigValidationError, Failure, config_failure, sort_failures


@dataclass
class Dataset:
    description: dict[str, Any]
    sample_ids: list[str]
    rows: list[dict[str, Any]]
    columns: list[str]
    target_field: str
    target: list[bool]
    fingerprint: str


def _seed_for(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _normal(rng: StableRNG) -> float:
    u1 = max(rng.random(), 1.0e-15)
    u2 = rng.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _anomaly_indices(n: int, ratio: float, seed: int) -> set[int]:
    count = _round_half_up(n * ratio)
    if count <= 0:
        return set()
    indices = list(range(n))
    rng = StableRNG(seed ^ 0xD1A6_0F1A_5EED)
    # Partial Fisher-Yates: only the first `count` positions need shuffling.
    for i in range(count):
        j = i + rng.randbelow(n - i)
        indices[i], indices[j] = indices[j], indices[i]
    return set(indices[:count])


def _validate_generation_inputs(description: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[Failure] = []
    dataset = description.get("dataset") if isinstance(description, dict) else None
    if not isinstance(dataset, dict):
        return sort_failures([config_failure("dataset", "target structure is missing")])
    columns = dataset.get("columns")
    if not isinstance(columns, list) or not columns:
        failures.append(config_failure("dataset.columns", "at least one column is required"))
        return sort_failures(failures)
    names: set[str] = set()
    for index, item in enumerate(columns):
        if not isinstance(item, dict):
            failures.append(config_failure(f"dataset.columns[{index}]", "column must be a mapping"))
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            failures.append(config_failure(f"dataset.columns[{index}].name", "invalid column name"))
            continue
        if name in names:
            failures.append(config_failure(f"dataset.columns[{index}].name", f"duplicate column {name!r}"))
        names.add(name)
        kind = item.get("type")
        if kind == "numeric" and item.get("distribution") == "redundant":
            source = item.get("source")
            if not isinstance(source, str) or source not in {x.get("name") for x in columns if isinstance(x, dict)}:
                failures.append(config_failure(f"dataset.columns[{index}].source", "unknown redundant source"))
        elif kind not in ("numeric", "categorical"):
            failures.append(config_failure(f"dataset.columns[{index}].type", "column type must be numeric or categorical"))
    return sort_failures(failures)


def generate_dataset(description: dict[str, Any]) -> Dataset:
    failures = _validate_generation_inputs(description)
    if failures:
        raise ConfigValidationError([Failure(**x) for x in failures])
    dataset = description["dataset"]
    n = int(dataset["sample_count"])
    seed = int(dataset["seed"])
    target_spec = dataset["target"]
    target_field = str(target_spec["field"])
    anomaly_ratio = float(target_spec["anomaly_ratio"])
    missing_ratio = float(dataset.get("missing_ratio", 0.0))
    prefix = str(dataset.get("id_prefix", "S"))
    width = int(dataset.get("id_width", 6))
    columns = [item for item in dataset["columns"]]
    column_names = [str(item["name"]) for item in columns]
    if target_field in column_names:
        raise ConfigValidationError(
            [Failure(code="E_CONFIG_INVALID", stage="config", field="dataset.target.field", message="target field duplicates a feature column")]
        )

    ids = [f"{prefix}{index:0{width}d}" for index in range(1, n + 1)]
    anomaly_ids = _anomaly_indices(n, anomaly_ratio, seed)
    values: dict[str, list[Any]] = {}

    for column_index, column in enumerate(columns):
        name = str(column["name"])
        kind = str(column["type"])
        rng = StableRNG(_seed_for(seed, f"column:{column_index}:{name}"))
        generated: list[Any] = []
        if kind == "categorical":
            categories = [str(x) for x in column["categories"]]
            rare = column.get("anomaly_category")
            rare = str(rare) if rare is not None else categories[-1]
            for row_index in range(n):
                if row_index in anomaly_ids and rare in categories:
                    generated.append(rare)
                else:
                    generated.append(categories[rng.randbelow(len(categories))])
        elif column.get("distribution") == "redundant":
            source = str(column["source"])
            noise = float(column.get("noise", 0.0))
            source_values = values.get(source)
            if source_values is None:
                raise ConfigValidationError(
                    [
                        Failure(
                            code="E_CONFIG_INVALID",
                            stage="config",
                            field=f"dataset.columns[{column_index}].source",
                            message="redundant source must appear earlier in the column list",
                        )
                    ]
                )
            for row_index in range(n):
                source_value = source_values[row_index]
                if source_value is None:
                    generated.append(None)
                else:
                    generated.append(float(source_value) + noise * _normal(rng))
        else:
            mean = float(column.get("mean", 0.0))
            std = float(column.get("std", 1.0))
            shift = column.get("anomaly_shift")
            shift = float(shift) if shift is not None else 4.0 * std
            for row_index in range(n):
                value = mean + std * _normal(rng)
                if row_index in anomaly_ids:
                    value += shift
                generated.append(float(value))
        values[name] = generated

    # Missingness is applied after all values exist, with independent deterministic masks.
    if missing_ratio > 0.0:
        cell_count = n * len(columns)
        target_missing = min(cell_count, int(math.floor(cell_count * missing_ratio + 0.5)))
        chosen: set[int] = set()
        rng = StableRNG(_seed_for(seed, "missing-cells"))
        while len(chosen) < target_missing:
            chosen.add(rng.randbelow(cell_count))
        for cell in sorted(chosen):
            row_index, column_index = divmod(cell, len(columns))
            values[column_names[column_index]][row_index] = None

    target = [index in anomaly_ids for index in range(n)]
    rows: list[dict[str, Any]] = []
    for row_index in range(n):
        row: dict[str, Any] = {name: values[name][row_index] for name in column_names}
        row[target_field] = bool(target[row_index])
        rows.append(row)

    fingerprint = _dataset_fingerprint(description, ids, rows, column_names, target_field)
    return Dataset(
        description=description,
        sample_ids=ids,
        rows=rows,
        columns=column_names,
        target_field=target_field,
        target=target,
        fingerprint=fingerprint,
    )


def _dataset_fingerprint(
    description: dict[str, Any],
    sample_ids: list[str],
    rows: list[dict[str, Any]],
    columns: list[str],
    target_field: str,
) -> str:
    h = hashlib.sha256()
    h.update(b"dataset-v1\n")
    h.update(canonical_json(description).encode("utf-8"))
    h.update(b"\n")
    for sample_id, row in zip(sample_ids, rows):
        h.update(sample_id.encode("utf-8"))
        for name in columns:
            value = row[name]
            h.update(b"|")
            if value is None:
                h.update(b"N")
            elif isinstance(value, bool):
                h.update(b"B1" if value else b"B0")
            elif isinstance(value, (int, float)):
                h.update(stable_float(float(value)).encode("ascii"))
            else:
                h.update(str(value).encode("utf-8"))
        h.update(b"|target=")
        h.update(b"1" if row[target_field] else b"0")
        h.update(b"\n")
    return h.hexdigest()


def dataset_feature_values(dataset: Dataset, feature_columns: list[str]) -> list[list[Any]]:
    """Return feature columns in configured order, excluding the target."""

    return [[row[name] for name in feature_columns] for row in dataset.rows]


def feature_data_fingerprint(dataset: Dataset, feature_columns: list[str]) -> str:
    return sha256_text(
        "features\n"
        + dataset.fingerprint
        + "\n"
        + canonical_json(feature_columns)
    )
