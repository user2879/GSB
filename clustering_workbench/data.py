"""Deterministic synthetic table generation and structural validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .codes import Failure
from .config import dataset_descriptor_path, load_yaml
from .injections import InjectionManager
from .utils import hash_object, json_safe, sha256_text


def _column_rng(seed: int, name: str) -> np.random.Generator:
    digest = sha256_text(f"column:{seed}:{name}")
    return np.random.default_rng(int(digest[:16], 16))


@dataclass
class GeneratedDataset:
    descriptor: dict[str, Any]
    columns: dict[str, list[Any]]
    column_types: dict[str, str]
    sample_ids: list[str]
    target_columns: list[str]
    target_values: dict[str, list[Any]]
    group_ids: list[int]
    rows_requested: int
    rows_generated: int
    max_rows: int
    content_fingerprint: str
    descriptor_fingerprint: str
    non_finite_locations: list[dict[str, Any]] = field(default_factory=list)

    def preview(self, limit: int = 20) -> dict[str, Any]:
        count = min(limit, self.rows_generated)
        columns = {}
        for name in sorted(self.columns):
            values = []
            for value in self.columns[name][:count]:
                if isinstance(value, float) and not math.isfinite(value):
                    values.append(None)
                else:
                    values.append(value)
            columns[name] = values
        return {
            "sample_ids": self.sample_ids[:count],
            "columns": columns,
            "target": {
                name: self.target_values.get(name, [])[:count]
                for name in self.target_columns
            },
        }


def _generate_numeric(
    descriptor: dict[str, Any],
    groups: np.ndarray,
    rows: int,
    seed: int,
    generated: dict[str, list[Any]],
) -> list[Any]:
    rng = _column_rng(seed, descriptor["name"])
    distribution = descriptor.get("distribution")
    source = descriptor.get("derived", {}).get("source") if isinstance(descriptor.get("derived"), dict) else None
    if source:
        if source not in generated:
            raise ValueError(f"Derived column {descriptor['name']} references unknown source {source}")
        derived = descriptor.get("derived", {})
        multiplier = float(derived.get("multiplier", 1.0))
        offset = float(derived.get("offset", 0.0))
        values: list[Any] = []
        for value in generated[source]:
            values.append(None if value is None else float(value) * multiplier + offset)
        return values

    if distribution == "normal":
        group_means = descriptor.get("group_means")
        group_stds = descriptor.get("group_stds")
        if group_means is not None:
            means = np.asarray(group_means, dtype=np.float64)
            stds = np.asarray(group_stds if group_stds is not None else [1.0] * len(means), dtype=np.float64)
            values_array = means[groups] + stds[groups] * rng.normal(size=rows)
        else:
            values_array = rng.normal(
                float(descriptor.get("mean", 0.0)),
                float(descriptor.get("std", 1.0)),
                size=rows,
            )
    elif distribution == "uniform":
        low = float(descriptor.get("low", 0.0))
        high = float(descriptor.get("high", 1.0))
        values_array = rng.uniform(low, high, size=rows)
    elif distribution == "lognormal":
        values_array = rng.lognormal(
            float(descriptor.get("mean", 0.0)),
            float(descriptor.get("sigma", 1.0)),
            size=rows,
        )
    elif distribution == "beta":
        values_array = rng.beta(
            float(descriptor.get("a", 2.0)),
            float(descriptor.get("b", 5.0)),
            size=rows,
        )
    elif distribution == "constant":
        values_array = np.full(rows, float(descriptor.get("value", 0.0)), dtype=np.float64)
    else:
        raise ValueError(f"Unsupported numeric distribution for {descriptor['name']}: {distribution}")

    missing_rate = float(descriptor.get("missing_rate", 0.0))
    missing = rng.random(rows) < missing_rate if missing_rate > 0 else np.zeros(rows, dtype=bool)
    values = [None if missing[index] else float(values_array[index]) for index in range(rows)]
    return values


def _generate_categorical(
    descriptor: dict[str, Any],
    groups: np.ndarray,
    rows: int,
    seed: int,
    latent_groups: int,
) -> list[Any]:
    rng = _column_rng(seed, descriptor["name"])
    if descriptor.get("latent_target"):
        labels = list(descriptor.get("group_labels") or [f"group_{index}" for index in range(latent_groups)])
        return [str(labels[int(group)]) for group in groups]

    cardinality = int(descriptor.get("cardinality", len(descriptor.get("categories", [])) or 1))
    prefix = str(descriptor.get("prefix", descriptor["name"].upper()))
    categories = descriptor.get("categories")
    if not categories:
        categories = [f"{prefix}_{index:02d}" for index in range(cardinality)]
    categories = [str(value) for value in categories]

    group_categories = descriptor.get("group_categories")
    if group_categories:
        group_labels = [str(value) for value in group_categories]
        values = []
        for group in groups:
            if rng.random() < float(descriptor.get("group_alignment", 0.72)):
                values.append(group_labels[int(group) % len(group_labels)])
            else:
                values.append(categories[int(rng.integers(0, len(categories)))])
    else:
        rank = np.arange(1, len(categories) + 1, dtype=np.float64)
        weights = 1.0 / np.power(rank, float(descriptor.get("zipf_exponent", 0.0)))
        weights = weights / weights.sum()
        indices = rng.choice(len(categories), size=rows, p=weights)
        values = [categories[int(index)] for index in indices]

    missing_rate = float(descriptor.get("missing_rate", 0.0))
    if missing_rate > 0:
        missing = rng.random(rows) < missing_rate
        values = [None if missing[index] else values[index] for index in range(rows)]
    return values


def _target_structure_failures(
    descriptor: dict[str, Any],
    column_types: dict[str, str],
    config: dict[str, Any],
) -> tuple[list[str], list[Failure]]:
    structure = descriptor.get("structure")
    config_targets = config.get("data", {}).get("target_columns", [])
    failures: list[Failure] = []
    target_columns: list[str] = []
    if isinstance(config_targets, list) and config_targets:
        target_columns = [str(value) for value in config_targets]
    elif isinstance(structure, dict):
        raw_targets = structure.get("target_columns", [])
        if isinstance(raw_targets, list):
            target_columns = [str(value) for value in raw_targets]
    if not isinstance(structure, dict):
        failures.append(
            Failure(
                code="CFG_TARGET_STRUCTURE_MISSING",
                stage="load",
                item="dataset.structure",
                message="The dataset target structure declaration is missing.",
                details={"configured_target_columns": target_columns},
            )
        )
        return target_columns, failures
    if not target_columns:
        failures.append(
            Failure(
                code="CFG_TARGET_STRUCTURE_MISSING",
                stage="load",
                item="dataset.structure.target_columns",
                message="The dataset target_columns declaration is missing or empty.",
                details={"configured_target_columns": target_columns},
            )
        )
        return target_columns, failures
    for column in target_columns:
        if column not in column_types:
            failures.append(
                Failure(
                    code="CFG_TARGET_STRUCTURE_MISSING",
                    stage="load",
                    item=column,
                    message=f"Target structure column '{column}' is absent from the data description.",
                    details={"target_column": column},
                )
            )
    return target_columns, failures


def generate_dataset(
    config: dict[str, Any],
    config_path: Path,
    injections: InjectionManager,
) -> tuple[GeneratedDataset | None, list[Failure]]:
    failures: list[Failure] = []
    descriptor_path = dataset_descriptor_path(config, config_path)
    descriptor_root = load_yaml(descriptor_path)
    dataset = descriptor_root.get("dataset", descriptor_root)
    if not isinstance(dataset, dict):
        raise ValueError(f"Dataset descriptor must be a mapping: {descriptor_path}")

    columns_descriptors = dataset.get("columns", [])
    if not isinstance(columns_descriptors, list) or not columns_descriptors:
        return None, [
            Failure(
                code="CFG_UNKNOWN_COLUMN",
                stage="load",
                item="dataset.columns",
                message="The dataset declares no columns.",
                details={"descriptor": str(descriptor_path)},
            )
        ]

    seed = int(dataset.get("seed", config.get("seed", 0)))
    latent_groups = max(1, int(dataset.get("latent_groups", 1)))
    group_rng = _column_rng(seed, "__latent_groups__")
    base_rows = int(dataset.get("rows", 200))
    scale = float(config.get("data", {}).get("sample_scale", 1.0) or 1.0)
    requested_rows = max(1, int(round(base_rows * scale)))
    max_rows = int(config.get("data", {}).get("max_rows", dataset.get("max_rows", 10000)) or 10000)
    rows, injection_failures = injections.cap_rows(requested_rows, max_rows)
    failures.extend(injection_failures)
    groups = group_rng.integers(0, latent_groups, size=rows)

    column_types: dict[str, str] = {}
    columns: dict[str, list[Any]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    for raw_descriptor in columns_descriptors:
        if not isinstance(raw_descriptor, dict) or not raw_descriptor.get("name"):
            continue
        name = str(raw_descriptor["name"])
        column_type = str(raw_descriptor.get("type", "")).lower()
        if column_type not in {"numeric", "categorical"}:
            continue
        declarations[name] = raw_descriptor
        column_types[name] = column_type
        if column_type == "numeric":
            try:
                columns[name] = _generate_numeric(raw_descriptor, groups, rows, seed, columns)
            except Exception as error:  # isolation: retain remaining columns
                failures.append(
                    Failure(
                        code="DATA_NON_FINITE_VALUE",
                        stage="load",
                        item=name,
                        message=f"Could not generate numeric column '{name}': {error}",
                        details={"column": name},
                    )
                )
                columns[name] = [None] * rows
        else:
            columns[name] = _generate_categorical(raw_descriptor, groups, rows, seed, latent_groups)

    prefix = str(dataset.get("id_prefix", "S"))
    width = int(dataset.get("id_width", 5))
    sample_ids = [f"{prefix}{index:0{width}d}" for index in range(rows)]

    overrides = config.get("data", {}).get("sample_overrides", {})
    if isinstance(overrides, dict):
        sample_index = {sample_id: index for index, sample_id in enumerate(sample_ids)}
        for sample_id, column_values in overrides.items():
            if sample_id not in sample_index or not isinstance(column_values, dict):
                continue
            row_index = sample_index[str(sample_id)]
            for column, value in column_values.items():
                if column in columns:
                    columns[column][row_index] = value

    non_finite_failures = injections.apply_non_finite(columns, sample_ids)
    failures.extend(non_finite_failures)
    non_finite_locations = []
    for name in sorted(columns):
        for row_index, value in enumerate(columns[name]):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    non_finite_locations.append(
                        {"sample_id": sample_ids[row_index], "column": name, "row_index": row_index}
                    )

    target_columns, target_failures = _target_structure_failures(dataset, column_types, config)
    failures.extend(target_failures)
    target_values = {
        column: list(columns[column])
        for column in target_columns
        if column in columns
    }

    content_value = {
        "seed": seed,
        "rows": rows,
        "sample_ids": sample_ids,
        "columns": columns,
        "target_columns": target_columns,
    }
    content_fingerprint = hash_object("data", json_safe(content_value))
    descriptor_fingerprint = hash_object(
        "schema",
        {
            "descriptor": dataset,
            "content": content_fingerprint,
            "column_types": column_types,
            "sample_ids": sample_ids,
            "non_finite_locations": non_finite_locations,
        },
    )
    result = GeneratedDataset(
        descriptor=dataset,
        columns=columns,
        column_types=column_types,
        sample_ids=sample_ids,
        target_columns=target_columns,
        target_values=target_values,
        group_ids=[int(value) for value in groups],
        rows_requested=requested_rows,
        rows_generated=rows,
        max_rows=max_rows,
        content_fingerprint=content_fingerprint,
        descriptor_fingerprint=descriptor_fingerprint,
        non_finite_locations=non_finite_locations,
    )
    return result, failures
