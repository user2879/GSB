"""Deterministic synthetic table generation and categorical encoding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .errors import (
    E_CATEGORY_UNMAPPED,
    E_CONFIG_INVALID,
    E_INPUT_NON_FINITE,
    BusinessError,
    Failure,
)
from .util import canonical_json, jsonable, sha256_value


NUMERIC_KINDS = {"numeric", "number", "float", "integer", "int", "redundant"}
CATEGORICAL_KINDS = {"categorical", "category", "string", "object"}


def normalize_kind(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    kind = value.strip().lower()
    if kind in NUMERIC_KINDS:
        return "numeric"
    if kind in CATEGORICAL_KINDS:
        return "categorical"
    return None


def valid_seed(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**63


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str
    declared_kind: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataDescription:
    path: Path
    raw: dict[str, Any]
    seed: int
    seed_fallback: bool
    rows: int
    columns: list[ColumnSpec]
    groups: list[dict[str, Any]]
    failures: list[Failure]
    content_hash: str

    @property
    def column_map(self) -> dict[str, ColumnSpec]:
        return {column.name: column for column in self.columns}


@dataclass
class SyntheticDataset:
    seed: int
    rows: int
    columns: list[ColumnSpec]
    values: dict[str, list[Any]]
    group_index: list[int]
    content_hash: str
    description_hash: str

    @property
    def column_map(self) -> dict[str, ColumnSpec]:
        return {column.name: column for column in self.columns}

    def to_summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "seed": self.seed,
            "columns": [
                {
                    "name": column.name,
                    "kind": column.kind,
                    "declared_kind": column.declared_kind,
                }
                for column in self.columns
            ],
            "content_hash": self.content_hash,
            "description_hash": self.description_hash,
            "group_count": len(set(self.group_index)),
        }


@dataclass
class PreparedData:
    matrix: list[list[float]]
    feature_names: list[str]
    feature_blocks: dict[str, list[str]]
    row_ids: list[int]
    raw_values: dict[str, list[Any]]
    numeric_fill_values: dict[str, float]
    categorical_fill_values: dict[str, str]
    category_maps: dict[str, list[str]]
    category_column_kinds: dict[str, str]
    train_count: int
    source_hash: str
    data_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix,
            "feature_names": self.feature_names,
            "feature_blocks": self.feature_blocks,
            "row_ids": self.row_ids,
            "raw_values": self.raw_values,
            "numeric_fill_values": self.numeric_fill_values,
            "categorical_fill_values": self.categorical_fill_values,
            "category_maps": self.category_maps,
            "category_column_kinds": self.category_column_kinds,
            "train_count": self.train_count,
            "source_hash": self.source_hash,
            "data_hash": self.data_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreparedData":
        return cls(
            matrix=payload["matrix"],
            feature_names=payload["feature_names"],
            feature_blocks=payload["feature_blocks"],
            row_ids=payload["row_ids"],
            raw_values=payload["raw_values"],
            numeric_fill_values=payload["numeric_fill_values"],
            categorical_fill_values=payload["categorical_fill_values"],
            category_maps=payload["category_maps"],
            category_column_kinds=payload["category_column_kinds"],
            train_count=payload["train_count"],
            source_hash=payload["source_hash"],
            data_hash=payload["data_hash"],
        )


def _load_yaml(path: Path, stage: str) -> dict[str, Any]:
    if not path.is_file():
        raise BusinessError(
            E_CONFIG_INVALID,
            f"data description file does not exist: {path}",
            stage=stage,
            path=str(path),
            fatal=True,
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"cannot parse data description: {exc}",
            stage=stage,
            path=str(path),
            fatal=True,
        ) from exc
    if not isinstance(raw, dict):
        raise BusinessError(
            E_CONFIG_INVALID,
            "data description root must be a mapping",
            stage=stage,
            path=str(path),
            fatal=True,
        )
    return raw


def _normalize_description(raw: dict[str, Any], path: Path) -> DataDescription:
    failures: list[Failure] = []
    seed_raw = raw.get("seed")
    if valid_seed(seed_raw):
        seed = int(seed_raw)
        seed_fallback = False
    else:
        seed = 0
        seed_fallback = True
        failures.append(
            Failure(
                code=E_CONFIG_INVALID,
                message="data seed must be a non-negative integer below 2**63",
                stage="config",
                path="data.seed",
            )
        )

    rows_raw = raw.get("rows")
    if isinstance(rows_raw, int) and not isinstance(rows_raw, bool) and rows_raw > 0:
        rows = rows_raw
    else:
        rows = 120
        failures.append(
            Failure(
                code=E_CONFIG_INVALID,
                message="data rows must be a positive integer",
                stage="config",
                path="data.rows",
            )
        )

    columns_raw = raw.get("columns")
    if not isinstance(columns_raw, list):
        raise BusinessError(
            E_CONFIG_INVALID,
            "data.columns must be a non-empty list",
            stage="config",
            path=str(path),
            fatal=True,
        )

    columns: list[ColumnSpec] = []
    names: set[str] = set()
    for index, item in enumerate(columns_raw):
        location = f"data.columns[{index}]"
        if not isinstance(item, dict):
            failures.append(Failure(E_CONFIG_INVALID, "column must be a mapping", "config", path=location))
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            failures.append(Failure(E_CONFIG_INVALID, "column name must be a non-empty string", "config", path=location))
            continue
        name = name.strip()
        if name in names:
            failures.append(
                Failure(E_CONFIG_INVALID, f"duplicate column name: {name}", "config", path=location)
            )
            continue
        declared_kind = str(item.get("kind", ""))
        kind = normalize_kind(declared_kind)
        if kind is None:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    f"unknown column kind for {name}: {declared_kind!r}",
                    "config",
                    path=location,
                )
            )
            continue
        options = dict(item)
        missing_rate = options.get("missing_rate", 0.0)
        if not isinstance(missing_rate, (int, float)) or isinstance(missing_rate, bool) or not 0 <= float(missing_rate) <= 1:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    f"missing_rate for {name} must be within [0, 1]",
                    "config",
                    path=location,
                )
            )
            options["missing_rate"] = 0.0
        else:
            options["missing_rate"] = float(missing_rate)

        if kind == "categorical":
            values = options.get("values")
            cardinality = options.get("cardinality")
            if values is not None:
                if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
                    failures.append(
                        Failure(
                            E_CONFIG_INVALID,
                            f"categorical values for {name} must be a non-empty string list",
                            "config",
                            path=location,
                        )
                    )
                    continue
                if len(set(values)) != len(values):
                    failures.append(
                        Failure(
                            E_CONFIG_INVALID,
                            f"categorical values for {name} contain duplicates",
                            "config",
                            path=location,
                        )
                    )
                    continue
            elif not (isinstance(cardinality, int) and not isinstance(cardinality, bool) and cardinality > 0):
                failures.append(
                    Failure(
                        E_CONFIG_INVALID,
                        f"categorical column {name} needs values or positive cardinality",
                        "config",
                        path=location,
                    )
                )
                continue
        if kind == "numeric":
            distribution = options.get("distribution", "normal")
            if distribution not in {"normal", "lognormal", "uniform"}:
                failures.append(
                    Failure(
                        E_CONFIG_INVALID,
                        f"unsupported distribution for {name}: {distribution!r}",
                        "config",
                        path=location,
                    )
                )
                options["distribution"] = "normal"
        columns.append(ColumnSpec(name=name, kind=kind, declared_kind=declared_kind, options=options))
        names.add(name)

    seen_column_names: set[str] = set()
    valid_columns: list[ColumnSpec] = []
    for spec in columns:
        if spec.declared_kind.lower() == "redundant":
            source = spec.options.get("source")
            if not isinstance(source, str) or source not in seen_column_names:
                failures.append(
                    Failure(
                        E_CONFIG_INVALID,
                        f"redundant column {spec.name} requires an earlier source column; got {source!r}",
                        "config",
                        path=f"columns.{spec.name}.source",
                    )
                )
                continue
        valid_columns.append(spec)
        seen_column_names.add(spec.name)
    columns = valid_columns

    if not columns:
        raise BusinessError(
            E_CONFIG_INVALID,
            "data description has no usable columns",
            stage="config",
            path=str(path),
            fatal=True,
        )

    groups_raw = raw.get("groups", [])
    if not isinstance(groups_raw, list):
        failures.append(Failure(E_CONFIG_INVALID, "data.groups must be a list", "config", path="data.groups"))
        groups_raw = []
    groups = [group for group in groups_raw if isinstance(group, dict)]
    if len(groups) != len(groups_raw):
        failures.append(Failure(E_CONFIG_INVALID, "each data group must be a mapping", "config", path="data.groups"))

    content_hash = sha256_value({"description": raw})
    return DataDescription(
        path=path,
        raw=raw,
        seed=seed,
        seed_fallback=seed_fallback,
        rows=rows,
        columns=columns,
        groups=groups,
        failures=failures,
        content_hash=content_hash,
    )


def load_data_description(path: Path) -> DataDescription:
    path = Path(path).resolve()
    raw = _load_yaml(path, "config")
    return _normalize_description(raw, path)


def _group_indices(rows: int, groups: list[dict[str, Any]]) -> list[int]:
    if not groups:
        return [0] * rows
    sizes_raw = [group.get("size") for group in groups]
    sizes: list[int] = []
    for index, size in enumerate(sizes_raw):
        if isinstance(size, int) and not isinstance(size, bool) and size > 0:
            sizes.append(size)
        else:
            sizes.append(0)
    if sum(sizes) == 0:
        return [index % len(groups) for index in range(rows)]
    assigned: list[int] = []
    for group_index, size in enumerate(sizes):
        assigned.extend([group_index] * size)
    if len(assigned) < rows:
        assigned.extend([len(groups) - 1] * (rows - len(assigned)))
    return assigned[:rows]


def _numeric_values(
    spec: ColumnSpec,
    rows: int,
    group_index: list[int],
    groups: list[dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    options = spec.options
    distribution = options.get("distribution", "normal")
    if distribution == "lognormal":
        meanlog = float(options.get("meanlog", options.get("mean", 0.0)))
        sigma = max(float(options.get("sigma", options.get("std", 1.0))), 0.0)
        values = rng.lognormal(mean=meanlog, sigma=sigma, size=rows)
    elif distribution == "uniform":
        low = float(options.get("low", 0.0))
        high = float(options.get("high", 1.0))
        if high < low:
            low, high = high, low
        values = rng.uniform(low=low, high=high, size=rows)
    else:
        mean = float(options.get("mean", 0.0))
        std = max(float(options.get("std", 1.0)), 0.0)
        values = rng.normal(loc=mean, scale=std, size=rows)

    offsets = options.get("group_offsets")
    offsets = offsets if isinstance(offsets, list) else []
    for index, group in enumerate(groups):
        group_offsets = group.get("numeric_offsets", {})
        if isinstance(group_offsets, dict) and spec.name in group_offsets:
            offsets = list(offsets) + [0.0] * max(0, len(groups) - len(offsets))
            offsets[index] = group_offsets[spec.name]
    if offsets:
        for row, group in enumerate(group_index):
            if group < len(offsets) and isinstance(offsets[group], (int, float)):
                values[row] += float(offsets[group])

    if "min" in options:
        values = np.maximum(values, float(options["min"]))
    if "max" in options:
        values = np.minimum(values, float(options["max"]))
    if options.get("dtype") in {"int", "integer"} or options.get("integer") is True:
        values = np.rint(values)
    decimals = options.get("decimals")
    if isinstance(decimals, int) and not isinstance(decimals, bool) and 0 <= decimals <= 15:
        values = np.round(values, decimals=decimals)
    return np.asarray(values, dtype=float)


def _categorical_values(
    spec: ColumnSpec,
    rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    options = spec.options
    values = options.get("values")
    if isinstance(values, list) and values:
        categories = [str(value) for value in values]
    else:
        cardinality = int(options["cardinality"])
        prefix = str(options.get("prefix", spec.name))
        categories = [f"{prefix}_{index:03d}" for index in range(cardinality)]
    probabilities = options.get("probabilities")
    if isinstance(probabilities, list) and len(probabilities) == len(categories):
        try:
            weights = np.asarray(probabilities, dtype=float)
        except (TypeError, ValueError):
            weights = np.ones(len(categories), dtype=float)
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or float(weights.sum()) <= 0:
            weights = np.ones(len(categories), dtype=float)
    else:
        weights = np.ones(len(categories), dtype=float)
    weights = weights / weights.sum()
    selected = rng.choice(len(categories), size=rows, p=weights)
    return np.asarray([categories[int(index)] for index in selected], dtype=object)


def generate_dataset(description: DataDescription, seed: int | None = None) -> SyntheticDataset:
    used_seed = description.seed if seed is None else int(seed)
    rng = np.random.Generator(np.random.PCG64(used_seed))
    rows = description.rows
    group_index = _group_indices(rows, description.groups)
    values: dict[str, list[Any]] = {}
    columns: list[ColumnSpec] = []

    for spec in description.columns:
        if spec.kind == "numeric" and spec.declared_kind.lower() not in {"redundant"}:
            column_values = _numeric_values(spec, rows, group_index, description.groups, rng)
            missing_rate = float(spec.options.get("missing_rate", 0.0))
            if missing_rate > 0:
                mask = rng.random(rows) < missing_rate
                column_values = column_values.astype(float)
                column_values[mask] = np.nan
            values[spec.name] = [float(value) for value in column_values]
        elif spec.declared_kind.lower() == "redundant":
            source = spec.options.get("source")
            if not isinstance(source, str) or source not in values:
                continue
            base = np.asarray(values[source], dtype=float)
            scale = float(spec.options.get("scale", 1.0))
            offset = float(spec.options.get("offset", 0.0))
            noise_std = max(float(spec.options.get("noise_std", 0.0)), 0.0)
            column_values = base * scale + offset
            if noise_std > 0:
                column_values = column_values + rng.normal(0.0, noise_std, size=rows)
            missing_rate = float(spec.options.get("missing_rate", 0.0))
            if missing_rate > 0:
                mask = rng.random(rows) < missing_rate
                column_values = column_values.astype(float)
                column_values[mask] = np.nan
            decimals = spec.options.get("decimals")
            if isinstance(decimals, int) and not isinstance(decimals, bool) and 0 <= decimals <= 15:
                column_values = np.round(column_values, decimals=decimals)
            values[spec.name] = [float(value) for value in column_values]
        else:
            column_values = _categorical_values(spec, rows, rng)
            missing_rate = float(spec.options.get("missing_rate", 0.0))
            if missing_rate > 0:
                mask = rng.random(rows) < missing_rate
                column_values = column_values.astype(object)
                column_values[mask] = None
            values[spec.name] = [
                None if value is None else str(value) for value in column_values
            ]
        columns.append(spec)

    content = {
        "seed": used_seed,
        "rows": rows,
        "columns": [
            {"name": column.name, "kind": column.kind, "declared_kind": column.declared_kind}
            for column in columns
        ],
        "values": values,
        "group_index": group_index,
    }
    return SyntheticDataset(
        seed=used_seed,
        rows=rows,
        columns=columns,
        values=values,
        group_index=group_index,
        content_hash=sha256_value(content),
        description_hash=description.content_hash,
    )


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _stable_median(values: list[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _stable_mode(values: list[str], fallback: str) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return fallback
    return sorted(counts, key=lambda value: (-counts[value], value))[0]


def assert_finite_raw_values(
    dataset: SyntheticDataset,
    stage: str = "data",
) -> None:
    for column in dataset.columns:
        if column.kind != "numeric":
            continue
        for row, value in enumerate(dataset.values[column.name]):
            if not _is_missing(value) and not math.isfinite(float(value)):
                raise BusinessError(
                    E_INPUT_NON_FINITE,
                    f"non-finite value at sample {row}, column {column.name}: {value!r}",
                    stage=stage,
                    path=f"sample[{row}].{column.name}",
                    extra={"sample": row, "column": column.name},
                )


def prepare_dataset(dataset: SyntheticDataset, fit_fraction: float) -> PreparedData:
    assert_finite_raw_values(dataset, stage="data")
    rows = dataset.rows
    train_count = max(1, min(rows, int(math.ceil(fit_fraction * rows))))
    numeric_fill_values: dict[str, float] = {}
    categorical_fill_values: dict[str, str] = {}
    category_maps: dict[str, list[str]] = {}
    category_column_kinds: dict[str, str] = {}

    for column in dataset.columns:
        source = dataset.values[column.name]
        if column.kind == "numeric":
            train_values = [
                float(value)
                for value in source[:train_count]
                if not _is_missing(value)
            ]
            numeric_fill_values[column.name] = _stable_median(train_values)
        else:
            train_values = [
                str(value)
                for value in source[:train_count]
                if not _is_missing(value)
            ]
            fill = _stable_mode(train_values, "<MISSING>")
            categorical_fill_values[column.name] = fill
            all_values = sorted(
                {
                    str(value)
                    for value in source
                    if not _is_missing(value)
                }
                | {fill}
            )
            category_maps[column.name] = all_values
            category_column_kinds[column.name] = column.kind

    feature_names: list[str] = []
    feature_blocks: dict[str, list[str]] = {}
    matrix: list[list[float]] = []
    for row in range(rows):
        encoded_row: list[float] = []
        for column in dataset.columns:
            raw_value = dataset.values[column.name][row]
            if column.kind == "numeric":
                value = numeric_fill_values[column.name] if _is_missing(raw_value) else float(raw_value)
                name = column.name
                if row == 0:
                    feature_names.append(name)
                encoded_row.append(value)
            else:
                value = categorical_fill_values[column.name] if _is_missing(raw_value) else str(raw_value)
                mapping = category_maps[column.name]
                if value not in mapping:
                    raise BusinessError(
                        E_CATEGORY_UNMAPPED,
                        f"unmapped category for column {column.name}: {value!r}",
                        stage="data",
                        path=f"sample[{row}].{column.name}",
                    )
                block = [f"{column.name}={category}" for category in mapping]
                if row == 0:
                    feature_names.extend(block)
                    feature_blocks[column.name] = block
                encoded_row.extend(1.0 if category == value else 0.0 for category in mapping)
        matrix.append(encoded_row)

    payload = {
        "feature_names": feature_names,
        "feature_blocks": feature_blocks,
        "row_ids": list(range(rows)),
        "raw_values": dataset.values,
        "numeric_fill_values": numeric_fill_values,
        "categorical_fill_values": categorical_fill_values,
        "category_maps": category_maps,
        "category_column_kinds": category_column_kinds,
        "train_count": train_count,
        "source_hash": dataset.content_hash,
        "data_hash": sha256_value({"source_hash": dataset.content_hash, "matrix": matrix}),
    }
    return PreparedData(
        matrix=matrix,
        feature_names=feature_names,
        feature_blocks=feature_blocks,
        row_ids=payload["row_ids"],
        raw_values=payload["raw_values"],
        numeric_fill_values=numeric_fill_values,
        categorical_fill_values=categorical_fill_values,
        category_maps=category_maps,
        category_column_kinds=category_column_kinds,
        train_count=train_count,
        source_hash=payload["source_hash"],
        data_hash=payload["data_hash"],
    )


def encode_raw_category(prepared: PreparedData, column: str, value: str) -> list[float]:
    """Invoke the same category-map path used by data preparation."""
    mapping = prepared.category_maps.get(column)
    if mapping is None:
        raise BusinessError(
            E_CONFIG_INVALID,
            f"column is not a categorical feature: {column}",
            stage="preprocess",
            path=column,
        )
    if value not in mapping:
        raise BusinessError(
            E_CATEGORY_UNMAPPED,
            f"unmapped category for column {column}: {value!r}",
            stage="preprocess",
            path=column,
            extra={"column": column, "value": value},
        )
    return [1.0 if category == value else 0.0 for category in mapping]


def prepared_summary(prepared: PreparedData) -> dict[str, Any]:
    return {
        "rows": len(prepared.matrix),
        "feature_count": len(prepared.feature_names),
        "feature_names": prepared.feature_names,
        "feature_blocks": prepared.feature_blocks,
        "train_count": prepared.train_count,
        "numeric_fill_values": prepared.numeric_fill_values,
        "categorical_fill_values": prepared.categorical_fill_values,
        "category_maps": prepared.category_maps,
        "source_hash": prepared.source_hash,
        "data_hash": prepared.data_hash,
    }
