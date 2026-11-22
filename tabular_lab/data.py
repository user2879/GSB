"""Deterministic synthetic tabular data generation and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .errors import E_CONFIG_INVALID, fail
from .util import child_rng, content_hash, to_jsonable


COLUMN_KINDS = {"numeric", "categorical", "redundant"}


@dataclass
class DatasetBundle:
    target_name: str
    column_names: list[str]
    column_kinds: dict[str, str]
    raw_columns: dict[str, np.ndarray]
    target: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    holdout_indices: np.ndarray
    feature_names: list[str]
    feature_to_raw: dict[str, str]
    raw_to_features: dict[str, list[str]]
    feature_indices: dict[str, int]
    matrix_encoded: np.ndarray
    matrix_standardized: np.ndarray
    feature_means: np.ndarray
    feature_scales: np.ndarray
    numeric_fill_means: dict[str, float]
    categorical_fill_modes: dict[str, str]
    column_fingerprints: dict[str, str]
    target_fingerprint: str
    split_fingerprint: str
    schema_fingerprint: str
    data_fingerprint: str
    train_sample_fingerprint: str
    validation_sample_fingerprint: str

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def resolve_features(self, requested: Iterable[str] | None) -> list[int]:
        if requested is None:
            return list(range(self.num_features))
        requested_list = [str(item) for item in requested]
        if not requested_list:
            return list(range(self.num_features))

        indices: set[int] = set()
        unknown: list[str] = []
        for name in requested_list:
            if name in self.feature_indices:
                indices.add(self.feature_indices[name])
            elif name in self.raw_to_features:
                indices.update(self.feature_indices[feature] for feature in self.raw_to_features[name])
            else:
                unknown.append(name)
        if unknown:
            fail(
                E_CONFIG_INVALID,
                "feature whitelist contains unknown column(s)",
                unknown_columns=sorted(set(unknown)),
                known_columns=sorted(set(self.raw_to_features) | set(self.feature_indices)),
            )
        return sorted(indices)

    def selected(self, indices: list[int]) -> "SelectedDataset":
        canonical = tuple(sorted(set(indices)))
        if not canonical:
            fail(E_CONFIG_INVALID, "a model must select at least one feature")
        names = [self.feature_names[index] for index in canonical]
        return SelectedDataset(bundle=self, indices=canonical, feature_names=names)


@dataclass
class SelectedDataset:
    bundle: DatasetBundle
    indices: tuple[int, ...]
    feature_names: list[str]

    @property
    def x_train(self) -> np.ndarray:
        return self.bundle.matrix_standardized[np.ix_(self.bundle.train_indices, self.indices)]

    @property
    def x_validation(self) -> np.ndarray:
        return self.bundle.matrix_standardized[np.ix_(self.bundle.validation_indices, self.indices)]

    @property
    def y_train(self) -> np.ndarray:
        return self.bundle.target[self.bundle.train_indices]

    @property
    def y_validation(self) -> np.ndarray:
        return self.bundle.target[self.bundle.validation_indices]

    def data_slice_fingerprint(self) -> str:
        encoded = self.bundle.matrix_encoded[np.ix_(self.bundle.train_indices, self.indices)]
        validation_encoded = self.bundle.matrix_encoded[np.ix_(self.bundle.validation_indices, self.indices)]
        payload = {
            "features": self.feature_names,
            "target": self.bundle.target_name,
            "train_x": _matrix_payload(encoded),
            "validation_x": _matrix_payload(validation_encoded),
            "train_y": _array_payload(self.y_train),
            "validation_y": _array_payload(self.y_validation),
            "train_indices": self.bundle.train_indices.tolist(),
            "validation_indices": self.bundle.validation_indices.tolist(),
        }
        return content_hash("slice", payload)

    def standardization_fingerprint(self) -> str:
        means = self.bundle.feature_means[list(self.indices)]
        scales = self.bundle.feature_scales[list(self.indices)]
        raw_columns = sorted({self.bundle.feature_to_raw[name] for name in self.feature_names})
        imputation = {
            "numeric_means": {
                name: self.bundle.numeric_fill_means[name]
                for name in raw_columns
                if name in self.bundle.numeric_fill_means
            },
            "categorical_modes": {
                name: self.bundle.categorical_fill_modes[name]
                for name in raw_columns
                if name in self.bundle.categorical_fill_modes
            },
        }
        payload = {
            "features": self.feature_names,
            "means": means.tolist(),
            "scales": scales.tolist(),
            "imputation": imputation,
            "train_indices": self.bundle.train_indices.tolist(),
        }
        return content_hash("stats", payload)

    def sample_fingerprint(self) -> str:
        return content_hash(
            "sample",
            {
                "train": self.bundle.train_indices.tolist(),
                "validation": self.bundle.validation_indices.tolist(),
                "train_y": _array_payload(self.y_train),
                "validation_y": _array_payload(self.y_validation),
            },
        )


def _array_payload(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value is None:
            result.append({"missing": True})
        elif isinstance(value, (float, np.floating)) and np.isnan(float(value)):
            result.append({"missing": True})
        else:
            result.append(to_jsonable(value))
    return result


def _matrix_payload(matrix: np.ndarray) -> list[list[Any]]:
    return [_array_payload(row) for row in matrix]


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(E_CONFIG_INVALID, f"{field_name} must be numeric", value=value)
    result = float(value)
    if not np.isfinite(result):
        fail(E_CONFIG_INVALID, f"{field_name} must be finite", value=value)
    return result


def _validate_weights(value: Any, field_name: str) -> list[float]:
    if value is None:
        return []
    if not isinstance(value, list):
        fail(E_CONFIG_INVALID, f"{field_name} must be a list")
    return [_finite_float(item, f"{field_name}[{index}]") for index, item in enumerate(value)]


def validate_data_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        fail(E_CONFIG_INVALID, "data description must be a mapping")
    if "seed" in spec:
        data_seed = spec.get("seed")
        if isinstance(data_seed, bool) or not isinstance(data_seed, int):
            fail(E_CONFIG_INVALID, "data.seed must be an integer")
    rows = spec.get("rows")
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 4:
        fail(E_CONFIG_INVALID, "data.rows must be an integer >= 4")

    columns = spec.get("columns")
    if not isinstance(columns, list) or not columns:
        fail(E_CONFIG_INVALID, "data.columns must be a non-empty list")
    seen: set[str] = set()
    for index, column in enumerate(columns):
        if not isinstance(column, dict):
            fail(E_CONFIG_INVALID, "each data column must be a mapping", index=index)
        name = str(column.get("name", "")).strip()
        if not name:
            fail(E_CONFIG_INVALID, "data column name is required", index=index)
        if name in seen:
            fail(E_CONFIG_INVALID, f"duplicate data column: {name}")
        seen.add(name)
        kind = str(column.get("kind", "numeric"))
        if kind not in COLUMN_KINDS:
            fail(E_CONFIG_INVALID, f"unknown data column kind: {kind}", column=name)
        missing_rate = _finite_float(column.get("missing_rate", 0.0), f"column[{name}].missing_rate")
        if missing_rate < 0.0 or missing_rate >= 1.0:
            fail(E_CONFIG_INVALID, "missing_rate must be in [0, 1)", column=name, value=missing_rate)
        variant = column.get("variant", 0)
        if isinstance(variant, bool) or not isinstance(variant, int):
            fail(E_CONFIG_INVALID, "column variant must be an integer", column=name)
        if kind in {"numeric", "categorical"}:
            _validate_weights(column.get("latent_weights", [1.0, 0.0]), f"column[{name}].latent_weights")
        if kind == "numeric":
            _finite_float(column.get("mean", 0.0), f"column[{name}].mean")
            if _finite_float(column.get("scale", 1.0), f"column[{name}].scale") <= 0.0:
                fail(E_CONFIG_INVALID, "numeric column scale must be positive", column=name)
            noise = _finite_float(column.get("noise_scale", 0.5), f"column[{name}].noise_scale")
            if noise < 0.0:
                fail(E_CONFIG_INVALID, "noise_scale must be non-negative", column=name)
        if kind == "categorical":
            categories = column.get("categories")
            if not isinstance(categories, list) or len(categories) < 2:
                fail(E_CONFIG_INVALID, "categorical columns require at least two categories", column=name)
            normalized_categories = [str(item) for item in categories]
            if len(set(normalized_categories)) != len(normalized_categories):
                fail(E_CONFIG_INVALID, "categorical values must be unique", column=name)
            if not isinstance(column.get("drop_first", True), bool):
                fail(E_CONFIG_INVALID, "drop_first must be boolean", column=name)
            noise = _finite_float(column.get("noise_scale", 0.25), f"column[{name}].noise_scale")
            if noise < 0.0:
                fail(E_CONFIG_INVALID, "noise_scale must be non-negative", column=name)
        if kind == "redundant":
            source = str(column.get("source", "")).strip()
            if not source:
                fail(E_CONFIG_INVALID, "redundant columns require a source", column=name)
            _finite_float(column.get("multiplier", 1.0), f"column[{name}].multiplier")
            _finite_float(column.get("offset", 0.0), f"column[{name}].offset")
            noise = _finite_float(column.get("noise_scale", 0.0), f"column[{name}].noise_scale")
            if noise < 0.0:
                fail(E_CONFIG_INVALID, "noise_scale must be non-negative", column=name)

    target = spec.get("target")
    if not isinstance(target, dict):
        fail(E_CONFIG_INVALID, "data.target must be a mapping")
    target_name = str(target.get("name", "target")).strip()
    if not target_name:
        fail(E_CONFIG_INVALID, "target name is required")
    if target_name in seen:
        fail(E_CONFIG_INVALID, "target name must not collide with a feature column")
    if str(target.get("kind", "binary")) != "binary":
        fail(E_CONFIG_INVALID, "only a binary target is supported")
    if float(target.get("missing_rate", 0.0)) != 0.0:
        fail(E_CONFIG_INVALID, "target missing_rate must be zero")
    _validate_weights(target.get("latent_weights", [1.0, -0.5]), "target.latent_weights")
    _finite_float(target.get("noise_scale", 0.8), "target.noise_scale")
    _finite_float(target.get("offset", 0.0), "target.offset")

    split = spec.get("split", {})
    if not isinstance(split, dict):
        fail(E_CONFIG_INVALID, "data.split must be a mapping")
    train_fraction = _finite_float(split.get("train_fraction", 0.7), "split.train_fraction")
    validation_fraction = _finite_float(split.get("validation_fraction", 0.3), "split.validation_fraction")
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        fail(E_CONFIG_INVALID, "split fractions must be positive")
    if train_fraction + validation_fraction > 1.0 + 1e-12:
        fail(E_CONFIG_INVALID, "train_fraction + validation_fraction must not exceed 1")


def _pad_weights(weights: list[float], dimension: int) -> np.ndarray:
    padded = np.zeros(dimension, dtype=np.float64)
    if weights:
        padded[: min(dimension, len(weights))] = weights[:dimension]
    return padded


def _latent_dimension(spec: dict[str, Any]) -> int:
    lengths = [len(spec.get("target", {}).get("latent_weights", [1.0, -0.5]))]
    for column in spec["columns"]:
        lengths.append(len(column.get("latent_weights", [1.0, 0.0])))
    return max(1, max(lengths))


def _mode(values: list[str], category_order: list[str]) -> str:
    counts = {category: 0 for category in category_order}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(category_order, key=lambda category: (counts.get(category, 0), -category_order.index(category)))


def generate_dataset(spec: dict[str, Any], seed: int) -> DatasetBundle:
    validate_data_spec(spec)
    # A data-file seed is an explicit override; otherwise the experiment seed is
    # used. The effective seed is part of all derived content fingerprints.
    if "seed" in spec:
        seed = int(spec["seed"])
    rows = int(spec["rows"])
    latent_dim = _latent_dimension(spec)
    latent_rng = child_rng(seed, "latents", latent_dim)
    latents = latent_rng.standard_normal((rows, latent_dim))

    columns_spec = {str(column["name"]): column for column in spec["columns"]}
    column_order = [str(column["name"]) for column in spec["columns"]]
    raw_columns: dict[str, np.ndarray] = {}
    column_kinds: dict[str, str] = {}
    generating: set[str] = set()

    def generate_column(name: str) -> np.ndarray:
        if name in raw_columns:
            return raw_columns[name]
        if name not in columns_spec:
            fail(E_CONFIG_INVALID, f"redundant column references unknown source: {name}")
        if name in generating:
            fail(E_CONFIG_INVALID, f"cyclic redundant column dependency at {name}")
        generating.add(name)
        column = columns_spec[name]
        kind = str(column.get("kind", "numeric"))
        variant = int(column.get("variant", 0))
        rng = child_rng(seed, "column", name, variant)
        missing_rate = float(column.get("missing_rate", 0.0))
        if kind == "numeric":
            weights = _pad_weights(
                [float(item) for item in column.get("latent_weights", [1.0, 0.0])], latent_dim
            )
            signal = latents @ weights
            noise = rng.normal(0.0, float(column.get("noise_scale", 0.5)), rows)
            values = float(column.get("mean", 0.0)) + float(column.get("scale", 1.0)) * (signal + noise)
            values = np.asarray(values, dtype=np.float64)
            if missing_rate > 0.0:
                missing = rng.random(rows) < missing_rate
                values[missing] = np.nan
        elif kind == "categorical":
            weights = _pad_weights(
                [float(item) for item in column.get("latent_weights", [1.0, 0.0])], latent_dim
            )
            signal = latents @ weights
            noise = rng.normal(0.0, float(column.get("noise_scale", 0.25)), rows)
            scores = signal + noise
            categories = [str(item) for item in column["categories"]]
            order = np.argsort(scores, kind="mergesort")
            labels = np.empty(rows, dtype=object)
            for rank, row_index in enumerate(order):
                labels[row_index] = categories[min(len(categories) - 1, rank * len(categories) // rows)]
            if missing_rate > 0.0:
                missing = rng.random(rows) < missing_rate
                labels[missing] = None
            values = labels
        elif kind == "redundant":
            source = str(column["source"])
            source_values = generate_column(source)
            if columns_spec[source].get("kind", "numeric") == "categorical":
                fail(E_CONFIG_INVALID, "redundant columns cannot use a categorical source", column=name)
            numeric_source = source_values.astype(np.float64, copy=True)
            values = (
                numeric_source * float(column.get("multiplier", 1.0))
                + float(column.get("offset", 0.0))
                + rng.normal(0.0, float(column.get("noise_scale", 0.0)), rows)
            )
            if missing_rate > 0.0:
                missing = rng.random(rows) < missing_rate
                values[missing] = np.nan
        else:
            fail(E_CONFIG_INVALID, f"unknown column kind: {kind}", column=name)
        generating.remove(name)
        raw_columns[name] = values
        column_kinds[name] = kind
        return values

    for column_name in column_order:
        generate_column(column_name)
    raw_columns = {name: raw_columns[name] for name in column_order}

    target_spec = spec["target"]
    target_name = str(target_spec.get("name", "target"))
    target_weights = _pad_weights(
        [float(item) for item in target_spec.get("latent_weights", [1.0, -0.5])], latent_dim
    )
    target_rng = child_rng(seed, "target")
    target_noise = target_rng.normal(0.0, float(target_spec.get("noise_scale", 0.8)), rows)
    target_score = latents @ target_weights + float(target_spec.get("offset", 0.0)) + target_noise
    target = (target_score > 0.0).astype(np.float64)
    if np.unique(target).size < 2:
        fail(E_CONFIG_INVALID, "generated binary target contains only one class; adjust target parameters")

    split_spec = spec.get("split", {})
    train_fraction = float(split_spec.get("train_fraction", 0.7))
    validation_fraction = float(split_spec.get("validation_fraction", 0.3))
    train_count = max(1, int(rows * train_fraction))
    validation_count = max(1, int(rows * validation_fraction))
    if train_count + validation_count > rows:
        validation_count = max(1, rows - train_count)
    if train_count + validation_count > rows:
        train_count = max(1, rows - 1)
        validation_count = 1
    split_rng = child_rng(seed, "split")
    permutation = split_rng.permutation(rows)
    train_indices = np.sort(permutation[:train_count]).astype(np.int64)
    validation_indices = np.sort(permutation[train_count : train_count + validation_count]).astype(np.int64)
    holdout_indices = np.sort(permutation[train_count + validation_count :]).astype(np.int64)

    numeric_fill_means: dict[str, float] = {}
    categorical_fill_modes: dict[str, str] = {}
    filled_columns: dict[str, np.ndarray] = {}
    for name in column_order:
        values = raw_columns[name]
        kind = column_kinds[name]
        if kind in {"numeric", "redundant"}:
            train_values = values[train_indices].astype(np.float64)
            finite = train_values[~np.isnan(train_values)]
            if finite.size == 0:
                fail(E_CONFIG_INVALID, "numeric column has no observed training values", column=name)
            mean = float(np.mean(finite, dtype=np.float64))
            numeric_fill_means[name] = mean
            filled = values.astype(np.float64, copy=True)
            filled[np.isnan(filled)] = mean
            filled_columns[name] = filled
        else:
            categories = [str(item) for item in columns_spec[name]["categories"]]
            observed = [str(value) for value in values[train_indices] if value is not None]
            if not observed:
                fail(E_CONFIG_INVALID, "categorical column has no observed training values", column=name)
            mode = _mode(observed, categories)
            categorical_fill_modes[name] = mode
            filled = np.asarray([mode if value is None else str(value) for value in values], dtype=object)
            filled_columns[name] = filled

    feature_names: list[str] = []
    feature_to_raw: dict[str, str] = {}
    raw_to_features: dict[str, list[str]] = {}
    feature_sources: list[tuple[str, str | None]] = []
    for name in column_order:
        kind = column_kinds[name]
        if kind in {"numeric", "redundant"}:
            feature_names.append(name)
            feature_to_raw[name] = name
            raw_to_features.setdefault(name, []).append(name)
            feature_sources.append((name, None))
        else:
            categories = [str(item) for item in columns_spec[name]["categories"]]
            drop_first = bool(columns_spec[name].get("drop_first", True))
            used_categories = categories[1:] if drop_first and len(categories) > 1 else categories
            for category in used_categories:
                feature = f"{name}={category}"
                feature_names.append(feature)
                feature_to_raw[feature] = name
                raw_to_features.setdefault(name, []).append(feature)
                feature_sources.append((name, category))

    matrix = np.empty((rows, len(feature_names)), dtype=np.float64)
    for feature_index, (column_name, category) in enumerate(feature_sources):
        if category is None:
            matrix[:, feature_index] = filled_columns[column_name]
        else:
            matrix[:, feature_index] = (filled_columns[column_name] == category).astype(np.float64)

    feature_means = np.zeros(len(feature_names), dtype=np.float64)
    feature_scales = np.ones(len(feature_names), dtype=np.float64)
    for feature_index, (column_name, category) in enumerate(feature_sources):
        if category is None:
            train_values = matrix[train_indices, feature_index]
            mean = float(np.mean(train_values, dtype=np.float64))
            scale = float(np.std(train_values, dtype=np.float64))
            if scale < 1e-12:
                scale = 1.0
            feature_means[feature_index] = mean
            feature_scales[feature_index] = scale
    standardized = (matrix - feature_means.reshape(1, -1)) / feature_scales.reshape(1, -1)
    # Keep signed zero out of fingerprints and JSON artifacts.
    standardized[standardized == 0.0] = 0.0
    matrix[matrix == 0.0] = 0.0

    column_fingerprints = {
        name: content_hash("column", {"kind": column_kinds[name], "values": _array_payload(raw_columns[name])})
        for name in column_order
    }
    target_fingerprint = content_hash("target", {target_name: _array_payload(target)})
    split_fingerprint = content_hash(
        "split",
        {
            "train": train_indices.tolist(),
            "validation": validation_indices.tolist(),
            "holdout": holdout_indices.tolist(),
        },
    )
    schema_fingerprint = content_hash(
        "schema",
        {
            "columns": [
                {
                    "name": name,
                    "kind": column_kinds[name],
                    "categories": columns_spec[name].get("categories"),
                    "drop_first": columns_spec[name].get("drop_first", True),
                }
                for name in column_order
            ],
            "features": [
                {"name": feature_names[index], "raw": feature_to_raw[feature_names[index]], "index": index}
                for index in range(len(feature_names))
            ],
            "target": target_name,
        },
    )
    data_fingerprint = content_hash(
        "data",
        {
            "seed": seed,
            "rows": rows,
            "columns": column_fingerprints,
            "target": target_fingerprint,
            "split": split_fingerprint,
            "schema": schema_fingerprint,
        },
    )
    train_sample_fingerprint = content_hash(
        "samples", {"train": train_indices.tolist(), "y": _array_payload(target[train_indices])}
    )
    validation_sample_fingerprint = content_hash(
        "samples", {"validation": validation_indices.tolist(), "y": _array_payload(target[validation_indices])}
    )

    return DatasetBundle(
        target_name=target_name,
        column_names=column_order,
        column_kinds=column_kinds,
        raw_columns=raw_columns,
        target=target,
        train_indices=train_indices,
        validation_indices=validation_indices,
        holdout_indices=holdout_indices,
        feature_names=feature_names,
        feature_to_raw=feature_to_raw,
        raw_to_features=raw_to_features,
        feature_indices={name: index for index, name in enumerate(feature_names)},
        matrix_encoded=matrix,
        matrix_standardized=standardized,
        feature_means=feature_means,
        feature_scales=feature_scales,
        numeric_fill_means=numeric_fill_means,
        categorical_fill_modes=categorical_fill_modes,
        column_fingerprints=column_fingerprints,
        target_fingerprint=target_fingerprint,
        split_fingerprint=split_fingerprint,
        schema_fingerprint=schema_fingerprint,
        data_fingerprint=data_fingerprint,
        train_sample_fingerprint=train_sample_fingerprint,
        validation_sample_fingerprint=validation_sample_fingerprint,
    )


def dataset_summary(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "target": bundle.target_name,
        "rows": int(bundle.target.size),
        "train_rows": int(bundle.train_indices.size),
        "validation_rows": int(bundle.validation_indices.size),
        "holdout_rows": int(bundle.holdout_indices.size),
        "train_indices": bundle.train_indices.tolist(),
        "validation_indices": bundle.validation_indices.tolist(),
        "holdout_indices": bundle.holdout_indices.tolist(),
        "columns": [
            {
                "name": name,
                "kind": bundle.column_kinds[name],
                "fingerprint": bundle.column_fingerprints[name],
                "numeric_fill_mean": bundle.numeric_fill_means.get(name),
                "categorical_fill_mode": bundle.categorical_fill_modes.get(name),
            }
            for name in bundle.column_names
        ],
        "features": [
            {
                "index": index,
                "name": name,
                "raw_column": bundle.feature_to_raw[name],
                "mean": float(bundle.feature_means[index]),
                "scale": float(bundle.feature_scales[index]),
            }
            for index, name in enumerate(bundle.feature_names)
        ],
        "target_fingerprint": bundle.target_fingerprint,
        "split_fingerprint": bundle.split_fingerprint,
        "schema_fingerprint": bundle.schema_fingerprint,
        "data_fingerprint": bundle.data_fingerprint,
        "train_sample_fingerprint": bundle.train_sample_fingerprint,
        "validation_sample_fingerprint": bundle.validation_sample_fingerprint,
    }


def write_dataset_csv(bundle: DatasetBundle, path: str) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(bundle.column_names + [bundle.target_name])
        for row_index in range(bundle.target.size):
            row: list[Any] = []
            for name in bundle.column_names:
                value = bundle.raw_columns[name][row_index]
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    row.append("")
                else:
                    row.append(value)
            row.append(int(bundle.target[row_index]))
            writer.writerow(row)
