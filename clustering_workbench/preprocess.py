"""Deterministic missing-value handling, encoding, scaling, and fingerprinting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .codes import Failure
from .data import GeneratedDataset
from .utils import hash_array, hash_object, json_safe


PREPROCESS_VERSION = "preprocess-v1"


def preprocess_artifact_fingerprint(
    content_fingerprint: str,
    X: np.ndarray,
    feature_names: list[str],
    statistics: dict[str, Any],
    dropped_columns: list[dict[str, Any]],
    train_mask: np.ndarray | list[bool],
) -> str:
    mask = np.asarray(train_mask, dtype=np.float64)
    column_statistics = {
        name: value for name, value in statistics.items() if name != "_pipeline"
    }
    return hash_object(
        "preart",
        {
            "content_fingerprint": content_fingerprint,
            "matrix_hash": hash_array(X),
            "feature_names": feature_names,
            "statistics": column_statistics,
            "dropped_columns": dropped_columns,
            "train_mask_hash": hash_array(mask),
        },
    )


@dataclass
class PreprocessResult:
    X: np.ndarray
    sample_ids: list[str]
    feature_names: list[str]
    column_types: dict[str, str]
    category_mappings: dict[str, dict[str, int]]
    statistics: dict[str, Any]
    dropped_columns: list[dict[str, Any]]
    train_row_count: int
    train_mask: list[bool]
    method: str
    parameter_fingerprint: str
    content_fingerprint: str
    artifact_fingerprint: str
    shape: tuple[int, int]

    def preview(self, limit: int = 20) -> dict[str, Any]:
        count = min(limit, int(self.X.shape[0]))
        return {
            "sample_ids": self.sample_ids[:count],
            "feature_names": list(self.feature_names),
            "rows": json_safe(self.X[:count]),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "shape": list(self.shape),
            "sample_ids": list(self.sample_ids),
            "feature_names": list(self.feature_names),
            "column_types": dict(self.column_types),
            "category_mappings": self.category_mappings,
            "statistics": self.statistics,
            "dropped_columns": self.dropped_columns,
            "train_row_count": self.train_row_count,
            "train_mask": self.train_mask,
            "parameter_fingerprint": self.parameter_fingerprint,
            "content_fingerprint": self.content_fingerprint,
            "artifact_fingerprint": self.artifact_fingerprint,
        }


def _selected_columns(dataset: GeneratedDataset, config: dict[str, Any]) -> list[str]:
    configured = config.get("data", {}).get("include_columns") or []
    if configured:
        return [str(value) for value in configured]
    targets = set(dataset.target_columns)
    return [name for name in dataset.column_types if name not in targets]


def preprocess_identity(
    dataset: GeneratedDataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    preprocess_config = config.get("preprocessing", {})
    method = str(preprocess_config.get("method", "standard"))
    if method not in {"standard", "quantile"}:
        method = "standard"
    train_fraction = min(
        1.0,
        max(0.01, float(preprocess_config.get("train_fraction", 0.7))),
    )
    selected = _selected_columns(dataset, config)
    parameters = {
        "version": PREPROCESS_VERSION,
        "method": method,
        "train_fraction": train_fraction,
        "drop_redundant": bool(preprocess_config.get("drop_redundant", True)),
        "epsilon": float(preprocess_config.get("epsilon", 1e-9)),
        "quantile_points": int(preprocess_config.get("quantile_points", 65)),
        "selected_columns": selected,
    }
    parameter_fingerprint = hash_object("preparam", parameters)
    content_fingerprint = hash_object(
        "pre",
        {
            "column_description_fingerprint": dataset.descriptor_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
        },
    )
    return {
        "method": method,
        "parameters": parameters,
        "parameter_fingerprint": parameter_fingerprint,
        "content_fingerprint": content_fingerprint,
        "selected_columns": selected,
    }


def validate_preprocess_columns(
    dataset: GeneratedDataset,
    config: dict[str, Any],
    failures: list[Failure],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, int]]]:
    accepted, arrays, mappings = _prepare_columns(dataset, config, failures)
    if accepted:
        _detect_non_finite(dataset, accepted, arrays, failures)
    return accepted, arrays, mappings


def _prepare_columns(
    dataset: GeneratedDataset,
    config: dict[str, Any],
    failures: list[Failure],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, dict[str, int]]]:
    selected = _selected_columns(dataset, config)
    expected_types = config.get("data", {}).get("expected_types", {}) or {}
    for column in sorted(expected_types):
        if column not in dataset.column_types:
            failures.append(
                Failure(
                    code="CFG_UNKNOWN_COLUMN",
                    stage="load",
                    item=column,
                    message=f"Configured expected_types references unknown column '{column}'.",
                    details={"column": column},
                )
            )

    arrays: dict[str, np.ndarray] = {}
    mappings: dict[str, dict[str, int]] = {}
    accepted: list[str] = []
    for column in selected:
        if column not in dataset.columns:
            failures.append(
                Failure(
                    code="CFG_UNKNOWN_COLUMN",
                    stage="load",
                    item=column,
                    message=f"Configured feature column '{column}' does not exist; other columns continue.",
                    details={"column": column},
                )
            )
            continue
        actual_type = dataset.column_types[column]
        expected_type = expected_types.get(column)
        if expected_type is not None and str(expected_type).lower() != actual_type:
            failures.append(
                Failure(
                    code="CFG_TYPE_MISMATCH",
                    stage="load",
                    item=column,
                    message=(
                        f"Column '{column}' is {actual_type} but expected {expected_type}; "
                        "the column is skipped and other columns continue."
                    ),
                    details={"column": column, "actual_type": actual_type, "expected_type": expected_type},
                )
            )
            continue
        if column in dataset.target_columns:
            continue

        values = dataset.columns[column]
        if actual_type == "categorical":
            non_missing = sorted({str(value) for value in values if value is not None})
            mapping = {value: index for index, value in enumerate(non_missing)}
            encoded = np.asarray(
                [np.nan if value is None else float(mapping[str(value)]) for value in values],
                dtype=np.float64,
            )
            mappings[column] = mapping
        else:
            converted = []
            for value in values:
                if value is None:
                    converted.append(np.nan)
                else:
                    try:
                        converted.append(float(value))
                    except (TypeError, ValueError):
                        converted.append(np.nan)
            encoded = np.asarray(converted, dtype=np.float64)
        arrays[column] = encoded
        accepted.append(column)
    return accepted, arrays, mappings


def _detect_non_finite(
    dataset: GeneratedDataset,
    accepted: list[str],
    arrays: dict[str, np.ndarray],
    failures: list[Failure],
) -> None:
    for column in accepted:
        non_finite = ~np.isfinite(arrays[column])
        original_missing = np.asarray([value is None for value in dataset.columns[column]], dtype=bool)
        injected = non_finite & ~original_missing
        if np.any(injected):
            indices = np.flatnonzero(injected)
            locations = [
                {"sample_id": dataset.sample_ids[int(index)], "row_index": int(index)}
                for index in indices[:20]
            ]
            failures.append(
                Failure(
                    code="DATA_NON_FINITE_VALUE",
                    stage="preprocess",
                    item=column,
                    message=f"Column '{column}' contained non-finite values; they were imputed in isolation.",
                    details={
                        "column": column,
                        "count": int(indices.size),
                        "locations": locations,
                        "locations_truncated": int(indices.size) > 20,
                    },
                )
            )


def _fit_and_transform(
    X_raw: np.ndarray,
    feature_names: list[str],
    train_mask: np.ndarray,
    method: str,
    epsilon: float,
    quantile_points: int,
    column_types: dict[str, str],
    mappings: dict[str, dict[str, int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    n_rows, n_features = X_raw.shape
    X_filled = X_raw.copy()
    statistics: dict[str, Any] = {}

    for column_index, name in enumerate(feature_names):
        training = X_filled[train_mask, column_index]
        finite_training = training[np.isfinite(training)]
        if finite_training.size:
            fill_value = float(np.median(finite_training))
        else:
            fill_value = 0.0
        column_values = X_filled[:, column_index]
        missing = ~np.isfinite(column_values)
        filled_count = int(np.count_nonzero(missing))
        column_values[missing] = fill_value
        X_filled[:, column_index] = column_values
        training = X_filled[train_mask, column_index]
        stats: dict[str, Any] = {
            "type": column_types.get(name, "numeric"),
            "fill_value": fill_value,
            "training_missing_filled": filled_count,
            "accumulation_order": "ascending row index",
        }
        if name in mappings:
            stats["category_mapping"] = mappings[name]
        if method == "standard":
            mean = float(np.sum(training, dtype=np.float64) / training.size)
            centered = training - mean
            variance = float(np.sum(centered * centered, dtype=np.float64) / training.size)
            std = math.sqrt(max(variance, 0.0))
            scale = max(std, epsilon)
            stats.update({"center": mean, "scale": scale, "variance": variance})
        else:
            q_count = max(2, min(int(quantile_points), int(training.size)))
            probabilities = np.linspace(0.0, 1.0, q_count, dtype=np.float64)
            reference_values = np.quantile(training, probabilities, method="linear")
            if float(reference_values[-1] - reference_values[0]) <= epsilon:
                reference_values = np.full_like(reference_values, float(reference_values[0]))
            stats.update(
                {
                    "quantile_probabilities": probabilities.tolist(),
                    "quantile_reference_values": reference_values.tolist(),
                    "output_domain": [0.0, 1.0],
                }
            )
        statistics[name] = stats

    X_scaled = np.empty_like(X_filled, dtype=np.float64)
    for column_index, name in enumerate(feature_names):
        values = X_filled[:, column_index]
        stats = statistics[name]
        if method == "standard":
            X_scaled[:, column_index] = (values - float(stats["center"])) / float(stats["scale"])
        else:
            probabilities = np.asarray(stats["quantile_probabilities"], dtype=np.float64)
            references = np.asarray(stats["quantile_reference_values"], dtype=np.float64)
            if float(references[-1] - references[0]) <= epsilon:
                X_scaled[:, column_index] = np.full(values.shape, 0.5, dtype=np.float64)
            else:
                # np.interp requires the abscissa to be increasing. Constant
                # reference values are handled above; ties elsewhere are harmless.
                unique_references, unique_indices = np.unique(references, return_index=True)
                unique_probabilities = probabilities[unique_indices]
                X_scaled[:, column_index] = np.interp(
                    values,
                    unique_references,
                    unique_probabilities,
                    left=float(unique_probabilities[0]),
                    right=float(unique_probabilities[-1]),
                )
    return X_scaled, statistics


def preprocess_dataset(
    dataset: GeneratedDataset,
    config: dict[str, Any],
    failures: list[Failure],
) -> PreprocessResult | None:
    preprocess_config = config.get("preprocessing", {})
    identity = preprocess_identity(dataset, config)
    method = str(identity["method"])
    accepted, arrays, mappings = _prepare_columns(dataset, config, failures)
    if not accepted:
        failures.append(
            Failure(
                code="PREPROCESS_NO_FEATURES",
                stage="preprocess",
                item="features",
                message="No usable feature columns remain after configuration and type validation.",
            )
        )
        return None

    _detect_non_finite(dataset, accepted, arrays, failures)
    X_raw = np.column_stack([arrays[name] for name in accepted])
    n_rows = X_raw.shape[0]
    train_fraction = float(preprocess_config.get("train_fraction", 0.7))
    train_fraction = min(1.0, max(0.01, train_fraction))
    train_count = max(1, min(n_rows, int(round(n_rows * train_fraction))))
    train_mask = np.zeros(n_rows, dtype=bool)
    train_mask[:train_count] = True

    X_filled_for_redundancy = X_raw.copy()
    for column_index in range(X_filled_for_redundancy.shape[1]):
        training = X_filled_for_redundancy[train_mask, column_index]
        finite_training = training[np.isfinite(training)]
        fill_value = float(np.median(finite_training)) if finite_training.size else 0.0
        column_values = X_filled_for_redundancy[:, column_index]
        column_values[~np.isfinite(column_values)] = fill_value
        X_filled_for_redundancy[:, column_index] = column_values

    dropped_columns: list[dict[str, Any]] = []
    kept: list[str] = []
    if bool(preprocess_config.get("drop_redundant", True)):
        for index, name in enumerate(accepted):
            duplicate_of = None
            for kept_index, kept_name in enumerate(kept):
                original_index = accepted.index(kept_name)
                if np.array_equal(
                    X_filled_for_redundancy[train_mask, index],
                    X_filled_for_redundancy[train_mask, original_index],
                ):
                    duplicate_of = kept_name
                    break
            if duplicate_of is None:
                kept.append(name)
            else:
                dropped_columns.append(
                    {
                        "column": name,
                        "duplicate_of": duplicate_of,
                        "reason": "exact_duplicate_on_training_rows",
                    }
                )
    else:
        kept = list(accepted)

    kept_indices = [accepted.index(name) for name in kept]
    X_kept = X_raw[:, kept_indices]
    kept_mappings = {name: mappings[name] for name in kept if name in mappings}
    X, statistics = _fit_and_transform(
        X_kept,
        kept,
        train_mask,
        method,
        float(preprocess_config.get("epsilon", 1e-9)),
        int(preprocess_config.get("quantile_points", 65)),
        dataset.column_types,
        kept_mappings,
    )
    parameter_fingerprint = str(identity["parameter_fingerprint"])
    content_fingerprint = str(identity["content_fingerprint"])
    artifact_fingerprint = preprocess_artifact_fingerprint(
        content_fingerprint,
        X,
        kept,
        statistics,
        dropped_columns,
        train_mask,
    )
    column_types = {name: dataset.column_types[name] for name in kept}
    statistics["_pipeline"] = {
        "selected_columns": accepted,
        "kept_columns": kept,
        "dropped_columns": dropped_columns,
        "train_fraction": train_fraction,
        "train_row_count": train_count,
        "method": method,
    }
    return PreprocessResult(
        X=X,
        sample_ids=list(dataset.sample_ids),
        feature_names=kept,
        column_types=column_types,
        category_mappings=kept_mappings,
        statistics=statistics,
        dropped_columns=dropped_columns,
        train_row_count=train_count,
        train_mask=train_mask.tolist(),
        method=method,
        parameter_fingerprint=parameter_fingerprint,
        content_fingerprint=content_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        shape=(int(X.shape[0]), int(X.shape[1])),
    )
