from __future__ import annotations

import bisect
import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Table, build_feature_matrix
from .errors import (
    E_ALL_MISSING_TRAIN,
    E_NONFINITE_FEATURE,
    E_UNKNOWN_CATEGORY,
    Issue,
)
from .util import hash_array, stable_hash, to_jsonable


@dataclass
class PreprocessPlan:
    cache_key: str
    columns: list[str]
    train_fraction: float
    train_indices: list[int]
    params_fingerprint: str
    column_description_fingerprint: str
    source_content_fingerprint: str
    column_fingerprints: dict[str, str]


@dataclass
class PreprocessResult:
    matrix: np.ndarray
    metadata: dict[str, Any]
    issues: list[Issue]

    @property
    def content_fingerprint(self) -> str:
        return str(self.metadata["content_fingerprint"])

    @property
    def matrix_fingerprint(self) -> str:
        return str(self.metadata["matrix_fingerprint"])


def _train_indices(rows: int, fraction: float, data_fingerprint: str, columns: list[str]) -> list[int]:
    if rows <= 0:
        return []
    count = max(1, min(rows, int(math.ceil(rows * fraction))))
    if count == rows:
        return list(range(rows))
    digest = hashlib.sha256(
        (data_fingerprint + "|" + ",".join(columns) + f"|{fraction!r}").encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:16], "big", signed=False)
    permutation = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed))).permutation(rows)
    return sorted(int(index) for index in permutation[:count])


def make_preprocess_plan(
    table: Table,
    columns: list[str],
    *,
    preprocessing: dict[str, Any],
    train_fraction: float,
) -> PreprocessPlan:
    selected_descriptions = {
        column: table.column_description_fingerprints.get(column, "<missing>") for column in columns
    }
    selected_content = {
        column: table.column_content_fingerprints.get(column, "<missing>") for column in columns
    }
    description_fingerprint = stable_hash({"columns": columns, "descriptions": selected_descriptions})
    source_fingerprint = stable_hash({"columns": columns, "content": selected_content})
    train_indices = _train_indices(table.rows, train_fraction, source_fingerprint, columns)
    parameters = {
        "preprocessing": to_jsonable(preprocessing),
        "train_fraction": float(train_fraction),
        "train_indices": train_indices,
        "column_types": {column: table.column_specs.get(column, {}).get("type") for column in columns},
    }
    parameters_fingerprint = stable_hash(parameters)
    key = stable_hash(
        {
            "kind": "preprocess",
            "schema": 1,
            "column_description_fingerprint": description_fingerprint,
            "source_content_fingerprint": source_fingerprint,
            "parameters_fingerprint": parameters_fingerprint,
            "columns": columns,
        }
    )
    return PreprocessPlan(
        cache_key=key,
        columns=list(columns),
        train_fraction=float(train_fraction),
        train_indices=train_indices,
        params_fingerprint=parameters_fingerprint,
        column_description_fingerprint=description_fingerprint,
        source_content_fingerprint=source_fingerprint,
        column_fingerprints=selected_content,
    )


def fit_transform(
    table: Table,
    plan: PreprocessPlan,
    *,
    preprocessing: dict[str, Any],
    candidate_id: str,
) -> PreprocessResult:
    issues: list[Issue] = []
    columns = plan.columns
    raw_matrix, _, unknown_counts = build_feature_matrix(table, columns)

    for column, count in unknown_counts.items():
        issues.append(
            Issue(
                E_UNKNOWN_CATEGORY,
                f"{count} values in {column!r} are outside the declared category set",
                severity="warning",
                context={"scope": "preprocess", "candidate": candidate_id, "column": column, "count": count},
            )
        )

    nonfinite_mask = ~np.isfinite(raw_matrix) & ~np.isnan(raw_matrix)
    if np.any(nonfinite_mask):
        row_index, column_index = (int(value) for value in np.argwhere(nonfinite_mask)[0])
        column = columns[column_index]
        issues.append(
            Issue(
                E_NONFINITE_FEATURE,
                f"non-finite feature in column {column!r}",
                context={
                    "scope": "preprocess",
                    "candidate": candidate_id,
                    "column": column,
                    "row": row_index,
                },
            )
        )
        return PreprocessResult(
            matrix=np.empty((0, 0), dtype=np.float64),
            metadata={
                "failed": True,
                "feature_columns": columns,
                "quantile_tie_policy": preprocessing.get("quantile_tie_policy", "average_rank"),
                "content_fingerprint": stable_hash(
                    {"failed": E_NONFINITE_FEATURE, "column": column, "row": row_index, "plan": plan.cache_key}
                ),
            },
            issues=issues,
        )

    train_indices = np.asarray(plan.train_indices, dtype=np.int64)
    if train_indices.size == 0:
        train_indices = np.arange(table.rows, dtype=np.int64)
    fill_values: dict[str, float] = {}
    missing_counts: dict[str, int] = {}
    for column_index, column in enumerate(columns):
        column_values = raw_matrix[:, column_index]
        missing_counts[column] = int(np.isnan(column_values).sum())
        train_values = column_values[train_indices]
        observed = np.asarray([value for value in train_values if not np.isnan(value)], dtype=np.float64)
        if observed.size == 0:
            fill = 0.0
            issues.append(
                Issue(
                    E_ALL_MISSING_TRAIN,
                    f"column {column!r} has no observed values in the training partition; fill=0",
                    severity="warning",
                    context={"scope": "preprocess", "candidate": candidate_id, "column": column},
                )
            )
        else:
            ordered = sorted(float(value) for value in observed)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                fill = ordered[middle]
            else:
                fill = (ordered[middle - 1] + ordered[middle]) / 2.0
        fill_values[column] = float(fill)
        raw_matrix[:, column_index] = np.where(np.isnan(column_values), fill, column_values)

    transformed = np.empty_like(raw_matrix, dtype=np.float64)
    metadata: dict[str, Any] = {
        "candidate": candidate_id,
        "method": preprocessing["method"],
        "feature_columns": columns,
        "column_types": {column: table.column_specs[column].get("type") for column in columns},
        "train_fraction": plan.train_fraction,
        "train_indices": plan.train_indices,
        "fill_values": fill_values,
        "missing_counts": missing_counts,
        "missing_fill_policy": "median_of_candidate_training_partition",
        "quantile_tie_policy": preprocessing.get("quantile_tie_policy", "average_rank"),
        "finite_accumulation_order": "ascending row index within candidate training partition",
        "column_description_fingerprint": plan.column_description_fingerprint,
        "source_content_fingerprint": plan.source_content_fingerprint,
        "parameters_fingerprint": plan.params_fingerprint,
        "statistics": {},
    }

    if preprocessing["method"] == "standardize":
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        variances: dict[str, float] = {}
        for column_index, column in enumerate(columns):
            train_values = raw_matrix[train_indices, column_index]
            total = 0.0
            for value in train_values:
                total += float(value)
            mean = total / float(train_values.size)
            squared_total = 0.0
            for value in train_values:
                delta = float(value) - mean
                squared_total += delta * delta
            variance = squared_total / float(train_values.size)
            scale = math.sqrt(max(0.0, variance))
            if scale <= 1e-15:
                scale = 1.0
            means[column] = mean
            variances[column] = variance
            scales[column] = scale
            transformed[:, column_index] = (raw_matrix[:, column_index] - mean) / scale
        metadata["statistics"] = {
            "means": means,
            "variances": variances,
            "scales": scales,
            "variance_denominator": "training_count",
        }
    elif preprocessing["method"] == "quantile":
        tie_policy = preprocessing.get("quantile_tie_policy", "average_rank")
        quantiles_metadata: dict[str, Any] = {}
        for column_index, column in enumerate(columns):
            train_values = np.asarray(raw_matrix[train_indices, column_index], dtype=np.float64)
            knots, mapped = _quantile_knots(train_values, tie_policy)
            quantiles_metadata[column] = {
                "training_values": [float(value) for value in knots],
                "quantiles": [float(value) for value in mapped],
                "tie_policy": tie_policy,
            }
            transformed[:, column_index] = _apply_quantile_transform(
                raw_matrix[:, column_index], knots, mapped
            )
        metadata["statistics"] = {"quantiles": quantiles_metadata}
    else:  # defensive: normalize_config replaces unsupported methods
        transformed = raw_matrix.copy()

    matrix_fingerprint = hash_array(transformed)
    content_metadata = {key: value for key, value in metadata.items() if key != "candidate"}
    content_payload = {
        "column_description_fingerprint": plan.column_description_fingerprint,
        "source_content_fingerprint": plan.source_content_fingerprint,
        "parameters_fingerprint": plan.params_fingerprint,
        "matrix_fingerprint": matrix_fingerprint,
        "metadata": content_metadata,
        "shape": list(transformed.shape),
    }
    metadata["matrix_fingerprint"] = matrix_fingerprint
    metadata["content_fingerprint"] = stable_hash(content_payload)
    metadata["cache_key"] = plan.cache_key
    return PreprocessResult(matrix=transformed, metadata=metadata, issues=issues)


def _quantile_knots(values: np.ndarray, tie_policy: str) -> tuple[list[float], list[float]]:
    ordered = sorted(float(value) for value in values if not np.isnan(value))
    if not ordered:
        return [0.0], [0.5]
    unique: list[float] = []
    positions: list[list[int]] = []
    for index, value in enumerate(ordered):
        if not unique or value != unique[-1]:
            unique.append(value)
            positions.append([index])
        else:
            positions[-1].append(index)
    denominator = max(1, len(ordered) - 1)
    mapped: list[float] = []
    for indexes in positions:
        if tie_policy == "lower_rank":
            rank = indexes[0]
        elif tie_policy == "upper_rank":
            rank = indexes[-1]
        else:
            rank = (indexes[0] + indexes[-1]) / 2.0
        mapped.append(rank / denominator)
    if len(unique) == 1:
        mapped = [0.5]
    return unique, mapped


def _apply_quantile_transform(
    values: np.ndarray, knots: list[float], quantiles: list[float]
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    if len(knots) == 1:
        result.fill(quantiles[0])
        return result
    for index, value in enumerate(values):
        numeric = float(value)
        if numeric <= knots[0]:
            result[index] = quantiles[0]
            continue
        if numeric >= knots[-1]:
            result[index] = quantiles[-1]
            continue
        position = bisect.bisect_left(knots, numeric)
        if position < len(knots) and knots[position] == numeric:
            result[index] = quantiles[position]
            continue
        left = position - 1
        right = position
        span = knots[right] - knots[left]
        ratio = 0.0 if span == 0 else (numeric - knots[left]) / span
        result[index] = quantiles[left] + ratio * (quantiles[right] - quantiles[left])
    return result
