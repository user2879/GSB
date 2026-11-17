"""Deterministic imputation, category encoding and feature scaling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .canonical import fsum, quantize_score, stable_median
from .config import PREPROCESS_VERSION
from .data import Dataset


@dataclass
class PreprocessedData:
    sample_ids: list[str]
    columns: list[str]
    matrix: np.ndarray
    numeric_columns: list[str]
    categorical_columns: list[str]
    fill_values: dict[str, Any]
    category_maps: dict[str, dict[str, float]]

    def to_payload(self) -> dict[str, Any]:
        return {
            "sample_ids": self.sample_ids,
            "columns": self.columns,
            "matrix": self.matrix,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "fill_values": self.fill_values,
            "category_maps": self.category_maps,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PreprocessedData":
        matrix = np.asarray(payload["matrix"], dtype=np.float64, order="C")
        return cls(
            sample_ids=list(payload["sample_ids"]),
            columns=list(payload["columns"]),
            matrix=matrix,
            numeric_columns=list(payload["numeric_columns"]),
            categorical_columns=list(payload["categorical_columns"]),
            fill_values=dict(payload["fill_values"]),
            category_maps={k: dict(v) for k, v in payload["category_maps"].items()},
        )


def _fill_numeric(values: list[Any], strategy: str) -> tuple[list[float], float]:
    finite = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if strategy == "zero":
        fill = 0.0
    elif strategy == "mean":
        fill = fsum(finite) / len(finite) if finite else 0.0
    elif strategy == "median":
        fill = stable_median(np.asarray(finite, dtype=np.float64)) if finite else 0.0
    else:
        raise ValueError(f"unknown numeric fill strategy: {strategy}")
    filled = [fill if value is None else float(value) for value in values]
    return filled, quantize_score(fill)


def _fill_categorical(values: list[Any], strategy: str, constant: str) -> tuple[list[str], str]:
    present = [str(x) for x in values if x is not None]
    if strategy == "constant":
        fill = constant
    elif strategy == "mode":
        if not present:
            fill = constant
        else:
            counts: dict[str, int] = {}
            for value in present:
                counts[value] = counts.get(value, 0) + 1
            fill = sorted(counts, key=lambda x: (-counts[x], x))[0]
    else:
        raise ValueError(f"unknown categorical fill strategy: {strategy}")
    return [fill if value is None else str(value) for value in values], fill


def _robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = stable_median(values)
    deviations = np.abs(values - median)
    mad = stable_median(deviations)
    scale = 1.4826 * mad
    if scale > 1.0e-12:
        return median, scale
    sorted_values = np.sort(values, kind="mergesort")
    if sorted_values.size >= 2:
        q1 = _quantile(sorted_values, 0.25)
        q3 = _quantile(sorted_values, 0.75)
        iqr_scale = (q3 - q1) / 1.349
        if iqr_scale > 1.0e-12:
            return median, float(iqr_scale)
    return median, 1.0


def _quantile(sorted_values: np.ndarray, q: float) -> float:
    if sorted_values.size == 0:
        return 0.0
    position = (sorted_values.size - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def preprocess_dataset(dataset: Dataset, config: dict[str, Any]) -> PreprocessedData:
    pre = config["preprocessing"]
    numeric_columns = list(pre["numeric_columns"])
    categorical_columns = list(pre["categorical_columns"])
    feature_columns = list(config["data"]["feature_columns"])
    if set(feature_columns) != set(numeric_columns) | set(categorical_columns):
        raise ValueError("preprocessing column sets do not cover configured feature columns")
    n = len(dataset.rows)
    d = len(feature_columns)
    raw_matrix = np.empty((n, d), dtype=np.float64, order="C")
    fill_values: dict[str, Any] = {}
    category_maps: dict[str, dict[str, float]] = {}
    column_lookup = {str(item["name"]): item for item in dataset.description["dataset"]["columns"]}

    for index, name in enumerate(feature_columns):
        values = [row[name] for row in dataset.rows]
        if name in numeric_columns:
            filled, fill = _fill_numeric(values, str(pre["numeric_fill"]))
            fill_values[name] = fill
            for row_index, value in enumerate(filled):
                raw_matrix[row_index, index] = value
        else:
            filled, fill = _fill_categorical(
                values,
                str(pre["categorical_fill"]),
                str(pre.get("constant_value", "__MISSING__")),
            )
            fill_values[name] = fill
            configured = column_lookup.get(name, {}).get("categories", [])
            observed = sorted(set(filled))
            categories = sorted(set(str(x) for x in configured) | set(observed) | {fill})
            mapping = {category: float(code) for code, category in enumerate(categories)}
            category_maps[name] = mapping
            for row_index, value in enumerate(filled):
                raw_matrix[row_index, index] = mapping[value]

    mode = str(pre["mode"])
    if mode == "raw":
        matrix = raw_matrix.copy(order="C")
    elif mode == "standardize":
        matrix = np.empty_like(raw_matrix, order="C")
        for j in range(d):
            column = raw_matrix[:, j]
            mean = fsum(column.tolist()) / n if n else 0.0
            if n > 1:
                variance = fsum(float((float(column[i]) - mean) ** 2) for i in range(n)) / (n - 1)
                scale = math.sqrt(variance) if variance > 0.0 else 1.0
            else:
                scale = 1.0
            for i in range(n):
                matrix[i, j] = (raw_matrix[i, j] - mean) / scale
    elif mode == "robust_standardize":
        matrix = np.empty_like(raw_matrix, order="C")
        for j in range(d):
            column = raw_matrix[:, j]
            center, scale = _robust_scale(column)
            for i in range(n):
                matrix[i, j] = (raw_matrix[i, j] - center) / scale
    else:
        raise ValueError(f"unknown preprocessing mode: {mode}")

    if not np.all(np.isfinite(matrix)):
        bad_row = int(np.argwhere(~np.isfinite(matrix))[0][0])
        raise ValueError(f"preprocessing produced a non-finite value for sample {dataset.sample_ids[bad_row]}")
    # Quantize transformed features as well as scores so cache and fresh runs are byte-stable.
    matrix = np.asarray([[quantize_score(float(x)) for x in row] for row in matrix], dtype=np.float64, order="C")
    return PreprocessedData(
        sample_ids=list(dataset.sample_ids),
        columns=feature_columns,
        matrix=matrix,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        fill_values=fill_values,
        category_maps=category_maps,
    )


def preprocessing_version() -> str:
    return PREPROCESS_VERSION
