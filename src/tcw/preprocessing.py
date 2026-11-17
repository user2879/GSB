"""Deterministic imputation-aware feature preprocessing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from .data import PreparedData, prepared_summary
from .errors import E_INPUT_NON_FINITE, BusinessError
from .util import sha256_value


@dataclass
class PreprocessResult:
    matrix: list[list[float]]
    feature_names: list[str]
    method: str
    params: dict[str, Any]
    stats: dict[str, Any]
    fingerprint: str
    input_hash: str
    output_hash: str
    source_data_hash: str
    rows: int
    columns: int
    train_count: int
    data_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix,
            "feature_names": self.feature_names,
            "method": self.method,
            "params": self.params,
            "stats": self.stats,
            "fingerprint": self.fingerprint,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "source_data_hash": self.source_data_hash,
            "rows": self.rows,
            "columns": self.columns,
            "train_count": self.train_count,
            "data_summary": self.data_summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessResult":
        return cls(**payload)


def _select_columns(prepared: PreparedData, include_columns: list[str]) -> tuple[list[list[float]], list[str]]:
    matrix: list[list[float]] = []
    selected_names: list[str] = []
    index_map = {name: index for index, name in enumerate(prepared.feature_names)}
    blocks: dict[str, list[str]] = {}

    for row in prepared.matrix:
        selected_row: list[float] = []
        for column in include_columns:
            if column in prepared.feature_blocks:
                block = prepared.feature_blocks[column]
            elif column in index_map:
                block = [column]
            else:
                raise BusinessError(
                    E_INPUT_NON_FINITE,
                    f"prepared features do not contain configured column: {column}",
                    stage="preprocess",
                    path=column,
                )
            for name in block:
                selected_row.append(float(row[index_map[name]]))
        matrix.append(selected_row)

    for column in include_columns:
        blocks[column] = (
            prepared.feature_blocks[column]
            if column in prepared.feature_blocks
            else [column]
        )
        selected_names.extend(blocks[column])
    return matrix, selected_names


def _finite_matrix(prepared: PreparedData) -> None:
    for row_index, row in enumerate(prepared.matrix):
        for column_index, value in enumerate(row):
            if not math.isfinite(float(value)):
                raise BusinessError(
                    E_INPUT_NON_FINITE,
                    f"non-finite encoded value at sample {row_index}, feature {column_index}",
                    stage="preprocess",
                    path=f"matrix[{row_index}][{column_index}]",
                    extra={"sample": row_index, "feature_index": column_index},
                )


def _stable_quantile(ordered: list[float], probability: float) -> float:
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _standardize_stats(train: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {"kind": "standardize", "features": {}}
    for index, name in enumerate(feature_names):
        values = [float(value) for value in train[:, index]]
        mean = math.fsum(values) / len(values)
        squared = [((float(value) - mean) ** 2) for value in values]
        variance = math.fsum(squared) / len(values)
        scale = math.sqrt(max(variance, 0.0))
        constant = scale <= 1e-15
        if constant:
            scale = 1.0
        stats["features"][name] = {
            "count": len(values),
            "mean": mean,
            "variance": variance,
            "scale": scale,
            "constant": constant,
            "accumulation_order": "training_row_ascending",
        }
    return stats


def _quantile_stats(
    train: np.ndarray,
    feature_names: list[str],
    n_quantiles: int,
    output_distribution: str,
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "kind": "quantile",
        "n_quantiles": n_quantiles,
        "output_distribution": output_distribution,
        "features": {},
    }
    effective_quantiles = max(2, min(n_quantiles, train.shape[0]))
    references = np.linspace(0.0, 1.0, effective_quantiles).tolist()
    for index, name in enumerate(feature_names):
        ordered = sorted(float(value) for value in train[:, index])
        quantile_values = [_stable_quantile(ordered, float(q)) for q in references]
        constant = quantile_values[0] == quantile_values[-1]
        stats["features"][name] = {
            "count": len(ordered),
            "references": references,
            "quantile_values": quantile_values,
            "constant": constant,
            "accumulation_order": "sorted_value_ascending",
        }
    return stats


def _transform_value(value: float, feature_stats: dict[str, Any], method: str, output_distribution: str) -> float:
    if method == "standardize":
        return (value - feature_stats["mean"]) / feature_stats["scale"]
    references = np.asarray(feature_stats["references"], dtype=float)
    quantile_values = np.asarray(feature_stats["quantile_values"], dtype=float)
    if feature_stats["constant"] or float(quantile_values[-1] - quantile_values[0]) <= 1e-15:
        transformed = 0.5
    else:
        transformed = float(np.interp(value, quantile_values, references, left=references[0], right=references[-1]))
    if output_distribution == "normal":
        clipped = min(max(transformed, 1e-6), 1.0 - 1e-6)
        return NormalDist().inv_cdf(clipped)
    return transformed


def fit_transform(prepared: PreparedData, preprocessing: dict[str, Any]) -> PreprocessResult:
    _finite_matrix(prepared)
    matrix, feature_names = _select_columns(prepared, preprocessing["include_columns"])
    array = np.asarray(matrix, dtype=float)
    train_count = max(1, min(prepared.train_count, array.shape[0]))
    train = array[:train_count]
    method = preprocessing["method"]
    quantile_params = preprocessing.get("quantile", {})
    output_distribution = quantile_params.get("output_distribution", "uniform")

    if method == "standardize":
        stats = _standardize_stats(train, feature_names)
        for row_index in range(array.shape[0]):
            for column_index, name in enumerate(feature_names):
                feature_stats = stats["features"][name]
                array[row_index, column_index] = (
                    array[row_index, column_index] - feature_stats["mean"]
                ) / feature_stats["scale"]
    elif method == "quantile":
        stats = _quantile_stats(
            train,
            feature_names,
            int(quantile_params.get("n_quantiles", 32)),
            output_distribution,
        )
        for row_index in range(array.shape[0]):
            for column_index, name in enumerate(feature_names):
                array[row_index, column_index] = _transform_value(
                    float(array[row_index, column_index]),
                    stats["features"][name],
                    method,
                    output_distribution,
                )
    else:
        raise BusinessError(
            E_INPUT_NON_FINITE,
            f"unsupported preprocessing method: {method!r}",
            stage="preprocess",
            path="preprocessing.method",
        )

    if not np.all(np.isfinite(array)):
        raise BusinessError(
            E_INPUT_NON_FINITE,
            "preprocessing produced a non-finite feature value",
            stage="preprocess",
            path="preprocessing.output",
        )

    params = {
        "method": method,
        "fit_fraction": preprocessing["fit_fraction"],
        "include_columns": preprocessing["include_columns"],
        "quantile": quantile_params,
    }
    input_hash = sha256_value(
        {
            "data_hash": prepared.data_hash,
            "include_columns": preprocessing["include_columns"],
            "matrix": matrix,
        }
    )
    fingerprint = sha256_value(
        {
            "column_descriptions": preprocessing["column_descriptions"],
            "parameters": params,
            "fill_statistics": {
                "numeric": prepared.numeric_fill_values,
                "categorical": prepared.categorical_fill_values,
                "train_count": prepared.train_count,
            },
            "transform_statistics": stats,
            "selected_features": feature_names,
            "input_hash": input_hash,
        }
    )
    output_matrix = [[float(value) for value in row] for row in array]
    output_hash = sha256_value({"fingerprint": fingerprint, "matrix": output_matrix})
    return PreprocessResult(
        matrix=output_matrix,
        feature_names=feature_names,
        method=method,
        params=params,
        stats=stats,
        fingerprint=fingerprint,
        input_hash=input_hash,
        output_hash=output_hash,
        source_data_hash=prepared.data_hash,
        rows=array.shape[0],
        columns=array.shape[1],
        train_count=train_count,
        data_summary=prepared_summary(prepared),
    )


def verify_preprocess_result(result: PreprocessResult) -> bool:
    output_hash = sha256_value({"fingerprint": result.fingerprint, "matrix": result.matrix})
    return output_hash == result.output_hash
