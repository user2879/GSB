"""Standardization and deterministic PCA / random projection / variance filtering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fingerprint import array_fingerprint, fingerprint, stable_normal, stable_seed


def _dtype(value: str) -> np.dtype:
    if value not in {"float32", "float64"}:
        raise ValueError(f"unsupported precision: {value}")
    return np.dtype(value)


def standardize(design: dict[str, Any], precision: str) -> dict[str, Any]:
    dtype = _dtype(precision)
    matrix = np.asarray(design["matrix"], dtype=dtype)
    mean = np.mean(matrix, axis=0, dtype=dtype)
    std = np.std(matrix, axis=0, dtype=dtype, ddof=0)
    epsilon = np.asarray(np.finfo(dtype).eps, dtype=dtype)
    safe_std = np.where(std > epsilon, std, np.ones_like(std, dtype=dtype))
    standardized = ((matrix - mean) / safe_std).astype(dtype, copy=False)
    parameter_fingerprint = fingerprint(
        {
            "kind": "standardization",
            "precision": precision,
            "ddof": 0,
            "epsilon": float(epsilon),
            "upstream": design["fingerprint"],
        }
    )
    standard_fingerprint = fingerprint(
        {
            "upstream": design["fingerprint"],
            "parameters": parameter_fingerprint,
            "precision": precision,
            "matrix": array_fingerprint(standardized),
        }
    )
    return {
        "scenario_id": design["scenario_id"],
        "sample_size": design["sample_size"],
        "order_limit": design["order_limit"],
        "precision": precision,
        "sample_ids": np.asarray(design["sample_ids"], dtype=object).copy(),
        "target": np.asarray(design["target"], dtype=dtype).copy(),
        "feature_names": list(design["feature_names"]),
        "matrix": standardized,
        "mean": mean,
        "std": std,
        "upstream_fingerprint": design["fingerprint"],
        "parameter_fingerprint": parameter_fingerprint,
        "fingerprint": standard_fingerprint,
    }


def _candidate_values(
    configured: list[int],
    maximum: int,
    metrics: np.ndarray,
) -> list[dict[str, Any]]:
    selected = sorted({min(int(value), maximum) for value in configured if int(value) > 0})
    if not selected:
        selected = [maximum]
    output: list[dict[str, Any]] = []
    for dimension in selected:
        output.append(
            {
                "dimension": int(dimension),
                "metric": float(metrics[dimension - 1]),
            }
        )
    return output


def _pca(standard: dict[str, Any], path_config: dict[str, Any], precision: str) -> dict[str, Any]:
    dtype = _dtype(precision)
    matrix = np.asarray(standard["matrix"], dtype=dtype)
    _, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    all_singular_values = np.asarray(singular_values, dtype=dtype)
    maximum = min(int(path_config["components"]), len(all_singular_values), matrix.shape[1])
    if maximum <= 0:
        raise ValueError("PCA has no available component")
    singular_values = np.asarray(all_singular_values[:maximum], dtype=dtype)
    components = np.asarray(vt[:maximum], dtype=dtype).copy()
    scores_full = np.asarray(np.dot(matrix, components.T), dtype=dtype)
    for index in range(maximum):
        dominant = int(np.argmax(np.abs(components[index])))
        if components[index, dominant] < 0:
            components[index] *= dtype.type(-1)
            scores_full[:, index] *= dtype.type(-1)
    denominator = max(1, matrix.shape[0] - 1)
    explained_variance = (singular_values * singular_values) / dtype.type(denominator)
    all_explained = (all_singular_values * all_singular_values) / dtype.type(denominator)
    total_variance = float(np.sum(all_explained, dtype=dtype))
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        ratios = np.zeros_like(explained_variance)
    else:
        ratios = explained_variance / dtype.type(total_variance)
    cumulative = np.cumsum(ratios)
    scores = scores_full[:, :maximum].copy()
    candidates = _candidate_values(path_config["candidates"], maximum, cumulative)
    summary = {
        "kind": "pca",
        "explained_variance": [float(value) for value in explained_variance],
        "explained_variance_ratio": [float(value) for value in ratios],
        "cumulative_explained_variance_ratio": [float(value) for value in cumulative],
        "total_variance": total_variance,
        "component_sign_rule": "dominant_absolute_component_positive",
        "component_dominant_indices": [int(np.argmax(np.abs(components[index]))) for index in range(maximum)],
        "component_fingerprint": array_fingerprint(components),
    }
    return {
        "path": "pca",
        "retained_dimension": maximum,
        "scores": scores,
        "components": components,
        "candidate_dimensions": candidates,
        "summary": summary,
    }


def _random_projection(
    standard: dict[str, Any],
    path_config: dict[str, Any],
    precision: str,
    scenario_fingerprint: str,
) -> dict[str, Any]:
    dtype = _dtype(precision)
    matrix = np.asarray(standard["matrix"], dtype=dtype)
    component_count = min(int(path_config["components"]), matrix.shape[1], matrix.shape[0])
    if component_count <= 0:
        raise ValueError("random projection has no available component")
    feature_names = list(standard["feature_names"])
    seed_material = stable_seed(
        "random_projection",
        scenario_fingerprint,
        json_like(feature_names),
        json_like(path_config),
        component_count,
    )
    projection64 = np.empty((matrix.shape[1], component_count), dtype=np.float64)
    scale = 1.0 / math.sqrt(float(component_count))
    for row in range(matrix.shape[1]):
        for column in range(component_count):
            projection64[row, column] = stable_normal(seed_material, "projection", row, column) * scale
    projection = projection64.astype(dtype)
    scores = np.asarray(np.dot(matrix, projection), dtype=dtype)
    variances = np.var(scores, axis=0, dtype=dtype, ddof=1) if scores.shape[0] > 1 else np.zeros(component_count, dtype=dtype)
    input_variances = np.var(matrix, axis=0, dtype=dtype, ddof=1) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1], dtype=dtype)
    total_variance = float(np.sum(input_variances, dtype=dtype))
    if not math.isfinite(total_variance) or total_variance <= 0.0:
        ratios = np.zeros_like(variances)
    else:
        ratios = variances / dtype.type(total_variance)
    cumulative = np.cumsum(ratios)
    candidates = _candidate_values(path_config["candidates"], component_count, cumulative)
    summary = {
        "kind": "random_projection",
        "projection_matrix_shape": [int(matrix.shape[1]), int(component_count)],
        "projection_seed": int(seed_material),
        "projection_scale": scale,
        "projection_matrix_fingerprint": array_fingerprint(projection64),
        "projection_matrix_norm": float(np.linalg.norm(projection64)),
        "component_variance": [float(value) for value in variances],
        "cumulative_variance_ratio": [float(value) for value in cumulative],
        "total_input_variance": total_variance,
    }
    return {
        "path": "random_projection",
        "retained_dimension": component_count,
        "scores": scores,
        "candidate_dimensions": candidates,
        "summary": summary,
    }


def _variance_filter(standard: dict[str, Any], path_config: dict[str, Any], precision: str) -> dict[str, Any]:
    dtype = _dtype(precision)
    matrix = np.asarray(standard["matrix"], dtype=dtype)
    variances = np.var(matrix, axis=0, dtype=dtype, ddof=1)
    minimum = dtype.type(path_config.get("min_variance", 0.0))
    eligible = [index for index, value in enumerate(variances) if value >= minimum]
    if not eligible:
        eligible = list(range(matrix.shape[1]))
    ranked = sorted(
        eligible,
        key=lambda index: (-round(float(variances[index]), 4), index),
    )
    maximum = min(int(path_config["components"]), len(ranked))
    if maximum <= 0:
        raise ValueError("variance filter has no available feature")
    selected = ranked[:maximum]
    scores = np.asarray(matrix[:, selected], dtype=dtype)
    all_total = float(np.sum(variances, dtype=dtype))
    cumulative_variance = np.cumsum(np.asarray([variances[index] for index in ranked[:maximum]], dtype=dtype))
    if not math.isfinite(all_total) or all_total <= 0.0:
        cumulative_ratios = np.zeros_like(cumulative_variance)
    else:
        cumulative_ratios = cumulative_variance / dtype.type(all_total)
    candidates = _candidate_values(path_config["candidates"], maximum, cumulative_ratios)
    summary = {
        "kind": "variance_filter",
        "selected_features": [standard["feature_names"][index] for index in selected],
        "selected_indices": selected,
        "feature_variances": {standard["feature_names"][index]: float(variances[index]) for index in selected},
        "min_variance": float(minimum),
        "retained_variance_ratio": float(cumulative_ratios[-1]) if len(cumulative_ratios) else 0.0,
        "candidate_variance_ratios": [float(value) for value in cumulative_ratios],
    }
    return {
        "path": "variance_filter",
        "retained_dimension": maximum,
        "scores": scores,
        "candidate_dimensions": candidates,
        "summary": summary,
    }


def json_like(value: Any) -> str:
    """Small stable string representation used only for projection seeding."""

    return fingerprint(value)


def reduction_parameter_fingerprint(
    standard_fingerprint: str,
    path: str,
    path_config: dict[str, Any],
    precision: str,
    scenario_scope: str,
) -> str:
    return fingerprint(
        {
            "kind": "reduction",
            "schema": 3,
            "path": path,
            "configuration": path_config,
            "scenario": scenario_scope,
            "precision": precision,
            "upstream": standard_fingerprint,
        }
    )


def reduce_path(
    standard: dict[str, Any],
    path: str,
    path_config: dict[str, Any],
    precision: str,
    scenario_fingerprint: str,
) -> dict[str, Any]:
    if path == "pca":
        reduction = _pca(standard, path_config, precision)
    elif path == "random_projection":
        reduction = _random_projection(standard, path_config, precision, scenario_fingerprint)
    elif path == "variance_filter":
        reduction = _variance_filter(standard, path_config, precision)
    else:
        raise ValueError(f"unsupported reduction path: {path}")

    parameter_fingerprint = reduction_parameter_fingerprint(
        standard["fingerprint"],
        path,
        path_config,
        precision,
        scenario_fingerprint,
    )
    result_fingerprint = fingerprint(
        {
            "upstream": standard["fingerprint"],
            "parameters": parameter_fingerprint,
            "precision": precision,
            "scores": array_fingerprint(reduction["scores"]),
            "summary": reduction["summary"],
        }
    )
    result = {
        "scenario_id": standard["scenario_id"],
        "sample_size": standard["sample_size"],
        "order_limit": standard["order_limit"],
        "path": path,
        "precision": precision,
        "sample_ids": np.asarray(standard["sample_ids"], dtype=object).copy(),
        "target": np.asarray(standard["target"], dtype=_dtype(precision)).copy(),
        "projection_input_features": list(standard["feature_names"]),
        "retained_dimension": int(reduction["retained_dimension"]),
        "scores": reduction["scores"],
        "candidate_dimensions": reduction["candidate_dimensions"],
        "summary": reduction["summary"],
        "upstream_fingerprint": standard["fingerprint"],
        "parameter_fingerprint": parameter_fingerprint,
        "fingerprint": result_fingerprint,
    }
    if "components" in reduction:
        result["components"] = reduction["components"]
    return result
