"""Three deterministic dimensionality-reduction paths."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ENGINE_REVISION, PipelineConfig
from .standardize import dtype_for_tier
from .util import fingerprint, sha256_bytes


def quantization_profile(config: PipelineConfig) -> dict[str, Any]:
    return config.quantization_profile()


def apply_fixed_point(array: np.ndarray, profile: dict[str, Any]) -> np.ndarray:
    if not bool(profile.get("enabled", False)):
        return array.copy()
    bits = int(profile.get("fractional_bits", 0))
    scale = float(2**bits)
    return (np.rint(array * scale) / scale).astype(array.dtype, copy=False)


def reduce_pca(
    matrix: np.ndarray,
    feature_keys: list[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    dimensions = min(int(settings["dimensions"]), matrix.shape[1], matrix.shape[0])
    centered = matrix - matrix.mean(axis=0, dtype=matrix.dtype)
    if matrix.shape[1] <= 1:
        singular_values = np.asarray([np.linalg.norm(centered[:, 0])], dtype=matrix.dtype)
        components = np.asarray([[1.0]], dtype=matrix.dtype)
    else:
        _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    components = np.array(components[:dimensions], dtype=matrix.dtype, copy=True)
    for row_index in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[row_index])))
        if float(components[row_index, pivot]) < 0.0:
            components[row_index] *= np.asarray(-1.0, dtype=matrix.dtype)
    singular_values = np.asarray(singular_values[:dimensions], dtype=matrix.dtype)
    denominator = float(np.sum(np.square(singular_values), dtype=matrix.dtype))
    if denominator > 0.0:
        ratios = np.square(singular_values, dtype=matrix.dtype) / np.asarray(denominator, dtype=matrix.dtype)
    else:
        ratios = np.zeros_like(singular_values)
    projected_raw = centered @ components.T
    return {
        "output_feature_names": [f"pca_{index}" for index in range(dimensions)],
        "projected_raw": projected_raw,
        "summary": {
            "method": "pca",
            "explained_variance": [float(value) for value in np.square(singular_values)],
            "explained_variance_ratio": [float(value) for value in ratios],
            "total_explained_variance_ratio": float(np.sum(ratios, dtype=matrix.dtype)),
            "components": [[float(value) for value in row] for row in components],
            "sign_normalization": "largest_absolute_component_positive",
        },
    }


def _random_projection_seed(
    config: PipelineConfig, feature_keys: list[str], settings: dict[str, Any]
) -> tuple[int, str]:
    path_fingerprint = fingerprint(
        {
            "engine_revision": ENGINE_REVISION,
            "path": "random_projection",
            "settings": settings,
            "feature_keys": feature_keys,
        }
    )
    seed = int.from_bytes(
        bytes.fromhex(sha256_bytes(path_fingerprint.encode("utf-8")))[:8], "big"
    ) % (2**32)
    return seed, path_fingerprint


def reduce_random_projection(
    matrix: np.ndarray,
    feature_keys: list[str],
    settings: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    dimensions = min(int(settings["dimensions"]), matrix.shape[1], matrix.shape[0])
    seed, path_fingerprint = _random_projection_seed(config, feature_keys, settings)
    rng = np.random.default_rng(seed)
    projection = rng.standard_normal((matrix.shape[1], dimensions)).astype(matrix.dtype)
    projection /= np.asarray(np.sqrt(dimensions), dtype=matrix.dtype)
    projected_raw = matrix @ projection
    return {
        "output_feature_names": [f"rp_{index}" for index in range(dimensions)],
        "projected_raw": projected_raw,
        "summary": {
            "method": "random_projection",
            "seed": seed,
            "path_fingerprint": path_fingerprint,
            "matrix_shape": [int(matrix.shape[1]), int(dimensions)],
            "matrix_fingerprint": fingerprint(
                [[float(value) for value in row] for row in projection]
            ),
            "projection_matrix": [[float(value) for value in row] for row in projection],
            "distribution": "gaussian",
        },
    }


def reduce_variance(
    matrix: np.ndarray,
    feature_keys: list[str],
    settings: dict[str, Any],
) -> dict[str, Any]:
    dimensions = min(int(settings["dimensions"]), matrix.shape[1], matrix.shape[0])
    variances = matrix.var(axis=0, dtype=matrix.dtype)
    ranked = sorted(
        range(len(feature_keys)),
        key=lambda index: (-float(variances[index]), feature_keys[index]),
    )
    selected = sorted(ranked[:dimensions], key=lambda index: feature_keys[index])
    projected_raw = matrix[:, selected]
    return {
        "output_feature_names": [feature_keys[index] for index in selected],
        "projected_raw": projected_raw,
        "summary": {
            "method": "variance",
            "selected_feature_keys": [feature_keys[index] for index in selected],
            "variances": [float(variances[index]) for index in selected],
            "selection_rule": "descending_variance_then_feature_key",
        },
    }


def reduce_path(
    path: str,
    matrix: np.ndarray,
    feature_keys: list[str],
    settings: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    if path == "pca":
        result = reduce_pca(matrix, feature_keys, settings)
    elif path == "random_projection":
        result = reduce_random_projection(matrix, feature_keys, settings, config)
    elif path == "variance":
        result = reduce_variance(matrix, feature_keys, settings)
    else:
        raise ValueError(f"unknown reduction path: {path}")
    projected_raw = np.asarray(result.pop("projected_raw"))
    projected = apply_fixed_point(projected_raw, quantization_profile(config))
    result["content"] = {
        "path": path,
        "input_feature_keys": list(feature_keys),
        "retained_dimensions": int(projected.shape[1]),
        "output_feature_names": result["output_feature_names"],
        "projected_raw": [[float(value) for value in row] for row in projected_raw],
        "projected": [[float(value) for value in row] for row in projected],
        "summary": result["summary"],
        "quantization": {
            "tier": str(config.raw["precision"]["quantization"]["tier"]),
            **quantization_profile(config),
        },
    }
    return result


def reduction_cache_key(
    config: PipelineConfig,
    tier: str,
    path: str,
    upstream_standard: list[dict[str, str]],
    settings: dict[str, Any],
) -> str:
    return fingerprint(
        {
            "stage": "dimensionality_reduction",
            "engine_revision": ENGINE_REVISION,
            "precision_fingerprint": config.precision_fingerprint(tier),
            "path": path,
            "path_settings": settings,
            "upstream_standard": upstream_standard,
        }
    )
