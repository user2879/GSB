"""Precision-aware standardization."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import ENGINE_REVISION, PipelineConfig
from .util import fingerprint


def dtype_for_tier(tier: str) -> np.dtype[Any]:
    if tier == "single":
        return np.dtype(np.float32)
    if tier == "double":
        return np.dtype(np.float64)
    raise ValueError(f"unknown precision tier: {tier}")


def standardize_values(
    feature_key: str,
    values: list[int],
    *,
    tier: str,
    method: str,
    epsilon: float,
) -> dict[str, Any]:
    if method != "zscore":
        raise ValueError(f"unsupported standardization method: {method}")
    dtype = dtype_for_tier(tier)
    array = np.asarray(values, dtype=dtype)
    if array.size == 0:
        raise ValueError("cannot standardize an empty feature")
    mean = np.asarray(array.mean(dtype=dtype), dtype=dtype)
    std = np.asarray(array.std(dtype=dtype), dtype=dtype)
    threshold = np.asarray(epsilon, dtype=dtype)
    if bool(std <= threshold):
        standardized = np.zeros_like(array)
        constant = True
    else:
        standardized = (array - mean) / std
        constant = False
    return {
        "content": {
            "feature_key": feature_key,
            "values": [float(value) for value in standardized],
            "mean": float(mean),
            "std": float(std),
            "constant": constant,
            "method": method,
        }
    }


def standard_cache_key(
    config: PipelineConfig,
    tier: str,
    feature_key: str,
    upstream_cross_content_fingerprint: str,
) -> str:
    return fingerprint(
        {
            "stage": "standardization",
            "engine_revision": ENGINE_REVISION,
            "precision_fingerprint": config.precision_fingerprint(tier),
            "upstream_cross_content_fingerprint": upstream_cross_content_fingerprint,
            "feature_key": feature_key,
            "params": config.raw["standardization"],
        }
    )
