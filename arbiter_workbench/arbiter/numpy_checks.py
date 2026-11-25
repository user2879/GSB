"""NumPy cross-checks for fixed-point quantization and cost estimates.

The deterministic data path stays on Python integers; NumPy is used only as an
independent vectorized oracle in diagnostics, so any divergence is detectable.
"""
from __future__ import annotations

import numpy as np

from .fixedpoint import MAX_INT64, MIN_INT64, FixedPoint


def vectorized_quantize(raw_values: list[str], fp: FixedPoint) -> np.ndarray:
    """Quantize a batch of decimal strings toward zero using integer NumPy."""
    decimals = np.array([float(v) for v in raw_values], dtype=np.float64)
    scaled = np.trunc(decimals * fp.factor).astype(np.int64)
    return np.clip(scaled, MIN_INT64, MAX_INT64)


def cross_check_quantization(raw_values: list[str], fp: FixedPoint) -> dict:
    vector = vectorized_quantize(raw_values, fp)
    scalar = np.array([fp.quantize(v).q for v in raw_values], dtype=np.int64)
    mismatch = np.flatnonzero(vector != scalar).tolist()
    return {
        "checked": len(raw_values),
        "mismatch_indexes": mismatch,
        "consistent": not mismatch,
    }


def vectorized_cost_grid(env_counts, key_counts, avg_obs) -> np.ndarray:
    """Full-path cost grid E * K * N_avg as an integer NumPy matrix."""
    e = np.asarray(env_counts, dtype=np.int64)
    k = np.asarray(key_counts, dtype=np.int64)
    grid = np.multiply.outer(e, k) * int(avg_obs)
    return grid.astype(np.int64)
