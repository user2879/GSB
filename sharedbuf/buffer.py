"""Deterministic shared-buffer model.

Sector contents are derived from a seed so the platform and every node
compute bit-identical data without exchanging the buffers themselves.
Batch execution is a pure function so the platform can recompute the
expected summary of any lease and compare it with the node report.
"""
from __future__ import annotations

import numpy as np

SECTOR_SIZE = 8
AGGS = ("sum", "mean", "min", "max")


def sector_seed(seed: int, sector_index: int) -> int:
    return seed * 100003 + sector_index


def initial_sector_data(seed: int, sector_index: int) -> np.ndarray:
    rng = np.random.default_rng(sector_seed(seed, sector_index))
    return np.round(rng.uniform(-1.0, 1.0, SECTOR_SIZE), 6)


def aggregate(arr: np.ndarray, agg: str) -> float:
    if agg == "sum":
        return float(np.sum(arr))
    if agg == "mean":
        return float(np.mean(arr))
    if agg == "min":
        return float(np.min(arr))
    if agg == "max":
        return float(np.max(arr))
    raise ValueError(f"unknown aggregation {agg!r}")


def run_batch(datas, op: str, agg: str, delta: float, count: int):
    """Execute ``count`` deterministic sub-operations over sector arrays.

    For writes each sub-operation adds ``delta`` to every sector; the
    per-step aggregate is accumulated into the returned value. Returns
    ``(value, post_arrays)``.
    """
    arrs = [np.array(a, dtype=float, copy=True) for a in datas]
    total = 0.0
    for _ in range(count):
        if op == "write":
            for arr in arrs:
                arr += delta
        total += aggregate(np.concatenate(arrs), agg)
    return total, arrs
