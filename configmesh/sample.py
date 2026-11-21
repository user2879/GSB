"""Deterministic local sample generation and fixed-point NumPy aggregation.

The sample set is a pure function of (node_id, shard_index, shard_count), so
platform and every node subprocess compute identical integers without sharing
data.  Aggregation uses NumPy int64 dot products and integer floor division;
platform-level saturation clamps the final digest to signed 64-bit.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from .util import canonical_json

SAMPLE_SIZE = 96
INT32_SPAN = 1_000_000


def _seed(node_id: str, shard_index: int, shard_count: int) -> int:
    digest = hashlib.sha256(
        canonical_json({
            "scope": "configmesh-sample-v1",
            "node_id": node_id,
            "shard_index": shard_index,
            "shard_count": shard_count,
        })
    ).digest()
    return int.from_bytes(digest[:8], "big")


def sample_matrix(node_id: str, shard_index: int, shard_count: int,
                  size: int = SAMPLE_SIZE) -> np.ndarray:
    """Return a deterministic int64 matrix with shape (size, 3)."""
    rng = np.random.default_rng(_seed(node_id, shard_index, shard_count))
    return rng.integers(-INT32_SPAN, INT32_SPAN, size=(size, 3), dtype=np.int64)


def shard_rows(node_id: str, shard_index: int, shard_count: int,
               size: int = SAMPLE_SIZE) -> np.ndarray:
    """Rows owned by a shard: contiguous deterministic partition of the matrix."""
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    matrix = sample_matrix(node_id, shard_index, shard_count, size)
    bounds = np.linspace(0, size, shard_count + 1, dtype=int)
    start, end = int(bounds[shard_index]), int(bounds[shard_index + 1])
    return matrix[start:end]


def aggregate_shards(qvalues: Sequence[int], node_id: str,
                     shard_count: int) -> dict:
    """Aggregate every shard using NumPy integer dot products.

    The three qvalues weight the three deterministic sample columns.  Division
    is floor division on the summed dot product; the result is saturated to
    signed 64-bit.
    """
    raw_weights = np.asarray(qvalues, dtype=np.int64)
    # Deterministic samples have three columns; zero-pad/truncate so any number
    # of configuration keys can weight the fixed-shape local sample.
    column_count = 3
    weights = np.zeros(column_count, dtype=np.int64)
    limit = min(column_count, raw_weights.shape[0])
    weights[:limit] = raw_weights[:limit]
    shard_digests: list[int] = []
    running = np.int64(0)
    for shard_index in range(shard_count):
        rows = shard_rows(node_id, shard_index, shard_count)
        products = rows * weights  # int64 element-wise
        dot_sum = int(products.sum(dtype=np.int64))
        divisor = max(rows.shape[0], 1)
        shard_value = dot_sum // divisor
        shard_digests.append(int(shard_value))
        running = np.int64(int(running) + shard_value)
    digest = int(running)
    if digest > 2**63 - 1:
        digest = 2**63 - 1
    elif digest < -(2**63):
        digest = -(2**63)
    return {
        "node_id": node_id,
        "shard_count": shard_count,
        "qvalues": [int(v) for v in qvalues],
        "shard_digests": shard_digests,
        "digest": digest,
    }


def aggregate_digest(qvalues: Sequence[int], node_id: str,
                     shard_count: int) -> str:
    return hashlib.sha256(
        canonical_json(aggregate_shards(qvalues, node_id, shard_count))
    ).hexdigest()
