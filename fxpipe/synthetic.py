"""Deterministic synthetic data helpers built on NumPy 2.x.

The fixed-point kernels themselves stay in pure Python integers so results are
independent of host word size and thread scheduling. NumPy is used here, with a
fixed ``numpy.random.default_rng(seed)`` stream, to materialize the deterministic
synthetic embedding matrix and integer segment weight tables.
"""

from __future__ import annotations

import numpy as np


def synthetic_embedding_matrix(vocab_size: int, dim: int, seed: int, span: int = 4096) -> np.ndarray:
    """Reproducible integer embedding matrix of shape (vocab_size, dim)."""
    rng = np.random.default_rng(seed)
    return rng.integers(-span // 2, span // 2, size=(vocab_size, dim), dtype=np.int64)


def synthetic_bias_vector(dim: int, seed: int, span: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed + 7)
    return rng.integers(-span // 2, span // 2, size=(dim,), dtype=np.int64)


def integer_permille_table(rows: int, cols: int, seed: int) -> np.ndarray:
    """Rows of integer per-mille weights; each row sums to exactly 1000."""
    rng = np.random.default_rng(seed)
    raw = rng.integers(1, 100_000, size=(rows, cols), dtype=np.int64)
    weights = (raw * 1000 // raw.sum(axis=1, keepdims=True))
    remainder = 1000 - weights.sum(axis=1, keepdims=True)
    for row_index in range(rows):
        cursor = 0
        while remainder[row_index, 0] > 0:
            weights[row_index, cursor % cols] += 1
            remainder[row_index, 0] -= 1
            cursor += 1
    return weights
