from __future__ import annotations

import numpy as np
import pytest

from spmm_diag.config import SpMMConfig
from spmm_diag.csr import DistributionSpec, generate_csr


@pytest.fixture
def sample_bundle():
    csr = generate_csr(16, 9, DistributionSpec("extreme", 0, 9), seed=41)
    rng = np.random.default_rng(42)
    b = rng.normal(0.0, 1.0, size=(9, 5))
    c = rng.normal(0.0, 0.5, size=(16, 5))
    config = SpMMConfig(
        alpha=0.75,
        beta=0.3,
        row_block_size=3,
        nnz_chunk_size=2,
        order="col_asc",
        precision_mode="fp32",
        fractional_bits=6,
        accumulator_bits=16,
        max_workers=4,
        recompute_budget=10_000,
        hot_capacity=12,
    )
    return csr, b, c, config


@pytest.fixture
def tiny_csr():
    from spmm_diag.csr import CSRMatrix

    offsets = np.array([0, 2, 2, 4, 6], dtype=np.int64)
    columns = np.array([0, 2, 1, 3, 0, 2], dtype=np.int64)
    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0], dtype=np.float64)
    return CSRMatrix(4, 4, offsets, columns, values, distribution="manual")

