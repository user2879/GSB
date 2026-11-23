from __future__ import annotations

import numpy as np

from spmm_diag.config import (
    MODE_FIXED,
    ORDER_ASCENDING,
    ORDER_B_ABS_DESC,
    ORDER_DESCENDING,
    SpMMConfig,
)
from spmm_diag.csr import CSRMatrix
from spmm_diag.engine import full_rebuild
from spmm_diag.kernels import compare_float_and_fixed


def _single_row(values, b_rows, config):
    csr = CSRMatrix(
        1,
        len(b_rows),
        np.array([0, len(values)], dtype=np.int64),
        np.arange(len(values), dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )
    b = np.asarray(b_rows, dtype=np.float64).reshape(-1, 1)
    c = np.zeros((1, 1), dtype=np.float64)
    return full_rebuild(csr, b, c, config)


def test_float32_order_switch_changes_rounding():
    base = dict(
        alpha=1.0,
        beta=0.0,
        row_block_size=1,
        nnz_chunk_size=8,
        precision_mode="fp32",
        max_workers=1,
    )
    ascending = _single_row(
        [1.0, 16_777_216.0, -16_777_216.0],
        [1.0, 1.0, 1.0],
        SpMMConfig(order=ORDER_ASCENDING, **base),
    )
    descending = _single_row(
        [1.0, 16_777_216.0, -16_777_216.0],
        [1.0, 1.0, 1.0],
        SpMMConfig(order=ORDER_DESCENDING, **base),
    )
    b_abs = _single_row(
        [1.0, 16_777_216.0, -16_777_216.0],
        [1.0, 2.0, 100.0],
        SpMMConfig(order=ORDER_B_ABS_DESC, **base),
    )
    assert float(ascending.c_matrix[0, 0]) == 0.0
    assert float(descending.c_matrix[0, 0]) == 1.0
    assert float(b_abs.c_matrix[0, 0]) == -1_644_167_168.0
    assert ascending.row_fingerprints != descending.row_fingerprints
    assert ascending.row_fingerprints != b_abs.row_fingerprints


def test_fixed_saturation_records_first_overflow_and_order_direction():
    values = [1.0, 1000.0, -1000.0]
    b_rows = [1.0, 1.0, 1.0]
    common = dict(
        alpha=1.0,
        beta=0.0,
        row_block_size=1,
        nnz_chunk_size=8,
        precision_mode=MODE_FIXED,
        fractional_bits=0,
        accumulator_bits=10,
        max_workers=1,
    )
    ascending = _single_row(values, b_rows, SpMMConfig(order=ORDER_ASCENDING, **common))
    descending = _single_row(values, b_rows, SpMMConfig(order=ORDER_DESCENDING, **common))
    assert ascending.blocks[0].first_overflow is not None
    assert descending.blocks[0].first_overflow is not None
    assert ascending.blocks[0].first_overflow.direction == "+"
    assert descending.blocks[0].first_overflow.direction == "-"
    assert ascending.blocks[0].first_overflow.row == 0
    assert ascending.blocks[0].first_overflow.column == 0
    assert float(ascending.c_matrix[0, 0]) != float(descending.c_matrix[0, 0])


def test_boundary_chunks_empty_rows_and_float_fixed_comparability(sample_bundle):
    csr, b, c, config = sample_bundle
    empty_rows = [row for row in range(csr.rows) if csr.row_nnz(row) == 0]
    assert empty_rows
    float_result = full_rebuild(csr, b, c, config)
    fixed_config = config.with_changes(
        precision_mode=MODE_FIXED, fractional_bits=5, accumulator_bits=24
    )
    fixed_result = full_rebuild(csr, b, c, fixed_config)
    for block in float_result.blocks.values():
        assert all(block.row_end - block.row_start <= config.row_block_size for _ in [0])
    fixed_empty_rows = sorted(
        row for block in fixed_result.blocks.values() for row in block.empty_rows
    )
    assert fixed_empty_rows == empty_rows
    comparison = compare_float_and_fixed(
        list(float_result.blocks.values()), list(fixed_result.blocks.values())
    )
    assert comparison["comparable"]
    assert comparison["element_count"] == csr.rows * b.shape[1]
    assert comparison["within_tolerance"]
