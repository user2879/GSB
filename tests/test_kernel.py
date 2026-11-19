import numpy as np
import pytest

from spmm.config import EngineConfig
from spmm.csr import generate_csr
from spmm.errors import E_BLOCK_INVALID, SpmmError
from spmm.kernel import (EMPTY_ROW_MARKER, ORDERS, b_row_abs_keys,
                         compute_row, reference_spmm)


def _dense_inputs(seed=11, m=24, k=20, n=12):
    a = generate_csr(m, k, "uniform", seed)
    rng = np.random.default_rng(seed + 1)
    b = rng.standard_normal((k, n)).astype(np.float32)
    c = rng.standard_normal((m, n)).astype(np.float32)
    return a, b, c


def test_reference_kernel_matches_numpy_fp64():
    a, b, c = _dense_inputs()
    out, metas = reference_spmm(a, b, c, 1.5, 0.25)
    dense = np.zeros((a.n_rows, a.n_cols), dtype=np.float64)
    for i in range(a.n_rows):
        dense[i, a.row_cols(i)] = a.row_vals(i)
    expected = (1.5 * dense @ b.astype(np.float64) + 0.25 * c.astype(np.float64))
    assert np.allclose(out.astype(np.float64), expected, rtol=1e-5, atol=1e-5)
    assert all(m["order"] == "asc" for m in metas)


def test_reference_kernel_is_bit_deterministic():
    a, b, c = _dense_inputs()
    out1, metas1 = reference_spmm(a, b, c, 1.25, 0.5)
    out2, metas2 = reference_spmm(a, b, c, 1.25, 0.5)
    assert np.array_equal(out1, out2)
    assert [m["fingerprint"] for m in metas1] == [m["fingerprint"] for m in metas2]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("chunk", [1, 3, 16, 64])
def test_blocked_kernel_fp64_matches_reference(order, chunk):
    a, b, c = _dense_inputs()
    abs_keys = b_row_abs_keys(b)
    dense = np.zeros((a.n_rows, a.n_cols), dtype=np.float64)
    for i in range(a.n_rows):
        dense[i, a.row_cols(i)] = a.row_vals(i)
    expected = 0.75 * dense @ b.astype(np.float64) + 1.5 * c.astype(np.float64)
    for i in range(a.n_rows):
        out, meta = compute_row(a, b, c[i], i, 0.75, 1.5, "fp64", chunk, order,
                                abs_keys)
        assert np.allclose(out.astype(np.float64), expected[i], rtol=1e-9,
                           atol=1e-6)
        assert meta["chunks"] == (a.row_nnz(i) + chunk - 1) // chunk


def test_empty_row_marker_and_value():
    a, b, c = _dense_inputs()
    a.indptr[1:10] = a.indptr[0]  # 强制第 0..8 行为空行（仅测试内核行为）
    abs_keys = b_row_abs_keys(b)
    out, meta = compute_row(a, b, c[0], 0, 2.0, 3.0, "fp32", 4, "asc", abs_keys)
    assert meta["empty"] is True
    assert meta["marker"] == EMPTY_ROW_MARKER
    assert np.array_equal(out, (np.float32(3.0) * c[0]).astype(np.float32))


def test_block_param_validation():
    cfg = EngineConfig(row_block_size=0)
    with pytest.raises(SpmmError) as exc:
        cfg.validate(64, 48)
    assert exc.value.code == E_BLOCK_INVALID
    cfg = EngineConfig(row_block_size=65)
    with pytest.raises(SpmmError) as exc:
        cfg.validate(64, 48)
    assert exc.value.code == E_BLOCK_INVALID
    cfg = EngineConfig(nnz_chunk_size=0)
    with pytest.raises(SpmmError) as exc:
        cfg.validate(64, 48)
    assert exc.value.code == E_BLOCK_INVALID
