import numpy as np
import pytest

from spmm.csr import (DISTRIBUTIONS, append_rows, delete_rows, generate_csr,
                      replace_rows, validate_csr)
from spmm.errors import E_CSR_INVALID, SpmmError


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_generation_is_deterministic(dist):
    a1 = generate_csr(32, 24, dist, 123)
    a2 = generate_csr(32, 24, dist, 123)
    assert np.array_equal(a1.indptr, a2.indptr)
    assert np.array_equal(a1.indices, a2.indices)
    assert np.array_equal(a1.values, a2.values)


@pytest.mark.parametrize("dist", DISTRIBUTIONS)
def test_generated_columns_strictly_increasing(dist):
    a = generate_csr(48, 40, dist, 7)
    validate_csr(a)
    for i in range(a.n_rows):
        cols = a.row_cols(i)
        assert np.all(cols[1:] > cols[:-1])


def test_validate_rejects_non_monotone_indptr():
    a = generate_csr(8, 8, "uniform", 1)
    a.indptr[2] = a.indptr[3] + 1
    with pytest.raises(SpmmError) as exc:
        validate_csr(a)
    assert exc.value.code == E_CSR_INVALID
    assert any(v["field"] == "indptr" and v["row"] == 2 for v in exc.value.violations)


def test_validate_rejects_out_of_range_index():
    a = generate_csr(8, 8, "uniform", 2)
    a.indices[0] = a.n_cols
    with pytest.raises(SpmmError) as exc:
        validate_csr(a)
    assert exc.value.code == E_CSR_INVALID
    assert any(v["field"] == "indices" and v["row"] == 0 for v in exc.value.violations)


def test_validate_rejects_unsorted_columns():
    a = generate_csr(64, 16, "powerlaw", 3)
    row = next(i for i in range(a.n_rows) if a.row_nnz(i) >= 2)
    cols = a.row_cols(row)
    cols[0], cols[1] = cols[1], cols[0]
    with pytest.raises(SpmmError) as exc:
        validate_csr(a)
    assert exc.value.code == E_CSR_INVALID
    assert any(v["field"] == "indices_order" and v["row"] == row
               for v in exc.value.violations)


def test_validate_rejects_nnz_mismatch():
    a = generate_csr(8, 8, "uniform", 4)
    a.indptr[-1] += 1
    with pytest.raises(SpmmError) as exc:
        validate_csr(a)
    assert exc.value.code == E_CSR_INVALID
    assert any(v["field"] == "nnz" for v in exc.value.violations)


def test_row_edits_keep_structure_valid():
    a = generate_csr(16, 12, "uniform", 5)
    a2 = replace_rows(a, {3: (np.array([1, 5, 9]), np.array([1.0, 2.0, 3.0], np.float32))})
    validate_csr(a2)
    assert a2.row_nnz(3) == 3
    a3 = delete_rows(a2, [0, 15])
    validate_csr(a3)
    assert a3.n_rows == 14
    a4 = append_rows(a3, [(np.array([0, 2]), np.array([0.5, 0.25], np.float32))])
    validate_csr(a4)
    assert a4.n_rows == 15
    assert a4.row_nnz(14) == 2
