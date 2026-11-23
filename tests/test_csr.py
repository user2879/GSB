from __future__ import annotations

from dataclasses import replace

import numpy as np

from spmm_diag.csr import DistributionSpec, generate_csr
from spmm_diag.diagnostics import E_CSR_INVALID


def test_all_distributions_are_sorted_and_deterministic():
    fingerprints = {}
    for kind in ("uniform", "powerlaw", "extreme"):
        spec = DistributionSpec(kind, 0, 11, power=3.0)
        first = generate_csr(40, 17, spec, seed=123)
        second = generate_csr(40, 17, spec, seed=123)
        assert first.fingerprint() == second.fingerprint()
        assert not first.validate()
        assert first.nnz > 0
        for row in range(first.rows):
            columns = first.row_columns(row)
            assert np.all(np.diff(columns) > 0)
        fingerprints[kind] = first.fingerprint()
    assert len(set(fingerprints.values())) == 3


def test_extreme_distribution_contains_empty_and_dense_rows():
    csr = generate_csr(
        80,
        20,
        DistributionSpec(
            "extreme", 0, 20, zero_probability=0.55, extreme_fraction=0.05
        ),
        seed=9,
    )
    counts = np.diff(csr.row_offsets)
    assert np.any(counts == 0)
    assert np.any(counts == 20)


def test_invalid_csr_reports_stable_code_and_nonfinite():
    csr = generate_csr(8, 6, DistributionSpec("uniform", 1, 4), seed=7)
    columns = csr.col_indices.copy()
    columns[0] = csr.cols + 1
    invalid = replace(csr, col_indices=columns)
    failures = invalid.validate()
    assert any(item.code == E_CSR_INVALID for item in failures)

    bad_values = csr.values.copy()
    bad_values[0] = np.nan
    nonfinite = replace(csr, values=bad_values)
    assert any(item.code == "E_MATRIX_NONFINITE" for item in nonfinite.validate())


def test_row_edit_operations_preserve_csr_invariants():
    csr = generate_csr(10, 8, DistributionSpec("powerlaw", 0, 6), seed=13)
    replaced = csr.replace_row(2, [7, 1, 3], [1.0, -2.0, 4.0]).delete_row(4)
    appended = replaced.append_row([2, 5], [0.5, -0.25], origin=999)
    assert not appended.validate()
    assert appended.rows == csr.rows
    assert tuple(appended.row_columns(0)) == tuple(sorted(appended.row_columns(0)))
    assert appended.row_origins[-1] == 999

