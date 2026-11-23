from __future__ import annotations

from spmm_diag.injections import run_injection_suite


def test_all_fault_injections_are_isolated_and_tagged(sample_bundle):
    csr, b, c, config = sample_bundle
    report = run_injection_suite(csr, b, c, config)
    assert report["all_injections_detected"]
    assert report["isolation"] == {
        "csr_unchanged": True,
        "b_unchanged": True,
        "config_unchanged": True,
    }
    codes = {item["kind"]: item["failure_codes"] for item in report["results"]}
    assert codes["csr_column_out_of_range"] == ["E_CSR_INVALID"]
    assert codes["dense_nonfinite"] == ["E_MATRIX_NONFINITE"]
    assert codes["accumulator_width_invalid"] == ["E_ACC_WIDTH_INVALID"]
    assert codes["block_size_zero"] == ["E_BLOCK_SIZE_INVALID"]
    assert all(item["trace"]["path"] for item in report["results"])
    assert all(item["failures"][0].get("injection_id") for item in report["results"])

