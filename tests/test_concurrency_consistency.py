from __future__ import annotations

from feature_diag.config import load_experiment
from feature_diag.evaluation import run_experiment


def test_parallel_and_serial_reports_are_field_identical(make_experiment, tmp_path) -> None:
    serial_config = load_experiment(
        make_experiment(cache_dir="serial-cache", snapshot_dir="serial-snapshot", max_workers=3)
    )
    parallel_config = load_experiment(
        make_experiment(cache_dir="parallel-cache", snapshot_dir="parallel-snapshot", max_workers=3)
    )
    assert serial_config.config_hash == parallel_config.config_hash

    serial = run_experiment(
        serial_config,
        output_dir=tmp_path / "serial",
        cache_dir=tmp_path / "serial-cache",
        snapshot_dir=tmp_path / "serial-snapshot",
        max_workers=1,
    )
    parallel = run_experiment(
        parallel_config,
        output_dir=tmp_path / "parallel",
        cache_dir=tmp_path / "parallel-cache",
        snapshot_dir=tmp_path / "parallel-snapshot",
        max_workers=3,
    )
    assert serial.report is not None and parallel.report is not None
    assert serial.report["analysis_fingerprint"] == parallel.report["analysis_fingerprint"]
    assert serial.report["scheme_metrics"] == parallel.report["scheme_metrics"]
    assert serial.report["precision_sensitive"] == parallel.report["precision_sensitive"]
    assert serial.report["boundary_diagnostics"] == parallel.report["boundary_diagnostics"]
    assert serial.report["failures"] == parallel.report["failures"]
    assert serial.report["cross_diagnostics"] == parallel.report["cross_diagnostics"]
    assert serial.report["reduction_diagnostics"] == parallel.report["reduction_diagnostics"]
    assert serial.report["cache_ledger"]["entries"] == parallel.report["cache_ledger"]["entries"]
    assert serial.report["execution"]["max_workers_effective"] == 1
    assert parallel.report["execution"]["max_workers_effective"] == 3
    assert parallel.report["execution"]["max_workers_effective"] <= parallel_config.max_workers
