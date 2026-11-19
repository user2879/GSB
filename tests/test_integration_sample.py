from __future__ import annotations

from pathlib import Path

from feature_diag.config import load_experiment
from feature_diag.errors import E_BOUNDARY_SENSITIVE
from feature_diag.evaluation import run_experiment


def test_sample_workbench_demonstrates_precision_and_boundary_sensitivity(tmp_path) -> None:
    experiment = load_experiment(Path("sample/experiment.yaml"))
    result = run_experiment(
        experiment,
        output_dir=tmp_path / "report",
        cache_dir=tmp_path / "cache",
        snapshot_dir=tmp_path / "snapshot",
        max_workers=2,
    )
    assert result.report is not None
    report = result.report
    assert report["summary"]["evaluated_task_count"] == 27
    assert report["summary"]["precision_sensitive_sample_count"] > 0
    assert report["summary"]["boundary_sensitive_count"] > 0
    assert any(item["business_code"] == E_BOUNDARY_SENSITIVE for item in report["boundary_diagnostics"])
    assert set(report["snapshot"]["resumed_stages"]) == set()
    assert set(report["snapshot"]["computed_stages"]) == {
        "crossing",
        "reductions",
        "metrics",
        "diagnostics",
    }
    for row in report["scheme_metrics"]:
        assert [item["stage"] for item in row["stage_summary"]] == [
            "cross_encoding",
            "standardization",
            "projection",
            "final_metric",
        ]
        assert len(row["stage_deltas"]) == row["sample_size"]

    resumed = run_experiment(
        experiment,
        output_dir=tmp_path / "report-resumed",
        cache_dir=tmp_path / "cache",
        snapshot_dir=tmp_path / "snapshot",
        max_workers=2,
    )
    assert resumed.report is not None
    assert resumed.report["analysis_fingerprint"] == report["analysis_fingerprint"]
    assert resumed.report["snapshot"]["recovery_point"] == "diagnostics"
    assert set(resumed.report["snapshot"]["resumed_stages"]) == {
        "crossing",
        "reductions",
        "metrics",
        "diagnostics",
    }
