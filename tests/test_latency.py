from __future__ import annotations

from rerank_lab.engine import run_experiment
from tests.conftest import patch_config


def test_stale_response_references_previous_completed_version(sample_copy, tmp_path):
    sample = sample_copy("default")
    patch_config(
        sample,
        interaction_latency_budget_ms=0.05,
        min_command_batch=5,
        latency_units_per_ms=1000,
    )
    report = run_experiment(sample, tmp_path / "out")
    assert report["commands"]
    for record in report["commands"]:
        response = record["response"]
        assert response["status"] == "stale"
        assert "E_RESULT_STALE" in response["error_codes"]
        assert response["measurement"]["scope"] == "min_command_batch"
        assert response["measurement"]["command_batch_size"] == 5
        assert response["version_source"] != response["requested_version"]
        assert response["computed_version"] == response["requested_version"]


def test_normal_budget_returns_current_summary(tmp_path):
    report = run_experiment("samples/default", tmp_path / "out")
    for record in report["commands"]:
        assert record["response"]["status"] == "current"
        assert record["response"]["version_source"] == record["version"]
