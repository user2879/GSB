from __future__ import annotations

import json

from rerank_lab.engine import run_experiment
from tests.conftest import patch_config


def test_repeated_clean_runs_are_byte_identical(tmp_path):
    report_a = run_experiment("samples/default", tmp_path / "a")
    report_b = run_experiment("samples/default", tmp_path / "b")
    text_a = json.dumps(report_a, ensure_ascii=False, sort_keys=True)
    text_b = json.dumps(report_b, ensure_ascii=False, sort_keys=True)
    assert text_a == text_b
    assert report_a["consistency"]["fresh_cold_match"] is True
    assert report_a["summary"]["status"] == "ok"


def test_stage_invalidation_and_reuse_are_precise(sample_copy, tmp_path):
    sample = sample_copy("default")
    patch_config(sample, cache_residency_budget=64)
    report = run_experiment(sample, tmp_path / "out")
    command_one = report["commands"][0]
    command_four = report["commands"][3]
    assert command_one["command"]["type"] == "set_weights"
    for telemetry in command_one["evaluation"]["request_cache_telemetry"].values():
        assert telemetry["cache_reuse"]["normalization"] is True
        assert telemetry["recomputed"]["normalization"] is False
        assert telemetry["recomputed"]["fusion"]
        assert telemetry["recomputed"]["rerank"] is True

    assert command_four["command"]["type"] == "set_rerank"
    for telemetry in command_four["evaluation"]["request_cache_telemetry"].values():
        assert telemetry["cache_reuse"]["normalization"] is True
        assert telemetry["recomputed"]["fusion"] == []
        assert telemetry["recomputed"]["rerank"] is True

    for request in report["requests"]:
        assert request["cache_keys"]["normalization"].startswith("sha256:")
        assert request["cache_keys"]["fusion"]
        assert request["cache_keys"]["rerank"].startswith("sha256:")
        assert request["cache_keys"]["final_sort"].startswith("sha256:")
        assert all(
            ref["feature_dimension"] == request["feature_dimension"]
            and ref["candidate_ids"] == request["candidate_ids"]
            for ref in request["cache_refs"]["fusion"]
        )
        for score in request["fusion"]["candidates"].values():
            assert (
                sum(c["contribution_fixed"] for c in score["contributions"])
                == score["raw_score_fixed"]
            )


def test_rerank_output_and_final_order_are_self_consistent(tmp_path):
    report = run_experiment("samples/default", tmp_path / "out")
    for request in report["requests"]:
        rerank = request["rerank"]
        assert rerank["order"] == [
            *rerank["window_final_order"],
            *rerank["tail_order"],
        ]
        assert [row["item_id"] for row in request["scores"]] == request["candidate_ids"]
        for event in rerank["reductions"]:
            assert event["new_score_fixed"] >= rerank["scores_after_rerank"].get(
                event["item_id"], rerank["lower_bound_fixed"]
            ) or event["new_score_fixed"] == rerank["lower_bound_fixed"]
            assert event["cache_key"].startswith("sha256:")


def test_snapshot_manifest_has_required_fields(tmp_path):
    output = tmp_path / "out"
    report = run_experiment("samples/default", output)
    manifest = json.loads((output / "snapshot" / "manifest.json").read_text())
    assert manifest["global_version"] == report["run"]["final_version"]
    assert manifest["config_hash"] == report["input"]["config_hash"]
    assert manifest["feature_groups_hash"] == report["input"]["feature_groups_hash"]
    assert manifest["completed_request_cursor"] == "r-004"
    assert set(manifest["stage_artifacts"]) == {
        "validation",
        "normalization",
        "fusion",
        "rerank",
    }
    assert not (output / ".tmp-snapshot").exists()
    assert not (output / ".snapshot-old").exists()
