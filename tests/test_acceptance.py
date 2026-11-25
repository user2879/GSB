from __future__ import annotations

from log_sampler.demo import run_verification


def test_full_reproducible_acceptance(sample_bundle, tmp_path):
    report = run_verification(
        config_path=sample_bundle["config_path"],
        batches_path=sample_bundle["batches_path"],
        injections_path=sample_bundle["injections_path"],
        commands_path=sample_bundle["commands_path"],
        state_root=tmp_path / "verification",
        fresh=True,
    )
    assert report["status"] == "ok"
    assert all(report["checks"].values())
    assert report["crash_recovery"]["crash_exit_code"] == 137
    assert report["crash_recovery"]["completed_after_kill"] >= 1
    assert report["crash_recovery"]["resumed_buckets"]
    assert report["replay"]["difference_count"] == 0
    assert report["evidence_roundtrip"]["checks"]["p3_to_p1_to_p3"]["equal"]
    assert report["evidence_roundtrip"]["checks"]["p3_to_p2_to_p3"]["equal"]
    assert report["budget_probe"]["changed"]
    assert report["capacity_probe"]["other_bucket_changes"] == []
    assert report["metrics"]["diagnostic_business_codes"] == [
        "E_D_FLIP_RANDOM_SOURCE",
        "E_D_FLIP_TRUNCATION_TIE",
    ] or set(report["metrics"]["diagnostic_business_codes"]) == {
        "E_D_FLIP_RANDOM_SOURCE",
        "E_D_FLIP_TRUNCATION_TIE",
    }
