from __future__ import annotations

from pathlib import Path

from feature_diag.config import load_experiment
from feature_diag.evaluation import run_experiment
from feature_diag.errors import E_SNAPSHOT_STALE


def test_partial_stage_recovery_ignores_incomplete_temp_artifacts(make_experiment, tmp_path) -> None:
    config = load_experiment(make_experiment(snapshot_dir="recovery-snapshot", cache_dir="recovery-cache"))
    partial = run_experiment(
        config,
        output_dir=tmp_path / "partial",
        cache_dir=tmp_path / "recovery-cache",
        snapshot_dir=tmp_path / "recovery-snapshot",
        max_workers=3,
        stop_after_stage="crossing",
    )
    assert partial.partial is True
    assert partial.snapshot_info["computed_stages"] == ["crossing"]

    orphan = tmp_path / "recovery-snapshot" / ".tmp" / "crossing-incomplete"
    orphan.mkdir(parents=True)
    (orphan / "state.json").write_text("{not valid json", encoding="utf-8")

    resumed = run_experiment(
        config,
        output_dir=tmp_path / "resumed",
        cache_dir=tmp_path / "recovery-cache",
        snapshot_dir=tmp_path / "recovery-snapshot",
        max_workers=3,
    )
    assert resumed.report is not None
    snapshot = resumed.report["snapshot"]
    assert snapshot["resumed_stages"][0] == "crossing"
    assert "crossing-incomplete" in snapshot["ignored_incomplete_temp_entries"]
    assert set(snapshot["computed_stages"]) == {"reductions", "metrics", "diagnostics"}
    assert resumed.report["summary"]["cache"]["snapshot_reuse"] > 0


def test_version_change_refuses_stale_snapshot(make_experiment, tmp_path) -> None:
    first_config = load_experiment(make_experiment(version=1, snapshot_dir="stale-snapshot"))
    first = run_experiment(
        first_config,
        output_dir=tmp_path / "fresh",
        cache_dir=tmp_path / "stale-cache",
        snapshot_dir=tmp_path / "stale-snapshot",
        max_workers=3,
    )
    assert first.report is not None
    assert first.report["snapshot"]["stale_event"] is None

    second_config = load_experiment(make_experiment(version=2, snapshot_dir="stale-snapshot"))
    second = run_experiment(
        second_config,
        output_dir=tmp_path / "stale",
        cache_dir=tmp_path / "stale-cache",
        snapshot_dir=tmp_path / "stale-snapshot",
        max_workers=3,
    )
    assert second.report is not None
    snapshot = second.report["snapshot"]
    assert snapshot["stale_event"]["code"] == E_SNAPSHOT_STALE
    assert snapshot["resumed_stages"] == []
    assert any(failure["code"] == E_SNAPSHOT_STALE for failure in second.report["failures"])
