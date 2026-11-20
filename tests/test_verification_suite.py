from __future__ import annotations

from pathlib import Path

from pref_reward_lab.verify import run_verification


def test_edit_sequence_matches_full_rebuild_and_roundtrips(tmp_path: Path) -> None:
    report = run_verification(tmp_path / "verification", pair_count=30, seed=5)
    assert report["all_incremental_full_matches"] is True
    assert report["concurrency_serial_parallel_match"] is True
    assert report["conversion_roundtrip"]["restore_mismatch_pair_ids"] == []
    assert report["conversion_roundtrip"]["sidecar_all_discarded_fields_recorded"] is True
    assert report["crash_simulation"]["termination_observed"] is True
    assert report["crash_simulation"]["recovered_from_complete_snapshot"] is True
    assert len(report["steps"]) == 9

