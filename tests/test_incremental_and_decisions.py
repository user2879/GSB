from __future__ import annotations

import json
from pathlib import Path

from aedwb.decision import decide_probabilities
from aedwb.pipeline import run_workbench
from aedwb.samplegen import generate_sample


def _run(sample: Path, state: Path) -> dict:
    return run_workbench(sample, state, state / "report.json")


def _stage_status(report: dict, packet_id: str) -> dict[str, str]:
    entry = next(item for item in report["ledger"]["entries"] if item["packet_id"] == packet_id)
    return {item["stage"]: item["status"] for item in entry["stage_ledger"]}


def _config(sample: Path) -> dict:
    return json.loads((sample / "config.json").read_text(encoding="utf-8"))


def _save_config(sample: Path, config: dict) -> None:
    (sample / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_rerun_reuses_all_stages(tmp_path: Path):
    sample = tmp_path / "sample"
    state = tmp_path / "state"
    generate_sample(sample)
    first = _run(sample, state)
    second = _run(sample, state)

    assert first["consistency_signature"] == second["consistency_signature"]
    assert second["run"]["snapshot_status"]["status"] == "resumed"
    assert second["ledger"]["cache_summary"]["computed_stage_count"] == 0
    assert second["ledger"]["cache_summary"]["reused_stage_count"] == 21 * 6

    (state / "manifest.json").unlink()
    partial = _run(sample, state)
    assert partial["run"]["snapshot_status"]["status"] == "partial_resume"
    assert partial["ledger"]["cache_summary"]["computed_stage_count"] == 0
    assert partial["consistency_signature"] == first["consistency_signature"]


def test_threshold_and_quantum_invalidate_only_downstream_stages(tmp_path: Path):
    sample = tmp_path / "sample"
    state = tmp_path / "state"
    generate_sample(sample)
    first = _run(sample, state)

    config = _config(sample)
    config["decision_threshold"] = 0.85
    config["probability_quantum"] = 0.20
    _save_config(sample, config)
    changed = _run(sample, state)

    assert changed["run"]["snapshot_status"]["error_code"] == "E_SNAPSHOT_STALE"
    assert "config_hash changed" in changed["run"]["snapshot_status"]["reasons"]
    assert changed["decision_counts"] != first["decision_counts"]
    assert changed["boundary_samples"] != first["boundary_samples"]
    statuses = _stage_status(changed, "a0_base_new")
    assert statuses["parse"] == "reused"
    assert statuses["map"] == "reused"
    assert statuses["align"] == "reused"
    assert statuses["calibrate"] == "reused"
    assert statuses["decide"] == "computed"
    assert statuses["drift"] == "reused"


def test_packet_replacement_is_local_and_matches_fresh_full_run(tmp_path: Path):
    sample = tmp_path / "sample"
    state = tmp_path / "state"
    generate_sample(sample)
    first = _run(sample, state)

    packet_path = next((sample / "packets").glob("*_a0_drift_new.json"))
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["frames"][0] = [0.55, 0.15, 0.15, 0.15]
    packet["frames"][1] = [0.10, 0.60, 0.20, 0.10]
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    incremental = _run(sample, state)
    assert incremental["run"]["snapshot_status"]["status"] == "resumed"
    assert _stage_status(incremental, "a0_drift_new") == {
        "parse": "computed",
        "map": "computed",
        "align": "computed",
        "calibrate": "computed",
        "decide": "computed",
        "drift": "computed",
    }
    assert _stage_status(incremental, "a0_base_new") == {
        "parse": "reused",
        "map": "reused",
        "align": "reused",
        "calibrate": "reused",
        "decide": "reused",
        "drift": "reused",
    }
    assert incremental["consistency_signature"] != first["consistency_signature"]

    fresh_state = tmp_path / "fresh-state"
    fresh = _run(sample, fresh_state)
    assert fresh["consistency_signature"] == incremental["consistency_signature"]
    assert fresh["global_frames"] == incremental["global_frames"]
    assert fresh["drift_alerts"] == incremental["drift_alerts"]
    assert fresh["boundary_samples"] == incremental["boundary_samples"]
    assert [item["sort_key"] for item in fresh["failures"]] == [
        item["sort_key"] for item in incremental["failures"]
    ]
    assert [
        (entry["packet_id"], [stage["stage"] for stage in entry["stage_ledger"]])
        for entry in fresh["ledger"]["entries"]
    ] == [
        (entry["packet_id"], [stage["stage"] for stage in entry["stage_ledger"]])
        for entry in incremental["ledger"]["entries"]
    ]
    assert [
        (entry["packet_id"], [(stage["stage"], stage["cache_key"]) for stage in entry["stage_ledger"]])
        for entry in fresh["ledger"]["entries"]
    ] == [
        (entry["packet_id"], [(stage["stage"], stage["cache_key"]) for stage in entry["stage_ledger"]])
        for entry in incremental["ledger"]["entries"]
    ]


def test_global_version_change_marks_snapshot_stale_and_recomputes_all(tmp_path: Path):
    sample = tmp_path / "sample"
    state = tmp_path / "state"
    generate_sample(sample)
    _run(sample, state)

    config = _config(sample)
    config["global_version"] = "g-next-version"
    _save_config(sample, config)
    changed = _run(sample, state)

    assert changed["run"]["snapshot_status"]["error_code"] == "E_SNAPSHOT_STALE"
    assert "global_version changed" in changed["run"]["snapshot_status"]["reasons"]
    assert changed["ledger"]["cache_summary"]["reused_stage_count"] == 0
    assert changed["ledger"]["cache_summary"]["computed_stage_count"] == 21 * 6


def test_priority_tie_break_and_boundary_detection():
    config = {
        "classes": ["speech", "music", "noise", "alarm"],
        "class_priority": ["alarm", "speech", "music", "noise"],
        "probability_quantum": 0.05,
        "decision_threshold": 0.40,
    }
    result = decide_probabilities([0.40, 0.40, 0.10, 0.10], config)
    assert result["boundary_sample"] is True
    assert result["decision"] == "speech"
    assert "quantum_tie_priority" in result["boundary_reasons"]

    result = decide_probabilities([0.15, 0.15, 0.45, 0.45], config)
    assert result["decision"] == "alarm"
    assert result["boundary_sample"] is True

    quantum_coarse = {**config, "decision_threshold": 0.30, "probability_quantum": 0.05}
    quantum_fine = {**config, "decision_threshold": 0.30, "probability_quantum": 0.01}
    coarse = decide_probabilities([0.34, 0.36, 0.15, 0.15], quantum_coarse)
    fine = decide_probabilities([0.34, 0.36, 0.15, 0.15], quantum_fine)
    assert coarse["decision"] == "speech"
    assert fine["decision"] == "music"
    assert coarse["boundary_sample"] is True
    assert fine["boundary_sample"] is False

    threshold_equal = decide_probabilities([0.40, 0.30, 0.20, 0.10], config)
    threshold_above = decide_probabilities(
        [0.40, 0.30, 0.20, 0.10], {**config, "decision_threshold": 0.30}
    )
    assert threshold_equal["boundary_sample"] is True
    assert threshold_above["boundary_sample"] is False
