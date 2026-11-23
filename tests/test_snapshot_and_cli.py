from __future__ import annotations

import json
import shutil
from pathlib import Path

from motion_pipeline.cli import main

from .helpers import SAMPLE_CONFIG, SAMPLE_SEQUENCES, run_pipeline, write_config


def test_snapshot_recovery_stale_detection_and_temp_cleanup(tmp_path: Path) -> None:
    state = tmp_path / "state"
    config = write_config(tmp_path / "config.yaml")
    first = run_pipeline(
        config_path=config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-first",
        fresh=True,
    )
    manifest_path = state / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["global_version"] >= 1
    assert manifest["completed_sequences"] == ["seq-0001", "seq-0002"]
    assert set(manifest["stage_versions"]) == {"encode", "motion", "predict", "decode"}
    assert manifest["config_hash"] == first.diagnostic_report["snapshot"]["config_hash"]

    orphan = state / "tmp" / "orphan-directory"
    orphan.mkdir(parents=True)
    (orphan / "partial.txt").write_text("partial", encoding="utf-8")
    resumed = run_pipeline(
        config_path=config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-resumed",
        expected_version=manifest["global_version"],
    )
    assert resumed.diagnostic_report["stage_cache_totals"] == {"hit": 8, "recompute": 0}
    assert resumed.diagnostic_report["snapshot"]["stale_events"] == []
    assert not orphan.exists()

    worker_config = write_config(
        tmp_path / "config-workers.yaml",
        lambda raw: raw.__setitem__("thread_pool_workers", 2),
    )
    worker_change = run_pipeline(
        config_path=worker_config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-worker-change",
    )
    assert worker_change.diagnostic_report["stage_cache_totals"] == {
        "hit": 8,
        "recompute": 0,
    }
    assert worker_change.diagnostic_report["snapshot"]["stale_events"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_hash"] == worker_change.diagnostic_report["snapshot"]["config_hash"]
    worker_rerun = run_pipeline(
        config_path=worker_config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-worker-rerun",
    )
    assert worker_rerun.diagnostic_report["snapshot"]["stale_events"] == []
    assert worker_rerun.diagnostic_report["stage_cache_totals"] == {"hit": 8, "recompute": 0}

    current_version = worker_rerun.diagnostic_report["snapshot"]["global_version"]
    mismatched = run_pipeline(
        config_path=config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-mismatched",
        expected_version=current_version - 1,
    )
    assert any(
        event["business_code"] == "E_SNAPSHOT_STALE"
        and event["reason"] == "EXPECTED_VERSION_MISMATCH"
        for event in mismatched.diagnostic_report["snapshot"]["stale_events"]
    )
    assert mismatched.diagnostic_report["stage_cache_totals"]["recompute"] == 8

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    predict_version = manifest["stage_versions"]["predict"]
    shutil.rmtree(state / "stages" / "predict" / f"{predict_version:08d}")
    missing_stage = run_pipeline(
        config_path=config,
        sequence_path=SAMPLE_SEQUENCES,
        tier=1,
        state_dir=state,
        output_dir=tmp_path / "out-missing-stage",
    )
    assert any(
        event["reason"] == "STAGE_SNAPSHOT_MISSING"
        for event in missing_stage.diagnostic_report["snapshot"]["stale_events"]
    )
    assert missing_stage.diagnostic_report["stage_cache"]["predict"]["recompute"] == 2


def test_cli_emits_reports_and_keeps_invalid_sequence_out_of_results(
    tmp_path: Path, capsys
) -> None:
    lines = SAMPLE_SEQUENCES.read_text(encoding="utf-8").splitlines()
    lines.append('{"sequence_id":"seq-0001","frames":[]}')
    sequences = tmp_path / "sequences-with-error.jsonl"
    sequences.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = main(
        [
            "run",
            "--config",
            str(SAMPLE_CONFIG),
            "--sequences",
            str(sequences),
            "--tier",
            "1",
            "--state-dir",
            str(tmp_path / "state"),
            "--output-dir",
            str(output_dir),
            "--fresh",
        ]
    )
    capsys.readouterr()

    assert exit_code == 0
    diagnostic = json.loads((output_dir / "diagnostic_report.json").read_text(encoding="utf-8"))
    failures = json.loads((output_dir / "failures.json").read_text(encoding="utf-8"))
    predictions = json.loads((output_dir / "predictions.json").read_text(encoding="utf-8"))
    assert diagnostic["failure_business_codes"] == ["E_SEQ_INVALID"]
    assert len(failures) == 1
    assert failures[0]["sequence_id"] == "seq-0001"
    assert failures[0]["business_code"] == "E_SEQ_INVALID"
    assert [row["sequence_id"] for row in predictions["sequences"]] == [
        "seq-0001",
        "seq-0002",
    ]
    assert "selected_strategy" in diagnostic
    assert "quantization_level" in diagnostic
    assert "stage_cache" in diagnostic
    assert "per_sequence_prediction_error" in diagnostic
    assert "monotonic_clock" in diagnostic
