from __future__ import annotations

import json
from pathlib import Path

import pytest

from cross_coding.cli import main
from cross_coding.pipeline import comparison_view, run_pipeline
from cross_coding.store import SnapshotStore, SnapshotVersionConflict


def test_concurrent_and_serial_reports_match(bundle) -> None:
    serial = run_pipeline(bundle, parallel=False)
    concurrent = run_pipeline(bundle, parallel=True)
    assert comparison_view(serial) == comparison_view(concurrent)
    assert concurrent["execution"]["configured_max_workers"] == bundle["policy"]["max_workers"]
    assert concurrent["execution"]["observed_max_concurrency"] <= bundle["policy"]["max_workers"]
    assert concurrent["execution"]["thread_pool_max_workers"] <= bundle["policy"]["max_workers"]


def test_snapshot_commit_versions_conflict_and_recovery(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "state")
    first = store.commit(
        stage_payloads={"enumeration": {"combinations": []}},
        policy_hash="policy:1",
        width_profile={"encoding_width_bits": 8},
        command_cursor=1,
        expected_version=0,
    )
    assert first["global_version"] == 1
    second = store.commit(
        stage_payloads={"enumeration": {"combinations": [1]}},
        policy_hash="policy:1",
        width_profile={"encoding_width_bits": 8},
        command_cursor=2,
        expected_version=1,
    )
    assert second["global_version"] == 2

    with pytest.raises(SnapshotVersionConflict) as exc_info:
        store.commit(
            stage_payloads={},
            policy_hash="policy:1",
            width_profile={},
            command_cursor=3,
            expected_version=1,
        )
    assert exc_info.value.code == "E_VERSION_CONFLICT"

    store.manifest_path.write_text("{broken", encoding="utf-8")
    recovery = store.recover()
    assert recovery["status"] == "recovered_from_snapshot"
    assert recovery["global_version"] == 2
    recovered = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert recovered["global_version"] == 2


def test_cli_run_and_injection_command(tmp_path, capsys) -> None:
    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "run",
            "--features",
            "config/features.yaml",
            "--policy",
            "config/policy.yaml",
            "--state",
            str(tmp_path / "state"),
            "--report",
            str(report_path),
        ]
    )
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["manifest"]["global_version"] == 1
    assert report_path.with_suffix(".md").exists()

    injection_out = tmp_path / "injections.json"
    code = main(["inject", "--out", str(injection_out), "--report-dir", str(tmp_path)])
    assert code == 0
    injection = json.loads(injection_out.read_text(encoding="utf-8"))
    assert injection["ok"] is True
    assert injection["business_code_counts"]["E_CONFIG_INVALID"] == 3


def test_invalid_cli_writes_failure_report(tmp_path) -> None:
    report_path = tmp_path / "invalid.json"
    code = main(
        [
            "run",
            "--features",
            "config/injections/invalid_width/features.yaml",
            "--policy",
            "config/injections/invalid_width/policy.yaml",
            "--state",
            str(tmp_path / "state"),
            "--report",
            str(report_path),
        ]
    )
    assert code == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "config_invalid"
    assert report["failures"][0]["code"] == "E_CONFIG_INVALID"
    assert report["failures"][0]["detail_code"] == "E_WIDTH_NOT_ALLOWED"
