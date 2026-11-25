from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from conftest import EXAMPLES, ROOT
from ins_diag.cli import main
from ins_diag.pipeline import run_pipeline


def test_all_four_faults_are_isolated_and_mapped(tmp_path: Path) -> None:
    output = tmp_path / "faults.json"
    exit_code = main(
        [
            "faults",
            "--root",
            str(ROOT / "examples" / "faults"),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    by_name = {item["fault"]: item for item in report["faults"]}
    assert by_name["cross_window_reference"]["hit"] is True
    assert by_name["invalid_kernel_bandwidth"]["hit"] is True
    assert by_name["non_finite_sample"]["hit"] is True
    assert by_name["lease_generation_rollback"]["hit"] is True
    assert by_name["cross_window_reference"]["status"] == "partial"
    assert by_name["non_finite_sample"]["failed_windows"] == 2
    assert by_name["invalid_kernel_bandwidth"]["status"] == "error"
    assert by_name["lease_generation_rollback"]["status"] == "failed"


def test_non_finite_window_failure_does_not_abort_other_windows(tmp_path: Path) -> None:
    fault_dir = ROOT / "examples" / "faults" / "non_finite_sample"
    report = run_pipeline(
        fault_dir / "config.yaml",
        state_dir=tmp_path / "state",
    )
    assert report["status"] == "partial"
    assert report["data"]["window_count"] == 10
    assert len(report["failed_windows"]) == 2
    assert report["clusters"] is not None
    assert report["density"] is not None
    assert any(
        error["code"] == "E_NONFINITE_SAMPLE"
        for error in report["business_errors"]
    )
    assert len([window for window in report["windows"] if window["status"] == "ok"]) == 8
    assert report["failed_windows"] == sorted(
        report["failed_windows"],
        key=lambda error: (error["stage"], error["path"], error["code"]),
    )


def test_cross_window_edit_failure_is_reported_but_other_windows_run(
    tmp_path: Path,
) -> None:
    fault_dir = ROOT / "examples" / "faults" / "cross_window_reference"
    report = run_pipeline(fault_dir / "config.yaml", state_dir=tmp_path / "state")
    assert report["status"] == "partial"
    assert report["edits"]["failures"][0]["code"] == "E_WINDOW_REF_OUT_OF_RANGE"
    assert len(report["windows"]) == 10
    assert report["clusters"] is not None
    assert report["snapshot"]["commit"]["committed"] is True


def _subprocess_env() -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    src = str(ROOT / "src")
    environment["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return environment


def test_sigkill_restart_reclaims_orphans_and_detects_stale_snapshot(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    baseline = subprocess.run(
        [
            sys.executable,
            "-m",
            "ins_diag",
            "run",
            "--config",
            str(EXAMPLES / "config.yaml"),
            "--state-dir",
            str(state),
            "--output",
            str(tmp_path / "baseline.json"),
        ],
        cwd=ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert baseline.returncode == 0, baseline.stderr

    killed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ins_diag",
            "run",
            "--config",
            str(EXAMPLES / "config.yaml"),
            "--state-dir",
            str(state),
            "--crash-after",
            "after_lease",
        ],
        cwd=ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert killed.returncode in {-9, 137}

    recovered_output = tmp_path / "recovered.json"
    recovered = subprocess.run(
        [
            sys.executable,
            "-m",
            "ins_diag",
            "run",
            "--config",
            str(EXAMPLES / "config.yaml"),
            "--state-dir",
            str(state),
            "--output",
            str(recovered_output),
        ],
        cwd=ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert recovered.returncode == 0, recovered.stderr
    report = json.loads(recovered_output.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["recovery"]["snapshot_present_before_run"] is True
    assert report["recovery"]["restoreable_four_way"] is False
    assert report["recovery"]["resume_scope"] == "cache_only_superseded"
    assert len(report["recovery"]["reclaimed_orphan_leases"]) == 4
    assert all(
        item["reason"] == "owner_process_not_alive"
        for item in report["recovery"]["reclaimed_orphan_leases"]
    )
    assert "E_SNAPSHOT_STALE" in report["error_codes"]
    assert report["snapshot"]["commit"]["committed"] is True
    assert report["cache"]["reused_count"] == 87

