"""快照版本判定、优雅退出与强制终止后的恢复一致性。"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from conftest import reconstruction_artifacts, run_pipeline
from mvrwb.errors import E_SNAPSHOT_STALE
from mvrwb.snapshot import SNAPSHOT_VERSION
from mvrwb.util import read_json, write_json_atomic


def test_stale_snapshot_version_is_flagged_then_rebuilt(sample_dir, work_dirs):
    report = run_pipeline(sample_dir, work_dirs)
    manifest_path = os.path.join(work_dirs["snapshot"], "manifest.json")
    manifest = read_json(manifest_path)
    # 人为把快照版本改旧
    manifest["snapshot_version"] = SNAPSHOT_VERSION - 1
    write_json_atomic(manifest_path, manifest)

    rerun = run_pipeline(sample_dir, {
        "work": work_dirs["work"],
        "snapshot": work_dirs["snapshot"],
        "report": os.path.join(os.path.dirname(work_dirs["report"]), "stale.json"),
    })
    codes = [failure["code"] for failure in rerun["failures"]]
    assert E_SNAPSHOT_STALE in codes
    stale = next(f for f in rerun["failures"] if f["code"] == E_SNAPSHOT_STALE)
    assert "recovery_view_id" in stale
    # 陈旧快照已被全量重建替换，产物恢复完整
    assert rerun["cloud"]["point_count"] == report["cloud"]["point_count"]


def test_snapshot_manifest_fields(sample_dir, work_dirs):
    run_pipeline(sample_dir, work_dirs)
    manifest = read_json(os.path.join(work_dirs["snapshot"], "manifest.json"))
    assert manifest["snapshot_version"] == SNAPSHOT_VERSION
    assert set(manifest["stage_versions"]) == {"feature", "matching", "depth", "cloud"}
    assert manifest["completed_views"] == sorted(manifest["completed_views"])
    assert isinstance(manifest["consumed_point_budget"], int)
    assert manifest["config_fingerprint"] and manifest["camera_bundle_fingerprint"]


def _cli_env(extra):
    env = dict(os.environ)
    env.update(extra)
    return env


def test_graceful_signal_exit_code_zero_and_resume_matches_full(sample_dir, tmp_path):
    snapshot = str(tmp_path / "snap_grace")
    report_path = str(tmp_path / "grace_report.json")
    work = str(tmp_path / "work_grace")
    cmd = [
        sys.executable, "-m", "mvrwb.cli", "run",
        "--input", sample_dir, "--work", work,
        "--snapshot", snapshot, "--report", report_path,
    ]
    env = _cli_env({"MVR_GRACEFUL_AFTER_STAGE": "feature", "MVR_GRACEFUL_AFTER_VIEW": "3"})
    result = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    partial = read_json(report_path)
    assert partial.get("interrupted") is True
    assert partial["recovery_point_view_id"] >= 3

    # 重启续跑（不带注入环境变量），结果与一次性全量一致
    resume_report_path = str(tmp_path / "resume_report.json")
    cmd[-1] = resume_report_path
    result = subprocess.run(cmd, env=_cli_env({}), capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.dirname(__file__)))
    assert result.returncode == 0, result.stderr
    resumed = read_json(resume_report_path)

    full_snapshot = str(tmp_path / "snap_full")
    full = run_pipeline(sample_dir, {
        "work": str(tmp_path / "work_full"),
        "snapshot": full_snapshot,
        "report": str(tmp_path / "full.json"),
    })
    assert reconstruction_artifacts(resumed) == reconstruction_artifacts(full)


def test_forced_termination_recovers_only_from_committed_snapshot(sample_dir, tmp_path):
    snapshot = str(tmp_path / "snap_kill")
    work = str(tmp_path / "work_kill")
    report_path = str(tmp_path / "kill_report.json")
    cmd = [
        sys.executable, "-m", "mvrwb.cli", "run",
        "--input", sample_dir, "--work", work,
        "--snapshot", snapshot, "--report", report_path,
    ]
    env = _cli_env({"MVR_CRASH_AFTER_STAGE": "depth", "MVR_CRASH_AFTER_VIEW": "2"})
    result = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.dirname(__file__)),
                            capture_output=True, text=True)
    # 强制终止：非零退出且不承诺退出报告
    assert result.returncode != 0
    assert not os.path.exists(report_path)

    # 重启仅依据已完整替换的快照恢复，最终结果与全量一致
    resume_report = str(tmp_path / "kill_resume.json")
    cmd[-1] = resume_report
    result = subprocess.run(cmd, env=_cli_env({}), capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.dirname(__file__)))
    assert result.returncode == 0, result.stderr
    resumed = read_json(resume_report)
    full = run_pipeline(sample_dir, {
        "work": str(tmp_path / "work_full2"),
        "snapshot": str(tmp_path / "snap_full2"),
        "report": str(tmp_path / "full2.json"),
    })
    assert reconstruction_artifacts(resumed) == reconstruction_artifacts(full)
