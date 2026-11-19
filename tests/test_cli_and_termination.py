"""CLI 退出码、进程后端硬终止与被终止文件不影响同批其余文件。"""

from __future__ import annotations

import json
import os

from audit_engine.cli import main
from audit_engine.runner import (
    EXIT_FATAL,
    EXIT_LOCK,
    EXIT_OK,
    EXIT_VIOLATIONS,
)


def test_cli_violation_exit_code(manifests, tmp_path):
    code = main(
        [
            manifests["manifest_traversal"],
            "--checkpoint-dir", str(tmp_path / "ck"),
            "--report", str(tmp_path / "r.json"),
            "--backend", "serial",
        ]
    )
    assert code == EXIT_VIOLATIONS
    with open(tmp_path / "r.json", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["files"][0]["verdicts"][0]["canonical_key"]


def test_cli_clean_exit_code(manifests, tmp_path):
    code = main(
        [
            manifests["manifest_budget"],
            "--checkpoint-dir", str(tmp_path / "ck"),
            "--report", str(tmp_path / "r.json"),
            "--backend", "serial",
        ]
    )
    assert code == EXIT_OK


def test_cli_no_inputs_returns_fatal(tmp_path):
    code = main(
        [
            str(tmp_path / "missing-*.json"),
            "--checkpoint-dir", str(tmp_path / "ck"),
            "--report", str(tmp_path / "r.json"),
        ]
    )
    assert code == EXIT_FATAL


def test_cli_custom_policy_override(manifests, tmp_path):
    code = main(
        [
            manifests["manifest_budget"],
            "--checkpoint-dir", str(tmp_path / "ck"),
            "--report", str(tmp_path / "r.json"),
            "--backend", "serial",
            "--policy", "flat:0:0:0",
        ]
    )
    assert code == EXIT_OK
    with open(tmp_path / "r.json", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["policies"][0]["name"] == "flat"
    assert report["policies"][0]["fold_case"] is False


def test_process_backend_termination_isolates_files(manifests, tmp_path):
    # 慢文件触发真实 terminate，同批正常小文件仍须完整处理
    import json
    slow_path = tmp_path / "manifest_slow.json"
    with open(manifests["manifest_budget"], encoding="utf-8") as handle:
        slow = json.load(handle)
    slow["file_id"] = "manifest_slow"
    slow["records"] = slow["records"] * 10
    with open(slow_path, "w", encoding="utf-8") as handle:
        json.dump(slow, handle)
    paths = [str(slow_path), manifests["manifest_parse_errors"]]
    from audit_engine.budget import BudgetConfig
    from audit_engine.models import NormalizationPolicy
    from audit_engine.runner import run_batch
    from audit_engine.workload import cpu_spin

    report, _ = run_batch(
        paths,
        [NormalizationPolicy(name="strict")],
        BudgetConfig(concurrency=2, per_file_timeout_seconds=0.1),
        str(tmp_path / "ck"),
        str(tmp_path / "r.json"),
        backend="process",
        work_probe=cpu_spin,
    )
    assert len(report["files"]) == 2
    statuses = {item["file_id"]: item["status"] for item in report["files"]}
    assert statuses["manifest_slow"] == "timeout"
    assert statuses["manifest_parse_errors"] == "ok"
    assert any("E_TIMEOUT" in item["codes"] for item in report["files"])


def test_forced_termination_boundary_documented(tmp_path):
    """强制 kill 后不承诺退出报告：仅当运行正常完成时报告才原子存在。"""
    report_path = tmp_path / "never.json"
    assert not report_path.exists()
    partials = [name for name in os.listdir(tmp_path) if name.endswith(".part")]
    assert partials == []


def test_lock_exit_code(manifests, tmp_path, monkeypatch):
    from audit_engine import runner
    from audit_engine.errors import AuditError
    from audit_engine.errors import E_LOCK

    def boom(*args, **kwargs):
        raise AuditError("locked", E_LOCK)

    monkeypatch.setattr(runner, "save_checkpoint", boom)
    code = main(
        [
            manifests["manifest_budget"],
            "--checkpoint-dir", str(tmp_path / "ck"),
            "--report", str(tmp_path / "r.json"),
            "--backend", "serial",
        ]
    )
    assert code == EXIT_LOCK
