"""并发等价、进程后端、检查点恢复与文件锁边界测试。"""

from __future__ import annotations

import json
import os
import threading

import pytest

from audit_engine.budget import BudgetConfig
from audit_engine.checkpoint import (
    CHECKPOINT_FILENAME,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
    should_rebuild,
)
from audit_engine.errors import E_LOCK
from audit_engine.locking import checkpoint_lock
from audit_engine.models import NormalizationPolicy
from audit_engine.runner import run_batch

POLICIES = [
    NormalizationPolicy(name="strict"),
    NormalizationPolicy(fold_case=False, name="case_sensitive"),
]


def _stable_signature(report: dict) -> dict:
    """抽取与并发度无关的确定性裁决签名。"""
    signature = {}
    for file_result in report["files"]:
        per_record = {}
        for verdict in file_result["verdicts"]:
            key = (verdict["record_id"], verdict["policy_name"])
            per_record[key] = (
                tuple(verdict["codes"]),
                verdict["canonical_key"],
                verdict["reachable_path"],
                verdict["critical"],
                verdict["perm_epsilon"],
            )
        signature[file_result["file_id"]] = per_record
    return signature


def test_serial_thread_equivalence(manifests, tmp_path):
    paths = sorted(manifests.values())
    serial_report, _ = run_batch(
        paths, POLICIES, BudgetConfig(concurrency=1),
        str(tmp_path / "ck_s"), str(tmp_path / "s.json"), backend="serial",
    )
    thread_report, _ = run_batch(
        paths, POLICIES, BudgetConfig(concurrency=4),
        str(tmp_path / "ck_t"), str(tmp_path / "t.json"), backend="thread",
    )
    assert _stable_signature(serial_report) == _stable_signature(thread_report)


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_all_concurrency_levels_equivalent(manifests, tmp_path, workers):
    paths = sorted(manifests.values())
    reference, _ = run_batch(
        paths, POLICIES, BudgetConfig(concurrency=1),
        str(tmp_path / f"ref_{workers}"), str(tmp_path / f"ref_{workers}.json"),
        backend="serial",
    )
    candidate, _ = run_batch(
        paths, POLICIES, BudgetConfig(concurrency=workers),
        str(tmp_path / f"cand_{workers}"), str(tmp_path / f"cand_{workers}.json"),
        backend="thread",
    )
    assert _stable_signature(reference) == _stable_signature(candidate)


def test_process_backend_runs(manifests, tmp_path):
    paths = [manifests["manifest_budget"], manifests["manifest_traversal"]]
    report, _ = run_batch(
        paths, POLICIES, BudgetConfig(concurrency=2, per_file_timeout_seconds=20),
        str(tmp_path / "ck_p"), str(tmp_path / "p.json"), backend="process",
    )
    assert len(report["files"]) == 2


def test_checkpoint_resume_skips_completed(manifests, tmp_path):
    paths = sorted(manifests.values())
    ck_dir = tmp_path / "ck_resume"
    report_path = tmp_path / "r.json"
    first, _ = run_batch(
        paths[:3], POLICIES, BudgetConfig(concurrency=1),
        str(ck_dir), str(report_path), backend="serial",
    )
    assert len(first["files"]) == 3

    second, _ = run_batch(
        paths, POLICIES, BudgetConfig(concurrency=1),
        str(ck_dir), str(tmp_path / "r2.json"), backend="serial",
    )
    assert len(second["files"]) == len(paths)
    with open(os.path.join(ck_dir, CHECKPOINT_FILENAME), encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    assert len(checkpoint["completed_files"]) == len(paths)


def test_checkpoint_version_change_forces_rebuild(manifests, tmp_path):
    ck_dir = tmp_path / "ck_ver"
    save_checkpoint(str(ck_dir), build_checkpoint([], {}, POLICIES))
    import audit_engine.checkpoint as cp
    original = cp.CHECKPOINT_STRUCT_VERSION
    try:
        cp.CHECKPOINT_STRUCT_VERSION = original + 100
        loaded = load_checkpoint(str(ck_dir), POLICIES)
        assert should_rebuild(loaded) is True
    finally:
        cp.CHECKPOINT_STRUCT_VERSION = original


def test_policy_change_forces_rebuild(manifests, tmp_path):
    ck_dir = tmp_path / "ck_pol"
    save_checkpoint(str(ck_dir), build_checkpoint([], {}, POLICIES))
    changed = [NormalizationPolicy(fold_case=False, name="strict")]
    loaded = load_checkpoint(str(ck_dir), changed)
    assert should_rebuild(loaded) is True


def test_lock_conflict_returns_e_lock(tmp_path):
    lock_path = str(tmp_path / "cp.lock")
    errors: list[Exception] = []

    def second_holder():
        try:
            with checkpoint_lock(lock_path):
                pass
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    with checkpoint_lock(lock_path):
        thread = threading.Thread(target=second_holder)
        thread.start()
        thread.join(timeout=5)
    assert errors and getattr(errors[0], "code", None) == E_LOCK


def test_report_written_atomically(manifests, tmp_path):
    report_path = tmp_path / "report.json"
    run_batch(
        [sorted(manifests.values())[0]], POLICIES, BudgetConfig(concurrency=1),
        str(tmp_path / "ck_atom"), str(report_path), backend="serial",
    )
    assert report_path.exists()
    leftovers = [name for name in os.listdir(tmp_path) if name.endswith(".part")]
    assert leftovers == []
    with open(report_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["files"][0]["verdicts"]


def test_duplicate_file_id_rejected(manifests, tmp_path):
    import shutil

    dup_dir = tmp_path / "dup"
    dup_dir.mkdir()
    source = manifests["manifest_budget"]
    shutil.copy(source, dup_dir / "a.json")
    shutil.copy(source, dup_dir / "b.json")
    with pytest.raises(ValueError, match="file_id"):
        run_batch(
            [str(dup_dir / "a.json"), str(dup_dir / "b.json")],
            POLICIES,
            BudgetConfig(concurrency=1),
            str(tmp_path / "ck_dup"),
            str(tmp_path / "dup.json"),
            backend="serial",
        )
