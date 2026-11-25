from __future__ import annotations

import copy
import os
import sqlite3
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from vscan.errors import DivergenceError
from vscan.executor import batch_fingerprint
from vscan.ledger import Ledger
from vscan.planner import compile_plan
from vscan.session import Session
from vscan.spec import parse_spec
from vscan.store import build_store


def test_ledger_commit_and_reopen_without_replacement(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    ledger_path = tmp_path / "ledger.sqlite"
    with Session(spec, tmp_path / "data", ledger_path) as session:
        report = session.run()
        snapshot = session.store.snapshot_digest
        assert len(report["batch_results"]) == 3
    ledger = Ledger(ledger_path, snapshot_digest=snapshot, spec_digest="spec")
    try:
        assert ledger.open_result.replaced is False
        records = ledger.committed_records()
        assert len(records) == 3
        assert all(record["state"] == "committed" for record in records.values())
        assert all(record["progress_cursor"] != "{}" for record in records.values())
    finally:
        ledger.close()
    with Session(spec, tmp_path / "data", ledger_path, rebuild_store=False) as resumed:
        report = resumed.run()
        assert report["ledger"]["replaced"] is False
        assert report["incremental"]["recovery_comparisons"] == 3


def test_snapshot_mismatch_replaces_whole_ledger_and_keeps_readonly_backup(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    ledger_path = tmp_path / "ledger.sqlite"
    with Session(spec, tmp_path / "data", ledger_path) as session:
        session.run()
        snapshot = session.store.snapshot_digest
    original_bytes = ledger_path.read_bytes()
    ledger = Ledger(ledger_path, snapshot_digest=snapshot + "-changed", spec_digest="spec")
    try:
        assert ledger.open_result.replaced is True
        assert ledger.open_result.backup_path is not None
        backup = Path(ledger.open_result.backup_path)
        assert backup.exists()
        assert stat.S_IMODE(backup.stat().st_mode) == 0o444
        assert backup.read_bytes() == original_bytes
        assert ledger.committed_records() == {}
    finally:
        ledger.close()


def test_content_digest_mismatch_replaces_whole_ledger(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    ledger_path = tmp_path / "ledger.sqlite"
    with Session(spec, tmp_path / "data", ledger_path) as session:
        session.run()
        snapshot = session.store.snapshot_digest
    conn = sqlite3.connect(ledger_path)
    try:
        conn.execute("UPDATE meta SET value='tampered' WHERE key='content_digest'")
        conn.commit()
    finally:
        conn.close()
    ledger = Ledger(ledger_path, snapshot_digest=snapshot, spec_digest="spec")
    try:
        assert ledger.open_result.replaced is True
        assert ledger.open_result.backup_path is not None
        assert Path(ledger.open_result.backup_path).exists()
        assert ledger.committed_records() == {}
    finally:
        ledger.close()


def test_pending_batch_is_unknown_after_restart(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    ledger_path = tmp_path / "ledger.sqlite"
    with Session(spec, tmp_path / "data", ledger_path) as session:
        session.run()
        snapshot = session.store.snapshot_digest
        task_map = session._build_tasks(session.plan)
        task = task_map[sorted(task_map)[0]]

    ledger = Ledger(ledger_path, snapshot_digest=snapshot, spec_digest="spec")
    try:
        ledger.begin_batch(task.plan, task.fingerprint + "-new")
    finally:
        ledger.close()

    reopened = Ledger(ledger_path, snapshot_digest=snapshot, spec_digest="spec")
    try:
        assert reopened.open_result.replaced is False
        report = reopened.reconcile_expected([(task.plan, task.fingerprint + "-new")])
        assert len(report.unknown) == 1
        assert report.unknown[0].segment_id == task.plan.segment_id
        assert report.unknown[0].batch_no == task.plan.batch_no
    finally:
        reopened.close()


def test_forced_process_exit_leaves_only_committed_groups_recoverable(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    ledger_path = tmp_path / "ledger.sqlite"
    data_root = tmp_path / "data"
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        __import__("json").dumps(spec.canonical_dict(), ensure_ascii=False), encoding="utf-8"
    )
    with Session(spec, data_root, ledger_path) as session:
        session.run()
        snapshot = session.store.snapshot_digest
        task_map = session._build_tasks(session.plan)
        new_task = task_map[sorted(task_map)[0]]

    script = f"""
import os
from pathlib import Path
from vscan.spec import parse_spec
from vscan.store import build_store
from vscan.planner import compile_plan
from vscan.executor import batch_fingerprint
from vscan.ledger import Ledger
spec = parse_spec(__import__('json').loads(Path({str(spec_path)!r}).read_text(encoding='utf-8')))
store = build_store(spec, Path({str(data_root)!r}))
plan = compile_plan(spec)
batch = plan.batches()[0]
fp = batch_fingerprint(spec, store, Path({str(data_root)!r}), batch) + '-uncommitted-attempt'
ledger = Ledger(Path({str(ledger_path)!r}), snapshot_digest=store.snapshot_digest, spec_digest='spec')
ledger.begin_batch(batch, fp)
os._exit(17)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path.cwd())
    completed = subprocess.run([sys.executable, "-c", script], env=env, cwd=Path.cwd())
    assert completed.returncode == 17

    reopened = Ledger(ledger_path, snapshot_digest=snapshot, spec_digest="spec")
    try:
        assert reopened.open_result.replaced is False
        records = reopened.committed_records()
        assert len(records) == 3
        expected = []
        reopened_store = build_store(spec, data_root)
        for batch in compile_plan(spec).batches():
            fingerprint = batch_fingerprint(spec, reopened_store, data_root, batch)
            if batch.key == new_task.plan.key:
                fingerprint += "-uncommitted-attempt"
            expected.append((batch, fingerprint))
        report = reopened.reconcile_expected(expected)
        assert len(report.unknown) == 1
        assert report.unknown[0].batch_no == new_task.plan.batch_no
    finally:
        reopened.close()


def test_verify_results_locates_first_divergence(tmp_path, spec_dict):
    spec = parse_spec(copy.deepcopy(spec_dict))
    with Session(spec, tmp_path / "data", tmp_path / "ledger.sqlite") as session:
        report = session.run()
        first = session.results[sorted(session.results)[0]]
        corrupted = replace(first, fingerprint=first.fingerprint + "-different")
        with pytest.raises(DivergenceError) as caught:
            session.ledger.verify_results([corrupted])
        assert caught.value.code == "E_DIVERGE"
        assert caught.value.detail["field"] == "fingerprint"
        assert caught.value.detail["batch_no"] == first.plan.batch_no
        assert report["ledger"]["format_version"] == 1
