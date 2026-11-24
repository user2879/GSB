"""Ledger, snapshot replacement, crash recovery and replay verification."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, replace

import pytest

from obsflow.canonical import canonical_bytes
from obsflow.config import load_engine_config, load_source_set
from obsflow.engine import ObsFlowEngine
from obsflow.errors import ReplayError, SnapshotError
from obsflow.injection import (
    inject_order_regression,
    inject_snapshot_digest,
    inject_truncation,
)
from obsflow.process_runner import launch_crashed_worker


def _run(declaration_path, window_path, ledger_path):
    engine = ObsFlowEngine.from_files(declaration_path, window_path,
                                      ledger_path)
    engine.read_records()
    report = engine.advance()
    return engine, report


def test_snapshot_replay_field_by_field(sample_dir, workdir):
    ledger = f"{workdir}/ok.db"
    engine, report = _run(sample_dir["declaration_path"],
                          sample_dir["window_path"], ledger)
    persisted = engine.ledger.all_windows()
    engine.close()

    reopened = ObsFlowEngine.from_files(sample_dir["declaration_path"],
                                        sample_dir["window_path"], ledger)
    windows, unknown = reopened.recover()
    assert not unknown
    assert len(windows) == len(persisted)
    reopened.read_records()
    replayed = reopened.replay_verify()
    assert len(replayed) == len(report.windows)
    reopened.close()


def test_force_kill_leaves_unknown_windows_and_rerun(sample_dir, workdir):
    ledger = f"{workdir}/crash.db"
    engine, _ = _run(sample_dir["declaration_path"],
                     sample_dir["window_path"], ledger)
    engine.close()

    result = launch_crashed_worker(ledger, kill_after_ms=200)
    assert result.returncode != 0

    reopened = ObsFlowEngine.from_files(sample_dir["declaration_path"],
                                        sample_dir["window_path"], ledger)
    windows, unknown = reopened.recover()
    assert len(unknown) == len(windows)
    assert all(w.status == "unknown" for w in reopened.query_windows())
    # Rerun the uncommitted work then replay must match field by field.
    reopened.read_records()
    reopened.advance()
    reopened.replay_verify()
    assert all(w.status == "ready" for w in reopened.query_windows())
    reopened.close()


def test_snapshot_digest_tamper_is_snapshot_error(sample_dir, workdir):
    ledger = f"{workdir}/tamper.db"
    engine, _ = _run(sample_dir["declaration_path"],
                     sample_dir["window_path"], ledger)
    engine.close()
    inject_snapshot_digest(ledger)
    reopened = ObsFlowEngine.from_files(sample_dir["declaration_path"],
                                        sample_dir["window_path"], ledger)
    reopened.recover()
    with pytest.raises(SnapshotError) as info:
        reopened.verify_snapshot()
    assert info.value.code == "E_SNAPSHOT"
    reopened.close()


def test_version_evolution_replaces_state_with_backup(sample_dir, workdir):
    ledger = f"{workdir}/evolve.db"
    engine, _ = _run(sample_dir["declaration_path"],
                     sample_dir["window_path"], ledger)
    engine.close()

    source_set = load_source_set(sample_dir["declaration_path"])
    changed = replace(source_set.sources[0], format_version=2)
    evolved = source_set.__class__(
        source_set_version=99,
        sources=(changed,) + source_set.sources[1:],
        base_dir=source_set.base_dir)
    config = load_engine_config(sample_dir["window_path"])
    reopened = ObsFlowEngine(config, evolved, ledger_path=ledger,
                             backup_dir=f"{workdir}/backups")
    windows, unknown = reopened.recover()
    assert windows == [] and unknown == []
    assert reopened.ledger.all_windows() == []
    backups = list(__import__("pathlib").Path(f"{workdir}/backups").glob("*.db"))
    assert backups and not __import__("os").access(backups[0], __import__("os").W_OK)
    reopened.close()


def test_replay_mismatch_locates_metric_window_field(sample_dir, workdir,
                                                     monkeypatch):
    ledger = f"{workdir}/mismatch.db"
    engine, _ = _run(sample_dir["declaration_path"],
                     sample_dir["window_path"], ledger)
    conn = sqlite3.connect(ledger)
    conn.execute("UPDATE windows SET sum_q = sum_q + 1 LIMIT 1")
    conn.commit()
    conn.close()

    reopened = ObsFlowEngine.from_files(sample_dir["declaration_path"],
                                        sample_dir["window_path"], ledger)
    reopened.recover()
    reopened.read_records()
    with pytest.raises(ReplayError) as info:
        reopened.replay_verify()
    assert info.value.code == "E_REPLAY"
    assert info.value.field == "sum_q"
    assert "metric_key" in info.value.detail
    reopened.close()


def test_injection_chain_business_codes(sample_dir, workdir):
    config = load_source_set(sample_dir["declaration_path"])
    first = config.sources[0]
    original = f"{sample_dir['output_dir']}/{first.file_path}"

    truncated = inject_truncation(
        original, f"{workdir}/truncated.jsonl")
    regressed = inject_order_regression(
        original, f"{workdir}/regressed.jsonl")

    from obsflow.codec import load_stream
    from obsflow.errors import IntegrityError, OrderError
    with pytest.raises(IntegrityError) as info:
        load_stream(truncated, first)
    assert info.value.code == "E_INTEGRITY"
    with pytest.raises(OrderError) as info:
        load_stream(regressed, first)
    assert info.value.code == "E_ORDER"
