from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from batch_diagnostic.engine import DiagnosticEngine
from batch_diagnostic.errors import ReplayError, SnapshotError
from batch_diagnostic.ledger import LEDGER_FORMAT_VERSION, SnapshotLedger
from batch_diagnostic.planning import build_plan
from batch_diagnostic.replay import replay_ledger
from batch_diagnostic.verifier import verify_ledger


def _run_full(spec, ledger_path):
    ledger = SnapshotLedger.create(ledger_path, spec)
    engine = DiagnosticEngine(spec, ledger)
    engine.start_workers()
    try:
        engine.run()
    finally:
        engine.stop_workers()
    ledger.close()


def test_ledger_round_trip_and_progress(system_spec, tmp_path):
    path = str(tmp_path / "diag.sqlite")
    _run_full(system_spec, path)
    ledger = SnapshotLedger.open(path, system_spec)
    try:
        plan = build_plan(system_spec)
        assert len(ledger.committed_batch_ids()) == len(plan.batches)
        assert ledger.unknown_batches(plan) == ()
        assert ledger.header.format_version == LEDGER_FORMAT_VERSION
        assert ledger.total_budget_ms() >= 0
        first = plan.batches[0]
        receipts = ledger.load_receipts(first.batch_id)
        assert len(receipts) == len(first.request_ids)
    finally:
        ledger.close()


def test_crash_leaves_unknown_batches_for_rerun(system_spec, tmp_path):
    path = str(tmp_path / "crash.sqlite")
    ledger = SnapshotLedger.create(path, system_spec)
    engine = DiagnosticEngine(system_spec, ledger)
    engine.start_workers()
    try:
        plan = build_plan(system_spec)
        first_batch = plan.batches[0]
        engine.run(only_batch_ids={first_batch.batch_id})
    finally:
        engine.stop_workers()
    ledger.close()

    reopened = SnapshotLedger.open(path, system_spec)
    try:
        plan = build_plan(system_spec)
        unknown = reopened.unknown_batches(plan)
        assert first_batch.batch_id not in unknown
        assert len(unknown) == len(plan.batches) - 1
    finally:
        reopened.close()


def test_snapshot_mismatch_replace_creates_readonly_backup(system_spec, tmp_path):
    path = str(tmp_path / "snap.sqlite")
    _run_full(system_spec, path)
    changed_worker = replace(
        system_spec.workers[0], seed=system_spec.workers[0].seed + 9, param_version=2
    )
    drifted_spec = replace(system_spec, workers=(changed_worker, system_spec.workers[1]))
    ledger = SnapshotLedger.open(path, drifted_spec, on_mismatch="replace")
    try:
        assert os.path.exists(ledger.backup_path)
        assert oct(os.stat(ledger.backup_path).st_mode & 0o777) == "0o444"
        assert ledger.committed_batch_ids() == ()
    finally:
        ledger.close()


def test_snapshot_mismatch_raises_when_requested(system_spec, tmp_path):
    path = str(tmp_path / "snap2.sqlite")
    _run_full(system_spec, path)
    changed_worker = replace(
        system_spec.workers[0], seed=system_spec.workers[0].seed + 9, param_version=2
    )
    drifted_spec = replace(system_spec, workers=(changed_worker, system_spec.workers[1]))
    with pytest.raises(SnapshotError) as exc:
        SnapshotLedger.open(path, drifted_spec, on_mismatch="raise")
    assert exc.value.code == "E_SNAPSHOT"


def test_replay_matches_and_detects_first_divergence(system_spec, tmp_path):
    path = str(tmp_path / "replay.sqlite")
    _run_full(system_spec, path)
    ledger = SnapshotLedger.open(path, system_spec)
    try:
        report = replay_ledger(system_spec, ledger)
        assert report["status"] == "match"
        plan = build_plan(system_spec)
        target = plan.batches[0]
        ledger.conn.execute(
            "UPDATE batches SET output_digest = ? WHERE batch_id = ?",
            ("tampered-digest", target.batch_id),
        )
        ledger.conn.commit()
        with pytest.raises(ReplayError) as exc:
            replay_ledger(system_spec, ledger)
        assert exc.value.code == "E_REPLAY"
        assert exc.value.context["batch_id"] == target.batch_id
        assert exc.value.context["field"] == "output_digest"
    finally:
        ledger.close()


def test_independent_verifier(system_spec, tmp_path):
    requests_path = str(tmp_path / "requests.json")
    workers_path = str(tmp_path / "workers.json")
    budget_path = str(tmp_path / "budget.json")
    ledger_path = str(tmp_path / "verify.sqlite")
    _write_declarations(system_spec, requests_path, workers_path, budget_path)
    _run_full(system_spec, ledger_path)
    ledger = SnapshotLedger.open(ledger_path, system_spec)
    try:
        report = verify_ledger(requests_path, workers_path, budget_path, ledger)
        assert report["status"] == "match"
    finally:
        ledger.close()


def _write_declarations(spec, requests_path, workers_path, budget_path):
    with open(requests_path, "w", encoding="utf-8") as handle:
        json.dump([request.__dict__ for request in spec.requests], handle)
    with open(workers_path, "w", encoding="utf-8") as handle:
        json.dump([worker.__dict__ for worker in spec.workers], handle)
    with open(budget_path, "w", encoding="utf-8") as handle:
        json.dump(spec.budget.__dict__, handle)
