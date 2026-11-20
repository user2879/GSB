"""Orchestrating scheduler binding the engine to ledger and snapshot stores.

The in-memory :class:`~gpu_scheduler.engine.SimulationEngine` remains the single
source of scheduling truth; this class runs it one round at a time and persists
each round atomically.  A ``crash_after_round`` argument models a forced process
termination by stopping *before* the commit of that round, so recovery can
demonstrate UNKNOWN marking and replay equivalence.
"""
from __future__ import annotations

import os
from typing import List, Optional

from . import snapshot as snapshot_mod
from .engine import SimulationEngine
from .events import TraceEvent
from .ledger import Ledger
from .model import Declaration
from . import replay as replay_mod


class SchedulingSession:
    def __init__(self, declaration: Declaration, ledger_path: str,
                 snapshot_dir: str, backup_dir: str) -> None:
        self.declaration = declaration
        self.ledger_path = ledger_path
        self.snapshot_dir = snapshot_dir
        self.backup_dir = backup_dir
        self.ledger = Ledger(ledger_path)
        self.ledger.initialize(declaration)
        self.engine = SimulationEngine(declaration)
        self._prev_digest = ""
        self.crashed = False

    def run(self, crash_after_round: Optional[int] = None) -> dict:
        engine = self.engine
        while not engine._all_terminal():
            engine.rt.round_no += 1
            trace: List[TraceEvent] = []
            engine._finalize_due(trace)
            preemption = engine._schedule_pass(trace)
            record = engine._build_round(trace, preemption)
            if crash_after_round is not None and \
                    record.round_no == crash_after_round:
                # SIGKILL simulation: in-memory round exists, never committed.
                self.crashed = True
                engine.rt.rounds.append(record)
                return {
                    "status": "crashed",
                    "committed_rounds": self.ledger.last_round_no(),
                    "uncommitted_round": record.round_no,
                    "unknown_shards": list(record.running),
                    "note": "forced termination before commit; "
                            "no exit report generated",
                }
            digest = self.ledger.append_round(
                record, self._prev_digest,
                engine.persistence_snapshot())
            self._prev_digest = digest
            engine.rt.rounds.append(record)
            if not engine._advance_clock(trace):
                break
        digest = self.ledger.verify()
        snap = snapshot_mod.build_snapshot(
            self.declaration, self.ledger, digest)
        snapshot_path = snapshot_mod.write_snapshot(snap, self.snapshot_dir)
        return {
            "status": "completed",
            "rounds": self.ledger.last_round_no(),
            "ledger_digest": digest,
            "snapshot_path": snapshot_path,
            "unknown_shards": [],
        }

    def close(self) -> None:
        self.ledger.close()


def _fresh_session(declaration: Declaration, ledger_path: str,
                   snapshot_dir: str, backup_dir: str) -> SchedulingSession:
    if os.path.exists(ledger_path):
        os.remove(ledger_path)
    os.makedirs(snapshot_dir, exist_ok=True)
    return SchedulingSession(
        declaration, ledger_path, snapshot_dir, backup_dir)


def resume_session(declaration: Declaration, ledger_path: str,
                   snapshot_dir: str, backup_dir: str,
                   reference_rounds: Optional[List] = None) -> dict:
    """Resume from committed state; version drift triggers full replacement."""
    stored_snapshot = snapshot_mod.read_snapshot(snapshot_dir)
    if stored_snapshot is not None and not snapshot_mod.versions_match(
            declaration, stored_snapshot):
        backup_path = snapshot_mod.replace_stale_snapshot(
            snapshot_dir, backup_dir)
        session = _fresh_session(
            declaration, ledger_path, snapshot_dir, backup_dir)
        result = session.run()
        session.close()
        result["replacement_backup"] = backup_path
        result["status"] = "replaced"
        return result

    ledger = Ledger(ledger_path)
    committed_before = ledger.last_round_no()
    engine, unknown, committed_count = replay_mod.reconstruct_engine(
        declaration, ledger)
    # New rounds continue at committed_count + 1; comparisons therefore start
    # at that same number so equal round numbers carry equal semantics.
    recovered_rounds = replay_mod.continue_recovered(engine, ledger)
    digest = ledger.verify()
    snap = snapshot_mod.build_snapshot(declaration, ledger, digest)
    snapshot_path = snapshot_mod.write_snapshot(snap, snapshot_dir)

    replay_ok = True
    replay_detail = "no reference supplied"
    if reference_rounds is not None:
        replay_mod.compare_round_intervals(
            reference_rounds, recovered_rounds, committed_count)
        replay_detail = "shards, GPU usage and preemptions match reference"
    ledger.close()
    return {
        "status": "recovered",
        "committed_rounds_before_recovery": committed_before,
        "unknown_shards": [f"{task}#S{index:02d}" for task, index in unknown],
        "rerun_shards": [f"{task}#S{index:02d}" for task, index in unknown],
        "rounds": len(recovered_rounds),
        "ledger_digest": digest,
        "snapshot_path": snapshot_path,
        "replay_consistent": replay_ok,
        "replay_detail": replay_detail,
    }
