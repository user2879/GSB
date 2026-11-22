"""High-level orchestration: run, persist, recover and authoritatively replay."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import E_LEDGER, E_PREVIEW, DiagError, StructuredLogger
from .loader import load_declaration
from .models import Declaration, ShardState
from .persist.ledger import Ledger
from .persist.snapshot import (build_snapshot, diff_versions, load_snapshot,
                               replay_compare, save_snapshot)
from .sim.engine import SchedulerEngine


@dataclass
class SessionResult:
    rounds: list[dict[str, Any]]
    outcomes: dict[str, dict[str, Any]]
    unknown_shards: list[dict[str, Any]]
    declaration_digest: str


def reset_state_dir(state_dir: str | Path) -> None:
    state_dir = Path(state_dir)
    if state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)


def _snapshot_shard_states(decl: Declaration) -> dict[str, dict[str, Any]]:
    states = {}
    for sid in sorted(decl.shards):
        shard = decl.shards[sid]
        states[sid] = {
            "state": shard.state.value,
            "pool_id": shard.pool_id,
            "digest": shard.result_digest,
            "attempts": shard.attempts,
        }
    return states


def run_session(pools_path: str | Path, tasks_path: str | Path,
                params_path: str | Path, state_dir: str | Path,
                logger: StructuredLogger | None = None,
                fail_shards: set[str] | None = None,
                crash_after_round: int | None = None,
                max_workers: int = 4) -> SessionResult:
    state_dir = Path(state_dir)
    reset_state_dir(state_dir)
    decl = load_declaration(pools_path, tasks_path, params_path)
    ledger = Ledger(state_dir / "ledger.sqlite",
                    decl.params.ledger_format_version,
                    _decl_digest(decl))
    engine = SchedulerEngine(decl, max_workers=max_workers)
    if fail_shards:
        engine.fail_inject = set(fail_shards)
    engine.crash_after_round = crash_after_round
    unknown: list[dict[str, Any]] = []

    crashed = False
    try:
        guard = 0
        max_rounds = sum(t.shard_count for t in decl.tasks.values()) * 200 + 100
        while not engine.is_done():
            try:
                record = engine.run_round()
            except RuntimeError as exc:
                if "forced termination" not in str(exc):
                    raise
                crash_record = engine.pending_crash_record
                # Dispatch intents survive; the round itself is never committed.
                ledger.log_dispatch_intent(
                    crash_record["round"],
                    [{"shard_id": sid,
                      "pool_id": decl.shards[sid].pool_id}
                     for sid in crash_record["running_set"]])
                ledger.conn.commit()
                raise
            # Dispatch intent is durably logged BEFORE the round commit; a
            # kill -9 leaves these rows behind so they can be marked UNKNOWN.
            ledger.log_dispatch_intent(
                record["round"],
                [{"shard_id": sid,
                  "pool_id": decl.shards[sid].pool_id}
                 for sid in record["running_set"]])
            ledger.commit_round(record, _snapshot_shard_states(decl),
                                record["round_digest"])
            guard += 1
            if guard > max_rounds:  # pragma: no cover - defensive
                raise RuntimeError("scheduler failed to converge")
    except RuntimeError as exc:
        if "forced termination" not in str(exc):
            raise
        crashed = True
        if logger:
            logger.log("E_STATE", "simulated forced termination",
                       round=engine.round_no + 1)

    # Dispatch intents are deliberately left in the journal when crashed so an
    # independent `recover` invocation reconciles them into unknown_shards.
    if not crashed:
        ledger.reconcile_crash()
    unknown = ledger.reconcile_crash() if not crashed else [
        {"shard_id": row[0], "round_no": row[1], "pool_id": row[2],
         "reason": "dispatch intent present but round never committed; rerun required"}
        for row in ledger.conn.execute(
            "SELECT shard_id, round_no, pool_id FROM dispatch_journal "
            "ORDER BY shard_id").fetchall()]

    rounds = ledger.load_rounds()
    outcomes = engine.shard_outcomes()
    for entry in unknown:
        if entry["shard_id"] in outcomes:
            outcomes[entry["shard_id"]]["state"] = ShardState.UNKNOWN.value

    snapshot = build_snapshot(decl, rounds, outcomes,
                              ledger.last_committed_round())
    save_snapshot(state_dir / "snapshot.json", snapshot)
    ledger.verify_digests()
    ledger.close()
    return SessionResult(rounds=rounds, outcomes=outcomes,
                         unknown_shards=unknown,
                         declaration_digest=_decl_digest(decl))


def recover_session(pools_path: str | Path, tasks_path: str | Path,
                    params_path: str | Path, state_dir: str | Path,
                    logger: StructuredLogger | None = None) -> dict[str, Any]:
    """Reopen solely from committed rounds; unknown shards must be rerun."""
    decl = load_declaration(pools_path, tasks_path, params_path)
    ledger = Ledger(Path(state_dir) / "ledger.sqlite",
                    decl.params.ledger_format_version, _decl_digest(decl))
    ledger.verify_digests()
    unknown = ledger.reconcile_crash()
    cursor = ledger.last_committed_round()
    snapshot = load_snapshot(Path(state_dir) / "snapshot.json")
    version_diffs = diff_versions(snapshot, decl)
    ledger.close()
    if logger and unknown:
        logger.log("E_STATE", "uncommitted shards marked unknown after recovery",
                   unknown_shards=unknown, last_committed_round=cursor)
    return {"last_committed_round": cursor, "unknown_shards": unknown,
            "version_diffs": version_diffs}


def authoritative_recompute(pools_path: str | Path, tasks_path: str | Path,
                            params_path: str | Path, state_dir: str | Path
                            ) -> dict[str, Any]:
    """Independent replay from declarations, compared byte-for-byte to ledger."""
    decl = load_declaration(pools_path, tasks_path, params_path)
    ledger = Ledger(Path(state_dir) / "ledger.sqlite",
                    decl.params.ledger_format_version, _decl_digest(decl))
    stored_rounds = ledger.load_rounds()
    ledger.close()

    engine = SchedulerEngine(Declaration(
        pools=decl.pools, tasks=decl.tasks,
        shards={sid: _fresh_shard(shard) for sid, shard in decl.shards.items()},
        params=decl.params))
    replay: list[dict[str, Any]] = []
    for _ in range(len(stored_rounds)):
        replay.append(engine.run_round())

    for index, (stored, actual) in enumerate(
            zip(stored_rounds, replay), start=1):
        for field in ("water_levels", "preemptions", "completed",
                      "running_set", "ready_set", "queued", "round_digest"):
            from .digest import canonical_bytes
            if canonical_bytes(stored.get(field)) != canonical_bytes(
                    actual.get(field)):
                raise DiagError(E_PREVIEW,
                                "authoritative recompute diverges from ledger", [{
                                    "round": index, "field": field,
                                    "ledger": stored.get(field),
                                    "recompute": actual.get(field),
                                }])
    return {"rounds_compared": len(stored_rounds),
            "outcomes": engine.shard_outcomes(),
            "byte_identical": True}


def restore_from_snapshot(pools_path: str | Path, tasks_path: str | Path,
                          params_path: str | Path, state_dir: str | Path
                          ) -> dict[str, Any]:
    """Whole-state replacement restore followed by E_REPLAY validation."""
    decl = load_declaration(pools_path, tasks_path, params_path)
    snapshot = load_snapshot(Path(state_dir) / "snapshot.json")
    diffs = diff_versions(snapshot, decl)
    engine = SchedulerEngine(Declaration(
        pools=decl.pools, tasks=decl.tasks,
        shards={sid: _fresh_shard(shard) for sid, shard in decl.shards.items()},
        params=decl.params))
    engine.run_all()
    replay_compare(snapshot, engine.rounds, engine.shard_outcomes())
    return {"version_diffs": diffs, "replay_ok": True,
            "rounds": len(engine.rounds)}


def _fresh_shard(shard):
    from .models import Shard
    return Shard(shard_id=shard.shard_id, task_id=shard.task_id,
                 index=shard.index, deps=shard.deps)


def _decl_digest(decl: Declaration) -> str:
    from .digest import digest_of
    return digest_of({
        "pools": {pid: pool.__dict__ for pid, pool in decl.pools.items()},
        "tasks": {tid: task.__dict__ for tid, task in decl.tasks.items()},
        "params": decl.params.__dict__,
        "shards": {sid: {"deps": shard.deps, "index": shard.index}
                   for sid, shard in decl.shards.items()},
    })


def verify_ledger_digest(state_dir: str | Path) -> None:
    db = Path(state_dir) / "ledger.sqlite"
    ledger = Ledger.__new__(Ledger)
    ledger.path = db
    ledger.conn = __import__("sqlite3").connect(str(db), isolation_level=None)
    ledger.conn.row_factory = __import__("sqlite3").Row
    try:
        ledger.verify_digests()
    finally:
        ledger.conn.close()
