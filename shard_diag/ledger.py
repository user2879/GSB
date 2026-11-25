from __future__ import annotations

import copy
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .canonical import canonical_json, monotonic_ms, sha256_digest
from .errors import DiagnosticError, ErrorCode


LEDGER_FORMAT_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    session_id TEXT PRIMARY KEY,
    generation INTEGER NOT NULL,
    snapshot_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_intents (
    batch_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    snapshot_digest TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    event_seqs_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'unknown', 'committed')),
    reason TEXT,
    created_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    snapshot_digest TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    committed_at_ms INTEGER NOT NULL,
    UNIQUE (session_id, batch_index)
);
CREATE TABLE IF NOT EXISTS responses (
    batch_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    hit_shards_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, event_seq, db_id, table_name)
);
CREATE TABLE IF NOT EXISTS routing_results (
    batch_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    old_shard TEXT,
    new_shard TEXT,
    changed INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, event_seq, db_id, table_name)
);
CREATE TABLE IF NOT EXISTS drift_records (
    batch_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    old_route_version INTEGER,
    new_route_version INTEGER NOT NULL,
    old_shard TEXT,
    new_shard TEXT,
    drift_direction TEXT NOT NULL,
    boundary_index INTEGER,
    boundary_offset INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, event_seq, db_id, table_name)
);
CREATE TABLE IF NOT EXISTS leases (
    batch_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    db_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    route_version INTEGER NOT NULL,
    lease_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, lease_id)
);
CREATE TABLE IF NOT EXISTS lease_transitions (
    batch_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    transition_index INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    db_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status TEXT NOT NULL,
    reason TEXT NOT NULL,
    basis TEXT NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, lease_id, transition_index)
);
CREATE TABLE IF NOT EXISTS ownership (
    batch_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    db_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    route_version INTEGER NOT NULL,
    decision_reason TEXT NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, event_seq, lease_id)
);
CREATE TABLE IF NOT EXISTS quota (
    batch_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    db_id TEXT,
    active INTEGER NOT NULL,
    remaining INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, scope, db_id)
);
CREATE TABLE IF NOT EXISTS progress (
    session_id TEXT PRIMARY KEY,
    snapshot_digest TEXT NOT NULL,
    last_batch_index INTEGER NOT NULL,
    last_seq INTEGER NOT NULL,
    last_batch_id TEXT NOT NULL,
    updated_at_ms INTEGER NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    session_id: str
    committed_batch_indices: tuple[int, ...]
    unknown_batch_ids: tuple[str, ...]
    unknown_event_seqs: tuple[int, ...]
    resume_after_seq: int | None
    next_batch_index: int
    reclaimed_event_seqs: tuple[int, ...]
    exit_report_available: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "committed_batch_indices": list(self.committed_batch_indices),
            "unknown_batch_ids": list(self.unknown_batch_ids),
            "unknown_event_seqs": list(self.unknown_event_seqs),
            "resume_after_seq": self.resume_after_seq,
            "next_batch_index": self.next_batch_index,
            "reclaimed_event_seqs": list(self.reclaimed_event_seqs),
            "exit_report_available": self.exit_report_available,
            "note": "forced child termination cannot produce an exit report",
        }


@dataclass(frozen=True, slots=True)
class ReplayMismatch:
    event_seq: int | None
    db_id: str | None
    field: str
    expected: Any
    actual: Any
    batch_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_seq": self.event_seq,
            "db_id": self.db_id,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "batch_index": self.batch_index,
        }


def _replay_error(mismatch: ReplayMismatch) -> DiagnosticError:
    return DiagnosticError(
        ErrorCode.REPLAY.value,
        f"ledger replay mismatch at event_seq={mismatch.event_seq!r}, db_id={mismatch.db_id!r}, "
        f"field={mismatch.field!r}",
        db_id=mismatch.db_id,
        field=mismatch.field,
        details=mismatch.as_dict(),
    )


def _normalized_replay_projection(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove scheduling/clock noise while retaining every semantic result."""

    projected = copy.deepcopy(payload)
    projected.pop("thread_name", None)
    projected.pop("wait_records", None)
    projected.pop("worker_elapsed_ms", None)
    projected.pop("high_water", None)
    quota_snapshot = projected.get("quota_snapshot")
    if isinstance(quota_snapshot, dict):
        quota_snapshot.pop("high_water", None)
        quota_snapshot.pop("wait_records", None)
    events = projected.get("events", [])
    for event in events:
        event.pop("worker_elapsed_ms", None)
        event.pop("thread_name", None)
        for lease in event.get("leases", []):
            lease.pop("granted_at_ms", None)
            lease.pop("terminal_at_ms", None)
            lease.pop("expires_at_ms", None)
    for lease in projected.get("leases", []):
        lease.pop("granted_at_ms", None)
        lease.pop("terminal_at_ms", None)
        lease.pop("expires_at_ms", None)
    for transition in projected.get("lease_transitions", []):
        transition.pop("at_ms", None)
    projected.pop("progress", None)
    return projected


class Ledger:
    """SQLite ledger with one transaction per completed probe batch."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            str(self.path),
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self.connection.executescript(SCHEMA)
            self.connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )

    @classmethod
    def reset(cls, path: str | Path) -> None:
        path = Path(path)
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix)
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def format_version(self) -> int:
        with self._lock:
            row = self.connection.execute(
                "SELECT value FROM meta WHERE key='ledger_format_version'"
            ).fetchone()
        return int(row["value"])

    def replace_snapshot(
        self,
        session_id: str,
        snapshot_payload: dict[str, Any],
        *,
        generation: int | None = None,
    ) -> dict[str, Any]:
        snapshot_digest = sha256_digest(snapshot_payload)
        now = monotonic_ms()
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                previous = connection.execute(
                    "SELECT generation FROM snapshots WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if generation is None:
                    next_generation = int(previous["generation"]) + 1 if previous else 1
                else:
                    if generation < 1:
                        raise DiagnosticError(
                            ErrorCode.STATE.value,
                            "snapshot generation must be positive",
                            field="generation",
                        )
                    if previous and generation <= int(previous["generation"]):
                        raise DiagnosticError(
                            ErrorCode.STATE.value,
                            "snapshot generation must increase on whole replacement",
                            field="generation",
                        )
                    next_generation = generation
                for table in (
                    "progress",
                    "quota",
                    "ownership",
                    "lease_transitions",
                    "leases",
                    "drift_records",
                    "routing_results",
                    "responses",
                    "batches",
                    "batch_intents",
                ):
                    connection.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
                connection.execute("DELETE FROM snapshots WHERE session_id = ?", (session_id,))
                payload_json = canonical_json(snapshot_payload)
                connection.execute(
                    "INSERT INTO snapshots(session_id, generation, snapshot_digest, payload_json, updated_at_ms) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (session_id, next_generation, snapshot_digest, payload_json, now),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
                    (f"snapshot_digest:{session_id}", snapshot_digest),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {
            "session_id": session_id,
            "generation": next_generation,
            "snapshot_digest": snapshot_digest,
            "whole_replacement": True,
        }

    def get_snapshot(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM snapshots WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise DiagnosticError(
                ErrorCode.STATE.value,
                f"session snapshot {session_id!r} does not exist",
                field="session_id",
            )
        payload = json_loads(row["payload_json"])
        actual_digest = sha256_digest(payload)
        if actual_digest != row["snapshot_digest"]:
            raise DiagnosticError(
                ErrorCode.REPLAY.value,
                "snapshot digest does not match stored payload",
                field="snapshot_digest",
                details={"session_id": session_id},
            )
        return {
            "session_id": row["session_id"],
            "generation": row["generation"],
            "snapshot_digest": row["snapshot_digest"],
            "payload": payload,
        }

    def begin_batch(
        self,
        *,
        session_id: str,
        snapshot_digest: str,
        batch_id: str,
        batch_index: int,
        event_seqs: Sequence[int],
    ) -> dict[str, Any]:
        if not event_seqs:
            raise DiagnosticError(ErrorCode.STATE.value, "batch must contain at least one event", field="events")
        event_seqs = tuple(sorted(event_seqs))
        now = monotonic_ms()
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                snapshot = connection.execute(
                    "SELECT snapshot_digest FROM snapshots WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if snapshot is None:
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        "session snapshot must be installed before beginning a batch",
                        field="session_id",
                    )
                if snapshot["snapshot_digest"] != snapshot_digest:
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        "batch snapshot digest does not match the installed session snapshot",
                        field="snapshot_digest",
                    )
                existing = connection.execute(
                    "SELECT status FROM batch_intents WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if existing is not None:
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        f"batch intent {batch_id!r} already exists with status {existing['status']!r}",
                        field="batch_id",
                    )
                connection.execute(
                    "INSERT INTO batch_intents(batch_id, session_id, snapshot_digest, batch_index, start_seq, "
                    "end_seq, event_seqs_json, status, reason, created_at_ms, updated_at_ms) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, 'pending', NULL, ?, ?)",
                    (
                        batch_id,
                        session_id,
                        snapshot_digest,
                        batch_index,
                        event_seqs[0],
                        event_seqs[-1],
                        canonical_json(list(event_seqs)),
                        now,
                        now,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {
            "batch_id": batch_id,
            "batch_index": batch_index,
            "status": "pending",
            "event_seqs": list(event_seqs),
        }

    def abandon_batch(self, batch_id: str, *, reason: str) -> dict[str, Any]:
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT * FROM batch_intents WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if row is None:
                    raise DiagnosticError(ErrorCode.STATE.value, f"unknown batch intent {batch_id!r}")
                if row["status"] == "committed":
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        f"cannot abandon committed batch {batch_id!r}",
                        field="batch_id",
                    )
                connection.execute(
                    "UPDATE batch_intents SET status='unknown', reason=?, updated_at_ms=? WHERE batch_id=?",
                    (reason, monotonic_ms(), batch_id),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {
            "batch_id": row["batch_id"],
            "session_id": row["session_id"],
            "batch_index": row["batch_index"],
            "event_seqs": json_loads(row["event_seqs_json"]),
            "status": "unknown",
            "reason": reason,
        }

    def commit_batch(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Atomically persist all batch result, ownership, quota, and progress rows."""

        batch_id = payload.get("batch_id")
        session_id = payload.get("session_id")
        batch_index = payload.get("batch_index")
        snapshot_digest = payload.get("snapshot_digest")
        events = payload.get("events", [])
        if not isinstance(batch_id, str) or not isinstance(session_id, str):
            raise DiagnosticError(ErrorCode.STATE.value, "batch payload needs batch_id and session_id", field="batch_id")
        if not isinstance(batch_index, int) or batch_index < 1:
            raise DiagnosticError(ErrorCode.STATE.value, "batch_index must be a positive integer", field="batch_index")
        if not isinstance(events, list) or not events:
            raise DiagnosticError(ErrorCode.STATE.value, "batch events must be non-empty", field="events")
        event_seqs = [event.get("event_seq") for event in events]
        if not all(isinstance(seq, int) for seq in event_seqs):
            raise DiagnosticError(ErrorCode.STATE.value, "every event needs an integer event_seq", field="event_seq")
        event_seqs = sorted(event_seqs)
        if len(event_seqs) != len(set(event_seqs)):
            raise DiagnosticError(ErrorCode.STATE.value, "batch event sequences must be unique", field="event_seq")
        content_payload = copy.deepcopy(payload)
        content_payload["format_version"] = LEDGER_FORMAT_VERSION
        content_digest = sha256_digest(content_payload)
        now = monotonic_ms()
        payload_json = canonical_json(content_payload)

        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                snapshot = connection.execute(
                    "SELECT snapshot_digest FROM snapshots WHERE session_id = ?", (session_id,)
                ).fetchone()
                if snapshot is None or snapshot["snapshot_digest"] != snapshot_digest:
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        "batch snapshot no longer matches the installed snapshot",
                        field="snapshot_digest",
                    )
                intent = connection.execute(
                    "SELECT * FROM batch_intents WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if intent is None or intent["status"] != "pending":
                    status = intent["status"] if intent is not None else "missing"
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        f"batch {batch_id!r} is not pending (status={status})",
                        field="batch_id",
                    )
                if intent["session_id"] != session_id or intent["batch_index"] != batch_index:
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        "batch payload does not match the pending intent",
                        field="batch_index",
                    )
                if json_loads(intent["event_seqs_json"]) != event_seqs:
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        "batch event sequence set does not match the pending intent",
                        field="event_seqs",
                    )
                previous = connection.execute(
                    "SELECT last_batch_index, last_seq, last_batch_id FROM progress WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                expected_index = 1 if previous is None else int(previous["last_batch_index"]) + 1
                if batch_index != expected_index:
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        f"batch index {batch_index} is not the next contiguous index {expected_index}",
                        field="batch_index",
                    )
                if previous is not None and event_seqs[0] <= int(previous["last_seq"]):
                    raise DiagnosticError(
                        ErrorCode.STATE.value,
                        "batch event sequences must advance beyond the progress cursor",
                        field="event_seq",
                    )
                for event_seq in event_seqs:
                    existing = connection.execute(
                        "SELECT 1 FROM responses WHERE session_id=? AND event_seq=? LIMIT 1",
                        (session_id, event_seq),
                    ).fetchone()
                    if existing is not None:
                        raise DiagnosticError(
                            ErrorCode.STATE.value,
                            "batch contains an event sequence already committed in this session",
                            field="event_seq",
                            details={"event_seq": event_seq},
                        )

                connection.execute(
                    "INSERT INTO batches(batch_id, session_id, snapshot_digest, batch_index, start_seq, end_seq, "
                    "format_version, content_digest, payload_json, committed_at_ms) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        batch_id,
                        session_id,
                        snapshot_digest,
                        batch_index,
                        event_seqs[0],
                        event_seqs[-1],
                        LEDGER_FORMAT_VERSION,
                        content_digest,
                        payload_json,
                        now,
                    ),
                )
                for event in events:
                    event_seq = event["event_seq"]
                    db_id = event["db_id"]
                    table_name = event["table"]
                    hits = event.get("hit_shards", [])
                    response_json = canonical_json(event)
                    connection.execute(
                        "INSERT INTO responses(batch_id, session_id, event_seq, db_id, table_name, "
                        "hit_shards_json, response_json) VALUES(?, ?, ?, ?, ?, ?, ?)",
                        (batch_id, session_id, event_seq, db_id, table_name, canonical_json(hits), response_json),
                    )
                for record in payload.get("route_records", []):
                    connection.execute(
                        "INSERT INTO routing_results(batch_id, session_id, event_seq, db_id, table_name, old_shard, "
                        "new_shard, changed, record_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            batch_id,
                            session_id,
                            record["event_seq"],
                            record["db_id"],
                            record["table"],
                            record.get("old_shard"),
                            record.get("new_shard"),
                            1 if record.get("changed") else 0,
                            canonical_json(record),
                        ),
                    )
                for record in payload.get("drift_records", []):
                    connection.execute(
                        "INSERT INTO drift_records(batch_id, session_id, event_seq, db_id, table_name, "
                        "old_route_version, new_route_version, old_shard, new_shard, drift_direction, boundary_index, "
                        "boundary_offset, record_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            batch_id,
                            session_id,
                            record["event_seq"],
                            record["db_id"],
                            record["table"],
                            record.get("old_route_version"),
                            record["new_route_version"],
                            record.get("old_shard"),
                            record.get("new_shard"),
                            record["drift_direction"],
                            record.get("boundary_index"),
                            record.get("boundary_offset", 0),
                            canonical_json(record),
                        ),
                    )
                for lease in payload.get("leases", []):
                    decision_reason = lease.get("terminal_reason") or lease.get("status", "active")
                    connection.execute(
                        "INSERT INTO leases(batch_id, session_id, lease_id, db_id, shard_id, event_seq, "
                        "route_version, lease_version, status, record_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            batch_id,
                            session_id,
                            lease["lease_id"],
                            lease["db_id"],
                            lease["shard_id"],
                            lease["event_seq"],
                            lease["route_version"],
                            lease["lease_version"],
                            lease["status"],
                            canonical_json(lease),
                        ),
                    )
                    connection.execute(
                        "INSERT INTO ownership(batch_id, session_id, db_id, shard_id, lease_id, event_seq, "
                        "route_version, decision_reason, record_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            batch_id,
                            session_id,
                            lease["db_id"],
                            lease["shard_id"],
                            lease["lease_id"],
                            lease["event_seq"],
                            lease["route_version"],
                            decision_reason,
                            canonical_json(lease),
                        ),
                    )
                for transition_index, transition in enumerate(payload.get("lease_transitions", [])):
                    connection.execute(
                        "INSERT INTO lease_transitions(batch_id, session_id, lease_id, transition_index, event_seq, db_id, "
                        "shard_id, from_status, to_status, reason, basis, record_json) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            batch_id,
                            session_id,
                            transition["lease_id"],
                            transition_index,
                            transition["event_seq"],
                            transition["db_id"],
                            transition["shard_id"],
                            transition["from_status"],
                            transition["to_status"],
                            transition["reason"],
                            transition["basis"],
                            canonical_json(transition),
                        ),
                    )
                quota_snapshot = payload.get("quota_snapshot", {})
                connection.execute(
                    "INSERT INTO quota(batch_id, session_id, scope, db_id, active, remaining, record_json) "
                    "VALUES(?, ?, 'global', NULL, ?, ?, ?)",
                    (
                        batch_id,
                        session_id,
                        int(quota_snapshot.get("active_leases", 0)),
                        int(quota_snapshot.get("remaining_global", 0)),
                        canonical_json(quota_snapshot),
                    ),
                )
                for db_id, db_quota in quota_snapshot.get("by_db", {}).items():
                    connection.execute(
                        "INSERT INTO quota(batch_id, session_id, scope, db_id, active, remaining, record_json) "
                        "VALUES(?, ?, 'database', ?, ?, ?, ?)",
                        (
                            batch_id,
                            session_id,
                            db_id,
                            int(db_quota["active_leases"]),
                            int(db_quota["remaining_leases"]),
                            canonical_json(db_quota),
                        ),
                    )
                connection.execute(
                    "UPDATE batch_intents SET status='committed', reason='atomic batch commit', updated_at_ms=? "
                    "WHERE batch_id=?",
                    (now, batch_id),
                )
                connection.execute(
                    "INSERT INTO progress(session_id, snapshot_digest, last_batch_index, last_seq, last_batch_id, "
                    "updated_at_ms) VALUES(?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET snapshot_digest=excluded.snapshot_digest, "
                    "last_batch_index=excluded.last_batch_index, last_seq=excluded.last_seq, "
                    "last_batch_id=excluded.last_batch_id, updated_at_ms=excluded.updated_at_ms",
                    (session_id, snapshot_digest, batch_index, event_seqs[-1], batch_id, now),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return {
            "batch_id": batch_id,
            "batch_index": batch_index,
            "format_version": LEDGER_FORMAT_VERSION,
            "content_digest": content_digest,
            "event_seqs": event_seqs,
            "progress": {"last_batch_index": batch_index, "last_seq": event_seqs[-1]},
        }

    def recover_session(self, session_id: str) -> RecoveryReport:
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                pending = connection.execute(
                    "SELECT * FROM batch_intents WHERE session_id=? AND status='pending' "
                    "ORDER BY batch_index, batch_id",
                    (session_id,),
                ).fetchall()
                for row in pending:
                    connection.execute(
                        "UPDATE batch_intents SET status='unknown', reason=?, updated_at_ms=? WHERE batch_id=?",
                        ("parent recovered before atomic batch commit; rerun required", monotonic_ms(), row["batch_id"]),
                    )
                committed = connection.execute(
                    "SELECT batch_index FROM batches WHERE session_id=? ORDER BY batch_index", (session_id,)
                ).fetchall()
                progress = connection.execute(
                    "SELECT last_batch_index, last_seq FROM progress WHERE session_id=?", (session_id,)
                ).fetchone()
                last_committed_index = int(progress["last_batch_index"]) if progress is not None else 0
                unresolved = connection.execute(
                    "SELECT * FROM batch_intents WHERE session_id=? AND status='unknown' "
                    "AND batch_index > ? ORDER BY batch_index, batch_id",
                    (session_id, last_committed_index),
                ).fetchall()
                unknown_ids = [row["batch_id"] for row in unresolved]
                unknown_seqs: list[int] = []
                for row in unresolved:
                    unknown_seqs.extend(json_loads(row["event_seqs_json"]))
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        committed_indices = tuple(int(row["batch_index"]) for row in committed)
        last_seq = int(progress["last_seq"]) if progress is not None else None
        next_index = int(progress["last_batch_index"]) + 1 if progress is not None else 1
        return RecoveryReport(
            session_id=session_id,
            committed_batch_indices=committed_indices,
            unknown_batch_ids=tuple(sorted(unknown_ids)),
            unknown_event_seqs=tuple(sorted(set(unknown_seqs))),
            resume_after_seq=last_seq,
            next_batch_index=next_index,
            reclaimed_event_seqs=tuple(sorted(unknown_seqs)),
        )

    def get_progress(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM progress WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def read_batches(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM batches WHERE session_id=? ORDER BY batch_index", (session_id,)
            ).fetchall()
        return [
            {
                "batch_id": row["batch_id"],
                "batch_index": row["batch_index"],
                "start_seq": row["start_seq"],
                "end_seq": row["end_seq"],
                "format_version": row["format_version"],
                "content_digest": row["content_digest"],
                "payload": json_loads(row["payload_json"]),
            }
            for row in rows
        ]

    def list_intents(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM batch_intents WHERE session_id=? ORDER BY batch_index, batch_id",
                (session_id,),
            ).fetchall()
        return [
            {
                "batch_id": row["batch_id"],
                "batch_index": row["batch_index"],
                "event_seqs": json_loads(row["event_seqs_json"]),
                "status": row["status"],
                "reason": row["reason"],
            }
            for row in rows
        ]

    def verify_integrity(self, session_id: str) -> dict[str, Any]:
        batches = self.read_batches(session_id)
        for batch in batches:
            stored = batch["payload"]
            if batch["format_version"] != LEDGER_FORMAT_VERSION:
                raise DiagnosticError(
                    ErrorCode.REPLAY.value,
                    "ledger batch format version mismatch",
                    field="format_version",
                    details={"batch_id": batch["batch_id"]},
                )
            content = copy.deepcopy(stored)
            content["format_version"] = LEDGER_FORMAT_VERSION
            actual = sha256_digest(content)
            if actual != batch["content_digest"]:
                first_event = stored.get("events", [{}])[0]
                mismatch = ReplayMismatch(
                    first_event.get("event_seq"),
                    first_event.get("db_id"),
                    "content_digest",
                    batch["content_digest"],
                    actual,
                    batch["batch_index"],
                )
                raise _replay_error(mismatch)
        progress = self.get_progress(session_id)
        return {
            "session_id": session_id,
            "batches": len(batches),
            "format_version": LEDGER_FORMAT_VERSION,
            "progress": progress,
            "verified": True,
        }

    def replay_against(
        self,
        session_id: str,
        recomputed_batches: Iterable[dict[str, Any]],
        *,
        require_complete: bool = True,
    ) -> dict[str, Any]:
        self.verify_integrity(session_id)
        stored = self.read_batches(session_id)
        recomputed = sorted(recomputed_batches, key=lambda payload: int(payload["batch_index"]))
        if require_complete and len(recomputed) != len(stored):
            next_index = stored[-1]["batch_index"] + 1 if stored else 1
            mismatch = ReplayMismatch(
                stored[-1]["end_seq"] if stored else None,
                None,
                "batch_index",
                len(stored),
                len(recomputed),
                next_index,
            )
            raise _replay_error(mismatch)
        max_batches = min(len(stored), len(recomputed))
        for index in range(max_batches):
            expected_batch = stored[index]
            actual_batch = recomputed[index]
            if int(actual_batch.get("batch_index", -1)) != expected_batch["batch_index"]:
                mismatch = ReplayMismatch(
                    expected_batch["start_seq"],
                    None,
                    "batch_index",
                    expected_batch["batch_index"],
                    actual_batch.get("batch_index"),
                    expected_batch["batch_index"],
                )
                raise _replay_error(mismatch)
            expected = _normalized_replay_projection(expected_batch["payload"])
            actual = _normalized_replay_projection(actual_batch)
            mismatch = _first_semantic_mismatch(expected, actual, batch_index=expected_batch["batch_index"])
            if mismatch is not None:
                raise _replay_error(mismatch)
        return {
            "session_id": session_id,
            "replayed_batches": len(recomputed),
            "matching_batches": max_batches,
            "verified": True,
        }

    def replace_batch_for_test(self, session_id: str, batch_index: int, payload: dict[str, Any]) -> None:
        """Test hook: rewrite a committed payload and its digest atomically."""

        content = copy.deepcopy(payload)
        content["format_version"] = LEDGER_FORMAT_VERSION
        digest = sha256_digest(content)
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                row = self.connection.execute(
                    "SELECT batch_id FROM batches WHERE session_id=? AND batch_index=?",
                    (session_id, batch_index),
                ).fetchone()
                if row is None:
                    raise DiagnosticError(ErrorCode.STATE.value, "batch does not exist", field="batch_index")
                self.connection.execute(
                    "UPDATE batches SET payload_json=?, content_digest=? WHERE batch_id=?",
                    (canonical_json(content), digest, row["batch_id"]),
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise


def json_loads(value: str) -> Any:
    import json

    return json.loads(value)


def _first_semantic_mismatch(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    batch_index: int,
) -> ReplayMismatch | None:
    expected_events = expected.get("events", [])
    actual_events = actual.get("events", [])
    expected_by_seq = {event.get("event_seq"): event for event in expected_events}
    actual_by_seq = {event.get("event_seq"): event for event in actual_events}
    all_seqs = sorted(
        set(expected_by_seq) | set(actual_by_seq),
        key=lambda value: (value is None, value),
    )
    for seq in all_seqs:
        expected_event = expected_by_seq.get(seq)
        actual_event = actual_by_seq.get(seq)
        if expected_event is None or actual_event is None:
            db_id = (expected_event or actual_event or {}).get("db_id")
            return ReplayMismatch(seq, db_id, "event_seq", expected_event, actual_event, batch_index)
        expected_db = expected_event.get("db_id")
        actual_db = actual_event.get("db_id")
        if expected_db != actual_db:
            return ReplayMismatch(seq, expected_db, "db_id", expected_db, actual_db, batch_index)
        expected_record = expected_event.get("route_record", {})
        actual_record = actual_event.get("route_record", {})
        for field in (
            "event_seq",
            "db_id",
            "table",
            "key_column",
            "route_key",
            "old_route_version",
            "new_route_version",
            "old_shard",
            "new_shard",
            "changed",
            "drift_direction",
            "boundary_index",
            "boundary_old",
            "boundary_new",
            "boundary_offset",
            "reason",
        ):
            if expected_record.get(field) != actual_record.get(field):
                return ReplayMismatch(
                    seq,
                    expected_db,
                    field,
                    expected_record.get(field),
                    actual_record.get(field),
                    batch_index,
                )
        if sorted(expected_event.get("hit_shards", [])) != sorted(actual_event.get("hit_shards", [])):
            return ReplayMismatch(
                seq,
                expected_db,
                "hit_shards",
                expected_event.get("hit_shards"),
                actual_event.get("hit_shards"),
                batch_index,
            )
        expected_lease_ids = sorted(lease.get("lease_id") for lease in expected_event.get("leases", []))
        actual_lease_ids = sorted(lease.get("lease_id") for lease in actual_event.get("leases", []))
        if expected_lease_ids != actual_lease_ids:
            return ReplayMismatch(
                seq,
                expected_db,
                "lease_id",
                expected_lease_ids,
                actual_lease_ids,
                batch_index,
            )
        expected_event_leases = {
            lease.get("lease_id"): lease for lease in expected_event.get("leases", [])
        }
        actual_event_leases = {
            lease.get("lease_id"): lease for lease in actual_event.get("leases", [])
        }
        for lease_id in sorted(expected_event_leases):
            expected_lease = expected_event_leases[lease_id]
            actual_lease = actual_event_leases[lease_id]
            for lease_field in ("db_id", "shard_id", "route_version", "lease_version", "status", "renew_count"):
                if expected_lease.get(lease_field) != actual_lease.get(lease_field):
                    return ReplayMismatch(
                        seq,
                        expected_db,
                        f"lease.{lease_field}",
                        expected_lease.get(lease_field),
                        actual_lease.get(lease_field),
                        batch_index,
                    )
        for event_field in (
            "table",
            "route_key",
            "worker_route_version",
            "sample_keys",
        ):
            if expected_event.get(event_field) != actual_event.get(event_field):
                return ReplayMismatch(
                    seq,
                    expected_db,
                    event_field,
                    expected_event.get(event_field),
                    actual_event.get(event_field),
                    batch_index,
                )
    for field in ("route_records", "drift_records", "leases", "quota_snapshot"):
        if expected.get(field) != actual.get(field):
            first = expected.get("events", [{}])[0]
            return ReplayMismatch(
                first.get("event_seq"),
                first.get("db_id"),
                field,
                expected.get(field),
                actual.get(field),
                batch_index,
            )
    return None
