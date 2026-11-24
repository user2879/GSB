"""SQLite ledger, crash recovery, snapshots, and wholesale restoration."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .declarations import DeclarationSet
from .errors import GrantError, LedgerError
from .logging import StructuredLogger

SCHEMA_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecoveryReport:
    recovered_attempts: tuple[dict[str, Any], ...]
    rerun_required: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovered_attempts": list(self.recovered_attempts),
            "rerun_required": list(self.rerun_required),
        }


@dataclass(frozen=True)
class RestoreReport:
    replaced: bool
    backup_path: str | None
    descriptor_mismatches: tuple[dict[str, Any], ...]
    added_sessions: tuple[str, ...]
    removed_sessions: tuple[str, ...]
    changed_sessions: tuple[str, ...]
    added_probes: tuple[str, ...]
    removed_probes: tuple[str, ...]
    changed_probes: tuple[str, ...]
    state_digest: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    cycle_cursor INTEGER NOT NULL,
    consecutive_unrenewed INTEGER NOT NULL,
    last_ratio_bp INTEGER,
    pause_reason TEXT,
    last_round_cycle INTEGER,
    updated_mono_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS round_attempts (
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_mono_ns INTEGER NOT NULL,
    completed_mono_ns INTEGER,
    note TEXT,
    PRIMARY KEY (session_id, cycle, attempt)
);
CREATE TABLE IF NOT EXISTS _stage_attempts (
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_mono_ns INTEGER NOT NULL,
    completed_mono_ns INTEGER,
    note TEXT,
    PRIMARY KEY (session_id, cycle, attempt)
);
CREATE TABLE IF NOT EXISTS _stage_datagrams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    probe_id TEXT NOT NULL,
    tx_id TEXT NOT NULL UNIQUE,
    request_payload TEXT NOT NULL,
    request_lease_version INTEGER NOT NULL,
    planned_mono_ns INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    start_mono_ns INTEGER NOT NULL,
    end_mono_ns INTEGER NOT NULL,
    acked INTEGER NOT NULL,
    total INTEGER NOT NULL,
    ratio_bp INTEGER NOT NULL,
    request_lease_version INTEGER NOT NULL,
    lease_version INTEGER NOT NULL,
    lease_renewed INTEGER NOT NULL,
    status_after TEXT NOT NULL,
    pause_reason_after TEXT,
    UNIQUE(session_id, cycle)
);
CREATE TABLE IF NOT EXISTS datagrams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    probe_id TEXT NOT NULL,
    tx_id TEXT NOT NULL UNIQUE,
    request_payload TEXT NOT NULL,
    request_lease_version INTEGER NOT NULL,
    planned_mono_ns INTEGER NOT NULL,
    send_mono_ns INTEGER,
    receive_mono_ns INTEGER,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    response_seq INTEGER,
    round_id INTEGER,
    FOREIGN KEY(round_id) REFERENCES rounds(id)
);
CREATE INDEX IF NOT EXISTS idx_datagrams_round ON datagrams(round_id, probe_id);
CREATE INDEX IF NOT EXISTS idx_datagrams_attempt ON datagrams(session_id, cycle, attempt);
CREATE TABLE IF NOT EXISTS pause_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    lease_version INTEGER NOT NULL,
    cursor_before INTEGER NOT NULL,
    cursor_after INTEGER NOT NULL,
    mono_ns INTEGER NOT NULL,
    details_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    cycle INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    kind TEXT NOT NULL,
    reason TEXT NOT NULL,
    mono_ns INTEGER NOT NULL,
    details_json TEXT NOT NULL
);
"""


TABLES = (
    "sessions",
    "round_attempts",
    "rounds",
    "datagrams",
    "pause_events",
    "recovery_events",
)


class Ledger:
    """A single-process writer ledger with durable SQLite transactions."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        declarations: DeclarationSet,
        *,
        logger: StructuredLogger | None = None,
        recover_incomplete: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.declarations = declarations
        self.logger = logger or StructuredLogger()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path), timeout=5.0, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.executescript(SCHEMA)
        self._ensure_header()
        self.recovery_report = (
            self.recover_incomplete_attempts() if recover_incomplete else RecoveryReport((), ())
        )

    def _ensure_header(self) -> None:
        with self._lock:
            existing = {
                row["key"]: row["value"]
                for row in self._conn.execute("SELECT key, value FROM meta")
            }
            expected = self._expected_meta()
            if not existing:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    for key, value in expected.items():
                        self._conn.execute(
                            "INSERT INTO meta(key, value) VALUES (?, ?)", (key, value)
                        )
                    now = time.monotonic_ns()
                    for session in self.declarations.sessions:
                        self._conn.execute(
                            """
                            INSERT INTO sessions(
                                session_id, status, lease_version, cycle_cursor,
                                consecutive_unrenewed, last_ratio_bp, pause_reason,
                                last_round_cycle, updated_mono_ns
                            ) VALUES (?, 'ACTIVE', 1, 0, 0, NULL, NULL, NULL, ?)
                            """,
                            (session.session_id, now),
                        )
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                self.logger.emit(
                    "ledger.created",
                    path=str(self.path),
                    content_digest=self.declarations.content_digest,
                )
                return

            actual_format = _parse_int(existing.get("format_version"))
            if actual_format != SCHEMA_VERSION:
                raise LedgerError(
                    "unsupported ledger format version",
                    "ledger.format_version",
                    {"supported": SCHEMA_VERSION, "actual": existing.get("format_version")},
                    code="E_FORMAT",
                )
            mismatches = []
            for key, expected_value in expected.items():
                actual_value = existing.get(key)
                if actual_value != expected_value:
                    mismatches.append(
                        {"field": key, "expected": expected_value, "actual": actual_value}
                    )
            if mismatches:
                raise GrantError(
                    "declaration versions or content digest do not match ledger",
                    f"ledger.{mismatches[0]['field']}",
                    {"mismatches": mismatches},
                )
            session_ids = {
                row["session_id"] for row in self._conn.execute("SELECT session_id FROM sessions")
            }
            expected_ids = {session.session_id for session in self.declarations.sessions}
            if session_ids != expected_ids:
                raise GrantError(
                    "ledger session set does not match declarations",
                    "ledger.sessions",
                    {
                        "missing": sorted(expected_ids - session_ids),
                        "extra": sorted(session_ids - expected_ids),
                    },
                )

    def _expected_meta(self) -> dict[str, str]:
        return {
            "format_version": str(SCHEMA_VERSION),
            "session_set_version": str(self.declarations.session_set_version),
            "probe_parameter_version": str(self.declarations.probe_parameter_version),
            "ack_policy_version": str(self.declarations.ack_policy_version),
            "content_digest": self.declarations.content_digest,
            "session_definitions": canonical_json(
                [s.canonical() for s in sorted(self.declarations.sessions, key=lambda s: s.session_id)]
            ),
            "probe_definitions": canonical_json(
                {
                    probe_id: self.declarations.probes[probe_id].canonical()
                    for probe_id in sorted(self.declarations.probes)
                }
            ),
        }

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_conn", None) is not None:
                self._conn.close()
                self._conn = None  # type: ignore[assignment]

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def header(self) -> dict[str, Any]:
        with self._lock:
            meta = {
                row["key"]: row["value"]
                for row in self._conn.execute("SELECT key, value FROM meta")
            }
        for key in (
            "format_version",
            "session_set_version",
            "probe_parameter_version",
            "ack_policy_version",
        ):
            meta[key] = _parse_int(meta.get(key))
        return meta

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise LedgerError(
                "unknown session in ledger", "session_id", {"session_id": session_id}
            )
        return dict(row)

    def all_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM sessions ORDER BY session_id").fetchall()
        return [dict(row) for row in rows]

    def max_attempt(self, session_id: str, cycle: int) -> int:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT COALESCE(MAX(attempt), 0) AS n FROM (
                    SELECT attempt FROM round_attempts WHERE session_id = ? AND cycle = ?
                    UNION ALL
                    SELECT attempt FROM _stage_attempts WHERE session_id = ? AND cycle = ?
                )
                """,
                (session_id, cycle, session_id, cycle),
            ).fetchone()
        return int(row["n"])

    def stage_attempt(
        self,
        *,
        session_id: str,
        cycle: int,
        attempt: int,
        records: Iterable[dict[str, Any]],
        started_mono_ns: int,
    ) -> None:
        record_list = list(records)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO _stage_attempts(
                        session_id, cycle, attempt, status, started_mono_ns,
                        completed_mono_ns, note
                    ) VALUES (?, ?, ?, 'RUNNING', ?, NULL, NULL)
                    """,
                    (session_id, cycle, attempt, started_mono_ns),
                )
                for record in record_list:
                    self._conn.execute(
                        """
                        INSERT INTO _stage_datagrams(
                            session_id, cycle, attempt, probe_id, tx_id,
                            request_payload, request_lease_version, planned_mono_ns
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            cycle,
                            attempt,
                            record["probe_id"],
                            record["tx_id"],
                            record["request_payload"],
                            record["request_lease_version"],
                            record["planned_mono_ns"],
                        ),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        self.logger.emit(
            "round.attempt_staged",
            session_id=session_id,
            cycle=cycle,
            attempt=attempt,
            probes=len(record_list),
        )

    def complete_round(
        self,
        *,
        session_id: str,
        cycle: int,
        attempt: int,
        start_mono_ns: int,
        end_mono_ns: int,
        acked: int,
        total: int,
        ratio_bp: int,
        request_lease_version: int,
        lease_version: int,
        lease_renewed: bool,
        consecutive_unrenewed: int,
        status_after: str,
        pause_reason_after: str | None,
        records: Iterable[dict[str, Any]],
        pause_event: dict[str, Any] | None = None,
    ) -> int:
        record_list = list(records)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                attempt_row = self._conn.execute(
                    """
                    SELECT status FROM _stage_attempts
                    WHERE session_id = ? AND cycle = ? AND attempt = ?
                    """,
                    (session_id, cycle, attempt),
                ).fetchone()
                if attempt_row is None or attempt_row["status"] != "RUNNING":
                    raise LedgerError(
                        "round attempt is not staged and running",
                        "_stage_attempts.status",
                        {"session_id": session_id, "cycle": cycle, "attempt": attempt},
                    )
                staged_rows = {
                    row["tx_id"]: dict(row)
                    for row in self._conn.execute(
                        """
                        SELECT * FROM _stage_datagrams
                        WHERE session_id = ? AND cycle = ? AND attempt = ?
                        """,
                        (session_id, cycle, attempt),
                    )
                }
                expected_tx_ids = {record["tx_id"] for record in record_list}
                if set(staged_rows) != expected_tx_ids:
                    raise LedgerError(
                        "staged datagram set does not match completed round",
                        "datagrams.tx_id",
                        {
                            "missing": sorted(expected_tx_ids - set(staged_rows)),
                            "extra": sorted(set(staged_rows) - expected_tx_ids),
                        },
                    )
                cursor = self._conn.execute(
                    """
                    INSERT INTO rounds(
                        session_id, cycle, attempt, start_mono_ns, end_mono_ns,
                        acked, total, ratio_bp, request_lease_version,
                        lease_version, lease_renewed, status_after,
                        pause_reason_after
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        cycle,
                        attempt,
                        start_mono_ns,
                        end_mono_ns,
                        acked,
                        total,
                        ratio_bp,
                        request_lease_version,
                        lease_version,
                        int(lease_renewed),
                        status_after,
                        pause_reason_after,
                    ),
                )
                round_id = int(cursor.lastrowid)
                for record in record_list:
                    staged = staged_rows[record["tx_id"]]
                    self._conn.execute(
                        """
                        INSERT INTO datagrams(
                            session_id, cycle, attempt, probe_id, tx_id,
                            request_payload, request_lease_version, planned_mono_ns,
                            send_mono_ns, receive_mono_ns, outcome, reason,
                            response_seq, round_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            cycle,
                            attempt,
                            record["probe_id"],
                            record["tx_id"],
                            staged["request_payload"],
                            staged["request_lease_version"],
                            staged["planned_mono_ns"],
                            record.get("send_mono_ns"),
                            record.get("receive_mono_ns"),
                            record["outcome"],
                            record["reason"],
                            record.get("response_seq"),
                            round_id,
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO round_attempts(
                        session_id, cycle, attempt, status, started_mono_ns,
                        completed_mono_ns, note
                    ) VALUES (?, ?, ?, 'COMPLETED', ?, ?, NULL)
                    """,
                    (session_id, cycle, attempt, start_mono_ns, end_mono_ns),
                )
                self._conn.execute(
                    """
                    UPDATE sessions SET
                        status = ?, lease_version = ?, cycle_cursor = ?,
                        consecutive_unrenewed = ?, last_ratio_bp = ?,
                        pause_reason = ?, last_round_cycle = ?, updated_mono_ns = ?
                    WHERE session_id = ?
                    """,
                    (
                        status_after,
                        lease_version,
                        cycle,
                        consecutive_unrenewed,
                        ratio_bp,
                        pause_reason_after,
                        cycle,
                        end_mono_ns,
                        session_id,
                    ),
                )
                if pause_event is not None:
                    self._insert_pause_event(pause_event)
                self._conn.execute(
                    "DELETE FROM _stage_datagrams WHERE session_id = ? AND cycle = ? AND attempt = ?",
                    (session_id, cycle, attempt),
                )
                self._conn.execute(
                    "DELETE FROM _stage_attempts WHERE session_id = ? AND cycle = ? AND attempt = ?",
                    (session_id, cycle, attempt),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        self.logger.emit(
            "round.committed",
            session_id=session_id,
            cycle=cycle,
            attempt=attempt,
            acked=acked,
            total=total,
            ratio_bp=ratio_bp,
            lease_version=lease_version,
            lease_renewed=lease_renewed,
            status=status_after,
            pause_reason=pause_reason_after,
        )
        return round_id

    def commit_dangling_pause(
        self,
        *,
        session_id: str,
        cycle: int,
        attempt: int,
        events: Iterable[dict[str, Any]],
        records: Iterable[dict[str, Any]],
        pause_event: dict[str, Any],
        updated_mono_ns: int,
        lease_version: int,
        cursor: int,
        pause_reason: str,
    ) -> None:
        event_list = list(events)
        record_list = list(records)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                staged_rows = {
                    row["tx_id"]: dict(row)
                    for row in self._conn.execute(
                        """
                        SELECT * FROM _stage_datagrams
                        WHERE session_id = ? AND cycle = ? AND attempt = ?
                        """,
                        (session_id, cycle, attempt),
                    )
                }
                expected_tx_ids = {record["tx_id"] for record in record_list}
                if set(staged_rows) != expected_tx_ids:
                    raise LedgerError(
                        "staged datagram set does not match dangling pause",
                        "datagrams.tx_id",
                        {
                            "missing": sorted(expected_tx_ids - set(staged_rows)),
                            "extra": sorted(set(staged_rows) - expected_tx_ids),
                        },
                    )
                for record in record_list:
                    staged = staged_rows[record["tx_id"]]
                    self._conn.execute(
                        """
                        INSERT INTO datagrams(
                            session_id, cycle, attempt, probe_id, tx_id,
                            request_payload, request_lease_version, planned_mono_ns,
                            send_mono_ns, receive_mono_ns, outcome, reason,
                            response_seq, round_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'DANGLING_UNKNOWN',
                                  'session_paused_before_receive', NULL, NULL)
                        """,
                        (
                            session_id,
                            cycle,
                            attempt,
                            record["probe_id"],
                            record["tx_id"],
                            staged["request_payload"],
                            staged["request_lease_version"],
                            staged["planned_mono_ns"],
                            record.get("send_mono_ns"),
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO round_attempts(
                        session_id, cycle, attempt, status, started_mono_ns,
                        completed_mono_ns, note
                    ) VALUES (?, ?, ?, 'PAUSED', ?, ?, 'paused_before_receive')
                    """,
                    (
                        session_id,
                        cycle,
                        attempt,
                        int(pause_event.get("started_mono_ns", updated_mono_ns)),
                        updated_mono_ns,
                    ),
                )
                self._conn.execute(
                    """
                    UPDATE sessions SET status = 'PAUSED', cycle_cursor = ?,
                        pause_reason = ?, updated_mono_ns = ?
                    WHERE session_id = ?
                    """,
                    (cursor, pause_reason, updated_mono_ns, session_id),
                )
                self._insert_pause_event(pause_event)
                for event in event_list:
                    self._insert_pause_event(event)
                self._conn.execute(
                    "DELETE FROM _stage_datagrams WHERE session_id = ? AND cycle = ? AND attempt = ?",
                    (session_id, cycle, attempt),
                )
                self._conn.execute(
                    "DELETE FROM _stage_attempts WHERE session_id = ? AND cycle = ? AND attempt = ?",
                    (session_id, cycle, attempt),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        self.logger.emit(
            "session.paused_inflight",
            session_id=session_id,
            cycle=cycle,
            attempt=attempt,
            reason=pause_reason,
            dangling=len(record_list),
            lease_version=lease_version,
        )

    def mark_paused_idle(
        self,
        *,
        session_id: str,
        reason: str,
        event_cycle: int,
        lease_version: int,
        cursor: int,
        mono_ns: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    UPDATE sessions SET status = 'PAUSED', pause_reason = ?,
                        updated_mono_ns = ? WHERE session_id = ?
                    """,
                    (reason, mono_ns, session_id),
                )
                self._insert_pause_event(
                    {
                        "session_id": session_id,
                        "kind": "PAUSE",
                        "reason": reason,
                        "cycle": event_cycle,
                        "lease_version": lease_version,
                        "cursor_before": cursor,
                        "cursor_after": cursor,
                        "mono_ns": mono_ns,
                        "details": details or {},
                    }
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        self.logger.emit(
            "session.paused",
            session_id=session_id,
            reason=reason,
            cycle=event_cycle,
            cursor=cursor,
            lease_version=lease_version,
        )

    def mark_resumed(
        self,
        *,
        session_id: str,
        reason: str,
        event_cycle: int,
        lease_version: int,
        cursor: int,
        mono_ns: int,
        reset_misses: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if reset_misses:
                    self._conn.execute(
                        """
                        UPDATE sessions SET status = 'ACTIVE', pause_reason = NULL,
                            consecutive_unrenewed = 0, updated_mono_ns = ?
                        WHERE session_id = ?
                        """,
                        (mono_ns, session_id),
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE sessions SET status = 'ACTIVE', pause_reason = NULL,
                            updated_mono_ns = ? WHERE session_id = ?
                        """,
                        (mono_ns, session_id),
                    )
                self._insert_pause_event(
                    {
                        "session_id": session_id,
                        "kind": "RESUME",
                        "reason": reason,
                        "cycle": event_cycle,
                        "lease_version": lease_version,
                        "cursor_before": cursor,
                        "cursor_after": cursor,
                        "mono_ns": mono_ns,
                        "details": details or {},
                    }
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        self.logger.emit(
            "session.resumed",
            session_id=session_id,
            reason=reason,
            cycle=event_cycle,
            cursor=cursor,
            lease_version=lease_version,
        )

    def mark_terminated(
        self,
        *,
        session_id: str,
        reason: str,
        event_cycle: int,
        lease_version: int,
        cursor: int,
        mono_ns: int,
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    UPDATE sessions SET status = 'TERMINATED', pause_reason = ?,
                        updated_mono_ns = ? WHERE session_id = ?
                    """,
                    (reason, mono_ns, session_id),
                )
                self._insert_pause_event(
                    {
                        "session_id": session_id,
                        "kind": "TERMINATE",
                        "reason": reason,
                        "cycle": event_cycle,
                        "lease_version": lease_version,
                        "cursor_before": cursor,
                        "cursor_after": cursor,
                        "mono_ns": mono_ns,
                        "details": {},
                    }
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        self.logger.emit(
            "session.terminated",
            session_id=session_id,
            reason=reason,
            cycle=event_cycle,
        )

    def _insert_pause_event(self, event: dict[str, Any]) -> None:
        self._conn.execute(
            """
            INSERT INTO pause_events(
                session_id, kind, reason, cycle, lease_version,
                cursor_before, cursor_after, mono_ns, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["session_id"],
                event["kind"],
                event["reason"],
                event["cycle"],
                event["lease_version"],
                event["cursor_before"],
                event["cursor_after"],
                event["mono_ns"],
                canonical_json(event.get("details", {})),
            ),
        )

    def recover_incomplete_attempts(self) -> RecoveryReport:
        recovered: list[dict[str, Any]] = []
        rerun: list[dict[str, Any]] = []
        now = time.monotonic_ns()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM _stage_attempts WHERE status = 'RUNNING'
                ORDER BY session_id, cycle, attempt
                """
            ).fetchall()
            if not rows:
                return RecoveryReport((), ())
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for row in rows:
                    staged_datagrams = self._conn.execute(
                        """
                        SELECT * FROM _stage_datagrams
                        WHERE session_id = ? AND cycle = ? AND attempt = ?
                        ORDER BY id
                        """,
                        (row["session_id"], row["cycle"], row["attempt"]),
                    ).fetchall()
                    for datagram in staged_datagrams:
                        self._conn.execute(
                            """
                            INSERT INTO datagrams(
                                session_id, cycle, attempt, probe_id, tx_id,
                                request_payload, request_lease_version, planned_mono_ns,
                                send_mono_ns, receive_mono_ns, outcome, reason,
                                response_seq, round_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 'UNKNOWN',
                                      'uncommitted_round_unknown_rerun_required', NULL, NULL)
                            """,
                            (
                                datagram["session_id"],
                                datagram["cycle"],
                                datagram["attempt"],
                                datagram["probe_id"],
                                datagram["tx_id"],
                                datagram["request_payload"],
                                datagram["request_lease_version"],
                                datagram["planned_mono_ns"],
                            ),
                        )
                    self._conn.execute(
                        """
                        INSERT INTO round_attempts(
                            session_id, cycle, attempt, status, started_mono_ns,
                            completed_mono_ns, note
                        ) VALUES (?, ?, ?, 'UNKNOWN', ?, ?,
                                  'uncommitted_round_unknown_rerun_required')
                        """,
                        (
                            row["session_id"],
                            row["cycle"],
                            row["attempt"],
                            row["started_mono_ns"],
                            now,
                        ),
                    )
                    details = {
                        "action": "mark_unknown_and_rerun",
                        "round_was_committed": False,
                    }
                    self._conn.execute(
                        """
                        INSERT INTO recovery_events(
                            session_id, cycle, attempt, kind, reason,
                            mono_ns, details_json
                        ) VALUES (?, ?, ?, 'INCOMPLETE_ROUND', 'E_RERUN_REQUIRED', ?, ?)
                        """,
                        (
                            row["session_id"],
                            row["cycle"],
                            row["attempt"],
                            now,
                            canonical_json(details),
                        ),
                    )
                    item = {
                        "session_id": row["session_id"],
                        "cycle": row["cycle"],
                        "attempt": row["attempt"],
                        "action": "mark_unknown_rerun_required",
                    }
                    recovered.append(item)
                    rerun.append(item)
                    self._conn.execute(
                        "DELETE FROM _stage_datagrams WHERE session_id = ? AND cycle = ? AND attempt = ?",
                        (row["session_id"], row["cycle"], row["attempt"]),
                    )
                    self._conn.execute(
                        "DELETE FROM _stage_attempts WHERE session_id = ? AND cycle = ? AND attempt = ?",
                        (row["session_id"], row["cycle"], row["attempt"]),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        output = RecoveryReport(tuple(recovered), tuple(rerun))
        self.logger.emit(
            "ledger.incomplete_attempts_recovered",
            recovered=len(recovered),
            report=output.to_dict(),
        )
        return output

    def round_records(
        self,
        session_id: str | None = None,
        *,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("r.session_id = ?")
            params.append(session_id)
        if start_cycle is not None:
            clauses.append("r.cycle >= ?")
            params.append(start_cycle)
        if end_cycle is not None:
            clauses.append("r.cycle <= ?")
            params.append(end_cycle)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            round_rows = self._conn.execute(
                f"SELECT r.* FROM rounds r {where} ORDER BY r.session_id, r.cycle",
                tuple(params),
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in round_rows:
                datagram_rows = self._conn.execute(
                    """
                    SELECT * FROM datagrams WHERE round_id = ?
                    ORDER BY probe_id
                    """,
                    (row["id"],),
                ).fetchall()
                result.append(
                    {
                        "session_id": row["session_id"],
                        "cycle": row["cycle"],
                        "attempt": row["attempt"],
                        "start_mono_ns": row["start_mono_ns"],
                        "end_mono_ns": row["end_mono_ns"],
                        "acked": row["acked"],
                        "total": row["total"],
                        "ratio_bp": row["ratio_bp"],
                        "request_lease_version": row["request_lease_version"],
                        "lease_version": row["lease_version"],
                        "lease_renewed": bool(row["lease_renewed"]),
                        "status_after": row["status_after"],
                        "pause_reason_after": row["pause_reason_after"],
                        "probes": {
                            d["probe_id"]: {
                                "outcome": d["outcome"],
                                "reason": d["reason"],
                                "response_seq": d["response_seq"],
                                "request_lease_version": d["request_lease_version"],
                                "send_mono_ns": d["send_mono_ns"],
                                "receive_mono_ns": d["receive_mono_ns"],
                            }
                            for d in datagram_rows
                        },
                    }
                )
        return result

    def logical_round_records(
        self,
        session_id: str | None = None,
        *,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self.round_records(
            session_id, start_cycle=start_cycle, end_cycle=end_cycle
        )
        result: list[dict[str, Any]] = []
        for record in records:
            result.append(
                {
                    "acked": record["acked"],
                    "attempt": record["attempt"],
                    "cycle": record["cycle"],
                    "lease_renewed": record["lease_renewed"],
                    "lease_version": record["lease_version"],
                    "pause_reason_after": record["pause_reason_after"],
                    "probes": {
                        probe_id: {
                            "outcome": probe["outcome"],
                            "reason": probe["reason"],
                            "response_seq": probe["response_seq"],
                        }
                        for probe_id, probe in record["probes"].items()
                    },
                    "ratio_bp": record["ratio_bp"],
                    "request_lease_version": record["request_lease_version"],
                    "session_id": record["session_id"],
                    "status_after": record["status_after"],
                    "total": record["total"],
                }
            )
        return result

    def logical_ledger_bytes(
        self,
        session_id: str | None = None,
        *,
        start_cycle: int | None = None,
        end_cycle: int | None = None,
    ) -> bytes:
        records = self.logical_round_records(
            session_id, start_cycle=start_cycle, end_cycle=end_cycle
        )
        return b"".join(
            (canonical_json(record) + "\n").encode("utf-8") for record in records
        )

    def pause_events(self, session_id: str | None = None) -> list[dict[str, Any]]:
        params: tuple[Any, ...] = () if session_id is None else (session_id,)
        where = "" if session_id is None else "WHERE session_id = ?"
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM pause_events {where} ORDER BY id", params
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result

    def recovery_events(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM recovery_events ORDER BY id"
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json"))
            result.append(item)
        return result

    def state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                state = {
                    table: [
                        dict(row)
                        for row in self._conn.execute(
                            f"SELECT * FROM {table} ORDER BY rowid"
                        )
                    ]
                    for table in TABLES
                }
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return state

    def save_snapshot(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        snapshot_path = Path(path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_snapshot()
        state_digest = digest(state)
        snapshot = {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "format_version": self.declarations.format_version,
            "session_set_version": self.declarations.session_set_version,
            "probe_parameter_version": self.declarations.probe_parameter_version,
            "ack_policy_version": self.declarations.ack_policy_version,
            "content_digest": self.declarations.content_digest,
            "session_definitions": json.loads(
                self._expected_meta()["session_definitions"]
            ),
            "probe_definitions": json.loads(
                self._expected_meta()["probe_definitions"]
            ),
            "created_monotonic_ns": time.monotonic_ns(),
            "state_digest": state_digest,
            "state": state,
        }
        payload = canonical_json(snapshot).encode("utf-8")
        temporary = snapshot_path.with_suffix(snapshot_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, snapshot_path)
        self.logger.emit(
            "snapshot.saved",
            path=str(snapshot_path),
            state_digest=state_digest,
            bytes=len(payload),
        )
        return {
            "path": str(snapshot_path),
            "state_digest": state_digest,
            "bytes": len(payload),
        }


def _parse_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_snapshot(path: str | os.PathLike[str]) -> dict[str, Any]:
    snapshot_path = Path(path)
    try:
        value = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LedgerError(
            "snapshot file does not exist", "snapshot", {"path": str(snapshot_path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise LedgerError(
            "invalid snapshot JSON", "snapshot", {"reason": exc.msg}, code="E_SNAPSHOT"
        ) from exc
    if not isinstance(value, dict):
        raise LedgerError("snapshot must be an object", "snapshot", code="E_SNAPSHOT")
    if value.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
        raise LedgerError(
            "unsupported snapshot format version",
            "snapshot.snapshot_format_version",
            {
                "supported": SNAPSHOT_FORMAT_VERSION,
                "actual": value.get("snapshot_format_version"),
            },
            code="E_FORMAT",
        )
    state = value.get("state")
    if not isinstance(state, dict):
        raise LedgerError("snapshot state is missing", "snapshot.state", code="E_SNAPSHOT")
    actual_digest = digest(state)
    if actual_digest != value.get("state_digest"):
        raise LedgerError(
            "snapshot state digest mismatch",
            "snapshot.state_digest",
            {"expected": value.get("state_digest"), "actual": actual_digest},
            code="E_SNAPSHOT",
        )
    return value


def restore_from_snapshot(
    *,
    snapshot_path: str | os.PathLike[str],
    db_path: str | os.PathLike[str],
    declarations: DeclarationSet,
    logger: StructuredLogger | None = None,
) -> RestoreReport:
    """Wholesale restore a snapshot; never merge row sets.

    The snapshot must describe the current declarations.  A mismatching disk
    ledger is first backed up as a read-only SQLite file and then replaced in
    full.
    """

    logger = logger or StructuredLogger()
    snapshot = load_snapshot(snapshot_path)
    mismatches: list[dict[str, Any]] = []
    expected_descriptor = {
        "format_version": declarations.format_version,
        "session_set_version": declarations.session_set_version,
        "probe_parameter_version": declarations.probe_parameter_version,
        "ack_policy_version": declarations.ack_policy_version,
        "content_digest": declarations.content_digest,
    }
    for field, expected in expected_descriptor.items():
        actual = snapshot.get(field)
        if actual != expected:
            mismatches.append({"field": field, "expected": expected, "actual": actual})
    if mismatches:
        raise GrantError(
            "snapshot descriptors do not match current declarations",
            f"snapshot.{mismatches[0]['field']}",
            {"mismatches": mismatches},
        )

    target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    old_summary = _read_ledger_summary(target)
    new_sessions = {
        row["session_id"] for row in snapshot["state"].get("sessions", []) if "session_id" in row
    }
    old_sessions = set(old_summary.get("sessions", {}))
    new_session_definitions = {
        item["session_id"]: item
        for item in snapshot.get("session_definitions", [])
        if isinstance(item, dict) and "session_id" in item
    }
    old_session_definitions = old_summary.get("session_definitions", {})
    new_probes = set(snapshot.get("probe_definitions", {}))
    old_probes = set(old_summary.get("probes", {}))
    added_sessions = tuple(sorted(new_sessions - old_sessions))
    removed_sessions = tuple(sorted(old_sessions - new_sessions))
    changed_sessions = tuple(
        sorted(
            session_id
            for session_id in old_sessions & new_sessions
            if (
                old_summary.get("sessions", {}).get(session_id)
                != _snapshot_session_logical(snapshot, session_id)
                or old_session_definitions.get(session_id)
                != new_session_definitions.get(session_id)
            )
        )
    )
    added_probes = tuple(sorted(new_probes - old_probes))
    removed_probes = tuple(sorted(old_probes - new_probes))
    changed_probes = tuple(
        sorted(
            probe_id
            for probe_id in old_probes & new_probes
            if old_summary.get("probes", {}).get(probe_id)
            != snapshot.get("probe_definitions", {}).get(probe_id)
        )
    )
    disk_mismatches: list[dict[str, Any]] = []
    if old_summary.get("exists"):
        disk_descriptor = old_summary.get("descriptor", {})
        for field, expected in expected_descriptor.items():
            actual = disk_descriptor.get(field)
            if actual != expected:
                disk_mismatches.append(
                    {"field": f"ledger.{field}", "expected": expected, "actual": actual}
                )
        if old_summary.get("has_staging"):
            disk_mismatches.append(
                {
                    "field": "ledger.staging",
                    "expected": "no_uncommitted_attempts",
                    "actual": "uncommitted_attempts_present",
                }
            )
        if old_summary.get("state_digest") != snapshot["state_digest"]:
            disk_mismatches.append(
                {
                    "field": "ledger.state_digest",
                    "expected": snapshot["state_digest"],
                    "actual": old_summary.get("state_digest"),
                }
            )

    if old_summary.get("exists") and not disk_mismatches:
        report = RestoreReport(
            replaced=False,
            backup_path=None,
            descriptor_mismatches=(),
            added_sessions=added_sessions,
            removed_sessions=removed_sessions,
            changed_sessions=changed_sessions,
            added_probes=added_probes,
            removed_probes=removed_probes,
            changed_probes=changed_probes,
            state_digest=snapshot["state_digest"],
        )
        logger.emit("snapshot.restore_skipped", report=report.to_dict())
        return report

    backup_path: str | None = None
    if old_summary.get("exists"):
        backup = target.with_name(
            f"{target.name}.readonly-backup.{time.time_ns()}.sqlite3"
        )
        _backup_database(target, backup)
        backup_path = str(backup)
        logger.emit("ledger.readonly_backup", path=backup_path)

    temporary = target.with_name(f"{target.name}.restore-{os.getpid()}.tmp")
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(temporary) + suffix).unlink()
        except FileNotFoundError:
            pass
    _write_snapshot_database(temporary, snapshot, declarations)
    for suffix in ("-wal", "-shm"):
        try:
            Path(str(target) + suffix).unlink()
        except FileNotFoundError:
            pass
    os.replace(temporary, target)
    report = RestoreReport(
        replaced=True,
        backup_path=backup_path,
        descriptor_mismatches=tuple(disk_mismatches),
        added_sessions=added_sessions,
        removed_sessions=removed_sessions,
        changed_sessions=changed_sessions,
        added_probes=added_probes,
        removed_probes=removed_probes,
        changed_probes=changed_probes,
        state_digest=snapshot["state_digest"],
    )
    logger.emit("snapshot.restored", report=report.to_dict())
    return report


def _snapshot_session_logical(snapshot: dict[str, Any], session_id: str) -> dict[str, Any] | None:
    for row in snapshot.get("state", {}).get("sessions", []):
        if row.get("session_id") == session_id:
            return {
                key: row.get(key)
                for key in (
                    "session_id",
                    "status",
                    "lease_version",
                    "cycle_cursor",
                    "consecutive_unrenewed",
                    "last_ratio_bp",
                    "pause_reason",
                    "last_round_cycle",
                )
            }
    return None


def _read_ledger_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "sessions": {}, "probes": {}, "descriptor": {}}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            meta = {
                row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM meta")
            }
            sessions = {
                row["session_id"]: {
                    key: row[key]
                    for key in (
                        "session_id",
                        "status",
                        "lease_version",
                        "cycle_cursor",
                        "consecutive_unrenewed",
                        "last_ratio_bp",
                        "pause_reason",
                        "last_round_cycle",
                    )
                }
                for row in conn.execute("SELECT * FROM sessions")
            }
            state = {
                table: [
                    dict(row)
                    for row in conn.execute(f"SELECT * FROM {table} ORDER BY rowid")
                ]
                for table in TABLES
            }
            stage_attempts = conn.execute(
                "SELECT COUNT(*) FROM _stage_attempts"
            ).fetchone()[0]
            stage_datagrams = conn.execute(
                "SELECT COUNT(*) FROM _stage_datagrams"
            ).fetchone()[0]
        finally:
            conn.close()
        probes = json.loads(meta.get("probe_definitions", "{}"))
        session_definition_list = json.loads(meta.get("session_definitions", "[]"))
        session_definitions = {
            item["session_id"]: item
            for item in session_definition_list
            if isinstance(item, dict) and "session_id" in item
        }
        descriptor = {
            key: _parse_int(meta.get(key))
            for key in (
                "format_version",
                "session_set_version",
                "probe_parameter_version",
                "ack_policy_version",
            )
        }
        descriptor["content_digest"] = meta.get("content_digest")
        return {
            "exists": True,
            "sessions": sessions,
            "session_definitions": session_definitions,
            "probes": probes,
            "descriptor": descriptor,
            "state_digest": digest(state),
            "has_staging": bool(stage_attempts or stage_datagrams),
        }
    except sqlite3.Error as exc:
        return {
            "exists": True,
            "sessions": {},
            "probes": {},
            "descriptor": {},
            "state_digest": None,
            "error": str(exc),
        }


def _backup_database(source: Path, destination: Path) -> None:
    try:
        src = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(str(destination))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
    except sqlite3.Error as exc:
        raise LedgerError(
            "failed to create read-only ledger backup",
            "ledger.backup",
            {"source": str(source), "reason": str(exc)},
        ) from exc
    try:
        os.chmod(destination, 0o444)
    except OSError:
        pass


def _write_snapshot_database(
    path: Path, snapshot: dict[str, Any], declarations: DeclarationSet
) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode = DELETE")
        conn.execute("PRAGMA synchronous = FULL")
        conn.executescript(SCHEMA)
        meta = {
            "format_version": str(SCHEMA_VERSION),
            "session_set_version": str(declarations.session_set_version),
            "probe_parameter_version": str(declarations.probe_parameter_version),
            "ack_policy_version": str(declarations.ack_policy_version),
            "content_digest": declarations.content_digest,
            "session_definitions": canonical_json(
                [s.canonical() for s in sorted(declarations.sessions, key=lambda s: s.session_id)]
            ),
            "probe_definitions": canonical_json(
                {
                    probe_id: declarations.probes[probe_id].canonical()
                    for probe_id in sorted(declarations.probes)
                }
            ),
        }
        conn.execute("BEGIN IMMEDIATE")
        try:
            for key, value in meta.items():
                conn.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (key, value))
            _insert_state(conn, snapshot["state"])
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    except sqlite3.Error as exc:
        raise LedgerError(
            "failed to write restored ledger", "ledger.restore", {"reason": str(exc)}
        ) from exc
    finally:
        conn.close()


def _insert_state(conn: sqlite3.Connection, state: dict[str, Any]) -> None:
    for table in TABLES:
        rows = state.get(table, [])
        if not rows:
            continue
        # Preserve row ids for rounds/datagrams/events while allowing explicit ids.
        columns = list(rows[0].keys())
        placeholders = ",".join("?" for _ in columns)
        column_sql = ",".join(columns)
        for row in rows:
            conn.execute(
                f"INSERT INTO {table}({column_sql}) VALUES ({placeholders})",
                tuple(row[column] for column in columns),
            )
