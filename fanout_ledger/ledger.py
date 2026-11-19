"""SQLite ledger for attempt records, outcomes and budget consumption.

Durability contract:

* every attempt is persisted in a *single* SQLite transaction that writes the
  attempt row, its outcome classification and the consumed budget together;
* a process killed while an attempt is still in flight leaves no row for it
  (the row is only inserted at attempt end), so recovery can never observe a
  half-written attempt.  On resume, attempts whose tracked sequence was never
  committed are explicitly marked ``unknown`` in the recovery ledger and must
  be re-run; an exit report is never produced from unknown state.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEDGER_SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    delivery_seq INTEGER NOT NULL,
    attempt_seq INTEGER NOT NULL,
    wait_start_ms INTEGER NOT NULL,
    wait_end_ms INTEGER NOT NULL,
    started_ms INTEGER NOT NULL,
    ended_ms INTEGER,
    elapsed_ms INTEGER,
    outcome TEXT NOT NULL,
    budget_consumed_ms INTEGER NOT NULL,
    backoff_ms INTEGER NOT NULL DEFAULT 0,
    http_status INTEGER,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS deliveries (
    session_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    delivery_seq INTEGER NOT NULL,
    final_outcome TEXT NOT NULL,
    attempts_used INTEGER NOT NULL,
    total_backoff_ms INTEGER NOT NULL,
    PRIMARY KEY (session_id, node_id, delivery_seq)
);
CREATE TABLE IF NOT EXISTS slo_judgments (
    session_id TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    judged_at_ms INTEGER NOT NULL,
    PRIMARY KEY (session_id)
);
CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    delivery_seq INTEGER NOT NULL,
    attempt_seq INTEGER NOT NULL,
    reason TEXT NOT NULL,
    detected_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS retry_exhaustions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    delivery_seq INTEGER NOT NULL,
    attempts_used INTEGER NOT NULL,
    total_backoff_ms INTEGER NOT NULL,
    last_outcome TEXT NOT NULL,
    marked_at_ms INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class AttemptRecord:
    node_id: str
    delivery_seq: int
    attempt_seq: int
    wait_start_ms: int
    wait_end_ms: int
    started_ms: int
    ended_ms: int
    elapsed_ms: int
    outcome: str
    budget_consumed_ms: int
    backoff_ms: int
    http_status: int | None
    note: str


class Ledger:
    def __init__(self, db_path: str | Path, session_id: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._lock = threading.RLock()
        # autocommit mode (isolation_level=None); every transaction is explicit.
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False,
                                    isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.row_factory = sqlite3.Row
        self._initialize()

    def _tx(self, body) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                body()
                self.conn.execute("COMMIT")
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise

    def _initialize(self) -> None:
        def body():
            # execute statements one by one: executescript() would force an
            # implicit commit and break the surrounding explicit transaction.
            for statement in [s.strip() for s in SCHEMA_SQL.split(";") if s.strip()]:
                self.conn.execute(statement)
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("schema_version", str(LEDGER_SCHEMA_VERSION)),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
                ("session_id", self.session_id),
            )
        self._tx(body)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def commit_attempt(self, record: AttemptRecord) -> None:
        """Atomically persist one finished attempt (record + outcome + budget)."""
        def body():
            self.conn.execute(
                """
                INSERT INTO attempts (
                    session_id, node_id, delivery_seq, attempt_seq,
                    wait_start_ms, wait_end_ms, started_ms, ended_ms,
                    elapsed_ms, outcome, budget_consumed_ms, backoff_ms,
                    http_status, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id, record.node_id, record.delivery_seq,
                    record.attempt_seq, record.wait_start_ms, record.wait_end_ms,
                    record.started_ms, record.ended_ms, record.elapsed_ms,
                    record.outcome, record.budget_consumed_ms, record.backoff_ms,
                    record.http_status, record.note,
                ),
            )
        self._tx(body)

    def commit_delivery(self, node_id: str, delivery_seq: int, final_outcome: str,
                        attempts_used: int, total_backoff_ms: int) -> None:
        def body():
            self.conn.execute(
                """
                INSERT OR REPLACE INTO deliveries (
                    session_id, node_id, delivery_seq, final_outcome,
                    attempts_used, total_backoff_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.session_id, node_id, delivery_seq, final_outcome,
                 attempts_used, total_backoff_ms),
            )
        self._tx(body)

    def save_judgment(self, canonical_json_text: str, judged_at_ms: int) -> None:
        def body():
            self.conn.execute(
                "INSERT OR REPLACE INTO slo_judgments"
                "(session_id, canonical_json, judged_at_ms) VALUES (?, ?, ?)",
                (self.session_id, canonical_json_text, judged_at_ms),
            )
        self._tx(body)

    def record_unknown(self, node_id: str, delivery_seq: int, attempt_seq: int,
                       reason: str, detected_at_ms: int) -> None:
        def body():
            self.conn.execute(
                """
                INSERT INTO recovery_events
                    (session_id, node_id, delivery_seq, attempt_seq, reason, detected_at_ms)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (self.session_id, node_id, delivery_seq, attempt_seq, reason,
                 detected_at_ms),
            )
        self._tx(body)

    def mark_retry_exhaustion(self, node_id: str, delivery_seq: int, attempts_used: int,
                              total_backoff_ms: int, last_outcome: str,
                              marked_at_ms: int) -> None:
        """Persist the E_RETRY terminal marker for a delivery (not a network attempt)."""
        def body():
            self.conn.execute(
                """
                INSERT INTO retry_exhaustions (
                    session_id, node_id, delivery_seq, attempts_used,
                    total_backoff_ms, last_outcome, marked_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (self.session_id, node_id, delivery_seq, attempts_used,
                 total_backoff_ms, last_outcome, marked_at_ms),
            )
        self._tx(body)

    def fetch_attempts(self) -> list[AttemptRecord]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM attempts WHERE session_id = ? "
                "ORDER BY node_id, delivery_seq, attempt_seq",
                (self.session_id,),
            ).fetchall()
        return [
            AttemptRecord(
                node_id=row["node_id"], delivery_seq=row["delivery_seq"],
                attempt_seq=row["attempt_seq"], wait_start_ms=row["wait_start_ms"],
                wait_end_ms=row["wait_end_ms"], started_ms=row["started_ms"],
                ended_ms=row["ended_ms"], elapsed_ms=row["elapsed_ms"],
                outcome=row["outcome"], budget_consumed_ms=row["budget_consumed_ms"],
                backoff_ms=row["backoff_ms"], http_status=row["http_status"],
                note=row["note"],
            )
            for row in rows
        ]

    def fetch_deliveries(self) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM deliveries WHERE session_id = ? "
                "ORDER BY node_id, delivery_seq", (self.session_id,),
            ).fetchall()

    def fetch_meta(self) -> dict[str, str]:
        with self._lock:
            rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def fetch_recovery_events(self) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM recovery_events WHERE session_id = ? ORDER BY id",
                (self.session_id,),
            ).fetchall()

    def fetch_retry_exhaustions(self) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT * FROM retry_exhaustions WHERE session_id = ? "
                "ORDER BY node_id, delivery_seq", (self.session_id,),
            ).fetchall()

    def has_recovery_event(self, node_id: str, delivery_seq: int) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM recovery_events WHERE session_id = ? "
                "AND node_id = ? AND delivery_seq = ? LIMIT 1",
                (self.session_id, node_id, delivery_seq),
            ).fetchone()
        return row is not None

    def max_committed_attempt(self, node_id: str, delivery_seq: int) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT MAX(attempt_seq) AS m FROM attempts WHERE session_id = ? "
                "AND node_id = ? AND delivery_seq = ?",
                (self.session_id, node_id, delivery_seq),
            ).fetchone()
        return -1 if row is None or row["m"] is None else int(row["m"])


def read_meta(db_path: str | Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {row["key"]: row["value"] for row in rows}
    finally:
        conn.close()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
