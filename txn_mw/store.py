"""SQLite persistence, append-only event sequencing and ledger digests."""

from __future__ import annotations

import contextlib
import copy
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from . import FORMAT_VERSION
from .canonical import canonical_json, ledger_digest
from .codec import encode_record


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS txns (
    txn_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('half','committable','consumed')),
    disposition TEXT NOT NULL,
    body_json TEXT NOT NULL,
    body_digest TEXT NOT NULL,
    half_at_wall_ms INTEGER NOT NULL,
    deadline_mono_ns INTEGER NOT NULL,
    query_host TEXT NOT NULL,
    query_port INTEGER NOT NULL,
    query_max INTEGER NOT NULL,
    query_count INTEGER NOT NULL DEFAULT 0,
    first_query_mono_ns INTEGER,
    last_query_mono_ns INTEGER,
    next_query_mono_ns INTEGER,
    query_inflight INTEGER NOT NULL DEFAULT 0,
    last_query_result TEXT,
    ack_sent INTEGER NOT NULL DEFAULT 0,
    pending_command TEXT,
    pending_command_seq INTEGER,
    effect_id TEXT,
    consumed_at_mono_ns INTEGER,
    terminal_reason TEXT,
    conflict_reason TEXT,
    updated_mono_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq INTEGER NOT NULL UNIQUE,
    txn_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL,
    body_digest TEXT NOT NULL,
    event_type TEXT NOT NULL,
    record BLOB NOT NULL,
    created_mono_ns INTEGER NOT NULL,
    valid INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS events_txn_seq ON events(txn_id, event_seq);

CREATE TABLE IF NOT EXISTS idempotency (
    idem_key TEXT PRIMARY KEY,
    txn_id TEXT NOT NULL UNIQUE,
    effect_id TEXT NOT NULL,
    consumer_id TEXT NOT NULL,
    created_mono_ns INTEGER NOT NULL,
    last_access_mono_ns INTEGER NOT NULL,
    consume_count INTEGER NOT NULL DEFAULT 1,
    result_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS idempotency_evictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    idem_key TEXT NOT NULL,
    txn_id TEXT NOT NULL,
    effect_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    evicted_mono_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS pulls (
    request_id TEXT PRIMARY KEY,
    txn_id TEXT NOT NULL,
    consumer_id TEXT NOT NULL,
    started_mono_ns INTEGER NOT NULL,
    completed_mono_ns INTEGER,
    status TEXT NOT NULL,
    effect_id TEXT,
    duplicate INTEGER NOT NULL DEFAULT 0,
    result_json TEXT
);

CREATE TABLE IF NOT EXISTS latency (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL CHECK (operation IN ('query','pull')),
    txn_id TEXT NOT NULL,
    request_id TEXT,
    start_mono_ns INTEGER NOT NULL,
    end_mono_ns INTEGER NOT NULL,
    duration_us INTEGER NOT NULL,
    budget_us INTEGER NOT NULL,
    exceeded INTEGER NOT NULL,
    truth_source TEXT NOT NULL DEFAULT 'sqlite.latency.row'
);
CREATE INDEX IF NOT EXISTS latency_txn_op ON latency(txn_id, operation);

CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    winner TEXT NOT NULL,
    created_mono_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS quarantine (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    row_id INTEGER,
    event_seq INTEGER,
    txn_id TEXT,
    reason TEXT NOT NULL,
    path TEXT,
    raw_hex TEXT,
    quarantined_mono_ns INTEGER NOT NULL
);
"""


def ledger_content_digest(snapshot: dict[str, Any]) -> str:
    """Digest the complete ledger except the digest field and runtime clocks."""
    unsigned = {k: copy.deepcopy(v) for k, v in snapshot.items() if k != "content_digest"}
    for row in unsigned.get("transactions", []):
        row.pop("updated_mono_ns", None)
    return ledger_digest(unsigned)


class Store:
    """One serialized SQLite connection safe for the threaded TCP server.

    SQLite remains the source of truth.  The process-wide lock serializes state
    transitions, while network waits always happen outside that lock.
    """

    def __init__(self, db_path: str | os.PathLike[str]):
        self.path = Path(db_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=10.0
        )
        self._conn.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.executescript(SCHEMA)
            self._ensure_meta()

    def _ensure_meta(self) -> None:
        defaults = {
            "format_version": str(FORMAT_VERSION),
            "next_event_seq": "1",
            "content_digest": "",
            "created_mono_ns": str(time.monotonic_ns()),
        }
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for key, value in defaults.items():
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (key, value)
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    @contextlib.contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        with self.lock:
            yield self._conn

    def close(self) -> None:
        with self.lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self._conn.close()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.read() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def next_event_seq(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM meta WHERE key='next_event_seq'").fetchone()
        seq = int(row["value"])
        self.set_meta(conn, "next_event_seq", str(seq + 1))
        return seq

    def append_event(
        self,
        conn: sqlite3.Connection,
        *,
        txn_id: str,
        version: int,
        state: str,
        body_digest: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        created_mono_ns: int | None = None,
    ) -> int:
        seq = self.next_event_seq(conn)
        payload = {
            "txn_id": txn_id,
            "version": version,
            "state": state,
            "event_seq": seq,
            "body_digest": body_digest,
            "type": event_type,
            "data": data or {},
        }
        record = encode_record(payload)
        now = created_mono_ns or time.monotonic_ns()
        conn.execute(
            """INSERT INTO events(event_seq,txn_id,version,state,body_digest,event_type,record,created_mono_ns)
               VALUES(?,?,?,?,?,?,?,?)""",
            (seq, txn_id, version, state, body_digest, event_type, sqlite3.Binary(record), now),
        )
        return seq

    def ledger_snapshot(self, conn: sqlite3.Connection, *, include_latency: bool = True) -> dict[str, Any]:
        def rows(table: str, order: str = "rowid") -> list[dict[str, Any]]:
            return [dict(r) for r in conn.execute(f"SELECT * FROM {table} ORDER BY {order}")]

        # Event records are bytes in SQLite.  The ledger export converts them to
        # base64 while the digest uses the exact same canonical JSON structure.
        import base64

        events = []
        for row in conn.execute("SELECT * FROM events WHERE valid=1 ORDER BY event_seq"):
            item = dict(row)
            item["record_b64"] = base64.b64encode(item.pop("record")).decode("ascii")
            events.append(item)
        snapshot: dict[str, Any] = {
            "format_version": int(self.get_meta("format_version", str(FORMAT_VERSION)) or FORMAT_VERSION),
            "meta": {
                "next_event_seq": int(self.get_meta("next_event_seq", "1") or "1"),
                "created_mono_ns": int(self.get_meta("created_mono_ns", "0") or "0"),
            },
            "transactions": rows("txns", "txn_id"),
            "events": events,
            "idempotency": rows("idempotency", "idem_key"),
            "idempotency_evictions": rows("idempotency_evictions", "id"),
            "pulls": rows("pulls", "started_mono_ns, request_id"),
            "conflicts": rows("conflicts", "id"),
        }
        if include_latency:
            snapshot["latency"] = rows("latency", "id")
        return snapshot

    def digest_payload(self, snapshot: dict[str, Any]) -> str:
        # Runtime update clocks are observational metadata, not ledger
        # content.  Excluding them makes restart-time replay deterministic while
        # still covering every state, event, idempotency and latency field.
        return ledger_content_digest(snapshot)

    def refresh_digest(self, conn: sqlite3.Connection) -> str:
        snapshot = self.ledger_snapshot(conn)
        digest = self.digest_payload(snapshot)
        self.set_meta(conn, "content_digest", digest)
        return digest

    def current_digest(self) -> str:
        with self.read() as conn:
            return self.refresh_digest(conn)

    def export_ledger(self, out_path: str | os.PathLike[str]) -> dict[str, Any]:
        with self.read() as conn:
            # Take a consistent read snapshot.
            conn.execute("BEGIN")
            try:
                snapshot = self.ledger_snapshot(conn)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        snapshot["content_digest"] = self.digest_payload(snapshot)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        temp = out.with_suffix(out.suffix + ".tmp")
        temp.write_text(canonical_json(snapshot) + "\n", encoding="utf-8")
        os.replace(temp, out)
        return {
            "ok": True,
            "path": str(out),
            "format_version": snapshot["format_version"],
            "content_digest": snapshot["content_digest"],
        }

    def sqlite_backup_readonly(self, target_path: str | os.PathLike[str]) -> str:
        target = Path(target_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        with self.lock:
            backup = sqlite3.connect(str(target))
            try:
                self._conn.backup(backup)
            finally:
                backup.close()
        try:
            os.chmod(target, 0o444)
        except OSError:
            pass
        return str(target)

    def quarantine_row(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row | dict[str, Any],
        reason: str,
        path: str | None,
        *,
        valid: bool = False,
    ) -> None:
        raw = row["record"] if row["record"] is not None else b""
        conn.execute(
            """INSERT INTO quarantine(row_id,event_seq,txn_id,reason,path,raw_hex,quarantined_mono_ns)
               VALUES(?,?,?,?,?,?,?)""",
            (
                row["row_id"],
                row["event_seq"],
                row["txn_id"],
                reason,
                path,
                bytes(raw).hex(),
                time.monotonic_ns(),
            ),
        )
        if not valid:
            conn.execute("UPDATE events SET valid=0 WHERE row_id=?", (row["row_id"],))
