"""SQLite ledger: receipts, lease spans, watermarks and progress cursor.

One *batch* (a group of processed events) is written inside a single SQLite
transaction. Only fully committed batches are visible to recovery; rows inside
an uncommitted transaction are rolled back by SQLite when the process dies.

Crash-window semantics
----------------------
Before a batch transaction opens, each event whose outcome is about to become
durable is registered in ``lease_inflight`` (status ``IN_FLIGHT``). If the
process is forcibly killed before the batch commits, recovery finds those rows
without matching committed receipts, marks them ``UNKNOWN`` and requires event
replay from the returned cursor. The middleware never fabricates an exit report
for the killed process.

The ledger stores an integer ``format_version`` and a content digest
(SHA-256 over the canonical concatenation of all durable rows).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .models import LEDGER_FORMAT_VERSION

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    committed_at_ms INTEGER NOT NULL,
    seq_from INTEGER NOT NULL,
    seq_to INTEGER NOT NULL,
    digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    seq INTEGER PRIMARY KEY,
    domain_id TEXT NOT NULL,
    holder_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    remaining_ms INTEGER NOT NULL,
    clock_ms INTEGER NOT NULL,
    state TEXT NOT NULL,
    queue_length INTEGER NOT NULL,
    active_leases INTEGER NOT NULL,
    free_capacity INTEGER NOT NULL,
    ratio_fixed INTEGER NOT NULL,
    scale INTEGER NOT NULL,
    error_bound_fixed INTEGER NOT NULL,
    reason TEXT NOT NULL,
    unknown INTEGER NOT NULL DEFAULT 0,
    batch_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    domain_id TEXT NOT NULL,
    holder_id TEXT NOT NULL,
    state_from TEXT,
    state_to TEXT NOT NULL,
    reason TEXT NOT NULL,
    started_ms INTEGER,
    expires_ms INTEGER,
    domain_version INTEGER NOT NULL,
    clock_ms INTEGER NOT NULL,
    batch_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS watermarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    domain_id TEXT NOT NULL,
    active_leases INTEGER NOT NULL,
    capacity INTEGER NOT NULL,
    free_capacity INTEGER NOT NULL,
    queue_length INTEGER NOT NULL,
    ratio_fixed INTEGER NOT NULL,
    scale INTEGER NOT NULL,
    truncation TEXT NOT NULL,
    error_bound_fixed INTEGER NOT NULL,
    batch_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_inflight (
    seq INTEGER PRIMARY KEY,
    domain_id TEXT NOT NULL,
    holder_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    opened_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS unknown_events (
    seq INTEGER PRIMARY KEY,
    domain_id TEXT NOT NULL,
    holder_id TEXT NOT NULL,
    action TEXT NOT NULL,
    detected_at_ms INTEGER NOT NULL,
    note TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=FULL;")
        self.conn.executescript(SCHEMA)
        self._ensure_meta()

    def _ensure_meta(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),))
        elif int(row["value"]) != LEDGER_FORMAT_VERSION:
            raise sqlite3.DatabaseError(
                f"unsupported ledger format_version {row['value']} "
                f"(want {LEDGER_FORMAT_VERSION})")

    # ------------------------------------------------------------- crash window
    def mark_inflight(self, events: Iterable[Any], now_ms: int) -> None:
        rows = [(e.seq, e.domain_id, e.holder_id, e.action, "IN_FLIGHT", now_ms)
                for e in events]
        self.conn.executemany(
            "INSERT OR REPLACE INTO lease_inflight"
            "(seq, domain_id, holder_id, action, status, opened_at_ms) "
            "VALUES (?,?,?,?,?,?)", rows)

    def resolve_inflight(self, seqs: Iterable[int]) -> None:
        seqs = list(seqs)
        if not seqs:
            return
        self.conn.executemany("DELETE FROM lease_inflight WHERE seq=?",
                              [(s,) for s in seqs])

    def recover_unknown(self, now_ms: int) -> list[dict[str, Any]]:
        """Mark inflight rows lacking a committed receipt as UNKNOWN."""
        rows = self.conn.execute(
            "SELECT i.* FROM lease_inflight i "
            "LEFT JOIN receipts r ON r.seq = i.seq "
            "WHERE i.status='IN_FLIGHT' AND r.seq IS NULL "
            "ORDER BY i.seq").fetchall()
        unknown = [dict(r) for r in rows]
        for r in unknown:
            self.conn.execute(
                "INSERT OR REPLACE INTO unknown_events"
                "(seq, domain_id, holder_id, action, detected_at_ms, note) "
                "VALUES (?,?,?,?,?,?)",
                (r["seq"], r["domain_id"], r["holder_id"], r["action"],
                 now_ms, "uncommitted batch at process restart; replay required"))
            self.conn.execute(
                "UPDATE lease_inflight SET status='UNKNOWN' WHERE seq=?",
                (r["seq"],))
        return unknown

    # ------------------------------------------------------------------ batch
    @contextmanager
    def batch(self, seq_from: int, seq_to: int, now_ms: int):
        """Open the single transaction covering one processed event batch."""
        self.conn.execute("BEGIN IMMEDIATE")
        ctx = _BatchContext(self, seq_from, seq_to, now_ms)
        try:
            yield ctx
            digest = self.content_digest()
            cur = self.conn.execute(
                "INSERT INTO batches(committed_at_ms, seq_from, seq_to, digest) "
                "VALUES (?,?,?,?)", (now_ms, seq_from, seq_to, digest))
            batch_id = cur.lastrowid
            ctx.assign_batch_id(batch_id)
            self.conn.execute("COMMIT")
            ctx.committed = True
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def content_digest(self) -> str:
        h = hashlib.sha256()
        for table in ("receipts", "transitions", "watermarks"):
            h.update(table.encode())
            for row in self.conn.execute(
                    f"SELECT * FROM {table} ORDER BY 1").fetchall():
                h.update(json.dumps(dict(row), sort_keys=True,
                                    ensure_ascii=False, separators=(",", ":")).encode())
        return h.hexdigest()

    # --------------------------------------------------------------- readings
    def progress_cursor(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(seq_to), 0) AS c FROM batches").fetchone()
        return int(row["c"])

    def batch_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) AS c FROM batches").fetchone()["c"])

    def latest_digest(self) -> str | None:
        row = self.conn.execute(
            "SELECT digest FROM batches ORDER BY batch_id DESC LIMIT 1").fetchone()
        return row["digest"] if row else None

    def receipts_after(self, seq_cursor: int) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM receipts WHERE seq > ? ORDER BY seq", (seq_cursor,)).fetchall()]

    def all_receipts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM receipts ORDER BY seq").fetchall()]

    def all_watermarks(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM watermarks ORDER BY seq, domain_id").fetchall()]

    def all_transitions(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM transitions ORDER BY id").fetchall()]

    def unknown_events(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM unknown_events ORDER BY seq").fetchall()]

    def close(self) -> None:
        self.conn.close()


class _BatchContext:
    def __init__(self, ledger: Ledger, seq_from: int, seq_to: int, now_ms: int):
        self.ledger = ledger
        self.seq_from = seq_from
        self.seq_to = seq_to
        self.now_ms = now_ms
        self._batch_id: int | None = None
        self.committed = False

    def assign_batch_id(self, batch_id: int) -> None:
        self._batch_id = batch_id
        # Move staged rows onto the committed batch id.
        for table in ("receipts", "transitions", "watermarks"):
            self.ledger.conn.execute(
                f"UPDATE {table} SET batch_id=? WHERE batch_id=-1", (batch_id,))

    def add_receipt(self, receipt: dict[str, Any]) -> None:
        self.ledger.conn.execute(
            "INSERT INTO receipts(seq, domain_id, holder_id, verdict, remaining_ms,"
            " clock_ms, state, queue_length, active_leases, free_capacity,"
            " ratio_fixed, scale, error_bound_fixed, reason, unknown, batch_id) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (receipt["seq"], receipt["domain_id"], receipt["holder_id"],
             receipt["verdict"], receipt["remaining_ms"], receipt["clock_ms"],
             receipt["state"], receipt["queue_length"], receipt["active_leases"],
             receipt["free_capacity"], receipt["ratio_fixed"], receipt["scale"],
             receipt["error_bound_fixed"], receipt.get("reason", ""),
             1 if receipt.get("unknown") else 0, -1))

    def add_transitions(self, transitions: list[dict[str, Any]]) -> None:
        for t in transitions:
            self.ledger.conn.execute(
                "INSERT INTO transitions(seq, domain_id, holder_id, state_from,"
                " state_to, reason, started_ms, expires_ms, domain_version,"
                " clock_ms, batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (t["seq"], t["domain_id"], t["holder_id"], t["state_from"],
                 t["state_to"], t["reason"], t["started_ms"], t["expires_ms"],
                 t["domain_version"], t["clock_ms"], -1))

    def add_watermark(self, wm: dict[str, Any]) -> None:
        self.ledger.conn.execute(
            "INSERT INTO watermarks(seq, domain_id, active_leases, capacity,"
            " free_capacity, queue_length, ratio_fixed, scale, truncation,"
            " error_bound_fixed, batch_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (wm["seq"], wm["domain_id"], wm["active_leases"], wm["capacity"],
             wm["free_capacity"], wm["queue_length"], wm["ratio_fixed"],
             wm["scale"], wm["truncation"], wm["error_bound_fixed"], -1))
