"""SQLite-backed routing ledger with batch commits and wholesale staleness.

Probe outcomes are buffered into a batch and committed atomically; after a
forced kill only fully committed batches are recoverable. When rule or ring
versions advance, all rows of older versions are marked stale in one
statement (wholesale replacement, never row-by-row patching) and excluded
from every current view and aggregate.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading

from .canon import canonical_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    committed_ms INTEGER NOT NULL,
    rule_version INTEGER NOT NULL,
    ring_version INTEGER NOT NULL,
    dep_graph_version INTEGER NOT NULL,
    budget_version INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tasks (
    batch_id INTEGER NOT NULL REFERENCES batches(batch_id),
    seq INTEGER NOT NULL,
    unit_id TEXT NOT NULL,
    predicate_digest TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    targets TEXT NOT NULL,
    reasons TEXT NOT NULL,
    lease TEXT NOT NULL,
    wait_ms INTEGER NOT NULL,
    exec_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    ring_version INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (batch_id, seq)
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    db INTEGER NOT NULL,
    granted_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL,
    reclaim_reason TEXT NOT NULL DEFAULT '',
    rule_version INTEGER NOT NULL,
    ring_version INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0
);
"""


class Ledger:
    def __init__(self, path: str, clock):
        self.path = path
        self.clock = clock
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self._pending = None  # buffered batch rows, not yet committed

    # ---------- batch lifecycle ----------
    def begin_batch(self):
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("a batch is already open")
            self._pending = []

    def buffer_task(self, record: dict, versions):
        """Buffer one probe record into the open batch (nothing persisted yet)."""
        with self._lock:
            if self._pending is None:
                raise RuntimeError("no open batch")
            self._pending.append((record, versions))

    def commit_batch(self, versions) -> int:
        with self._lock:
            rows, self._pending = self._pending, None
        if rows is None:
            raise RuntimeError("no open batch")
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO batches(committed_ms, rule_version, ring_version,"
                " dep_graph_version, budget_version) VALUES (?,?,?,?,?)",
                (
                    self.clock(),
                    versions.rule_version,
                    versions.ring_version,
                    versions.dep_graph_version,
                    versions.budget_version,
                ),
            )
            batch_id = cur.lastrowid
            for seq, (rec, v) in enumerate(rows):
                self._conn.execute(
                    "INSERT INTO tasks(batch_id, seq, unit_id, predicate_digest,"
                    " plan_digest, targets, reasons, lease, wait_ms, exec_ms,"
                    " status, rule_version, ring_version)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        batch_id,
                        seq,
                        rec["unit_id"],
                        rec["predicate_digest"],
                        rec["plan_digest"],
                        canonical_text(rec["targets"]),
                        canonical_text(rec["reasons"]),
                        canonical_text(rec.get("lease", {})),
                        rec["wait_ms"],
                        rec["exec_ms"],
                        rec["status"],
                        v.rule_version,
                        v.ring_version,
                    ),
                )
        return batch_id

    def abort_batch(self):
        with self._lock:
            self._pending = None

    # ---------- leases ----------
    def record_lease(self, lease: dict, rule_version: int, ring_version: int = 0):
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO leases(lease_id, db, granted_ms, expires_ms,"
                " version, state, reclaim_reason, rule_version, ring_version)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    lease["lease_id"],
                    lease["db"],
                    lease["granted_ms"],
                    lease["expires_ms"],
                    lease["version"],
                    lease["state"],
                    lease.get("reclaim_reason", ""),
                    rule_version,
                    ring_version,
                ),
            )

    def mark_uncommitted_leases_unknown(self) -> list:
        """Crash recovery: leases still 'held' were never released -> unknown."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE leases SET state='unknown',"
                " reclaim_reason='process_killed: lease never released; rerun required'"
                " WHERE state='held'"
            )
            return [r[0] for r in self._conn.execute(
                "SELECT lease_id FROM leases WHERE state='unknown'"
            ).fetchall()] if cur.rowcount else []

    # ---------- staleness / wholesale replacement ----------
    def mark_stale(self, rule_version: int, ring_version: int) -> dict:
        """Mark every row not matching the current versions as stale."""
        with self._lock, self._conn:
            counts = {}
            for table in ("batches", "tasks", "leases"):
                cur = self._conn.execute(
                    f"UPDATE {table} SET stale=1"
                    " WHERE stale=0 AND (rule_version<>? OR ring_version<>?)",
                    (rule_version, ring_version),
                )
                counts[table] = cur.rowcount
        return counts

    # ---------- current views (stale rows never participate) ----------
    def current_tasks(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT batch_id, seq, unit_id, predicate_digest, plan_digest,"
                " targets, reasons, lease, wait_ms, exec_ms, status,"
                " rule_version, ring_version FROM tasks WHERE stale=0"
                " ORDER BY batch_id, seq"
            ).fetchall()
        cols = ["batch_id", "seq", "unit_id", "predicate_digest", "plan_digest",
                "targets", "reasons", "lease", "wait_ms", "exec_ms", "status",
                "rule_version", "ring_version"]
        out = []
        for row in rows:
            rec = dict(zip(cols, row))
            rec["targets"] = json.loads(rec["targets"])
            rec["reasons"] = json.loads(rec["reasons"])
            rec["lease"] = json.loads(rec["lease"])
            out.append(rec)
        return out

    def current_leases(self):
        with self._lock:
            rows = self._conn.execute(
                "SELECT lease_id, db, granted_ms, expires_ms, version, state,"
                " reclaim_reason, rule_version, ring_version"
                " FROM leases WHERE stale=0 ORDER BY lease_id"
            ).fetchall()
        cols = ["lease_id", "db", "granted_ms", "expires_ms", "version",
                "state", "reclaim_reason", "rule_version", "ring_version"]
        return [dict(zip(cols, r)) for r in rows]

    def summary(self):
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(wait_ms),0), COALESCE(SUM(exec_ms),0)"
                " FROM tasks WHERE stale=0"
            ).fetchone()
            per_unit = self._conn.execute(
                "SELECT unit_id, COUNT(*) FROM tasks WHERE stale=0"
                " GROUP BY unit_id ORDER BY unit_id"
            ).fetchall()
        return {
            "tasks": row[0],
            "total_wait_ms": row[1],
            "total_exec_ms": row[2],
            "per_unit": {u: c for u, c in per_unit},
        }

    def last_committed_batch(self):
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(batch_id), 0) FROM batches"
            ).fetchone()
        return row[0]

    def close(self):
        self._conn.close()
