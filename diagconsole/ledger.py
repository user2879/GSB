"""SQLite result-node ledger with atomic batches and crash recovery support."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .errors import DiagError, canonical_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    task_id       TEXT NOT NULL,
    cluster_id    TEXT NOT NULL,
    version       INTEGER NOT NULL,
    parent        INTEGER,
    aggregate_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    monotonic_ms  INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'committed',
    PRIMARY KEY (task_id, cluster_id, version)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS roots (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    root_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id      TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    created_ms    INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def begin_batch(self, batch_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO batches(batch_id, status, created_ms) "
                "VALUES (?, 'staging', ?)", (batch_id, self.now_ms()))

    def finish_batch(self, batch_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE batches SET status='committed' WHERE batch_id=?",
                         (batch_id,))

    def upsert_node(self, node: dict, status: str = "committed",
                    allow_replace: bool = False) -> str:
        """Idempotent node write inside the caller's transaction.

        Returns ``inserted``, ``identical`` or ``replaced``. Nothing is visible
        to other readers until the surrounding transaction commits. A content
        change is accepted only with ``allow_replace`` (invalidation/rollback).
        """
        aggregate_json = canonical_json(node["aggregate"])
        existing = self.conn.execute(
            "SELECT aggregate_json, content_digest, parent FROM nodes "
            "WHERE task_id=? AND cluster_id=? AND version=?",
            (node["task_id"], node["cluster_id"], node["version"])).fetchone()
        if existing is not None:
            if existing["aggregate_json"] == aggregate_json and \
                    existing["content_digest"] == node["content_digest"] and \
                    existing["parent"] == node["parent"]:
                return "identical"
            if not allow_replace:
                raise DiagError(
                    "E_LEDGER",
                    "non-idempotent rewrite for the same node key with different content",
                    {"task": node["task_id"], "cluster": node["cluster_id"],
                     "revision": node["version"]})
            self.conn.execute(
                "UPDATE nodes SET parent=?, aggregate_json=?, content_digest=?, "
                "monotonic_ms=?, status=? WHERE task_id=? AND cluster_id=? "
                "AND version=?",
                (node["parent"], aggregate_json, node["content_digest"],
                 node["monotonic_ms"], status, node["task_id"],
                 node["cluster_id"], node["version"]))
            return "replaced"
        self.conn.execute(
            "INSERT INTO nodes(task_id, cluster_id, version, parent, "
            "aggregate_json, content_digest, monotonic_ms, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (node["task_id"], node["cluster_id"], node["version"],
             node["parent"], aggregate_json, node["content_digest"],
             node["monotonic_ms"], status))
        return "inserted"

    def assert_parent_committed(self, node: dict) -> None:
        """Parent reference must already be committed (same transaction allowed)."""
        if node["parent"] is None:
            return
        row = self.conn.execute(
            "SELECT 1 FROM nodes WHERE task_id=? AND cluster_id=? AND version=?",
            (node["task_id"], node["cluster_id"], node["parent"])).fetchone()
        if row is None:
            raise DiagError(
                "E_REF",
                "node parent reference is missing from the ledger",
                {"task": node["task_id"], "cluster": node["cluster_id"],
                 "revision": node["version"], "parent": node["parent"]})

    def get_node(self, task_id: str, cluster_id: str, version: int):
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE task_id=? AND cluster_id=? AND version=?",
            (task_id, cluster_id, version)).fetchone()
        return dict(row) if row else None

    def list_nodes(self, status: str | None = None) -> list[dict]:
        sql = "SELECT * FROM nodes"
        params: tuple = ()
        if status is not None:
            sql += " WHERE status=?"
            params = (status,)
        sql += " ORDER BY version, task_id, cluster_id"
        return [dict(row) for row in self.conn.execute(sql, params)]

    def list_unknown_nodes(self) -> list[dict]:
        return self.list_nodes(status="unknown")

    def mark_unknown(self, task_id: str, cluster_id: str, version: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE nodes SET status='unknown' WHERE task_id=? AND "
                "cluster_id=? AND version=?", (task_id, cluster_id, version))

    def insert_unknown(self, node: dict) -> None:
        """Durable crash-journal row for a write interrupted before commit."""
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO nodes(task_id, cluster_id, version, "
                "parent, aggregate_json, content_digest, monotonic_ms, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'unknown')",
                (node["task_id"], node["cluster_id"], node["version"],
                 node["parent"], canonical_json(node["aggregate"]),
                 node["content_digest"], self.now_ms()))

    def delete_node(self, task_id: str, cluster_id: str, version: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM nodes WHERE task_id=? AND cluster_id=? AND version=?",
                (task_id, cluster_id, version))

    def set_root(self, version: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO roots(id, root_version) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET root_version=excluded.root_version",
                (version,))

    def get_root(self) -> int | None:
        row = self.conn.execute("SELECT root_version FROM roots WHERE id=1").fetchone()
        return None if row is None else int(row["root_version"])

    def staging_batches(self) -> list[str]:
        return [row["batch_id"] for row in self.conn.execute(
            "SELECT batch_id FROM batches WHERE status='staging' ORDER BY batch_id")]

    def reset(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                "DELETE FROM nodes; DELETE FROM roots; DELETE FROM batches;")
