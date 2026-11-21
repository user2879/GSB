"""SQLite durability layer.

All state transitions happen inside an SQLite transaction, so a process killed
at any instant leaves either the previous fully committed record or the next
fully committed record -- never a half written row.  The platform recovers
only from committed records and treats leases stuck in ``active`` as unknown.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS revisions (
    rev_seq INTEGER PRIMARY KEY,
    parent  INTEGER NOT NULL,
    payload TEXT NOT NULL,
    superseded INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    rev_seq INTEGER NOT NULL,
    low_seq INTEGER NOT NULL,
    high_seq INTEGER NOT NULL,
    keys_json TEXT NOT NULL,
    issued_ms INTEGER NOT NULL,
    deadline_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS node_cursors (
    node_id TEXT PRIMARY KEY,
    applied_rev INTEGER NOT NULL,
    view_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS budgets (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    limit_value INTEGER NOT NULL,
    consumed INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    session_id TEXT NOT NULL,
    event TEXT NOT NULL,
    node_id TEXT,
    rev_seq INTEGER,
    code TEXT,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    name TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    state_json TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ---- revisions -------------------------------------------------------
    def upsert_revision(self, rev: dict[str, Any], superseded: bool = False) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO revisions(rev_seq, parent, payload, superseded) "
                "VALUES(?,?,?,?) ON CONFLICT(rev_seq) DO UPDATE SET "
                "payload=excluded.payload, superseded=excluded.superseded",
                (rev["rev_seq"], rev["parent"], json.dumps(rev, sort_keys=True),
                 1 if superseded else 0),
            )

    def mark_revision_superseded(self, rev_seq: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE revisions SET superseded=1 WHERE rev_seq=?", (rev_seq,))

    def list_revisions(self, include_superseded: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT payload, superseded FROM revisions ORDER BY rev_seq"
        rows = self.conn.execute(sql).fetchall()
        result = []
        for row in rows:
            if not include_superseded and row["superseded"]:
                continue
            result.append(json.loads(row["payload"]))
        return result

    def superseded_sequences(self) -> set[int]:
        rows = self.conn.execute(
            "SELECT rev_seq FROM revisions WHERE superseded=1"
        ).fetchall()
        return {row["rev_seq"] for row in rows}

    # ---- leases ----------------------------------------------------------
    def put_lease(self, lease: dict[str, Any], status: str, detail: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO leases(lease_id, request_id, node_id, rev_seq, low_seq, high_seq, "
                "keys_json, issued_ms, deadline_ms, status, detail) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(lease_id) DO UPDATE SET status=excluded.status, detail=excluded.detail",
                (lease["lease_id"], lease["request_id"], lease["node_id"], lease["rev_seq"],
                 lease["low_seq"], lease["high_seq"], json.dumps(sorted(lease["keys"])),
                 lease["issued_ms"], lease["deadline_ms"], status, detail),
            )

    def update_lease_status(self, lease_id: str, status: str, detail: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE leases SET status=?, detail=? WHERE lease_id=?",
                (status, detail, lease_id),
            )

    def list_leases(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM leases ORDER BY rowid").fetchall()
        return [_lease_row_to_dict(r) for r in rows]

    def active_leases(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM leases WHERE status='active' ORDER BY rowid"
        ).fetchall()
        return [_lease_row_to_dict(r) for r in rows]

    # ---- cursors ---------------------------------------------------------
    def put_cursor(self, node_id: str, applied_rev: int, view: dict[str, int]) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO node_cursors(node_id, applied_rev, view_json) VALUES(?,?,?) "
                "ON CONFLICT(node_id) DO UPDATE SET applied_rev=excluded.applied_rev, "
                "view_json=excluded.view_json",
                (node_id, applied_rev, json.dumps(view, sort_keys=True)),
            )

    def list_cursors(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM node_cursors").fetchall()
        return {
            row["node_id"]: {
                "node_id": row["node_id"],
                "applied_rev": row["applied_rev"],
                "view": json.loads(row["view_json"]),
            }
            for row in rows
        }

    # ---- budgets ---------------------------------------------------------
    def put_budget(self, name: str, kind: str, limit_value: int, consumed: int) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO budgets(name, kind, limit_value, consumed) VALUES(?,?,?,?) "
                "ON CONFLICT(name) DO UPDATE SET limit_value=excluded.limit_value, "
                "consumed=excluded.consumed",
                (name, kind, limit_value, consumed),
            )

    def list_budgets(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM budgets ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    # ---- events ----------------------------------------------------------
    def insert_event(self, record: dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO events(ts, session_id, event, node_id, rev_seq, code, payload) "
                "VALUES(?,?,?,?,?,?,?)",
                (record.get("ts", 0.0), record.get("session_id", ""),
                 record.get("event", ""), record.get("node_id"),
                 record.get("rev_seq"), record.get("code"),
                 json.dumps(record, sort_keys=True)),
            )

    def query_events(self, node_id: str | None = None, rev_seq: int | None = None,
                     code: str | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if node_id is not None:
            clauses.append("node_id = ?")
            params.append(node_id)
        if rev_seq is not None:
            clauses.append("rev_seq = ?")
            params.append(rev_seq)
        if code is not None:
            clauses.append("code = ?")
            params.append(code)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"SELECT payload FROM events{where} ORDER BY id", params
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    # ---- snapshots -------------------------------------------------------
    def save_snapshot(self, name: str, version: int, content_hash: str,
                      state: dict[str, Any], updated_ms: int) -> None:
        with self.tx() as c:
            c.execute(
                "INSERT INTO snapshots(name, version, content_hash, state_json, updated_ms) "
                "VALUES(?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET version=excluded.version, "
                "content_hash=excluded.content_hash, state_json=excluded.state_json, "
                "updated_ms=excluded.updated_ms",
                (name, version, content_hash, json.dumps(state, sort_keys=True), updated_ms),
            )

    def load_snapshot(self, name: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        return {
            "name": row["name"],
            "version": row["version"],
            "content_hash": row["content_hash"],
            "state": json.loads(row["state_json"]),
            "updated_ms": row["updated_ms"],
        }


def _lease_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "lease_id": row["lease_id"],
        "request_id": row["request_id"],
        "node_id": row["node_id"],
        "rev_seq": row["rev_seq"],
        "low_seq": row["low_seq"],
        "high_seq": row["high_seq"],
        "keys": json.loads(row["keys_json"]),
        "issued_ms": row["issued_ms"],
        "deadline_ms": row["deadline_ms"],
        "status": row["status"],
        "detail": row["detail"],
    }
