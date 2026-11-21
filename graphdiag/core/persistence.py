"""SQLite persistence for the traversal ledger and session snapshots.

The database holds:

* ``ledger``  - append-only per-query traversal records (one row per run)
* ``snapshots`` - full session snapshots keyed by snapshot id
* ``snapshot_events`` - restore / replay audit events
"""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from .models import INTEGER_FORMAT_VERSION, canonical_json, content_digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL,
    graph_version INTEGER NOT NULL,
    query_version INTEGER NOT NULL,
    scale_exponent INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    created_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    integer_format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    event TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    created_wall_ms INTEGER NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- ledger -------------------------------------------------------------
    def append_ledger(
        self,
        query_id: str,
        graph_version: int,
        query_version: int,
        scale_exponent: int,
        record: dict[str, Any],
        monotonic_ms: int,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO ledger(query_id, graph_version, query_version, "
            "scale_exponent, record_json, created_monotonic_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                query_id,
                graph_version,
                query_version,
                scale_exponent,
                canonical_json(record),
                monotonic_ms,
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def read_ledger(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM ledger ORDER BY id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    # -- snapshots ----------------------------------------------------------
    def save_snapshot(self, snapshot_id: str, payload: dict[str, Any]) -> str:
        payload = dict(payload)
        payload["integer_format_version"] = INTEGER_FORMAT_VERSION
        body = {k: v for k, v in payload.items() if k != "integer_format_version"}
        digest = content_digest(body)
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots(snapshot_id, integer_format_version, "
            "content_digest, payload_json, created_wall_ms) VALUES (?, ?, ?, ?, ?)",
            (
                snapshot_id,
                INTEGER_FORMAT_VERSION,
                digest,
                canonical_json(payload),
                int(time.time() * 1000),
            ),
        )
        self.conn.commit()
        return digest

    def load_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"snapshot not found: {snapshot_id}")
        payload = json.loads(row["payload_json"])
        body = {k: v for k, v in payload.items() if k != "integer_format_version"}
        recomputed = content_digest(body)
        if recomputed != row["content_digest"]:
            raise ValueError(
                f"snapshot {snapshot_id} digest mismatch: "
                f"{row['content_digest']} != {recomputed}"
            )
        return payload

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT snapshot_id, integer_format_version, content_digest, created_wall_ms "
            "FROM snapshots ORDER BY created_wall_ms ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def record_event(self, snapshot_id: str, event: str, detail: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO snapshot_events(snapshot_id, event, detail_json, created_wall_ms) "
            "VALUES (?, ?, ?, ?)",
            (snapshot_id, event, canonical_json(detail), int(time.time() * 1000)),
        )
        self.conn.commit()

    def read_events(self, snapshot_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM snapshot_events WHERE snapshot_id = ? ORDER BY id ASC",
            (snapshot_id,),
        ).fetchall()
        return [dict(row) for row in rows]
