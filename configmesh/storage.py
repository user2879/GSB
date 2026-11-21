"""SQLite-backed session snapshots with content digests and full-replace resume.

A snapshot stores manifest version identity, precision caliber, dispatch plan,
receipts, fixed-point error bounds and the state-machine transition.  Reopening
a snapshot compares version + content digest first; any mismatch triggers a
full replacement of plan/receipt state rather than incremental merging.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from .util import canonical_json, content_digest, wall_ms

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    session_id TEXT NOT NULL,
    manifest_version INTEGER NOT NULL,
    grid_exp INTEGER NOT NULL,
    state TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL,
    PRIMARY KEY (session_id, manifest_version)
);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_json TEXT NOT NULL,
    recorded_ms INTEGER NOT NULL
);
"""


@dataclass
class Snapshot:
    session_id: str
    manifest_version: int
    grid_exp: int
    state: str
    payload: dict[str, Any]
    content_digest: str
    updated_ms: int


class SnapshotStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        parent = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(parent, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def save_snapshot(self, session_id: str, manifest_version: int,
                      grid_exp: int, state: str, payload: dict[str, Any],
                      replace: bool = False) -> Snapshot:
        digest = content_digest(payload)
        snapshot = Snapshot(
            session_id=session_id,
            manifest_version=manifest_version,
            grid_exp=grid_exp,
            state=state,
            payload=payload,
            content_digest=digest,
            updated_ms=wall_ms(),
        )
        self.conn.execute(
            """INSERT INTO snapshots(session_id, manifest_version, grid_exp, state,
                                     payload_json, content_digest, updated_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, manifest_version) DO UPDATE SET
                 grid_exp=excluded.grid_exp,
                 state=excluded.state,
                 payload_json=excluded.payload_json,
                 content_digest=excluded.content_digest,
                 updated_ms=excluded.updated_ms""",
            (session_id, manifest_version, grid_exp, state,
             canonical_json(payload).decode("utf-8"), digest, snapshot.updated_ms),
        )
        self.conn.commit()
        return snapshot

    def load_latest(self, session_id: str) -> Snapshot | None:
        row = self.conn.execute(
            """SELECT * FROM snapshots WHERE session_id = ?
               ORDER BY manifest_version DESC LIMIT 1""",
            (session_id,),
        ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def load_version(self, session_id: str, manifest_version: int) -> Snapshot | None:
        row = self.conn.execute(
            "SELECT * FROM snapshots WHERE session_id = ? AND manifest_version = ?",
            (session_id, manifest_version),
        ).fetchone()
        return self._row_to_snapshot(row) if row else None

    def list_versions(self, session_id: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT manifest_version FROM snapshots WHERE session_id = ? ORDER BY manifest_version",
            (session_id,),
        ).fetchall()
        return [int(row["manifest_version"]) for row in rows]

    def replace_all_session_state(self, session_id: str) -> int:
        """Wipe plan/receipt rows for a session before whole-state replacement."""
        cur = self.conn.execute("DELETE FROM snapshots WHERE session_id = ?", (session_id,))
        self.conn.commit()
        return cur.rowcount

    def append_ledger(self, session_id: str, event: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO ledger(session_id, event_json, recorded_ms) VALUES (?, ?, ?)",
            (session_id, canonical_json(event).decode("utf-8"), wall_ms()),
        )
        self.conn.commit()

    def ledger(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT event_json FROM ledger WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def identity_matches(self, snapshot: Snapshot, manifest_version: int,
                         grid_exp: int, current_payload: dict[str, Any]) -> bool:
        return (
            snapshot.manifest_version == manifest_version
            and snapshot.grid_exp == grid_exp
            and snapshot.content_digest == content_digest(current_payload)
        )

    @staticmethod
    def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
        return Snapshot(
            session_id=row["session_id"],
            manifest_version=row["manifest_version"],
            grid_exp=row["grid_exp"],
            state=row["state"],
            payload=json.loads(row["payload_json"]),
            content_digest=row["content_digest"],
            updated_ms=row["updated_ms"],
        )
