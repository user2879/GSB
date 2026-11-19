"""SQLite persistence with per-record and whole-session content digests."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..util.canonical import canonical_bytes, digest
from ..util.logging_util import StructuredLogger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at_ms INTEGER NOT NULL,
    grid_q INTEGER NOT NULL,
    state TEXT NOT NULL,
    state_history_json TEXT NOT NULL,
    session_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    manifest_digest TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    error_digest TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (session_id, version)
);
CREATE TABLE IF NOT EXISTS items (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    key TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    artifact_bytes BLOB NOT NULL,
    error_json TEXT NOT NULL,
    PRIMARY KEY (session_id, version, key)
);
CREATE TABLE IF NOT EXISTS attempts (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    request_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    record_json TEXT NOT NULL,
    PRIMARY KEY (session_id, version, request_id, attempt)
);
CREATE TABLE IF NOT EXISTS receipts (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    receipt_json TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    PRIMARY KEY (session_id, version, node_id)
);
CREATE TABLE IF NOT EXISTS invalidations (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    step_index INTEGER NOT NULL,
    key TEXT NOT NULL,
    trigger TEXT NOT NULL,
    source TEXT NOT NULL,
    PRIMARY KEY (session_id, version, step_index)
);
CREATE TABLE IF NOT EXISTS views (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    view_json TEXT NOT NULL,
    view_digest TEXT NOT NULL,
    PRIMARY KEY (session_id, version)
);
"""


class Database:
    def __init__(self, path: str | Path, logger: StructuredLogger) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Whole-session replacement semantics
    # ------------------------------------------------------------------
    def replace_snapshot(self, session_id: str, version: int, payload: dict[str, Any]) -> None:
        """Replace all rows for (session, version); never merge incrementally."""
        self.conn.execute(
            "DELETE FROM snapshots WHERE session_id=? AND version=?", (session_id, version)
        )
        for table in ("items", "attempts", "receipts", "invalidations", "views"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE session_id=? AND version=?",
                (session_id, version),
            )

        manifest_digest = digest(payload["manifest"])
        plan_digest = digest(payload["plan"])
        error_digest = digest(payload["errors"])
        receipt_digest = digest(payload["receipts"])
        self.conn.execute(
            """INSERT INTO snapshots
               (session_id, version, manifest_digest, plan_digest, error_digest,
                receipt_digest, payload_json)
               VALUES (?,?,?,?,?,?,?)""",
            (
                session_id,
                version,
                manifest_digest,
                plan_digest,
                error_digest,
                receipt_digest,
                json.dumps(payload, sort_keys=True, ensure_ascii=False),
            ),
        )

        for key, artifact in sorted(payload["artifacts"].items()):
            self.conn.execute(
                """INSERT INTO items (session_id, version, key, artifact_id,
                                      artifact_bytes, error_json)
                   VALUES (?,?,?,?,?,?)""",
                (
                    session_id,
                    version,
                    key,
                    artifact["artifact_id"],
                    bytes.fromhex(artifact["artifact_hex"]),
                    json.dumps(artifact["error"], sort_keys=True, ensure_ascii=False),
                ),
            )

        for attempt in payload["attempts"]:
            self.conn.execute(
                """INSERT INTO attempts (session_id, version, request_id, attempt, record_json)
                   VALUES (?,?,?,?,?)""",
                (
                    session_id,
                    version,
                    attempt["request_id"],
                    attempt["attempt"],
                    json.dumps(attempt, sort_keys=True, ensure_ascii=False),
                ),
            )

        for node_id, receipt in sorted(payload["receipts"].items()):
            self.conn.execute(
                """INSERT INTO receipts (session_id, version, node_id, receipt_json, receipt_digest)
                   VALUES (?,?,?,?,?)""",
                (
                    session_id,
                    version,
                    node_id,
                    json.dumps(receipt, sort_keys=True, ensure_ascii=False),
                    digest(receipt),
                ),
            )

        for index, step in enumerate(payload["invalidation_steps"]):
            self.conn.execute(
                """INSERT INTO invalidations
                   (session_id, version, step_index, key, trigger, source)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, version, index, step["key"], step["trigger"], step["source"]),
            )

        if payload.get("view") is not None:
            self.conn.execute(
                """INSERT INTO views (session_id, version, view_json, view_digest)
                   VALUES (?,?,?,?)""",
                (
                    session_id,
                    version,
                    json.dumps(payload["view"], sort_keys=True, ensure_ascii=False),
                    digest(payload["view"]),
                ),
            )
        self.conn.commit()
        self.logger.log(
            "snapshot_replaced",
            source="database",
            key=str(version),
            manifest_digest=manifest_digest,
        )

    def upsert_session(
        self,
        session_id: str,
        created_at_ms: int,
        grid_q: int,
        state: str,
        state_history: list[str],
    ) -> None:
        session_digest = digest(
            {
                "session_id": session_id,
                "grid_q": grid_q,
                "state": state,
                "state_history": state_history,
            }
        )
        self.conn.execute(
            """INSERT INTO sessions (session_id, created_at_ms, grid_q, state,
                                     state_history_json, session_digest)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(session_id) DO UPDATE SET
                 state=excluded.state,
                 state_history_json=excluded.state_history_json,
                 session_digest=excluded.session_digest""",
            (
                session_id,
                created_at_ms,
                grid_q,
                state,
                json.dumps(state_history, sort_keys=True),
                session_digest,
            ),
        )
        self.conn.commit()

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()

    def get_snapshot(self, session_id: str, version: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM snapshots WHERE session_id=? AND version=?",
            (session_id, version),
        ).fetchone()

    def list_versions(self, session_id: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT version FROM snapshots WHERE session_id=? ORDER BY version",
            (session_id,),
        ).fetchall()
        return [row["version"] for row in rows]

    def load_snapshot_payload(self, session_id: str, version: int) -> dict[str, Any]:
        row = self.get_snapshot(session_id, version)
        if row is None:
            raise KeyError(f"no snapshot for session={session_id} version={version}")
        return json.loads(row["payload_json"])

    def load_items(self, session_id: str, version: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM items WHERE session_id=? AND version=? ORDER BY key",
                (session_id, version),
            ).fetchall()
        )

    def load_views(self, session_id: str, version: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM views WHERE session_id=? AND version=?",
            (session_id, version),
        ).fetchone()


def artifact_hex(blob: bytes) -> str:
    return blob.hex()


def verify_digest(value: Any, expected_hex: str) -> bool:
    return digest(value) == expected_hex
