"""SQLite-backed session store, state machine and idempotent commits."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import (
    CURRENT_FORMAT_VERSION,
    STATE_COMMITTED,
    STATE_DISTRIBUTING,
    STATE_FAILED,
    STATE_MACHINE_STATES,
    STATE_PREPARED,
)
from .errors import BizError, ErrorCode
from .logging_setup import utc_now_iso
from .manifest import canonical_json

ALLOWED_TRANSITIONS = {
    STATE_PREPARED: {STATE_DISTRIBUTING, STATE_FAILED},
    STATE_DISTRIBUTING: {STATE_COMMITTED, STATE_FAILED},
    STATE_COMMITTED: set(),
    STATE_FAILED: set(),
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    config_version INTEGER NOT NULL,
    manifest_version INTEGER NOT NULL,
    state TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS state_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    from_state TEXT,
    to_state TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    confirmed_nodes_json TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    node TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    receipt_id TEXT,
    ok INTEGER NOT NULL,
    retry INTEGER NOT NULL,
    duplicate INTEGER NOT NULL,
    digest TEXT,
    raw_json TEXT NOT NULL,
    ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS committed_versions (
    config_version INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    committed_at TEXT NOT NULL
);
"""


class SessionStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    # ----- state machine ------------------------------------------------
    def create_session(
        self,
        session_id: str,
        config_version: int,
        plan: dict,
        manifest_version: int = CURRENT_FORMAT_VERSION,
    ) -> dict:
        now = utc_now_iso()
        self.conn.execute(
            """INSERT INTO sessions
               (session_id, config_version, manifest_version, state, plan_json,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                config_version,
                manifest_version,
                STATE_PREPARED,
                canonical_json(plan),
                now,
                now,
            ),
        )
        self._record_transition(session_id, None, STATE_PREPARED, config_version, [])
        self.conn.commit()
        return self.get_session(session_id)

    def transition(
        self,
        session_id: str,
        to_state: str,
        confirmed_nodes: list[str],
    ) -> dict:
        if to_state not in STATE_MACHINE_STATES:
            raise BizError(
                ErrorCode.E_STATE,
                f"unknown target state {to_state}",
                {"session_id": session_id},
            )
        session = self.get_session(session_id)
        from_state = session["state"]
        if to_state not in ALLOWED_TRANSITIONS[from_state]:
            raise BizError(
                ErrorCode.E_STATE,
                f"illegal transition {from_state} -> {to_state}",
                {"session_id": session_id, "from": from_state, "to": to_state},
            )
        now = utc_now_iso()
        self.conn.execute(
            "UPDATE sessions SET state = ?, updated_at = ? WHERE session_id = ?",
            (to_state, now, session_id),
        )
        self._record_transition(
            session_id,
            from_state,
            to_state,
            session["config_version"],
            confirmed_nodes,
        )
        self.conn.commit()
        return self.get_session(session_id)

    def _record_transition(
        self,
        session_id: str,
        from_state: str | None,
        to_state: str,
        config_version: int,
        confirmed_nodes: list[str],
    ) -> None:
        self.conn.execute(
            """INSERT INTO state_transitions
               (session_id, from_state, to_state, config_version,
                confirmed_nodes_json, ts)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                from_state,
                to_state,
                config_version,
                canonical_json(sorted(confirmed_nodes)),
                utc_now_iso(),
            ),
        )

    def get_session(self, session_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise BizError(
                ErrorCode.E_STATE,
                f"session {session_id} not found",
                {"session_id": session_id},
            )
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json"))
        return result

    def list_transitions(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM state_transitions WHERE session_id = ?
               ORDER BY id ASC""",
            (session_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["confirmed_nodes"] = json.loads(item.pop("confirmed_nodes_json"))
            result.append(item)
        return result

    # ----- receipts -----------------------------------------------------
    def record_receipt(self, session_id: str, node: str, attempt: int, receipt: dict) -> None:
        session = self.get_session(session_id)
        self.conn.execute(
            """INSERT INTO receipts
               (session_id, config_version, node, attempt, receipt_id, ok,
                retry, duplicate, digest, raw_json, ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                session["config_version"],
                node,
                attempt,
                receipt.get("receipt_id"),
                1 if receipt.get("ok") else 0,
                1 if receipt.get("retry") else 0,
                1 if receipt.get("duplicate") else 0,
                receipt.get("digest"),
                canonical_json(receipt),
                utc_now_iso(),
            ),
        )
        self.conn.commit()

    def list_receipts(self, session_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM receipts WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["raw"] = json.loads(item.pop("raw_json"))
            result.append(item)
        return result

    def latest_successful_receipts(self, session_id: str) -> dict[str, dict]:
        rows = self.conn.execute(
            """SELECT * FROM receipts WHERE session_id = ? AND ok = 1
               ORDER BY id ASC""",
            (session_id,),
        ).fetchall()
        latest: dict[str, dict] = {}
        for row in rows:
            latest[row["node"]] = json.loads(row["raw_json"])
        return latest

    # ----- idempotent commits ------------------------------------------
    def is_committed(self, config_version: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM committed_versions WHERE config_version = ?",
            (config_version,),
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json"))
        return result

    def commit(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        config_version = session["config_version"]
        existing = self.is_committed(config_version)
        if existing is not None:
            return {"idempotent": True, "committed": existing}
        now = utc_now_iso()
        self.conn.execute(
            """INSERT INTO committed_versions
               (config_version, session_id, plan_json, committed_at)
               VALUES (?, ?, ?, ?)""",
            (
                config_version,
                session_id,
                canonical_json(session["plan"]),
                now,
            ),
        )
        self.transition(session_id, STATE_COMMITTED, [])
        return {"idempotent": False, "committed": self.is_committed(config_version)}

    def latest_committed_version(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM committed_versions ORDER BY config_version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["plan"] = json.loads(result.pop("plan_json"))
        return result

    def all_committed_versions(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT config_version FROM committed_versions ORDER BY config_version"
        ).fetchall()
        return [row["config_version"] for row in rows]
