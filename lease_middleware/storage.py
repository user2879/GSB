"""SQLite persistence: lease records, append-only ledger, subscription view."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .codec import LeaseRecord
from .errors import E_CURSOR, E_UPGRADE, MiddlewareError

SCHEMA_VERSION = 1

# Allowed state transitions of the lease state machine.
LEGAL_TRANSITIONS = {
    "granted": {"renewed", "expired", "revoked"},
    "renewed": {"renewed", "expired", "revoked"},
    "expired": set(),
    "revoked": set(),
}


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    tmp.replace(path)


class Storage:
    def __init__(self, db_path: str | Path, checkpoint_path: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else \
            self.db_path.with_suffix(".upgrade.json")
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    # ---------------------------------------------------------------- schema
    def _create_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS lease_records(
                    lease_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    config_key TEXT NOT NULL,
                    format TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    mono_deadline REAL,
                    upgraded INTEGER NOT NULL DEFAULT 0,
                    backup_json TEXT
                );
                CREATE TABLE IF NOT EXISTS ledger(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    client_id TEXT,
                    lease_id TEXT,
                    config_key TEXT,
                    config_version INTEGER,
                    lease_version INTEGER,
                    state TEXT,
                    renewal_count INTEGER,
                    digest TEXT,
                    payload_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS subscription_view(
                    client_id TEXT NOT NULL,
                    config_key TEXT NOT NULL,
                    lease_id TEXT,
                    state TEXT NOT NULL,
                    seen_config_version INTEGER NOT NULL,
                    PRIMARY KEY(client_id, config_key)
                );
                CREATE TABLE IF NOT EXISTS client_cursors(
                    client_id TEXT PRIMARY KEY,
                    seen_config_version INTEGER NOT NULL
                );
                """
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # ------------------------------------------------------------------- meta
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, sort_keys=True, ensure_ascii=False)),
            )

    # ------------------------------------------------------------- lease rows
    def upsert_lease(self, record: LeaseRecord, state: str,
                     mono_deadline: float | None, *, upgraded: bool = False,
                     backup_json: str | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO lease_records(lease_id,client_id,config_key,format,record_json,"
                "state,mono_deadline,upgraded,backup_json) VALUES(?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(lease_id) DO UPDATE SET client_id=excluded.client_id,"
                "config_key=excluded.config_key,format=excluded.format,"
                "record_json=excluded.record_json,state=excluded.state,"
                "mono_deadline=excluded.mono_deadline,upgraded=excluded.upgraded,"
                "backup_json=COALESCE(excluded.backup_json,lease_records.backup_json)",
                (record.lease_id, record.client_id, record.config_key, record.format,
                 json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False),
                 state, mono_deadline, 1 if upgraded else 0, backup_json),
            )

    def list_lease_rows(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM lease_records ORDER BY lease_id").fetchall())

    def get_lease_row(self, lease_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM lease_records WHERE lease_id=?", (lease_id,)).fetchone()

    def update_lease_state(self, lease_id: str, state: str,
                           mono_deadline: float | None = None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE lease_records SET state=?, mono_deadline=COALESCE(?,mono_deadline) "
                "WHERE lease_id=?", (state, mono_deadline, lease_id))

    def update_lease_record(self, record: LeaseRecord, state: str,
                            mono_deadline: float | None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE lease_records SET format=?,record_json=?,state=?,"
                "mono_deadline=COALESCE(?,mono_deadline),upgraded=1 WHERE lease_id=?",
                (record.format,
                 json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False),
                 state, mono_deadline, record.lease_id))

    # ----------------------------------------------------------------- ledger
    def append_ledger(self, event_type: str, *, created_at_ms: int, client_id: str | None = None,
                      lease_id: str | None = None, config_key: str | None = None,
                      config_version: int | None = None, lease_version: int | None = None,
                      state: str | None = None, renewal_count: int | None = None,
                      digest: str | None = None, payload: dict[str, Any] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO ledger(event_type,client_id,lease_id,config_key,config_version,"
            "lease_version,state,renewal_count,digest,payload_json,created_at_ms) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (event_type, client_id, lease_id, config_key, config_version, lease_version,
             state, renewal_count, digest,
             json.dumps(payload or {}, sort_keys=True, ensure_ascii=False), created_at_ms),
        )
        return int(cur.lastrowid)

    def list_ledger(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM ledger ORDER BY seq").fetchall())

    # ----------------------------------------------------- subscription view
    def replace_subscription_view(self, rows: list[tuple[str, str, str | None, str, int]]) -> None:
        """Whole-table replacement; never an incremental merge."""
        with self.conn:
            self.conn.execute("DELETE FROM subscription_view")
            self.conn.executemany(
                "INSERT INTO subscription_view(client_id,config_key,lease_id,state,"
                "seen_config_version) VALUES(?,?,?,?,?)", rows,
            )

    def list_subscription_view(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM subscription_view ORDER BY client_id,config_key").fetchall())

    def mark_view_entry(self, client_id: str, config_key: str, state: str,
                        lease_id: str | None) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE subscription_view SET state=?, lease_id=? "
                "WHERE client_id=? AND config_key=?",
                (state, lease_id, client_id, config_key),
            )

    # ---------------------------------------------------------------- cursors
    def get_cursor(self, client_id: str) -> int:
        row = self.conn.execute(
            "SELECT seen_config_version FROM client_cursors WHERE client_id=?",
            (client_id,)).fetchone()
        return int(row["seen_config_version"]) if row else 0

    def get_all_cursors(self) -> dict[str, int]:
        return {r["client_id"]: int(r["seen_config_version"])
                for r in self.conn.execute(
                    "SELECT client_id,seen_config_version FROM client_cursors").fetchall()}

    def set_cursor(self, client_id: str, version: int) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO client_cursors(client_id,seen_config_version) VALUES(?,?) "
                "ON CONFLICT(client_id) DO UPDATE SET seen_config_version=excluded."
                "seen_config_version", (client_id, version))

    def replace_all_cursors(self, cursors: dict[str, int]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM client_cursors")
            self.conn.executemany(
                "INSERT INTO client_cursors(client_id,seen_config_version) VALUES(?,?)",
                sorted(cursors.items()),
            )

    def assert_cursor_within_ledger(self, client_id: str, new_version: int) -> None:
        """A cursor can never advance beyond a ledger-recorded pushed version."""
        row = self.conn.execute(
            "SELECT MAX(config_version) AS m FROM ledger WHERE event_type='config_push' "
            "AND client_id=?", (client_id,)).fetchone()
        max_pushed = int(row["m"] or 0)
        if new_version > max_pushed:
            raise MiddlewareError(
                E_CURSOR, "cursor would advance beyond ledger-recorded pushes",
                "client_cursors.seen_config_version", client_id=client_id,
                context={"attempted": new_version, "max_pushed": max_pushed})

    # ------------------------------------------------------------- checkpoint
    def load_checkpoint(self) -> dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {"upgraded": [], "hashes": {}}
        try:
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MiddlewareError(E_UPGRADE, f"checkpoint unreadable: {exc}", "checkpoint")

    def save_checkpoint(self, payload: dict[str, Any]) -> None:
        atomic_write_json(self.checkpoint_path, payload)

    def record_hash(self, record_json: str) -> str:
        return hashlib.sha256(record_json.encode("utf-8")).hexdigest()
