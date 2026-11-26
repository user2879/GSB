from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import FORMAT_VERSION
from .hashing import chain_digest, sha256_digest


GENESIS_DIGEST = sha256_digest("LEASE_LEDGER_GENESIS_V1")


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cursors (
                    name TEXT PRIMARY KEY,
                    value INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS entries (
                    seq INTEGER PRIMARY KEY,
                    event_seq INTEGER,
                    ts_monotonic_ms INTEGER NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    ok INTEGER NOT NULL DEFAULT 1,
                    error_code TEXT,
                    lease_id TEXT,
                    node_id TEXT,
                    state_before TEXT,
                    state_after TEXT,
                    generation_used INTEGER,
                    generation_before INTEGER,
                    generation_after INTEGER,
                    global_used INTEGER,
                    global_remaining INTEGER,
                    node_used INTEGER,
                    node_remaining INTEGER,
                    wait_start_ms INTEGER,
                    wait_end_ms INTEGER,
                    wait_ms INTEGER,
                    progress_cursor INTEGER NOT NULL,
                    format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    prev_digest TEXT NOT NULL,
                    entry_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entries_lease ON entries(lease_id, seq);
                CREATE INDEX IF NOT EXISTS idx_entries_kind ON entries(kind, seq);
                """
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('format_version', ?)",
                (str(FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO cursors(name,value) VALUES('global', 0)"
            )
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def last_digest(self) -> str:
        row = self.conn.execute("SELECT entry_digest FROM entries ORDER BY seq DESC LIMIT 1").fetchone()
        return row["entry_digest"] if row else GENESIS_DIGEST

    def cursor(self, name: str = "global") -> int:
        row = self.conn.execute("SELECT value FROM cursors WHERE name=?", (name,)).fetchone()
        return int(row["value"]) if row else 0

    def append(
        self,
        *,
        kind: str,
        ts_monotonic_ms: int,
        payload: dict[str, Any],
        event_seq: int | None = None,
        ok: bool = True,
        error_code: str | None = None,
        lease_id: str | None = None,
        node_id: str | None = None,
        state_before: str | None = None,
        state_after: str | None = None,
        generation_used: int | None = None,
        generation_before: int | None = None,
        generation_after: int | None = None,
        global_used: int | None = None,
        global_remaining: int | None = None,
        node_used: int | None = None,
        node_remaining: int | None = None,
        wait_start_ms: int | None = None,
        wait_end_ms: int | None = None,
        progress_cursor_name: str = "global",
    ) -> dict[str, Any]:
        with self.lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                cursor_row = self.conn.execute(
                    "SELECT value FROM cursors WHERE name=?",
                    (progress_cursor_name,),
                ).fetchone()
                previous_cursor = int(cursor_row["value"]) if cursor_row else 0
                next_cursor = previous_cursor + 1
                prev_digest = self.last_digest()
                content_digest = sha256_digest(payload)
                body = {
                    "format_version": FORMAT_VERSION,
                    "progress_cursor": next_cursor,
                    "event_seq": event_seq,
                    "kind": kind,
                    "ok": ok,
                    "error_code": error_code,
                    "lease_id": lease_id,
                    "node_id": node_id,
                    "state_before": state_before,
                    "state_after": state_after,
                    "generation_used": generation_used,
                    "generation_before": generation_before,
                    "generation_after": generation_after,
                    "global_used": global_used,
                    "global_remaining": global_remaining,
                    "node_used": node_used,
                    "node_remaining": node_remaining,
                    "wait_start_ms": wait_start_ms,
                    "wait_end_ms": wait_end_ms,
                    "wait_ms": (
                        max(0, wait_end_ms - wait_start_ms)
                        if wait_start_ms is not None and wait_end_ms is not None
                        else None
                    ),
                    "content_digest": content_digest,
                    "payload": payload,
                }
                entry_digest = chain_digest(prev_digest, body)
                now_utc = datetime.now(timezone.utc).isoformat()
                self.conn.execute(
                    """
                    INSERT INTO entries(
                        seq, event_seq, ts_monotonic_ms, recorded_at_utc, kind, ok, error_code,
                        lease_id, node_id, state_before, state_after,
                        generation_used, generation_before, generation_after,
                        global_used, global_remaining, node_used, node_remaining,
                        wait_start_ms, wait_end_ms, wait_ms, progress_cursor,
                        format_version, content_digest, prev_digest, entry_digest, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        next_cursor,
                        event_seq,
                        int(ts_monotonic_ms),
                        now_utc,
                        kind,
                        1 if ok else 0,
                        error_code,
                        lease_id,
                        node_id,
                        state_before,
                        state_after,
                        generation_used,
                        generation_before,
                        generation_after,
                        global_used,
                        global_remaining,
                        node_used,
                        node_remaining,
                        wait_start_ms,
                        wait_end_ms,
                        body["wait_ms"],
                        next_cursor,
                        FORMAT_VERSION,
                        content_digest,
                        prev_digest,
                        entry_digest,
                        __import__("json").dumps(payload, ensure_ascii=False, sort_keys=True),
                    ),
                )
                self.conn.execute(
                    """
                    INSERT INTO cursors(name,value) VALUES(?,?)
                    ON CONFLICT(name) DO UPDATE SET value=excluded.value
                    """,
                    (progress_cursor_name, next_cursor),
                )
                self.conn.commit()
                return {"seq": next_cursor, "entry_digest": entry_digest, **body}
            except Exception:
                self.conn.rollback()
                raise

    def entries(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM entries ORDER BY seq ASC").fetchall())

    def verify_hash_chain(self) -> None:
        prev = GENESIS_DIGEST
        for row in self.entries():
            payload = __import__("json").loads(row["payload_json"])
            body = {
                "format_version": row["format_version"],
                "progress_cursor": row["progress_cursor"],
                "event_seq": row["event_seq"],
                "kind": row["kind"],
                "ok": bool(row["ok"]),
                "error_code": row["error_code"],
                "lease_id": row["lease_id"],
                "node_id": row["node_id"],
                "state_before": row["state_before"],
                "state_after": row["state_after"],
                "generation_used": row["generation_used"],
                "generation_before": row["generation_before"],
                "generation_after": row["generation_after"],
                "global_used": row["global_used"],
                "global_remaining": row["global_remaining"],
                "node_used": row["node_used"],
                "node_remaining": row["node_remaining"],
                "wait_start_ms": row["wait_start_ms"],
                "wait_end_ms": row["wait_end_ms"],
                "wait_ms": row["wait_ms"],
                "content_digest": row["content_digest"],
                "payload": payload,
            }
            expected = chain_digest(prev, body)
            if row["prev_digest"] != prev or row["entry_digest"] != expected:
                raise ValueError(f"ledger hash chain mismatch at seq {row['seq']}")
            prev = row["entry_digest"]
