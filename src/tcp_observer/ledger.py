from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .canonical import canonical_text, sha256_hex

INTEGER_FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_index INTEGER PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    start_seq INTEGER NOT NULL,
                    end_seq INTEGER NOT NULL,
                    integer_format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observation_items (
                    event_seq INTEGER PRIMARY KEY,
                    batch_index INTEGER NOT NULL,
                    item_id TEXT NOT NULL,
                    integer_format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observation_bounds (
                    event_seq INTEGER PRIMARY KEY,
                    batch_index INTEGER NOT NULL,
                    integer_format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS schedule_trajectory (
                    batch_index INTEGER PRIMARY KEY,
                    integer_format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_committed_batch_index INTEGER NOT NULL,
                    last_committed_event_seq INTEGER NOT NULL,
                    integer_format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recovery_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    batch_index INTEGER,
                    integer_format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def record_recovery_event(self, event_type: str, payload: dict[str, Any]) -> None:
        digest = sha256_hex(payload)
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute(
                """
                INSERT INTO recovery_events
                (event_type, batch_index, integer_format_version, content_digest, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    payload.get("batch_index"),
                    INTEGER_FORMAT_VERSION,
                    digest,
                    canonical_text(payload),
                ),
            )
            self.conn.execute("COMMIT")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def set_meta(self, key: str, value: Any) -> None:
        text = value if isinstance(value, str) else canonical_text(value)
        with self.lock:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, text),
            )

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def committed_batch_indices(self) -> list[int]:
        rows = self.conn.execute("SELECT batch_index FROM batches ORDER BY batch_index").fetchall()
        return [int(row["batch_index"]) for row in rows]

    def last_progress(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT payload_json FROM progress WHERE id=1").fetchone()
        return None if row is None else json.loads(row["payload_json"])

    def commit_batch(
        self,
        batch: dict[str, Any],
        batch_items: list[dict[str, Any]],
        schedule_payload: dict[str, Any],
        cursor: dict[str, Any],
        *,
        crash_before_commit: bool = False,
    ) -> str:
        batch_index = int(batch["batch_index"])
        batch_digest = str(batch["content_digest"])
        schedule_digest = sha256_hex(schedule_payload)
        cursor_digest = sha256_hex(cursor)
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO batches
                    (batch_index, batch_id, start_seq, end_seq, integer_format_version, content_digest, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_index,
                        batch["batch_id"],
                        batch["start_seq"],
                        batch["end_seq"],
                        INTEGER_FORMAT_VERSION,
                        batch_digest,
                        canonical_text(batch),
                    ),
                )
                for item in batch_items:
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO observation_items
                        (event_seq, batch_index, item_id, integer_format_version, content_digest, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            int(item["event_seq"]),
                            batch_index,
                            item["item_id"],
                            INTEGER_FORMAT_VERSION,
                            item["content_digest"],
                            canonical_text(item),
                        ),
                    )
                    bounds_payload = {
                        "event_seq": item["event_seq"],
                        "item_id": item["item_id"],
                        "batch_index": batch_index,
                        "error_bounds": item["error_bounds"],
                        "error_bounds_raw": item["error_bounds_raw"],
                        "local_quantization_error": item["local_quantization_error"],
                    }
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO observation_bounds
                        (event_seq, batch_index, integer_format_version, content_digest, payload_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            int(item["event_seq"]),
                            batch_index,
                            INTEGER_FORMAT_VERSION,
                            sha256_hex(bounds_payload),
                            canonical_text(bounds_payload),
                        ),
                    )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO schedule_trajectory
                    (batch_index, integer_format_version, content_digest, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (batch_index, INTEGER_FORMAT_VERSION, schedule_digest, canonical_text(schedule_payload)),
                )
                self.conn.execute(
                    """
                    INSERT INTO progress(id, last_committed_batch_index, last_committed_event_seq,
                                         integer_format_version, content_digest, payload_json)
                    VALUES (1, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        last_committed_batch_index=excluded.last_committed_batch_index,
                        last_committed_event_seq=excluded.last_committed_event_seq,
                        integer_format_version=excluded.integer_format_version,
                        content_digest=excluded.content_digest,
                        payload_json=excluded.payload_json
                    """,
                    (
                        batch_index,
                        int(batch["end_seq"]),
                        INTEGER_FORMAT_VERSION,
                        cursor_digest,
                        canonical_text(cursor),
                    ),
                )
                if crash_before_commit:
                    os.kill(os.getpid(), 9)
                self.conn.execute("COMMIT")
            except BaseException:
                # If os.kill is somehow delivered as an exception, roll back.
                try:
                    self.conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
        return batch_digest

    def core_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        meta_rows = self.conn.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        for row in meta_rows:
            try:
                value: Any = json.loads(row["value"])
            except json.JSONDecodeError:
                value = row["value"]
            records.append({"table": "meta", "key": row["key"], "payload": value})
        for table, order in [
            ("batches", "batch_index"),
            ("observation_items", "event_seq"),
            ("observation_bounds", "event_seq"),
            ("schedule_trajectory", "batch_index"),
            ("progress", "id"),
        ]:
            rows = self.conn.execute(f"SELECT payload_json FROM {table} ORDER BY {order}").fetchall()
            for row in rows:
                records.append({"table": table, "payload": json.loads(row["payload_json"])})
        return records

    def export_core_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        records = self.core_records()
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_text(record) + "\n")
        return path
