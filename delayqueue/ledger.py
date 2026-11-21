"""SQLite ledger: message state, migrations, leases, dedup and progress.

Every state migration is written inside an explicit SQLite transaction so a
hard process kill can only leave fully committed batches on disk.  The ledger
carries an integer format version and a content digest; reopening verifies
both and supports full replacement (old file kept as a read-only backup).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
from contextlib import contextmanager

from .errors import SnapshotMismatchError
from .models import (
    ACKED,
    DEAD,
    INFLIGHT,
    READY,
    STATES,
    UNKNOWN_MARKER,
    mono_ms,
    payload_digest,
    wall_ms,
)

FORMAT_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS topics (
    topic TEXT PRIMARY KEY,
    visibility_timeout_ms INTEGER NOT NULL,
    max_deliveries INTEGER NOT NULL,
    backoff_ms TEXT NOT NULL,
    concurrency INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    deliver_at_abs INTEGER NOT NULL,
    origin_offset_ms INTEGER NOT NULL,
    payload BLOB NOT NULL,
    payload_summary TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    suppressed INTEGER NOT NULL,
    unknown_count INTEGER NOT NULL,
    ready_monotonic_ms INTEGER NOT NULL,
    first_error TEXT NOT NULL,
    last_error TEXT NOT NULL,
    seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    mono_ms INTEGER NOT NULL,
    wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS visibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    lease_mono_ms INTEGER NOT NULL,
    timeout_ms INTEGER NOT NULL,
    closed_mono_ms INTEGER,
    close_reason TEXT
);
CREATE TABLE IF NOT EXISTS dedup (
    idempotency_key TEXT PRIMARY KEY,
    payload_summary TEXT NOT NULL,
    suppressed_count INTEGER NOT NULL,
    message_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    batch_id INTEGER PRIMARY KEY,
    cursor_seq INTEGER NOT NULL,
    committed_at_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transitions_msg ON transitions(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_state ON messages(state);
CREATE INDEX IF NOT EXISTS idx_visibility_msg ON visibility(message_id);
"""


def _read_only(path: str) -> None:
    """Mark a file read-only (Windows read-only attribute / POSIX mode)."""
    os.chmod(path, 0o444)


class Ledger:
    """Thin, thread-safe SQLite wrapper used by :class:`DelayQueue`."""

    def __init__(self, path: str, conn: sqlite3.Connection):
        self.path = path
        self.conn = conn
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ open
    @classmethod
    def create(cls, path: str, spec, epoch_ms: int) -> "Ledger":
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.executescript(_SCHEMA)
        ledger = cls(path, conn)
        with ledger.batch() as cur:
            ledger._set_meta(cur, "format_version", str(FORMAT_VERSION))
            ledger._set_meta(cur, "spec_version", spec.spec_version)
            ledger._set_meta(cur, "spec_digest", spec.canonical_digest())
            ledger._set_meta(cur, "dedup_version", "0")
            ledger._set_meta(cur, "epoch_ms", str(epoch_ms))
            ledger._set_meta(cur, "global_concurrency",
                             str(spec.global_concurrency))
            ledger._set_meta(cur, "next_batch_id", "1")
            ledger._set_meta(cur, "content_digest", ledger._digest(cur))
            for topic in spec.topics.values():
                cur.execute(
                    "INSERT INTO topics VALUES (?,?,?,?,?)",
                    (topic.topic, topic.visibility_timeout_ms,
                     topic.max_deliveries, json.dumps(list(topic.backoff_ms)),
                     topic.concurrency),
                )
        return ledger

    @classmethod
    def open(cls, path: str) -> "Ledger":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA busy_timeout=5000")
        ledger = cls(path, conn)
        ledger.verify()
        return ledger

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------- transactions
    @contextmanager
    def batch(self):
        """One atomic commit: migrations + leases + progress cursor together."""
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                yield cur
                self._set_meta(cur, "content_digest", self._digest(cur))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                cur.close()

    def next_batch_id(self, cur) -> int:
        row = cur.execute(
            "SELECT value FROM meta WHERE key='next_batch_id'").fetchone()
        batch_id = int(row[0])
        self._set_meta(cur, "next_batch_id", str(batch_id + 1))
        return batch_id

    # ------------------------------------------------------------------- meta
    @staticmethod
    def _set_meta(cur, key: str, value: str) -> None:
        cur.execute(
            "INSERT INTO meta(key, value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row[0] if row else default

    # ----------------------------------------------------------------- digest
    def _digest(self, cur) -> str:
        import hashlib
        lines = []
        for row in cur.execute(
            "SELECT message_id, state, attempts, suppressed, unknown_count, "
            "payload_summary FROM messages ORDER BY message_id"
        ):
            lines.append("|".join(str(part) for part in row))
        for row in cur.execute(
            "SELECT idempotency_key, payload_summary, suppressed_count, "
            "message_id FROM dedup ORDER BY idempotency_key"
        ):
            lines.append("dedup|" + "|".join(str(part) for part in row))
        blob = "\n".join(lines).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def actual_digest(self) -> str:
        with self._lock:
            return self._digest(self.conn.cursor())

    # ----------------------------------------------------------------- verify
    def verify(self) -> None:
        """Compare format version and content digest with disk state."""
        with self._lock:
            cur = self.conn.cursor()
            row = cur.execute(
                "SELECT value FROM meta WHERE key='format_version'"
            ).fetchone()
            stored_version = int(row[0]) if row else -1
            row = cur.execute(
                "SELECT value FROM meta WHERE key='content_digest'"
            ).fetchone()
            stored_digest = row[0] if row else ""
            actual = self._digest(cur)
            message_count = cur.execute(
                "SELECT COUNT(*) FROM messages").fetchone()[0]
            transition_count = cur.execute(
                "SELECT COUNT(*) FROM transitions").fetchone()[0]
            cur.close()
        if stored_version != FORMAT_VERSION or stored_digest != actual:
            report = {
                "format_version_stored": stored_version,
                "format_version_expected": FORMAT_VERSION,
                "version_diff": FORMAT_VERSION - stored_version,
                "message_count": message_count,
                "discarded_transitions": transition_count,
                "digest_stored": stored_digest,
                "digest_actual": actual,
                "path": self.path,
            }
            raise SnapshotMismatchError(
                "ledger format version or content digest mismatch", report)

    def tamper_digest(self) -> None:
        """Test hook: corrupt the stored digest to trigger E_SNAPSHOT."""
        with self._lock:
            self.conn.execute(
                "UPDATE meta SET value='tampered' WHERE key='content_digest'")
            self.conn.commit()

    # -------------------------------------------------------------- replacement
    @staticmethod
    def replace(path: str, backup_path: str, spec, inputs,
                epoch_ms: int) -> dict:
        """Full replacement; old ledger retained as a read-only backup."""
        old_conn = sqlite3.connect(path)
        try:
            old_version = int(old_conn.execute(
                "SELECT value FROM meta WHERE key='format_version'"
            ).fetchone()[0])
            old_messages = old_conn.execute(
                "SELECT COUNT(*) FROM messages").fetchone()[0]
            old_dedup = old_conn.execute(
                "SELECT COUNT(*) FROM dedup").fetchone()[0]
            discarded_transitions = old_conn.execute(
                "SELECT COUNT(*) FROM transitions").fetchone()[0]
        finally:
            old_conn.close()

        os.makedirs(os.path.dirname(os.path.abspath(backup_path)),
                    exist_ok=True)
        shutil.copy2(path, backup_path)
        _read_only(backup_path)
        for suffix in ("-wal", "-shm"):
            sidecar = path + suffix
            if os.path.exists(sidecar):
                shutil.copy2(sidecar, backup_path + suffix)
                _read_only(backup_path + suffix)
                os.remove(sidecar)
        os.remove(path)

        new_ledger = Ledger.create(path, spec, epoch_ms)
        return {
            "backup_path": os.path.abspath(backup_path),
            "format_version_old": old_version,
            "format_version_new": FORMAT_VERSION,
            "version_diff": FORMAT_VERSION - old_version,
            "message_count_old": old_messages,
            "message_count_new": len(inputs),
            "message_count_diff": len(inputs) - old_messages,
            "dedup_count_old": old_dedup,
            "discarded_transitions": discarded_transitions,
        }

    # -------------------------------------------------------------- row mapping
    @staticmethod
    def row_to_message(row) -> dict:
        return {
            "message_id": row[0],
            "topic": row[1],
            "deliver_at_abs": row[2],
            "origin_offset_ms": row[3],
            "payload": bytes(row[4]),
            "payload_summary": row[5],
            "idempotency_key": row[6],
            "state": row[7],
            "attempts": row[8],
            "suppressed": row[9],
            "unknown_count": row[10],
            "ready_monotonic_ms": row[11],
            "first_error": row[12],
            "last_error": row[13],
            "seq": row[14],
        }

    _MESSAGE_COLUMNS = (
        "message_id, topic, deliver_at_abs, origin_offset_ms, payload, "
        "payload_summary, idempotency_key, state, attempts, suppressed, "
        "unknown_count, ready_monotonic_ms, first_error, last_error, seq"
    )

    def load_messages(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {self._MESSAGE_COLUMNS} FROM messages ORDER BY seq"
            ).fetchall()
        return [self.row_to_message(row) for row in rows]

    # ------------------------------------------------------------- recovery
    def recover_inflight(self) -> dict:
        """Re-drive committed inflight leases after a hard process kill.

        Monotonic time cannot survive process restarts, so an open lease whose
        resolution never committed is annotated ``unknown`` and moved back to
        ready via the visibility-timeout transition (attempts untouched,
        visibility rows retained).
        """
        report = {"recovered_messages": 0, "unknown_messages": 0,
                  "dead_messages": 0, "redelivered_after_visibility": []}
        with self.batch() as cur:
            batch_id = self.next_batch_id(cur)
            rows = cur.execute(
                f"SELECT {self._MESSAGE_COLUMNS} FROM messages "
                "WHERE state=? ORDER BY seq", (INFLIGHT,)).fetchall()
            now = mono_ms()
            for row in rows:
                message = self.row_to_message(row)
                cur.execute(
                    "UPDATE messages SET state=?, unknown_count=?, "
                    "ready_monotonic_ms=? WHERE message_id=?",
                    (READY, message["unknown_count"] + 1, now,
                     message["message_id"]),
                )
                cur.execute(
                    "UPDATE visibility SET closed_mono_ms=?, close_reason=? "
                    "WHERE message_id=? AND closed_mono_ms IS NULL",
                    (now, UNKNOWN_MARKER, message["message_id"]),
                )
                cur.execute(
                    "INSERT INTO transitions(batch_id, message_id, from_state,"
                    " to_state, reason, attempts, mono_ms, wall_ms) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (batch_id, message["message_id"], INFLIGHT, READY,
                     "visibility_timeout", message["attempts"], now,
                     wall_ms()),
                )
                report["unknown_messages"] += 1
                report["redelivered_after_visibility"].append(
                    message["message_id"])
            report["recovered_messages"] = cur.execute(
                "SELECT COUNT(*) FROM messages").fetchone()[0]
            report["dead_messages"] = cur.execute(
                "SELECT COUNT(*) FROM messages WHERE state=?",
                (DEAD,)).fetchone()[0]
            cur.execute(
                "INSERT INTO progress(batch_id, cursor_seq, committed_at_ms) "
                "VALUES(?,?,?)",
                (batch_id, report["recovered_messages"], wall_ms()),
            )
        return report

    def migration_count(self, message_id: str) -> int:
        with self._lock:
            return self.conn.execute(
                "SELECT COUNT(*) FROM transitions WHERE message_id=?",
                (message_id,)).fetchone()[0]

    def topic_rows(self) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(
                "SELECT topic, visibility_timeout_ms, max_deliveries, "
                "backoff_ms, concurrency FROM topics ORDER BY topic"
            ).fetchall()

    def bump_dedup_version(self, cur) -> None:
        current = int(self.get_meta("dedup_version", "0"))
        self._set_meta(cur, "dedup_version", str(current + 1))

    def assert_valid_state(self, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
