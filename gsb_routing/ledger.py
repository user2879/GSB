"""SQLite session ledger with atomic batches and whole-snapshot replacement."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, first_field_difference, sha256_digest
from .errors import E_REPLAY, RouteError, replay_error


FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self.conn is not None:
                self.conn.close()

    def _create_schema(self) -> None:
        with self._lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    spec_digest TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    route_version INTEGER NOT NULL,
                    start_seq INTEGER NOT NULL,
                    end_seq INTEGER NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','committed','unknown')),
                    content_digest TEXT,
                    content_json TEXT,
                    error_code TEXT,
                    created_monotonic_ms INTEGER NOT NULL,
                    committed_monotonic_ms INTEGER
                );
                CREATE INDEX IF NOT EXISTS batches_session_cursor
                    ON batches(session_id, status, end_seq, batch_id);
                CREATE TABLE IF NOT EXISTS event_records (
                    session_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    route_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, seq),
                    FOREIGN KEY(batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS lease_records (
                    session_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    database_id TEXT NOT NULL,
                    route_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    PRIMARY KEY(session_id, batch_id, lease_id)
                );
                CREATE TABLE IF NOT EXISTS progress (
                    session_id TEXT PRIMARY KEY,
                    next_seq INTEGER NOT NULL,
                    last_batch_id TEXT,
                    route_version INTEGER NOT NULL,
                    updated_monotonic_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    snapshot_version INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_monotonic_ms INTEGER NOT NULL
                );
                """
            )
            existing = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
            if existing is None:
                self.conn.execute("INSERT INTO meta(key, value) VALUES('format_version', ?)", (str(FORMAT_VERSION),))
            elif int(existing["value"]) != FORMAT_VERSION:
                raise RouteError(
                    E_REPLAY,
                    "ledger.meta.format_version",
                    "ledger format version is unsupported",
                    {"expected": FORMAT_VERSION, "actual": existing["value"]},
                )

    def begin_batch(
        self,
        *,
        batch_id: str,
        session_id: str,
        spec_digest: str,
        route_version: int,
        start_seq: int,
        end_seq: int,
    ) -> None:
        created_ms = time.monotonic_ns() // 1_000_000
        with self._lock, self.conn:
            existing = self.conn.execute("SELECT status FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            if existing is not None and existing["status"] == "committed":
                return
            if existing is not None:
                self.conn.execute(
                    """
                    UPDATE batches
                       SET session_id=?, spec_digest=?, format_version=?, route_version=?,
                           start_seq=?, end_seq=?, status='pending', content_digest=NULL,
                           content_json=NULL, error_code=NULL, created_monotonic_ms=?,
                           committed_monotonic_ms=NULL
                     WHERE batch_id=?
                    """,
                    (
                        session_id,
                        spec_digest,
                        FORMAT_VERSION,
                        route_version,
                        start_seq,
                        end_seq,
                        created_ms,
                        batch_id,
                    ),
                )
            else:
                self.conn.execute(
                    """
                    INSERT INTO batches(
                        batch_id, session_id, spec_digest, format_version, route_version,
                        start_seq, end_seq, status, created_monotonic_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (batch_id, session_id, spec_digest, FORMAT_VERSION, route_version, start_seq, end_seq, created_ms),
                )

    def mark_unknown(self, batch_id: str, *, reason: str = "uncommitted_batch_requires_rerun") -> None:
        with self._lock, self.conn:
            row = self.conn.execute("SELECT status FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if row is None or row["status"] == "committed":
                return
            self.conn.execute(
                "UPDATE batches SET status='unknown', error_code=? WHERE batch_id=?",
                (reason, batch_id),
            )

    def recover_unknown(self) -> list[dict[str, Any]]:
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE batches SET status='unknown', error_code=COALESCE(error_code, 'process_interrupted_before_commit') WHERE status='pending'"
            )
            rows = self.conn.execute(
                "SELECT * FROM batches WHERE status='unknown' ORDER BY start_seq, batch_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def commit_batch(self, batch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        digest = sha256_digest(payload)
        content = canonical_json(payload)
        committed_ms = time.monotonic_ns() // 1_000_000
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
                if row is None:
                    raise RouteError(E_REPLAY, f"batches[{batch_id}]", "batch intent was not started")
                if row["status"] == "committed":
                    if row["content_digest"] != digest:
                        raise RouteError(
                            E_REPLAY,
                            f"batches[{batch_id}].content_digest",
                            "committed batch content changed",
                        )
                    self.conn.commit()
                    return payload
                self.conn.execute(
                    """
                    UPDATE batches
                       SET status='committed', content_digest=?, content_json=?,
                           committed_monotonic_ms=?, error_code=NULL
                     WHERE batch_id=?
                    """,
                    (digest, content, committed_ms, batch_id),
                )
                events = payload.get("events", [])
                for event in events:
                    event_digest = sha256_digest(event)
                    self.conn.execute(
                        """
                        INSERT INTO event_records(
                            session_id, batch_id, seq, route_version, content_digest, content_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, seq) DO UPDATE SET
                            batch_id=excluded.batch_id,
                            route_version=excluded.route_version,
                            content_digest=excluded.content_digest,
                            content_json=excluded.content_json
                        """,
                        (
                            row["session_id"],
                            batch_id,
                            int(event["seq"]),
                            int(event["route"]["route_version"]),
                            event_digest,
                            canonical_json(event),
                        ),
                    )
                leases = payload.get("leases", [])
                for lease in leases:
                    lease_digest = sha256_digest(lease)
                    self.conn.execute(
                        """
                        INSERT INTO lease_records(
                            session_id, batch_id, lease_id, event_seq, database_id,
                            route_version, status, content_digest, content_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(session_id, batch_id, lease_id) DO UPDATE SET
                            event_seq=excluded.event_seq,
                            database_id=excluded.database_id,
                            route_version=excluded.route_version,
                            status=excluded.status,
                            content_digest=excluded.content_digest,
                            content_json=excluded.content_json
                        """,
                        (
                            row["session_id"],
                            batch_id,
                            str(lease["lease_id"]),
                            int(lease["event_seq"]),
                            str(lease["database_id"]),
                            int(lease["route_version"]),
                            str(lease["status"]),
                            lease_digest,
                            canonical_json(lease),
                        ),
                    )
                cursor = payload.get("progress_cursor") or {}
                next_seq = int(cursor.get("next_seq", row["end_seq"] + 1))
                self.conn.execute(
                    """
                    INSERT INTO progress(session_id, next_seq, last_batch_id, route_version, updated_monotonic_ms)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        next_seq=excluded.next_seq,
                        last_batch_id=excluded.last_batch_id,
                        route_version=excluded.route_version,
                        updated_monotonic_ms=excluded.updated_monotonic_ms
                    """,
                    (row["session_id"], next_seq, batch_id, int(payload["route_version"]), committed_ms),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return payload

    def get_batch_payload(self, batch_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT status, content_json, content_digest FROM batches WHERE batch_id=?", (batch_id,)
            ).fetchone()
        if row is None or row["status"] != "committed" or row["content_json"] is None:
            return None
        payload = json.loads(row["content_json"])
        if sha256_digest(payload) != row["content_digest"]:
            raise RouteError(E_REPLAY, f"batches[{batch_id}].content_digest", "ledger content digest mismatch")
        return payload

    def committed_batches(self, session_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM batches WHERE status='committed'"
        params: list[Any] = []
        if session_id is not None:
            query += " AND session_id=?"
            params.append(session_id)
        query += " ORDER BY start_seq, end_seq, batch_id"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def unknown_batches(self, session_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM batches WHERE status='unknown'"
        params: list[Any] = []
        if session_id is not None:
            query += " AND session_id=?"
            params.append(session_id)
        query += " ORDER BY start_seq, batch_id"
        with self._lock:
            rows = self.conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def progress(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM progress WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row is not None else None

    def committed_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM event_records WHERE session_id=? ORDER BY seq", (session_id,)
            ).fetchall()
        events: list[dict[str, Any]] = []
        for row in rows:
            event = json.loads(row["content_json"])
            if sha256_digest(event) != row["content_digest"]:
                raise replay_error(int(row["seq"]), "content_digest", "stored event content digest mismatch")
            events.append(event)
        return events

    def committed_leases(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM lease_records WHERE session_id=? ORDER BY event_seq, database_id, lease_id",
                (session_id,),
            ).fetchall()
        leases: list[dict[str, Any]] = []
        for row in rows:
            lease = json.loads(row["content_json"])
            if sha256_digest(lease) != row["content_digest"]:
                raise replay_error(int(row["event_seq"]), "lease.content_digest", "stored lease digest mismatch")
            leases.append(lease)
        return leases

    def replay_verify(self, expected_events: Iterable[dict[str, Any]], *, session_id: str) -> None:
        stored = self.committed_events(session_id)
        expected_by_seq = {int(event["seq"]): event for event in expected_events}
        stored_by_seq = {int(event["seq"]): event for event in stored}
        for seq in sorted(set(stored_by_seq) | set(expected_by_seq)):
            stored_event = stored_by_seq.get(seq)
            expected_event = expected_by_seq.get(seq)
            if stored_event is None:
                continue
            if expected_event is None:
                raise replay_error(seq, "presence", "ledger contains an event absent from replay input")
            difference = first_field_difference(stored_event, expected_event)
            if difference is not None:
                raise replay_error(seq, difference, "ledger replay differs from deterministic recomputation")

    def verify_integrity(self, session_id: str | None = None) -> None:
        batches = self.committed_batches(session_id)
        for batch in batches:
            payload = self.get_batch_payload(batch["batch_id"])
            if payload is None:
                raise RouteError(E_REPLAY, f"batches[{batch['batch_id']}]", "committed batch is unreadable")
            if int(payload["start_seq"]) != int(batch["start_seq"]) or int(payload["end_seq"]) != int(batch["end_seq"]):
                raise RouteError(E_REPLAY, f"batches[{batch['batch_id']}].range", "batch range mismatch")
        if session_id is not None:
            self.committed_events(session_id)
            self.committed_leases(session_id)

    def save_snapshot(self, *, session_id: str, snapshot_version: int, state: dict[str, Any]) -> str:
        digest = sha256_digest(state)
        content = canonical_json(state)
        created_ms = time.monotonic_ns() // 1_000_000
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                old = self.conn.execute("SELECT snapshot_version FROM snapshots WHERE id=1").fetchone()
                if old is not None and snapshot_version <= int(old["snapshot_version"]):
                    raise RouteError(
                        E_REPLAY,
                        "snapshot.snapshot_version",
                        "snapshot version must increase for whole replacement",
                        {"current": old["snapshot_version"], "requested": snapshot_version},
                    )
                self.conn.execute(
                    """
                    INSERT INTO snapshots(id, snapshot_version, session_id, format_version, content_digest, content_json, created_monotonic_ms)
                    VALUES(1, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        snapshot_version=excluded.snapshot_version,
                        session_id=excluded.session_id,
                        format_version=excluded.format_version,
                        content_digest=excluded.content_digest,
                        content_json=excluded.content_json,
                        created_monotonic_ms=excluded.created_monotonic_ms
                    """,
                    (snapshot_version, session_id, FORMAT_VERSION, digest, content, created_ms),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return digest

    def load_snapshot(self, *, snapshot_version: int | None = None, session_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM snapshots WHERE id=1").fetchone()
        if row is None:
            raise RouteError(E_REPLAY, "snapshot", "no snapshot is available")
        if snapshot_version is not None and int(row["snapshot_version"]) != snapshot_version:
            raise RouteError(
                E_REPLAY,
                "snapshot.snapshot_version",
                "snapshot version does not match the requested restore version",
                {"expected": snapshot_version, "actual": row["snapshot_version"]},
            )
        if session_id is not None and row["session_id"] != session_id:
            raise RouteError(
                E_REPLAY,
                "snapshot.session_id",
                "snapshot belongs to another session",
                {"expected": session_id, "actual": row["session_id"]},
            )
        state = json.loads(row["content_json"])
        if sha256_digest(state) != row["content_digest"]:
            raise RouteError(E_REPLAY, "snapshot.content_digest", "snapshot content digest mismatch")
        return {
            "snapshot_version": int(row["snapshot_version"]),
            "session_id": row["session_id"],
            "format_version": int(row["format_version"]),
            "content_digest": row["content_digest"],
            "state": state,
        }


__all__ = ["FORMAT_VERSION", "Ledger"]
