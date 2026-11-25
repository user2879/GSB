from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, digest, digest_text
from .errors import LedgerError
from .spec import Spec


FORMAT_VERSION = 1


class Ledger:
    def __init__(
        self,
        path: str | Path,
        spec: Spec,
        initial_snapshot: dict[str, Any],
        *,
        fresh: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self._connection = sqlite3.connect(str(self.path), timeout=10.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=10000")
        self._create_schema()
        self.snapshot_replaced = False
        self.replacement_reason = ""
        self._open_session(initial_snapshot, fresh=fresh)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.commit()
        except BaseException:
            self._connection.rollback()
            raise

    def close(self) -> None:
        try:
            self._connection.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        with self.transaction() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS session (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    run_id TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    declaration_version INTEGER NOT NULL,
                    spec_digest TEXT NOT NULL,
                    snapshot_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    updated_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_replacements (
                    replacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    old_declaration_version INTEGER,
                    old_spec_digest TEXT,
                    old_snapshot_version INTEGER,
                    new_declaration_version INTEGER NOT NULL,
                    new_spec_digest TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS commits (
                    commit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    format_version INTEGER NOT NULL,
                    declaration_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_ns INTEGER NOT NULL,
                    UNIQUE (run_id, event_seq)
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    lease_id TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    declaration_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ownership (
                    ownership_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    lease_id TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    declaration_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending (
                    request_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    format_version INTEGER NOT NULL,
                    declaration_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS recoveries (
                    recovery_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    format_version INTEGER NOT NULL,
                    declaration_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    released_global_slots TEXT NOT NULL,
                    released_node_slots TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_ns INTEGER NOT NULL,
                    UNIQUE (run_id, request_id)
                );
                CREATE INDEX IF NOT EXISTS idx_commits_run_seq ON commits(run_id, event_seq);
                CREATE INDEX IF NOT EXISTS idx_transitions_run_seq ON transitions(run_id, event_seq);
                CREATE INDEX IF NOT EXISTS idx_ownership_run_seq ON ownership(run_id, event_seq);
                """
            )

    def _open_session(self, initial_snapshot: dict[str, Any], *, fresh: bool) -> None:
        row = self._connection.execute("SELECT * FROM session WHERE id = 1").fetchone()
        reason = ""
        if fresh:
            reason = "fresh_requested" if row is not None else "initial"
        elif row is None:
            reason = "initial"
        elif int(row["format_version"]) != FORMAT_VERSION:
            reason = "format_version_changed"
        elif int(row["declaration_version"]) != self.spec.declaration_version:
            reason = "declaration_version_changed"
        elif row["spec_digest"] != self.spec.digest:
            reason = "declaration_digest_changed"
        elif digest_text(row["payload_json"]) != row["content_digest"]:
            reason = "snapshot_digest_mismatch"
        if reason:
            self._replace_session(initial_snapshot, reason, row)
            self.snapshot_replaced = True
            self.replacement_reason = reason
        else:
            self.snapshot_replaced = False
            self.replacement_reason = "preserved"

    def _replace_session(
        self,
        initial_snapshot: dict[str, Any],
        reason: str,
        old_row: sqlite3.Row | None,
    ) -> None:
        run_id = uuid.uuid4().hex
        payload_json = canonical_json(initial_snapshot)
        content_digest = digest_text(payload_json)
        now_ns = time.time_ns()
        replacement_payload = {
            "run_id": run_id,
            "reason": reason,
            "new_declaration_version": self.spec.declaration_version,
            "new_spec_digest": self.spec.digest,
            "new_snapshot_version": 1,
        }
        with self.transaction() as conn:
            conn.execute("DELETE FROM commits")
            conn.execute("DELETE FROM transitions")
            conn.execute("DELETE FROM ownership")
            conn.execute("DELETE FROM pending")
            conn.execute("DELETE FROM recoveries")
            conn.execute(
                """
                INSERT INTO session_replacements(
                    run_id, format_version, old_declaration_version, old_spec_digest,
                    old_snapshot_version, new_declaration_version, new_spec_digest,
                    reason, content_digest, created_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    FORMAT_VERSION,
                    int(old_row["declaration_version"]) if old_row is not None else None,
                    old_row["spec_digest"] if old_row is not None else None,
                    int(old_row["snapshot_version"]) if old_row is not None else None,
                    self.spec.declaration_version,
                    self.spec.digest,
                    reason,
                    digest(replacement_payload),
                    now_ns,
                ),
            )
            conn.execute(
                """
                INSERT INTO session(
                    id, run_id, format_version, declaration_version, spec_digest,
                    snapshot_version, payload_json, content_digest, updated_ns
                ) VALUES (1, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    run_id=excluded.run_id,
                    format_version=excluded.format_version,
                    declaration_version=excluded.declaration_version,
                    spec_digest=excluded.spec_digest,
                    snapshot_version=excluded.snapshot_version,
                    payload_json=excluded.payload_json,
                    content_digest=excluded.content_digest,
                    updated_ns=excluded.updated_ns
                """,
                (
                    run_id,
                    FORMAT_VERSION,
                    self.spec.declaration_version,
                    self.spec.digest,
                    payload_json,
                    content_digest,
                    now_ns,
                ),
            )

    def session_info(self) -> dict[str, Any]:
        row = self._connection.execute("SELECT * FROM session WHERE id = 1").fetchone()
        if row is None:
            raise LedgerError("ledger has no active session")
        return {
            "run_id": row["run_id"],
            "format_version": int(row["format_version"]),
            "declaration_version": int(row["declaration_version"]),
            "spec_digest": row["spec_digest"],
            "snapshot_version": int(row["snapshot_version"]),
            "content_digest": row["content_digest"],
            "updated_ns": int(row["updated_ns"]),
            "snapshot_replaced": self.snapshot_replaced,
            "replacement_reason": self.replacement_reason,
        }

    @property
    def run_id(self) -> str:
        return str(self.session_info()["run_id"])

    def load_snapshot(self, *, verify: bool = True) -> dict[str, Any]:
        row = self._connection.execute("SELECT * FROM session WHERE id = 1").fetchone()
        if row is None:
            raise LedgerError("ledger has no active snapshot")
        payload_json = row["payload_json"]
        if verify and digest_text(payload_json) != row["content_digest"]:
            raise LedgerError("snapshot content digest mismatch")
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise LedgerError("snapshot payload is not an object")
        return payload

    def record_pending(
        self,
        event_seq: int,
        payload: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> str:
        request_id = request_id or uuid.uuid4().hex
        payload_json = canonical_json(payload)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO pending(
                    request_id, run_id, event_seq, format_version, declaration_version,
                    status, payload_json, content_digest, created_ns
                ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    request_id,
                    self.run_id,
                    event_seq,
                    FORMAT_VERSION,
                    self.spec.declaration_version,
                    payload_json,
                    digest_text(payload_json),
                    time.time_ns(),
                ),
            )
        return request_id

    def commit_event(
        self,
        *,
        event_seq: int,
        outcome: dict[str, Any],
        reclaims: list[dict[str, Any]],
        transitions: list[dict[str, Any]],
        ownership_changes: list[dict[str, Any]],
        snapshot: dict[str, Any],
        pending_request_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = self.run_id
        now_ns = time.time_ns()
        commit_payload = {
            "event_seq": event_seq,
            "outcome": outcome,
            "reclaims": reclaims,
            "transitions": transitions,
            "ownership_changes": ownership_changes,
        }
        commit_json = canonical_json(commit_payload)
        snapshot_json = canonical_json(snapshot)
        snapshot_digest = digest_text(snapshot_json)
        with self.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO commits(
                        run_id, event_seq, format_version, declaration_version,
                        payload_json, content_digest, created_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        event_seq,
                        FORMAT_VERSION,
                        self.spec.declaration_version,
                        commit_json,
                        digest_text(commit_json),
                        now_ns,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise LedgerError("event has already been committed", event_seq=event_seq) from exc
            for item in transitions:
                payload_json = canonical_json(item)
                conn.execute(
                    """
                    INSERT INTO transitions(
                        run_id, event_seq, lease_id, format_version, declaration_version,
                        payload_json, content_digest, created_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        event_seq,
                        item["lease_id"],
                        FORMAT_VERSION,
                        self.spec.declaration_version,
                        payload_json,
                        digest_text(payload_json),
                        now_ns,
                    ),
                )
            for item in ownership_changes:
                payload_json = canonical_json(item)
                conn.execute(
                    """
                    INSERT INTO ownership(
                        run_id, event_seq, lease_id, format_version, declaration_version,
                        payload_json, content_digest, created_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        event_seq,
                        item["lease_id"],
                        FORMAT_VERSION,
                        self.spec.declaration_version,
                        payload_json,
                        digest_text(payload_json),
                        now_ns,
                    ),
                )
            current = conn.execute("SELECT snapshot_version FROM session WHERE id = 1").fetchone()
            if current is None:
                raise LedgerError("ledger session disappeared during commit")
            next_version = int(current["snapshot_version"]) + 1
            conn.execute(
                """
                UPDATE session SET
                    snapshot_version = ?,
                    payload_json = ?,
                    content_digest = ?,
                    updated_ns = ?
                WHERE id = 1
                """,
                (next_version, snapshot_json, snapshot_digest, now_ns),
            )
            if pending_request_id is not None:
                conn.execute(
                    "DELETE FROM pending WHERE request_id = ? AND status = 'pending'",
                    (pending_request_id,),
                )
        return {
            "event_seq": event_seq,
            "snapshot_version": next_version,
            "snapshot_digest": snapshot_digest,
            "commit_digest": digest_text(commit_json),
        }

    def pending_requests(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM pending ORDER BY created_ns, request_id"
        ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            result.append(
                {
                    "request_id": row["request_id"],
                    "event_seq": int(row["event_seq"]),
                    "status": row["status"],
                    "payload": payload,
                    "content_digest": row["content_digest"],
                    "format_version": int(row["format_version"]),
                }
            )
        return result

    def recover_pending(self) -> list[dict[str, Any]]:
        run_id = self.run_id
        pending_rows = self._connection.execute(
            "SELECT * FROM pending WHERE run_id = ? AND status = 'pending' ORDER BY created_ns, request_id",
            (run_id,),
        ).fetchall()
        now_ns = time.time_ns()
        if pending_rows:
            with self.transaction() as conn:
                for row in pending_rows:
                    payload = json.loads(row["payload_json"])
                    reservations = payload.get("reservations", [])
                    released_global = sorted(
                        int(reservation["global_slot"])
                        for reservation in reservations
                        if reservation.get("global_slot") is not None
                    )
                    released_node = sorted(
                        [
                            str(reservation.get("node_id")),
                            int(reservation["node_slot"]),
                        ]
                        for reservation in reservations
                        if reservation.get("node_slot") is not None
                    )
                    recovery_payload = {
                        "request_id": row["request_id"],
                        "event_seq": int(row["event_seq"]),
                        "status": "unknown",
                        "released_global_slots": released_global,
                        "released_node_slots": released_node,
                        "original_payload": payload,
                    }
                    payload_json = canonical_json(recovery_payload)
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO recoveries(
                            run_id, request_id, event_seq, format_version, declaration_version,
                            status, released_global_slots, released_node_slots,
                            payload_json, content_digest, created_ns
                        ) VALUES (?, ?, ?, ?, ?, 'unknown', ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            row["request_id"],
                            int(row["event_seq"]),
                            FORMAT_VERSION,
                            self.spec.declaration_version,
                            canonical_json(released_global),
                            canonical_json(released_node),
                            payload_json,
                            digest_text(payload_json),
                            now_ns,
                        ),
                    )
                    conn.execute(
                        "UPDATE pending SET status = 'unknown' WHERE request_id = ?",
                        (row["request_id"],),
                    )
        recovery_rows = self._connection.execute(
            "SELECT * FROM recoveries WHERE run_id = ? ORDER BY event_seq, recovery_id",
            (run_id,),
        ).fetchall()
        recovered: list[dict[str, Any]] = []
        for row in recovery_rows:
            if digest_text(row["payload_json"]) != row["content_digest"]:
                raise LedgerError("recovery content digest mismatch", request_id=row["request_id"])
            recovered.append(
                {
                    "request_id": row["request_id"],
                    "event_seq": int(row["event_seq"]),
                    "status": row["status"],
                    "released_global_slots": json.loads(row["released_global_slots"]),
                    "released_node_slots": json.loads(row["released_node_slots"]),
                }
            )
        return recovered

    def read_commits(self, *, verify: bool = True) -> list[dict[str, Any]]:
        run_id = self.run_id
        rows = self._connection.execute(
            "SELECT * FROM commits WHERE run_id = ? ORDER BY event_seq, commit_id",
            (run_id,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            if verify and digest_text(row["payload_json"]) != row["content_digest"]:
                raise LedgerError("commit content digest mismatch", event_seq=int(row["event_seq"]))
            payload = json.loads(row["payload_json"])
            result.append(
                {
                    "event_seq": int(row["event_seq"]),
                    "format_version": int(row["format_version"]),
                    "declaration_version": int(row["declaration_version"]),
                    "content_digest": row["content_digest"],
                    "payload": payload,
                }
            )
        return result

    def utf8_commit_bytes(self) -> bytes:
        return "\n".join(
            canonical_json(item["payload"]) for item in self.read_commits()
        ).encode("utf-8")
