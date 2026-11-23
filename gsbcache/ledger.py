"""SQLite arbitration ledger with one transaction per custody transition."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .errors import ArbiterError, LedgerError
from .spec import Spec
from .util import atomic_write, canonical_bytes, canonical_json, now_ms, sha256_value

LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
GENESIS_HASH = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    capacity_slots INTEGER NOT NULL,
    capacity_used INTEGER NOT NULL,
    capacity_remaining INTEGER NOT NULL,
    lease_limit INTEGER NOT NULL,
    active_leases INTEGER NOT NULL,
    quota_remaining INTEGER NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    last_event_seq INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    artifact_keys TEXT NOT NULL,
    digests TEXT NOT NULL,
    requested_at_ms INTEGER NOT NULL,
    expires_at_ms INTEGER NOT NULL,
    deadline_mono_ns INTEGER NOT NULL,
    state TEXT NOT NULL,
    created_seq INTEGER NOT NULL,
    closed_seq INTEGER,
    release_kind TEXT,
    release_reason TEXT,
    owner_session TEXT,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_leases_state ON leases(state);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    arbitration_index INTEGER,
    request_id TEXT,
    lease_id TEXT,
    node_id TEXT,
    artifact_keys TEXT,
    digests TEXT,
    expected_digest TEXT,
    expires_at_ms INTEGER,
    release_kind TEXT,
    details TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    node_id TEXT,
    lease_id TEXT,
    state TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    ended_at_ms INTEGER,
    owner_session TEXT,
    details TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_state ON attempts(state);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    state TEXT NOT NULL,
    started_at_ms INTEGER NOT NULL,
    ended_at_ms INTEGER,
    exit_report INTEGER NOT NULL DEFAULT 0,
    details TEXT NOT NULL
);
"""


def _json(value: Any) -> str:
    return canonical_json(value)


def _loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def _integer(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LedgerError("ledger integer field is invalid", field=field, actual=value)
    return value


class Ledger:
    def __init__(
        self,
        path: str | Path,
        spec: Spec | None = None,
        *,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.spec = spec
        self.read_only = read_only
        self.lock = threading.RLock()
        self.closed = False
        if read_only:
            if not self.path.exists():
                raise LedgerError("ledger does not exist", path=str(self.path), field="path")
            uri = f"file:{self.path.resolve().as_posix()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=30.0)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA query_only=ON")
            self._verify_existing()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30.0)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.execute("PRAGMA busy_timeout=30000")
            self.conn.executescript(SCHEMA)
            self._initialize_or_verify()

    @classmethod
    def open(
        cls,
        path: str | Path,
        spec: Spec,
        *,
        read_only: bool = False,
    ) -> "Ledger":
        return cls(path, spec, read_only=read_only)

    def close(self) -> None:
        if not self.closed:
            try:
                self.conn.close()
            finally:
                self.closed = True

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def transaction(self, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        with self.lock:
            if self.read_only:
                raise LedgerError("ledger is read-only")
            self.conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield self.conn
            except Exception:
                self.conn.rollback()
                raise
            else:
                self.conn.commit()

    def _meta_get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_int(self, key: str) -> int | None:
        value = self._meta_get(key)
        return None if value is None else int(value)

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def _initialize_or_verify(self) -> None:
        with self.lock:
            version = self._meta_int("format_version")
            if version is None:
                if self.spec is None:
                    raise LedgerError("new ledger requires a spec", field="spec")
                with self.transaction():
                    self._set_meta(self.conn, "format_version", LEDGER_FORMAT_VERSION)
                    self._set_meta(self.conn, "cursor", 0)
                    self._set_meta(self.conn, "last_event_hash", GENESIS_HASH)
                    self._set_meta(self.conn, "pipeline_id", self.spec.pipeline.pipeline_id)
                    self._set_meta(self.conn, "spec_digest", self.spec.digest)
                    self._set_meta(self.conn, "created_at_ms", now_ms())
                    for node in self.spec.nodes:
                        self.conn.execute(
                            """
                            INSERT INTO nodes(node_id,capacity_slots,capacity_used,capacity_remaining,lease_limit,active_leases,quota_remaining,generation,last_event_seq)
                            VALUES(?,?,?,?,?,?,?,?,?)
                            """,
                            (node.node_id, node.capacity_slots, 0, node.capacity_slots, node.lease_limit, 0, node.lease_limit, 0, 0),
                        )
                    self._refresh_digest(self.conn)
                return
            self._verify_existing()

    def _verify_existing(self) -> None:
        with self.lock:
            version = self._meta_int("format_version")
            if version != LEDGER_FORMAT_VERSION:
                raise LedgerError(
                    "ledger format version mismatch",
                    path=str(self.path),
                    field="format_version",
                    expected=LEDGER_FORMAT_VERSION,
                    actual=version,
                    differences=[f"format_version expected={LEDGER_FORMAT_VERSION} actual={version}"],
                )
            stored_digest = self._meta_get("content_digest")
            computed_digest = self._compute_state_digest()
            differences: list[str] = []
            if stored_digest != computed_digest:
                differences.append(f"content_digest expected={computed_digest} actual={stored_digest}")
            if self.spec is not None:
                stored_spec = self._meta_get("spec_digest")
                if stored_spec != self.spec.digest:
                    differences.append(
                        f"spec_digest expected={self.spec.digest} actual={stored_spec}"
                    )
            if differences:
                raise LedgerError(
                    "ledger version or content digest does not match disk state",
                    path=str(self.path),
                    field="content_digest" if stored_digest != computed_digest else "spec_digest",
                    expected=computed_digest,
                    actual=stored_digest,
                    differences=differences,
                    replacement_required=True,
                )

    def _table_dicts(self, table: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(f"SELECT * FROM {table}").fetchall()
        return [dict(row) for row in rows]

    def _read_state(self) -> dict[str, Any]:
        nodes = sorted(self._table_dicts("nodes"), key=lambda x: x["node_id"])
        leases = []
        for row in sorted(self._table_dicts("leases"), key=lambda x: x["lease_id"]):
            item = dict(row)
            item["artifact_keys"] = _loads(item["artifact_keys"], [])
            item["digests"] = _loads(item["digests"], {})
            leases.append(item)
        attempts = []
        for row in sorted(self._table_dicts("attempts"), key=lambda x: x["attempt_id"]):
            item = dict(row)
            item["details"] = _loads(item["details"], {})
            attempts.append(item)
        sessions = []
        for row in sorted(self._table_dicts("sessions"), key=lambda x: x["session_id"]):
            item = dict(row)
            item["details"] = _loads(item["details"], {})
            sessions.append(item)
        events: list[dict[str, Any]] = []
        for row in self.conn.execute("SELECT * FROM events ORDER BY seq").fetchall():
            item = dict(row)
            for field in ("artifact_keys", "digests", "details"):
                item[field] = _loads(item[field], [] if field == "artifact_keys" else {})
            events.append(item)
        meta = {
            "cursor": int(self._meta_get("cursor") or 0),
            "last_event_hash": self._meta_get("last_event_hash") or GENESIS_HASH,
            "pipeline_id": self._meta_get("pipeline_id"),
            "spec_digest": self._meta_get("spec_digest"),
            "created_at_ms": int(self._meta_get("created_at_ms") or 0),
        }
        return {
            "format_version": int(self._meta_get("format_version") or 0),
            "meta": meta,
            "nodes": nodes,
            "leases": leases,
            "attempts": attempts,
            "sessions": sessions,
            "events": events,
        }

    def _compute_state_digest(self, state: dict[str, Any] | None = None) -> str:
        return sha256_value(state if state is not None else self._read_state())

    def _refresh_digest(self, conn: sqlite3.Connection) -> str:
        state = self._read_state()
        digest = sha256_value(state)
        self._set_meta(conn, "content_digest", digest)
        return digest

    def state_snapshot(self) -> dict[str, Any]:
        with self.lock:
            state = self._read_state()
            state["snapshot_format"] = SNAPSHOT_FORMAT_VERSION
            state["content_digest"] = sha256_value({key: value for key, value in state.items() if key not in {"snapshot_format", "content_digest"}})
            return state

    def info(self) -> dict[str, Any]:
        with self.lock:
            return {
                "format_version": self._meta_int("format_version"),
                "content_digest": self._meta_get("content_digest"),
                "cursor": self._meta_int("cursor"),
                "pipeline_id": self._meta_get("pipeline_id"),
                "spec_digest": self._meta_get("spec_digest"),
                "events": self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
                "active_leases": self.conn.execute("SELECT COUNT(*) FROM leases WHERE state='active'").fetchone()[0],
                "nodes": self._table_dicts("nodes"),
            }

    def export_snapshot(self, path: str | Path) -> dict[str, Any]:
        snapshot = self.state_snapshot()
        atomic_write(path, (json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
        return {
            "path": str(path),
            "format_version": snapshot["format_version"],
            "content_digest": snapshot["content_digest"],
            "cursor": snapshot["meta"]["cursor"],
        }

    def begin_session(self, pid: int, details: dict[str, Any] | None = None) -> str:
        session_id = "session_" + uuid.uuid4().hex
        with self.transaction():
            self.conn.execute(
                "INSERT INTO sessions(session_id,pid,state,started_at_ms,exit_report,details) VALUES(?,?,?,?,?,?)",
                (session_id, int(pid), "running", now_ms(), 0, _json(details or {})),
            )
            self._refresh_digest(self.conn)
        return session_id

    def close_session(self, session_id: str, *, exit_report: bool = True, state: str = "completed") -> None:
        with self.transaction():
            self.conn.execute(
                "UPDATE sessions SET state=?,ended_at_ms=?,exit_report=? WHERE session_id=?",
                (state, now_ms(), 1 if exit_report else 0, session_id),
            )
            self._refresh_digest(self.conn)

    def recover(self, current_session: str | None = None) -> dict[str, Any]:
        """Handle only completed BEGIN records left behind by a forced kill."""
        result: dict[str, Any] = {
            "unknown_attempts": [],
            "stale_leases": [],
            "aborted_sessions": [],
            "exit_report": None,
            "requires_reapply": False,
        }
        with self.transaction():
            rows = self.conn.execute("SELECT * FROM sessions WHERE state='running'").fetchall()
            for row in rows:
                if current_session is not None and row["session_id"] == current_session:
                    continue
                self.conn.execute(
                    "UPDATE sessions SET state='aborted',ended_at_ms=?,exit_report=0 WHERE session_id=?",
                    (now_ms(), row["session_id"]),
                )
                result["aborted_sessions"].append(row["session_id"])
            attempts = self.conn.execute("SELECT * FROM attempts WHERE state='pending'").fetchall()
            for row in attempts:
                self.conn.execute(
                    "UPDATE attempts SET state='unknown',ended_at_ms=?,details=? WHERE attempt_id=?",
                    (
                        now_ms(),
                        _json({**_loads(row["details"], {}), "requires_reapply": True, "reason": "uncommitted_attempt_after_restart"}),
                        row["attempt_id"],
                    ),
                )
                self._append_event(
                    self.conn,
                    event_type="attempt_unknown",
                    arbitration_index=None,
                    request_id=row["request_id"],
                    lease_id=row["lease_id"],
                    node_id=row["node_id"],
                    artifact_keys=[row["artifact_key"]],
                    digests={},
                    expected_digest=None,
                    expires_at_ms=None,
                    release_kind=None,
                    details={
                        "attempt_id": row["attempt_id"],
                        "requires_reapply": True,
                        "reason": "uncommitted_attempt_after_restart",
                    },
                )
                result["unknown_attempts"].append(row["attempt_id"])
                result["requires_reapply"] = True
            leases = self.conn.execute("SELECT * FROM leases WHERE state='active'").fetchall()
            for row in leases:
                if current_session is not None and row["owner_session"] == current_session:
                    continue
                closed = self._close_lease_tx(
                    self.conn,
                    row["lease_id"],
                    release_kind="unknown",
                    reason="owner_session_terminated_before_restart",
                    event_type="lease_unknown",
                    extra_details={"requires_reapply": True},
                )
                if closed:
                    result["stale_leases"].append(row["lease_id"])
                    result["requires_reapply"] = True
            if result["unknown_attempts"] or result["stale_leases"] or result["aborted_sessions"]:
                self._refresh_digest(self.conn)
        return result

    def begin_attempt(
        self,
        request_id: str,
        artifact_key: str,
        node_id: str | None,
        *,
        owner_session: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        attempt_id = "attempt_" + uuid.uuid4().hex
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO attempts(attempt_id,request_id,artifact_key,node_id,lease_id,state,started_at_ms,owner_session,details)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    attempt_id,
                    request_id,
                    artifact_key,
                    node_id,
                    None,
                    "pending",
                    now_ms(),
                    owner_session,
                    _json(details or {}),
                ),
            )
            self._refresh_digest(self.conn)
        return attempt_id

    def commit_attempt(
        self,
        attempt_id: str,
        *,
        node_id: str,
        lease_id: str,
        request_id: str,
        artifact_keys: list[str],
        digests: dict[str, str],
        expected_digest: str,
        expires_at_ms: int,
        deadline_mono_ns: int,
        owner_session: str,
        arbitration_index: int,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        keys = list(artifact_keys)
        if not keys or len(set(keys)) != len(keys):
            raise LedgerError("grant requires a non-empty unique artifact key set", field="artifact_keys")
        if not isinstance(expires_at_ms, int) or isinstance(expires_at_ms, bool):
            raise LedgerError("expires_at_ms must be an integer", field="expires_at_ms", actual=expires_at_ms)
        with self.transaction():
            attempt = self.conn.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            if attempt is None:
                raise LedgerError("attempt does not exist", attempt_id=attempt_id, field="attempt_id")
            if attempt["state"] != "pending":
                raise LedgerError("attempt is not pending", attempt_id=attempt_id, field="state", actual=attempt["state"])
            node_row = self.conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
            if node_row is None:
                raise LedgerError("node does not exist", node_id=node_id, field="node_id")
            if node_row["active_leases"] + 1 > node_row["lease_limit"]:
                raise ArbiterError(
                    "per-node lease limit would be exceeded",
                    node_id=node_id,
                    field="lease_limit",
                    lease_limit=node_row["lease_limit"],
                    active_leases=node_row["active_leases"],
                )
            if node_row["capacity_used"] + len(keys) > node_row["capacity_slots"]:
                raise ArbiterError(
                    "node capacity would be exceeded",
                    node_id=node_id,
                    field="capacity_slots",
                    capacity_slots=node_row["capacity_slots"],
                    capacity_used=node_row["capacity_used"],
                    requested=len(keys),
                )
            placeholders = ",".join("?" for _ in keys)
            conflicts = self.conn.execute(
                f"SELECT lease_id,node_id,artifact_keys FROM leases WHERE state='active' AND node_id<>?",
                (node_id,),
            ).fetchall()
            for other in conflicts:
                other_keys = set(_loads(other["artifact_keys"], []))
                overlap = sorted(other_keys & set(keys))
                if overlap:
                    raise ArbiterError(
                        "artifact key is already leased by another node",
                        artifact_key=overlap[0],
                        field="artifact_keys",
                        node_id=node_id,
                        conflicting_lease_id=other["lease_id"],
                        conflicting_node_id=other["node_id"],
                    )
            # Even a duplicate key on the same node violates one-key-unique.
            same_node = self.conn.execute(
                "SELECT lease_id,artifact_keys FROM leases WHERE state='active' AND node_id=?",
                (node_id,),
            ).fetchall()
            for other in same_node:
                overlap = sorted(set(_loads(other["artifact_keys"], [])) & set(keys))
                if overlap:
                    raise ArbiterError(
                        "artifact key is already leased on this node",
                        artifact_key=overlap[0],
                        field="artifact_keys",
                        node_id=node_id,
                        conflicting_lease_id=other["lease_id"],
                    )
            seq = self._next_seq()
            requested_at = now_ms()
            self.conn.execute(
                """
                INSERT INTO leases(lease_id,node_id,artifact_keys,digests,requested_at_ms,expires_at_ms,
                    deadline_mono_ns,state,created_seq,closed_seq,release_kind,release_reason,owner_session,version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    lease_id,
                    node_id,
                    _json(keys),
                    _json(digests),
                    requested_at,
                    expires_at_ms,
                    deadline_mono_ns,
                    "active",
                    seq,
                    None,
                    None,
                    None,
                    owner_session,
                    1,
                ),
            )
            self.conn.execute(
                "UPDATE nodes SET capacity_used=capacity_used+?,capacity_remaining=capacity_remaining-?,"
                "active_leases=active_leases+1,quota_remaining=quota_remaining-1,last_event_seq=? WHERE node_id=?",
                (len(keys), len(keys), seq, node_id),
            )
            self.conn.execute(
                "UPDATE attempts SET state='committed',lease_id=?,ended_at_ms=?,node_id=?,details=? WHERE attempt_id=?",
                (lease_id, now_ms(), node_id, _json({**_loads(attempt["details"], {}), **(details or {})}), attempt_id),
            )
            event = self._append_event(
                self.conn,
                event_type="lease_granted",
                arbitration_index=arbitration_index,
                request_id=request_id,
                lease_id=lease_id,
                node_id=node_id,
                artifact_keys=keys,
                digests=digests,
                expected_digest=expected_digest,
                expires_at_ms=expires_at_ms,
                release_kind=None,
                details={
                    "attempt_id": attempt_id,
                    "capacity_slots_acquired": len(keys),
                    "quota_units_acquired": 1,
                    "capacity_remaining_after": int(node_row["capacity_slots"]) - (int(node_row["capacity_used"]) + len(keys)),
                    "quota_remaining_after": int(node_row["lease_limit"]) - (int(node_row["active_leases"]) + 1),
                    "requested_at_ms": requested_at,
                    **(details or {}),
                },
            )
            self._set_meta(self.conn, "cursor", seq)
            self._refresh_digest(self.conn)
        return event

    def renew_lease(
        self,
        lease_id: str,
        *,
        extend_ms: int,
        expires_at_ms: int,
        deadline_mono_ns: int,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(extend_ms, int) or extend_ms <= 0:
            raise LedgerError("extend_ms must be a positive integer", field="extend_ms")
        with self.transaction():
            row = self.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None or row["state"] != "active":
                raise ArbiterError("lease is not active", lease_id=lease_id, field="state")
            seq = self._next_seq()
            version = int(row["version"]) + 1
            self.conn.execute(
                "UPDATE leases SET expires_at_ms=?,deadline_mono_ns=?,version=? WHERE lease_id=?",
                (expires_at_ms, deadline_mono_ns, version, lease_id),
            )
            self.conn.execute(
                "UPDATE nodes SET last_event_seq=? WHERE node_id=?",
                (seq, row["node_id"]),
            )
            event = self._append_event(
                self.conn,
                event_type="lease_renewed",
                arbitration_index=None,
                request_id=None,
                lease_id=lease_id,
                node_id=row["node_id"],
                artifact_keys=_loads(row["artifact_keys"], []),
                digests=_loads(row["digests"], {}),
                expected_digest=None,
                expires_at_ms=expires_at_ms,
                release_kind=None,
                details={
                    "extend_ms": extend_ms,
                    "lease_version": version,
                    "capacity_slots_changed": 0,
                    "quota_units_changed": 0,
                    **(details or {}),
                },
            )
            self._set_meta(self.conn, "cursor", seq)
            self._refresh_digest(self.conn)
        return event

    def _next_seq(self) -> int:
        cursor = int(self._meta_get("cursor") or 0)
        seq = cursor + 1
        row = self.conn.execute("SELECT MAX(seq) AS max_seq FROM events").fetchone()
        if row and row["max_seq"] is not None and int(row["max_seq"]) >= seq:
            seq = int(row["max_seq"]) + 1
        return seq

    def _event_record(
        self,
        *,
        seq: int,
        event_type: str,
        arbitration_index: int | None,
        request_id: str | None,
        lease_id: str | None,
        node_id: str | None,
        artifact_keys: list[str],
        digests: dict[str, str],
        expected_digest: str | None,
        expires_at_ms: int | None,
        release_kind: str | None,
        details: dict[str, Any],
        created_at_ms: int,
        prev_hash: str,
    ) -> dict[str, Any]:
        return {
            "seq": seq,
            "event_type": event_type,
            "arbitration_index": arbitration_index,
            "request_id": request_id,
            "lease_id": lease_id,
            "node_id": node_id,
            "artifact_keys": list(artifact_keys),
            "digests": dict(digests),
            "expected_digest": expected_digest,
            "expires_at_ms": expires_at_ms,
            "release_kind": release_kind,
            "details": details,
            "created_at_ms": created_at_ms,
            "prev_hash": prev_hash,
        }

    @staticmethod
    def event_hash(record_without_hash: dict[str, Any]) -> str:
        return sha256_value(record_without_hash)

    def _append_event(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        arbitration_index: int | None,
        request_id: str | None,
        lease_id: str | None,
        node_id: str | None,
        artifact_keys: list[str],
        digests: dict[str, str],
        expected_digest: str | None,
        expires_at_ms: int | None,
        release_kind: str | None,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        seq = self._next_seq()
        prev_hash = self._meta_get("last_event_hash") or GENESIS_HASH
        record = self._event_record(
            seq=seq,
            event_type=event_type,
            arbitration_index=arbitration_index,
            request_id=request_id,
            lease_id=lease_id,
            node_id=node_id,
            artifact_keys=artifact_keys,
            digests=digests,
            expected_digest=expected_digest,
            expires_at_ms=expires_at_ms,
            release_kind=release_kind,
            details=details,
            created_at_ms=now_ms(),
            prev_hash=prev_hash,
        )
        event_hash = self.event_hash(record)
        conn.execute(
            """
            INSERT INTO events(seq,event_type,arbitration_index,request_id,lease_id,node_id,artifact_keys,digests,
                expected_digest,expires_at_ms,release_kind,details,created_at_ms,prev_hash,event_hash)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                seq,
                event_type,
                arbitration_index,
                request_id,
                lease_id,
                node_id,
                _json(artifact_keys),
                _json(digests),
                expected_digest,
                expires_at_ms,
                release_kind,
                _json(details),
                record["created_at_ms"],
                prev_hash,
                event_hash,
            ),
        )
        self._set_meta(conn, "last_event_hash", event_hash)
        self._set_meta(conn, "cursor", seq)
        return {**record, "event_hash": event_hash}

    def _close_lease_tx(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        *,
        release_kind: str,
        reason: str,
        event_type: str | None = None,
        extra_details: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        row = conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        if row is None:
            return None
        # Conditional transition is the single ownership decision.  Expiry and
        # quota reclamation race on this row; exactly one can win.
        changed = conn.execute(
            "UPDATE leases SET state='closed',closed_seq=?,release_kind=?,release_reason=?,version=version+1 WHERE lease_id=? AND state='active'",
            (None, release_kind, reason, lease_id),
        ).rowcount
        if changed == 0:
            return {
                "closed": False,
                "already_closed": True,
                "lease_id": lease_id,
                "winner_release_kind": row["release_kind"],
            }
        keys = _loads(row["artifact_keys"], [])
        seq = self._next_seq()
        conn.execute(
            "UPDATE leases SET closed_seq=? WHERE lease_id=?",
            (seq, lease_id),
        )
        node = conn.execute("SELECT * FROM nodes WHERE node_id=?", (row["node_id"],)).fetchone()
        if node is None:
            raise LedgerError("lease node missing", lease_id=lease_id, node_id=row["node_id"])
        conn.execute(
            "UPDATE nodes SET capacity_used=MAX(0,capacity_used-?),"
            "capacity_remaining=MIN(capacity_slots,capacity_remaining+?),"
            "active_leases=MAX(0,active_leases-1),"
            "quota_remaining=MIN(lease_limit,quota_remaining+1),last_event_seq=? WHERE node_id=?",
            (len(keys), len(keys), seq, row["node_id"]),
        )
        mapped_event = event_type or {
            "expiry": "lease_expired",
            "quota_recycle": "quota_recycled",
            "unknown": "lease_unknown",
        }.get(release_kind, "lease_released")
        event = self._append_event(
            conn,
            event_type=mapped_event,
            arbitration_index=None,
            request_id=None,
            lease_id=lease_id,
            node_id=row["node_id"],
            artifact_keys=list(keys),
            digests=_loads(row["digests"], {}),
            expected_digest=None,
            expires_at_ms=int(row["expires_at_ms"]),
            release_kind=release_kind,
            details={
                "custody_token": lease_id,
                "release_kind": release_kind,
                "reason": reason,
                "capacity_slots_released": len(keys),
                "quota_units_released": 1,
                "capacity_slots_after": max(0, int(node["capacity_slots"]) - (int(node["capacity_used"]) - len(keys))),
                "quota_units_after": max(0, int(node["lease_limit"]) - (int(node["active_leases"]) - 1)),
                "capacity_remaining_after": min(
                    int(node["capacity_slots"]),
                    int(node["capacity_remaining"]) + len(keys),
                ),
                "quota_remaining_after": min(
                    int(node["lease_limit"]),
                    int(node["quota_remaining"]) + 1,
                ),
                **(extra_details or {}),
            },
        )
        self._set_meta(conn, "cursor", seq)
        return event

    def close_lease(self, lease_id: str, release_kind: str = "manual", reason: str = "explicit_close") -> dict[str, Any]:
        if release_kind not in {"expiry", "quota_recycle", "quota", "manual", "unknown"}:
            raise LedgerError("unsupported release_kind", field="release_kind", actual=release_kind)
        kind = "quota_recycle" if release_kind == "quota" else release_kind
        with self.transaction():
            event = self._close_lease_tx(
                self.conn,
                lease_id,
                release_kind=kind,
                reason=reason,
            )
            if event is not None and "event_hash" in event:
                self._refresh_digest(self.conn)
                return {**event, "closed": True, "already_closed": False}
            if event is not None:
                return event
            return {"closed": False, "already_closed": True, "lease_id": lease_id}

    def expire_lease(self, lease_id: str) -> dict[str, Any]:
        return self.close_lease(lease_id, "expiry", "monotonic_deadline_reached")

    def recycle_quota(self, lease_id: str, reason: str = "quota_reclaim") -> dict[str, Any]:
        return self.close_lease(lease_id, "quota_recycle", reason)

    def abort_attempt(self, attempt_id: str, reason: str) -> None:
        with self.transaction():
            self.conn.execute(
                "UPDATE attempts SET state='aborted',ended_at_ms=?,details=? WHERE attempt_id=? AND state='pending'",
                (
                    now_ms(),
                    _json({"reason": reason}),
                    attempt_id,
                ),
            )
            self._refresh_digest(self.conn)

    def get_active_leases(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM leases WHERE state='active' ORDER BY lease_id").fetchall()
            return [self._lease_row(row) for row in rows]

    def get_lease(self, lease_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            return None if row is None else self._lease_row(row)

    def get_nodes(self) -> list[dict[str, Any]]:
        with self.lock:
            return sorted(self._table_dicts("nodes"), key=lambda x: x["node_id"])

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
            return None if row is None else dict(row)

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            sql = "SELECT * FROM events"
            params: tuple[Any, ...] = ()
            if event_type:
                sql += " WHERE event_type=?"
                params = (event_type,)
            sql += " ORDER BY seq"
            rows = self.conn.execute(sql, params).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["artifact_keys"] = _loads(item["artifact_keys"], [])
                item["digests"] = _loads(item["digests"], {})
                item["details"] = _loads(item["details"], {})
                result.append(item)
            return result

    def get_attempts(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM attempts ORDER BY started_at_ms,attempt_id").fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["details"] = _loads(item["details"], {})
                result.append(item)
            return result

    def get_sessions(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM sessions ORDER BY started_at_ms,session_id").fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["details"] = _loads(item["details"], {})
                result.append(item)
            return result

    @staticmethod
    def _lease_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["artifact_keys"] = _loads(item["artifact_keys"], [])
        item["digests"] = _loads(item["digests"], {})
        return item


# ---------------------------------------------------------------------------
# Raw snapshot replacement helpers


def _read_snapshot(path: str | Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except FileNotFoundError as exc:
        raise LedgerError("snapshot file not found", path=str(path), field="path") from exc
    except json.JSONDecodeError as exc:
        raise LedgerError("snapshot JSON is invalid", path=str(path), field="json", error=str(exc)) from exc
    if not isinstance(snapshot, dict):
        raise LedgerError("snapshot root must be an object", field="snapshot")
    return snapshot


def validate_snapshot(snapshot: dict[str, Any]) -> str:
    if snapshot.get("snapshot_format") != SNAPSHOT_FORMAT_VERSION:
        raise LedgerError(
            "snapshot format version mismatch",
            field="snapshot_format",
            expected=SNAPSHOT_FORMAT_VERSION,
            actual=snapshot.get("snapshot_format"),
        )
    version = snapshot.get("format_version")
    if not isinstance(version, int) or isinstance(version, bool) or version != LEDGER_FORMAT_VERSION:
        raise LedgerError(
            "ledger format version mismatch",
            field="format_version",
            expected=LEDGER_FORMAT_VERSION,
            actual=version,
        )
    for key in ("meta", "nodes", "leases", "attempts", "sessions", "events"):
        if key not in snapshot:
            raise LedgerError("snapshot is missing a required section", field=key)
    digest = snapshot.get("content_digest")
    if not isinstance(digest, str):
        raise LedgerError("snapshot content_digest must be a string", field="content_digest")
    state = {key: value for key, value in snapshot.items() if key not in {"snapshot_format", "content_digest"}}
    computed = sha256_value(state)
    if digest != computed:
        raise LedgerError(
            "snapshot content digest mismatch",
            field="content_digest",
            expected=computed,
            actual=digest,
            differences=[f"content_digest expected={computed} actual={digest}"],
        )
    # Event chains are validated independently before replacing the ledger.
    previous = GENESIS_HASH
    expected_seq = 0
    for event in snapshot["events"]:
        seq = _integer(event.get("seq"), "seq")
        if seq != expected_seq + 1:
            raise LedgerError(
                "snapshot event sequence is not contiguous",
                field="seq",
                expected=expected_seq + 1,
                actual=seq,
            )
        expected_seq = seq
        if event.get("prev_hash") != previous:
            raise LedgerError(
                "snapshot event hash chain break",
                field="prev_hash",
                seq=seq,
                expected=previous,
                actual=event.get("prev_hash"),
            )
        record = {key: value for key, value in event.items() if key != "event_hash"}
        computed_hash = sha256_value(record)
        if event.get("event_hash") != computed_hash:
            raise LedgerError(
                "snapshot event hash mismatch",
                field="event_hash",
                seq=seq,
                expected=computed_hash,
                actual=event.get("event_hash"),
            )
        previous = computed_hash
    meta = snapshot["meta"]
    if expected_seq != int(meta.get("cursor", 0)):
        raise LedgerError(
            "snapshot cursor does not match the event chain",
            field="cursor",
            expected=expected_seq,
            actual=meta.get("cursor"),
        )
    if previous != meta.get("last_event_hash"):
        raise LedgerError(
            "snapshot final event hash does not match metadata",
            field="last_event_hash",
            expected=previous,
            actual=meta.get("last_event_hash"),
        )
    return digest


def _raw_stats(path: str | Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {"exists": False}
    conn = sqlite3.connect(str(path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        meta = {str(row["key"]): str(row["value"]) for row in conn.execute("SELECT key,value FROM meta").fetchall()}
        stats: dict[str, Any] = {
            "exists": True,
            "format_version": int(meta["format_version"]) if meta.get("format_version") else None,
            "content_digest": meta.get("content_digest"),
            "cursor": int(meta["cursor"]) if meta.get("cursor") else None,
            "spec_digest": meta.get("spec_digest"),
        }
        for table in ("nodes", "leases", "events", "attempts", "sessions"):
            stats[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return stats
    except (sqlite3.Error, KeyError, ValueError) as exc:
        return {"exists": True, "error": str(exc)}
    finally:
        conn.close()


def _snapshot_stats(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "exists": True,
        "format_version": snapshot.get("format_version"),
        "content_digest": snapshot.get("content_digest"),
        "cursor": snapshot.get("meta", {}).get("cursor"),
        "spec_digest": snapshot.get("meta", {}).get("spec_digest"),
        "nodes": len(snapshot.get("nodes", [])),
        "leases": len(snapshot.get("leases", [])),
        "events": len(snapshot.get("events", [])),
        "attempts": len(snapshot.get("attempts", [])),
        "sessions": len(snapshot.get("sessions", [])),
    }


def snapshot_differences(current: dict[str, Any], snapshot: dict[str, Any]) -> list[str]:
    target = _snapshot_stats(snapshot)
    differences: list[str] = []
    for field in ("format_version", "content_digest", "cursor", "spec_digest"):
        if current.get(field) != target.get(field):
            differences.append(f"{field}: {current.get(field)!r} -> {target.get(field)!r}")
    for field in ("nodes", "leases", "events", "attempts", "sessions"):
        if current.get(field) != target.get(field):
            differences.append(f"{field}.count: {current.get(field)!r} -> {target.get(field)!r}")
    if current.get("error"):
        differences.append(f"current_db.error: {current['error']}")
    return sorted(differences)


def _create_db_from_snapshot(temp_path: Path, snapshot: dict[str, Any], digest: str) -> None:
    if temp_path.exists():
        temp_path.unlink()
    conn = sqlite3.connect(str(temp_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.executescript(SCHEMA)
        conn.execute("BEGIN IMMEDIATE")
        meta = dict(snapshot["meta"])
        meta["format_version"] = str(snapshot["format_version"])
        meta["content_digest"] = digest
        for key, value in meta.items():
            conn.execute("INSERT INTO meta(key,value) VALUES(?,?)", (key, str(value)))
        for row in snapshot["nodes"]:
            conn.execute(
                """
                INSERT INTO nodes(node_id,capacity_slots,capacity_used,capacity_remaining,lease_limit,active_leases,quota_remaining,generation,last_event_seq)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["node_id"],
                    int(row["capacity_slots"]),
                    int(row["capacity_used"]),
                    int(row.get("capacity_remaining", int(row["capacity_slots"]) - int(row["capacity_used"]))),
                    int(row["lease_limit"]),
                    int(row["active_leases"]),
                    int(row.get("quota_remaining", int(row["lease_limit"]) - int(row["active_leases"]))),
                    int(row.get("generation", 0)),
                    int(row.get("last_event_seq", 0)),
                ),
            )
        for row in snapshot["leases"]:
            conn.execute(
                """
                INSERT INTO leases(lease_id,node_id,artifact_keys,digests,requested_at_ms,expires_at_ms,
                    deadline_mono_ns,state,created_seq,closed_seq,release_kind,release_reason,owner_session,version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["lease_id"],
                    row["node_id"],
                    _json(row["artifact_keys"]),
                    _json(row["digests"]),
                    int(row["requested_at_ms"]),
                    int(row["expires_at_ms"]),
                    int(row["deadline_mono_ns"]),
                    row["state"],
                    int(row["created_seq"]),
                    row.get("closed_seq"),
                    row.get("release_kind"),
                    row.get("release_reason"),
                    row.get("owner_session"),
                    int(row.get("version", 1)),
                ),
            )
        for row in snapshot["attempts"]:
            conn.execute(
                """
                INSERT INTO attempts(attempt_id,request_id,artifact_key,node_id,lease_id,state,started_at_ms,ended_at_ms,owner_session,details)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["attempt_id"],
                    row["request_id"],
                    row["artifact_key"],
                    row.get("node_id"),
                    row.get("lease_id"),
                    row["state"],
                    int(row["started_at_ms"]),
                    row.get("ended_at_ms"),
                    row.get("owner_session"),
                    _json(row.get("details", {})),
                ),
            )
        for row in snapshot["sessions"]:
            conn.execute(
                """
                INSERT INTO sessions(session_id,pid,state,started_at_ms,ended_at_ms,exit_report,details)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    row["session_id"],
                    int(row["pid"]),
                    row["state"],
                    int(row["started_at_ms"]),
                    row.get("ended_at_ms"),
                    int(row.get("exit_report", 0)),
                    _json(row.get("details", {})),
                ),
            )
        for row in snapshot["events"]:
            conn.execute(
                """
                INSERT INTO events(seq,event_type,arbitration_index,request_id,lease_id,node_id,artifact_keys,digests,
                    expected_digest,expires_at_ms,release_kind,details,created_at_ms,prev_hash,event_hash)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["seq"],
                    row["event_type"],
                    row.get("arbitration_index"),
                    row.get("request_id"),
                    row.get("lease_id"),
                    row.get("node_id"),
                    _json(row.get("artifact_keys", [])),
                    _json(row.get("digests", {})),
                    row.get("expected_digest"),
                    row.get("expires_at_ms"),
                    row.get("release_kind"),
                    _json(row.get("details", {})),
                    int(row["created_at_ms"]),
                    row["prev_hash"],
                    row["event_hash"],
                ),
            )
        conn.commit()
        # Verify the independently reconstructed state before atomically
        # swapping the file into place.
        state = {
            "format_version": snapshot["format_version"],
            "meta": {
                "cursor": int(meta.get("cursor", 0)),
                "last_event_hash": meta.get("last_event_hash", GENESIS_HASH),
                "pipeline_id": meta.get("pipeline_id"),
                "spec_digest": meta.get("spec_digest"),
                "created_at_ms": int(meta.get("created_at_ms", 0)),
            },
            "nodes": sorted(
                [dict(row) for row in conn.execute("SELECT * FROM nodes").fetchall()],
                key=lambda x: x["node_id"],
            ),
            "leases": [],
            "attempts": [],
            "sessions": [],
            "events": [],
        }
        for row in conn.execute("SELECT * FROM leases ORDER BY lease_id").fetchall():
            item = dict(row)
            item["artifact_keys"] = _loads(item["artifact_keys"], [])
            item["digests"] = _loads(item["digests"], {})
            state["leases"].append(item)
        for row in conn.execute("SELECT * FROM attempts ORDER BY attempt_id").fetchall():
            item = dict(row)
            item["details"] = _loads(item["details"], {})
            state["attempts"].append(item)
        for row in conn.execute("SELECT * FROM sessions ORDER BY session_id").fetchall():
            item = dict(row)
            item["details"] = _loads(item["details"], {})
            state["sessions"].append(item)
        for row in conn.execute("SELECT * FROM events ORDER BY seq").fetchall():
            item = dict(row)
            item["artifact_keys"] = _loads(item["artifact_keys"], [])
            item["digests"] = _loads(item["digests"], {})
            item["details"] = _loads(item["details"], {})
            state["events"].append(item)
        if sha256_value(state) != digest:
            raise LedgerError("restored snapshot state digest mismatch", field="content_digest")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def replace_ledger_from_snapshot(ledger_path: str | Path, snapshot_path: str | Path) -> dict[str, Any]:
    ledger_path = Path(ledger_path)
    snapshot = _read_snapshot(snapshot_path)
    digest = validate_snapshot(snapshot)
    current = _raw_stats(ledger_path)
    differences = snapshot_differences(current, snapshot)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = ledger_path.with_name(ledger_path.name + ".replace.tmp")
    _create_db_from_snapshot(temp_path, snapshot, digest)
    # WAL/shm files from the old generation must not survive a whole-file
    # replacement.  The caller must not hold an open Ledger on this path.
    for suffix in ("-wal", "-shm"):
        old_sidecar = Path(str(ledger_path) + suffix)
        if old_sidecar.exists():
            try:
                old_sidecar.unlink()
            except OSError:
                pass
    os.replace(temp_path, ledger_path)
    return {
        "ok": True,
        "mode": "whole_snapshot_replace",
        "incremental_merge": False,
        "ledger": str(ledger_path),
        "snapshot": str(snapshot_path),
        "format_version": snapshot["format_version"],
        "content_digest": digest,
        "differences": differences,
        "restored": _snapshot_stats(snapshot),
    }
