from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import FORMAT_VERSION
from .canonical import canonical_text, content_digest
from .errors import VerifyError


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def digest_record(record: dict[str, Any]) -> str:
    return content_digest({"format_version": FORMAT_VERSION, "record": record})


class Ledger:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS event_log (
        log_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        stream TEXT NOT NULL,
        event_seq INTEGER NOT NULL,
        kind TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        payload_digest TEXT NOT NULL,
        created_monotonic_ms INTEGER NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL,
        UNIQUE(stream, event_seq)
    );
    CREATE TABLE IF NOT EXISTS ownership (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stream TEXT NOT NULL,
        sequence_no INTEGER NOT NULL,
        event_seq INTEGER NOT NULL,
        quota_version INTEGER NOT NULL,
        at_ms INTEGER,
        entitlement_id TEXT NOT NULL,
        lease_id TEXT,
        shard_id TEXT,
        node_id TEXT,
        previous_holder_key TEXT,
        holder_key TEXT,
        change TEXT NOT NULL,
        reason TEXT,
        record_json TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL,
        UNIQUE(stream, sequence_no)
    );
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        stream TEXT NOT NULL,
        event_seq INTEGER NOT NULL,
        outcome TEXT NOT NULL,
        record_json TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS leases (
        lease_id TEXT PRIMARY KEY,
        shard_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        lease_version INTEGER NOT NULL,
        renewal_count INTEGER NOT NULL,
        status TEXT NOT NULL,
        granted_ms INTEGER NOT NULL,
        expires_ms INTEGER NOT NULL,
        ttl_ms INTEGER NOT NULL,
        quota_version INTEGER NOT NULL,
        resource_units INTEGER NOT NULL,
        allocation_json TEXT NOT NULL,
        last_event_seq INTEGER NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL,
        record_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS attempts (
        shard_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        revision INTEGER NOT NULL,
        node_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        status TEXT NOT NULL,
        resource_units INTEGER NOT NULL,
        priority INTEGER NOT NULL,
        wait_started_ms INTEGER,
        wait_ended_ms INTEGER,
        dispatched_ms INTEGER,
        finished_ms INTEGER,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        record_json TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL,
        PRIMARY KEY(shard_id, attempt)
    );
    CREATE TABLE IF NOT EXISTS outcomes (
        shard_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        attempt INTEGER NOT NULL,
        node_id TEXT NOT NULL,
        lease_id TEXT NOT NULL,
        state TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        summary_digest TEXT NOT NULL,
        receipt_json TEXT NOT NULL,
        elapsed_ms REAL NOT NULL,
        resource_units INTEGER NOT NULL,
        wait_started_ms INTEGER,
        wait_ended_ms INTEGER,
        record_json TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL,
        committed_log_seq INTEGER NOT NULL,
        PRIMARY KEY(shard_id, revision, attempt)
    );
    CREATE TABLE IF NOT EXISTS current_view (
        shard_id TEXT PRIMARY KEY,
        revision INTEGER NOT NULL,
        attempt INTEGER NOT NULL,
        outcome_key TEXT NOT NULL,
        summary_digest TEXT NOT NULL,
        outcome_canonical TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS occupancy (
        occupancy_id INTEGER PRIMARY KEY AUTOINCREMENT,
        shard_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        lease_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        resource_units INTEGER NOT NULL,
        state TEXT NOT NULL,
        quota_version INTEGER NOT NULL,
        event_seq INTEGER NOT NULL,
        started_ms INTEGER NOT NULL,
        finished_ms INTEGER,
        occupancy_json TEXT NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS progress (
        id INTEGER PRIMARY KEY CHECK(id = 1),
        revision INTEGER NOT NULL,
        completed_count INTEGER NOT NULL,
        topology_order_json TEXT NOT NULL,
        cursor_json TEXT NOT NULL,
        topology_cursor TEXT,
        committed_log_seq INTEGER NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS recovery_marks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        shard_id TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        status TEXT NOT NULL,
        reason TEXT NOT NULL,
        detail_json TEXT NOT NULL,
        recovered_ms INTEGER NOT NULL,
        format_version INTEGER NOT NULL,
        content_digest TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS snapshots (
        snapshot_revision INTEGER PRIMARY KEY,
        snapshot_json TEXT NOT NULL,
        snapshot_digest TEXT NOT NULL,
        ledger_digest TEXT NOT NULL,
        created_ms INTEGER NOT NULL,
        format_version INTEGER NOT NULL
    );
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._lock = threading.RLock()
        if read_only:
            uri = f"file:{self.path.resolve()}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute("PRAGMA foreign_keys=ON")
        if not read_only:
            self.conn.executescript(self.SCHEMA)

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def initialize_meta(self, spec_digest: str, spec_path: str) -> None:
        if self.read_only:
            return
        record = {"spec_digest": spec_digest, "spec_path": spec_path, "ledger_format_version": FORMAT_VERSION}
        digest = digest_record(record)
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key,value,format_version,content_digest) VALUES(?,?,?,?)",
                ("spec", canonical_text(record), FORMAT_VERSION, digest),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value,format_version,content_digest) VALUES(?,?,?,?)",
                ("ledger_format_version", str(FORMAT_VERSION), FORMAT_VERSION, digest_record({"version": FORMAT_VERSION})),
            )

    def append_event(self, stream: str, event: dict[str, Any], *, kind: str | None = None) -> int:
        event_seq = int(event["seq"])
        payload_digest = content_digest(event)
        record = {"stream": stream, "event": event}
        digest = digest_record(record)
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT OR IGNORE INTO event_log
                (stream,event_seq,kind,payload_json,payload_digest,created_monotonic_ms,
                 format_version,content_digest)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    stream,
                    event_seq,
                    kind or event.get("type", "event"),
                    canonical_text(event),
                    payload_digest,
                    monotonic_ms(),
                    FORMAT_VERSION,
                    digest,
                ),
            )
            if cur.rowcount == 0:
                row = self.conn.execute(
                    "SELECT log_seq FROM event_log WHERE stream=? AND event_seq=?", (stream, event_seq)
                ).fetchone()
                return int(row["log_seq"])
            return int(cur.lastrowid)

    def record_ownership(self, stream: str, record: dict[str, Any]) -> None:
        digest = digest_record(record)
        with self._lock:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO ownership
                (stream,sequence_no,event_seq,quota_version,at_ms,entitlement_id,lease_id,shard_id,
                 node_id,previous_holder_key,holder_key,change,reason,record_json,format_version,content_digest)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    stream,
                    int(record["sequence"]),
                    int(record["event_seq"]),
                    int(record.get("quota_version", 0)),
                    record.get("at_ms"),
                    record["entitlement_id"],
                    record.get("lease_id"),
                    record.get("shard_id"),
                    record.get("node_id"),
                    record.get("previous_holder_key"),
                    record.get("holder_key"),
                    record["change"],
                    record.get("reason"),
                    canonical_text(record),
                    FORMAT_VERSION,
                    digest,
                ),
            )

    def record_decision(self, stream: str, event_seq: int, decision: dict[str, Any]) -> None:
        digest = digest_record(decision)
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO decisions(stream,event_seq,outcome,record_json,format_version,content_digest)
                VALUES(?,?,?,?,?,?)
                """,
                (stream, event_seq, decision.get("outcome", "decision"), canonical_text(decision), FORMAT_VERSION, digest),
            )

    def record_dispatch(self, attempt_record: dict[str, Any], ownership_record: dict[str, Any]) -> None:
        digest = digest_record(attempt_record)
        allocation = attempt_record.get("allocated_entitlements", {})
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO attempts
                    (shard_id,attempt,revision,node_id,lease_id,status,resource_units,priority,
                     wait_started_ms,wait_ended_ms,dispatched_ms,finished_ms,cancel_requested,
                     record_json,format_version,content_digest)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        attempt_record["shard_id"],
                        attempt_record["attempt"],
                        attempt_record["revision"],
                        attempt_record["node_id"],
                        attempt_record["lease_id"],
                        attempt_record["status"],
                        attempt_record["resource_units"],
                        attempt_record["priority"],
                        attempt_record.get("wait_started_ms"),
                        attempt_record.get("wait_ended_ms"),
                        attempt_record.get("dispatched_ms"),
                        attempt_record.get("finished_ms"),
                        1 if attempt_record.get("cancel_requested") else 0,
                        canonical_text(attempt_record),
                        FORMAT_VERSION,
                        digest,
                    ),
                )
                lease = attempt_record["lease"]
                lease_digest = digest_record(lease)
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO leases
                    (lease_id,shard_id,node_id,attempt,revision,lease_version,renewal_count,status,
                     granted_ms,expires_ms,ttl_ms,quota_version,resource_units,allocation_json,
                     last_event_seq,format_version,content_digest,record_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        lease["lease_id"], lease["shard_id"], lease["node_id"], attempt_record["attempt"],
                        attempt_record["revision"], lease["lease_version"], lease["renewal_count"], lease["status"],
                        lease["granted_at_ms"], lease["expires_at_ms"], lease["ttl_ms"], lease["quota_version"],
                        lease["resource_units"], canonical_text(allocation), lease["last_event_seq"],
                        FORMAT_VERSION, lease_digest, canonical_text(lease),
                    ),
                )
                occ = attempt_record["occupancy"]
                occ_digest = digest_record(occ)
                self.conn.execute(
                    """
                    INSERT INTO occupancy
                    (shard_id,attempt,lease_id,node_id,revision,resource_units,state,quota_version,
                     event_seq,started_ms,finished_ms,occupancy_json,format_version,content_digest)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        occ["shard_id"], occ["attempt"], occ["lease_id"], occ["node_id"], occ["revision"],
                        occ["resource_units"], occ["state"], occ["quota_version"], occ["event_seq"],
                        occ["started_ms"], occ.get("finished_ms"), canonical_text(occ),
                        FORMAT_VERSION, occ_digest,
                    ),
                )
                self.record_ownership("runtime", ownership_record)
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def next_attempt(self, shard_id: str, revision: int) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(attempt), 0) + 1 AS next_attempt FROM attempts WHERE shard_id=? AND revision=?",
            (shard_id, revision),
        ).fetchone()
        return int(row["next_attempt"])

    def commit_shard_result(self, result: dict[str, Any], *, topology_order: list[str], revision: int) -> int:
        result_digest = digest_record(result)
        summary_canonical = canonical_text(result["summary"])
        outcome_canonical = canonical_text(
            {
                "shard_id": result["shard_id"],
                "revision": result["revision"],
                "summary": result["summary"],
                "summary_digest": result["summary_digest"],
            }
        )
        view_digest = content_digest(outcome_canonical)
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                if result.get("release_event"):
                    self._insert_event_locked("runtime", result["release_event"])
                event = {
                    "seq": result["completion_seq"],
                    "stream": "runtime",
                    "type": "shard_committed",
                    "shard_id": result["shard_id"],
                    "revision": result["revision"],
                    "attempt": result["attempt"],
                    "lease_id": result["lease_id"],
                }
                log_seq = self._insert_event_locked("runtime", event)
                self.conn.execute(
                    """
                    INSERT INTO outcomes
                    (shard_id,revision,attempt,node_id,lease_id,state,summary_json,summary_digest,
                     receipt_json,elapsed_ms,resource_units,wait_started_ms,wait_ended_ms,record_json,
                     format_version,content_digest,committed_log_seq)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        result["shard_id"], result["revision"], result["attempt"], result["node_id"],
                        result["lease_id"], result.get("state", "COMMITTED"), summary_canonical,
                        result["summary_digest"], canonical_text(result.get("receipt", {})),
                        float(result["elapsed_ms"]), result["resource_units"],
                        result.get("wait_started_ms"), result.get("wait_ended_ms"),
                        canonical_text(result), FORMAT_VERSION, result_digest, log_seq,
                    ),
                )
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO current_view
                    (shard_id,revision,attempt,outcome_key,summary_digest,outcome_canonical,
                     format_version,content_digest)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        result["shard_id"], result["revision"], result["attempt"],
                        f"{result['revision']}:{result['attempt']}", result["summary_digest"],
                        outcome_canonical, FORMAT_VERSION, view_digest,
                    ),
                )
                self.conn.execute(
                    """
                    UPDATE occupancy SET state='RELEASED', finished_ms=?
                    WHERE lease_id=? AND attempt=? AND shard_id=?
                    """,
                    (monotonic_ms(), result["lease_id"], result["attempt"], result["shard_id"]),
                )
                self.conn.execute(
                    "UPDATE attempts SET status='COMMITTED', finished_ms=? WHERE shard_id=? AND attempt=?",
                    (monotonic_ms(), result["shard_id"], result["attempt"]),
                )
                self.conn.execute(
                    "UPDATE leases SET status='released' WHERE lease_id=?",
                    (result["lease_id"],),
                )
                ownership = result["ownership_record"]
                self._insert_ownership_locked("runtime", ownership)
                completed = [
                    r["shard_id"]
                    for r in self.conn.execute("SELECT shard_id FROM current_view ORDER BY shard_id")
                ]
                cursor = {
                    "revision": revision,
                    "completed_shard_ids": completed,
                    "completed_count": len(completed),
                    "last_committed_shard_id": result["shard_id"],
                    "topology_order": topology_order,
                }
                cursor_digest = digest_record(cursor)
                position = topology_order.index(result["shard_id"]) if result["shard_id"] in topology_order else None
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO progress
                    (id,revision,completed_count,topology_order_json,cursor_json,topology_cursor,
                     committed_log_seq,format_version,content_digest)
                    VALUES(1,?,?,?,?,?,?,?,?)
                    """,
                    (
                        revision, len(completed), canonical_text(topology_order), canonical_text(cursor),
                        result["shard_id"] if position is not None else None, log_seq,
                        FORMAT_VERSION, cursor_digest,
                    ),
                )
                self.conn.commit()
                return log_seq
            except Exception:
                self.conn.rollback()
                raise

    def _insert_event_locked(self, stream: str, event: dict[str, Any]) -> int:
        payload_digest = content_digest(event)
        record_digest = digest_record({"stream": stream, "event": event})
        cur = self.conn.execute(
            """
            INSERT INTO event_log
            (stream,event_seq,kind,payload_json,payload_digest,created_monotonic_ms,
             format_version,content_digest)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                stream, int(event["seq"]), event.get("type", "event"), canonical_text(event),
                payload_digest, monotonic_ms(), FORMAT_VERSION, record_digest,
            ),
        )
        return int(cur.lastrowid)

    def _insert_ownership_locked(self, stream: str, record: dict[str, Any]) -> None:
        digest = digest_record(record)
        self.conn.execute(
            """
            INSERT INTO ownership
            (stream,sequence_no,event_seq,quota_version,at_ms,entitlement_id,lease_id,shard_id,
             node_id,previous_holder_key,holder_key,change,reason,record_json,format_version,content_digest)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                stream, int(record["sequence"]), int(record["event_seq"]), int(record.get("quota_version", 0)),
                record.get("at_ms"), record["entitlement_id"], record.get("lease_id"), record.get("shard_id"),
                record.get("node_id"), record.get("previous_holder_key"), record.get("holder_key"),
                record["change"], record.get("reason"), canonical_text(record), FORMAT_VERSION, digest,
            ),
        )

    def record_recovery(self, marks: list[dict[str, Any]]) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for mark in marks:
                    digest = digest_record(mark)
                    if mark.get("event"):
                        self._insert_event_locked("runtime", mark["event"])
                    if mark.get("ownership"):
                        self._insert_ownership_locked("runtime", mark["ownership"])
                    self.conn.execute(
                        """
                        INSERT INTO recovery_marks
                        (shard_id,attempt,status,reason,detail_json,recovered_ms,format_version,content_digest)
                        VALUES(?,?,?,?,?,?,?,?)
                        """,
                        (
                            mark["shard_id"], mark["attempt"], mark["status"], mark["reason"],
                            canonical_text(mark), monotonic_ms(), FORMAT_VERSION, digest,
                        ),
                    )
                    self.conn.execute(
                        "UPDATE attempts SET status='UNKNOWN_RERUN', finished_ms=? WHERE shard_id=? AND attempt=?",
                        (monotonic_ms(), mark["shard_id"], mark["attempt"]),
                    )
                    self.conn.execute(
                        "UPDATE occupancy SET state='RECOVERED', finished_ms=? WHERE shard_id=? AND attempt=? AND state='HELD'",
                        (monotonic_ms(), mark["shard_id"], mark["attempt"]),
                    )
                    self.conn.execute(
                        "UPDATE leases SET status='recovered' WHERE lease_id=?",
                        (mark["lease_id"],),
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def open_uncommitted_outcome_for_crash(self, result: dict[str, Any]) -> None:
        """Open a transaction and insert an outcome without committing.

        The acceptance harness SIGKILLs the process while this transaction is
        open, proving SQLite rolls it back and recovery uses committed attempts.
        """

        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute(
            """
            INSERT INTO outcomes
            (shard_id,revision,attempt,node_id,lease_id,state,summary_json,summary_digest,
             receipt_json,elapsed_ms,resource_units,wait_started_ms,wait_ended_ms,record_json,
             format_version,content_digest,committed_log_seq)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result["shard_id"], result["revision"], result["attempt"], result["node_id"],
                result["lease_id"], "UNKNOWN_CRASH_WINDOW", canonical_text(result["summary"]),
                result["summary_digest"], canonical_text(result.get("receipt", {})),
                float(result["elapsed_ms"]), result["resource_units"], result.get("wait_started_ms"),
                result.get("wait_ended_ms"), canonical_text(result), FORMAT_VERSION,
                digest_record(result), -1,
            ),
        )

    def ledger_digest(self) -> str:
        rows: list[tuple[str, int, str]] = []
        for table in ("meta", "event_log", "ownership", "decisions", "leases", "attempts", "outcomes", "occupancy", "progress", "recovery_marks"):
            try:
                for row in self.conn.execute(f"SELECT rowid AS rid, content_digest FROM {table} ORDER BY rowid"):
                    rows.append((table, int(row["rid"]), str(row["content_digest"])))
            except sqlite3.OperationalError:
                continue
        return content_digest(rows)

    def write_snapshot_record(self, revision: int, snapshot: dict[str, Any]) -> None:
        digest = content_digest(snapshot)
        ledger_digest = self.ledger_digest()
        with self._lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO snapshots
                (snapshot_revision,snapshot_json,snapshot_digest,ledger_digest,created_ms,format_version)
                VALUES(?,?,?,?,?,?)
                """,
                (revision, canonical_text(snapshot), digest, ledger_digest, monotonic_ms(), FORMAT_VERSION),
            )


def read_only_ledger(path: str | Path) -> Ledger:
    if not Path(path).exists():
        raise VerifyError(f"ledger does not exist: {path}", path=str(path), field="ledger")
    return Ledger(path, read_only=True)
