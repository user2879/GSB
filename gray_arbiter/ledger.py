"""SQLite ledger with atomic batches, restart recovery, and replay checks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from . import LEDGER_FORMAT_VERSION
from .errors import BusinessError, StructuredLogger


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_digest(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("content_digest", None)
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


class ReplayMismatch(BusinessError):
    def __init__(
        self,
        *,
        request_id: str | None,
        field: str,
        expected: Any,
        actual: Any,
        batch_id: str,
        basis: str,
    ) -> None:
        super().__init__(
            "E_REPLAY",
            "ledger replay verification found a mismatch",
            field=field,
            request_id=request_id,
            pool_id=None,
            phase="replay",
            details={
                "batch_id": batch_id,
                "expected": expected,
                "actual": actual,
                "basis": basis,
            },
        )


class Ledger:
    """One local SQLite ledger.

    A committed batch is written in a single transaction together with its
    request rows, pool watermarks, borrows, lifecycle events, progress cursor,
    and content digest.  The inflight marker is intentionally committed
    separately before processing; after a hard kill it is the only evidence
    that the batch must be rerun.
    """

    def __init__(self, path: str | Path, logger: StructuredLogger | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or StructuredLogger()
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(
            str(self.path),
            timeout=15.0,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS version_epochs (
                    version INTEGER PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_digest TEXT NOT NULL,
                    created_at_ns INTEGER NOT NULL,
                    stale_at_ns INTEGER
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    config_version INTEGER NOT NULL,
                    ledger_format_version INTEGER NOT NULL,
                    start_cursor INTEGER NOT NULL,
                    end_cursor INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    committed_at_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS request_ledger (
                    batch_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    destination TEXT,
                    pool_id TEXT,
                    connection_id TEXT,
                    reuse_sequence INTEGER,
                    borrow_id TEXT,
                    lender_pool TEXT,
                    status TEXT NOT NULL,
                    error_code TEXT,
                    watermark_json TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    PRIMARY KEY (batch_id, ordinal),
                    UNIQUE (batch_id, request_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS pool_watermarks (
                    batch_id TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    stats_json TEXT NOT NULL,
                    PRIMARY KEY (batch_id, pool_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS borrow_ledger (
                    batch_id TEXT NOT NULL,
                    borrow_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (batch_id, borrow_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS mirror_ledger (
                    batch_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    pool_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (batch_id, request_id),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS connection_events (
                    batch_id TEXT NOT NULL,
                    event_seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (batch_id, event_seq),
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS inflight_batches (
                    batch_id TEXT PRIMARY KEY,
                    config_version INTEGER NOT NULL,
                    start_cursor INTEGER NOT NULL,
                    end_cursor INTEGER NOT NULL,
                    started_at_ns INTEGER NOT NULL,
                    resource_snapshot_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS interrupted_batches (
                    batch_id TEXT PRIMARY KEY,
                    config_version INTEGER NOT NULL,
                    start_cursor INTEGER NOT NULL,
                    end_cursor INTEGER NOT NULL,
                    detected_at_ns INTEGER NOT NULL,
                    rerun_required INTEGER NOT NULL,
                    detail_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS unknown_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    resource_kind TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    pool_id TEXT,
                    state TEXT NOT NULL,
                    needs_rerun INTEGER NOT NULL,
                    detected_at_ns INTEGER NOT NULL,
                    detail_json TEXT NOT NULL,
                    resolved INTEGER NOT NULL DEFAULT 0,
                    resolved_at_ns INTEGER
                );
                """
            )
            unknown_columns = {
                row["name"]
                for row in self.connection.execute("PRAGMA table_info(unknown_resources)").fetchall()
            }
            if "resolved" not in unknown_columns:
                self.connection.execute(
                    "ALTER TABLE unknown_resources ADD COLUMN resolved INTEGER NOT NULL DEFAULT 0"
                )
            if "resolved_at_ns" not in unknown_columns:
                self.connection.execute(
                    "ALTER TABLE unknown_resources ADD COLUMN resolved_at_ns INTEGER"
                )
            self.connection.execute(
                "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES (?, ?)",
                ("ledger_format_version", str(LEDGER_FORMAT_VERSION)),
            )
            self.connection.execute(
                "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES (?, ?)",
                ("content_digest_algorithm", "sha256"),
            )
            format_row = self.connection.execute(
                "SELECT value FROM ledger_meta WHERE key = 'ledger_format_version'"
            ).fetchone()
            if format_row is None or int(format_row["value"]) != LEDGER_FORMAT_VERSION:
                raise BusinessError(
                    "E_SNAPSHOT",
                    "ledger format version is incompatible",
                    field="ledger_meta.ledger_format_version",
                    request_id=None,
                    pool_id=None,
                    phase="ledger_open",
                    details={
                        "expected": LEDGER_FORMAT_VERSION,
                        "actual": None if format_row is None else format_row["value"],
                    },
                )

    # ------------------------------------------------------------------
    # Epochs and batch start
    # ------------------------------------------------------------------
    def ensure_epoch(self, version: int, config_digest: str, *, status: str = "active") -> None:
        with self._lock, self.connection:
            self.connection.execute(
                """
                INSERT INTO version_epochs(version, status, config_digest, created_at_ns)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    status = excluded.status,
                    config_digest = excluded.config_digest
                """,
                (version, status, config_digest, time.time_ns()),
            )

    def mark_epoch_stale(self, old_version: int, new_version: int, new_config_digest: str) -> None:
        now = time.time_ns()
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE version_epochs SET status = 'stale', stale_at_ns = ? WHERE version = ?",
                (now, old_version),
            )
            self.connection.execute(
                """
                INSERT INTO version_epochs(version, status, config_digest, created_at_ns)
                VALUES (?, 'active', ?, ?)
                ON CONFLICT(version) DO UPDATE SET
                    status = 'active',
                    config_digest = excluded.config_digest
                """,
                (new_version, new_config_digest, now),
            )

    def begin_batch(
        self,
        batch_id: str,
        *,
        config_version: int,
        start_cursor: int,
        end_cursor: int,
        resource_snapshot: Mapping[str, Any] | None = None,
    ) -> None:
        snapshot = dict(resource_snapshot or {})
        with self._lock:
            if self.has_batch(batch_id):
                raise BusinessError(
                    "E_REPLAY",
                    "batch is already committed and must not be overwritten",
                    field="batch_id",
                    request_id=None,
                    pool_id=None,
                    phase="ledger_begin",
                    details={"batch_id": batch_id},
                )
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO inflight_batches(
                        batch_id, config_version, start_cursor, end_cursor,
                        started_at_ns, resource_snapshot_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(batch_id) DO UPDATE SET
                        config_version = excluded.config_version,
                        start_cursor = excluded.start_cursor,
                        end_cursor = excluded.end_cursor,
                        started_at_ns = excluded.started_at_ns,
                        resource_snapshot_json = excluded.resource_snapshot_json
                    """,
                    (
                        batch_id,
                        config_version,
                        start_cursor,
                        end_cursor,
                        time.time_ns(),
                        canonical_json(snapshot),
                    ),
                )

    def update_inflight_resources(self, batch_id: str, snapshot: Mapping[str, Any]) -> None:
        with self._lock, self.connection:
            self.connection.execute(
                "UPDATE inflight_batches SET resource_snapshot_json = ? WHERE batch_id = ?",
                (canonical_json(snapshot), batch_id),
            )

    # ------------------------------------------------------------------
    # Atomic commit
    # ------------------------------------------------------------------
    def commit_batch(self, payload: Mapping[str, Any]) -> str:
        payload_copy = json.loads(canonical_json(payload))
        batch_id = payload_copy["batch_id"]
        fmt_version = int(payload_copy.get("ledger_format_version", LEDGER_FORMAT_VERSION))
        if fmt_version != LEDGER_FORMAT_VERSION:
            raise BusinessError(
                "E_SNAPSHOT",
                "ledger payload format version mismatch",
                field="ledger_format_version",
                request_id=None,
                pool_id=None,
                phase="ledger_commit",
                details={"expected": LEDGER_FORMAT_VERSION, "actual": fmt_version},
            )
        digest = content_digest(payload_copy)
        payload_json = canonical_json(payload_copy)
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                inflight = self.connection.execute(
                    "SELECT batch_id FROM inflight_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if inflight is None:
                    raise BusinessError(
                        "E_REPLAY",
                        "batch commit requires a committed inflight marker",
                        field="batch_id",
                        request_id=None,
                        pool_id=None,
                        phase="ledger_commit",
                        details={"batch_id": batch_id},
                    )
                existing = self.connection.execute(
                    "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
                ).fetchone()
                if existing is not None:
                    raise BusinessError(
                        "E_REPLAY",
                        "refusing to overwrite a committed batch",
                        field="batch_id",
                        request_id=None,
                        pool_id=None,
                        phase="ledger_commit",
                        details={"batch_id": batch_id},
                    )
                self.connection.execute(
                    """
                    INSERT INTO batches(
                        batch_id, config_version, ledger_format_version,
                        start_cursor, end_cursor, content_digest,
                        payload_json, committed_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        int(payload_copy["config_version"]),
                        fmt_version,
                        int(payload_copy["start_cursor"]),
                        int(payload_copy["end_cursor"]),
                        digest,
                        payload_json,
                        time.time_ns(),
                    ),
                )
                for request in sorted(payload_copy["requests"], key=lambda item: item["ordinal"]):
                    self.connection.execute(
                        """
                        INSERT INTO request_ledger(
                            batch_id, ordinal, request_id, destination, pool_id,
                            connection_id, reuse_sequence, borrow_id, lender_pool,
                            status, error_code, watermark_json, document_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            batch_id,
                            int(request["ordinal"]),
                            request["request_id"],
                            request.get("main_destination"),
                            request.get("main_pool_id"),
                            request.get("connection_id"),
                            request.get("reuse_sequence"),
                            request.get("borrow_id"),
                            request.get("lender_pool"),
                            request["main_status"],
                            request.get("main_error_code"),
                            canonical_json(request.get("watermarks", {})),
                            canonical_json(request),
                        ),
                    )
                for stat in sorted(payload_copy.get("pool_stats", []), key=lambda item: item["pool_id"]):
                    self.connection.execute(
                        "INSERT INTO pool_watermarks(batch_id, pool_id, stats_json) VALUES (?, ?, ?)",
                        (batch_id, stat["pool_id"], canonical_json(stat)),
                    )
                for borrow in sorted(
                    payload_copy.get("borrows", []), key=lambda item: item["borrow_id"]
                ):
                    self.connection.execute(
                        """
                        INSERT INTO borrow_ledger(batch_id, borrow_id, state, record_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        (batch_id, borrow["borrow_id"], borrow["state"], canonical_json(borrow)),
                    )
                for mirror in sorted(
                    payload_copy.get("mirror_records", []), key=lambda item: item["ordinal"]
                ):
                    self.connection.execute(
                        """
                        INSERT INTO mirror_ledger(batch_id, request_id, pool_id, status, record_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            batch_id,
                            mirror["request_id"],
                            mirror["pool_id"],
                            mirror["status"],
                            canonical_json(mirror),
                        ),
                    )
                for event in sorted(
                    payload_copy.get("connection_events", []),
                    key=lambda item: item["event_seq"],
                ):
                    self.connection.execute(
                        "INSERT INTO connection_events(batch_id, event_seq, event_json) VALUES (?, ?, ?)",
                        (batch_id, int(event["event_seq"]), canonical_json(event)),
                    )
                self.connection.execute(
                    """
                    INSERT INTO version_epochs(version, status, config_digest, created_at_ns)
                    VALUES (?, 'active', ?, ?)
                    ON CONFLICT(version) DO UPDATE SET
                        status = 'active',
                        config_digest = excluded.config_digest
                    """,
                    (
                        int(payload_copy["config_version"]),
                        payload_copy.get("config_digest", ""),
                        time.time_ns(),
                    ),
                )
                self.connection.execute(
                    "DELETE FROM inflight_batches WHERE batch_id = ?", (batch_id,)
                )
                self.connection.execute(
                    "UPDATE interrupted_batches SET "
                    "start_cursor = max(start_cursor, ?), "
                    "rerun_required = CASE WHEN max(start_cursor, ?) > end_cursor THEN 0 ELSE 1 END "
                    "WHERE start_cursor <= ? AND end_cursor >= ?",
                    (
                        int(payload_copy["end_cursor"]) + 1,
                        int(payload_copy["end_cursor"]) + 1,
                        int(payload_copy["end_cursor"]),
                        int(payload_copy["start_cursor"]),
                    ),
                )
                self.connection.execute(
                    "UPDATE unknown_resources SET resolved = 1, resolved_at_ns = ? "
                    "WHERE resolved = 0 AND batch_id IN ("
                    "SELECT batch_id FROM interrupted_batches WHERE rerun_required = 0)",
                    (time.time_ns(),),
                )
                self.connection.commit()
                return digest
            except Exception:
                self.connection.rollback()
                raise

    # ------------------------------------------------------------------
    # Recovery and replay
    # ------------------------------------------------------------------
    def recover(self) -> dict[str, Any]:
        now = time.time_ns()
        with self._lock:
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                rows = self.connection.execute(
                    "SELECT * FROM inflight_batches ORDER BY start_cursor, batch_id"
                ).fetchall()
                for row in rows:
                    batch_id = row["batch_id"]
                    start_cursor = int(row["start_cursor"])
                    end_cursor = int(row["end_cursor"])
                    snapshot = json.loads(row["resource_snapshot_json"] or "{}")
                    detail = {
                        "config_version": int(row["config_version"]),
                        "start_cursor": start_cursor,
                        "end_cursor": end_cursor,
                    }
                    self.connection.execute(
                        """
                        INSERT OR REPLACE INTO interrupted_batches(
                            batch_id, config_version, start_cursor, end_cursor,
                            detected_at_ns, rerun_required, detail_json
                        ) VALUES (?, ?, ?, ?, ?, 1, ?)
                        """,
                        (
                            batch_id,
                            int(row["config_version"]),
                            start_cursor,
                            end_cursor,
                            now,
                            canonical_json(detail),
                        ),
                    )
                    resources: list[dict[str, Any]] = []
                    for conn in snapshot.get("connections", []):
                        if conn.get("state") == "closed":
                            continue
                        resources.append(
                            {
                                "resource_kind": "connection",
                                "resource_id": conn.get("connection_id"),
                                "pool_id": conn.get("pool_id"),
                                "state": "unknown",
                                "detail": conn,
                            }
                        )
                    for borrow in snapshot.get("borrows", []):
                        if borrow.get("state") != "active":
                            continue
                        resources.append(
                            {
                                "resource_kind": "borrow",
                                "resource_id": borrow.get("borrow_id"),
                                "pool_id": borrow.get("borrower_pool"),
                                "state": "unknown",
                                "detail": borrow,
                            }
                        )
                    if not resources:
                        resources.append(
                            {
                                "resource_kind": "batch",
                                "resource_id": batch_id,
                                "pool_id": None,
                                "state": "unknown",
                                "detail": detail,
                            }
                        )
                    for resource in resources:
                        self.connection.execute(
                            """
                            INSERT INTO unknown_resources(
                                batch_id, resource_kind, resource_id, pool_id,
                                state, needs_rerun, detected_at_ns, detail_json
                            ) VALUES (?, ?, ?, ?, 'unknown', 1, ?, ?)
                            """,
                            (
                                batch_id,
                                resource["resource_kind"],
                                resource["resource_id"],
                                resource["pool_id"],
                                now,
                                canonical_json(resource["detail"]),
                            ),
                        )
                    self.connection.execute(
                        "DELETE FROM inflight_batches WHERE batch_id = ?", (batch_id,)
                    )
                self.connection.commit()
            except Exception:
                self.connection.rollback()
                raise
            committed = self.list_batches()
            next_cursor = max((batch["end_cursor"] for batch in committed), default=0) + 1
            unresolved_rows = self.connection.execute(
                "SELECT * FROM unknown_resources WHERE resolved = 0 ORDER BY id"
            ).fetchall()
            pending_interruptions = self.connection.execute(
                "SELECT * FROM interrupted_batches WHERE rerun_required = 1 "
                "ORDER BY start_cursor, batch_id"
            ).fetchall()
            unresolved = [
                {
                    "batch_id": row["batch_id"],
                    "resource_kind": row["resource_kind"],
                    "resource_id": row["resource_id"],
                    "pool_id": row["pool_id"],
                    "state": row["state"],
                    "detail": json.loads(row["detail_json"]),
                }
                for row in unresolved_rows
            ]
            pending = [
                {
                    "batch_id": row["batch_id"],
                    "config_version": row["config_version"],
                    "start_cursor": row["start_cursor"],
                    "end_cursor": row["end_cursor"],
                    "rerun_required": True,
                }
                for row in pending_interruptions
            ]
            rerun_from = min(
                [next_cursor] + [int(item["start_cursor"]) for item in pending],
            )
            return {
                "committed_batches": committed,
                "interrupted_batches": pending,
                "unknown_connections": [
                    item for item in unresolved if item["resource_kind"] == "connection"
                ],
                "unknown_borrows": [
                    item for item in unresolved if item["resource_kind"] == "borrow"
                ],
                "unknown_resources": unresolved,
                "next_cursor": next_cursor,
                "rerun_from_cursor": rerun_from,
                "resume_cursor": rerun_from,
                "exit_report_available": not unresolved and not pending,
                "evidence": "only rows present in committed batches may be replayed",
            }

    def replay_self_check(self) -> dict[str, Any]:
        batches = self.list_batches()
        if not batches:
            return {
                "status": "ok",
                "batches_checked": 0,
                "requests_checked": 0,
                "first_request_id": None,
                "next_cursor": 1,
            }
        expected_cursor = 1
        previous_end = 0
        checked_requests = 0
        first_request_id: str | None = None
        for batch_row in batches:
            batch_id = batch_row["batch_id"]
            payload = self.get_batch_payload(batch_id)
            actual_digest = content_digest(payload)
            if actual_digest != batch_row["content_digest"]:
                first = (payload.get("requests") or [{}])[0]
                raise BusinessError(
                    "E_SNAPSHOT",
                    "ledger content digest mismatch",
                    field="content_digest",
                    request_id=first.get("request_id"),
                    pool_id=None,
                    phase="replay",
                    details={
                        "batch_id": batch_id,
                        "expected": batch_row["content_digest"],
                        "actual": actual_digest,
                        "basis": "sha256(canonical_payload)",
                    },
                )
            start_cursor = int(payload["start_cursor"])
            end_cursor = int(payload["end_cursor"])
            if start_cursor != previous_end + 1:
                raise ReplayMismatch(
                    request_id=None,
                    field="start_cursor",
                    expected=previous_end + 1,
                    actual=start_cursor,
                    batch_id=batch_id,
                    basis="committed batches must be cursor-contiguous in replay order",
                )
            if end_cursor < start_cursor:
                raise ReplayMismatch(
                    request_id=None,
                    field="end_cursor",
                    expected=f">= {start_cursor}",
                    actual=end_cursor,
                    batch_id=batch_id,
                    basis="batch cursor range must be non-empty",
                )
            previous_end = end_cursor
            expected_cursor = end_cursor + 1
            payload_requests = sorted(
                payload.get("requests", []), key=lambda item: item["ordinal"]
            )
            row_requests = self.connection.execute(
                "SELECT * FROM request_ledger WHERE batch_id = ? ORDER BY ordinal",
                (batch_id,),
            ).fetchall()
            if len(payload_requests) != len(row_requests):
                raise ReplayMismatch(
                    request_id=payload_requests[0].get("request_id") if payload_requests else None,
                    field="request_count",
                    expected=len(payload_requests),
                    actual=len(row_requests),
                    batch_id=batch_id,
                    basis="request ledger row count must match the signed payload",
                )
            for expected, row in zip(payload_requests, row_requests):
                if first_request_id is None:
                    first_request_id = expected["request_id"]
                checked_requests += 1
                expected_fields = {
                    "request_id": expected["request_id"],
                    "destination": expected.get("main_destination"),
                    "pool_id": expected.get("main_pool_id"),
                    "connection_id": expected.get("connection_id"),
                    "reuse_sequence": expected.get("reuse_sequence"),
                    "borrow_id": expected.get("borrow_id"),
                    "lender_pool": expected.get("lender_pool"),
                    "status": expected["main_status"],
                    "error_code": expected.get("main_error_code"),
                    "watermark_json": canonical_json(expected.get("watermarks", {})),
                }
                actual_fields = {
                    "request_id": row["request_id"],
                    "destination": row["destination"],
                    "pool_id": row["pool_id"],
                    "connection_id": row["connection_id"],
                    "reuse_sequence": row["reuse_sequence"],
                    "borrow_id": row["borrow_id"],
                    "lender_pool": row["lender_pool"],
                    "status": row["status"],
                    "error_code": row["error_code"],
                    "watermark_json": row["watermark_json"],
                }
                for field in expected_fields:
                    if expected_fields[field] != actual_fields[field]:
                        raise ReplayMismatch(
                            request_id=expected["request_id"],
                            field=field,
                            expected=expected_fields[field],
                            actual=actual_fields[field],
                            batch_id=batch_id,
                            basis="row and signed payload disagree; first mismatch by ordinal",
                        )
                stored_doc = json.loads(row["document_json"])
                if stored_doc != expected:
                    for field in sorted(expected):
                        if stored_doc.get(field) != expected.get(field):
                            raise ReplayMismatch(
                                request_id=expected["request_id"],
                                field=field,
                                expected=expected.get(field),
                                actual=stored_doc.get(field),
                                batch_id=batch_id,
                                basis="per-request signed document mismatch",
                            )
            for stat in payload.get("pool_stats", []):
                row = self.connection.execute(
                    "SELECT stats_json FROM pool_watermarks WHERE batch_id = ? AND pool_id = ?",
                    (batch_id, stat["pool_id"]),
                ).fetchone()
                if row is None or json.loads(row["stats_json"]) != stat:
                    raise ReplayMismatch(
                        request_id=first_request_id,
                        field=f"pool_stats.{stat['pool_id']}",
                        expected=stat,
                        actual=None if row is None else json.loads(row["stats_json"]),
                        batch_id=batch_id,
                        basis="pool watermark row mismatch",
                    )
            payload_borrows = sorted(
                payload.get("borrows", []), key=lambda item: item["borrow_id"]
            )
            borrow_rows = self.connection.execute(
                "SELECT * FROM borrow_ledger WHERE batch_id = ? ORDER BY borrow_id",
                (batch_id,),
            ).fetchall()
            if len(payload_borrows) != len(borrow_rows):
                raise ReplayMismatch(
                    request_id=first_request_id,
                    field="borrow_count",
                    expected=len(payload_borrows),
                    actual=len(borrow_rows),
                    batch_id=batch_id,
                    basis="borrow ledger row count must match the signed payload",
                )
            for expected, row in zip(payload_borrows, borrow_rows):
                actual_record = json.loads(row["record_json"])
                for field, expected_value, actual_value in (
                    ("borrow_id", expected["borrow_id"], row["borrow_id"]),
                    ("state", expected["state"], row["state"]),
                    ("record", expected, actual_record),
                ):
                    if expected_value != actual_value:
                        raise ReplayMismatch(
                            request_id=expected.get("request_id"),
                            field=f"borrows.{field}",
                            expected=expected_value,
                            actual=actual_value,
                            batch_id=batch_id,
                            basis="borrow ownership row mismatch",
                        )
            payload_mirrors = sorted(
                payload.get("mirror_records", []), key=lambda item: item["ordinal"]
            )
            mirror_rows = self.connection.execute(
                "SELECT * FROM mirror_ledger WHERE batch_id = ? ORDER BY record_json",
                (batch_id,),
            ).fetchall()
            mirror_rows_by_request = {row["request_id"]: row for row in mirror_rows}
            if len(payload_mirrors) != len(mirror_rows):
                raise ReplayMismatch(
                    request_id=first_request_id,
                    field="mirror_record_count",
                    expected=len(payload_mirrors),
                    actual=len(mirror_rows),
                    batch_id=batch_id,
                    basis="mirror ledger row count must match the signed payload",
                )
            for expected in payload_mirrors:
                row = mirror_rows_by_request.get(expected["request_id"])
                if row is None:
                    raise ReplayMismatch(
                        request_id=expected["request_id"],
                        field="mirror_record",
                        expected=expected,
                        actual=None,
                        batch_id=batch_id,
                        basis="missing independent mirror record",
                    )
                actual_record = json.loads(row["record_json"])
                for field, expected_value, actual_value in (
                    ("pool_id", expected["pool_id"], row["pool_id"]),
                    ("status", expected["status"], row["status"]),
                    ("record", expected, actual_record),
                ):
                    if expected_value != actual_value:
                        raise ReplayMismatch(
                            request_id=expected["request_id"],
                            field=f"mirror.{field}",
                            expected=expected_value,
                            actual=actual_value,
                            batch_id=batch_id,
                            basis="independent mirror record mismatch",
                        )
            payload_events = sorted(
                payload.get("connection_events", []), key=lambda item: item["event_seq"]
            )
            event_rows = self.connection.execute(
                "SELECT * FROM connection_events WHERE batch_id = ? ORDER BY event_seq",
                (batch_id,),
            ).fetchall()
            if len(payload_events) != len(event_rows):
                raise ReplayMismatch(
                    request_id=first_request_id,
                    field="connection_event_count",
                    expected=len(payload_events),
                    actual=len(event_rows),
                    batch_id=batch_id,
                    basis="connection lifecycle row count mismatch",
                )
            for expected, row in zip(payload_events, event_rows):
                actual_event = json.loads(row["event_json"])
                if expected != actual_event or int(row["event_seq"]) != int(expected["event_seq"]):
                    raise ReplayMismatch(
                        request_id=first_request_id,
                        field=f"connection_events.{expected['event_seq']}",
                        expected=expected,
                        actual=actual_event,
                        batch_id=batch_id,
                        basis="connection lifecycle replay mismatch",
                    )
        return {
            "status": "ok",
            "batches_checked": len(batches),
            "requests_checked": checked_requests,
            "first_request_id": first_request_id,
            "next_cursor": expected_cursor,
        }

    def verify_batch(
        self,
        batch_id: str,
        *,
        force_digest_mismatch: bool = False,
    ) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            raise BusinessError(
                "E_REPLAY",
                "batch not found in committed ledger",
                field="batch_id",
                request_id=None,
                pool_id=None,
                phase="replay",
                details={"batch_id": batch_id},
            )
        payload = self.get_batch_payload(batch_id)
        actual = content_digest(payload)
        if force_digest_mismatch or actual != row["content_digest"]:
            first = (payload.get("requests") or [{}])[0]
            raise BusinessError(
                "E_SNAPSHOT",
                "ledger content digest mismatch",
                field="content_digest",
                request_id=first.get("request_id"),
                pool_id=None,
                phase="replay",
                details={
                    "batch_id": batch_id,
                    "expected": row["content_digest"],
                    "actual": actual,
                    "forced": force_digest_mismatch,
                    "basis": "sha256(canonical_payload)",
                },
            )
        return {"batch_id": batch_id, "content_digest": actual, "status": "ok"}

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------
    def has_batch(self, batch_id: str) -> bool:
        with self._lock:
            row = self.connection.execute(
                "SELECT 1 FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            return row is not None

    def is_empty(self) -> bool:
        with self._lock:
            row = self.connection.execute("SELECT COUNT(*) AS n FROM batches").fetchone()
            return int(row["n"]) == 0

    def list_batches(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT b.*, COALESCE(e.status, 'unknown') AS epoch_status
                FROM batches b
                LEFT JOIN version_epochs e ON e.version = b.config_version
                ORDER BY b.start_cursor, b.batch_id
                """
            ).fetchall()
            return [
                {
                    "batch_id": row["batch_id"],
                    "config_version": row["config_version"],
                    "ledger_format_version": row["ledger_format_version"],
                    "start_cursor": row["start_cursor"],
                    "end_cursor": row["end_cursor"],
                    "content_digest": row["content_digest"],
                    "committed_at_ns": row["committed_at_ns"],
                    "epoch_status": row["epoch_status"],
                    "stale": row["epoch_status"] == "stale",
                }
                for row in rows
            ]

    def get_batch_payload(self, batch_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT payload_json FROM batches WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            if row is None:
                raise BusinessError(
                    "E_REPLAY",
                    "batch not found",
                    field="batch_id",
                    request_id=None,
                    pool_id=None,
                    phase="replay",
                    details={"batch_id": batch_id},
                )
            return json.loads(row["payload_json"])

    def load_requests(self, *, current_version_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            if current_version_only:
                rows = self.connection.execute(
                    """
                    SELECT r.document_json
                    FROM request_ledger r
                    JOIN version_epochs e ON e.version = (
                        SELECT b.config_version FROM batches b WHERE b.batch_id = r.batch_id
                    )
                    WHERE e.status = 'active'
                    ORDER BY r.batch_id, r.ordinal
                    """
                ).fetchall()
            else:
                rows = self.connection.execute(
                    "SELECT document_json FROM request_ledger ORDER BY batch_id, ordinal"
                ).fetchall()
            return [json.loads(row["document_json"]) for row in rows]

    def ledger_snapshot(self) -> dict[str, Any]:
        meta = {
            row["key"]: row["value"]
            for row in self.connection.execute(
                "SELECT key, value FROM ledger_meta ORDER BY key"
            ).fetchall()
        }
        epochs = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM version_epochs ORDER BY version"
            ).fetchall()
        ]
        interrupted = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM interrupted_batches ORDER BY start_cursor, batch_id"
            ).fetchall()
        ]
        unknown = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM unknown_resources ORDER BY id"
            ).fetchall()
        ]
        return {
            "path": str(self.path),
            "meta": meta,
            "batch_list": self.list_batches(),
            "epochs": epochs,
            "interrupted_batches": interrupted,
            "unknown_resources": unknown,
        }

    def close(self) -> None:
        with self._lock:
            try:
                self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass
            self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
