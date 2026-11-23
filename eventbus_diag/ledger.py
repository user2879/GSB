"""Durable SQLite ledger, state machine, and crash-recovery verification."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .canonical import digest_value, stable_serialize, wall_ms
from .errors import BusDiagnosticError, LeaseError, ReplayError, SpecError
from .logging_utils import StructuredLogger
from .models import Ack, Acquisition, Declaration, Lease, PlanBatch
from .worker import compute_ack


LEDGER_FORMAT_VERSION = 1
QUANTIZATION_VERSION = "eventbus-quant-v1"

DISPATCHING = "DISPATCHING"
CONFIRMED = "CONFIRMED"
FAILED = "FAILED"
UNKNOWN = "UNKNOWN"

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "NONE": frozenset({DISPATCHING, FAILED, UNKNOWN}),
    DISPATCHING: frozenset({CONFIRMED, FAILED, UNKNOWN}),
    FAILED: frozenset({DISPATCHING}),
    UNKNOWN: frozenset({DISPATCHING, FAILED}),
    CONFIRMED: frozenset(),
}


@dataclass(frozen=True)
class PreparedRun:
    run_id: str
    declaration_version: int
    resumed: bool
    previous_version: int | None
    progress_cursor: int
    replacement: dict[str, Any] | None


@dataclass(frozen=True)
class VerificationResult:
    verified_batches: tuple[str, ...]
    unknown_batches: tuple[str, ...]
    remaining_batches: tuple[str, ...]
    first_mismatch: str | None = None


@dataclass(frozen=True)
class CompletionResult:
    state: str
    progress_cursor: int
    ledger_digest: str


class Ledger:
    def __init__(self, db_path: str | Path, logger: StructuredLogger | None = None) -> None:
        self.db_path = str(db_path)
        self.logger = logger
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        if self.db_path != ":memory:":
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = FULL")
        self._initialize_schema()

    def close(self) -> None:
        with self._lock:
            try:
                self.connection.commit()
            except sqlite3.Error:
                pass
            self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except Exception:
                self.connection.rollback()
                raise
            else:
                self.connection.commit()

    def _initialize_schema(self) -> None:
        with self._lock:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_info (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    updated_wall_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    declaration_version INTEGER NOT NULL,
                    spec_digest TEXT NOT NULL,
                    events_digest TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    quantization_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    progress_cursor INTEGER NOT NULL DEFAULT 0,
                    in_flight_count INTEGER NOT NULL DEFAULT 0,
                    replacement_json TEXT,
                    created_wall_ms INTEGER NOT NULL,
                    updated_wall_ms INTEGER NOT NULL,
                    UNIQUE(declaration_version)
                );

                CREATE TABLE IF NOT EXISTS batches (
                    run_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    declaration_version INTEGER NOT NULL,
                    subscriber_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    batch_index INTEGER NOT NULL,
                    event_ids_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    quantization_version TEXT NOT NULL,
                    lease_id TEXT,
                    lease_version INTEGER,
                    state TEXT NOT NULL,
                    wait_start_mono_ms REAL,
                    wait_end_mono_ms REAL,
                    wait_ms REAL,
                    wait_reason TEXT,
                    quota_json TEXT,
                    ack_summary_json TEXT,
                    ack_digest TEXT,
                    processing_ms INTEGER,
                    failure_code TEXT,
                    failure_message TEXT,
                    progress_cursor INTEGER NOT NULL DEFAULT 0,
                    stale INTEGER NOT NULL DEFAULT 0,
                    updated_wall_ms INTEGER NOT NULL,
                    PRIMARY KEY (run_id, batch_id)
                );
                CREATE INDEX IF NOT EXISTS idx_batches_run_ordinal
                    ON batches(run_id, ordinal);

                CREATE TABLE IF NOT EXISTS transitions (
                    run_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    at_mono_ms REAL NOT NULL,
                    at_wall_ms INTEGER NOT NULL,
                    reason TEXT,
                    stale INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, batch_id, seq)
                );

                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    subscriber_id TEXT NOT NULL,
                    channel_id TEXT NOT NULL,
                    lease_version INTEGER NOT NULL,
                    granted_mono_ms REAL NOT NULL,
                    expires_mono_ms REAL NOT NULL,
                    revoked_mono_ms REAL,
                    state TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    updated_wall_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_leases_run_batch
                    ON leases(run_id, batch_id);

                CREATE TABLE IF NOT EXISTS quota_usage (
                    run_id TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    global_limit INTEGER NOT NULL,
                    subscriber_limit INTEGER NOT NULL,
                    channel_rate_limit INTEGER NOT NULL,
                    global_active_after_grant INTEGER NOT NULL,
                    subscriber_active_after_grant INTEGER NOT NULL,
                    channel_sends_in_window INTEGER NOT NULL,
                    recorded_mono_ms REAL NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, batch_id)
                );

                CREATE TABLE IF NOT EXISTS errors (
                    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    batch_id TEXT,
                    code TEXT NOT NULL,
                    field TEXT,
                    source_line INTEGER,
                    message TEXT NOT NULL,
                    payload_json TEXT,
                    at_wall_ms INTEGER NOT NULL,
                    at_mono_ms REAL NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_errors_run
                    ON errors(run_id, error_id);

                CREATE TABLE IF NOT EXISTS replay_checks (
                    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    batch_id TEXT,
                    mismatch_field TEXT,
                    expected_json TEXT,
                    actual_json TEXT,
                    matched INTEGER NOT NULL,
                    at_wall_ms INTEGER NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            row = self.connection.execute(
                "SELECT format_version FROM ledger_info WHERE id = 1"
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO ledger_info(id, format_version, content_digest, updated_wall_ms) VALUES(1, ?, ?, ?)",
                    (LEDGER_FORMAT_VERSION, digest_value({"format": LEDGER_FORMAT_VERSION}), wall_ms()),
                )
            elif int(row["format_version"]) != LEDGER_FORMAT_VERSION:
                raise BusDiagnosticError(
                    "E_LEDGER",
                    f"unsupported ledger format version {row['format_version']}",
                    details={
                        "expected": LEDGER_FORMAT_VERSION,
                        "actual": int(row["format_version"]),
                    },
                )

    def latest_run(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM runs ORDER BY declaration_version DESC, created_wall_ms DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def current_run(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM runs WHERE stale = 0 ORDER BY declaration_version DESC, created_wall_ms DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def prepare_run(
        self,
        declaration: Declaration,
        *,
        spec_digest: str,
        events_digest: str,
        content_digest: str,
        replacement: dict[str, Any] | None = None,
    ) -> PreparedRun:
        run_id = f"run_v{declaration.declaration_version}_{content_digest[:16]}"
        now_wall = wall_ms()
        now_mono = self._mono_ms()
        with self._transaction() as conn:
            latest = conn.execute(
                "SELECT * FROM runs ORDER BY declaration_version DESC, created_wall_ms DESC LIMIT 1"
            ).fetchone()
            if latest is not None:
                latest_version = int(latest["declaration_version"])
                if declaration.declaration_version < latest_version:
                    raise SpecError(
                        f"declaration_version must increase; latest is {latest_version}",
                        field="declaration_version",
                    )
                if declaration.declaration_version == latest_version:
                    if (
                        str(latest["spec_digest"]) != spec_digest
                        or str(latest["events_digest"]) != events_digest
                    ):
                        raise SpecError(
                            "declaration_version was reused with different declaration or event content",
                            field="declaration_version",
                            details={
                                "existing_version": latest_version,
                                "existing_spec_digest": latest["spec_digest"],
                                "new_spec_digest": spec_digest,
                                "existing_events_digest": latest["events_digest"],
                                "new_events_digest": events_digest,
                            },
                        )
                    return PreparedRun(
                        run_id=str(latest["run_id"]),
                        declaration_version=latest_version,
                        resumed=True,
                        previous_version=latest_version - 1,
                        progress_cursor=int(latest["progress_cursor"]),
                        replacement=(
                            json.loads(latest["replacement_json"])
                            if latest["replacement_json"]
                            else None
                        ),
                    )

                previous_version = latest_version
                conn.execute(
                    "UPDATE runs SET stale = 1, updated_wall_ms = ? WHERE stale = 0",
                    (now_wall,),
                )
                conn.execute(
                    "UPDATE batches SET stale = 1 WHERE run_id IN (SELECT run_id FROM runs WHERE stale = 1)"
                )
                conn.execute(
                    "UPDATE transitions SET stale = 1 WHERE run_id IN (SELECT run_id FROM runs WHERE stale = 1)"
                )
                conn.execute(
                    "UPDATE quota_usage SET stale = 1 WHERE run_id IN (SELECT run_id FROM runs WHERE stale = 1)"
                )
                conn.execute(
                    "UPDATE errors SET stale = 1 WHERE run_id IN (SELECT run_id FROM runs WHERE stale = 1)"
                )
                conn.execute(
                    "UPDATE replay_checks SET stale = 1 WHERE run_id IN (SELECT run_id FROM runs WHERE stale = 1)"
                )
                conn.execute(
                    "UPDATE leases SET stale = 1, state = 'REVOKED', revoked_mono_ms = COALESCE(revoked_mono_ms, ?) WHERE run_id IN (SELECT run_id FROM runs WHERE stale = 1)",
                    (now_mono,),
                )
            else:
                previous_version = None

            replacement_json = stable_serialize(replacement) if replacement is not None else None
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, declaration_version, spec_digest, events_digest,
                    content_digest, quantization_version, status, stale,
                    progress_cursor, in_flight_count, replacement_json,
                    created_wall_ms, updated_wall_ms
                ) VALUES(?, ?, ?, ?, ?, ?, 'RUNNING', 0, 0, 0, ?, ?, ?)
                """,
                (
                    run_id,
                    declaration.declaration_version,
                    spec_digest,
                    events_digest,
                    content_digest,
                    QUANTIZATION_VERSION,
                    replacement_json,
                    now_wall,
                    now_wall,
                ),
            )
            self._update_ledger_info_locked(conn, content_digest)
            return PreparedRun(
                run_id=run_id,
                declaration_version=declaration.declaration_version,
                resumed=False,
                previous_version=previous_version,
                progress_cursor=0,
                replacement=replacement,
            )

    def begin_batch(
        self,
        run_id: str,
        batch: PlanBatch,
        acquisition: Acquisition,
        *,
        reason: str = "lease_granted",
    ) -> None:
        lease = acquisition.lease
        wait = acquisition.wait
        quota = acquisition.quota
        now_wall = wall_ms()
        now_mono = self._mono_ms()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT state FROM batches WHERE run_id = ? AND batch_id = ?",
                (run_id, batch.batch_id),
            ).fetchone()
            previous_state = str(row["state"]) if row else "NONE"
            self._assert_transition(previous_state, DISPATCHING, batch.batch_id)
            conn.execute(
                """
                INSERT INTO batches(
                    run_id, batch_id, ordinal, declaration_version, subscriber_id,
                    channel_id, topic, batch_index, event_ids_json, content_digest,
                    quantization_version, lease_id, lease_version, state, wait_start_mono_ms,
                    wait_end_mono_ms, wait_ms, wait_reason, quota_json,
                    ack_summary_json, ack_digest, processing_ms, failure_code,
                    failure_message, progress_cursor, stale, updated_wall_ms
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, 0, 0, ?)
                ON CONFLICT(run_id, batch_id) DO UPDATE SET
                    state = excluded.state,
                    quantization_version = excluded.quantization_version,
                    lease_id = excluded.lease_id,
                    lease_version = excluded.lease_version,
                    wait_start_mono_ms = excluded.wait_start_mono_ms,
                    wait_end_mono_ms = excluded.wait_end_mono_ms,
                    wait_ms = excluded.wait_ms,
                    wait_reason = excluded.wait_reason,
                    quota_json = excluded.quota_json,
                    ack_summary_json = NULL,
                    ack_digest = NULL,
                    processing_ms = NULL,
                    failure_code = NULL,
                    failure_message = NULL,
                    stale = 0,
                    updated_wall_ms = excluded.updated_wall_ms
                """,
                (
                    run_id,
                    batch.batch_id,
                    batch.ordinal,
                    self._declaration_version_for_run_locked(conn, run_id),
                    batch.subscriber_id,
                    batch.channel_id,
                    batch.topic,
                    batch.batch_index,
                    stable_serialize(list(batch.event_ids)),
                    batch.content_digest,
                    QUANTIZATION_VERSION,
                    lease.lease_id,
                    lease.version,
                    DISPATCHING,
                    wait.start_mono_ms,
                    wait.end_mono_ms,
                    wait.wait_ms,
                    wait.reason,
                    stable_serialize(quota.as_dict()),
                    now_wall,
                ),
            )
            conn.execute(
                """
                INSERT INTO leases(
                    lease_id, run_id, batch_id, subscriber_id, channel_id,
                    lease_version, granted_mono_ms, expires_mono_ms,
                    revoked_mono_ms, state, stale, updated_wall_ms
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, NULL, 'ACTIVE', 0, ?)
                ON CONFLICT(lease_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    batch_id = excluded.batch_id,
                    subscriber_id = excluded.subscriber_id,
                    channel_id = excluded.channel_id,
                    lease_version = excluded.lease_version,
                    granted_mono_ms = excluded.granted_mono_ms,
                    expires_mono_ms = excluded.expires_mono_ms,
                    revoked_mono_ms = NULL,
                    state = 'ACTIVE',
                    stale = 0,
                    updated_wall_ms = excluded.updated_wall_ms
                """,
                (
                    lease.lease_id,
                    run_id,
                    batch.batch_id,
                    lease.subscriber_id,
                    lease.channel_id,
                    lease.version,
                    lease.granted_mono_ms,
                    lease.expires_mono_ms,
                    now_wall,
                ),
            )
            conn.execute(
                """
                INSERT INTO quota_usage(
                    run_id, batch_id, global_limit, subscriber_limit,
                    channel_rate_limit, global_active_after_grant,
                    subscriber_active_after_grant, channel_sends_in_window,
                    recorded_mono_ms, stale
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(run_id, batch_id) DO UPDATE SET
                    global_limit = excluded.global_limit,
                    subscriber_limit = excluded.subscriber_limit,
                    channel_rate_limit = excluded.channel_rate_limit,
                    global_active_after_grant = excluded.global_active_after_grant,
                    subscriber_active_after_grant = excluded.subscriber_active_after_grant,
                    channel_sends_in_window = excluded.channel_sends_in_window,
                    recorded_mono_ms = excluded.recorded_mono_ms,
                    stale = 0
                """,
                (
                    run_id,
                    batch.batch_id,
                    quota.global_limit,
                    quota.subscriber_limit,
                    quota.channel_rate_limit,
                    quota.global_active_after_grant,
                    quota.subscriber_active_after_grant,
                    quota.channel_sends_in_window,
                    lease.granted_mono_ms,
                ),
            )
            self._insert_transition_locked(
                conn,
                run_id=run_id,
                batch_id=batch.batch_id,
                from_state=previous_state,
                to_state=DISPATCHING,
                at_mono_ms=now_mono,
                at_wall_ms=now_wall,
                reason=reason,
            )
            self._refresh_in_flight_locked(conn, run_id)

    def complete_batch(
        self,
        run_id: str,
        batch: PlanBatch,
        *,
        state: str,
        lease: Lease,
        acquisition: Acquisition,
        ack: Ack | None = None,
        error: BusDiagnosticError | None = None,
    ) -> CompletionResult:
        if state not in (CONFIRMED, FAILED):
            raise ValueError(f"invalid completion state {state!r}")
        now_wall = wall_ms()
        now_mono = self._mono_ms()
        ack_summary = ack.summary() if ack else None
        ack_json = stable_serialize(ack_summary) if ack_summary is not None else None
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT state FROM batches WHERE run_id = ? AND batch_id = ?",
                (run_id, batch.batch_id),
            ).fetchone()
            if row is None:
                raise LeaseError(
                    "cannot complete a batch that was never granted",
                    lease_id=lease.lease_id,
                    batch_id=batch.batch_id,
                    subscriber_id=lease.subscriber_id,
                )
            previous_state = str(row["state"])
            self._assert_transition(previous_state, state, batch.batch_id)
            progress_cursor = self._compute_progress_cursor_locked(conn, run_id, batch.ordinal, state)
            conn.execute(
                """
                UPDATE batches SET
                    state = ?,
                    lease_version = ?,
                    wait_start_mono_ms = ?,
                    wait_end_mono_ms = ?,
                    wait_ms = ?,
                    wait_reason = ?,
                    quota_json = ?,
                    ack_summary_json = ?,
                    ack_digest = ?,
                    processing_ms = ?,
                    failure_code = ?,
                    failure_message = ?,
                    progress_cursor = ?,
                    stale = 0,
                    updated_wall_ms = ?
                WHERE run_id = ? AND batch_id = ?
                """,
                (
                    state,
                    lease.version,
                    acquisition.wait.start_mono_ms,
                    acquisition.wait.end_mono_ms,
                    acquisition.wait.wait_ms,
                    acquisition.wait.reason,
                    stable_serialize(acquisition.quota.as_dict()),
                    ack_json,
                    ack.ack_digest if ack else None,
                    ack.processing_ms if ack else None,
                    error.code if error else None,
                    error.message if error else None,
                    progress_cursor,
                    now_wall,
                    run_id,
                    batch.batch_id,
                ),
            )
            lease_end_state = "EXPIRED" if error is not None and error.code == "E_LEASE" else "RELEASED"
            conn.execute(
                """
                UPDATE leases SET
                    lease_version = ?,
                    expires_mono_ms = ?,
                    revoked_mono_ms = ?,
                    state = ?,
                    stale = 0,
                    updated_wall_ms = ?
                WHERE lease_id = ?
                """,
                (
                    lease.version,
                    lease.expires_mono_ms,
                    now_mono,
                    lease_end_state,
                    now_wall,
                    lease.lease_id,
                ),
            )
            self._insert_transition_locked(
                conn,
                run_id=run_id,
                batch_id=batch.batch_id,
                from_state=DISPATCHING,
                to_state=state,
                at_mono_ms=now_mono,
                at_wall_ms=now_wall,
                reason="ack_received" if state == CONFIRMED else (error.code if error else "failure"),
            )
            if error is not None:
                self._insert_error_locked(
                    conn,
                    run_id=run_id,
                    batch_id=batch.batch_id,
                    error=error,
                    now_wall=now_wall,
                    now_mono=now_mono,
                )
            conn.execute(
                "UPDATE runs SET progress_cursor = ?, status = 'RUNNING', updated_wall_ms = ? WHERE run_id = ?",
                (progress_cursor, now_wall, run_id),
            )
            self._refresh_in_flight_locked(conn, run_id)
            ledger_digest = self._update_ledger_info_locked(
                conn, self._run_content_digest_locked(conn, run_id)
            )
            return CompletionResult(
                state=state,
                progress_cursor=progress_cursor,
                ledger_digest=ledger_digest,
            )

    def record_unsent_failure(
        self,
        run_id: str,
        batch: PlanBatch,
        error: BusDiagnosticError,
    ) -> CompletionResult:
        """Record a plan batch that could not be dispatched at all."""

        now_wall = wall_ms()
        now_mono = self._mono_ms()
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT state FROM batches WHERE run_id = ? AND batch_id = ?",
                (run_id, batch.batch_id),
            ).fetchone()
            previous_state = str(row["state"]) if row else "NONE"
            if previous_state != FAILED:
                self._assert_transition(previous_state, FAILED, batch.batch_id)
            conn.execute(
                """
                INSERT INTO batches(
                    run_id, batch_id, ordinal, declaration_version, subscriber_id,
                    channel_id, topic, batch_index, event_ids_json, content_digest,
                    quantization_version, lease_id, lease_version, state, failure_code, failure_message,
                    progress_cursor, stale, updated_wall_ms
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, 0, 0, ?)
                ON CONFLICT(run_id, batch_id) DO UPDATE SET
                    state = excluded.state,
                    failure_code = excluded.failure_code,
                    failure_message = excluded.failure_message,
                    stale = 0,
                    updated_wall_ms = excluded.updated_wall_ms
                """,
                (
                    run_id,
                    batch.batch_id,
                    batch.ordinal,
                    self._declaration_version_for_run_locked(conn, run_id),
                    batch.subscriber_id,
                    batch.channel_id,
                    batch.topic,
                    batch.batch_index,
                    stable_serialize(list(batch.event_ids)),
                    batch.content_digest,
                    QUANTIZATION_VERSION,
                    FAILED,
                    error.code,
                    error.message,
                    now_wall,
                ),
            )
            if previous_state != FAILED:
                self._insert_transition_locked(
                    conn,
                    run_id=run_id,
                    batch_id=batch.batch_id,
                    from_state=previous_state,
                    to_state=FAILED,
                    at_mono_ms=now_mono,
                    at_wall_ms=now_wall,
                    reason=error.code,
                )
            self._insert_error_locked(
                conn,
                run_id=run_id,
                batch_id=batch.batch_id,
                error=error,
                now_wall=now_wall,
                now_mono=now_mono,
            )
            progress_cursor = self._compute_progress_cursor_locked(conn, run_id, batch.ordinal, FAILED)
            conn.execute(
                "UPDATE runs SET progress_cursor = ?, updated_wall_ms = ? WHERE run_id = ?",
                (progress_cursor, now_wall, run_id),
            )
            self._refresh_in_flight_locked(conn, run_id)
            ledger_digest = self._update_ledger_info_locked(
                conn, self._run_content_digest_locked(conn, run_id)
            )
            return CompletionResult(FAILED, progress_cursor, ledger_digest)

    def record_error(
        self,
        error: BusDiagnosticError,
        *,
        run_id: str | None = None,
        batch_id: str | None = None,
    ) -> None:
        with self._transaction() as conn:
            self._insert_error_locked(
                conn,
                run_id=run_id,
                batch_id=batch_id,
                error=error,
                now_wall=wall_ms(),
                now_mono=self._mono_ms(),
            )

    def verify_and_mark_unknown(
        self,
        run_id: str,
        plan_batches: Iterable[PlanBatch],
    ) -> VerificationResult:
        plan = tuple(plan_batches)
        expected = {batch.batch_id: batch for batch in plan}
        now_wall = wall_ms()
        now_mono = self._mono_ms()
        verified: list[str] = []
        unknown: list[str] = []
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM batches WHERE run_id = ? ORDER BY ordinal, batch_id",
                (run_id,),
            ).fetchall()
            rows_by_id = {str(row["batch_id"]): row for row in rows}
            conn.execute(
                """
                UPDATE leases SET
                    state = CASE
                        WHEN batch_id IN (
                            SELECT batch_id FROM batches WHERE run_id = ? AND state = ?
                        ) THEN 'RELEASED'
                        ELSE 'REVOKED'
                    END,
                    revoked_mono_ms = COALESCE(revoked_mono_ms, ?),
                    updated_wall_ms = ?
                WHERE run_id = ? AND state = 'ACTIVE'
                """,
                (run_id, CONFIRMED, now_mono, now_wall, run_id),
            )
            for row in rows:
                batch_id = str(row["batch_id"])
                if batch_id not in expected:
                    raise ReplayError(
                        "ledger contains a batch not present in the deterministic plan",
                        batch_id=batch_id,
                        mismatch_field="batch_id",
                        ordinal=int(row["ordinal"]),
                    )

            for batch in plan:
                row = rows_by_id.get(batch.batch_id)
                if row is None:
                    self._insert_unknown_locked(conn, run_id, batch, now_mono, now_wall)
                    unknown.append(batch.batch_id)
                    continue
                state = str(row["state"])
                if state == CONFIRMED:
                    self._verify_confirmed_locked(conn, run_id, batch, row)
                    verified.append(batch.batch_id)
                elif state == DISPATCHING:
                    self._verify_event_identity_locked(batch, row)
                    conn.execute(
                        "UPDATE batches SET state = ?, failure_code = ?, failure_message = ?, updated_wall_ms = ? WHERE run_id = ? AND batch_id = ?",
                        (
                            UNKNOWN,
                            "E_RECOVERY",
                            "uncommitted dispatch found after forced termination; rerun required",
                            now_wall,
                            run_id,
                            batch.batch_id,
                        ),
                    )
                    self._insert_transition_locked(
                        conn,
                        run_id=run_id,
                        batch_id=batch.batch_id,
                        from_state=DISPATCHING,
                        to_state=UNKNOWN,
                        at_mono_ms=now_mono,
                        at_wall_ms=now_wall,
                        reason="E_RECOVERY",
                    )
                    unknown.append(batch.batch_id)
                elif state in (UNKNOWN, FAILED):
                    self._verify_event_identity_locked(batch, row)
                else:
                    raise ReplayError(
                        f"unsupported ledger batch state {state!r}",
                        batch_id=batch.batch_id,
                        mismatch_field="state",
                        ordinal=batch.ordinal,
                    )

            remaining_rows = conn.execute(
                "SELECT batch_id FROM batches WHERE run_id = ? AND state != ? ORDER BY ordinal, batch_id",
                (run_id, CONFIRMED),
            ).fetchall()
            remaining = tuple(str(row["batch_id"]) for row in remaining_rows)
            progress = self._compute_contiguous_cursor_locked(conn, run_id)
            conn.execute(
                "UPDATE runs SET progress_cursor = ?, updated_wall_ms = ? WHERE run_id = ?",
                (progress, now_wall, run_id),
            )
            self._refresh_in_flight_locked(conn, run_id)
            self._update_ledger_info_locked(
                conn, self._run_content_digest_locked(conn, run_id)
            )
        if self.logger:
            self.logger.info(
                "LEDGER_RECOVERY_SCAN",
                "ledger recovery scan completed",
                run_id=run_id,
                verified_batches=len(verified),
                unknown_batches=len(unknown),
                remaining_batches=len(remaining),
            )
        return VerificationResult(
            verified_batches=tuple(verified),
            unknown_batches=tuple(unknown),
            remaining_batches=remaining,
            first_mismatch=None,
        )

    def verify_committed(self, run_id: str, plan_batches: Iterable[PlanBatch]) -> VerificationResult:
        plan = tuple(plan_batches)
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM batches WHERE run_id = ? ORDER BY ordinal, batch_id",
                (run_id,),
            ).fetchall()
            rows_by_id = {str(row["batch_id"]): row for row in rows}
            expected = {batch.batch_id for batch in plan}
            for row in rows:
                if str(row["batch_id"]) not in expected:
                    raise ReplayError(
                        "ledger contains an unexpected batch",
                        batch_id=str(row["batch_id"]),
                        mismatch_field="batch_id",
                        ordinal=int(row["ordinal"]),
                    )
            verified: list[str] = []
            remaining: list[str] = []
            for batch in plan:
                row = rows_by_id.get(batch.batch_id)
                if row is None:
                    remaining.append(batch.batch_id)
                    continue
                state = str(row["state"])
                if state == CONFIRMED:
                    self._verify_confirmed_locked(self.connection, run_id, batch, row)
                    verified.append(batch.batch_id)
                else:
                    self._verify_event_identity_locked(batch, row)
                    remaining.append(batch.batch_id)
            return VerificationResult(tuple(verified), (), tuple(remaining))

    def pending_batches(self, run_id: str, plan_batches: Iterable[PlanBatch]) -> tuple[PlanBatch, ...]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT batch_id, state FROM batches WHERE run_id = ?",
                (run_id,),
            ).fetchall()
            states = {str(row["batch_id"]): str(row["state"]) for row in rows}
            return tuple(
                batch
                for batch in plan_batches
                if states.get(batch.batch_id) != CONFIRMED
            )

    def finalize_run(self, run_id: str) -> str:
        with self._transaction() as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) AS count FROM batches WHERE run_id = ? AND state != ?",
                (run_id, CONFIRMED),
            ).fetchone()["count"]
            status = "COMPLETED" if int(remaining) == 0 else "PARTIAL"
            progress = self._compute_contiguous_cursor_locked(conn, run_id)
            conn.execute(
                "UPDATE runs SET status = ?, progress_cursor = ?, updated_wall_ms = ? WHERE run_id = ?",
                (status, progress, wall_ms(), run_id),
            )
            self._update_ledger_info_locked(
                conn, self._run_content_digest_locked(conn, run_id)
            )
            return status

    def snapshot(self, run_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            if run_id is None:
                current = self.connection.execute(
                    "SELECT * FROM runs WHERE stale = 0 ORDER BY declaration_version DESC LIMIT 1"
                ).fetchone()
            else:
                current = self.connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
            if current is None:
                return {
                    "run": None,
                    "batches": [],
                    "transitions": [],
                    "leases": [],
                    "quota_usage": [],
                    "errors": [],
                    "stats": {},
                    "ledger_info": self._ledger_info_snapshot_locked(),
                }
            current_run_id = str(current["run_id"])
            batches = [
                dict(row)
                for row in self.connection.execute(
                    "SELECT * FROM batches WHERE run_id = ? AND stale = 0 ORDER BY ordinal, batch_id",
                    (current_run_id,),
                ).fetchall()
            ]
            for batch in batches:
                batch["event_ids"] = json.loads(batch.pop("event_ids_json"))
                batch["quota"] = (
                    json.loads(batch.pop("quota_json")) if batch.get("quota_json") else None
                )
                batch["ack_summary"] = (
                    json.loads(batch.pop("ack_summary_json"))
                    if batch.get("ack_summary_json")
                    else None
                )
            transitions = [
                dict(row)
                for row in self.connection.execute(
                    "SELECT * FROM transitions WHERE run_id = ? AND stale = 0 ORDER BY batch_id, seq",
                    (current_run_id,),
                ).fetchall()
            ]
            leases = [
                dict(row)
                for row in self.connection.execute(
                    "SELECT * FROM leases WHERE run_id = ? AND stale = 0 ORDER BY lease_id",
                    (current_run_id,),
                ).fetchall()
            ]
            quota_usage = [
                dict(row)
                for row in self.connection.execute(
                    "SELECT * FROM quota_usage WHERE run_id = ? AND stale = 0 ORDER BY batch_id",
                    (current_run_id,),
                ).fetchall()
            ]
            errors = [
                dict(row)
                for row in self.connection.execute(
                    "SELECT * FROM errors WHERE run_id = ? AND stale = 0 ORDER BY error_id",
                    (current_run_id,),
                ).fetchall()
            ]
            for error in errors:
                error["payload"] = (
                    json.loads(error.pop("payload_json")) if error.get("payload_json") else None
                )
            state_counts: dict[str, int] = {}
            for batch in batches:
                state_counts[batch["state"]] = state_counts.get(batch["state"], 0) + 1
            stats = {
                "batch_count": len(batches),
                "state_counts": dict(sorted(state_counts.items())),
                "confirmed_batches": state_counts.get(CONFIRMED, 0),
                "failed_batches": state_counts.get(FAILED, 0),
                "unknown_batches": state_counts.get(UNKNOWN, 0),
                "in_flight_batches": state_counts.get(DISPATCHING, 0),
                "active_leases": sum(1 for lease in leases if lease["state"] == "ACTIVE"),
                "error_count": len(errors),
                "total_wait_ms": round(
                    sum(float(batch.get("wait_ms") or 0.0) for batch in batches), 3
                ),
            }
            run = dict(current)
            if run.get("replacement_json"):
                run["replacement"] = json.loads(run["replacement_json"])
            return {
                "run": run,
                "batches": batches,
                "transitions": transitions,
                "leases": leases,
                "quota_usage": quota_usage,
                "errors": errors,
                "stats": stats,
                "ledger_info": self._ledger_info_snapshot_locked(),
            }

    def mark_lease_expired(self, run_id: str, expiry) -> None:
        with self._transaction() as conn:
            conn.execute(
                "UPDATE leases SET state = 'EXPIRED', revoked_mono_ms = ?, updated_wall_ms = ? WHERE lease_id = ?",
                (expiry.revoked_mono_ms, wall_ms(), expiry.lease_id),
            )
            self._update_ledger_info_locked(
                conn, self._run_content_digest_locked(conn, run_id)
            )

    def record_replay_check(
        self,
        run_id: str,
        *,
        batch_id: str | None,
        mismatch_field: str | None,
        expected: Any,
        actual: Any,
        matched: bool,
    ) -> None:
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO replay_checks(
                    run_id, batch_id, mismatch_field, expected_json,
                    actual_json, matched, at_wall_ms, stale
                ) VALUES(?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    run_id,
                    batch_id,
                    mismatch_field,
                    stable_serialize(expected),
                    stable_serialize(actual),
                    1 if matched else 0,
                    wall_ms(),
                ),
            )

    def _verify_confirmed_locked(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        batch: PlanBatch,
        row: sqlite3.Row,
    ) -> None:
        self._verify_event_identity_locked(batch, row)
        ack_json = row["ack_summary_json"]
        if not ack_json:
            raise ReplayError(
                "confirmed batch has no confirmation summary",
                batch_id=batch.batch_id,
                mismatch_field="ack_summary",
                ordinal=batch.ordinal,
            )
        try:
            summary = json.loads(ack_json)
        except json.JSONDecodeError as exc:
            raise ReplayError(
                "confirmed batch confirmation summary is invalid JSON",
                batch_id=batch.batch_id,
                mismatch_field="ack_summary",
                ordinal=batch.ordinal,
            ) from exc
        if not isinstance(summary, dict):
            raise ReplayError(
                "confirmed batch confirmation summary is not an object",
                batch_id=batch.batch_id,
                mismatch_field="ack_summary",
                ordinal=batch.ordinal,
            )
        required = {"subscriber_id", "batch_id", "event_ids", "processing_ms", "ack_digest"}
        missing = sorted(required - set(summary))
        if missing:
            raise ReplayError(
                "confirmed batch confirmation summary is missing fields",
                batch_id=batch.batch_id,
                mismatch_field="ack_summary",
                ordinal=batch.ordinal,
                details={"missing_fields": missing},
            )
        if summary["subscriber_id"] != batch.subscriber_id or summary["batch_id"] != batch.batch_id:
            raise ReplayError(
                "confirmation summary identity does not match the ledger batch",
                batch_id=batch.batch_id,
                mismatch_field="ack_summary",
                ordinal=batch.ordinal,
            )
        if sorted(int(value) for value in summary["event_ids"]) != list(batch.event_ids):
            raise ReplayError(
                "confirmation summary event ids do not match the ledger batch",
                batch_id=batch.batch_id,
                mismatch_field="ack_event_ids",
                ordinal=batch.ordinal,
            )
        if row["ack_digest"] != summary["ack_digest"]:
            raise ReplayError(
                "ledger ack_digest column disagrees with the confirmation summary",
                batch_id=batch.batch_id,
                mismatch_field="ack_digest",
                ordinal=batch.ordinal,
                details={
                    "ledger_ack_digest": row["ack_digest"],
                    "summary_ack_digest": summary["ack_digest"],
                },
            )
        if row["processing_ms"] != summary["processing_ms"]:
            raise ReplayError(
                "ledger processing_ms column disagrees with the confirmation summary",
                batch_id=batch.batch_id,
                mismatch_field="processing_ms",
                ordinal=batch.ordinal,
            )
        digest_basis = dict(summary)
        claimed_digest = digest_basis.pop("ack_digest")
        if digest_value(digest_basis) != claimed_digest:
            raise ReplayError(
                "confirmation summary digest is not self-consistent",
                batch_id=batch.batch_id,
                mismatch_field="ack_digest",
                ordinal=batch.ordinal,
            )
        contract = batch.content.get("processing_contract", {})
        expected_ack = compute_ack(
            subscriber_id=batch.subscriber_id,
            batch_id=batch.batch_id,
            events=batch.content["events"],
            seed=int(contract.get("seed", 0)),
            budget_ms=int(contract.get("processing_budget_ms", 1)),
        )
        if summary != expected_ack:
            raise ReplayError(
                "confirmation summary differs from deterministic acknowledgement replay",
                batch_id=batch.batch_id,
                mismatch_field="ack_summary_replay",
                ordinal=batch.ordinal,
                details={
                    "expected_processing_ms": expected_ack.get("processing_ms"),
                    "actual_processing_ms": summary.get("processing_ms"),
                    "expected_ack_digest": expected_ack.get("ack_digest"),
                    "actual_ack_digest": summary.get("ack_digest"),
                },
            )

        lease_id = row["lease_id"]
        lease_version = row["lease_version"]
        if not lease_id or lease_version is None:
            raise ReplayError(
                "confirmed batch has no lease ownership",
                batch_id=batch.batch_id,
                mismatch_field="lease",
                ordinal=batch.ordinal,
            )
        lease_row = conn.execute(
            "SELECT * FROM leases WHERE lease_id = ? AND run_id = ?",
            (lease_id, run_id),
        ).fetchone()
        if lease_row is None:
            raise ReplayError(
                "confirmed batch lease row is missing",
                batch_id=batch.batch_id,
                mismatch_field="lease",
                ordinal=batch.ordinal,
            )
        if (
            str(lease_row["batch_id"]) != batch.batch_id
            or str(lease_row["subscriber_id"]) != batch.subscriber_id
            or int(lease_row["lease_version"]) != int(lease_version)
        ):
            raise ReplayError(
                "lease ownership disagrees with the confirmed batch",
                batch_id=batch.batch_id,
                mismatch_field="lease_owner",
                ordinal=batch.ordinal,
            )

        transitions = conn.execute(
            "SELECT * FROM transitions WHERE run_id = ? AND batch_id = ? ORDER BY seq",
            (run_id, batch.batch_id),
        ).fetchall()
        if not transitions:
            raise ReplayError(
                "confirmed batch has no transition history",
                batch_id=batch.batch_id,
                mismatch_field="transitions",
                ordinal=batch.ordinal,
            )
        previous = "NONE"
        for transition in transitions:
            from_state = str(transition["from_state"])
            to_state = str(transition["to_state"])
            if from_state != previous:
                raise ReplayError(
                    "transition history is not contiguous",
                    batch_id=batch.batch_id,
                    mismatch_field="transitions",
                    ordinal=batch.ordinal,
                )
            self._assert_transition(from_state, to_state, batch.batch_id)
            previous = to_state
        if previous != CONFIRMED:
            raise ReplayError(
                "transition history does not end in CONFIRMED",
                batch_id=batch.batch_id,
                mismatch_field="transitions",
                ordinal=batch.ordinal,
            )

    def _verify_event_identity_locked(self, batch: PlanBatch, row: sqlite3.Row) -> None:
        if str(row["batch_id"]) != batch.batch_id:
            raise ReplayError(
                "ledger batch id mismatch",
                batch_id=batch.batch_id,
                mismatch_field="batch_id",
                ordinal=batch.ordinal,
            )
        if str(row["subscriber_id"]) != batch.subscriber_id:
            raise ReplayError(
                "ledger subscriber mismatch",
                batch_id=batch.batch_id,
                mismatch_field="subscriber_id",
                ordinal=batch.ordinal,
            )
        if str(row["channel_id"]) != batch.channel_id:
            raise ReplayError(
                "ledger channel mismatch",
                batch_id=batch.batch_id,
                mismatch_field="channel_id",
                ordinal=batch.ordinal,
            )
        if str(row["topic"]) != batch.topic:
            raise ReplayError(
                "ledger topic mismatch",
                batch_id=batch.batch_id,
                mismatch_field="topic",
                ordinal=batch.ordinal,
            )
        try:
            ledger_ids = [int(value) for value in json.loads(row["event_ids_json"])]
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReplayError(
                "ledger event id set is invalid",
                batch_id=batch.batch_id,
                mismatch_field="event_ids",
                ordinal=batch.ordinal,
            ) from exc
        if ledger_ids != list(batch.event_ids):
            raise ReplayError(
                "ledger event id set differs from deterministic batch content",
                batch_id=batch.batch_id,
                mismatch_field="event_ids",
                ordinal=batch.ordinal,
                details={"ledger": ledger_ids, "plan": list(batch.event_ids)},
            )
        if str(row["content_digest"]) != batch.content_digest:
            raise ReplayError(
                "ledger batch content digest differs from deterministic batch content",
                batch_id=batch.batch_id,
                mismatch_field="content_digest",
                ordinal=batch.ordinal,
            )

    def _insert_unknown_locked(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        batch: PlanBatch,
        now_mono: float,
        now_wall: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO batches(
                run_id, batch_id, ordinal, declaration_version, subscriber_id,
                channel_id, topic, batch_index, event_ids_json, content_digest,
                quantization_version, lease_id, lease_version, state, failure_code, failure_message,
                progress_cursor, stale, updated_wall_ms
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, 0, 0, ?)
            """,
            (
                run_id,
                batch.batch_id,
                batch.ordinal,
                self._declaration_version_for_run_locked(conn, run_id),
                batch.subscriber_id,
                batch.channel_id,
                batch.topic,
                batch.batch_index,
                stable_serialize(list(batch.event_ids)),
                batch.content_digest,
                QUANTIZATION_VERSION,
                UNKNOWN,
                "E_RECOVERY",
                "batch had no committed dispatch record when recovery started; rerun required",
                now_wall,
            ),
        )
        self._insert_transition_locked(
            conn,
            run_id=run_id,
            batch_id=batch.batch_id,
            from_state="NONE",
            to_state=UNKNOWN,
            at_mono_ms=now_mono,
            at_wall_ms=now_wall,
            reason="E_RECOVERY",
        )

    def _insert_transition_locked(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        batch_id: str,
        from_state: str,
        to_state: str,
        at_mono_ms: float,
        at_wall_ms: int,
        reason: str | None,
    ) -> None:
        self._assert_transition(from_state, to_state, batch_id)
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) AS seq FROM transitions WHERE run_id = ? AND batch_id = ?",
            (run_id, batch_id),
        ).fetchone()
        seq = int(row["seq"]) + 1
        conn.execute(
            """
            INSERT INTO transitions(
                run_id, batch_id, seq, from_state, to_state,
                at_mono_ms, at_wall_ms, reason, stale
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (run_id, batch_id, seq, from_state, to_state, at_mono_ms, at_wall_ms, reason),
        )

    @staticmethod
    def _assert_transition(from_state: str, to_state: str, batch_id: str) -> None:
        allowed = ALLOWED_TRANSITIONS.get(from_state, frozenset())
        if to_state not in allowed:
            raise LeaseError(
                f"illegal batch transition {from_state} -> {to_state}",
                batch_id=batch_id,
                details={"from_state": from_state, "to_state": to_state},
            )

    def _insert_error_locked(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str | None,
        batch_id: str | None,
        error: BusDiagnosticError,
        now_wall: int,
        now_mono: float,
    ) -> None:
        payload = error.as_dict()
        details = payload.pop("details", None)
        conn.execute(
            """
            INSERT INTO errors(
                run_id, batch_id, code, field, source_line, message,
                payload_json, at_wall_ms, at_mono_ms, stale
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                run_id,
                batch_id,
                error.code,
                error.field,
                error.line,
                error.message,
                stable_serialize(details) if details is not None else None,
                now_wall,
                now_mono,
            ),
        )
        if self.logger:
            self.logger.diagnostic(error, run_id=run_id, batch_id=batch_id)

    def _declaration_version_for_run_locked(self, conn: sqlite3.Connection, run_id: str) -> int:
        row = conn.execute(
            "SELECT declaration_version FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise BusDiagnosticError("E_LEDGER", f"unknown run_id {run_id!r}")
        return int(row["declaration_version"])

    def _compute_progress_cursor_locked(
        self,
        conn: sqlite3.Connection,
        run_id: str,
        ordinal: int,
        state: str,
    ) -> int:
        # A cursor advances only while every lower ordinal is terminal confirmed.
        _ = ordinal, state
        return self._compute_contiguous_cursor_locked(conn, run_id)

    def _compute_contiguous_cursor_locked(self, conn: sqlite3.Connection, run_id: str) -> int:
        rows = conn.execute(
            "SELECT ordinal, state FROM batches WHERE run_id = ? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        states = {int(row["ordinal"]): str(row["state"]) for row in rows}
        cursor = 0
        while states.get(cursor) == CONFIRMED:
            cursor += 1
        return cursor

    def _refresh_in_flight_locked(self, conn: sqlite3.Connection, run_id: str) -> None:
        conn.execute(
            """
            UPDATE runs SET in_flight_count = (
                SELECT COUNT(*) FROM batches
                WHERE run_id = ? AND state = ?
            ) WHERE run_id = ?
            """,
            (run_id, DISPATCHING, run_id),
        )

    def _run_content_digest_locked(self, conn: sqlite3.Connection, run_id: str) -> str:
        run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        batches = conn.execute(
            "SELECT batch_id, ordinal, state, event_ids_json, content_digest, quantization_version, lease_id, lease_version, ack_digest, progress_cursor FROM batches WHERE run_id = ? ORDER BY ordinal, batch_id",
            (run_id,),
        ).fetchall()
        transitions = conn.execute(
            "SELECT batch_id, seq, from_state, to_state, at_mono_ms, reason FROM transitions WHERE run_id = ? ORDER BY batch_id, seq",
            (run_id,),
        ).fetchall()
        run_content = None
        if run:
            run_content = {
                key: run[key]
                for key in (
                    "run_id",
                    "declaration_version",
                    "spec_digest",
                    "events_digest",
                    "content_digest",
                    "quantization_version",
                )
            }
        return digest_value(
            {
                "format_version": LEDGER_FORMAT_VERSION,
                "run": run_content,
                "batches": [dict(row) for row in batches],
                "transitions": [dict(row) for row in transitions],
            }
        )

    def _update_ledger_info_locked(self, conn: sqlite3.Connection, content_digest: str) -> str:
        conn.execute(
            "UPDATE ledger_info SET format_version = ?, content_digest = ?, updated_wall_ms = ? WHERE id = 1",
            (LEDGER_FORMAT_VERSION, content_digest, wall_ms()),
        )
        return content_digest

    def _ledger_info_snapshot_locked(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM ledger_info WHERE id = 1").fetchone()
        return dict(row) if row else {}

    @staticmethod
    def _mono_ms() -> float:
        import time

        return time.monotonic() * 1000.0
