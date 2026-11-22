from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_hex
from .errors import MeshError
from .logging_utils import StructuredLogger
from .model import RequestSpec

LEDGER_FORMAT_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    request_seq INTEGER NOT NULL UNIQUE,
    service TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    run_epoch INTEGER NOT NULL DEFAULT 1,
    final_outcome TEXT,
    final_error_code TEXT,
    attempts_committed INTEGER NOT NULL DEFAULT 0,
    completed_at_wall_ms INTEGER,
    registered_at_wall_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS attempt_intents (
    intent_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    run_epoch INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL,
    service TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    select_seq INTEGER NOT NULL,
    started_wall_ms INTEGER NOT NULL,
    started_mono_ms INTEGER NOT NULL,
    state TEXT NOT NULL,
    UNIQUE(request_id, run_epoch, attempt_no)
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    request_seq INTEGER NOT NULL,
    run_epoch INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL,
    select_seq INTEGER NOT NULL,
    service TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    route_revision INTEGER NOT NULL,
    instance_version TEXT NOT NULL,
    outcome TEXT NOT NULL,
    status TEXT,
    error_code TEXT,
    transport_ok INTEGER NOT NULL,
    started_wall_ms INTEGER NOT NULL,
    started_mono_ms INTEGER NOT NULL,
    finished_mono_ms INTEGER NOT NULL,
    wait_start_mono_ms INTEGER NOT NULL,
    wait_end_mono_ms INTEGER NOT NULL,
    budget_wait_ms INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    upstream_delay_ms INTEGER NOT NULL,
    budget_used INTEGER NOT NULL,
    global_inflight INTEGER NOT NULL,
    service_inflight INTEGER NOT NULL,
    final_attempt INTEGER NOT NULL,
    detail TEXT,
    breaker_after_json TEXT NOT NULL,
    UNIQUE(request_id, run_epoch, attempt_no)
);
CREATE INDEX IF NOT EXISTS idx_attempts_request ON attempts(request_id, run_epoch, attempt_no);
CREATE INDEX IF NOT EXISTS idx_attempts_instance ON attempts(instance_id, attempt_id);
CREATE TABLE IF NOT EXISTS budget_usage (
    attempt_id INTEGER PRIMARY KEY REFERENCES attempts(attempt_id),
    service TEXT NOT NULL,
    global_slots INTEGER NOT NULL,
    service_slots INTEGER NOT NULL,
    global_limit INTEGER NOT NULL,
    service_quota INTEGER NOT NULL,
    budget_used_after INTEGER NOT NULL,
    service_budget_used_after INTEGER NOT NULL,
    wait_ms INTEGER NOT NULL,
    running_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS outcomes (
    request_id TEXT NOT NULL,
    run_epoch INTEGER NOT NULL,
    request_seq INTEGER NOT NULL,
    service TEXT NOT NULL,
    final_outcome TEXT NOT NULL,
    error_code TEXT,
    attempt_count INTEGER NOT NULL,
    total_duration_ms INTEGER NOT NULL,
    completed_at_wall_ms INTEGER NOT NULL,
    PRIMARY KEY(request_id, run_epoch)
);
CREATE TABLE IF NOT EXISTS unknown_attempts (
    intent_id INTEGER PRIMARY KEY,
    request_id TEXT NOT NULL,
    run_epoch INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL,
    service TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    select_seq INTEGER NOT NULL,
    started_wall_ms INTEGER NOT NULL,
    detected_wall_ms INTEGER NOT NULL,
    state TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class AttemptIntent:
    request_id: str
    run_epoch: int
    attempt_no: int
    service: str
    instance_id: str
    select_seq: int
    started_wall_ms: int
    started_mono_ms: int


@dataclass(frozen=True)
class AttemptCommit:
    request_id: str
    request_seq: int
    run_epoch: int
    attempt_no: int
    select_seq: int
    service: str
    instance_id: str
    route_revision: int
    instance_version: str
    outcome: str
    status: str | None
    error_code: str | None
    transport_ok: bool
    started_wall_ms: int
    started_mono_ms: int
    finished_mono_ms: int
    wait_start_mono_ms: int
    wait_end_mono_ms: int
    budget_wait_ms: int
    duration_ms: int
    upstream_delay_ms: int
    budget_used: bool
    global_inflight: int
    service_inflight: int
    final_attempt: bool
    detail: str | None
    breaker_after: dict[str, Any]


@dataclass(frozen=True)
class FinalOutcome:
    final_outcome: str
    error_code: str | None
    attempt_count: int
    total_duration_ms: int
    completed_at_wall_ms: int


@dataclass
class RecoverySnapshot:
    budget_used_global: int
    budget_used_by_service: dict[str, int]
    breakers: dict[str, dict[str, Any]]
    completed_cursor: int
    requests_to_rerun: list[dict[str, Any]]
    unknown_attempts: list[dict[str, Any]]
    exit_report_available: bool


class Ledger:
    def __init__(
        self,
        path: str | Path,
        *,
        reset: bool = False,
        logger: StructuredLogger | None = None,
        verify: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logger
        self._lock = threading.RLock()
        if reset:
            for suffix in ("", "-wal", "-shm"):
                candidate = Path(str(self.path) + suffix)
                if candidate.exists():
                    candidate.unlink()
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=FULL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._init_or_verify(verify=verify)

    def _table_names(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return {str(row[0]) for row in rows}

    def _init_or_verify(self, *, verify: bool) -> None:
        tables = self._table_names()
        if not tables:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.execute(f"PRAGMA user_version={LEDGER_FORMAT_VERSION}")
            self._conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES(?, ?)",
                ("format_version", str(LEDGER_FORMAT_VERSION)),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES(?, ?)",
                ("created_wall_ms", str(int(time.time() * 1000))),
            )
            digest = self._compute_digest_locked()
            self._conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES(?, ?)",
                ("content_digest", digest),
            )
            self._conn.commit()
            return
        required = {
            "ledger_meta",
            "requests",
            "attempt_intents",
            "attempts",
            "budget_usage",
            "outcomes",
            "unknown_attempts",
        }
        missing = sorted(required - tables)
        if missing:
            raise MeshError(
                "E_SNAPSHOT",
                "ledger schema is missing required tables: " + ", ".join(missing),
                field="ledger.schema",
                location=str(self.path),
            )
        user_version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        meta = self.meta_dict()
        meta_version = int(meta.get("format_version", "-1"))
        if user_version != LEDGER_FORMAT_VERSION or meta_version != LEDGER_FORMAT_VERSION:
            raise MeshError(
                "E_SNAPSHOT",
                f"ledger format version mismatch: pragma={user_version}, meta={meta_version}, expected={LEDGER_FORMAT_VERSION}",
                field="ledger.format_version",
                location=str(self.path),
            )
        if verify:
            stored = meta.get("content_digest")
            actual = self._compute_digest_locked()
            if stored != actual:
                raise MeshError(
                    "E_SNAPSHOT",
                    "ledger content digest mismatch",
                    field="ledger.content_digest",
                    location=str(self.path),
                    details={"stored": stored, "actual": actual},
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _rows_for_digest(self, table: str, order_by: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
        return [dict(row) for row in rows]

    def _compute_digest_locked(self) -> str:
        meta_rows = self._conn.execute(
            "SELECT key, value FROM ledger_meta WHERE key != 'content_digest' ORDER BY key"
        ).fetchall()
        payload = {
            "format_version": LEDGER_FORMAT_VERSION,
            "meta": {str(row[0]): str(row[1]) for row in meta_rows},
            "requests": self._rows_for_digest("requests", "request_seq, request_id"),
            "attempt_intents": self._rows_for_digest("attempt_intents", "intent_id"),
            "attempts": self._rows_for_digest("attempts", "attempt_id"),
            "budget_usage": self._rows_for_digest("budget_usage", "attempt_id"),
            "outcomes": self._rows_for_digest(
                "outcomes", "request_seq, run_epoch, request_id"
            ),
            "unknown_attempts": self._rows_for_digest(
                "unknown_attempts", "intent_id"
            ),
        }
        return sha256_hex(payload)

    def _refresh_digest_locked(self) -> str:
        digest = self._compute_digest_locked()
        self._conn.execute(
            "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES(?, ?)",
            ("content_digest", digest),
        )
        return digest

    def _write(self, operation: Any) -> Any:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = operation()
                self._refresh_digest_locked()
                self._conn.commit()
                return result
            except Exception:
                self._conn.rollback()
                raise

    def content_digest(self) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM ledger_meta WHERE key='content_digest'"
            ).fetchone()
            return "" if row is None else str(row[0])

    def verify_digest(self) -> None:
        with self._lock:
            stored = self.content_digest()
            actual = self._compute_digest_locked()
            if stored != actual:
                raise MeshError(
                    "E_SNAPSHOT",
                    "ledger content digest mismatch",
                    field="ledger.content_digest",
                    location=str(self.path),
                    details={"stored": stored, "actual": actual},
                )

    def meta_dict(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM ledger_meta ORDER BY key"
            ).fetchall()
            return {str(row[0]): str(row[1]) for row in rows}

    def set_meta(self, values: dict[str, Any]) -> None:
        def operation() -> None:
            for key in sorted(values):
                self._conn.execute(
                    "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES(?, ?)",
                    (key, str(values[key])),
                )

        self._write(operation)

    def register_request(self, request: RequestSpec, request_seq: int) -> int:
        now_wall = int(time.time() * 1000)

        def operation() -> int:
            existing = self._conn.execute(
                "SELECT request_seq, service, payload_json FROM requests WHERE request_id=?",
                (request.id,),
            ).fetchone()
            payload_json = canonical_json(request.payload)
            if existing is not None:
                if existing["service"] != request.service or existing["payload_json"] != payload_json:
                    raise MeshError(
                        "E_SPEC",
                        "request id already exists with different content",
                        field="requests.id",
                        location=request.source_path or "$",
                        request_id=request.id,
                        service=request.service,
                    )
                return int(existing["request_seq"])
            self._conn.execute(
                """
                INSERT INTO requests(
                    request_id, request_seq, service, payload_json, status,
                    run_epoch, attempts_committed, registered_at_wall_ms
                ) VALUES(?, ?, ?, ?, 'pending', 1, 0, ?)
                """,
                (request.id, request_seq, request.service, payload_json, now_wall),
            )
            return request_seq

        return int(self._write(operation))

    def begin_intent(self, intent: AttemptIntent) -> None:
        def operation() -> None:
            self._conn.execute(
                """
                INSERT INTO attempt_intents(
                    request_id, run_epoch, attempt_no, service, instance_id,
                    select_seq, started_wall_ms, started_mono_ms, state
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'inflight')
                ON CONFLICT(request_id, run_epoch, attempt_no) DO UPDATE SET
                    service=excluded.service,
                    instance_id=excluded.instance_id,
                    select_seq=excluded.select_seq,
                    started_wall_ms=excluded.started_wall_ms,
                    started_mono_ms=excluded.started_mono_ms,
                    state='inflight'
                """,
                (
                    intent.request_id,
                    intent.run_epoch,
                    intent.attempt_no,
                    intent.service,
                    intent.instance_id,
                    intent.select_seq,
                    intent.started_wall_ms,
                    intent.started_mono_ms,
                ),
            )
            self._conn.execute(
                "UPDATE requests SET status='running' WHERE request_id=? AND status!='complete'",
                (intent.request_id,),
            )

        self._write(operation)

    def commit_attempt(self, attempt: AttemptCommit, final: FinalOutcome | None) -> int:
        def operation() -> int:
            cursor = self._conn.execute(
                """
                INSERT INTO attempts(
                    request_id, request_seq, run_epoch, attempt_no, select_seq,
                    service, instance_id, route_revision, instance_version,
                    outcome, status, error_code, transport_ok,
                    started_wall_ms, started_mono_ms, finished_mono_ms,
                    wait_start_mono_ms, wait_end_mono_ms, budget_wait_ms,
                    duration_ms, upstream_delay_ms, budget_used,
                    global_inflight, service_inflight, final_attempt, detail,
                    breaker_after_json
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.request_id,
                    attempt.request_seq,
                    attempt.run_epoch,
                    attempt.attempt_no,
                    attempt.select_seq,
                    attempt.service,
                    attempt.instance_id,
                    attempt.route_revision,
                    attempt.instance_version,
                    attempt.outcome,
                    attempt.status,
                    attempt.error_code,
                    1 if attempt.transport_ok else 0,
                    attempt.started_wall_ms,
                    attempt.started_mono_ms,
                    attempt.finished_mono_ms,
                    attempt.wait_start_mono_ms,
                    attempt.wait_end_mono_ms,
                    attempt.budget_wait_ms,
                    attempt.duration_ms,
                    attempt.upstream_delay_ms,
                    1 if attempt.budget_used else 0,
                    attempt.global_inflight,
                    attempt.service_inflight,
                    1 if attempt.final_attempt else 0,
                    attempt.detail,
                    canonical_json(attempt.breaker_after),
                ),
            )
            attempt_id = int(cursor.lastrowid)
            global_used = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM attempts WHERE budget_used=1"
                ).fetchone()[0]
            )
            service_used = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM attempts WHERE budget_used=1 AND service=?",
                    (attempt.service,),
                ).fetchone()[0]
            )
            self._conn.execute(
                """
                INSERT INTO budget_usage(
                    attempt_id, service, global_slots, service_slots,
                    global_limit, service_quota, budget_used_after,
                    service_budget_used_after, wait_ms, running_ms
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    attempt.service,
                    1 if attempt.budget_used else 0,
                    1 if attempt.budget_used else 0,
                    int(self._meta_int_locked("global_concurrency", 0)),
                    int(self._meta_int_locked(f"quota:{attempt.service}", 0)),
                    global_used,
                    service_used,
                    attempt.budget_wait_ms,
                    attempt.duration_ms,
                ),
            )
            self._conn.execute(
                "DELETE FROM attempt_intents WHERE request_id=? AND run_epoch=? AND attempt_no=?",
                (attempt.request_id, attempt.run_epoch, attempt.attempt_no),
            )
            self._conn.execute(
                "UPDATE requests SET attempts_committed=attempts_committed+1 WHERE request_id=? AND run_epoch=?",
                (attempt.request_id, attempt.run_epoch),
            )
            if final is not None:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO outcomes(
                        request_id, run_epoch, request_seq, service, final_outcome,
                        error_code, attempt_count, total_duration_ms, completed_at_wall_ms
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.request_id,
                        attempt.run_epoch,
                        attempt.request_seq,
                        attempt.service,
                        final.final_outcome,
                        final.error_code,
                        final.attempt_count,
                        final.total_duration_ms,
                        final.completed_at_wall_ms,
                    ),
                )
                self._conn.execute(
                    """
                    UPDATE requests SET
                        status='complete', final_outcome=?, final_error_code=?,
                        completed_at_wall_ms=?
                    WHERE request_id=? AND run_epoch=?
                    """,
                    (
                        final.final_outcome,
                        final.error_code,
                        final.completed_at_wall_ms,
                        attempt.request_id,
                        attempt.run_epoch,
                    ),
                )
            return attempt_id

        return int(self._write(operation))

    def _meta_int_locked(self, key: str, default: int) -> int:
        row = self._conn.execute(
            "SELECT value FROM ledger_meta WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return default
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return default

    def recover(self) -> RecoverySnapshot:
        """Mark unresolved intents unknown and rebuild only committed state."""

        detected_wall = int(time.time() * 1000)

        def operation() -> None:
            intents = self._conn.execute(
                "SELECT * FROM attempt_intents ORDER BY intent_id"
            ).fetchall()
            for row in intents:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO unknown_attempts(
                        intent_id, request_id, run_epoch, attempt_no, service,
                        instance_id, select_seq, started_wall_ms, detected_wall_ms, state
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'unknown')
                    """,
                    (
                        row["intent_id"],
                        row["request_id"],
                        row["run_epoch"],
                        row["attempt_no"],
                        row["service"],
                        row["instance_id"],
                        row["select_seq"],
                        row["started_wall_ms"],
                        detected_wall,
                    ),
                )
                self._conn.execute(
                    "UPDATE attempt_intents SET state='unknown' WHERE intent_id=?",
                    (row["intent_id"],),
                )
                self._conn.execute(
                    """
                    UPDATE requests SET status='unknown'
                    WHERE request_id=? AND status!='complete'
                    """,
                    (row["request_id"],),
                )
            self._conn.execute(
                "UPDATE requests SET status='unknown' WHERE status='running'"
            )

        self._write(operation)
        with self._lock:
            budget_rows = self._conn.execute(
                "SELECT service, COUNT(*) AS n FROM attempts WHERE budget_used=1 GROUP BY service ORDER BY service"
            ).fetchall()
            budgets = {str(row["service"]): int(row["n"]) for row in budget_rows}
            global_used = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM attempts WHERE budget_used=1"
                ).fetchone()[0]
            )
            latest_breakers: dict[str, dict[str, Any]] = {}
            for row in self._conn.execute(
                "SELECT instance_id, breaker_after_json FROM attempts ORDER BY attempt_id"
            ):
                try:
                    parsed = json.loads(row["breaker_after_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(parsed, dict):
                    latest_breakers[str(row["instance_id"])] = parsed
            cursor_row = self._conn.execute("SELECT COALESCE(MAX(attempt_id), 0) AS c FROM attempts").fetchone()
            completed_cursor = int(cursor_row["c"])
            rerun_rows = self._conn.execute(
                "SELECT request_id, request_seq, service, payload_json, run_epoch, status FROM requests WHERE status!='complete' ORDER BY request_seq, request_id"
            ).fetchall()
            requests_to_rerun = [dict(row) for row in rerun_rows]
            unknown_rows = self._conn.execute(
                "SELECT * FROM unknown_attempts ORDER BY intent_id"
            ).fetchall()
            unknown_attempts = [dict(row) for row in unknown_rows]
        if self._logger is not None:
            self._logger.warning(
                "ledger_recovery",
                code="E_RECOVERY",
                committed_cursor=completed_cursor,
                unknown_count=len(unknown_attempts),
                rerun_count=len(requests_to_rerun),
                ledger=str(self.path),
            )
        return RecoverySnapshot(
            budget_used_global=global_used,
            budget_used_by_service=budgets,
            breakers=latest_breakers,
            completed_cursor=completed_cursor,
            requests_to_rerun=requests_to_rerun,
            unknown_attempts=unknown_attempts,
            exit_report_available=False,
        )

    def mark_rerun(self, request_id: str) -> int:
        def operation() -> int:
            row = self._conn.execute(
                "SELECT run_epoch, status FROM requests WHERE request_id=?",
                (request_id,),
            ).fetchone()
            if row is None:
                raise MeshError(
                    "E_REPLAY",
                    "request is missing from ledger",
                    request_id=request_id,
                    field="request_id",
                )
            if row["status"] == "complete":
                return int(row["run_epoch"])
            new_epoch = int(row["run_epoch"]) + 1
            self._conn.execute(
                "UPDATE requests SET run_epoch=?, status='pending', final_outcome=NULL, final_error_code=NULL, completed_at_wall_ms=NULL WHERE request_id=?",
                (new_epoch, request_id),
            )
            return new_epoch

        return int(self._write(operation))

    def requests_for_rerun(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM requests WHERE status!='complete' ORDER BY request_seq, request_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def request_epoch(self, request_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT run_epoch FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
            return 0 if row is None else int(row[0])

    def request_sequence(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM requests ORDER BY request_seq, request_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def latest_outcomes(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT o.* FROM outcomes o
                JOIN requests r ON r.request_id=o.request_id AND r.run_epoch=o.run_epoch
                ORDER BY o.service, o.request_seq, o.request_id
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def latest_attempts(self, limit: int | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT a.*, b.global_slots, b.service_slots, b.global_limit,
                   b.service_quota, b.budget_used_after, b.service_budget_used_after
            FROM attempts a
            JOIN requests r ON r.request_id=a.request_id AND r.run_epoch=a.run_epoch
            LEFT JOIN budget_usage b ON b.attempt_id=a.attempt_id
            ORDER BY a.request_seq, a.attempt_no, a.attempt_id
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query = """
                SELECT * FROM (
                    SELECT a.*, b.global_slots, b.service_slots, b.global_limit,
                           b.service_quota, b.budget_used_after, b.service_budget_used_after
                    FROM attempts a
                    JOIN requests r ON r.request_id=a.request_id AND r.run_epoch=a.run_epoch
                    LEFT JOIN budget_usage b ON b.attempt_id=a.attempt_id
                    ORDER BY a.attempt_id DESC LIMIT ?
                ) ORDER BY request_seq, attempt_no, attempt_id
            """
            params = (int(limit),)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def all_attempts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM attempts ORDER BY request_seq, run_epoch, attempt_no, attempt_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def attempts_for_request(self, request_id: str, epoch: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if epoch is None:
                epoch = self.request_epoch(request_id)
            rows = self._conn.execute(
                "SELECT * FROM attempts WHERE request_id=? AND run_epoch=? ORDER BY attempt_no, attempt_id",
                (request_id, epoch),
            ).fetchall()
            return [dict(row) for row in rows]

    def unknown_attempts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM unknown_attempts ORDER BY intent_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def attempts_by_instance(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT instance_id, COUNT(*) AS n FROM attempts GROUP BY instance_id ORDER BY instance_id"
            ).fetchall()
            return {str(row[0]): int(row[1]) for row in rows}

    def requests_by_service(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT service, COUNT(*) AS n FROM outcomes GROUP BY service ORDER BY service"
            ).fetchall()
            return {str(row[0]): int(row[1]) for row in rows}

    def count_attempts(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])

    def count_requests(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM requests").fetchone()[0])

    def budget_totals(self) -> tuple[int, dict[str, int]]:
        with self._lock:
            global_used = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM attempts WHERE budget_used=1"
                ).fetchone()[0]
            )
            rows = self._conn.execute(
                "SELECT service, COUNT(*) AS n FROM attempts WHERE budget_used=1 GROUP BY service ORDER BY service"
            ).fetchall()
            return global_used, {str(row[0]): int(row[1]) for row in rows}

    def completed_cursor(self) -> int:
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COALESCE(MAX(attempt_id), 0) FROM attempts"
                ).fetchone()[0]
            )

    def latest_breakers(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for row in self._conn.execute(
                "SELECT instance_id, breaker_after_json FROM attempts ORDER BY attempt_id"
            ):
                try:
                    parsed = json.loads(row["breaker_after_json"])
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(parsed, dict):
                    result[str(row["instance_id"])] = parsed
            return {key: result[key] for key in sorted(result)}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self.path),
                "format_version": LEDGER_FORMAT_VERSION,
                "content_digest": self.content_digest(),
                "requests": self.count_requests(),
                "attempts": self.count_attempts(),
                "unknown_attempts": len(self.unknown_attempts()),
                "meta": self.meta_dict(),
            }
