"""Crash-safe SQLite receipt ledger with deterministic content digests."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DiagnosticError, is_int
from .jsonio import canonical_json, parse_json_text, sha256_canonical
from .planner import BatchPlan, DispatchPlan


LEDGER_FORMAT_VERSION = 1
ROW_FIELDS = (
    "batch_id",
    "worker_id",
    "batch_seq",
    "param_version_seq",
    "param_digest",
    "request_ids",
    "arrival_seqs",
    "request_binding_digest",
    "plan_digest",
    "status",
    "receipt_digest",
    "normalized_receipts",
    "invalidation_trace",
)


def ledger_content_digest(
    rows: Iterable[dict[str, Any]],
    *,
    format_version: int,
    progress_cursor: int,
) -> str:
    ordered = sorted(
        rows,
        key=lambda row: (row["worker_id"], row["batch_seq"], row["batch_id"]),
    )
    payload = {
        "format_version": format_version,
        "progress_cursor": progress_cursor,
        "rows": ordered,
    }
    return sha256_canonical(payload)


def _pending_row(batch: BatchPlan) -> dict[str, Any]:
    return {
        "batch_id": batch.batch_id,
        "worker_id": batch.worker_id,
        "batch_seq": batch.batch_seq,
        "param_version_seq": batch.param_version_seq,
        "param_digest": batch.param_digest,
        "request_ids": list(batch.request_ids),
        "arrival_seqs": list(batch.arrival_seqs),
        "request_binding_digest": batch.request_binding_digest,
        "plan_digest": batch.plan_digest,
        "status": "pending",
        "receipt_digest": None,
        "normalized_receipts": [],
        "invalidation_trace": {},
    }


def _row_matches_batch(row: dict[str, Any], batch: BatchPlan) -> bool:
    return (
        row["batch_id"] == batch.batch_id
        and row["worker_id"] == batch.worker_id
        and row["batch_seq"] == batch.batch_seq
        and row["param_version_seq"] == batch.param_version_seq
        and row["param_digest"] == batch.param_digest
        and row["request_ids"] == list(batch.request_ids)
        and row["arrival_seqs"] == list(batch.arrival_seqs)
        and row["request_binding_digest"] == batch.request_binding_digest
        and row["plan_digest"] == batch.plan_digest
    )


def validate_row_payload(row: Any, *, code: str = "E_LEDGER") -> dict[str, Any]:
    if not isinstance(row, dict):
        raise DiagnosticError(
            code=code, message="ledger row must be an object", field="rows"
        )
    for field in ROW_FIELDS:
        if field not in row:
            raise DiagnosticError(
                code=code,
                message=f"ledger row is missing {field!r}",
                object=row.get("batch_id", "<row>"),
                field=field,
            )
    for field in ("batch_id", "worker_id", "param_digest", "request_binding_digest", "plan_digest"):
        if not isinstance(row[field], str) or not row[field]:
            raise DiagnosticError(
                code=code,
                message=f"ledger row field {field!r} must be a non-empty string",
                object=row.get("batch_id", "<row>"),
                field=field,
            )
    for field in ("batch_seq", "param_version_seq"):
        if not is_int(row[field]) or row[field] < 0:
            raise DiagnosticError(
                code=code,
                message=f"ledger row field {field!r} must be a non-negative integer",
                object=row.get("batch_id", "<row>"),
                field=field,
            )
    if row["status"] not in {"pending", "completed"}:
        raise DiagnosticError(
            code=code,
            message="ledger row status must be pending or completed",
            object=row.get("batch_id", "<row>"),
            field="status",
        )
    if not isinstance(row["request_ids"], list) or not all(
        isinstance(item, str) and item for item in row["request_ids"]
    ):
        raise DiagnosticError(
            code=code,
            message="request_ids must be an array of strings",
            object=row.get("batch_id", "<row>"),
            field="request_ids",
        )
    if not isinstance(row["arrival_seqs"], list) or not all(
        is_int(item) and item >= 0 for item in row["arrival_seqs"]
    ):
        raise DiagnosticError(
            code=code,
            message="arrival_seqs must be an array of non-negative integers",
            object=row.get("batch_id", "<row>"),
            field="arrival_seqs",
        )
    if row["receipt_digest"] is not None and not isinstance(
        row["receipt_digest"], str
    ):
        raise DiagnosticError(
            code=code,
            message="receipt_digest must be a string or null",
            object=row.get("batch_id", "<row>"),
            field="receipt_digest",
        )
    if not isinstance(row["normalized_receipts"], list):
        raise DiagnosticError(
            code=code,
            message="normalized_receipts must be an array",
            object=row.get("batch_id", "<row>"),
            field="normalized_receipts",
        )
    if not isinstance(row["invalidation_trace"], dict):
        raise DiagnosticError(
            code=code,
            message="invalidation_trace must be an object",
            object=row.get("batch_id", "<row>"),
            field="invalidation_trace",
        )
    if row["status"] == "completed":
        if not isinstance(row["receipt_digest"], str) or not row["receipt_digest"]:
            raise DiagnosticError(
                code=code,
                message="completed rows require a receipt digest",
                object=row["batch_id"],
                field="receipt_digest",
            )
    else:
        if row["receipt_digest"] is not None or row["normalized_receipts"]:
            raise DiagnosticError(
                code=code,
                message="pending rows cannot claim receipt data",
                object=row.get("batch_id", "<row>"),
                field="receipt_digest",
            )
    return row


@dataclass(frozen=True)
class RecoveryReport:
    completed_batch_ids: tuple[str, ...]
    unknown_batch_ids: tuple[str, ...]
    progress_cursor: int
    requires_rerun: bool
    interrupted_process_exit_report: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_batch_ids": list(self.completed_batch_ids),
            "unknown_batch_ids": list(self.unknown_batch_ids),
            "progress_cursor": self.progress_cursor,
            "requires_rerun": self.requires_rerun,
            "interrupted_process_exit_report": None,
            "note": (
                "Only fully committed batches were recovered; a forced-terminated "
                "process has no trustworthy exit report."
            ),
        }


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA busy_timeout=10000")
        self._create_schema()
        self.verify_integrity()

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    worker_id TEXT NOT NULL,
                    batch_seq INTEGER NOT NULL,
                    param_version_seq INTEGER NOT NULL,
                    param_digest TEXT NOT NULL,
                    request_ids TEXT NOT NULL,
                    arrival_seqs TEXT NOT NULL,
                    request_binding_digest TEXT NOT NULL,
                    plan_digest TEXT NOT NULL,
                    status TEXT NOT NULL,
                    receipt_digest TEXT,
                    normalized_receipts TEXT NOT NULL,
                    invalidation_trace TEXT NOT NULL,
                    row_payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS batches_order
                    ON batches(worker_id, batch_seq, batch_id);
                """
            )
            meta = dict(self._connection.execute("SELECT key, value FROM meta"))
            if "format_version" not in meta:
                self._connection.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self._connection.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('progress_cursor', '0')"
                )
                digest = ledger_content_digest(
                    [], format_version=LEDGER_FORMAT_VERSION, progress_cursor=0
                )
                self._connection.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
                    (digest,),
                )

    def _meta(self) -> dict[str, str]:
        return dict(self._connection.execute("SELECT key, value FROM meta"))

    @property
    def format_version(self) -> int:
        with self._lock:
            value = self._meta().get("format_version")
        try:
            return int(value) if value is not None else 0
        except ValueError as exc:
            raise DiagnosticError(
                code="E_LEDGER",
                message="ledger format_version is not an integer",
                object=str(self.path),
                field="format_version",
            ) from exc

    @property
    def progress_cursor(self) -> int:
        with self._lock:
            value = self._meta().get("progress_cursor", "0")
        try:
            return int(value)
        except ValueError as exc:
            raise DiagnosticError(
                code="E_LEDGER",
                message="ledger progress cursor is not an integer",
                object=str(self.path),
                field="progress_cursor",
            ) from exc

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [
                parse_json_text(value, object_name=str(self.path), code="E_LEDGER")
                for (value,) in self._connection.execute(
                    "SELECT row_payload FROM batches ORDER BY worker_id, batch_seq, batch_id"
                )
            ]
        return [validate_row_payload(row) for row in rows]

    def row_by_id(self) -> dict[str, dict[str, Any]]:
        return {row["batch_id"]: row for row in self.rows()}

    def completed_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            row["batch_id"] for row in self.rows() if row["status"] == "completed"
        )

    def unknown_batch_ids(self) -> tuple[str, ...]:
        return tuple(
            row["batch_id"] for row in self.rows() if row["status"] == "pending"
        )

    def recovery_report(self) -> RecoveryReport:
        completed = self.completed_batch_ids()
        unknown = self.unknown_batch_ids()
        return RecoveryReport(
            completed_batch_ids=completed,
            unknown_batch_ids=unknown,
            progress_cursor=self.progress_cursor,
            requires_rerun=bool(unknown),
        )

    @property
    def content_digest(self) -> str:
        with self._lock:
            value = self._meta().get("content_digest")
        if value is None:
            raise DiagnosticError(
                code="E_LEDGER",
                message="ledger has no content digest",
                object=str(self.path),
                field="content_digest",
            )
        return value

    def _computed_digest(self) -> str:
        rows = self.rows()
        completed = sum(1 for row in rows if row["status"] == "completed")
        return ledger_content_digest(
            rows,
            format_version=self.format_version,
            progress_cursor=completed,
        )

    def verify_integrity(self) -> None:
        with self._lock:
            if self.format_version != LEDGER_FORMAT_VERSION:
                raise DiagnosticError(
                    code="E_LEDGER",
                    message=(
                        f"unsupported ledger format {self.format_version}; "
                        f"expected {LEDGER_FORMAT_VERSION}"
                    ),
                    object=str(self.path),
                    field="format_version",
                )
            rows = self.rows()
            completed = sum(1 for row in rows if row["status"] == "completed")
            if completed != self.progress_cursor:
                raise DiagnosticError(
                    code="E_LEDGER",
                    message="progress cursor disagrees with completed rows",
                    object=str(self.path),
                    field="progress_cursor",
                )
            expected = ledger_content_digest(
                rows,
                format_version=self.format_version,
                progress_cursor=completed,
            )
            if expected != self.content_digest:
                raise DiagnosticError(
                    code="E_LEDGER",
                    message="ledger content digest mismatch",
                    object=str(self.path),
                    field="content_digest",
                    details={"expected": expected, "actual": self.content_digest},
                )

    def _insert_row(self, row: dict[str, Any]) -> None:
        validate_row_payload(row)
        self._connection.execute(
            """
            INSERT INTO batches(
                batch_id, worker_id, batch_seq, param_version_seq, param_digest,
                request_ids, arrival_seqs, request_binding_digest, plan_digest,
                status, receipt_digest, normalized_receipts, invalidation_trace,
                row_payload
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["batch_id"],
                row["worker_id"],
                row["batch_seq"],
                row["param_version_seq"],
                row["param_digest"],
                canonical_json(row["request_ids"]),
                canonical_json(row["arrival_seqs"]),
                row["request_binding_digest"],
                row["plan_digest"],
                row["status"],
                row["receipt_digest"],
                canonical_json(row["normalized_receipts"]),
                canonical_json(row["invalidation_trace"]),
                canonical_json(row),
            ),
        )

    def _replace_row(self, row: dict[str, Any]) -> None:
        self._connection.execute(
            "DELETE FROM batches WHERE batch_id = ?", (row["batch_id"],)
        )
        self._insert_row(row)

    def _write_meta(self) -> None:
        rows = self.rows()
        completed = sum(1 for row in rows if row["status"] == "completed")
        digest = ledger_content_digest(
            rows,
            format_version=self.format_version,
            progress_cursor=completed,
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('progress_cursor', ?)",
            (str(completed),),
        )
        self._connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
            (digest,),
        )

    def apply_plan(
        self,
        plan: DispatchPlan,
        *,
        invalid_batch_ids: Iterable[str] = (),
    ) -> None:
        invalid = set(invalid_batch_ids)
        plan_by_id = plan.by_id()
        with self._lock:
            self.verify_integrity()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                existing = self.row_by_id()
                for batch_id in sorted(set(existing) - set(plan_by_id)):
                    self._connection.execute(
                        "DELETE FROM batches WHERE batch_id = ?", (batch_id,)
                    )
                    existing.pop(batch_id)
                for batch in plan.sorted_batches():
                    prior = existing.get(batch.batch_id)
                    if (
                        prior is not None
                        and batch.batch_id not in invalid
                        and _row_matches_batch(prior, batch)
                    ):
                        continue
                    if prior is not None:
                        self._connection.execute(
                            "DELETE FROM batches WHERE batch_id = ?", (batch.batch_id,)
                        )
                    self._insert_row(_pending_row(batch))
                self._write_meta()
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def complete_batch(
        self,
        batch: BatchPlan,
        *,
        receipt_digest: str,
        normalized_receipts: list[dict[str, Any]],
        invalidation_trace: dict[str, Any],
    ) -> dict[str, Any]:
        row = {
            "batch_id": batch.batch_id,
            "worker_id": batch.worker_id,
            "batch_seq": batch.batch_seq,
            "param_version_seq": batch.param_version_seq,
            "param_digest": batch.param_digest,
            "request_ids": list(batch.request_ids),
            "arrival_seqs": list(batch.arrival_seqs),
            "request_binding_digest": batch.request_binding_digest,
            "plan_digest": batch.plan_digest,
            "status": "completed",
            "receipt_digest": receipt_digest,
            "normalized_receipts": normalized_receipts,
            "invalidation_trace": invalidation_trace,
        }
        with self._lock:
            self.verify_integrity()
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._replace_row(row)
                self._write_meta()
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return row

    def snapshot_rows(self) -> list[dict[str, Any]]:
        return self.rows()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def write_ledger_rows_to_new_database(
    path: str | Path,
    *,
    rows: Iterable[dict[str, Any]],
    format_version: int,
    progress_cursor: int,
    content_digest: str,
    code: str = "E_SNAPSHOT",
) -> None:
    target = Path(path)
    if target.exists():
        target.unlink()
    connection = sqlite3.connect(target, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(
            """
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE batches (
                batch_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                batch_seq INTEGER NOT NULL,
                param_version_seq INTEGER NOT NULL,
                param_digest TEXT NOT NULL,
                request_ids TEXT NOT NULL,
                arrival_seqs TEXT NOT NULL,
                request_binding_digest TEXT NOT NULL,
                plan_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                receipt_digest TEXT,
                normalized_receipts TEXT NOT NULL,
                invalidation_trace TEXT NOT NULL,
                row_payload TEXT NOT NULL
            );
            CREATE INDEX batches_order ON batches(worker_id, batch_seq, batch_id);
            """
        )
        normalized_rows = [validate_row_payload(row, code=code) for row in rows]
        completed = sum(1 for row in normalized_rows if row["status"] == "completed")
        if completed != progress_cursor:
            raise DiagnosticError(
                code=code,
                message="snapshot cursor disagrees with completed rows",
                object=str(target),
                field="cursor",
            )
        computed = ledger_content_digest(
            normalized_rows,
            format_version=format_version,
            progress_cursor=progress_cursor,
        )
        if computed != content_digest:
            raise DiagnosticError(
                code=code,
                message="snapshot ledger digest mismatch",
                object=str(target),
                field="ledger_digest",
                details={"expected": computed, "actual": content_digest},
            )
        for row in normalized_rows:
            connection.execute(
                """
                INSERT INTO batches VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["batch_id"],
                    row["worker_id"],
                    row["batch_seq"],
                    row["param_version_seq"],
                    row["param_digest"],
                    canonical_json(row["request_ids"]),
                    canonical_json(row["arrival_seqs"]),
                    row["request_binding_digest"],
                    row["plan_digest"],
                    row["status"],
                    row["receipt_digest"],
                    canonical_json(row["normalized_receipts"]),
                    canonical_json(row["invalidation_trace"]),
                    canonical_json(row),
                ),
            )
        connection.execute(
            "INSERT INTO meta VALUES('format_version', ?)", (str(format_version),)
        )
        connection.execute(
            "INSERT INTO meta VALUES('progress_cursor', ?)", (str(progress_cursor),)
        )
        connection.execute(
            "INSERT INTO meta VALUES('content_digest', ?)", (content_digest,)
        )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
