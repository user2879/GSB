"""SQLite receipt ledger and whole-snapshot replacement semantics."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import atomic_write_json, canonical_json, digest_json
from .errors import DiagnosticError, fail


LEDGER_FORMAT_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LedgerAssignment:
    request_id: str
    rank: int
    duplicate_refs: tuple[str, ...]
    vector_digest: str
    conclusion_digest: str
    receipt_json: Mapping[str, Any]


@dataclass(frozen=True)
class LedgerBatch:
    batch_id: str
    worker_id: str
    batch_seq: int
    batch_version: int
    request_ids: tuple[str, ...]
    arrival_seqs: tuple[int, ...]
    param_version: int
    param_digest: str
    output_digest: str
    receipt_digest: str
    elapsed_ms: int
    wait_start_ms: float
    wait_end_ms: float
    run_start_ms: float
    run_end_ms: float
    status: str
    run_id: str
    worker_pid: int
    assignments: tuple[LedgerAssignment, ...]


@dataclass(frozen=True)
class LedgerTrace:
    run_id: str
    worker_id: str
    batch_id: str
    reason: str
    dependency_path: tuple[str, ...]
    request_ids: tuple[str, ...]
    meta: Mapping[str, Any] = field(default_factory=dict)


def _decode_json(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - database corruption boundary
        fail("E_LEDGER", "ledger.json", f"invalid stored JSON: {exc.msg}")


def _stable_assignment(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "rank": row["rank"],
        "duplicate_refs": sorted(row.get("duplicate_refs") or []),
        "vector_digest": row["vector_digest"],
        "conclusion_digest": row["conclusion_digest"],
    }


def _stable_batch(row: Mapping[str, Any]) -> dict[str, Any]:
    assignments = sorted(
        (_stable_assignment(item) for item in row.get("assignments", [])),
        key=lambda item: item["request_id"],
    )
    return {
        "batch_id": row["batch_id"],
        "worker_id": row["worker_id"],
        "batch_seq": row["batch_seq"],
        "batch_version": row["batch_version"],
        "request_ids": list(row["request_ids"]),
        "arrival_seqs": list(row["arrival_seqs"]),
        "param_version": row["param_version"],
        "param_digest": row["param_digest"],
        "output_digest": row["output_digest"],
        "receipt_digest": row["receipt_digest"],
        "status": row["status"],
        "run_id": row["run_id"],
        "assignments": assignments,
    }


def _stable_trace(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "worker_id": row["worker_id"],
        "batch_id": row["batch_id"],
        "reason": row["reason"],
        "dependency_path": list(row["dependency_path"]),
        "request_ids": list(row["request_ids"]),
        "meta": row.get("meta", {}),
    }


def _logical_content(
    *,
    meta: Mapping[str, Any],
    session: Mapping[str, Any],
    progress: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stable_batches = sorted(
        (_stable_batch(batch) for batch in batches),
        key=lambda item: (item["worker_id"], item["batch_seq"], item["batch_id"]),
    )
    stable_traces = sorted(
        (_stable_trace(trace) for trace in traces),
        key=lambda item: canonical_json(item),
    )
    return {
        "schema": "receipt-ledger-content-v1",
        "ledger_format_version": LEDGER_FORMAT_VERSION,
        "meta": {
            "plan_fingerprint": meta.get("plan_fingerprint", ""),
            "spec_fingerprint": meta.get("spec_fingerprint", ""),
            "receipt_format": meta.get("receipt_format", ""),
        },
        "session": {
            "run_id": session.get("run_id", ""),
            "state": session.get("state", "empty"),
        },
        "progress": {
            "completed_count": progress.get("completed_count", 0),
            "total_count": progress.get("total_count", 0),
            "updated_run_id": progress.get("updated_run_id", ""),
        },
        "batches": stable_batches,
        "traces": stable_traces,
    }


class Ledger:
    """Thread-safe ledger writer with one connection per ledger instance."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), timeout=30.0, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._initialize_or_verify()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_connection", None) is not None:
                self._connection.close()
                self._connection = None  # type: ignore[assignment]

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            fail("E_LEDGER", "ledger", "connection is closed")
        return self._connection

    def _create_schema(self) -> None:
        connection = self.connection
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                run_id TEXT NOT NULL,
                state TEXT NOT NULL,
                started_ns INTEGER,
                finished_ns INTEGER,
                exit_report_json TEXT
            );
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                completed_count INTEGER NOT NULL,
                total_count INTEGER NOT NULL,
                updated_run_id TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                worker_id TEXT NOT NULL,
                batch_seq INTEGER NOT NULL,
                batch_version INTEGER NOT NULL,
                request_ids_json TEXT NOT NULL,
                arrival_seqs_json TEXT NOT NULL,
                param_version INTEGER NOT NULL,
                param_digest TEXT NOT NULL,
                output_digest TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                wait_start_ms REAL NOT NULL,
                wait_end_ms REAL NOT NULL,
                run_start_ms REAL NOT NULL,
                run_end_ms REAL NOT NULL,
                status TEXT NOT NULL,
                run_id TEXT NOT NULL,
                worker_pid INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assignments (
                batch_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                duplicate_refs_json TEXT NOT NULL,
                vector_digest TEXT NOT NULL,
                conclusion_digest TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, request_id),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS traces (
                trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                dependency_path_json TEXT NOT NULL,
                request_ids_json TEXT NOT NULL,
                meta_json TEXT NOT NULL
            );
            """
        )

    def _initialize_or_verify(self) -> None:
        connection = self.connection
        existing_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not existing_tables:
            with connection:
                self._create_schema()
                connection.executemany(
                    "INSERT INTO meta(key, value) VALUES(?, ?)",
                    [
                        ("format_version", str(LEDGER_FORMAT_VERSION)),
                        ("content_digest", ""),
                        ("snapshot_version", "1"),
                        ("plan_fingerprint", ""),
                        ("spec_fingerprint", ""),
                        ("receipt_format", ""),
                    ],
                )
                connection.execute(
                    "INSERT INTO session(id, run_id, state, started_ns, finished_ns, exit_report_json) "
                    "VALUES(1, '', 'empty', NULL, NULL, NULL)"
                )
                connection.execute(
                    "INSERT INTO progress(id, completed_count, total_count, updated_run_id) "
                    "VALUES(1, 0, 0, '')"
                )
                self._refresh_digest(connection)
            return
        required = {"meta", "session", "progress", "batches", "assignments", "traces"}
        if not required.issubset(existing_tables):
            fail("E_LEDGER", "ledger.schema", "incomplete ledger schema")
        version_text = self._meta_get("format_version")
        try:
            version = int(version_text)
        except (TypeError, ValueError):
            fail("E_LEDGER", "ledger.format_version", "expected an integer")
        if version != LEDGER_FORMAT_VERSION:
            fail(
                "E_LEDGER",
                "ledger.format_version",
                f"unsupported ledger format version {version}",
            )
        self.verify_integrity()

    def _meta_get(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_map(self) -> dict[str, str]:
        return {
            str(row["key"]): str(row["value"])
            for row in self.connection.execute("SELECT key, value FROM meta").fetchall()
        }

    def _meta_set(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _session_row(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM session WHERE id=1").fetchone()
        if row is None:
            fail("E_LEDGER", "ledger.session", "session row is missing")
        return dict(row)

    def _progress_row(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM progress WHERE id=1").fetchone()
        if row is None:
            fail("E_LEDGER", "ledger.progress", "progress row is missing")
        return dict(row)

    def _batch_rows(self) -> list[dict[str, Any]]:
        batch_rows = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM batches ORDER BY worker_id, batch_seq, batch_id"
            ).fetchall()
        ]
        assignment_rows = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM assignments ORDER BY batch_id, request_id"
            ).fetchall()
        ]
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in assignment_rows:
            grouped.setdefault(str(row["batch_id"]), []).append(
                {
                    "request_id": row["request_id"],
                    "rank": int(row["rank"]),
                    "duplicate_refs": _decode_json(row["duplicate_refs_json"], []),
                    "vector_digest": row["vector_digest"],
                    "conclusion_digest": row["conclusion_digest"],
                    "receipt_json": _decode_json(row["receipt_json"], {}),
                }
            )
        result: list[dict[str, Any]] = []
        for row in batch_rows:
            row["request_ids"] = _decode_json(row.pop("request_ids_json"), [])
            row["arrival_seqs"] = _decode_json(row.pop("arrival_seqs_json"), [])
            row["assignments"] = grouped.get(str(row["batch_id"]), [])
            result.append(row)
        return result

    def _trace_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.connection.execute(
            "SELECT * FROM traces ORDER BY trace_id"
        ).fetchall():
            item = dict(row)
            item["dependency_path"] = _decode_json(item.pop("dependency_path_json"), [])
            item["request_ids"] = _decode_json(item.pop("request_ids_json"), [])
            item["meta"] = _decode_json(item.pop("meta_json"), {})
            item.pop("trace_id", None)
            rows.append(item)
        return rows

    def _compute_digest(self, connection: sqlite3.Connection | None = None) -> str:
        # The helper reads through instance methods; callers always hold _lock.
        content = _logical_content(
            meta=self._meta_map(),
            session=self._session_row(),
            progress=self._progress_row(),
            batches=self._batch_rows(),
            traces=self._trace_rows(),
        )
        return digest_json(content)

    def _refresh_digest(self, connection: sqlite3.Connection | None = None) -> str:
        digest = self._compute_digest(connection)
        self._meta_set("content_digest", digest)
        return digest

    def verify_integrity(self) -> str:
        with self._lock:
            stored = self._meta_get("content_digest")
            actual = self._compute_digest()
            if stored != actual:
                fail(
                    "E_LEDGER",
                    "ledger.content_digest",
                    f"stored {stored!r} does not match computed {actual}",
                )
            return actual

    @property
    def content_digest(self) -> str:
        with self._lock:
            return self.verify_integrity()

    @property
    def format_version(self) -> int:
        with self._lock:
            return int(self._meta_get("format_version") or "0")

    @property
    def snapshot_version(self) -> int:
        with self._lock:
            return int(self._meta_get("snapshot_version") or "1")

    @property
    def plan_fingerprint(self) -> str:
        with self._lock:
            return self._meta_get("plan_fingerprint") or ""

    def batch_versions(self) -> dict[str, int]:
        with self._lock:
            return {
                str(row["batch_id"]): int(row["batch_version"])
                for row in self.connection.execute(
                    "SELECT batch_id, batch_version FROM batches"
                ).fetchall()
            }

    def committed_batch_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                str(row["batch_id"])
                for row in self.connection.execute(
                    "SELECT batch_id FROM batches ORDER BY worker_id, batch_seq, batch_id"
                ).fetchall()
            )

    def begin_session(
        self,
        *,
        run_id: str,
        plan_fingerprint: str,
        spec_fingerprint: str,
        receipt_format: str,
        total_count: int,
        reset: bool = False,
        new_generation: bool = True,
    ) -> None:
        with self._lock:
            connection = self.connection
            session = self._session_row()
            if session["state"] == "running" and not reset:
                fail(
                    "E_LEDGER",
                    "ledger.session",
                    "a running session exists; use recover or an explicit rebuild",
                )
            with connection:
                if reset:
                    connection.execute("DELETE FROM traces")
                    connection.execute("DELETE FROM assignments")
                    connection.execute("DELETE FROM batches")
                snapshot_version = int(self._meta_get("snapshot_version") or "1")
                if new_generation:
                    snapshot_version += 1
                    self._meta_set("snapshot_version", str(snapshot_version))
                self._meta_set("plan_fingerprint", plan_fingerprint)
                self._meta_set("spec_fingerprint", spec_fingerprint)
                self._meta_set("receipt_format", receipt_format)
                now_ns = time.time_ns()
                connection.execute(
                    "UPDATE session SET run_id=?, state='running', started_ns=?, "
                    "finished_ns=NULL, exit_report_json=NULL WHERE id=1",
                    (run_id, now_ns),
                )
                completed = int(
                    connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
                )
                connection.execute(
                    "UPDATE progress SET completed_count=?, total_count=?, updated_run_id=? WHERE id=1",
                    (completed, total_count, run_id),
                )
                self._refresh_digest(connection)

    def resume_session(self, run_id: str, total_count: int) -> None:
        with self._lock:
            current = self._session_row()
            if current["state"] != "running":
                fail(
                    "E_LEDGER",
                    "ledger.session",
                    f"session is {current['state']}, not running",
                )
            with self.connection:
                self.connection.execute(
                    "UPDATE session SET run_id=? WHERE id=1", (run_id,)
                )
                self.connection.execute(
                    "UPDATE progress SET total_count=?, updated_run_id=? WHERE id=1",
                    (total_count, run_id),
                )
                self._refresh_digest(self.connection)

    def commit_batch(
        self,
        batch: LedgerBatch,
        traces: Sequence[LedgerTrace],
    ) -> int:
        with self._lock:
            connection = self.connection
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO batches(
                        batch_id, worker_id, batch_seq, batch_version,
                        request_ids_json, arrival_seqs_json, param_version,
                        param_digest, output_digest, receipt_digest, elapsed_ms,
                        wait_start_ms, wait_end_ms, run_start_ms, run_end_ms,
                        status, run_id, worker_pid
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        batch.batch_id,
                        batch.worker_id,
                        batch.batch_seq,
                        batch.batch_version,
                        canonical_json(list(batch.request_ids)),
                        canonical_json(list(batch.arrival_seqs)),
                        batch.param_version,
                        batch.param_digest,
                        batch.output_digest,
                        batch.receipt_digest,
                        batch.elapsed_ms,
                        batch.wait_start_ms,
                        batch.wait_end_ms,
                        batch.run_start_ms,
                        batch.run_end_ms,
                        batch.status,
                        batch.run_id,
                        batch.worker_pid,
                    ),
                )
                for assignment in batch.assignments:
                    connection.execute(
                        """
                        INSERT INTO assignments(
                            batch_id, request_id, rank, duplicate_refs_json,
                            vector_digest, conclusion_digest, receipt_json
                        ) VALUES(?,?,?,?,?,?,?)
                        """,
                        (
                            batch.batch_id,
                            assignment.request_id,
                            assignment.rank,
                            canonical_json(list(assignment.duplicate_refs)),
                            assignment.vector_digest,
                            assignment.conclusion_digest,
                            canonical_json(assignment.receipt_json),
                        ),
                    )
                for trace in traces:
                    self._insert_trace(trace)
                completed = int(
                    connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
                )
                connection.execute(
                    "UPDATE progress SET completed_count=?, updated_run_id=? WHERE id=1",
                    (completed, batch.run_id),
                )
                connection.execute(
                    "UPDATE session SET state='running', run_id=? WHERE id=1",
                    (batch.run_id,),
                )
                self._refresh_digest(connection)
                connection.commit()
                return completed
            except BaseException:
                connection.rollback()
                raise

    def _insert_trace(self, trace: LedgerTrace) -> None:
        self.connection.execute(
            """
            INSERT INTO traces(
                run_id, worker_id, batch_id, reason,
                dependency_path_json, request_ids_json, meta_json
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                trace.run_id,
                trace.worker_id,
                trace.batch_id,
                trace.reason,
                canonical_json(list(trace.dependency_path)),
                canonical_json(list(trace.request_ids)),
                canonical_json(dict(trace.meta)),
            ),
        )

    def prune_for_rebuild(
        self,
        *,
        keep_batch_ids: Iterable[str],
        run_id: str,
        plan_fingerprint: str,
        spec_fingerprint: str,
        receipt_format: str,
        total_count: int,
        removed_traces: Sequence[LedgerTrace] = (),
    ) -> tuple[str, ...]:
        keep = set(keep_batch_ids)
        with self._lock:
            connection = self.connection
            current_ids = set(self.committed_batch_ids())
            removed = tuple(sorted(current_ids - keep))
            with connection:
                if removed:
                    placeholders = ",".join("?" for _ in removed)
                    connection.execute(
                        f"DELETE FROM batches WHERE batch_id IN ({placeholders})",
                        removed,
                    )
                for trace in removed_traces:
                    self._insert_trace(trace)
                snapshot_version = int(self._meta_get("snapshot_version") or "1") + 1
                self._meta_set("snapshot_version", str(snapshot_version))
                self._meta_set("plan_fingerprint", plan_fingerprint)
                self._meta_set("spec_fingerprint", spec_fingerprint)
                self._meta_set("receipt_format", receipt_format)
                completed = int(
                    connection.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
                )
                connection.execute(
                    "UPDATE progress SET completed_count=?, total_count=?, updated_run_id=? WHERE id=1",
                    (completed, total_count, run_id),
                )
                connection.execute(
                    "UPDATE session SET run_id=?, state='running', started_ns=?, "
                    "finished_ns=NULL, exit_report_json=NULL WHERE id=1",
                    (run_id, time.time_ns()),
                )
                self._refresh_digest(connection)
            return removed

    def finalize_session(self, exit_report: Mapping[str, Any]) -> None:
        with self._lock:
            with self.connection:
                self.connection.execute(
                    "UPDATE session SET state='completed', finished_ns=?, exit_report_json=? WHERE id=1",
                    (time.time_ns(), canonical_json(dict(exit_report))),
                )
                self._refresh_digest(self.connection)

    def update_exit_report(self, exit_report: Mapping[str, Any]) -> None:
        """Update the report envelope after finalize without changing data digest."""

        with self._lock:
            session = self._session_row()
            if session["state"] != "completed":
                fail("E_LEDGER", "ledger.session", "exit report requires a completed session")
            with self.connection:
                self.connection.execute(
                    "UPDATE session SET exit_report_json=? WHERE id=1",
                    (canonical_json(dict(exit_report)),),
                )
            self.verify_integrity()

    def recovery_status(self) -> dict[str, Any]:
        with self._lock:
            session = self._session_row()
            progress = self._progress_row()
            committed = set(self.committed_batch_ids())
            state = str(session["state"])
            report = {
                "session_state": state,
                "prior_unclean_termination": state == "running",
                "can_emit_previous_exit_report": state == "completed"
                and session.get("exit_report_json") is not None,
                "committed_batch_ids": tuple(sorted(committed)),
                "completed_count": int(progress["completed_count"]),
                "total_count": int(progress["total_count"]),
                "exit_report": _decode_json(session.get("exit_report_json"), None),
            }
            return report

    def unknown_plan_batches(self, planned_batch_ids: Iterable[str]) -> tuple[str, ...]:
        with self._lock:
            committed = set(self.committed_batch_ids())
            return tuple(sorted(set(planned_batch_ids) - committed))

    def batches_as_dict(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {str(row["batch_id"]): row for row in self._batch_rows()}

    def traces_as_list(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._trace_rows()

    def corrupt_content_digest_for_injection(self) -> "None":
        """Deliberately create an E_LEDGER digest mismatch for acceptance tests."""

        with self._lock:
            with self.connection:
                self._meta_set("content_digest", "0" * 64)
            self.verify_integrity()

    def recompute_content_digest(self) -> str:
        """Explicit maintenance hook used by integrity tooling and tests."""

        with self._lock:
            with self.connection:
                digest = self._refresh_digest(self.connection)
            return digest

    def _state_for_snapshot(self) -> dict[str, Any]:
        meta = self._meta_map()
        return {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "snapshot_version": int(meta.get("snapshot_version", "1")),
            "ledger_format_version": int(meta.get("format_version", "1")),
            "content_digest": meta.get("content_digest", ""),
            "meta": {
                "plan_fingerprint": meta.get("plan_fingerprint", ""),
                "spec_fingerprint": meta.get("spec_fingerprint", ""),
                "receipt_format": meta.get("receipt_format", ""),
            },
            "session": self._session_row(),
            "progress": self._progress_row(),
            "batches": self._batch_rows(),
            "traces": self._trace_rows(),
        }

    def export_snapshot(self, path: str | Path) -> dict[str, Any]:
        with self._lock:
            self.verify_integrity()
            snapshot = self._state_for_snapshot()
            atomic_write_json(path, snapshot)
            return snapshot

    @staticmethod
    def load_snapshot(path: str | Path) -> dict[str, Any]:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                snapshot = json.load(handle)
        except FileNotFoundError:
            fail("E_IO", "snapshot", f"file not found: {path}")
        except json.JSONDecodeError as exc:
            fail("E_LEDGER", "snapshot", f"invalid JSON: {exc.msg}")
        except OSError as exc:
            fail("E_IO", "snapshot", f"cannot read {path}: {exc.strerror}")
        if not isinstance(snapshot, dict):
            fail("E_LEDGER", "snapshot", "expected an object")
        return snapshot

    @staticmethod
    def _validate_snapshot(snapshot: Mapping[str, Any]) -> str:
        if type(snapshot.get("snapshot_schema_version")) is not int:
            fail("E_LEDGER", "snapshot.snapshot_schema_version", "expected an integer")
        if snapshot["snapshot_schema_version"] != SNAPSHOT_SCHEMA_VERSION:
            fail(
                "E_LEDGER",
                "snapshot.snapshot_schema_version",
                f"unsupported schema version {snapshot['snapshot_schema_version']}",
            )
        if type(snapshot.get("snapshot_version")) is not int or snapshot["snapshot_version"] < 0:
            fail("E_LEDGER", "snapshot.snapshot_version", "expected a non-negative integer")
        if snapshot.get("ledger_format_version") != LEDGER_FORMAT_VERSION:
            fail(
                "E_LEDGER",
                "snapshot.ledger_format_version",
                f"unsupported ledger format version {snapshot.get('ledger_format_version')}",
            )
        for key in ("meta", "session", "progress"):
            if not isinstance(snapshot.get(key), dict):
                fail("E_LEDGER", f"snapshot.{key}", "expected an object")
        for key in ("batches", "traces"):
            if not isinstance(snapshot.get(key), list):
                fail("E_LEDGER", f"snapshot.{key}", "expected an array")
        expected = digest_json(
            _logical_content(
                meta=snapshot["meta"],
                session=snapshot["session"],
                progress=snapshot["progress"],
                batches=snapshot["batches"],
                traces=snapshot["traces"],
            )
        )
        if snapshot.get("content_digest") != expected:
            fail(
                "E_LEDGER",
                "snapshot.content_digest",
                f"declared {snapshot.get('content_digest')!r} does not match computed {expected}",
            )
        return expected

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> tuple[str, ...]:
        """Whole replacement, never an incremental merge."""

        incoming_digest = self._validate_snapshot(snapshot)
        with self._lock:
            self.verify_integrity()
            current_ids = set(self.committed_batch_ids())
            incoming_ids = {
                str(batch["batch_id"])
                for batch in snapshot["batches"]
                if isinstance(batch, dict) and "batch_id" in batch
            }
            replaced = tuple(sorted(current_ids | incoming_ids))
            current_version = self.snapshot_version
            current_digest = self.content_digest
            if (
                int(snapshot["snapshot_version"]) == current_version
                and incoming_digest == current_digest
            ):
                return ()

            connection = self.connection
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM traces")
                connection.execute("DELETE FROM assignments")
                connection.execute("DELETE FROM batches")
                for batch in snapshot["batches"]:
                    connection.execute(
                        """
                        INSERT INTO batches(
                            batch_id, worker_id, batch_seq, batch_version,
                            request_ids_json, arrival_seqs_json, param_version,
                            param_digest, output_digest, receipt_digest, elapsed_ms,
                            wait_start_ms, wait_end_ms, run_start_ms, run_end_ms,
                            status, run_id, worker_pid
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            batch["batch_id"],
                            batch["worker_id"],
                            batch["batch_seq"],
                            batch["batch_version"],
                            canonical_json(batch["request_ids"]),
                            canonical_json(batch["arrival_seqs"]),
                            batch["param_version"],
                            batch["param_digest"],
                            batch["output_digest"],
                            batch["receipt_digest"],
                            batch.get("elapsed_ms", 0),
                            batch.get("wait_start_ms", 0.0),
                            batch.get("wait_end_ms", 0.0),
                            batch.get("run_start_ms", 0.0),
                            batch.get("run_end_ms", 0.0),
                            batch.get("status", "committed"),
                            batch.get("run_id", ""),
                            batch.get("worker_pid", 0),
                        ),
                    )
                    for assignment in batch.get("assignments", []):
                        connection.execute(
                            """
                            INSERT INTO assignments(
                                batch_id, request_id, rank, duplicate_refs_json,
                                vector_digest, conclusion_digest, receipt_json
                            ) VALUES(?,?,?,?,?,?,?)
                            """,
                            (
                                batch["batch_id"],
                                assignment["request_id"],
                                assignment["rank"],
                                canonical_json(assignment.get("duplicate_refs", [])),
                                assignment["vector_digest"],
                                assignment["conclusion_digest"],
                                canonical_json(assignment.get("receipt_json", {})),
                            ),
                        )
                for trace in snapshot["traces"]:
                    connection.execute(
                        """
                        INSERT INTO traces(
                            run_id, worker_id, batch_id, reason,
                            dependency_path_json, request_ids_json, meta_json
                        ) VALUES(?,?,?,?,?,?,?)
                        """,
                        (
                            trace["run_id"],
                            trace["worker_id"],
                            trace["batch_id"],
                            trace["reason"],
                            canonical_json(trace.get("dependency_path", [])),
                            canonical_json(trace.get("request_ids", [])),
                            canonical_json(trace.get("meta", {})),
                        ),
                    )
                session = snapshot["session"]
                progress = snapshot["progress"]
                connection.execute(
                    "UPDATE session SET run_id=?, state=?, started_ns=?, finished_ns=?, "
                    "exit_report_json=? WHERE id=1",
                    (
                        session.get("run_id", ""),
                        session.get("state", "empty"),
                        session.get("started_ns"),
                        session.get("finished_ns"),
                        session.get("exit_report_json"),
                    ),
                )
                connection.execute(
                    "UPDATE progress SET completed_count=?, total_count=?, updated_run_id=? WHERE id=1",
                    (
                        int(progress.get("completed_count", 0)),
                        int(progress.get("total_count", 0)),
                        progress.get("updated_run_id", ""),
                    ),
                )
                self._meta_set("format_version", str(LEDGER_FORMAT_VERSION))
                self._meta_set("snapshot_version", str(int(snapshot["snapshot_version"])))
                self._meta_set("plan_fingerprint", snapshot["meta"].get("plan_fingerprint", ""))
                self._meta_set("spec_fingerprint", snapshot["meta"].get("spec_fingerprint", ""))
                self._meta_set("receipt_format", snapshot["meta"].get("receipt_format", ""))
                self._meta_set("content_digest", incoming_digest)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            self.verify_integrity()
            return replaced
