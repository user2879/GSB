"""SQLite ledger with transactionally coupled batch observations and cursors."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_hex
from .errors import WorkbenchError
from .logs import JsonlLogger, utc_now
from .models import BatchObservation, PreparedSpec, RunResult, StageComputation

SCHEMA_VERSION = 1


class Ledger:
    def __init__(self, db_path: str | Path, logger: JsonlLogger | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.initialize()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def _tx(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def initialize(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS specs (
            spec_version INTEGER PRIMARY KEY,
            fingerprint TEXT NOT NULL UNIQUE,
            dataset_digest TEXT NOT NULL,
            pipeline_digest TEXT NOT NULL,
            fixed_scale INTEGER NOT NULL,
            sharding_json TEXT NOT NULL,
            dataset_json TEXT NOT NULL,
            pipeline_json TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            spec_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            dataset_digest TEXT NOT NULL,
            normalize_scale INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            status TEXT NOT NULL,
            base_run_id TEXT,
            base_spec_version INTEGER,
            base_fingerprint TEXT,
            crash_after_stage TEXT,
            command_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            completed_at TEXT,
            final_data_digest TEXT
        );
        CREATE TABLE IF NOT EXISTS stage_runs (
            run_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            ordinal INTEGER NOT NULL,
            stage_type TEXT NOT NULL,
            stage_version INTEGER NOT NULL,
            spec_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            status TEXT NOT NULL,
            input_digest TEXT,
            output_digest TEXT,
            row_count INTEGER,
            error_code TEXT,
            started_at TEXT NOT NULL,
            committed_at TEXT,
            PRIMARY KEY (run_id, stage_id, attempt)
        );
        CREATE TABLE IF NOT EXISTS stage_batches (
            run_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            batch_index INTEGER NOT NULL,
            start_row INTEGER NOT NULL,
            end_row INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            input_digest TEXT NOT NULL,
            output_digest TEXT NOT NULL,
            spec_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            trusted INTEGER NOT NULL DEFAULT 1,
            committed_at TEXT NOT NULL,
            PRIMARY KEY (run_id, stage_id, attempt, batch_index)
        );
        CREATE TABLE IF NOT EXISTS stage_progress (
            run_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            batch_index INTEGER NOT NULL,
            rows_done INTEGER NOT NULL,
            state TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, stage_id, attempt)
        );
        CREATE TABLE IF NOT EXISTS shards (
            run_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            stage_attempt INTEGER NOT NULL,
            shard_index INTEGER NOT NULL,
            start_row INTEGER NOT NULL,
            end_row INTEGER NOT NULL,
            row_count INTEGER NOT NULL,
            rows_digest TEXT NOT NULL,
            shard_fingerprint TEXT NOT NULL,
            source_output_digest TEXT NOT NULL,
            source_stage_version INTEGER NOT NULL,
            spec_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            reused_from_spec_version INTEGER,
            trusted INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, stage_id, stage_attempt, shard_index)
        );
        CREATE TABLE IF NOT EXISTS conclusion_views (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            run_id TEXT NOT NULL,
            spec_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            view_json TEXT NOT NULL,
            view_digest TEXT NOT NULL,
            replaced_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conclusion_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            spec_version INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            view_json TEXT NOT NULL,
            view_digest TEXT NOT NULL,
            replaced_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evolution_reports (
            run_id TEXT PRIMARY KEY,
            old_spec_version INTEGER,
            new_spec_version INTEGER NOT NULL,
            old_fingerprint TEXT,
            new_fingerprint TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS business_errors (
            error_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            code TEXT NOT NULL,
            stage_id TEXT,
            param_name TEXT,
            param_value_json TEXT,
            affected_rows INTEGER,
            location TEXT,
            message TEXT NOT NULL,
            details_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            event TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS reports (
            run_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, kind)
        );
        CREATE INDEX IF NOT EXISTS idx_stage_runs_run ON stage_runs(run_id, ordinal, attempt);
        CREATE INDEX IF NOT EXISTS idx_batches_run_stage ON stage_batches(run_id, stage_id, attempt, batch_index);
        CREATE INDEX IF NOT EXISTS idx_shards_run ON shards(run_id, stage_id, stage_attempt, shard_index);
        CREATE INDEX IF NOT EXISTS idx_errors_run ON business_errors(run_id, error_id);
        CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, event_id);
        """
        with self._lock:
            self.conn.executescript(ddl)
            self.conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    def allocate_spec(self, prepared: PreparedSpec) -> tuple[int, bool, dict[str, Any] | None]:
        """Allocate a monotonic revision or return the existing revision.

        The explicit BEGIN IMMEDIATE is the concurrency boundary for two
        processes loading the same fingerprint at once.
        """

        with self._tx(immediate=True) as conn:
            existing = conn.execute(
                "SELECT * FROM specs WHERE fingerprint = ?", (prepared.fingerprint,)
            ).fetchone()
            if existing is not None:
                current = conn.execute(
                    "SELECT * FROM conclusion_views WHERE singleton = 1"
                ).fetchone()
                current_payload = dict(current) if current is not None else None
                if current_payload is not None:
                    current_payload["view"] = json.loads(current_payload["view_json"])
                return int(existing["spec_version"]), False, current_payload
            version = int(
                conn.execute("SELECT COALESCE(MAX(spec_version), 0) + 1 FROM specs").fetchone()[0]
            )
            now = utc_now()
            conn.execute(
                """
                INSERT INTO specs(
                    spec_version, fingerprint, dataset_digest, pipeline_digest,
                    fixed_scale, sharding_json, dataset_json, pipeline_json,
                    plan_json, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    version,
                    prepared.fingerprint,
                    prepared.dataset_digest,
                    prepared.pipeline_digest,
                    prepared.pipeline.fixed_scale,
                    canonical_json(prepared.pipeline.sharding.as_dict()),
                    canonical_json(prepared.dataset.canonical),
                    canonical_json(prepared.pipeline.canonical),
                    canonical_json([stage.as_dict() for stage in prepared.pipeline.stages]),
                    now,
                ),
            )
            current = conn.execute(
                "SELECT * FROM conclusion_views WHERE singleton = 1"
            ).fetchone()
            current_payload = dict(current) if current is not None else None
            if current_payload is not None:
                current_payload["view"] = json.loads(current_payload["view_json"])
            return version, True, current_payload

    def create_run(
        self,
        run_id: str,
        prepared: PreparedSpec,
        spec_version: int,
        batch_size: int,
        normalize_scale: int,
        base_view: dict[str, Any] | None,
        command: dict[str, Any],
        crash_after_stage: str | None = None,
    ) -> None:
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO runs(
                    run_id, spec_version, fingerprint, dataset_digest,
                    normalize_scale, batch_size, status, base_run_id,
                    base_spec_version, base_fingerprint, crash_after_stage,
                    command_json, started_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    spec_version,
                    prepared.fingerprint,
                    prepared.dataset_digest,
                    normalize_scale,
                    batch_size,
                    "RUNNING",
                    base_view["run_id"] if base_view else None,
                    base_view["spec_version"] if base_view else None,
                    base_view["fingerprint"] if base_view else None,
                    crash_after_stage,
                    canonical_json(command),
                    now,
                    now,
                ),
            )

    def start_stage(
        self,
        run_id: str,
        stage: Any,
        attempt: int,
        spec_version: int,
        fingerprint: str,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO stage_runs(
                    run_id, stage_id, attempt, ordinal, stage_type,
                    stage_version, spec_version, fingerprint, status, started_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    stage.id,
                    attempt,
                    stage.ordinal,
                    stage.type,
                    stage.version,
                    spec_version,
                    fingerprint,
                    "RUNNING",
                    utc_now(),
                ),
            )

    def commit_batch(
        self,
        run_id: str,
        stage_id: str,
        attempt: int,
        observation: BatchObservation,
        spec_version: int,
        fingerprint: str,
    ) -> None:
        """Commit one observation and its progress cursor atomically."""

        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO stage_batches(
                    run_id, stage_id, attempt, batch_index, start_row, end_row,
                    row_count, input_digest, output_digest, spec_version,
                    fingerprint, trusted, committed_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    stage_id,
                    attempt,
                    observation.batch_index,
                    observation.start_row,
                    observation.end_row,
                    observation.row_count,
                    observation.input_digest,
                    observation.output_digest,
                    spec_version,
                    fingerprint,
                    1 if attempt > 0 else 0,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO stage_progress(
                    run_id, stage_id, attempt, batch_index, rows_done, state, updated_at
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(run_id, stage_id, attempt) DO UPDATE SET
                    batch_index=excluded.batch_index,
                    rows_done=excluded.rows_done,
                    state=excluded.state,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    stage_id,
                    attempt,
                    observation.batch_index,
                    observation.end_row,
                    "BATCH_COMMITTED",
                    now,
                ),
            )

    def commit_stage(
        self,
        run_id: str,
        computation: StageComputation,
        attempt: int,
        spec_version: int,
        fingerprint: str,
        shards: list[dict[str, Any]],
        reused_versions: dict[int, int] | None = None,
    ) -> None:
        reused_versions = reused_versions or {}
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE stage_runs SET status='COMMITTED', input_digest=?,
                    output_digest=?, row_count=?, committed_at=?
                WHERE run_id=? AND stage_id=? AND attempt=?
                """,
                (
                    computation.input_digest,
                    computation.output_digest,
                    computation.row_count,
                    now,
                    run_id,
                    computation.stage.id,
                    attempt,
                ),
            )
            conn.execute(
                """
                UPDATE stage_progress SET state='COMMITTED', updated_at=?
                WHERE run_id=? AND stage_id=? AND attempt=?
                """,
                (now, run_id, computation.stage.id, attempt),
            )
            for shard in shards:
                index = int(shard["shard_index"])
                conn.execute(
                    """
                    INSERT INTO shards(
                        run_id, stage_id, stage_attempt, shard_index, start_row,
                        end_row, row_count, rows_digest, shard_fingerprint,
                        source_output_digest, source_stage_version, spec_version,
                        fingerprint, reused_from_spec_version, trusted, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        computation.stage.id,
                        attempt,
                        index,
                        shard["start_row"],
                        shard["end_row"],
                        shard["row_count"],
                        shard["rows_digest"],
                        shard["shard_fingerprint"],
                        shard["source_output_digest"],
                        shard["source_stage_version"],
                        spec_version,
                        fingerprint,
                        reused_versions.get(index),
                        1,
                        now,
                    ),
                )

    def fail_stage(
        self,
        run_id: str,
        stage_id: str,
        attempt: int,
        code: str,
        mark_run_failed: bool = True,
    ) -> None:
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE stage_runs SET status='FAILED', error_code=?, committed_at=? "
                "WHERE run_id=? AND stage_id=? AND attempt=?",
                (code, now, run_id, stage_id, attempt),
            )
            conn.execute(
                "UPDATE stage_progress SET state='FAILED', updated_at=? "
                "WHERE run_id=? AND stage_id=? AND attempt=?",
                (now, run_id, stage_id, attempt),
            )
            if mark_run_failed:
                conn.execute(
                    "UPDATE runs SET status='FAILED', updated_at=?, completed_at=? WHERE run_id=?",
                    (now, now, run_id),
                )

    def complete_run(
        self,
        run_id: str,
        status: str,
        view: dict[str, Any],
        evolution: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        view_json = canonical_json(view)
        view_digest = sha256_hex(b"MLPIPE-VIEW1\0" + view_json.encode("utf-8"))
        with self._tx(immediate=True) as conn:
            conn.execute(
                "UPDATE runs SET status=?, final_data_digest=?, updated_at=?, completed_at=? "
                "WHERE run_id=?",
                (status, view["final_data_digest"], now, now, run_id),
            )
            conn.execute("DELETE FROM conclusion_views WHERE singleton=1")
            conn.execute(
                """
                INSERT INTO conclusion_views(
                    singleton, run_id, spec_version, fingerprint, view_json,
                    view_digest, replaced_at
                ) VALUES(1,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    view["spec_version"],
                    view["fingerprint"],
                    view_json,
                    view_digest,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO conclusion_history(
                    run_id, spec_version, fingerprint, view_json, view_digest, replaced_at
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    run_id,
                    view["spec_version"],
                    view["fingerprint"],
                    view_json,
                    view_digest,
                    now,
                ),
            )
            if evolution is not None:
                conn.execute(
                    """
                    INSERT INTO evolution_reports(
                        run_id, old_spec_version, new_spec_version, old_fingerprint,
                        new_fingerprint, payload_json, created_at
                    ) VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        old_spec_version=excluded.old_spec_version,
                        new_spec_version=excluded.new_spec_version,
                        old_fingerprint=excluded.old_fingerprint,
                        new_fingerprint=excluded.new_fingerprint,
                        payload_json=excluded.payload_json,
                        created_at=excluded.created_at
                    """,
                    (
                        run_id,
                        evolution.get("old_spec_version"),
                        view["spec_version"],
                        evolution.get("old_fingerprint"),
                        view["fingerprint"],
                        canonical_json(evolution),
                        now,
                    ),
                )

    def record_error(self, run_id: str | None, error: WorkbenchError) -> None:
        payload = error.to_dict()
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO business_errors(
                    run_id, code, stage_id, param_name, param_value_json,
                    affected_rows, location, message, details_json, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    error.code,
                    error.stage_id,
                    error.param,
                    canonical_json(error.param_value) if error.param_value is not None else None,
                    error.affected_rows,
                    error.location,
                    error.message,
                    canonical_json(error.details) if error.details else None,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO events(run_id, event, payload_json, created_at) VALUES(?,?,?,?)",
                (run_id, error.code, canonical_json(payload), now),
            )
        if self.logger is not None:
            log_payload = {"run_id": run_id, **payload}
            self.logger.write(error.code, log_payload)

    def record_event(self, run_id: str | None, event: str, payload: dict[str, Any]) -> None:
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO events(run_id, event, payload_json, created_at) VALUES(?,?,?,?)",
                (run_id, event, canonical_json(payload), now),
            )
        if self.logger is not None:
            self.logger.write(event, {"run_id": run_id, **payload})

    def record_report(self, run_id: str, kind: str, payload: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO reports(run_id, kind, payload_json, created_at)
                VALUES(?,?,?,?)
                ON CONFLICT(run_id, kind) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    created_at=excluded.created_at
                """,
                (run_id, kind, canonical_json(payload), utc_now()),
            )

    def mark_stage_unknown(self, run_id: str, stage_id: str, attempt: int) -> None:
        now = utc_now()
        with self._tx() as conn:
            conn.execute(
                "UPDATE stage_runs SET status='UNKNOWN' WHERE run_id=? AND stage_id=? AND attempt=?",
                (run_id, stage_id, attempt),
            )
            conn.execute(
                "UPDATE stage_batches SET trusted=0 WHERE run_id=? AND stage_id=? AND attempt=?",
                (run_id, stage_id, attempt),
            )
            conn.execute(
                "UPDATE stage_progress SET state='UNKNOWN', updated_at=? "
                "WHERE run_id=? AND stage_id=? AND attempt=?",
                (now, run_id, stage_id, attempt),
            )
            conn.execute(
                "UPDATE runs SET status='INTERRUPTED', updated_at=? WHERE run_id=?",
                (now, run_id),
            )

    def update_run_status(self, run_id: str, status: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "UPDATE runs SET status=?, updated_at=? WHERE run_id=?",
                (status, utc_now(), run_id),
            )

    def get_spec(self, spec_version: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM specs WHERE spec_version=?", (spec_version,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_spec_by_fingerprint(self, fingerprint: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM specs WHERE fingerprint=?", (fingerprint,)
            ).fetchone()
        return dict(row) if row is not None else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def current_conclusion(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM conclusion_views WHERE singleton=1"
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["view"] = json.loads(payload["view_json"])
        return payload

    def conclusion_for_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM conclusion_history WHERE run_id=? "
                "ORDER BY history_id DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["view"] = json.loads(payload["view_json"])
        return payload

    def stage_attempts(self, run_id: str) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM stage_runs WHERE run_id=? ORDER BY ordinal, stage_id, attempt",
                (run_id,),
            ).fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            payload = dict(row)
            result.setdefault(payload["stage_id"], []).append(payload)
        return result

    def latest_stage_run(self, run_id: str, stage_id: str) -> dict[str, Any] | None:
        attempts = self.stage_attempts(run_id).get(stage_id, [])
        return attempts[-1] if attempts else None

    def latest_stage_runs(self, run_id: str) -> list[dict[str, Any]]:
        attempts = self.stage_attempts(run_id)
        latest = [values[-1] for values in attempts.values() if values]
        latest.sort(key=lambda row: (row["ordinal"], row["stage_id"]))
        return latest

    def get_batches(
        self,
        run_id: str,
        stage_id: str,
        attempt: int,
        trusted_only: bool = True,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM stage_batches WHERE run_id=? AND stage_id=? AND attempt=?"
            + (" AND trusted=1" if trusted_only else "")
            + " ORDER BY batch_index"
        )
        with self._lock:
            rows = self.conn.execute(sql, (run_id, stage_id, attempt)).fetchall()
        return [dict(row) for row in rows]

    def get_progress(self, run_id: str, stage_id: str, attempt: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM stage_progress WHERE run_id=? AND stage_id=? AND attempt=?",
                (run_id, stage_id, attempt),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_shards(
        self,
        run_id: str,
        stage_id: str,
        attempt: int,
        trusted_only: bool = True,
    ) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM shards WHERE run_id=? AND stage_id=? AND stage_attempt=?"
            + (" AND trusted=1" if trusted_only else "")
            + " ORDER BY shard_index"
        )
        with self._lock:
            rows = self.conn.execute(sql, (run_id, stage_id, attempt)).fetchall()
        return [dict(row) for row in rows]

    def all_shards_for_spec(self, spec_version: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM shards WHERE spec_version=? AND trusted=1 "
                "ORDER BY stage_id, shard_index",
                (spec_version,),
            ).fetchall()
        return [dict(row) for row in rows]

    def shards_for_run(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM shards WHERE run_id=? AND trusted=1 "
                "ORDER BY stage_id, shard_index",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM runs ORDER BY run_seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_errors(self, run_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            if run_id is None:
                rows = self.conn.execute(
                    "SELECT * FROM business_errors ORDER BY error_id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM business_errors WHERE run_id=? ORDER BY error_id DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def list_events(self, run_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            if run_id is None:
                rows = self.conn.execute(
                    "SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY event_id DESC LIMIT ?",
                    (run_id, limit),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_report(self, run_id: str, kind: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM reports WHERE run_id=? AND kind=?", (run_id, kind)
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["payload"] = json.loads(payload["payload_json"])
        return payload

    def get_evolution_report(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM evolution_reports WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["payload"] = json.loads(payload["payload_json"])
        return payload

    def table_counts(self) -> dict[str, int]:
        tables = (
            "specs",
            "runs",
            "stage_runs",
            "stage_batches",
            "shards",
            "business_errors",
            "events",
            "conclusion_history",
        )
        with self._lock:
            return {
                table: int(self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }

    def run_result(self, run_id: str, elapsed_ms: int) -> RunResult:
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        stages = self.latest_stage_runs(run_id)
        shard_stage = next((stage for stage in stages if stage["stage_type"] == "shard_pack"), None)
        shards = (
            self.get_shards(run_id, shard_stage["stage_id"], shard_stage["attempt"])
            if shard_stage is not None
            else []
        )
        evolution = self.get_evolution_report(run_id)
        return RunResult(
            run_id=run_id,
            status=run["status"],
            spec_version=run["spec_version"],
            fingerprint=run["fingerprint"],
            dataset_digest=run["dataset_digest"],
            stage_statuses=[
                {
                    "ordinal": stage["ordinal"],
                    "stage_id": stage["stage_id"],
                    "type": stage["stage_type"],
                    "version": stage["stage_version"],
                    "status": stage["status"],
                    "row_count": stage["row_count"],
                    "output_digest": stage["output_digest"],
                }
                for stage in stages
            ],
            shards=[
                {
                    "shard_index": shard["shard_index"],
                    "start_row": shard["start_row"],
                    "end_row": shard["end_row"],
                    "row_count": shard["row_count"],
                    "shard_fingerprint": shard["shard_fingerprint"],
                    "source_stage_version": shard["source_stage_version"],
                    "reused_from_spec_version": shard["reused_from_spec_version"],
                }
                for shard in shards
            ],
            final_data_digest=run["final_data_digest"],
            elapsed_ms=elapsed_ms,
            evolution=evolution["payload"] if evolution else None,
        )
