"""SQLite persistence with atomic replay batches and strict invariants."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .constants import CLOCK_BASIS, SCHEMA_VERSION
from .errors import LogValidationError, ParameterError, SnapshotError
from .models import (
    AggregationSpec,
    MasterRecord,
    Progress,
    ReplayBatch,
    ReplayParams,
    ReplayRun,
    Sample,
)
from .util import canonical_json, sha256_json, utc_now_iso
from .validation import validate_progress


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS master_log (
    commit_seq INTEGER PRIMARY KEY CHECK (commit_seq > 0),
    txn_id TEXT NOT NULL UNIQUE,
    commit_ts_ms INTEGER NOT NULL CHECK (commit_ts_ms >= 0),
    txn_bytes INTEGER NOT NULL CHECK (txn_bytes > 0)
);

CREATE TRIGGER IF NOT EXISTS master_log_contiguous_before_insert
BEFORE INSERT ON master_log
WHEN NEW.commit_seq != COALESCE((SELECT MAX(commit_seq) FROM master_log), 0) + 1
BEGIN
    SELECT RAISE(ABORT, 'E_LOG commit_seq must be contiguous and start at 1');
END;

CREATE TABLE IF NOT EXISTS replay_runs (
    param_version TEXT PRIMARY KEY,
    rate_rps INTEGER NOT NULL CHECK (rate_rps > 0),
    concurrency INTEGER NOT NULL CHECK (concurrency > 0),
    clock_mode TEXT NOT NULL,
    clock_base_ms INTEGER NOT NULL CHECK (clock_base_ms >= 0),
    estimator_digest TEXT NOT NULL,
    estimator_json TEXT NOT NULL,
    params_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS replay_batches (
    batch_key INTEGER PRIMARY KEY AUTOINCREMENT,
    param_version TEXT NOT NULL REFERENCES replay_runs(param_version) ON DELETE CASCADE,
    batch_index INTEGER NOT NULL CHECK (batch_index >= 0),
    attempt INTEGER NOT NULL CHECK (attempt >= 1),
    first_seq INTEGER NOT NULL CHECK (first_seq > 0),
    last_seq INTEGER NOT NULL CHECK (last_seq >= first_seq),
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'COMPLETE', 'UNKNOWN')),
    started_mono_ns INTEGER NOT NULL,
    completed_mono_ns INTEGER,
    started_utc TEXT NOT NULL,
    completed_utc TEXT,
    notes TEXT NOT NULL DEFAULT '',
    UNIQUE (param_version, batch_index, attempt),
    FOREIGN KEY (first_seq) REFERENCES master_log(commit_seq),
    FOREIGN KEY (last_seq) REFERENCES master_log(commit_seq)
);

CREATE TABLE IF NOT EXISTS replay_samples (
    param_version TEXT NOT NULL,
    commit_seq INTEGER NOT NULL REFERENCES master_log(commit_seq),
    batch_key INTEGER NOT NULL REFERENCES replay_batches(batch_key),
    replay_completed_ns INTEGER NOT NULL,
    delay_ns INTEGER NOT NULL CHECK (delay_ns >= 0),
    fixed_delay INTEGER NOT NULL CHECK (fixed_delay >= 0),
    PRIMARY KEY (param_version, commit_seq),
    FOREIGN KEY (param_version) REFERENCES replay_runs(param_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS replay_progress (
    param_version TEXT PRIMARY KEY REFERENCES replay_runs(param_version) ON DELETE CASCADE,
    last_replayed_seq INTEGER NOT NULL CHECK (last_replayed_seq >= 0),
    last_completed_batch_key INTEGER REFERENCES replay_batches(batch_key),
    completed_batches INTEGER NOT NULL CHECK (completed_batches >= 0),
    status TEXT NOT NULL CHECK (
        status IN ('IN_PROGRESS', 'COMPLETE', 'REQUIRES_RERUN', 'UNKNOWN',
                   'RESTORED_STALE_LOG_MISMATCH')
    ),
    updated_utc TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS replay_progress_not_ahead_before_insert
BEFORE INSERT ON replay_progress
WHEN NEW.last_replayed_seq > COALESCE((SELECT MAX(commit_seq) FROM master_log), 0)
BEGIN
    SELECT RAISE(ABORT, 'E_LOG replay progress ahead of current log tail');
END;

CREATE TRIGGER IF NOT EXISTS replay_progress_not_ahead_before_update
BEFORE UPDATE OF last_replayed_seq ON replay_progress
WHEN NEW.last_replayed_seq > COALESCE((SELECT MAX(commit_seq) FROM master_log), 0)
BEGIN
    SELECT RAISE(ABORT, 'E_LOG replay progress ahead of current log tail');
END;

CREATE TABLE IF NOT EXISTS snapshot_restore_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_sha256 TEXT NOT NULL,
    snapshot_path TEXT NOT NULL,
    restored_utc TEXT NOT NULL,
    report_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_replay_batches_run_index
ON replay_batches(param_version, batch_index, attempt);
CREATE INDEX IF NOT EXISTS idx_replay_samples_run_seq
ON replay_samples(param_version, commit_seq);
"""


class Repository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=10.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 10000")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.initialize()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def initialize(self) -> None:
        self.connection.executescript(SCHEMA_SQL)
        with self.transaction() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('clock_basis', ?)",
                (CLOCK_BASIS,),
            )
        current = self.get_meta("schema_version")
        if current != str(SCHEMA_VERSION):
            raise ParameterError(
                "数据库结构版本不兼容",
                field="schema_version",
                details={"expected": SCHEMA_VERSION, "actual": current},
            )

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def replace_master_log(self, records: Sequence[MasterRecord]) -> dict[str, Any]:
        if not records:
            raise LogValidationError("主库日志不能为空", seq=0, field="records")
        digest = sha256_json([record.to_dict() for record in records])
        with self.transaction() as connection:
            connection.execute("DELETE FROM replay_samples")
            connection.execute("DELETE FROM replay_progress")
            connection.execute("DELETE FROM replay_batches")
            connection.execute("DELETE FROM replay_runs")
            connection.execute("DELETE FROM master_log")
            for record in records:
                connection.execute(
                    "INSERT INTO master_log(commit_seq, txn_id, commit_ts_ms, txn_bytes) "
                    "VALUES(?, ?, ?, ?)",
                    (record.commit_seq, record.txn_id, record.commit_ts_ms, record.txn_bytes),
                )
            self._set_meta(connection, "log_digest", digest)
            self._set_meta(connection, "log_count", str(len(records)))
            self._set_meta(connection, "log_first_seq", str(records[0].commit_seq))
            self._set_meta(connection, "log_last_seq", str(records[-1].commit_seq))
        return self.log_stats()

    @staticmethod
    def _set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_master_log(self) -> list[MasterRecord]:
        rows = self.connection.execute(
            "SELECT commit_seq, txn_id, commit_ts_ms, txn_bytes "
            "FROM master_log ORDER BY commit_seq"
        ).fetchall()
        return [
            MasterRecord(
                commit_seq=int(row["commit_seq"]),
                txn_id=str(row["txn_id"]),
                commit_ts_ms=int(row["commit_ts_ms"]),
                txn_bytes=int(row["txn_bytes"]),
            )
            for row in rows
        ]

    def log_stats(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count, MIN(commit_seq) AS first_seq, "
            "MAX(commit_seq) AS last_seq FROM master_log"
        ).fetchone()
        records = self.get_master_log()
        digest = sha256_json([record.to_dict() for record in records]) if records else None
        return {
            "count": int(row["count"]),
            "first_seq": row["first_seq"],
            "last_seq": row["last_seq"],
            "digest": digest,
            "clock_basis": CLOCK_BASIS,
        }

    def ensure_run(self, params: ReplayParams, spec: AggregationSpec) -> ReplayRun:
        params_json = canonical_json(params.binding_dict())
        now = utc_now_iso()
        row = self.connection.execute(
            "SELECT * FROM replay_runs WHERE param_version = ?", (params.version,)
        ).fetchone()
        if row is not None:
            if row["params_json"] != params_json or row["estimator_digest"] != spec.digest:
                raise ParameterError(
                    "参数版本摘要冲突",
                    field="param_version",
                    details={"param_version": params.version},
                )
            return self._run_from_row(row)
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO replay_runs(param_version, rate_rps, concurrency, clock_mode, "
                "clock_base_ms, estimator_digest, estimator_json, params_json, created_utc) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    params.version,
                    params.rate_rps,
                    params.concurrency,
                    params.clock_mode,
                    params.clock_base_ms,
                    spec.digest,
                    canonical_json(spec.to_dict()),
                    params_json,
                    now,
                ),
            )
        return ReplayRun(
            param_version=params.version,
            rate_rps=params.rate_rps,
            concurrency=params.concurrency,
            clock_mode=params.clock_mode,
            clock_base_ms=params.clock_base_ms,
            estimator_digest=spec.digest,
            estimator_json=canonical_json(spec.to_dict()),
            params_json=params_json,
            created_utc=now,
        )

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> ReplayRun:
        return ReplayRun(
            param_version=str(row["param_version"]),
            rate_rps=int(row["rate_rps"]),
            concurrency=int(row["concurrency"]),
            clock_mode=str(row["clock_mode"]),
            clock_base_ms=int(row["clock_base_ms"]),
            estimator_digest=str(row["estimator_digest"]),
            estimator_json=str(row["estimator_json"]),
            params_json=str(row["params_json"]),
            created_utc=str(row["created_utc"]),
        )

    def get_run(self, param_version: str) -> ReplayRun | None:
        row = self.connection.execute(
            "SELECT * FROM replay_runs WHERE param_version = ?", (param_version,)
        ).fetchone()
        return None if row is None else self._run_from_row(row)

    def list_runs(self) -> list[ReplayRun]:
        rows = self.connection.execute(
            "SELECT * FROM replay_runs ORDER BY created_utc, param_version"
        ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def progress_for(self, param_version: str) -> Progress | None:
        row = self.connection.execute(
            "SELECT * FROM replay_progress WHERE param_version = ?", (param_version,)
        ).fetchone()
        return None if row is None else self._progress_from_row(row)

    def list_progress(self) -> list[Progress]:
        rows = self.connection.execute(
            "SELECT * FROM replay_progress ORDER BY updated_utc, param_version"
        ).fetchall()
        return [self._progress_from_row(row) for row in rows]

    @staticmethod
    def _progress_from_row(row: sqlite3.Row) -> Progress:
        return Progress(
            param_version=str(row["param_version"]),
            last_replayed_seq=int(row["last_replayed_seq"]),
            last_completed_batch_key=(
                None
                if row["last_completed_batch_key"] is None
                else int(row["last_completed_batch_key"])
            ),
            completed_batches=int(row["completed_batches"]),
            status=str(row["status"]),
            updated_utc=str(row["updated_utc"]),
        )

    def next_attempt(self, param_version: str, batch_index: int) -> int:
        row = self.connection.execute(
            "SELECT MAX(attempt) AS attempt FROM replay_batches "
            "WHERE param_version = ? AND batch_index = ?",
            (param_version, batch_index),
        ).fetchone()
        return 1 if row is None or row["attempt"] is None else int(row["attempt"]) + 1

    def begin_batch(
        self,
        param_version: str,
        batch_index: int,
        attempt: int,
        first_seq: int,
        last_seq: int,
        started_mono_ns: int,
    ) -> int:
        now = utc_now_iso()
        with self.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO replay_batches(param_version, batch_index, attempt, first_seq, "
                "last_seq, status, started_mono_ns, completed_mono_ns, started_utc, "
                "completed_utc, notes) VALUES(?, ?, ?, ?, ?, 'RUNNING', ?, NULL, ?, NULL, '')",
                (param_version, batch_index, attempt, first_seq, last_seq, started_mono_ns, now),
            )
            batch_key = int(cursor.lastrowid)
            progress = self.progress_for(param_version)
            if progress is None:
                connection.execute(
                    "INSERT INTO replay_progress(param_version, last_replayed_seq, "
                    "last_completed_batch_key, completed_batches, status, updated_utc) "
                    "VALUES(?, 0, NULL, 0, 'IN_PROGRESS', ?)",
                    (param_version, now),
                )
            else:
                connection.execute(
                    "UPDATE replay_progress SET status = 'IN_PROGRESS', updated_utc = ? "
                    "WHERE param_version = ?",
                    (now, param_version),
                )
            return batch_key

    def complete_batch(
        self,
        *,
        param_version: str,
        batch_key: int,
        last_replayed_seq: int,
        completed_batches: int,
        completed_mono_ns: int,
        is_final: bool,
        last_log_seq: int,
        samples: Sequence[Sample],
    ) -> Progress:
        validate_progress(last_replayed_seq, last_log_seq)
        now = utc_now_iso()
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE replay_batches SET status = 'COMPLETE', completed_mono_ns = ?, "
                "completed_utc = ? WHERE batch_key = ? AND status = 'RUNNING'",
                (completed_mono_ns, now, batch_key),
            )
            if cursor.rowcount != 1:
                raise ParameterError(
                    "批次不处于 RUNNING 状态，无法完成", field="batch_key", details={"batch_key": batch_key}
                )
            for sample in samples:
                connection.execute(
                    "INSERT INTO replay_samples(param_version, commit_seq, batch_key, "
                    "replay_completed_ns, delay_ns, fixed_delay) VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        param_version,
                        sample.commit_seq,
                        batch_key,
                        sample.replay_completed_ns,
                        sample.delay_ns,
                        sample.fixed_delay,
                    ),
                )
            status = "COMPLETE" if is_final else "IN_PROGRESS"
            connection.execute(
                "INSERT INTO replay_progress(param_version, last_replayed_seq, "
                "last_completed_batch_key, completed_batches, status, updated_utc) "
                "VALUES(?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(param_version) DO UPDATE SET "
                "last_replayed_seq = excluded.last_replayed_seq, "
                "last_completed_batch_key = excluded.last_completed_batch_key, "
                "completed_batches = excluded.completed_batches, "
                "status = excluded.status, updated_utc = excluded.updated_utc",
                (
                    param_version,
                    last_replayed_seq,
                    batch_key,
                    completed_batches,
                    status,
                    now,
                ),
            )
        progress = self.progress_for(param_version)
        if progress is None:
            raise ParameterError("批次完成后进度缺失", field="replay_progress")
        return progress

    def mark_running_batches_unknown(self) -> list[ReplayBatch]:
        now = utc_now_iso()
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM replay_batches WHERE status = 'RUNNING' "
                "ORDER BY param_version, batch_index, attempt"
            ).fetchall()
            affected_versions = sorted({str(row["param_version"]) for row in rows})
            for row in rows:
                connection.execute(
                    "UPDATE replay_batches SET status = 'UNKNOWN', completed_utc = ?, "
                    "notes = 'process terminated before atomic batch commit; rerun required' "
                    "WHERE batch_key = ?",
                    (now, row["batch_key"]),
                )
            for version in affected_versions:
                connection.execute(
                    "UPDATE replay_progress SET status = 'REQUIRES_RERUN', updated_utc = ? "
                    "WHERE param_version = ?",
                    (now, version),
                )
            batch_keys = [int(row["batch_key"]) for row in rows]
            if not batch_keys:
                return []
            placeholders = ",".join("?" for _ in batch_keys)
            result_rows = connection.execute(
                f"SELECT * FROM replay_batches WHERE batch_key IN ({placeholders}) "
                "ORDER BY param_version, batch_index, attempt",
                batch_keys,
            ).fetchall()
            return [self._batch_from_row(row) for row in result_rows]

    @staticmethod
    def _batch_from_row(row: sqlite3.Row) -> ReplayBatch:
        return ReplayBatch(
            batch_key=int(row["batch_key"]),
            param_version=str(row["param_version"]),
            batch_index=int(row["batch_index"]),
            attempt=int(row["attempt"]),
            first_seq=int(row["first_seq"]),
            last_seq=int(row["last_seq"]),
            status=str(row["status"]),
            started_mono_ns=int(row["started_mono_ns"]),
            completed_mono_ns=(
                None if row["completed_mono_ns"] is None else int(row["completed_mono_ns"])
            ),
            started_utc=str(row["started_utc"]),
            completed_utc=(
                None if row["completed_utc"] is None else str(row["completed_utc"])
            ),
            notes=str(row["notes"]),
        )

    def get_batches(self, param_version: str) -> list[ReplayBatch]:
        rows = self.connection.execute(
            "SELECT * FROM replay_batches WHERE param_version = ? "
            "ORDER BY batch_index, attempt",
            (param_version,),
        ).fetchall()
        return [self._batch_from_row(row) for row in rows]

    def list_batches(self) -> list[ReplayBatch]:
        rows = self.connection.execute(
            "SELECT * FROM replay_batches ORDER BY param_version, batch_index, attempt"
        ).fetchall()
        return [self._batch_from_row(row) for row in rows]

    def get_samples(self, param_version: str) -> list[Sample]:
        rows = self.connection.execute(
            "SELECT param_version, commit_seq, batch_key, replay_completed_ns, delay_ns, "
            "fixed_delay FROM replay_samples WHERE param_version = ? ORDER BY commit_seq",
            (param_version,),
        ).fetchall()
        return [
            Sample(
                param_version=str(row["param_version"]),
                commit_seq=int(row["commit_seq"]),
                batch_id=int(row["batch_key"]),
                replay_completed_ns=int(row["replay_completed_ns"]),
                delay_ns=int(row["delay_ns"]),
                fixed_delay=int(row["fixed_delay"]),
            )
            for row in rows
        ]

    def list_samples(self) -> list[Sample]:
        rows = self.connection.execute(
            "SELECT param_version, commit_seq, batch_key, replay_completed_ns, delay_ns, "
            "fixed_delay FROM replay_samples ORDER BY param_version, commit_seq"
        ).fetchall()
        result: list[Sample] = []
        for row in rows:
            result.append(
                Sample(
                    param_version=str(row["param_version"]),
                    commit_seq=int(row["commit_seq"]),
                    batch_id=int(row["batch_key"]),
                    replay_completed_ns=int(row["replay_completed_ns"]),
                    delay_ns=int(row["delay_ns"]),
                    fixed_delay=int(row["fixed_delay"]),
                )
            )
        return result

    def replay_state_counts(self) -> dict[str, int]:
        tables = ("replay_runs", "replay_batches", "replay_samples", "replay_progress")
        return {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in tables
        }

    def clear_replay_state(self) -> dict[str, int]:
        before = self.replay_state_counts()
        with self.transaction() as connection:
            connection.execute("DELETE FROM replay_samples")
            connection.execute("DELETE FROM replay_progress")
            connection.execute("DELETE FROM replay_batches")
            connection.execute("DELETE FROM replay_runs")
        return before

    def replace_replay_state(
        self,
        *,
        run: ReplayRun,
        batches: Sequence[ReplayBatch],
        samples: Sequence[Sample],
        progress: Progress,
        status_override: str | None = None,
    ) -> None:
        last_log_seq = int(self.log_stats()["last_seq"] or 0)
        validate_progress(progress.last_replayed_seq, last_log_seq)
        log_seq_set = {record.commit_seq for record in self.get_master_log()}
        bad_samples = sorted(sample.commit_seq for sample in samples if sample.commit_seq not in log_seq_set)
        if bad_samples:
            raise SnapshotError(
                "快照样本引用了当前日志中不存在的提交序号",
                seq=bad_samples[0],
                field="samples.commit_seq",
            )
        with self.transaction() as connection:
            connection.execute("DELETE FROM replay_samples")
            connection.execute("DELETE FROM replay_progress")
            connection.execute("DELETE FROM replay_batches")
            connection.execute("DELETE FROM replay_runs")
            connection.execute(
                "INSERT INTO replay_runs(param_version, rate_rps, concurrency, clock_mode, "
                "clock_base_ms, estimator_digest, estimator_json, params_json, created_utc) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.param_version,
                    run.rate_rps,
                    run.concurrency,
                    run.clock_mode,
                    run.clock_base_ms,
                    run.estimator_digest,
                    run.estimator_json,
                    run.params_json,
                    run.created_utc,
                ),
            )
            for batch in batches:
                connection.execute(
                    "INSERT INTO replay_batches(batch_key, param_version, batch_index, attempt, "
                    "first_seq, last_seq, status, started_mono_ns, completed_mono_ns, "
                    "started_utc, completed_utc, notes) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        batch.batch_key,
                        batch.param_version,
                        batch.batch_index,
                        batch.attempt,
                        batch.first_seq,
                        batch.last_seq,
                        batch.status,
                        batch.started_mono_ns,
                        batch.completed_mono_ns,
                        batch.started_utc,
                        batch.completed_utc,
                        batch.notes,
                    ),
                )
            for sample in samples:
                connection.execute(
                    "INSERT INTO replay_samples(param_version, commit_seq, batch_key, "
                    "replay_completed_ns, delay_ns, fixed_delay) VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        sample.param_version,
                        sample.commit_seq,
                        sample.batch_id,
                        sample.replay_completed_ns,
                        sample.delay_ns,
                        sample.fixed_delay,
                    ),
                )
            connection.execute(
                "INSERT INTO replay_progress(param_version, last_replayed_seq, "
                "last_completed_batch_key, completed_batches, status, updated_utc) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                (
                    progress.param_version,
                    progress.last_replayed_seq,
                    progress.last_completed_batch_key,
                    progress.completed_batches,
                    status_override or progress.status,
                    progress.updated_utc,
                ),
            )

    def add_snapshot_restore_audit(
        self, snapshot_sha256: str, snapshot_path: str, report: dict[str, Any]
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO snapshot_restore_audit(snapshot_sha256, snapshot_path, "
                "restored_utc, report_json) VALUES(?, ?, ?, ?)",
                (snapshot_sha256, snapshot_path, utc_now_iso(), canonical_json(report)),
            )

    def list_snapshot_restore_audits(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT snapshot_sha256, snapshot_path, restored_utc, report_json "
            "FROM snapshot_restore_audit ORDER BY audit_id"
        ).fetchall()
        return [
            {
                "snapshot_sha256": str(row["snapshot_sha256"]),
                "snapshot_path": str(row["snapshot_path"]),
                "restored_utc": str(row["restored_utc"]),
                "report": json.loads(row["report_json"]),
            }
            for row in rows
        ]
