from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, sha256_obj, utc_now
from .execution import aggregate_state_json, merge_aggregate_maps, serialize_aggregate_map
from .model import Dataset, Query, RunConfig
from .scheduler import ScheduleJournal, TaskRecord


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                run_id TEXT PRIMARY KEY,
                parent_run_id TEXT,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                query_id TEXT NOT NULL,
                data_digest TEXT NOT NULL,
                table_digests_json TEXT NOT NULL,
                table_versions_json TEXT NOT NULL,
                budget_version INTEGER NOT NULL,
                budget_digest TEXT NOT NULL,
                config_digest TEXT NOT NULL,
                split_threshold INTEGER NOT NULL,
                split_parts INTEGER NOT NULL,
                row_group_size INTEGER NOT NULL,
                parallelism INTEGER NOT NULL,
                memory_budget_bytes INTEGER NOT NULL,
                effective_memory_budget_bytes INTEGER NOT NULL,
                split_strategy TEXT NOT NULL,
                skew_safety_limit INTEGER NOT NULL,
                injection_json TEXT NOT NULL,
                config_json TEXT NOT NULL,
                query_json TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                result_json TEXT,
                result_digest TEXT,
                snapshot_id TEXT,
                snapshot_path TEXT,
                error_code TEXT,
                error_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                replay_order INTEGER NOT NULL,
                parent_task_id TEXT,
                split_source TEXT,
                table_name TEXT NOT NULL,
                table_version INTEGER NOT NULL,
                table_digest TEXT NOT NULL,
                row_start INTEGER NOT NULL,
                row_end INTEGER NOT NULL,
                referenced_columns_json TEXT NOT NULL,
                split_eligible INTEGER NOT NULL,
                status TEXT NOT NULL,
                included INTEGER NOT NULL,
                children_json TEXT NOT NULL,
                rows_scanned INTEGER NOT NULL,
                rows_completed INTEGER NOT NULL,
                aggregate_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                estimate_bytes INTEGER NOT NULL,
                start_ms INTEGER NOT NULL,
                end_ms INTEGER NOT NULL,
                queue_wait_ms INTEGER NOT NULL,
                queue_started_ms INTEGER,
                queue_count INTEGER NOT NULL,
                PRIMARY KEY (run_id, task_id)
            );

            CREATE TABLE IF NOT EXISTS skew_decisions (
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                action TEXT NOT NULL,
                group_key_json TEXT NOT NULL,
                hot_count INTEGER NOT NULL,
                threshold INTEGER NOT NULL,
                strategy TEXT NOT NULL,
                requested_parts INTEGER NOT NULL,
                actual_parts INTEGER NOT NULL,
                hot_groups_json TEXT NOT NULL,
                children_json TEXT NOT NULL,
                safety_limit INTEGER NOT NULL,
                PRIMARY KEY (run_id, task_id, action)
            );

            CREATE TABLE IF NOT EXISTS budget_events (
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event TEXT NOT NULL,
                monotonic_ms INTEGER NOT NULL,
                task_id TEXT NOT NULL,
                estimate_bytes INTEGER NOT NULL,
                active_tasks INTEGER NOT NULL,
                used_bytes INTEGER NOT NULL,
                budget_bytes INTEGER NOT NULL,
                reason TEXT,
                PRIMARY KEY (run_id, seq)
            );

            CREATE TABLE IF NOT EXISTS batch_state (
                run_id TEXT NOT NULL,
                batch_no INTEGER NOT NULL,
                completed_markers INTEGER NOT NULL,
                included_tasks INTEGER NOT NULL,
                aggregate_json TEXT NOT NULL,
                peak_memory_bytes INTEGER NOT NULL,
                split_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, batch_no)
            );

            CREATE TABLE IF NOT EXISTS snapshot_replacements (
                replacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_run_id TEXT,
                new_run_id TEXT NOT NULL,
                old_snapshot_id TEXT,
                reason TEXT NOT NULL,
                details_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshot_index (
                snapshot_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                run_id TEXT NOT NULL,
                ledger_digest TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        task_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        if "queue_started_ms" not in task_columns:
            self.connection.execute("ALTER TABLE tasks ADD COLUMN queue_started_ms INTEGER")
        self.connection.commit()

    def reset_run(self, run_id: str) -> None:
        with self.connection:
            for table in ("tasks", "skew_decisions", "budget_events", "batch_state"):
                self.connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
            self.connection.execute("DELETE FROM snapshot_index WHERE run_id = ?", (run_id,))
            self.connection.execute("DELETE FROM sessions WHERE run_id = ?", (run_id,))

    def begin_run(
        self,
        *,
        run_id: str,
        parent_run_id: str | None,
        kind: str,
        status: str,
        dataset: Dataset,
        config: RunConfig,
        plan_payload: Sequence[Mapping[str, Any]],
    ) -> None:
        now = utc_now()
        config_json = canonical_json(config.to_json_dict())
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO sessions (
                    run_id, parent_run_id, kind, status, query_id, data_digest,
                    table_digests_json, table_versions_json, budget_version,
                    budget_digest, config_digest, split_threshold, split_parts,
                    row_group_size, parallelism, memory_budget_bytes,
                    effective_memory_budget_bytes, split_strategy, skew_safety_limit,
                    injection_json, config_json, query_json, plan_json,
                    result_json, result_digest, snapshot_id, snapshot_path,
                    error_code, error_json, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, ?, ?
                )
                """,
                (
                    run_id,
                    parent_run_id,
                    kind,
                    status,
                    dataset.query.query_id,
                    dataset.data_digest,
                    canonical_json(dataset.table_digests()),
                    canonical_json(dataset.table_versions()),
                    config.budget_version,
                    config.budget_digest,
                    config.config_digest,
                    config.split_threshold,
                    config.split_parts,
                    config.row_group_size,
                    config.parallelism,
                    config.memory_budget_bytes,
                    config.effective_memory_budget(),
                    config.split_strategy,
                    config.skew_safety_limit,
                    canonical_json(dataset.injection),
                    config_json,
                    canonical_json(dataset.query.canonical_payload()),
                    canonical_json(list(plan_payload)),
                    now,
                    now,
                ),
            )

    def _upsert_records(self, run_id: str, journal: ScheduleJournal) -> None:
        records = journal.records_in_order()
        for replay_order, record in enumerate(records):
            task = record.task
            self.connection.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    run_id, task_id, replay_order, parent_task_id, split_source,
                    table_name, table_version, table_digest, row_start, row_end,
                    referenced_columns_json, split_eligible, status, included,
                    children_json, rows_scanned, rows_completed, aggregate_json,
                    digest, estimate_bytes, start_ms, end_ms, queue_wait_ms,
                    queue_started_ms, queue_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task.task_id,
                    replay_order,
                    task.parent_task_id,
                    task.split_source,
                    task.table_name,
                    task.table_version,
                    task.table_digest,
                    task.row_start,
                    task.row_end,
                    canonical_json(list(task.referenced_columns)),
                    1 if task.split_eligible else 0,
                    record.status,
                    1 if record.included else 0,
                    canonical_json(list(record.children)),
                    record.outcome.rows_scanned,
                    record.outcome.rows_completed,
                    aggregate_state_json(record.outcome),
                    record.outcome.digest,
                    task.estimated_bytes,
                    record.start_ms,
                    record.end_ms,
                    record.queue_wait_ms,
                    record.queue_started_ms,
                    record.queue_count,
                ),
            )

    def _replace_skew_decisions(self, run_id: str, journal: ScheduleJournal) -> None:
        self.connection.execute("DELETE FROM skew_decisions WHERE run_id = ?", (run_id,))
        for decision in sorted(journal.skew_decisions, key=lambda item: item.task_id):
            self.connection.execute(
                """
                INSERT OR REPLACE INTO skew_decisions (
                    run_id, task_id, action, group_key_json, hot_count, threshold,
                    strategy, requested_parts, actual_parts, hot_groups_json,
                    children_json, safety_limit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    decision.task_id,
                    decision.action,
                    canonical_json(decision.group_key),
                    decision.hot_count,
                    decision.threshold,
                    decision.strategy,
                    decision.requested_parts,
                    decision.actual_parts,
                    canonical_json(decision.hot_groups),
                    canonical_json(decision.children),
                    decision.safety_limit,
                ),
            )

    def _replace_budget_events(self, run_id: str, journal: ScheduleJournal) -> None:
        self.connection.execute("DELETE FROM budget_events WHERE run_id = ?", (run_id,))
        for event in journal.budget_events:
            self.connection.execute(
                """
                INSERT INTO budget_events (
                    run_id, seq, event, monotonic_ms, task_id, estimate_bytes,
                    active_tasks, used_bytes, budget_bytes, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.seq,
                    event.event,
                    event.monotonic_ms,
                    event.task_id,
                    event.estimate_bytes,
                    event.active_tasks,
                    event.used_bytes,
                    event.budget_bytes,
                    event.reason,
                ),
            )

    def checkpoint(
        self,
        run_id: str,
        batch_no: int,
        journal: ScheduleJournal,
        query: Query,
        group_types: Sequence[str],
    ) -> None:
        merged = merge_aggregate_maps(journal.included_outcomes(), query, group_types)
        aggregate_json = canonical_json(serialize_aggregate_map(merged, query, group_types))
        with self.connection:
            self._upsert_records(run_id, journal)
            self._replace_skew_decisions(run_id, journal)
            self._replace_budget_events(run_id, journal)
            self.connection.execute(
                """
                INSERT OR REPLACE INTO batch_state (
                    run_id, batch_no, completed_markers, included_tasks,
                    aggregate_json, peak_memory_bytes, split_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    batch_no,
                    journal.completed_markers,
                    len(journal.included_outcomes()),
                    aggregate_json,
                    journal.peak_memory_bytes,
                    journal.split_count,
                    utc_now(),
                ),
            )
            self.connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE run_id = ?",
                (utc_now(), run_id),
            )

    def finish_run(
        self,
        run_id: str,
        *,
        result_json: str,
        result_digest: str,
        snapshot_id: str | None = None,
        snapshot_path: str | None = None,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE sessions SET
                    status = 'complete', result_json = ?, result_digest = ?,
                    snapshot_id = COALESCE(?, snapshot_id),
                    snapshot_path = COALESCE(?, snapshot_path),
                    error_code = NULL, error_json = NULL, updated_at = ?
                WHERE run_id = ?
                """,
                (result_json, result_digest, snapshot_id, snapshot_path, utc_now(), run_id),
            )

    def fail_run(self, run_id: str, error: Mapping[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE sessions SET status = 'error', error_code = ?, error_json = ?,
                    updated_at = ?
                WHERE run_id = ?
                """,
                (error.get("code"), canonical_json(dict(error)), utc_now(), run_id),
            )

    def get_session(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE run_id = ?", (run_id,)
        ).fetchone()
        return None if row is None else dict(row)

    def get_tasks(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM tasks WHERE run_id = ? ORDER BY replay_order, task_id",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_skew_decisions(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM skew_decisions WHERE run_id = ? ORDER BY task_id, action",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_budget_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM budget_events WHERE run_id = ? ORDER BY seq",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_batch_states(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM batch_state WHERE run_id = ? ORDER BY batch_no",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def semantic_digest(self, run_id: str) -> str:
        session = self.get_session(run_id)
        if session is None:
            raise KeyError(run_id)
        session_semantic = {
            key: session[key]
            for key in (
                "run_id",
                "query_id",
                "data_digest",
                "table_digests_json",
                "table_versions_json",
                "budget_version",
                "budget_digest",
                "config_digest",
                "split_threshold",
                "split_parts",
                "row_group_size",
                "parallelism",
                "memory_budget_bytes",
                "effective_memory_budget_bytes",
                "split_strategy",
                "skew_safety_limit",
                "injection_json",
                "config_json",
                "query_json",
                "plan_json",
                "result_digest",
            )
        }
        tasks = self.get_tasks(run_id)
        skews = self.get_skew_decisions(run_id)
        batches = self.get_batch_states(run_id)
        payload = {
            "session": session_semantic,
            "tasks": tasks,
            "skew_decisions": skews,
            "batch_states": batches,
        }
        return sha256_obj(payload)

    def record_snapshot_replacement(
        self,
        *,
        old_run_id: str | None,
        new_run_id: str,
        old_snapshot_id: str | None,
        reason: str,
        details: Mapping[str, Any],
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO snapshot_replacements (
                    old_run_id, new_run_id, old_snapshot_id, reason,
                    details_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    old_run_id,
                    new_run_id,
                    old_snapshot_id,
                    reason,
                    canonical_json(dict(details)),
                    utc_now(),
                ),
            )

    def register_snapshot(self, snapshot: Mapping[str, Any], path: str | Path) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO snapshot_index (
                    snapshot_id, path, run_id, ledger_digest, result_digest,
                    snapshot_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot["snapshot_id"],
                    str(path),
                    snapshot["run_id"],
                    snapshot["ledger_digest"],
                    snapshot["result_digest"],
                    canonical_json(dict(snapshot)),
                    utc_now(),
                ),
            )

    def get_snapshot_index(self, snapshot_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM snapshot_index WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        return None if row is None else dict(row)
