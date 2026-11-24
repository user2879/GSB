from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .errors import AppError
from .models import BudgetConfig, TaskResult
from .util import canonical_json


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    is_current INTEGER NOT NULL,
    budget_version INTEGER NOT NULL,
    encoding_version INTEGER NOT NULL,
    segment_digest TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    cursor INTEGER NOT NULL DEFAULT 0,
    tasks_total INTEGER NOT NULL DEFAULT 0,
    committed_batches INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id),
    committed_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    is_current INTEGER NOT NULL,
    task_count INTEGER NOT NULL,
    first_ordinal INTEGER,
    last_ordinal INTEGER,
    cursor_before INTEGER NOT NULL,
    cursor_after INTEGER NOT NULL,
    budget_version INTEGER NOT NULL,
    encoding_version INTEGER NOT NULL,
    segment_digest TEXT NOT NULL,
    plan_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_ledger (
    ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER REFERENCES batches(batch_id),
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id),
    task_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    row_group_id TEXT NOT NULL,
    block_id TEXT NOT NULL,
    block_ordinal INTEGER NOT NULL,
    column_name TEXT NOT NULL,
    completed_rows INTEGER NOT NULL,
    decode_memory_bytes INTEGER NOT NULL,
    queue_started_ms REAL NOT NULL,
    wait_ended_ms REAL NOT NULL,
    run_started_ms REAL NOT NULL,
    run_ended_ms REAL NOT NULL,
    duration_ms REAL NOT NULL,
    block_digest TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    result_json TEXT NOT NULL,
    stats_json TEXT NOT NULL,
    budget_version INTEGER NOT NULL,
    encoding_version INTEGER NOT NULL,
    is_current INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scan_ledger_generation_current
ON scan_ledger(generation_id, is_current, block_ordinal);
CREATE INDEX IF NOT EXISTS idx_scan_ledger_task ON scan_ledger(task_id);

CREATE TABLE IF NOT EXISTS query_results (
    query_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER REFERENCES batches(batch_id),
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id),
    query_id TEXT NOT NULL,
    row_group_id TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    rows_json TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    select_json TEXT NOT NULL,
    is_current INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_query_results_generation_current
ON query_results(generation_id, is_current);

CREATE TABLE IF NOT EXISTS plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER REFERENCES batches(batch_id),
    generation_id INTEGER NOT NULL REFERENCES generations(generation_id),
    query_id TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    is_current INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plans_generation_current
ON plans(generation_id, is_current);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def set_meta(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, canonical_json(value)),
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def current_generation(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM generations WHERE is_current=1 ORDER BY generation_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def begin_generation(
        self,
        *,
        reason: str,
        budget: BudgetConfig,
        budget_version: int,
        encoding_version: int,
        segment_digest: str,
        spec_digest: str,
        plan_digest: str,
        tasks_total: int,
        cursor: int = 0,
    ) -> int:
        with self.conn:
            self._stale_all()
            cursor_row = self.conn.execute(
                """
                INSERT INTO generations(
                    created_at, reason, status, is_current, budget_version,
                    encoding_version, segment_digest, spec_digest, plan_digest,
                    budget_json, cursor, tasks_total, committed_batches
                ) VALUES(?, ?, 'active', 1, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    _now_iso(),
                    reason,
                    budget_version,
                    encoding_version,
                    segment_digest,
                    spec_digest,
                    plan_digest,
                    canonical_json(budget.to_dict()),
                    cursor,
                    tasks_total,
                ),
            )
            return int(cursor_row.lastrowid)

    def _stale_all(self) -> None:
        self.conn.execute("UPDATE generations SET status='stale', is_current=0")
        self.conn.execute("UPDATE batches SET status='stale', is_current=0")
        self.conn.execute("UPDATE scan_ledger SET is_current=0")
        self.conn.execute("UPDATE query_results SET is_current=0")
        self.conn.execute("UPDATE plans SET is_current=0")

    def update_generation_progress(
        self,
        *,
        generation_id: int,
        cursor: int,
        tasks_total: int,
        plan_digest: str | None = None,
        segment_digest: str | None = None,
    ) -> None:
        with self.conn:
            if plan_digest is not None and segment_digest is not None:
                self.conn.execute(
                    """
                    UPDATE generations
                    SET cursor=?, tasks_total=?, plan_digest=?, segment_digest=?,
                        committed_batches=(
                            SELECT COUNT(*) FROM batches WHERE generation_id=? AND is_current=1
                        )
                    WHERE generation_id=? AND is_current=1
                    """,
                    (
                        cursor,
                        tasks_total,
                        plan_digest,
                        segment_digest,
                        generation_id,
                        generation_id,
                    ),
                )
            else:
                self.conn.execute(
                    """
                    UPDATE generations
                    SET cursor=?, tasks_total=?,
                        committed_batches=(
                            SELECT COUNT(*) FROM batches WHERE generation_id=? AND is_current=1
                        )
                    WHERE generation_id=? AND is_current=1
                    """,
                    (cursor, tasks_total, generation_id, generation_id),
                )

    def commit_batch(
        self,
        *,
        generation_id: int,
        reason: str,
        task_results: Iterable[TaskResult],
        query_results: dict[tuple[str, str], dict[str, Any]],
        plans: dict[str, dict[str, Any]],
        cursor_before: int,
        cursor_after: int,
        tasks_total: int,
        budget_version: int,
        encoding_version: int,
        segment_digest: str,
        plan_digest: str,
        replace_task_ids: Iterable[str] = (),
        replace_query_result_keys: Iterable[tuple[str, str]] = (),
        replace_query_ids: Iterable[str] = (),
        fail_after_rows: int | None = None,
    ) -> int:
        results = tuple(task_results)
        if not results and not query_results and not plans:
            raise AppError(
                "E_LEDGER",
                "批次必须包含扫描任务、查询结论或计划中的至少一项",
                field="task_results",
            )
        ordinals = [result.task.block_ordinal for result in results]
        try:
            with self.conn:
                generation = self.conn.execute(
                    "SELECT * FROM generations WHERE generation_id=? AND is_current=1",
                    (generation_id,),
                ).fetchone()
                if generation is None:
                    raise AppError(
                        "E_LEDGER", "当前生成不存在或已陈旧", field="generation_id"
                    )
                replace_task_ids = tuple(replace_task_ids)
                replace_query_result_keys = tuple(replace_query_result_keys)
                replace_query_ids = tuple(replace_query_ids)
                for task_id in replace_task_ids:
                    self.conn.execute(
                        "UPDATE scan_ledger SET is_current=0 "
                        "WHERE generation_id=? AND task_id=? AND is_current=1",
                        (generation_id, task_id),
                    )
                for query_id, row_group_id in replace_query_result_keys:
                    self.conn.execute(
                        "UPDATE query_results SET is_current=0 "
                        "WHERE generation_id=? AND query_id=? AND row_group_id=? AND is_current=1",
                        (generation_id, query_id, row_group_id),
                    )
                for query_id in replace_query_ids:
                    self.conn.execute(
                        "UPDATE plans SET is_current=0 "
                        "WHERE generation_id=? AND query_id=? AND is_current=1",
                        (generation_id, query_id),
                    )
                batch_cursor = self.conn.execute(
                    """
                    INSERT INTO batches(
                        generation_id, committed_at, reason, status, is_current,
                        task_count, first_ordinal, last_ordinal, cursor_before,
                        cursor_after, budget_version, encoding_version,
                        segment_digest, plan_digest
                    ) VALUES(?, ?, ?, 'committed', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        generation_id,
                        _now_iso(),
                        reason,
                        len(results),
                        min(ordinals) if ordinals else None,
                        max(ordinals) if ordinals else None,
                        cursor_before,
                        cursor_after,
                        budget_version,
                        encoding_version,
                        segment_digest,
                        plan_digest,
                    ),
                )
                batch_id = int(batch_cursor.lastrowid)
                for index, result in enumerate(results, start=1):
                    self._insert_task_result(
                        batch_id=batch_id,
                        generation_id=generation_id,
                        result=result,
                        budget_version=budget_version,
                        encoding_version=encoding_version,
                    )
                    if fail_after_rows is not None and index >= fail_after_rows:
                        raise RuntimeError("injected batch failure")
                for (query_id, row_group_id), value in sorted(query_results.items()):
                    self.conn.execute(
                        """
                        INSERT INTO query_results(
                            batch_id, generation_id, query_id, row_group_id,
                            result_digest, row_count, rows_json, plan_digest,
                            select_json, is_current
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id,
                            generation_id,
                            query_id,
                            row_group_id,
                            value["result_digest"],
                            value["row_count"],
                            canonical_json(value["rows"]),
                            value["plan_digest"],
                            canonical_json(value["select"]),
                        ),
                    )
                for query_id, plan_payload in sorted(plans.items()):
                    self.conn.execute(
                        """
                        INSERT INTO plans(
                            batch_id, generation_id, query_id, plan_digest,
                            plan_json, is_current
                        ) VALUES(?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id,
                            generation_id,
                            query_id,
                            plan_payload["plan_digest"],
                            canonical_json(plan_payload),
                        ),
                    )
                self.conn.execute(
                    """
                    UPDATE generations
                    SET cursor=?, tasks_total=?, plan_digest=?, segment_digest=?,
                        budget_version=?, encoding_version=?,
                        committed_batches=(
                            SELECT COUNT(*) FROM batches
                            WHERE generation_id=? AND is_current=1
                        )
                    WHERE generation_id=?
                    """,
                    (
                        cursor_after,
                        tasks_total,
                        plan_digest,
                        segment_digest,
                        budget_version,
                        encoding_version,
                        generation_id,
                        generation_id,
                    ),
                )
                return batch_id
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                "E_LEDGER",
                f"账本批次事务失败，未提交任何部分: {error}",
                field="commit_batch",
            ) from error

    def _insert_task_result(
        self,
        *,
        batch_id: int,
        generation_id: int,
        result: TaskResult,
        budget_version: int,
        encoding_version: int,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO scan_ledger(
                batch_id, generation_id, task_id, query_id, segment_id,
                row_group_id, block_id, block_ordinal, column_name,
                completed_rows, decode_memory_bytes, queue_started_ms,
                wait_ended_ms, run_started_ms, run_ended_ms, duration_ms,
                block_digest, result_digest, result_json, stats_json,
                budget_version, encoding_version, is_current
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                batch_id,
                generation_id,
                result.task.task_id,
                result.task.query_id,
                result.task.segment_id,
                result.task.row_group_id,
                result.task.block_id,
                result.task.block_ordinal,
                result.task.column,
                result.completed_rows,
                result.decode_memory_bytes,
                result.queue_started_ms,
                result.wait_ended_ms,
                result.run_started_ms,
                result.run_ended_ms,
                result.duration_ms,
                result.block_digest,
                result.result_digest,
                canonical_json(result.result_json),
                canonical_json(result.stats),
                budget_version,
                encoding_version,
            ),
        )

    def current_task_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT l.* FROM scan_ledger l
            JOIN generations g ON g.generation_id=l.generation_id
            WHERE g.is_current=1 AND l.is_current=1
            ORDER BY l.block_ordinal, l.ledger_id
            """
        ).fetchall()
        return [self._decode_task_row(row) for row in rows]

    def current_query_results(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT q.* FROM query_results q
            JOIN generations g ON g.generation_id=q.generation_id
            WHERE g.is_current=1 AND q.is_current=1
            ORDER BY q.query_id, q.row_group_id, q.query_result_id
            """
        ).fetchall()
        return [self._decode_query_row(row) for row in rows]

    def current_plans(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT p.* FROM plans p
            JOIN generations g ON g.generation_id=p.generation_id
            WHERE g.is_current=1 AND p.is_current=1
            ORDER BY p.query_id, p.plan_id
            """
        ).fetchall()
        return [
            {
                **dict(row),
                "plan_json": json.loads(row["plan_json"]),
            }
            for row in rows
        ]

    def current_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT b.* FROM batches b
            JOIN generations g ON g.generation_id=b.generation_id
            WHERE g.is_current=1 AND b.is_current=1
            ORDER BY b.batch_id
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def statistics(self) -> dict[str, Any]:
        tasks = self.current_task_rows()
        query_results = self.current_query_results()
        return {
            "task_count": len(tasks),
            "completed_rows": sum(int(task["completed_rows"]) for task in tasks),
            "decode_memory_bytes": sum(
                int(task["decode_memory_bytes"]) for task in tasks
            ),
            "query_result_count": len(query_results),
            "query_row_count": sum(int(item["row_count"]) for item in query_results),
            "total_task_duration_ms": round(
                sum(float(task["duration_ms"]) for task in tasks), 6
            ),
            "leader_task_id": tasks[0]["task_id"] if tasks else None,
            "tail_task_id": tasks[-1]["task_id"] if tasks else None,
        }

    def _decode_task_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["result_json"] = json.loads(payload["result_json"])
        payload["stats_json"] = json.loads(payload["stats_json"])
        return payload

    def _decode_query_row(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = dict(row)
        payload["rows"] = json.loads(payload["rows_json"])
        payload["select"] = json.loads(payload["select_json"])
        return payload

    def export_active_state(self) -> dict[str, Any]:
        generation = self.current_generation()
        if generation is None:
            raise AppError("E_LEDGER", "当前没有活动账本生成", field="generation")
        return {
            "generation": generation,
            "batches": self.current_batches(),
            "tasks": self.current_task_rows(),
            "query_results": self.current_query_results(),
            "plans": self.current_plans(),
        }

    def import_as_current(
        self,
        state: dict[str, Any],
        *,
        reason: str,
    ) -> int:
        """Whole-view replacement used by snapshot recovery."""

        generation = dict(state["generation"])
        tasks = list(state["tasks"])
        query_results = list(state["query_results"])
        plans = list(state["plans"])
        batches = list(state.get("batches", []))
        try:
            with self.conn:
                self._stale_all()
                cursor = self.conn.execute(
                    """
                    INSERT INTO generations(
                        created_at, reason, status, is_current, budget_version,
                        encoding_version, segment_digest, spec_digest, plan_digest,
                        budget_json, cursor, tasks_total, committed_batches
                    ) VALUES(?, ?, 'active', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _now_iso(),
                        reason,
                        int(generation["budget_version"]),
                        int(generation["encoding_version"]),
                        str(generation["segment_digest"]),
                        str(generation["spec_digest"]),
                        str(generation["plan_digest"]),
                        str(generation["budget_json"]),
                        int(generation.get("cursor", 0)),
                        int(generation.get("tasks_total", len(tasks))),
                        int(generation.get("committed_batches", len(batches))),
                    ),
                )
                new_generation_id = int(cursor.lastrowid)
                batch_id_map: dict[int, int] = {}
                for batch in sorted(batches, key=lambda item: int(item["batch_id"])):
                    inserted = self.conn.execute(
                        """
                        INSERT INTO batches(
                            generation_id, committed_at, reason, status, is_current,
                            task_count, first_ordinal, last_ordinal, cursor_before,
                            cursor_after, budget_version, encoding_version,
                            segment_digest, plan_digest
                        ) VALUES(?, ?, ?, 'committed', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_generation_id,
                            str(batch["committed_at"]),
                            str(batch["reason"]),
                            int(batch["task_count"]),
                            batch.get("first_ordinal"),
                            batch.get("last_ordinal"),
                            int(batch["cursor_before"]),
                            int(batch["cursor_after"]),
                            int(batch["budget_version"]),
                            int(batch["encoding_version"]),
                            str(batch["segment_digest"]),
                            str(batch["plan_digest"]),
                        ),
                    )
                    batch_id_map[int(batch["batch_id"])] = int(inserted.lastrowid)
                for task in sorted(tasks, key=lambda item: int(item["block_ordinal"])):
                    self.conn.execute(
                        """
                        INSERT INTO scan_ledger(
                            batch_id, generation_id, task_id, query_id, segment_id,
                            row_group_id, block_id, block_ordinal, column_name,
                            completed_rows, decode_memory_bytes, queue_started_ms,
                            wait_ended_ms, run_started_ms, run_ended_ms, duration_ms,
                            block_digest, result_digest, result_json, stats_json,
                            budget_version, encoding_version, is_current
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id_map.get(task.get("batch_id")),
                            new_generation_id,
                            task["task_id"],
                            task["query_id"],
                            task["segment_id"],
                            task["row_group_id"],
                            task["block_id"],
                            int(task["block_ordinal"]),
                            task["column_name"],
                            int(task["completed_rows"]),
                            int(task["decode_memory_bytes"]),
                            float(task["queue_started_ms"]),
                            float(task["wait_ended_ms"]),
                            float(task["run_started_ms"]),
                            float(task["run_ended_ms"]),
                            float(task["duration_ms"]),
                            task["block_digest"],
                            task["result_digest"],
                            canonical_json(task["result_json"]),
                            canonical_json(task["stats_json"]),
                            int(task["budget_version"]),
                            int(task["encoding_version"]),
                        ),
                    )
                for item in sorted(
                    query_results,
                    key=lambda item: (item["query_id"], item["row_group_id"]),
                ):
                    self.conn.execute(
                        """
                        INSERT INTO query_results(
                            batch_id, generation_id, query_id, row_group_id,
                            result_digest, row_count, rows_json, plan_digest,
                            select_json, is_current
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id_map.get(item.get("batch_id")),
                            new_generation_id,
                            item["query_id"],
                            item["row_group_id"],
                            item["result_digest"],
                            int(item["row_count"]),
                            canonical_json(item["rows"]),
                            item["plan_digest"],
                            canonical_json(item["select"]),
                        ),
                    )
                for item in sorted(plans, key=lambda item: item["query_id"]):
                    self.conn.execute(
                        """
                        INSERT INTO plans(
                            batch_id, generation_id, query_id, plan_digest,
                            plan_json, is_current
                        ) VALUES(?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id_map.get(item.get("batch_id")),
                            new_generation_id,
                            item["query_id"],
                            item["plan_digest"],
                            canonical_json(item["plan_json"]),
                        ),
                    )
                return new_generation_id
        except Exception as error:
            raise AppError(
                "E_LEDGER",
                f"快照整体替换失败，未合并任何部分: {error}",
                field="import_as_current",
            ) from error

    def clone_current_generation(
        self,
        *,
        reason: str,
        budget: BudgetConfig,
        budget_version: int,
        encoding_version: int,
        segment_digest: str,
        spec_digest: str,
        plan_digest: str,
        tasks_total: int,
        cursor: int,
    ) -> int:
        state = self.export_active_state()
        new_generation_id = self.begin_generation(
            reason=reason,
            budget=budget,
            budget_version=budget_version,
            encoding_version=encoding_version,
            segment_digest=segment_digest,
            spec_digest=spec_digest,
            plan_digest=plan_digest,
            tasks_total=tasks_total,
            cursor=cursor,
        )
        try:
            with self.conn:
                batch = self.conn.execute(
                    """
                    INSERT INTO batches(
                        generation_id, committed_at, reason, status, is_current,
                        task_count, first_ordinal, last_ordinal, cursor_before,
                        cursor_after, budget_version, encoding_version,
                        segment_digest, plan_digest
                    ) VALUES(?, ?, ?, 'committed', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_generation_id,
                        _now_iso(),
                        "generation-clone",
                        len(state["tasks"]),
                        min(
                            (int(item["block_ordinal"]) for item in state["tasks"]),
                            default=None,
                        ),
                        max(
                            (int(item["block_ordinal"]) for item in state["tasks"]),
                            default=None,
                        ),
                        0,
                        cursor,
                        budget_version,
                        encoding_version,
                        segment_digest,
                        plan_digest,
                    ),
                )
                batch_id = int(batch.lastrowid)
                for task in state["tasks"]:
                    self.conn.execute(
                        """
                        INSERT INTO scan_ledger(
                            batch_id, generation_id, task_id, query_id, segment_id,
                            row_group_id, block_id, block_ordinal, column_name,
                            completed_rows, decode_memory_bytes, queue_started_ms,
                            wait_ended_ms, run_started_ms, run_ended_ms, duration_ms,
                            block_digest, result_digest, result_json, stats_json,
                            budget_version, encoding_version, is_current
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id,
                            new_generation_id,
                            task["task_id"],
                            task["query_id"],
                            task["segment_id"],
                            task["row_group_id"],
                            task["block_id"],
                            int(task["block_ordinal"]),
                            task["column_name"],
                            int(task["completed_rows"]),
                            int(task["decode_memory_bytes"]),
                            float(task["queue_started_ms"]),
                            float(task["wait_ended_ms"]),
                            float(task["run_started_ms"]),
                            float(task["run_ended_ms"]),
                            float(task["duration_ms"]),
                            task["block_digest"],
                            task["result_digest"],
                            canonical_json(task["result_json"]),
                            canonical_json(task["stats_json"]),
                            budget_version,
                            encoding_version,
                        ),
                    )
                for item in state["query_results"]:
                    self.conn.execute(
                        """
                        INSERT INTO query_results(
                            batch_id, generation_id, query_id, row_group_id,
                            result_digest, row_count, rows_json, plan_digest,
                            select_json, is_current
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id,
                            new_generation_id,
                            item["query_id"],
                            item["row_group_id"],
                            item["result_digest"],
                            int(item["row_count"]),
                            canonical_json(item["rows"]),
                            item["plan_digest"],
                            canonical_json(item["select"]),
                        ),
                    )
                for item in state["plans"]:
                    self.conn.execute(
                        """
                        INSERT INTO plans(
                            batch_id, generation_id, query_id, plan_digest,
                            plan_json, is_current
                        ) VALUES(?, ?, ?, ?, ?, 1)
                        """,
                        (
                            batch_id,
                            new_generation_id,
                            item["query_id"],
                            item["plan_digest"],
                            canonical_json(item["plan_json"]),
                        ),
                    )
                return new_generation_id
        except Exception as error:
            raise AppError(
                "E_LEDGER", f"生成克隆失败: {error}", field="clone_generation"
            ) from error

    def generation_rows(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM generations ORDER BY generation_id"
            ).fetchall()
        ]
