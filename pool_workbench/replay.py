"""Snapshot replay validation and non-reproducibility reporting.

Replay rebuilds the pool parameters and task sequence recorded inside a
snapshot, schedules them against an isolated *clone* of the sample database and
compares every trace (outcome, connection slot and acquisition wait) with the
snapshot.  The original sample tables are never modified.

When the parameter version or sample data version no longer matches the
snapshot, replay cannot be reproducible and the first diverging field is
reported with business code ``E_REPLAY``.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from .config import PoolConfig, Task, WorkbenchConfig
from .errors import BusinessError, ErrorCode
from .planner import build_plan
from .sample_db import clone_database, open_connection, read_sample_data_version
from .scheduler import run_batch
from .pool import ConnectionPool
from .structured_log import StructuredLogger

_COMPARED_FIELDS = ("outcome", "slot_seq", "wait_ms")


def _pool_config_from_snapshot(snapshot_config: dict[str, Any]) -> PoolConfig:
    return PoolConfig(
        max_connections=int(snapshot_config["max_connections"]),
        workload_quotas=dict(snapshot_config["workload_quotas"]),
        hold_timeout_ms=int(snapshot_config["hold_timeout_ms"]),
        acquire_timeout_ms=int(snapshot_config["acquire_timeout_ms"]),
        param_version=int(snapshot_config["param_version"]),
    )


def _tasks_from_snapshot(raw_tasks: list[dict[str, Any]]) -> list[Task]:
    return [
        Task(
            task_id=item["task_id"],
            workload=item["workload"],
            conn_type=item["conn_type"],
            estimated_ms=int(item["estimated_ms"]),
            sql=item["sql"],
        )
        for item in raw_tasks
    ]


def _table_checksum(db_path: str | Path, table: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        total = 0
        for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1"):
            total += hash(tuple(row))
        return total
    finally:
        conn.close()


def replay_snapshot(
    document: dict[str, Any],
    live_db_path: str | Path,
    current_param_version: int,
    logger: StructuredLogger,
    allowed_tables: tuple[str, ...],
) -> dict[str, Any]:
    snapshot_config = document["pool_config"]
    snapshot_param_version = int(snapshot_config["param_version"])
    snapshot_data_version = str(document["sample_data_version"])
    live_data_version = read_sample_data_version(live_db_path)

    checksums_before = {
        table: _table_checksum(live_db_path, table) for table in allowed_tables
    }

    if snapshot_param_version != int(current_param_version):
        raise BusinessError(
            ErrorCode.E_REPLAY,
            "snapshot is not reproducible: parameter version diverged",
            field_paths=["pool_config.param_version"],
            details={
                "task_id": document["tasks"][0]["task_id"] if document.get("tasks") else None,
                "field": "pool_config.param_version",
                "expected": snapshot_param_version,
                "actual": current_param_version,
            },
        )
    if snapshot_data_version != live_data_version:
        raise BusinessError(
            ErrorCode.E_REPLAY,
            "snapshot is not reproducible: sample data version diverged",
            field_paths=["sample_data_version"],
            details={
                "task_id": document["tasks"][0]["task_id"] if document.get("tasks") else None,
                "field": "sample_data_version",
                "expected": snapshot_data_version,
                "actual": live_data_version,
            },
        )

    pool_config = _pool_config_from_snapshot(snapshot_config)
    tasks = _tasks_from_snapshot(document["tasks"])
    expected_traces = document["traces"]

    with tempfile.TemporaryDirectory(prefix="wb_replay_") as tmp:
        clone_path = Path(tmp) / "replay.sqlite"
        clone_database(live_db_path, clone_path)
        pool = ConnectionPool(
            pool_config, str(clone_path), open_connection, logger,
        )
        try:
            result = run_batch(pool_config, tasks, pool, logger)
        finally:
            pool.close()

    actual_by_id = {trace.task_id: trace.to_dict() for trace in result.traces}
    expected_by_id = {trace["task_id"]: trace for trace in expected_traces}

    for index, expected in enumerate(expected_traces):
        task_id = expected["task_id"]
        actual = actual_by_id.get(task_id)
        if actual is None:
            raise BusinessError(
                ErrorCode.E_REPLAY,
                "snapshot is not reproducible: task missing from replay",
                field_paths=[f"traces[{index}].task_id"],
                details={"task_id": task_id, "field": "trace",
                         "expected": expected, "actual": None},
            )
        for field in _COMPARED_FIELDS:
            if expected.get(field) != actual.get(field):
                raise BusinessError(
                    ErrorCode.E_REPLAY,
                    "snapshot is not reproducible: replay trace diverged",
                    field_paths=[f"traces[{index}].{field}"],
                    details={"task_id": task_id, "field": field,
                             "expected": expected.get(field),
                             "actual": actual.get(field)},
                )

    checksums_after = {
        table: _table_checksum(live_db_path, table) for table in allowed_tables
    }
    if checksums_before != checksums_after:
        raise BusinessError(
            ErrorCode.E_REPLAY,
            "replay modified the sample tables, which is forbidden",
            field_paths=["sample_tables"],
            details={"before": checksums_before, "after": checksums_after},
        )

    report = {
        "status": "REPRODUCIBLE",
        "compared_tasks": len(expected_traces),
        "compared_fields": list(_COMPARED_FIELDS),
        "outcome_counts": result.outcome_counts,
        "sample_tables_unchanged": True,
        "param_version": snapshot_param_version,
        "sample_data_version": snapshot_data_version,
    }
    logger.event("replay_finished", **report)
    return report


def document_for_snapshot(
    config: WorkbenchConfig,
    pool: ConnectionPool,
    result,
    queued_task_ids: list[str],
    snapshot_id: str,
) -> dict[str, Any]:
    pool_snapshot = pool.pool_snapshot()
    return {
        "snapshot_id": snapshot_id,
        "param_version": config.pool.param_version,
        "sample_data_version": config.sample_data_version,
        "pool_config": pool_snapshot["config"],
        "pool_state": {
            key: value
            for key, value in pool_snapshot.items()
            if key != "config"
        },
        "queued_tasks": list(queued_task_ids),
        "tasks": [
            {
                "task_id": task.task_id,
                "workload": task.workload,
                "conn_type": task.conn_type,
                "estimated_ms": task.estimated_ms,
                "sql": task.sql,
            }
            for task in config.tasks
        ],
        "traces": [trace.to_dict() for trace in result.traces],
        "outcome_counts": dict(result.outcome_counts),
    }


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")
