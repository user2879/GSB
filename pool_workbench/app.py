"""High-level workbench session orchestrating config, pool, snapshots, replay."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from .config import PoolConfig, WorkbenchConfig, load_config
from .errors import BusinessError, ErrorCode
from .planner import build_plan
from .replay import document_for_snapshot, replay_snapshot
from .sample_db import (
    ALLOWED_TABLES,
    open_connection,
    read_sample_data_version,
    verify_schema,
)
from .scheduler import SchedulerResult, run_batch
from .snapshot import SnapshotStore, compute_version_diff
from .pool import ConnectionPool
from .structured_log import StructuredLogger


class WorkbenchSession:
    def __init__(
        self,
        workspace: str | Path,
        log_path: str | Path | None = None,
        connector: Callable = open_connection,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.workspace / "snapshots"
        self.store = SnapshotStore(self.snapshot_dir)
        self.logger = StructuredLogger(log_path or (self.workspace / "events.jsonl"))
        self._connector = connector
        self.config: WorkbenchConfig | None = None
        self.pool: ConnectionPool | None = None
        self.result: SchedulerResult | None = None
        self.last_diff: dict[str, Any] | None = None
        self.session_param_version: int | None = None
        self.pool_replaced_from_snapshot: str | None = None

    # ----- loading and gating -------------------------------------------------
    def load(self, config_path: str | Path, db_path: str | Path) -> WorkbenchConfig:
        verify_schema(db_path, ALLOWED_TABLES)
        config = load_config(config_path, ALLOWED_TABLES)
        data_version = read_sample_data_version(db_path)
        if data_version != config.sample_data_version:
            raise BusinessError(
                ErrorCode.E_SCHEMA,
                "config sample_data_version does not match the sample library",
                field_paths=["sample_data_version"],
                details={"expected": config.sample_data_version, "actual": data_version},
            )
        self.config = config
        self.session_param_version = config.pool.param_version
        self.db_path = str(db_path)
        self.pool = ConnectionPool(
            config.pool, self.db_path, self._connector, self.logger
        )
        self.logger.event(
            "session_loaded",
            param_version=config.pool.param_version,
            tasks=len(config.tasks),
            max_connections=config.pool.max_connections,
        )
        return config

    # ----- execution ----------------------------------------------------------
    def execute(self) -> SchedulerResult:
        if self.config is None or self.pool is None:
            raise RuntimeError("session must be loaded before execute()")
        plan = build_plan(self.config.pool, self.config.tasks)
        self.logger.event(
            "plan_built",
            peak_active=plan.peak_active,
            peak_workload=plan.peak_workload,
            open_events=plan.open_events,
        )
        self.result = run_batch(
            self.config.pool, self.config.tasks, self.pool, self.logger
        )
        return self.result

    def execute_with_failure(self, injector: Callable) -> SchedulerResult:
        if self.config is None or self.pool is None:
            raise RuntimeError("session must be loaded before execute()")
        self.result = run_batch(
            self.config.pool, self.config.tasks, self.pool, self.logger,
            failure_injector=injector,
        )
        return self.result

    # ----- snapshots ----------------------------------------------------------
    def save_snapshot(self) -> Path:
        if self.config is None or self.pool is None or self.result is None:
            raise RuntimeError("nothing to snapshot; execute a batch first")
        queued = [
            trace.task_id for trace in self.result.traces
            if trace.outcome == "GET_TIMEOUT"
        ]
        snapshot_id = time.strftime("snap-%Y%m%d%H%M%S")
        document = document_for_snapshot(
            self.config, self.pool, self.result, queued, snapshot_id
        )
        path = self.store.save(document)
        self.logger.event("snapshot_saved", path=str(path),
                          param_version=document["param_version"])
        return path

    def restore_snapshot(self, path: str | Path) -> dict[str, Any]:
        """Restore a snapshot, replacing the whole pool state on version drift."""
        document = self.store.load(path)
        current_config = (
            self.config.pool if self.config is not None
            else PoolConfig(
                max_connections=document["pool_config"]["max_connections"],
                workload_quotas=dict(document["pool_config"]["workload_quotas"]),
                hold_timeout_ms=document["pool_config"]["hold_timeout_ms"],
                acquire_timeout_ms=document["pool_config"]["acquire_timeout_ms"],
                param_version=document["pool_config"]["param_version"],
            )
        )
        current_dict = {
            "max_connections": current_config.max_connections,
            "workload_quotas": dict(current_config.workload_quotas),
            "hold_timeout_ms": current_config.hold_timeout_ms,
            "acquire_timeout_ms": current_config.acquire_timeout_ms,
            "param_version": current_config.param_version,
        }
        diff = compute_version_diff(
            document, current_dict, list(document.get("queued_tasks", []))
        )
        self.last_diff = diff.to_dict()
        # Whole replacement whenever the parameter version OR any parameter
        # value diverges; never incrementally merge into the live pool.
        replaced = bool(diff.changed_params)
        if replaced:
            # Whole replacement: tear down the current pool and rebuild from
            # the snapshot parameters instead of merging anything.
            if self.pool is not None:
                self.pool.close()
            snapshot_pool_config = PoolConfig(
                max_connections=document["pool_config"]["max_connections"],
                workload_quotas=dict(document["pool_config"]["workload_quotas"]),
                hold_timeout_ms=document["pool_config"]["hold_timeout_ms"],
                acquire_timeout_ms=document["pool_config"]["acquire_timeout_ms"],
                param_version=document["pool_config"]["param_version"],
            )
            self.pool = ConnectionPool(
                snapshot_pool_config, self.db_path, self._connector, self.logger
            )
            self.pool_replaced_from_snapshot = document["snapshot_id"]
            self.logger.event(
                "snapshot_restored_replaced",
                snapshot_id=document["snapshot_id"],
                diff=self.last_diff,
            )
        else:
            self.logger.event(
                "snapshot_restored_aligned",
                snapshot_id=document["snapshot_id"],
            )
        return {
            "snapshot_id": document["snapshot_id"],
            "replaced": replaced,
            "diff": self.last_diff,
            "document": document,
        }

    def load_from_snapshot_for_replay(self, document: dict[str, Any], db_path: str | Path) -> None:
        """Build an in-memory config matching the snapshot purely for replay."""
        from .config import Task
        verify_schema(db_path, self.config.allowed_tables if self.config else ALLOWED_TABLES)
        snapshot_pool = document["pool_config"]
        pool_config = PoolConfig(
            max_connections=int(snapshot_pool["max_connections"]),
            workload_quotas=dict(snapshot_pool["workload_quotas"]),
            hold_timeout_ms=int(snapshot_pool["hold_timeout_ms"]),
            acquire_timeout_ms=int(snapshot_pool["acquire_timeout_ms"]),
            param_version=int(snapshot_pool["param_version"]),
        )
        tasks = [
            Task(
                task_id=item["task_id"],
                workload=item["workload"],
                conn_type=item["conn_type"],
                estimated_ms=int(item["estimated_ms"]),
                sql=item["sql"],
            )
            for item in document["tasks"]
        ]
        self.config = WorkbenchConfig(
            pool=pool_config,
            tasks=tasks,
            allowed_tables=ALLOWED_TABLES,
            sample_data_version=str(document["sample_data_version"]),
        )
        self.db_path = str(db_path)
        self.pool = ConnectionPool(
            pool_config, self.db_path, self._connector, self.logger
        )

    def replay(self, document: dict[str, Any]) -> dict[str, Any]:
        if self.config is None:
            raise RuntimeError("session must be loaded before replay()")
        # Replay compares against the *session* parameter version, which stays
        # pinned to the loaded config even after a snapshot replaces the pool.
        current_version = (
            self.session_param_version
            if self.session_param_version is not None
            else self.config.pool.param_version
        )
        return replay_snapshot(
            document,
            self.db_path,
            current_version,
            self.logger,
            self.config.allowed_tables,
        )

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
