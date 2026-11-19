"""SQLite execution ledger and whole-state JSON snapshots."""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json_text, digest_value, loads_strict, write_json_atomic
from .errors import AppError
from .model import Workspace, workspace_state_from_export
from .scheduler import ScheduleRecord


LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class LedgerShard:
    shard_id: str
    task_id: str
    shard_index: int
    start: int
    end: int
    status: str
    resources: int
    queued_at_ms: int | None
    started_at_ms: int | None
    ended_at_ms: int | None
    wait_duration_ms: int | None
    outcome: str | None
    result_digest: str | None
    error_code: str | None
    error_message: str | None
    result_json: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "task_id": self.task_id,
            "shard_index": self.shard_index,
            "start": self.start,
            "end": self.end,
            "status": self.status,
            "resources": self.resources,
            "queued_at_ms": self.queued_at_ms,
            "started_at_ms": self.started_at_ms,
            "ended_at_ms": self.ended_at_ms,
            "wait_duration_ms": self.wait_duration_ms,
            "outcome": self.outcome,
            "result_digest": self.result_digest,
            "error_code": self.error_code,
            "error_message": self.error_message,
        }


@dataclass(frozen=True)
class LedgerBatch:
    batch_id: str
    format_version: int
    parameter_version: int
    content_digest: str
    plan_digest: str | None
    state_json: str
    status: str
    started_at: str
    ended_at: str | None
    result_digest: str | None
    result_json: str | None
    cumulative_digest: str | None
    error_code: str | None
    error_message: str | None
    exit_report_allowed: bool
    exit_report_json: str | None
    replay_status: str | None
    replay_batch_id: str | None
    replay_details: str | None
    shards: tuple[LedgerShard, ...]

    def shard_map(self) -> dict[str, LedgerShard]:
        return {shard.shard_id: shard for shard in self.shards}

    def state_payload(self) -> Any:
        return loads_strict(self.state_json, source=f"ledger:{self.batch_id}:state")

    def result_payload(self) -> Any:
        if self.result_json is None:
            return None
        return loads_strict(self.result_json, source=f"ledger:{self.batch_id}:result")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "format_version": self.format_version,
            "parameter_version": self.parameter_version,
            "content_digest": self.content_digest,
            "plan_digest": self.plan_digest,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "result_digest": self.result_digest,
            "cumulative_digest": self.cumulative_digest,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "exit_report_allowed": self.exit_report_allowed,
            "replay_status": self.replay_status,
            "replay_batch_id": self.replay_batch_id,
            "shards": [shard.to_dict() for shard in self.shards],
        }


class ExecutionLedger:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            self._initialize()
        except sqlite3.Error as exc:
            raise AppError("E_LEDGER", f"无法打开执行账本 {self.path}: {exc}", {"path": str(self.path)}) from exc

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _initialize(self) -> None:
        with self._lock, self._conn:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, LEDGER_FORMAT_VERSION):
                raise AppError(
                    "E_LEDGER",
                    f"账本格式版本 {version} 不受支持，当前为 {LEDGER_FORMAT_VERSION}",
                    {"path": str(self.path), "format_version": version},
                )
            self._conn.execute("PRAGMA user_version = 1")
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    batch_id TEXT PRIMARY KEY,
                    format_version INTEGER NOT NULL,
                    parameter_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    plan_digest TEXT,
                    state_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    result_digest TEXT,
                    result_json TEXT,
                    cumulative_digest TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    exit_report_allowed INTEGER NOT NULL DEFAULT 0,
                    exit_report_json TEXT,
                    replay_status TEXT,
                    replay_batch_id TEXT,
                    replay_details TEXT
                );
                CREATE TABLE IF NOT EXISTS shard_records (
                    batch_id TEXT NOT NULL,
                    shard_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    shard_index INTEGER NOT NULL,
                    start_row INTEGER NOT NULL,
                    end_row INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    resources INTEGER NOT NULL,
                    queued_at_ms INTEGER,
                    started_at_ms INTEGER,
                    ended_at_ms INTEGER,
                    wait_duration_ms INTEGER,
                    outcome TEXT,
                    result_digest TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    result_json TEXT,
                    PRIMARY KEY (batch_id, shard_id),
                    FOREIGN KEY (batch_id) REFERENCES runs(batch_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_shard_batch_order
                    ON shard_records(batch_id, task_id, shard_index);
                """
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )

    def reset(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM shard_records")
            self._conn.execute("DELETE FROM runs")
            self._conn.execute("DELETE FROM ledger_meta WHERE key IN ('last_content_digest', 'last_parameter_version')")

    def begin_batch(self, *, batch_id: str, state_payload: dict[str, Any], plan_payload: dict[str, Any], planned_shards: Iterable[dict[str, Any]]) -> None:
        state_json = canonical_json_text(state_payload)
        parameter_version = int(state_payload["parameter_version"])
        content_digest = str(plan_payload["content_digest"])
        plan_digest = str(plan_payload.get("plan_digest", "")) or None
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO runs(
                    batch_id, format_version, parameter_version, content_digest, plan_digest,
                    state_json, status, started_at, exit_report_allowed
                ) VALUES (?, ?, ?, ?, ?, ?, 'running', ?, 0)
                """,
                (batch_id, LEDGER_FORMAT_VERSION, parameter_version, content_digest, plan_digest, state_json, _utc_now()),
            )
            for shard in planned_shards:
                self._conn.execute(
                    """
                    INSERT INTO shard_records(
                        batch_id, shard_id, task_id, shard_index, start_row, end_row,
                        status, resources
                    ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)
                    """,
                    (
                        batch_id,
                        shard["shard_id"],
                        shard["task_id"],
                        shard["index"],
                        shard["start"],
                        shard["end"],
                        shard["resources"],
                    ),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES('last_content_digest', ?)",
                (content_digest,),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO ledger_meta(key, value) VALUES('last_parameter_version', ?)",
                (str(parameter_version),),
            )

    def mark_running(self, batch_id: str, shard_id: str, started_at_ms: int) -> None:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE shard_records
                   SET status = 'running', started_at_ms = ?
                 WHERE batch_id = ? AND shard_id = ? AND status = 'queued'
                """,
                (started_at_ms, batch_id, shard_id),
            )
            if cursor.rowcount != 1:
                raise AppError(
                    "E_LEDGER",
                    f"无法将分片标记为运行中: {shard_id}",
                    {"batch_id": batch_id, "shard_id": shard_id},
                )

    def _cumulative_digest(self, batch_id: str) -> str:
        rows = self._conn.execute(
            """
            SELECT task_id, shard_index, shard_id, outcome, result_digest, resources,
                   error_code, status
              FROM shard_records
             WHERE batch_id = ? AND status IN ('completed', 'failed')
             ORDER BY task_id, shard_index, shard_id
            """,
            (batch_id,),
        ).fetchall()
        payload = [
            {
                "task_id": row["task_id"],
                "shard_index": row["shard_index"],
                "shard_id": row["shard_id"],
                "outcome": row["outcome"],
                "result_digest": row["result_digest"],
                "resources": row["resources"],
                "error_code": row["error_code"],
            }
            for row in rows
        ]
        return digest_value(payload)

    def commit_shard_batch(self, batch_id: str, records: Iterable[ScheduleRecord]) -> str:
        records = list(records)
        with self._lock, self._conn:
            for record in records:
                result_json = None
                if record.result is not None and hasattr(record.result, "to_dict"):
                    result_json = canonical_json_text(record.result.to_dict())
                status = "completed" if record.outcome == "success" else "failed"
                self._conn.execute(
                    """
                    UPDATE shard_records
                       SET status = ?, resources = ?, queued_at_ms = ?, started_at_ms = ?,
                           ended_at_ms = ?, wait_duration_ms = ?, outcome = ?,
                           result_digest = ?, error_code = ?, error_message = ?, result_json = ?
                     WHERE batch_id = ? AND shard_id = ?
                    """,
                    (
                        status,
                        record.resources,
                        record.queued_at_ms,
                        record.started_at_ms,
                        record.ended_at_ms,
                        record.wait_duration_ms,
                        record.outcome,
                        getattr(record.result, "result_digest", None),
                        record.error_code,
                        record.error_message,
                        result_json,
                        batch_id,
                        record.shard_id,
                    ),
                )
            cumulative = self._cumulative_digest(batch_id)
            self._conn.execute(
                "UPDATE runs SET cumulative_digest = ? WHERE batch_id = ?",
                (cumulative, batch_id),
            )
        return cumulative

    def finish_batch(
        self,
        batch_id: str,
        *,
        status: str,
        result_digest: str | None,
        result_payload: dict[str, Any] | None,
        error_code: str | None = None,
        error_message: str | None = None,
        exit_report_allowed: bool = True,
    ) -> None:
        result_json = canonical_json_text(result_payload) if result_payload is not None else None
        exit_json = result_json if exit_report_allowed and result_payload is not None else None
        with self._lock, self._conn:
            self._conn.execute(
                """
                UPDATE runs
                   SET status = ?, ended_at = ?, result_digest = ?, result_json = ?,
                       error_code = ?, error_message = ?, exit_report_allowed = ?,
                       exit_report_json = ?
                 WHERE batch_id = ?
                """,
                (
                    status,
                    _utc_now(),
                    result_digest,
                    result_json,
                    error_code,
                    error_message,
                    1 if exit_json is not None else 0,
                    exit_json,
                    batch_id,
                ),
            )

    def mark_interrupted(self, batch_id: str) -> list[str]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT shard_id FROM shard_records WHERE batch_id = ? AND status IN ('queued', 'running') ORDER BY task_id, shard_index",
                (batch_id,),
            ).fetchall()
            unknown = [str(row["shard_id"]) for row in rows]
            self._conn.execute(
                """
                UPDATE shard_records
                   SET status = 'unknown', outcome = 'unknown',
                       error_code = 'E_RECOVERY', error_message = '进程强制终止，提交状态未知，必须重跑'
                 WHERE batch_id = ? AND status IN ('queued', 'running')
                """,
                (batch_id,),
            )
            self._conn.execute(
                """
                UPDATE runs
                   SET status = 'interrupted', ended_at = ?, error_code = 'E_RECOVERY',
                       error_message = '进程强制终止；不生成退出报告',
                       exit_report_allowed = 0, exit_report_json = NULL
                 WHERE batch_id = ? AND status = 'running'
                """,
                (_utc_now(), batch_id),
            )
        return sorted(unknown)

    def mark_budget_blocked(self, batch_id: str) -> list[str]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT shard_id FROM shard_records WHERE batch_id = ? AND status = 'queued' ORDER BY task_id, shard_index",
                (batch_id,),
            ).fetchall()
            blocked = [str(row["shard_id"]) for row in rows]
            self._conn.execute(
                """
                UPDATE shard_records
                   SET status = 'budget_blocked', outcome = 'blocked', error_code = 'E_BUDGET',
                       error_message = '全局资源预算或任务并发配额持续不足'
                 WHERE batch_id = ? AND status = 'queued'
                """,
                (batch_id,),
            )
        return sorted(blocked)

    def mark_stale_incomplete(self, batch_id: str) -> list[str]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT shard_id FROM shard_records WHERE batch_id = ? AND status IN ('queued', 'running') ORDER BY task_id, shard_index",
                (batch_id,),
            ).fetchall()
            stale = [str(row["shard_id"]) for row in rows]
            self._conn.execute(
                """
                UPDATE shard_records
                   SET status = 'stale', outcome = 'stale', error_code = 'E_STALE',
                       error_message = '参数版本已变更，旧分片轨迹不得参与当前汇总'
                 WHERE batch_id = ? AND status IN ('queued', 'running')
                """,
                (batch_id,),
            )
        return sorted(stale)

    def mark_stale_batch(self, batch_id: str) -> list[str]:
        with self._lock, self._conn:
            rows = self._conn.execute(
                "SELECT shard_id FROM shard_records WHERE batch_id = ? ORDER BY task_id, shard_index",
                (batch_id,),
            ).fetchall()
            shard_ids = [str(row["shard_id"]) for row in rows]
            self._conn.execute(
                """
                UPDATE shard_records
                   SET status = 'stale', outcome = 'stale', error_code = 'E_STALE',
                       error_message = '参数版本已变更，旧分片轨迹不得参与当前汇总'
                 WHERE batch_id = ? AND status != 'unknown'
                """,
                (batch_id,),
            )
        return sorted(shard_ids)

    def recover_incomplete(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT batch_id FROM runs WHERE status = 'running' ORDER BY rowid"
            ).fetchall()
        recovered: list[dict[str, Any]] = []
        for row in rows:
            batch_id = str(row["batch_id"])
            unknown = self.mark_interrupted(batch_id)
            batch = self.get_batch(batch_id)
            recovered.append(
                {
                    "batch_id": batch_id,
                    "status": batch.status,
                    "unknown_shards": unknown,
                    "exit_report_available": False,
                }
            )
        return recovered

    def get_batch(self, batch_id: str) -> LedgerBatch:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE batch_id = ?", (batch_id,)).fetchone()
            if row is None:
                raise AppError("E_LEDGER", f"账本批次不存在: {batch_id}", {"batch_id": batch_id})
            shard_rows = self._conn.execute(
                """
                SELECT * FROM shard_records
                 WHERE batch_id = ?
                 ORDER BY task_id, shard_index, shard_id
                """,
                (batch_id,),
            ).fetchall()
        return self._batch_from_rows(row, shard_rows)

    def latest_batch(self, *, recover: bool = True) -> LedgerBatch | None:
        if recover:
            self.recover_incomplete()
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
            if row is None:
                return None
            batch_id = str(row["batch_id"])
            shard_rows = self._conn.execute(
                "SELECT * FROM shard_records WHERE batch_id = ? ORDER BY task_id, shard_index, shard_id",
                (batch_id,),
            ).fetchall()
        return self._batch_from_rows(row, shard_rows)

    def _batch_from_rows(self, row: sqlite3.Row, shard_rows: list[sqlite3.Row]) -> LedgerBatch:
        shards = tuple(
            LedgerShard(
                shard_id=str(shard["shard_id"]),
                task_id=str(shard["task_id"]),
                shard_index=int(shard["shard_index"]),
                start=int(shard["start_row"]),
                end=int(shard["end_row"]),
                status=str(shard["status"]),
                resources=int(shard["resources"]),
                queued_at_ms=shard["queued_at_ms"],
                started_at_ms=shard["started_at_ms"],
                ended_at_ms=shard["ended_at_ms"],
                wait_duration_ms=shard["wait_duration_ms"],
                outcome=shard["outcome"],
                result_digest=shard["result_digest"],
                error_code=shard["error_code"],
                error_message=shard["error_message"],
                result_json=shard["result_json"],
            )
            for shard in shard_rows
        )
        return LedgerBatch(
            batch_id=str(row["batch_id"]),
            format_version=int(row["format_version"]),
            parameter_version=int(row["parameter_version"]),
            content_digest=str(row["content_digest"]),
            plan_digest=row["plan_digest"],
            state_json=str(row["state_json"]),
            status=str(row["status"]),
            started_at=str(row["started_at"]),
            ended_at=row["ended_at"],
            result_digest=row["result_digest"],
            result_json=row["result_json"],
            cumulative_digest=row["cumulative_digest"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            exit_report_allowed=bool(row["exit_report_allowed"]),
            exit_report_json=row["exit_report_json"],
            replay_status=row["replay_status"],
            replay_batch_id=row["replay_batch_id"],
            replay_details=row["replay_details"],
            shards=shards,
        )

    def set_replay_result(self, batch_id: str, *, status: str, replay_batch_id: str | None, details: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE runs SET replay_status = ?, replay_batch_id = ?, replay_details = ? WHERE batch_id = ?",
                (status, replay_batch_id, canonical_json_text(details), batch_id),
            )

    def ledger_info(self) -> dict[str, Any]:
        latest = self.latest_batch(recover=False)
        with self._lock:
            meta = {
                str(row["key"]): str(row["value"])
                for row in self._conn.execute("SELECT key, value FROM ledger_meta").fetchall()
            }
        return {
            "format_version": LEDGER_FORMAT_VERSION,
            "meta": dict(sorted(meta.items())),
            "latest": latest.to_dict() if latest is not None else None,
        }


@dataclass(frozen=True)
class SnapshotRestore:
    path: str
    replaced: bool
    mismatched_fields: tuple[str, ...]
    parameter_version: int
    content_digest: str
    run_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "replaced": self.replaced,
            "mismatched_fields": list(self.mismatched_fields),
            "parameter_version": self.parameter_version,
            "content_digest": self.content_digest,
            "run_id": self.run_id,
        }


class SnapshotStore:
    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)

    @property
    def default_path(self) -> Path:
        return self.directory / "session.json"

    def save(self, path: str | os.PathLike[str] | None, *, workspace: Workspace, run: Any) -> Path:
        target = Path(path) if path is not None else self.default_path
        payload = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "parameter_version": workspace.parameter_version,
            "content_digest": workspace.content_digest,
            "state": workspace.export_payload(),
            "run": run.to_dict() if run is not None and hasattr(run, "to_dict") else None,
        }
        try:
            write_json_atomic(target, payload)
        except OSError as exc:
            raise AppError("E_DATA", f"保存会话快照失败: {target}: {exc}", {"path": str(target)}) from exc
        return target

    def restore(self, path: str | os.PathLike[str], *, workspace: Workspace) -> SnapshotRestore:
        target = Path(path)
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise AppError("E_DATA", f"无法读取快照: {target}: {exc}", {"path": str(target)}) from exc
        payload = loads_strict(text, source=str(target))
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("format_version"), int)
            or isinstance(payload.get("format_version"), bool)
            or payload.get("format_version") != SNAPSHOT_FORMAT_VERSION
        ):
            raise AppError("E_DATA", f"快照格式版本不受支持: {target}", {"path": str(target)})
        if not isinstance(payload.get("parameter_version"), int) or isinstance(payload.get("parameter_version"), bool):
            raise AppError("E_DATA", f"快照参数版本必须是整数: {target}", {"path": str(target)})
        state = workspace_state_from_export(payload.get("state"))
        content_digest = state.content_digest()
        declared_digest = payload.get("content_digest")
        if declared_digest != content_digest:
            raise AppError(
                "E_DATA",
                "快照内容摘要与状态不一致，拒绝恢复",
                {"path": str(target), "expected": declared_digest, "actual": content_digest},
            )
        mismatched: list[str] = []
        if int(payload.get("parameter_version", -1)) != state.parameter_version:
            mismatched.append("parameter_version")
        if declared_digest != workspace.content_digest:
            mismatched.append("content_digest")
        if workspace.parameter_version != state.parameter_version and "parameter_version" not in mismatched:
            mismatched.append("parameter_version")

        workspace.restore_state(state)
        run_payload = payload.get("run")
        workspace.current_run = run_payload
        return SnapshotRestore(
            path=str(target),
            replaced=True,
            mismatched_fields=tuple(sorted(mismatched)),
            parameter_version=state.parameter_version,
            content_digest=content_digest,
            run_id=run_payload.get("run_id") if isinstance(run_payload, dict) else None,
        )
