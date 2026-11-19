"""Transactional execution service, injection, recovery, replay, and snapshots."""

from __future__ import annotations

import copy
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from .canonical import canonical_json_bytes, digest_value, load_json_file
from .engine import BlockTensorBank, TaskResult, aggregate_task, evaluate_shard
from .errors import AppError, as_app_error
from .ledger import ExecutionLedger, SnapshotRestore, SnapshotStore
from .logging_utils import JsonLogger
from .model import Workspace, parse_data_payload, parse_tasks_payload, workspace_state_from_parts
from .planner import ExecutionPlan, compile_workspace
from .scheduler import (
    RunControl,
    ScheduleReport,
    SchedulerBudgetError,
    WorkItem,
    build_work_items,
    run_schedule,
)


@dataclass
class RunReport:
    run_id: str
    batch_id: str
    status: str
    parameter_version: int
    content_digest: str
    plan_digest: str
    global_budget: int
    threads: int
    schedule: ScheduleReport
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    deterministic_payload: dict[str, Any] | None = None
    overall_digest: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    stale: bool = False
    ledger_status: str | None = None
    on_stale: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    @property
    def result_bytes(self) -> bytes | None:
        return canonical_json_bytes(self.deterministic_payload) if self.deterministic_payload is not None else None

    def mark_stale(self) -> None:
        if self.stale:
            return
        self.stale = True
        if self.status == "committed":
            self.status = "stale"
        if self.on_stale is not None:
            self.on_stale()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "batch_id": self.batch_id,
            "status": "stale" if self.stale else self.status,
            "parameter_version": self.parameter_version,
            "content_digest": self.content_digest,
            "plan_digest": self.plan_digest,
            "global_budget": self.global_budget,
            "threads": self.threads,
            "schedule": self.schedule.to_dict(),
            "task_results": {task_id: result.to_dict() for task_id, result in sorted(self.task_results.items())},
            "deterministic_payload": self.deterministic_payload,
            "overall_digest": self.overall_digest,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "stale": self.stale,
            "ledger_status": self.ledger_status,
        }


@dataclass(frozen=True)
class InjectionResult:
    kind: str
    data_payload: dict[str, Any]
    tasks_payload: dict[str, Any]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "note": self.note}


def _normalize_injection(kind: str | None) -> str | None:
    if kind is None:
        return None
    normalized = kind.strip().lower().replace("_", "-")
    aliases = {
        "range": "range",
        "e-range": "range",
        "nullmap": "nullmap",
        "e-nullmap": "nullmap",
        "budget": "budget",
        "e-budget": "budget",
        "none": None,
    }
    if normalized not in aliases:
        raise AppError("E_DATA", f"未知异常注入类型: {kind}", {"allowed": ["range", "nullmap", "budget"]})
    return aliases[normalized]


def apply_injection(data_payload: dict[str, Any], tasks_payload: dict[str, Any], kind: str | None) -> InjectionResult:
    """Mutate only a detached copy of input parameters; never synthesize outcomes."""

    normalized = _normalize_injection(kind)
    data = copy.deepcopy(data_payload)
    tasks = copy.deepcopy(tasks_payload)
    if normalized is None:
        return InjectionResult("none", data, tasks, "未注入异常")

    if normalized == "range":
        blocks = data.get("blocks") if isinstance(data, dict) else None
        raw_tasks = tasks.get("tasks") if isinstance(tasks, dict) else None
        if not isinstance(blocks, list) or not blocks or not isinstance(raw_tasks, list) or not raw_tasks:
            raise AppError("E_DATA", "range 注入需要非空数据块与任务")
        first_block = blocks[0]
        row_count = first_block.get("row_count")
        if not isinstance(row_count, int):
            columns = first_block.get("columns", [])
            row_count = len(columns[0].get("values", [])) if isinstance(columns, list) and columns else 0
        task = raw_tasks[0]
        if isinstance(task.get("shards"), list) and task["shards"]:
            task["shards"][0]["end"] = row_count + 1
        else:
            resources = task.get("resources_per_shard", task.get("resources", 1))
            if isinstance(resources, list):
                resources = resources[0] if resources else 1
            task.pop("shard_rows", None)
            task.pop("resources_per_shard", None)
            task.pop("resources", None)
            task["shards"] = [{"start": 0, "end": row_count + 1, "resources": int(resources or 1)}]
        tasks["parameter_version"] = int(tasks.get("parameter_version", 1)) + 1
        return InjectionResult("range", data, tasks, "首个任务分片终点被设为 row_count + 1，等待 E_RANGE 校验")

    if normalized == "nullmap":
        blocks = data.get("blocks") if isinstance(data, dict) else None
        if not isinstance(blocks, list) or not blocks:
            raise AppError("E_DATA", "nullmap 注入需要非空数据块")
        columns = blocks[0].get("columns") if isinstance(blocks[0], dict) else None
        if not isinstance(columns, list) or not columns:
            raise AppError("E_DATA", "nullmap 注入需要非空列")
        column = columns[0]
        bitmap = column.get("null_bitmap")
        if isinstance(bitmap, str):
            bitmap = [int(ch) for ch in bitmap]
        if not isinstance(bitmap, list):
            raise AppError("E_DATA", "nullmap 注入需要 null_bitmap")
        if bitmap:
            bitmap = bitmap[:-1]
        else:
            bitmap = [0]
        column["null_bitmap"] = bitmap
        return InjectionResult("nullmap", data, tasks, "首个列的空值位图被错位，等待 E_NULLMAP 校验")

    tasks["global_budget"] = 0
    tasks["parameter_version"] = int(tasks.get("parameter_version", 1)) + 1
    return InjectionResult("budget", data, tasks, "全局预算被设为 0，等待真实调度报告 E_BUDGET")


def load_injected_workspace(
    data_path: str | os.PathLike[str],
    tasks_path: str | os.PathLike[str],
    *,
    injection: str | None = None,
) -> tuple[Workspace, InjectionResult]:
    data_payload = load_json_file(data_path)
    tasks_payload = load_json_file(tasks_path)
    injected = apply_injection(data_payload, tasks_payload, injection)
    blocks = parse_data_payload(injected.data_payload)
    tasks = parse_tasks_payload(injected.tasks_payload)
    state = workspace_state_from_parts(blocks, tasks)
    if injected.kind == "none":
        workspace = Workspace(state, data_path=data_path, tasks_path=tasks_path)
    else:
        workspace = Workspace(state, data_path=None, tasks_path=None)
    return workspace, injected


class ExecutionService:
    def __init__(
        self,
        workspace: Workspace,
        ledger: ExecutionLedger,
        *,
        snapshot_store: SnapshotStore | None = None,
        logger: JsonLogger | None = None,
    ) -> None:
        self.workspace = workspace
        self.ledger = ledger
        self.snapshot_store = snapshot_store
        self.logger = logger or JsonLogger(None)
        self.last_report: RunReport | None = None

    def _report_from_schedule(
        self,
        *,
        run_id: str,
        plan: ExecutionPlan,
        schedule: ScheduleReport,
        status: str,
        task_results: dict[str, TaskResult] | None = None,
        deterministic_payload: dict[str, Any] | None = None,
        overall_digest: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        stale: bool = False,
        ledger_status: str | None = None,
    ) -> RunReport:
        return RunReport(
            run_id=run_id,
            batch_id=run_id,
            status=status,
            parameter_version=plan.parameter_version,
            content_digest=plan.content_digest,
            plan_digest=plan.plan_digest,
            global_budget=plan.global_budget,
            threads=plan.threads,
            schedule=schedule,
            task_results=task_results or {},
            deterministic_payload=deterministic_payload,
            overall_digest=overall_digest,
            error_code=error_code,
            error_message=error_message,
            stale=stale,
            ledger_status=ledger_status,
        )

    def execute(self, *, force_crash_after: int | None = None, save_snapshot: bool = True) -> RunReport:
        if force_crash_after is not None and force_crash_after < 1:
            raise AppError("E_DATA", "force_crash_after 必须 >= 1")
        plan = compile_workspace(self.workspace.state)
        run_id = uuid.uuid4().hex
        work_items = build_work_items(plan)
        control = RunControl(run_id, work_items)
        self.workspace.register_run(control)
        bank = BlockTensorBank(self.workspace.state)
        committed_count = 0
        ledger_started = False
        report: RunReport | None = None

        self.logger.log(
            "run_start",
            run_id=run_id,
            parameter_version=plan.parameter_version,
            content_digest=plan.content_digest,
            plan_digest=plan.plan_digest,
            shard_count=plan.shard_count,
            global_budget=plan.global_budget,
            threads=plan.threads,
        )

        def execute_item(item: WorkItem) -> Any:
            if item.task.delay_ms:
                time.sleep(item.task.delay_ms / 1000.0)
            return evaluate_shard(item.task, item.shard, bank)

        def on_start(record: Any) -> None:
            self.ledger.mark_running(run_id, record.shard_id, record.started_at_ms or 0)

        def on_batch(records: list[Any]) -> None:
            nonlocal committed_count
            self.ledger.commit_shard_batch(run_id, records)
            committed_count += len(records)
            self.logger.log(
                "shard_batch_committed",
                run_id=run_id,
                shard_ids=[record.shard_id for record in records],
                committed_count=committed_count,
            )
            if force_crash_after is not None and committed_count >= force_crash_after:
                self.logger.log(
                    "forced_crash",
                    level="ERROR",
                    code="E_RECOVERY",
                    run_id=run_id,
                    committed_count=committed_count,
                    message="模拟进程强制终止；不会生成退出报告",
                )
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                finally:
                    os._exit(137)

        try:
            self.ledger.begin_batch(
                batch_id=run_id,
                state_payload=self.workspace.export_payload(),
                plan_payload=plan.to_dict(),
                planned_shards=[shard.to_dict() for task in plan.tasks for shard in task.shards],
            )
            ledger_started = True
            schedule = run_schedule(
                work_items,
                global_budget=plan.global_budget,
                task_quotas={task.task_id: task.quota for task in plan.tasks},
                max_workers=plan.threads,
                execute_fn=execute_item,
                control=control,
                on_start=on_start,
                on_batch=on_batch,
            )

            def make_stale(reason: str) -> RunReport:
                for record in schedule.records:
                    record.status = "stale"
                self.ledger.mark_stale_batch(run_id)
                self.ledger.finish_batch(
                    run_id,
                    status="stale",
                    result_digest=None,
                    result_payload=None,
                    error_code="E_STALE",
                    error_message=reason,
                    exit_report_allowed=False,
                )
                stale_report = self._report_from_schedule(
                    run_id=run_id,
                    plan=plan,
                    schedule=schedule,
                    status="stale",
                    error_code="E_STALE",
                    error_message=reason,
                    stale=True,
                    ledger_status="stale",
                )
                self.last_report = stale_report
                self.logger.log("run_stale", level="WARNING", code="E_STALE", run_id=run_id, message=reason)
                return stale_report

            if control.cancelled or not self.workspace.is_current(plan.parameter_version, plan.content_digest):
                return make_stale(control.reason or "parameters changed before completion")

            if schedule.stale:
                return make_stale(schedule.error_message or "parameters changed")

            failed = [record for record in schedule.records if record.outcome != "success"]
            if failed:
                first = failed[0]
                error = AppError(
                    first.error_code or "E_INTERNAL",
                    f"分片 {first.shard_id} 执行失败: {first.error_message or 'unknown error'}",
                    {"run_id": run_id, "shard_id": first.shard_id},
                )
                self.ledger.finish_batch(
                    run_id,
                    status="failed",
                    result_digest=None,
                    result_payload=None,
                    error_code=error.code,
                    error_message=error.message,
                    exit_report_allowed=False,
                )
                report = self._report_from_schedule(
                    run_id=run_id,
                    plan=plan,
                    schedule=schedule,
                    status="failed",
                    error_code=error.code,
                    error_message=error.message,
                    ledger_status="failed",
                )
                self.last_report = report
                raise error

            records_by_task: dict[str, list[Any]] = {task.task_id: [] for task in plan.tasks}
            for record in schedule.records:
                records_by_task[record.task_id].append(record.result)
            block_map = self.workspace.state.by_block()
            task_results: dict[str, TaskResult] = {}
            for task in plan.tasks:
                task_results[task.task_id] = aggregate_task(
                    task,
                    records_by_task[task.task_id],
                    block_map[task.block_id].row_count,
                )
            deterministic_payload = {
                "status": "committed",
                "parameter_version": plan.parameter_version,
                "content_digest": plan.content_digest,
                "plan_digest": plan.plan_digest,
                "tasks": [task_results[task.task_id].payload for task in plan.tasks],
            }
            overall_digest = digest_value(deterministic_payload)

            if control.cancelled or not self.workspace.is_current(plan.parameter_version, plan.content_digest):
                return make_stale(control.reason or "parameters changed during aggregation")

            self.ledger.finish_batch(
                run_id,
                status="committed",
                result_digest=overall_digest,
                result_payload=deterministic_payload,
                exit_report_allowed=True,
            )
            report = self._report_from_schedule(
                run_id=run_id,
                plan=plan,
                schedule=schedule,
                status="committed",
                task_results=task_results,
                deterministic_payload=deterministic_payload,
                overall_digest=overall_digest,
                ledger_status="committed",
            )
            def mark_published_run_stale() -> None:
                try:
                    self.ledger.mark_stale_batch(run_id)
                    self.ledger.finish_batch(
                        run_id,
                        status="stale",
                        result_digest=overall_digest,
                        result_payload=None,
                        error_code="E_STALE",
                        error_message="参数版本已变更，旧批次整体失效",
                        exit_report_allowed=False,
                    )
                    self.logger.log("published_run_marked_stale", level="WARNING", code="E_STALE", run_id=run_id)
                except AppError as exc:
                    self.logger.error("published_run_stale_mark_failed", exc, run_id=run_id)

            report.on_stale = mark_published_run_stale
            if not self.workspace.publish_run_if_current(report, plan.parameter_version, plan.content_digest):
                report.mark_stale()
                self.last_report = report
                return report
            if save_snapshot and self.snapshot_store is not None:
                try:
                    path = self.snapshot_store.save(None, workspace=self.workspace, run=report)
                    self.logger.log("snapshot_saved", run_id=run_id, path=str(path))
                except AppError as exc:
                    self.logger.error("snapshot_save_failed", exc, run_id=run_id)
            self.last_report = report
            self.logger.log(
                "run_committed",
                run_id=run_id,
                result_digest=overall_digest,
                task_count=len(task_results),
                peak_resources=schedule.peak_resources,
            )
            return report

        except SchedulerBudgetError as exc:
            schedule = exc.report
            if ledger_started:
                self.ledger.mark_budget_blocked(run_id)
                self.ledger.finish_batch(
                    run_id,
                    status="budget_blocked",
                    result_digest=None,
                    result_payload=None,
                    error_code="E_BUDGET",
                    error_message=exc.message,
                    exit_report_allowed=False,
                )
            report = self._report_from_schedule(
                run_id=run_id,
                plan=plan,
                schedule=schedule,
                status="budget_blocked",
                error_code="E_BUDGET",
                error_message=exc.message,
                ledger_status="budget_blocked",
            )
            self.last_report = report
            self.workspace.current_run = report
            self.logger.error("run_budget_blocked", exc, run_id=run_id)
            exc.details["run_report"] = report.to_dict()
            raise
        except AppError:
            raise
        except Exception as exc:
            error = as_app_error(exc)
            if ledger_started:
                try:
                    self.ledger.finish_batch(
                        run_id,
                        status="failed",
                        result_digest=None,
                        result_payload=None,
                        error_code=error.code,
                        error_message=error.message,
                        exit_report_allowed=False,
                    )
                except Exception:
                    pass
            self.logger.error("run_failed", error, run_id=run_id)
            raise error from exc
        finally:
            self.workspace.unregister_run(control)

    def recover(self) -> dict[str, Any]:
        interrupted = self.ledger.recover_incomplete()
        latest = self.ledger.latest_batch(recover=False)
        return {
            "interrupted_batches": interrupted,
            "latest": latest.to_dict() if latest is not None else None,
            "exit_report_available": bool(latest.exit_report_allowed) if latest is not None else False,
            "exit_report": latest.result_payload() if latest is not None and latest.exit_report_allowed else None,
            "note": "强制终止时的退出报告不可用；未提交分片只能标记 unknown 并重跑",
        }

    def replay(self, *, batch_id: str | None = None, save_snapshot: bool = True) -> dict[str, Any]:
        self.ledger.recover_incomplete()
        if batch_id is None:
            batch = self.ledger.latest_batch(recover=False)
        else:
            batch = self.ledger.get_batch(batch_id)
        if batch is None:
            raise AppError("E_REPLAY", "账本中没有可重放批次")
        if batch.format_version != 1:
            raise AppError("E_REPLAY", f"账本批次格式版本不支持: {batch.format_version}", {"batch_id": batch.batch_id})
        try:
            replay_workspace = Workspace.from_export_payload(batch.state_payload())
        except AppError as exc:
            raise AppError(
                "E_REPLAY",
                f"无法依据账本参数版本重建状态: {exc.message}",
                {"batch_id": batch.batch_id, "cause_code": exc.code},
            ) from exc
        if replay_workspace.parameter_version != batch.parameter_version:
            raise AppError(
                "E_REPLAY",
                "账本参数版本与状态快照不一致",
                {"batch_id": batch.batch_id, "ledger_parameter_version": batch.parameter_version, "state_parameter_version": replay_workspace.parameter_version},
            )
        if replay_workspace.content_digest != batch.content_digest:
            raise AppError(
                "E_REPLAY",
                "账本内容摘要与参数状态不一致",
                {"batch_id": batch.batch_id, "ledger_content_digest": batch.content_digest, "state_content_digest": replay_workspace.content_digest},
            )
        replay_service = ExecutionService(
            replay_workspace,
            self.ledger,
            snapshot_store=self.snapshot_store,
            logger=self.logger,
        )
        replay_report = replay_service.execute(save_snapshot=save_snapshot)
        if replay_report.status != "committed":
            details = {
                "batch_id": batch.batch_id,
                "replay_batch_id": replay_report.batch_id,
                "status": replay_report.status,
                "error_code": replay_report.error_code,
                "error_message": replay_report.error_message,
            }
            self.ledger.set_replay_result(batch.batch_id, status="mismatch", replay_batch_id=replay_report.batch_id, details=details)
            raise AppError("E_REPLAY", "重放批次未能完整提交", details)

        actual_by_id = {record.shard_id: record for record in replay_report.schedule.records}
        compared = 0
        unknown_rerun: list[str] = []
        for expected in batch.shards:
            actual = actual_by_id.get(expected.shard_id)
            if expected.status == "unknown":
                unknown_rerun.append(expected.shard_id)
                if actual is None or actual.outcome != "success":
                    details = {
                        "batch_id": batch.batch_id,
                        "replay_batch_id": replay_report.batch_id,
                        "shard_id": expected.shard_id,
                        "field": "outcome",
                        "expected": "success (unknown shard rerun)",
                        "actual": None if actual is None else actual.outcome,
                        "basis": "recovery requires rerunning an uncommitted running/queued shard",
                    }
                    self.ledger.set_replay_result(batch.batch_id, status="mismatch", replay_batch_id=replay_report.batch_id, details=details)
                    raise AppError("E_REPLAY", f"重放不一致: {expected.shard_id}.outcome", details)
                continue
            if expected.status not in {"completed", "failed"}:
                continue
            compared += 1
            if actual is None:
                details = {
                    "batch_id": batch.batch_id,
                    "replay_batch_id": replay_report.batch_id,
                    "shard_id": expected.shard_id,
                    "field": "presence",
                    "expected": "present",
                    "actual": None,
                    "basis": "committed ledger shard record has no replay counterpart",
                }
                self.ledger.set_replay_result(batch.batch_id, status="mismatch", replay_batch_id=replay_report.batch_id, details=details)
                raise AppError("E_REPLAY", f"重放不一致: {expected.shard_id} 缺失", details)
            actual_outcome = actual.outcome
            actual_digest = getattr(actual.result, "result_digest", None)
            fields = [
                ("outcome", expected.outcome, actual_outcome),
                ("result_digest", expected.result_digest, actual_digest),
                ("resources", expected.resources, actual.resources),
            ]
            for field_name, expected_value, actual_value in fields:
                if expected_value != actual_value:
                    details = {
                        "batch_id": batch.batch_id,
                        "replay_batch_id": replay_report.batch_id,
                        "shard_id": expected.shard_id,
                        "field": field_name,
                        "expected": expected_value,
                        "actual": actual_value,
                        "basis": "first mismatch between committed ledger record and deterministic replay",
                    }
                    self.ledger.set_replay_result(batch.batch_id, status="mismatch", replay_batch_id=replay_report.batch_id, details=details)
                    raise AppError("E_REPLAY", f"重放不一致: {expected.shard_id}.{field_name}", details)

        if batch.result_digest is not None and replay_report.overall_digest != batch.result_digest:
            details = {
                "batch_id": batch.batch_id,
                "replay_batch_id": replay_report.batch_id,
                "shard_id": "<run>",
                "field": "result_digest",
                "expected": batch.result_digest,
                "actual": replay_report.overall_digest,
                "basis": "aggregate digest mismatch after all shard-level comparisons passed",
            }
            self.ledger.set_replay_result(batch.batch_id, status="mismatch", replay_batch_id=replay_report.batch_id, details=details)
            raise AppError("E_REPLAY", "重放不一致: 任务总结果摘要", details)

        details = {
            "batch_id": batch.batch_id,
            "replay_batch_id": replay_report.batch_id,
            "parameter_version": batch.parameter_version,
            "content_digest": batch.content_digest,
            "compared_shards": compared,
            "unknown_rerun_shards": unknown_rerun,
            "result_digest": replay_report.overall_digest,
            "exit_report_available_for_interrupted_run": False,
        }
        self.ledger.set_replay_result(batch.batch_id, status="consistent", replay_batch_id=replay_report.batch_id, details=details)
        self.logger.log("replay_consistent", batch_id=batch.batch_id, replay_batch_id=replay_report.batch_id, compared=compared)
        return details

    def restore_snapshot(self, path: str | os.PathLike[str]) -> SnapshotRestore:
        if self.snapshot_store is None:
            raise AppError("E_DATA", "未配置快照目录")
        restored = self.snapshot_store.restore(path, workspace=self.workspace)
        self.logger.log(
            "snapshot_restored",
            path=restored.path,
            parameter_version=restored.parameter_version,
            content_digest=restored.content_digest,
            mismatched_fields=list(restored.mismatched_fields),
        )
        return restored
