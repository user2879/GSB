"""训练运行器：把确定性规划交给线程池执行，并管理快照生命周期。

规划（grid.build_schedule_plan）与执行分离：线程池只负责并发执行纯确定性
训练，所有预算/配额裁决、决策顺序都来自稳定规划，因此 max_workers 变化
不会改变任何可比较结果。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from . import grid
from .config import ExperimentConfig
from .data import PreparedData, prepare_data
from .errors import (
    E_BUDGET_EXHAUSTED,
    E_GRACEFUL_STOP,
    E_STAGE_WRITE_FAILED,
    E_TASK_FAILED,
    E_TASK_QUOTA_EXCEEDED,
    FailureLog,
)
from .grid import SchedulePlan
from .models import Checkpoint, TaskResult, train_task
from .snapshot import SnapshotStore
from .util import monotonic

STAGE_SUMMARY = "summary"


@dataclass
class RunOutcome:
    config_hash: str
    data: PreparedData
    plan: SchedulePlan
    results: dict[str, TaskResult]
    failures: FailureLog
    summary: dict[str, Any]
    resume_info: Any
    graceful_stopped: bool
    snapshot_versions: dict[str, int]
    monotonic_events: list[dict[str, Any]] = field(default_factory=list)


class StopController:
    """跨线程的优雅退出信号。"""

    def __init__(self) -> None:
        self._event = threading.Event()

    def request_stop(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()


def run_training(
    config: ExperimentConfig,
    store: SnapshotStore,
    failure_log: FailureLog | None = None,
    stop_controller: StopController | None = None,
    failpoint: Callable[[str, int, str], None] | None = None,
    force_full: bool = False,
) -> RunOutcome:
    failure_log = failure_log if failure_log is not None else FailureLog()
    stop_controller = stop_controller or StopController()
    config_hash = config.config_hash

    resume_info = store.resume_or_raise(config_hash)
    if force_full or resume_info is None:
        if force_full:
            store.reset()
            resume_info = None
        data = prepare_data(config)
        store.save_data(data, config_hash)
    else:
        data = store.load_data()
        fresh_fingerprint = prepare_data(config).data_fingerprint
        if data.data_fingerprint != fresh_fingerprint:
            data = prepare_data(config)
            store.save_data(data, config_hash)

    tasks = grid.expand_grid(config, data)
    plan = grid.build_schedule_plan(config, tasks)
    planned_by_key = plan.by_key()

    manifest = store.load_manifest()
    completed: set[str] = set(manifest.get("completed_tasks", []))
    # 配额标记需要跨恢复保留：按任务键重建，避免重复与丢失。
    for record in manifest.get("failures", []):
        if record.get("code") == E_TASK_QUOTA_EXCEEDED and record.get("task_key"):
            failure_log.add(
                code=E_TASK_QUOTA_EXCEEDED,
                stage=record.get("stage", "train"),
                message=record.get("message", "单方案轮次配额触及"),
                task_key=record["task_key"],
                epoch=record.get("epoch"),
            )
    results: dict[str, TaskResult] = {}
    for name in store.list_result_keys():
        loaded = _load_result_by_file(store, name)
        if loaded is not None:
            results[loaded.task_key] = loaded
            completed.add(loaded.task_key)

    runnable = [
        item
        for item in plan.ordered_tasks
        if item.starts and item.task.task_key not in completed
    ]

    clock_events: list[dict[str, Any]] = []
    clock_lock = threading.Lock()
    consumed_epochs_holder = {"value": int(manifest.get("consumed_epochs", 0))}
    graceful_stopped = False

    def execute(planned: grid.PlannedTask) -> tuple[str, TaskResult | None, Checkpoint | None]:
        task = planned.task
        if failpoint is not None:
            failpoint("data_split", 0, task.task_key)
        started = monotonic()
        with clock_lock:
            clock_events.append(
                {"task_key": task.task_key, "phase": "start", "monotonic": started}
            )
        try:
            checkpoint = store.load_checkpoint(task.task_key)
            already = checkpoint.epoch if checkpoint is not None else 0
            remaining_epochs = max(0, planned.planned_epochs - already)

            def task_failpoint(stage: str, epoch: int) -> None:
                if failpoint is not None:
                    failpoint(stage, epoch, task.task_key)

            snapshot_interval = config.scheduler.snapshot_interval

            def on_epoch(epoch_no: int, epoch_checkpoint: Checkpoint) -> None:
                if epoch_no % snapshot_interval != 0:
                    return
                if failpoint is not None:
                    failpoint("snapshot_write", epoch_no, task.task_key)
                store.save_checkpoint(task.task_key, epoch_checkpoint)

            result, new_checkpoint = train_task(
                spec=task.spec,
                data=data,
                task_key=task.task_key,
                cache_key=task.cache_key,
                seed=task.seed,
                epoch_limit=remaining_epochs,
                checkpoint=checkpoint,
                stop_requested=stop_controller.is_set,
                failpoint=task_failpoint,
                on_epoch=on_epoch,
            )
            # 完成当前轮次后先提交检查点（整体替换），再提交结果。
            store.save_checkpoint(task.task_key, new_checkpoint)
            finished_plan = result.converged or (
                result.epochs_executed >= planned.planned_epochs
            )
            if planned.quota_capped and not result.converged:
                result.status = grid.QUOTA_EXCEEDED
                result.quota_touched = True
                failure_log.add(
                    code=E_TASK_QUOTA_EXCEEDED,
                    stage="train",
                    message=(
                        f"任务触及单方案轮次配额 "
                        f"{config.scheduler.per_scheme_epoch_quota}，已中止"
                    ),
                    task_key=task.task_key,
                    epoch=result.epochs_executed,
                )
            elif planned.budget_limited and not result.converged:
                result.status = grid.BUDGET_TRUNCATED
            ended = monotonic()
            result.started_monotonic = started
            result.ended_monotonic = ended
            with clock_lock:
                clock_events.append(
                    {"task_key": task.task_key, "phase": "end", "monotonic": ended}
                )
                consumed_epochs_holder["value"] += result.epochs_executed - already
                completed_snapshot = sorted(
                    (completed | {task.task_key}) if finished_plan else completed
                )
                consumed_snapshot = consumed_epochs_holder["value"]
            if finished_plan:
                store.save_result(
                    result,
                    new_checkpoint,
                    completed_snapshot,
                    consumed_snapshot,
                    failure_log.to_list(),
                )
            else:
                store.persist_progress(
                    completed_snapshot,
                    consumed_snapshot,
                    failure_log.to_list(),
                )
                store.update_failures(failure_log.to_list())
            if finished_plan:
                return task.task_key, result, new_checkpoint
            return task.task_key, None, new_checkpoint
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # 异常隔离：记录后继续其余任务
            try:
                failed_checkpoint = store.load_checkpoint(task.task_key)
            except Exception:
                failed_checkpoint = None
            code = (
                E_STAGE_WRITE_FAILED
                if "snapshot" in type(exc).__name__.lower()
                or "write" in type(exc).__name__.lower()
                else E_TASK_FAILED
            )
            failure_log.add(
                code=code,
                stage="train",
                message=str(exc),
                task_key=task.task_key,
                epoch=failed_checkpoint.epoch if failed_checkpoint is not None else None,
                exception_type=type(exc).__name__,
            )
            store.update_failures(failure_log.to_list())
            ended = monotonic()
            with clock_lock:
                clock_events.append(
                    {"task_key": task.task_key, "phase": "failed", "monotonic": ended}
                )
            return task.task_key, None, None

    if runnable and not stop_controller.is_set():
        with ThreadPoolExecutor(
            max_workers=config.scheduler.max_workers,
            thread_name_prefix="regbench-worker",
        ) as pool:
            futures = [pool.submit(execute, item) for item in runnable]
            for future in futures:
                task_key, result, _ = future.result()
                if result is not None:
                    results[task_key] = result
                    completed.add(task_key)
                if stop_controller.is_set():
                    graceful_stopped = True

    if stop_controller.is_set() and runnable:
        graceful_stopped = True
        failure_log.add(
            code=E_GRACEFUL_STOP,
            stage="runner",
            message="收到可处理退出信号，已完成当前轮次快照原子替换",
        )
        store.update_failures(failure_log.to_list())

    # 预算耗尽但从未启动的任务进入失败清单，保留就绪位置。
    for task_key in plan.exhausted_tasks:
        failure_log.add(
            code=E_BUDGET_EXHAUSTED,
            stage="scheduler",
            message="全局轮次预算耗尽，任务保留在就绪队列，未启动",
            task_key=task_key,
        )
    store.update_failures(failure_log.to_list())

    summary = build_summary(config, plan, results, failure_log, store)
    store.save_summary(summary)
    manifest_after = store.load_manifest()

    return RunOutcome(
        config_hash=config_hash,
        data=data,
        plan=plan,
        results=results,
        failures=failure_log,
        summary=summary,
        resume_info=resume_info,
        graceful_stopped=graceful_stopped,
        snapshot_versions=manifest_after.get("stage_versions", {}),
        monotonic_events=clock_events,
    )


def build_summary(
    config: ExperimentConfig,
    plan: SchedulePlan,
    results: dict[str, TaskResult],
    failure_log: FailureLog,
    store: SnapshotStore,
) -> dict[str, Any]:
    by_scheme: dict[str, list[dict[str, Any]]] = {}
    failed_keys = {
        record.task_key
        for record in failure_log.records
        if record.code in (E_TASK_FAILED, E_STAGE_WRITE_FAILED)
    }
    for planned in plan.ordered_tasks:
        task_key = planned.task.task_key
        if task_key in failed_keys or task_key not in results:
            continue
        result = results[task_key]
        by_scheme.setdefault(result.scheme, []).append(
            {
                "task_key": result.task_key,
                "penalty": result.penalty,
                "l1_ratio": result.l1_ratio,
                "status": result.status,
                "epochs_executed": result.epochs_executed,
                "convergence_epoch": result.convergence_epoch,
                "validation": result.validation,
                "final_weight_summary": result.final_weight_summary,
                "quota_touched": result.quota_touched,
            }
        )
    grid_total = len(plan.ordered_tasks)
    grid_done = sum(
        1
        for item in plan.ordered_tasks
        if item.task.task_key in results
        and item.task.task_key not in failed_keys
    )
    summary = {
        "schemes": {key: sorted(value, key=lambda row: row["task_key"]) for key, value in by_scheme.items()},
        "grid_completion": {
            "total": grid_total,
            "completed": grid_done,
            "ratio": (grid_done / grid_total) if grid_total else 0.0,
        },
        "budget": {
            "global_budget": config.scheduler.global_epoch_budget,
            "consumed": plan.consumed_budget,
            "remaining": plan.remaining_budget,
            "per_scheme_quota": config.scheduler.per_scheme_epoch_quota,
            "quota_exceeded_tasks": list(plan.quota_tasks),
            "budget_exhausted_tasks": list(plan.exhausted_tasks),
        },
        "failures": failure_log.to_list(),
    }
    return summary


def _load_result_by_file(store: SnapshotStore, file_name: str) -> TaskResult | None:
    import json
    import os

    path = os.path.join(store.train_dir, file_name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        from .snapshot import task_result_from_dict

        return task_result_from_dict(payload)
    except (OSError, KeyError):
        return None
