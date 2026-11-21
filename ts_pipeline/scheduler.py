"""Stable task keys, deterministic budget allocation and bounded parallelism."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from .canonical import fingerprint
from .config import Config
from .errors import (
    E_POINT_BUDGET_EXHAUSTED,
    E_SERIES_QUOTA_EXHAUSTED,
    Failure,
)


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskPlan:
    series_id: str
    window_index: int
    model_name: str
    window_key: str
    train_key: str
    prediction_key: str
    metric_key: str
    series_fingerprint: str
    model_param_fingerprint: str
    quantization_fingerprint: str
    window_spec_fingerprint: str
    tolerance: float
    cost_points: int
    window: dict[str, Any]
    status: str = "pending"

    @property
    def task_key(self) -> str:
        return f"{self.series_id}#W{self.window_index:08d}#{self.model_name}"

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return (self.series_id, self.window_index, self.model_name)

    def cache_plan(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "series_id": self.series_id,
            "window_index": self.window_index,
            "model_name": self.model_name,
            "window_key": self.window_key,
            "train_key": self.train_key,
            "prediction_key": self.prediction_key,
            "metric_key": self.metric_key,
            "series_fingerprint": self.series_fingerprint,
            "window_spec_fingerprint": self.window_spec_fingerprint,
            "model_parameter_fingerprint": self.model_param_fingerprint,
            "quantization_fingerprint": self.quantization_fingerprint,
            "tolerance": self.tolerance,
            "cost_points": self.cost_points,
            "layers": ["window", "train", "prediction", "metric"],
        }

    def fingerprint_record(self) -> dict[str, Any]:
        return {
            "series_fingerprint": self.series_fingerprint,
            "window_spec_fingerprint": self.window_spec_fingerprint,
            "window_key": self.window_key,
            "model_parameter_fingerprint": self.model_param_fingerprint,
            "quantization_fingerprint": self.quantization_fingerprint,
            "train_key": self.train_key,
            "prediction_key": self.prediction_key,
            "metric_key": self.metric_key,
            "tolerance": self.tolerance,
        }


@dataclass
class SeriesAllocation:
    sequence_id: str
    requested_points: int
    quota_cap: int
    allocated_points: int
    used_points: int = 0
    quota_exhausted: bool = False
    budget_exhausted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "requested_points": self.requested_points,
            "quota_cap": self.quota_cap,
            "allocated_points": self.allocated_points,
            "used_points": self.used_points,
            "quota_exhausted": self.quota_exhausted,
            "budget_exhausted": self.budget_exhausted,
        }


@dataclass
class ScheduleResult:
    tasks: list[TaskPlan]
    failures: list[Failure]
    allocations: dict[str, SeriesAllocation]
    budget_used: int
    budget_remaining: int
    pending_cursor: dict[str, Any] | None

    @property
    def ready_tasks(self) -> list[TaskPlan]:
        return [task for task in self.tasks if task.status == "ready"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_used": self.budget_used,
            "budget_remaining": self.budget_remaining,
            "pending_cursor": self.pending_cursor,
            "allocations": [
                self.allocations[key].to_dict() for key in sorted(self.allocations)
            ],
            "ready_task_count": sum(1 for task in self.tasks if task.status == "ready"),
            "quota_failed_task_count": sum(
                1 for task in self.tasks if task.status == E_SERIES_QUOTA_EXHAUSTED
            ),
            "budget_failed_task_count": sum(
                1 for task in self.tasks if task.status == E_POINT_BUDGET_EXHAUSTED
            ),
        }


def make_task(
    *,
    config: Config,
    window: dict[str, Any],
    model_name: str,
) -> TaskPlan:
    series_fingerprint = str(window["series_fingerprint"])
    window_spec_fingerprint = fingerprint(
        {"schema": "window-spec-v1", **window["window_spec"]}
    )
    model_param_fingerprint = config.models.parameter_fingerprint(model_name)
    quantization_fingerprint = config.quantization.fingerprint()
    train_key = fingerprint(
        {
            "schema": "train-key-v1",
            "window_key": window["window_key"],
            "model_name": model_name,
            "model_parameter_fingerprint": model_param_fingerprint,
            "model_implementation_version": 1,
        }
    )
    prediction_key = fingerprint(
        {
            "schema": "prediction-key-v1",
            "train_key": train_key,
            "forecast_horizon": config.backtest.forecast_horizon,
            "quantization_fingerprint": quantization_fingerprint,
            "prediction_implementation_version": 1,
        }
    )
    metric_key = fingerprint(
        {
            "schema": "metric-key-v1",
            "prediction_key": prediction_key,
            "actual_fingerprint": window["actual_fingerprint"],
            "error_tolerance": config.backtest.error_tolerance,
            "metric_implementation_version": 1,
        }
    )
    return TaskPlan(
        series_id=window["series_id"],
        window_index=window["window_index"],
        model_name=model_name,
        window_key=window["window_key"],
        train_key=train_key,
        prediction_key=prediction_key,
        metric_key=metric_key,
        series_fingerprint=series_fingerprint,
        model_param_fingerprint=model_param_fingerprint,
        quantization_fingerprint=quantization_fingerprint,
        window_spec_fingerprint=window_spec_fingerprint,
        tolerance=config.backtest.error_tolerance,
        cost_points=config.backtest.window_length + config.backtest.forecast_horizon,
        window=window,
    )


def _quota_failure(task: TaskPlan, quota: int | None, used: int) -> Failure:
    return Failure(
        code=E_SERIES_QUOTA_EXHAUSTED,
        stage="schedule",
        message=(
            f"series point quota {quota} would be exceeded "
            f"(already used {used}, task costs {task.cost_points})"
        ),
        sequence_id=task.series_id,
        window_index=task.window_index,
        model_name=task.model_name,
        detail={"used_points": used, "task_cost": task.cost_points, "quota": quota},
    )


def _budget_failure(task: TaskPlan, allocated: int, used: int) -> Failure:
    return Failure(
        code=E_POINT_BUDGET_EXHAUSTED,
        stage="schedule",
        message=(
            f"batch point budget exhausted for series allocation {allocated} "
            f"(already used {used}, task costs {task.cost_points})"
        ),
        sequence_id=task.series_id,
        window_index=task.window_index,
        model_name=task.model_name,
        detail={"allocated_points": allocated, "used_points": used, "task_cost": task.cost_points},
    )


def schedule_tasks(tasks: Iterable[TaskPlan], config: Config) -> ScheduleResult:
    ordered = sorted(tasks, key=lambda task: task.sort_key)
    by_series: dict[str, list[TaskPlan]] = {}
    for task in ordered:
        by_series.setdefault(task.series_id, []).append(task)
    allocations: dict[str, SeriesAllocation] = {}
    remaining = config.scheduler.batch_point_budget
    for sequence_id in sorted(by_series):
        requested = sum(task.cost_points for task in by_series[sequence_id])
        quota_cap = requested
        if config.scheduler.per_series_point_quota is not None:
            quota_cap = min(quota_cap, config.scheduler.per_series_point_quota)
        allocated = min(quota_cap, remaining)
        remaining -= allocated
        allocations[sequence_id] = SeriesAllocation(
            sequence_id=sequence_id,
            requested_points=requested,
            quota_cap=quota_cap,
            allocated_points=allocated,
        )

    failures: list[Failure] = []
    pending_cursor: dict[str, Any] | None = None
    for task in ordered:
        allocation = allocations[task.series_id]
        if allocation.quota_exhausted:
            task.status = E_SERIES_QUOTA_EXHAUSTED
            failures.append(_quota_failure(task, config.scheduler.per_series_point_quota, allocation.used_points))
            continue
        if allocation.budget_exhausted:
            task.status = E_POINT_BUDGET_EXHAUSTED
            failures.append(_budget_failure(task, allocation.allocated_points, allocation.used_points))
            if pending_cursor is None:
                pending_cursor = {
                    "series_id": task.series_id,
                    "window_index": task.window_index,
                    "model_name": task.model_name,
                    "task_key": task.task_key,
                }
            continue
        projected = allocation.used_points + task.cost_points
        if projected > allocation.quota_cap:
            allocation.quota_exhausted = True
            task.status = E_SERIES_QUOTA_EXHAUSTED
            failures.append(_quota_failure(task, config.scheduler.per_series_point_quota, allocation.used_points))
            continue
        if projected > allocation.allocated_points:
            allocation.budget_exhausted = True
            task.status = E_POINT_BUDGET_EXHAUSTED
            failures.append(_budget_failure(task, allocation.allocated_points, allocation.used_points))
            if pending_cursor is None:
                pending_cursor = {
                    "series_id": task.series_id,
                    "window_index": task.window_index,
                    "model_name": task.model_name,
                    "task_key": task.task_key,
                }
            continue
        allocation.used_points = projected
        task.status = "ready"

    budget_used = sum(allocation.used_points for allocation in allocations.values())
    return ScheduleResult(
        tasks=ordered,
        failures=failures,
        allocations=allocations,
        budget_used=budget_used,
        budget_remaining=max(0, config.scheduler.batch_point_budget - budget_used),
        pending_cursor=pending_cursor,
    )


def execute_parallel(
    tasks: list[TaskPlan],
    worker: Callable[[TaskPlan], R],
    concurrency: int,
) -> list[R]:
    if not tasks:
        return []
    if concurrency <= 1 or len(tasks) == 1:
        return [worker(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=min(concurrency, len(tasks))) as executor:
        return list(executor.map(worker, tasks))


def chunk_tasks_by_windows(tasks: list[TaskPlan], window_interval: int) -> list[list[TaskPlan]]:
    grouped: dict[tuple[str, int], list[TaskPlan]] = {}
    for task in sorted(tasks, key=lambda item: item.sort_key):
        grouped.setdefault((task.series_id, task.window_index), []).append(task)
    window_keys = sorted(grouped)
    chunks: list[list[TaskPlan]] = []
    for start in range(0, len(window_keys), window_interval):
        chunk: list[TaskPlan] = []
        for key in window_keys[start : start + window_interval]:
            chunk.extend(grouped[key])
        chunks.append(chunk)
    return chunks


def invalidation_report(
    previous: dict[str, Any],
    tasks: list[TaskPlan],
) -> dict[str, Any]:
    invalidated: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda item: item.sort_key):
        current = task.fingerprint_record()
        prior = previous.get(task.task_key)
        if not isinstance(prior, dict):
            invalidated.append(
                {
                    "task_key": task.task_key,
                    "series_id": task.series_id,
                    "window_index": task.window_index,
                    "model_name": task.model_name,
                    "reason": "no_previous_snapshot",
                    "changed_layers": ["window", "train", "prediction", "metric"],
                }
            )
            continue
        changed_layers = []
        if prior.get("window_key") != current["window_key"]:
            changed_layers.append("window")
        if prior.get("train_key") != current["train_key"]:
            changed_layers.append("train")
        if prior.get("prediction_key") != current["prediction_key"]:
            changed_layers.append("prediction")
        if prior.get("metric_key") != current["metric_key"]:
            changed_layers.append("metric")
        if not changed_layers:
            unchanged.append(
                {
                    "task_key": task.task_key,
                    "series_id": task.series_id,
                    "window_index": task.window_index,
                    "model_name": task.model_name,
                }
            )
            continue
        if prior.get("series_fingerprint") != current["series_fingerprint"]:
            reason = "sequence_content_changed"
        elif prior.get("window_spec_fingerprint") != current["window_spec_fingerprint"]:
            reason = "window_parameters_changed"
        elif prior.get("model_parameter_fingerprint") != current["model_parameter_fingerprint"]:
            reason = "model_parameters_changed"
        elif prior.get("quantization_fingerprint") != current["quantization_fingerprint"]:
            reason = "quantization_changed"
        elif prior.get("tolerance") != current["tolerance"]:
            reason = "tolerance_changed"
        else:
            reason = "upstream_content_changed"
        invalidated.append(
            {
                "task_key": task.task_key,
                "series_id": task.series_id,
                "window_index": task.window_index,
                "model_name": task.model_name,
                "reason": reason,
                "changed_layers": changed_layers,
            }
        )
    return {
        "invalidated_tasks": invalidated,
        "unchanged_tasks": unchanged,
        "invalidated_count": len(invalidated),
        "unchanged_count": len(unchanged),
    }
