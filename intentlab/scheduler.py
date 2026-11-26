"""Deterministic bounded-concurrency scheduling."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, TypeVar

from .errors import Codes


@dataclass(frozen=True)
class RecomputeTask:
    query_id: str
    source: str
    cost: int
    requires_match: bool
    requires_score: bool
    pending: bool = False
    priority: int = 0
    session_id: str = ""
    turn: int = 0

    def order_key(self) -> tuple[object, ...]:
        return (
            0 if self.pending else 1,
            self.priority,
            self.session_id,
            self.turn,
            self.query_id,
        )


@dataclass(frozen=True)
class SchedulePlan:
    selected: tuple[RecomputeTask, ...]
    deferred: tuple[dict[str, str], ...]
    reserved_points: int
    source_usage: Mapping[str, int]


@dataclass(frozen=True)
class TaskOutcome:
    query_id: str
    ok: bool
    value: object | None = None
    error_code: str = ""
    error_message: str = ""
    order_key: tuple[object, ...] = ()

    def to_failure(self) -> dict[str, object] | None:
        if self.ok:
            return None
        return {
            "query_id": self.query_id,
            "code": self.error_code,
            "message": self.error_message,
            "order_key": list(self.order_key),
        }


def select_tasks(
    tasks: Iterable[RecomputeTask],
    *,
    match_budget_points: int,
    source_quotas: Mapping[str, int],
    batch_query_limit: int,
    default_source_quota: int | None = None,
) -> SchedulePlan:
    selected: list[RecomputeTask] = []
    deferred: list[dict[str, str]] = []
    remaining = max(0, match_budget_points)
    usage: dict[str, int] = {}
    ordered = sorted(tasks, key=lambda task: task.order_key())
    batch_full = False
    for task in ordered:
        if batch_full:
            deferred.append({"query_id": task.query_id, "code": Codes.PENDING_BATCH_LIMIT})
            continue
        if len(selected) >= batch_query_limit:
            batch_full = True
            deferred.append({"query_id": task.query_id, "code": Codes.PENDING_BATCH_LIMIT})
            continue
        quota = source_quotas.get(task.source)
        if quota is None:
            quota = default_source_quota if default_source_quota is not None else batch_query_limit
        if usage.get(task.source, 0) >= quota:
            deferred.append({"query_id": task.query_id, "code": Codes.PENDING_SOURCE_QUOTA})
            continue
        reserved = task.cost if task.requires_match else 0
        if reserved > remaining:
            deferred.append({"query_id": task.query_id, "code": Codes.PENDING_BUDGET})
            continue
        selected.append(task)
        usage[task.source] = usage.get(task.source, 0) + 1
        remaining -= reserved
    return SchedulePlan(
        selected=tuple(selected),
        deferred=tuple(deferred),
        reserved_points=match_budget_points - remaining,
        source_usage=dict(sorted(usage.items())),
    )


T = TypeVar("T")


def execute_tasks(
    tasks: Iterable[RecomputeTask],
    worker: Callable[[RecomputeTask], T],
    *,
    workers: int,
) -> tuple[dict[str, T], list[dict[str, object]]]:
    """Execute concurrently while publishing outcomes in query-id order."""

    ordered_tasks = sorted(tasks, key=lambda task: (task.session_id, task.turn, task.query_id))
    outcomes: dict[str, TaskOutcome] = {}
    if workers <= 1:
        for task in ordered_tasks:
            try:
                value = worker(task)
                outcomes[task.query_id] = TaskOutcome(
                    query_id=task.query_id,
                    ok=True,
                    value=value,
                    order_key=task.order_key(),
                )
            except Exception as exc:  # business failure list remains deterministic
                outcomes[task.query_id] = TaskOutcome(
                    query_id=task.query_id,
                    ok=False,
                    error_code=Codes.TASK_FAILED,
                    error_message=f"{type(exc).__name__}: {exc}",
                    order_key=task.order_key(),
                )
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="intentlab") as executor:
            future_map: dict[Future[T], RecomputeTask] = {
                executor.submit(worker, task): task for task in ordered_tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                try:
                    value = future.result()
                    outcomes[task.query_id] = TaskOutcome(
                        query_id=task.query_id,
                        ok=True,
                        value=value,
                        order_key=task.order_key(),
                    )
                except Exception as exc:
                    outcomes[task.query_id] = TaskOutcome(
                        query_id=task.query_id,
                        ok=False,
                        error_code=Codes.TASK_FAILED,
                        error_message=f"{type(exc).__name__}: {exc}",
                        order_key=task.order_key(),
                    )

    values: dict[str, T] = {}
    failures: list[dict[str, object]] = []
    for query_id in sorted(outcomes):
        outcome = outcomes[query_id]
        if outcome.ok:
            values[query_id] = outcome.value  # type: ignore[assignment]
        else:
            failure = outcome.to_failure()
            if failure is not None:
                failures.append(failure)
    failures.sort(key=lambda item: (str(item["code"]), str(item["query_id"])))
    return values, failures


def pending_entries(
    deferred: Iterable[Mapping[str, str]],
    *,
    tasks: Iterable[RecomputeTask],
    failures: Iterable[Mapping[str, object]] = (),
) -> tuple[dict[str, str], ...]:
    task_map = {task.query_id: task for task in tasks}
    result: dict[str, dict[str, str]] = {}
    for item in deferred:
        query_id = str(item["query_id"])
        task = task_map[query_id]
        result[query_id] = {
            "query_id": query_id,
            "code": str(item["code"]),
            "source": task.source,
            "cost": str(task.cost),
        }
    for item in failures:
        query_id = str(item["query_id"])
        task = task_map.get(query_id)
        result.setdefault(
            query_id,
            {
                "query_id": query_id,
                "code": str(item["code"]),
                "source": task.source if task else "",
                "cost": str(task.cost) if task else "0",
            },
        )
    return tuple(result[query_id] for query_id in sorted(result))
