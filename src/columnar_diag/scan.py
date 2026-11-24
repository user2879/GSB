from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from .models import BudgetConfig, ScanTask, SegmentSpec, TaskResult
from .planner import (
    PlanBundle,
    merge_query_row_group,
    task_result_payload,
)
from .scheduler import ScheduleReport, run_scheduled
from .segment import SegmentImage
from .util import sha256_json


@dataclass(frozen=True)
class ScanReport:
    task_results: tuple[TaskResult, ...]
    schedule: ScheduleReport


def run_block_task(
    task: ScanTask,
    *,
    ast: dict[str, Any],
    image: SegmentImage,
    delay_ms: float = 0.0,
) -> dict[str, Any]:
    values = image.values(task.block_id)
    result = task_result_payload(
        task=task,
        ast=ast,
        values=values,
        block_digest=image.blocks[task.block_id].block_digest,
    )
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)
    return result


def execute_tasks(
    tasks: Iterable[ScanTask],
    *,
    bundle: PlanBundle,
    image: SegmentImage,
    budget: BudgetConfig,
    delay_ms: float = 0.0,
) -> ScanReport:
    task_list = tuple(tasks)
    schedule = run_scheduled(
        task_list,
        lambda task: run_block_task(
            task,
            ast=bundle.query_asts[task.query_id],
            image=image,
            delay_ms=delay_ms,
        ),
        budget,
    )
    results: list[TaskResult] = []
    for outcome in schedule.outcomes:
        payload = outcome.value
        values = image.values(outcome.task.block_id)
        stats = {
            "scanned_rows": len(values),
            "matched_rows": len(payload["matched_row_indices"]),
            "min_value": min(values) if values else None,
            "max_value": max(values) if values else None,
            "sum_value": sum(values),
        }
        results.append(
            TaskResult(
                task=outcome.task,
                completed_rows=len(values),
                decode_memory_bytes=outcome.task.decode_memory_bytes,
                queue_started_ms=outcome.queue_started_ms,
                wait_ended_ms=outcome.wait_ended_ms,
                run_started_ms=outcome.run_started_ms,
                run_ended_ms=outcome.run_ended_ms,
                duration_ms=outcome.duration_ms,
                block_digest=image.blocks[outcome.task.block_id].block_digest,
                result_digest=payload["result_digest"],
                result_json=payload,
                stats=stats,
            )
        )
    return ScanReport(task_results=tuple(results), schedule=schedule)


def row_group_values(
    spec: SegmentSpec, row_group_id: str
) -> dict[str, tuple[int, ...]]:
    return {
        block.column: block.values
        for block in spec.blocks
        if block.row_group == row_group_id
    }


def compute_query_results(
    spec: SegmentSpec,
    bundle: PlanBundle,
    task_results: Iterable[TaskResult | dict[str, Any]],
    *,
    query_ids: set[str] | None = None,
    row_group_ids: set[str] | None = None,
    decoded_row_values: dict[str, dict[str, tuple[int, ...]]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    by_task: dict[str, dict[str, Any]] = {}
    for item in task_results:
        if isinstance(item, TaskResult):
            by_task[item.task.task_id] = item.result_json
        else:
            by_task[str(item["task_id"])] = item
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for plan in bundle.plans:
        if query_ids is not None and plan.query_id not in query_ids:
            continue
        for detail in plan.row_groups:
            row_group_id = str(detail["row_group_id"])
            if row_group_ids is not None and row_group_id not in row_group_ids:
                continue
            candidate_blocks = list(detail["candidate_blocks"])
            payloads: list[dict[str, Any]] = []
            complete = True
            for block_id in candidate_blocks:
                task_id = f"q:{plan.query_id}|rg:{row_group_id}|block:{block_id}"
                payload = by_task.get(task_id)
                if payload is None:
                    complete = False
                    break
                payloads.append(payload)
            if not complete:
                continue
            values = (
                decoded_row_values[row_group_id]
                if decoded_row_values is not None
                else row_group_values(spec, row_group_id)
            )
            merged = merge_query_row_group(
                query_id=plan.query_id,
                row_group_id=row_group_id,
                select=plan.select,
                predicate_columns=plan.predicate_columns,
                task_payloads=payloads,
                row_values=values,
            )
            merged["plan_digest"] = plan.digest
            merged["select"] = list(plan.select)
            result[(plan.query_id, row_group_id)] = merged
    return result


def query_overall_digest(
    query_id: str, results: Iterable[dict[str, Any]]
) -> str:
    rows: list[list[int]] = []
    for result in sorted(results, key=lambda item: str(item["row_group_id"])):
        if result["query_id"] == query_id:
            rows.extend(result["rows"])
    return sha256_json({"query_id": query_id, "rows": rows})
