"""Stable filename/column scheduling under a per-batch keypoint budget."""

from __future__ import annotations

from typing import Any

from .config import EffectiveConfig
from .models import Dataset, FailureLog


def build_schedule(dataset: Dataset, effective: EffectiveConfig, failures: FailureLog) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    remaining = effective.point_budget
    for entry in sorted(dataset.images, key=lambda item: item.name):
        truth = dataset.image_truths.get(entry.name)
        if truth is None:
            continue
        for column in sorted(truth.columns, key=lambda item: (item.order, -item.center_x, item.id)):
            requested = column.char_count
            if remaining <= 0:
                allocated = 0
                status = "pending"
            elif remaining >= requested:
                allocated = requested
                status = "complete"
            else:
                allocated = remaining
                status = "partial"
            stable_key = f"{entry.name}#{column.order:04d}#{column.id}"
            task = {
                "sort_key": [entry.name, column.order, column.id],
                "stable_key": stable_key,
                "image": entry.name,
                "column": column.id,
                "column_order": column.order,
                "requested_points": requested,
                "allocated_points": allocated,
                "status": status,
            }
            entries.append(task)
            if status != "complete":
                pending_points = requested - allocated
                pending_entry = {
                    **task,
                    "pending_points": pending_points,
                    "code": "E_POINT_BUDGET_PENDING",
                }
                pending.append(pending_entry)
                failures.add(
                    "E_POINT_BUDGET_PENDING",
                    "schedule",
                    f"{entry.name}:{column.id}",
                    "column exceeds the remaining per-batch keypoint budget",
                    image=entry.name,
                    column=column.id,
                    requested=requested,
                    allocated=allocated,
                    pending=pending_points,
                    status=status,
                )
            remaining -= allocated

    runnable = [entry for entry in entries if entry["allocated_points"] > 0]
    runnable.sort(key=lambda item: tuple(item["sort_key"]))
    pending.sort(key=lambda item: tuple(item["sort_key"]))
    return {
        "budget": effective.point_budget,
        "remaining_after_plan": remaining,
        "allocated_total": sum(entry["allocated_points"] for entry in entries),
        "tasks": runnable,
        "all_entries": entries,
        "pending": pending,
        "max_workers": effective.max_workers,
    }


def stable_task_key(task: dict[str, Any]) -> tuple[str, int, str]:
    return (task["image"], int(task["column_order"]), task["column"])
