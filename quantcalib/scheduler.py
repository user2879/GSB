"""Deterministic three-budget scheduler independent of concurrency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import E_BATCH_BUDGET_PENDING, E_QUOTA_EXCEEDED
from .model import Config, Layer
from .utils import chunk_ranges


@dataclass(frozen=True)
class ScheduledGroup:
    layer_name: str
    group_index: int
    start: int
    end: int

    @property
    def num_points(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PendingGroup:
    layer_name: str
    group_index: int
    start: int
    end: int
    code: str
    reason: str

    @property
    def num_points(self) -> int:
        return self.end - self.start


@dataclass
class BudgetPlan:
    scheduled: list[ScheduledGroup] = field(default_factory=list)
    pending: list[PendingGroup] = field(default_factory=list)
    consumption_sequence: list[dict[str, Any]] = field(default_factory=list)
    per_layer_consumed: dict[str, int] = field(default_factory=dict)
    global_consumed: int = 0
    layer_aborted: dict[str, str] = field(default_factory=dict)
    batch_events: list[dict[str, Any]] = field(default_factory=list)
    execution_batches: list[list[ScheduledGroup]] = field(default_factory=list)

    def by_layer(self) -> dict[str, list[ScheduledGroup]]:
        result: dict[str, list[ScheduledGroup]] = {}
        for group in self.scheduled:
            result.setdefault(group.layer_name, []).append(group)
        return result

    def pending_by_layer(self) -> dict[str, list[PendingGroup]]:
        result: dict[str, list[PendingGroup]] = {}
        for group in self.pending:
            result.setdefault(group.layer_name, []).append(group)
        return result


def plan_budget(layers: Iterable[Layer], config: Config) -> BudgetPlan:
    plan = BudgetPlan()
    scheduling = config.scheduling
    global_used = 0
    global_exhausted = False
    ordered_layers = sorted(layers, key=lambda item: item.name)

    for layer in ordered_layers:
        cfg = config.for_layer(layer)
        layer_used = 0
        plan.per_layer_consumed[layer.name] = 0
        group_count = layer.numel // cfg.group_granularity
        aborted = False
        for group_index in range(group_count):
            start = group_index * cfg.group_granularity
            end = start + cfg.group_granularity
            points = end - start
            if global_exhausted:
                plan.pending.append(
                    PendingGroup(
                        layer.name,
                        group_index,
                        start,
                        end,
                        E_BATCH_BUDGET_PENDING,
                        "global calibration point budget is exhausted",
                    )
                )
                continue
            if aborted:
                plan.pending.append(
                    PendingGroup(
                        layer.name,
                        group_index,
                        start,
                        end,
                        E_QUOTA_EXCEEDED,
                        "layer already aborted after exceeding per-layer calibration quota",
                    )
                )
                continue
            if layer_used + points > scheduling.per_layer_calibration_quota:
                aborted = True
                plan.layer_aborted[layer.name] = E_QUOTA_EXCEEDED
                plan.pending.append(
                    PendingGroup(
                        layer.name,
                        group_index,
                        start,
                        end,
                        E_QUOTA_EXCEEDED,
                        f"per-layer quota {scheduling.per_layer_calibration_quota} would be exceeded; layer stopped",
                    )
                )
                continue
            if global_used + points > scheduling.global_calibration_point_budget:
                global_exhausted = True
                plan.pending.append(
                    PendingGroup(
                        layer.name,
                        group_index,
                        start,
                        end,
                        E_BATCH_BUDGET_PENDING,
                        "global calibration point budget insufficient for this group; remaining work is pending",
                    )
                )
                continue
            scheduled = ScheduledGroup(layer.name, group_index, start, end)
            plan.scheduled.append(scheduled)
            layer_used += points
            global_used += points
            plan.per_layer_consumed[layer.name] = layer_used
            plan.consumption_sequence.append(
                {
                    "sequence": len(plan.consumption_sequence),
                    "layer": layer.name,
                    "group_index": group_index,
                    "start": start,
                    "end": end,
                    "points": points,
                    "layer_consumed_after": layer_used,
                    "global_consumed_after": global_used,
                }
            )
    plan.global_consumed = global_used
    plan.batch_events = _pack_batch_events(plan.scheduled, scheduling.batch_quantization_point_budget)
    plan.execution_batches = _pack_execution_batches(
        plan.scheduled, scheduling.batch_quantization_point_budget
    )
    return plan


def _pack_batch_events(
    scheduled: list[ScheduledGroup], limit: int
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    current_chunks: list[dict[str, Any]] = []
    current_points = 0
    for group in scheduled:
        for start, end in chunk_ranges(group.start, group.end, limit):
            points = end - start
            if current_chunks and current_points + points > limit:
                batches.append(
                    {
                        "batch_index": len(batches),
                        "points": current_points,
                        "chunks": current_chunks,
                    }
                )
                current_chunks = []
                current_points = 0
            current_chunks.append(
                {
                    "layer": group.layer_name,
                    "group_index": group.group_index,
                    "start": start,
                    "end": end,
                    "points": points,
                }
            )
            current_points += points
    if current_chunks:
        batches.append(
            {
                "batch_index": len(batches),
                "points": current_points,
                "chunks": current_chunks,
            }
        )
    return batches


def _pack_execution_batches(
    scheduled: list[ScheduledGroup], limit: int
) -> list[list[ScheduledGroup]]:
    batches: list[list[ScheduledGroup]] = []
    current: list[ScheduledGroup] = []
    current_points = 0
    for group in scheduled:
        if current and current_points + group.num_points > limit:
            batches.append(current)
            current = []
            current_points = 0
        current.append(group)
        current_points += group.num_points
        if group.num_points >= limit:
            batches.append(current)
            current = []
            current_points = 0
    if current:
        batches.append(current)
    return batches


def budget_report(plan: BudgetPlan, config: Config) -> dict[str, Any]:
    return {
        "limits": {
            "thread_pool_workers": config.scheduling.thread_pool_workers,
            "global_calibration_point_budget": config.scheduling.global_calibration_point_budget,
            "per_layer_calibration_quota": config.scheduling.per_layer_calibration_quota,
            "batch_quantization_point_budget": config.scheduling.batch_quantization_point_budget,
            "snapshot_commit_interval_batches": config.scheduling.snapshot_commit_interval_batches,
        },
        "global_consumed": plan.global_consumed,
        "global_remaining": max(
            0,
            config.scheduling.global_calibration_point_budget - plan.global_consumed,
        ),
        "per_layer_consumed": dict(sorted(plan.per_layer_consumed.items())),
        "layer_aborted": dict(sorted(plan.layer_aborted.items())),
        "consumption_sequence": plan.consumption_sequence,
        "batch_count": len(plan.batch_events),
        "batch_events": plan.batch_events,
        "pending_items": [
            {
                "layer": item.layer_name,
                "group_index": item.group_index,
                "start": item.start,
                "end": item.end,
                "points": item.num_points,
                "code": item.code,
                "reason": item.reason,
            }
            for item in plan.pending
        ],
        "determinism": "layers sorted by name; groups by integer index; budgets reserved before worker dispatch",
    }
