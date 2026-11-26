"""Stable sample scheduling under point budget and per-sample quota."""
from __future__ import annotations

from typing import Any

from .codes import E_BUDGET_PENDING, E_POINT_QUOTA_EXCEEDED


def point_cost(sample: dict[str, Any], dimension: int) -> int:
    return int(dimension) * len(sample.get("key_positions", []))


def plan_batch(
    invalid_sample_ids: list[str],
    samples_by_id: dict[str, dict[str, Any]],
    dimension: int,
    point_budget: int,
    sample_quota: int,
    previous_pending: list[str] | None = None,
    cumulative_consumed: int = 0,
) -> dict[str, Any]:
    previous_pending = list(previous_pending or [])
    known_previous = [sid for sid in previous_pending if sid in samples_by_id]
    fresh = sorted(sid for sid in invalid_sample_ids if sid not in set(known_previous))
    candidate_ids = list(dict.fromkeys(known_previous + fresh))

    remaining = int(point_budget)
    admitted: list[dict[str, Any]] = []
    quota_exceeded: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    pending: list[str] = []
    stopped_for_budget = False

    for sample_id in candidate_ids:
        sample = samples_by_id.get(sample_id)
        if sample is None:
            continue
        cost = point_cost(sample, dimension)
        if cost > int(sample_quota):
            item = {"sample_id": sample_id, "cost": cost, "sample_quota": int(sample_quota)}
            quota_exceeded.append(item)
            events.append(
                {
                    "code": E_POINT_QUOTA_EXCEEDED,
                    "sample_id": sample_id,
                    "cost": cost,
                    "sample_quota": int(sample_quota),
                }
            )
            continue

        if cost > remaining:
            pending_start = candidate_ids.index(sample_id)
            pending = [sid for sid in candidate_ids[pending_start:] if sid in samples_by_id]
            stopped_for_budget = True
            events.append(
                {
                    "code": E_BUDGET_PENDING,
                    "sample_id": sample_id,
                    "cost": cost,
                    "remaining_budget": remaining,
                    "point_budget": int(point_budget),
                    "pending_count": len(pending),
                }
            )
            break

        admitted.append({"sample_id": sample_id, "cost": cost})
        remaining -= cost

    if not stopped_for_budget:
        pending = []

    batch_consumed = int(point_budget) - remaining
    return {
        "candidate_order": candidate_ids,
        "admitted": admitted,
        "admitted_sample_ids": [item["sample_id"] for item in admitted],
        "quota_exceeded": quota_exceeded,
        "quota_exceeded_sample_ids": [item["sample_id"] for item in quota_exceeded],
        "pending": pending,
        "remaining_budget": remaining,
        "batch_consumed": batch_consumed,
        "cumulative_consumed": int(cumulative_consumed) + batch_consumed,
        "point_budget": int(point_budget),
        "sample_quota": int(sample_quota),
        "events": events,
    }
