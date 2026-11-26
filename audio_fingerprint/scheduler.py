from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable

from .biz import BATCH_BUDGET_EXHAUSTED, CLIP_QUOTA_EXCEEDED
from .canonical import fingerprint


def plan_sort_key(plan: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(plan["track_id"]),
        int(plan["version"]),
        str(plan["clip_id"]),
        int(plan["frame_index"]),
        str(plan["product_id"]),
    )


def allocate_frame_budget(
    plans: list[dict[str, Any]],
    batch_sample_budget: int,
    clip_sample_quota: int,
) -> dict[str, Any]:
    ordered = sorted(plans, key=plan_sort_key)
    remaining = int(batch_sample_budget)
    used_by_clip: dict[str, int] = defaultdict(int)
    aborted_clips: set[str] = set()
    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    budget_stop = False
    budget_event_emitted = False

    for plan in ordered:
        clip_id = str(plan["clip_id"])
        cost = int(plan["points_cost"])
        if budget_stop:
            deferred.append(_deferred_plan(plan, BATCH_BUDGET_EXHAUSTED, "batch_sample_budget_exhausted"))
            continue
        if clip_id in aborted_clips:
            deferred.append(_deferred_plan(plan, CLIP_QUOTA_EXCEEDED, "clip_sample_quota_aborted_for_batch"))
            continue
        if cost > remaining:
            if not budget_event_emitted:
                events.append(
                    {
                        "code": BATCH_BUDGET_EXHAUSTED,
                        "target": "scheduler.batch_sample_budget",
                        "message": "remaining batch budget is insufficient for next stable unit",
                        "remaining_points": remaining,
                        "required_points": cost,
                        "next_track_id": plan["track_id"],
                        "next_frame_index": plan["frame_index"],
                    }
                )
                budget_event_emitted = True
            budget_stop = True
            deferred.append(_deferred_plan(plan, BATCH_BUDGET_EXHAUSTED, "batch_sample_budget_exhausted"))
            continue

        selected.append(plan)
        remaining -= cost
        used_by_clip[clip_id] += cost
        if used_by_clip[clip_id] > int(clip_sample_quota):
            events.append(
                {
                    "code": CLIP_QUOTA_EXCEEDED,
                    "target": f"clip:{clip_id}",
                    "message": "clip cumulative sample points exceeded quota; clip paused for this batch",
                    "track_id": plan["track_id"],
                    "version": int(plan["version"]),
                    "used_points": used_by_clip[clip_id],
                    "quota_points": int(clip_sample_quota),
                    "completed_through_frame": int(plan["frame_index"]),
                }
            )
            aborted_clips.add(clip_id)

    todos = [_todo_for_deferred(item) for item in deferred]
    return {
        "selected": selected,
        "deferred": deferred,
        "todos": todos,
        "events": events,
        "remaining_budget_points": remaining,
        "used_budget_points": int(batch_sample_budget) - remaining,
        "used_by_clip": dict(sorted(used_by_clip.items())),
    }


def _deferred_plan(plan: dict[str, Any], code: str, reason: str) -> dict[str, Any]:
    result = dict(plan)
    result["defer_code"] = code
    result["defer_reason"] = reason
    return result


def _todo_for_deferred(deferred: dict[str, Any]) -> dict[str, Any]:
    material = {
        "product_id": deferred["product_id"],
        "cache_key": deferred["cache_key"],
        "reason": deferred["defer_reason"],
    }
    return {
        "todo_id": "todo_sched_" + fingerprint(material)[:20],
        "kind": "frame_processing",
        "business_code": deferred["defer_code"],
        "status": "pending",
        "priority": 0,
        "track_id": deferred["track_id"],
        "version": int(deferred["version"]),
        "clip_id": deferred["clip_id"],
        "frame_index": int(deferred["frame_index"]),
        "product_id": deferred["product_id"],
        "cache_key": deferred["cache_key"],
        "reason": deferred["defer_reason"],
    }


def ordered_map(
    items: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    max_concurrency: int,
    serial: bool = False,
) -> list[dict[str, Any]]:
    if serial or max_concurrency <= 1 or len(items) <= 1:
        return [worker(item) for item in items]
    workers = min(int(max_concurrency), len(items))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="afp-deterministic") as executor:
        return list(executor.map(worker, items))
