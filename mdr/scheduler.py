from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable


def run_ordered(
    worker: Callable[[Any], Any],
    items: list[Any],
    max_workers: int,
    parallel: bool = True,
) -> list[Any]:
    if not items:
        return []
    if not parallel or max_workers <= 1:
        return [worker(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, items))


def point_stable_key(point: dict[str, Any], sequence_by_view: dict[str, int]) -> tuple[Any, ...]:
    left, right = point["pair"]
    left_obs = point["observations"][left]
    right_obs = point["observations"][right]
    return (
        min(sequence_by_view.get(left, 10**9), sequence_by_view.get(right, 10**9)),
        max(sequence_by_view.get(left, 10**9), sequence_by_view.get(right, 10**9)),
        int(left_obs[0]),
        int(left_obs[1]),
        int(right_obs[0]),
        int(right_obs[1]),
        point["point_id"],
    )


def allocate_point_budget(
    points: list[dict[str, Any]],
    sequence_by_view: dict[str, int],
    config: dict[str, Any],
    committed_ids: set[str] | None = None,
    previous_pending_ids: list[str] | None = None,
    previous_usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    budgets = config["budgets"]
    global_limit = int(budgets["global_point_budget"])
    view_limit = int(budgets["view_point_quota"])
    committed = committed_ids or set()
    previous_usage = previous_usage or {}
    global_used = int(previous_usage.get("global", 0))
    view_used = {view: int(count) for view, count in previous_usage.get("views", {}).items()}

    uncommitted = [point for point in points if point["point_id"] not in committed]
    stable = sorted(uncommitted, key=lambda p: point_stable_key(p, sequence_by_view))
    priority = set(previous_pending_ids or [])
    stable.sort(key=lambda p: (p["point_id"] not in priority, point_stable_key(p, sequence_by_view)))

    allowed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    stopped = False
    for point in stable:
        if stopped:
            pending.append(_pending_point(point, sequence_by_view, "budget allocation stopped after exhaustion"))
            continue
        views = point["pair"]
        next_global = global_used + 1
        next_views = {view: view_used.get(view, 0) + 1 for view in views}
        if next_global > global_limit or any(next_views[view] > view_limit for view in views):
            stopped = True
            pending.append(_pending_point(point, sequence_by_view, "global_point_budget or view_point_quota exhausted"))
            continue
        global_used = next_global
        for view in views:
            view_used[view] = next_views[view]
        allowed.append(point)

    allowed_ids = {point["point_id"] for point in allowed} | committed
    return {
        "allowed_ids": sorted(allowed_ids),
        "new_allowed_ids": [point["point_id"] for point in allowed],
        "pending": pending,
        "usage": {"global": global_used, "views": dict(sorted(view_used.items()))},
        "limit": {"global_point_budget": global_limit, "view_point_quota": view_limit},
    }


def _pending_point(point: dict[str, Any], sequences: dict[str, int], reason: str) -> dict[str, Any]:
    return {
        "unit_type": "triangulation_point",
        "unit_id": point["point_id"],
        "stable_key": "|".join(str(value) for value in point_stable_key(point, sequences)),
        "reason": reason,
        "code": "E_BUDGET_PENDING",
    }
