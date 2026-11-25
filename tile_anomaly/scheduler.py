"""Stable thread-scheduled window execution under quota and budget constraints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

from .errors import E_BUDGET_PENDING, E_TILE_QUOTA_EXCEEDED, BusinessEvent
from .windows import WindowSpec


Item = TypeVar("Item")


@dataclass
class Allocation:
    selected: list[tuple[str, WindowSpec]]
    pending: list[dict[str, Any]]
    events: list[BusinessEvent] = field(default_factory=list)
    quota_used: dict[str, int] = field(default_factory=dict)
    aborted: bool = False
    consumed_increment: int = 0


def stable_grouped_windows(
    owned_windows: dict[str, list[WindowSpec]],
    tile_order: list[str],
    needed_ids: set[str],
) -> list[tuple[str, list[WindowSpec]]]:
    grouped = []
    for tile_id in tile_order:
        selected = [
            window
            for window in owned_windows.get(tile_id, [])
            if window.window_id in needed_ids
        ]
        if selected:
            grouped.append((tile_id, selected))
    return grouped


def allocate_windows(
    grouped: list[tuple[str, list[WindowSpec]]],
    *,
    stage: str,
    budget_limit: int,
    consumed_budget: int,
    tile_quota: int,
    quota_used: dict[str, int] | None = None,
) -> Allocation:
    remaining_budget = max(0, budget_limit - consumed_budget)
    used = dict(quota_used or {})
    selected: list[tuple[str, WindowSpec]] = []
    pending: list[dict[str, Any]] = []
    events: list[BusinessEvent] = []
    aborted = False

    for group_index, (tile_id, windows) in enumerate(grouped):
        if aborted:
            for window in windows:
                pending.append(
                    {
                        "tile_id": tile_id,
                        "window_id": window.window_id,
                        "stage": stage,
                        "code": E_TILE_QUOTA_EXCEEDED,
                    }
                )
            continue
        already_used = used.get(tile_id, 0)
        if already_used + len(windows) > tile_quota:
            aborted = True
            events.append(
                BusinessEvent(
                    tile_id=tile_id,
                    stage="scheduler",
                    code=E_TILE_QUOTA_EXCEEDED,
                    message="tile window quota would be exceeded; batch aborted",
                    detail={
                        "operation_stage": stage,
                        "tile_window_quota": tile_quota,
                        "already_used": already_used,
                        "requested": len(windows),
                    },
                )
            )
            for window in windows:
                pending.append(
                    {
                        "tile_id": tile_id,
                        "window_id": window.window_id,
                        "stage": stage,
                        "code": E_TILE_QUOTA_EXCEEDED,
                    }
                )
            continue
        if remaining_budget <= 0:
            aborted = True
            events.append(
                BusinessEvent(
                    tile_id=tile_id,
                    stage="scheduler",
                    code=E_BUDGET_PENDING,
                    message="global window budget exhausted; work remains pending",
                    detail={
                        "operation_stage": stage,
                        "budget_limit": budget_limit,
                        "consumed_budget": consumed_budget,
                    },
                )
            )
            for window in windows:
                pending.append(
                    {
                        "tile_id": tile_id,
                        "window_id": window.window_id,
                        "stage": stage,
                        "code": E_BUDGET_PENDING,
                    }
                )
            continue
        allowed = min(len(windows), remaining_budget, tile_quota - already_used)
        for window in windows[:allowed]:
            selected.append((tile_id, window))
        used[tile_id] = already_used + allowed
        remaining_budget -= allowed
        if allowed < len(windows):
            aborted = True
            for window in windows[allowed:]:
                pending.append(
                    {
                        "tile_id": tile_id,
                        "window_id": window.window_id,
                        "stage": stage,
                        "code": E_BUDGET_PENDING,
                    }
                )
            events.append(
                BusinessEvent(
                    tile_id=tile_id,
                    stage="scheduler",
                    code=E_BUDGET_PENDING,
                    message="global window budget exhausted during tile allocation",
                    detail={
                        "operation_stage": stage,
                        "budget_limit": budget_limit,
                        "consumed_budget": consumed_budget,
                        "scheduled": allowed,
                        "remaining_from_tile": len(windows) - allowed,
                    },
                )
            )
            for later_tile_id, later_windows in grouped[group_index + 1 :]:
                for window in later_windows:
                    pending.append(
                        {
                            "tile_id": later_tile_id,
                            "window_id": window.window_id,
                            "stage": stage,
                            "code": E_BUDGET_PENDING,
                        }
                    )
            break

    return Allocation(
        selected=selected,
        pending=pending,
        events=events,
        quota_used=used,
        aborted=aborted,
        consumed_increment=len(selected),
    )


def run_threaded(
    items: Iterable[Item],
    worker: Callable[[Item], Any],
    max_workers: int,
) -> list[Any]:
    from concurrent.futures import ThreadPoolExecutor

    item_list = list(items)
    if max_workers <= 1 or len(item_list) <= 1:
        return [worker(item) for item in item_list]
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tile-window") as pool:
        futures = [pool.submit(worker, item) for item in item_list]
        return [future.result() for future in futures]

