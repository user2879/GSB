"""Deterministic budget allocation and bounded parallel channel execution."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import FusionConfig
from .diagnostics import failure, sort_failures


@dataclass
class BudgetAllocation:
    requests: dict[str, list[str]]
    granted: dict[str, list[str]]
    pending_queue: list[str] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    consumed: int = 0
    remaining: int = 0

    def report(self, config: FusionConfig, max_observed: int = 0) -> dict[str, Any]:
        return {
            "audit": self.audit,
            "channel_quota": config.channel_quota,
            "concurrency": {
                "configured_max_workers": config.max_workers,
                "max_observed": max_observed,
            },
            "consumed": self.consumed,
            "global_budget": config.global_budget,
            "max_workers": config.max_workers,
            "pending_queue": self.pending_queue,
            "remaining": self.remaining,
        }


def allocate_budget(requests: dict[str, list[str]], config: FusionConfig) -> BudgetAllocation:
    """Reserve all work in channel-name order before any worker starts.

    One merge-budget unit is one channel-candidate contribution that must be
    recomputed. Because allocation happens serially in sorted channel order,
    the audit trail is independent of worker completion timing.
    """

    remaining = int(config.global_budget)
    consumed = 0
    granted: dict[str, list[str]] = {}
    pending_queue: list[str] = []
    audit: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seq = 0

    for channel in sorted(requests):
        ids = sorted(set(requests[channel]))
        requested = len(ids)
        if requested == 0:
            continue
        before = remaining
        quota_cap = min(requested, int(config.channel_quota))
        grant_count = min(quota_cap, before)
        granted_ids = ids[:grant_count]
        denied_ids = ids[grant_count:]
        granted[channel] = granted_ids
        remaining -= grant_count
        consumed += grant_count

        quota_limited = requested > int(config.channel_quota)
        global_limited = grant_count < requested and grant_count == before
        status = "ok"
        if quota_limited and global_limited:
            status = "channel_quota_and_global_budget_exhausted"
        elif quota_limited:
            status = "channel_quota_exhausted"
        elif global_limited:
            status = "global_budget_exhausted"

        audit.append(
            {
                "channel": channel,
                "granted": grant_count,
                "remaining_global_after": remaining,
                "requested": requested,
                "seq": seq,
                "status": status,
            }
        )
        seq += 1

        if quota_limited:
            failures.append(
                failure(
                    "fuse",
                    "E_CHANNEL_QUOTA_EXHAUSTED",
                    (
                        f"channel {channel!r} requested {requested} merge operations, "
                        f"exceeding quota {config.channel_quota}; {len(denied_ids)} left pending"
                    ),
                    channel=channel,
                )
            )
        if global_limited:
            failures.append(
                failure(
                    "fuse",
                    "E_GLOBAL_BUDGET_EXHAUSTED",
                    (
                        f"global budget exhausted while allocating channel {channel!r}; "
                        f"granted {grant_count}/{requested}"
                    ),
                    channel=channel,
                )
            )
        pending_queue.extend(f"{channel}/{candidate_id}" for candidate_id in denied_ids)

    return BudgetAllocation(
        requests={channel: sorted(set(ids)) for channel, ids in requests.items()},
        granted=granted,
        pending_queue=pending_queue,
        audit=audit,
        failures=sort_failures(failures),
        consumed=consumed,
        remaining=remaining,
    )


def execute_channel_tasks(
    granted: dict[str, list[str]],
    max_workers: int,
    worker: Callable[[str, list[str]], Any],
) -> tuple[dict[str, Any], int]:
    """Execute channel tasks in a thread pool, returning deterministic results."""

    tasks = [(channel, sorted(ids)) for channel, ids in sorted(granted.items()) if ids]
    if not tasks:
        return {}, 0
    results: dict[str, Any] = {}
    lock = threading.Lock()
    active = 0
    max_observed = 0
    first_wave_size = min(len(tasks), int(max_workers))
    start_barrier = threading.Barrier(first_wave_size) if first_wave_size > 1 else None
    first_wave_remaining = first_wave_size

    def wrapped(channel: str, ids: list[str]) -> Any:
        nonlocal active, max_observed, first_wave_remaining
        with lock:
            active += 1
            max_observed = max(max_observed, active)
            participate_in_barrier = first_wave_remaining > 0
            if participate_in_barrier:
                first_wave_remaining -= 1
        try:
            if participate_in_barrier and start_barrier is not None:
                start_barrier.wait(timeout=5.0)
            return worker(channel, ids)
        finally:
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=int(max_workers), thread_name_prefix="fusion-channel") as executor:
        futures = {
            channel: executor.submit(wrapped, channel, ids)
            for channel, ids in tasks
        }
        for channel, _ in tasks:
            results[channel] = futures[channel].result()
    return results, max_observed
