"""Deterministic triple-constraint scheduling.

The plan phase is fully serial and fixes budget deduction order, abort lists,
pending lists and retained fusion lists; the execution phase may run in a
thread pool without affecting any recorded ordering.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import errors


@dataclass
class Plan:
    classify_order: list = field(default_factory=list)   # tile ids, stable order
    quota_aborted: list = field(default_factory=list)    # E_TILE_BUDGET_EXHAUSTED
    global_pending: list = field(default_factory=list)   # E_GLOBAL_BUDGET_EXHAUSTED
    fuse_order: list = field(default_factory=list)       # pair keys, stable order
    fuse_pending: list = field(default_factory=list)     # E_MERGE_BATCH_PENDING
    budget_trace: list = field(default_factory=list)


def make_plan(dirty_tiles, tile_pixels: int, dirty_pairs,
              global_budget: int, per_tile_quota: int, merge_batch_limit: int) -> Plan:
    plan = Plan()
    remaining = global_budget
    seq = 0
    for tid in dirty_tiles:  # already sorted by (row, col, id)
        if tile_pixels > per_tile_quota:
            plan.quota_aborted.append(tid)
            plan.budget_trace.append({
                "seq": seq, "tile_id": tid, "needed": tile_pixels,
                "quota": per_tile_quota, "consumed": 0,
                "remaining_global": remaining, "code": errors.E_TILE_BUDGET_EXHAUSTED})
            seq += 1
            continue
        if tile_pixels > remaining:
            plan.global_pending.append(tid)
            plan.budget_trace.append({
                "seq": seq, "tile_id": tid, "needed": tile_pixels,
                "quota": 0, "consumed": 0,
                "remaining_global": remaining, "code": errors.E_GLOBAL_BUDGET_EXHAUSTED})
            seq += 1
            continue
        quota = min(per_tile_quota, remaining)
        remaining -= tile_pixels
        plan.classify_order.append(tid)
        plan.budget_trace.append({
            "seq": seq, "tile_id": tid, "needed": tile_pixels,
            "quota": quota, "consumed": tile_pixels,
            "remaining_global": remaining, "code": None})
        seq += 1
    for idx, pkey in enumerate(dirty_pairs):  # already sorted by pair key
        if idx < merge_batch_limit:
            plan.fuse_order.append(pkey)
        else:
            plan.fuse_pending.append(pkey)
    return plan


def execute_parallel(items, fn, workers: int, stop_check=None) -> dict:
    """Run fn(item) for each item; results keyed by item, order-independent.

    ``stop_check`` is consulted between submissions; when it returns True the
    remaining items are left unexecuted and reported via the ``__stopped__``
    sentinel key.
    """
    results = {}
    stopped = False
    if workers <= 1:
        for item in items:
            if stop_check is not None and stop_check():
                stopped = True
                break
            results[item] = fn(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {}
            for item in items:
                if stop_check is not None and stop_check():
                    stopped = True
                    break
                futures[pool.submit(fn, item)] = item
            for fut, item in futures.items():
                results[item] = fut.result()
    if stopped:
        results["__stopped__"] = True
    return results
