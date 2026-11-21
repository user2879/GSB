"""Budget-constrained concurrent scheduler (requirement 6).

Turns are scheduled through a thread pool whose max worker count comes
from the config and is never exceeded.  The per-batch fusion point budget
is consumed in stable ascending turn-id order; turns that do not fit are
kept as pending and flagged E_FUSE_BUDGET_PENDING.  Every task carries a
stable sort key derived from (turn_id, stage name), so the final turn
order, per-stage diff order, budget deduction sequence and failure list
are identical to a single-threaded run in turn-id order, regardless of
thread scheduling, concurrency level or budget exhaustion timing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from . import codes

POINTS_PER_TURN = 3  # one fusion point per evidence channel


def plan_budget(turn_ids, budget):
    """Consume the fusion point budget in ascending turn-id order."""
    in_batch = []
    pending = []
    deductions = []
    spent = 0
    for turn_id in sorted(turn_ids):
        if spent + POINTS_PER_TURN <= budget:
            spent += POINTS_PER_TURN
            in_batch.append(turn_id)
            deductions.append({"turn_id": turn_id, "cost": POINTS_PER_TURN, "spent": spent})
        else:
            pending.append(turn_id)
    return in_batch, pending, deductions


def run_tasks(tasks, max_workers):
    """Run tasks concurrently; return results sorted by stable sort key.

    tasks: list of (sort_key, callable).  max_workers is taken from the
    config and never exceeded.
    """
    if max_workers <= 1 or len(tasks) <= 1:
        results = [fn() for _, fn in sorted(tasks, key=lambda t: t[0])]
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [(key, pool.submit(fn)) for key, fn in tasks]
        results = [(key, future.result()) for key, future in futures]
    return [value for _, value in sorted(results, key=lambda item: item[0])]


def budget_report(budget, deductions, pending):
    return {
        "budget": budget,
        "cost_per_turn": POINTS_PER_TURN,
        "spent": deductions[-1]["spent"] if deductions else 0,
        "deductions": deductions,
        "pending": [{"turn_id": t, "code": codes.E_FUSE_BUDGET_PENDING} for t in pending],
    }
