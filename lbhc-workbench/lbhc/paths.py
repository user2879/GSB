"""Probe path algorithms and adaptive selection.

full       - every non-draining instance is probed once per round.
stratified - per pool, instances that have reached a stable verdict are
             ranked by (consecutive failures desc, weight desc, id); the
             low-priority tail is probed at reduced frequency (every
             REDOWN_EVERY rounds). While deprioritised an instance keeps
             its last verdict: it is never counted as success or failure,
             so it cannot silently come back to life.

Because deprioritisation only starts AFTER an instance has stabilised,
and stable instances produce no further transitions under deterministic
backends, both paths emit byte-identical verdict and transition sequences
for the same input.
"""

from __future__ import annotations

FULL = "full"
STRATIFIED = "stratified"
REDOWN_EVERY = 3  # deprioritised instances are re-probed every N rounds
TAIL_FRACTION = 0.5  # bottom half of stable instances may be deprioritised


def select_path(spec, forced="auto"):
    """Pick the probe path and explain the choice with scale parameters."""
    n_inst = spec.instance_count
    n_pools = spec.pool_count
    budget = spec.concurrency_budget
    quotas = sum(p.concurrency_quota for p in spec.pools)
    headroom = budget - min(quotas, budget)
    capacity_per_round = min(budget, quotas)
    params = {
        "instances": n_inst,
        "pools": n_pools,
        "global_budget": budget,
        "pool_quota_sum": quotas,
        "budget_headroom": headroom,
    }
    if forced in (FULL, STRATIFIED):
        params["reason"] = f"path forced by operator: {forced}"
        return forced, params
    if n_inst <= max(2 * capacity_per_round, 8):
        params["reason"] = (
            f"full path: {n_inst} instances fit comfortably within the "
            f"per-round capacity {capacity_per_round} "
            f"(headroom {headroom})")
        return FULL, params
    params["reason"] = (
        f"stratified path: {n_inst} instances across {n_pools} pools "
        f"exceed 2x per-round capacity {capacity_per_round}; probing "
        f"every instance each round would exhaust the budget")
    return STRATIFIED, params


def _active_backends(spec):
    for pool in spec.pools:
        for b in pool.backends:
            if b.role != "draining":
                yield pool, b


def due_full(spec, health_map, round_no):
    """Every non-draining instance, deterministic order."""
    return [(p, b) for p, b in _active_backends(spec)]


def due_stratified(spec, health_map, round_no):
    """Per-pool priority sampling with down-frequency for the stable tail."""
    due = []
    for pool in spec.pools:
        active = [b for b in pool.backends if b.role != "draining"]
        if round_no == 0:
            due.extend((pool, b) for b in active)
            continue
        unstable, stable = [], []
        for b in active:
            h = health_map[b.id]
            is_stable = (
                (h.state == "healthy"
                 and h.consec_success >= pool.success_threshold)
                or (h.state == "unhealthy"
                    and h.consec_failure >= pool.failure_threshold))
            (stable if is_stable else unstable).append(b)
        # priority: consecutive failures first, then weight, then id
        stable.sort(key=lambda b: (-health_map[b.id].consec_failure,
                                   -b.weight, b.id))
        keep = len(stable) - int(len(stable) * TAIL_FRACTION)
        priority, tail = stable[:keep], stable[keep:]
        due.extend((pool, b) for b in unstable)
        due.extend((pool, b) for b in priority)
        if round_no % REDOWN_EVERY == 0:
            due.extend((pool, b) for b in tail)
        # deprioritised tail instances are skipped this round: their last
        # verdict stands and they are NOT counted as success or failure.
    due.sort(key=lambda pb: (pb[0].id, pb[1].id))
    return due


def due_instances(path, spec, health_map, round_no):
    if path == FULL:
        return due_full(spec, health_map, round_no)
    return due_stratified(spec, health_map, round_no)
