"""Resource budgets and the deterministic degradation path.

Given a memory budget and a per-batch time budget the planner decides the
resident coefficient set (order, clustering) and the interpolation strategy.
When over budget it degrades in a fixed priority order:

1. lower the SH order (3 -> 2 -> 1),
2. switch to cluster representatives (block-mean coefficient field),
3. reduce the sample-direction count (halved, floored at the per-order
   minimum; projection time scales with directions).

Every step records the reason and the estimated error bound before/after.
The plan never drops a query request; if even the fully degraded
configuration exceeds a budget the planner raises E_BUDGET_DENIED.  The
decision sequence is a pure function of its inputs and therefore
reproducible.
"""
from . import sh
from .errors import BudgetDeniedError
from .sampling import min_directions
from .strategies import (CLUSTER_BLOCK, COEFF_BYTES, OPS_PER_MS,
                         ORDER_ERROR_FACTOR, est_error_bound)

MIN_ORDER = 1


def _coeff_memory(grid_n: int, order: int, clustered: bool) -> int:
    probes = (grid_n // CLUSTER_BLOCK) ** 3 if clustered else grid_n ** 3
    return probes * sh.num_coeffs(order) * 3 * COEFF_BYTES


def _projection_ops(grid_n: int, order: int, n_dirs: int) -> float:
    return float(grid_n ** 3) * n_dirs * sh.num_coeffs(order) * 3.0


def _query_ops(grid_n: int, order: int, clustered: bool,
               n_queries: int) -> float:
    taps = 8.0  # trilinear taps in both full and cluster-resident modes
    return taps * sh.num_coeffs(order) * 3.0 * n_queries


def _error_bound(grid_n: int, order: int, clustered: bool,
                 field_scale: float) -> float:
    strategy = "cluster" if clustered else "full"
    return est_error_bound(strategy, grid_n, order, field_scale)


def plan_budget(grid_n: int, order: int, n_directions: int, n_queries: int,
                field_scale: float, memory_budget_bytes: int,
                time_budget_ms: float) -> dict:
    cur_order, clustered, cur_dirs = int(order), False, int(n_directions)
    steps = []

    def state():
        return {"order": cur_order, "clustered": clustered,
                "n_directions": cur_dirs}

    def memory():
        return _coeff_memory(grid_n, cur_order, clustered)

    def time_ms():
        ops = (_projection_ops(grid_n, cur_order, cur_dirs)
               + _query_ops(grid_n, cur_order, clustered, n_queries))
        return ops / OPS_PER_MS

    def record(action, reason, before_err):
        steps.append({
            "action": action,
            "reason": reason,
            "state": state(),
            "error_bound_before": before_err,
            "error_bound_after": _error_bound(grid_n, cur_order, clustered,
                                              field_scale),
        })

    # --- memory budget -------------------------------------------------
    while memory() > memory_budget_bytes:
        before = _error_bound(grid_n, cur_order, clustered, field_scale)
        if cur_order > MIN_ORDER:
            cur_order -= 1
            record("reduce_order", "memory budget exceeded", before)
        elif not clustered:
            clustered = True
            record("cluster_representatives", "memory budget exceeded", before)
        elif cur_dirs > min_directions(cur_order):
            cur_dirs = max(min_directions(cur_order), cur_dirs // 2)
            record("reduce_directions", "memory budget exceeded", before)
        else:
            raise BudgetDeniedError(
                "memory budget cannot be met even fully degraded",
                memory_budget_bytes=memory_budget_bytes,
                required_bytes=memory(), **state())

    # --- time budget ---------------------------------------------------
    while time_ms() > time_budget_ms:
        before = _error_bound(grid_n, cur_order, clustered, field_scale)
        if cur_order > MIN_ORDER:
            cur_order -= 1
            record("reduce_order", "time budget exceeded", before)
        elif not clustered:
            clustered = True
            record("cluster_representatives", "time budget exceeded", before)
        elif cur_dirs > min_directions(cur_order):
            cur_dirs = max(min_directions(cur_order), cur_dirs // 2)
            record("reduce_directions", "time budget exceeded", before)
        else:
            raise BudgetDeniedError(
                "time budget cannot be met even fully degraded",
                time_budget_ms=time_budget_ms,
                required_ms=time_ms(), **state())

    return {
        "final": state(),
        "resident_memory_bytes": memory(),
        "est_batch_time_ms": time_ms(),
        "est_error_bound": _error_bound(grid_n, cur_order, clustered,
                                        field_scale),
        "degradation_steps": steps,
        "queries_planned": n_queries,
        "queries_dropped": 0,
        "budgets": {"memory_bytes": memory_budget_bytes,
                    "time_ms": time_budget_ms},
    }
