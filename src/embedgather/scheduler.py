"""Parallel gather scheduling with a global point budget.

Determinism contract: quota allocation and budget deduction happen in a
single-threaded pass ordered by request index before any worker runs,
and refunds happen in request order after all workers finish. Thread
interleaving therefore cannot change result order, contribution order
or the budget deduction sequence.
"""

from concurrent.futures import ThreadPoolExecutor

from .errors import E_POINT_BUDGET_EXHAUSTED
from .gather import execute_gather


def run_gather_stage(requests, table, config, injector=None,
                     only_indices=None):
    """Run gather requests through the thread pool.

    Returns (results, budget_report, pending_indices).
    """
    selected = [r for r in requests
                if only_indices is None or r.index in only_indices]
    selected.sort(key=lambda r: r.index)

    # Deterministic allocation pass in request-index order.
    remaining = config.global_point_budget
    quotas = {}
    deduction_log = []
    for req in selected:
        quota = min(config.per_request_quota, remaining)
        if injector is not None:
            quota = injector.quota_for(req.index, quota)
        quotas[req.index] = quota
        remaining -= quota
        deduction_log.append({
            "request_index": req.index,
            "reserved": quota,
            "sort_key": [req.index, 0],
        })

    # Parallel execution; results keyed by request index.
    results = {}
    if selected:
        with ThreadPoolExecutor(
                max_workers=config.max_concurrency) as pool:
            futures = {}
            for req in selected:
                extra = ()
                if injector is not None:
                    extra = injector.extra_ids_for(req.index)
                futures[req.index] = pool.submit(
                    execute_gather, req, table, config.agg_mode,
                    config.quant_level, config.frac_bits,
                    quotas[req.index], extra)
            for index, fut in futures.items():
                results[index] = fut.result()

    # Deterministic refund pass in request-index order.
    for req in selected:
        used = results[req.index].points_used
        refund = quotas[req.index] - used
        remaining += refund
        deduction_log.append({
            "request_index": req.index,
            "used": used,
            "refunded": refund,
            "sort_key": [req.index, 1],
        })

    pending = sorted(idx for idx, res in results.items()
                     if res.code == E_POINT_BUDGET_EXHAUSTED)
    budget_report = {
        "global_budget": config.global_point_budget,
        "consumed": config.global_point_budget - remaining,
        "remaining": remaining,
        "deduction_sequence": deduction_log,
    }
    return results, budget_report, pending
