"""Fixed-point adapter merge, reversible split and the merge-budget scheduler."""

from __future__ import annotations

import numpy as np

from .errors import (
    E_IRREVERSIBLE_MERGE,
    E_MERGE_BUDGET_EXHAUSTED,
    E_MERGE_PENDING,
    failure,
)
from .fixedpoint import (
    MERGE_GUARD_BITS,
    Ledger,
    _safe_matmul,
    _safe_product,
    quantize,
    round_shift,
    saturate,
)


def scaled_delta_guard(adapter, frac_bits: int, ledger: Ledger, op_prefix: str = "merge"):
    """Adapter delta at frac_bits + MERGE_GUARD_BITS precision (int64)."""
    guard = frac_bits + MERGE_GUARD_BITS
    # B and A carry ``frac_bits`` fractional bits each; their exact product
    # carries 2*frac_bits.  Reduce to ``guard`` bits (shift by frac_bits - G).
    raw = _safe_matmul(adapter.B, adapter.A)
    delta = round_shift(raw, frac_bits - MERGE_GUARD_BITS, ledger, f"{op_prefix}_delta")
    scaling_q = quantize(np.float64(adapter.scaling), guard, ledger, f"{op_prefix}_scaling_q")
    return round_shift(_safe_product(delta, scaling_q), guard, ledger, f"{op_prefix}_scale")


def merge_layer(W_fixed, adapters, frac_bits: int, ledger: Ledger | None = None, reverse: bool = False):
    """Merge adapters onto a base weight in fixed task-name order.

    Each adapter's delta is computed at guard precision and rounded once to
    the tier (recorded in the ledger); the rounded deltas are then summed
    exactly in fixed task-name order.  The running sum is saturated to the
    storage band, which is the only order-sensitive (and irreversible) path.
    Returns (merged, steps, saturated).
    """
    ledger = ledger or Ledger()
    guard = MERGE_GUARD_BITS
    ordered = sorted(adapters, key=lambda a: a.task_name, reverse=reverse)
    acc = np.asarray(W_fixed, dtype=np.int64).copy()
    steps = []
    saturated = False
    for ad in ordered:
        delta_g = scaled_delta_guard(ad, frac_bits, ledger)
        delta_f = round_shift(delta_g, guard, ledger, "merge_round")
        new_acc = acc + delta_f
        new_acc, mask = saturate(new_acc, ledger, "merge_saturate")
        saturated = saturated or bool(np.any(mask))
        inc = new_acc - acc
        steps.append(
            {
                "task": ad.task_name,
                "layer": ad.target_layer,
                "increment_fixed": inc,
                "saturated": bool(np.any(mask)),
            }
        )
        acc = new_acc
    return acc, steps, saturated


def split_layer(merged, adapters, frac_bits: int):
    """Rebuild the base weight from the merged product and the exported
    adapters only: subtract each adapter's rounded delta in reverse
    task-name order."""
    ledger = Ledger()
    recon = np.asarray(merged, dtype=np.int64).copy()
    ordered = sorted(adapters, key=lambda a: a.task_name, reverse=True)
    for ad in ordered:
        delta_g = scaled_delta_guard(ad, frac_bits, ledger, "split")
        delta_f = round_shift(delta_g, MERGE_GUARD_BITS, ledger, "split_round")
        recon = recon - delta_f
    return recon


def check_reversible(W_fixed, merged, steps, adapters, frac_bits: int):
    """Returns list of E_IRREVERSIBLE_MERGE failures (empty when reversible).

    Restoration error above half a quantum of the configured tier is fatal.
    """
    recon = split_layer(merged, adapters, frac_bits)
    err_quanta = np.abs(recon - np.asarray(W_fixed, dtype=np.int64))
    if err_quanta.size == 0 or float(np.max(err_quanta)) <= 0.5:
        return []
    bad_layers = sorted({s["layer"] for s in steps})
    bad_tasks = sorted({s["task"] for s in steps})
    return [
        failure(
            E_IRREVERSIBLE_MERGE,
            f"restore error {float(np.max(err_quanta))} quanta > 0.5 on layers {bad_layers}",
            task=",".join(bad_tasks),
            layer=",".join(bad_layers),
        )
    ]


def merge_cost(task, layer_numel: int) -> int:
    """Estimated merge compute: adapter rank x target layer element count."""
    return int(task.rank) * int(layer_numel)


def schedule_merges(tasks, layer_numel, budget: int):
    """Deterministic budget scheduler.

    Per-task quota = budget // n_tasks.  A task whose cost exceeds its quota
    is aborted with E_MERGE_BUDGET_EXHAUSTED; when the global budget runs out
    the remaining tasks stay in a reproducible pending queue (E_MERGE_PENDING).
    Returns (approved, failures, pending, consumption).
    """
    ordered = sorted(tasks, key=lambda t: t.sort_key)
    quota = budget // len(ordered) if ordered else 0
    remaining = budget
    approved, failures, pending, consumption = [], [], [], []
    for task in ordered:
        cost = merge_cost(task, layer_numel(task.target_layer))
        if cost > quota:
            failures.append(
                failure(
                    E_MERGE_BUDGET_EXHAUSTED,
                    f"task cost {cost} exceeds per-task quota {quota}",
                    task=task.name,
                    layer=task.target_layer,
                )
            )
            consumption.append({"task": task.name, "cost": 0, "status": "exhausted"})
            continue
        if cost > remaining:
            pending.append(task)
            consumption.append({"task": task.name, "cost": 0, "status": "pending"})
            continue
        remaining -= cost
        approved.append(task)
        consumption.append({"task": task.name, "cost": cost, "status": "approved"})
    for task in pending:
        failures.append(
            failure(
                E_MERGE_PENDING,
                "global merge budget exhausted; task kept in pending queue",
                task=task.name,
                layer=task.target_layer,
            )
        )
    return approved, failures, pending, consumption
