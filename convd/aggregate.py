"""NumPy-based deterministic aggregation cross-check of the ledger."""

import numpy as np


def verify_aggregates(ledger):
    """Rebuild the convergence matrix and check ledger summary counts."""
    entries = ledger.round_entries()
    if not entries:
        return {"rounds": 0, "executors": 0, "ok": True}
    rounds = sorted({r for r, _, _ in entries})
    executors = sorted({e for _, e, _ in entries})
    matrix = np.zeros((len(rounds), len(executors)), dtype=np.int64)
    for round_no, executor_id, entry in entries:
        matrix[rounds.index(round_no),
               executors.index(executor_id)] = \
            1 if entry["conclusion"] == "converged" else 0
    summary = ledger.summary()
    ok = True
    details = {}
    for idx, executor_id in enumerate(executors):
        expected = int(matrix[:, idx].sum())
        actual = summary[executor_id]["converged_rounds"]
        details[executor_id] = {"expected": expected, "actual": actual}
        ok = ok and expected == actual
    return {"rounds": len(rounds), "executors": len(executors),
            "converged_total": int(matrix.sum()), "per_executor": details,
            "ok": ok}
