"""Fixed-point integer statistics for latency observations.

Representation: values are integer milliseconds; aggregates are emitted as
integers scaled by 10**3 (milli-milliseconds). Truncation direction is
toward zero. Quantization error upper bound: the mean carries at most
1 scaled unit (0.001 ms) of truncation error; p95 and max are exact
multiples of 1000 (observations are integer ms, no quantization loss).
"""

STATS_SCALE = 1000
ERROR_BOUND_X1000 = 1
TRUNCATION = "toward_zero"


def aggregate(values_ms):
    """values_ms: iterable of non-negative ints. Returns fixed-point dict."""
    vals = sorted(int(v) for v in values_ms)
    n = len(vals)
    if n == 0:
        return {"count": 0, "mean_x1000": 0, "p95_x1000": 0,
                "max_x1000": 0, "scale": STATS_SCALE,
                "error_bound_x1000": ERROR_BOUND_X1000,
                "truncation": TRUNCATION}
    total = sum(vals)
    mean_x1000 = (total * STATS_SCALE) // n  # toward zero (vals >= 0)
    rank = (95 * n + 99) // 100  # nearest-rank ceil(0.95*n), 1-based
    p95 = vals[rank - 1]
    return {"count": n,
            "mean_x1000": mean_x1000,
            "p95_x1000": p95 * STATS_SCALE,
            "max_x1000": vals[-1] * STATS_SCALE,
            "scale": STATS_SCALE,
            "error_bound_x1000": ERROR_BOUND_X1000,
            "truncation": TRUNCATION}


def aggregate_by_scope(rows):
    """rows: sqlite rows with scope/elapsed_ms. Returns {scope: agg}."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scope"], []).append(row["elapsed_ms"])
    return {scope: aggregate(vals) for scope, vals in
            sorted(grouped.items())}
