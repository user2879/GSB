"""Integer fixed-point statistics with explicit truncation direction.

All rates/quantiles are expressed as integers scaled by the session scale
factor (``scale``). Truncation direction is always *toward zero*; the
quantization error upper bound for a truncated division is exactly one
scale unit (i.e. the true value v satisfies
``fixed/scale <= v < (fixed+1)/scale`` for non-negative values).
"""

TRUNCATION = "toward_zero"


def trunc_div(num, den):
    """Integer division truncated toward zero (works for negatives too)."""
    if den == 0:
        raise ZeroDivisionError("trunc_div by zero")
    q = abs(num) // abs(den)
    if (num < 0) != (den < 0):
        q = -q
    return q


def rate_fixed(success, total, scale):
    """Success rate in fixed point. Returns (value, err_bound) in scale units."""
    if total <= 0:
        return 0, 1
    return (success * scale) // total, 1  # err bound = 1 scale unit (truncation)


def quantile_fixed(sorted_values, q_num, q_den, scale):
    """Quantile of integer millisecond samples via truncated-rank method.

    Rank index = trunc(n * q_num / q_den), clamped to n-1. Inputs are integer
    milliseconds, so the scaled result is exact and the quantization error
    bound is 0 scale units.
    """
    n = len(sorted_values)
    if n == 0:
        return 0, 0
    idx = (q_num * n) // q_den
    if idx >= n:
        idx = n - 1
    return sorted_values[idx] * scale, 0


def summarize(latencies, successes, scale, min_samples):
    """Build an observation summary dict from raw samples.

    latencies: sorted-able iterable of integer ms (aligned with successes).
    successes: iterable of 0/1 aligned with latencies.
    """
    lat = sorted(int(x) for x in latencies)
    ok = sum(int(s) for s in successes)
    total = len(lat)
    rate, rate_bound = rate_fixed(ok, total, scale)
    p50, p50_bound = quantile_fixed(lat, 1, 2, scale)
    p95, p95_bound = quantile_fixed(lat, 95, 100, scale)
    return {
        "samples": total,
        "success": ok,
        "fail": total - ok,
        "rate_fixed": rate,
        "rate_err_bound": rate_bound,
        "p50_fixed": p50,
        "p95_fixed": p95,
        "quant_err_bound": max(p50_bound, p95_bound),
        "scale": scale,
        "truncation": TRUNCATION,
        "decidable": 1 if total >= min_samples else 0,
    }
