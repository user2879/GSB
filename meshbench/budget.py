"""Fixed-point SLI folding and per-window error-budget receipts.

All ratios are represented as integers scaled by :data:`SCALE` (1,000,000),
i.e. micro-units of the true ratio.  Every scaled division uses explicit floor
truncation, so the absolute quantization error of one ratio is strictly below
``1 / SCALE``.  Receipts are pure functions of the four outcome counters and
the route parameters; arrival order cannot change the result.
"""

from dataclasses import dataclass

from .canonical import canonical_dumps, content_digest
from .config import RouteConfig

SCALE = 1_000_000
TRUNCATION = "floor"
QUANT_ERROR_NUMERATOR = 1
QUANT_ERROR_DENOMINATOR = SCALE

# The four mutually exclusive terminal outcomes.
SUCCESS = "success"   # completed below the slow threshold
SLOW = "slow"         # completed successfully, but real latency >= threshold
FAILURE = "failure"   # upstream error / connection error / circuit fast-fail
TIMEOUT = "timeout"   # real socket timeout exhausted the attempt budget
OUTCOMES = (SUCCESS, FAILURE, TIMEOUT, SLOW)
TERMINAL_OUTCOMES = set(OUTCOMES)

WINDOW_MS_PER_MINUTE = 60_000


def window_span_ms(route: RouteConfig) -> int:
    return route.window_minutes * WINDOW_MS_PER_MINUTE


def window_index_for(now_ms: int, route: RouteConfig) -> int:
    return int(now_ms) // window_span_ms(route)


def window_bounds(route: RouteConfig, window_index: int):
    span = window_span_ms(route)
    start = window_index * span
    return start, start + span


def _scaled_ratio(numerator: int, total: int) -> int:
    """Floor-truncated fixed-point ratio; error < 1/SCALE of the true ratio."""
    if total == 0:
        return 0
    return (numerator * SCALE) // total


@dataclass(frozen=True)
class WindowCounters:
    success: int = 0
    failure: int = 0
    timeout: int = 0
    slow: int = 0

    @property
    def total(self):
        return self.success + self.failure + self.timeout + self.slow

    @property
    def bad(self):
        # Everything that is not a clean, fast success consumes the budget.
        return self.failure + self.timeout + self.slow

    def add(self, outcome):
        return WindowCounters(
            self.success + (outcome == SUCCESS),
            self.failure + (outcome == FAILURE),
            self.timeout + (outcome == TIMEOUT),
            self.slow + (outcome == SLOW),
        )

    def as_dict(self):
        return {
            "success": self.success,
            "failure": self.failure,
            "timeout": self.timeout,
            "slow": self.slow,
            "total": self.total,
        }


def compute_receipt(route: RouteConfig, rule_version: int, window_index: int,
                    counters: WindowCounters):
    """Deterministically fold counters into a canonical error-budget receipt."""
    start_ms, end_ms = window_bounds(route, window_index)
    total = counters.total
    good = counters.success + counters.slow
    availability_sli = _scaled_ratio(good, total)
    latency_sli = _scaled_ratio(counters.success, total)
    bad_rate = _scaled_ratio(counters.bad, total)

    # Allowed bad events expressed on the same fixed-point scale (floor).
    allowed_bad_scaled = (total * route.budget_permyriad * SCALE) // 10_000
    bad_scaled = counters.bad * SCALE
    remaining_scaled = allowed_bad_scaled - bad_scaled

    insufficient = total < route.min_samples
    budget_exhausted = (not insufficient) and (remaining_scaled < 0)

    window_id = f"{route.route_id}:v{rule_version}:w{window_index}"
    payload = {
        "schema": "meshbench.receipt/v1",
        "window_id": window_id,
        "route_id": route.route_id,
        "target_service": route.target_service,
        "rule_version": rule_version,
        "window_index": window_index,
        "window_start_ms": start_ms,
        "window_end_ms": end_ms,
        "window_minutes": route.window_minutes,
        "budget_permyriad": route.budget_permyriad,
        "min_samples": route.min_samples,
        "slow_threshold_ms": route.slow_threshold_ms,
        "counters": counters.as_dict(),
        "scale": SCALE,
        "truncation": TRUNCATION,
        "quant_error_max": {
            "numerator": QUANT_ERROR_NUMERATOR,
            "denominator": QUANT_ERROR_DENOMINATOR,
        },
        "availability_sli_scaled": availability_sli,
        "latency_sli_scaled": latency_sli,
        "bad_rate_scaled": bad_rate,
        "allowed_bad_scaled": allowed_bad_scaled,
        "bad_scaled": bad_scaled,
        "remaining_scaled": remaining_scaled,
        "insufficient_sample": insufficient,
        "budget_exhausted": budget_exhausted,
    }
    canonical = canonical_dumps(payload)
    return Receipt(
        window_id=window_id,
        route_id=route.route_id,
        rule_version=rule_version,
        window_index=window_index,
        payload=payload,
        canonical_bytes=canonical,
        digest=content_digest(payload),
    )


@dataclass(frozen=True)
class Receipt:
    window_id: str
    route_id: str
    rule_version: int
    window_index: int
    payload: dict
    canonical_bytes: bytes
    digest: str
