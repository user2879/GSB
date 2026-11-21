"""Per-instance health state machine.

States: unknown -> healthy / unhealthy, plus draining (declared role).
Consecutive success/failure counters reset each other; a probe outcome
only ever increments its own counter. Deprioritised (down-frequency)
instances are simply not probed, so they can never be counted.
"""

from __future__ import annotations

from dataclasses import dataclass

STATES = ("unknown", "healthy", "unhealthy", "draining")

# Reason categories recorded on transitions.
REASON_THRESHOLD = "threshold_met"      # 阈值达成
REASON_TIMEOUT = "timeout"              # 超时
REASON_CONN_REFUSED = "conn_refused"    # 连接拒绝
REASON_DIGEST = "digest_mismatch"       # 摘要不符
FAILURE_REASONS = (REASON_TIMEOUT, REASON_CONN_REFUSED, REASON_DIGEST)


@dataclass
class Health:
    state: str = "unknown"
    consec_success: int = 0
    consec_failure: int = 0
    last_reason: str = ""

    def snapshot(self):
        return (self.state, self.consec_success, self.consec_failure,
                self.last_reason)


def initial_health(role):
    if role == "draining":
        return Health(state="draining", last_reason="declared_draining")
    return Health()


def apply_outcome(health, outcome, reason, success_threshold,
                  failure_threshold):
    """Apply one probe outcome. Returns (from, to, reason) or None.

    Only committed probe results may call this; skipped (deprioritised)
    instances never reach this function, hence are never counted.
    """
    if health.state == "draining":
        return None
    if outcome == "success":
        health.consec_success += 1
        health.consec_failure = 0
        if (health.state in ("unknown", "unhealthy")
                and health.consec_success >= success_threshold):
            prev = health.state
            health.state = "healthy"
            health.last_reason = REASON_THRESHOLD
            return (prev, "healthy", REASON_THRESHOLD)
    else:
        health.consec_failure += 1
        health.consec_success = 0
        if (health.state in ("unknown", "healthy")
                and health.consec_failure >= failure_threshold):
            prev = health.state
            health.state = "unhealthy"
            health.last_reason = reason
            return (prev, "unhealthy", reason)
    return None
