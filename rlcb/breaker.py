"""Pure sliding-window circuit breaker and degradation policy.

This module is free of sockets, threads and wall-clock timing. It consumes a
per-service request sequence in canonical arrival order and emits one
immutable decision per request. Ratio arithmetic is integer fixed-point in
basis points, so the same request stream yields byte-identical decisions
regardless of batch splitting or dictionary traversal order.

State machine (streaming evaluation)
------------------------------------
* ``closed``: requests execute and accumulate in the current window. As soon
  as the running failure rate reaches the threshold (including threshold 0 on
  the first failure), the breaker trips, that request is degraded and the
  breaker becomes ``open``.
* ``open``: remaining requests of the tripping window and all requests until
  the window boundary are degraded without executing. At the next window
  boundary the breaker becomes ``half_open`` and the new window is a probe.
* ``half_open``: probe requests execute and accumulate. When the probe window
  closes, a rate below the threshold counts one clean recovery window; after
  ``recovery_windows`` consecutive clean windows the breaker closes, while a
  failing probe reopens it immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fixedpoint import ratio_bp
from .spec import ServiceSpec

STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

OUTCOME_EXECUTED = "executed"
OUTCOME_DEGRADED = "degraded"
OUTCOME_RATE_REJECTED = "rate_rejected"


@dataclass(frozen=True)
class Decision:
    request_id: str
    service_id: str
    batch_id: str
    arrival_seq: int
    config_version: int
    breaker_state_before: str
    outcome: str
    degrade_level: int
    window_index: int
    window_failure_count: int
    window_size: int
    failure_rate_bp: int
    threshold_bp: int
    threshold_comparison: str
    triggered_window: int | None
    recovery_passed: int
    failed: int
    basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_id": self.service_id,
            "batch_id": self.batch_id,
            "arrival_seq": self.arrival_seq,
            "config_version": self.config_version,
            "breaker_state_before": self.breaker_state_before,
            "outcome": self.outcome,
            "degrade_level": self.degrade_level,
            "window_index": self.window_index,
            "window_failure_count": self.window_failure_count,
            "window_size": self.window_size,
            "failure_rate_bp": self.failure_rate_bp,
            "threshold_bp": self.threshold_bp,
            "threshold_comparison": self.threshold_comparison,
            "triggered_window": self.triggered_window,
            "recovery_passed": self.recovery_passed,
            "failed": self.failed,
            "basis": self.basis,
        }


@dataclass
class _Window:
    index: int
    failures: int = 0
    size: int = 0


@dataclass
class BreakerState:
    state: str = STATE_CLOSED
    triggered_window: int | None = None
    recovery_passed: int = 0
    decisions: list[Decision] = field(default_factory=list)


def _compare(rate_bp: int, threshold_bp: int) -> str:
    if rate_bp >= threshold_bp:
        return f"{rate_bp}>={threshold_bp}"
    return f"{rate_bp}<{threshold_bp}"


def evaluate_service(
    spec: ServiceSpec,
    requests: list[Any],
    *,
    rate_rejected_ids: frozenset[str] | None = None,
) -> list[Decision]:
    rate_rejected_ids = rate_rejected_ids or frozenset()
    ordered = sorted(requests, key=lambda r: (r.arrival_seq, r.request_id))
    bs = BreakerState()
    window = _Window(index=0)
    last_window_index = -1
    final_index = len(ordered) - 1

    for position, req in enumerate(ordered):
        window_index = req.arrival_seq // spec.window_length
        if window_index != last_window_index:
            window = _Window(index=window_index)
            last_window_index = window_index
            if bs.state == STATE_OPEN:
                # Crossing into a new window ends open degradation and starts
                # a half-open recovery probe for this window.
                bs.state = STATE_HALF_OPEN
                bs.recovery_passed = 0

        state_before = bs.state
        is_window_end = (
            position == final_index
            or (position + 1 < len(ordered)
                and (ordered[position + 1].arrival_seq // spec.window_length
                     != window_index))
        )

        if req.request_id in rate_rejected_ids:
            bs.decisions.append(_decision(
                spec, req, state_before, OUTCOME_RATE_REJECTED, window,
                "not_evaluated", bs, "rps_limit_rejected_before_window"))
            continue

        if state_before == STATE_OPEN:
            bs.decisions.append(_decision(
                spec, req, STATE_OPEN, OUTCOME_DEGRADED, window,
                _compare(ratio_bp(window.failures, max(1, window.size)),
                         spec.failure_threshold_bp),
                bs, "open_degrade_request_not_executed"))
            continue

        # closed or half_open -> the request executes and enters the window.
        window.size += 1
        window.failures += req.failed
        rate_bp = ratio_bp(window.failures, window.size)
        comparison = _compare(rate_bp, spec.failure_threshold_bp)

        if state_before == STATE_HALF_OPEN:
            outcome, basis = self_half_open_step(
                spec, bs, window, rate_bp, comparison, is_window_end)
        else:
            outcome, basis = self_closed_step(
                spec, bs, window, rate_bp, comparison)

        bs.decisions.append(_decision(
            spec, req, state_before, outcome, window, comparison, bs, basis))
    return bs.decisions


def self_closed_step(spec, bs, window, rate_bp, comparison):
    if rate_bp >= spec.failure_threshold_bp:
        bs.state = STATE_OPEN
        bs.triggered_window = window.index
        bs.recovery_passed = 0
        return OUTCOME_DEGRADED, "streaming_threshold_exceeded_opened"
    return OUTCOME_EXECUTED, "window_below_threshold_running"


def self_half_open_step(spec, bs, window, rate_bp, comparison, is_window_end):
    # A failing rate inside the probe reopens immediately.
    if rate_bp >= spec.failure_threshold_bp:
        bs.state = STATE_OPEN
        bs.triggered_window = window.index
        bs.recovery_passed = 0
        return OUTCOME_DEGRADED, "probe_threshold_exceeded_reopened"
    if not is_window_end:
        return OUTCOME_EXECUTED, "half_open_probe_in_window"
    # Probe window closed below threshold: count one clean recovery window.
    bs.recovery_passed += 1
    if bs.recovery_passed >= spec.recovery_windows:
        bs.state = STATE_CLOSED
        bs.triggered_window = None
        bs.recovery_passed = 0
        return OUTCOME_EXECUTED, "recovery_window_passed_breaker_closed"
    return OUTCOME_EXECUTED, "recovery_window_passed_probe_continues"


def _decision(spec, req, state_before, outcome, window, comparison,
              bs: BreakerState, basis: str) -> Decision:
    return Decision(
        request_id=req.request_id,
        service_id=spec.service_id,
        batch_id=req.batch_id,
        arrival_seq=req.arrival_seq,
        config_version=spec.config_version,
        breaker_state_before=state_before,
        outcome=outcome,
        degrade_level=spec.degrade_level,
        window_index=window.index,
        window_failure_count=window.failures,
        window_size=window.size,
        failure_rate_bp=ratio_bp(window.failures, max(1, window.size)),
        threshold_bp=spec.failure_threshold_bp,
        threshold_comparison=comparison,
        triggered_window=bs.triggered_window,
        recovery_passed=bs.recovery_passed,
        failed=req.failed,
        basis=basis,
    )
