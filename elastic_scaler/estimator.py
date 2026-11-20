"""Pure-integer per-minute demand estimation and unified budget trimming.

Integer conventions (documented in every rule string):
* floor division ``a // b`` for per-replica throughput;
* ceiling division :func:`ceil_div` for replica demand (explicit upward
  rounding, no floating point is ever consulted);
* every clamp and trim is recorded with a stable reason code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import ceil_div
from .spec import ClusterSpec, ServiceSpec

MS_PER_MINUTE = 60_000

# Stable reason codes used by both the ledger and the GUI.
REASON_CLAMP_MIN = "CLAMP_MIN_REPLICAS"
REASON_CLAMP_MAX = "CLAMP_MAX_REPLICAS"
REASON_QUOTA = "TRIM_CONCURRENCY_QUOTA"
REASON_BUDGET = "TRIM_GLOBAL_BUDGET"
REASON_SATURATED = "SATURATED"


@dataclass(frozen=True)
class Estimate:
    service: str
    minute: int
    requests: int
    per_replica_capacity: int
    effective_capacity: int
    raw_demand: int
    min_demand: int
    quota_replica_cap: int
    quota_demand: int
    demand: int
    clamped_min: bool
    clamped_max: bool
    quota_limited: bool
    rules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "minute": self.minute,
            "requests": self.requests,
            "per_replica_capacity": self.per_replica_capacity,
            "effective_capacity": self.effective_capacity,
            "raw_demand": self.raw_demand,
            "min_demand": self.min_demand,
            "quota_replica_cap": self.quota_replica_cap,
            "quota_demand": self.quota_demand,
            "demand": self.demand,
            "clamped_min": self.clamped_min,
            "clamped_max": self.clamped_max,
            "quota_limited": self.quota_limited,
            "rules": list(self.rules),
        }


@dataclass(frozen=True)
class Allocation:
    service: str
    minute: int
    allocated_replicas: int
    servable_requests: int
    rejected_requests: int
    trim_reasons: tuple[str, ...]
    estimate: Estimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "minute": self.minute,
            "allocated_replicas": self.allocated_replicas,
            "servable_requests": self.servable_requests,
            "rejected_requests": self.rejected_requests,
            "trim_reasons": list(self.trim_reasons),
            "estimate": self.estimate.to_dict(),
        }


@dataclass(frozen=True)
class MinuteDecision:
    minute: int
    spec_digest: str
    spec_version: int
    budget: int
    budget_used: int
    budget_remaining: int
    total_rejected: int
    allocations: tuple[Allocation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "minute": self.minute,
            "spec_digest": self.spec_digest,
            "spec_version": self.spec_version,
            "budget": self.budget,
            "budget_used": self.budget_used,
            "budget_remaining": self.budget_remaining,
            "total_rejected": self.total_rejected,
            "allocations": [alloc.to_dict() for alloc in self.allocations],
        }


def estimate_service(svc: ServiceSpec, minute: int, requests: int) -> Estimate:
    """Derive the untrimmed replica demand for one service at one minute."""
    # Floor division: only whole requests completed inside a 60_000 ms window.
    capacity = (svc.concurrency_limit * MS_PER_MINUTE) // svc.request_ms
    # The scale threshold lowers the utilization target; floor keeps it integer.
    # max(1, ...) preserves a defined ceiling-division denominator even when a
    # single request cannot finish within one minute.
    effective_capacity = max(1, (capacity * svc.scale_threshold_pct) // 100)

    # Explicit ceiling division (round up): a partial replica's worth of load
    # still requires one more whole replica.
    raw_demand = ceil_div(requests, effective_capacity)

    rules = [
        "per_replica_capacity=floor(concurrency_limit*60000/request_ms)",
        "effective_capacity=max(1,floor(per_replica_capacity*scale_threshold_pct/100))",
        "raw_demand=ceil_div(requests,effective_capacity) [integer ceiling, never float]",
    ]

    clamped_min = False
    min_demand = raw_demand
    if raw_demand < svc.min_replicas:
        min_demand = svc.min_replicas
        clamped_min = True
        rules.append("demand=max(raw_demand,min_replicas) -> CLAMP_MIN_REPLICAS")

    quota_cap = svc.concurrency_quota // svc.concurrency_limit
    quota_demand = min(min_demand, quota_cap)
    quota_limited = min_demand > quota_cap
    if quota_limited:
        rules.append("quota_demand=min(min_demand,concurrency_quota//concurrency_limit) -> QUOTA_CAP")

    demand = min(quota_demand, svc.max_replicas)
    clamped_max = quota_demand > svc.max_replicas
    if clamped_max:
        rules.append("demand=min(quota_demand,max_replicas) -> CLAMP_MAX_REPLICAS")
    rules.append("output is deterministic for identical (spec, minute, requests)")

    return Estimate(
        service=svc.name,
        minute=minute,
        requests=requests,
        per_replica_capacity=capacity,
        effective_capacity=effective_capacity,
        raw_demand=raw_demand,
        min_demand=min_demand,
        quota_replica_cap=quota_cap,
        quota_demand=quota_demand,
        demand=demand,
        clamped_min=clamped_min,
        clamped_max=clamped_max,
        quota_limited=quota_limited,
        rules=tuple(rules),
    )


def decide_minute(spec: ClusterSpec, minute: int, spec_version: int) -> MinuteDecision:
    """Estimate every service, then trim and commit once under one budget."""
    estimates: list[Estimate] = []
    for svc in spec.services:  # services are stored in lexicographic order
        estimates.append(estimate_service(svc, minute, svc.requests[minute]))

    budget = spec.global_replica_budget
    allocations: dict[str, Allocation] = {}

    # Phase 1: minimum replicas are reserved first; spec validation guarantees
    # the reserve fits both the global budget and every concurrency quota.
    reserved = sum(svc.min_replicas for svc in spec.services)
    remaining = budget - reserved
    if remaining < 0:  # defensive; loader already rejects this as E_BUDGET
        raise RuntimeError("minimum replicas exceed global budget")

    # Phase 2: deterministic stable priority = lexicographic service name order.
    for est in estimates:
        svc = spec.service(est.service)
        allocated = svc.min_replicas
        wanted_extra = est.demand - svc.min_replicas
        grant = min(wanted_extra, remaining)
        allocated += grant
        remaining -= grant

        reasons: list[str] = []
        if est.clamped_max:
            reasons.append(REASON_CLAMP_MAX)
        if est.quota_limited and allocated >= est.quota_replica_cap:
            reasons.append(REASON_QUOTA)
        if grant < wanted_extra and remaining == 0:
            reasons.append(REASON_BUDGET)
        if (
            allocated >= svc.max_replicas
            and REASON_BUDGET not in reasons
            and REASON_QUOTA not in reasons
        ):
            reasons.append(REASON_SATURATED)

        servable = allocated * est.per_replica_capacity
        rejected = est.requests - servable
        if rejected < 0:
            rejected = 0
        if est.requests > 0 and rejected > 0 and not reasons:
            reasons.append(REASON_SATURATED)

        allocations[svc.name] = Allocation(
            service=svc.name,
            minute=minute,
            allocated_replicas=allocated,
            servable_requests=servable,
            rejected_requests=rejected,
            trim_reasons=tuple(reasons),
            estimate=est,
        )

    ordered = tuple(allocations[svc.name] for svc in spec.services)
    budget_used = sum(item.allocated_replicas for item in ordered)
    assert budget_used <= budget
    for item in ordered:
        svc = spec.service(item.service)
        assert item.allocated_replicas <= svc.max_replicas
        assert item.allocated_replicas <= svc.quota_replica_cap
        assert item.rejected_requests >= 0
        assert item.rejected_requests == max(
            0, item.estimate.requests - item.servable_requests
        )

    return MinuteDecision(
        minute=minute,
        spec_digest=spec.digest,
        spec_version=spec_version,
        budget=budget,
        budget_used=budget_used,
        budget_remaining=budget - budget_used,
        total_rejected=sum(item.rejected_requests for item in ordered),
        allocations=ordered,
    )
