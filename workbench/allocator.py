"""Single-stage, stable quota/budget trimming and rejection accounting."""

from __future__ import annotations

from .models import AllocationResult, ClusterSpec, MinuteEstimate

TRIM_NONE = "none"
TRIM_MIN_FLOOR = "min_floor"
TRIM_CAPACITY_OVERFLOW = "capacity_overflow"
TRIM_QUOTA = "concurrency_quota"
TRIM_BUDGET = "global_budget"

_COMPARE_FIELDS = (
    "minute",
    "service_id",
    "demand",
    "effective_cap",
    "allocated",
    "trimmed",
    "trim_reason",
    "capacity_requests",
    "rejected",
)


def allocate_minute(
    spec: ClusterSpec,
    minute: int,
    estimates: tuple[MinuteEstimate, ...],
) -> tuple[AllocationResult, ...]:
    """Commit one minute's allocations atomically.

    Trimming rule (documented and explicit):
      1. Per-service effective cap = min(max_replicas, concurrency_quota).
      2. Services are visited in service-id order; ties on demand are broken
         by service id ascending (the primary key), giving a total order.
      3. Global budget is consumed greedily in that order, so under pressure
         services sorted first keep capacity and later services are trimmed.
    """

    by_id = {estimate.service_id: estimate for estimate in estimates}
    results: list[AllocationResult] = []
    remaining_budget = spec.global_replica_budget

    # Explicit stable ordering: demand descending, then service id ascending.
    ordered = sorted(estimates, key=lambda item: (-item.demand, item.service_id))

    for estimate in ordered:
        service = next(item for item in spec.services if item.service_id == estimate.service_id)
        effective_cap = min(service.max_replicas, service.concurrency_quota)
        capped_demand = min(estimate.demand, effective_cap)

        # Explicit reason priority: saturation > quota > min floor > none.
        # Only one reason is emitted so downstream comparisons are stable.
        if estimate.raw_demand > service.max_replicas:
            cap_reason = TRIM_CAPACITY_OVERFLOW
        elif estimate.demand > effective_cap:
            cap_reason = TRIM_QUOTA
        elif estimate.raw_demand < service.min_replicas and estimate.demand >= service.min_replicas:
            cap_reason = TRIM_MIN_FLOOR
        else:
            cap_reason = TRIM_NONE

        allocated = min(capped_demand, remaining_budget)
        remaining_budget -= allocated

        if allocated < capped_demand:
            trim_reason = TRIM_BUDGET
        else:
            trim_reason = cap_reason

        trimmed = estimate.demand - allocated
        capacity_requests = allocated * estimate.per_replica_capacity
        rejected = max(0, estimate.requests - capacity_requests)

        results.append(
            AllocationResult(
                minute=minute,
                service_id=estimate.service_id,
                demand=estimate.demand,
                effective_cap=effective_cap,
                allocated=allocated,
                trimmed=trimmed,
                trim_reason=trim_reason,
                capacity_requests=capacity_requests,
                rejected=rejected,
            )
        )

    # Re-emit in canonical service-id order for ledger and byte comparisons.
    return tuple(sorted(results, key=lambda item: item.service_id))


def assert_allocations_equal(
    left: tuple[AllocationResult, ...],
    right: tuple[AllocationResult, ...],
    minute: int,
) -> None:
    for left_item, right_item in zip(
        sorted(left, key=lambda item: item.service_id),
        sorted(right, key=lambda item: item.service_id),
    ):
        for field_name, left_value, right_value in zip(
            _COMPARE_FIELDS, left_item.as_tuple(), right_item.as_tuple()
        ):
            if left_value != right_value:
                from .errors import DivergenceError

                raise DivergenceError(
                    "allocation paths diverged",
                    details={
                        "minute": minute,
                        "service_id": left_item.service_id,
                        "field": field_name,
                        "full": left_value,
                        "incremental": right_value,
                    },
                )
