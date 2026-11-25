from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .algorithms import PathResult
from .canonical import canonical_bytes
from .spec import Spec


@dataclass(frozen=True)
class MinuteAllocation:
    minute: int
    allocation_order: int
    service_id: str
    clamped_replicas: int
    service_concurrency_quota: int
    service_replica_quota: int
    quota_capped_replicas: int
    global_replica_budget: int
    global_remaining_before: int
    global_remaining_after: int
    service_quota_remaining_after: int
    concurrency_quota_remaining_after: int
    allocated_replicas: int
    shortfall_replicas: int
    lease_count: int
    lease_ids: tuple[str, ...]

    def comparable_dict(self) -> dict[str, Any]:
        return {
            "allocated_replicas": self.allocated_replicas,
            "allocation_order": self.allocation_order,
            "clamped_replicas": self.clamped_replicas,
            "final_replica_allocation": self.allocated_replicas,
            "global_remaining_after": self.global_remaining_after,
            "global_remaining_before": self.global_remaining_before,
            "lease_count": self.lease_count,
            "lease_ids": list(self.lease_ids),
            "lease_occupancy": {
                "lease_count": self.lease_count,
                "lease_ids": list(self.lease_ids),
                "replica_units": self.lease_count,
            },
            "minute": self.minute,
            "quota_capped_replicas": self.quota_capped_replicas,
            "service_concurrency_quota": self.service_concurrency_quota,
            "service_replica_quota": self.service_replica_quota,
            "service_id": self.service_id,
            "service_quota_remaining_after": self.service_quota_remaining_after,
            "concurrency_quota_remaining_after": self.concurrency_quota_remaining_after,
            "shortfall_replicas": self.shortfall_replicas,
        }


def allocate_minute(
    spec: Spec,
    minute: int,
    estimates_by_service: dict[str, Any],
) -> list[MinuteAllocation]:
    budget = spec.budget.global_replica_budget
    remaining = budget
    minute_items: list[MinuteAllocation] = []
    for order, service in enumerate(spec.services):
        estimate = estimates_by_service[service.service_id]
        concurrency_quota = spec.budget.service_concurrency_quotas[service.service_id]
        replica_quota = concurrency_quota // service.concurrency_per_replica
        quota_capped = min(estimate.clamped_replicas, replica_quota)
        before = remaining
        allocated = min(quota_capped, remaining)
        remaining -= allocated
        lease_ids = tuple(
            f"m{minute:05d}:{service.service_id}:r{index:02d}"
            for index in range(allocated)
        )
        minute_items.append(
            MinuteAllocation(
                minute=minute,
                allocation_order=order,
                service_id=service.service_id,
                clamped_replicas=estimate.clamped_replicas,
                service_concurrency_quota=concurrency_quota,
                service_replica_quota=replica_quota,
                quota_capped_replicas=quota_capped,
                global_replica_budget=budget,
                global_remaining_before=before,
                global_remaining_after=remaining,
                service_quota_remaining_after=replica_quota - allocated,
                concurrency_quota_remaining_after=(
                    concurrency_quota
                    - allocated * service.concurrency_per_replica
                ),
                allocated_replicas=allocated,
                shortfall_replicas=estimate.clamped_replicas - allocated,
                lease_count=allocated,
                lease_ids=lease_ids,
            )
        )
    return minute_items


def allocate_path(spec: Spec, path_result: PathResult) -> dict[int, list[MinuteAllocation]]:
    allocations: dict[int, list[MinuteAllocation]] = {}
    for minute in range(spec.minute_count):
        estimates = {
            service_id: path_result.minutes[service_id][minute]
            for service_id in spec.service_ids
        }
        allocations[minute] = allocate_minute(spec, minute, estimates)
    return allocations


def conclusion_document(
    spec: Spec,
    path_result: PathResult,
    allocations: dict[int, list[MinuteAllocation]] | None = None,
) -> dict[str, Any]:
    allocations = allocations or allocate_path(spec, path_result)
    minutes = [
        item.comparable_dict()
        for service_id in spec.service_ids
        for item in path_result.minutes[service_id]
    ]
    allocation_rows = [
        allocation.comparable_dict()
        for minute in sorted(allocations)
        for allocation in allocations[minute]
    ]
    return {
        "allocation_rule": "sorted_service_ids_then_global_budget_and_service_quota",
        "allocations": allocation_rows,
        "minutes": minutes,
        "schema_version": 1,
    }


def conclusion_bytes(
    spec: Spec,
    path_result: PathResult,
    allocations: dict[int, list[MinuteAllocation]] | None = None,
) -> bytes:
    return canonical_bytes(conclusion_document(spec, path_result, allocations))
