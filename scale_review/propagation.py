from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .algorithms import (
    PathResult,
    append_incremental_minutes,
    compute_full_minute,
    run_full_path,
    run_incremental_path,
    select_path,
)
from .allocation import (
    MinuteAllocation,
    allocate_minute,
    allocate_path,
    conclusion_bytes,
)
from .canonical import canonical_bytes, sha256_hex
from .consistency import verify_minute_range, verify_paths
from .errors import SpecError
from .estimator import estimate_minute
from .spec import BudgetConfig, Spec, validate_spec_dict


class VersionedPlan:
    def __init__(self, spec: Spec, *, verify: bool = True) -> None:
        self.spec = spec
        self.version = 1
        self.full_result = run_full_path(spec)
        self.incremental_result = run_incremental_path(spec)
        if verify:
            full_allocations, _ = verify_paths(
                spec, self.full_result, self.incremental_result
            )
        else:
            full_allocations = allocate_path(spec, self.full_result)
        self.allocations = full_allocations
        self.demand_bytes: dict[tuple[str, int], bytes] = {
            (service_id, item.minute): canonical_bytes(item.comparable_dict())
            for service_id in spec.service_ids
            for item in self.full_result.minutes[service_id]
        }
        self.allocation_rows: dict[tuple[int, str], MinuteAllocation] = {
            (minute, row.service_id): row
            for minute, rows in full_allocations.items()
            for row in rows
        }
        self.allocation_bytes: dict[tuple[int, str], bytes] = {
            key: canonical_bytes(row.comparable_dict())
            for key, row in self.allocation_rows.items()
        }
        self.selection = select_path(spec)
        self.versions: list[dict[str, Any]] = [
            {
                "version": 1,
                "status": "current",
                "change_type": "initial",
                "digest": self.digest(),
                "selection": self.selection.ledger_basis(),
            }
        ]
        self.propagation_log: list[dict[str, Any]] = []

    def digest(self) -> str:
        return sha256_hex(self.conclusion_bytes())

    def allocation_map(self) -> dict[int, list[MinuteAllocation]]:
        result: dict[int, list[MinuteAllocation]] = {}
        for minute in range(self.spec.minute_count):
            result[minute] = [
                self.allocation_rows[(minute, service_id)]
                for service_id in self.spec.service_ids
            ]
        return result

    def conclusion_bytes(self) -> bytes:
        return conclusion_bytes(self.spec, self.full_result, self.allocation_map())

    def current_display_state(self) -> dict[str, Any]:
        return {
            "plan_version": self.version,
            "spec": self.spec.canonical_dict(),
            "selection": self.selection.ledger_basis(),
            "conclusion": conclusion_bytes(
                self.spec, self.full_result, self.allocation_map()
            ).decode("utf-8"),
            "digest": self.digest(),
            "versions": self.versions,
        }

    def _replace_spec(self, spec: Spec) -> Spec:
        return validate_spec_dict(spec.canonical_dict())

    def _bump_version(
        self,
        change_type: str,
        propagation: dict[str, Any],
        new_spec: Spec,
        new_full: PathResult,
        new_incremental: PathResult,
    ) -> dict[str, Any]:
        self.spec = new_spec
        self.full_result = new_full
        self.incremental_result = new_incremental
        self.selection = select_path(new_spec)
        self.version += 1
        for record in self.versions:
            record["status"] = "stale"
        propagation["new_version"] = self.version
        propagation["new_digest"] = self.digest()
        self.propagation_log.append(propagation)
        self.versions.append(
            {
                "version": self.version,
                "status": "current",
                "change_type": change_type,
                "digest": self.digest(),
                "selection": self.selection.ledger_basis(),
                "propagation": propagation,
            }
        )
        return propagation

    def append_minutes(self, new_requests: Mapping[str, Sequence[int]]) -> dict[str, Any]:
        old_length = self.spec.minute_count
        if set(new_requests) != set(self.spec.service_ids):
            raise SpecError(
                "append batch must contain exactly every declared service",
                field="append_requests",
                details={
                    "expected": self.spec.service_ids,
                    "received": sorted(new_requests),
                },
            )
        append_length: int | None = None
        for service_id in self.spec.service_ids:
            values = new_requests[service_id]
            if not isinstance(values, list) and not isinstance(values, tuple):
                raise SpecError(
                    "appended requests must be an array",
                    field=f"append_requests.{service_id}",
                )
            for index, value in enumerate(values):
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise SpecError(
                        "appended request count must be a non-negative integer",
                        field=f"append_requests.{service_id}[{index}]",
                    )
            if append_length is None:
                append_length = len(values)
            elif len(values) != append_length:
                raise SpecError(
                    "appended batches must have equal minute counts",
                    field=f"append_requests.{service_id}",
                )
        if append_length is None or append_length == 0:
            raise SpecError(
                "append batch must contain at least one minute",
                field="append_requests",
            )

        new_services = tuple(
            replace(
                service,
                requests_per_minute=service.requests_per_minute
                + tuple(new_requests[service.service_id]),
            )
            for service in self.spec.services
        )
        new_spec = self._replace_spec(replace(self.spec, services=new_services))
        new_length = old_length + append_length

        full_minutes: dict[str, tuple[Any, ...]] = {}
        incremental_minutes: dict[str, tuple[Any, ...]] = {}
        reasons: list[dict[str, Any]] = []
        for service in new_spec.services:
            old_full = self.full_result.minutes[service.service_id]
            old_incremental = self.incremental_result.minutes[service.service_id]
            added_incremental = append_incremental_minutes(
                new_spec,
                service,
                old_incremental,
                tuple(new_requests[service.service_id]),
            )
            added_full_items = []
            for offset in range(append_length):
                if offset == 0:
                    baseline = (
                        old_full[-1].clamped_replicas
                        if old_full
                        else service.min_replicas
                    )
                else:
                    baseline = added_full_items[-1].clamped_replicas
                added_full_items.append(
                    compute_full_minute(
                        new_spec,
                        service,
                        old_length + offset,
                        baseline,
                    )
                )
            added_full = tuple(added_full_items)
            full_minutes[service.service_id] = old_full + added_full
            incremental_minutes[service.service_id] = (
                old_incremental + added_incremental
            )
            for offset in range(append_length):
                minute = old_length + offset
                window_start = max(
                    0, minute - new_spec.algorithm.window_length + 1
                )
                reasons.append(
                    {
                        "service_id": service.service_id,
                        "minute": minute,
                        "affected": True,
                        "recomputed_path": "targeted_full_and_ring_incremental",
                        "references": {
                            "raw_minute": minute,
                            "window_minutes": [window_start, minute],
                            "previous_clamped_minute": minute - 1,
                            "ring_index": minute % new_spec.algorithm.window_length,
                        },
                        "basis": "appended_raw_point_dirties_new_trailing_window",
                    }
                )

        new_full = PathResult("full", full_minutes)
        new_incremental = PathResult("incremental", incremental_minutes)
        new_allocations = verify_minute_range(
            new_spec,
            new_full,
            new_incremental,
            old_length,
            new_length,
        )
        allocation_lease_refs = {
            (minute, row.service_id): list(row.lease_ids)
            for minute, rows in new_allocations.items()
            for row in rows
        }
        for reason in reasons:
            key = (reason["minute"], reason["service_id"])
            reason["references"]["lease_ids"] = allocation_lease_refs.get(key, [])
            reason["references"]["lease_node"] = "allocation.lease_occupancy"

        for minute in range(old_length, new_length):
            for row in new_allocations[minute]:
                key = (minute, row.service_id)
                self.allocation_rows[key] = row
                self.allocation_bytes[key] = canonical_bytes(row.comparable_dict())
            for service_id in new_spec.service_ids:
                item = new_full.minutes[service_id][minute]
                self.demand_bytes[(service_id, minute)] = canonical_bytes(
                    item.comparable_dict()
                )

        propagation = {
            "change_type": "append_minutes",
            "old_version": self.version,
            "old_minute_count": old_length,
            "new_minute_count": new_length,
            "affected_services": list(new_spec.service_ids),
            "affected_minutes": list(range(old_length, new_length)),
            "recomputed_minutes": list(range(old_length, new_length)),
            "unchanged_minutes": list(range(0, old_length)),
            "reasons": sorted(
                reasons, key=lambda item: (item["service_id"], item["minute"])
            ),
        }
        return self._bump_version(
            "append_minutes", propagation, new_spec, new_full, new_incremental
        )

    def update_budget(
        self,
        *,
        global_replica_budget: int | None = None,
        service_concurrency_quotas: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        old_spec = self.spec
        new_global = (
            old_spec.budget.global_replica_budget
            if global_replica_budget is None
            else global_replica_budget
        )
        if not isinstance(new_global, int) or isinstance(new_global, bool) or new_global < 0:
            raise SpecError(
                "global_replica_budget must be a non-negative integer",
                field="budget.global_replica_budget",
            )
        new_quotas = dict(old_spec.budget.service_concurrency_quotas)
        changed_quota_services: set[str] = set()
        if service_concurrency_quotas is not None:
            for service_id, quota in service_concurrency_quotas.items():
                if service_id not in new_quotas:
                    raise SpecError(
                        "quota key does not match a declared service_id",
                        field=f"budget.service_concurrency_quotas.{service_id}",
                    )
                if not isinstance(quota, int) or isinstance(quota, bool) or quota < 0:
                    raise SpecError(
                        "service concurrency quota must be a non-negative integer",
                        field=f"budget.service_concurrency_quotas.{service_id}",
                    )
                if new_quotas[service_id] != quota:
                    changed_quota_services.add(service_id)
                new_quotas[service_id] = quota

        new_spec = self._replace_spec(
            replace(
                old_spec,
                budget=BudgetConfig(
                    global_replica_budget=new_global,
                    service_concurrency_quotas=dict(sorted(new_quotas.items())),
                ),
            )
        )

        dirty_minutes: set[int] = set()
        minute_count = old_spec.minute_count
        for minute in range(minute_count):
            if new_global != old_spec.budget.global_replica_budget:
                eligible_total = 0
                for service in old_spec.services:
                    clamped = self.full_result.minutes[service.service_id][
                        minute
                    ].clamped_replicas
                    effective_quota = (
                        new_quotas[service.service_id]
                        // service.concurrency_per_replica
                    )
                    eligible_total += min(clamped, effective_quota)
                if eligible_total > min(
                    old_spec.budget.global_replica_budget, new_global
                ):
                    dirty_minutes.add(minute)

            for service_id in changed_quota_services:
                service = old_spec.service(service_id)
                clamped = self.full_result.minutes[service_id][minute].clamped_replicas
                old_quota = old_spec.budget.service_concurrency_quotas[service_id]
                new_quota = new_quotas[service_id]
                old_effective = old_quota // service.concurrency_per_replica
                new_effective = new_quota // service.concurrency_per_replica
                if min(clamped, old_effective) != min(clamped, new_effective):
                    dirty_minutes.add(minute)

        reasons: list[dict[str, Any]] = []
        changed_allocation_keys: list[list[Any]] = []
        for minute in sorted(dirty_minutes):
            estimates = {
                service_id: self.full_result.minutes[service_id][minute]
                for service_id in new_spec.service_ids
            }
            rows = allocate_minute(new_spec, minute, estimates)
            for row in rows:
                key = (minute, row.service_id)
                new_bytes = canonical_bytes(row.comparable_dict())
                if self.allocation_bytes.get(key) != new_bytes:
                    changed_allocation_keys.append([minute, row.service_id, "allocated_replicas"])
                    self.allocation_rows[key] = row
                    self.allocation_bytes[key] = new_bytes
                for changed_field in (
                    "global_budget",
                    "service_concurrency_quota",
                    "final_replica_allocation",
                    "lease_occupancy",
                ):
                    reasons.append(
                        {
                            "service_id": row.service_id,
                            "minute": minute,
                            "field": changed_field,
                            "affected": True,
                            "lease_ids": list(row.lease_ids),
                            "basis": "budget_or_quota_reference_in_allocation_to_lease_node",
                        }
                    )

        propagation = {
            "change_type": "budget_update",
            "old_version": self.version,
            "old_global_replica_budget": old_spec.budget.global_replica_budget,
            "new_global_replica_budget": new_global,
            "changed_quota_services": sorted(changed_quota_services),
            "affected_services": list(new_spec.service_ids) if dirty_minutes else [],
            "affected_minutes": sorted(dirty_minutes),
            "recomputed_minutes": sorted(dirty_minutes),
            "actually_changed_allocation_keys": sorted(
                changed_allocation_keys, key=lambda item: (item[0], item[1])
            ),
            "unchanged_demand_minutes": list(range(minute_count)),
            "reasons": sorted(
                reasons, key=lambda item: (item["service_id"], item["minute"], item["field"])
            ),
        }
        return self._bump_version(
            "budget_update",
            propagation,
            new_spec,
            self.full_result,
            self.incremental_result,
        )

    def update_scaling_factor(self, scaling_factor_permille: int) -> dict[str, Any]:
        if (
            not isinstance(scaling_factor_permille, int)
            or isinstance(scaling_factor_permille, bool)
            or scaling_factor_permille <= 0
        ):
            raise SpecError(
                "scaling_factor_permille must be a positive integer",
                field="scaling_factor_permille",
            )
        old_spec = self.spec
        new_spec = self._replace_spec(
            replace(old_spec, scaling_factor_permille=scaling_factor_permille)
        )
        window_length = new_spec.algorithm.window_length
        new_minutes_by_service: dict[str, tuple[Any, ...]] = {}
        dirty_minutes: set[int] = set()
        reasons: list[dict[str, Any]] = []

        for service in new_spec.services:
            old_items = self.full_result.minutes[service.service_id]
            seed_minutes: set[int] = set()
            for minute, request_count in enumerate(service.requests_per_minute):
                raw_demand, clamped = estimate_minute(
                    request_count,
                    service.concurrency_per_replica,
                    service.min_replicas,
                    service.max_replicas,
                    scaling_factor_permille,
                )
                old = old_items[minute]
                if raw_demand != old.unclamped_replicas or clamped != old.clamped_replicas:
                    seed_minutes.add(minute)

            service_dirty: set[int] = set()
            for seed in sorted(seed_minutes):
                for minute in range(seed, min(new_spec.minute_count, seed + window_length)):
                    service_dirty.add(minute)
                if seed + 1 < new_spec.minute_count:
                    service_dirty.add(seed + 1)

            updated: list[Any] = list(old_items)
            newly_computed: dict[int, Any] = {}
            for minute in sorted(service_dirty):
                if minute == 0:
                    baseline = service.min_replicas
                elif minute - 1 in newly_computed:
                    baseline = newly_computed[minute - 1].clamped_replicas
                else:
                    baseline = old_items[minute - 1].clamped_replicas
                new_item = compute_full_minute(
                    new_spec, service, minute, baseline_replicas=baseline
                )
                newly_computed[minute] = new_item
                key = (service.service_id, minute)
                new_bytes = canonical_bytes(new_item.comparable_dict())
                if self.demand_bytes.get(key) != new_bytes:
                    self.demand_bytes[key] = new_bytes
                    updated[minute] = new_item
                else:
                    updated[minute] = old_items[minute]

            new_minutes_by_service[service.service_id] = tuple(updated)
            dirty_minutes.update(service_dirty)
            for minute in sorted(service_dirty):
                seeds = [
                    seed
                    for seed in sorted(seed_minutes)
                    if seed <= minute <= seed + window_length - 1 or minute == seed + 1
                ]
                reasons.append(
                    {
                        "service_id": service.service_id,
                        "minute": minute,
                        "affected": True,
                        "seed_minutes": seeds,
                        "references": {
                            "factor_node": "scaling_factor_permille",
                            "window_minutes": [
                                max(0, minute - window_length + 1),
                                minute,
                            ],
                            "action_baseline_minute": minute - 1,
                        },
                        "basis": "factor_estimate_seed_propagates_through_window_and_action",
                    }
                )

        new_full = PathResult("full", new_minutes_by_service)
        new_incremental = PathResult("incremental", new_minutes_by_service)
        factor_lease_refs: dict[tuple[str, int], list[str]] = {}
        for minute in sorted(dirty_minutes):
            estimates = {
                service_id: new_minutes_by_service[service_id][minute]
                for service_id in new_spec.service_ids
            }
            rows = allocate_minute(new_spec, minute, estimates)
            for row in rows:
                key = (minute, row.service_id)
                factor_lease_refs[(row.service_id, minute)] = list(row.lease_ids)
                new_bytes = canonical_bytes(row.comparable_dict())
                if self.allocation_bytes.get(key) != new_bytes:
                    self.allocation_rows[key] = row
                    self.allocation_bytes[key] = new_bytes
        for reason in reasons:
            reason["lease_ids"] = factor_lease_refs.get(
                (reason["service_id"], reason["minute"]), []
            )

        propagation = {
            "change_type": "scaling_factor_update",
            "old_version": self.version,
            "old_scaling_factor_permille": old_spec.scaling_factor_permille,
            "new_scaling_factor_permille": scaling_factor_permille,
            "affected_services": list(new_spec.service_ids) if dirty_minutes else [],
            "affected_minutes": sorted(dirty_minutes),
            "recomputed_minutes": sorted(dirty_minutes),
            "reasons": sorted(
                reasons, key=lambda item: (item["service_id"], item["minute"])
            ),
        }
        return self._bump_version(
            "scaling_factor_update",
            propagation,
            new_spec,
            new_full,
            new_incremental,
        )

    def audit_algorithm_paths(self) -> dict[str, Any]:
        full = run_full_path(self.spec)
        incremental = run_incremental_path(self.spec)
        full_allocations, incremental_allocations = verify_paths(
            self.spec, full, incremental
        )
        full_bytes = conclusion_bytes(self.spec, full, full_allocations)
        incremental_bytes = conclusion_bytes(
            self.spec, incremental, incremental_allocations
        )
        selection = select_path(self.spec)
        return {
            "equal": full_bytes == incremental_bytes,
            "full_digest": sha256_hex(full_bytes),
            "incremental_digest": sha256_hex(incremental_bytes),
            "byte_length": len(full_bytes),
            "selection": selection.ledger_basis(),
        }
