"""Deterministic integer fixed-point entitlement apportionment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .spec import Declaration, SHARE_SCALE


@dataclass(frozen=True)
class Apportionment:
    row: dict[str, Any]


def _apportion(units: int, weights: Iterable[int], tie_break: dict[str, Any] | None = None) -> list[int]:
    """Hamilton apportionment with an explicit deterministic tie break."""

    weight_values = list(weights)
    total_weight = sum(weight_values)
    if units <= 0 or total_weight <= 0:
        return [0 for _ in weight_values]
    shares = [(units * weight) // total_weight for weight in weight_values]
    remainders = [(units * weight) % total_weight for weight in weight_values]
    left = units - sum(shares)
    order = sorted(
        range(len(weight_values)),
        key=lambda index: (
            -remainders[index],
            -(weight_values[index]),
            (tie_break or {}).get(str(index), ""),
            index,
        ),
    )
    for index in order[:left]:
        shares[index] += 1
    return shares


def derive_entitlements(
    declaration: Declaration,
    *,
    workload_units: dict[str, int] | None = None,
    pressure_units: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Derive tenant entitlements and the live allocation matrix.

    The fixed-point rule is explicit:

    1. ``floor_i = cluster_total * quota_bp_i // 10000``.
    2. ``share_budget = cluster_total * sum(quota_bp) // 10000``.
    3. The remaining ``share_budget - sum(floor_i)`` units are assigned by
       largest remainder (``cluster_total * quota_bp_i % 10000``). Ties use
       larger ``quota_bp``, then lexicographically smaller ``tenant_id``.
    4. A tenant receives at least its reserve. If raised entitlements exceed
       cluster capacity, the overflow is removed from only the reducible part
       above reserves, using largest-remainder apportionment of the overflow.
       This keeps every result >= reserve and total <= cluster capacity.

    Tenant input order never affects any step.
    """

    tenants = declaration.sorted_tenants()
    cluster = declaration.cluster_total_units
    share_sum = sum(tenant.quota_bp for tenant in tenants)
    reserve_total = sum(tenant.reserve_units for tenant in tenants)
    if reserve_total > cluster:
        raise AssertionError("reserve total must be validated before apportionment")

    shared_budget = (cluster * share_sum) // SHARE_SCALE
    floors: dict[str, int] = {}
    remainders: dict[str, int] = {}
    for tenant in tenants:
        numerator = cluster * tenant.quota_bp
        floors[tenant.tenant_id] = numerator // SHARE_SCALE
        remainders[tenant.tenant_id] = numerator % SHARE_SCALE

    floor_total = sum(floors.values())
    bonus_left = shared_budget - floor_total
    bonus_order = sorted(
        tenants,
        key=lambda tenant: (
            -remainders[tenant.tenant_id],
            -tenant.quota_bp,
            tenant.tenant_id,
        ),
    )
    bonuses = {tenant.tenant_id: 0 for tenant in tenants}
    for tenant in bonus_order[: max(0, bonus_left)]:
        bonuses[tenant.tenant_id] = 1

    share_entitlements = {
        tenant.tenant_id: floors[tenant.tenant_id] + bonuses[tenant.tenant_id]
        for tenant in tenants
    }
    tentative = {
        tenant.tenant_id: max(share_entitlements[tenant.tenant_id], tenant.reserve_units)
        for tenant in tenants
    }
    tentative_total = sum(tentative.values())
    capacity_reductions = {tenant.tenant_id: 0 for tenant in tenants}
    if tentative_total > cluster:
        overflow = tentative_total - cluster
        reducible = {
            tenant.tenant_id: tentative[tenant.tenant_id] - tenant.reserve_units
            for tenant in tenants
        }
        reducible_ids = [tenant.tenant_id for tenant in tenants if reducible[tenant.tenant_id] > 0]
        reductions = _apportion(
            overflow,
            [reducible[tenant_id] for tenant_id in reducible_ids],
            {str(index): tenant_id for index, tenant_id in enumerate(reducible_ids)},
        )
        for tenant_id, reduction in zip(reducible_ids, reductions):
            if reduction > reducible[tenant_id]:
                raise AssertionError("capacity reduction exceeded reducible amount")
            capacity_reductions[tenant_id] = reduction

    entitlements = {
        tenant.tenant_id: tentative[tenant.tenant_id] - capacity_reductions[tenant.tenant_id]
        for tenant in tenants
    }
    if sum(entitlements.values()) > cluster:
        raise AssertionError("entitlements exceed cluster capacity")
    if any(entitlements[t.tenant_id] < t.reserve_units for t in tenants):
        raise AssertionError("entitlement crossed a reserve floor")

    workload_units = workload_units or {}
    pressure_units = pressure_units or {}
    rows: list[dict[str, Any]] = []
    for tenant in tenants:
        tenant_id = tenant.tenant_id
        share_entitlement = share_entitlements[tenant_id]
        entitlement = entitlements[tenant_id]
        workload_value = int(workload_units.get(tenant_id, 0))
        pressure_value = int(pressure_units.get(tenant_id, 0))
        actual = workload_value + pressure_value
        rows.append(
            {
                "tenant_id": tenant_id,
                "priority": tenant.priority,
                "share_bp": tenant.quota_bp,
                "reserve_units": tenant.reserve_units,
                "share_floor_units": floors[tenant_id],
                "share_remainder_numerator": remainders[tenant_id],
                "share_rounding_bonus_units": bonuses[tenant_id],
                "share_entitlement_units": share_entitlement,
                "reserve_uplift_units": max(0, tenant.reserve_units - share_entitlement),
                "capacity_reduction_units": capacity_reductions[tenant_id],
                "entitlement_units": entitlement,
                "share_quantization_error_scaled_1e4": share_entitlement * SHARE_SCALE
                - cluster * tenant.quota_bp,
                "entitlement_error_scaled_1e4": entitlement * SHARE_SCALE
                - cluster * tenant.quota_bp,
                "workload_units": workload_value,
                "pressure_units": pressure_value,
                "actual_units": actual,
                "excess_units": max(0, actual - entitlement),
                "deviation_units": actual - entitlement,
                "headroom_units": max(0, entitlement - actual),
                "reserve_remaining_units": max(0, tenant.reserve_units - actual),
            }
        )

    total_actual = sum(row["actual_units"] for row in rows)
    total_entitlement = sum(row["entitlement_units"] for row in rows)
    tenant_excess_total = sum(row["excess_units"] for row in rows)
    return {
        "cluster_total_units": cluster,
        "share_scale": SHARE_SCALE,
        "share_bp_sum": share_sum,
        "shared_budget_units": shared_budget,
        "floor_entitlement_units": floor_total,
        "reserve_total_units": reserve_total,
        "entitlement_total_units": total_entitlement,
        "unassigned_units": cluster - total_entitlement,
        "workload_total_units": sum(row["workload_units"] for row in rows),
        "pressure_total_units": sum(row["pressure_units"] for row in rows),
        "actual_total_units": total_actual,
        "cluster_excess_units": max(0, total_actual - cluster),
        "tenant_excess_total_units": tenant_excess_total,
        "tenants": rows,
    }


def allocation_by_tenant(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["tenant_id"]: row for row in matrix["tenants"]}


def matrix_digest_payload(matrix: dict[str, Any]) -> dict[str, Any]:
    return {
        "cluster_total_units": matrix["cluster_total_units"],
        "entitlement_total_units": matrix["entitlement_total_units"],
        "unassigned_units": matrix["unassigned_units"],
        "actual_total_units": matrix["actual_total_units"],
        "cluster_excess_units": matrix["cluster_excess_units"],
        "tenant_excess_total_units": matrix["tenant_excess_total_units"],
        "tenants": matrix["tenants"],
    }
