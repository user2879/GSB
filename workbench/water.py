from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_WATER_INT = 2**63 - 1


def saturate_int64(value: int) -> tuple[int, bool]:
    value = int(value)
    if value > MAX_WATER_INT:
        return MAX_WATER_INT, True
    if value < -MAX_WATER_INT - 1:
        return -MAX_WATER_INT - 1, True
    return value, False


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-int(numerator)) // denominator)


@dataclass(frozen=True)
class ErrorAttribution:
    version_edges: int = 0
    version_error_fp: int = 0
    dependency_uncertain: int = 0
    dependency_error_fp: int = 0
    lease_boundaries: int = 0
    lease_error_fp: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "version_edges": int(self.version_edges),
            "version_error_fp": int(self.version_error_fp),
            "dependency_uncertain": int(self.dependency_uncertain),
            "dependency_error_fp": int(self.dependency_error_fp),
            "lease_boundaries": int(self.lease_boundaries),
            "lease_error_fp": int(self.lease_error_fp),
            "source_error_fp": self.version_error_fp
            + self.dependency_error_fp
            + self.lease_error_fp,
        }


class WaterModel:
    """Integer fixed-point water marks.

    ``factor = 10 ** scale_exponent``.  A ratio ``used / capacity`` is stored
    as ``floor(used * factor / capacity)``.  All divisions are integer floor
    divisions for nonnegative quantities.  Values outside signed int64 are
    saturated to the signed int64 boundary and flagged.
    """

    def __init__(self, scale_exponent: int = 0) -> None:
        if not isinstance(scale_exponent, int) or isinstance(scale_exponent, bool):
            raise ValueError("scale_exponent must be an integer")
        if not 0 <= scale_exponent <= 3:
            raise ValueError("scale_exponent must be in [0, 3]")
        self.scale_exponent = int(scale_exponent)
        self.factor = 10**self.scale_exponent

    def scale_amount(self, value: int) -> dict[str, int | bool]:
        scaled, saturated = saturate_int64(int(value) * self.factor)
        return {"raw": int(value), "scaled": scaled, "saturated": saturated}

    def ratio_water_fp(self, used: int, capacity: int) -> dict[str, Any]:
        used = max(0, int(used))
        capacity = int(capacity)
        scaled_used, used_saturated = saturate_int64(used * self.factor)
        scaled_capacity, capacity_saturated = saturate_int64(capacity * self.factor)
        if capacity <= 0:
            water = self.factor
            remainder = 0
            raw_bound = 0
            saturated = True
        else:
            raw = used * self.factor
            water = raw // capacity
            remainder = raw % capacity
            raw_bound = ceil_div(capacity, self.factor)
            water, saturated = saturate_int64(water)
            if water > self.factor:
                water = self.factor
                saturated = True
            saturated = saturated or used_saturated or capacity_saturated
        return {
            "used": used,
            "capacity": capacity,
            "scaled_used": scaled_used,
            "scaled_capacity": scaled_capacity,
            "water_fp": int(water),
            "factor": self.factor,
            "division_remainder": int(remainder),
            "truncation_error_fp_bound": 1,
            "truncation_error_raw_bound": int(raw_bound),
            "saturated": bool(saturated),
            "truncation_direction": "floor_nonnegative",
            "overflow_rule": "saturate_signed_int64",
        }

    def shard_error_record(
        self,
        *,
        shard_id: str,
        resource_units: int,
        global_budget: int,
        attribution: ErrorAttribution | None = None,
    ) -> dict[str, Any]:
        attribution = attribution or ErrorAttribution()
        global_water = self.ratio_water_fp(resource_units, global_budget)
        source = attribution.as_dict()
        synthetic_bound = (
            int(global_water["truncation_error_fp_bound"])
            + int(source["source_error_fp"])
        )
        raw_bound = (
            ceil_div(synthetic_bound * max(1, global_budget), self.factor)
            if global_budget > 0
            else MAX_WATER_INT
        )
        return {
            "shard_id": shard_id,
            "scale_exponent": self.scale_exponent,
            "scale_factor": self.factor,
            "resource_units": int(resource_units),
            "global_budget_units": int(global_budget),
            "global_share_water_fp": global_water["water_fp"],
            "division_remainder": global_water["division_remainder"],
            "quantization_truncation_error_fp": 1,
            "quantization_truncation_error_raw_bound": global_water[
                "truncation_error_raw_bound"
            ],
            "synthetic_error_bound_fp": synthetic_bound,
            "synthetic_error_bound_raw_global_units": raw_bound,
            **source,
        }

    def node_water(
        self,
        *,
        active_shards: int,
        concurrency_quota: int,
        active_leases: int,
        lease_limit: int,
        node_resource_units: int,
        global_budget: int,
        attribution: ErrorAttribution | None = None,
    ) -> dict[str, Any]:
        attribution = attribution or ErrorAttribution()
        quota = self.ratio_water_fp(active_shards, concurrency_quota)
        lease = self.ratio_water_fp(active_leases, lease_limit)
        resource = self.ratio_water_fp(node_resource_units, global_budget)
        water_fp = max(
            int(quota["water_fp"]),
            int(lease["water_fp"]),
            int(resource["water_fp"]),
        )
        source = attribution.as_dict()
        comparison_bound_fp = 2 + int(source["source_error_fp"])
        return {
            "quota_water": quota,
            "lease_water": lease,
            "resource_water": resource,
            "node_water_fp": water_fp,
            "comparison_error_bound_fp": comparison_bound_fp,
            "tie_rule": "stable_topological_order_then_shard_id",
            **source,
        }

    def global_error_summary(
        self, shard_errors: list[dict[str, Any]]
    ) -> dict[str, Any]:
        count = len(shard_errors)
        quant_bound = count
        version_bound = sum(int(item["version_error_fp"]) for item in shard_errors)
        dependency_bound = sum(
            int(item["dependency_error_fp"]) for item in shard_errors
        )
        lease_bound = sum(int(item["lease_error_fp"]) for item in shard_errors)
        source_bound = version_bound + dependency_bound + lease_bound
        raw_bound = sum(
            int(item["quantization_truncation_error_raw_bound"])
            for item in shard_errors
        )
        return {
            "shard_count": count,
            "aggregate_quantization_error_bound_fp": quant_bound,
            "version_chain_error_fp": version_bound,
            "dependency_error_fp": dependency_bound,
            "lease_error_fp": lease_bound,
            "source_error_bound_fp": source_bound,
            "global_aggregate_error_bound_fp": quant_bound + source_bound,
            "global_aggregate_error_bound_raw_units": raw_bound,
            "direct_global_water_bound_fp": 1,
            "note": "sum of per-shard floor bounds is conservative; direct global ratio has bound 1 fp unit",
        }

    def budget_gap_is_ambiguous(
        self, *, remaining_budget: int, demand: int, shard_error: dict[str, Any]
    ) -> bool:
        gap = int(remaining_budget) - int(demand)
        bound = int(shard_error["synthetic_error_bound_raw_global_units"])
        return abs(gap) <= bound

    def water_difference_is_ambiguous(
        self, water_a_fp: int, water_b_fp: int, comparison_bound_fp: int
    ) -> bool:
        return abs(int(water_a_fp) - int(water_b_fp)) <= int(comparison_bound_fp)


def stable_shard_key(topo_index: dict[str, int], shard_id: str) -> tuple[int, str]:
    return (int(topo_index.get(shard_id, 10**12)), shard_id)


def choose_by_water_and_order(
    candidates: list[dict[str, Any]],
    *,
    topo_index: dict[str, int],
    water_model: WaterModel,
) -> dict[str, Any]:
    """Choose a candidate using water with an explicit stable fallback.

    Candidate dictionaries need ``shard_id`` and ``water_fp`` and may carry
    ``comparison_error_bound_fp``.  When the leading two water marks are inside
    the error bound, topology order then shard id decides and the winner is
    marked ``error_affected``.
    """

    if not candidates:
        raise ValueError("no candidates")
    ordered_by_water = sorted(candidates, key=lambda item: int(item["water_fp"]))
    winner = dict(ordered_by_water[0])
    affected = False
    if len(ordered_by_water) > 1:
        second = ordered_by_water[1]
        bound = max(
            int(winner.get("comparison_error_bound_fp", 2)),
            int(second.get("comparison_error_bound_fp", 2)),
        )
        affected = water_model.water_difference_is_ambiguous(
            int(winner["water_fp"]), int(second["water_fp"]), bound
        )
        if affected:
            stable_winner = min(
                (winner, second),
                key=lambda item: stable_shard_key(topo_index, str(item["shard_id"])),
            )
            winner = dict(stable_winner)
    stable_order_winner = min(
        candidates, key=lambda item: stable_shard_key(topo_index, str(item["shard_id"]))
    )
    if affected:
        winner = dict(stable_order_winner)
    winner["error_affected"] = affected
    winner["decision_rule"] = (
        "error_bound:topological_order:shard_id"
        if affected
        else "lowest_water:topological_order:shard_id"
    )
    return winner
