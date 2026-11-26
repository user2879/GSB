"""Stable integer-only arbitration at ambiguous fixed-point boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fixed import FixedErrorModel, WaterLevel


@dataclass(frozen=True)
class WaterCandidate:
    shard_id: str
    node_id: str
    demand_units: int
    water: WaterLevel

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "demand_units": self.demand_units,
            "water_vector": self.water.as_dict(),
        }


def _stable_key(candidate: WaterCandidate, topo_rank: dict[str, int]) -> tuple[int, str]:
    return topo_rank[candidate.shard_id], candidate.shard_id


def arbitrate_water(
    first: WaterCandidate,
    second: WaterCandidate,
    *,
    topo_rank: dict[str, int],
    error_model: FixedErrorModel,
) -> dict[str, Any]:
    if first.shard_id == second.shard_id:
        raise ValueError("water arbitration needs two distinct candidates")
    utilization_delta = first.water.utilization_fp - second.water.utilization_fp
    headroom_delta = first.water.headroom_fp - second.water.headroom_fp
    bound = error_model.per_node_comparison_bound_fp
    indistinguishable = abs(utilization_delta) <= bound

    first_key = _stable_key(first, topo_rank)
    second_key = _stable_key(second, topo_rank)
    if indistinguishable:
        chosen = min((first, second), key=lambda candidate: _stable_key(candidate, topo_rank))
        basis = ["water_difference_at_or_within_error_bound", "stable_topological_order", "shard_id"]
        affected = True
        reason = (
            f"|{first.water.utilization_fp} - {second.water.utilization_fp}| = "
            f"{abs(utilization_delta)} <= {bound}; stable key "
            f"({chosen.shard_id}) selected"
        )
    else:
        preferred = first if utilization_delta < 0 else second
        if first.water.utilization_fp == second.water.utilization_fp:
            preferred = min((first, second), key=lambda candidate: _stable_key(candidate, topo_rank))
            basis = ["equal_utilization_fp", "stable_topological_order", "shard_id"]
        else:
            basis = ["lower_utilization_fp_outside_error_bound"]
        chosen = preferred
        affected = False
        reason = (
            f"|{first.water.utilization_fp} - {second.water.utilization_fp}| = "
            f"{abs(utilization_delta)} > {bound}; lower integer utilization selected"
        )

    return {
        "decision": chosen.shard_id,
        "decision_basis": basis,
        "decision_source": "fixed_integer_water_v1",
        "error_affected": affected,
        "reason": reason,
        "candidates": [first.as_dict(), second.as_dict()],
        "topological_keys": {
            first.shard_id: list(first_key),
            second.shard_id: list(second_key),
        },
        "difference": {
            "utilization_fp": utilization_delta,
            "headroom_fp": headroom_delta,
            "compared_field": "utilization_fp",
        },
        "error_bound": error_model.as_dict(),
    }


def arbitrate_budget(
    remaining_units: int,
    first: WaterCandidate,
    second: WaterCandidate,
    *,
    topo_rank: dict[str, int],
    error_model: FixedErrorModel,
    budget_error_bound_units: int = 0,
) -> dict[str, Any]:
    if remaining_units < 0 or budget_error_bound_units < 0:
        raise ValueError("remaining budget and error bound must be non-negative")
    if budget_error_bound_units * error_model.scale > error_model.total_bound_fp:
        raise ValueError("budget error bound exceeds the declared total fixed error bound")

    residual_first = remaining_units - first.demand_units
    residual_second = remaining_units - second.demand_units
    residual_delta = residual_first - residual_second
    indistinguishable = abs(residual_delta) <= budget_error_bound_units
    feasible = [candidate for candidate in (first, second) if remaining_units >= candidate.demand_units]

    if indistinguishable:
        pool = feasible or [first, second]
        chosen = min(pool, key=lambda candidate: _stable_key(candidate, topo_rank))
        basis = ["budget_residual_at_or_within_error_bound", "stable_topological_order", "shard_id"]
        affected = True
    elif feasible:
        chosen = max(feasible, key=lambda candidate: (remaining_units - candidate.demand_units, -_stable_key(candidate, topo_rank)[0], candidate.shard_id))
        basis = ["larger_integer_budget_residual"]
        affected = False
    else:
        chosen = min((first, second), key=lambda candidate: _stable_key(candidate, topo_rank))
        basis = ["neither_candidate_fits", "defer"]
        affected = False

    admits = remaining_units >= chosen.demand_units
    return {
        "decision": chosen.shard_id,
        "admitted": admits,
        "defer": not admits,
        "decision_basis": basis,
        "decision_source": "integer_resource_units_v1",
        "error_affected": affected,
        "remaining_units": remaining_units,
        "residuals_units": {first.shard_id: residual_first, second.shard_id: residual_second},
        "difference_units": residual_delta,
        "budget_error_bound_units": budget_error_bound_units,
        "candidates": [first.as_dict(), second.as_dict()],
        "error_bound": error_model.as_dict(),
        "topological_keys": {
            first.shard_id: list(_stable_key(first, topo_rank)),
            second.shard_id: list(_stable_key(second, topo_rank)),
        },
    }

