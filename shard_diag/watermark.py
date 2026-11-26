from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)


def monotonic_ms() -> int:
    # CLOCK_MONOTONIC excludes wall-clock jumps; nanoseconds are floored to ms.
    import time

    return time.monotonic_ns() // 1_000_000


def saturating_fixed_int(value: int) -> int:
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return int(value)


def quantize_units(raw_units: int, scale: int) -> dict[str, int | bool]:
    """Multiply integer units by the session scale.

    Truncation is irrelevant for this exact multiplication.  Results outside
    signed SQLite/int64 storage saturate at the corresponding boundary.
    """
    exact = int(raw_units) * int(scale)
    q = saturating_fixed_int(exact)
    return {
        "raw": int(raw_units),
        "q": q,
        "scale": int(scale),
        "truncation_direction": "none_for_multiplication",
        "saturated": q != exact,
    }


def floor_ratio_q(numerator: int, denominator: int, scale: int) -> dict[str, int]:
    """Floor numerator/denominator * scale for non-negative water ratios.

    Fixed-point remainder satisfies 0 <= remainder < denominator.  Therefore
    the discarded error is remainder/denominator of one fixed-point ulp.
    """
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    scaled = int(numerator) * int(scale)
    q = saturating_fixed_int(scaled // denominator)
    remainder = scaled % denominator
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "scale": int(scale),
        "q": int(q),
        "remainder_num": int(remainder),
        "error_num": int(remainder),
        "error_den": int(denominator),
        "error_bound_q_num": 1,
        "error_bound_q_den": 1,
    }


@dataclass(frozen=True)
class BudgetAttribution:
    budget: int
    used_units: int
    remaining_units: int
    scale: int
    participants: tuple[dict[str, Any], ...]
    sum_of_floors_q: int
    aggregate_q: int
    aggregate_remainder_num: int
    carry_q: int
    error_bound_q_inclusive: int
    error_bound_q_exclusive: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "used_units": self.used_units,
            "remaining_units": self.remaining_units,
            "scale": self.scale,
            "participants": list(self.participants),
            "sum_of_floors_q": self.sum_of_floors_q,
            "aggregate_q": self.aggregate_q,
            "aggregate_remainder_num": self.aggregate_remainder_num,
            "carry_q": self.carry_q,
            "error_bound_q_inclusive": self.error_bound_q_inclusive,
            "error_bound_q_exclusive": self.error_bound_q_exclusive,
            "rounding_rule": "floor_nonnegative_toward_zero",
            "saturation_rule": "signed_int64_boundary",
        }


def global_budget_attribution(
    allocations: Iterable[tuple[str, int]],
    budget: int,
    scale: int,
) -> BudgetAttribution:
    items = list(allocations)
    used = sum(units for _, units in items)
    participants: list[dict[str, Any]] = []
    sum_floors = 0
    for shard_id, units in items:
        ratio = floor_ratio_q(units, budget, scale)
        sum_floors += ratio["q"]
        participants.append(
            {
                "object_id": shard_id,
                "raw_units": int(units),
                "quantized_units_q": int(quantize_units(units, scale)["q"]),
                "budget_share_q": ratio["q"],
                "error_fraction_num": ratio["remainder_num"],
                "error_fraction_den": budget,
                "error_bound_q_exclusive": 1,
            }
        )
    aggregate = floor_ratio_q(used, budget, scale)
    carry = aggregate["q"] - sum_floors
    count = len(items)
    return BudgetAttribution(
        budget=int(budget),
        used_units=int(used),
        remaining_units=int(budget - used),
        scale=int(scale),
        participants=tuple(participants),
        sum_of_floors_q=int(sum_floors),
        aggregate_q=aggregate["q"],
        aggregate_remainder_num=aggregate["remainder_num"],
        carry_q=int(carry),
        # Each of n floor errors is < 1 ulp. Integer rounded carry can be at
        # most n-1; n is the safe exclusive upper bound.
        error_bound_q_inclusive=max(0, count - 1),
        error_bound_q_exclusive=count,
    )


def budget_candidate_judgement(
    *,
    used_units: int,
    candidate_units: int,
    budget: int,
    scale: int,
) -> dict[str, Any]:
    exact_slack = int(budget) - int(used_units) - int(candidate_units)
    remaining = floor_ratio_q(max(0, budget - used_units), budget, scale)
    demand = floor_ratio_q(candidate_units, budget, scale)
    delta_q = remaining["q"] - demand["q"]
    # Two independent floor operations each discard < 1 ulp.
    error_bound_q = 2
    error_affected = abs(delta_q) <= error_bound_q
    return {
        "exact_slack_units": exact_slack,
        "exact_admissible": exact_slack >= 0,
        "remaining_share_q": remaining["q"],
        "candidate_share_q": demand["q"],
        "water_delta_q": delta_q,
        "error_bound_q": error_bound_q,
        "error_affected": error_affected,
        "tie_rule": "higher priority, then stable Kahn topological order, then lexicographic shard_id",
    }


def node_water_report(
    *,
    node_id: str,
    active_shards: int,
    active_leases: int,
    concurrency_quota: int,
    max_leases: int,
    scale: int,
    participant_ids: Iterable[str] = (),
) -> dict[str, Any]:
    slot = floor_ratio_q(active_shards, concurrency_quota, scale)
    lease = floor_ratio_q(active_leases, max_leases, scale)
    ids = list(participant_ids)
    return {
        "scope": "node",
        "node_id": node_id,
        "active_shards": int(active_shards),
        "active_leases": int(active_leases),
        "concurrency_quota": int(concurrency_quota),
        "max_leases_per_node": int(max_leases),
        "slot_water_q": slot["q"],
        "lease_water_q": lease["q"],
        "slot_rounding": slot,
        "lease_rounding": lease,
        "participant_ids": ids,
        "comparison_error_bound_q": 2,
        "rounding_rule": "floor_nonnegative_toward_zero",
    }


def compare_fixed_water(a_q: int, b_q: int, error_bound_q: int = 2) -> dict[str, Any]:
    delta = int(a_q) - int(b_q)
    return {
        "a_q": int(a_q),
        "b_q": int(b_q),
        "delta_q": delta,
        "error_bound_q": int(error_bound_q),
        "error_affected": abs(delta) <= int(error_bound_q),
        "conclusion": "a_lower" if delta < 0 else "b_lower" if delta > 0 else "equal_or_indistinguishable",
        "tie_break": "stable topological order then lexicographic shard_id",
    }


def stable_candidate_sort_key(
    shard_id: str,
    priority: int,
    topo_rank: dict[str, int],
) -> tuple[int, int, str]:
    return (-int(priority), int(topo_rank[shard_id]), shard_id)
