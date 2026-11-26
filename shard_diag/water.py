from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


FP_MAX = 2**63 - 1
DOMINANCE_SHARE_NUM = 6
DOMINANCE_SHARE_DEN = 10


def scale_for_power(power: int) -> int:
    if not isinstance(power, int) or isinstance(power, bool) or not 0 <= power <= 3:
        raise ValueError("scale_power must be an integer in [0, 3]")
    return 10**power


def saturated_nonnegative(value: int) -> tuple[int, bool]:
    if value < 0:
        raise ValueError("fixed-point values must be non-negative")
    if value > FP_MAX:
        return FP_MAX, True
    return int(value), False


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-int(numerator)) // int(denominator))


def quantize_units(value: int, scale_power: int) -> dict[str, Any]:
    scale = scale_for_power(scale_power)
    quantized, saturated = saturated_nonnegative(int(value) * scale)
    return {"q": quantized, "scale": scale, "saturated": saturated}


@dataclass(frozen=True, slots=True)
class WaterAttribution:
    node_id: str
    shard_id: str
    scale_power: int
    scale: int
    delay_ms_q: int
    resource_units_q: int
    lease_limit_q: int
    delay_ms: int
    jitter_ms: int
    revision_lag: int
    max_revision_lag: int
    resource_units: int
    lease_limit: int
    lease_rearrangements: int
    water_q: int
    delay_limit_q: int
    quantization_bound_q: int
    jitter_bound_q: int
    lease_bound_q: int
    total_error_bound_q: int
    dominant_segment: str
    dominance_threshold: str
    gate: str
    exact_delay_allowed: bool
    revision_allowed: bool
    error_affected: bool
    saturated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "shard_id": self.shard_id,
            "scale_power": self.scale_power,
            "scale": self.scale,
            "quantized_inputs_q": {
                "replay_delay_ms": self.delay_ms_q,
                "resource_units": self.resource_units_q,
                "lease_limit": self.lease_limit_q,
            },
            "delay_ms": self.delay_ms,
            "jitter_ms": self.jitter_ms,
            "revision_lag": self.revision_lag,
            "max_revision_lag": self.max_revision_lag,
            "resource_units": self.resource_units,
            "lease_limit": self.lease_limit,
            "lease_rearrangements": self.lease_rearrangements,
            "water_q": self.water_q,
            "delay_limit_q": self.delay_limit_q,
            "error_segments_q": {
                "quantization_truncation": self.quantization_bound_q,
                "replay_observation_jitter": self.jitter_bound_q,
                "lease_rearrangement": self.lease_bound_q,
            },
            "total_error_bound_q": self.total_error_bound_q,
            "dominant_segment": self.dominant_segment,
            "dominance_threshold": self.dominance_threshold,
            "gate": self.gate,
            "exact_delay_allowed": self.exact_delay_allowed,
            "revision_allowed": self.revision_allowed,
            "error_affected": self.error_affected,
            "saturated": self.saturated,
            "truncation_rule": "non-negative floor division toward zero",
            "overflow_rule": f"saturate at int64 max {FP_MAX}",
        }


def _dominant_segment(quant: int, jitter: int, lease: int) -> str:
    segments = {
        "quantization_truncation": quant,
        "replay_observation_jitter": jitter,
        "lease_rearrangement": lease,
    }
    total = sum(segments.values())
    name, value = max(segments.items(), key=lambda item: (item[1], item[0]))
    if total > 0 and value * DOMINANCE_SHARE_DEN >= DOMINANCE_SHARE_NUM * total:
        return name
    return "mixed"


def candidate_water(
    *,
    node_id: str,
    shard_id: str,
    delay_ms: int,
    jitter_ms: int,
    revision_lag: int,
    max_revision_lag: int,
    delay_limit_ms: int,
    resource_units: int,
    lease_limit: int,
    scale_power: int,
    lease_rearrangements: int = 0,
    reorder_granule_ms: int = 1,
) -> WaterAttribution:
    scale = scale_for_power(scale_power)
    if min(delay_ms, jitter_ms, revision_lag, resource_units, lease_limit) < 0:
        raise ValueError("water inputs must be non-negative")
    if resource_units <= 0 or lease_limit <= 0:
        raise ValueError("resource_units and lease_limit must be positive")

    delay_q, delay_saturated = saturated_nonnegative(int(delay_ms) * scale)
    resource_q, resource_saturated = saturated_nonnegative(int(resource_units) * scale)
    lease_q, lease_saturated = saturated_nonnegative(int(lease_limit) * scale)

    water_raw = delay_ms * scale * resource_units
    limit_raw = delay_limit_ms * scale * resource_units
    water_q, water_saturated = saturated_nonnegative(water_raw // lease_limit)
    delay_limit_q, limit_saturated = saturated_nonnegative(limit_raw // lease_limit)

    quantization_bound_q = 1
    jitter_bound_q = ceil_div(jitter_ms * scale * resource_units, lease_limit)
    lease_bound_q = ceil_div(
        max(0, lease_rearrangements) * max(1, reorder_granule_ms) * scale * resource_units,
        lease_limit,
    )
    saturated = any(
        (
            delay_saturated,
            resource_saturated,
            lease_saturated,
            water_saturated,
            limit_saturated,
        )
    )
    total_error_bound_q = quantization_bound_q + jitter_bound_q + lease_bound_q
    if saturated:
        total_error_bound_q = FP_MAX

    margin_q = delay_limit_q - water_q
    if saturated:
        gate = "indeterminate"
    elif margin_q > total_error_bound_q:
        gate = "allow"
    elif margin_q < -total_error_bound_q:
        gate = "deny"
    else:
        gate = "indeterminate"

    revision_allowed = revision_lag <= max_revision_lag
    if not revision_allowed and gate == "allow":
        gate = "deny"

    return WaterAttribution(
        node_id=node_id,
        shard_id=shard_id,
        scale_power=scale_power,
        scale=scale,
        delay_ms_q=delay_q,
        resource_units_q=resource_q,
        lease_limit_q=lease_q,
        delay_ms=int(delay_ms),
        jitter_ms=int(jitter_ms),
        revision_lag=int(revision_lag),
        max_revision_lag=int(max_revision_lag),
        resource_units=int(resource_units),
        lease_limit=int(lease_limit),
        lease_rearrangements=int(lease_rearrangements),
        water_q=water_q,
        delay_limit_q=delay_limit_q,
        quantization_bound_q=quantization_bound_q,
        jitter_bound_q=jitter_bound_q,
        lease_bound_q=lease_bound_q,
        total_error_bound_q=total_error_bound_q,
        dominant_segment=_dominant_segment(
            quantization_bound_q, jitter_bound_q, lease_bound_q
        ),
        dominance_threshold="largest_segment_share >= 0.60",
        gate=gate,
        exact_delay_allowed=delay_ms <= delay_limit_ms,
        revision_allowed=revision_allowed,
        error_affected=gate == "indeterminate",
        saturated=saturated,
    )


@dataclass(frozen=True, slots=True)
class WaterComparison:
    left_shard_id: str
    right_shard_id: str
    delta_q: int
    error_bound_q: int
    indistinguishable: bool
    winner: str
    stable_tie_winner: str
    water_winner: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_shard_id": self.left_shard_id,
            "right_shard_id": self.right_shard_id,
            "delta_q": self.delta_q,
            "error_bound_q": self.error_bound_q,
            "indistinguishable": self.indistinguishable,
            "winner": self.winner,
            "stable_tie_winner": self.stable_tie_winner,
            "water_winner": self.water_winner,
            "reason": self.reason,
        }


def compare_candidate_water(
    left: WaterAttribution,
    right: WaterAttribution,
    left_topo_rank: int,
    right_topo_rank: int,
) -> WaterComparison:
    delta = abs(left.water_q - right.water_q)
    bound = left.total_error_bound_q + right.total_error_bound_q
    indistinguishable = delta <= bound
    stable_winner = (
        left.shard_id
        if (left_topo_rank, left.shard_id) <= (right_topo_rank, right.shard_id)
        else right.shard_id
    )
    if left.water_q < right.water_q:
        water_winner = left.shard_id
    elif right.water_q < left.water_q:
        water_winner = right.shard_id
    else:
        water_winner = None

    if indistinguishable:
        reason = "fixed-point water gap is inside composite error bound; stable topology/id wins"
        winner = stable_winner
    elif water_winner is not None:
        reason = "water gap exceeds error bound; lower water wins"
        winner = water_winner
    else:
        reason = "equal water; stable topology/id wins"
        winner = stable_winner
    return WaterComparison(
        left_shard_id=left.shard_id,
        right_shard_id=right.shard_id,
        delta_q=delta,
        error_bound_q=bound,
        indistinguishable=indistinguishable,
        winner=winner,
        stable_tie_winner=stable_winner,
        water_winner=water_winner,
        reason=reason,
    )


def global_budget_error_bound(
    *,
    shard_count: int,
    scale_power: int,
    jitter_bounds_q: Sequence[int] = (),
    lease_bounds_q: Sequence[int] = (),
    saturated: bool = False,
) -> dict[str, Any]:
    scale = scale_for_power(scale_power)
    aggregate_water_bound_q = 1
    distributed_sum_bound_q = max(0, int(shard_count))
    jitter_total_q = sum(int(value) for value in jitter_bounds_q)
    lease_total_q = sum(int(value) for value in lease_bounds_q)
    safe_upper_bound_q = distributed_sum_bound_q + jitter_total_q + lease_total_q
    if saturated:
        safe_upper_bound_q = FP_MAX
    return {
        "scale": scale,
        "hard_resource_quantization_bound_q": 0,
        "aggregate_water_truncation_bound_q": aggregate_water_bound_q,
        "distributed_per_shard_sum_bound_q": distributed_sum_bound_q,
        "observation_jitter_total_q": jitter_total_q,
        "lease_rearrangement_total_q": lease_total_q,
        "safe_upper_bound_q": safe_upper_bound_q,
        "saturated": saturated,
        "derivation": "integer resource units are multiplied exactly by 10**p; floor of one aggregate ratio loses <1 q; summing n independent floors loses <n q",
    }


def budget_slack_assessment(
    *,
    remaining_units: int,
    demand_units: int,
    scale_power: int,
    global_error_bound_q: int,
) -> dict[str, Any]:
    scale = scale_for_power(scale_power)
    slack_units = int(remaining_units) - int(demand_units)
    slack_q, saturated = saturated_nonnegative(abs(slack_units) * scale)
    affected = slack_q <= max(0, int(global_error_bound_q)) or saturated
    hard_allowed = slack_units >= 0
    return {
        "remaining_units": int(remaining_units),
        "demand_units": int(demand_units),
        "slack_units": slack_units,
        "scale": scale,
        "slack_magnitude_q": slack_q,
        "global_error_bound_q": int(global_error_bound_q),
        "indistinguishable": affected,
        "hard_allowed": hard_allowed,
        "error_affected": affected and hard_allowed,
        "rule": "exact integer ledger remains the hard gate; error flag only marks the boundary",
    }
