from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable


Q_MAX = (1 << 63) - 1


def saturate_q(value: int) -> tuple[int, bool]:
    if value > Q_MAX:
        return Q_MAX, True
    if value < -Q_MAX:
        return -Q_MAX, True
    return int(value), False


@dataclass(frozen=True)
class FixedQuantum:
    q: int
    scale: int
    saturated: bool
    numerator: int
    denominator: int

    @property
    def truncation_error_q(self) -> Fraction:
        """Exact floor remainder, in fixed-point q units."""

        scaled = Fraction(self.numerator * self.scale, self.denominator)
        return scaled - Fraction(self.q)

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "scale": self.scale,
            "saturated": self.saturated,
            "truncation_error_q_num": self.truncation_error_q.numerator,
            "truncation_error_q_den": self.truncation_error_q.denominator,
        }


def floor_scaled_fraction(numerator: int, denominator: int, scale: int) -> FixedQuantum:
    """Quantize numerator/denominator as floor(x * scale).

    Non-negative watermarks are truncated towards zero/floor; signed values
    use Python floor division (towards negative infinity). Values outside
    int64 saturation bounds clamp to +/-Q_MAX and set saturated=true.
    """

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if scale <= 0:
        raise ValueError("scale must be positive")
    full = numerator * scale
    q = full // denominator
    q, saturated = saturate_q(q)
    return FixedQuantum(q=q, scale=scale, saturated=saturated, numerator=numerator, denominator=denominator)


def quantize_units(raw_units: int, scale: int) -> FixedQuantum:
    if raw_units < 0:
        raise ValueError("raw_units must be non-negative")
    q, saturated = saturate_q(raw_units * scale)
    return FixedQuantum(q=q, scale=scale, saturated=saturated, numerator=raw_units, denominator=1)


def rescale_q(q: int, from_scale: int, to_scale: int) -> FixedQuantum:
    """Rescale a fixed-point integer using integer multiply/divide only."""

    if from_scale <= 0 or to_scale <= 0:
        raise ValueError("scales must be positive")
    value = (q * int(to_scale)) // int(from_scale)
    value, saturated = saturate_q(value)
    return FixedQuantum(q=value, scale=to_scale, saturated=saturated, numerator=q, denominator=1)


@dataclass(frozen=True)
class ErrorBound:
    scale: int
    term_count: int
    truncation_q: int
    accumulation_q: int
    comparison_q: int
    total_q: int

    def as_resource_units(self, denominator: int) -> Fraction:
        """Convert normalized-ratio q units to resource-unit bounds.

        Ratio q = floor(scale * raw / denominator), hence one q unit maps to
        denominator / scale raw units.
        """

        return Fraction(self.total_q * denominator, self.scale)

    def to_dict(self, denominator: int | None = None) -> dict[str, Any]:
        payload = {
            "scale": self.scale,
            "term_count": self.term_count,
            "segments_q": {
                "quantization_truncation": self.truncation_q,
                "per_node_accumulation": self.accumulation_q,
                "threshold_comparison": self.comparison_q,
            },
            "total_bound_q": self.total_q,
        }
        if denominator is not None:
            bound = self.as_resource_units(denominator)
            payload["resource_unit_bound_num"] = bound.numerator
            payload["resource_unit_bound_den"] = bound.denominator
            payload["resource_unit_bound_float"] = float(bound)
        return payload


def error_bound_for_terms(term_count: int, scale: int) -> ErrorBound:
    """Conservative three-segment bound for n independently quantized terms.

    first-term truncation = 1 ulp; remaining n-1 truncations are exposed by
    per-node accumulation; final threshold comparison contributes one ulp.
    """

    if term_count < 1:
        term_count = 1
    truncation = 1
    accumulation = max(0, term_count - 1)
    comparison = 1
    return ErrorBound(
        scale=scale,
        term_count=term_count,
        truncation_q=truncation,
        accumulation_q=accumulation,
        comparison_q=comparison,
        total_q=truncation + accumulation + comparison,
    )


def node_water_error_bound(active_term_count: int, candidate_count: int, scale: int) -> ErrorBound:
    return error_bound_for_terms(max(1, active_term_count + candidate_count), scale)


def global_water_error_bound(shard_count: int, scale: int) -> ErrorBound:
    return error_bound_for_terms(max(1, shard_count), scale)


def single_shard_error_attribution(
    resource_units: int,
    budget_units: int,
    scale: int,
) -> dict[str, Any]:
    quantum = floor_scaled_fraction(resource_units, budget_units, scale)
    worst_q = Fraction(1)
    actual_q = quantum.truncation_error_q
    actual_units = actual_q * budget_units / scale
    return {
        "resource_units": resource_units,
        "budget_units": budget_units,
        "scale": scale,
        "water_q": quantum.q,
        "saturated": quantum.saturated,
        "actual_truncation_error_q_num": actual_q.numerator,
        "actual_truncation_error_q_den": actual_q.denominator,
        "worst_case_truncation_error_q": 1,
        "actual_error_resource_units_num": actual_units.numerator,
        "actual_error_resource_units_den": actual_units.denominator,
        "worst_case_error_resource_units": Fraction(budget_units, scale).__float__(),
    }


def budget_boundary_view(
    *,
    used_units: int,
    demand_units: int,
    budget_units: int,
    scale: int,
    active_term_count: int = 0,
) -> dict[str, Any]:
    remaining = budget_units - used_units
    remaining_q = floor_scaled_fraction(max(0, remaining), budget_units, scale).q
    demand_q = floor_scaled_fraction(demand_units, budget_units, scale).q
    delta_q = remaining_q - demand_q
    bound = node_water_error_bound(active_term_count, 1, scale)
    exact_feasible = used_units + demand_units <= budget_units
    affected = abs(delta_q) <= bound.total_q
    return {
        "used_units": used_units,
        "demand_units": demand_units,
        "budget_units": budget_units,
        "remaining_units": remaining,
        "exact_feasible": exact_feasible,
        "remaining_water_q": remaining_q,
        "demand_water_q": demand_q,
        "delta_q": delta_q,
        "error_bound": bound.to_dict(budget_units),
        "error_affected": affected,
        "quantized_verdict": "admit" if delta_q >= 0 else "reject",
    }


def candidate_water(
    resource_units: int,
    budget_units: int,
    scale: int,
) -> int:
    return floor_scaled_fraction(resource_units, budget_units, scale).q


def adjudicate_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    budget_units: int,
    scale: int,
) -> dict[str, Any]:
    """Pick the first candidate in stable topology/id order.

    Candidates require shard_id and topo_index. resource_units drives the
    quantized water comparison. If the winner/runner-up water delta is within
    the synthetic bound, the decision is marked error-affected.
    """

    ordered = sorted(
        list(candidates),
        key=lambda item: (int(item.get("topo_index", 10**9)), str(item["shard_id"])),
    )
    if not ordered:
        return {"winner": None, "ordered": [], "error_affected": False}

    enriched: list[dict[str, Any]] = []
    for candidate in ordered:
        water_q = int(candidate.get("water_q") if candidate.get("water_q") is not None else candidate_water(
            int(candidate["resource_units"]), budget_units, scale
        ))
        enriched.append({**candidate, "water_q": water_q})

    winner = enriched[0]
    bound = error_bound_for_terms(2, scale)
    ambiguous_with: list[str] = []
    for runner_up in enriched[1:]:
        if abs(winner["water_q"] - runner_up["water_q"]) <= bound.total_q:
            ambiguous_with.append(runner_up["shard_id"])

    return {
        "winner": winner["shard_id"],
        "winner_node_id": winner.get("node_id"),
        "ordered": enriched,
        "tie_break_rule": "topo_index_then_shard_id",
        "error_bound_q": bound.total_q,
        "error_affected": bool(ambiguous_with),
        "ambiguous_with": ambiguous_with,
    }


def saturation_audit(values: Iterable[int], scale: int) -> dict[str, Any]:
    saturated: list[int] = []
    total = 0
    for value in values:
        quantum = quantize_units(value, scale)
        total += quantum.q
        if quantum.saturated:
            saturated.append(value)
    total_q, total_saturated = saturate_q(total)
    return {
        "total_q": total_q,
        "scale": scale,
        "saturated_values": saturated,
        "aggregate_saturated": total_saturated,
        "rule": "integer floor/truncation downward; int64 saturation at +/- (2^63-1)",
    }


def unified_water_report(
    spec: dict[str, Any],
    *,
    used_units: int = 0,
    lease_used: dict[str, int] | None = None,
    shard_used: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Quantize resource, lease caps, concurrency caps and current usage."""

    scale = int(spec["scale_factor"])
    lease_used = lease_used or {}
    shard_used = shard_used or {}
    budget = int(spec["budget_units"])
    used_units = int(used_units)

    nodes: dict[str, Any] = {}
    for node in spec["nodes"]:
        node_id = node["node_id"]
        lease_limit = int(node["lease_limit"])
        concurrency = int(node["concurrency"])
        nodes[node_id] = {
            "lease_limit_raw": lease_limit,
            "lease_limit_q": quantize_units(lease_limit, scale).q,
            "lease_used_raw": int(lease_used.get(node_id, 0)),
            "lease_used_q": quantize_units(int(lease_used.get(node_id, 0)), scale).q,
            "concurrency_limit_raw": concurrency,
            "concurrency_limit_q": quantize_units(concurrency, scale).q,
            "active_shards_raw": int(shard_used.get(node_id, 0)),
            "active_shards_q": quantize_units(int(shard_used.get(node_id, 0)), scale).q,
            "lease_headroom_q": quantize_units(
                max(0, lease_limit - int(lease_used.get(node_id, 0))), scale
            ).q,
            "concurrency_headroom_q": quantize_units(
                max(0, concurrency - int(shard_used.get(node_id, 0))), scale
            ).q,
        }

    shard_water = {
        shard["shard_id"]: quantize_units(int(shard["resource_units"]), scale).q
        for shard in spec["shards"]
    }
    shard_count = len(spec["shards"])
    return {
        "scale_factor": scale,
        "truncation_rule": "non-negative values floor downward; integer raw counts are exact after multiplication",
        "saturation_rule": "clamp to +/- (2^63-1) and set saturated=true",
        "global_budget_raw": budget,
        "global_budget_q": quantize_units(budget, scale).q,
        "used_units_raw": used_units,
        "used_units_q": quantize_units(used_units, scale).q,
        "remaining_budget_q": quantize_units(max(0, budget - used_units), scale).q,
        "shard_requirement_q": shard_water,
        "nodes": nodes,
        "global_error_bound": global_water_error_bound(shard_count, scale).to_dict(budget),
    }
