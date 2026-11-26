from __future__ import annotations

from typing import Any

from .aggregation import aggregate_samples, shard_watermark
from .canonical import short_digest
from .fixedpoint import contribution_report, error_vector, floor_divide, quantize
from .spec import SessionSpec, ShardSpec


def deterministic_lease_id(spec: SessionSpec, shard: ShardSpec) -> str:
    node = spec.node(shard.node_id)
    identity = {
        "subject": "quotalab-lease-v1",
        "session_scale_power": spec.scale_power,
        "shard_id": shard.shard_id,
        "node_id": shard.node_id,
        "capacity_version": node.capacity_version,
        "resource_units": shard.resource_units,
        "sample_rows": shard.sample_rows,
        "depends_on": list(shard.depends_on),
    }
    return "lease_" + short_digest(identity, 20)


def build_shard_accounting(
    spec: SessionSpec,
    shard_id: str,
    samples: list[list[int]],
) -> dict[str, Any]:
    shard = spec.shard(shard_id)
    summary = aggregate_samples(
        samples,
        q=spec.q,
        pressure_den=spec.accounting.pressure_den,
    )
    watermark = shard_watermark(
        shard.resource_units * spec.q,
        int(summary["fixed_point"]["pressure_num"]),
        spec.accounting.pressure_den,
    )
    local_bound = int(summary["fixed_point"]["mean_bound_q"]) + int(watermark["trunc_bound_q"])
    topo_rank = list(spec.topo_order).index(shard_id)
    return {
        "shard_id": shard_id,
        "node_id": shard.node_id,
        "topo_rank": topo_rank,
        "priority": shard.priority,
        "demand_units": shard.resource_units,
        "demand_q": shard.resource_units * spec.q,
        "summary": summary,
        "watermark_q": int(watermark["water_q"]),
        "watermark_calculation": watermark,
        "local_trunc_bound_q": local_bound,
        "stable_order_key": [topo_rank, shard_id],
    }


def _active_members_for_node(
    spec: SessionSpec,
    node_id: str,
    active_ids: set[str],
    candidate_id: str | None,
) -> set[str]:
    members = {sid for sid in active_ids if spec.shard(sid).node_id == node_id}
    if candidate_id and spec.shard(candidate_id).node_id == node_id:
        members.add(candidate_id)
    return members


def candidate_error_vector(
    spec: SessionSpec,
    accounting: dict[str, dict[str, Any]],
    active_ids: set[str],
    candidate_id: str,
) -> dict[str, Any]:
    relevant_ids = set(active_ids) | {candidate_id}
    shard_bound = sum(int(accounting[sid]["local_trunc_bound_q"]) for sid in relevant_ids)

    node_round_bound = 0
    node_projections: dict[str, dict[str, Any]] = {}
    for node in spec.nodes:
        member_ids = _active_members_for_node(spec, node.node_id, active_ids, candidate_id)
        if not member_ids:
            continue
        water_sum = sum(int(accounting[sid]["watermark_q"]) for sid in member_ids)
        denominator = max(1, node.concurrency_quota)
        division = floor_divide(water_sum, denominator)
        node_round_bound += division.bound_q
        node_shard_bound = sum(int(accounting[sid]["local_trunc_bound_q"]) for sid in member_ids)
        node_judgement_bound = (
            node_shard_bound
            + division.bound_q
            + spec.tolerances.node_watermark_q
            + spec.tolerances.shard_q
        )
        node_projections[node.node_id] = {
            "member_shard_ids": sorted(member_ids, key=lambda sid: accounting[sid]["stable_order_key"]),
            "water_sum_q": water_sum,
            "slot_denominator": denominator,
            "slot_water_q": division.quotient,
            "slot_remainder": division.residual,
            "round_bound_q": division.bound_q,
            "propagated_shard_bound_q": node_shard_bound,
            "node_judgement_bound_q": node_judgement_bound,
        }

    total_water = sum(int(accounting[sid]["watermark_q"]) for sid in relevant_ids)
    global_division = floor_divide(total_water, max(1, len(relevant_ids)))
    vector = error_vector(
        shard_bound_q=shard_bound,
        node_bound_q=node_round_bound,
        global_bound_q=global_division.bound_q,
        shard_tolerance_q=spec.tolerances.shard_q,
        node_tolerance_q=spec.tolerances.node_watermark_q,
        global_tolerance_q=spec.tolerances.global_budget_q,
    )
    candidate_node = spec.shard(candidate_id).node_id
    vector.update(
        {
            "candidate_shard_id": candidate_id,
            "candidate_watermark_q": int(accounting[candidate_id]["watermark_q"]),
            "candidate_demand_q": int(accounting[candidate_id]["demand_q"]),
            "relevant_shard_ids": sorted(relevant_ids, key=lambda sid: accounting[sid]["stable_order_key"]),
            "node_projections": node_projections,
            "candidate_node_judgement_bound_q": node_projections[candidate_node]["node_judgement_bound_q"],
            "global_average_water_q": global_division.quotient,
            "global_rounding_remainder": global_division.residual,
        }
    )
    return vector


def node_watermark_report(spec: SessionSpec, accounting: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"nodes": {}}
    node_round_total = 0
    for node in spec.nodes:
        shard_ids = [shard.shard_id for shard in spec.shards_for_node(node.node_id)]
        water_sum = sum(int(accounting[sid]["watermark_q"]) for sid in shard_ids)
        denominator = max(1, node.concurrency_quota)
        division = floor_divide(water_sum, denominator)
        shard_bound = sum(int(accounting[sid]["local_trunc_bound_q"]) for sid in shard_ids)
        judgement_bound = shard_bound + division.bound_q + spec.tolerances.node_watermark_q + spec.tolerances.shard_q
        node_round_total += division.bound_q
        report["nodes"][node.node_id] = {
            "shard_ids": shard_ids,
            "water_sum_q": water_sum,
            "slot_water_q": division.quotient,
            "slot_denominator": denominator,
            "remainder": division.residual,
            "shard_propagated_bound_q": shard_bound,
            "round_bound_q": division.bound_q,
            "watermark_judgement_bound_q": judgement_bound,
            "concurrency_quota": node.concurrency_quota,
            "lease_limit": node.lease_limit,
            "concurrency_quota_q": quantize(node.concurrency_quota, spec.q),
            "lease_limit_q": quantize(node.lease_limit, spec.q),
        }
    total_water = sum(int(item["watermark_q"]) for item in accounting.values())
    global_division = floor_divide(total_water, max(1, len(accounting)))
    report["global"] = {
        "water_sum_q": total_water,
        "average_water_q": global_division.quotient,
        "remainder": global_division.residual,
        "round_bound_q": global_division.bound_q,
        "budget_q": spec.budget_q,
        "aggregate_budget_error_bound_q": (
            sum(int(item["local_trunc_bound_q"]) for item in accounting.values())
            + node_round_total
            + global_division.bound_q
            + spec.tolerances.shard_q
            + spec.tolerances.node_watermark_q
            + spec.tolerances.global_budget_q
        ),
    }
    return report


def global_error_report(spec: SessionSpec, accounting: dict[str, dict[str, Any]]) -> dict[str, Any]:
    shard_contribution = sum(int(item["local_trunc_bound_q"]) for item in accounting.values()) + spec.tolerances.shard_q
    node_report = node_watermark_report(spec, accounting)
    node_contribution = (
        sum(int(node_info["round_bound_q"]) for node_info in node_report["nodes"].values())
        + spec.tolerances.node_watermark_q
    )
    global_contribution = int(node_report["global"]["round_bound_q"]) + spec.tolerances.global_budget_q
    contributions = contribution_report(
        shard_contribution,
        node_contribution,
        global_contribution,
        dominance_threshold_bp=spec.tolerances.dominance_threshold_bp,
    )
    return {
        **contributions,
        "node_watermarks": node_report,
        "propagation_order": ["shard_truncation", "node_watermark_accumulation", "global_budget_aggregation"],
    }
