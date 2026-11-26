from __future__ import annotations

from typing import Any

from .aggregate import aggregate_shard
from .fixedpoint import attribute_decision, global_budget_record, node_water_record, shard_error_record
from .spec import CapacityVersion, SpecBundle, stable_topological_order


def canonical_water_and_error(
    shard_id: str,
    bundle: SpecBundle,
    capacity: CapacityVersion | None = None,
    scale_factor: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    capacity = capacity or bundle.active_capacity()
    if scale_factor is None:
        scale_factor = bundle.active_fixed().scale_factor
    order = stable_topological_order(bundle.shards)
    position = order.index(shard_id)
    prefix_ids = set(order[: position + 1])
    shard = bundle.shards_by_id[shard_id]
    global_used = sum(bundle.shards_by_id[item].resource_units for item in order[: position + 1])
    node_order = [item for item in order if bundle.shards_by_id[item].node_id == shard.node_id]
    node_rank = node_order.index(shard_id) + 1
    node_units = sum(bundle.shards_by_id[item].resource_units for item in node_order[:node_rank])
    limit = capacity.node_limits[shard.node_id]

    shard_error = shard_error_record(shard_id, shard.resource_units, capacity.budget_units, scale_factor)
    node_water = node_water_record(
        shard.node_id,
        node_rank,
        max(1, limit.concurrency_quota),
        node_rank,
        max(1, limit.lease_limit),
        scale_factor,
        candidate_units=0,
    )
    node_water["dimension_note"] = "used values are stable-order node rank/count for reproducible field comparison"
    global_water = global_budget_record(global_used, capacity.budget_units, len(order), scale_factor)
    attribution = attribute_decision(
        candidate_ids=[shard_id],
        candidate_levels_q=[int(node_water["projected_resource_level_q"])],
        slack_q=int(global_water["remaining_q"]),
        scale=scale_factor,
        capacity_changed_dimensions=len(capacity.node_limits),
        dependency_propagation_hops=len(shard.dependencies),
        renewal_count=0,
    )
    water = {
        "topology_position": position,
        "stable_order_prefix": list(order[: position + 1]),
        "global": global_water,
        "node": node_water,
        "node_stable_rank": node_rank,
        "node_cumulative_units": node_units,
    }
    error = {
        "shard": shard_error,
        "node_comparison_error_bound_q": node_water["comparison_error_bound_q"],
        "global_budget_aggregation_error_upper_bound_q": global_water["budget_aggregation_error_upper_bound_q"],
        "attribution": attribution,
    }
    semantic = {
        "topology_order": order,
        "global_used_prefix_units": global_used,
        "node_stable_rank": node_rank,
        "capacity_version": capacity.version,
        "scale_factor": scale_factor,
    }
    return water, error, semantic


def compute_semantic_outcome(
    bundle: SpecBundle,
    shard_id: str,
    *,
    capacity_version: int | None = None,
    scale_factor: int | None = None,
) -> dict[str, Any]:
    capacity = bundle.capacity_by_version[capacity_version] if capacity_version is not None else bundle.active_capacity()
    scale = scale_factor if scale_factor is not None else bundle.active_fixed().scale_factor
    shard = bundle.shards_by_id[shard_id]
    aggregate = aggregate_shard(
        shard,
        bundle.samples_by_id,
        capacity_version=capacity.version,
        scale_factor=scale,
        node_fixed_scale=bundle.nodes_by_id[shard.node_id].fixed_scale,
    )
    water, error, semantic_extra = canonical_water_and_error(shard_id, bundle, capacity, scale)
    semantic = {
        **semantic_extra,
        "aggregate_groups": aggregate["groups"],
        "sample_snapshot": aggregate["sample_snapshot"],
    }
    return {
        "shard_id": shard_id,
        "capacity_version": capacity.version,
        "node_id": shard.node_id,
        "resource_units": shard.resource_units,
        "summary": aggregate["summary"],
        "summary_prefix": aggregate["summary_prefix"],
        "sample_snapshot": aggregate["sample_snapshot"],
        "water": water,
        "error": error,
        "semantic": semantic,
    }
