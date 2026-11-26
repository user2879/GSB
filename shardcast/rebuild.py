"""Full-from-scratch rebuild used as the reference path for parity checks."""

from __future__ import annotations

from .propagation import compare_field_by_field, plan_invalidation
from .sample import aggregate_rows, canonical_json, generate_sample
from .spec import Spec


def compute_summary(spec: Spec, node_id: str, shard_id: str, capacity_version: int) -> dict:
    """Pure reference aggregation for one shard without any worker process."""
    node = spec.node(node_id)
    shard = spec.shard(shard_id)
    data = generate_sample(node.seed, shard_id, shard.sample_rows)
    partial = aggregate_rows(data)
    return {
        "node_id": node_id,
        "shard_id": shard_id,
        "sample_rows": shard.sample_rows,
        "capacity_version": capacity_version,
        "participating_rows": partial["participating_rows"],
        "groups": partial["groups"],
    }


def full_rebuild_state(
    spec: Spec,
    assignments: dict[str, str],
    capacity_version: int,
) -> dict:
    """Rebuild every summary, ownership and water mark from an empty view."""
    summaries: dict[str, dict] = {}
    ownership: dict[str, dict] = {}
    water_marks: dict[str, dict] = {}
    for shard_id in spec.shard_ids:
        node_id = assignments[shard_id]
        summary = compute_summary(spec, node_id, shard_id, capacity_version)
        summaries[shard_id] = summary
        ownership[shard_id] = {
            "node_id": node_id,
            "capacity_version": capacity_version,
        }
        mark = water_marks.setdefault(
            node_id,
            {
                "node_id": node_id,
                "capacity_version": capacity_version,
                "high_water_ms": 0,
                "shard_count": 0,
                "resource_total": 0,
            },
        )
        mark["shard_count"] += 1
        mark["resource_total"] += spec.shard(shard_id).resource_units
    return {
        "summaries": summaries,
        "ownership": ownership,
        "water_marks": water_marks,
    }


def incremental_state(
    spec: Spec,
    assignments: dict[str, str],
    capacity_version: int,
    invalidated: list[str],
    preserved: dict,
) -> dict:
    """Recompute only invalidated shards; untouched bytes come from ``preserved``."""
    invalid_set = set(invalidated)
    summaries = dict(preserved.get("summaries", {}))
    ownership = dict(preserved.get("ownership", {}))
    water_marks: dict[str, dict] = {}
    for shard_id in invalidated:
        node_id = assignments[shard_id]
        summaries[shard_id] = compute_summary(
            spec, node_id, shard_id, capacity_version
        )
        ownership[shard_id] = {
            "node_id": node_id,
            "capacity_version": capacity_version,
        }
        mark = water_marks.setdefault(
            node_id,
            {
                "node_id": node_id,
                "capacity_version": capacity_version,
                "high_water_ms": 0,
                "shard_count": 0,
                "resource_total": 0,
            },
        )
        mark["shard_count"] = mark["shard_count"] + 1
        mark["resource_total"] = mark["resource_total"] + spec.shard(shard_id).resource_units
    for node_id, mark in preserved.get("water_marks", {}).items():
        if node_id not in water_marks:
            water_marks[node_id] = dict(mark)
    return {
        "summaries": summaries,
        "ownership": ownership,
        "water_marks": water_marks,
    }


def verify_incremental_parity(
    old_spec: Spec,
    new_spec: Spec,
    assignments: dict[str, str],
    capacity_version: int,
    preserved: dict,
) -> dict:
    steps = plan_invalidation(old_spec, new_spec)
    invalidated = [step["shard_id"] for step in steps]
    incremental = incremental_state(
        new_spec, assignments, capacity_version, invalidated, preserved
    )
    full = full_rebuild_state(new_spec, assignments, capacity_version)
    compare_field_by_field(
        {"summaries": incremental["summaries"], "ownership": incremental["ownership"]},
        {"summaries": full["summaries"], "ownership": full["ownership"]},
        context="rebuild",
    )
    invalid_nodes = {assignments[shard_id] for shard_id in invalidated}
    compare_field_by_field(
        {
            "water_marks": {
                node_id: incremental["water_marks"][node_id]
                for node_id in sorted(invalid_nodes)
            }
        },
        {
            "water_marks": {
                node_id: {
                    "node_id": node_id,
                    "capacity_version": capacity_version,
                    "high_water_ms": 0,
                    "shard_count": sum(
                        1 for shard_id in invalidated if assignments[shard_id] == node_id
                    ),
                    "resource_total": sum(
                        new_spec.shard(shard_id).resource_units
                        for shard_id in invalidated
                        if assignments[shard_id] == node_id
                    ),
                }
                for node_id in sorted(invalid_nodes)
            }
        },
        context="rebuild",
    )
    unaffected = sorted(set(new_spec.shard_ids) - set(invalidated))
    for shard_id in unaffected:
        before = canonical_json(preserved["summaries"][shard_id])
        after = canonical_json(incremental["summaries"][shard_id])
        if before != after:
            from .errors import DivergeError

            raise DivergeError(
                "unaffected shard summary changed during incremental recompute",
                path=f"summaries.{shard_id}",
            )
    return {
        "invalidated": steps,
        "unaffected": unaffected,
        "incremental": incremental,
        "full_rebuild": full,
    }
