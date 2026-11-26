from __future__ import annotations

import hashlib
from typing import Any

from .canonical import content_digest
from .graph import transitive_invalid
from .spec import Spec


def rendezvous_node_id(node_ids: list[str] | tuple[str, ...], shard_id: str) -> str:
    if not node_ids:
        raise ValueError("at least one node is required")

    def score(node_id: str) -> int:
        return int.from_bytes(
            hashlib.sha256(f"{node_id}\x1f{shard_id}".encode("utf-8")).digest()[:8],
            "big",
        )

    return min(node_ids, key=lambda node_id: (-score(node_id), node_id))


def rendezvous_rank(node_ids: list[str] | tuple[str, ...], shard_id: str) -> list[str]:
    def score(node_id: str) -> int:
        return int.from_bytes(
            hashlib.sha256(f"{node_id}\x1f{shard_id}".encode("utf-8")).digest()[:8],
            "big",
        )

    return sorted(node_ids, key=lambda node_id: (-score(node_id), node_id))


def attribution_from_spec_or_history(
    spec: Spec, history_node_by_shard: dict[str, str] | None = None
) -> dict[str, str]:
    history = history_node_by_shard or {}
    node_ids = [node["node_id"] for node in spec.nodes]
    result: dict[str, str] = {}
    for shard in spec.shards:
        shard_id = shard["shard_id"]
        if shard_id in history:
            result[shard_id] = history[shard_id]
        elif shard.get("node_id"):
            result[shard_id] = str(shard["node_id"])
        else:
            result[shard_id] = rendezvous_node_id(tuple(node_ids), shard_id)
    return result


def _topo_key(spec: Spec, shard_id: str) -> tuple[int, str]:
    try:
        index = spec.topo_order.index(shard_id)
    except ValueError:
        index = 10**9
    return index, shard_id


def plan_evolution(
    old: Spec,
    new: Spec,
    *,
    old_attribution: dict[str, str] | None = None,
) -> dict[str, Any]:
    old_nodes = old.node_by_id
    new_nodes = new.node_by_id
    old_shards = old.shard_by_id
    new_shards = new.shard_by_id
    attribution = attribution_from_spec_or_history(old, old_attribution)

    aspects: dict[str, set[str]] = {}
    direct_basis: dict[str, list[dict[str, str]]] = {}

    def mark(shard_id: str, aspect: str, basis: str, field: str) -> None:
        aspects.setdefault(shard_id, set()).add(aspect)
        direct_basis.setdefault(shard_id, []).append(
            {"stage": "direct", "aspect": aspect, "basis": basis, "field": field}
        )

    if old.scale_exponent != new.scale_exponent:
        for shard_id in new_shards:
            mark(
                shard_id,
                "water",
                f"session.scale_exponent {old.scale_exponent}->{new.scale_exponent}",
                "session.scale_exponent",
            )
    if old.raw.get("global_budget_units") != new.raw.get("global_budget_units"):
        for shard_id in new_shards:
            mark(
                shard_id,
                "water",
                "global_budget_units changed",
                "global_budget_units",
            )

    for node_id in sorted(set(old_nodes) | set(new_nodes)):
        old_node = old_nodes.get(node_id)
        new_node = new_nodes.get(node_id)
        affected_shards = sorted(
            (
                shard_id
                for shard_id, assigned_node in attribution.items()
                if assigned_node == node_id and shard_id in new_shards
            ),
            key=lambda sid: _topo_key(new, sid),
        )
        if old_node is None:
            continue
        if new_node is None:
            for shard_id in affected_shards:
                mark(shard_id, "placement", f"node {node_id} removed", "nodes")
            continue
        field_aspects = {
            "capacity_version": "summary",
            "seed": "summary",
            "fixed_point_exponent": "summary",
            "port": "placement",
            "host": "placement",
            "concurrency_quota": "water",
            "lease_limit": "water",
        }
        for field, aspect in field_aspects.items():
            if old_node.get(field) != new_node.get(field):
                for shard_id in affected_shards:
                    mark(
                        shard_id,
                        aspect,
                        f"node {node_id}.{field} {old_node.get(field)!r}->{new_node.get(field)!r}",
                        f"nodes.{node_id}.{field}",
                    )

    for shard_id in sorted(set(old_shards) & set(new_shards)):
        old_shard = dict(old_shards[shard_id])
        new_shard = dict(new_shards[shard_id])
        if content_digest(old_shard) == content_digest(new_shard):
            continue
        for field in sorted(set(old_shard) | set(new_shard)):
            if old_shard.get(field) == new_shard.get(field):
                continue
            aspect = "scheduling"
            if field in {"sample_rows"}:
                aspect = "summary"
            elif field in {"resource_units"}:
                aspect = "water"
            elif field in {"upstreams"}:
                aspect = "dependency"
            elif field in {"node_id"}:
                aspect = "placement"
            mark(
                shard_id,
                aspect,
                f"shard {shard_id}.{field} changed",
                f"shards.{shard_id}.{field}",
            )

    for shard_id in set(new_shards) - set(old_shards):
        mark(shard_id, "summary", "shard added", f"shards.{shard_id}")

    dependencies = new.dependencies
    seed_ids = set(direct_basis)
    invalid = transitive_invalid(seed_ids, dependencies)
    invalid &= set(new_shards)

    propagation_steps: list[dict[str, Any]] = []
    for shard_id in new.topo_order:
        if shard_id in direct_basis:
            propagation_steps.append(
                {
                    "shard_id": shard_id,
                    "stage": "direct",
                    "reasons": direct_basis[shard_id],
                }
            )
            continue
        if shard_id not in invalid:
            continue
        bad_upstreams = [up for up in dependencies[shard_id] if up in invalid]
        propagation_steps.append(
            {
                "shard_id": shard_id,
                "stage": "propagated",
                "invalid_upstreams": sorted(bad_upstreams, key=lambda sid: _topo_key(new, sid)),
                "basis": "all upstream outcomes must be complete and digest-available",
            }
        )
        aspects.setdefault(shard_id, set()).add("dependency")

    ordered_invalid = [sid for sid in new.topo_order if sid in invalid]
    return {
        "old_spec_digest": old.digest,
        "new_spec_digest": new.digest,
        "invalid_shards": ordered_invalid,
        "recompute_order": ordered_invalid,
        "unchanged_shards": [sid for sid in new.topo_order if sid not in invalid],
        "aspects": {sid: sorted(aspects.get(sid, {"dependency"})) for sid in ordered_invalid},
        "propagation_steps": propagation_steps,
        "attribution": attribution,
    }
