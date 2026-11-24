"""Stable invalidation closure and propagation evidence."""
from __future__ import annotations

from typing import Mapping, Sequence

from .spec import Spec


def propagate_invalidation(
    spec: Spec,
    direct_reasons: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    """Return direct and transitive invalidations in stable topological order."""
    children: dict[str, list[str]] = {shard_id: [] for shard_id in spec.shards}
    for shard in spec.shards.values():
        for dep in shard.dependencies:
            children[dep].append(shard.id)
    for shard_id in children:
        children[shard_id].sort()

    active: set[str] = set(direct_reasons)
    reasons: dict[str, str] = {}
    direct_set: set[str] = set(direct_reasons)
    for shard_id in spec.topo_order:
        if shard_id in direct_reasons:
            reasons[shard_id] = ";".join(sorted(direct_reasons[shard_id]))
        if shard_id in active:
            for child in children[shard_id]:
                if child not in active:
                    active.add(child)
                    reasons[child] = f"dependency:{shard_id}->{child}"
                elif child not in direct_set and child not in reasons:
                    reasons[child] = f"dependency:{shard_id}->{child}"
    entries: list[dict[str, object]] = []
    for rank, shard_id in enumerate(spec.topo_order):
        if shard_id in active:
            entries.append(
                {
                    "shard_id": shard_id,
                    "direct": shard_id in direct_set,
                    "reason": reasons.get(shard_id, "dependency:unknown"),
                    "topo_rank": rank,
                }
            )
    return entries
