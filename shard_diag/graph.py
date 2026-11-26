from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class Topology:
    order: tuple[str, ...]
    cycles: tuple[str, ...]
    missing: dict[str, tuple[str, ...]]

    @property
    def rank(self) -> dict[str, int]:
        return {shard_id: index for index, shard_id in enumerate(self.order)}


def missing_upstream_ids(shard: dict[str, Any], known_ids: set[str]) -> list[str]:
    return sorted(up for up in shard["upstream"] if up not in known_ids)


def stable_topology(shards: Iterable[dict[str, Any]]) -> Topology:
    shard_list = list(shards)
    by_id = {shard["shard_id"]: shard for shard in shard_list}
    known = set(by_id)
    indegree: dict[str, int] = {shard_id: 0 for shard_id in sorted(by_id)}
    children: dict[str, list[str]] = {shard_id: [] for shard_id in sorted(by_id)}
    missing: dict[str, tuple[str, ...]] = {}

    for shard_id in sorted(by_id):
        shard = by_id[shard_id]
        missing_ids = missing_upstream_ids(shard, known)
        if missing_ids:
            missing[shard_id] = tuple(missing_ids)
        for upstream in shard["upstream"]:
            if upstream in known:
                indegree[shard_id] += 1
                children[upstream].append(shard_id)

    ready: list[str] = [shard_id for shard_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        shard_id = heapq.heappop(ready)
        order.append(shard_id)
        for child in sorted(children[shard_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)

    cyclic = tuple(sorted(shard_id for shard_id, degree in indegree.items() if degree > 0))
    if cyclic:
        order.extend(cyclic)
    return Topology(order=tuple(order), cycles=cyclic, missing=missing)


def descendant_set(
    changed_ids: Iterable[str],
    shards: Iterable[dict[str, Any]],
    *,
    include_start: bool = True,
) -> dict[str, list[str]]:
    """Return affected id -> deterministic propagation path/reason list."""

    shard_list = list(shards)
    by_id = {shard["shard_id"]: shard for shard in shard_list}
    children: dict[str, list[str]] = {shard_id: [] for shard_id in sorted(by_id)}
    for shard in shard_list:
        for upstream in shard["upstream"]:
            if upstream in by_id:
                children[upstream].append(shard["shard_id"])

    affected_reasons: dict[str, list[str]] = {}
    queue: list[tuple[str, list[str]]] = []
    for changed in sorted(set(changed_ids)):
        if changed not in by_id:
            continue
        if include_start:
            affected_reasons.setdefault(changed, [])
            reason = "direct_change"
            if reason not in affected_reasons[changed]:
                affected_reasons[changed].append(reason)
        queue.append((changed, [changed]))

    seen_edges: set[tuple[str, str]] = set()
    while queue:
        parent, path = queue.pop(0)
        for child in sorted(children.get(parent, [])):
            edge = (parent, child)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            reason = f"upstream:{parent}"
            reasons = affected_reasons.setdefault(child, [])
            if reason not in reasons:
                reasons.append(reason)
            queue.append((child, path + [child]))

    return dict(sorted(affected_reasons.items()))


def shards_for_nodes(node_ids: Iterable[str], shards: Iterable[dict[str, Any]]) -> set[str]:
    targets = set(node_ids)
    return {shard["shard_id"] for shard in shards if shard["node_id"] in targets}


def ordered_affected(
    affected: dict[str, Any] | Iterable[str],
    shards: Iterable[dict[str, Any]],
) -> list[str]:
    ids = list(affected.keys()) if isinstance(affected, dict) else list(affected)
    rank = stable_topology(shards).rank
    return sorted(ids, key=lambda shard_id: (rank.get(shard_id, 10**9), shard_id))
