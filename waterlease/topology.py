from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .errors import SpecError


ShardLike = Any


def shard_map(shards: Mapping[str, ShardLike] | Iterable[ShardLike]) -> dict[str, ShardLike]:
    if isinstance(shards, Mapping):
        return dict(shards)
    result: dict[str, ShardLike] = {}
    for shard in shards:
        result[shard.shard_id] = shard
    return result


def topological_order(
    shards: Mapping[str, ShardLike] | Iterable[ShardLike],
) -> list[str]:
    """Return a deterministic Kahn order.

    Zero-indegree ties are resolved by shard id, making the order stable across
    processes, submission orders and batch splits.
    """

    by_id = shard_map(shards)
    indegree = {shard_id: len(set(_upstreams(shard))) for shard_id, shard in by_id.items()}
    dependents: dict[str, list[str]] = defaultdict(list)
    for shard_id, shard in by_id.items():
        for upstream in _upstreams(shard):
            if upstream in by_id:
                dependents[upstream].append(shard_id)

    ready = [shard_id for shard_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        shard_id = heapq.heappop(ready)
        order.append(shard_id)
        for downstream in sorted(dependents.get(shard_id, ())):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                heapq.heappush(ready, downstream)

    if len(order) != len(by_id):
        remaining = sorted(shard_id for shard_id, degree in indegree.items() if degree > 0)
        cycle = _find_cycle(by_id, remaining)
        raise SpecError(
            "dependency graph contains a cycle",
            path="$.shards.upstreams",
            details={"remaining": remaining, "cycle": cycle},
        )
    return order


def _upstreams(shard: ShardLike) -> tuple[str, ...] | list[str]:
    value = getattr(shard, "upstreams", None)
    if value is None and isinstance(shard, Mapping):
        value = shard.get("upstreams", ())
    return tuple(value or ())


def _find_cycle(by_id: dict[str, ShardLike], candidates: list[str]) -> list[str]:
    candidate_set = set(candidates)
    visited: set[str] = set()
    for start in candidates:
        if start in visited:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        seen_on_path: set[str] = {start}
        path: list[str] = []
        while stack:
            node, index = stack[-1]
            if index == 0:
                path.append(node)
            ups = [item for item in _upstreams(by_id[node]) if item in candidate_set]
            if index >= len(ups):
                stack.pop()
                path.pop()
                seen_on_path.discard(node)
                continue
            upstream = ups[index]
            stack[-1] = (node, index + 1)
            if upstream in seen_on_path:
                cycle_start = path.index(upstream)
                return path[cycle_start:] + [upstream]
            if upstream not in visited:
                seen_on_path.add(upstream)
                stack.append((upstream, 0))
        visited.update(path)
    return candidates[:1]


def reverse_dependency_map(
    shards: Mapping[str, ShardLike] | Iterable[ShardLike],
) -> dict[str, tuple[str, ...]]:
    by_id = shard_map(shards)
    result: dict[str, set[str]] = defaultdict(set)
    for shard_id, shard in by_id.items():
        for upstream in _upstreams(shard):
            result[upstream].add(shard_id)
    return {key: tuple(sorted(value)) for key, value in result.items()}


def stable_invalidation_sort(
    invalid_ids: Iterable[str], order: Iterable[str]
) -> list[str]:
    rank = {shard_id: index for index, shard_id in enumerate(order)}
    return sorted(invalid_ids, key=lambda shard_id: (rank.get(shard_id, 10**12), shard_id))
