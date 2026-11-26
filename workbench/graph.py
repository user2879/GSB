from __future__ import annotations

import heapq
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence


def stable_topological_order(
    shard_ids: Sequence[str],
    dependencies: Mapping[str, Iterable[str]],
) -> list[str]:
    """Kahn order with shard id as the deterministic ready-set tie break."""

    ids = list(shard_ids)
    known = set(ids)
    indegree: dict[str, int] = {shard_id: 0 for shard_id in ids}
    children: dict[str, list[str]] = defaultdict(list)
    for shard_id in ids:
        for upstream in dependencies.get(shard_id, ()):  # type: ignore[arg-type]
            if upstream in known and upstream != shard_id:
                indegree[shard_id] += 1
                children[upstream].append(shard_id)

    ready = [shard_id for shard_id in ids if indegree[shard_id] == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        shard_id = heapq.heappop(ready)
        order.append(shard_id)
        for child in sorted(children[shard_id]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, child)

    if len(order) != len(ids):
        cyclic = sorted(shard_id for shard_id, degree in indegree.items() if degree > 0)
        raise ValueError(f"dependency cycle involves: {','.join(cyclic)}")
    return order


def descendant_map(dependencies: Mapping[str, Iterable[str]]) -> dict[str, set[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    for shard_id, ups in dependencies.items():
        for upstream in ups:
            children[upstream].add(shard_id)

    result: dict[str, set[str]] = {}

    def collect(node: str) -> set[str]:
        if node in result:
            return result[node]
        seen: set[str] = set()
        stack = list(children.get(node, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(children.get(current, ()))
        result[node] = seen
        return seen

    for node in dependencies:
        collect(node)
    return result


def transitive_invalid(seed: Iterable[str], dependencies: Mapping[str, Iterable[str]]) -> set[str]:
    descendants = descendant_map(dependencies)
    invalid: set[str] = set()
    stack = list(seed)
    while stack:
        current = stack.pop()
        if current in invalid:
            continue
        invalid.add(current)
        stack.extend(descendants.get(current, ()))
    return invalid
