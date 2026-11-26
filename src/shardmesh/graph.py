"""Stable dependency graph and invalidation propagation."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping

from .errors import spec_error


def build_dependents(deps: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    dependents: dict[str, list[str]] = {shard_id: [] for shard_id in deps}
    for shard_id, upstream_ids in deps.items():
        for upstream_id in upstream_ids:
            dependents.setdefault(upstream_id, []).append(shard_id)
    for values in dependents.values():
        values.sort()
    return dependents


def topological_order(deps: Mapping[str, Iterable[str]]) -> list[str]:
    """Kahn order with lexicographic tie-breaking.

    The result is deterministic and is the explicit stable topology order used
    by indistinguishable-boundary decisions.
    """
    ids = set(deps)
    indegree = {shard_id: 0 for shard_id in deps}
    forward: dict[str, list[str]] = {shard_id: [] for shard_id in deps}
    for shard_id, upstream_ids in deps.items():
        seen = set()
        for upstream_id in upstream_ids:
            if upstream_id not in ids:
                raise spec_error(
                    None,
                    "deps",
                    f"shard {shard_id!r} references missing upstream {upstream_id!r}",
                    category="shard",
                )
            if upstream_id in seen:
                continue
            seen.add(upstream_id)
            forward[upstream_id].append(shard_id)
            indegree[shard_id] += 1

    ready = sorted(shard_id for shard_id, degree in indegree.items() if degree == 0)
    queue = deque(ready)
    order: list[str] = []
    while queue:
        shard_id = queue.popleft()
        order.append(shard_id)
        newly_ready = []
        for downstream in forward[shard_id]:
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                newly_ready.append(downstream)
        if newly_ready:
            queue.extend(sorted(newly_ready))
    if len(order) != len(deps):
        cyclic = sorted(shard_id for shard_id, degree in indegree.items() if degree > 0)
        raise spec_error(
            None,
            "deps",
            f"dependency graph contains a cycle involving {cyclic}",
            category="dependency_graph",
            details={"cycle_candidates": cyclic},
        )
    return order


def transitive_downstream(
    direct_ids: Iterable[str],
    deps: Mapping[str, Iterable[str]],
) -> tuple[list[str], list[dict[str, str]]]:
    dependents = build_dependents(deps)
    topo_index = {shard_id: index for index, shard_id in enumerate(topological_order(deps))}
    invalid: set[str] = set()
    steps: list[dict[str, str]] = []
    queue = deque()
    for shard_id in sorted(direct_ids, key=lambda item: topo_index.get(item, 10**9)):
        if shard_id not in invalid:
            invalid.add(shard_id)
            queue.append(shard_id)
            steps.append({"shard_id": shard_id, "basis": "direct_parameter_reference"})
    while queue:
        current = queue.popleft()
        for child in dependents.get(current, []):
            if child not in invalid:
                invalid.add(child)
                queue.append(child)
                steps.append(
                    {
                        "shard_id": child,
                        "basis": f"upstream_{current}_invalidated",
                        "from_shard_id": current,
                    }
                )
    ordered = sorted(invalid, key=lambda shard_id: topo_index.get(shard_id, 10**9))
    steps.sort(key=lambda step: topo_index.get(step["shard_id"], 10**9))
    return ordered, steps


def all_upstreams(shard_id: str, deps: Mapping[str, Iterable[str]]) -> set[str]:
    result: set[str] = set()
    stack = list(deps.get(shard_id, []))
    while stack:
        upstream = stack.pop()
        if upstream in result:
            continue
        result.add(upstream)
        stack.extend(deps.get(upstream, []))
    return result
