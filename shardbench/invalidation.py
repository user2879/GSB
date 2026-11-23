"""Dependency-graph invalidation propagation and subset recomputation."""

from __future__ import annotations

import heapq


def topo_order(units) -> list:
    """Stable topological order (Kahn with lexicographic tie-breaking)."""
    indegree = {u: 0 for u in units}
    dependents = {u: [] for u in units}
    for u in units.values():
        for dep in u.depends_on:
            indegree[u.unit_id] += 1
            dependents[dep].append(u.unit_id)
    heap = [u for u, d in indegree.items() if d == 0]
    heapq.heapify(heap)
    order = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for child in sorted(dependents[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, child)
    return order


def invalidated_subset(config, changed_unit_ids):
    """Propagate invalidation along the dependency graph.

    Returns (ordered_invalidated, steps) where ordered_invalidated follows the
    stable topological order and steps records each propagation hop with its
    reason. `changed_unit_ids` are the roots (rule change / ring adjustment /
    upstream result change).
    """
    dependents = {u: [] for u in config.units}
    for u in config.units.values():
        for dep in u.depends_on:
            dependents[dep].append(u.unit_id)

    invalidated = set(changed_unit_ids)
    steps = []
    queue = list(sorted(changed_unit_ids))
    while queue:
        src = queue.pop(0)
        for dst in sorted(dependents.get(src, [])):
            if dst not in invalidated:
                invalidated.add(dst)
                steps.append(
                    {
                        "from": src,
                        "to": dst,
                        "reason": f"{dst} depends_on {src}; upstream result changed",
                    }
                )
                queue.append(dst)
    order = [u for u in topo_order(config.units) if u in invalidated]
    return order, steps
