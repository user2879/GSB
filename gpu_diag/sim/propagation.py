"""Dependency failure/invalidation propagation over shard and task graphs.

Propagation is order-independent: the affected set is computed as a fixed point
and then emitted in a stable topological order together with the edge that first
reached each node, regardless of the iteration order of the input seeds.
"""
from __future__ import annotations

from collections import defaultdict, deque

from ..models import Declaration, ShardState


def _stable_topological_order(
        nodes: set[str],
        edges: dict[str, tuple[str, ...]]) -> list[str]:
    indegree: dict[str, int] = {node: 0 for node in nodes}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dep in edges.get(node, ()):
            if dep in nodes:
                adjacency[dep].append(node)
                indegree[node] += 1
    ready = deque(sorted(node for node, deg in indegree.items() if deg == 0))
    order: list[str] = []
    while ready:
        node = ready.popleft()
        order.append(node)
        for nxt in sorted(adjacency[node]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return order


def propagate_shard_invalidation(decl: Declaration,
                                 seeds: list[str]) -> tuple[list[str], list[dict]]:
    """Return (stable ordered affected shard ids, step-by-step rationale)."""
    shards = decl.shards
    dependents: dict[str, list[str]] = defaultdict(list)
    for sid, shard in shards.items():
        for dep in shard.deps:
            dependents[dep].append(sid)

    affected: set[str] = set(seeds)
    rationale: list[dict] = []
    queue: deque[str] = deque(sorted(seeds))
    while queue:
        current = queue.popleft()
        for nxt in sorted(dependents.get(current, ())):
            if nxt not in affected:
                affected.add(nxt)
                rationale.append({
                    "source_shard": current, "affected_shard": nxt,
                    "reason": "upstream shard outcome changed "
                              "(failed/preempted/reset)",
                })
                queue.append(nxt)

    edges = {sid: shard.deps for sid, shard in shards.items()}
    ordered = _stable_topological_order(affected, edges)
    seed_set = set(seeds)
    for sid in ordered:
        if sid not in seed_set:
            shards[sid].state = ShardState.AFFECTED
    return ordered, rationale


def derive_task_dependency_graph(decl: Declaration
                                 ) -> dict[str, tuple[str, ...]]:
    """Task edges implied by explicit deps plus cross-task shard references."""
    graph: dict[str, set[str]] = {tid: set(task.deps)
                                  for tid, task in decl.tasks.items()}
    for shard in decl.shards.values():
        for ref in shard.deps:
            upstream_task = ref.split("#", 1)[0]
            if upstream_task != shard.task_id and upstream_task in graph:
                graph[shard.task_id].add(upstream_task)
    return {tid: tuple(sorted(deps)) for tid, deps in graph.items()}


def propagate_task_invalidation(decl: Declaration,
                                seed_tasks: list[str]) -> tuple[list[str], list[dict]]:
    graph = derive_task_dependency_graph(decl)
    dependents: dict[str, list[str]] = defaultdict(list)
    for tid, deps in graph.items():
        for dep in deps:
            dependents[dep].append(tid)

    affected: set[str] = set(seed_tasks)
    rationale: list[dict] = []
    queue: deque[str] = deque(sorted(seed_tasks))
    while queue:
        current = queue.popleft()
        for nxt in sorted(dependents.get(current, ())):
            if nxt not in affected:
                affected.add(nxt)
                rationale.append({
                    "source_task": current, "affected_task": nxt,
                    "reason": "upstream task shard outcome changed",
                })
                queue.append(nxt)
    ordered = _stable_topological_order(affected, graph)
    return ordered, rationale
