"""Stable dependency graph helpers and failure/invalidation propagation.

Two edge kinds are merged into one global shard graph:

* intra-task edges from a shard to its ``depends_on`` indices;
* cross-task edges from every shard of a task to every shard of the tasks in
  its ``depends_on_tasks`` list.

All traversals use sorted identifiers and Kahn topological order so the
affected-set ordering is independent of dictionary/hash iteration order.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple

from .errors import E_CYCLE, E_DEPEND, Problem, SchedulerError
from .model import Declaration, Shard

ShardKey = Tuple[str, int]


def shard_key(shard: Shard) -> ShardKey:
    return (shard.task_id, shard.shard_index)


def _key_text(key: ShardKey) -> str:
    return f"{key[0]}#S{key[1]:02d}"


def build_global_graph(declaration: Declaration) -> Dict[ShardKey, List[ShardKey]]:
    """Return ``shard -> upstream shards`` adjacency for the full session."""
    index = declaration.shard_index()
    tasks = declaration.task_map()
    graph: Dict[ShardKey, List[ShardKey]] = {}
    for shard in declaration.all_shards():
        upstream: List[ShardKey] = []
        for dep_index in shard.depends_on:
            ref = (shard.task_id, dep_index)
            if ref not in index:
                raise SchedulerError(
                    E_DEPEND,
                    "missing intra-task shard reference",
                    [Problem(_key_text(shard_key(shard)),
                             "existing shard reference", _key_text(ref),
                             E_DEPEND)],
                )
            upstream.append(ref)
        for task_id in shard.depends_on_tasks:
            if task_id not in tasks:
                raise SchedulerError(
                    E_DEPEND,
                    "missing cross-task reference",
                    [Problem(_key_text(shard_key(shard)),
                             "existing task id", task_id, E_DEPEND)],
                )
            upstream_task = tasks[task_id]
            for other in upstream_task.shards:
                upstream.append(shard_key(other))
        graph[shard_key(shard)] = sorted(set(upstream))
    assert_acyclic(graph)
    return graph


def assert_acyclic(graph: Dict[ShardKey, List[ShardKey]]) -> None:
    indegree: Dict[ShardKey, int] = {node: 0 for node in graph}
    for node, upstreams in graph.items():
        for upstream in upstreams:
            if upstream in indegree:
                indegree[node] += 1
    # Kahn on the downstream direction: start with nodes having no upstreams.
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    seen: Set[ShardKey] = set()
    while ready:
        node = ready.pop(0)
        seen.add(node)
        for downstream in sorted(downstream_of(graph, node)):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                ready.append(downstream)
        ready.sort()
    cycle = sorted(node for node in graph if node not in seen)
    if cycle:
        raise SchedulerError(
            E_CYCLE,
            "dependency graph contains a cycle",
            [Problem("global_shard_graph", "acyclic graph",
                     [_key_text(node) for node in cycle], E_CYCLE)],
        )


def downstream_of(graph: Dict[ShardKey, List[ShardKey]],
                  node: ShardKey) -> Iterable[ShardKey]:
    for candidate, upstreams in graph.items():
        if node in upstreams:
            yield candidate


def topological_order(graph: Dict[ShardKey, List[ShardKey]]) -> List[ShardKey]:
    """Stable upstream-first Kahn ordering over shard keys."""
    indegree: Dict[ShardKey, int] = {
        node: len([up for up in upstreams if up in graph])
        for node, upstreams in graph.items()
    }
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    order: List[ShardKey] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for downstream in sorted(downstream_of(graph, node)):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                ready.append(downstream)
        ready.sort()
    if len(order) != len(graph):
        raise SchedulerError(E_CYCLE, "cycle prevents topological ordering", [])
    return order


def transitive_downstream(
    graph: Dict[ShardKey, List[ShardKey]],
    origins: Iterable[ShardKey],
) -> List[ShardKey]:
    """Return all direct/indirect downstream shards of ``origins``.

    The origins themselves are excluded; the result is returned in stable
    topological order so invalidation can be applied deterministically.
    """
    affected: Set[ShardKey] = set()
    stack = list(origins)
    while stack:
        node = stack.pop()
        for downstream in downstream_of(graph, node):
            if downstream not in affected:
                affected.add(downstream)
                stack.append(downstream)
    order = topological_order(graph)
    return [node for node in order if node in affected]


def reverse_dependency_count(graph: Dict[ShardKey, List[ShardKey]]) -> int:
    counts: Dict[ShardKey, int] = defaultdict(int)
    for upstreams in graph.values():
        for upstream in upstreams:
            counts[upstream] += 1
    return sum(counts.values())
