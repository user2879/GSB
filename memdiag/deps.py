"""Directed dependency graph, deterministic topology and invalidation traces."""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from .errors import DependencyError
from .models import InvalidationStep, Plan


@dataclass(frozen=True)
class DependencyGraph:
    edges: Mapping[str, tuple[str, ...]]
    reverse: Mapping[str, tuple[str, ...]]
    edge_labels: Mapping[tuple[str, str], tuple[str, ...]]
    topo: tuple[str, ...]

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self.edges))

    def parents(self, layer_id: str) -> tuple[str, ...]:
        return tuple(self.reverse.get(layer_id, ()))

    def children(self, layer_id: str) -> tuple[str, ...]:
        return tuple(self.edges.get(layer_id, ()))


def _find_cycle(graph: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    color: dict[str, int] = {node: 0 for node in sorted(graph)}
    stack: list[str] = []
    stack_index: dict[str, int] = {}

    def visit(node: str) -> tuple[str, ...] | None:
        color[node] = 1
        stack_index[node] = len(stack)
        stack.append(node)
        for child in graph.get(node, ()):
            if color.get(child, 0) == 0:
                found = visit(child)
                if found:
                    return found
            elif color.get(child, 0) == 1:
                return tuple(stack[stack_index[child] :] + [child])
        stack.pop()
        stack_index.pop(node, None)
        color[node] = 2
        return None

    for node in sorted(graph):
        if color[node] == 0:
            found = visit(node)
            if found:
                return found
    return ()


def build_dependency_graph(plan: Plan) -> DependencyGraph:
    nodes = set(plan.layer_ids)
    edge_labels: dict[tuple[str, str], set[str]] = defaultdict(set)
    for producer in sorted(plan.consumers):
        for consumer in plan.consumers[producer]:
            edge_labels[(producer, consumer)].add("output")
    for owner in sorted(plan.memory_derived_from):
        for derived_from in plan.memory_derived_from[owner]:
            edge_labels[(derived_from, owner)].add("memory")

    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    reverse: dict[str, set[str]] = {node: set() for node in nodes}
    indegree: dict[str, int] = {node: 0 for node in nodes}
    for producer, consumer in sorted(edge_labels):
        if consumer not in adjacency[producer]:
            adjacency[producer].add(consumer)
            reverse[consumer].add(producer)
            indegree[consumer] += 1

    heap = [node for node in sorted(nodes) if indegree[node] == 0]
    heapq.heapify(heap)
    ordered: list[str] = []
    while heap:
        node = heapq.heappop(heap)
        ordered.append(node)
        for child in sorted(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, child)
    if len(ordered) != len(nodes):
        graph = {node: tuple(sorted(adjacency[node])) for node in sorted(adjacency)}
        cycle = _find_cycle(graph)
        raise DependencyError(
            "dependency graph contains a cycle",
            layer_id=cycle[0] if cycle else "<plan>",
            cycle=list(cycle),
            referrers=list(cycle[1:]) if len(cycle) > 1 else [],
        )

    return DependencyGraph(
        edges={node: tuple(sorted(adjacency[node])) for node in sorted(adjacency)},
        reverse={node: tuple(sorted(reverse[node])) for node in sorted(reverse)},
        edge_labels={
            edge: tuple(sorted(labels)) for edge, labels in sorted(edge_labels.items())
        },
        topo=tuple(ordered),
    )


def _union_edges(old_plan: Plan | None, new_plan: Plan) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node: set() for node in new_plan.layer_ids}
    plans = [new_plan] if old_plan is None else [old_plan, new_plan]
    for plan in plans:
        for producer in sorted(plan.consumers):
            adjacency.setdefault(producer, set())
            for consumer in plan.consumers[producer]:
                adjacency.setdefault(consumer, set())
                adjacency[producer].add(consumer)
        for owner in sorted(plan.memory_derived_from):
            adjacency.setdefault(owner, set())
            for source in plan.memory_derived_from[owner]:
                adjacency.setdefault(source, set())
                adjacency[source].add(owner)
    return adjacency


def _edge_reason(old_plan: Plan | None, new_plan: Plan, source: str, target: str) -> str:
    labels: set[str] = set()
    for plan in ([new_plan] if old_plan is None else [old_plan, new_plan]):
        if target in plan.consumers.get(source, ()):
            labels.add("output")
        if source in plan.memory_derived_from.get(target, ()):
            labels.add("memory")
    if labels == {"output", "memory"}:
        return "output_and_memory_dependency"
    if labels == {"memory"}:
        return "memory_derivation"
    return "upstream_output"


def propagate_invalidation(
    new_plan: Plan,
    old_plan: Plan | None,
    changed_sources: Mapping[str, str] | Iterable[str],
) -> tuple[tuple[str, ...], tuple[InvalidationStep, ...]]:
    if isinstance(changed_sources, Mapping):
        trigger_reasons = {str(key): str(value) for key, value in changed_sources.items()}
    else:
        trigger_reasons = {str(key): "input_change" for key in changed_sources}
    if old_plan is None:
        trigger_reasons = {layer_id: "initial_compute" for layer_id in new_plan.layer_ids}

    adjacency = _union_edges(old_plan, new_plan)
    sources = sorted(trigger_reasons)
    new_nodes = set(new_plan.layer_ids)

    distances: dict[str, tuple[int, tuple[str, ...]]] = {}
    queue: deque[tuple[str, int, tuple[str, ...]]] = deque()
    for source in sources:
        if old_plan is None or source in new_nodes or source in adjacency:
            distances[source] = (0, (source,))
            queue.append((source, 0, (source,)))
    while queue:
        node, depth, path = queue.popleft()
        candidate = (depth, path)
        if node in distances and distances[node] < candidate:
            continue
        for child in sorted(adjacency.get(node, ())):
            next_path = path + (child,)
            next_state = (depth + 1, next_path)
            previous = distances.get(child)
            if previous is None or next_state < previous:
                distances[child] = next_state
                queue.append((child, depth + 1, next_path))

    invalidated: list[str] = []
    trace: list[InvalidationStep] = []
    for layer_id in build_dependency_graph(new_plan).topo:
        if layer_id not in distances:
            continue
        depth, path = distances[layer_id]
        trigger = path[0]
        via = path[-2] if len(path) > 1 else None
        if depth == 0:
            reason = trigger_reasons.get(layer_id, "input_change")
        else:
            reason = _edge_reason(old_plan, new_plan, path[-2], layer_id)
        invalidated.append(layer_id)
        trace.append(
            InvalidationStep(
                layer_id=layer_id,
                trigger_source=trigger,
                via=via,
                depth=depth,
                reason=reason,
                path=path,
            )
        )
    return tuple(invalidated), tuple(trace)
