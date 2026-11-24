from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

from .errors import DependencyCycleError
from .spec import Spec


@dataclass(frozen=True)
class PropagationStep:
    shard_id: str
    order: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"order": self.order, "shard_id": self.shard_id, "reasons": list(self.reasons)}


class DependencyGraph:
    def __init__(
        self,
        nodes: tuple[str, ...],
        edges: dict[str, set[str]],
        incoming: dict[str, set[str]],
    ):
        self.nodes = tuple(sorted(nodes))
        self.edges = {node: set(edges.get(node, set())) for node in self.nodes}
        self.incoming = {node: set(incoming.get(node, set())) for node in self.nodes}

    @classmethod
    def build(cls, spec: Spec) -> "DependencyGraph":
        nodes: set[str] = set()
        edges: dict[str, set[str]] = defaultdict(set)
        incoming: dict[str, set[str]] = defaultdict(set)

        for task in spec.tasks:
            nodes.update(task.shard_ids)
            for shard_index, refs in task.shard_dependencies:
                downstream = task.shard_id(shard_index)
                nodes.add(downstream)
                for upstream in refs:
                    nodes.add(upstream)
                    edges[upstream].add(downstream)
                    incoming[downstream].add(upstream)

        for task in spec.tasks:
            downstream_gate = task.shard_id(0)
            for upstream_task_id in task.depends_on:
                upstream_task = spec.task_by_id[upstream_task_id]
                for upstream_shard in upstream_task.shard_ids:
                    edges[upstream_shard].add(downstream_gate)
                    incoming[downstream_gate].add(upstream_shard)

        return cls(tuple(sorted(nodes)), edges, incoming)

    def stable_topological_order(self, subset: Iterable[str] | None = None) -> list[str]:
        allowed = set(self.nodes if subset is None else subset)
        indegree = {node: 0 for node in allowed}
        for node in allowed:
            indegree[node] = sum(1 for prev in self.incoming.get(node, ()) if prev in allowed)
        ready = sorted(node for node, degree in indegree.items() if degree == 0)
        queue = deque(ready)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            newly_ready: list[str] = []
            for nxt in self.edges.get(node, ()):
                if nxt not in allowed:
                    continue
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    newly_ready.append(nxt)
            for nxt in sorted(newly_ready):
                queue.append(nxt)
        if len(order) != len(allowed):
            cycle = self.find_cycle()
            raise DependencyCycleError(
                "dependency graph contains a cycle",
                locator={"field": "shard_dependencies/depends_on", "cycle": cycle},
            )
        return order

    def ensure_acyclic(self) -> list[str]:
        return self.stable_topological_order()

    def find_cycle(self) -> list[str]:
        color = {node: 0 for node in self.nodes}
        stack: list[str] = []

        def visit(node: str) -> list[str] | None:
            color[node] = 1
            stack.append(node)
            for nxt in sorted(self.edges.get(node, ())):
                if color[nxt] == 0:
                    found = visit(nxt)
                    if found:
                        return found
                elif color[nxt] == 1:
                    start = stack.index(nxt)
                    return stack[start:] + [nxt]
            stack.pop()
            color[node] = 2
            return None

        for node in self.nodes:
            if color[node] == 0:
                found = visit(node)
                if found:
                    return found
        return []

    def descendants(self, sources: Iterable[str]) -> set[str]:
        source_set = set(sources)
        affected: set[str] = set(source_set)
        queue = deque(sorted(source_set))
        while queue:
            node = queue.popleft()
            for nxt in sorted(self.edges.get(node, ())):
                if nxt not in affected:
                    affected.add(nxt)
                    queue.append(nxt)
        return affected

    def propagation_trace(
        self,
        sources: Iterable[str],
        *,
        statuses: dict[str, str] | None = None,
        details: dict[str, str] | None = None,
    ) -> list[PropagationStep]:
        source_set = set(sources)
        affected = self.descendants(source_set)
        order = [node for node in self.stable_topological_order(affected)]
        statuses = statuses or {}
        details = details or {}
        steps: list[PropagationStep] = []
        for index, node in enumerate(order):
            if node in source_set:
                basis = details.get(node, "trigger_source")
                step_reasons = (f"source:{statuses.get(node, 'changed')}:{basis}",)
            else:
                parent_reasons = []
                for prev in sorted(self.incoming.get(node, ())):
                    if prev in affected:
                        status = statuses.get(prev, "affected")
                        detail = details.get(prev, "dependency changed")
                        parent_reasons.append(f"{prev}:{status}:{detail}")
                step_reasons = tuple(parent_reasons or ("reachable_dependency",))
            steps.append(PropagationStep(node, index, step_reasons))
        return steps
