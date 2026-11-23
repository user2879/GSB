"""Stable dependency graph traversal for invalidation and deletion checks."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import dependency_error
from .schema import Plan, referenced_by


@dataclass(frozen=True)
class Edge:
    source: str
    dependent: str
    kind: str

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "dependent": self.dependent,
            "kind": self.kind,
        }


class DependencyGraph:
    """A deterministic graph with activation and memory-derivation edges.

    Edges point from a dependency to its dependent.  Topological ties are
    broken by declaration order first and layer id second.
    """

    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.rank = plan.layer_index
        self.edges_by_source: dict[str, set[str]] = {
            layer.id: set() for layer in plan.layers
        }
        self.reasons: dict[tuple[str, str], set[str]] = {}
        self.edges: list[Edge] = []
        for layer in plan.layers:
            for dependency in layer.depends_on:
                self._add_edge(dependency, layer.id, "activation")
            for term in layer.memory_derivation.terms:
                self._add_edge(term.layer_id, layer.id, "memory")
        self.order = self._stable_topological_order()
        self.position = {layer_id: index for index, layer_id in enumerate(self.order)}

    def _key(self, layer_id: str) -> tuple[int, str]:
        return (self.rank[layer_id], layer_id)

    def _add_edge(self, source: str, dependent: str, kind: str) -> None:
        if dependent not in self.edges_by_source[source]:
            self.edges_by_source[source].add(dependent)
            self.edges.append(Edge(source, dependent, kind))
        self.reasons.setdefault((source, dependent), set()).add(kind)

    def _stable_topological_order(self) -> tuple[str, ...]:
        indegree = {layer.id: 0 for layer in self.plan.layers}
        for source, dependents in self.edges_by_source.items():
            for dependent in dependents:
                indegree[dependent] += 1
        ready = [(self.rank[layer_id], layer_id) for layer_id, degree in indegree.items() if degree == 0]
        heapq.heapify(ready)
        ordered: list[str] = []
        while ready:
            _, current = heapq.heappop(ready)
            ordered.append(current)
            for dependent in sorted(self.edges_by_source[current], key=self._key):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    heapq.heappush(ready, self._key(dependent))
        if len(ordered) != len(self.plan.layers):
            # parse_plan normally makes this unreachable, but keep graph usable
            # as a diagnostic object for malformed plans constructed in tests.
            remaining = sorted(
                (layer_id for layer_id, degree in indegree.items() if degree > 0),
                key=self._key,
            )
            raise ValueError(f"dependency cycle: {remaining}")
        return tuple(ordered)

    def upstream(self, layer_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                (source for source, dependents in self.edges_by_source.items() if layer_id in dependents),
                key=self._key,
            )
        )

    def downstream(self, layer_ids: Iterable[str]) -> tuple[str, ...]:
        changed = set(layer_ids)
        affected = set(changed)
        queue = sorted(changed, key=self._key)
        while queue:
            current = queue.pop(0)
            for dependent in sorted(self.edges_by_source.get(current, ()), key=self._key):
                if dependent not in affected:
                    affected.add(dependent)
                    queue.append(dependent)
        return tuple(layer_id for layer_id in self.order if layer_id in affected)

    def invalidation_report(
        self,
        changed_layers: Iterable[str],
        *,
        reason: str = "declaration_change",
        changed_fields: dict[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        changed = sorted(set(changed_layers), key=self._key)
        for layer_id in changed:
            if layer_id not in self.rank:
                raise dependency_error(
                    "changed layer does not exist", layer_id=layer_id, field="id"
                )
        affected = self.downstream(changed)
        affected_set = set(affected)
        changed_set = set(changed)
        steps: list[dict[str, Any]] = []
        for layer_id in affected:
            if layer_id in changed_set:
                fields = sorted((changed_fields or {}).get(layer_id, []))
                steps.append(
                    {
                        "layer_id": layer_id,
                        "level": "changed",
                        "basis": reason,
                        "fields": fields,
                        "via": [],
                    }
                )
                continue
            via: list[dict[str, str]] = []
            for source in self.upstream(layer_id):
                if source not in affected_set:
                    continue
                for kind in sorted(self.reasons[(source, layer_id)]):
                    via.append({"source": source, "kind": kind})
            direct_sources = [item for item in via if item["source"] in changed_set]
            if direct_sources:
                level = "direct"
                basis = "direct_dependency_changed"
            else:
                level = "indirect"
                basis = "transitive_dependency_changed"
            steps.append(
                {
                    "layer_id": layer_id,
                    "level": level,
                    "basis": basis,
                    "fields": [],
                    "via": via,
                }
            )
        return {
            "changed": changed,
            "changed_fields": {
                layer_id: sorted(fields)
                for layer_id, fields in sorted((changed_fields or {}).items())
            },
            "invalidated": list(affected),
            "unaffected": [layer_id for layer_id in self.order if layer_id not in affected_set],
            "topological_order": list(self.order),
            "propagation": steps,
        }

    def assert_deletable(self, layer_id: str) -> None:
        if layer_id not in self.rank:
            raise dependency_error(
                "layer does not exist", layer_id=layer_id, field="id"
            )
        referrers = referenced_by(self.plan, layer_id)
        if referrers:
            raise dependency_error(
                "cannot delete a layer that is still referenced",
                layer_id=layer_id,
                field="depends_on",
                referrers=referrers,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": list(self.order),
            "edges": [edge.to_dict() for edge in sorted(
                self.edges,
                key=lambda edge: (self.position.get(edge.source, -1), self.position.get(edge.dependent, -1), edge.kind),
            )],
            "topological_order": list(self.order),
        }


def build_graph(plan: Plan) -> DependencyGraph:
    return DependencyGraph(plan)
