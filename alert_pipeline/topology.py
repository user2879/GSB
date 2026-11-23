"""Service dependency graph: validation, stable topology order, path search."""
from __future__ import annotations

from collections import deque

from .errors import SpecError
from .util import canonical_digest


class DependencyGraph:
    def __init__(self, graph_doc: dict):
        if not isinstance(graph_doc, dict) or "services" not in graph_doc:
            raise SpecError("graph document must contain a 'services' array",
                            0, "services")
        services = graph_doc["services"]
        if not isinstance(services, list):
            raise SpecError("'services' must be an array", 0, "services")

        ids: list[str] = []
        seen: set[str] = set()
        for index, svc in enumerate(services, start=1):
            if not isinstance(svc, dict) or "service_id" not in svc:
                raise SpecError("service entry missing service_id", index,
                                "service_id")
            sid = svc["service_id"]
            if not isinstance(sid, str) or not sid:
                raise SpecError("service_id must be a non-empty string", index,
                                "service_id")
            if sid in seen:
                raise SpecError("duplicate service_id", index, "service_id",
                                sid)
            seen.add(sid)
            ids.append(sid)

        self.services: tuple[str, ...] = tuple(ids)
        # Edge a -> b means: a depends on b (b is upstream of a / a calls b).
        self.edges: dict[str, set[str]] = {sid: set() for sid in ids}
        for index, svc in enumerate(services, start=1):
            sid = svc["service_id"]
            deps = svc.get("depends_on", [])
            if not isinstance(deps, list) or not all(
                    isinstance(d, str) for d in deps):
                raise SpecError("depends_on must be a list of service ids",
                                index, "depends_on")
            for dep in deps:
                if dep not in seen:
                    raise SpecError("dependency references unknown service",
                                    index, "depends_on", dep)
                self.edges[sid].add(dep)

        # Reverse adjacency: who depends on me (my downstream callers).
        self.reverse: dict[str, set[str]] = {sid: set() for sid in ids}
        for src, targets in self.edges.items():
            for tgt in targets:
                self.reverse[tgt].add(src)
        self._acyclic_check()
        self.topo_order = self._kahn_topo()
        self.topo_rank = {sid: i for i, sid in enumerate(self.topo_order)}
        self.digest = canonical_digest(
            {sid: sorted(self.edges[sid]) for sid in self.topo_order})

    def _acyclic_check(self) -> None:
        indeg = {sid: 0 for sid in self.services}
        for src in self.services:
            for tgt in self.edges[src]:
                indeg[tgt] += 1
        # Deterministic queue order independent of insertion order.
        queue = deque(sorted(sid for sid, d in indeg.items() if d == 0))
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for tgt in sorted(self.edges[node]):
                indeg[tgt] -= 1
                if indeg[tgt] == 0:
                    queue.append(tgt)
        if visited != len(self.services):
            offenders = sorted(sid for sid, d in indeg.items() if d > 0)
            raise SpecError("dependency graph contains a cycle", 0,
                            "depends_on", "->".join(offenders))

    def _kahn_topo(self) -> tuple[str, ...]:
        indeg = {sid: len(self.edges[sid]) for sid in self.services}
        # Kahn over edges (src -> dependency): dependencies first. The ready
        # set is kept sorted so the order is independent of node insertion.
        ready = sorted(sid for sid, d in indeg.items() if d == 0)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for caller in sorted(self.reverse.get(node, set())):
                indeg[caller] -= 1
                if indeg[caller] == 0:
                    ready.append(caller)
                    ready.sort()
        if len(order) != len(self.services):
            raise SpecError("dependency graph contains a cycle", 0,
                            "depends_on")
        return tuple(order)

    def shortest_paths_from(self, origin: str) -> dict[str, tuple[str, ...]]:
        """BFS over outgoing edges: origin -> its upstream dependencies."""
        paths = {origin: (origin,)}
        queue = deque([origin])
        while queue:
            node = queue.popleft()
            for nxt in sorted(self.edges[node]):
                if nxt not in paths:
                    paths[nxt] = paths[node] + (nxt,)
                    queue.append(nxt)
        return paths

    def shortest_paths_to(self, origin: str) -> dict[str, tuple[str, ...]]:
        """BFS over reverse edges: origin -> its downstream callers."""
        paths = {origin: (origin,)}
        queue = deque([origin])
        while queue:
            node = queue.popleft()
            for nxt in sorted(self.reverse[node]):
                if nxt not in paths:
                    paths[nxt] = paths[node] + (nxt,)
                    queue.append(nxt)
        return paths

