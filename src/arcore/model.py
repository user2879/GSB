"""Core immutable data model and service dependency topology."""

from dataclasses import dataclass, field

from .errors import SpecError


@dataclass(frozen=True)
class Alert:
    """One normalized alert record."""

    alert_id: str
    source_id: str
    timestamp_ms: int
    service_id: str
    level: str
    fingerprint: str
    labels: tuple[str, ...]
    status: str  # OPEN or CLOSED
    seq: int

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "source_id": self.source_id,
            "timestamp_ms": self.timestamp_ms,
            "service_id": self.service_id,
            "level": self.level,
            "fingerprint": self.fingerprint,
            "labels": list(self.labels),
            "status": self.status,
            "seq": self.seq,
        }

    @staticmethod
    def from_dict(data: dict) -> "Alert":
        return Alert(
            alert_id=data["alert_id"],
            source_id=data["source_id"],
            timestamp_ms=int(data["timestamp_ms"]),
            service_id=data["service_id"],
            level=data["level"],
            fingerprint=data["fingerprint"],
            labels=tuple(data["labels"]),
            status=data["status"],
            seq=int(data["seq"]),
        )


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    seed: int
    record_count: int
    services: tuple[str, ...]
    levels: tuple[str, ...]
    start_ts_ms: int
    max_gap_ms: int
    fingerprint_namespace: int
    label_vocabulary: tuple[str, ...]
    max_labels: int
    closed_probability: float


@dataclass(frozen=True)
class Topology:
    """Directed acyclic dependency graph.

    Edge ``u -> v`` means service ``u`` depends on service ``v`` (a failure
    of ``v`` may surface as alerts in ``u``). Layer 0 contains root-cause
    services with no outgoing dependency edges.
    """

    edges: tuple[tuple[str, str], ...]
    nodes: tuple[str, ...]
    _adj: dict[str, tuple[str, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    _layer: dict[str, int] = field(
        default_factory=dict, compare=False, repr=False
    )

    @staticmethod
    def build(edges: list[tuple[str, str]],
              extra_nodes: list[str] | None = None) -> "Topology":
        node_set: set[str] = set(extra_nodes or [])
        for u, v in edges:
            node_set.add(u)
            node_set.add(v)
        nodes = tuple(sorted(node_set))
        adj: dict[str, list[str]] = {n: [] for n in nodes}
        indeg: dict[str, int] = {n: 0 for n in nodes}
        for u, v in edges:
            adj[u].append(v)
            indeg[v] += 1
        for n in adj:
            adj[n] = sorted(set(adj[n]))
        # Kahn topological sort doubles as cycle detection.
        ready = sorted(n for n in nodes if indeg[n] == 0)
        order: list[str] = []
        indeg_work = dict(indeg)
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in adj[node]:
                indeg_work[child] -= 1
                if indeg_work[child] == 0:
                    ready.append(child)
                    ready.sort()
        if len(order) != len(nodes):
            cycle = sorted(n for n in nodes if n not in set(order))
            raise SpecError(
                f"dependency graph contains a cycle involving {cycle}",
                field="topology.edges",
            )
        # Layer = longest dependency path from the node. Roots get layer 0.
        # DFS memoization guarantees every dependency is resolved first even
        # though the Kahn ordering above follows edge direction oppositely.
        layer: dict[str, int] = {}

        def depth(node: str) -> int:
            if node in layer:
                return layer[node]
            deps = adj[node]
            layer[node] = 0 if not deps else 1 + max(depth(d) for d in deps)
            return layer[node]

        for node in nodes:
            depth(node)
        return Topology(
            edges=tuple(sorted((u, v) for u, v in edges)),
            nodes=nodes,
            _adj={n: tuple(adj[n]) for n in nodes},
            _layer=layer,
        )

    def depends_on(self, service: str) -> tuple[str, ...]:
        return self._adj.get(service, ())

    def are_related(self, a: str, b: str) -> bool:
        if a == b:
            return True
        return b in self._adj.get(a, ()) or a in self._adj.get(b, ())

    def layer_of(self, service: str) -> int:
        return self._layer[service]

    def reachable_from(self, service: str) -> set[str]:
        """Transitive dependency closure of ``service``."""
        seen: set[str] = set()
        stack = [service]
        while stack:
            current = stack.pop()
            for dep in self._adj.get(current, ()):
                if dep not in seen:
                    seen.add(dep)
                    stack.append(dep)
        return seen

    def to_dict(self) -> dict:
        return {"nodes": list(self.nodes), "edges": [list(e) for e in self.edges]}
