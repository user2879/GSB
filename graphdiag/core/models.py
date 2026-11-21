"""Immutable domain models and deterministic serialization helpers.

Every on-disk artifact (declaration files, SQLite snapshots) is reduced to a
canonical JSON encoding with sorted keys and compact separators, so identical
logical content always yields byte-identical digests regardless of the order
in which members were originally inserted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any

MICRO = 1_000_000
INTEGER_FORMAT_VERSION = 1


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, compact separators, no ASCII escape."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_digest(obj: Any) -> str:
    """SHA-256 over the canonical JSON encoding of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    # ``bool`` is an ``int`` subclass but must never be accepted as a weight.
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class Node:
    node_id: int
    label: int


@dataclass(frozen=True)
class Edge:
    source: int
    target: int
    weight_micro: int  # raw integer micro units, >= 0


@dataclass(frozen=True)
class Query:
    query_id: str
    source: int
    targets: tuple[int, ...]
    cap_micro: int  # raw positive integer micro units


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    adjacency: dict[int, tuple[Edge, ...]] = field(default_factory=dict, compare=False)
    node_ids: frozenset[int] = field(default_factory=frozenset, compare=False)

    def outgoing(self, node_id: int) -> tuple[Edge, ...]:
        return self.adjacency.get(node_id, ())


@dataclass(frozen=True)
class GraphDeclaration:
    graph: Graph
    queries: tuple[Query, ...]
    scale_exponent: int  # 0..6, scale factor = 10 ** exponent

    def graph_payload(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.graph.nodes],
            "edges": [asdict(e) for e in self.graph.edges],
        }

    def queries_payload(self) -> list[dict[str, Any]]:
        return [
            {
                "query_id": q.query_id,
                "source": q.source,
                "targets": list(q.targets),
                "cap_micro": q.cap_micro,
            }
            for q in self.queries
        ]
