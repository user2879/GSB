"""Versioned parameter-to-tile dependency graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .errors import DiffRenderError, E_CLOSURE_MISMATCH, E_DEPENDENCY_STALE


@dataclass
class Node:
    key: str
    kind: str
    content_hash: str
    version: int
    version_stamp: str
    scope: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind,
            "content_hash": self.content_hash,
            "version": self.version,
            "version_stamp": self.version_stamp,
            "scope": list(self.scope),
        }


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    triggers: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {"src": self.src, "dst": self.dst, "triggers": list(self.triggers)}


@dataclass
class DependencyGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[tuple[str, str], Edge] = field(default_factory=dict)
    _outgoing: dict[str, set[str]] = field(default_factory=dict, init=False)
    _incoming: dict[str, set[str]] = field(default_factory=dict, init=False)

    def add_node(self, node: Node) -> None:
        if node.key in self.nodes:
            raise DiffRenderError(
                E_DEPENDENCY_STALE, f"duplicate graph node {node.key!r}"
            )
        self.nodes[node.key] = node
        self._outgoing.setdefault(node.key, set())
        self._incoming.setdefault(node.key, set())

    def add_edge(self, src: str, dst: str, triggers: Iterable[str] = ()) -> None:
        if src not in self.nodes or dst not in self.nodes:
            raise DiffRenderError(
                E_DEPENDENCY_STALE,
                "dependency edge references an unknown node",
                {"src": src, "dst": dst},
            )
        trigger_tuple = tuple(sorted(set(triggers)))
        key = (src, dst)
        existing = self.edges.get(key)
        if existing is not None:
            trigger_tuple = tuple(sorted(set(existing.triggers) | set(trigger_tuple)))
        self.edges[key] = Edge(src, dst, trigger_tuple)
        self._outgoing.setdefault(src, set()).add(dst)
        self._incoming.setdefault(dst, set()).add(src)

    def direct_changed_seeds(self, changed_params: Iterable[str]) -> set[str]:
        changed = set(changed_params)
        seeds: set[str] = set()
        for edge in self.edges.values():
            if edge.triggers and changed.intersection(edge.triggers):
                seeds.add(edge.dst)
        return seeds

    def invalidate(
        self,
        changed_params: Iterable[str],
        extra_roots: Iterable[str] = (),
        include_param_nodes: bool = True,
    ) -> tuple[str, ...]:
        """Return the exact downstream closure of a parameter edit.

        ``extra_roots`` lets the geometry layer add a tile that is newly covered
        by a moved sphere.  Once a node is a seed, every dependent is invalidated
        regardless of the trigger labels on structural edges.
        """

        changed = tuple(sorted(set(changed_params)))
        seeds = self.direct_changed_seeds(changed)
        for root in extra_roots:
            if root not in self.nodes:
                raise DiffRenderError(
                    E_DEPENDENCY_STALE,
                    "closure root is not present in dependency graph",
                    {"node": root},
                )
            seeds.add(root)
        if include_param_nodes:
            for key in changed:
                node_key = f"param:{key}"
                if node_key in self.nodes:
                    seeds.add(node_key)
        seen = set(seeds)
        stack = sorted(seeds)
        while stack:
            current = stack.pop()
            for dependent in sorted(self._outgoing.get(current, ())):
                if dependent not in seen:
                    seen.add(dependent)
                    stack.append(dependent)
        return tuple(sorted(seen))

    def assert_closure(
        self,
        actual: Sequence[str],
        expected: Sequence[str],
        context: Mapping[str, object] | None = None,
    ) -> None:
        actual_set, expected_set = set(actual), set(expected)
        if actual_set != expected_set:
            raise DiffRenderError(
                E_CLOSURE_MISMATCH,
                "invalidated closure does not match exact tile closure",
                {
                    "missing": sorted(expected_set - actual_set),
                    "unexpected": sorted(actual_set - expected_set),
                    "context": dict(context or {}),
                },
            )

    def verify_nodes(self, expected_hashes: Mapping[str, str]) -> None:
        for key in sorted(expected_hashes):
            node = self.nodes.get(key)
            if node is None:
                raise DiffRenderError(
                    E_DEPENDENCY_STALE,
                    "persisted node is missing after graph rebuild",
                    {"node": key},
                )
            if node.content_hash != expected_hashes[key]:
                raise DiffRenderError(
                    E_DEPENDENCY_STALE,
                    "node hash differs from committed snapshot",
                    {
                        "node": key,
                        "expected": expected_hashes[key],
                        "actual": node.content_hash,
                    },
                )

    def hash_manifest(self) -> dict[str, str]:
        return {key: self.nodes[key].content_hash for key in sorted(self.nodes)}

    def stamp_manifest(self) -> dict[str, str]:
        return {key: self.nodes[key].version_stamp for key in sorted(self.nodes)}

    def to_dict(self) -> dict[str, object]:
        return {
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "edges": [
                self.edges[key].to_dict()
                for key in sorted(self.edges, key=lambda item: (item[0], item[1]))
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DependencyGraph":
        try:
            graph = cls()
            for raw in data["nodes"]:  # type: ignore[index]
                graph.add_node(
                    Node(
                        key=str(raw["key"]),
                        kind=str(raw["kind"]),
                        content_hash=str(raw["content_hash"]),
                        version=int(raw["version"]),
                        version_stamp=str(raw["version_stamp"]),
                        scope=tuple(str(v) for v in raw.get("scope", [])),
                    )
                )
            for raw in data["edges"]:  # type: ignore[index]
                graph.add_edge(
                    str(raw["src"]),
                    str(raw["dst"]),
                    tuple(str(v) for v in raw.get("triggers", [])),
                )
            return graph
        except (KeyError, TypeError, ValueError) as exc:
            raise DiffRenderError(
                E_DEPENDENCY_STALE, f"invalid serialized dependency graph: {exc}"
            ) from exc

