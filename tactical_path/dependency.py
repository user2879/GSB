"""Persistent dependency graph and reverse invalidation closure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import sha256_json
from .errors import ErrorCode, TacticalError


@dataclass
class DependencyNode:
    id: str
    kind: str
    content_hash: str
    version: str
    deps: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "content_hash": self.content_hash,
            "version": self.version,
            "deps": list(self.deps),
            "metadata": dict(sorted(self.metadata.items())),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DependencyNode":
        return cls(
            id=str(value["id"]),
            kind=str(value["kind"]),
            content_hash=str(value["content_hash"]),
            version=str(value["version"]),
            deps=tuple(str(item) for item in value.get("deps", [])),
            metadata=dict(value.get("metadata", {})),
        )


class DependencyGraph:
    def __init__(self, tier: str) -> None:
        self.tier = tier
        self.nodes: dict[str, DependencyNode] = {}

    def add(self, node: DependencyNode) -> None:
        self.nodes[node.id] = node

    def remove(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    def get(self, node_id: str) -> DependencyNode:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise TacticalError(
                ErrorCode.DEPENDENCY_STALE,
                "dependency node is absent",
                context={"tier": self.tier, "node_id": node_id},
            ) from exc

    def reverse_edges(self) -> dict[str, set[str]]:
        reverse: dict[str, set[str]] = {node_id: set() for node_id in self.nodes}
        for node in self.nodes.values():
            for dependency in node.deps:
                reverse.setdefault(dependency, set()).add(node.id)
        return reverse

    def closure(self, seeds: Iterable[str], *, include_seeds: bool = True) -> list[str]:
        reverse = self.reverse_edges()
        seen: set[str] = set()
        stack = list(sorted(set(seeds)))
        if include_seeds:
            seen.update(stack)
        while stack:
            current = stack.pop()
            for dependent in sorted(reverse.get(current, ())):
                if dependent not in seen:
                    seen.add(dependent)
                    stack.append(dependent)
        return sorted(seen)

    def cache_closure(self, seeds: Iterable[str]) -> list[str]:
        return [
            node_id
            for node_id in self.closure(seeds)
            if self.nodes.get(node_id) is not None
            and self.nodes[node_id].kind in {"flow", "jump", "path"}
        ]

    def dependent_versions(self, node_id: str) -> dict[str, str]:
        node = self.get(node_id)
        return {dep: self.nodes[dep].version for dep in node.deps if dep in self.nodes}

    def assert_dependencies_current(
        self, node_id: str, expected: dict[str, str]
    ) -> None:
        current = self.dependent_versions(node_id)
        for dependency, version in expected.items():
            if current.get(dependency) != version:
                raise TacticalError(
                    ErrorCode.DEPENDENCY_STALE,
                    "cached result references a stale dependency",
                    context={
                        "tier": self.tier,
                        "node_id": node_id,
                        "dependency": dependency,
                        "expected_version": version,
                        "actual_version": current.get(dependency),
                    },
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "dependency-graph-v1",
            "tier": self.tier,
            "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            "content_hash": self.content_hash(),
        }

    def content_hash(self) -> str:
        return sha256_json(
            {
                "tier": self.tier,
                "nodes": [self.nodes[key].to_dict() for key in sorted(self.nodes)],
            }
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DependencyGraph":
        if value.get("schema") != "dependency-graph-v1":
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "unsupported dependency graph schema",
                context={"schema": value.get("schema")},
            )
        graph = cls(str(value.get("tier", "")))
        for item in value.get("nodes", []):
            node = DependencyNode.from_dict(item)
            graph.add(node)
        expected = value.get("content_hash")
        if expected is not None and graph.content_hash() != expected:
            raise TacticalError(
                ErrorCode.SNAPSHOT_VERSION,
                "dependency graph content hash mismatch",
                context={"tier": graph.tier},
            )
        return graph

