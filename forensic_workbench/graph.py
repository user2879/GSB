"""Dependency DAG structures and node-level incremental consistency checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .errors import closure_mismatch, dependency_stale
from .util import sha256_json


@dataclass
class GraphNode:
    node_id: str
    node_type: str
    payload: dict[str, Any]
    param_versions: dict[str, str]
    input_hashes: dict[str, str]
    produced_in_version: int
    content_hash: str = ""

    def compute_hash(self) -> str:
        return sha256_json(
            {
                "node_id": self.node_id,
                "node_type": self.node_type,
                "payload": self.payload,
                "param_versions": self.param_versions,
                "input_hashes": self.input_hashes,
            }
        )

    def seal(self) -> "GraphNode":
        self.content_hash = self.compute_hash()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "payload": self.payload,
            "param_versions": self.param_versions,
            "input_hashes": self.input_hashes,
            "produced_in_version": self.produced_in_version,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphNode":
        node = cls(
            node_id=str(value["node_id"]),
            node_type=str(value["node_type"]),
            payload=dict(value["payload"]),
            param_versions={str(key): str(val) for key, val in value["param_versions"].items()},
            input_hashes={str(key): str(val) for key, val in value["input_hashes"].items()},
            produced_in_version=int(value["produced_in_version"]),
            content_hash=str(value["content_hash"]),
        )
        if node.compute_hash() != node.content_hash:
            raise dependency_stale(
                "persisted node content hash does not match its payload",
                node_id=node.node_id,
                declared=node.content_hash,
                actual=node.compute_hash(),
            )
        return node


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    trigger_keys: tuple[str, ...]
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "trigger_keys": list(self.trigger_keys),
            "relation": self.relation,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphEdge":
        return cls(
            source=str(value["source"]),
            target=str(value["target"]),
            trigger_keys=tuple(str(item) for item in value["trigger_keys"]),
            relation=str(value["relation"]),
        )

    def sort_key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.relation)


@dataclass
class DependencyGraph:
    version: int
    param_versions: dict[str, str]
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]
    audit_history: list[dict[str, Any]] = field(default_factory=list)

    def seal(self) -> "DependencyGraph":
        for node in self.nodes.values():
            node.seal()
        self.edges = sorted(self.edges, key=GraphEdge.sort_key)
        self.validate()
        return self

    def validate(self) -> None:
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise dependency_stale(
                    "dependency edge source node is missing",
                    source=edge.source,
                    target=edge.target,
                    relation=edge.relation,
                )
            if edge.target not in self.nodes:
                raise dependency_stale(
                    "dependency edge target node is missing",
                    source=edge.source,
                    target=edge.target,
                    relation=edge.relation,
                )
        for node in self.nodes.values():
            actual = node.compute_hash()
            if node.content_hash and actual != node.content_hash:
                raise dependency_stale(
                    "node content hash mismatch",
                    node_id=node.node_id,
                    declared=node.content_hash,
                    actual=actual,
                )
            for dependency in node.input_hashes:
                if dependency not in self.nodes:
                    raise dependency_stale(
                        "node references a missing dependency",
                        node_id=node.node_id,
                        dependency=dependency,
                    )

    def node_hashes(self) -> dict[str, str]:
        return {node_id: node.content_hash for node_id, node in sorted(self.nodes.items())}

    def graph_hash(self) -> str:
        return sha256_json(
            {
                "param_versions": self.param_versions,
                "nodes": self.node_hashes(),
                "edges": [edge.to_dict() for edge in self.edges],
            }
        )

    def edges_for(self, node_id: str) -> list[GraphEdge]:
        return [
            edge
            for edge in self.edges
            if edge.source == node_id or edge.target == node_id
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "forensic-dependency-graph-v1",
            "version": self.version,
            "param_versions": self.param_versions,
            "nodes": {
                node_id: self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)
            },
            "edges": [edge.to_dict() for edge in self.edges],
            "graph_hash": self.graph_hash(),
            "audit_history": self.audit_history,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DependencyGraph":
        try:
            graph = cls(
                version=int(value["version"]),
                param_versions={
                    str(key): str(item) for key, item in value["param_versions"].items()
                },
                nodes={
                    str(node_id): GraphNode.from_dict(node_value)
                    for node_id, node_value in value["nodes"].items()
                },
                edges=[GraphEdge.from_dict(item) for item in value["edges"]],
                audit_history=list(value.get("audit_history", [])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise dependency_stale(
                "dependency graph has an invalid structure",
                reason=str(exc),
            ) from exc
        graph.validate()
        declared = value.get("graph_hash")
        actual = graph.graph_hash()
        if declared != actual:
            raise dependency_stale(
                "dependency graph hash mismatch",
                expected=declared,
                actual=actual,
            )
        return graph


def ensure_nodes(graph: DependencyGraph, node_ids: Iterable[str]) -> None:
    missing = sorted(node_id for node_id in node_ids if node_id not in graph.nodes)
    if missing:
        raise dependency_stale(
            "required dependency nodes are missing",
            missing_nodes=missing,
            graph_version=graph.version,
        )


def compare_graphs(
    incremental: DependencyGraph,
    full: DependencyGraph,
    *,
    context: dict[str, Any] | None = None,
) -> bool:
    base_context = dict(context or {})
    incremental_ids = set(incremental.nodes)
    full_ids = set(full.nodes)
    if incremental_ids != full_ids:
        raise closure_mismatch(
            "incremental and full graphs contain different node sets",
            **base_context,
            missing_from_incremental=sorted(full_ids - incremental_ids),
            missing_from_full=sorted(incremental_ids - full_ids),
        )
    if incremental.param_versions != full.param_versions:
        raise closure_mismatch(
            "incremental and full graphs use different parameter versions",
            **base_context,
            incremental=incremental.param_versions,
            full=full.param_versions,
        )
    for node_id in sorted(incremental.nodes):
        incremental_node = incremental.nodes[node_id]
        full_node = full.nodes[node_id]
        if incremental_node.content_hash == full_node.content_hash:
            continue
        affected_edges = [
            edge.to_dict()
            for edge in sorted(
                incremental.edges + full.edges, key=GraphEdge.sort_key
            )
            if edge.source == node_id or edge.target == node_id
        ]
        item = None
        if full_node.node_type == "cross_gop":
            item = full_node.payload.get("comparison_item")
        raise closure_mismatch(
            "incremental and full node results differ",
            **base_context,
            node_id=node_id,
            node_type=full_node.node_type,
            comparison_item=item,
            incremental_hash=incremental_node.content_hash,
            full_hash=full_node.content_hash,
            invalidated_edges=affected_edges,
            upstream_edges=[
                edge.to_dict()
                for edge in full.edges
                if edge.target == node_id
            ],
            downstream_edges=[
                edge.to_dict()
                for edge in full.edges
                if edge.source == node_id
            ],
            incremental_payload=incremental_node.payload,
            full_payload=full_node.payload,
        )
    incremental_edges = {edge.to_dict()["source"] + "|" + edge.to_dict()["target"] + "|" + edge.relation for edge in incremental.edges}
    full_edges = {edge.to_dict()["source"] + "|" + edge.to_dict()["target"] + "|" + edge.relation for edge in full.edges}
    if incremental_edges != full_edges:
        raise closure_mismatch(
            "incremental and full graphs contain different dependency edges",
            **base_context,
            missing_from_incremental=sorted(full_edges - incremental_edges),
            missing_from_full=sorted(incremental_edges - full_edges),
        )
    return True
