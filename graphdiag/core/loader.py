"""JSON declaration loading with strict, fail-fast validation.

The declaration is rejected as a whole with :class:`E_GRAPH` on the first
conflict.  The error detail always identifies the offending node id, edge
(``source -> target``) or query id so the caller can report it precisely.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .errors import GraphValidationError
from .models import Edge, Graph, GraphDeclaration, Node, Query, _is_int


def _fail(message: str, **detail: Any) -> None:
    raise GraphValidationError(message, detail=detail)


def _require_object(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        _fail("declaration root must be a JSON object")
    return data


def load_declaration(path: str) -> GraphDeclaration:
    """Load and fully validate a graph + query declaration JSON file."""
    if not os.path.isfile(path):
        _fail(f"declaration file not found: {path}", path=path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        _fail(f"invalid JSON: {exc.msg} at line {exc.lineno}", path=path)
    data = _require_object(data)

    scale_exponent = _validate_config(data)
    graph = _validate_graph(data)
    queries = _validate_queries(data, graph)
    return GraphDeclaration(graph=graph, queries=queries, scale_exponent=scale_exponent)


def _validate_config(data: dict[str, Any]) -> int:
    config = data.get("config", {})
    if not isinstance(config, dict):
        _fail("'config' must be an object")
    exponent = config.get("scale_exponent", 6)
    if not _is_int(exponent) or not 0 <= exponent <= 6:
        _fail("scale_exponent must be an integer in [0, 6]", value=exponent)
    return exponent


def _validate_graph(data: dict[str, Any]) -> Graph:
    raw_nodes = data.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        _fail("'nodes' must be a non-empty list")

    seen_ids: set[int] = set()
    nodes: list[Node] = []
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            _fail(f"node at index {index} must be an object", node_index=index)
        node_id = raw.get("node_id")
        label = raw.get("label")
        if not _is_int(node_id):
            _fail("node_id must be an integer", node_index=index)
        if not _is_int(label):
            _fail(f"node {node_id} label must be an integer", node_id=node_id)
        if node_id in seen_ids:
            _fail(f"duplicate node id: {node_id}", node_id=node_id)
        seen_ids.add(node_id)
        nodes.append(Node(node_id=node_id, label=label))

    raw_edges = data.get("edges", [])
    if not isinstance(raw_edges, list):
        _fail("'edges' must be a list")

    edges: list[Edge] = []
    adjacency: dict[int, list[Edge]] = {}
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, dict):
            _fail(f"edge at index {index} must be an object", edge_index=index)
        source = raw.get("source")
        target = raw.get("target")
        weight = raw.get("weight_micro")
        edge_desc = f"{source} -> {target}"
        if not _is_int(source) or source not in seen_ids:
            _fail(f"edge {edge_desc} has missing source node", edge=edge_desc)
        if not _is_int(target) or target not in seen_ids:
            _fail(f"edge {edge_desc} has missing target node", edge=edge_desc)
        if not _is_int(weight) or weight < 0:
            _fail(
                f"edge {edge_desc} weight_micro must be a non-negative integer",
                edge=edge_desc,
            )
        edge = Edge(source=source, target=target, weight_micro=weight)
        edges.append(edge)
        adjacency.setdefault(source, []).append(edge)

    # Deterministic adjacency: explicit (weight, target, insertion) ordering,
    # never dict / set / hash iteration order.
    ordered_adjacency: dict[int, tuple[Edge, ...]] = {}
    for source in sorted(adjacency):
        ordered_adjacency[source] = tuple(
            sorted(
                adjacency[source],
                key=lambda e: (e.weight_micro, e.target, e.source),
            )
        )
    return Graph(
        nodes=tuple(sorted(nodes, key=lambda n: n.node_id)),
        edges=tuple(edges),
        adjacency=ordered_adjacency,
        node_ids=frozenset(seen_ids),
    )


def _validate_queries(data: dict[str, Any], graph: Graph) -> tuple[Query, ...]:
    raw_queries = data.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        _fail("'queries' must be a non-empty list")

    seen_query_ids: set[str] = set()
    queries: list[Query] = []
    for index, raw in enumerate(raw_queries):
        if not isinstance(raw, dict):
            _fail(f"query at index {index} must be an object", query_index=index)
        query_id = raw.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            _fail("query_id must be a non-empty string", query_index=index)
        if query_id in seen_query_ids:
            _fail(f"duplicate query id: {query_id}", query_id=query_id)
        seen_query_ids.add(query_id)

        source = raw.get("source")
        if not _is_int(source) or source not in graph.node_ids:
            _fail(f"query {query_id} references missing source node", query_id=query_id)

        raw_targets = raw.get("targets")
        if not isinstance(raw_targets, list) or not raw_targets:
            _fail(f"query {query_id} targets must be a non-empty list", query_id=query_id)
        targets: list[int] = []
        for target in raw_targets:
            if not _is_int(target) or target not in graph.node_ids:
                _fail(
                    f"query {query_id} references missing target node {target}",
                    query_id=query_id,
                )
            targets.append(target)
        # De-duplicate while keeping a stable sorted order.
        targets = sorted(set(targets))

        cap = raw.get("cap_micro")
        if not _is_int(cap) or cap <= 0:
            _fail(
                f"query {query_id} cap_micro must be a positive integer",
                query_id=query_id,
            )
        queries.append(
            Query(query_id=query_id, source=source, targets=tuple(targets), cap_micro=cap)
        )
    return tuple(queries)
