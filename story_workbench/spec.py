"""YAML parser and semantic validation for narrative specifications."""

from __future__ import annotations

import collections
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .canonical import double
from .errors import E_SPEC_IO, SpecInvalid, WorkbenchError
from .expressions import (
    coefficient_count,
    parse_expression,
    render_expression,
    validate_expression,
)
from .model import Condition, Edge, Effect, NarrativeSpec, Node, SequenceSample


SCHEMA = "narrative-workbench/spec/v1"
_COMPONENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPS = {">=", ">", "<=", "<"}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    if not isinstance(node, yaml.MappingNode):
        raise SpecInvalid("expected a YAML mapping")
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            mark = key_node.start_mark
            raise SpecInvalid(
                f"duplicate YAML mapping key {key!r}",
                location={"line": mark.line + 1, "column": mark.column + 1},
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_spec(path: str | Path) -> NarrativeSpec:
    spec_path = Path(path)
    try:
        text = spec_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise WorkbenchError(
            E_SPEC_IO,
            f"cannot read specification: {exc}",
            location={"path": str(spec_path)},
        ) from exc
    return parse_spec(text, source_name=str(spec_path))


def parse_spec(text: str, *, source_name: str = "<memory>") -> NarrativeSpec:
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except SpecInvalid:
        raise
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = {"source": source_name}
        if mark is not None:
            location.update({"line": mark.line + 1, "column": mark.column + 1})
        raise SpecInvalid(f"invalid YAML: {exc}", location=location) from exc

    root = _mapping(raw, location={"source": source_name})
    _only_keys(
        root,
        {
            "schema",
            "id",
            "title",
            "seed",
            "components",
            "initial_state",
            "boundary_tolerance",
            "merge_tolerance",
            "start",
            "nodes",
            "edges",
            "samples",
        },
        location={"field": "<root>"},
    )

    schema = _string(root.get("schema"), "schema", location={"field": "schema"})
    if schema != SCHEMA:
        raise SpecInvalid(
            f"unsupported schema {schema!r}; expected {SCHEMA!r}",
            location={"field": "schema"},
        )
    spec_id = _string(root.get("id"), "id", location={"field": "id"})
    title = _string(root.get("title", spec_id), "title", location={"field": "title"})
    seed = _integer(root.get("seed"), "seed", location={"field": "seed"})
    if seed < 0:
        raise SpecInvalid("seed must be non-negative", location={"field": "seed"})

    components_raw = _sequence(root.get("components"), "components", location={"field": "components"})
    if not components_raw:
        raise SpecInvalid("components must not be empty", location={"field": "components"})
    components: list[str] = []
    for index, component in enumerate(components_raw):
        name = _string(component, "component", location={"field": "components", "index": index})
        if not _COMPONENT_RE.match(name):
            raise SpecInvalid(
                f"component {name!r} must be an ASCII identifier usable in expressions",
                location={"field": "components", "index": index},
            )
        if name in components:
            raise SpecInvalid(
                f"duplicate component {name!r}",
                location={"field": "components", "index": index},
            )
        components.append(name)

    initial_raw = _mapping(root.get("initial_state"), location={"field": "initial_state"})
    _only_keys(initial_raw, set(components), location={"field": "initial_state"})
    initial_values: list[float] = []
    for component in components:
        if component not in initial_raw:
            raise SpecInvalid(
                f"initial_state is missing component {component!r}",
                location={"field": "initial_state", "component": component},
            )
        initial_values.append(
            _finite_number(
                initial_raw[component],
                "initial_state",
                location={"field": "initial_state", "component": component},
            )
        )

    boundary_tolerance = _finite_number(
        root.get("boundary_tolerance"),
        "boundary_tolerance",
        location={"field": "boundary_tolerance"},
    )
    merge_tolerance = _finite_number(
        root.get("merge_tolerance"),
        "merge_tolerance",
        location={"field": "merge_tolerance"},
    )
    if boundary_tolerance < 0.0:
        raise SpecInvalid("boundary_tolerance must be non-negative", location={"field": "boundary_tolerance"})
    if merge_tolerance < 0.0:
        raise SpecInvalid("merge_tolerance must be non-negative", location={"field": "merge_tolerance"})
    start = _string(root.get("start"), "start", location={"field": "start"})

    nodes: list[Node] = []
    node_ids: set[str] = set()
    nodes_raw = _sequence(root.get("nodes"), "nodes", location={"field": "nodes"})
    for index, value in enumerate(nodes_raw):
        location = {"field": "nodes", "index": index, "node_index": index}
        node_raw = _mapping(value, location=location)
        _only_keys(node_raw, {"id", "kind", "title", "ending"}, location={**location, "node_index": index})
        node_id = _string(node_raw.get("id"), "node.id", location=location)
        if node_id in node_ids:
            raise SpecInvalid(f"duplicate node id {node_id!r}", location={**location, "node_id": node_id})
        node_ids.add(node_id)
        kind = _string(node_raw.get("kind"), "node.kind", location={**location, "node_id": node_id})
        if kind not in {"scene", "choice"}:
            raise SpecInvalid(
                f"node kind must be 'scene' or 'choice', got {kind!r}",
                location={**location, "node_id": node_id},
            )
        node_title = _string(
            node_raw.get("title", node_id),
            "node.title",
            location={**location, "node_id": node_id},
        )
        ending = node_raw.get("ending")
        if ending is not None:
            ending = _string(ending, "node.ending", location={**location, "node_id": node_id})
        nodes.append(Node(index=index, id=node_id, kind=kind, title=node_title, ending=ending))

    if start not in node_ids:
        raise SpecInvalid(f"start node {start!r} does not exist", location={"field": "start"})

    node_lookup = {node.id: node for node in nodes}
    edges: list[Edge] = []
    edge_ids: set[str] = set()
    edges_raw = _sequence(root.get("edges"), "edges", location={"field": "edges"})
    for index, value in enumerate(edges_raw):
        edge_location = {"field": "edges", "index": index, "edge_index": index}
        edge_raw = _mapping(value, location=edge_location)
        _only_keys(
            edge_raw,
            {"id", "from", "to", "label", "effects", "conditions"},
            location={**edge_location, "edge_index": index},
        )
        edge_id = _string(edge_raw.get("id"), "edge.id", location=edge_location)
        if edge_id in edge_ids:
            raise SpecInvalid(f"duplicate edge id {edge_id!r}", location={**edge_location, "edge_id": edge_id})
        edge_ids.add(edge_id)
        source = _string(edge_raw.get("from"), "edge.from", location={**edge_location, "edge_id": edge_id})
        target = _string(edge_raw.get("to"), "edge.to", location={**edge_location, "edge_id": edge_id})
        if source not in node_lookup:
            raise SpecInvalid(
                f"edge {edge_id!r} references unknown source node {source!r}",
                location={**edge_location, "edge_id": edge_id, "node_id": source},
            )
        if target not in node_lookup:
            raise SpecInvalid(
                f"edge {edge_id!r} references unknown target node {target!r}",
                location={**edge_location, "edge_id": edge_id, "node_id": target},
            )
        label = _string(
            edge_raw.get("label", edge_id),
            "edge.label",
            location={**edge_location, "edge_id": edge_id},
        )

        effects_raw = edge_raw.get("effects", [])
        if effects_raw is None:
            effects_raw = []
        effects_seq = _sequence(effects_raw, "edge.effects", location={**edge_location, "edge_id": edge_id})
        effects: list[Effect] = []
        effect_ids: set[str] = set()
        for effect_index, effect_value in enumerate(effects_seq):
            effect_location = {
                **edge_location,
                "edge_id": edge_id,
                "effect_index": effect_index,
            }
            effect_raw = _mapping(effect_value, location=effect_location)
            _only_keys(effect_raw, {"id", "target", "expr"}, location=effect_location)
            effect_id = _string(effect_raw.get("id"), "effect.id", location=effect_location)
            if effect_id in effect_ids:
                raise SpecInvalid(
                    f"duplicate effect id {effect_id!r} on edge {edge_id!r}",
                    location={**effect_location, "effect_id": effect_id},
                )
            effect_ids.add(effect_id)
            target_component = _string(effect_raw.get("target"), "effect.target", location={**effect_location, "effect_id": effect_id})
            if target_component not in components:
                raise SpecInvalid(
                    f"effect {effect_id!r} writes unknown component {target_component!r}",
                    location={**effect_location, "effect_id": effect_id, "component": target_component},
                    details={"allowed_components": components},
                )
            expression = _string(
                effect_raw.get("expr"),
                "effect.expr",
                location={**effect_location, "effect_id": effect_id},
            )
            ast = parse_expression(expression, location={**effect_location, "effect_id": effect_id})
            validate_expression(
                ast,
                components=components,
                location={**effect_location, "effect_id": effect_id},
            )
            count = coefficient_count(ast)
            coefficient_keys = tuple(
                f"{edge_id}:{effect_id}:coef:{ordinal}" for ordinal in range(count)
            )
            effects.append(
                Effect(
                    id=effect_id,
                    target=target_component,
                    expression=render_expression(ast),
                    ast=ast,
                    coefficient_keys=coefficient_keys,
                    declaration_index=effect_index,
                )
            )

        conditions_raw = edge_raw.get("conditions", [])
        if conditions_raw is None:
            conditions_raw = []
        conditions_seq = _sequence(
            conditions_raw,
            "edge.conditions",
            location={**edge_location, "edge_id": edge_id},
        )
        conditions: list[Condition] = []
        condition_ids: set[str] = set()
        for condition_index, condition_value in enumerate(conditions_seq):
            condition_location = {
                **edge_location,
                "edge_id": edge_id,
                "condition_index": condition_index,
            }
            condition_raw = _mapping(condition_value, location=condition_location)
            _only_keys(
                condition_raw,
                {"id", "component", "op", "threshold"},
                location=condition_location,
            )
            condition_id = _string(condition_raw.get("id"), "condition.id", location=condition_location)
            if condition_id in condition_ids:
                raise SpecInvalid(
                    f"duplicate condition id {condition_id!r} on edge {edge_id!r}",
                    location={**condition_location, "condition_id": condition_id},
                )
            condition_ids.add(condition_id)
            component = _string(
                condition_raw.get("component"),
                "condition.component",
                location={**condition_location, "condition_id": condition_id},
            )
            if component not in components:
                raise SpecInvalid(
                    f"condition {condition_id!r} references unknown component {component!r}",
                    location={**condition_location, "condition_id": condition_id, "component": component},
                    details={"allowed_components": components},
                )
            op = _string(
                condition_raw.get("op"),
                "condition.op",
                location={**condition_location, "condition_id": condition_id},
            )
            if op not in _OPS:
                raise SpecInvalid(
                    f"condition {condition_id!r} has unsupported op {op!r}",
                    location={**condition_location, "condition_id": condition_id},
                    details={"allowed_ops": sorted(_OPS)},
                )
            threshold = _finite_number(
                condition_raw.get("threshold"),
                "condition.threshold",
                location={**condition_location, "condition_id": condition_id},
            )
            conditions.append(
                Condition(
                    id=condition_id,
                    component=component,
                    op=op,
                    threshold=threshold,
                    threshold_key=f"{edge_id}:{condition_id}:threshold",
                    declaration_index=condition_index,
                )
            )
        edges.append(
            Edge(
                index=index,
                id=edge_id,
                source=source,
                target=target,
                label=label,
                effects=tuple(effects),
                conditions=tuple(conditions),
            )
        )

    _validate_acyclic(nodes=nodes, edges=edges)

    samples: list[SequenceSample] = []
    sample_ids: set[str] = set()
    samples_raw = _sequence(root.get("samples", []), "samples", location={"field": "samples"})
    for index, value in enumerate(samples_raw):
        location = {"field": "samples", "index": index, "sample_index": index}
        sample_raw = _mapping(value, location=location)
        _only_keys(sample_raw, {"id", "description", "edges"}, location=location)
        sample_id = _string(sample_raw.get("id"), "sample.id", location=location)
        if sample_id in sample_ids:
            raise SpecInvalid(f"duplicate sample id {sample_id!r}", location={**location, "sample_id": sample_id})
        sample_ids.add(sample_id)
        description = _string(
            sample_raw.get("description", sample_id),
            "sample.description",
            location={**location, "sample_id": sample_id},
        )
        sequence_raw = _sequence(
            sample_raw.get("edges"),
            "sample.edges",
            location={**location, "sample_id": sample_id},
        )
        sequence: list[str] = []
        current = start
        for step_index, edge_value in enumerate(sequence_raw):
            chosen_id = _string(
                edge_value,
                "sample edge id",
                location={**location, "sample_id": sample_id, "step_index": step_index},
            )
            if chosen_id not in edge_ids:
                raise SpecInvalid(
                    f"sample {sample_id!r} references unknown edge {chosen_id!r}",
                    location={**location, "sample_id": sample_id, "step_index": step_index},
                )
            chosen = next(edge for edge in edges if edge.id == chosen_id)
            if chosen.source != current:
                raise SpecInvalid(
                    f"sample {sample_id!r} cannot take edge {chosen_id!r} from node {current!r}",
                    location={"field": "samples", "index": index, "sample_id": sample_id, "step_index": step_index},
                    details={"expected_source": current, "actual_source": chosen.source},
                )
            if node_lookup[current].ending is not None:
                raise SpecInvalid(
                    f"sample {sample_id!r} continues after ending {node_lookup[current].ending!r}",
                    location={"field": "samples", "index": index, "sample_id": sample_id, "step_index": step_index},
                )
            sequence.append(chosen_id)
            current = chosen.target
        samples.append(SequenceSample(id=sample_id, description=description, edges=tuple(sequence)))

    return NarrativeSpec(
        schema=schema,
        id=spec_id,
        title=title,
        seed=seed,
        components=tuple(components),
        initial_state=tuple(initial_values),
        boundary_tolerance=boundary_tolerance,
        merge_tolerance=merge_tolerance,
        start=start,
        nodes=tuple(nodes),
        edges=tuple(edges),
        samples=tuple(samples),
        source_name=source_name,
    )


def _mapping(value: Any, *, location: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecInvalid("expected a mapping", location=location)
    for key in value:
        if not isinstance(key, str):
            raise SpecInvalid("mapping keys must be strings", location=location)
    return value


def _sequence(value: Any, name: str, *, location: dict[str, Any]) -> list[Any]:
    if not isinstance(value, list):
        raise SpecInvalid(f"{name} must be a list", location=location)
    return value


def _string(value: Any, name: str, *, location: dict[str, Any]) -> str:
    if not isinstance(value, str) or not value:
        raise SpecInvalid(f"{name} must be a non-empty string", location=location)
    return value


def _integer(value: Any, name: str, *, location: dict[str, Any]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecInvalid(f"{name} must be an integer", location=location)
    return value


def _finite_number(value: Any, name: str, *, location: dict[str, Any]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecInvalid(f"{name} must be a finite number", location=location)
    try:
        result = double(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SpecInvalid(f"{name} must be finite", location=location) from exc
    if not math.isfinite(result):
        raise SpecInvalid(f"{name} must be finite", location=location)
    return result


def _only_keys(mapping: Mapping[str, Any], allowed: set[str], *, location: dict[str, Any]) -> None:
    for key in mapping:
        if key not in allowed:
            raise SpecInvalid(
                f"unknown field {key!r}",
                location={**location, "unknown_field": key},
                details={"allowed_fields": sorted(allowed)},
            )


def _validate_acyclic(*, nodes: Sequence[Node], edges: Sequence[Edge]) -> None:
    indegree = {node.id: 0 for node in nodes}
    outgoing: dict[str, list[Edge]] = {node.id: [] for node in nodes}
    for edge in edges:
        indegree[edge.target] += 1
        outgoing[edge.source].append(edge)

    ready = collections.deque(node.id for node in nodes if indegree[node.id] == 0)
    visited = 0
    while ready:
        node_id = ready.popleft()
        visited += 1
        for edge in outgoing[node_id]:
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                ready.append(edge.target)

    if visited == len(nodes):
        return

    cycle_start = next(node.id for node in nodes if indegree[node.id] > 0)
    path: list[str] = []
    seen: dict[str, int] = {}
    current = cycle_start
    cycle_edges: list[Edge] = []
    while current not in seen:
        seen[current] = len(path)
        path.append(current)
        candidates = [edge for edge in outgoing[current] if indegree[edge.target] > 0]
        if not candidates:
            break
        chosen = candidates[0]
        cycle_edges.append(chosen)
        current = chosen.target
    if current in seen:
        cycle_edges = cycle_edges[seen[current] :]
        cycle_nodes = path[seen[current] :] + [current]
    else:  # pragma: no cover - defensive fallback for malformed topologies
        cycle_nodes = path + [path[-1]]
    raise SpecInvalid(
        "narrative graph contains a cycle",
        location={"field": "edges", "edge_index": cycle_edges[0].index if cycle_edges else None},
        details={
            "cycle_nodes": cycle_nodes,
            "cycle_edge_ids": [edge.id for edge in cycle_edges],
            "cycle_edge_indices": [edge.index for edge in cycle_edges],
        },
    )
