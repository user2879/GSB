"""Strict YAML specification parser, expression whitelist and DAG validation."""

from __future__ import annotations

import ast
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Mapping

import yaml

from .canonical import sha256_bytes
from .errors import E_GRAPH_CYCLE, E_SPEC_INVALID, BusinessError
from .model import (
    Coefficient,
    Condition,
    Edge,
    Effect,
    Expr,
    Node,
    NumericSpec,
    Spec,
    StateSpec,
)


_TOP_LEVEL_KEYS = {
    "spec_version",
    "seed",
    "start",
    "comparison",
    "boundary_epsilon",
    "path_tolerance",
    "numeric",
    "state",
    "conditions",
    "nodes",
    "edges",
    "metadata",
}
_NUMERIC_KEYS = {
    "coefficient_relative_error",
    "initial_relative_error",
    "unit_roundoff",
    "max_abs_error",
}
_STATE_KEYS = {"components", "initial"}
_CONDITION_KEYS = {"id", "component", "threshold", "direction"}
_NODE_KEYS = {"id", "type", "outcome"}
_EDGE_KEYS = {"id", "source", "from", "target", "to", "choice", "when", "preconditions", "effects"}
_DIRECTION_ALIASES = {
    ">": ">",
    "strict_greater": ">",
    "strict_gt": ">",
    ">=": ">=",
    "greater_equal": ">=",
    "gte": ">=",
    "<": "<",
    "strict_less": "<",
    "strict_lt": "<",
    "<=": "<=",
    "less_equal": "<=",
    "lte": "<=",
    "==": "==",
    "equal": "==",
    "!=": "!=",
    "not_equal": "!=",
}
_NODE_TYPES = {"scene", "choice", "ending"}
_MAX_EXPR_NODES = 256
_MAX_EXPR_DEPTH = 32


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            mark = getattr(key_node, "start_mark", None)
            detail = f"line {mark.line + 1}, column {mark.column + 1}" if mark else "unknown location"
            raise BusinessError(
                E_SPEC_INVALID,
                f"duplicate YAML mapping key {key!r}",
                {"location": detail, "key": str(key)},
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _invalid(message: str, **details: Any) -> BusinessError:
    return BusinessError(E_SPEC_INVALID, message, details)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(f"{path} must be a mapping", path=path)
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise _invalid(f"{path} must be a list", path=path)
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _invalid(f"{path} must be a non-empty string", path=path)
    return value.strip()


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid(f"{path} must be an integer", path=path)
    return value


def _float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid(f"{path} must be a number", path=path)
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise _invalid(f"{path} cannot be represented as a double", path=path) from exc
    if not math.isfinite(result):
        raise _invalid(f"{path} must be finite", path=path)
    return result


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = [str(key) for key in mapping if key not in allowed]
    if unknown:
        raise _invalid(
            f"{path} contains unknown keys",
            path=path,
            unknown=sorted(unknown),
        )


def _parse_direction(value: Any, path: str) -> str:
    text = _string(value, path)
    direction = _DIRECTION_ALIASES.get(text)
    if direction is None:
        raise _invalid(f"{path} uses an unsupported comparison direction", path=path, value=text)
    return direction


def _compile_expr(
    node: ast.AST,
    components: set[str],
    prefix: str,
    counter: list[int],
    depth: int = 0,
) -> Expr:
    if depth > _MAX_EXPR_DEPTH:
        raise _invalid("effect expression nesting is too deep", max_depth=_MAX_EXPR_DEPTH)
    if counter[0] >= _MAX_EXPR_NODES:
        raise _invalid("effect expression has too many nodes", max_nodes=_MAX_EXPR_NODES)
    counter[0] += 1

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _invalid("only numeric constants are allowed in effects")
        try:
            value = float(node.value)
        except (OverflowError, ValueError) as exc:
            raise _invalid("effect constant cannot be represented as a double") from exc
        if not math.isfinite(value):
            raise _invalid("effect constant must be finite")
        key = f"{prefix}/coef[{counter[0] - 1}]"
        return Expr(kind="number", value=value, coefficient_key=key)

    if isinstance(node, ast.Name):
        if node.id not in components:
            raise _invalid("effect expression references an unknown state variable", variable=node.id)
        return Expr(kind="variable", name=node.id)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        child = _compile_expr(node.operand, components, prefix, counter, depth + 1)
        if isinstance(node.op, ast.UAdd):
            return child
        return Expr(kind="negate", left=child)

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
        left = _compile_expr(node.left, components, prefix, counter, depth + 1)
        right = _compile_expr(node.right, components, prefix, counter, depth + 1)
        kind = "add" if isinstance(node.op, ast.Add) else "multiply"
        return Expr(kind=kind, left=left, right=right)

    raise _invalid(
        "effect expression only permits addition, multiplication, numeric constants and state variables",
        ast_node=type(node).__name__,
    )


def _parse_effect(
    text: Any,
    components: set[str],
    edge_id: str,
    effect_index: int,
) -> tuple[Effect, list[Coefficient]]:
    source = _string(text, f"edges[{edge_id}].effects[{effect_index}]")
    if "\n" in source or "\r" in source:
        raise _invalid("effect expressions must be single-line", edge_id=edge_id, effect_index=effect_index)
    try:
        parsed = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise _invalid(
            "effect expression has invalid syntax",
            edge_id=edge_id,
            effect_index=effect_index,
            expression=source,
        ) from exc
    if len(parsed.body) != 1:
        raise _invalid("each effect must be one assignment", edge_id=edge_id, effect_index=effect_index)
    statement = parsed.body[0]

    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise _invalid("effect assignment target must be one state variable", edge_id=edge_id, effect_index=effect_index)
        target = statement.targets[0].id
        expression_node = statement.value
        mode = "assign"
    elif isinstance(statement, ast.AugAssign):
        if not isinstance(statement.target, ast.Name):
            raise _invalid("effect assignment target must be one state variable", edge_id=edge_id, effect_index=effect_index)
        if isinstance(statement.op, ast.Add):
            mode = "add"
        elif isinstance(statement.op, ast.Mult):
            mode = "multiply"
        else:
            raise _invalid("augmented effects only permit += and *=", edge_id=edge_id, effect_index=effect_index)
        target = statement.target.id
        expression_node = statement.value
    else:
        raise _invalid("each effect must be an assignment or augmented assignment", edge_id=edge_id, effect_index=effect_index)

    if target not in components:
        raise _invalid("effect assignment targets an unknown state variable", edge_id=edge_id, effect_index=effect_index, variable=target)

    prefix = f"{edge_id}/effect[{effect_index}]"
    expression = _compile_expr(expression_node, components, prefix, [0])
    effect = Effect(target=target, mode=mode, expression=expression, source=source, effect_index=effect_index)
    coefficients: list[Coefficient] = []

    def collect_ordered(node: Expr) -> None:
        if node.kind == "number":
            assert node.coefficient_key is not None and node.value is not None
            coefficients.append(
                Coefficient(
                    key=node.coefficient_key,
                    edge_id=edge_id,
                    effect_index=effect_index,
                    value=node.value,
                )
            )
            return
        if node.left is not None:
            collect_ordered(node.left)
        if node.right is not None:
            collect_ordered(node.right)

    coefficients.clear()
    collect_ordered(expression)
    return effect, coefficients


def _parse_when(value: Any, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_string(value, path),)
    if isinstance(value, list):
        return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise _invalid(f"{path} must be a condition id or a list of condition ids", path=path)


def _validate_acyclic(nodes: tuple[Node, ...], edges: tuple[Edge, ...]) -> None:
    node_order = [node.id for node in nodes]
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_order}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_order}
    for edge in edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1
    queue = deque(node_id for node_id in node_order if indegree[node_id] == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(node_order):
        remaining = [node_id for node_id in node_order if indegree[node_id] > 0]
        raise BusinessError(
            E_GRAPH_CYCLE,
            "narrative graph contains a cycle",
            {"cycle_nodes": remaining},
        )


def parse_spec(data: str | bytes, source_name: str = "<memory>") -> Spec:
    raw_bytes = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        loaded = yaml.load(raw_bytes, Loader=_UniqueKeyLoader)
    except BusinessError:
        raise
    except yaml.YAMLError as exc:
        raise _invalid("YAML parsing failed", source=source_name, yaml_error=str(exc)) from exc

    root = _mapping(loaded, "$")
    _check_keys(root, _TOP_LEVEL_KEYS, "$")

    spec_version = _integer(root.get("spec_version"), "spec_version")
    if spec_version != 1:
        raise _invalid("unsupported spec_version", spec_version=spec_version)
    seed = _integer(root.get("seed"), "seed")
    start = _string(root.get("start"), "start")
    comparison = _parse_direction(root.get("comparison", ">"), "comparison")
    boundary_epsilon = _float(root.get("boundary_epsilon"), "boundary_epsilon")
    if boundary_epsilon < 0.0:
        raise _invalid("boundary_epsilon must be non-negative")
    path_tolerance = _float(root.get("path_tolerance"), "path_tolerance")
    if path_tolerance < 0.0:
        raise _invalid("path_tolerance must be non-negative")

    numeric_raw = _mapping(root.get("numeric"), "numeric")
    _check_keys(numeric_raw, _NUMERIC_KEYS, "numeric")
    coefficient_relative_error = _float(
        numeric_raw.get("coefficient_relative_error"), "numeric.coefficient_relative_error"
    )
    initial_relative_error = _float(
        numeric_raw.get("initial_relative_error", 0.0), "numeric.initial_relative_error"
    )
    unit_roundoff = _float(numeric_raw.get("unit_roundoff"), "numeric.unit_roundoff")
    max_abs_error = _float(numeric_raw.get("max_abs_error"), "numeric.max_abs_error")
    if coefficient_relative_error < 0.0 or initial_relative_error < 0.0 or unit_roundoff < 0.0:
        raise _invalid("numeric relative errors and unit_roundoff must be non-negative")
    if max_abs_error <= 0.0:
        raise _invalid("numeric.max_abs_error must be positive")
    numeric = NumericSpec(
        coefficient_relative_error=coefficient_relative_error,
        initial_relative_error=initial_relative_error,
        unit_roundoff=unit_roundoff,
        max_abs_error=max_abs_error,
    )

    state_raw = _mapping(root.get("state"), "state")
    _check_keys(state_raw, _STATE_KEYS, "state")
    components_raw = _list(state_raw.get("components"), "state.components")
    if not components_raw:
        raise _invalid("state.components must not be empty")
    components = tuple(_string(item, f"state.components[{index}]") for index, item in enumerate(components_raw))
    if len(set(components)) != len(components):
        raise _invalid("state.components contains duplicates")
    component_set = set(components)
    initial_raw = _mapping(state_raw.get("initial"), "state.initial")
    initial_keys = set(initial_raw)
    unknown_initial = sorted(str(key) for key in initial_keys - component_set)
    missing_initial = [component for component in components if component not in initial_keys]
    if unknown_initial:
        raise _invalid("state.initial contains unknown components", components=unknown_initial)
    if missing_initial:
        raise _invalid("state.initial is missing components", components=missing_initial)
    initial = tuple(_float(initial_raw[component], f"state.initial.{component}") for component in components)
    state = StateSpec(components=components, initial=initial)

    conditions_raw = _list(root.get("conditions", []), "conditions")
    conditions: list[Condition] = []
    condition_ids: set[str] = set()
    for index, item in enumerate(conditions_raw):
        path = f"conditions[{index}]"
        mapping = _mapping(item, path)
        _check_keys(mapping, _CONDITION_KEYS, path)
        condition_id = _string(mapping.get("id"), f"{path}.id")
        if condition_id in condition_ids:
            raise _invalid("duplicate condition id", condition_id=condition_id)
        condition_ids.add(condition_id)
        component = _string(mapping.get("component"), f"{path}.component")
        if component not in component_set:
            raise _invalid("condition references an unknown state component", condition_id=condition_id, component=component)
        # A missing threshold is intentionally distinguished from a bad numeric
        # value so the declared business branch is observable in reports.
        if "threshold" not in mapping or mapping.get("threshold") is None:
            raise _invalid("condition is missing a threshold", condition_id=condition_id)
        threshold = _float(mapping.get("threshold"), f"{path}.threshold")
        direction = _parse_direction(mapping.get("direction", comparison), f"{path}.direction")
        conditions.append(Condition(condition_id, component, threshold, direction))

    nodes_raw = _list(root.get("nodes"), "nodes")
    if not nodes_raw:
        raise _invalid("nodes must not be empty")
    nodes: list[Node] = []
    node_ids: set[str] = set()
    for index, item in enumerate(nodes_raw):
        path = f"nodes[{index}]"
        mapping = _mapping(item, path)
        _check_keys(mapping, _NODE_KEYS, path)
        node_id = _string(mapping.get("id"), f"{path}.id")
        if node_id in node_ids:
            raise _invalid("duplicate node id", node_id=node_id)
        node_ids.add(node_id)
        node_type = _string(mapping.get("type"), f"{path}.type")
        if node_type not in _NODE_TYPES:
            raise _invalid("node type must be scene, choice or ending", node_id=node_id, node_type=node_type)
        outcome_value = mapping.get("outcome")
        outcome = None if outcome_value is None else _string(outcome_value, f"{path}.outcome")
        if node_type == "ending" and outcome is None:
            outcome = node_id
        nodes.append(Node(node_id, node_type, outcome, index))

    if start not in node_ids:
        raise _invalid("start node does not exist", start=start)
    start_node = next(node for node in nodes if node.id == start)
    if start_node.type == "ending":
        raise _invalid("start node cannot be an ending", start=start)

    edges_raw = _list(root.get("edges"), "edges")
    edges: list[Edge] = []
    edge_ids: set[str] = set()
    coefficients: list[Coefficient] = []
    for index, item in enumerate(edges_raw):
        path = f"edges[{index}]"
        mapping = _mapping(item, path)
        _check_keys(mapping, _EDGE_KEYS, path)
        edge_id = _string(mapping.get("id"), f"{path}.id")
        if edge_id in edge_ids:
            raise _invalid("duplicate edge id", edge_id=edge_id)
        edge_ids.add(edge_id)
        if "source" in mapping and "from" in mapping:
            raise _invalid("edge cannot declare both source and from", edge_id=edge_id)
        if "target" in mapping and "to" in mapping:
            raise _invalid("edge cannot declare both target and to", edge_id=edge_id)
        source = _string(mapping.get("source", mapping.get("from")), f"{path}.source")
        target = _string(mapping.get("target", mapping.get("to")), f"{path}.target")
        if source not in node_ids:
            raise _invalid("edge source node does not exist", edge_id=edge_id, node_id=source)
        if target not in node_ids:
            raise _invalid("edge target node does not exist", edge_id=edge_id, node_id=target)
        choice = _string(mapping.get("choice"), f"{path}.choice")
        if "when" in mapping and "preconditions" in mapping:
            raise _invalid("edge cannot declare both when and preconditions", edge_id=edge_id)
        when_value = mapping.get("when", mapping.get("preconditions"))
        when = _parse_when(when_value, f"{path}.when")
        for condition_id in when:
            if condition_id not in condition_ids:
                raise _invalid("edge references an unknown condition", edge_id=edge_id, condition_id=condition_id)
        effects_raw = _list(mapping.get("effects", []), f"{path}.effects")
        effects: list[Effect] = []
        for effect_index, effect_text in enumerate(effects_raw):
            effect, effect_coefficients = _parse_effect(effect_text, component_set, edge_id, effect_index)
            effects.append(effect)
            coefficients.extend(effect_coefficients)
        edges.append(Edge(edge_id, source, target, choice, when, tuple(effects), index))

    edge_tuple = tuple(edges)
    node_tuple = tuple(nodes)
    outgoing: dict[str, list[Edge]] = defaultdict(list)
    for edge in edge_tuple:
        outgoing[edge.source].append(edge)
    ending_ids = {node.id for node in node_tuple if node.type == "ending"}
    for node_id in ending_ids:
        if outgoing.get(node_id):
            raise _invalid("ending node cannot have outgoing edges", node_id=node_id)

    # A deterministic conditional route may reuse one choice token, but only
    # one unconditional candidate is permitted for that token.
    for node_id, node_edges in outgoing.items():
        grouped: dict[str, list[Edge]] = defaultdict(list)
        for edge in node_edges:
            grouped[edge.choice].append(edge)
        for choice, candidates in grouped.items():
            unconditional = [edge.id for edge in candidates if not edge.when]
            if len(unconditional) > 1:
                raise _invalid(
                    "duplicate unconditional choice token at one node",
                    node_id=node_id,
                    choice=choice,
                    edge_ids=unconditional,
                )

    _validate_acyclic(node_tuple, edge_tuple)

    return Spec(
        spec_version=spec_version,
        seed=seed,
        start=start,
        comparison=comparison,
        boundary_epsilon=boundary_epsilon,
        path_tolerance=path_tolerance,
        numeric=numeric,
        state=state,
        conditions=tuple(conditions),
        nodes=node_tuple,
        edges=edge_tuple,
        coefficients=tuple(coefficients),
        source_sha256=sha256_bytes(raw_bytes),
    )


def load_spec(path: str | Path) -> Spec:
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise _invalid("cannot read narrative specification", path=str(source), error=str(exc)) from exc
    return parse_spec(data, source_name=str(source))
