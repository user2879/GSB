"""Branch-merge path consistency auditing."""

from __future__ import annotations

from typing import Any, Mapping

from .engine import (
    _State,
    _apply_effect,
    _check_error_bound,
    _evaluate_condition,
    _state_payload,
)
from .errors import E_PATH_DIVERGE, BusinessError
from .model import Edge, Spec


def _enumerate_paths(spec: Spec, target: str, max_paths: int) -> list[tuple[Edge, ...]]:
    outgoing = spec.outgoing()
    paths: list[tuple[Edge, ...]] = []
    stack_path: list[Edge] = []

    def visit(node_id: str) -> None:
        if len(paths) > max_paths:
            raise BusinessError(
                E_PATH_DIVERGE,
                "path enumeration exceeded the configured limit",
                {"target_node": target, "max_paths": max_paths},
            )
        if node_id == target:
            paths.append(tuple(stack_path))
            return
        for edge in outgoing[node_id]:
            stack_path.append(edge)
            visit(edge.target)
            stack_path.pop()

    visit(spec.start)
    return paths


def _run_edges(
    spec: Spec,
    edges: tuple[Edge, ...],
    component_index: Mapping[str, int],
) -> _State:
    state = _State(list(spec.state.initial), [])
    for index, component in enumerate(spec.state.components):
        state.errors.append(abs(state.values[index]) * spec.numeric.initial_relative_error)
        _check_error_bound(
            state.values[index],
            state.errors[index],
            spec,
            edge_id="<initial>",
            effect_index=None,
            component=component,
        )
    for edge in edges:
        for effect_index in range(len(edge.effects)):
            _apply_effect(edge, effect_index, state, component_index, spec, {})
    return state


def _outgoing_decisions(
    spec: Spec,
    node_id: str,
    state: _State,
    component_index: Mapping[str, int],
) -> list[dict[str, Any]]:
    condition_map = spec.condition_map()
    result: list[dict[str, Any]] = []
    for edge in spec.outgoing()[node_id]:
        for condition_id in edge.when:
            condition = condition_map[condition_id]
            index = component_index[condition.component]
            result.append(
                _evaluate_condition(
                    condition,
                    state.values[index],
                    state.errors[index],
                    spec,
                    {},
                    step_index=-1,
                    from_node=node_id,
                    edge_id=edge.id,
                )
            )
    return result


def analyze_paths(
    spec: Spec,
    *,
    max_paths: int = 1024,
    raise_on_diverge: bool = False,
) -> dict[str, Any]:
    component_index = {component: index for index, component in enumerate(spec.state.components)}
    indegree = {node.id: 0 for node in spec.nodes}
    for edge in spec.edges:
        indegree[edge.target] += 1
    join_nodes = [node.id for node in spec.nodes if indegree[node.id] >= 2]

    report_joins: list[dict[str, Any]] = []
    overall_diverge = False
    overall_first: dict[str, Any] | None = None

    for join_node in join_nodes:
        paths = _enumerate_paths(spec, join_node, max_paths)
        if len(paths) < 2:
            continue
        path_states = [_run_edges(spec, path, component_index) for path in paths]
        path_decisions = [
            _outgoing_decisions(spec, join_node, state, component_index)
            for state in path_states
        ]
        path_reports: list[dict[str, Any]] = []
        for path_index, (path, state, decisions) in enumerate(zip(paths, path_states, path_decisions)):
            path_reports.append(
                {
                    "path_index": path_index,
                    "path_id": f"{join_node}:path[{path_index}]",
                    "edge_ids": [edge.id for edge in path],
                    "state": _state_payload(spec, state),
                    "outgoing_decisions": decisions,
                }
            )

        pair_reports: list[dict[str, Any]] = []
        for left_index in range(len(paths)):
            for right_index in range(left_index + 1, len(paths)):
                left_state = path_states[left_index]
                right_state = path_states[right_index]
                component_diffs: list[dict[str, Any]] = []
                first_component: dict[str, Any] | None = None
                for component_index_value, component in enumerate(spec.state.components):
                    left_value = left_state.values[component_index_value]
                    right_value = right_state.values[component_index_value]
                    difference = left_value - right_value
                    over_tolerance = abs(difference) > spec.path_tolerance
                    entry = {
                        "component": component,
                        "left_value": left_value,
                        "right_value": right_value,
                        "difference": difference,
                        "absolute_difference": abs(difference),
                        "left_error_upper": left_state.errors[component_index_value],
                        "right_error_upper": right_state.errors[component_index_value],
                        "over_tolerance": over_tolerance,
                    }
                    component_diffs.append(entry)
                    if over_tolerance and first_component is None:
                        first_component = entry

                left_decisions = {f"{d['edge_id']}|{d['condition_id']}": d for d in path_decisions[left_index]}
                right_decisions = {f"{d['edge_id']}|{d['condition_id']}": d for d in path_decisions[right_index]}
                judgment_diffs: list[dict[str, Any]] = []
                judgment_comparisons: list[dict[str, Any]] = []
                for key in sorted(set(left_decisions) | set(right_decisions)):
                    left_decision = left_decisions.get(key)
                    right_decision = right_decisions.get(key)
                    changed = (
                        left_decision is None
                        or right_decision is None
                        or left_decision["reachable"] != right_decision["reachable"]
                        or left_decision["uncertain"] != right_decision["uncertain"]
                    )
                    comparison = {
                        "judgment": key,
                        "left_present": left_decision is not None,
                        "right_present": right_decision is not None,
                        "left_reachable": None if left_decision is None else left_decision["reachable"],
                        "right_reachable": None if right_decision is None else right_decision["reachable"],
                        "left_uncertain": None if left_decision is None else left_decision["uncertain"],
                        "right_uncertain": None if right_decision is None else right_decision["uncertain"],
                        "left_signed_margin": None if left_decision is None else left_decision["signed_margin"],
                        "right_signed_margin": None if right_decision is None else right_decision["signed_margin"],
                        "changed": changed,
                    }
                    judgment_comparisons.append(comparison)
                    if changed:
                        judgment_diffs.append(comparison)

                pair_diverge = first_component is not None
                if pair_diverge:
                    overall_diverge = True
                    if overall_first is None:
                        overall_first = {
                            "join_node": join_node,
                            "component": first_component["component"],
                            "left_path_id": f"{join_node}:path[{left_index}]",
                            "right_path_id": f"{join_node}:path[{right_index}]",
                            "left_edge_ids": [edge.id for edge in paths[left_index]],
                            "right_edge_ids": [edge.id for edge in paths[right_index]],
                            "left_value": first_component["left_value"],
                            "right_value": first_component["right_value"],
                            "difference": first_component["difference"],
                            "absolute_difference": first_component["absolute_difference"],
                            "tolerance": spec.path_tolerance,
                        }
                pair_reports.append(
                    {
                        "left_path_id": f"{join_node}:path[{left_index}]",
                        "right_path_id": f"{join_node}:path[{right_index}]",
                        "diverge": pair_diverge,
                        "first_divergent_component": None if first_component is None else first_component["component"],
                        "component_diffs": component_diffs,
                        "judgment_comparisons": judgment_comparisons,
                        "judgment_diffs": judgment_diffs,
                    }
                )

        report_joins.append(
            {
                "join_node": join_node,
                "path_count": len(paths),
                "paths": path_reports,
                "pairwise": pair_reports,
            }
        )

    business_code = E_PATH_DIVERGE if overall_diverge else "OK"
    report = {
        "business_code": business_code,
        "spec_sha256": spec.source_sha256,
        "path_tolerance": spec.path_tolerance,
        "join_nodes": report_joins,
        "first_divergence": overall_first,
    }
    if overall_diverge and raise_on_diverge:
        raise BusinessError(
            E_PATH_DIVERGE,
            "branch paths differ beyond the declared tolerance",
            {"first_divergence": overall_first},
        )
    return report
