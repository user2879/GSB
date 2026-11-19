"""Multi-path merge consistency checks without choosing a privileged path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .canonical import double, sha256_json, state_hash
from .engine import (
    Arbitration,
    CausalChain,
    apply_edge,
    evaluate_edge_conditions,
    initial_causal_chains,
)
from .model import Edge, NarrativeSpec


@dataclass(frozen=True)
class PathTrace:
    path_index: int
    edge_sequence: tuple[str, ...]
    state: tuple[float, ...]
    state_hash: str
    condition_feasible: bool
    boundary: bool
    conditions: tuple[dict[str, Any], ...]
    steps: tuple[dict[str, Any], ...]
    causal_chains: tuple[CausalChain, ...]
    arbitrations: tuple[Arbitration, ...]

    def as_dict(self, spec: NarrativeSpec) -> dict[str, Any]:
        return {
            "path_index": self.path_index,
            "edge_sequence": list(self.edge_sequence),
            "edge_indices": [spec.edge_by_id(edge_id).index for edge_id in self.edge_sequence],
            "state_vector": [double(value) for value in self.state],
            "state": {
                component: double(self.state[index])
                for index, component in enumerate(spec.components)
            },
            "state_hash": self.state_hash,
            "condition_feasible": self.condition_feasible,
            "boundary": self.boundary,
            "conditions": [dict(item) for item in self.conditions],
            "steps": [dict(item) for item in self.steps],
            "causal_chains": [chain.as_dict() for chain in self.causal_chains],
            "arbitrations": [arbitration.as_dict() for arbitration in self.arbitrations],
        }


def _enumerate_paths(spec: NarrativeSpec, target_node: str, limit: int = 4096) -> tuple[tuple[Edge, ...], ...]:
    paths: list[tuple[Edge, ...]] = []
    stack: list[tuple[str, tuple[Edge, ...], frozenset[str]]] = [(spec.start, (), frozenset({spec.start}))]
    while stack:
        node_id, path, visited = stack.pop()
        if node_id == target_node:
            paths.append(path)
            continue
        outgoing = spec.outgoing(node_id)
        for edge in reversed(outgoing):
            if edge.target in visited:
                continue
            stack.append((edge.target, (*path, edge), visited | {edge.target}))
        if len(paths) + len(stack) > limit:
            raise ValueError(f"too many paths for merge node {target_node!r}; limit={limit}")
    # DFS stack traversal above already emits declaration order after reversal,
    # but sort by edge index tuples anyway so the result is independent of stack details.
    paths.sort(key=lambda path: tuple(edge.index for edge in path))
    return tuple(paths)


def _trace_path(spec: NarrativeSpec, path_index: int, edges: Sequence[Edge]) -> PathTrace:
    state = list(spec.initial_state)
    chains = initial_causal_chains(spec.components, state)
    all_conditions: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    arbitration_list: list[Arbitration] = []
    feasible = True
    boundary = False

    for step_index, edge in enumerate(edges):
        judgments = evaluate_edge_conditions(spec, edge, state)
        condition_payload = [
            {
                "step_index": step_index,
                "edge_id": edge.id,
                "edge_index": edge.index,
                **judgment.as_dict(),
            }
            for judgment in judgments
        ]
        all_conditions.extend(condition_payload)
        if any(judgment.status == "unsatisfied" for judgment in judgments):
            feasible = False
        if any(judgment.status == "boundary" for judgment in judgments):
            boundary = True
            feasible = False
        before_state = tuple(state)
        application = apply_edge(spec, edge, state, chains)
        arbitration_list.extend(application.arbitrations)
        state = list(application.state)
        chains = application.causal_chains
        steps.append(
            {
                "step_index": step_index,
                "edge_id": edge.id,
                "edge_index": edge.index,
                "state_before": {
                    component: before_state[index]
                    for index, component in enumerate(spec.components)
                },
                "state_after_vector": [double(value) for value in state],
                "effect_runs": [run.as_dict() for run in application.effect_runs],
                "effective_runs": [run.as_dict() for run in application.effective_runs],
                "arbitrations": [arbitration.as_dict() for arbitration in application.arbitrations],
            }
        )

    return PathTrace(
        path_index=path_index,
        edge_sequence=tuple(edge.id for edge in edges),
        state=tuple(double(value) for value in state),
        state_hash=state_hash(spec.components, state),
        condition_feasible=feasible,
        boundary=boundary,
        conditions=tuple(all_conditions),
        steps=tuple(steps),
        causal_chains=chains,
        arbitrations=tuple(arbitration_list),
    )


def check_merge_node(spec: NarrativeSpec, node_id: str) -> dict[str, Any]:
    paths = _enumerate_paths(spec, node_id)
    traces = tuple(_trace_path(spec, index, path) for index, path in enumerate(paths))
    feasible_traces = tuple(trace for trace in traces if trace.condition_feasible)
    comparisons: list[dict[str, Any]] = []
    divergent_pairs: list[dict[str, Any]] = []
    first_divergence: dict[str, Any] | None = None

    for left_index in range(len(feasible_traces)):
        left = feasible_traces[left_index]
        for right_index in range(left_index + 1, len(feasible_traces)):
            right = feasible_traces[right_index]
            component_diffs: list[dict[str, Any]] = []
            pair_divergence: dict[str, Any] | None = None
            for component_index, component in enumerate(spec.components):
                left_value = double(left.state[component_index])
                right_value = double(right.state[component_index])
                difference = abs(double(left_value - right_value))
                within = difference <= spec.merge_tolerance
                diff_payload = {
                    "component": component,
                    "component_index": component_index,
                    "path_a_value": left_value,
                    "path_b_value": right_value,
                    "absolute_difference": difference,
                    "tolerance": double(spec.merge_tolerance),
                    "within_tolerance": within,
                }
                component_diffs.append(diff_payload)
                if not within and pair_divergence is None:
                    pair_divergence = {
                        "component": component,
                        "component_index": component_index,
                        "path_a_index": left.path_index,
                        "path_b_index": right.path_index,
                        "path_a_edges": list(left.edge_sequence),
                        "path_b_edges": list(right.edge_sequence),
                        "path_a_state": {
                            spec.components[index]: double(left.state[index])
                            for index in range(len(spec.components))
                        },
                        "path_b_state": {
                            spec.components[index]: double(right.state[index])
                            for index in range(len(spec.components))
                        },
                        "absolute_difference": difference,
                        "tolerance": double(spec.merge_tolerance),
                        "recompute": {
                            "path_a_edges": list(left.edge_sequence),
                            "path_b_edges": list(right.edge_sequence),
                            "formula": "abs(path_a_state[component] - path_b_state[component])",
                        },
                    }
            pair_payload = {
                "path_a_index": left.path_index,
                "path_b_index": right.path_index,
                "path_a_edges": list(left.edge_sequence),
                "path_b_edges": list(right.edge_sequence),
                "component_differences": component_diffs,
                "within_tolerance": pair_divergence is None,
            }
            comparisons.append(pair_payload)
            if pair_divergence is not None:
                divergent_pairs.append(pair_payload)
                if first_divergence is None:
                    first_divergence = pair_divergence

    if len(feasible_traces) < 2:
        status = "insufficient-feasible-paths"
    elif first_divergence is None:
        status = "consistent"
    else:
        status = "divergent"

    report: dict[str, Any] = {
        "schema": "narrative-workbench/merge-check/v1",
        "target_node": node_id,
        "tolerance": double(spec.merge_tolerance),
        "path_count": len(traces),
        "feasible_path_count": len(feasible_traces),
        "status": status,
        "paths": [trace.as_dict(spec) for trace in traces],
        "pairwise_differences": comparisons,
        "first_divergence": first_divergence,
        "method": "enumerate every simple start-to-merge path, accumulate independently, compare every feasible pair",
    }
    report["report_hash"] = sha256_json(report)
    return report


def check_all_merge_nodes(spec: NarrativeSpec) -> dict[str, Any]:
    indegree: dict[str, int] = {node.id: 0 for node in spec.nodes}
    for edge in spec.edges:
        indegree[edge.target] += 1
    merge_nodes = tuple(node.id for node in spec.nodes if indegree[node.id] > 1)
    reports = [check_merge_node(spec, node_id) for node_id in merge_nodes]
    payload: dict[str, Any] = {
        "schema": "narrative-workbench/merge-report/v1",
        "spec_id": spec.id,
        "spec_hash": spec.spec_hash,
        "merge_nodes": [
            {
                "node_id": node_id,
                "indegree": indegree[node_id],
            }
            for node_id in merge_nodes
        ],
        "checks": reports,
    }
    payload["report_hash"] = sha256_json(payload)
    return payload
