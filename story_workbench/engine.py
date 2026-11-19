"""Deterministic numeric execution, condition margins, and conflict arbitration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Sequence

from .canonical import add_double, double
from .errors import E_STATE_NONFINITE, WorkbenchError
from .expressions import evaluate_expression
from .model import Condition, Edge, Effect, NarrativeSpec, state_mapping


@dataclass(frozen=True)
class ConditionJudgment:
    condition_id: str
    component: str
    op: str
    value: float
    threshold: float
    signed_margin: float
    directed_margin: float
    distance: float
    boundary: bool
    satisfied_by_value: bool
    status: str
    threshold_key: str
    declaration_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "component": self.component,
            "op": self.op,
            "value": double(self.value),
            "threshold": double(self.threshold),
            "signed_margin": double(self.signed_margin),
            "directed_margin": double(self.directed_margin),
            "distance": double(self.distance),
            "boundary": self.boundary,
            "satisfied_by_value": self.satisfied_by_value,
            "status": self.status,
            "threshold_key": self.threshold_key,
            "declaration_index": self.declaration_index,
        }


@dataclass(frozen=True)
class BranchEvaluation:
    edge_id: str
    edge_index: int
    target_node: str
    label: str
    conditions: tuple[ConditionJudgment, ...]
    status: str
    reason: str

    @property
    def reachable(self) -> bool:
        return self.status == "reachable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_index": self.edge_index,
            "target_node": self.target_node,
            "label": self.label,
            "conditions": [condition.as_dict() for condition in self.conditions],
            "status": self.status,
            "reachable": self.reachable,
            "boundary": self.status == "boundary",
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EffectRun:
    edge_id: str
    edge_index: int
    effect_id: str
    effect_index: int
    target: str
    expression: str
    delta: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_index": self.edge_index,
            "effect_id": self.effect_id,
            "effect_index": self.effect_index,
            "target": self.target,
            "expression": self.expression,
            "delta": double(self.delta),
        }


@dataclass(frozen=True)
class Arbitration:
    scope: str
    edge_id: str
    target: str
    reason: str
    winner_effect_id: str
    winner_effect_index: int
    overwritten_effect_id: str
    overwritten_effect_index: int
    overwritten_value: float
    effective_value: float
    declaration_order: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "edge_id": self.edge_id,
            "target": self.target,
            "reason": self.reason,
            "winner_effect_id": self.winner_effect_id,
            "winner_effect_index": self.winner_effect_index,
            "overwritten_effect_id": self.overwritten_effect_id,
            "overwritten_effect_index": self.overwritten_effect_index,
            "overwritten_value": double(self.overwritten_value),
            "effective_value": double(self.effective_value),
            "declaration_order": list(self.declaration_order),
        }


@dataclass(frozen=True)
class CausalContribution:
    edge_id: str
    edge_index: int
    effect_id: str
    effect_index: int
    before: float
    delta: float
    after: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "edge_index": self.edge_index,
            "effect_id": self.effect_id,
            "effect_index": self.effect_index,
            "before": double(self.before),
            "delta": double(self.delta),
            "after": double(self.after),
        }


@dataclass(frozen=True)
class CausalChain:
    component: str
    initial: float
    value: float
    contributions: tuple[CausalContribution, ...]

    def edge_ids(self) -> tuple[str, ...]:
        return tuple(contribution.edge_id for contribution in self.contributions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "initial": double(self.initial),
            "value": double(self.value),
            "edge_ids": list(self.edge_ids()),
            "contributions": [contribution.as_dict() for contribution in self.contributions],
        }


@dataclass(frozen=True)
class GroupApplication:
    state: tuple[float, ...]
    causal_chains: tuple[CausalChain, ...]
    effect_runs: tuple[EffectRun, ...]
    effective_runs: tuple[EffectRun, ...]
    arbitrations: tuple[Arbitration, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": [double(value) for value in self.state],
            "effect_runs": [run.as_dict() for run in self.effect_runs],
            "effective_runs": [run.as_dict() for run in self.effective_runs],
            "arbitrations": [arbitration.as_dict() for arbitration in self.arbitrations],
        }


def initial_causal_chains(components: Sequence[str], values: Sequence[float]) -> tuple[CausalChain, ...]:
    return tuple(
        CausalChain(component=component, initial=double(values[index]), value=double(values[index]), contributions=())
        for index, component in enumerate(components)
    )


def evaluate_condition(condition: Condition, value: float, boundary_tolerance: float) -> ConditionJudgment:
    current = double(value)
    threshold = double(condition.threshold)
    signed_margin = double(current - threshold)
    if condition.op in {">=", ">"}:
        directed_margin = signed_margin
        satisfied = signed_margin >= 0.0 if condition.op == ">=" else signed_margin > 0.0
    elif condition.op in {"<=", "<"}:
        directed_margin = double(-signed_margin)
        satisfied = directed_margin >= 0.0 if condition.op == "<=" else directed_margin > 0.0
    else:  # Parser guarantees this branch is unreachable.
        raise ValueError(f"unsupported condition op {condition.op!r}")
    boundary = abs(signed_margin) < boundary_tolerance
    if boundary:
        status = "boundary"
    elif satisfied:
        status = "satisfied"
    else:
        status = "unsatisfied"
    return ConditionJudgment(
        condition_id=condition.id,
        component=condition.component,
        op=condition.op,
        value=current,
        threshold=threshold,
        signed_margin=signed_margin,
        directed_margin=directed_margin,
        distance=abs(signed_margin),
        boundary=boundary,
        satisfied_by_value=satisfied,
        status=status,
        threshold_key=condition.threshold_key,
        declaration_index=condition.declaration_index,
    )


def evaluate_edge_conditions(
    spec: NarrativeSpec,
    edge: Edge,
    state_values: Sequence[float],
) -> tuple[ConditionJudgment, ...]:
    state = state_mapping(spec.components, state_values)
    return tuple(
        evaluate_condition(condition, state[condition.component], spec.boundary_tolerance)
        for condition in edge.conditions
    )


def evaluate_branches(spec: NarrativeSpec, node_id: str, state_values: Sequence[float]) -> tuple[BranchEvaluation, ...]:
    branches: list[BranchEvaluation] = []
    for edge in spec.outgoing(node_id):
        judgments = evaluate_edge_conditions(spec, edge, state_values)
        if not judgments:
            status = "reachable"
            reason = "no-conditions"
        elif any(judgment.status == "unsatisfied" for judgment in judgments):
            status = "unreachable"
            reason = "hard-condition-failed"
        elif any(judgment.status == "boundary" for judgment in judgments):
            status = "boundary"
            reason = "condition-on-boundary"
        else:
            status = "reachable"
            reason = "all-conditions-satisfied"
        branches.append(
            BranchEvaluation(
                edge_id=edge.id,
                edge_index=edge.index,
                target_node=edge.target,
                label=edge.label,
                conditions=judgments,
                status=status,
                reason=reason,
            )
        )
    return tuple(branches)


def _effect_refs_for_edges(edges: Iterable[Edge]) -> tuple[tuple[Edge, Effect], ...]:
    refs: list[tuple[Edge, Effect]] = []
    for edge in edges:
        for effect in edge.effects:
            refs.append((edge, effect))
    return tuple(refs)


def apply_effect_group(
    spec: NarrativeSpec,
    edges: Sequence[Edge],
    state_values: Sequence[float],
    causal_chains: Sequence[CausalChain],
    *,
    scope: str = "edge",
) -> GroupApplication:
    """Apply effects using pre-group RHS values and component-level last writer.

    Every effect RHS is evaluated against the immutable pre-group state in
    declaration order (edge order, then effect order for a multi-edge group).
    Only the last declaration targeting a component is effective. Effective
    winners are then applied in declaration order. No dict/set iteration is
    used for arithmetic or output ordering.
    """

    if len(state_values) != len(spec.components):
        raise ValueError("state length does not match component count")
    if len(causal_chains) != len(spec.components):
        raise ValueError("causal chain length does not match component count")

    pre_state = state_mapping(spec.components, state_values)
    refs = _effect_refs_for_edges(edges)
    runs: list[EffectRun] = []
    for edge, effect in refs:
        def _nonfinite(edge: Edge = edge, effect: Effect = effect) -> None:
            raise WorkbenchError(
                E_STATE_NONFINITE,
                f"effect {effect.id!r} on edge {edge.id!r} produced a non-finite double",
                location={"edge_index": edge.index, "edge_id": edge.id, "effect_id": effect.id},
            )

        try:
            delta = evaluate_expression(effect.ast, pre_state, nonfinite_callback=_nonfinite)
        except WorkbenchError:
            raise
        except (ValueError, OverflowError) as exc:
            _nonfinite()
            raise AssertionError("unreachable") from exc
        runs.append(
            EffectRun(
                edge_id=edge.id,
                edge_index=edge.index,
                effect_id=effect.id,
                effect_index=effect.declaration_index,
                target=effect.target,
                expression=effect.expression,
                delta=delta,
            )
        )

    winner_by_component: dict[str, int] = {}
    for run_index, run in enumerate(runs):
        winner_by_component[run.target] = run_index

    output_state = [double(value) for value in state_values]
    output_chains = list(causal_chains)
    component_positions = {component: index for index, component in enumerate(spec.components)}
    arbitrations: list[Arbitration] = []

    declaration_order = tuple(
        {
            "edge_id": run.edge_id,
            "edge_index": run.edge_index,
            "effect_id": run.effect_id,
            "effect_index": run.effect_index,
        }
        for run in runs
    )

    effective_runs: list[EffectRun] = []
    for run_index, run in enumerate(runs):
        winner_index = winner_by_component[run.target]
        winner = runs[winner_index]
        if winner_index != run_index:
            arbitrations.append(
                Arbitration(
                    scope=scope,
                    edge_id=run.edge_id,
                    target=run.target,
                    reason="component-last-writer",
                    winner_effect_id=winner.effect_id,
                    winner_effect_index=winner.effect_index,
                    overwritten_effect_id=run.effect_id,
                    overwritten_effect_index=run.effect_index,
                    overwritten_value=run.delta,
                    effective_value=winner.delta,
                    declaration_order=declaration_order,
                )
            )
            continue

        component_index = component_positions[run.target]
        before = output_state[component_index]
        after = add_double(before, run.delta)
        output_state[component_index] = after
        chain = output_chains[component_index]
        contribution = CausalContribution(
            edge_id=run.edge_id,
            edge_index=run.edge_index,
            effect_id=run.effect_id,
            effect_index=run.effect_index,
            before=before,
            delta=run.delta,
            after=after,
        )
        output_chains[component_index] = replace(
            chain,
            value=after,
            contributions=(*chain.contributions, contribution),
        )
        effective_runs.append(run)

    return GroupApplication(
        state=tuple(output_state),
        causal_chains=tuple(output_chains),
        effect_runs=tuple(runs),
        effective_runs=tuple(effective_runs),
        arbitrations=tuple(arbitrations),
    )


def apply_edge(
    spec: NarrativeSpec,
    edge: Edge,
    state_values: Sequence[float],
    causal_chains: Sequence[CausalChain],
) -> GroupApplication:
    return apply_effect_group(spec, (edge,), state_values, causal_chains, scope="edge")


def project_node_outgoing_effects(
    spec: NarrativeSpec,
    node_id: str,
    state_values: Sequence[float],
) -> dict[str, Any]:
    """Diagnostic projection for simultaneous outgoing-edge effect stacking.

    Branches are mutually exclusive in the normal timeline. This projection
    still defines and records what happens if a caller stacks all outgoing
    effects at one node: edge declaration order then effect declaration order,
    with component-level last-writer arbitration.
    """

    edges = spec.outgoing(node_id)
    chains = initial_causal_chains(spec.components, state_values)
    application = apply_effect_group(spec, edges, state_values, chains, scope="node-outgoing")
    return {
        "node_id": node_id,
        "edge_order": [edge.id for edge in edges],
        "projected_state": {component: application.state[index] for index, component in enumerate(spec.components)},
        "arbitrations": [arbitration.as_dict() for arbitration in application.arbitrations],
        "effective_effects": [run.as_dict() for run in application.effective_runs],
        "all_effect_runs": [run.as_dict() for run in application.effect_runs],
    }


def causal_chain_payload(chains: Sequence[CausalChain]) -> list[dict[str, Any]]:
    return [chain.as_dict() for chain in chains]


def condition_boundary_payload(branches: Sequence[BranchEvaluation]) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for branch in branches:
        for judgment in branch.conditions:
            if judgment.boundary:
                boundaries.append(
                    {
                        "edge_id": branch.edge_id,
                        "edge_index": branch.edge_index,
                        "branch_status": branch.status,
                        **judgment.as_dict(),
                    }
                )
    return boundaries
