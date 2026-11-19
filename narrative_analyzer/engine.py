"""Deterministic double-precision replay and error-bound propagation."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .canonical import canonical_bytes, exact_float, float_bits, sha256_value
from .errors import (
    E_INJECT_CONFIG,
    E_PRECISION_BOUND,
    E_SELECTION_UNKNOWN,
    BusinessError,
)
from .model import Condition, Edge, Expr, Spec


_REPLAY_DOMAIN = b"narrative-replay-v1\n"


@dataclass(frozen=True)
class ReplayOptions:
    """Parameter-layer overrides. A spec object is never mutated."""

    coefficient_overrides: Mapping[str, float] = field(default_factory=dict)
    threshold_offsets: Mapping[str, float] = field(default_factory=dict)
    initial_offsets: Mapping[str, float] = field(default_factory=dict)
    enforce_reachability: bool = True

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "coefficient_overrides": {key: self.coefficient_overrides[key] for key in sorted(self.coefficient_overrides)},
            "threshold_offsets": {key: self.threshold_offsets[key] for key in sorted(self.threshold_offsets)},
            "initial_offsets": {key: self.initial_offsets[key] for key in sorted(self.initial_offsets)},
            "enforce_reachability": self.enforce_reachability,
        }


@dataclass
class _State:
    values: list[float]
    errors: list[float]

    def clone(self) -> "_State":
        return _State(list(self.values), list(self.errors))


@dataclass(frozen=True)
class _Bound:
    value: float
    propagated: float = 0.0
    coefficient: float = 0.0
    rounding: float = 0.0

    @property
    def total(self) -> float:
        return self.propagated + self.coefficient + self.rounding


@dataclass
class _BranchEvaluation:
    edge: Edge
    decisions: list[dict[str, Any]]
    reachable: bool
    uncertain: bool
    status: str


def _precision_error(message: str, **details: Any) -> BusinessError:
    return BusinessError(E_PRECISION_BOUND, message, details)


def _ensure_finite(value: float, *, what: str, **details: Any) -> None:
    if not math.isfinite(value):
        raise _precision_error(f"{what} is not finite", what=what, value=repr(value), **details)


def _check_error_bound(
    value: float,
    error: float,
    spec: Spec,
    *,
    edge_id: str,
    effect_index: int | None,
    component: str,
) -> None:
    _ensure_finite(value, what="state value", edge_id=edge_id, effect_index=effect_index, component=component)
    _ensure_finite(error, what="state error bound", edge_id=edge_id, effect_index=effect_index, component=component)
    if error > spec.numeric.max_abs_error:
        raise _precision_error(
            "state error bound exceeds numeric.max_abs_error",
            edge_id=edge_id,
            effect_index=effect_index,
            component=component,
            error_upper=error,
            max_abs_error=spec.numeric.max_abs_error,
        )


def _coef_value(coefficient_key: str, base: float, overrides: Mapping[str, float]) -> float:
    return float(overrides.get(coefficient_key, base))


def _eval_expr(
    expr: Expr,
    state: _State,
    component_index: Mapping[str, int],
    spec: Spec,
    coefficient_overrides: Mapping[str, float],
) -> _Bound:
    if expr.kind == "number":
        assert expr.value is not None and expr.coefficient_key is not None
        value = _coef_value(expr.coefficient_key, expr.value, coefficient_overrides)
        _ensure_finite(value, what="coefficient", coefficient=expr.coefficient_key)
        coefficient_error = abs(value) * spec.numeric.coefficient_relative_error
        return _Bound(value=value, coefficient=coefficient_error)
    if expr.kind == "variable":
        assert expr.name is not None
        index = component_index[expr.name]
        return _Bound(
            value=state.values[index],
            propagated=state.errors[index],
        )
    if expr.kind == "negate":
        assert expr.left is not None
        child = _eval_expr(expr.left, state, component_index, spec, coefficient_overrides)
        return _Bound(
            value=-child.value,
            propagated=child.propagated,
            coefficient=child.coefficient,
            rounding=child.rounding,
        )
    if expr.kind == "add":
        assert expr.left is not None and expr.right is not None
        left = _eval_expr(expr.left, state, component_index, spec, coefficient_overrides)
        right = _eval_expr(expr.right, state, component_index, spec, coefficient_overrides)
        value = left.value + right.value
        _ensure_finite(value, what="addition result")
        rounding = left.rounding + right.rounding + spec.numeric.unit_roundoff * abs(value)
        return _Bound(
            value=value,
            propagated=left.propagated + right.propagated,
            coefficient=left.coefficient + right.coefficient,
            rounding=rounding,
        )
    if expr.kind == "multiply":
        assert expr.left is not None and expr.right is not None
        left = _eval_expr(expr.left, state, component_index, spec, coefficient_overrides)
        right = _eval_expr(expr.right, state, component_index, spec, coefficient_overrides)
        value = left.value * right.value
        _ensure_finite(value, what="multiplication result")
        # Conservative upper bound for |x*y| representation error:
        # |x|Ey + |y|Ex + Ex*Ey, plus one IEEE-754 rounded operation.
        # The category split below intentionally over-approximates the
        # coefficient/cross terms so each reported component remains an upper
        # bound after chained operations.
        propagated = abs(left.value) * right.propagated + abs(right.value) * left.propagated
        coefficient = (
            abs(left.value) * right.coefficient
            + abs(right.value) * left.coefficient
            + left.coefficient * right.coefficient
        )
        rounding = (
            abs(left.value) * right.rounding
            + abs(right.value) * left.rounding
            + left.total * right.total
            + spec.numeric.unit_roundoff * abs(value)
        )
        return _Bound(value=value, propagated=propagated, coefficient=coefficient, rounding=rounding)
    raise _precision_error("unsupported compiled expression node", kind=expr.kind)


def _add_bounds(left: _Bound, right: _Bound, unit_roundoff: float) -> _Bound:
    value = left.value + right.value
    _ensure_finite(value, what="addition result")
    return _Bound(
        value=value,
        propagated=left.propagated + right.propagated,
        coefficient=left.coefficient + right.coefficient,
        rounding=left.rounding + right.rounding + unit_roundoff * abs(value),
    )


def _multiply_bounds(left: _Bound, right: _Bound, unit_roundoff: float) -> _Bound:
    value = left.value * right.value
    _ensure_finite(value, what="multiplication result")
    propagated = abs(left.value) * right.propagated + abs(right.value) * left.propagated
    coefficient = (
        abs(left.value) * right.coefficient
        + abs(right.value) * left.coefficient
        + left.coefficient * right.coefficient
    )
    rounding = (
        abs(left.value) * right.rounding
        + abs(right.value) * left.rounding
        + left.total * right.total
        + unit_roundoff * abs(value)
    )
    return _Bound(value=value, propagated=propagated, coefficient=coefficient, rounding=rounding)


def _apply_effect(
    edge: Edge,
    effect_index: int,
    state: _State,
    component_index: Mapping[str, int],
    spec: Spec,
    coefficient_overrides: Mapping[str, float],
) -> dict[str, Any]:
    effect = edge.effects[effect_index]
    index = component_index[effect.target]
    before_value = state.values[index]
    before_error = state.errors[index]
    expression = _eval_expr(effect.expression, state, component_index, spec, coefficient_overrides)
    old_bound = _Bound(value=before_value, propagated=before_error)
    if effect.mode == "assign":
        result = expression
    elif effect.mode == "add":
        result = _add_bounds(old_bound, expression, spec.numeric.unit_roundoff)
    elif effect.mode == "multiply":
        result = _multiply_bounds(old_bound, expression, spec.numeric.unit_roundoff)
    else:
        raise _precision_error("unsupported effect mode", edge_id=edge.id, mode=effect.mode)

    _check_error_bound(
        result.value,
        result.total,
        spec,
        edge_id=edge.id,
        effect_index=effect_index,
        component=effect.target,
    )
    state.values[index] = result.value
    state.errors[index] = result.total
    return {
        "effect_index": effect_index,
        "source": effect.source,
        "target": effect.target,
        "mode": effect.mode,
        "before_value": before_value,
        "before_bits": float_bits(before_value),
        "before_error_upper": before_error,
        "expression_value": expression.value,
        "expression_hex": exact_float(expression.value),
        "expression_error_upper": expression.total,
        "propagated_error_upper": result.propagated,
        "coefficient_error_upper": result.coefficient,
        "roundoff_error_upper": result.rounding,
        "after_value": result.value,
        "after_bits": float_bits(result.value),
        "after_error_upper": result.total,
    }


def _interval_uncertain(
    direction: str,
    value: float,
    value_error: float,
    threshold: float,
    threshold_error: float,
) -> bool:
    lo_v = value - value_error
    hi_v = value + value_error
    lo_t = threshold - threshold_error
    hi_t = threshold + threshold_error
    if direction == ">":
        return not (lo_v > hi_t or hi_v <= lo_t)
    if direction == ">=":
        return not (lo_v >= hi_t or hi_v < lo_t)
    if direction == "<":
        return not (hi_v < lo_t or lo_v >= hi_t)
    if direction == "<=":
        return not (hi_v <= lo_t or lo_v > hi_t)
    if direction == "==":
        if value_error == 0.0 and threshold_error == 0.0:
            return False
        return not (hi_v < lo_t or lo_v > hi_t)
    if direction == "!=":
        if value_error == 0.0 and threshold_error == 0.0:
            return False
        return not (hi_v < lo_t or lo_v > hi_t)
    raise _precision_error("unsupported condition direction", direction=direction)


def _raw_condition_result(direction: str, value: float, threshold: float) -> bool:
    if direction == ">":
        return value > threshold
    if direction == ">=":
        return value >= threshold
    if direction == "<":
        return value < threshold
    if direction == "<=":
        return value <= threshold
    if direction == "==":
        return value == threshold
    if direction == "!=":
        return value != threshold
    raise _precision_error("unsupported condition direction", direction=direction)


def _evaluate_condition(
    condition: Condition,
    value: float,
    value_error: float,
    spec: Spec,
    threshold_offsets: Mapping[str, float],
    *,
    step_index: int,
    from_node: str,
    edge_id: str,
) -> dict[str, Any]:
    offset = float(threshold_offsets.get(condition.id, 0.0))
    threshold = condition.threshold + offset
    _ensure_finite(threshold, what="condition threshold", condition_id=condition.id)
    threshold_error = abs(threshold) * spec.numeric.coefficient_relative_error
    signed_margin = value - threshold
    _ensure_finite(signed_margin, what="condition margin", condition_id=condition.id)
    subtraction_roundoff = spec.numeric.unit_roundoff * abs(signed_margin)
    margin_error = value_error + threshold_error + subtraction_roundoff
    margin_lower = (value - value_error) - (threshold + threshold_error)
    margin_upper = (value + value_error) - (threshold - threshold_error)
    raw_result = _raw_condition_result(condition.direction, value, threshold)
    uncertain = _interval_uncertain(
        condition.direction,
        value,
        value_error,
        threshold,
        threshold_error,
    )
    boundary = abs(signed_margin) < spec.boundary_epsilon
    return {
        "condition_id": condition.id,
        "step_index": step_index,
        "from_node": from_node,
        "edge_id": edge_id,
        "component": condition.component,
        "direction": condition.direction,
        "current": value,
        "current_hex": exact_float(value),
        "current_error_upper": value_error,
        "state_interval": [value - value_error, value + value_error],
        "threshold_base": condition.threshold,
        "threshold_offset": offset,
        "threshold": threshold,
        "threshold_hex": exact_float(threshold),
        "threshold_error_upper": threshold_error,
        "threshold_interval": [threshold - threshold_error, threshold + threshold_error],
        "signed_margin": signed_margin,
        "absolute_margin": abs(signed_margin),
        "margin_hex": exact_float(signed_margin),
        "margin_error_upper": margin_error,
        "margin_lower": margin_lower,
        "margin_upper": margin_upper,
        "rounded_subtraction_error_upper": subtraction_roundoff,
        "reachable": raw_result,
        "uncertain": uncertain,
        "boundary": boundary,
    }


def _state_payload(spec: Spec, state: _State) -> dict[str, Any]:
    return {
        "components": list(spec.state.components),
        "values": list(state.values),
        "bits": [float_bits(value) for value in state.values],
        "hex": [value.hex() for value in state.values],
        "errors": list(state.errors),
        "error_bits": [float_bits(error) for error in state.errors],
        "intervals": [
            [value - error, value + error]
            for value, error in zip(state.values, state.errors)
        ],
    }


def _resolve_edge(
    spec: Spec,
    current: str,
    token: str,
    branches: dict[str, _BranchEvaluation],
    enforce_reachability: bool,
    step_index: int,
) -> Edge:
    outgoing = [edge for edge in spec.edges if edge.source == current]
    id_matches = [edge for edge in outgoing if edge.id == token]
    candidates = id_matches if id_matches else [edge for edge in outgoing if edge.choice == token]
    if not candidates:
        raise BusinessError(
            E_SELECTION_UNKNOWN,
            "selection token does not match an outgoing edge",
            {
                "step_index": step_index,
                "node_id": current,
                "selection": token,
                "available_choices": [edge.choice for edge in outgoing],
                "available_edges": [edge.id for edge in outgoing],
            },
        )
    if enforce_reachability:
        for edge in candidates:
            if branches[edge.id].reachable:
                return edge
        raise BusinessError(
            E_SELECTION_UNKNOWN,
            "selection token has no reachable edge",
            {
                "step_index": step_index,
                "node_id": current,
                "selection": token,
                "candidate_edges": [edge.id for edge in candidates],
                "candidate_status": {edge.id: branches[edge.id].status for edge in candidates},
            },
        )
    return candidates[0]


def _hash_step(step_payload: dict[str, Any], previous_chain_hash: str) -> tuple[str, str]:
    step_hash = sha256_value(step_payload)
    chain = hashlib.sha256(bytes.fromhex(previous_chain_hash) + bytes.fromhex(step_hash) + _REPLAY_DOMAIN).hexdigest()
    return step_hash, chain


def replay(spec: Spec, selections: Iterable[str], options: ReplayOptions | None = None) -> dict[str, Any]:
    """Replay one selection sequence under explicit, reproducible semantics."""

    options = options or ReplayOptions()
    selections_tuple = tuple(selections)
    coefficient_overrides = {key: float(value) for key, value in options.coefficient_overrides.items()}
    threshold_offsets = {key: float(value) for key, value in options.threshold_offsets.items()}
    initial_offsets = {key: float(value) for key, value in options.initial_offsets.items()}

    coefficient_map = spec.coefficient_map()
    for key in coefficient_overrides:
        if key not in coefficient_map:
            raise BusinessError(E_INJECT_CONFIG, "unknown coefficient override", {"coefficient": key})
    condition_map = spec.condition_map()
    for key in threshold_offsets:
        if key not in condition_map:
            raise BusinessError(E_INJECT_CONFIG, "unknown condition threshold offset", {"condition": key})
    component_set = set(spec.state.components)
    for key in initial_offsets:
        if key not in component_set:
            raise BusinessError(E_INJECT_CONFIG, "unknown initial-state component offset", {"component": key})

    node_map = spec.node_map()
    component_index = {component: index for index, component in enumerate(spec.state.components)}
    state = _State(list(spec.state.initial), [])
    for index, component in enumerate(spec.state.components):
        offset = initial_offsets.get(component, 0.0)
        _ensure_finite(offset, what="initial-state offset", component=component)
        state.values[index] = state.values[index] + offset
        _ensure_finite(state.values[index], what="initial state value", component=component)
        state.errors.append(
            abs(state.values[index]) * spec.numeric.initial_relative_error
        )
        _check_error_bound(
            state.values[index],
            state.errors[index],
            spec,
            edge_id="<initial>",
            effect_index=None,
            component=component,
        )

    initial_chain = hashlib.sha256(
        canonical_bytes(
            {
                "spec_sha256": spec.source_sha256,
                "selections": list(selections_tuple),
                "options": options.canonical_dict(),
            }
        )
    ).hexdigest()
    chain_hash = initial_chain
    current = spec.start
    edge_ids: list[str] = []
    steps: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    selected_branches: list[dict[str, Any]] = []

    def selection_error(message: str, **details: Any) -> BusinessError:
        return BusinessError(
            E_SELECTION_UNKNOWN,
            message,
            {
                "spec_sha256": spec.source_sha256,
                "selections": list(selections_tuple),
                "node_id": current,
                **details,
            },
        )

    for step_index, token in enumerate(selections_tuple):
        if node_map[current].type == "ending":
            raise selection_error("selection sequence continues after an ending", step_index=step_index)
        outgoing = [edge for edge in spec.edges if edge.source == current]
        if not outgoing:
            raise selection_error("non-ending node has no outgoing edge", step_index=step_index)

        branch_evaluations: dict[str, _BranchEvaluation] = {}
        branch_reports: list[dict[str, Any]] = []
        for edge in outgoing:
            edge_decisions: list[dict[str, Any]] = []
            for condition_id in edge.when:
                condition = condition_map[condition_id]
                index = component_index[condition.component]
                decision = _evaluate_condition(
                    condition,
                    state.values[index],
                    state.errors[index],
                    spec,
                    threshold_offsets,
                    step_index=step_index,
                    from_node=current,
                    edge_id=edge.id,
                )
                edge_decisions.append(decision)
                decisions.append(decision)
            reachable = all(decision["reachable"] for decision in edge_decisions)
            if any(not decision["reachable"] for decision in edge_decisions):
                status = "blocked"
            elif any(decision["uncertain"] for decision in edge_decisions):
                status = "uncertain"
            else:
                status = "reachable"
            evaluation = _BranchEvaluation(
                edge=edge,
                decisions=edge_decisions,
                reachable=reachable,
                uncertain=any(decision["uncertain"] for decision in edge_decisions),
                status=status,
            )
            branch_evaluations[edge.id] = evaluation
            branch_reports.append(
                {
                    "edge_id": edge.id,
                    "choice": edge.choice,
                    "target": edge.target,
                    "condition_ids": list(edge.when),
                    "reachable": reachable,
                    "uncertain": evaluation.uncertain,
                    "status": status,
                }
            )

        selected = _resolve_edge(
            spec,
            current,
            token,
            branch_evaluations,
            options.enforce_reachability,
            step_index,
        )
        before_state = state.clone()
        effect_reports: list[dict[str, Any]] = []
        for effect_index in range(len(selected.effects)):
            effect_reports.append(
                _apply_effect(
                    selected,
                    effect_index,
                    state,
                    component_index,
                    spec,
                    coefficient_overrides,
                )
            )
        after_state = state.clone()
        edge_ids.append(selected.id)
        step_decisions = [
            decision
            for edge in outgoing
            for decision in branch_evaluations[edge.id].decisions
        ]
        edge_error_summary = {
            "coefficient_error_upper": sum(
                effect["coefficient_error_upper"] for effect in effect_reports
            ),
            "roundoff_error_upper": sum(
                effect["roundoff_error_upper"] for effect in effect_reports
            ),
            "propagated_error_upper": sum(
                effect["propagated_error_upper"] for effect in effect_reports
            ),
            "max_after_component_error_upper": max(
                (effect["after_error_upper"] for effect in effect_reports),
                default=0.0,
            ),
        }

        step_payload: dict[str, Any] = {
            "step_index": step_index,
            "from_node": current,
            "selection": token,
            "selected_edge_id": selected.id,
            "selected_choice": selected.choice,
            "to_node": selected.target,
            "before_state_bits": [float_bits(value) for value in before_state.values],
            "before_error_bits": [float_bits(error) for error in before_state.errors],
            "after_state_bits": [float_bits(value) for value in after_state.values],
            "after_error_bits": [float_bits(error) for error in after_state.errors],
            "decisions": step_decisions,
            "branches": branch_reports,
            "effects": effect_reports,
            "edge_error_summary": edge_error_summary,
        }
        step_hash, chain_hash = _hash_step(step_payload, chain_hash)
        step_report = dict(step_payload)
        step_report["selected_branch_status"] = branch_evaluations[selected.id].status
        step_report["branches"] = branch_reports
        step_report["decisions"] = step_decisions
        step_report["before_state"] = _state_payload(spec, before_state)
        step_report["after_state"] = _state_payload(spec, after_state)
        step_report["effects"] = effect_reports
        step_report["edge_error_summary"] = edge_error_summary
        step_report["step_content_hash"] = step_hash
        step_report["chain_hash"] = chain_hash
        steps.append(step_report)
        selected_branches.append(
            {
                "step_index": step_index,
                "edge_id": selected.id,
                "choice": selected.choice,
                "reachable": branch_evaluations[selected.id].reachable,
                "uncertain": branch_evaluations[selected.id].uncertain,
                "status": branch_evaluations[selected.id].status,
            }
        )
        current = selected.target

    if node_map[current].type != "ending":
        raise selection_error(
            "selection sequence ended before an ending node",
            step_index=len(selections_tuple),
        )

    final_node = node_map[current]
    edge_sequence_hash = sha256_value({"edge_ids": edge_ids})
    final_state_payload = _state_payload(spec, state)
    content_payload = {
        "spec_sha256": spec.source_sha256,
        "selections": list(selections_tuple),
        "edge_ids": edge_ids,
        "edge_sequence_hash": edge_sequence_hash,
        "final_state_bits": final_state_payload["bits"],
        "final_errors": final_state_payload["errors"],
        "outcome": final_node.outcome,
        "final_node": final_node.id,
        "step_hashes": [step["step_content_hash"] for step in steps],
    }
    content_hash = sha256_value(content_payload)
    return {
        "business_code": "OK",
        "spec_sha256": spec.source_sha256,
        "spec_seed": spec.seed,
        "selections": list(selections_tuple),
        "options": options.canonical_dict(),
        "options_sha256": sha256_value(options.canonical_dict()),
        "start": spec.start,
        "steps": steps,
        "selected_branches": selected_branches,
        "decisions": decisions,
        "edge_ids": edge_ids,
        "edge_sequence_hash": edge_sequence_hash,
        "chain_hash": chain_hash,
        "final_node": final_node.id,
        "outcome": final_node.outcome,
        "final_state": final_state_payload,
        "content_hash": content_hash,
    }
