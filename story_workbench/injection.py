"""Serializable parameter injections and first-condition-flip localization."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .canonical import add_double, canonical_json, double, sha256_json
from .errors import InjectionInvalid
from .expressions import coefficient_count, render_expression, replace_coefficient
from .model import Edge, NarrativeSpec
from .timeline import TimelineResult


SCHEMA = "narrative-workbench/injections/v1"
_KINDS = {
    "coefficient_perturb",
    "threshold_shift",
    "initial_offset",
    "remove_effects",
}


@dataclass(frozen=True)
class InjectionConfig:
    id: str
    operations: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "id": self.id,
            "operations": [dict(operation) for operation in self.operations],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.as_dict())


def parse_injection_config(value: str | Mapping[str, Any]) -> InjectionConfig:
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InjectionInvalid(f"invalid injection JSON: {exc}") from exc
    else:
        raw = dict(value)
    if not isinstance(raw, dict):
        raise InjectionInvalid("injection config must be an object")
    if raw.get("schema") != SCHEMA:
        raise InjectionInvalid(f"injection schema must be {SCHEMA!r}")
    config_id = raw.get("id")
    if not isinstance(config_id, str) or not config_id:
        raise InjectionInvalid("injection id must be a non-empty string")
    operations_raw = raw.get("operations")
    if not isinstance(operations_raw, list):
        raise InjectionInvalid("injection operations must be a list")
    operations: list[dict[str, Any]] = []
    for index, operation in enumerate(operations_raw):
        if not isinstance(operation, dict):
            raise InjectionInvalid(f"injection operation {index} must be an object")
        kind = operation.get("kind")
        if kind not in _KINDS:
            raise InjectionInvalid(f"injection operation {index} has unknown kind {kind!r}")
        operations.append(_validate_operation(index, operation))
    return InjectionConfig(id=config_id, operations=tuple(operations))


def _validate_operation(index: int, operation: Mapping[str, Any]) -> dict[str, Any]:
    kind = operation["kind"]
    if kind == "coefficient_perturb":
        _only_keys(operation, {"kind", "edge_id", "effect_id", "coefficient_index", "mode", "value"}, index)
        edge_id = _string(operation, "edge_id", index)
        effect_id = _string(operation, "effect_id", index)
        coefficient_index = _integer(operation, "coefficient_index", index)
        mode = operation.get("mode")
        if mode not in {"add", "multiply"}:
            raise InjectionInvalid(f"operation {index} coefficient mode must be 'add' or 'multiply'")
        value = _number(operation, "value", index)
        return {
            "kind": kind,
            "edge_id": edge_id,
            "effect_id": effect_id,
            "coefficient_index": coefficient_index,
            "mode": mode,
            "value": value,
        }
    if kind == "threshold_shift":
        _only_keys(operation, {"kind", "edge_id", "condition_id", "delta"}, index)
        return {
            "kind": kind,
            "edge_id": _string(operation, "edge_id", index),
            "condition_id": _string(operation, "condition_id", index),
            "delta": _number(operation, "delta", index),
        }
    if kind == "initial_offset":
        _only_keys(operation, {"kind", "component", "delta"}, index)
        return {
            "kind": kind,
            "component": _string(operation, "component", index),
            "delta": _number(operation, "delta", index),
        }
    if kind == "remove_effects":
        _only_keys(operation, {"kind", "edge_id"}, index)
        return {"kind": kind, "edge_id": _string(operation, "edge_id", index)}
    raise InjectionInvalid(f"unknown injection kind {kind!r}")


def _only_keys(operation: Mapping[str, Any], allowed: set[str], index: int) -> None:
    for key in operation:
        if key not in allowed:
            raise InjectionInvalid(f"operation {index} has unknown field {key!r}")


def _string(operation: Mapping[str, Any], key: str, index: int) -> str:
    value = operation.get(key)
    if not isinstance(value, str) or not value:
        raise InjectionInvalid(f"operation {index} field {key!r} must be a non-empty string")
    return value


def _integer(operation: Mapping[str, Any], key: str, index: int) -> int:
    value = operation.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InjectionInvalid(f"operation {index} field {key!r} must be a non-negative integer")
    return value


def _number(operation: Mapping[str, Any], key: str, index: int) -> float:
    value = operation.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InjectionInvalid(f"operation {index} field {key!r} must be a finite number")
    try:
        return double(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InjectionInvalid(f"operation {index} field {key!r} must be finite") from exc


def apply_injections(
    spec: NarrativeSpec,
    config: InjectionConfig | Mapping[str, Any] | str,
) -> tuple[NarrativeSpec, list[dict[str, Any]]]:
    if not isinstance(config, InjectionConfig):
        config = parse_injection_config(config)
    current = spec
    records: list[dict[str, Any]] = []
    for operation_index, operation in enumerate(config.operations):
        before_spec_hash = current.spec_hash
        operation_before: Any
        operation_after: Any
        snippet: str
        kind = operation["kind"]
        if kind == "coefficient_perturb":
            current, operation_before, operation_after, snippet = _apply_coefficient(current, operation)
        elif kind == "threshold_shift":
            current, operation_before, operation_after, snippet = _apply_threshold(current, operation)
        elif kind == "initial_offset":
            current, operation_before, operation_after, snippet = _apply_initial(current, operation)
        elif kind == "remove_effects":
            current, operation_before, operation_after, snippet = _apply_remove_effects(current, operation)
        else:  # pragma: no cover - validated earlier
            raise InjectionInvalid(f"unknown injection kind {kind!r}")
        records.append(
            {
                "operation_index": operation_index,
                "operation": dict(operation),
                "injection_fragment": dict(operation),
                "snippet": snippet,
                "before": operation_before,
                "after": operation_after,
                "spec_hash_before": before_spec_hash,
                "spec_hash_after": current.spec_hash,
            }
        )
    return current, records


def _find_edge(spec: NarrativeSpec, edge_id: str) -> tuple[Edge, int]:
    for index, edge in enumerate(spec.edges):
        if edge.id == edge_id:
            return edge, index
    raise InjectionInvalid(f"injection references unknown edge {edge_id!r}")


def _replace_edge(spec: NarrativeSpec, edge_index: int, edge: Edge) -> NarrativeSpec:
    edges = list(spec.edges)
    edges[edge_index] = edge
    return spec.with_edges(edges)


def _apply_coefficient(
    spec: NarrativeSpec,
    operation: Mapping[str, Any],
) -> tuple[NarrativeSpec, float, float, str]:
    edge, edge_index = _find_edge(spec, operation["edge_id"])
    effect_index = next((index for index, effect in enumerate(edge.effects) if effect.id == operation["effect_id"]), None)
    if effect_index is None:
        raise InjectionInvalid(f"injection references unknown effect {operation['effect_id']!r} on edge {edge.id!r}")
    effect = edge.effects[effect_index]
    ordinal = operation["coefficient_index"]
    count = coefficient_count(effect.ast)
    if ordinal >= count:
        raise InjectionInvalid(
            f"coefficient index {ordinal} out of range for effect {effect.id!r} (count={count})"
        )
    original = _coefficient_at(effect.ast, ordinal)
    mode = operation["mode"]
    value = double(operation["value"])
    if mode == "add":
        patched = add_double(original, value)
    else:
        patched = double(original * value)
    new_ast = replace_coefficient(effect.ast, ordinal, patched)
    new_effect = replace(effect, ast=new_ast, expression=render_expression(new_ast))
    effects = list(edge.effects)
    effects[effect_index] = new_effect
    new_edge = replace(edge, effects=tuple(effects))
    snippet = (
        f"{edge.id}:{effect.id}:coef:{ordinal} {mode} {value!r} "
        f"({original!r} -> {patched!r})"
    )
    return _replace_edge(spec, edge_index, new_edge), original, patched, snippet


def _coefficient_at(ast: tuple[Any, ...], ordinal: int) -> float:
    counter = [0]
    found: list[float] = []

    def visit(node: tuple[Any, ...]) -> None:
        kind = node[0]
        if kind == "num":
            index = counter[0]
            counter[0] += 1
            if index == ordinal:
                found.append(double(node[1]))
            return
        if kind == "var":
            return
        if kind == "neg":
            visit(node[1])
            return
        if kind in {"add", "mul"}:
            visit(node[1])
            visit(node[2])
            return
        raise ValueError(f"unknown expression node {kind!r}")

    visit(ast)
    if not found:
        raise InjectionInvalid(f"coefficient ordinal {ordinal} not found")
    return found[0]


def _apply_threshold(
    spec: NarrativeSpec,
    operation: Mapping[str, Any],
) -> tuple[NarrativeSpec, float, float, str]:
    edge, edge_index = _find_edge(spec, operation["edge_id"])
    condition_index = next(
        (index for index, condition in enumerate(edge.conditions) if condition.id == operation["condition_id"]),
        None,
    )
    if condition_index is None:
        raise InjectionInvalid(
            f"injection references unknown condition {operation['condition_id']!r} on edge {edge.id!r}"
        )
    condition = edge.conditions[condition_index]
    original = condition.threshold
    patched = add_double(original, double(operation["delta"]))
    new_condition = replace(condition, threshold=patched)
    conditions = list(edge.conditions)
    conditions[condition_index] = new_condition
    new_edge = replace(edge, conditions=tuple(conditions))
    snippet = f"{condition.threshold_key} {original!r} -> {patched!r} (delta {operation['delta']!r})"
    return _replace_edge(spec, edge_index, new_edge), original, patched, snippet


def _apply_initial(
    spec: NarrativeSpec,
    operation: Mapping[str, Any],
) -> tuple[NarrativeSpec, float, float, str]:
    component = operation["component"]
    try:
        component_index = spec.component_index(component)
    except KeyError as exc:
        raise InjectionInvalid(f"injection references unknown component {component!r}") from exc
    original = spec.initial_state[component_index]
    patched = add_double(original, double(operation["delta"]))
    values = list(spec.initial_state)
    values[component_index] = patched
    snippet = f"initial:{component}:value {original!r} -> {patched!r} (delta {operation['delta']!r})"
    return spec.with_initial_state(values), original, patched, snippet


def _apply_remove_effects(
    spec: NarrativeSpec,
    operation: Mapping[str, Any],
) -> tuple[NarrativeSpec, list[str], list[str], str]:
    edge, edge_index = _find_edge(spec, operation["edge_id"])
    before = [effect.id for effect in edge.effects]
    new_edge = replace(edge, effects=())
    snippet = f"effects:{edge.id} removed {before!r}"
    return _replace_edge(spec, edge_index, new_edge), before, [], snippet


def _condition_records(result: TimelineResult) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for snapshot in result.snapshots:
        history_key = tuple(snapshot.edge_history)
        for branch in snapshot.branches:
            for condition in branch.conditions:
                key = (history_key, snapshot.node_id, branch.edge_id, condition.condition_id)
                if key in seen:
                    continue
                seen.add(key)
                records.append(
                    {
                        "key": key,
                        "state_version": snapshot.state_version,
                        "version_stamp": snapshot.version_stamp,
                        "step_index": len(snapshot.edge_history),
                        "node_id": snapshot.node_id,
                        "edge_id": branch.edge_id,
                        "edge_index": branch.edge_index,
                        "branch_status": branch.status,
                        **condition.as_dict(),
                    }
                )
    return records


def locate_first_flip(
    baseline: TimelineResult,
    injected: TimelineResult,
    *,
    injection_records: Sequence[Mapping[str, Any]],
    injection_config: InjectionConfig | Mapping[str, Any] | str,
) -> dict[str, Any]:
    if not isinstance(injection_config, InjectionConfig):
        injection_config = parse_injection_config(injection_config)
    baseline_records = _condition_records(baseline)
    injected_records = _condition_records(injected)
    injected_by_key = {record["key"]: record for record in injected_records}

    ordered_records = list(baseline_records)
    baseline_keys = {record["key"] for record in baseline_records}
    ordered_records.extend(record for record in injected_records if record["key"] not in baseline_keys)

    first_decision_flip: dict[str, Any] | None = None
    first_margin_difference: dict[str, Any] | None = None
    for baseline_record in ordered_records:
        injected_record = injected_by_key.get(baseline_record["key"])
        if injected_record is None:
            continue
        decision_flip = (
            baseline_record["satisfied_by_value"] != injected_record["satisfied_by_value"]
            or baseline_record["branch_status"] != injected_record["branch_status"]
            or baseline_record["status"] != injected_record["status"]
        )
        margin_difference = double(
            injected_record["signed_margin"] - baseline_record["signed_margin"]
        )
        margin_payload = {
            "key": list(baseline_record["key"][0]),
            "node_id": baseline_record["node_id"],
            "edge_id": baseline_record["edge_id"],
            "condition_id": baseline_record["condition_id"],
            "before_margin": baseline_record["signed_margin"],
            "after_margin": injected_record["signed_margin"],
            "margin_delta": margin_difference,
            "margin_interval": {
                "low": min(baseline_record["signed_margin"], injected_record["signed_margin"]),
                "high": max(baseline_record["signed_margin"], injected_record["signed_margin"]),
                "contains_zero": min(baseline_record["signed_margin"], injected_record["signed_margin"])
                <= 0.0
                <= max(baseline_record["signed_margin"], injected_record["signed_margin"]),
            },
            "before": baseline_record,
            "after": injected_record,
        }
        if first_margin_difference is None and margin_difference != 0.0:
            first_margin_difference = margin_payload
        if decision_flip:
            first_decision_flip = margin_payload
            break

    matching_injection_records = list(injection_records)
    if first_decision_flip is not None:
        edge_id = first_decision_flip["edge_id"]
        condition_id = first_decision_flip["condition_id"]
        matching = []
        for record in injection_records:
            operation = record.get("operation", {})
            threshold_match = (
                operation.get("kind") == "threshold_shift"
                and operation.get("condition_id") == condition_id
            )
            coefficient_match = (
                operation.get("kind") == "coefficient_perturb"
                and bool(edge_id)
                and operation.get("edge_id") == edge_id
            )
            if threshold_match or coefficient_match:
                matching.append(record)
            elif operation.get("kind") == "initial_offset":
                matching.append(record)
        if matching:
            matching_injection_records = matching

    ending_changed = baseline.ending != injected.ending
    report: dict[str, Any] = {
        "schema": "narrative-workbench/flip-report/v1",
        "spec_id": baseline.spec_id,
        "sequence_id": baseline.sequence_id,
        "injection_config": injection_config.as_dict(),
        "injection_records": [dict(record) for record in injection_records],
        "applied_injection_records": matching_injection_records,
        "baseline": {
            "ending": baseline.ending,
            "status": baseline.status,
            "state_hash": baseline.state_hash,
            "replay_hash": baseline.replay_hash,
            "edge_history": list(baseline.edge_history),
        },
        "injected": {
            "ending": injected.ending,
            "status": injected.status,
            "state_hash": injected.state_hash,
            "replay_hash": injected.replay_hash,
            "edge_history": list(injected.edge_history),
        },
        "ending_changed": ending_changed,
        "first_decision_flip": first_decision_flip,
        "first_margin_difference": first_margin_difference,
        "flip_detected": bool(ending_changed and first_decision_flip is not None),
        "error_interval": (
            first_decision_flip["margin_interval"] if first_decision_flip is not None else None
        ),
    }
    report["report_hash"] = sha256_json(report)
    return report
