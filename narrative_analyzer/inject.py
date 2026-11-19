"""Serializable parameter-layer injection and first-flip localization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .canonical import sha256_bytes, sha256_value
from .engine import ReplayOptions, replay
from .errors import E_INJECT_CONFIG, BusinessError
from .model import Spec


@dataclass(frozen=True)
class InjectionPlan:
    config: dict[str, Any]
    digest: str


def _inject_error(message: str, **details: Any) -> BusinessError:
    return BusinessError(E_INJECT_CONFIG, message, details)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _inject_error("injection value must be numeric", path=path)
    result = float(value)
    if not math.isfinite(result):
        raise _inject_error("injection value must be finite", path=path)
    return result


def parse_injection(data: str | bytes, source_name: str = "<memory>") -> InjectionPlan:
    raw_bytes = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        loaded = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise _inject_error("injection configuration parsing failed", source=source_name, yaml_error=str(exc)) from exc
    if not isinstance(loaded, dict):
        raise _inject_error("injection configuration must be a mapping", source=source_name)
    allowed = {"version", "name", "coefficient_perturbations", "threshold_offsets", "initial_offsets"}
    unknown = sorted(str(key) for key in loaded if key not in allowed)
    if unknown:
        raise _inject_error("injection configuration contains unknown keys", unknown=unknown)
    if loaded.get("version", 1) != 1:
        raise _inject_error("unsupported injection configuration version", version=loaded.get("version"))

    normalized: dict[str, Any] = {"version": 1}
    if "name" in loaded:
        if not isinstance(loaded["name"], str) or not loaded["name"].strip():
            raise _inject_error("injection name must be a non-empty string")
        normalized["name"] = loaded["name"].strip()

    coefficients_raw = loaded.get("coefficient_perturbations", [])
    thresholds_raw = loaded.get("threshold_offsets", [])
    initials_raw = loaded.get("initial_offsets", [])
    for name, value in (
        ("coefficient_perturbations", coefficients_raw),
        ("threshold_offsets", thresholds_raw),
        ("initial_offsets", initials_raw),
    ):
        if not isinstance(value, list):
            raise _inject_error(f"{name} must be a list", field=name)

    coefficient_perturbations: list[dict[str, Any]] = []
    seen_coefficients: set[str] = set()
    for index, item in enumerate(coefficients_raw):
        path = f"coefficient_perturbations[{index}]"
        if not isinstance(item, dict) or set(item) != {"coefficient", "relative"}:
            raise _inject_error("coefficient perturbation requires exactly coefficient and relative", path=path)
        coefficient = item["coefficient"]
        if not isinstance(coefficient, str) or not coefficient:
            raise _inject_error("coefficient perturbation target must be a non-empty string", path=path)
        if coefficient in seen_coefficients:
            raise _inject_error("duplicate coefficient perturbation", coefficient=coefficient)
        seen_coefficients.add(coefficient)
        relative = _number(item["relative"], f"{path}.relative")
        if not math.isfinite(1.0 + relative):
            raise _inject_error("relative coefficient perturbation is not finite", coefficient=coefficient)
        coefficient_perturbations.append({"coefficient": coefficient, "relative": relative})
    normalized["coefficient_perturbations"] = coefficient_perturbations

    threshold_offsets: list[dict[str, Any]] = []
    seen_conditions: set[str] = set()
    for index, item in enumerate(thresholds_raw):
        path = f"threshold_offsets[{index}]"
        if not isinstance(item, dict) or set(item) != {"condition", "delta"}:
            raise _inject_error("threshold offset requires exactly condition and delta", path=path)
        condition = item["condition"]
        if not isinstance(condition, str) or not condition:
            raise _inject_error("threshold offset target must be a non-empty string", path=path)
        if condition in seen_conditions:
            raise _inject_error("duplicate threshold offset", condition=condition)
        seen_conditions.add(condition)
        threshold_offsets.append({"condition": condition, "delta": _number(item["delta"], f"{path}.delta")})
    normalized["threshold_offsets"] = threshold_offsets

    initial_offsets: list[dict[str, Any]] = []
    seen_components: set[str] = set()
    for index, item in enumerate(initials_raw):
        path = f"initial_offsets[{index}]"
        if not isinstance(item, dict) or set(item) != {"component", "delta"}:
            raise _inject_error("initial-state offset requires exactly component and delta", path=path)
        component = item["component"]
        if not isinstance(component, str) or not component:
            raise _inject_error("initial-state offset target must be a non-empty string", path=path)
        if component in seen_components:
            raise _inject_error("duplicate initial-state offset", component=component)
        seen_components.add(component)
        initial_offsets.append({"component": component, "delta": _number(item["delta"], f"{path}.delta")})
    normalized["initial_offsets"] = initial_offsets

    if not coefficient_perturbations and not threshold_offsets and not initial_offsets:
        raise _inject_error("injection configuration has no operations")
    normalized["config_sha256"] = sha256_value(normalized)
    return InjectionPlan(config=normalized, digest=sha256_bytes(raw_bytes))


def load_injection(path: str | Path) -> InjectionPlan:
    source = Path(path)
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise _inject_error("cannot read injection configuration", path=str(source), error=str(exc)) from exc
    return parse_injection(data, source_name=str(source))


def _effective_operations(spec: Spec, plan: InjectionPlan) -> tuple[list[dict[str, Any]], ReplayOptions]:
    coefficient_map = spec.coefficient_map()
    condition_map = spec.condition_map()
    component_set = set(spec.state.components)
    coefficient_overrides: dict[str, float] = {}
    threshold_offsets: dict[str, float] = {}
    initial_offsets: dict[str, float] = {}
    effective: list[dict[str, Any]] = []

    for item in plan.config["coefficient_perturbations"]:
        key = item["coefficient"]
        if key not in coefficient_map:
            raise _inject_error("unknown coefficient perturbation target", coefficient=key)
        base = coefficient_map[key].value
        relative = item["relative"]
        after = base * (1.0 + relative)
        if not math.isfinite(after):
            raise _inject_error("perturbed coefficient is not finite", coefficient=key)
        coefficient_overrides[key] = after
        effective.append(
            {
                "kind": "coefficient",
                "target": key,
                "before": base,
                "relative": relative,
                "after": after,
            }
        )
    for item in plan.config["threshold_offsets"]:
        key = item["condition"]
        if key not in condition_map:
            raise _inject_error("unknown condition threshold offset target", condition=key)
        base = condition_map[key].threshold
        delta = item["delta"]
        after = base + delta
        if not math.isfinite(after):
            raise _inject_error("perturbed threshold is not finite", condition=key)
        threshold_offsets[key] = delta
        effective.append(
            {
                "kind": "threshold",
                "target": key,
                "before": base,
                "delta": delta,
                "after": after,
            }
        )
    for item in plan.config["initial_offsets"]:
        key = item["component"]
        if key not in component_set:
            raise _inject_error("unknown initial-state offset target", component=key)
        base = spec.state.initial[spec.state.components.index(key)]
        delta = item["delta"]
        after = base + delta
        if not math.isfinite(after):
            raise _inject_error("perturbed initial value is not finite", component=key)
        initial_offsets[key] = delta
        effective.append(
            {
                "kind": "initial",
                "target": key,
                "before": base,
                "delta": delta,
                "after": after,
            }
        )
    options = ReplayOptions(
        coefficient_overrides=coefficient_overrides,
        threshold_offsets=threshold_offsets,
        initial_offsets=initial_offsets,
        enforce_reachability=True,
    )
    return effective, options


def _decision_key(decision: Mapping[str, Any]) -> str:
    return f"step[{decision['step_index']}]/{decision['edge_id']}/{decision['condition_id']}"


def run_injection(spec: Spec, selections: tuple[str, ...], plan: InjectionPlan) -> dict[str, Any]:
    effective, options = _effective_operations(spec, plan)
    baseline = replay(spec, selections, ReplayOptions(enforce_reachability=True))
    injected = replay(spec, selections, options)
    injected_by_key = {_decision_key(decision): decision for decision in injected["decisions"]}

    first_flip: dict[str, Any] | None = None
    for before in baseline["decisions"]:
        key = _decision_key(before)
        after = injected_by_key.get(key)
        if after is None:
            continue
        if before["reachable"] != after["reachable"]:
            first_flip = {
                "decision_key": key,
                "step_index": before["step_index"],
                "edge_id": before["edge_id"],
                "condition_id": before["condition_id"],
                "component": before["component"],
                "direction": before["direction"],
                "before": {
                    "reachable": before["reachable"],
                    "uncertain": before["uncertain"],
                    "boundary": before["boundary"],
                    "current": before["current"],
                    "threshold": before["threshold"],
                    "signed_margin": before["signed_margin"],
                    "absolute_margin": before["absolute_margin"],
                    "margin_error_upper": before["margin_error_upper"],
                },
                "after": {
                    "reachable": after["reachable"],
                    "uncertain": after["uncertain"],
                    "boundary": after["boundary"],
                    "current": after["current"],
                    "threshold": after["threshold"],
                    "signed_margin": after["signed_margin"],
                    "absolute_margin": after["absolute_margin"],
                    "margin_error_upper": after["margin_error_upper"],
                },
                "outcome_before": baseline["outcome"],
                "outcome_after": injected["outcome"],
                "injection_offsets": effective,
            }
            break

    return {
        "business_code": "OK",
        "spec_sha256": spec.source_sha256,
        "selections": list(selections),
        "injection": plan.config,
        "injection_source_sha256": plan.digest,
        "effective_operations": effective,
        "outcome_changed": baseline["outcome"] != injected["outcome"],
        "first_flip": first_flip,
        "baseline": baseline,
        "injected": injected,
    }
