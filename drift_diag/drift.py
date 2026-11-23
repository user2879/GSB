"""Deterministic field comparison, fixed-point metrics, and convergence rules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .canonical import stable_int
from .config import EvaluationConfig
from .models import FIELD_ORDER, NUMERIC_FIELDS, WorkloadTarget


INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1


def saturate_int64(value: int) -> int:
    if value < INT64_MIN:
        return INT64_MIN
    if value > INT64_MAX:
        return INT64_MAX
    return value


def truncate_toward_zero(numerator: int, denominator: int) -> int:
    """Integer-only truncation; negative values are rounded toward zero."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    magnitude = abs(numerator) // denominator
    return magnitude if numerator >= 0 else -magnitude


def quantize_difference(target: int, observed: int, scale_exp: int) -> int:
    """Scale an exact integer difference and saturate to signed 64-bit.

    Inputs are validated integers, so the rational denominator is one and no
    truncation occurs. The helper still documents the generic rule: division
    is integer-only and truncates toward zero before saturation.
    """

    factor = 10**scale_exp
    scaled = truncate_toward_zero((target - observed) * factor, 1)
    return saturate_int64(scaled)


def _field_basis(field: str, present: bool) -> str:
    if not present:
        return "observation_presence_exact"
    if field in NUMERIC_FIELDS:
        return "integer_fixed_point"
    if field == "image":
        return "string_exact"
    if field == "labels":
        return "label_map_canonical_exact"
    return "nullable_reference_exact"


def compare_field(
    workload: WorkloadTarget,
    field: str,
    actual_state: dict[str, dict[str, Any]],
    *,
    scale_exp: int,
) -> dict[str, Any]:
    observed_map = actual_state.get(workload.name, {})
    present = field in observed_map
    observed = deepcopy(observed_map.get(field)) if present else None
    target_value = deepcopy(workload.fields[field])

    if not present:
        drift = True
        reason = "missing_observation"
    else:
        drift = target_value != observed
        reason = "value_mismatch" if drift else "match"

    metric: dict[str, Any] | None = None
    if present and field in NUMERIC_FIELDS:
        fixed = quantize_difference(target_value, observed, scale_exp)
        metric = {
            "fixed_value": fixed,
            "absolute_fixed": abs(fixed),
            "scale_exp": scale_exp,
            "scale_factor": 10**scale_exp,
            "difference": target_value - observed,
            "truncation": "toward_zero",
            "overflow": "saturate_int64",
            "quantization_error_upper_units": 0,
            "metric_status": "measured",
        }
    field_version = stable_int(
        "field-version-v1",
        workload.name,
        field,
        target_value,
        workload.provenance[field].to_dict(),
        present,
        observed,
    )

    return {
        "workload": workload.name,
        "field": field,
        "target": target_value,
        "observed": observed,
        "observed_present": present,
        "drift": drift,
        "basis": _field_basis(field, present),
        "reason": reason,
        "field_version": field_version,
        "fixed_metric": metric,
    }


def aggregate_workload(
    workload_name: str,
    decisions: list[dict[str, Any]],
    *,
    tolerance_num: int,
    tolerance_den: int,
) -> dict[str, Any]:
    by_field = {decision["field"]: decision for decision in decisions}
    ordered = [by_field[field] for field in FIELD_ORDER if field in by_field]
    field_count = len(ordered)
    drift_decisions = [decision for decision in ordered if decision["drift"]]
    drift_count = len(drift_decisions)

    fixed_metric_units = sum(
        decision["fixed_metric"]["absolute_fixed"]
        for decision in ordered
        if decision["fixed_metric"] is not None
    )
    per_field_error_units = {
        decision["field"]: (
            decision["fixed_metric"]["quantization_error_upper_units"]
            if decision["fixed_metric"] is not None
            else 0
        )
        for decision in ordered
    }
    error_upper_units = sum(per_field_error_units.values())

    cross = drift_count * tolerance_den - tolerance_num * field_count
    converged = cross <= 0
    boundary_sensitive = cross == 0 and (tolerance_num > 0 or drift_count > 0)

    return {
        "workload": workload_name,
        "field_count": field_count,
        "drift_fields": [decision["field"] for decision in drift_decisions],
        "drift_field_count": drift_count,
        "drift_ratio": {
            "numerator": drift_count,
            "denominator": field_count,
            "reduced": _reduce_ratio(drift_count, field_count),
        },
        "tolerance": {
            "numerator": tolerance_num,
            "denominator": tolerance_den,
        },
        "numeric_drift_metric_units": fixed_metric_units,
        "numeric_fields_measured": sum(
            1
            for decision in ordered
            if decision["fixed_metric"] is not None
        ),
        "per_field_quantization_error_units": per_field_error_units,
        "workload_error_upper_units": error_upper_units,
        "non_numeric_drift_fields": [
            decision["field"]
            for decision in drift_decisions
            if decision["field"] not in NUMERIC_FIELDS
        ],
        "converged": converged,
        "boundary_sensitive": boundary_sensitive,
        "boundary_rule_applied": boundary_sensitive,
        "boundary_tie_break": "inclusive: drift_ratio <= tolerance converges",
        "error_affected": boundary_sensitive,
        "exact_integer_quantization": error_upper_units == 0,
        "decision_annotation": (
            "boundary_sensitive_exact_rational"
            if boundary_sensitive
            else "exact_rational"
        ),
        "quantization_policy": {
            "fixed_point_base": 10,
            "truncation": "toward_zero",
            "overflow": "saturate_int64",
        },
    }


def _reduce_ratio(numerator: int, denominator: int) -> dict[str, int]:
    if numerator == 0:
        return {"numerator": 0, "denominator": 1}
    a, b = numerator, denominator
    while b:
        a, b = b, a % b
    return {"numerator": numerator // a, "denominator": denominator // a}


def aggregate_all(
    target: Any,
    decisions: list[dict[str, Any]],
    config: EvaluationConfig,
) -> dict[str, Any]:
    by_workload: dict[str, list[dict[str, Any]]] = {name: [] for name in sorted(target.workloads)}
    for decision in decisions:
        by_workload[decision["workload"]].append(decision)

    workloads: dict[str, Any] = {}
    for name in sorted(by_workload):
        workloads[name] = aggregate_workload(
            name,
            by_workload[name],
            tolerance_num=config.tolerance_num,
            tolerance_den=config.tolerance_den,
        )
    return {
        "overall_converged": all(result["converged"] for result in workloads.values()),
        "workloads": workloads,
    }


def drift_only(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [decision for decision in decisions if decision["drift"]]
