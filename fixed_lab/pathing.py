"""Coefficient path summarization, zero policy, and adjacent-level events."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .dataset import Dataset
from .fixedpoint import FixedParams
from .metrics import fixed_objective, residual_statistics


def classify_zero(value_fixed: int, params: FixedParams, policy: str) -> dict[str, Any]:
    half = params.half_zero_units
    abs_value = abs(Decimal(value_fixed))
    exact_zero = value_fixed == 0
    equal_half_boundary = abs_value == half
    within_half = abs_value <= half
    summary_value = value_fixed
    decision = "nonzero"
    if exact_zero:
        decision = "zero_exact"
        summary_value = 0
    elif within_half:
        if equal_half_boundary:
            decision = "zero_half_step_tie" if policy == "zero" else "nonzero_half_step_tie"
        else:
            decision = "zero_within_half_step" if policy == "zero" else "nonzero_within_half_step"
        if policy == "zero":
            summary_value = 0
    return {
        "raw_fixed": value_fixed,
        "summary_fixed": summary_value,
        "decision": decision,
        "is_zero": summary_value == 0,
        "exact_zero": exact_zero,
        "within_half_step": within_half and not exact_zero,
        "boundary": equal_half_boundary,
        "half_step_fixed": format(half, "f"),
        "policy": policy,
    }


def sign_of_fixed(value: int) -> int:
    return (value > 0) - (value < 0)


def summarize_path(
    *,
    dataset: Dataset,
    qd,
    params: FixedParams,
    penalty: str,
    level_results: list[dict[str, Any]],
    solver_config: dict[str, Any],
) -> dict[str, Any]:
    policy = solver_config["zero_half_step_policy"]
    levels: list[dict[str, Any]] = []
    alpha_keys: list[str] = []

    for alpha_index, result in enumerate(level_results):
        alpha = result["requested_alpha_decimal"]
        alpha_key = format(alpha, "f")
        alpha_keys.append(alpha_key)
        raw_fixed = result["coefficients_fixed"]
        classifications = [classify_zero(value, params, policy) for value in raw_fixed]
        summary_fixed = [item["summary_fixed"] for item in classifications]
        nonzero_indices = [i for i, item in enumerate(classifications) if not item["is_zero"]]
        nonzero_columns = [dataset.columns[i] for i in nonzero_indices]
        objective = fixed_objective(qd, summary_fixed, alpha, penalty)
        residuals = residual_statistics(dataset, summary_fixed, params)
        levels.append({
            "alpha_index": alpha_index,
            "alpha": alpha_key,
            "alpha_key": alpha_key,
            "status": result.get("status", "OK"),
            "iterations": result.get("iterations"),
            "converged": result.get("converged"),
            "boundary_round": result.get("boundary_round"),
            "fallback_count": result.get("fallback_count", 0),
            "effective_alpha": result.get("effective_alpha", alpha_key),
            "coefficients_fixed_raw": raw_fixed,
            "coefficients_raw": [float(Decimal(v) / Decimal(params.scale)) for v in raw_fixed],
            "coefficients_fixed_summary": summary_fixed,
            "coefficients": [float(Decimal(v) / Decimal(params.scale)) for v in summary_fixed],
            "zero_classification": classifications,
            "nonzero_indices": nonzero_indices,
            "nonzero_columns": nonzero_columns,
            "nonzero_set": nonzero_columns,
            "objective": objective,
            "residuals": residuals,
            "solver_summary": {
                key: result.get(key)
                for key in [
                    "lambda_accum_scaled",
                    "equation_residual_max_abs",
                    "max_iterations_reached",
                    "unresolved_low_pivot",
                ]
                if key in result
            },
        })

    events: list[dict[str, Any]] = []
    for from_index in range(len(levels) - 1):
        before = levels[from_index]
        after = levels[from_index + 1]
        before_fixed = before["coefficients_fixed_summary"]
        after_fixed = after["coefficients_fixed_summary"]
        for coordinate, column in enumerate(dataset.columns):
            before_zero = before["zero_classification"][coordinate]["is_zero"]
            after_zero = after["zero_classification"][coordinate]["is_zero"]
            before_sign = 0 if before_zero else sign_of_fixed(before_fixed[coordinate])
            after_sign = 0 if after_zero else sign_of_fixed(after_fixed[coordinate])
            event_type = None
            if before_zero and not after_zero:
                event_type = "enter"
            elif not before_zero and after_zero:
                event_type = "exit"
            elif not before_zero and not after_zero and before_sign != after_sign:
                event_type = "sign_flip"
            if event_type is not None:
                events.append({
                    "event_id": f"{before['alpha_key']}->{after['alpha_key']}:{column}:{event_type}",
                    "type": event_type,
                    "coordinate": coordinate,
                    "column": column,
                    "alpha_from": before["alpha_key"],
                    "alpha_to": after["alpha_key"],
                    "alpha_from_index": from_index,
                    "alpha_to_index": from_index + 1,
                    "sign_from": before_sign,
                    "sign_to": after_sign,
                    "fixed_from": before_fixed[coordinate],
                    "fixed_to": after_fixed[coordinate],
                    "boundary": before["zero_classification"][coordinate]["boundary"]
                    or after["zero_classification"][coordinate]["boundary"],
                })

    return {
        "penalty": penalty,
        "coordinate_columns": list(dataset.columns),
        "alpha_keys": alpha_keys,
        "levels": levels,
        "events": events,
        "comparison_note": "Levels are keyed by canonical decimal alpha; coordinates are keyed by stable column names.",
    }


def path_comparison_index(path_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {level["alpha_key"]: level for level in path_report["levels"]}
