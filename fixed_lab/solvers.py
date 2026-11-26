"""Deterministic fixed-domain Ridge and Lasso solvers."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
from typing import Any

from .features import QuantizedDataset
from .fixedpoint import FixedParams, round_div, round_half_away_integer


def _soft_threshold(value: int, threshold: int) -> int:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0


def bareiss_solve(
    matrix: list[list[int]],
    rhs_scaled: list[int],
    *,
    pivot_threshold: int,
    stop_on_low_pivot: bool,
) -> dict[str, Any]:
    p = len(matrix)
    augmented = [matrix[i][:] + [rhs_scaled[i]] for i in range(p)]
    pivot_trace: list[dict[str, Any]] = []
    elimination_trace: list[dict[str, Any]] = []
    low_pivot: dict[str, Any] | None = None
    singular: dict[str, Any] | None = None
    previous_pivot = 1

    for k in range(p):
        pivot = augmented[k][k]
        below = abs(pivot) < pivot_threshold
        if below:
            low_pivot = {"step": k, "pivot": pivot, "pivot_abs": abs(pivot), "threshold": pivot_threshold}
        pivot_trace.append({
            "step": k,
            "rule": "fixed_diagonal_no_row_swap",
            "pivot": pivot,
            "pivot_abs": abs(pivot),
            "threshold": pivot_threshold,
            "below_threshold": below,
            "candidate_rows": [k],
        })
        if pivot == 0:
            singular = {"step": k, "pivot": pivot}
            return {"coefficients": None, "pivot_trace": pivot_trace, "elimination_trace": elimination_trace,
                    "back_substitution_trace": [], "low_pivot": low_pivot, "singular": singular,
                    "equation_residual": None}
        if below and stop_on_low_pivot:
            return {"coefficients": None, "pivot_trace": pivot_trace, "elimination_trace": elimination_trace,
                    "back_substitution_trace": [], "low_pivot": low_pivot, "singular": None,
                    "equation_residual": None}

        if k < p - 1:
            for i in range(k + 1, p):
                aik = augmented[i][k]
                for j in range(k + 1, p + 1):
                    numerator = augmented[i][j] * pivot - aik * augmented[k][j]
                    quotient, remainder, direction = round_div(numerator, previous_pivot)
                    augmented[i][j] = quotient
                    elimination_trace.append({
                        "k": k,
                        "i": i,
                        "j": j,
                        "numerator": numerator,
                        "denominator": previous_pivot,
                        "quotient": quotient,
                        "remainder": remainder,
                        "direction": direction,
                    })
                augmented[i][k] = 0
            previous_pivot = pivot

    coefficients = [0 for _ in range(p)]
    back_trace: list[dict[str, Any]] = []
    for i in range(p - 1, -1, -1):
        numerator = augmented[i][p]
        for j in range(i + 1, p):
            numerator -= augmented[i][j] * coefficients[j]
        quotient, remainder, direction = round_div(numerator, augmented[i][i])
        coefficients[i] = quotient
        back_trace.append({
            "coordinate": i,
            "numerator": numerator,
            "denominator": augmented[i][i],
            "quotient": quotient,
            "remainder": remainder,
            "direction": direction,
        })

    residual_vector: list[int] = []
    for i in range(p):
        residual = rhs_scaled[i] * 0
        for j in range(p):
            residual += matrix[i][j] * coefficients[j]
        residual -= rhs_scaled[i]
        residual_vector.append(residual)

    return {
        "coefficients": coefficients,
        "pivot_trace": pivot_trace,
        "elimination_trace": elimination_trace,
        "back_substitution_trace": back_trace,
        "low_pivot": low_pivot,
        "singular": None,
        "equation_residual_vector": residual_vector,
        "equation_residual_max_abs": max((abs(v) for v in residual_vector), default=0),
    }


def solve_ridge_path_level(
    g: list[list[int]],
    b: list[int],
    params: FixedParams,
    alpha: Decimal,
    solver_config: dict[str, Any],
) -> dict[str, Any]:
    threshold = int(solver_config["pivot_abs_threshold"])
    factor: Decimal = solver_config["fallback_factor"]
    max_fallbacks = int(solver_config["max_fallbacks"])
    current_alpha = alpha
    attempts: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    fallback_count = 0

    for attempt in range(max_fallbacks + 1):
        alpha_acc = round_half_away_integer(current_alpha * Decimal(params.accum_scale))
        matrix = deepcopy(g)
        for i in range(len(matrix)):
            matrix[i][i] += alpha_acc
        rhs_scaled = [params.scale * value for value in b]
        enforce = fallback_count < max_fallbacks
        result = bareiss_solve(
            matrix,
            rhs_scaled,
            pivot_threshold=threshold,
            stop_on_low_pivot=enforce,
        )
        attempts.append({
            "attempt": attempt,
            "requested_alpha": format(alpha, "f"),
            "effective_alpha": format(current_alpha, "f"),
            "alpha_acc": alpha_acc,
            "stopped_low_pivot": result["low_pivot"] is not None and result["coefficients"] is None,
            "low_pivot": result["low_pivot"],
        })
        if result["coefficients"] is not None:
            final_result = result
            break
        if result["low_pivot"] is not None and fallback_count < max_fallbacks:
            fallback_count += 1
            current_alpha = current_alpha * factor
            continue
        final_result = result
        break

    if final_result is None or final_result.get("coefficients") is None:
        return {
            "status": "E_SOLVER_LOW_PIVOT",
            "alpha": format(alpha, "f"),
            "requested_alpha_decimal": alpha,
            "fallback_count": fallback_count,
            "attempts": attempts,
            "pivot_trace": final_result.get("pivot_trace", []) if final_result else [],
            "elimination_trace": final_result.get("elimination_trace", []) if final_result else [],
            "back_substitution_trace": [],
            "coefficients_fixed": [],
            "coefficients": [],
        }

    coefficients_fixed = final_result["coefficients"]
    coefficients = [float(Decimal(value) / Decimal(params.scale)) for value in coefficients_fixed]
    unresolved_low = final_result.get("low_pivot")
    return {
        "status": "E_SOLVER_LOW_PIVOT" if unresolved_low is not None else "OK",
        "alpha": format(alpha, "f"),
        "requested_alpha_decimal": alpha,
        "effective_alpha": attempts[-1]["effective_alpha"],
        "fallback_count": fallback_count,
        "attempts": attempts,
        "pivot_trace": final_result["pivot_trace"],
        "elimination_trace": final_result["elimination_trace"],
        "back_substitution_trace": final_result["back_substitution_trace"],
        "equation_residual_vector": final_result["equation_residual_vector"],
        "equation_residual_max_abs": final_result["equation_residual_max_abs"],
        "unresolved_low_pivot": unresolved_low,
        "coefficients_fixed": coefficients_fixed,
        "coefficients": coefficients,
    }


def lasso_objective_x2(qd: QuantizedDataset, coefficients_fixed: list[int], objective_lambda: int) -> int:
    scale = qd.params.scale
    total = 0
    for row_index, sample_id in enumerate(qd.sample_ids):
        residual_fixed_numerator = scale * qd.y[row_index]
        row = qd.x[row_index]
        for j, value in enumerate(row):
            residual_fixed_numerator -= value * coefficients_fixed[j]
        total += qd.w[row_index] * residual_fixed_numerator * residual_fixed_numerator
    total += 2 * objective_lambda * sum(abs(value) for value in coefficients_fixed)
    return total


def solve_lasso_path_level(
    g: list[list[int]],
    b: list[int],
    qd: QuantizedDataset,
    alpha: Decimal,
    solver_config: dict[str, Any],
    *,
    order: str = "ascending",
) -> dict[str, Any]:
    params = qd.params
    p = len(qd.columns)
    accum_lambda = round_half_away_integer(alpha * Decimal(params.accum_scale) * Decimal(params.scale))
    objective_lambda = round_half_away_integer(alpha * Decimal(params.scale) ** 4)
    tolerance_units = Decimal(solver_config["coefficient_tolerance"]) * Decimal(params.scale)
    half_step = Decimal(params.grid) / Decimal(2)
    max_iterations = int(solver_config["max_iterations"])
    boundary_policy = solver_config["convergence_boundary_policy"]

    coefficients = [0 for _ in range(p)]
    sweeps: list[dict[str, Any]] = []
    converged = False
    boundary_round: int | None = None
    status = "E_SOLVER_NON_CONVERGED"

    for iteration in range(1, max_iterations + 1):
        max_delta = 0
        updates: list[dict[str, Any]] = []
        indices = range(p) if order == "ascending" else range(p - 1, -1, -1)
        for j in indices:
            z = params.scale * b[j]
            for k in range(p):
                if k != j:
                    z -= g[j][k] * coefficients[k]
            unthresholded = z
            thresholded = _soft_threshold(z, accum_lambda)
            denominator = g[j][j]
            curvature_code = "positive"
            if denominator == 0:
                new_value = 0
                direction = "zero_denominator"
                remainder = 0
                curvature_code = "zero"
            else:
                new_value, remainder, direction = round_div(thresholded, denominator)
                if denominator < 0:
                    curvature_code = "negative"
            old_value = coefficients[j]
            delta = abs(new_value - old_value)
            max_delta = max(max_delta, delta)
            coefficients[j] = new_value
            updates.append({
                "coordinate": j,
                "old": old_value,
                "new": new_value,
                "delta": delta,
                "z": unthresholded,
                "lambda": accum_lambda,
                "thresholded": thresholded,
                "denominator": denominator,
                "remainder": remainder,
                "direction": direction,
                "curvature": curvature_code,
            })

        objective = lasso_objective_x2(qd, coefficients, objective_lambda)
        previous_objective = sweeps[-1]["objective_fixed_x2"] if sweeps else None
        objective_delta = None if previous_objective is None else objective - previous_objective
        equal_threshold = Decimal(max_delta) == tolerance_units
        threshold_distance = abs(Decimal(max_delta) - tolerance_units)
        within_half_step = threshold_distance <= half_step
        threshold_converged = Decimal(max_delta) <= tolerance_units
        boundary = equal_threshold or (not threshold_converged and within_half_step)
        did_converge = threshold_converged
        boundary_action = "not_applicable"
        if not threshold_converged and within_half_step and boundary_policy == "converge":
            did_converge = True
            boundary_action = "converge_within_half_step"
        elif not threshold_converged and within_half_step:
            boundary_action = "continue_within_half_step"

        sweep_record = {
            "iteration": iteration,
            "order": order,
            "updates": updates,
            "coefficients_fixed_after_sweep": coefficients[:],
            "max_delta_fixed": max_delta,
            "tolerance_fixed": format(tolerance_units, "f"),
            "half_step_fixed": format(half_step, "f"),
            "equal_threshold": equal_threshold,
            "within_half_step": within_half_step,
            "boundary": boundary,
            "boundary_action": boundary_action,
            "objective_fixed_x2": objective,
            "objective_delta_fixed_x2": objective_delta,
            "converged": did_converge,
        }
        sweeps.append(sweep_record)
        if boundary and boundary_round is None:
            boundary_round = iteration
        if did_converge:
            converged = True
            status = "OK"
            break

    return {
        "status": status,
        "order": order,
        "alpha": format(alpha, "f"),
        "requested_alpha_decimal": alpha,
        "lambda_accum_scaled": accum_lambda,
        "lambda_objective_scaled": objective_lambda,
        "coefficients_fixed": coefficients,
        "coefficients": [float(Decimal(value) / Decimal(params.scale)) for value in coefficients],
        "iterations": len(sweeps),
        "converged": converged,
        "boundary_round": boundary_round,
        "max_iterations_reached": not converged,
        "sweeps": sweeps,
    }


def lasso_order_audit(
    g: list[list[int]],
    b: list[int],
    qd: QuantizedDataset,
    alpha: Decimal,
    solver_config: dict[str, Any],
    strong_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    ascending = solve_lasso_path_level(g, b, qd, alpha, solver_config, order="ascending")
    descending = solve_lasso_path_level(g, b, qd, alpha, solver_config, order="descending")
    first_asc = ascending["sweeps"][0]["coefficients_fixed_after_sweep"] if ascending["sweeps"] else []
    first_desc = descending["sweeps"][0]["coefficients_fixed_after_sweep"] if descending["sweeps"] else []
    first_diff = [d - a for a, d in zip(first_asc, first_desc)]
    full_diff = [d - a for a, d in zip(ascending["coefficients_fixed"], descending["coefficients_fixed"])]
    pair_effects = []
    for pair in strong_pairs:
        i, j = pair["i"], pair["j"]
        pair_effects.append({
            "columns": pair["columns"],
            "i": i,
            "j": j,
            "first_sweep_ascending": [first_asc[i], first_asc[j]],
            "first_sweep_descending": [first_desc[i], first_desc[j]],
            "first_sweep_difference": [first_diff[i], first_diff[j]],
            "full_solve_difference": [full_diff[i], full_diff[j]],
            "order_effect_observed": first_diff[i] != 0 or first_diff[j] != 0 or full_diff[i] != 0 or full_diff[j] != 0,
        })
    return {
        "rule": "coordinate indices ascending; ties never use runtime random state",
        "first_sweep_ascending_fixed": first_asc,
        "first_sweep_descending_fixed": first_desc,
        "first_sweep_difference_fixed": first_diff,
        "full_ascending_fixed": ascending["coefficients_fixed"],
        "full_descending_fixed": descending["coefficients_fixed"],
        "full_difference_fixed": full_diff,
        "pair_effects": pair_effects,
        "ascending_iterations": ascending["iterations"],
        "descending_iterations": descending["iterations"],
    }
