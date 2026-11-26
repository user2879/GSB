"""Staged float64 reference and fixed-point error propagation attribution."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import numpy as np

from .dataset import Dataset
from .features import QuantizedDataset


def raw_float_arrays(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = dataset.ids
    x = np.array([[float(dataset.rows[sid].features[j]) for j in range(len(dataset.columns))] for sid in ids], dtype=np.float64)
    y = np.array([float(dataset.rows[sid].target) for sid in ids], dtype=np.float64)
    w = np.array([float(dataset.rows[sid].weight) for sid in ids], dtype=np.float64)
    return x, y, w


def quant_float_arrays(qd: QuantizedDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.float64(qd.params.scale)
    x = np.array(qd.x, dtype=np.float64) / scale
    y = np.array(qd.y, dtype=np.float64) / scale
    w = np.array(qd.w, dtype=np.float64) / scale
    return x, y, w


def ordered_float_quadratics(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, p = x.shape
    g = np.zeros((p, p), dtype=np.float64)
    b = np.zeros(p, dtype=np.float64)
    for j in range(p):
        for i in range(j + 1):
            acc = np.float64(0.0)
            for s in range(n):
                acc = acc + np.float64(w[s]) * np.float64(x[s, i]) * np.float64(x[s, j])
            g[i, j] = np.float64(acc)
            g[j, i] = np.float64(acc)
        rhs_acc = np.float64(0.0)
        for s in range(n):
            rhs_acc = rhs_acc + np.float64(w[s]) * np.float64(x[s, j]) * np.float64(y[s])
        b[j] = rhs_acc
    return g, b


def float_ridge(g: np.ndarray, b: np.ndarray, alpha: Decimal) -> list[float]:
    p = g.shape[0]
    matrix = g + np.float64(float(alpha)) * np.eye(p, dtype=np.float64)
    beta = np.linalg.solve(matrix, b)
    return [float(v) for v in beta]


def _soft_float(value: np.float64, threshold: float) -> np.float64:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return np.float64(0.0)


def float_lasso(
    g: np.ndarray,
    b: np.ndarray,
    alpha: Decimal,
    solver_config: dict[str, Any],
    *,
    order: str = "ascending",
) -> list[float]:
    p = g.shape[0]
    beta = np.zeros(p, dtype=np.float64)
    alpha_float = float(alpha)
    tolerance = float(solver_config["coefficient_tolerance"])
    max_iterations = int(solver_config["max_iterations"])
    indices = range(p) if order == "ascending" else range(p - 1, -1, -1)
    for _iteration in range(max_iterations):
        max_delta = 0.0
        for j in indices:
            z = np.float64(b[j])
            for k in range(p):
                if k != j:
                    z = z - np.float64(g[j, k]) * beta[k]
            denominator = np.float64(g[j, j])
            new_value = np.float64(0.0) if denominator == 0 else _soft_float(z, alpha_float) / denominator
            max_delta = max(max_delta, abs(float(new_value - beta[j])))
            beta[j] = new_value
        if max_delta <= tolerance:
            break
    return [float(v) for v in beta]


def _vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [float(x - y) for x, y in zip(a, b)]


def _max_abs(values: list[float]) -> float:
    return max((abs(v) for v in values), default=0.0)


def _stage_event(beta: list[float], half_band: float, sign_policy_zero: bool = True) -> list[dict[str, Any]]:
    signs = []
    nonzero = []
    for value in beta:
        is_zero = abs(value) <= half_band if sign_policy_zero else value == 0.0
        nonzero.append(not is_zero)
        signs.append(0 if is_zero else (1 if value > 0 else -1))
    return [{"nonzero": nonzero, "signs": signs}]


def build_attribution(
    *,
    dataset: Dataset,
    qd: QuantizedDataset,
    quadratics,
    penalty: str,
    alphas: list[Decimal],
    solver_config: dict[str, Any],
    path_report: dict[str, Any],
    order_audits: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    x_raw, y_raw, w_raw = raw_float_arrays(dataset)
    x_q, _y_q_fixed, _w_q_fixed = quant_float_arrays(qd)
    g_ref, b_ref = ordered_float_quadratics(x_raw, y_raw, w_raw)
    g_q, b_q = ordered_float_quadratics(x_q, y_raw, w_raw)
    accum_scale = np.float64(qd.params.accum_scale)
    g_acc = np.array(quadratics.g, dtype=np.float64) / accum_scale
    b_acc = np.array(quadratics.b, dtype=np.float64) / accum_scale
    tolerance = float(solver_config.get("attribution_tolerance", Decimal("1e-6")))
    half_band = float(Decimal(qd.params.grid) / (Decimal(2) * Decimal(qd.params.scale)))

    stage_by_alpha: dict[str, dict[str, list[float]]] = {}
    per_alpha: list[dict[str, Any]] = []

    for alpha in alphas:
        key = format(alpha, "f")
        if penalty == "ridge":
            reference = float_ridge(g_ref, b_ref, alpha)
            feature_quant = float_ridge(g_q, b_q, alpha)
            accumulation = float_ridge(g_acc, b_acc, alpha)
        else:
            reference = float_lasso(g_ref, b_ref, alpha, solver_config)
            feature_quant = float_lasso(g_q, b_q, alpha, solver_config)
            accumulation = float_lasso(g_acc, b_acc, alpha, solver_config)

        level = next(item for item in path_report["levels"] if item["alpha_key"] == key)
        integer_solve = list(level["coefficients_raw"])
        path_summary = list(level["coefficients"])
        stages = {
            "reference_float64": reference,
            "feature_quantization": feature_quant,
            "accumulation_or_coordinate_float": accumulation,
            "integer_solve": integer_solve,
            "path_summary": path_summary,
        }
        stage_by_alpha[key] = stages

        contributions = {
            "quantization_truncation": _vector_sub(feature_quant, reference),
            "accumulation_rounding": _vector_sub(accumulation, feature_quant),
            "elimination_pivot_or_coordinate_update": _vector_sub(integer_solve, accumulation),
            "path_summary_zeroing": _vector_sub(path_summary, integer_solve),
        }
        if penalty == "lasso" and order_audits and key in order_audits:
            audit = order_audits[key]
            descending = [float(Decimal(v) / Decimal(qd.params.scale)) for v in audit["full_descending_fixed"]]
            contributions["coordinate_order_desc_minus_asc"] = _vector_sub(descending, integer_solve)

        per_coordinate = []
        for j, column in enumerate(dataset.columns):
            stage_errors = {
                name: abs(values[j] - reference[j])
                for name, values in stages.items()
                if name != "reference_float64"
            }
            first_stage = None
            equal_boundary = False
            for stage_name in [
                "feature_quantization",
                "accumulation_or_coordinate_float",
                "integer_solve",
                "path_summary",
            ]:
                error = stage_errors[stage_name]
                if error == tolerance:
                    equal_boundary = True
                if error > tolerance and first_stage is None:
                    first_stage = stage_name
            per_coordinate.append({
                "coordinate": j,
                "column": column,
                "reference": reference[j],
                "stages": {name: values[j] for name, values in stages.items()},
                "absolute_error_vs_reference": stage_errors,
                "first_stage_over_tolerance": first_stage,
                "equal_tolerance_boundary": equal_boundary,
                "contributions": {name: values[j] for name, values in contributions.items()},
            })

        global_first_stage = None
        for stage_name in [
            "feature_quantization",
            "accumulation_or_coordinate_float",
            "integer_solve",
            "path_summary",
        ]:
            if any(record["absolute_error_vs_reference"][stage_name] > tolerance for record in per_coordinate):
                global_first_stage = stage_name
                break

        per_alpha.append({
            "alpha": key,
            "tolerance": tolerance,
            "stage_max_abs_error": {name: _max_abs(_vector_sub(values, reference)) for name, values in stages.items() if name != "reference_float64"},
            "first_stage_over_tolerance": global_first_stage,
            "coefficient_records": per_coordinate,
            "contribution_max_abs": {name: _max_abs(values) for name, values in contributions.items()},
        })

    event_attribution = _attribute_events(path_report, stage_by_alpha, half_band)
    return {
        "reference": "float64 raw Decimal tokens converted to float; ordered scalar accumulation",
        "half_zero_band": half_band,
        "tolerance": tolerance,
        "per_alpha": per_alpha,
        "event_attribution": event_attribution,
    }


def _classify_vector(beta: list[float], half_band: float) -> tuple[list[bool], list[int]]:
    nonzero = [abs(value) > half_band for value in beta]
    signs = [0 if not flag else (1 if value > 0 else -1) for flag, value in zip(nonzero, beta)]
    return nonzero, signs


def _event_for_pair(before: list[float], after: list[float], coordinate: int, half_band: float) -> str | None:
    before_nonzero, before_signs = _classify_vector(before, half_band)
    after_nonzero, after_signs = _classify_vector(after, half_band)
    bz = not before_nonzero[coordinate]
    az = not after_nonzero[coordinate]
    if bz and not az:
        return "enter"
    if not bz and az:
        return "exit"
    if not bz and not az and before_signs[coordinate] != after_signs[coordinate]:
        return "sign_flip"
    return None


def _attribute_events(
    path_report: dict[str, Any],
    stage_by_alpha: dict[str, dict[str, list[float]]],
    half_band: float,
) -> list[dict[str, Any]]:
    results = []
    stage_names = [
        "reference_float64",
        "feature_quantization",
        "accumulation_or_coordinate_float",
        "integer_solve",
        "path_summary",
    ]
    contribution_names = [
        "quantization_truncation",
        "accumulation_rounding",
        "elimination_pivot_or_coordinate_update",
        "path_summary_zeroing",
    ]
    for event in path_report["events"]:
        alpha_from = event["alpha_from"]
        alpha_to = event["alpha_to"]
        coordinate = event["coordinate"]
        stage_presence: dict[str, str | None] = {}
        first_matching = None
        first_creating = None
        previous_event = None
        for stage_name in stage_names:
            event_type = _event_for_pair(
                stage_by_alpha[alpha_from][stage_name],
                stage_by_alpha[alpha_to][stage_name],
                coordinate,
                half_band,
            )
            stage_presence[stage_name] = event_type
            if stage_name != "reference_float64":
                if first_matching is None and event_type == event["type"]:
                    first_matching = stage_name
                if first_creating is None and event_type == event["type"] and previous_event != event["type"]:
                    first_creating = stage_name
            previous_event = event_type

        reference_event = stage_presence["reference_float64"]
        if reference_event == event["type"] and first_creating is None:
            causal_stage = "reference_already_present"
        else:
            causal_stage = first_creating or first_matching or "reference_already_present"

        transition_sources: dict[str, float] = {}
        for contribution_name in contribution_names:
            # Reconstruct contribution at each endpoint from stage records indirectly.
            # The named contribution is a stage-to-stage vector, so endpoint maps are
            # obtained from the per-alpha stage chain by callers? Here use differences.
            pass
        # Direct transition distortion by stage cumulative error:
        for stage_name in stage_names[1:]:
            err_from = _vector_sub(stage_by_alpha[alpha_from][stage_name], stage_by_alpha[alpha_from]["reference_float64"])
            err_to = _vector_sub(stage_by_alpha[alpha_to][stage_name], stage_by_alpha[alpha_to]["reference_float64"])
            transition_sources[stage_name] = float(err_to[coordinate] - err_from[coordinate])

        source_label = max(transition_sources, key=lambda key: abs(transition_sources[key])) if transition_sources else None
        results.append({
            "event_id": event["event_id"],
            "type": event["type"],
            "column": event["column"],
            "coordinate": coordinate,
            "alpha_from": alpha_from,
            "alpha_to": alpha_to,
            "final_event": event["type"],
            "stage_event_presence": stage_presence,
            "first_stage_matching_final": first_matching,
            "first_stage_creating_from_different_previous": first_creating,
            "causal_stage": causal_stage,
            "transition_source_contributions": transition_sources,
            "largest_abs_source_stage": source_label,
            "note": "transition source is change in cumulative stage-vs-reference error between adjacent alphas",
        })
    return results
