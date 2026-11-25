"""Per-stage fixed-point error attribution."""

from __future__ import annotations

from typing import Any, Sequence

from .reference import pad_vector

STAGE_ORDER = (
    "interpolation",
    "normal_equation_accumulation",
    "integer_elimination",
    "coefficient_back_substitution",
)


def _clean(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _coefficient_error(
    *,
    name: str,
    truth: float,
    stage_value: float,
    delta: float,
    cumulative: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    absolute_error = abs(cumulative)
    zero_truth = truth == 0.0
    if zero_truth:
        metric = absolute_error
        metric_kind = "absolute_zero_truth"
        tolerance = absolute_tolerance
        exceeded = metric > tolerance
    else:
        metric = absolute_error / abs(truth)
        metric_kind = "relative_nonzero_truth"
        tolerance = relative_tolerance
        exceeded = metric > tolerance
    return {
        "name": name,
        "truth_value": _clean(truth),
        "stage_value": _clean(stage_value),
        "stage_delta": _clean(delta),
        "cumulative_error": _clean(cumulative),
        "absolute_error": _clean(absolute_error),
        "metric": _clean(metric),
        "metric_kind": metric_kind,
        "tolerance": tolerance,
        "exceeded": exceeded,
    }


def attribute_stage_errors(
    *,
    names: Sequence[str],
    truth: Sequence[float],
    interpolated: Sequence[float],
    normal_accumulated: Sequence[float],
    eliminated: Sequence[float],
    final: Sequence[float],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    dimension = len(names)
    truth_vector = pad_vector(truth, dimension)
    stage_vectors = {
        "interpolation": pad_vector(interpolated, dimension),
        "normal_equation_accumulation": pad_vector(normal_accumulated, dimension),
        "integer_elimination": pad_vector(eliminated, dimension),
        "coefficient_back_substitution": pad_vector(final, dimension),
    }
    previous = truth_vector
    results: dict[str, Any] = {}
    first_stage: str | None = None
    first_coefficients: list[str] = []
    for stage in STAGE_ORDER:
        current = stage_vectors[stage]
        entries: list[dict[str, Any]] = []
        stage_exceeded: list[str] = []
        for index, name in enumerate(names):
            delta = current[index] - previous[index]
            cumulative = current[index] - truth_vector[index]
            entry = _coefficient_error(
                name=name,
                truth=truth_vector[index],
                stage_value=current[index],
                delta=delta,
                cumulative=cumulative,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
            )
            entries.append(entry)
            if entry["exceeded"]:
                stage_exceeded.append(name)
        results[stage] = {
            "cumulative_stage_value": [_clean(value) for value in current],
            "coefficients": entries,
            "exceeded_coefficients": stage_exceeded,
        }
        if first_stage is None and stage_exceeded:
            first_stage = stage
            first_coefficients = stage_exceeded
        previous = current

    final_entries = results["coefficient_back_substitution"]["coefficients"]
    max_relative = max(
        (
            entry["metric"]
            for entry in final_entries
            if entry["metric_kind"] == "relative_nonzero_truth"
        ),
        default=0.0,
    )
    max_absolute = max(
        (entry["absolute_error"] for entry in final_entries),
        default=0.0,
    )
    return {
        "truth_path": "float64_observed_only_no_imputation",
        "coefficient_names": list(names),
        "truth_coefficients": [_clean(value) for value in truth_vector],
        "stage_order": list(STAGE_ORDER),
        "stages": results,
        "first_exceeded_stage": first_stage,
        "first_exceeded_coefficients": first_coefficients,
        "final_max_relative_error_nonzero_truth": _clean(max_relative),
        "final_max_absolute_error": _clean(max_absolute),
        "zero_truth_branch_coefficients": [
            entry["name"]
            for entry in final_entries
            if entry["metric_kind"] == "absolute_zero_truth"
        ],
    }
