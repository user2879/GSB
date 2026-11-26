from __future__ import annotations

from typing import Any

from .pipeline import LAYER_EXTRAPOLATION, LAYER_INSIDE, layer_indices
from .quant import relative_or_absolute_error


def _mean(values: list[float]) -> float:
    total = 0.0
    for value in values:
        total += float(value)
    return total / len(values) if values else 0.0


def build_layer_stats(sample: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    groups = layer_indices(sample)
    score_rows = {row["key_index"]: row for row in bundle["score"]["rows"]}
    ref_rows = {row["key_index"]: row for row in bundle["reference"]["rows"]}
    result = {}
    for layer in (LAYER_INSIDE, LAYER_EXTRAPOLATION):
        indexes = groups[layer]
        score_values = [score_rows[i]["score_float"] for i in indexes]
        weight_values = [score_rows[i]["weight"] for i in indexes]
        score_deviations = [
            abs(score_rows[i]["score_float"] - ref_rows[i]["score"]) for i in indexes
        ]
        weight_deviations = [
            abs(score_rows[i]["weight"] - ref_rows[i]["weight"]) for i in indexes
        ]
        max_score_deviation = max(score_deviations) if score_deviations else 0.0
        max_weight_deviation = max(weight_deviations) if weight_deviations else 0.0
        result[layer] = {
            "sample_id": sample["id"],
            "scheme": bundle["position"]["scheme"],
            "sample_count": 1 if indexes else 0,
            "key_count": len(indexes),
            "key_indices": indexes,
            "score_mean": _mean(score_values),
            "weight_mean": _mean(weight_values),
            "max_score_deviation": max_score_deviation,
            "max_weight_deviation": max_weight_deviation,
        }
    return result


def _stage_summary(stage: str, comparisons: list[dict[str, Any]],
                   tolerance_rel: float, tolerance_abs: float) -> dict[str, Any]:
    first_exceedance = None
    max_relative = None
    max_absolute = 0.0
    absolute_branch_count = 0
    relative_branch_count = 0

    for comparison in comparisons:
        error = comparison["error"]
        if error["branch"] == "absolute_zero_reference":
            absolute_branch_count += 1
            threshold = tolerance_abs
            exceeded = error["absolute_error"] > tolerance_abs
        else:
            relative_branch_count += 1
            threshold = tolerance_rel
            exceeded = error["relative_error"] > tolerance_rel
            if max_relative is None or error["relative_error"] > max_relative:
                max_relative = error["relative_error"]
        max_absolute = max(max_absolute, error["absolute_error"])
        if exceeded and first_exceedance is None:
            first_exceedance = {
                "stage": stage,
                "threshold": threshold,
                "key_index": comparison.get("key_index"),
                "position": comparison.get("position"),
                "dimension": comparison.get("dimension"),
                "branch": error["branch"],
                "metric": error["metric"],
                "absolute_error": error["absolute_error"],
                "relative_error": error["relative_error"],
            }

    return {
        "stage": stage,
        "exceeded": first_exceedance is not None,
        "tolerance_rel": tolerance_rel,
        "tolerance_abs": tolerance_abs,
        "relative_branch_count": relative_branch_count,
        "absolute_zero_reference_count": absolute_branch_count,
        "max_relative_error": max_relative,
        "max_absolute_error": max_absolute,
        "first_exceedance": first_exceedance,
    }


def build_error_trace(sample: dict[str, Any], scheme: dict[str, Any],
                      config: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    tolerances = config["tolerances"]
    groups = layer_indices(sample)
    extrap_indexes = groups[LAYER_EXTRAPOLATION]
    position = bundle["position"]
    float_encoding = bundle["float_encoding"]
    score_rows = {row["key_index"]: row for row in bundle["score"]["rows"]}
    ref_rows = {row["key_index"]: row for row in bundle["reference"]["rows"]}

    position_comparisons: list[dict[str, Any]] = []
    query_position = sample["query_position"]
    if query_position >= int(sample["train_length"]):
        for dimension_index, quantization in enumerate(
                position["query"]["pe_quantization"]):
            error = relative_or_absolute_error(
                float_encoding["query"]["pe"][dimension_index],
                quantization["dequantized"],
            )
            position_comparisons.append({
                "key_index": None,
                "position": query_position,
                "dimension": dimension_index,
                "error": error,
            })

    for key_index in extrap_indexes:
        fixed_key = position["keys"][key_index]
        float_key = float_encoding["keys"][key_index]
        for dimension_index, quantization in enumerate(fixed_key["pe_quantization"]):
            error = relative_or_absolute_error(
                float_key["pe"][dimension_index],
                quantization["dequantized"],
            )
            position_comparisons.append({
                "key_index": key_index,
                "position": fixed_key["position"],
                "dimension": dimension_index,
                "error": error,
            })

    score_comparisons = []
    weight_comparisons = []
    for key_index in extrap_indexes:
        fixed_row = score_rows[key_index]
        ref_row = ref_rows[key_index]
        score_error = relative_or_absolute_error(ref_row["score"], fixed_row["score_float"])
        score_comparisons.append({
            "key_index": key_index,
            "position": fixed_row["position"],
            "dimension": None,
            "error": score_error,
        })
        weight_error = relative_or_absolute_error(ref_row["weight"], fixed_row["weight"])
        weight_comparisons.append({
            "key_index": key_index,
            "position": fixed_row["position"],
            "dimension": None,
            "error": weight_error,
        })

    stages = {
        "position": _stage_summary(
            "position", position_comparisons,
            tolerances["position_rel"], tolerances["position_abs"],
        ),
        "score": _stage_summary(
            "score", score_comparisons,
            tolerances["score_rel"], tolerances["score_abs"],
        ),
        "weight": _stage_summary(
            "weight", weight_comparisons,
            tolerances["weight_rel"], tolerances["weight_abs"],
        ),
    }
    first_stage = None
    first_metric = None
    for stage_name in ("position", "score", "weight"):
        if stages[stage_name]["exceeded"]:
            first_stage = stage_name
            first_metric = stages[stage_name]["first_exceedance"]["metric"]
            break

    return {
        "sample_id": sample["id"],
        "scheme": scheme["name"],
        "extrapolation_key_indices": extrap_indexes,
        "extrapolation_point_count": len(extrap_indexes),
        "stages": stages,
        "first_exceeded_stage": first_stage,
        "first_exceeded_metric": first_metric,
        "first_exceedance": stages[first_stage]["first_exceedance"] if first_stage else None,
    }
