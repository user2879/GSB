from __future__ import annotations

from typing import Any, Iterable

from .config import STAGES, FeatureConfig, TrainingConfig
from .features import FeatureSnapshot
from .fixed import fraction_scale
from .fusion import (
    compare_rankings,
    evaluation_metrics,
)


DIAGNOSTIC_SCHEMA = "diagnostic-summary-v1"


def _scalar_error(fixed: float, reference: float) -> dict[str, Any]:
    absolute = abs(fixed - reference)
    if reference == 0.0:
        return {
            "fixed": fixed,
            "reference": reference,
            "absolute_error": absolute,
            "relative_error": None,
            "comparison_branch": "ABSOLUTE_ZERO_REFERENCE",
            "deviation": absolute,
        }
    relative = absolute / abs(reference)
    return {
        "fixed": fixed,
        "reference": reference,
        "absolute_error": absolute,
        "relative_error": relative,
        "comparison_branch": "RELATIVE",
        "deviation": relative,
    }


def _vector_error(
    fixed_values: Iterable[float],
    reference_values: Iterable[float],
    *,
    element_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    fixed_list = list(fixed_values)
    reference_list = list(reference_values)
    names = list(element_names) if element_names is not None else [str(i) for i in range(len(fixed_list))]
    if not fixed_list:
        return {
            "fixed": [],
            "reference": [],
            "absolute_error": 0.0,
            "relative_error": None,
            "comparison_branch": "ABSOLUTE_ZERO_REFERENCE",
            "deviation": 0.0,
            "element": None,
        }
    per_element = [
        {"element": name, **_scalar_error(float(fixed), float(reference))}
        for name, fixed, reference in zip(names, fixed_list, reference_list)
    ]
    selected = max(per_element, key=lambda item: (item["deviation"], item["element"] or ""))
    return {
        "fixed": fixed_list,
        "reference": reference_list,
        "elements": per_element,
        "absolute_error": selected["absolute_error"],
        "relative_error": selected["relative_error"],
        "comparison_branch": selected["comparison_branch"],
        "deviation": selected["deviation"],
        "element": selected["element"],
    }


def _threshold_result(error: dict[str, Any], tolerance: float) -> dict[str, Any]:
    return {
        **error,
        "tolerance": tolerance,
        "exceeded": error["deviation"] > tolerance,
    }


def _pointwise_diagnostics(
    config: FeatureConfig,
    training: TrainingConfig,
    feature_snapshot: FeatureSnapshot,
    training_payload: dict[str, Any],
    fusion_payload: dict[str, Any],
    labels_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    scale = fraction_scale(training_payload["fractional_bits"])
    fusion_by_key = {
        (row["request_id"], row["item_id"]): row for row in fusion_payload["records"]
    }
    points: list[dict[str, Any]] = []
    selected = set(training_payload["selected_samples"])
    pending = set(training_payload["pending_samples"])
    labels = {
        (item["sample"].split(":", 1)[0], item["sample"].split(":", 1)[1]): item
        for item in labels_payload["samples"]
    }
    for sample_key in sorted(feature_snapshot.entries):
        sample_id = f"{sample_key[0]}:{sample_key[1]}"
        entry_rows = feature_snapshot.entries_sorted(sample_key)
        feature_errors = [
            {
                "feature": row["feature"],
                "kind": row["kind"],
                **_scalar_error(float(row["dequantized"]), float(row["raw_value"])),
            }
            for row in entry_rows
        ]
        selected_feature = max(
            feature_errors,
            key=lambda item: (item["deviation"], item["feature"]),
        )
        feature_stage = {
            "fixed": [row["dequantized"] for row in entry_rows],
            "reference": [row["raw_value"] for row in entry_rows],
            "elements": feature_errors,
            "absolute_error": selected_feature["absolute_error"],
            "relative_error": selected_feature["relative_error"],
            "comparison_branch": selected_feature["comparison_branch"],
            "deviation": selected_feature["deviation"],
            "element": selected_feature["feature"],
        }
        hidden_fixed = [value / scale for value in training_payload["hidden_fixed"][sample_id]]
        hidden_ref = training_payload["hidden_reference"][sample_id]
        hidden_stage = _vector_error(
            hidden_fixed,
            hidden_ref,
            element_names=[f"h{index}" for index in range(len(hidden_fixed))],
        )
        fusion_row = fusion_by_key[sample_key]
        ctr_stage = _scalar_error(
            float(fusion_row["ctr_score"]), float(fusion_row["ctr_reference_score"])
        )
        cvr_stage = _scalar_error(
            float(fusion_row["cvr_score"]), float(fusion_row["cvr_reference_score"])
        )
        fusion_stage = _scalar_error(
            float(fusion_row["fusion_score"]), float(fusion_row["fusion_reference_score"])
        )
        stage_map = {
            "feature_quantization": feature_stage,
            "shared_hidden": hidden_stage,
            "ctr_tower": ctr_stage,
            "cvr_tower": cvr_stage,
            "fusion": fusion_stage,
        }
        errors = {
            stage: _threshold_result(stage_map[stage], training.tolerances.get(stage))
            for stage in STAGES
        }
        first_exceeded = None
        for stage in STAGES:
            if errors[stage]["exceeded"]:
                first_exceeded = {
                    "stage": stage,
                    "deviation": errors[stage]["deviation"],
                    "absolute_error": errors[stage]["absolute_error"],
                    "relative_error": errors[stage]["relative_error"],
                    "comparison_branch": errors[stage]["comparison_branch"],
                    "element": errors[stage].get("element"),
                    "tolerance": errors[stage]["tolerance"],
                }
                break
        labels_row = labels[sample_key]
        points.append(
            {
                "sample_id": sample_id,
                "request_id": sample_key[0],
                "item_id": sample_key[1],
                "training_selected": sample_id in selected,
                "training_budget_pending": sample_id in pending,
                "fixed": {
                    "ctr_score_q": fusion_row["ctr_score_q"],
                    "ctr_score": fusion_row["ctr_score"],
                    "cvr_score_q": fusion_row["cvr_score_q"],
                    "cvr_score": fusion_row["cvr_score"],
                    "fusion_score_q": fusion_row["fusion_score_q"],
                    "fusion_score": fusion_row["fusion_score"],
                    "rank": fusion_row["fixed_rank"],
                    "global_rank": fusion_row["global_fixed_rank"],
                    "shared_hidden_q": training_payload["hidden_fixed"][sample_id],
                },
                "reference": {
                    "ctr_score": fusion_row["ctr_reference_score"],
                    "cvr_score": fusion_row["cvr_reference_score"],
                    "fusion_score": fusion_row["fusion_reference_score"],
                    "rank": fusion_row["reference_rank"],
                    "global_rank": fusion_row["global_reference_rank"],
                    "shared_hidden": training_payload["hidden_reference"][sample_id],
                },
                "labels": {
                    "click": labels_row["click"],
                    "conversion": labels_row["conversion"],
                    "conversion_delay_ms": labels_row["delay_ms"],
                },
                "cvr_visibility": labels_row["visibility"],
                "feature_quantization": entry_rows,
                "fusion_rounding": fusion_row["fusion_rounding"],
                "stage_errors": errors,
                "first_exceeded": first_exceeded,
            }
        )
    return points


def _stage_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in STAGES:
        pairs = [(point, point["stage_errors"][stage]) for point in points]
        errors = [error for _, error in pairs]
        max_pair = max(pairs, key=lambda item: item[1]["deviation"], default=None)
        relative_values = [item["relative_error"] for item in errors if item["relative_error"] is not None]
        result[stage] = {
            "sample_count": len(errors),
            "exceeded_count": sum(1 for item in errors if item["exceeded"]),
            "zero_reference_branch_count": sum(
                1 for item in errors if item["comparison_branch"] == "ABSOLUTE_ZERO_REFERENCE"
            ),
            "max_absolute_error": max((item["absolute_error"] for item in errors), default=0.0),
            "max_relative_error": max(relative_values, default=None),
            "max_deviation": max((item["deviation"] for item in errors), default=0.0),
            "max_deviation_sample": max_pair[0]["sample_id"] if max_pair else None,
            "max_deviation_element": max_pair[1].get("element") if max_pair else None,
            "tolerance": errors[0]["tolerance"] if errors else None,
        }
    return result


def _enrich_flips(
    flips: list[dict[str, Any]],
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    first_by_key = {
        (point["request_id"], point["item_id"]): point["first_exceeded"] for point in points
    }
    enriched = []
    for flip in flips:
        key = (flip["request_id"], flip["item_id"])
        other = (flip["request_id"], flip["other_item_id"])
        enriched.append(
            {
                **flip,
                "item_first_exceeded_stage": (
                    first_by_key[key]["stage"] if first_by_key.get(key) is not None else None
                ),
                "other_first_exceeded_stage": (
                    first_by_key[other]["stage"] if first_by_key.get(other) is not None else None
                ),
            }
        )
    return enriched


def _reference_flips(
    fusion_payload: dict[str, Any],
    mode: str,
    ctr_weight: float,
    cvr_weight: float,
) -> list[dict[str, Any]]:
    fixed_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for row in fusion_payload["records"]:
        fixed = dict(row)
        fixed["fusion_score"] = row["fusion_score"]
        fixed["ctr_score"] = row["ctr_score"]
        fixed["cvr_score"] = row["cvr_score"]
        reference = dict(row)
        reference["fusion_score"] = row["fusion_reference_score"]
        reference["ctr_score"] = row["ctr_reference_score"]
        reference["cvr_score"] = row["cvr_reference_score"]
        fixed_rows.append(fixed)
        reference_rows.append(reference)
    return compare_rankings(
        reference_rows,
        fixed_rows,
        mode=mode,
        ctr_weight=ctr_weight,
        cvr_weight=cvr_weight,
        comparison={"comparison_kind": "float64_reference_vs_fixed"},
    )


def build_diagnostic_payload(
    config: FeatureConfig,
    training: TrainingConfig,
    feature_snapshot: FeatureSnapshot,
    training_payload: dict[str, Any],
    fusion_payload: dict[str, Any],
    neighbor_results: list[dict[str, Any]],
    labels_payload: dict[str, Any],
) -> dict[str, Any]:
    points = _pointwise_diagnostics(
        config,
        training,
        feature_snapshot,
        training_payload,
        fusion_payload,
        labels_payload,
    )
    current_records = fusion_payload["records"]
    quantization_flips: list[dict[str, Any]] = []
    for neighbor in neighbor_results:
        comparison = {
            "comparison_kind": "adjacent_quantization",
            "variant_kind": neighbor["variant_kind"],
            "variant_from": neighbor["from"],
            "variant_to": neighbor["to"],
        }
        flips = compare_rankings(
            current_records,
            neighbor["fusion"]["records"],
            mode=config.fusion.mode,
            ctr_weight=config.fusion.ctr_weight,
            cvr_weight=config.fusion.cvr_weight,
            comparison=comparison,
        )
        quantization_flips.extend(flips)
    reference_flips = _reference_flips(
        fusion_payload,
        config.fusion.mode,
        config.fusion.ctr_weight,
        config.fusion.cvr_weight,
    )
    labels = {
        (item["sample"].split(":", 1)[0], item["sample"].split(":", 1)[1]): {
            "click": item["click"],
            "conversion": item["visibility"]["effective_target"],
        }
        for item in labels_payload["samples"]
    }
    metrics = evaluation_metrics(fusion_payload, labels)
    return {
        "schema": DIAGNOSTIC_SCHEMA,
        "points": points,
        "stage_summary": _stage_summary(points),
        "quantization_flips": _enrich_flips(quantization_flips, points),
        "reference_flips": _enrich_flips(reference_flips, points),
        "metrics": metrics,
        "neighbor_definitions": [
            {
                "variant_kind": neighbor["variant_kind"],
                "from": neighbor["from"],
                "to": neighbor["to"],
                "tier": neighbor["tier"],
                "fractional_bits": neighbor["fractional_bits"],
            }
            for neighbor in neighbor_results
        ],
    }
