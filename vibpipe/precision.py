from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import LoadedConfig, topological_stages
from .fixedpoint import convert_q, next_gear
from .hashing import fingerprint
from .samples import Sample


def build_attribution(
    config: LoadedConfig,
    samples: list[Sample],
    normal_products: dict[str, dict[str, Any]],
    normal_stats: dict[str, dict[str, list[dict[str, Any]]]],
    normal_drift: list[dict[str, Any]],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    from .engine import run_pipeline

    high_gear = next_gear(config.gear)
    high_result = run_pipeline(
        config,
        samples,
        None,
        gear_override=high_gear,
        include_attribution=False,
        use_cache=False,
        max_workers=1,
    )
    high_report = high_result["report"]
    high_stats = high_report["stats"]
    high_drift = high_report["drift"]
    canonical_scale = 10 ** config.decimal_width
    tolerance_q = config.tolerance_q
    stages = topological_stages(config.stages)
    ordered_stage_names = ["source"] + [stage["name"] for stage in stages]

    stage_rows: list[dict[str, Any]] = []
    first_exceeded_stage: str | None = None
    for stage_rank, stage_name in enumerate(ordered_stage_names):
        normal_windows = normal_stats.get(stage_name, {})
        high_windows = high_stats.get(stage_name, {})
        for window_name in ("reference", "observation"):
            normal_list = normal_windows.get(window_name, [])
            high_list = high_windows.get(window_name, [])
            for index, normal_stat in enumerate(normal_list):
                high_stat = high_list[index]
                diff_q = abs(int(high_stat["mean_q"]) - int(normal_stat["mean_q"]))
                diff_canonical_q = convert_q(
                    diff_q, int(normal_stat["scale"]), canonical_scale
                )
                exceeds = diff_canonical_q > tolerance_q
                if exceeds and first_exceeded_stage is None:
                    first_exceeded_stage = stage_name
                stage_rows.append(
                    {
                        "stage_rank": stage_rank,
                        "stage": stage_name,
                        "feature_index": index,
                        "feature_id": normal_stat["feature_id"],
                        "window": window_name,
                        "normal_mean_q": normal_stat["mean_q"],
                        "high_mean_q": high_stat["mean_q"],
                        "mean_diff_q": diff_q,
                        "diff_canonical_q": diff_canonical_q,
                        "canonical_scale": canonical_scale,
                        "tolerance_q": tolerance_q,
                        "exceeds_tolerance": exceeds,
                    }
                )

    disagreements = classify_disagreements(
        config,
        normal_products,
        normal_drift,
        high_drift,
        stage_rows,
        ordered_stage_names,
        tolerance_q,
    )
    return {
        "configured_gear": config.gear,
        "high_precision_gear": high_gear,
        "canonical_scale": canonical_scale,
        "tolerance_q": tolerance_q,
        "first_exceeded_stage": first_exceeded_stage,
        "stage_rows": stage_rows,
        "drift_disagreements": disagreements,
        "disagreement_count": len(disagreements),
        "high_precision_report_fingerprint": fingerprint(
            {"stats": high_stats, "drift": high_drift}
        ),
    }


def classify_disagreements(
    config: LoadedConfig,
    products: dict[str, dict[str, Any]],
    normal_drift: list[dict[str, Any]],
    high_drift: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    ordered_stage_names: list[str],
    tolerance_q: int,
) -> list[dict[str, Any]]:
    high_by_id = {row["feature_id"]: row for row in high_drift}
    disagreements: list[dict[str, Any]] = []
    final_stage = str(config.raw.get("feature_stage", ordered_stage_names[-1]))
    final_product = products[final_stage]

    for normal_row in normal_drift:
        feature_id = normal_row["feature_id"]
        high_row = high_by_id.get(feature_id)
        if high_row is None or normal_row["drift"] == high_row["drift"]:
            continue
        lineage_ids_by_stage = build_lineage(feature_id, final_product, products)
        lineage_diffs: list[dict[str, Any]] = []
        first_lineage_exceed: dict[str, Any] | None = None
        for stage_name in ordered_stage_names:
            ids = lineage_ids_by_stage.get(stage_name, {feature_id})
            rows = [
                row
                for row in stage_rows
                if row["stage"] == stage_name and row["feature_id"] in ids
            ]
            max_row = max(rows, key=lambda row: row["diff_canonical_q"], default=None)
            if max_row is None:
                continue
            lineage_diffs.append(
                {
                    "stage": stage_name,
                    "feature_ids": sorted(ids),
                    "max_diff_canonical_q": max_row["diff_canonical_q"],
                    "exceeds_tolerance": max_row["exceeds_tolerance"],
                }
            )
            if first_lineage_exceed is None and max_row["exceeds_tolerance"]:
                first_lineage_exceed = lineage_diffs[-1]

        boundary = bool(normal_row["boundary"] or high_row["boundary"])
        normal_distance = normal_row["distance_to_threshold_q"]
        high_distance = high_row["distance_to_threshold_q"]
        if boundary:
            classification = "threshold_boundary_decision"
            reason = "one or both decisions lie exactly on or within half a quantization step of the threshold"
        elif first_lineage_exceed and _stage_type(config, first_lineage_exceed["stage"]) == "log":
            classification = "log_domain_approximation"
            reason = "first tolerance exceedance on the feature lineage occurs in the integer log approximation stage"
        else:
            classification = "quantization_truncation"
            reason = "difference is introduced by fixed-point conversion/truncation before or outside the log approximation"

        disagreements.append(
            {
                "feature_id": feature_id,
                "classification": classification,
                "reason": reason,
                "normal_drift": normal_row["drift"],
                "high_precision_drift": high_row["drift"],
                "criteria": {
                    "threshold_bp": normal_row["threshold_bp"],
                    "normal_drift_bp_rounded": normal_row["drift_bp_rounded"],
                    "high_drift_bp_rounded": high_row["drift_bp_rounded"],
                    "normal_exact_compare": normal_row["exact_compare"],
                    "high_exact_compare": high_row["exact_compare"],
                    "normal_distance_to_threshold_q": normal_distance,
                    "high_distance_to_threshold_q": high_distance,
                    "normal_half_step_q": normal_row["half_gear_step_q"],
                    "high_half_step_q": high_row["half_gear_step_q"],
                    "normal_boundary_reason": normal_row["boundary_reason"],
                    "high_boundary_reason": high_row["boundary_reason"],
                    "tolerance_q": tolerance_q,
                    "first_lineage_exceeded_stage": None
                    if first_lineage_exceed is None
                    else first_lineage_exceed["stage"],
                    "lineage_diffs": lineage_diffs,
                },
            }
        )
    return disagreements


def _stage_type(config: LoadedConfig, stage_name: str) -> str | None:
    if stage_name == "source":
        return "source"
    for stage in config.stages:
        if stage["name"] == stage_name:
            return str(stage["type"])
    return None


def build_lineage(
    final_feature_id: str,
    final_product: dict[str, Any],
    products: dict[str, dict[str, Any]],
) -> dict[str, set[str]]:
    lineage: dict[str, set[str]] = {final_product["stage"]: {final_feature_id}}
    final_feature = next(
        (f for f in final_product["features"] if f["id"] == final_feature_id), None
    )
    source_feature_id = final_feature.get("source_feature") if final_feature else None
    interaction_product = next(
        (product for product in products.values() if product.get("type") == "interaction"),
        None,
    )
    if interaction_product is None:
        return lineage
    interaction_id = source_feature_id or final_feature_id.replace(
        final_product["stage"] + ":", interaction_product["stage"] + ":", 1
    )
    lineage[interaction_product["stage"]] = {interaction_id}
    interaction_feature = next(
        (f for f in interaction_product["features"] if f["id"] == interaction_id), None
    )
    pair = interaction_feature.get("pair") if interaction_feature else None
    if not pair:
        return lineage

    log_product = next((p for p in products.values() if p.get("type") == "log"), None)
    if log_product is not None:
        ids = set()
        for index in pair:
            if 0 <= index < len(log_product["features"]):
                ids.add(log_product["features"][index]["id"])
        lineage[log_product["stage"]] = ids
        for stage_type in ("standardize", "bucket"):
            product = next((p for p in products.values() if p.get("type") == stage_type), None)
            if product is not None:
                stage_ids = set()
                for index in pair:
                    if 0 <= index < len(product["features"]):
                        stage_ids.add(product["features"][index]["id"])
                lineage[product["stage"]] = stage_ids
        source_product = products.get("source")
        if source_product is not None:
            lineage["source"] = {
                source_product["features"][index]["id"]
                for index in pair
                if 0 <= index < len(source_product["features"])
            }
    return lineage
