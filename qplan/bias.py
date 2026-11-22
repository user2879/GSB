from __future__ import annotations

import math
from typing import Any

from .canonical import content_digest
from .paths import AccessPath
from .spec import QuerySpec, Spec


BIAS_MODEL_VERSION = "qplan-bias-v1"


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 1.0 if numerator <= 0.0 else float("inf")
    return numerator / denominator


def _finite(value: float) -> float | str:
    if math.isfinite(value):
        return round(value, 12)
    return "infinity"


def _bias_outside_threshold(ratio: float, threshold: float) -> bool:
    if not math.isfinite(ratio):
        return True
    return ratio > threshold or ratio < (1.0 / threshold)


def _classify_bias(
    *,
    coverage_mismatch: bool,
    selectivity_ratio: float,
    row_ratio: float,
    join_order_mismatch: bool,
    threshold: float,
) -> tuple[str, dict[str, Any]]:
    selectivity_flagged = _bias_outside_threshold(selectivity_ratio, threshold)
    rows_flagged = _bias_outside_threshold(row_ratio, threshold)
    if coverage_mismatch:
        return "index_coverage_decision_error", {
            "rule": "expected covering flag differs from EXPLAIN QUERY PLAN observation",
            "coverage_mismatch": True,
        }
    if selectivity_flagged:
        return "selectivity_error", {
            "rule": "estimated/actual filtered-row ratio exceeds the declared threshold",
            "selectivity_ratio": _finite(selectivity_ratio),
            "threshold": threshold,
        }
    if join_order_mismatch and (rows_flagged or selectivity_flagged):
        return "join_order_deviation", {
            "rule": "observed SQLite table order differs from the planned join order",
            "join_order_mismatch": True,
        }
    if rows_flagged:
        return "selectivity_error", {
            "rule": "final estimated/actual rows ratio exceeds threshold; no stronger coverage/order evidence",
            "row_ratio": _finite(row_ratio),
        }
    if join_order_mismatch:
        return "join_order_deviation", {
            "rule": "table order differs from the estimated join order",
            "join_order_mismatch": True,
        }
    return "selectivity_error", {
        "rule": "bias fallback: cost/row discrepancy without coverage or order evidence",
    }


def analyze_bias(
    spec: Spec,
    query_records: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_calibration = []
    for record in query_records:
        query: QuerySpec = record["query"]
        baseline_path_id = f"scan:{query.tables[0]}"
        baseline_estimate = record["estimates"][baseline_path_id]
        baseline_execution = record["executions"][baseline_path_id]
        baseline_calibration.append(
            {
                "query_id": query.id,
                "estimated_cost": int(baseline_estimate["cost"]),
                "real_elapsed_ms": float(baseline_execution["real_elapsed_ms"]),
            }
        )
    estimated_cost_sum = sum(item["estimated_cost"] for item in baseline_calibration)
    actual_time_sum = sum(item["real_elapsed_ms"] for item in baseline_calibration)
    unit_ms_per_cost = actual_time_sum / estimated_cost_sum if estimated_cost_sum > 0 and actual_time_sum > 0 else 1.0
    threshold = float(spec.bias.threshold)
    analyses: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    for record in query_records:
        query: QuerySpec = record["query"]
        paths: dict[str, AccessPath] = record["paths"]
        estimates: dict[str, dict[str, Any]] = record["estimates"]
        executions: dict[str, dict[str, Any]] = record["executions"]
        selected_path_id = record["selected_path_id"]
        baseline_path_id = f"scan:{query.tables[0]}"
        selected_path = paths[selected_path_id]
        selected_estimate = estimates[selected_path_id]
        selected_execution = executions[selected_path_id]
        baseline_estimate = estimates[baseline_path_id]
        baseline_execution = executions[baseline_path_id]
        estimated_cost = int(selected_estimate["cost"])
        estimated_ms = estimated_cost * unit_ms_per_cost
        actual_ms = float(selected_execution["real_elapsed_ms"])
        cost_ratio = _ratio(estimated_ms, actual_ms)
        estimated_rows = int(selected_estimate["estimated_output_rows"])
        actual_rows = int(selected_execution["row_count"])
        row_ratio = _ratio(float(estimated_rows), float(actual_rows)) if actual_rows != 0 else (1.0 if estimated_rows == 0 else float("inf"))
        baseline_rows = int(baseline_execution["row_count"])
        baseline_digest = baseline_execution["digest"]
        digest_match = selected_execution["digest"] == baseline_digest
        all_path_digest_match = all(
            execution["digest"] == baseline_digest for execution in executions.values()
        )
        observed_covering = bool(selected_execution["plan"]["observed_expected_covering"])
        coverage_mismatch = bool(selected_path.covering) != observed_covering
        observed_order = selected_execution["plan"]["observed_table_order"]
        expected_order = [table for table in selected_path.join_order if table in observed_order]
        observed_order_filtered = [table for table in observed_order if table in query.tables]
        join_order_mismatch = observed_order_filtered != expected_order
        driving_filter_detail = selected_estimate["filter_details"][selected_path.driving_table]
        estimated_filter_rows = int(driving_filter_detail["estimated_rows"])
        actual_filter_rows = int(
            record["actual_table_filter_rows"][selected_path.driving_table]["filtered_rows"]
        )
        selectivity_ratio = (
            _ratio(float(estimated_filter_rows), float(actual_filter_rows))
            if actual_filter_rows != 0
            else (1.0 if estimated_filter_rows == 0 else float("inf"))
        )
        biased = (
            not digest_match
            or not all_path_digest_match
            or _bias_outside_threshold(cost_ratio, threshold)
            or _bias_outside_threshold(row_ratio, threshold)
            or coverage_mismatch
        )
        attribution = None
        if biased:
            category, basis = _classify_bias(
                coverage_mismatch=coverage_mismatch,
                selectivity_ratio=selectivity_ratio,
                row_ratio=row_ratio,
                join_order_mismatch=join_order_mismatch,
                threshold=threshold,
            )
            attribution = {
                "category": category,
                "basis": basis,
                "estimated_filtered_rows": estimated_filter_rows,
                "actual_filtered_rows": actual_filter_rows,
                "estimated_output_rows": estimated_rows,
                "actual_output_rows": actual_rows,
                "coverage_mismatch": coverage_mismatch,
                "expected_covering": bool(selected_path.covering),
                "observed_covering": observed_covering,
                "expected_join_order": list(selected_path.join_order),
                "observed_join_order": observed_order_filtered,
                "join_order_mismatch": join_order_mismatch,
                "threshold": threshold,
            }
            diagnostics.append(
                {
                    "code": "E_BIAS",
                    "query_id": query.id,
                    "category": category,
                    "basis": basis,
                    "cost_ratio": _finite(cost_ratio),
                    "row_ratio": _finite(row_ratio),
                }
            )
        analysis = {
            "bias_model_version": BIAS_MODEL_VERSION,
            "threshold": threshold,
            "calibration": {
                "method": "global full-scan workload calibration",
                "unit_ms_per_cost": round(unit_ms_per_cost, 12),
                "estimated_baseline_cost_units": estimated_cost_sum,
                "measured_baseline_elapsed_ms": round(actual_time_sum, 6),
                "per_query": baseline_calibration,
            },
            "estimated_cost_units": estimated_cost,
            "calibrated_estimated_ms": round(estimated_ms, 6),
            "actual_selected_ms": round(actual_ms, 6),
            "cost_ratio_estimated_over_actual": _finite(cost_ratio),
            "estimated_rows": estimated_rows,
            "actual_rows": actual_rows,
            "row_ratio_estimated_over_actual": _finite(row_ratio),
            "estimated_baseline_cost_units": int(baseline_estimate["cost"]),
            "actual_baseline_ms": round(float(baseline_execution["real_elapsed_ms"]), 6),
            "selected_over_baseline_estimated_cost_ratio": _finite(
                _ratio(float(estimated_cost), float(baseline_estimate["cost"]))
            ),
            "selected_over_baseline_actual_time_ratio": _finite(
                _ratio(actual_ms, float(baseline_execution["real_elapsed_ms"]))
            ),
            "baseline_rows": baseline_rows,
            "result_digest_match": digest_match,
            "all_paths_match_baseline_digest": all_path_digest_match,
            "baseline_path_id": baseline_path_id,
            "selected_path_id": selected_path_id,
            "bias_triggered": biased,
            "attribution": attribution,
        }
        analysis["cost_ratio"] = analysis["cost_ratio_estimated_over_actual"]
        analysis["row_ratio"] = analysis["row_ratio_estimated_over_actual"]
        analysis["bias_digest"] = content_digest(analysis)
        analyses[query.id] = analysis
    return {
        "calibration": {
            "method": "global full-scan workload calibration",
            "unit_ms_per_cost": round(unit_ms_per_cost, 12),
            "estimated_baseline_cost_units": estimated_cost_sum,
            "measured_baseline_elapsed_ms": round(actual_time_sum, 6),
            "per_query": baseline_calibration,
        },
        "threshold": threshold,
        "queries": analyses,
        "diagnostics": diagnostics,
    }
