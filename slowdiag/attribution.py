from __future__ import annotations

from typing import Any

from .canonical import round_div
from .config import WorkbenchConfig
from .constants import ELAPSED_FACTOR_BP, METRICS, SCAN_FACTOR_BP
from .storage import SourceDataset


CAUSES = ("semantics", "format_version", "sample_period")
CAUSE_LABELS = {
    "semantics": "上报口径差异",
    "format_version": "格式版本差异",
    "sample_period": "采样周期差异",
}


def _semantic_factor(config: WorkbenchConfig, source_id: str, metric: str) -> int:
    source = config.source_by_id(source_id)
    if metric == "elapsed_ms":
        return ELAPSED_FACTOR_BP[source.elapsed_semantics]
    if metric == "scanned_rows":
        return SCAN_FACTOR_BP[source.scan_semantics]
    raise KeyError(metric)


def _format_weight_bp(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_versions = left.get("record_versions") or []
    right_versions = right.get("record_versions") or []
    left_mean_bp = 0
    right_mean_bp = 0
    if left_versions:
        left_mean_bp = round_div(sum(left_versions) * 10000, len(left_versions))
    if right_versions:
        right_mean_bp = round_div(sum(right_versions) * 10000, len(right_versions))
    histogram_gap = abs(left_mean_bp - right_mean_bp) // 10
    padded_gap = abs(
        len(set(left.get("padded_fields") or []))
        - len(set(right.get("padded_fields") or []))
    )
    return min(10000, histogram_gap + padded_gap * 500)


def _weights_bp(
    config: WorkbenchConfig,
    metric: str,
    left_source: str,
    right_source: str,
    left_metric: dict[str, Any],
    right_metric: dict[str, Any],
) -> dict[str, int]:
    left_factor = _semantic_factor(config, left_source, metric)
    right_factor = _semantic_factor(config, right_source, metric)
    semantic = abs(left_factor - right_factor) * 10000 // max(left_factor, right_factor)
    format_version = _format_weight_bp(left_metric, right_metric)
    left_period = config.source_by_id(left_source).sample_period_ms
    right_period = config.source_by_id(right_source).sample_period_ms
    sample_period = (
        abs(left_period - right_period) * 10000 // max(left_period, right_period)
    )
    return {
        "semantics": semantic,
        "format_version": format_version,
        "sample_period": sample_period,
    }


def _shares_bp(weights: dict[str, int]) -> dict[str, int]:
    total = sum(weights.values())
    if total == 0:
        return {cause: (10000 if cause == "sample_period" else 0) for cause in CAUSES}
    shares = {
        cause: weights[cause] * 10000 // total for cause in CAUSES
    }
    remainder = 10000 - sum(shares.values())
    order = sorted(CAUSES, key=lambda cause: (-weights[cause], CAUSES.index(cause)))
    for index in range(remainder):
        shares[order[index % len(order)]] += 1
    return shares


def _cause_rows(
    delta: int,
    weights: dict[str, int],
) -> list[dict[str, Any]]:
    shares = _shares_bp(weights)
    total_milli = delta * 1000
    numerators = {cause: delta * shares[cause] for cause in CAUSES}
    contributions = {
        cause: numerators[cause] // 10 for cause in CAUSES
    }
    remaining = total_milli - sum(contributions.values())
    remainder_order = sorted(
        CAUSES,
        key=lambda cause: (
            -(numerators[cause] % 10),
            -weights[cause],
            CAUSES.index(cause),
        ),
    )
    for cause in remainder_order[:remaining]:
        contributions[cause] += 1
    rows: list[dict[str, Any]] = []
    for cause in CAUSES:
        rows.append(
            {
                "cause": cause,
                "label": CAUSE_LABELS[cause],
                "weight_bp": weights[cause],
                "share_bp": shares[cause],
                "contribution_milli_units": contributions[cause],
                "basis": {
                    "semantics": "absolute calibration factor gap between source A and B",
                    "format_version": "record-version mean gap plus padded-field gap",
                    "sample_period": "absolute sampling-period gap / max(period A, period B)",
                }[cause],
            }
        )
    return rows


def _attribute_metric(
    config: WorkbenchConfig,
    result: dict[str, Any],
    metric_conflict: dict[str, Any],
) -> dict[str, Any]:
    metric = metric_conflict["metric"]
    left_id = result["source_a"]["source_id"]
    right_id = result["source_b"]["source_id"]
    left_metric = result["source_a"]["metrics"][metric]
    right_metric = result["source_b"]["metrics"][metric]
    delta = int(metric_conflict["delta"] or 0)
    weights = _weights_bp(
        config, metric, left_id, right_id, left_metric, right_metric
    )
    return {
        "metric": metric,
        "conflict_score_milli_units": delta * 1000,
        "delta": delta,
        "delta_pct_bp": metric_conflict["delta_pct_bp"],
        "tolerance_pct": metric_conflict["tolerance_pct"],
        "conflict_reasons": list(metric_conflict["reasons"]),
        "evidence_a": {
            "source_id": left_id,
            "value": left_metric.get("value"),
            "evidence_count": left_metric.get("evidence_count"),
            "row_numbers": left_metric.get("row_numbers", []),
            "record_versions": left_metric.get("record_versions", []),
            "padded_fields": left_metric.get("padded_fields", []),
            "semantics": config.source_by_id(left_id).elapsed_semantics
            if metric == "elapsed_ms"
            else config.source_by_id(left_id).scan_semantics,
            "sample_period_ms": config.source_by_id(left_id).sample_period_ms,
        },
        "evidence_b": {
            "source_id": right_id,
            "value": right_metric.get("value"),
            "evidence_count": right_metric.get("evidence_count"),
            "row_numbers": right_metric.get("row_numbers", []),
            "record_versions": right_metric.get("record_versions", []),
            "padded_fields": right_metric.get("padded_fields", []),
            "semantics": config.source_by_id(right_id).elapsed_semantics
            if metric == "elapsed_ms"
            else config.source_by_id(right_id).scan_semantics,
            "sample_period_ms": config.source_by_id(right_id).sample_period_ms,
        },
        "causes": _cause_rows(delta, weights),
    }


def attribute_conflicts(
    config: WorkbenchConfig,
    datasets: list[SourceDataset],
    migration_document: dict[str, Any],
) -> dict[str, Any]:
    del datasets  # Attribution is derived solely from the frozen migration evidence.
    results: list[dict[str, Any]] = []
    for pair in migration_document["pairs"]:
        for migration_result in pair["results"]:
            if migration_result["status"] != "not_migratable":
                continue
            conflicts = list(migration_result.get("conflicts", []))
            if not conflicts:
                conflicts = [
                    migration_result["metric_comparisons"][metric]
                    for metric in METRICS
                    if migration_result["metric_comparisons"][metric].get("conflict")
                ]
            attributed = [
                _attribute_metric(config, migration_result, conflict)
                for conflict in conflicts
            ]
            attributed.sort(
                key=lambda item: (-item["conflict_score_milli_units"], item["metric"])
            )
            results.append(
                {
                    "pair_id": pair["pair_id"],
                    "query_id": migration_result["query_id"],
                    "status": "attributed",
                    "conflict_count": len(attributed),
                    "conflicts": attributed,
                    "sort_rule": [
                        "conflict_contribution:desc",
                        "metric:asc",
                    ],
                }
            )
    results.sort(key=lambda item: (item["pair_id"], item["query_id"]))
    return {
        "cause_order": list(CAUSES),
        "cause_labels": CAUSE_LABELS,
        "results": results,
        "counts": {
            "not_migratable_queries": len(results),
            "metric_conflicts": sum(item["conflict_count"] for item in results),
        },
    }
