from __future__ import annotations

from typing import Any

from .config import WorkbenchConfig
from .constants import METRICS
from .reducer import all_metric_reductions
from .storage import SourceDataset


MIGRATION_RULES = {
    "value_tolerance_pct": "absolute normalized metric delta / max(value_a, value_b)",
    "insufficient": "source A or B has fewer than min_evidence unique observations for a metric",
    "migratable": "both conclusions have the same slow status and every metric delta <= tolerance",
    "not_migratable": "slow status differs, or at least one metric delta > tolerance",
}


def metric_is_slow(metric: str, value: int, config: WorkbenchConfig) -> bool:
    if metric == "elapsed_ms":
        return value >= config.dataset.slow_elapsed_ms
    if metric == "scanned_rows":
        return value >= config.dataset.slow_scanned_rows
    raise KeyError(metric)


def source_conclusions(
    config: WorkbenchConfig,
    dataset: SourceDataset,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for query_id in dataset.query_ids():
        rows = list(dataset.rows_for(query_id))
        reductions = all_metric_reductions(
            rows,
            config.migration.replay_count,
            config.migration.reducer,
            config.migration.trim_percent,
        )
        metrics: dict[str, Any] = {}
        metric_slow: dict[str, bool] = {}
        for metric in METRICS:
            reduction = reductions[metric]
            metrics[metric] = {
                "value": reduction.value,
                "raw_value": reduction.raw_value,
                "evidence_count": len(rows),
                "observations_used": reduction.observations_used,
                "row_numbers": list(reduction.row_numbers),
                "record_versions": sorted({row.record_version for row in rows}),
                "padded_fields": sorted(
                    {field for row in rows for field in row.padded_fields}
                ),
                "reducer": reduction.reducer,
                "trim_percent": reduction.trim_percent,
            }
            metric_slow[metric] = metric_is_slow(metric, reduction.value, config)
        if config.dataset.slow_rule == "both":
            slow = all(metric_slow.values())
        else:
            slow = any(metric_slow.values())
        results[query_id] = {
            "source_id": dataset.config.id,
            "slow": slow,
            "metrics": metrics,
            "metric_slow": metric_slow,
        }
    return results


def _empty_metric_conclusion() -> dict[str, Any]:
    return {
        "value": None,
        "raw_value": None,
        "evidence_count": 0,
        "observations_used": 0,
        "row_numbers": [],
        "record_versions": [],
        "padded_fields": [],
        "reducer": None,
        "trim_percent": None,
    }


def _comparison(
    config: WorkbenchConfig,
    metric: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    left_value = left.get("value")
    right_value = right.get("value")
    if left_value is None or right_value is None:
        delta = None
        delta_bp = None
    else:
        delta = abs(left_value - right_value)
        denominator = max(left_value, right_value, 1)
        delta_bp = (delta * 10000 + denominator // 2) // denominator
    left_slow = None if left_value is None else {
        "elapsed_ms": config.dataset.slow_elapsed_ms,
        "scanned_rows": config.dataset.slow_scanned_rows,
    }[metric] <= left_value
    right_slow = None if right_value is None else {
        "elapsed_ms": config.dataset.slow_elapsed_ms,
        "scanned_rows": config.dataset.slow_scanned_rows,
    }[metric] <= right_value
    tolerance_bp = config.migration.value_tolerance_pct * 100
    reasons: list[str] = []
    if delta_bp is not None and delta_bp > tolerance_bp:
        reasons.append("value_tolerance_exceeded")
    if left_slow is not None and right_slow is not None and left_slow != right_slow:
        reasons.append("metric_slow_status_conflict")
    return {
        "metric": metric,
        "source_a_value": left_value,
        "source_b_value": right_value,
        "delta": delta,
        "delta_pct_bp": delta_bp,
        "tolerance_pct": config.migration.value_tolerance_pct,
        "source_a_slow": left_slow,
        "source_b_slow": right_slow,
        "conflict": bool(reasons),
        "reasons": reasons,
    }


def _migration_result(
    config: WorkbenchConfig,
    query_id: str,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {
        metric: _comparison(config, metric, left["metrics"][metric], right["metrics"][metric])
        for metric in METRICS
    }
    insufficient_metrics = [
        metric
        for metric in METRICS
        if left["metrics"][metric]["evidence_count"] < config.migration.min_evidence
        or right["metrics"][metric]["evidence_count"] < config.migration.min_evidence
    ]
    conflict_metrics = [
        metric for metric in METRICS if comparisons[metric]["conflict"]
    ]
    status_conflict = left["slow"] != right["slow"]
    reasons: list[str] = []
    if insufficient_metrics:
        status = "insufficient"
        reasons.extend(f"insufficient_evidence:{metric}" for metric in insufficient_metrics)
    elif status_conflict or conflict_metrics:
        status = "not_migratable"
        if status_conflict:
            reasons.append("slow_status_conflict")
        reasons.extend(f"metric_conflict:{metric}" for metric in conflict_metrics)
    else:
        status = "migratable"
        reasons.append("same_status_and_within_tolerance")
    return {
        "query_id": query_id,
        "status": status,
        "consistent": (not status_conflict) and not conflict_metrics,
        "migrated_claim": {"slow": left["slow"]},
        "rejudged_on_b": {"slow": right["slow"]},
        "source_a": left,
        "source_b": right,
        "conflicts": [
            comparisons[metric] for metric in METRICS if comparisons[metric]["conflict"]
        ],
        "metric_comparisons": comparisons,
        "insufficient_metrics": insufficient_metrics,
        "reasons": reasons,
        "thresholds": {
            "replay_count": config.migration.replay_count,
            "reducer": config.migration.reducer,
            "trim_percent": config.migration.trim_percent,
            "min_evidence": config.migration.min_evidence,
            "value_tolerance_pct": config.migration.value_tolerance_pct,
        },
    }


def migrate_conclusions(
    config: WorkbenchConfig,
    datasets: list[SourceDataset],
) -> dict[str, Any]:
    by_source = {dataset.config.id: dataset for dataset in datasets}
    conclusions = {
        source_id: source_conclusions(config, dataset)
        for source_id, dataset in sorted(by_source.items())
    }
    query_ids = sorted(
        {query_id for source in conclusions.values() for query_id in source}
    )
    pair_documents: list[dict[str, Any]] = []
    for left_id, right_id in config.migration.pairs:
        results: list[dict[str, Any]] = []
        for query_id in query_ids:
            left = conclusions[left_id].get(query_id) or {
                "source_id": left_id,
                "slow": False,
                "metrics": {
                    metric: _empty_metric_conclusion() for metric in METRICS
                },
                "metric_slow": {metric: False for metric in METRICS},
            }
            right = conclusions[right_id].get(query_id) or {
                "source_id": right_id,
                "slow": False,
                "metrics": {
                    metric: _empty_metric_conclusion() for metric in METRICS
                },
                "metric_slow": {metric: False for metric in METRICS},
            }
            results.append(_migration_result(config, query_id, left, right))
        counts = {
            status: sum(1 for result in results if result["status"] == status)
            for status in ("migratable", "not_migratable", "insufficient")
        }
        pair_documents.append(
            {
                "pair_id": f"{left_id}->{right_id}",
                "from_source": left_id,
                "to_source": right_id,
                "results": results,
                "counts": counts,
            }
        )
    pair_documents.sort(key=lambda item: item["pair_id"])
    totals = {
        status: sum(
            pair["counts"][status] for pair in pair_documents
        )
        for status in ("migratable", "not_migratable", "insufficient")
    }
    return {
        "rules": MIGRATION_RULES,
        "replay": {
            "count": config.migration.replay_count,
            "reducer": config.migration.reducer,
            "trim_percent": config.migration.trim_percent,
            "selection": "stable_even_spacing_by_row_no",
            "median_rule": "odd: middle; even: round-half-up integer mean",
            "trimmed_mean_rule": "trim floor(n*trim_percent/100) from each side",
            "clock_for_run_metrics": "perf_counter_ns",
        },
        "thresholds": {
            "min_evidence": config.migration.min_evidence,
            "value_tolerance_pct": config.migration.value_tolerance_pct,
            "slow_rule": config.dataset.slow_rule,
            "slow_elapsed_ms": config.dataset.slow_elapsed_ms,
            "slow_scanned_rows": config.dataset.slow_scanned_rows,
        },
        "pairs": pair_documents,
        "counts": totals,
    }
