from __future__ import annotations

from typing import Any

from .canonical import canonical_json
from .config import WorkbenchConfig
from .constants import METRICS, SOURCE_TYPE_BIAS
from .errors import E_EMPTY, SlowDiagError
from .reducer import Reduction, all_metric_reductions
from .storage import SourceDataset


ARBITRATION_RULE = (
    "source_type_bias:desc",
    "format_version:desc",
    "source_id:asc",
)


def _metric_slow(metric: str, value: int, config: WorkbenchConfig) -> bool:
    if metric == "elapsed_ms":
        return value >= config.dataset.slow_elapsed_ms
    if metric == "scanned_rows":
        return value >= config.dataset.slow_scanned_rows
    raise KeyError(metric)


def _candidate(
    dataset: SourceDataset,
    query_id: str,
    metric: str,
    reduction: Reduction,
) -> dict[str, Any]:
    rows = [row for row in dataset.rows_for(query_id)]
    return {
        "source_id": dataset.config.id,
        "source_type": dataset.config.type,
        "format_version": dataset.config.format_version,
        "record_versions": sorted({row.record_version for row in rows}),
        "value": reduction.value,
        "raw_value": reduction.raw_value,
        "elapsed_semantics": dataset.config.elapsed_semantics,
        "scan_semantics": dataset.config.scan_semantics,
        "sample_period_ms": dataset.config.sample_period_ms,
        "evidence_count": len(rows),
        "observations_used": reduction.observations_used,
        "row_numbers": list(reduction.row_numbers),
        "reducer": reduction.reducer,
        "trim_percent": reduction.trim_percent,
        "source_type_bias": SOURCE_TYPE_BIAS[dataset.config.type],
    }


def _rank_key(candidate: dict[str, Any]) -> tuple[int, int, str]:
    return (
        -candidate["source_type_bias"],
        -candidate["format_version"],
        candidate["source_id"],
    )


def _rejection_reason(winner: dict[str, Any], candidate: dict[str, Any]) -> str:
    if candidate["source_type_bias"] != winner["source_type_bias"]:
        return "lower_source_type_bias"
    if candidate["format_version"] != winner["format_version"]:
        return "lower_format_version"
    return "source_id_order"


def _merge_metric(
    config: WorkbenchConfig,
    query_id: str,
    metric: str,
    datasets: list[SourceDataset],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for dataset in datasets:
        rows = list(dataset.rows_for(query_id))
        if not rows:
            continue
        reductions = all_metric_reductions(
            rows,
            config.migration.replay_count,
            config.migration.reducer,
            config.migration.trim_percent,
        )
        candidates.append(_candidate(dataset, query_id, metric, reductions[metric]))
    candidates.sort(key=_rank_key)
    if not candidates:
        raise SlowDiagError(
            E_EMPTY,
            "no evidence candidates remain for a merged metric",
            query_id=query_id,
            metric=metric,
        )
    winner = candidates[0]
    comparisons: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        comparison = dict(candidate)
        comparison["rejection_reason"] = _rejection_reason(winner, candidate)
        comparisons.append(comparison)
    adopted = {
        key: winner[key]
        for key in (
            "source_id",
            "source_type",
            "format_version",
            "record_versions",
            "value",
            "raw_value",
            "elapsed_semantics",
            "scan_semantics",
            "sample_period_ms",
            "evidence_count",
            "observations_used",
            "row_numbers",
            "reducer",
            "trim_percent",
        )
    }
    return {
        "adopted": adopted,
        "comparisons": comparisons,
        "basis": {
            "rule": list(ARBITRATION_RULE),
            "candidate_count": len(candidates),
            "ranking": [
                {
                    "source_id": candidate["source_id"],
                    "source_type_bias": candidate["source_type_bias"],
                    "format_version": candidate["format_version"],
                }
                for candidate in candidates
            ],
            "winner_key": {
                "source_type_bias": winner["source_type_bias"],
                "format_version": winner["format_version"],
                "source_id": winner["source_id"],
            },
        },
    }


def merge_evidence(
    config: WorkbenchConfig, datasets: list[SourceDataset]
) -> list[dict[str, Any]]:
    ordered_datasets = sorted(datasets, key=lambda item: item.config.id)
    query_ids = sorted(
        {row.query_id for dataset in ordered_datasets for row in dataset.evidence}
    )
    if not query_ids:
        raise SlowDiagError(
            E_EMPTY,
            "the merged query set is empty",
            details={"source_count": len(ordered_datasets)},
        )
    results: list[dict[str, Any]] = []
    for query_id in query_ids:
        metrics: dict[str, Any] = {}
        metric_slow: dict[str, bool] = {}
        for metric in METRICS:
            merged = _merge_metric(config, query_id, metric, ordered_datasets)
            metrics[metric] = merged
            metric_slow[metric] = _metric_slow(metric, merged["adopted"]["value"], config)
        if config.dataset.slow_rule == "both":
            slow = all(metric_slow.values())
        else:
            slow = any(metric_slow.values())
        diagnosis = {
            "slow": slow,
            "rule": config.dataset.slow_rule,
            "thresholds": {
                "elapsed_ms": config.dataset.slow_elapsed_ms,
                "scanned_rows": config.dataset.slow_scanned_rows,
            },
            "metric_slow": metric_slow,
            "basis": [
                f"{metric}:{metrics[metric]['adopted']['value']}"
                f"{'>=' if metric_slow[metric] else '<'}"
                f"{config.dataset.slow_elapsed_ms if metric == 'elapsed_ms' else config.dataset.slow_scanned_rows}"
                for metric in METRICS
            ],
        }
        results.append(
            {
                "query_id": query_id,
                "metrics": metrics,
                "diagnosis": diagnosis,
            }
        )
    return results


def merged_query_bytes(query_result: dict[str, Any]) -> bytes:
    return canonical_json(query_result).encode("utf-8")
