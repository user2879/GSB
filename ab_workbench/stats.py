"""Group metric accumulation with deterministic ordering.

Accumulation order is fixed: request ID lexicographic, then item ID lexicographic.
Concurrent workers compute partial accumulators over request-level tasks; partial
results are merged back in the same canonical order, so thread scheduling cannot
change any numerator/denominator pair or the resulting fixed-point ratio.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import ExperimentConfig
from .errors import E_LOG_INVALID, FailureRecord
from .fingerprint import stable_sort_key
from .fixedpoint import FixedPoint
from .loader import Exposure

CLICK_RATE = "ctr"
CONVERSION_RATE = "cvr"
SUPPORTED_METRICS = {CLICK_RATE, CONVERSION_RATE}


@dataclass
class MetricAccumulator:
    numerator: int = 0
    denominator: int = 0

    def add(self, other: "MetricAccumulator") -> None:
        self.numerator += other.numerator
        self.denominator += other.denominator


@dataclass
class GroupStat:
    group: str
    exposures: int
    requests: int
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    drop_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self):
        return {
            "group": self.group,
            "exposures": self.exposures,
            "requests": self.requests,
            "metrics": self.metrics,
            "drop_reasons": dict(sorted(self.drop_reasons.items())),
        }


def _empty_metric_rows(
    groups: List[str], metrics: List[str]
) -> Dict[str, Dict[str, MetricAccumulator]]:
    return {g: {m: MetricAccumulator() for m in metrics} for g in groups}


def _accumulate_request(
    rows: List[Exposure],
    metrics: List[str],
    valid_groups,
    assignments: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, MetricAccumulator]], int, List[FailureRecord]]:
    partial = {g: {m: MetricAccumulator() for m in metrics} for g in valid_groups}
    failures: List[FailureRecord] = []
    request_count = 0
    grouped = sorted(rows, key=lambda e: (e.request_id, e.item_id))
    current_request = None
    for exposure in grouped:
        if exposure.request_id != current_request:
            current_request = exposure.request_id
            request_count += 1
        assignment = assignments.get(exposure.user_id)
        if assignment is None:
            failures.append(
                FailureRecord(
                    stage="stats",
                    code=E_LOG_INVALID,
                    reason="user_missing_bucket_assignment",
                    request_id=exposure.request_id,
                    item_id=exposure.item_id,
                    group=exposure.group,
                )
            )
            continue
        bucket = partial[exposure.group]
        for metric in metrics:
            bucket[metric].denominator += 1
            flag = exposure.clicked if metric == CLICK_RATE else exposure.converted
            bucket[metric].numerator += flag
    return partial, request_count, failures


def _merge(
    target: Dict[str, Dict[str, MetricAccumulator]],
    partial: Dict[str, Dict[str, MetricAccumulator]],
    groups: List[str],
    metrics: List[str],
) -> None:
    for group in groups:
        for metric in metrics:
            target[group][metric].add(partial[group][metric])


def compute_group_stats(
    exposures: List[Exposure],
    config: ExperimentConfig,
    assignments: Dict[str, Any],
    fp: FixedPoint,
    concurrent: bool = True,
) -> Tuple[Dict[str, GroupStat], List[FailureRecord]]:
    groups = config.group_order
    metrics = config.all_metrics
    unknown = [m for m in metrics if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(f"unsupported metrics: {unknown}")

    by_request: Dict[str, List[Exposure]] = {}
    for exposure in exposures:
        by_request.setdefault(exposure.request_id, []).append(exposure)
    request_ids = sorted(by_request)
    tasks = [
        (stable_sort_key("stats-request", rid), rid, by_request[rid])
        for rid in request_ids
    ]
    tasks.sort(key=lambda t: t[0])

    totals = _empty_metric_rows(groups, metrics)
    all_failures: List[FailureRecord] = []
    request_counts: Dict[str, int] = {g: 0 for g in groups}

    def _run(task_rows: List[Tuple[str, str, List[Exposure]]]):
        local_failures: List[FailureRecord] = []
        local_totals = _empty_metric_rows(groups, metrics)
        local_requests: Dict[str, int] = {g: 0 for g in groups}
        for _key, rid, rows in task_rows:
            partial, _count, failures = _accumulate_request(
                rows, metrics, groups, assignments
            )
            local_failures.extend(failures)
            _merge(local_totals, partial, groups, metrics)
            owner = assignments.get(rows[0].user_id)
            if owner is not None and owner.group in groups:
                local_requests[owner.group] += 1
        return local_totals, local_requests, local_failures

    if concurrent and config.concurrency > 1 and tasks:
        workers = min(config.concurrency, len(tasks))
        chunks = [tasks[i::workers] for i in range(workers)]
        chunks = [c for c in chunks if c]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            partials = list(pool.map(lambda chunk: _run(chunk), chunks))
        for local_totals, local_requests, local_failures in partials:
            _merge(totals, local_totals, groups, metrics)
            for group in groups:
                request_counts[group] += local_requests[group]
            all_failures.extend(local_failures)
    else:
        local_totals, local_requests, local_failures = _run(tasks)
        _merge(totals, local_totals, groups, metrics)
        for group in groups:
            request_counts[group] = local_requests[group]
        all_failures.extend(local_failures)

    drop_reasons: Dict[str, Dict[str, int]] = {g: {} for g in groups}
    for failure in all_failures:
        drop_reasons.setdefault(failure.group, {})
        drop_reasons[failure.group][failure.reason] = (
            drop_reasons[failure.group].get(failure.reason, 0) + 1
        )

    result: Dict[str, GroupStat] = {}
    for group in groups:
        metric_rows: Dict[str, Dict[str, Any]] = {}
        for metric in metrics:
            acc = totals[group][metric]
            scaled = fp.quantize(fp.ratio(acc.numerator, acc.denominator)) if acc.denominator else 0
            metric_rows[metric] = {
                "metric": metric,
                "numerator": acc.numerator,
                "denominator": acc.denominator,
                "value_scaled": scaled,
                "value_text": fp.to_text(scaled),
                "precision": fp.precision_label,
            }
        result[group] = GroupStat(
            group=group,
            exposures=sum(totals[group][m].denominator for m in metrics) // max(len(metrics), 1),
            requests=request_counts[group],
            metrics=metric_rows,
            drop_reasons=drop_reasons.get(group, {}),
        )
    all_failures.sort(key=lambda f: f.sort_key())
    return result, all_failures
