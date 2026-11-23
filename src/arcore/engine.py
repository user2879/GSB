"""End-to-end round orchestration tying every component together."""

from dataclasses import dataclass

from .clock import MonotonicClock
from .correlation import (
    approximate_correlate,
    compare_paths,
    exact_correlate,
    select_path,
)
from .fixedpoint import attribute_misjudgments, compute_stats
from .lease import LeaseManager
from .rootcause import derive_root_causes
from .scheduler import CorrelationScheduler, build_batches
from .stream import StreamResult, generate_stream
from .suppression import suppress_closed
from .spec import Declaration


@dataclass
class RoundResult:
    round_id: int
    payload: dict
    stats: dict


def _dedup_pairs(pair_lists: list[list[dict]]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    for pairs in pair_lists:
        for record in pairs:
            key = tuple(record["pair"])
            merged.setdefault(key, record)
    return [merged[key] for key in sorted(merged)]


def run_round(declaration: Declaration, round_id: int,
              stream: StreamResult | None = None,
              lease_manager: LeaseManager | None = None,
              clock=None, work_hook=None) -> RoundResult:
    clock = clock or MonotonicClock()
    stream = stream or generate_stream(declaration)
    alerts = list(stream.alerts)
    params = declaration.parameters
    lease_manager = lease_manager or LeaseManager(
        params.global_concurrency, params.service_concurrency,
        params.lease_ttl_ms, clock=clock)

    decision = select_path(alerts, params)
    exact_pairs = exact_correlate(
        alerts, declaration.topology, params.correlation_window_ms)
    approximate_pairs = approximate_correlate(
        alerts, declaration.topology, params.correlation_window_ms)
    error_bounds = compare_paths(
        alerts, declaration.topology, params.correlation_window_ms,
        exact_pairs, approximate_pairs)

    batches = build_batches(alerts, params.correlation_window_ms)
    scheduler = CorrelationScheduler(
        lease_manager, declaration.topology, params.correlation_window_ms,
        params.global_concurrency, params.service_concurrency, clock=clock,
        work_hook=work_hook)
    scheduling = scheduler.execute(
        batches, alerts, use_approximate=(decision.selected_path == "approximate"))

    suppression = suppress_closed(alerts, params.silence_window_ms)
    root_causes = derive_root_causes(
        alerts, declaration.topology, params.layer_weights,
        suppression.suppressed_alert_ids)

    correlated_ids: set[str] = set()
    for record in exact_pairs:
        correlated_ids.update(record["pair"])
    waits = [report.wait_ms for report in scheduling.reports]
    stats = compute_stats(
        total_alerts=len(alerts),
        suppressed_count=suppression.suppressed_count,
        correlated_alert_count=len(correlated_ids),
        queue_wait_ms=waits,
        scale=params.fixed_point_scale,
        saturation_max=params.saturation_max)
    misattributions = attribute_misjudgments(
        error_bounds.measured_false_positives,
        error_bounds.measured_misses,
        error_bounds.structural_false_positive_upper_bound,
        error_bounds.structural_miss_upper_bound)

    payload = {
        "round_id": round_id,
        "global_digest": stream.global_digest,
        "source_summaries": [s.to_dict() for s in stream.source_summaries],
        "path_decision": decision.to_dict(),
        "error_bounds": error_bounds.to_dict(),
        "correlation": {
            "exact_pair_count": len(exact_pairs),
            "approximate_pair_count": len(approximate_pairs),
            "exact_pairs": exact_pairs,
            "approximate_pairs": approximate_pairs,
        },
        "suppression": suppression.to_dict(),
        "root_causes": root_causes.to_dict(),
        # batch_specs are deterministic and replayed; batch_observations and
        # lease_occupancy are genuine runtime artifacts (thread interleaving,
        # monotonic clock) recorded for audit but excluded from replay diffs.
        "lease_occupancy": lease_manager.snapshot_deterministic(),
        "batch_specs": [
            {
                "batch_id": report.batch.batch_id,
                "service_id": report.batch.service_id,
                "index_lo": report.batch.index_lo,
                "index_hi": report.batch.index_hi,
                "alert_ids": list(report.batch.alert_ids),
            }
            for report in scheduling.reports
        ],
        "batch_observations": [report.deterministic_dict()
                               for report in scheduling.reports],
        "misattributions": [item.to_dict() for item in misattributions],
        "progress_cursor": {
            "processed_alert_count": len(alerts),
            "processed_batch_count": len(batches),
            "last_alert_id": alerts[-1].alert_id if alerts else None,
        },
    }
    return RoundResult(round_id=round_id, payload=payload,
                       stats=stats.to_dict())
