from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def floor_fixed(numerator: int, denominator: int, scale: int) -> int:
    """Non-negative fixed-point division, truncated toward zero/floor."""

    if denominator <= 0:
        return 0
    return (int(numerator) * int(scale)) // int(denominator)


def nearest_rank_p95(values: Iterable[int]) -> int | None:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return None
    rank = (95 * len(ordered) + 99) // 100
    rank = min(max(rank, 1), len(ordered))
    return ordered[rank - 1]


def _window_ms(rows: list[dict[str, Any]], attempt_span: tuple[int, int] | None) -> int:
    completed = [int(row["completed_at_wall_ms"]) for row in rows]
    if not completed:
        return 1
    starts = []
    if attempt_span is not None:
        starts.append(int(attempt_span[0]))
    finishes = completed
    if starts:
        return max(1, max(finishes) - min(starts))
    return max(1, max(completed) - min(completed) + 1)


def _metric_row(
    service: str,
    rows: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    scale: int,
) -> dict[str, Any]:
    count = len(rows)
    successes = sum(1 for row in rows if row["final_outcome"] == "success")
    ratio = floor_fixed(successes, count, scale) if count else 0
    p95_ms = nearest_rank_p95(int(row["total_duration_ms"]) for row in rows)
    p95_fixed = 0 if p95_ms is None else p95_ms * scale
    if attempts:
        started = min(int(row["started_wall_ms"]) for row in attempts)
        span = (started, max(int(row["completed_at_wall_ms"]) for row in rows)) if rows else None
    else:
        span = None
    window = _window_ms(rows, span)
    throughput = (count * 1000 * scale) // window if count else 0
    final_distribution = Counter(str(row["final_outcome"]) for row in rows)
    error_distribution = Counter(
        str(row["error_code"]) for row in rows if row.get("error_code")
    )
    attempt_distribution = Counter(str(row["outcome"]) for row in attempts)
    result = {
        "service": service,
        "requests": count,
        "successes": successes,
        "success_ratio_fixed": ratio,
        "p95_latency_ms": p95_ms,
        "p95_latency_ms_fixed": p95_fixed,
        "throughput_rps_fixed": throughput,
        "window_ms": window,
        "final_distribution": dict(sorted(final_distribution.items())),
        "attempt_distribution": dict(sorted(attempt_distribution.items())),
        "error_distribution": dict(sorted(error_distribution.items())),
        "single_value_quantization_error": {
            "success_ratio_fixed": 1,
            "p95_latency_ms_fixed": scale,
            "throughput_rps_fixed": 1,
        },
    }
    return result


def _pairwise_comparisons(rows: list[dict[str, Any]], metric: str, bound: int) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: row["service"])
    for left_index in range(len(ordered)):
        for right_index in range(left_index + 1, len(ordered)):
            left = ordered[left_index]
            right = ordered[right_index]
            difference = abs(int(left[metric]) - int(right[metric]))
            affected = difference <= bound
            if metric == "p95_latency_ms_fixed":
                winner = left["service"] if int(left[metric]) < int(right[metric]) else right["service"]
                if affected:
                    winner = min(left["service"], right["service"])
            else:
                winner = left["service"] if int(left[metric]) > int(right[metric]) else right["service"]
                if affected:
                    winner = min(left["service"], right["service"])
            comparisons.append(
                {
                    "left": left["service"],
                    "right": right["service"],
                    "metric": metric,
                    "difference": difference,
                    "comparison_error_bound": bound,
                    "within_error_bound": affected,
                    "winner_by_stable_rule": winner,
                }
            )
    return comparisons


def _rank(
    rows: list[dict[str, Any]],
    metric: str,
    bound: int,
    *,
    higher_is_better: bool,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row[metric]) if higher_is_better else int(row[metric]),
            row["service"],
        ),
    )
    ranked: list[dict[str, Any]] = []
    index = 0
    tier = 0
    while index < len(ordered):
        tier += 1
        group = [ordered[index]]
        anchor = int(ordered[index][metric])
        next_index = index + 1
        while next_index < len(ordered):
            value = int(ordered[next_index][metric])
            if abs(value - anchor) <= bound:
                group.append(ordered[next_index])
                next_index += 1
            else:
                break
        group.sort(key=lambda row: row["service"])
        for row in group:
            ranked.append(
                {
                    "rank": tier,
                    "service": row["service"],
                    "value": int(row[metric]),
                    "tie_group": tier if len(group) > 1 else None,
                    "within_error_bound": len(group) > 1,
                    "tie_break": "service_id_asc" if len(group) > 1 else None,
                }
            )
        index = next_index
    return ranked


def build_stats(
    outcomes: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    scale: int,
) -> dict[str, Any]:
    if scale < 1:
        raise ValueError("scale must be positive")
    by_service_outcomes: dict[str, list[dict[str, Any]]] = {}
    by_service_attempts: dict[str, list[dict[str, Any]]] = {}
    for row in sorted(outcomes, key=lambda item: (item["service"], item["request_seq"], item["request_id"])):
        by_service_outcomes.setdefault(str(row["service"]), []).append(row)
    for row in sorted(attempts, key=lambda item: (item["service"], item["request_seq"], item["attempt_no"], item["attempt_id"])):
        by_service_attempts.setdefault(str(row["service"]), []).append(row)
    services = sorted(set(by_service_outcomes) | set(by_service_attempts))
    rows = [
        _metric_row(
            service,
            by_service_outcomes.get(service, []),
            by_service_attempts.get(service, []),
            scale,
        )
        for service in services
    ]
    all_outcomes = sorted(
        outcomes, key=lambda item: (item["request_seq"], item["request_id"])
    )
    all_attempts = sorted(
        attempts,
        key=lambda item: (item["request_seq"], item["attempt_no"], item["attempt_id"]),
    )
    overall = _metric_row("__overall__", all_outcomes, all_attempts, scale)
    ratio_bound = 2
    p95_bound = 2 * scale
    throughput_bound = 2
    rankings = {
        "success_ratio": _rank(rows, "success_ratio_fixed", ratio_bound, higher_is_better=True),
        "p95_latency": _rank(rows, "p95_latency_ms_fixed", p95_bound, higher_is_better=False),
        "throughput": _rank(rows, "throughput_rps_fixed", throughput_bound, higher_is_better=True),
    }
    comparisons = {
        "success_ratio": _pairwise_comparisons(rows, "success_ratio_fixed", ratio_bound),
        "p95_latency": _pairwise_comparisons(rows, "p95_latency_ms_fixed", p95_bound),
        "throughput": _pairwise_comparisons(rows, "throughput_rps_fixed", throughput_bound),
    }
    return {
        "scale": scale,
        "method": {
            "rounding": "non_negative_floor_truncation",
            "ratio_formula": "successes * scale // requests",
            "p95_formula": "nearest_rank_ceil(0.95*n) * scale",
            "throughput_formula": "requests * 1000 * scale // observed_window_ms",
            "single_value_quantization_error": {
                "ratio_scaled_units": 1,
                "p95_scaled_units": scale,
                "throughput_scaled_units": 1,
                "p95_source_measurement_error_ms_lt": 1,
            },
            "comparison_error_bound": {
                "ratio_scaled_units": ratio_bound,
                "p95_scaled_units": p95_bound,
                "throughput_scaled_units": throughput_bound,
            },
            "stable_tie_break": "service_id_ascending",
            "attribution_order_independent": True,
        },
        "services": rows,
        "overall": overall,
        "rankings": rankings,
        "pairwise_comparisons": comparisons,
    }
