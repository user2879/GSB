"""Integer fixed-point window estimation and error attribution.

No floating-point value participates in a persisted conclusion. The estimator
is a pure function of the canonical sample set, master record sequence, replay
rate and aggregation spec; concurrency metadata is not consulted. Concurrency
enters through the observed completion clusters recorded in the truth samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .constants import ESTIMATOR_VERSION, P95_METHOD, SEGMENT_ORDER, TRUNCATION_DIRECTION, TRUTH_SOURCE
from .models import AggregationSpec, MasterRecord, ReplayParams
from .util import (
    canonical_json,
    ceil_div,
    fixed_ceil_from_ns,
    fixed_from_ns,
    format_fixed,
    sha256_json,
)


@dataclass(frozen=True)
class SampleInput:
    commit_seq: int
    replay_completed_ns: int
    delay_ns: int
    fixed_delay: int | None = None


@dataclass(frozen=True)
class _PreparedWindow:
    window_index: int
    records: list[MasterRecord]
    full_samples: list[SampleInput]
    kept_samples: list[SampleInput]
    missing_seqs: list[int]
    duplicate_seqs: list[int]


def _prepare_windows(
    records: list[MasterRecord],
    samples: list[SampleInput],
    spec: AggregationSpec,
) -> tuple[list[_PreparedWindow], list[int]]:
    by_seq: dict[int, list[SampleInput]] = {}
    for sample in samples:
        by_seq.setdefault(sample.commit_seq, []).append(sample)
    record_seqs = {record.commit_seq for record in records}
    unexpected = sorted(seq for seq in by_seq if seq not in record_seqs)

    windows: list[_PreparedWindow] = []
    for window_index, start in enumerate(range(0, len(records), spec.window_size)):
        window_records = records[start : start + spec.window_size]
        expected = [record.commit_seq for record in window_records]
        missing = [seq for seq in expected if seq not in by_seq]
        duplicates = [seq for seq in expected if len(by_seq.get(seq, [])) > 1]
        full_samples = [
            by_seq[seq][0]
            for seq in expected
            if seq in by_seq and len(by_seq[seq]) == 1
        ]
        full_samples.sort(key=lambda sample: sample.commit_seq)
        kept = full_samples[: spec.max_samples_per_window]
        windows.append(
            _PreparedWindow(
                window_index=window_index,
                records=window_records,
                full_samples=full_samples,
                kept_samples=kept,
                missing_seqs=missing,
                duplicate_seqs=duplicates,
            )
        )
    return windows, unexpected


def _p95_nearest_rank(values_ns: list[int]) -> tuple[int, int, int]:
    """Return selected value, selected one-based rank, and Type-7 selection gap."""

    ordered = sorted(values_ns)
    count = len(ordered)
    if count == 0:
        raise ValueError("P95 requires at least one sample")
    rank = ceil_div(95 * count, 100)  # one-based nearest-rank ceil(0.95*n)
    selected = ordered[rank - 1]

    # Type-7 continuous quantile position h = 1 + (n - 1) * 0.95.
    h_numerator = 95 * count + 5
    lower_zero = (h_numerator // 100) - 1
    upper_zero = ceil_div(h_numerator, 100) - 1
    lower_zero = max(0, min(lower_zero, count - 1))
    upper_zero = max(lower_zero, min(upper_zero, count - 1))
    lower = ordered[lower_zero]
    upper = ordered[upper_zero]
    if selected < lower:
        gap = lower - selected
    elif selected > upper:
        gap = selected - upper
    else:
        gap = 0
    return selected, rank, gap


def _segments(
    sampling_fixed: int,
    fixed_point_fixed: int,
    rate_fixed: int,
    threshold_pct: int,
) -> dict[str, Any]:
    values = {
        "sampling_truncation": sampling_fixed,
        "fixed_point_truncation": fixed_point_fixed,
        "replay_rate_discretization": rate_fixed,
    }
    total = sum(values.values())
    segments: dict[str, Any] = {}
    for name in SEGMENT_ORDER:
        value = values[name]
        ratio_ppm = (value * 1_000_000) // total if total else 0
        segments[name] = {"upper_bound_fixed": value, "ratio_ppm": ratio_ppm}

    if total == 0:
        dominant = "NONE"
    else:
        ordered_segments = sorted(
            SEGMENT_ORDER, key=lambda name: (-values[name], SEGMENT_ORDER.index(name))
        )
        candidate = ordered_segments[0]
        if values[candidate] * 100 >= total * threshold_pct:
            dominant = candidate
        else:
            dominant = "MIXED"
    return {
        "segments": segments,
        "total_upper_bound_fixed": total,
        "dominant_segment": dominant,
        "dominant_rule": {
            "threshold_pct": threshold_pct,
            "comparison": "segment_value * 100 >= total * threshold_pct",
            "otherwise": "MIXED",
            "tie_break": "SEGMENT_ORDER",
        },
    }


def estimate_windows(
    records: list[MasterRecord],
    samples: Iterable[SampleInput],
    params: ReplayParams,
    spec: AggregationSpec,
) -> dict[str, Any]:
    sample_list = sorted(list(samples), key=lambda sample: sample.commit_seq)
    windows, unexpected = _prepare_windows(records, sample_list, spec)
    scale = spec.scale
    ceiling_ns = spec.latency_ceiling_ms * 1_000_000

    output_windows: list[dict[str, Any]] = []
    for window in windows:
        expected_seqs = [record.commit_seq for record in window.records]
        basis: dict[str, Any] = {
            "expected_commit_seqs": expected_seqs,
            "observed_commit_seqs": [sample.commit_seq for sample in window.kept_samples],
            "missing_commit_seqs": window.missing_seqs,
            "duplicate_commit_seqs": window.duplicate_seqs,
            "expected_count": len(window.records),
            "sample_count_full": len(window.full_samples),
            "sample_count_used": len(window.kept_samples),
            "truncated_count": len(window.full_samples) - len(window.kept_samples),
            "min_samples": spec.min_samples,
            "sample_cap": spec.max_samples_per_window,
            "truncation_policy": "commit_seq_ascending_prefix",
        }
        reasons: list[str] = []
        if unexpected:
            reasons.append("UNEXPECTED_COMMIT_SEQ")
        if window.missing_seqs:
            reasons.append("MISSING_LATENCY_SAMPLE")
        if window.duplicate_seqs:
            reasons.append("DUPLICATE_LATENCY_SAMPLE")
        if len(window.kept_samples) < spec.min_samples:
            reasons.append("SAMPLE_COUNT_BELOW_MIN")
        if any(sample.delay_ns > ceiling_ns for sample in window.full_samples):
            reasons.append("LATENCY_CEILING_EXCEEDED")

        determinable = not reasons
        window_report: dict[str, Any] = {
            "window_index": window.window_index,
            "first_commit_seq": expected_seqs[0],
            "last_commit_seq": expected_seqs[-1],
            "determinable": determinable,
            "decision": "DETERMINABLE" if determinable else "UNDECIDABLE",
            "decision_reasons": reasons,
            "basis": basis,
            "statistics": {
                "mean": None,
                "p95": None,
                "max": None,
            },
            "error_budget": None,
        }
        if not determinable:
            output_windows.append(window_report)
            continue

        values_ns = [sample.delay_ns for sample in window.kept_samples]
        count = len(values_ns)
        mean_ns_numerator = sum(values_ns)
        mean_fixed = (mean_ns_numerator * scale) // (count * 1_000_000)
        p95_ns, p95_rank, p95_selection_gap_ns = _p95_nearest_rank(values_ns)
        max_ns = max(values_ns)
        p95_fixed = fixed_from_ns(p95_ns, spec.scale_power)
        max_fixed = fixed_from_ns(max_ns, spec.scale_power)

        # Rate/concurrency contribution is derived from observed barrier clusters,
        # never from unverified concurrency metadata. This preserves byte identity
        # when the same sample set is re-aggregated with different run metadata.
        cluster_sizes: dict[int, int] = {}
        for sample in window.full_samples:
            cluster_sizes[sample.replay_completed_ns] = (
                cluster_sizes.get(sample.replay_completed_ns, 0) + 1
            )
        max_cluster = max(cluster_sizes.values(), default=0)
        rate_tick_ns = ceil_div(1_000_000_000, params.rate_rps)
        rate_ub_ns = max_cluster * rate_tick_ns
        rate_fixed = fixed_ceil_from_ns(rate_ub_ns, spec.scale_power)

        omitted = len(window.records) - len(window.kept_samples)
        sampling_mean_ns_numerator = omitted * ceiling_ns
        sampling_mean_fixed = ceil_div(
            sampling_mean_ns_numerator * scale,
            len(window.records) * 1_000_000,
        )
        sampling_max_fixed = fixed_ceil_from_ns(max(0, ceiling_ns - max_ns), spec.scale_power)
        sampling_p95_fixed = fixed_ceil_from_ns(ceiling_ns, spec.scale_power)

        p95_selection_fixed = fixed_ceil_from_ns(
            p95_selection_gap_ns, spec.scale_power
        )
        # One final fixed-point floor is required for each published value.
        fixed_mean = 1
        fixed_max = 1
        fixed_p95 = 1 + p95_selection_fixed

        mean_budget = _segments(
            sampling_mean_fixed, fixed_mean, rate_fixed, spec.dominant_threshold_pct
        )
        p95_budget = _segments(
            sampling_p95_fixed, fixed_p95, rate_fixed, spec.dominant_threshold_pct
        )
        max_budget = _segments(
            sampling_max_fixed, fixed_max, rate_fixed, spec.dominant_threshold_pct
        )

        window_report["statistics"] = {
            "mean": {
                "fixed": mean_fixed,
                "scale": scale,
                "ms": format_fixed(mean_fixed, spec.scale_power),
            },
            "p95": {
                "fixed": p95_fixed,
                "scale": scale,
                "ms": format_fixed(p95_fixed, spec.scale_power),
                "rank_one_based": p95_rank,
                "method": P95_METHOD,
            },
            "max": {
                "fixed": max_fixed,
                "scale": scale,
                "ms": format_fixed(max_fixed, spec.scale_power),
            },
        }
        window_report["error_budget"] = {
            "mean": mean_budget,
            "p95": p95_budget,
            "max": max_budget,
            "components": {
                "single_sample_quantization_upper_bound_fixed": 1,
                "window_aggregation_upper_bound_fixed": {
                    "mean": fixed_mean,
                    "p95": fixed_p95,
                    "max": fixed_max,
                },
                "p95_selection_upper_bound_fixed": p95_selection_fixed,
                "p95_selection_upper_bound_ns": p95_selection_gap_ns,
                "observed_max_batch_cluster": max_cluster,
                "rate_tick_ns": rate_tick_ns,
                "rate_upper_bound_ns": rate_ub_ns,
            },
        }
        output_windows.append(window_report)

    determinable_count = sum(1 for window in output_windows if window["determinable"])
    report: dict[str, Any] = {
        "estimator_version": ESTIMATOR_VERSION,
        "clock_basis": TRUTH_SOURCE,
        "scale": scale,
        "scale_power": spec.scale_power,
        "truncation_direction": TRUNCATION_DIRECTION,
        "p95_method": P95_METHOD,
        "rate_rps": params.rate_rps,
        "spec": spec.to_dict(),
        "unexpected_commit_seqs": unexpected,
        "window_count": len(output_windows),
        "determinable_window_count": determinable_count,
        "undecidable_window_count": len(output_windows) - determinable_count,
        "windows": output_windows,
        "precision_contract": {
            "single_sample_quantization_ub_fixed": 1,
            "single_sample_quantization_ub_ms": format_fixed(1, spec.scale_power),
            "window_aggregation": "one final floor operation per published statistic",
            "quantile_rank_error": 0,
            "quantile_method": P95_METHOD,
            "p95_selection_rule": (
                "nearest-rank selected value compared with the Type-7 interpolation "
                "interval; its distance to that interval is included in fixed-point "
                "truncation as a conservative value-selection bound"
            ),
        },
    }
    report["result_fingerprint"] = sha256_json(report)
    return report


def canonical_estimation_bytes(report: dict[str, Any]) -> bytes:
    return canonical_json(report).encode("utf-8")
