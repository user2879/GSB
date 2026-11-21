"""Integer-millisecond window aggregation with explicit quantization error."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .model import FlowDiagnosis, FlowRateSummary, FrameBundle, WindowRate


RATE_SCALE = 1_000_000
QUANTIZATION_ERROR_UPPER_PPM = 1
COMBINED_ERROR_BOUND_PPM = 2
ROUNDING_POLICY = "half_up"


def _round_half_up(numerator: int, denominator: int) -> int:
    if denominator == 0:
        return 0
    quotient, remainder = divmod(numerator, denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return quotient


def rate_ppm(loss: int, expected: int) -> int:
    return _round_half_up(loss * RATE_SCALE, expected)


def exact_fraction(loss: int, expected: int) -> str:
    return f"{loss}/{expected}"


def aggregate_rates(
    frame_bundle: FrameBundle,
    diagnoses: Iterable[FlowDiagnosis],
    window_ms: int,
) -> dict[str, Any]:
    if isinstance(window_ms, bool) or not isinstance(window_ms, int) or window_ms <= 0:
        raise ValueError("window_ms must be a positive integer")

    frames_by_flow = defaultdict(list)
    for frame in frame_bundle.frames:
        frames_by_flow[frame.flow_id].append(frame)

    flow_summaries: list[FlowRateSummary] = []
    for diagnosis in sorted(diagnoses, key=lambda item: item.flow_id):
        flow_frames = frames_by_flow.get(diagnosis.flow_id, [])
        first_send_by_seq: dict[int, int] = {}
        for frame in flow_frames:
            current = first_send_by_seq.get(frame.seq)
            if current is None or frame.send_ms < current:
                first_send_by_seq[frame.seq] = frame.send_ms

        window_observed: dict[int, int] = defaultdict(int)
        window_loss: dict[int, int] = defaultdict(int)
        window_assignments: dict[int, str] = {}
        for send_ms in first_send_by_seq.values():
            start = (send_ms // window_ms) * window_ms
            window_observed[start] += 1

        unwindowed = 0
        for gap in diagnosis.gaps:
            if gap.classification != "true_loss":
                continue
            if gap.time_window_ms is None:
                unwindowed += gap.count
                continue
            start = (gap.time_window_ms["start_ms"] // window_ms) * window_ms
            window_loss[start] += gap.count
            window_assignments[start] = "whole_gap_to_start_window"

        windows: list[WindowRate] = []
        for start in sorted(set(window_observed) | set(window_loss)):
            observed = window_observed.get(start, 0)
            loss = window_loss.get(start, 0)
            expected = observed + loss
            windows.append(
                WindowRate(
                    flow_id=diagnosis.flow_id,
                    window_start_ms=start,
                    window_end_ms=start + window_ms,
                    observed_unique=observed,
                    true_loss=loss,
                    expected=expected,
                    rate_ppm=rate_ppm(loss, expected),
                    exact_fraction=exact_fraction(loss, expected),
                    window_assignment=window_assignments.get(start, "observed_frames"),
                )
            )
        total_expected = diagnosis.observed_unique_count + diagnosis.true_loss_count
        flow_summaries.append(
            FlowRateSummary(
                flow_id=diagnosis.flow_id,
                true_loss=diagnosis.true_loss_count,
                expected=total_expected,
                rate_ppm=rate_ppm(diagnosis.true_loss_count, total_expected),
                exact_fraction=exact_fraction(diagnosis.true_loss_count, total_expected),
                unwindowed_true_loss=unwindowed,
                windows=windows,
            )
        )

    rate_sorted = sorted(
        flow_summaries,
        key=lambda item: (-item.rate_ppm, item.flow_id),
    )
    groups: list[list[FlowRateSummary]] = []
    for summary in rate_sorted:
        if (
            groups
            and abs(groups[-1][0].rate_ppm - summary.rate_ppm) <= COMBINED_ERROR_BOUND_PPM
        ):
            groups[-1].append(summary)
        else:
            groups.append([summary])

    ranking: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    rank = 0
    for group in groups:
        group_sorted = sorted(group, key=lambda item: item.flow_id)
        affected = len(group) > 1
        for item in group_sorted:
            rank += 1
            ranking.append(
                {
                    "rank": rank,
                    "flow_id": item.flow_id,
                    "rate_ppm": item.rate_ppm,
                    "error_affected": affected,
                }
            )
        for index in range(1, len(group_sorted)):
            left = group_sorted[index - 1]
            right = group_sorted[index]
            delta = abs(left.rate_ppm - right.rate_ppm)
            comparisons.append(
                {
                    "flow_a": left.flow_id,
                    "flow_b": right.flow_id,
                    "delta_ppm": delta,
                    "combined_error_bound_ppm": COMBINED_ERROR_BOUND_PPM,
                    "decision_affected": delta <= COMBINED_ERROR_BOUND_PPM,
                    "tie_breaker": "flow_id_asc",
                }
            )

    return {
        "window_ms": window_ms,
        "rate_unit": "ppm",
        "rate_scale": RATE_SCALE,
        "rounding": ROUNDING_POLICY,
        "quantization_error_upper_ppm": QUANTIZATION_ERROR_UPPER_PPM,
        "combined_error_bound_ppm": COMBINED_ERROR_BOUND_PPM,
        "critical_band_policy": "sort_by_flow_id_asc_within_combined_error_band",
        "flows": [summary.to_dict() for summary in flow_summaries],
        "ranking": ranking,
        "rate_comparison": comparisons,
    }


def rates_digest(rate_report: dict[str, Any]) -> str:
    from .util import content_digest

    return content_digest(rate_report)
