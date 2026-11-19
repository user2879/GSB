"""Candidate voting, sub-frame alignment, boundary refinement, and overlap policies."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .index import ReferenceIndex
from .models import (
    FingerprintMatch,
    Fragment,
    RetrievalConfig,
    ToleranceConfig,
    TrackFingerprint,
)


@dataclass(slots=True)
class _Candidate:
    matches: list[FingerprintMatch]
    reference_lo: int
    reference_hi: int
    query_lo: int
    query_hi: int


def _match_identity(match: FingerprintMatch) -> tuple[tuple[int, int, int, int, int, int, int], int, int, int, int]:
    return (
        match.key,
        match.reference_frame,
        match.query_frame,
        match.reference_target_frame,
        match.query_target_frame,
    )


def _collect_matches(
    index: ReferenceIndex,
    query: TrackFingerprint,
    score_scale: int,
) -> list[FingerprintMatch]:
    matches: list[FingerprintMatch] = []
    seen: set[tuple[tuple[int, int, int, int, int, int, int], str, int, int, int, int]] = set()
    for query_item in query.items:
        for reference_entry in index.lookup(query_item.key):
            reference = index.get(reference_entry.reference_id)
            identity = (
                query_item.key,
                reference_entry.reference_id,
                reference_entry.item.anchor_frame,
                query_item.anchor_frame,
                reference_entry.item.target_frame,
                query_item.target_frame,
            )
            if identity in seen:
                continue
            seen.add(identity)
            weight = min(query_item.quality, reference_entry.item.quality)
            fixed_units = max(1, int(math.floor(weight * score_scale + 0.5)))
            reference_seconds = float(reference.frames[reference_entry.item.anchor_frame].theoretical_start_seconds)
            query_seconds = float(query.frames[query_item.anchor_frame].theoretical_start_seconds)
            matches.append(
                FingerprintMatch(
                    key=query_item.key,
                    reference_id=reference_entry.reference_id,
                    query_id=query.track_id,
                    reference_frame=reference_entry.item.anchor_frame,
                    query_frame=query_item.anchor_frame,
                    reference_target_frame=reference_entry.item.target_frame,
                    query_target_frame=query_item.target_frame,
                    float_weight=float(weight),
                    fixed_units=fixed_units,
                    offset_frames=query_item.anchor_frame - reference_entry.item.anchor_frame,
                    reference_time_seconds=reference_seconds,
                    query_time_seconds=query_seconds,
                    offset_seconds=query_seconds - reference_seconds,
                )
            )
    matches.sort(
        key=lambda m: (
            m.query_frame,
            m.reference_frame,
            m.query_target_frame,
            m.reference_target_frame,
            m.key,
            m.reference_id,
        )
    )
    return matches


def _cluster_offsets(matches: list[FingerprintMatch], voting_window_frames: int) -> list[list[FingerprintMatch]]:
    by_offset: dict[int, list[FingerprintMatch]] = defaultdict(list)
    for match in matches:
        by_offset[match.offset_frames].append(match)
    if not by_offset:
        return []
    half_window = max(1, voting_window_frames // 2)
    offsets = sorted(by_offset)
    offset_weight = {
        offset: sum(match.fixed_units for match in by_offset[offset]) for offset in offsets
    }
    seeds = sorted(offsets, key=lambda offset: (-offset_weight[offset], offset))
    consumed: set[int] = set()
    groups: list[list[int]] = []
    for seed in seeds:
        if seed in consumed:
            continue
        window_offsets = [offset for offset in offsets if abs(offset - seed) <= half_window]
        if not window_offsets:
            continue
        consumed.update(window_offsets)
        groups.append(window_offsets)
    clustered: list[list[FingerprintMatch]] = []
    for offsets_group in groups:
        group_matches: list[FingerprintMatch] = []
        for offset in offsets_group:
            group_matches.extend(by_offset[offset])
        group_matches.sort(key=lambda m: (m.query_frame, m.reference_frame, m.key))
        clustered.append(group_matches)
    return clustered


def _axis_runs(frames: Iterable[int], max_index: int, gap: int, padding: int) -> list[tuple[int, int]]:
    unique = sorted(set(frames))
    if not unique:
        return []
    runs: list[list[int]] = [[unique[0], unique[0]]]
    for frame in unique[1:]:
        if frame - runs[-1][1] > gap:
            runs.append([frame, frame])
        else:
            runs[-1][1] = frame
    expanded = [
        (max(0, start - padding), min(max_index, end + padding)) for start, end in runs
    ]
    return expanded


def _split_query_components(matches: list[FingerprintMatch], gap: int) -> list[list[FingerprintMatch]]:
    """Split one offset vote into time-local candidates across insertions."""

    if not matches:
        return []
    ordered = sorted(matches, key=lambda match: (match.query_frame, match.reference_frame, match.key))
    components: list[list[FingerprintMatch]] = [[ordered[0]]]
    last_frame = ordered[0].query_frame
    for match in ordered[1:]:
        if match.query_frame - last_frame > gap:
            components.append([match])
        else:
            components[-1].append(match)
        last_frame = max(last_frame, match.query_frame)
    return components


def _dense_limits(frames: list[int], radius: int = 4, minimum_count: int = 2) -> tuple[int, int]:
    counts = Counter(frames)
    unique = sorted(counts)
    if not unique:
        return 0, 0
    low = unique[0]
    for frame in unique:
        if sum(counts.get(candidate, 0) for candidate in range(frame, frame + radius + 1)) >= minimum_count:
            low = frame
            break
    high = unique[-1]
    for frame in reversed(unique):
        if sum(counts.get(candidate, 0) for candidate in range(frame - radius, frame + 1)) >= minimum_count:
            high = frame
            break
    return low, high


def _trimmed_limits(frames: list[int], trim_fraction: float = 0.001) -> tuple[int, int]:
    ordered = sorted(frames)
    if not ordered:
        return 0, 0
    trim = int(len(ordered) * trim_fraction)
    return ordered[trim], ordered[max(trim, len(ordered) - 1 - trim)]


def _refine_candidate(
    matches: list[FingerprintMatch],
    reference: TrackFingerprint,
    query: TrackFingerprint,
    retrieval: RetrievalConfig,
) -> tuple[tuple[int, int], tuple[int, int], list[FingerprintMatch]]:
    query_frames: list[int] = []
    for match in matches:
        query_frames.extend((match.query_frame, match.query_target_frame))
    query_runs = _axis_runs(
        query_frames,
        max(0, query.frame_count - 1),
        retrieval.boundary_gap_frames,
        0,
    )
    if not query_runs:
        return (0, 0), (0, 0), []
    best: tuple[float, tuple[int, int], tuple[int, int], list[FingerprintMatch]] | None = None
    query_counts = Counter(query_frames)
    for query_run in query_runs:
        filtered = [
            match
            for match in matches
            if query_run[0] <= match.query_frame <= query_run[1]
            and query_run[0] <= match.query_target_frame <= query_run[1]
        ]
        if not filtered:
            continue
        reference_frames = []
        for match in filtered:
            reference_frames.extend((match.reference_frame, match.reference_target_frame))
        run_query_frames = [
            frame
            for match in filtered
            for frame in (match.query_frame, match.query_target_frame)
        ]
        query_low, query_high = _dense_limits(run_query_frames)
        dense_filtered = [
            match
            for match in filtered
            if query_low <= match.query_frame <= query_high
            and query_low <= match.query_target_frame <= query_high
        ]
        if dense_filtered:
            filtered = dense_filtered
        trimmed_query_frames = []
        for match in filtered:
            trimmed_query_frames.extend((match.query_frame, match.query_target_frame))
        query_low, query_high = _trimmed_limits(trimmed_query_frames)
        trimmed_filtered = [
            match
            for match in filtered
            if query_low <= match.query_frame <= query_high
            and query_low <= match.query_target_frame <= query_high
        ]
        if trimmed_filtered:
            filtered = trimmed_filtered
        dense_reference_frames = []
        for match in filtered:
            dense_reference_frames.extend((match.reference_frame, match.reference_target_frame))
        ref_dense_lo, ref_dense_hi = _trimmed_limits(dense_reference_frames)
        ref_lo = max(0, ref_dense_lo - retrieval.boundary_padding_frames)
        ref_hi = min(reference.frame_count - 1, ref_dense_hi + retrieval.boundary_padding_frames)
        query_low = max(0, query_low - retrieval.boundary_padding_frames)
        query_high = min(query.frame_count - 1, query_high + retrieval.boundary_padding_frames)
        query_run = (query_low, query_high)
        query_density = sum(query_counts[f] for f in range(query_run[0], query_run[1] + 1))
        score = float(query_density + len(filtered))
        key = (score, (ref_lo, ref_hi), query_run, filtered)
        if best is None or key[0] > best[0] or (key[0] == best[0] and (key[1], key[2]) < (best[1], best[2])):
            best = key
    if best is None:
        return (0, 0), (0, 0), []
    return best[1], best[2], best[3]


def _score_matches(
    matches: list[FingerprintMatch], tolerance: ToleranceConfig
) -> tuple[float, int, dict[str, Any] | None, list[dict[str, Any]]]:
    ordered = sorted(
        matches,
        key=lambda m: (
            m.query_frame,
            m.reference_frame,
            m.query_target_frame,
            m.reference_target_frame,
            m.key,
        ),
    )
    float_total = 0.0
    fixed_total = 0
    first_exceed: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []
    for step, match in enumerate(ordered, start=1):
        float_total += match.float_weight
        fixed_total += match.fixed_units
        fixed_float = fixed_total / tolerance.score_scale
        difference = abs(float_total - fixed_float)
        trace.append(
            {
                "step": step,
                "query_frame": match.query_frame,
                "reference_frame": match.reference_frame,
                "float_total": float_total,
                "fixed_units_total": fixed_total,
                "fixed_total": fixed_float,
                "abs_difference": difference,
            }
        )
        if first_exceed is None and difference > tolerance.score_tolerance:
            first_exceed = {
                "step": step,
                "query_frame": match.query_frame,
                "reference_frame": match.reference_frame,
                "float_total": float_total,
                "fixed_units_total": fixed_total,
                "fixed_total": fixed_float,
                "abs_difference": difference,
                "score_tolerance": tolerance.score_tolerance,
            }
    return float_total, fixed_total, first_exceed, trace


def _drift_and_offset(
    matches: list[FingerprintMatch],
    reference: TrackFingerprint,
    query: TrackFingerprint,
    reference_start_seconds: float,
    reference_end_seconds: float,
    base_sample_rate: int,
    hop_length: int,
    tolerance: ToleranceConfig,
) -> tuple[int, float, float, float, float, float, dict[str, Any], dict[str, Any]]:
    offsets = np.asarray([m.offset_seconds for m in matches], dtype=np.float64)
    offset_seconds = float(np.median(offsets))
    offset_analysis_samples = offset_seconds * base_sample_rate
    integer_frame_offset = int(math.floor(offset_analysis_samples / hop_length + 0.5))
    integer_analysis_samples = float(integer_frame_offset * hop_length)
    offset_reference_samples = offset_seconds * reference.sample_rate
    integer_reference_samples = integer_frame_offset * hop_length * reference.sample_rate / base_sample_rate
    native_conversion_error = abs(offset_reference_samples - integer_reference_samples)
    analysis_conversion_error = abs(offset_analysis_samples - integer_analysis_samples)

    ref_times = np.asarray([m.reference_time_seconds for m in matches], dtype=np.float64)
    query_times = np.asarray([m.query_time_seconds for m in matches], dtype=np.float64)
    rate_ratio = 1.0
    intercept = offset_seconds
    if ref_times.size >= 2 and float(np.ptp(ref_times)) > 1e-9:
        slope, fitted_intercept = np.polyfit(ref_times, query_times, 1)
        if math.isfinite(float(slope)):
            rate_ratio = float(slope)
            intercept = float(fitted_intercept)
    span = max(0.0, reference_end_seconds - reference_start_seconds)
    drift_seconds = (rate_ratio - 1.0) * span
    if abs(drift_seconds) <= 1e-6:
        direction = "stable"
    elif drift_seconds > 0:
        direction = "increasing"
    else:
        direction = "decreasing"
    drift = {
        "direction": direction,
        "rate_ratio": rate_ratio,
        "drift_percent": (rate_ratio - 1.0) * 100.0,
        "drift_seconds_over_span": drift_seconds,
        "drift_reference_samples_over_span": drift_seconds * reference.sample_rate,
        "note": "query_time ~= rate_ratio * reference_time + intercept",
    }
    alignment = {
        "model": "query_time = rate_ratio * reference_time + intercept_seconds",
        "rate_ratio": rate_ratio,
        "intercept_seconds": intercept,
        "offset_seconds_median": offset_seconds,
        "offset_analysis_samples_float": offset_analysis_samples,
        "integer_offset_analysis_samples": integer_analysis_samples,
        "offset_analysis_conversion_error_samples": analysis_conversion_error,
        "offset_reference_samples_float": offset_reference_samples,
        "integer_offset_reference_samples_float": integer_reference_samples,
        "offset_reference_conversion_error_samples": native_conversion_error,
        "time_offset_tolerance_samples": tolerance.time_offset_tolerance_samples,
        "within_time_offset_tolerance": native_conversion_error
        <= tolerance.time_offset_tolerance_samples,
    }
    return (
        integer_frame_offset,
        offset_seconds,
        offset_reference_samples,
        offset_analysis_samples,
        integer_analysis_samples,
        native_conversion_error,
        alignment,
        drift,
    )


def _finalize_candidate_parts(
    reference: TrackFingerprint,
    query: TrackFingerprint,
    matches: list[FingerprintMatch],
    retrieval: RetrievalConfig,
    tolerance: ToleranceConfig,
    base_sample_rate: int,
    hop_length: int,
) -> tuple[Fragment, list[FingerprintMatch]] | None:
    if len(matches) < retrieval.min_matches:
        return None
    (ref_lo, ref_hi), (query_lo, query_hi), filtered = _refine_candidate(matches, reference, query, retrieval)
    if not filtered or len(filtered) < retrieval.min_matches:
        return None
    float_score, fixed_units, first_exceed, trace = _score_matches(filtered, tolerance)
    ref_start_sample = reference.frames[ref_lo].source_start_sample
    ref_end_sample = reference.frames[ref_hi].source_end_sample
    query_start_sample = query.frames[query_lo].source_start_sample
    query_end_sample = query.frames[query_hi].source_end_sample
    ref_start_seconds = float(reference.frames[ref_lo].theoretical_start_seconds)
    ref_end_seconds = float(reference.frames[ref_hi].theoretical_end_seconds)
    query_start_seconds = float(query.frames[query_lo].theoretical_start_seconds)
    query_end_seconds = float(query.frames[query_hi].theoretical_end_seconds)
    (
        integer_frame_offset,
        offset_seconds,
        offset_reference_samples,
        offset_analysis_samples,
        integer_analysis_samples,
        conversion_error,
        alignment,
        drift,
    ) = _drift_and_offset(
        filtered,
        reference,
        query,
        ref_start_seconds,
        ref_end_seconds,
        base_sample_rate,
        hop_length,
        tolerance,
    )
    match_identity = min(_match_identity(match) for match in filtered)
    fragment = Fragment(
        reference_id=reference.track_id,
        query_id=query.track_id,
        match=match_identity,
        reference_start_sample=int(ref_start_sample),
        reference_end_sample=int(ref_end_sample),
        query_start_sample=int(query_start_sample),
        query_end_sample=int(query_end_sample),
        reference_start_seconds=ref_start_seconds,
        reference_end_seconds=ref_end_seconds,
        query_start_seconds=query_start_seconds,
        query_end_seconds=query_end_seconds,
        score_float=float_score,
        score_fixed_units=fixed_units,
        score_fixed=fixed_units / tolerance.score_scale,
        first_exceed=first_exceed,
        score_trace=trace,
        integer_frame_offset=integer_frame_offset,
        offset_seconds_float=offset_seconds,
        offset_reference_samples_float=offset_reference_samples,
        offset_analysis_samples_float=offset_analysis_samples,
        integer_offset_analysis_samples=integer_analysis_samples,
        offset_conversion_error_samples=conversion_error,
        alignment=alignment,
        drift=drift,
        anchor_count=len(filtered),
    )
    return fragment, filtered


def _overlaps(left: Fragment, right: Fragment) -> bool:
    return (
        left.reference_start_sample < right.reference_end_sample
        and right.reference_start_sample < left.reference_end_sample
        and left.query_start_sample < right.query_end_sample
        and right.query_start_sample < left.query_end_sample
    )


def _keep_highest(fragments: list[Fragment]) -> list[Fragment]:
    ordered = sorted(
        fragments,
        key=lambda f: (
            -f.score_fixed_units,
            f.reference_start_sample,
            f.query_start_sample,
            f.reference_end_sample,
            f.query_end_sample,
        ),
    )
    kept: list[Fragment] = []
    for fragment in ordered:
        if not any(_overlaps(fragment, existing) for existing in kept):
            kept.append(fragment)
    return kept


def _merge_overlaps(
    fragments: list[Fragment],
    candidate_matches: list[list[FingerprintMatch]],
    reference: TrackFingerprint,
    query: TrackFingerprint,
    retrieval: RetrievalConfig,
    tolerance: ToleranceConfig,
    base_sample_rate: int,
    hop_length: int,
) -> list[Fragment]:
    ids = list(range(len(fragments)))
    parent = list(range(len(fragments)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(len(fragments)):
        for j in range(i + 1, len(fragments)):
            if _overlaps(fragments[i], fragments[j]):
                union(i, j)
    components: dict[int, list[int]] = defaultdict(list)
    for index in ids:
        components[find(index)].append(index)
    merged: list[Fragment] = []
    for component in components.values():
        if len(component) == 1:
            merged.append(fragments[component[0]])
            continue
        unique: dict[tuple[tuple[int, int, int, int, int, int, int], int, int, int, int], FingerprintMatch] = {}
        for index in component:
            for match in candidate_matches[index]:
                unique[_match_identity(match)] = match
        combined = list(unique.values())
        result = _finalize_candidate_parts(
            reference, query, combined, retrieval, tolerance, base_sample_rate, hop_length
        )
        if result is not None:
            merged.append(result[0])
    return merged


def _fragments_for_reference(
    reference: TrackFingerprint,
    query: TrackFingerprint,
    matches: list[FingerprintMatch],
    retrieval: RetrievalConfig,
    tolerance: ToleranceConfig,
    base_sample_rate: int,
    hop_length: int,
) -> list[Fragment]:
    clustered = _cluster_offsets(matches, retrieval.voting_window_frames)
    raw: list[tuple[float, int, _Candidate]] = []
    for group in clustered:
        split_gap = max(1, retrieval.boundary_gap_frames)
        for component in _split_query_components(group, split_gap):
            if len(component) < retrieval.min_matches:
                continue
            reference_frames = [m.reference_frame for m in component] + [
                m.reference_target_frame for m in component
            ]
            query_frames = [m.query_frame for m in component] + [m.query_target_frame for m in component]
            candidate = _Candidate(
                matches=component,
                reference_lo=min(reference_frames),
                reference_hi=max(reference_frames),
                query_lo=min(query_frames),
                query_hi=max(query_frames),
            )
            weight = sum(m.fixed_units for m in component)
            raw.append((-weight, len(component), candidate))
    raw.sort(key=lambda value: (value[0], -value[1], value[2].reference_lo, value[2].query_lo))
    fragments: list[Fragment] = []
    matched_groups: list[list[FingerprintMatch]] = []
    for _, _, candidate in raw[: retrieval.max_candidates]:
        result = _finalize_candidate_parts(
            reference,
            query,
            candidate.matches,
            retrieval,
            tolerance,
            base_sample_rate,
            hop_length,
        )
        if result is not None:
            fragments.append(result[0])
            matched_groups.append(result[1])
    if retrieval.overlap_policy == "merge":
        return _merge_overlaps(
            fragments,
            matched_groups,
            reference,
            query,
            retrieval,
            tolerance,
            base_sample_rate,
            hop_length,
        )
    return _keep_highest(fragments)


def retrieve_similar_fragments(
    index: ReferenceIndex,
    query: TrackFingerprint,
    retrieval: RetrievalConfig,
    tolerance: ToleranceConfig,
    base_sample_rate: int,
    hop_length: int,
) -> list[Fragment]:
    all_matches = _collect_matches(index, query, tolerance.score_scale)
    by_reference: dict[str, list[FingerprintMatch]] = defaultdict(list)
    for match in all_matches:
        by_reference[match.reference_id].append(match)
    fragments: list[Fragment] = []
    for reference_id in index.reference_ids:
        matches = by_reference.get(reference_id, [])
        if not matches:
            continue
        reference = index.get(reference_id)
        fragments.extend(
            _fragments_for_reference(
                reference,
                query,
                matches,
                retrieval,
                tolerance,
                base_sample_rate,
                hop_length,
            )
        )
    fragments.sort(
        key=lambda f: (
            -f.score_fixed_units,
            f.reference_start_sample,
            f.query_start_sample,
            f.reference_id,
            f.reference_end_sample,
            f.query_end_sample,
        )
    )
    return fragments


def fragment_to_dict(fragment: Fragment) -> dict[str, Any]:
    trace_keys = [
        "step",
        "query_frame",
        "reference_frame",
        "float_total",
        "fixed_units_total",
        "fixed_total",
        "abs_difference",
    ]
    trace = {
        "format": "parallel_arrays",
        "field_order": trace_keys,
        **{key: [entry[key] for entry in fragment.score_trace] for key in trace_keys},
    }
    return {
        "reference_start_sample": fragment.reference_start_sample,
        "reference_end_sample": fragment.reference_end_sample,
        "query_start_sample": fragment.query_start_sample,
        "query_end_sample": fragment.query_end_sample,
        "reference_start_seconds": round(fragment.reference_start_seconds, 12),
        "reference_end_seconds": round(fragment.reference_end_seconds, 12),
        "query_start_seconds": round(fragment.query_start_seconds, 12),
        "query_end_seconds": round(fragment.query_end_seconds, 12),
        "score": fragment.score_fixed,
        "score_fixed_units": fragment.score_fixed_units,
        "score_float_reference": fragment.score_float,
        "first_exceed_score_tolerance": fragment.first_exceed,
        "score_trace": trace,
        "integer_frame_offset": fragment.integer_frame_offset,
        "offset_seconds_float": fragment.offset_seconds_float,
        "offset_reference_samples_float": fragment.offset_reference_samples_float,
        "offset_analysis_samples_float": fragment.offset_analysis_samples_float,
        "integer_offset_analysis_samples": fragment.integer_offset_analysis_samples,
        "offset_conversion_error_samples": fragment.offset_conversion_error_samples,
        "alignment": fragment.alignment,
        "drift": fragment.drift,
        "anchor_count": fragment.anchor_count,
    }
