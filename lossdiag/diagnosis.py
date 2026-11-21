"""Sequence-axis gap detection and deterministic cause classification."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .alignment import flow_five_tuples, translate_event_window
from .model import (
    EventBundle,
    EventRecord,
    FlowDiagnosis,
    FrameBundle,
    FrameObservation,
    Gap,
)
from .util import content_digest


TRUE_LOSS = "true_loss"
RETRANSMISSION = "retransmission"
REORDERING = "reordering"
CLASSIFICATIONS = (TRUE_LOSS, RETRANSMISSION, REORDERING)


def _frame_evidence(
    frames: list[FrameObservation],
) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    retransmission: dict[int, set[str]] = defaultdict(set)
    reordering: dict[int, set[str]] = defaultdict(set)
    by_seq: dict[int, list[FrameObservation]] = defaultdict(list)
    for frame in frames:
        by_seq[frame.seq].append(frame)
    for seq, observations in by_seq.items():
        if len(observations) > 1:
            retransmission[seq].add("duplicate_sequence_observation")
        if any(frame.retransmit for frame in observations):
            retransmission[seq].add("frame_retransmit_flag")
        if any(frame.reorder for frame in observations):
            reordering[seq].add("frame_reorder_flag")

    first_arrival: dict[int, int] = {}
    for seq, observations in by_seq.items():
        first_arrival[seq] = min(frame.arrival_ordinal for frame in observations)
    max_seen = -1
    for seq in sorted(first_arrival):
        if first_arrival[seq] < max_seen:
            reordering[seq].add("arrival_order_inversion")
        max_seen = max(max_seen, first_arrival[seq])
    return retransmission, reordering


def _event_classification(event: EventRecord) -> str:
    if event.retransmit:
        return RETRANSMISSION
    if event.reorder_depth > 0:
        return REORDERING
    return TRUE_LOSS


def _event_evidence(event: EventRecord) -> list[str]:
    evidence = [f"event_list:{event.source_version}"]
    if event.retransmit:
        evidence.append("event_retransmit_marker")
    if event.reorder_depth > 0:
        evidence.append("event_reorder_depth")
    if event.time_window_ms is None:
        evidence.append("event_window_missing")
    return evidence


def _merge_time_windows(windows: Iterable[tuple[int, int]]) -> tuple[dict[str, int] | None, int]:
    valid = sorted((start, end) for start, end in windows if end >= start)
    if not valid:
        return None, 0
    start = valid[0][0]
    end = max(item[1] for item in valid)
    count = len(valid)
    return {"start_ms": start, "end_ms": end}, count


def _build_gaps_for_flow(
    *,
    flow_id: str,
    frames: list[FrameObservation],
    events: list[EventRecord],
    frame_model_version: str,
) -> tuple[list[Gap], dict[str, int], int, int, int, int]:
    by_seq: dict[int, list[FrameObservation]] = defaultdict(list)
    for frame in frames:
        by_seq[frame.seq].append(frame)
    observed_seqs = set(by_seq)

    event_intervals = [(event.event_seq, event.event_seq + event.loss_count - 1) for event in events]
    envelope_starts: list[int] = []
    envelope_ends: list[int] = []
    if observed_seqs:
        envelope_starts.append(min(observed_seqs))
        envelope_ends.append(max(observed_seqs))
    for start, end in event_intervals:
        if end >= start:
            envelope_starts.append(start)
            envelope_ends.append(end)
    if not envelope_starts:
        return [], {"start_seq": 0, "end_seq": 0}, 0, 0, 0, 0
    envelope_start = min(envelope_starts)
    envelope_end = max(envelope_ends)

    candidates: set[int] = set()
    evidence: dict[int, set[str]] = defaultdict(set)
    event_classes: dict[int, set[str]] = defaultdict(set)
    event_windows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frame_windows: dict[int, list[tuple[int, int]]] = defaultdict(list)
    versions: dict[int, set[str]] = defaultdict(set)
    frame_retransmission, frame_reordering = _frame_evidence(frames)

    if observed_seqs:
        for seq in range(envelope_start, envelope_end + 1):
            if seq not in observed_seqs:
                candidates.add(seq)
                evidence[seq].add("frame_sequence_hole")
                versions[seq].add(frame_model_version)

    for event in events:
        start, end = event.event_seq, event.event_seq + event.loss_count - 1
        if end < start:
            continue
        classification = _event_classification(event)
        interval_retransmission = any(
            frame_retransmission.get(seq) for seq in range(start, end + 1)
        )
        interval_reordering = any(
            frame_reordering.get(seq) for seq in range(start, end + 1)
        )
        frame_hint_used = False
        if classification == TRUE_LOSS and interval_retransmission:
            classification = RETRANSMISSION
            frame_hint_used = True
        elif classification == TRUE_LOSS and interval_reordering:
            classification = REORDERING
            frame_hint_used = True
        window = translate_event_window(event)
        for seq in range(start, end + 1):
            candidates.add(seq)
            event_classes[seq].add(classification)
            evidence[seq].update(_event_evidence(event))
            if frame_hint_used:
                evidence[seq].add("frame_evidence_in_event_interval")
            versions[seq].add(event.source_version)
            if window is not None and event.time_basis is not None:
                event_windows[seq].append((window.start_ms, window.end_ms))
            if seq in observed_seqs:
                evidence[seq].add("event_observed_sequence_conflict")

    for seq in sorted(observed_seqs):
        observations = by_seq[seq]
        if len(observations) > 1:
            candidates.add(seq)
            evidence[seq].add("duplicate_sequence_observation")
            versions[seq].add(frame_model_version)
        if any(frame.retransmit for frame in observations):
            candidates.add(seq)
            evidence[seq].add("frame_retransmit_flag")
            versions[seq].add(frame_model_version)
        if any(frame.reorder for frame in observations):
            candidates.add(seq)
            evidence[seq].add("frame_reorder_flag")
            versions[seq].add(frame_model_version)
        for frame in observations:
            frame_windows[seq].append((frame.send_ms, frame.send_ms))

    for seq in sorted(set(frame_retransmission) | set(frame_reordering)):
        candidates.add(seq)
        evidence[seq].update(frame_retransmission.get(seq, set()))
        evidence[seq].update(frame_reordering.get(seq, set()))
        versions[seq].add(frame_model_version)

    classified: dict[int, str] = {}
    for seq in candidates:
        classes = event_classes.get(seq, set())
        if RETRANSMISSION in classes or frame_retransmission.get(seq):
            classification = RETRANSMISSION
            if frame_retransmission.get(seq):
                evidence[seq].update(frame_retransmission[seq])
        elif REORDERING in classes or frame_reordering.get(seq):
            classification = REORDERING
            if frame_reordering.get(seq):
                evidence[seq].update(frame_reordering[seq])
        else:
            classification = TRUE_LOSS
        classified[seq] = classification

    gaps: list[Gap] = []
    for seq in sorted(classified):
        classification = classified[seq]
        windows = event_windows.get(seq) or frame_windows.get(seq, [])
        time_source = "event_window_ms" if event_windows.get(seq) else (
            "frame_send_ms" if frame_windows.get(seq) else "unwindowed"
        )
        if gaps and gaps[-1].end_seq + 1 == seq and gaps[-1].classification == classification:
            previous = gaps[-1]
            combined_windows = bool(previous.time_window_ms) or bool(windows)
            if combined_windows:
                if previous.time_window_ms:
                    candidate_windows = [(previous.time_window_ms["start_ms"], previous.time_window_ms["end_ms"])]
                else:
                    candidate_windows = []
                candidate_windows.extend(windows)
                merged, _ = _merge_time_windows(candidate_windows)
            else:
                merged = None
            previous_sources = set(previous.source_versions)
            previous_sources.update(versions.get(seq, set()))
            previous_evidence = set(previous.evidence)
            previous_evidence.update(evidence.get(seq, set()))
            gaps[-1] = Gap(
                flow_id=flow_id,
                start_seq=previous.start_seq,
                end_seq=seq,
                classification=classification,
                count=seq - previous.start_seq + 1,
                evidence=tuple(sorted(previous_evidence)),
                source_versions=tuple(sorted(previous_sources)),
                time_window_ms=merged,
                time_source=previous.time_source if previous.time_source == time_source else "mixed_explicit",
            )
        else:
            merged, _ = _merge_time_windows(windows)
            gaps.append(
                Gap(
                    flow_id=flow_id,
                    start_seq=seq,
                    end_seq=seq,
                    classification=classification,
                    count=1,
                    evidence=tuple(sorted(evidence.get(seq, set()))),
                    source_versions=tuple(sorted(versions.get(seq, set()))),
                    time_window_ms=merged,
                    time_source=time_source,
                )
            )

    true_loss = sum(1 for classification in classified.values() if classification == TRUE_LOSS)
    retransmission = sum(1 for classification in classified.values() if classification == RETRANSMISSION)
    reordering = sum(1 for classification in classified.values() if classification == REORDERING)
    unwindowed_true_loss = sum(
        1
        for seq, classification in classified.items()
        if classification == TRUE_LOSS and not (event_windows.get(seq) or frame_windows.get(seq))
    )
    return (
        gaps,
        {"start_seq": envelope_start, "end_seq": envelope_end},
        true_loss,
        retransmission,
        reordering,
        unwindowed_true_loss,
    )


def diagnose_streams(
    frame_bundle: FrameBundle,
    event_bundles: list[EventBundle],
) -> list[FlowDiagnosis]:
    frames_by_flow: dict[str, list[FrameObservation]] = defaultdict(list)
    for frame in frame_bundle.frames:
        frames_by_flow[frame.flow_id].append(frame)
    events_by_flow: dict[str, list[EventRecord]] = defaultdict(list)
    for bundle in event_bundles:
        for event in bundle.records:
            events_by_flow[event.flow_id].append(event)
    five_tuples = flow_five_tuples(frame_bundle, event_bundles)

    diagnoses: list[FlowDiagnosis] = []
    for flow_id in sorted(set(frames_by_flow) | set(events_by_flow)):
        frames = frames_by_flow.get(flow_id, [])
        events = events_by_flow.get(flow_id, [])
        gaps, envelope, true_loss, retransmission, reordering, unwindowed = _build_gaps_for_flow(
            flow_id=flow_id,
            frames=frames,
            events=events,
            frame_model_version=frame_bundle.model_version,
        )
        source_versions = {frame_bundle.model_version} if frames else set()
        source_versions.update(event.source_version for event in events)
        observed_unique = len({frame.seq for frame in frames})
        core = {
            "serialization": "gap-v1",
            "flow_id": flow_id,
            "five_tuple": five_tuples[flow_id].to_dict(),
            "sequence_envelope": envelope,
            "observed_unique_count": observed_unique,
            "observed_frame_count": len(frames),
            "gaps": [gap.to_dict() for gap in gaps],
            "counts": {
                "true_loss": true_loss,
                "retransmission": retransmission,
                "reordering": reordering,
                "unwindowed_true_loss": unwindowed,
            },
            "source_versions": sorted(source_versions),
        }
        diagnoses.append(
            FlowDiagnosis(
                flow_id=flow_id,
                five_tuple=five_tuples[flow_id],
                sequence_envelope=envelope,
                observed_unique_count=observed_unique,
                observed_frame_count=len(frames),
                gaps=gaps,
                true_loss_count=true_loss,
                retransmission_count=retransmission,
                reordering_count=reordering,
                unwindowed_true_loss_count=unwindowed,
                gap_digest=content_digest(core),
                source_versions=sorted(source_versions),
            )
        )
    return diagnoses


def diagnosis_digest(diagnoses: Iterable[FlowDiagnosis]) -> str:
    return content_digest([diagnosis.to_dict() for diagnosis in diagnoses])
