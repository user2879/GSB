"""Five-tuple stream grouping and explicit clock-basis alignment."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .model import (
    AlignmentRecord,
    AlignmentResult,
    EventBundle,
    EventRecord,
    FiveTuple,
    FrameBundle,
    FrameObservation,
    TimeWindow,
)


def group_frames(frames: Iterable[FrameObservation]) -> dict[str, list[FrameObservation]]:
    grouped: dict[str, list[FrameObservation]] = defaultdict(list)
    for frame in frames:
        grouped[frame.flow_id].append(frame)
    return {flow_id: grouped[flow_id] for flow_id in sorted(grouped)}


def group_events(events: Iterable[EventRecord]) -> dict[str, list[EventRecord]]:
    grouped: dict[str, list[EventRecord]] = defaultdict(list)
    for event in events:
        grouped[event.flow_id].append(event)
    return {flow_id: grouped[flow_id] for flow_id in sorted(grouped)}


def translate_event_window(event: EventRecord) -> TimeWindow | None:
    if (
        event.time_window_ms is None
        or event.offset_to_frame_ms is None
        or event.time_basis is None
    ):
        return None
    return event.time_window_ms.translated(event.offset_to_frame_ms)


def align_observations(
    frame_bundle: FrameBundle,
    event_bundles: list[EventBundle],
) -> AlignmentResult:
    frames_by_flow = group_frames(frame_bundle.frames)
    events_by_flow: dict[str, list[EventRecord]] = defaultdict(list)
    for bundle in event_bundles:
        for event in bundle.records:
            events_by_flow[event.flow_id].append(event)

    flow_ids = sorted(set(frames_by_flow) | set(events_by_flow))
    records: list[AlignmentRecord] = []
    unaligned: list[str] = []
    for flow_id in flow_ids:
        frames = frames_by_flow.get(flow_id, [])
        events = events_by_flow.get(flow_id, [])
        missing: list[str] = []

        frame_sources = sorted({frame.source_id for frame in frames})
        event_sources = sorted({event.source_id for event in events})
        frame_clock = frame_bundle.time_basis.clock if frames else None
        event_clocks = sorted(
            {event.time_basis.clock for event in events if event.time_basis is not None}
        )

        if not frames:
            missing.append("frame_observation")
        if not events:
            missing.append("event_source")
        if frames and events and not event_clocks:
            missing.append("event_time_basis")
        if len(event_clocks) > 1:
            missing.append("event_time_basis_conflict")

        offsets = sorted(
            {event.offset_to_frame_ms for event in events if event.offset_to_frame_ms is not None}
        )
        offset = offsets[0] if len(offsets) == 1 else None
        if events and not offsets:
            missing.append("event_offset_to_frame_ms")
        if len(offsets) > 1:
            missing.append("event_offset_conflict")

        event_alignment_versions = sorted(
            {
                bundle.alignment_version
                for bundle in event_bundles
                for event in bundle.records
                if event.flow_id == flow_id and bundle.alignment_version
            }
        )
        if events and not event_alignment_versions:
            missing.append("event_alignment_basis_version")
        if any(version != frame_bundle.alignment_version for version in event_alignment_versions):
            missing.append("alignment_basis_version_mismatch")

        sequence_only = False
        if events and all(event.time_window_ms is None for event in events):
            sequence_only = True
            offset = None
            missing.append("event_time_window")

        translated = any(translate_event_window(event) is not None for event in events)
        aligned = not missing
        if not aligned:
            unaligned.append(flow_id)
        records.append(
            AlignmentRecord(
                flow_id=flow_id,
                frame_source_id=frame_sources[0] if len(frame_sources) == 1 else None,
                event_source_ids=event_sources,
                frame_clock=frame_clock,
                event_clock=event_clocks[0] if len(event_clocks) == 1 else None,
                offset_to_frame_ms=offset,
                translated_event_windows=translated,
                sequence_only=sequence_only,
                aligned=aligned,
                missing_basis=sorted(set(missing)),
            )
        )

    basis_version = frame_bundle.alignment_version
    return AlignmentResult(
        records=records,
        unaligned_flow_ids=unaligned,
        basis_version=basis_version,
        offset_policy="translation_only",
    )


def flow_five_tuples(
    frame_bundle: FrameBundle,
    event_bundles: list[EventBundle],
) -> dict[str, FiveTuple]:
    result: dict[str, FiveTuple] = {}
    for frame in frame_bundle.frames:
        result[frame.flow_id] = frame.five_tuple
    for bundle in event_bundles:
        for event in bundle.records:
            result[event.flow_id] = event.five_tuple
    return {flow_id: result[flow_id] for flow_id in sorted(result)}
