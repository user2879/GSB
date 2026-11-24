"""Frame-by-frame continuation replay against uninterrupted execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .engine import Engine, FrameRecord
from .errors import WorkbenchError
from .model import SessionSpec, digest_value
from .snapshot import SnapshotStore
from .state_machine import Transition


FRAME_COMPARE_FIELDS = (
    "frame_type",
    "stream_id",
    "data_bytes",
    "connection_window_before",
    "connection_window_after",
    "stream_window_before",
    "stream_window_after",
    "business_code",
    "strategy",
    "event_seq",
)

TRANSITION_COMPARE_FIELDS = (
    "stream_id",
    "from_state",
    "to_state",
    "reason",
    "event_seq",
)


@dataclass(frozen=True)
class ReplayResult:
    snapshot_event_seq: int
    start_frame_seq: int
    frames_compared: int
    transitions_compared: int
    full_digest: str
    replay_digest: str
    match: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_event_seq": self.snapshot_event_seq,
            "start_frame_seq": self.start_frame_seq,
            "frames_compared": self.frames_compared,
            "transitions_compared": self.transitions_compared,
            "full_digest": self.full_digest,
            "replay_digest": self.replay_digest,
            "match": self.match,
        }


def _replay_error(
    message: str,
    field: str,
    *,
    event_seq: int | None,
    frame_seq: int | None,
    **details: Any,
) -> WorkbenchError:
    return WorkbenchError(
        "E_REPLAY",
        message,
        field=field,
        event_seq=event_seq,
        frame_seq=frame_seq,
        details=details,
    )


def _frame_digest(frames: list[FrameRecord], transitions: list[Transition]) -> str:
    return digest_value(
        {
            "frames": [frame.to_dict() for frame in frames],
            "transitions": [transition.to_dict() for transition in transitions],
        }
    )


def _compare_frames(expected: FrameRecord, actual: FrameRecord) -> None:
    if expected.frame_seq != actual.frame_seq:
        raise _replay_error(
            "frame sequence diverged",
            "frame_seq",
            event_seq=actual.event_seq,
            frame_seq=actual.frame_seq,
            expected=expected.frame_seq,
            actual=actual.frame_seq,
        )
    for field in FRAME_COMPARE_FIELDS:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value != actual_value:
            raise _replay_error(
                f"frame field {field} diverged",
                field,
                event_seq=actual.event_seq,
                frame_seq=actual.frame_seq,
                expected=expected_value,
                actual=actual_value,
            )


def _compare_transitions(expected: Transition, actual: Transition) -> None:
    for field in TRANSITION_COMPARE_FIELDS:
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value != actual_value:
            raise _replay_error(
                f"transition field {field} diverged",
                field,
                event_seq=actual.event_seq,
                frame_seq=actual.frame_seq,
                expected=expected_value,
                actual=actual_value,
            )


def replay_from_state(
    snapshot_spec: SessionSpec,
    snapshot_state: dict[str, Any],
    *,
    reference_engine: Engine | None = None,
) -> ReplayResult:
    reference = reference_engine or Engine(snapshot_spec).run()
    restored = Engine.from_state_dict(snapshot_spec, snapshot_state)
    if restored.spec.content_digest != reference.spec.content_digest:
        raise _replay_error(
            "snapshot session and reference session digests differ",
            "session_digest",
            event_seq=restored.next_event_seq,
            frame_seq=None,
        )
    start_frame_index = len(restored.frames)
    start_transition_index = len(restored.transitions)
    snapshot_event_seq = restored.next_event_seq
    start_frame_seq = restored.last_frame_seq + 1

    prefix_frames = reference.frames[:start_frame_index]
    if len(prefix_frames) != len(restored.frames):
        raise _replay_error(
            "snapshot frame prefix is longer than the reference run",
            "frame_count",
            event_seq=snapshot_event_seq,
            frame_seq=start_frame_seq,
        )
    for index, (expected, actual) in enumerate(zip(prefix_frames, restored.frames)):
        if expected.to_dict() != actual.to_dict():
            _compare_frames(expected, actual)
            raise _replay_error(
                "snapshot frame prefix differs from the reference run",
                "frame_prefix",
                event_seq=actual.event_seq,
                frame_seq=actual.frame_seq,
            )

    restored.run(drain=True)
    reference_frames = reference.frames[start_frame_index:]
    replay_frames = restored.frames[start_frame_index:]
    expected_transitions = reference.transitions[start_transition_index:]
    replay_transitions = restored.transitions[start_transition_index:]

    compared = min(len(reference_frames), len(replay_frames))
    for index in range(compared):
        _compare_frames(reference_frames[index], replay_frames[index])
    if len(reference_frames) != len(replay_frames):
        index = compared
        event_seq = (
            replay_frames[index].event_seq
            if index < len(replay_frames)
            else reference_frames[index].event_seq if index < len(reference_frames) else snapshot_event_seq
        )
        frame_seq = start_frame_seq + index
        raise _replay_error(
            "frame count diverged during replay",
            "frame_count",
            event_seq=event_seq,
            frame_seq=frame_seq,
            expected=len(reference_frames),
            actual=len(replay_frames),
        )

    transition_count = min(len(expected_transitions), len(replay_transitions))
    for index in range(transition_count):
        _compare_transitions(expected_transitions[index], replay_transitions[index])
    if len(expected_transitions) != len(replay_transitions):
        transition = (
            replay_transitions[transition_count]
            if transition_count < len(replay_transitions)
            else expected_transitions[transition_count]
            if transition_count < len(expected_transitions)
            else None
        )
        raise _replay_error(
            "transition count diverged during replay",
            "transition_count",
            event_seq=None if transition is None else transition.event_seq,
            frame_seq=None if transition is None else transition.frame_seq,
            expected=len(expected_transitions),
            actual=len(replay_transitions),
        )

    full_digest = _frame_digest(reference_frames, expected_transitions)
    replay_digest = _frame_digest(replay_frames, replay_transitions)
    if full_digest != replay_digest:
        raise _replay_error(
            "replay digest differs without a field-level divergence",
            "digest",
            event_seq=snapshot_event_seq,
            frame_seq=start_frame_seq,
            expected=full_digest,
            actual=replay_digest,
        )
    return ReplayResult(
        snapshot_event_seq=snapshot_event_seq,
        start_frame_seq=start_frame_seq,
        frames_compared=len(replay_frames),
        transitions_compared=len(replay_transitions),
        full_digest=full_digest,
        replay_digest=replay_digest,
        match=True,
    )


def replay_from_snapshot(
    store: SnapshotStore,
    snapshot_id: int,
    *,
    reference_spec: SessionSpec | None = None,
    reference_engine: Engine | None = None,
) -> ReplayResult:
    _info, snapshot_spec, snapshot_state = store.load(snapshot_id)
    spec = reference_spec or snapshot_spec
    if reference_spec is not None and reference_spec.content_digest != snapshot_spec.content_digest:
        raise _replay_error(
            "snapshot and requested reference sessions are not identical",
            "session_digest",
            event_seq=None,
            frame_seq=None,
        )
    reference = reference_engine or Engine(spec).run()
    return replay_from_state(snapshot_spec, snapshot_state, reference_engine=reference)
