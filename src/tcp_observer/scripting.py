from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import DupAckError, ScriptError
from .spec import SessionSpec


@dataclass(frozen=True)
class ScriptGap:
    after_event_seq: int
    gap_start_seq: int
    next_present_seq: int | None
    missing_count: int


def find_first_gap(spec: SessionSpec) -> ScriptGap | None:
    present = {entry.event_seq for entry in spec.entries}
    expected = list(range(spec.initial_seq, spec.last_event_seq + 1))
    previous = spec.initial_seq - 1
    for seq in expected:
        if seq not in present:
            next_present = next((value for value in expected if value > seq and value in present), None)
            missing = 1
            cursor = seq + 1
            while cursor not in present and cursor <= spec.last_event_seq:
                missing += 1
                cursor += 1
            return ScriptGap(
                after_event_seq=previous,
                gap_start_seq=seq,
                next_present_seq=next_present,
                missing_count=missing,
            )
        previous = seq
    return None


def assert_no_gap(spec: SessionSpec) -> None:
    gap = find_first_gap(spec)
    if gap is not None:
        raise ScriptError(
            f"script event sequence gap starts at {gap.gap_start_seq}; missing {gap.missing_count} event(s)",
            field_name="entries.event_seq",
            event_seq=gap.gap_start_seq,
            details={
                "gap_start_seq": gap.gap_start_seq,
                "missing_count": gap.missing_count,
                "after_event_seq": gap.after_event_seq,
                "next_present_seq": gap.next_present_seq,
                "stopped_from_event_seq": gap.gap_start_seq,
            },
        )


def dupack_windows(spec: SessionSpec) -> list[dict[str, object]]:
    length = spec.seal_length
    result: list[dict[str, object]] = []
    for window_index, start in enumerate(range(spec.initial_seq, spec.last_event_seq + 1, length)):
        end = min(start + length - 1, spec.last_event_seq)
        entries = [
            entry
            for entry in spec.entries
            if start <= entry.event_seq <= end and entry.dup_count > 0
        ]
        count = sum(entry.dup_count for entry in entries)
        if count:
            result.append(
                {
                    "window_index": window_index,
                    "start_seq": start,
                    "end_seq": end,
                    "dup_ack_count": count,
                    "event_seqs": [entry.event_seq for entry in entries],
                }
            )
    return result


def assert_no_dupack_cluster(spec: SessionSpec) -> None:
    for window in dupack_windows(spec):
        count = int(window["dup_ack_count"])
        if count >= spec.dupack_cluster_limit:
            raise DupAckError(
                f"duplicate ACK cluster count {count} >= limit {spec.dupack_cluster_limit}",
                field_name="entries.dup_count",
                event_seq=int(window["start_seq"]),
                details={
                    "window_index": window["window_index"],
                    "window_start_seq": window["start_seq"],
                    "window_end_seq": window["end_seq"],
                    "count": count,
                    "limit": spec.dupack_cluster_limit,
                    "event_seqs": window["event_seqs"],
                },
            )


def validate_for_run(spec: SessionSpec) -> None:
    assert_no_gap(spec)
    assert_no_dupack_cluster(spec)
