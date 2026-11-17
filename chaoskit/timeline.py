"""Seed-driven deterministic injection timeline.

The timeline is derived from the experiment seed and the injection items
sorted by a stable key ``(target_id, type, injection_id)``. Textual order in
the JSON declaration therefore cannot change the result, and the same
declaration + seed always produces a byte-identical timeline digest.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List

from .errors import TimelineError
from .rng import SplitMix64
from .spec import ExperimentSpec, InjectionSpec, canonical_json_bytes

# Maximum extra gap between two starts. Kept small so demos stay quick while
# still exercising inter-injection spacing deterministically.
MAX_GAP_MS = 200


@dataclass(frozen=True)
class TimelineEntry:
    index: int
    start_offset_ms: int
    target_id: str
    injection_id: str
    injection_type: str
    duration_ms: int
    params: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_offset_ms": self.start_offset_ms,
            "target_id": self.target_id,
            "injection_id": self.injection_id,
            "injection_type": self.injection_type,
            "duration_ms": self.duration_ms,
            "params": self.params,
        }


def build_timeline(spec: ExperimentSpec) -> List[TimelineEntry]:
    ordered = spec.sorted_injections()
    rng = SplitMix64(spec.seed)
    entries: List[TimelineEntry] = []
    cursor = 0
    for index, item in enumerate(ordered):
        if index > 0:
            cursor += rng.next_below(MAX_GAP_MS + 1)
        entries.append(
            TimelineEntry(
                index=index,
                start_offset_ms=cursor,
                target_id=item.target_id,
                injection_id=item.injection_id,
                injection_type=item.injection_type,
                duration_ms=item.duration_ms,
                params=dict(sorted(item.params.items())),
            )
        )
        cursor += item.duration_ms
    return entries


def timeline_digest(entries: List[TimelineEntry]) -> str:
    payload = canonical_json_bytes([entry.to_dict() for entry in entries])
    return hashlib.sha256(payload).hexdigest()


def verify_timeline(expected_entries: List[TimelineEntry], actual_entries: List[TimelineEntry]) -> None:
    """Compare two timelines field by field and raise on first deviation."""
    if len(expected_entries) != len(actual_entries):
        raise TimelineError(
            "timeline length deviates from committed ledger",
            first_diverging_index=min(len(expected_entries), len(actual_entries)),
            expected=len(expected_entries),
            actual=len(actual_entries),
        )
    compare_fields = (
        "start_offset_ms",
        "target_id",
        "injection_id",
        "injection_type",
        "duration_ms",
        "params",
    )
    for index, (expected, actual) in enumerate(zip(expected_entries, actual_entries)):
        for field_name in compare_fields:
            if getattr(expected, field_name) != getattr(actual, field_name):
                raise TimelineError(
                    f"timeline deviation at injection {index} field {field_name}",
                    first_diverging_index=index,
                    expected=getattr(expected, field_name),
                    actual=getattr(actual, field_name),
                )
