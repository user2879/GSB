"""Shared sequence and edit-plan models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fingerprints import fingerprint


@dataclass(frozen=True)
class SequenceRecord:
    seq_id: str
    observations: tuple[Any, ...]
    timestamp_interval: Any
    segment_ids: tuple[Any, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.seq_id,
            "observations": list(self.observations),
            "timestamp_interval": self.timestamp_interval,
            "segment_ids": list(self.segment_ids),
        }

    def fingerprint(self) -> str:
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SequenceRecord":
        return cls(
            seq_id=raw["id"],
            observations=tuple(raw.get("observations", [])),
            timestamp_interval=raw.get("timestamp_interval", 1.0),
            segment_ids=tuple(raw.get("segment_ids", [])),
        )


@dataclass(frozen=True)
class EditBatch:
    batch: int
    edits: tuple[dict[str, Any], ...]


@dataclass
class LibraryData:
    records: dict[str, SequenceRecord] = field(default_factory=dict)
    invalid_entries: list[dict[str, Any]] = field(default_factory=list)
    raw_entries: list[dict[str, Any]] = field(default_factory=list)

    def ordered_records(self) -> list[SequenceRecord]:
        return [self.records[key] for key in sorted(self.records)]

    def copy(self) -> "LibraryData":
        return LibraryData(
            records=dict(self.records),
            invalid_entries=[dict(entry) for entry in self.invalid_entries],
            raw_entries=[dict(entry) for entry in self.raw_entries],
        )
