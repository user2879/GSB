"""Immutable declaration and snapshot models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EntrySpec:
    key: str
    size_bytes: int
    content_seed: str


@dataclass(frozen=True)
class SegmentSpec:
    segment_id: str
    capacity_bytes: int
    entries: tuple[EntrySpec, ...]

    @property
    def declared_used_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)


@dataclass(frozen=True)
class SnapshotEntrySpec:
    key: str
    digest: str | None = None


@dataclass(frozen=True)
class ViewSegmentSpec:
    segment_id: str
    entries: tuple[SnapshotEntrySpec, ...]


@dataclass(frozen=True)
class ViewSpec:
    view_id: str
    version: int
    segments: tuple[ViewSegmentSpec, ...]


@dataclass(frozen=True)
class CacheSpec:
    format_version: int
    segments: tuple[SegmentSpec, ...]
    views: tuple[ViewSpec, ...]
    source_path: Path | None = None

    def segment(self, segment_id: str) -> SegmentSpec:
        for segment in self.segments:
            if segment.segment_id == segment_id:
                return segment
        raise KeyError(segment_id)


@dataclass(frozen=True)
class OperationResult:
    ok: bool
    code: str
    operation_id: str | None = None
    operation_seq: int | None = None
    committed_cursor: int | None = None
    segment_id: str | None = None
    entry_key: str | None = None
    required_bytes: int | None = None
    available_bytes: int | None = None
    elapsed_ns: int | None = None
    idempotent: bool = False
    details: dict | None = None

    def to_dict(self) -> dict:
        payload: dict = {"ok": self.ok, "code": self.code}
        for key in (
            "operation_id",
            "operation_seq",
            "committed_cursor",
            "segment_id",
            "entry_key",
            "required_bytes",
            "available_bytes",
            "elapsed_ns",
        ):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.idempotent:
            payload["idempotent"] = True
        if self.details:
            payload["details"] = self.details
        return payload

