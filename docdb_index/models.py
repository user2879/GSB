"""Immutable domain data shared across validation, storage and querying."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .canonical import canonical_json, digest_bytes, entry_sort_key, is_number


@dataclass(frozen=True)
class IndexSpec:
    path: str
    segment_capacity: int
    allow_null: bool


@dataclass(frozen=True)
class CollectionSpec:
    name: str
    version: int
    primary_key: str
    documents: tuple[dict[str, Any], ...]
    indexes: tuple[IndexSpec, ...]


@dataclass(frozen=True)
class IndexEntry:
    value: int | float
    pk: int | float | str

    @property
    def value_json(self) -> str:
        return canonical_json(self.value)

    @property
    def pk_json(self) -> str:
        return canonical_json(self.pk)

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return entry_sort_key(self.value, self.pk)

    def identity(self) -> tuple[str, str]:
        return (self.value_json, self.pk_json)


@dataclass
class SegmentRecord:
    segment_id: str
    table_name: str
    version: int
    entry_count: int
    low_value_json: str
    low_pk_json: str
    high_value_json: str
    high_pk_json: str
    digest: str
    content_blob: bytes
    entries: list[IndexEntry] = field(default_factory=list)

    @property
    def low_key(self) -> tuple[Any, ...]:
        low_value = json.loads(self.low_value_json)
        low_pk = json.loads(self.low_pk_json)
        return entry_sort_key(low_value, low_pk)

    @property
    def high_key(self) -> tuple[Any, ...]:
        high_value = json.loads(self.high_value_json)
        high_pk = json.loads(self.high_pk_json)
        return entry_sort_key(high_value, high_pk)


def segment_content_blob(segment_id: str, version: int, entries: list[IndexEntry]) -> bytes:
    payload = {
        "entries": [[entry.value_json, entry.pk_json] for entry in entries],
        "segment_id": segment_id,
        "version": version,
    }
    return canonical_json(payload).encode("utf-8")


def segment_digest(blob: bytes) -> str:
    return digest_bytes(blob)


def validate_index_value(value: Any) -> bool:
    return is_number(value)
