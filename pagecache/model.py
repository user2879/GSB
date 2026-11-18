from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable, Sequence


SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)
CURRENT_FORMAT_VERSION = 3


class ColumnKind(IntEnum):
    INT = 1
    FLOAT = 2

    @property
    def name(self) -> str:
        return "int" if self is ColumnKind.INT else "float"


@dataclass(frozen=True, slots=True)
class ColumnSummary:
    name: str
    kind: ColumnKind
    null_count: int
    digest: bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.name,
            "null_count": self.null_count,
            "digest": self.digest.hex(),
        }


@dataclass(slots=True)
class PageEntry:
    page_id: str
    row_count: int
    record_version: int = CURRENT_FORMAT_VERSION
    column_summaries: tuple[ColumnSummary, ...] = ()
    payload_checksum: int | None = None
    compressed: bool = False
    overflow_ref: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "page_id": self.page_id,
            "row_count": self.row_count,
            "record_version": self.record_version,
            "column_summaries": [summary.to_dict() for summary in self.column_summaries],
            "payload_checksum": self.payload_checksum,
            "compressed": self.compressed,
            "overflow_ref": self.overflow_ref,
        }


@dataclass(frozen=True, slots=True)
class Manifest:
    header_version: int
    records: tuple[PageEntry, ...]

    @property
    def format_versions(self) -> tuple[int, ...]:
        return tuple(sorted({record.record_version for record in self.records}))

    @property
    def page_count(self) -> int:
        return len(self.records)

    def by_page_id(self) -> dict[str, PageEntry]:
        return {record.page_id: record for record in self.records}


@dataclass(frozen=True, slots=True)
class PageColumn:
    name: str
    kind: ColumnKind
    values: tuple[int | float | None, ...]
    null_bitmap: bytes


@dataclass(frozen=True, slots=True)
class PageData:
    page_id: str
    row_count: int
    columns: tuple[PageColumn, ...]
    raw_bytes: bytes = field(default=b"", repr=False, compare=False)

    def column(self, name: str) -> PageColumn:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(name)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


def iter_non_null(values: Iterable[int | float | None]) -> Iterable[int | float]:
    for value in values:
        if value is not None:
            yield value
