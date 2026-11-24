"""Multi-version binary frame codec and canonical normalization.

Frame layout (all integers big-endian, unsigned)::

    magic        4 bytes  = b"SDQF"
    header       7 bytes  = header_version u8 | source_type u8
                          | format_version u8 | record_count u32
    record * N
        frame_len   u16  = byte length of (version, field_count, fields)
        version     u8
        field_count u8
        fields      field_count * u64

The first field of every record is the integer ``query_id``. Historical
versions expose different field sets; missing fields are backfilled during
normalization and recorded on every normalized record.
"""

from __future__ import annotations

import struct

from .errors import FormatError
from .models import (
    NormalizedRecord,
    SourceDeclaration,
    SOURCE_APP,
    SOURCE_DB,
    SOURCE_SLOW,
    SUPPORTED_VERSIONS,
)

MAGIC = b"SDQF"
HEADER_VERSION = 1
_HEADER_STRUCT = struct.Struct(">4sBBBI")
_LEN_STRUCT = struct.Struct(">H")
_META_STRUCT = struct.Struct(">BB")
_U64_STRUCT = struct.Struct(">Q")

# version -> ordered field layout (field 0 is always query_id).
FIELD_LAYOUTS: dict[tuple[str, int], tuple[str, ...]] = {
    (SOURCE_DB, 1): ("query_id", "latency_us", "rows_scanned"),
    (SOURCE_DB, 2): ("query_id", "latency_us", "rows_scanned", "call_count"),
    (SOURCE_APP, 1): ("query_id", "latency_ms", "call_count"),
    (SOURCE_APP, 2): ("query_id", "latency_ms", "call_count", "rows_scanned"),
    (SOURCE_APP, 3): (
        "query_id", "latency_ms", "call_count", "rows_scanned", "error_count",
    ),
    (SOURCE_SLOW, 1): ("query_id", "low_ms", "high_ms", "sample_count"),
    (SOURCE_SLOW, 2): (
        "query_id", "low_ms", "high_ms", "sample_count", "rows_scanned",
    ),
}


class FrameBuilder:
    """Accumulates records and serializes a deterministic binary frame."""

    def __init__(self, source_type: str, format_version: int) -> None:
        if source_type not in SUPPORTED_VERSIONS:
            raise FormatError(f"unsupported source_type {source_type!r}")
        if format_version not in SUPPORTED_VERSIONS[source_type]:
            raise FormatError(
                f"unsupported format_version {format_version}"
            )
        self.source_type = source_type
        self.format_version = format_version
        self._records: list[tuple[int, tuple[int, ...]]] = []

    def add(self, version: int, fields: tuple[int, ...]) -> "FrameBuilder":
        layout = FIELD_LAYOUTS.get((self.source_type, version))
        if layout is None:
            raise FormatError(f"unsupported record version {version}")
        if len(fields) != len(layout):
            raise FormatError(
                f"record needs {len(layout)} fields, got {len(fields)}",
                row=len(self._records) + 1,
            )
        for value in fields:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise FormatError("fields must be non-negative integers",
                                  row=len(self._records) + 1)
        self._records.append((version, tuple(fields)))
        return self

    def build(self) -> bytes:
        out = bytearray()
        out += _HEADER_STRUCT.pack(
            MAGIC, HEADER_VERSION, _type_tag(self.source_type),
            self.format_version, len(self._records),
        )
        for version, fields in self._records:
            payload = _META_STRUCT.pack(version, len(fields))
            payload += b"".join(_U64_STRUCT.pack(v) for v in fields)
            out += _LEN_STRUCT.pack(len(payload))
            out += payload
        return bytes(out)


def _type_tag(source_type: str) -> int:
    return {SOURCE_DB: 1, SOURCE_APP: 2, SOURCE_SLOW: 3}[source_type]


_TAG_TO_TYPE = {1: SOURCE_DB, 2: SOURCE_APP, 3: SOURCE_SLOW}


def decode_and_normalize(data: bytes, source: SourceDeclaration) -> list[NormalizedRecord]:
    """Decode a frame file and return canonical, backfilled records."""

    offset = 0
    if len(data) < _HEADER_STRUCT.size:
        raise FormatError(
            "frame shorter than header", source=source.source_id
        )
    magic, header_version, type_tag, header_fmt, record_count = _HEADER_STRUCT.unpack_from(data, 0)
    offset = _HEADER_STRUCT.size
    if magic != MAGIC:
        raise FormatError("bad frame magic", source=source.source_id)
    if header_version != HEADER_VERSION:
        raise FormatError(
            f"unknown frame header version {header_version}",
            source=source.source_id, detail={"header_version": header_version},
        )
    source_type = _TAG_TO_TYPE.get(type_tag)
    if source_type is None or source_type != source.source_type:
        raise FormatError(
            f"frame source_type tag {type_tag} does not match declaration",
            source=source.source_id,
        )
    if header_fmt not in SUPPORTED_VERSIONS[source_type]:
        raise FormatError(
            f"unknown header format version {header_fmt}",
            source=source.source_id,
        )

    records: list[NormalizedRecord] = []
    for row_number in range(1, record_count + 1):
        record, offset = _decode_record(data, offset, row_number, source, source_type)
        records.append(record)

    if offset != len(data):
        raise FormatError(
            f"trailing {len(data) - offset} bytes after declared records",
            source=source.source_id, row=record_count,
        )
    return records


def _decode_record(data: bytes, offset: int, row_number: int,
                   source: SourceDeclaration, source_type: str):
    if offset + _LEN_STRUCT.size > len(data):
        raise FormatError("truncated record length field",
                          source=source.source_id, row=row_number)
    (frame_len,) = _LEN_STRUCT.unpack_from(data, offset)
    offset += _LEN_STRUCT.size
    if frame_len < _META_STRUCT.size or frame_len % 8 != 2:
        raise FormatError(
            f"frame_len {frame_len} is not self-consistent",
            source=source.source_id, row=row_number,
        )
    if offset + frame_len > len(data):
        raise FormatError("declared frame_len exceeds remaining bytes",
                          source=source.source_id, row=row_number)
    version, field_count = _META_STRUCT.unpack_from(data, offset)
    meta_offset = offset
    offset += _META_STRUCT.size

    layout = FIELD_LAYOUTS.get((source_type, version))
    if layout is None:
        raise FormatError(
            f"unknown format version {version} for {source_type}",
            source=source.source_id, row=row_number,
        )
    expected_count = len(layout)
    if field_count != expected_count:
        raise FormatError(
            f"required {expected_count} fields for version {version}, "
            f"frame carries {field_count}",
            source=source.source_id, row=row_number,
        )
    if frame_len != _META_STRUCT.size + _U64_STRUCT.size * field_count:
        raise FormatError(
            "frame_len does not match field_count",
            source=source.source_id, row=row_number,
        )

    values: list[int] = []
    for _ in range(field_count):
        (value,) = _U64_STRUCT.unpack_from(data, offset)
        values.append(value)
        offset += _U64_STRUCT.size
    if offset != meta_offset + frame_len:
        raise FormatError("record boundary mismatch",
                          source=source.source_id, row=row_number)
    named = dict(zip(layout, values))
    return _normalize(named, version, row_number, source), offset


def _normalize(named: dict[str, int], version: int, row_number: int,
               source: SourceDeclaration) -> NormalizedRecord:
    query_id = named["query_id"]
    filled: list[str] = []
    reason: dict[str, str] = {}
    source_type = source.source_type

    if source_type == SOURCE_DB:
        latency_us = named["latency_us"]
        rows_scanned = named["rows_scanned"]
        call_count = named.get("call_count", 0)
        if "call_count" not in named:
            filled.append("call_count")
            reason["call_count"] = (
                "db_exec_stats v1 has no call_count; backfilled to 0"
            )
    elif source_type == SOURCE_APP:
        latency_us = named["latency_ms"] * 1000
        call_count = named["call_count"]
        rows_scanned = named.get("rows_scanned", 0)
        if "rows_scanned" not in named:
            filled.append("rows_scanned")
            reason["rows_scanned"] = (
                "app_log_summary v1 does not report scanned rows; backfilled to 0"
            )
    else:  # SOURCE_SLOW
        low_ms, high_ms = named["low_ms"], named["high_ms"]
        if high_ms < low_ms:
            raise FormatError(
                "slow-log interval high_ms must be >= low_ms",
                source=source.source_id, row=row_number,
            )
        # Integer midpoint: truncating toward zero (floor for non-negatives).
        latency_us = ((low_ms + high_ms) // 2) * 1000
        call_count = named["sample_count"]
        rows_scanned = named.get("rows_scanned", 0)
        if "rows_scanned" not in named:
            filled.append("rows_scanned")
            reason["rows_scanned"] = (
                "slow_log_excerpt v1 has no row count; backfilled to 0"
            )

    if latency_us < 0 or rows_scanned < 0 or call_count < 0:
        raise FormatError(
            "latency/rows/call counts must be non-negative",
            source=source.source_id, row=row_number,
        )

    return NormalizedRecord(
        source_id=source.source_id,
        source_type=source_type,
        format_version=version,
        row_number=row_number,
        query_id=query_id,
        latency_us=latency_us,
        rows_scanned=rows_scanned,
        call_count=call_count,
        filled_fields=tuple(filled),
        fill_reason=reason,
    )


def load_frame_file(source: SourceDeclaration) -> list[NormalizedRecord]:
    with open(source.frame_file, "rb") as handle:
        data = handle.read()
    return decode_and_normalize(data, source)
