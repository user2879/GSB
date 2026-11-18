from __future__ import annotations

from pathlib import Path
import struct
import zlib

from .errors import EngineError, ErrorCode
from .model import (
    CURRENT_FORMAT_VERSION,
    SUPPORTED_FORMAT_VERSIONS,
    ColumnKind,
    ColumnSummary,
    Manifest,
    PageEntry,
)


MANIFEST_MAGIC = b"PCMANIF1"
MANIFEST_TRAILER = b"PCMEND01"
_HEADER = struct.Struct("<8sHHI")
_RECORD_LENGTH = struct.Struct("<I")
_TRAILER = struct.Struct("<8sI")
_COMMON_HEAD = struct.Struct("<BH")
_ROW_COUNT = struct.Struct("<I")
_SUMMARY_HEAD = struct.Struct("<HBQ16s")
_CHECKSUM = struct.Struct("<I")
_V3_HEAD = struct.Struct("<BH")
_RECORD_CRC = struct.Struct("<I")


def encode_record(entry: PageEntry, *, format_version: int = CURRENT_FORMAT_VERSION) -> bytes:
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "unsupported manifest format version",
            path="$manifest.record.format_version",
            details={"format_version": format_version},
        )
    page_id_bytes = entry.page_id.encode("utf-8")
    if not page_id_bytes or len(page_id_bytes) > 65535:
        raise EngineError(ErrorCode.E_FORMAT, "invalid page_id length", path="$manifest.record.page_id")
    if entry.row_count <= 0 or entry.row_count > 0xFFFFFFFF:
        raise EngineError(ErrorCode.E_FORMAT, "row_count must fit a positive u32", path="$manifest.record.row_count")
    body = bytearray()
    body.extend(_COMMON_HEAD.pack(format_version, len(page_id_bytes)))
    body.extend(page_id_bytes)
    body.extend(_ROW_COUNT.pack(entry.row_count))

    if format_version >= 2:
        summaries = entry.column_summaries
        if len(summaries) > 65535:
            raise EngineError(ErrorCode.E_FORMAT, "too many column summaries", path="$manifest.record.column_summaries")
        body.extend(struct.pack("<H", len(summaries)))
        for index, summary in enumerate(summaries):
            name_bytes = summary.name.encode("utf-8")
            if len(name_bytes) > 65535 or len(summary.digest) != 16:
                raise EngineError(
                    ErrorCode.E_FORMAT,
                    "invalid column summary",
                    path=f"$manifest.record.column_summaries[{index}]",
                )
            body.extend(_SUMMARY_HEAD.pack(len(name_bytes), int(summary.kind), summary.null_count, summary.digest))
            body.extend(name_bytes)
        if entry.payload_checksum is None or not 0 <= entry.payload_checksum <= 0xFFFFFFFF:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "v2/v3 record requires a 32-bit payload checksum",
                path="$manifest.record.payload_checksum",
            )
        body.extend(_CHECKSUM.pack(entry.payload_checksum))

    if format_version >= 3:
        ref_bytes = (entry.overflow_ref or "").encode("utf-8")
        if len(ref_bytes) > 65535:
            raise EngineError(ErrorCode.E_FORMAT, "overflow reference is too long", path="$manifest.record.overflow_ref")
        body.extend(_V3_HEAD.pack(1 if entry.compressed else 0, len(ref_bytes)))
        body.extend(ref_bytes)

    crc = zlib.crc32(body) & 0xFFFFFFFF
    record = bytes(body) + _RECORD_CRC.pack(crc)
    return _RECORD_LENGTH.pack(len(record)) + record


def encode_manifest(entries: list[PageEntry] | tuple[PageEntry, ...], *, format_version: int = CURRENT_FORMAT_VERSION) -> bytes:
    if format_version not in SUPPORTED_FORMAT_VERSIONS:
        raise EngineError(ErrorCode.E_FORMAT, "unsupported manifest version", path="$manifest.version")
    chunks = [_HEADER.pack(MANIFEST_MAGIC, format_version, 0, len(entries))]
    for entry in entries:
        chunks.append(encode_record(entry, format_version=format_version))
    body = b"".join(chunks)
    return body + _TRAILER.pack(MANIFEST_TRAILER, zlib.crc32(body) & 0xFFFFFFFF)


def write_manifest(
    path: str | Path,
    entries: list[PageEntry] | tuple[PageEntry, ...],
    *,
    format_version: int = CURRENT_FORMAT_VERSION,
    append_trailer: bool = True,
) -> bytes:
    data = encode_manifest(entries, format_version=format_version)
    if not append_trailer:
        data = data[: -_TRAILER.size]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def read_manifest(path: str | Path) -> Manifest:
    target = Path(path)
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise EngineError(ErrorCode.E_FORMAT, f"cannot read manifest: {exc}", path=str(target)) from exc
    return decode_manifest(data, source_path=str(target))


def decode_manifest(data: bytes, *, source_path: str = "$manifest") -> Manifest:
    minimum = _HEADER.size + _TRAILER.size
    if len(data) < minimum:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "manifest is truncated",
            path=f"{source_path}.footer",
            details={"size": len(data), "minimum": minimum},
        )
    magic, header_version, _flags, record_count = _HEADER.unpack_from(data, 0)
    if magic != MANIFEST_MAGIC:
        raise EngineError(ErrorCode.E_FORMAT, "invalid manifest magic", path=f"{source_path}.magic")
    if header_version not in SUPPORTED_FORMAT_VERSIONS:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "unsupported manifest format version",
            path=f"{source_path}.version",
            details={"version": header_version, "supported": list(SUPPORTED_FORMAT_VERSIONS)},
        )

    offset = _HEADER.size
    body_end_guess = len(data) - _TRAILER.size
    records: list[PageEntry] = []
    seen_ids: set[str] = set()
    for record_index in range(record_count):
        if offset + _RECORD_LENGTH.size > body_end_guess:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "manifest record length field is truncated",
                path=f"{source_path}.records[{record_index}].record_length",
                record_index=record_index,
            )
        record_length = _RECORD_LENGTH.unpack_from(data, offset)[0]
        offset += _RECORD_LENGTH.size
        if record_length < 1 + 4:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "manifest record length is too small",
                path=f"{source_path}.records[{record_index}].record_length",
                record_index=record_index,
                details={"record_length": record_length},
            )
        record_end = offset + record_length
        if record_end > body_end_guess:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "manifest record exceeds the available body length",
                path=f"{source_path}.records[{record_index}].record_length",
                record_index=record_index,
                details={"record_length": record_length, "available": max(0, body_end_guess - offset)},
            )
        record = data[offset:record_end]
        offset = record_end
        entry = _decode_record(record, record_index, source_path)
        if entry.page_id in seen_ids:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "duplicate page identifier",
                path=f"{source_path}.records[{record_index}].page_id",
                record_index=record_index,
                page_id=entry.page_id,
            )
        seen_ids.add(entry.page_id)
        records.append(entry)

    if offset != body_end_guess:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "manifest body length does not match the trailer position",
            path=f"{source_path}.length",
            details={"parsed_end": offset, "expected_end": body_end_guess},
        )
    trailer_magic, expected_crc = _TRAILER.unpack_from(data, offset)
    if trailer_magic != MANIFEST_TRAILER:
        raise EngineError(ErrorCode.E_FORMAT, "invalid manifest trailer", path=f"{source_path}.footer.magic")
    actual_crc = zlib.crc32(data[:offset]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "manifest file checksum mismatch",
            path=f"{source_path}.footer.checksum",
            details={"expected": expected_crc, "actual": actual_crc},
        )
    return Manifest(header_version=header_version, records=tuple(records))


def _decode_record(record: bytes, record_index: int, source_path: str) -> PageEntry:
    base = f"{source_path}.records[{record_index}]"
    if len(record) < _COMMON_HEAD.size + _ROW_COUNT.size + _RECORD_CRC.size:
        raise EngineError(ErrorCode.E_FORMAT, "record is truncated", path=base, record_index=record_index)
    expected_crc = _RECORD_CRC.unpack_from(record, len(record) - _RECORD_CRC.size)[0]
    actual_crc = zlib.crc32(record[: -_RECORD_CRC.size]) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "record checksum mismatch",
            path=f"{base}.record_crc",
            record_index=record_index,
            details={"expected": expected_crc, "actual": actual_crc},
        )
    body = record[: -_RECORD_CRC.size]
    offset = 0
    version, page_id_len = _COMMON_HEAD.unpack_from(body, offset)
    offset += _COMMON_HEAD.size
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "unsupported record format version",
            path=f"{base}.record_version",
            record_index=record_index,
            details={"version": version, "supported": list(SUPPORTED_FORMAT_VERSIONS)},
        )
    if page_id_len == 0 or offset + page_id_len + _ROW_COUNT.size > len(body):
        raise EngineError(
            ErrorCode.E_FORMAT,
            "record page_id length is invalid",
            path=f"{base}.page_id_len",
            record_index=record_index,
        )
    try:
        page_id = body[offset : offset + page_id_len].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EngineError(ErrorCode.E_FORMAT, "page_id is not valid UTF-8", path=f"{base}.page_id", record_index=record_index) from exc
    offset += page_id_len
    if not page_id or "/" in page_id or "\\" in page_id:
        raise EngineError(ErrorCode.E_FORMAT, "invalid page_id", path=f"{base}.page_id", record_index=record_index)
    row_count = _ROW_COUNT.unpack_from(body, offset)[0]
    offset += _ROW_COUNT.size
    if row_count <= 0:
        raise EngineError(ErrorCode.E_FORMAT, "row_count must be positive", path=f"{base}.row_count", record_index=record_index, page_id=page_id)

    summaries: tuple[ColumnSummary, ...] = ()
    payload_checksum: int | None = None
    if version >= 2:
        if offset + 2 > len(body):
            raise EngineError(ErrorCode.E_FORMAT, "column summary count is missing", path=f"{base}.column_summaries", record_index=record_index)
        summary_count = struct.unpack_from("<H", body, offset)[0]
        offset += 2
        parsed_summaries: list[ColumnSummary] = []
        for summary_index in range(summary_count):
            summary_base = f"{base}.column_summaries[{summary_index}]"
            if offset + _SUMMARY_HEAD.size > len(body):
                raise EngineError(ErrorCode.E_FORMAT, "column summary is truncated", path=summary_base, record_index=record_index)
            name_len, kind_value, null_count, digest = _SUMMARY_HEAD.unpack_from(body, offset)
            offset += _SUMMARY_HEAD.size
            if name_len == 0 or offset + name_len > len(body):
                raise EngineError(ErrorCode.E_FORMAT, "invalid column summary name length", path=f"{summary_base}.name_len", record_index=record_index)
            try:
                name = body[offset : offset + name_len].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EngineError(ErrorCode.E_FORMAT, "column name is not UTF-8", path=f"{summary_base}.name", record_index=record_index) from exc
            offset += name_len
            try:
                kind = ColumnKind(kind_value)
            except ValueError as exc:
                raise EngineError(ErrorCode.E_FORMAT, "unsupported column kind", path=f"{summary_base}.kind", record_index=record_index) from exc
            if null_count > row_count:
                raise EngineError(ErrorCode.E_FORMAT, "null_count exceeds row_count", path=f"{summary_base}.null_count", record_index=record_index)
            parsed_summaries.append(ColumnSummary(name=name, kind=kind, null_count=null_count, digest=digest))
        summaries = tuple(parsed_summaries)
        if offset + _CHECKSUM.size > len(body):
            raise EngineError(ErrorCode.E_FORMAT, "payload checksum is missing", path=f"{base}.payload_checksum", record_index=record_index)
        payload_checksum = _CHECKSUM.unpack_from(body, offset)[0]
        offset += _CHECKSUM.size

    compressed = False
    overflow_ref: str | None = None
    if version >= 3:
        if offset + _V3_HEAD.size > len(body):
            raise EngineError(ErrorCode.E_FORMAT, "v3 compression/reference fields are missing", path=f"{base}.compressed", record_index=record_index)
        compressed_value, ref_len = _V3_HEAD.unpack_from(body, offset)
        offset += _V3_HEAD.size
        if compressed_value not in (0, 1):
            raise EngineError(ErrorCode.E_FORMAT, "compression flag must be 0 or 1", path=f"{base}.compressed", record_index=record_index)
        compressed = bool(compressed_value)
        if offset + ref_len > len(body):
            raise EngineError(ErrorCode.E_FORMAT, "overflow reference is truncated", path=f"{base}.overflow_ref", record_index=record_index)
        if ref_len:
            try:
                overflow_ref = body[offset : offset + ref_len].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise EngineError(ErrorCode.E_FORMAT, "overflow reference is not UTF-8", path=f"{base}.overflow_ref", record_index=record_index) from exc
            if "/" in overflow_ref or "\\" in overflow_ref:
                raise EngineError(ErrorCode.E_FORMAT, "overflow reference must be a base filename", path=f"{base}.overflow_ref", record_index=record_index)
        offset += ref_len

    if offset != len(body):
        raise EngineError(
            ErrorCode.E_FORMAT,
            "record length is not self-consistent with its fields",
            path=f"{base}.record_length",
            record_index=record_index,
            details={"parsed_end": offset, "record_body_size": len(body)},
        )
    return PageEntry(
        page_id=page_id,
        row_count=row_count,
        record_version=version,
        column_summaries=summaries,
        payload_checksum=payload_checksum,
        compressed=compressed,
        overflow_ref=overflow_ref,
    )
