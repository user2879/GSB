from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import zlib

from .errors import EngineError, ErrorCode
from .model import ColumnKind, ColumnSummary, PageColumn, PageData


PAGE_MAGIC = b"PGDATA01"
PAGE_VERSION = 1
PAGE_TRAILER = b"PGEND001"
_HEADER = struct.Struct("<8sHHIHH")
_TRAILER = struct.Struct("<8sI")
_COLUMN = struct.Struct("<BH")


def page_file_name(page_id: str) -> str:
    _validate_page_id(page_id)
    return f"{page_id}.bin"


def page_path(pages_dir: str | Path, page_id: str) -> Path:
    return Path(pages_dir) / page_file_name(page_id)


def encode_page(page: PageData) -> bytes:
    _validate_page_id(page.page_id)
    if page.row_count <= 0:
        raise EngineError(ErrorCode.E_FORMAT, "row_count must be positive", path="$page.row_count")
    if not page.columns:
        raise EngineError(ErrorCode.E_FORMAT, "page must contain at least one column", path="$page.columns")

    header = _HEADER.pack(
        PAGE_MAGIC,
        PAGE_VERSION,
        len(page.columns),
        page.row_count,
        len(page.page_id.encode("utf-8")),
        0,
    )
    chunks = [header, page.page_id.encode("utf-8")]
    bitmaps: list[bytes] = []
    payloads: list[bytes] = []
    for index, column in enumerate(page.columns):
        _validate_page_id(column.name)
        if column.kind not in (ColumnKind.INT, ColumnKind.FLOAT):
            raise EngineError(
                ErrorCode.E_FORMAT,
                "unsupported column kind",
                path=f"$page.columns[{index}].kind",
            )
        name_bytes = column.name.encode("utf-8")
        if len(name_bytes) > 65535:
            raise EngineError(ErrorCode.E_FORMAT, "column name is too long", path=f"$page.columns[{index}].name")
        chunks.append(_COLUMN.pack(int(column.kind), len(name_bytes)))
        chunks.append(name_bytes)
        if len(column.values) != page.row_count:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "column value count differs from row_count",
                path=f"$page.columns[{index}].values",
            )
        bitmap = _encode_null_bitmap(column.values, index)
        bitmaps.append(bitmap)
        payload = bytearray()
        for row_index, value in enumerate(column.values):
            if value is None:
                payload.extend(b"\x00" * 8)
            elif column.kind is ColumnKind.INT:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise EngineError(
                        ErrorCode.E_FORMAT,
                        "integer column contains a non-integer value",
                        path=f"$page.columns[{index}].values[{row_index}]",
                    )
                payload.extend(struct.pack("<q", value))
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EngineError(
                        ErrorCode.E_FORMAT,
                        "float column contains a non-numeric value",
                        path=f"$page.columns[{index}].values[{row_index}]",
                    )
                number = float(value)
                if not math.isfinite(number):
                    raise EngineError(
                        ErrorCode.E_FORMAT,
                        "float column contains a non-finite value",
                        path=f"$page.columns[{index}].values[{row_index}]",
                    )
                payload.extend(struct.pack("<d", number))
        payloads.append(bytes(payload))

    chunks.extend(bitmaps)
    chunks.extend(payloads)
    body = b"".join(chunks)
    return body + _TRAILER.pack(PAGE_TRAILER, zlib.crc32(body) & 0xFFFFFFFF)


def write_page(path: str | Path, page: PageData) -> bytes:
    data = encode_page(page)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return data


def read_page_file(path: str | Path, *, expected_page_id: str | None = None) -> PageData:
    target = Path(path)
    try:
        data = target.read_bytes()
    except OSError as exc:
        raise EngineError(ErrorCode.E_FORMAT, f"cannot read page file: {exc}", path=str(target)) from exc
    return decode_page_bytes(data, expected_page_id=expected_page_id, source_path=str(target))


def decode_page_bytes(
    data: bytes,
    *,
    expected_page_id: str | None = None,
    source_path: str = "$page",
) -> PageData:
    if len(data) < _HEADER.size + _TRAILER.size:
        raise EngineError(ErrorCode.E_FORMAT, "page payload is truncated", path=source_path)
    try:
        magic, version, column_count, row_count, page_id_len, _reserved = _HEADER.unpack_from(data, 0)
    except struct.error as exc:
        raise EngineError(ErrorCode.E_FORMAT, "invalid page header", path=f"{source_path}.header") from exc
    if magic != PAGE_MAGIC:
        raise EngineError(ErrorCode.E_FORMAT, "invalid page magic", path=f"{source_path}.magic")
    if version != PAGE_VERSION:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "unsupported page format version",
            path=f"{source_path}.version",
            details={"version": version},
        )
    if column_count == 0 or row_count == 0:
        raise EngineError(ErrorCode.E_FORMAT, "empty page is invalid", path=f"{source_path}.header")

    offset = _HEADER.size
    if page_id_len == 0 or offset + page_id_len > len(data) - _TRAILER.size:
        raise EngineError(ErrorCode.E_FORMAT, "invalid page_id length", path=f"{source_path}.page_id_len")
    try:
        page_id = data[offset : offset + page_id_len].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EngineError(ErrorCode.E_FORMAT, "page_id is not valid UTF-8", path=f"{source_path}.page_id") from exc
    offset += page_id_len
    _validate_page_id(page_id)
    if expected_page_id is not None and page_id != expected_page_id:
        raise EngineError(
            ErrorCode.E_PAYLOAD,
            "page payload identifier does not match the expected identifier",
            page_id=expected_page_id,
            details={"payload_page_id": page_id},
        )

    specs: list[tuple[str, ColumnKind]] = []
    for index in range(column_count):
        if offset + _COLUMN.size > len(data) - _TRAILER.size:
            raise EngineError(ErrorCode.E_FORMAT, "column descriptor is truncated", path=f"{source_path}.columns[{index}]")
        kind_value, name_len = _COLUMN.unpack_from(data, offset)
        offset += _COLUMN.size
        if name_len == 0 or offset + name_len > len(data) - _TRAILER.size:
            raise EngineError(ErrorCode.E_FORMAT, "invalid column name length", path=f"{source_path}.columns[{index}].name_len")
        try:
            name = data[offset : offset + name_len].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EngineError(ErrorCode.E_FORMAT, "column name is not UTF-8", path=f"{source_path}.columns[{index}].name") from exc
        offset += name_len
        try:
            kind = ColumnKind(kind_value)
        except ValueError as exc:
            raise EngineError(
                ErrorCode.E_FORMAT,
                "unsupported column kind",
                path=f"{source_path}.columns[{index}].kind",
            ) from exc
        specs.append((name, kind))
    if len({name for name, _kind in specs}) != len(specs):
        raise EngineError(ErrorCode.E_FORMAT, "duplicate column name", path=f"{source_path}.columns")

    bitmap_size = (row_count + 7) // 8
    data_size = row_count * 8
    required_end = offset + column_count * (bitmap_size + data_size) + _TRAILER.size
    if required_end != len(data):
        raise EngineError(
            ErrorCode.E_FORMAT,
            "page length fields are not self-consistent",
            path=f"{source_path}.length",
            details={"expected_size": required_end, "actual_size": len(data)},
        )
    bitmaps: list[bytes] = []
    for index in range(column_count):
        bitmaps.append(data[offset : offset + bitmap_size])
        offset += bitmap_size

    columns: list[PageColumn] = []
    for index, (name, kind) in enumerate(specs):
        bitmap = bitmaps[index]
        values: list[int | float | None] = []
        for row_index in range(row_count):
            raw_value = data[offset : offset + 8]
            offset += 8
            is_null = bool(bitmap[row_index // 8] & (1 << (row_index % 8)))
            if is_null:
                values.append(None)
            elif kind is ColumnKind.INT:
                values.append(struct.unpack("<q", raw_value)[0])
            else:
                value = struct.unpack("<d", raw_value)[0]
                if not math.isfinite(value):
                    raise EngineError(
                        ErrorCode.E_FORMAT,
                        "non-finite float value in page",
                        path=f"{source_path}.columns[{index}].values[{row_index}]",
                    )
                values.append(value)
        columns.append(PageColumn(name=name, kind=kind, values=tuple(values), null_bitmap=bitmap))

    if offset != len(data) - _TRAILER.size:
        raise EngineError(ErrorCode.E_FORMAT, "page body length mismatch", path=f"{source_path}.length")
    trailer_magic, expected_crc = _TRAILER.unpack_from(data, offset)
    if trailer_magic != PAGE_TRAILER:
        raise EngineError(ErrorCode.E_FORMAT, "invalid page trailer", path=f"{source_path}.trailer")
    actual_crc = zlib.crc32(data[:offset]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise EngineError(
            ErrorCode.E_INTEGRITY,
            "page checksum mismatch",
            path=f"{source_path}.checksum",
            details={"expected": expected_crc, "actual": actual_crc},
        )
    return PageData(page_id=page_id, row_count=row_count, columns=tuple(columns), raw_bytes=bytes(data))


def page_checksum(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def column_summary(column: PageColumn) -> ColumnSummary:
    digest = hashlib.sha256()
    digest.update(bytes([int(column.kind)]))
    digest.update(column.name.encode("utf-8"))
    digest.update(column.null_bitmap)
    for value in column.values:
        if value is None:
            digest.update(b"\x00" * 8)
        elif column.kind is ColumnKind.INT:
            digest.update(struct.pack("<q", int(value)))
        else:
            digest.update(struct.pack("<d", float(value)))
    return ColumnSummary(
        name=column.name,
        kind=column.kind,
        null_count=sum(1 for value in column.values if value is None),
        digest=digest.digest()[:16],
    )


def summaries_for_page(page: PageData) -> tuple[ColumnSummary, ...]:
    return tuple(column_summary(column) for column in page.columns)


def _encode_null_bitmap(values: tuple[int | float | None, ...], column_index: int) -> bytes:
    bitmap = bytearray((len(values) + 7) // 8)
    for index, value in enumerate(values):
        if value is None:
            bitmap[index // 8] |= 1 << (index % 8)
    return bytes(bitmap)


def _validate_page_id(page_id: str) -> None:
    if not page_id or page_id in {".", ".."}:
        raise EngineError(ErrorCode.E_FORMAT, "invalid page identifier", path="$page.page_id")
    if "/" in page_id or "\\" in page_id or "\x00" in page_id:
        raise EngineError(ErrorCode.E_FORMAT, "page identifier cannot contain a path separator", path="$page.page_id")
