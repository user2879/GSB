"""Deterministic local columnar page format.

Layout (all integers little endian)::

    MAGIC (9 bytes: b"COLMVPG1\n")
    uint32 header_length
    UTF-8 canonical JSON header
    payload blocks referenced by header offsets

Each column owns a one-byte-per-row null bitmap (1 means NULL, 0 means
present) followed by fixed-width little-endian data.  The one-byte bitmap is
deliberately simple so the declared bitmap length is exactly the page row
count, as required by the data contract.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Sequence

import numpy as np

from .canonical import canonical_json, sha256_hex
from .errors import ColmvError


MAGIC = b"COLMVPG1\n"
FORMAT_VERSION = 1
DTYPES = {"int": "<i8", "float": "<f8"}


@dataclass
class PageData:
    page_id: str
    start_row: int
    end_row: int
    columns: list[dict[str, Any]]
    values: dict[str, list[Any]]
    nulls: dict[str, list[bool]]
    content_digest: str = ""

    @property
    def row_count(self) -> int:
        return self.end_row - self.start_row

    def rows(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for offset in range(self.row_count):
            row: dict[str, Any] = {}
            for column in self.columns:
                name = column["name"]
                row[name] = None if self.nulls[name][offset] else self.values[name][offset]
            result.append(row)
        return result


def _encode_column_values(column: dict[str, Any], values: Sequence[Any]) -> bytes:
    name = column["name"]
    column_type = column["type"]
    dtype = DTYPES[column_type]
    converted: list[int | float] = []
    for index, value in enumerate(values):
        if value is None:
            converted.append(0)
            continue
        if column_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ColmvError(
                    "E_VIEW",
                    f"integer column {name!r} has non-integer value at page row {index}",
                    field=name,
                )
            if not -(2**63) <= value <= 2**63 - 1:
                raise ColmvError(
                    "E_VIEW",
                    f"integer column {name!r} value exceeds int64 at page row {index}",
                    field=name,
                )
            converted.append(value)
        else:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ColmvError(
                    "E_VIEW",
                    f"float column {name!r} has non-numeric value at page row {index}",
                    field=name,
                )
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ColmvError(
                    "E_VIEW",
                    f"float column {name!r} has non-finite value at page row {index}",
                    field=name,
                )
            converted.append(numeric)
    return np.asarray(converted, dtype=dtype).tobytes(order="C")


def encode_page(
    *,
    page_id: str,
    start_row: int,
    end_row: int,
    columns: Sequence[dict[str, Any]],
    values: dict[str, Sequence[Any]],
) -> bytes:
    row_count = end_row - start_row
    if row_count < 0:
        raise ValueError("end_row must be >= start_row")

    blocks: list[tuple[str, str, bytes]] = []
    for column in columns:
        name = column["name"]
        column_type = column["type"]
        if column_type not in DTYPES:
            raise ValueError(f"unsupported column type: {column_type}")
        column_values = list(values[name])
        if len(column_values) != row_count:
            raise ValueError(f"column {name!r} has {len(column_values)} values, expected {row_count}")
        null_map = bytes(1 if value is None else 0 for value in column_values)
        data = _encode_column_values(column, column_values)
        blocks.append((name, "null", null_map))
        blocks.append((name, "data", data))

    header_columns: list[dict[str, Any]] = []
    cursor = 0
    for index, column in enumerate(columns):
        name = column["name"]
        null_map = blocks[index * 2][2]
        data = blocks[index * 2 + 1][2]
        header_columns.append(
            {
                "name": name,
                "type": column["type"],
                "dtype": DTYPES[column["type"]],
                "null_bitmap_offset": cursor,
                "null_bitmap_length": len(null_map),
                "data_offset": cursor + len(null_map),
                "data_length": len(data),
                "data_sha256": sha256_hex(data),
            }
        )
        cursor += len(null_map) + len(data)

    header = {
        "format_version": FORMAT_VERSION,
        "page_id": page_id,
        "start_row": start_row,
        "end_row": end_row,
        "row_count": row_count,
        "columns": header_columns,
        "payload_length": cursor,
    }
    header_bytes = canonical_json(header).encode("utf-8")
    payload = b"".join(block[2] for block in blocks)
    return MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload


def write_page_file(
    path: Path,
    *,
    page_id: str,
    start_row: int,
    end_row: int,
    columns: Sequence[dict[str, Any]],
    values: dict[str, Sequence[Any]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = encode_page(
        page_id=page_id,
        start_row=start_row,
        end_row=end_row,
        columns=columns,
        values=values,
    )
    path.write_bytes(content)
    return sha256_hex(content)


def _integrity(message: str, *, page_id: str | None, path: Path | None, **details: Any) -> NoReturn:
    raise ColmvError(
        "E_INTEGRITY",
        message,
        field=page_id,
        details={"page_id": page_id, "path": str(path) if path else None, **details},
    )


def _semantic_error(
    *,
    semantic_view_id: str | None,
    message: str,
    field: str,
    page_id: str | None,
    path: Path | None,
    **details: Any,
) -> NoReturn:
    if semantic_view_id is not None:
        raise ColmvError(
            "E_VIEW",
            message,
            view_id=semantic_view_id,
            field=field,
            details={"page_id": page_id, "path": str(path) if path else None, **details},
        )
    _integrity(message, page_id=page_id, path=path, **details)


def read_page_file(
    path: Path,
    *,
    expected_page_id: str | None = None,
    expected_start_row: int | None = None,
    expected_end_row: int | None = None,
    inject_truncate: bool = False,
    semantic_view_id: str | None = None,
) -> PageData:
    try:
        content = path.read_bytes()
    except OSError as exc:
        _integrity(f"cannot read columnar page file: {exc}", page_id=expected_page_id, path=path)

    original_length = len(content)
    if inject_truncate:
        content = content[: max(0, len(content) // 2)]

    if len(content) < len(MAGIC) + 4:
        _integrity(
            "columnar page is truncated before its header",
            page_id=expected_page_id,
            path=path,
            expected_min_bytes=len(MAGIC) + 4,
            actual_bytes=len(content),
            original_bytes=original_length,
        )
    if content[: len(MAGIC)] != MAGIC:
        _integrity("columnar page magic mismatch", page_id=expected_page_id, path=path)
    try:
        header_length = struct.unpack("<I", content[len(MAGIC) : len(MAGIC) + 4])[0]
    except struct.error as exc:
        _integrity(f"cannot decode columnar page header length: {exc}", page_id=expected_page_id, path=path)
    header_start = len(MAGIC) + 4
    header_end = header_start + header_length
    if header_end > len(content):
        _integrity(
            "columnar page is truncated inside its header",
            page_id=expected_page_id,
            path=path,
            expected_header_bytes=header_length,
            available_header_bytes=max(0, len(content) - header_start),
        )
    try:
        header = json.loads(content[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _integrity(f"columnar page header JSON is invalid: {exc}", page_id=expected_page_id, path=path)

    if not isinstance(header, dict):
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header must be a JSON object",
            field="base.page.format_version",
            page_id=expected_page_id,
            path=path,
        )
    if header.get("format_version") != FORMAT_VERSION or isinstance(header.get("format_version"), bool):
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page format_version is unsupported",
            field="base.page.format_version",
            page_id=expected_page_id,
            path=path,
            expected=FORMAT_VERSION,
            actual=header.get("format_version"),
        )

    page_id = header.get("page_id")
    if expected_page_id is not None and page_id != expected_page_id:
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header page_id does not match the page directory",
            field=f"base.pages.{expected_page_id}.page_id",
            page_id=expected_page_id,
            path=path,
            header_page_id=page_id,
        )
    if expected_start_row is not None and header.get("start_row") != expected_start_row:
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header start_row does not match the page directory",
            field=f"base.pages.{page_id}.start_row",
            page_id=page_id,
            path=path,
            expected_start_row=expected_start_row,
            actual_start_row=header.get("start_row"),
        )
    if expected_end_row is not None and header.get("end_row") != expected_end_row:
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header end_row does not match the page directory",
            field=f"base.pages.{page_id}.end_row",
            page_id=page_id,
            path=path,
            expected_end_row=expected_end_row,
            actual_end_row=header.get("end_row"),
        )

    header_start_row = header.get("start_row")
    header_end_row = header.get("end_row")
    if (
        not isinstance(header_start_row, int)
        or isinstance(header_start_row, bool)
        or not isinstance(header_end_row, int)
        or isinstance(header_end_row, bool)
    ):
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header start_row/end_row must be integers",
            field=f"base.pages.{page_id}.start_row",
            page_id=page_id,
            path=path,
        )

    row_count = header.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0:
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header has invalid row_count",
            field=f"base.pages.{page_id}.row_count",
            page_id=page_id,
            path=path,
        )
    if header_end_row - header_start_row != row_count:
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header row interval is inconsistent",
            field=f"base.pages.{page_id}.row_count",
            page_id=page_id,
            path=path,
        )
    payload_length = header.get("payload_length")
    if not isinstance(payload_length, int) or isinstance(payload_length, bool) or payload_length < 0:
        _integrity("columnar page header has invalid payload_length", page_id=page_id, path=path)
    expected_total = header_end + payload_length
    if len(content) != expected_total:
        _integrity(
            "columnar page length does not match its header",
            page_id=page_id,
            path=path,
            expected_total_bytes=expected_total,
            actual_bytes=len(content),
            original_bytes=original_length,
            injected_truncation=inject_truncate,
        )

    payload = content[header_end:]
    values: dict[str, list[Any]] = {}
    nulls: dict[str, list[bool]] = {}
    payload_intervals: list[tuple[int, int, str]] = []
    columns = header.get("columns")
    if not isinstance(columns, list):
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page header columns must be a list",
            field=f"base.pages.{page_id}.columns",
            page_id=page_id,
            path=path,
        )

    for column in columns:
        if not isinstance(column, dict):
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="columnar page column descriptor is not an object",
                field=f"base.pages.{page_id}.columns",
                page_id=page_id,
                path=path,
            )
        name = column.get("name")
        column_type = column.get("type")
        if not isinstance(name, str) or column_type not in DTYPES:
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="columnar page column descriptor has invalid name/type",
                field=f"base.pages.{page_id}.columns",
                page_id=page_id,
                path=path,
            )
        null_offset = column.get("null_bitmap_offset")
        null_length = column.get("null_bitmap_length")
        data_offset = column.get("data_offset")
        data_length = column.get("data_length")
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0
            for item in (null_offset, null_length, data_offset, data_length)
        ):
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="columnar page column offsets must be non-negative integers",
                field=f"base.pages.{page_id}.null_bitmap.{name}",
                page_id=page_id,
                path=path,
            )
        if null_length != row_count:
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="null bitmap length is not equal to page row_count",
                field=f"base.pages.{page_id}.null_bitmap.{name}",
                page_id=page_id,
                path=path,
                column=name,
                expected=row_count,
                actual=null_length,
            )
        expected_data_length = row_count * 8
        if data_length != expected_data_length:
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="column value block length is not 8 bytes per row",
                field=f"base.pages.{page_id}.columns.{name}",
                page_id=page_id,
                path=path,
                column=name,
                expected=expected_data_length,
                actual=data_length,
            )
        if null_offset + null_length > len(payload) or data_offset + data_length > len(payload):
            _integrity("columnar page column block exceeds payload", page_id=page_id, path=path)
        payload_intervals.append((null_offset, null_length, f"{name}.null_bitmap"))
        payload_intervals.append((data_offset, data_length, f"{name}.data"))
        null_block = payload[null_offset : null_offset + null_length]
        if any(item not in (0, 1) for item in null_block):
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="null bitmap contains a byte other than 0 or 1",
                field=f"base.pages.{page_id}.null_bitmap.{name}",
                page_id=page_id,
                path=path,
                column=name,
            )
        data_block = payload[data_offset : data_offset + data_length]
        digest = column.get("data_sha256")
        if digest is not None and digest != sha256_hex(data_block):
            _integrity("column data digest mismatch", page_id=page_id, path=path, column=name)
        dtype = DTYPES[column_type]
        array = np.frombuffer(data_block, dtype=dtype)
        if column_type == "float" and not bool(np.all(np.isfinite(array))):
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="float column contains non-finite value",
                field=f"base.pages.{page_id}.columns.{name}",
                page_id=page_id,
                path=path,
                column=name,
            )
        column_values: list[Any] = []
        column_nulls: list[bool] = []
        for offset in range(row_count):
            is_null = bool(null_block[offset])
            column_nulls.append(is_null)
            if is_null:
                column_values.append(None)
            elif column_type == "int":
                column_values.append(int(array[offset]))
            else:
                column_values.append(float(array[offset]))
        values[name] = column_values
        nulls[name] = column_nulls

    cursor = 0
    for offset, length, block_name in sorted(payload_intervals):
        if offset != cursor:
            _semantic_error(
                semantic_view_id=semantic_view_id,
                message="columnar page payload blocks must be contiguous and non-overlapping",
                field=f"base.pages.{page_id}.columns",
                page_id=page_id,
                path=path,
                block=block_name,
                expected_offset=cursor,
                actual_offset=offset,
            )
        cursor += length
    if cursor != payload_length:
        _semantic_error(
            semantic_view_id=semantic_view_id,
            message="columnar page payload block lengths do not cover payload_length",
            field=f"base.pages.{page_id}.columns",
            page_id=page_id,
            path=path,
            expected=payload_length,
            actual=cursor,
        )

    return PageData(
        page_id=page_id,
        start_row=header["start_row"],
        end_row=header["end_row"],
        columns=columns,
        values=values,
        nulls=nulls,
        content_digest=sha256_hex(content),
    )


def validate_page_columns(
    page: PageData,
    expected_columns: Sequence[dict[str, str]],
    *,
    view_id: str,
    page_id: str,
) -> None:
    expected = [(column["name"], column["type"]) for column in expected_columns]
    actual = [(column.get("name"), column.get("type")) for column in page.columns]
    if expected != actual:
        raise ColmvError(
            "E_VIEW",
            f"page {page_id!r} column layout does not match the catalog",
            view_id=view_id,
            field="base.pages",
            details={"expected": expected, "actual": actual},
        )
