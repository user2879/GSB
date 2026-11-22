"""Binary fixed-layout columnar page reader/writer.

Layout (little endian):
  magic      8 bytes  b"CPAGE001"
  version    u8
  page_id    i64
  rows       u32
  ncols      u16
  for each column:
    name_len u16, name utf-8
    type     u8 (0=int64, 1=float64)
    values   rows * 8 bytes
    nullmap  rows bytes, each exactly 0 or 1
    crc32    u32 of nullmap

The catalog carries the SHA-256 of the complete page file. Null bitmap faults
are checked before the catalog digest so they receive E_NULLMAP.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import sha256_bytes
from .errors import EPage, ENullMap
from .models import ColumnDef, PageData

MAGIC = b"CPAGE001"
PAGE_FORMAT_VERSION = 1
HEADER = struct.Struct("<8sBqIH")
TYPE_TO_BYTE = {"int": 0, "float": 1}
BYTE_TO_TYPE = {value: key for key, value in TYPE_TO_BYTE.items()}


def encode_page(
    page_id: int,
    rows: int,
    columns: list[ColumnDef],
    values: dict[str, np.ndarray],
    null_maps: dict[str, np.ndarray],
) -> bytes:
    parts = [HEADER.pack(MAGIC, PAGE_FORMAT_VERSION, page_id, rows, len(columns))]
    for column in columns:
        name = column.name.encode("utf-8")
        if len(name) > 65535:
            raise ValueError("column name too long")
        parts.append(struct.pack("<H", len(name)))
        parts.append(name)
        parts.append(struct.pack("<B", TYPE_TO_BYTE[column.column_type]))
        array = values[column.name]
        if column.column_type == "int":
            payload = np.asarray(array, dtype="<i8").tobytes(order="C")
        else:
            payload = np.asarray(array, dtype="<f8").tobytes(order="C")
        if len(payload) != rows * 8:
            raise ValueError(f"column {column.name} length does not equal rows")
        bitmap = np.asarray(null_maps[column.name], dtype=np.uint8).tobytes(order="C")
        if len(bitmap) != rows:
            raise ValueError(f"null map {column.name} length does not equal rows")
        parts.extend(
            [
                payload,
                bitmap,
                struct.pack("<I", zlib.crc32(bitmap) & 0xFFFFFFFF),
            ]
        )
    return b"".join(parts)


def write_page(
    path: str | Path,
    page_id: int,
    rows: int,
    columns: list[ColumnDef],
    values: dict[str, np.ndarray],
    null_maps: dict[str, np.ndarray],
) -> str:
    payload = encode_page(page_id, rows, columns, values, null_maps)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return sha256_bytes(payload)


def _read_exact(data: bytes, offset: int, length: int, field: str, page_id: int | None) -> tuple[bytes, int]:
    if length < 0 or offset + length > len(data):
        raise EPage(
            "page file is truncated",
            page_id=page_id,
            field=field,
            details={"offset": offset, "needed": length, "file_size": len(data)},
        )
    return data[offset : offset + length], offset + length


def decode_page(
    path: str | Path,
    *,
    expected_page_id: int | None = None,
    expected_rows: int | None = None,
    expected_columns: tuple[ColumnDef, ...] | None = None,
) -> tuple[PageData, str]:
    target = Path(path)
    try:
        data = target.read_bytes()
    except FileNotFoundError as exc:
        raise EPage(
            "page file does not exist",
            page_id=expected_page_id,
            field="file",
            details={"path": str(target)},
        ) from exc
    except OSError as exc:
        raise EPage(
            "page file cannot be read",
            page_id=expected_page_id,
            field="file",
            details={"path": str(target), "error": str(exc)},
        ) from exc

    if len(data) < HEADER.size:
        raise EPage(
            "page file is shorter than its header",
            page_id=expected_page_id,
            field="header",
            details={"file_size": len(data), "minimum": HEADER.size},
        )
    magic, version, page_id, rows, ncols = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise EPage(
            "page magic mismatch",
            page_id=expected_page_id,
            field="magic",
            details={"actual": magic.hex()},
        )
    if version != PAGE_FORMAT_VERSION:
        raise EPage(
            "unsupported page format version",
            page_id=page_id,
            field="format_version",
            details={"actual": version, "expected": PAGE_FORMAT_VERSION},
        )
    if expected_page_id is not None and page_id != expected_page_id:
        raise EPage(
            "page id in catalog and page header differ",
            page_id=expected_page_id,
            field="page_id",
            details={"catalog": expected_page_id, "header": page_id},
        )
    if expected_rows is not None and rows != expected_rows:
        raise EPage(
            "page row count does not match catalog",
            page_id=page_id,
            field="rows",
            details={"catalog": expected_rows, "header": rows},
        )
    if ncols == 0 or ncols > 4096:
        raise EPage(
            "invalid page column count",
            page_id=page_id,
            field="ncols",
            details={"ncols": ncols},
        )
    if expected_columns is not None and ncols != len(expected_columns):
        raise EPage(
            "page column count does not match catalog",
            page_id=page_id,
            field="ncols",
            details={"catalog": len(expected_columns), "header": ncols},
        )

    offset = HEADER.size
    values: dict[str, np.ndarray] = {}
    null_maps: dict[str, np.ndarray] = {}
    names: set[str] = set()
    for column_index in range(ncols):
        field_prefix = f"columns[{column_index}]"
        raw_len, offset = _read_exact(data, offset, 2, f"{field_prefix}.name_len", page_id)
        name_len = struct.unpack("<H", raw_len)[0]
        raw_name, offset = _read_exact(data, offset, name_len, f"{field_prefix}.name", page_id)
        try:
            name = raw_name.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EPage(
                "column name is not valid UTF-8",
                page_id=page_id,
                field=f"{field_prefix}.name",
            ) from exc
        if not name or name in names:
            raise EPage(
                "column name is empty or duplicated",
                page_id=page_id,
                field=f"{field_prefix}.name",
                details={"name": name},
            )
        names.add(name)
        raw_type, offset = _read_exact(data, offset, 1, f"{field_prefix}.type", page_id)
        type_byte = raw_type[0]
        if type_byte not in BYTE_TO_TYPE:
            raise EPage(
                "unsupported column type",
                page_id=page_id,
                field=f"columns[{name}].type",
                details={"type_byte": type_byte},
            )
        column_type = BYTE_TO_TYPE[type_byte]
        if expected_columns is not None:
            expected = expected_columns[column_index]
            if name != expected.name or column_type != expected.column_type:
                raise EPage(
                    "page column definition does not match catalog",
                    page_id=page_id,
                    field=f"columns[{column_index}]",
                    details={
                        "catalog": expected.payload(),
                        "header": {"name": name, "type": column_type},
                    },
                )
        raw_values, offset = _read_exact(data, offset, rows * 8, f"columns[{name}].values", page_id)
        if column_type == "int":
            array = np.frombuffer(raw_values, dtype="<i8", count=rows).copy()
        else:
            array = np.frombuffer(raw_values, dtype="<f8", count=rows).copy()
            if not bool(np.isfinite(array).all()):
                bad_index = int(np.flatnonzero(~np.isfinite(array))[0])
                raise EPage(
                    "float column contains a non-finite value",
                    page_id=page_id,
                    field=f"columns[{name}].values",
                    details={"row_index": bad_index, "value": str(array[bad_index])},
                )
        if offset + rows > len(data):
            raise ENullMap(
                "null bitmap length does not equal page rows",
                page_id=page_id,
                field=f"columns[{name}].null_bitmap",
                details={"expected": rows, "actual": max(0, len(data) - offset)},
            )
        raw_bitmap = data[offset : offset + rows]
        offset += rows
        bitmap = np.frombuffer(raw_bitmap, dtype=np.uint8, count=rows).copy()
        if not bool(np.isin(bitmap, (0, 1)).all()):
            bad_index = int(np.flatnonzero(~np.isin(bitmap, (0, 1)))[0])
            raise ENullMap(
                "null bitmap contains values other than 0/1",
                page_id=page_id,
                field=f"columns[{name}].null_bitmap",
                details={"row_index": bad_index, "value": int(bitmap[bad_index])},
            )
        if offset + 4 > len(data):
            raise ENullMap(
                "null bitmap checksum is missing or truncated",
                page_id=page_id,
                field=f"columns[{name}].null_bitmap_crc32",
                details={"expected_bytes": 4, "remaining_bytes": max(0, len(data) - offset)},
            )
        raw_crc = data[offset : offset + 4]
        offset += 4
        stored_crc = struct.unpack("<I", raw_crc)[0]
        computed_crc = zlib.crc32(raw_bitmap) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise ENullMap(
                "null bitmap checksum mismatch",
                page_id=page_id,
                field=f"columns[{name}].null_bitmap_crc32",
                details={
                    "stored": stored_crc,
                    "computed": computed_crc,
                    "interpretation": "null bitmap is truncated, shifted, or misaligned",
                },
            )
        values[name] = array
        null_maps[name] = bitmap

    if offset != len(data):
        raise EPage(
            "page file has trailing bytes",
            page_id=page_id,
            field="file_length",
            details={"expected": offset, "actual": len(data)},
        )
    content_sha256 = sha256_bytes(data)
    page = PageData(
        page_id=page_id,
        rows=rows,
        path=target,
        content_sha256=content_sha256,
        columns=values,
        null_maps=null_maps,
    )
    return page, content_sha256


def read_page_header(path: str | Path) -> dict[str, Any]:
    """Read only the fixed header, useful for diagnostics and tests."""
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        raise EPage("page file is shorter than its header", field="header")
    magic, version, page_id, rows, ncols = HEADER.unpack_from(data, 0)
    return {
        "magic": magic.decode("ascii", errors="replace"),
        "format_version": version,
        "page_id": page_id,
        "rows": rows,
        "ncols": ncols,
    }
