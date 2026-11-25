from __future__ import annotations

import bisect
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .canonical import INT64_MAX, INT64_MIN, atomic_write


MAGIC = b"VSC1"
FORMAT_VERSION = 1
ENCODING_ID = {"raw": 0, "dict": 1, "rle": 2}
ID_ENCODING = {value: key for key, value in ENCODING_ID.items()}
HEADER = struct.Struct("<4sHBBQQQ")
TRAILER_SIZE = 32
DECODED_CELL_BYTES = 9  # signed int64 value plus one validity byte per row


class CodecError(ValueError):
    pass


@dataclass(frozen=True)
class ColumnData:
    values: np.ndarray
    null_mask: np.ndarray

    def __post_init__(self) -> None:
        if self.values.dtype != np.dtype("<i8"):
            object.__setattr__(self, "values", self.values.astype("<i8", copy=False))
        if self.null_mask.dtype != np.dtype("?"):
            object.__setattr__(self, "null_mask", self.null_mask.astype("?", copy=False))
        if self.values.ndim != 1 or self.null_mask.ndim != 1:
            raise CodecError("column arrays must be one-dimensional")
        if self.values.shape != self.null_mask.shape:
            raise CodecError("column values and null mask must have the same length")

    @property
    def rows(self) -> int:
        return int(self.values.size)

    @property
    def non_null_count(self) -> int:
        return int(self.rows - np.count_nonzero(self.null_mask))


@dataclass(frozen=True)
class ColumnFileInfo:
    encoding: str
    rows: int
    non_null_count: int
    payload_bytes: int
    file_bytes: int
    file_digest: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "encoding": self.encoding,
            "rows": self.rows,
            "non_null_count": self.non_null_count,
            "payload_bytes": self.payload_bytes,
            "file_bytes": self.file_bytes,
            "file_digest": self.file_digest,
        }


def _values_array(values: Iterable[int | None]) -> ColumnData:
    materialized = list(values)
    null_mask = np.fromiter((item is None for item in materialized), dtype="?", count=len(materialized))
    base = np.fromiter(
        (INT64_MIN if item is None else int(item) for item in materialized),
        dtype="<i8",
        count=len(materialized),
    )
    if np.any((base[~null_mask] < INT64_MIN) | (base[~null_mask] > INT64_MAX)):
        raise CodecError("column values must fit signed int64")
    base[null_mask] = 0
    return ColumnData(base, null_mask)


def _pack_raw(data: ColumnData) -> bytes:
    bitmap = np.packbits(~data.null_mask, bitorder="little").tobytes()
    return data.values.astype("<i8", copy=False).tobytes(order="C") + bitmap


def _unpack_raw(payload: bytes, rows: int) -> ColumnData:
    bitmap_bytes = (rows + 7) // 8
    expected = rows * 8 + bitmap_bytes
    if len(payload) != expected:
        raise CodecError(f"raw payload length {len(payload)} != {expected}")
    values = np.frombuffer(payload, dtype="<i8", count=rows).copy()
    bitmap = np.frombuffer(payload, dtype=np.uint8, offset=rows * 8, count=bitmap_bytes)
    valid = np.unpackbits(bitmap, bitorder="little", count=rows).astype("?")
    null_mask = ~valid
    values[null_mask] = 0
    return ColumnData(values, null_mask)


def _code_width(dictionary_size: int) -> int:
    if dictionary_size <= 0xFE:
        return 1
    if dictionary_size <= 0xFFFE:
        return 2
    if dictionary_size <= 0xFFFFFFFE:
        return 4
    return 8


def _pack_dict(data: ColumnData) -> bytes:
    valid_values = data.values[~data.null_mask]
    dictionary = np.unique(valid_values.astype("<i8", copy=False))
    dictionary = np.sort(dictionary, kind="stable")
    width = _code_width(int(dictionary.size))
    if dictionary.size:
        positions = np.searchsorted(dictionary, valid_values)
    else:
        positions = np.empty(0, dtype=np.uint64)
    null_code = (1 << (width * 8)) - 1
    codes = np.full(data.rows, null_code, dtype=f"<u{width}")
    if dictionary.size:
        codes[~data.null_mask] = positions.astype(f"<u{width}")
    return (
        struct.pack("<IB3x", int(dictionary.size), width)
        + dictionary.astype("<i8", copy=False).tobytes()
        + codes.tobytes()
    )


def _unpack_dict(payload: bytes, rows: int) -> ColumnData:
    if len(payload) < 8:
        raise CodecError("dictionary payload is truncated")
    dictionary_size, width = struct.unpack_from("<IB", payload, 0)
    if width not in (1, 2, 4, 8):
        raise CodecError(f"invalid dictionary code width {width}")
    expected = 8 + dictionary_size * 8 + rows * width
    if len(payload) != expected:
        raise CodecError(f"dictionary payload length {len(payload)} != {expected}")
    dictionary = np.frombuffer(
        payload, dtype="<i8", count=dictionary_size, offset=8
    ).copy()
    if dictionary_size and np.any(dictionary[1:] <= dictionary[:-1]):
        raise CodecError("dictionary values must be strictly ascending")
    codes = np.frombuffer(
        payload,
        dtype=f"<u{width}",
        count=rows,
        offset=8 + dictionary_size * 8,
    ).copy()
    null_code = (1 << (width * 8)) - 1
    null_mask = codes == null_code
    values = np.zeros(rows, dtype="<i8")
    if dictionary_size:
        nonnull_codes = codes[~null_mask].astype(np.int64)
        if np.any(nonnull_codes >= dictionary_size):
            raise CodecError("dictionary code is outside the dictionary")
        values[~null_mask] = dictionary[nonnull_codes]
    elif np.any(~null_mask):
        raise CodecError("empty dictionary cannot have non-null codes")
    return ColumnData(values, null_mask)


def _pack_rle(data: ColumnData) -> bytes:
    runs: list[tuple[int, int, int]] = []
    index = 0
    while index < data.rows:
        is_null = bool(data.null_mask[index])
        value = 0 if is_null else int(data.values[index])
        end = index + 1
        while end < data.rows:
            next_null = bool(data.null_mask[end])
            next_value = 0 if next_null else int(data.values[end])
            if next_null != is_null or next_value != value:
                break
            end += 1
        runs.append((value, end - index, 1 if is_null else 0))
        index = end
    chunks = [struct.pack("<Q", len(runs))]
    for value, length, is_null in runs:
        chunks.append(struct.pack("<qQB", value, length, is_null))
    return b"".join(chunks)


def _unpack_rle(payload: bytes, rows: int) -> ColumnData:
    if len(payload) < 8:
        raise CodecError("RLE payload is truncated")
    run_count = struct.unpack_from("<Q", payload, 0)[0]
    expected = 8 + run_count * 17
    if len(payload) != expected:
        raise CodecError(f"RLE payload length {len(payload)} != {expected}")
    values = np.zeros(rows, dtype="<i8")
    null_mask = np.zeros(rows, dtype="?")
    cursor = 0
    for run_index in range(run_count):
        value, length, is_null = struct.unpack_from("<qQB", payload, 8 + run_index * 17)
        if length <= 0 or cursor + length > rows:
            raise CodecError("invalid RLE run length")
        if is_null not in (0, 1):
            raise CodecError("invalid RLE null flag")
        if is_null:
            null_mask[cursor : cursor + length] = True
        else:
            values[cursor : cursor + length] = value
        cursor += length
    if cursor != rows:
        raise CodecError(f"RLE runs cover {cursor} rows, expected {rows}")
    return ColumnData(values, null_mask)


def encode_column(values: Iterable[int | None], encoding: str) -> bytes:
    if encoding not in ENCODING_ID:
        raise CodecError(f"unsupported encoding {encoding!r}")
    data = _values_array(values)
    if encoding == "raw":
        payload = _pack_raw(data)
    elif encoding == "dict":
        payload = _pack_dict(data)
    else:
        payload = _pack_rle(data)
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        ENCODING_ID[encoding],
        0,
        data.rows,
        data.non_null_count,
        len(payload),
    )
    body = header + payload
    return body + hashlib.sha256(body).digest()


def _validate_blob(blob: bytes) -> tuple[str, int, int, bytes]:
    if len(blob) < HEADER.size + TRAILER_SIZE:
        raise CodecError("column file is too short")
    body, stored_digest = blob[:-TRAILER_SIZE], blob[-TRAILER_SIZE:]
    actual_digest = hashlib.sha256(body).digest()
    if actual_digest != stored_digest:
        raise CodecError("column file SHA-256 mismatch")
    magic, version, encoding_id, flags, rows, non_null_count, payload_len = HEADER.unpack_from(body)
    if magic != MAGIC:
        raise CodecError("invalid column file magic")
    if version != FORMAT_VERSION:
        raise CodecError(f"unsupported column format version {version}")
    if flags != 0:
        raise CodecError("nonzero reserved flags")
    if encoding_id not in ID_ENCODING:
        raise CodecError(f"unknown encoding id {encoding_id}")
    payload = body[HEADER.size:]
    if len(payload) != payload_len:
        raise CodecError("declared payload length does not match file")
    encoding = ID_ENCODING[encoding_id]
    return encoding, rows, non_null_count, payload


def decode_column(blob: bytes) -> tuple[ColumnData, str]:
    encoding, rows, non_null_count, payload = _validate_blob(blob)
    if encoding == "raw":
        data = _unpack_raw(payload, rows)
    elif encoding == "dict":
        data = _unpack_dict(payload, rows)
    else:
        data = _unpack_rle(payload, rows)
    if data.non_null_count != non_null_count:
        raise CodecError("declared non-null count does not match payload")
    return data, encoding


def write_column(path: str | Path, values: Iterable[int | None], encoding: str) -> ColumnFileInfo:
    blob = encode_column(values, encoding)
    atomic_write(path, blob)
    data, actual_encoding = decode_column(blob)
    return ColumnFileInfo(
        encoding=actual_encoding,
        rows=data.rows,
        non_null_count=data.non_null_count,
        payload_bytes=len(blob) - HEADER.size - TRAILER_SIZE,
        file_bytes=len(blob),
        file_digest=hashlib.sha256(blob).hexdigest(),
    )


def file_info(path: str | Path) -> ColumnFileInfo:
    source = Path(path)
    blob = source.read_bytes()
    data, encoding = decode_column(blob)
    return ColumnFileInfo(
        encoding=encoding,
        rows=data.rows,
        non_null_count=data.non_null_count,
        payload_bytes=len(blob) - HEADER.size - TRAILER_SIZE,
        file_bytes=len(blob),
        file_digest=hashlib.sha256(blob).hexdigest(),
    )


def read_column(path: str | Path) -> ColumnData:
    return decode_column(Path(path).read_bytes())[0]


def read_rows(path: str | Path, row_indices: np.ndarray | Iterable[int]) -> ColumnData:
    blob = Path(path).read_bytes()
    encoding, rows, non_null_count, payload = _validate_blob(blob)
    indices = np.asarray(
        list(row_indices) if not isinstance(row_indices, np.ndarray) else row_indices,
        dtype=np.int64,
    )
    if indices.ndim != 1:
        raise CodecError("row indices must be one-dimensional")
    if indices.size and (indices.min() < 0 or indices.max() >= rows):
        raise CodecError("row index outside column")
    if encoding == "raw":
        bitmap_bytes = (rows + 7) // 8
        all_values = np.frombuffer(payload, dtype="<i8", count=rows)
        bitmap = np.frombuffer(
            payload, dtype=np.uint8, offset=rows * 8, count=bitmap_bytes
        )
        values = all_values[indices].copy()
        valid = (((bitmap[indices >> 3] >> (indices & 7)) & 1) != 0)
        null_mask = ~valid
        values[null_mask] = 0
        return ColumnData(values, null_mask)
    if encoding == "dict":
        dictionary_size, width = struct.unpack_from("<IB", payload, 0)
        dictionary = np.frombuffer(payload, dtype="<i8", count=dictionary_size, offset=8)
        codes_offset = 8 + dictionary_size * 8
        all_codes = np.frombuffer(
            payload, dtype=f"<u{width}", count=rows, offset=codes_offset
        )
        codes = all_codes[indices].copy()
        null_code = (1 << (width * 8)) - 1
        null_mask = codes == np.array(null_code, dtype=f"<u{width}")
        values = np.zeros(indices.size, dtype="<i8")
        if dictionary_size:
            nonnull_codes = codes[~null_mask].astype(np.int64)
            if np.any(nonnull_codes >= dictionary_size):
                raise CodecError("dictionary code is outside the dictionary")
            values[~null_mask] = dictionary[nonnull_codes]
        elif np.any(~null_mask):
            raise CodecError("empty dictionary cannot have non-null codes")
        return ColumnData(values, null_mask)

    run_count = struct.unpack_from("<Q", payload, 0)[0]
    run_values = np.empty(run_count, dtype="<i8")
    run_nulls = np.empty(run_count, dtype="?")
    run_lengths = np.empty(run_count, dtype=np.uint64)
    cursor = 0
    for run_index in range(run_count):
        value, length, is_null = struct.unpack_from("<qQB", payload, 8 + run_index * 17)
        if length <= 0 or cursor + length > rows:
            raise CodecError("invalid RLE run length")
        if is_null not in (0, 1):
            raise CodecError("invalid RLE null flag")
        run_values[run_index] = value
        run_lengths[run_index] = length
        run_nulls[run_index] = bool(is_null)
        cursor += length
    if cursor != rows:
        raise CodecError(f"RLE runs cover {cursor} rows, expected {rows}")
    run_ends = np.cumsum(run_lengths, dtype=np.uint64)
    run_indices = np.searchsorted(run_ends, indices.astype(np.uint64), side="right")
    if indices.size and int(run_indices.max()) >= run_count:
        raise CodecError("RLE lookup escaped the run table")
    values = run_values[run_indices].copy()
    null_mask = run_nulls[run_indices].copy()
    values[null_mask] = 0
    return ColumnData(values, null_mask)


def encoded_decoded_bytes(rows: int, columns: int) -> int:
    return int(rows) * int(columns) * DECODED_CELL_BYTES


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
