"""Deterministic generators and columnar block-group codecs.

The on-disk GSB1 layout is:

* ``GSBBLK1`` magic and one format byte;
* big-endian uint32 canonical-JSON header length;
* UTF-8 JSON header;
* column payloads at the offsets recorded in the header.

Integers are signed 64-bit.  Raw columns begin with a little-endian bit map
(1 means non-null), RLE entries are ``int64 value + uint32 count + uint8 null``,
and dictionary columns use sorted distinct values plus uint32 indices with
``0xffffffff`` marking NULL.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable

from .hashing import canonical_bytes, canonical_hash, sha256_hex


BLOCK_MAGIC = b"GSBBLK1"
BLOCK_FORMAT_BYTE = b"\x00"
NULL_INDEX = 0xFFFFFFFF


def _stable_unit(seed: int, row_index: int, salt: str = "row") -> int:
    material = f"gsb-generator/v1|{salt}|seed={int(seed)}|row={int(row_index)}".encode()
    digest = sha256_hex(material)
    return int(digest[:16], 16)


def generate_column_values(column: dict[str, Any], row_count: int) -> list[int | None]:
    """Generate stable integer values from a fixed integer seed.

    The counter-based hash is part of the public deterministic rule.  A row is
    NULL when ``null_every > 0`` and ``(row_index + 1) % null_every == 0``.
    """

    seed = int(column.get("seed", 0))
    modulus = int(column.get("modulus") or 1000)
    offset = int(column.get("offset", 0))
    null_every = int(column.get("null_every", 0) or 0)
    values: list[int | None] = []
    for row_index in range(int(row_count)):
        if null_every and (row_index + 1) % null_every == 0:
            values.append(None)
            continue
        unit = _stable_unit(seed, row_index, str(column.get("name", "value")))
        values.append(offset + (unit % modulus))
    return values


def generate_block_columns(block_group: dict[str, Any]) -> dict[str, list[int | None]]:
    row_count = int(block_group["row_count"])
    return {
        column["name"]: generate_column_values(column, row_count)
        for column in sorted(block_group["columns"], key=lambda c: c["name"])
    }


def actual_index_stats(values: Iterable[int | None]) -> dict[str, Any]:
    materialized = list(values)
    non_null = [value for value in materialized if value is not None]
    return {
        "min": min(non_null) if non_null else None,
        "max": max(non_null) if non_null else None,
        "null_count": len(materialized) - len(non_null),
    }


def _bitmap(values: list[int | None]) -> bytes:
    size = (len(values) + 7) // 8
    mask = 0
    for index, value in enumerate(values):
        if value is not None:
            mask |= 1 << index
    return mask.to_bytes(size, "little")


def _decode_bitmap(data: bytes, row_count: int) -> list[bool]:
    size = (row_count + 7) // 8
    mask = int.from_bytes(data[:size], "little") if size else 0
    return [bool((mask >> index) & 1) for index in range(row_count)]


def encode_raw(values: list[int | None]) -> bytes:
    out = bytearray(_bitmap(values))
    for value in values:
        out.extend(struct.pack(">q", 0 if value is None else int(value)))
    return bytes(out)


def decode_raw(data: bytes, row_count: int) -> list[int | None]:
    bitmap_size = (row_count + 7) // 8
    non_null = _decode_bitmap(data, row_count)
    values: list[int | None] = []
    for index in range(row_count):
        start = bitmap_size + index * 8
        value = struct.unpack_from(">q", data, start)[0]
        values.append(value if non_null[index] else None)
    return values


def encode_rle(values: list[int | None]) -> bytes:
    runs: list[tuple[int, int, int]] = []
    for value in values:
        null_flag = 1 if value is None else 0
        encoded_value = 0 if value is None else int(value)
        if runs and runs[-1][0] == encoded_value and runs[-1][2] == null_flag:
            runs[-1] = (encoded_value, runs[-1][1] + 1, null_flag)
        else:
            runs.append((encoded_value, 1, null_flag))
    out = bytearray(struct.pack(">I", len(runs)))
    for value, count, null_flag in runs:
        out.extend(struct.pack(">qI B", value, count, null_flag))
    return bytes(out)


def decode_rle(data: bytes, row_count: int) -> list[int | None]:
    run_count = struct.unpack_from(">I", data, 0)[0]
    values: list[int | None] = []
    offset = 4
    for _ in range(run_count):
        value, count, null_flag = struct.unpack_from(">qIB", data, offset)
        offset += 13
        repeated = None if null_flag else value
        values.extend([repeated] * count)
    if len(values) != row_count:
        raise ValueError(f"RLE row count mismatch: decoded {len(values)}, header {row_count}")
    return values


def _dictionary_key(value: int) -> tuple[str, int | str]:
    return ("int", int(value))


def encode_dict(values: list[int | None]) -> bytes:
    dictionary = sorted({value for value in values if value is not None}, key=_dictionary_key)
    positions = {value: index for index, value in enumerate(dictionary)}
    out = bytearray(struct.pack(">I", len(dictionary)))
    for value in dictionary:
        out.extend(struct.pack(">q", int(value)))
    for value in values:
        out.extend(struct.pack(">I", NULL_INDEX if value is None else positions[value]))
    return bytes(out)


def decode_dict(data: bytes, row_count: int) -> list[int | None]:
    dictionary_size = struct.unpack_from(">I", data, 0)[0]
    dictionary = [struct.unpack_from(">q", data, 4 + index * 8)[0] for index in range(dictionary_size)]
    offset = 4 + dictionary_size * 8
    values: list[int | None] = []
    for index in range(row_count):
        encoded = struct.unpack_from(">I", data, offset + index * 4)[0]
        values.append(None if encoded == NULL_INDEX else dictionary[encoded])
    return values


def encode_logical(values: list[int | None], encoding: str) -> bytes:
    if encoding == "raw":
        return encode_raw(values)
    if encoding == "rle":
        return encode_rle(values)
    if encoding == "dict":
        return encode_dict(values)
    raise ValueError(f"unsupported encoding: {encoding}")


def decode_logical(data: bytes, encoding: str, row_count: int) -> list[int | None]:
    if encoding == "raw":
        return decode_raw(data, row_count)
    if encoding == "rle":
        return decode_rle(data, row_count)
    if encoding == "dict":
        return decode_dict(data, row_count)
    raise ValueError(f"unsupported encoding: {encoding}")


def compress_payload(data: bytes, compression: str) -> bytes:
    if compression == "none":
        return data
    if compression == "zlib" or compression == "zlib-6":
        return zlib.compress(data, 6)
    if compression.startswith("zlib-"):
        return zlib.compress(data, int(compression.split("-", 1)[1]))
    raise ValueError(f"unsupported compression: {compression}")


def decompress_payload(data: bytes, compression: str) -> bytes:
    if compression == "none":
        return data
    if compression.startswith("zlib"):
        return zlib.decompress(data)
    raise ValueError(f"unsupported compression: {compression}")


def logical_block_digest(block_group_id: str, row_count: int, columns: dict[str, list[Any]]) -> str:
    return canonical_hash(
        {
            "block_group_id": block_group_id,
            "row_count": row_count,
            "columns": columns,
        }
    )


def encode_block(
    block_group: dict[str, Any],
    columns: dict[str, list[int | None]] | None = None,
) -> dict[str, Any]:
    block_group_id = block_group["id"]
    row_count = int(block_group["row_count"])
    if columns is None:
        columns = generate_block_columns(block_group)
    columns = {name: list(values) for name, values in columns.items()}
    column_specs = {column["name"]: column for column in block_group["columns"]}

    logical_digest = logical_block_digest(block_group_id, row_count, columns)
    header_columns: list[dict[str, Any]] = []
    column_meta: dict[str, dict[str, Any]] = {}
    frames: list[bytes] = []
    offset = 0

    for name in sorted(columns):
        values = columns[name]
        if len(values) != row_count:
            raise ValueError(f"column {name} length {len(values)} != row_count {row_count}")
        spec_column = column_specs.get(name, {"name": name, "encoding": "raw", "compression": "none"})
        encoding = spec_column["encoding"]
        compression = spec_column["compression"]
        logical = encode_logical(values, encoding)
        stored = compress_payload(logical, compression)
        stats = actual_index_stats(values)
        meta = {
            "encoding": encoding,
            "compression": compression,
            "uncompressed_size": len(logical),
            "stored_size": len(stored),
            "payload_digest": sha256_hex(logical),
            "actual_min": stats["min"],
            "actual_max": stats["max"],
            "null_count": stats["null_count"],
        }
        column_meta[name] = dict(meta)
        header_entry = {"name": name, "offset": offset}
        header_entry.update(meta)
        header_columns.append(header_entry)
        frames.append(stored)
        offset += len(stored)

    header = {
        "format_version": 1,
        "block_group_id": block_group_id,
        "row_count": row_count,
        "logical_digest": logical_digest,
        "logical_digest_algorithm": "gsb-canonical-v1",
        "columns": header_columns,
    }
    header_bytes = canonical_bytes(header)
    block_bytes = BLOCK_MAGIC + BLOCK_FORMAT_BYTE + struct.pack(">I", len(header_bytes)) + header_bytes
    block_bytes += b"".join(frames)
    content_digest = sha256_hex(block_bytes)
    return {
        "block_group_id": block_group_id,
        "row_count": row_count,
        "columns": columns,
        "column_meta": column_meta,
        "header": header,
        "bytes": block_bytes,
        "content_digest": content_digest,
        "logical_digest": logical_digest,
    }


def decode_block_bytes(data: bytes) -> dict[str, Any]:
    if len(data) < 12 or data[:7] != BLOCK_MAGIC:
        raise ValueError("invalid GSB block magic")
    header_length = struct.unpack_from(">I", data, 8)[0]
    header_start = 12
    header_end = header_start + header_length
    header = json.loads(data[header_start:header_end].decode("utf-8"))
    row_count = int(header["row_count"])
    columns: dict[str, list[int | None]] = {}
    column_meta: dict[str, dict[str, Any]] = {}
    for entry in sorted(header["columns"], key=lambda item: item["name"]):
        name = entry["name"]
        start = header_end + int(entry.get("offset", 0))
        size = int(entry["stored_size"])
        stored = data[start : start + size]
        logical = decompress_payload(stored, entry["compression"])
        if sha256_hex(logical) != entry["payload_digest"]:
            raise ValueError(f"payload digest mismatch for {name}")
        values = decode_logical(logical, entry["encoding"], row_count)
        columns[name] = values
        meta = {key: value for key, value in entry.items() if key not in {"name", "offset"}}
        column_meta[name] = meta
    logical_digest = logical_block_digest(header["block_group_id"], row_count, columns)
    if header.get("logical_digest_algorithm") == "gsb-canonical-v1" and logical_digest != header["logical_digest"]:
        raise ValueError("logical digest mismatch")
    return {
        "block_group_id": header["block_group_id"],
        "row_count": row_count,
        "columns": columns,
        "column_meta": column_meta,
        "header": header,
        "content_digest": sha256_hex(data),
        "logical_digest": logical_digest,
        "bytes": data,
    }


def write_block_file(path: str | Path, block: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(block["bytes"])
    return path


def read_block_file(path: str | Path) -> dict[str, Any]:
    return decode_block_bytes(Path(path).read_bytes())
