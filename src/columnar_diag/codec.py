from __future__ import annotations

import json
import struct
import zlib
from typing import Iterable

import numpy as np

from .models import EncodedBlock
from .util import canonical_bytes, sha256_bytes


def _pack_raw(values: tuple[int, ...]) -> bytes:
    return np.asarray(values, dtype="<i8").tobytes(order="C")


def _unpack_raw(payload: bytes, row_count: int) -> tuple[int, ...]:
    if len(payload) != row_count * 8:
        raise ValueError("raw payload length does not match row count")
    return tuple(int(value) for value in np.frombuffer(payload, dtype="<i8", count=row_count))


def _pack_dictionary(values: tuple[int, ...]) -> bytes:
    dictionary = sorted(set(values))
    positions = {value: index for index, value in enumerate(dictionary)}
    parts = [struct.pack("<I", len(dictionary))]
    parts.extend(struct.pack("<q", value) for value in dictionary)
    parts.extend(struct.pack("<I", positions[value]) for value in values)
    return b"".join(parts)


def _unpack_dictionary(payload: bytes, row_count: int) -> tuple[int, ...]:
    if len(payload) < 4:
        raise ValueError("dictionary payload truncated")
    count = struct.unpack_from("<I", payload, 0)[0]
    expected = 4 + count * 8 + row_count * 4
    if len(payload) != expected:
        raise ValueError("dictionary payload length mismatch")
    dictionary = tuple(
        struct.unpack_from("<q", payload, 4 + index * 8)[0] for index in range(count)
    )
    if tuple(sorted(dictionary)) != dictionary or len(set(dictionary)) != len(dictionary):
        raise ValueError("dictionary entries are not strictly ascending")
    base = 4 + count * 8
    result = []
    for index in range(row_count):
        position = struct.unpack_from("<I", payload, base + index * 4)[0]
        if position >= count:
            raise ValueError("dictionary position out of range")
        result.append(dictionary[position])
    return tuple(result)


def _pack_rle(values: tuple[int, ...]) -> bytes:
    runs: list[tuple[int, int]] = []
    for value in values:
        if runs and runs[-1][1] == value:
            count, previous = runs[-1]
            runs[-1] = (count + 1, previous)
        else:
            runs.append((1, value))
    return struct.pack("<I", len(runs)) + b"".join(
        struct.pack("<Iq", count, value) for count, value in runs
    )


def _unpack_rle(payload: bytes, row_count: int) -> tuple[int, ...]:
    if len(payload) < 4:
        raise ValueError("rle payload truncated")
    run_count = struct.unpack_from("<I", payload, 0)[0]
    expected = 4 + run_count * 12
    if len(payload) != expected:
        raise ValueError("rle payload length mismatch")
    result: list[int] = []
    offset = 4
    for _ in range(run_count):
        count, value = struct.unpack_from("<Iq", payload, offset)
        if count == 0:
            raise ValueError("rle run count must be positive")
        result.extend([value] * count)
        offset += 12
    if len(result) != row_count:
        raise ValueError("rle run lengths do not match row count")
    return tuple(result)


def encode_payload(encoding: str, values: Iterable[int]) -> bytes:
    materialized = tuple(int(value) for value in values)
    if encoding == "raw":
        return _pack_raw(materialized)
    if encoding == "dictionary":
        return _pack_dictionary(materialized)
    if encoding == "rle":
        return _pack_rle(materialized)
    raise ValueError(f"unsupported encoding: {encoding}")


def decode_payload(encoding: str, payload: bytes, row_count: int) -> tuple[int, ...]:
    if encoding == "raw":
        return _unpack_raw(payload, row_count)
    if encoding == "dictionary":
        return _unpack_dictionary(payload, row_count)
    if encoding == "rle":
        return _unpack_rle(payload, row_count)
    raise ValueError(f"unsupported encoding: {encoding}")


def encode_block(
    *,
    block_id: str,
    column: str,
    row_group: str,
    encoding: str,
    compressed: bool,
    values: tuple[int, ...],
) -> EncodedBlock:
    raw_payload = encode_payload(encoding, values)
    stored_payload = zlib.compress(raw_payload, level=9) if compressed else raw_payload
    payload_digest = sha256_bytes(stored_payload)
    header = {
        "block_id": block_id,
        "column": column,
        "row_group": row_group,
        "encoding": encoding,
        "compressed": compressed,
        "row_count": len(values),
        "uncompressed_bytes": len(raw_payload),
        "compressed_bytes": len(stored_payload),
        "payload_digest": payload_digest,
    }
    header_bytes = canonical_bytes(header)
    record_bytes = (
        struct.pack("<I", len(header_bytes))
        + header_bytes
        + struct.pack("<I", len(stored_payload))
        + stored_payload
    )
    block_digest = sha256_bytes(header_bytes + stored_payload)
    return EncodedBlock(
        block_id=block_id,
        column=column,
        row_group=row_group,
        encoding=encoding,
        compressed=compressed,
        row_count=len(values),
        pre_bytes=len(raw_payload),
        post_bytes=len(stored_payload),
        payload=stored_payload,
        header=header,
        header_bytes=header_bytes,
        block_digest=block_digest,
        record_bytes=record_bytes,
    )


def decode_block(block: EncodedBlock) -> tuple[int, ...]:
    raw_payload = zlib.decompress(block.payload) if block.compressed else block.payload
    if len(raw_payload) != block.pre_bytes:
        raise ValueError("uncompressed byte count mismatch")
    return decode_payload(block.encoding, raw_payload, block.row_count)


def decode_record(header_bytes: bytes, payload: bytes) -> tuple[dict, tuple[int, ...]]:
    metadata = json.loads(header_bytes.decode("utf-8"))
    if sha256_bytes(payload) != metadata.get("payload_digest"):
        raise ValueError("block payload digest mismatch")
    raw_payload = zlib.decompress(payload) if metadata["compressed"] else payload
    if len(raw_payload) != metadata["uncompressed_bytes"]:
        raise ValueError("block uncompressed byte count mismatch")
    values = decode_payload(
        str(metadata["encoding"]), raw_payload, int(metadata["row_count"])
    )
    return metadata, values
