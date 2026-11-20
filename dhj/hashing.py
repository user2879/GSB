"""Explicit, platform-independent hash combination rules.

Two rules are declared and used everywhere:

* ``FNV1A64`` - 64-bit FNV-1a over canonical UTF-8 bytes (the only rule used
  for partition assignment and row summaries).
* Partition mapping - ``partition = FNV1A64(canonical(q_int)) mod n``, where
  ``q_int`` is the fixed-point quantized join key (or the exact int scaled by
  the session scale factor).
"""

import struct

from . import fixedpoint as fp
from .jsonio import dumps_canonical

FNV_OFFSET_BASIS_64 = 0xCBF29CE484222325
FNV_PRIME_64 = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET_BASIS_64
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME_64) & MASK64
    return h


def hash_int(value: int) -> int:
    """Hash a canonical fixed-point integer (decimal canonical string bytes)."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("hash_int expects an int")
    return fnv1a64(dumps_canonical(value))


def key_partition(quantized_key, num_partitions: int) -> int:
    """Map a quantized (non-null) join key to a partition index."""
    if not isinstance(num_partitions, int) or num_partitions <= 0:
        raise ValueError("num_partitions must be a positive integer")
    if quantized_key == fp.NULL_Q:
        raise ValueError("null join keys cannot be assigned to a partition")
    return hash_int(int(quantized_key)) % num_partitions


def row_digest(table_name, row_id, side, join_key_column, value, null_bitmap):
    """Stable per-row summary, independent of bucket version and partition."""
    payload = {
        "t": table_name,
        "r": row_id,
        "s": side,
        "k": join_key_column,
        "v": value,
        "n": null_bitmap,
    }
    return fnv1a64(dumps_canonical(payload))


def digest_hex_64(value_int):
    """Stable 16-char hex rendering of a 64-bit digest."""
    return format(value_int & MASK64, "016x")


def pack_uint64(value: int) -> bytes:
    return struct.pack(">Q", value & MASK64)


def unpack_uint64(data: bytes, offset: int):
    return struct.unpack_from(">Q", data, offset)[0]
