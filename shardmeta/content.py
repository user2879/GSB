"""Deterministic shard content and NumPy fixed-point checksum digests.

The rules below are explicit and depend only on integer inputs (node seed,
object seed, object key, shard index, declared shard size). Therefore the same
declaration produces byte-identical shards and identical digests regardless of
load order, batch splitting or dict iteration order.

Content generation
------------------
For a shard of length ``L`` bytes we use a ``numpy.random.Generator`` seeded by
a 64-bit integer derived from node seed, object seed, key hash and shard index
and draw uint8 values in ``[0, 250)``, leaving room for integer fixed-point
arithmetic without overflow concerns.

Digest (all arithmetic uses numpy uint64 fixed-point integer operations)
----------------------------------------------------------------------
* ``content_check``  – splitmix64 of a FNV-1a style hash of the shard bytes.
* ``fixed_mean``     – exact integer mean * 1000 (scaled fixed point).
* ``shard_digest``   – 16 hex chars binding node id, key, index and content.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

# uint64 wrap-around is the intended fixed-point behaviour; silence overflow.
np.seterr(over="ignore")

BYTE_MOD = np.uint64(250)
MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
FNV_OFFSET = np.uint64(0xCBF29CE484222325)
FNV_PRIME = np.uint64(0x100000001B3)
FIXED_SCALE = np.uint64(1000)


def _u64(value: int) -> np.uint64:
    return np.uint64(int(value) & 0xFFFFFFFFFFFFFFFF)


def mix64(value: int) -> int:
    """SplitMix64 finalizer over a 64-bit integer, returning a Python int."""
    z = _u64(value)
    z = (z + np.uint64(0x9E3779B97F4A7C15)) & MASK64
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    z = z ^ (z >> np.uint64(31))
    return int(z)


def _string_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def shard_stream_seed(node_seed: int, object_seed: int, key: str, shard_index: int) -> int:
    """Derive the reproducible RNG seed for one shard."""
    combined = (
        _u64(node_seed)
        ^ (_u64(object_seed) * np.uint64(0x9E3779B97F4A7C15))
        ^ _u64(_string_int(key))
        ^ (_u64(shard_index) * np.uint64(0x100000001B3))
    )
    return mix64(int(combined))


def generate_shard_bytes(
    node_seed: int, object_seed: int, key: str, shard_index: int, size: int
) -> bytes:
    """Generate deterministic shard content bytes for a declared shard."""
    if size <= 0:
        raise ValueError("shard size must be positive")
    seed = shard_stream_seed(node_seed, object_seed, key, shard_index)
    generator = np.random.default_rng(seed)
    values = generator.integers(0, int(BYTE_MOD), size=int(size), dtype=np.uint64)
    return values.astype(np.uint8).tobytes()


def _fnv1a_uint64(data: bytes) -> int:
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    h = FNV_OFFSET
    for byte in arr:
        h = ((h ^ byte) * FNV_PRIME) & MASK64
    return int(h)


def content_check_value(data: bytes) -> str:
    """Return the splitmix64 content check as 16-char lower-case hex."""
    return f"{mix64(_fnv1a_uint64(data)):016x}"


def fixed_point_mean_x1000(data: bytes) -> int:
    """Integer fixed-point mean of bytes, scaled by 1000 (exact, no float)."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    total = np.uint64(0)
    for value in arr:
        total = (total + value) & MASK64
    length = np.uint64(len(data))
    return int((total * FIXED_SCALE) // length)


def shard_digest(node_id: str, key: str, shard_index: int, data: bytes) -> str:
    """Bind node id + key + index + content into a 16 hex-char digest.

    The binding hashes a canonical text prefix together with the FNV hash of
    the content bytes, then applies splitmix64 so the result is a stable,
    explicitly reproducible function of exactly the four inputs.
    """
    prefix = f"node={node_id}|key={key}|idx={int(shard_index)}|".encode("utf-8")
    bound = _fnv1a_uint64(prefix + data)
    bound ^= _string_int(node_id)
    bound = int(bound & 0xFFFFFFFFFFFFFFFF)
    return f"{mix64(bound):016x}"


@dataclass(frozen=True)
class ShardChecksum:
    shard_digest: str
    content_check: str
    fixed_mean_x1000: int
    size: int

    def as_dict(self) -> dict:
        return {
            "shard_digest": self.shard_digest,
            "content_check": self.content_check,
            "fixed_mean_x1000": int(self.fixed_mean_x1000),
            "size": int(self.size),
        }


def compute_checksum(node_id: str, key: str, shard_index: int, data: bytes) -> ShardChecksum:
    return ShardChecksum(
        shard_digest=shard_digest(node_id, key, shard_index, data),
        content_check=content_check_value(data),
        fixed_mean_x1000=fixed_point_mean_x1000(data),
        size=len(data),
    )


__all__ = [
    "mix64",
    "shard_stream_seed",
    "generate_shard_bytes",
    "content_check_value",
    "fixed_point_mean_x1000",
    "shard_digest",
    "compute_checksum",
    "ShardChecksum",
]
