"""Deterministic hashing helpers.

No runtime random state (``random`` / global RNG) is used anywhere in this
project.  These helpers give stable 64 bit fingerprints for frame contents,
configuration objects and cache keys.
"""
from __future__ import annotations

import json
import struct

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & MASK64
    return h


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def canonical_bytes(obj) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def canonical_hash(obj) -> str:
    """Return a stable 16 hex digit fingerprint for a JSON-serialisable object."""
    return format(fnv1a64(canonical_bytes(obj)), "016x")


def hash_mix(*parts: str) -> str:
    """Combine several fingerprints into one stable fingerprint."""
    h = FNV_OFFSET
    for part in parts:
        h = fnv1a64(struct.pack("<Q", h) + str(part).encode("utf-8"))
    return format(h, "016x")
