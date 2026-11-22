"""Deterministic fixed-seed content samples and canonical byte digests.

No network access is involved: content bytes are generated locally with the
splitmix64 PRNG seeded from the declared content seed plus the (version-free)
content path.  Two independent FNV-1a 32-bit hashes describe the data:

* ``content_hash`` - hash of the full byte stream of a path;
* ``shard_hash``   - hash of one declared half-open byte interval.

All hashes are reproducible across processes and platform re-runs.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence, Tuple

FNV_OFFSET = 0x811C9DC5
FNV_PRIME = 0x01000193
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def splitmix64(state: int) -> Tuple[int, int]:
    """One splitmix64 step, returning ``(new_state, 32-bit word)``."""
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    mixed = state
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & MASK64
    mixed = mixed ^ (mixed >> 31)
    return state, mixed & MASK32


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace surprises."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def fnv1a_32(data: bytes) -> int:
    """32-bit FNV-1a over an explicit byte buffer."""
    digest = FNV_OFFSET
    for byte in data:
        digest ^= byte
        digest = (digest * FNV_PRIME) & MASK32
    return digest


def path_seed(content_seed: int, path: str) -> int:
    """Version-independent per-path PRNG seed so data is stable across versions."""
    return (int(content_seed) ^ fnv1a_32(path.encode("utf-8"))) & MASK64


def generate_bytes(content_seed: int, path: str, length: int) -> bytes:
    """Deterministic content bytes for ``length`` bytes of ``path``."""
    if length < 0:
        raise ValueError("content length must be non-negative")
    state = path_seed(content_seed, path)
    out = bytearray()
    while len(out) < length:
        state, word = splitmix64(state)
        out.extend(word.to_bytes(4, "little"))
    return bytes(out[:length])


def content_hash(content_seed: int, path: str, length: int) -> int:
    return fnv1a_32(generate_bytes(content_seed, path, length))


def shard_hash(
    content_seed: int, path: str, start: int, end: int, length: int
) -> int:
    """FNV-1a over the half-open ``[start, end)`` byte interval."""
    if start < 0 or end > length or start > end:
        raise ValueError(
            f"shard interval [{start},{end}) out of bounds for length {length}"
        )
    data = generate_bytes(content_seed, path, length)[start:end]
    return fnv1a_32(data)


def shard_hit_digest(hashes: Sequence[int]) -> int:
    """Deterministic ordered combination of per-shard hashes into one digest."""
    digest = FNV_OFFSET
    for value in hashes:
        for byte in int(value & MASK32).to_bytes(4, "big"):
            digest ^= byte
            digest = (digest * FNV_PRIME) & MASK32
    return digest


def shard_intervals(shards: Sequence[Dict[str, int]], length: int) -> List[Tuple[int, int]]:
    """Return validated contiguous half-open intervals from a shard declaration."""
    intervals: List[Tuple[int, int]] = []
    cursor = 0
    for shard in shards:
        start, end = int(shard["start"]), int(shard["end"])
        if start != cursor:
            raise ValueError(
                f"shards not contiguous: expected start {cursor}, got {start}"
            )
        if end <= start:
            raise ValueError(f"empty/negative shard [{start},{end})")
        if end > length:
            raise ValueError(f"shard end {end} exceeds content length {length}")
        intervals.append((start, end))
        cursor = end
    if cursor != length:
        raise ValueError(
            f"shards do not cover full content: covered {cursor}/{length}"
        )
    return intervals
