"""Deterministic 64-bit integer randomness.

Reproducibility is a hard requirement, so we never rely on the host ``random``
module (its stream has changed across Python releases and may be seeded from
OS entropy). ``SplitMix64`` is a tiny, well-known stateless counter-based
generator: the same seed always yields the identical integer sequence on every
platform and Python version.
"""
from __future__ import annotations

from typing import Iterable, Union

MASK64 = (1 << 64) - 1


def _u64(value: int) -> int:
    return value & MASK64


class SplitMix64:
    """Counter-based deterministic PRNG operating on unsigned 64-bit ints."""

    def __init__(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        self._state = _u64(seed)

    def next_u64(self) -> int:
        self._state = _u64(self._state + 0x9E3779B97F4A7C15)
        z = self._state
        z = _u64((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9)
        z = _u64((z ^ (z >> 27)) * 0x94D049BB133111EB)
        return _u64(z ^ (z >> 31))

    def next_below(self, exclusive_upper: int) -> int:
        """Return an int in ``[0, exclusive_upper)`` without modulo bias."""
        if exclusive_upper <= 0:
            raise ValueError("exclusive_upper must be positive")
        if exclusive_upper == 1:
            return 0
        threshold = (1 << 64) % exclusive_upper
        while True:
            value = self.next_u64()
            if value >= threshold:
                return value % exclusive_upper

    def next_in_range(self, lower: int, upper_inclusive: int) -> int:
        return lower + self.next_below(upper_inclusive - lower + 1)


def mix_u64(parts: Iterable[Union[str, int, bytes]]) -> int:
    """Deterministically fold any ordered set of strings/ints/bytes into one u64.

    Used to derive per-injection/per-request decisions from an experiment seed
    and stable identifiers without consuming a shared random stream (which would
    couple unrelated decisions and make replay harder to reason about).
    """
    h = 0xCBF29CE484222325
    for part in parts:
        if isinstance(part, bytes):
            data = part
        elif isinstance(part, str):
            data = part.encode("utf-8")
        elif isinstance(part, int) and not isinstance(part, bool):
            data = part.to_bytes(16, "little", signed=True)
        else:
            raise TypeError("mix_u64 only accepts str, int or bytes parts")
        h = _u64(h ^ 0x01000193)
        for byte in data:
            h = _u64(h ^ byte)
            h = _u64(h * 0x100000001B3)
    return h
