"""Deterministic, instance-local random source.

No module-global RNG and no :mod:`random` global state is used.  Every stream is
keyed by an explicit material (seed + fingerprints + purpose) and generates
integers via counter-mode BLAKE2b, giving reproducible streams independent of
call order or thread interleaving.
"""
from __future__ import annotations

import hashlib

_MASK64 = (1 << 64) - 1


class DeterministicRandom:
    def __init__(self, material: str):
        self._key = hashlib.blake2b(material.encode("utf-8"),
                                    digest_size=32).digest()
        self._counter = 0

    def _next_block(self) -> bytes:
        block = hashlib.blake2b(
            self._counter.to_bytes(8, "little") + self._key,
            digest_size=32,
        ).digest()
        self._counter += 1
        return block

    def next_uint(self, bound: int) -> int:
        """Uniform integer in [0, bound) with rejection sampling."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        # Rejection sampling on a 64-bit draw to avoid modulo bias.
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = int.from_bytes(self._next_block()[:8], "little") & _MASK64
            if value < limit:
                return value % bound

    def next_signed(self, bound: int) -> int:
        """Uniform integer in [-bound, bound]."""
        return self.next_uint(2 * bound + 1) - bound

    def ints(self, count: int, bound: int) -> list[int]:
        return [self.next_uint(bound) for _ in range(count)]

    def noise_table(self, length: int, amplitude: int) -> list[int]:
        """Signed integer noise entries in [-amplitude, amplitude]."""
        return [self.next_signed(amplitude) for _ in range(length)]
