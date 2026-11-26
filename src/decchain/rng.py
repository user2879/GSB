"""Deterministic, global-state-free pseudo random source.

The stream is derived purely from a seed (itself derived from request
fingerprint, step index and config fingerprint) plus an internal counter.
SHA-256 in counter mode yields 32-bit words; bounded integers use rejection
sampling so there is no modulo bias.  No ``random``/``numpy.random`` global
state is ever touched.
"""

from __future__ import annotations

import hashlib


class DeterministicRng:
    def __init__(self, seed_hex: str):
        self._base = bytes.fromhex(seed_hex)
        self._counter = 0
        self._buffer = 0
        self._buffered_bits = 0

    def _next_word(self) -> int:
        block = hashlib.sha256(
            self._base + self._counter.to_bytes(8, "big")
        ).digest()
        self._counter += 1
        return int.from_bytes(block[:4], "big")

    def _next_bits(self, nbits: int) -> int:
        while self._buffered_bits < nbits:
            self._buffer |= self._next_word() << self._buffered_bits
            self._buffered_bits += 32
        value = self._buffer & ((1 << nbits) - 1)
        self._buffer >>= nbits
        self._buffered_bits -= nbits
        return value

    def next_unit(self) -> float:
        """Uniform float in [0, 1) using 53 random bits."""
        return self._next_bits(53) / float(1 << 53)

    def next_bounded(self, n: int) -> int:
        """Uniform integer in [0, n) without modulo bias."""
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return 0
        limit = (1 << 32) - ((1 << 32) % n)
        while True:
            word = self._next_word()
            if word < limit:
                return word % n


def derive_seed(content_fp: str, seq_id: int, step_index: int,
                config_fp: str) -> str:
    payload = (f"{content_fp}|seq={int(seq_id)}|step={int(step_index)}"
               f"|cfg={config_fp}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
