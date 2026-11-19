"""Deterministic, hash-seeded pseudo random generator.

``DeterministicRNG`` derives every byte from SHA-256 over an explicit key, so:

* it never reads NumPy / Python global random state;
* the same key produces the same byte stream on every platform and run;
* distinct keys produce independent streams.

Integers are decoded big-endian and floats are produced via 53-bit mantissa
division, matching a well-defined, version-independent formula.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .hashing import canonical_bytes, sha256_hex


class DeterministicRNG:
    def __init__(self, key: object):
        # Mixing a constant domain prefix avoids cross-purpose key collisions.
        material = b"dbench.rng.v1\0" + canonical_bytes(key)
        self._base = hashlib.sha256(material).digest()
        self._counter = 0
        self._buffer = b""
        self._offset = 0

    def _refill(self) -> None:
        block = hashlib.sha256(
            self._base + self._counter.to_bytes(8, "big")
        ).digest()
        self._counter += 1
        self._buffer += block

    def _take(self, num_bytes: int) -> bytes:
        while len(self._buffer) - self._offset < num_bytes:
            self._refill()
        chunk = self._buffer[self._offset : self._offset + num_bytes]
        self._offset += num_bytes
        return chunk

    def random_bytes(self, num_bytes: int) -> bytes:
        if num_bytes < 0:
            raise ValueError("num_bytes must be non-negative")
        return b"".join(self._take(1) for _ in range(num_bytes))

    def standard_normal(self, shape: tuple[int, ...]) -> np.ndarray:
        """Standard normal samples via the Box-Muller transform."""
        size = int(np.prod(shape)) if shape else 1
        # Two uniforms per sample; generate raw bytes in 8-byte units.
        needed = size * 2
        raw = self._take(needed * 8)
        ints = np.frombuffer(raw, dtype=np.uint64).astype(np.float64)
        # Uniform in (0, 1], avoiding log(0).
        uniforms = (ints + 1.0) / (float(1 << 64) + 1.0)
        u1, u2 = uniforms[0::2], uniforms[1::2]
        r = np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        samples = r * np.cos(theta)
        return samples.reshape(shape)

    def uniform(self, shape: tuple[int, ...], low: float = 0.0, high: float = 1.0) -> np.ndarray:
        size = int(np.prod(shape)) if shape else 1
        raw = self._take(size * 8)
        ints = np.frombuffer(raw, dtype=np.uint64).astype(np.float64)
        samples = ints / float(1 << 64)
        return (low + (high - low) * samples).reshape(shape)


def derive_rng(*parts: object) -> DeterministicRNG:
    """Build an RNG keyed by the concatenation of explicit logical parts."""
    key = sha256_hex(canonical_bytes(list(parts)))
    return DeterministicRNG(key)
