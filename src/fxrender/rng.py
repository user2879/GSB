"""Seeded deterministic random source with no global runtime state.

A small SplitMix64-based generator is used so results are identical on every
machine and independent of numpy's global RNG. Each latent gets its own
generator derived from (fingerprint, seed, config fingerprint).
"""

from __future__ import annotations

import math

MASK64 = (1 << 64) - 1


class DeterministicRNG:
    def __init__(self, seed_material: str | int) -> None:
        if isinstance(seed_material, str):
            h = 0xCBF29CE484222325
            for ch in seed_material.encode("utf-8"):
                h ^= ch
                h = (h * 0x100000001B3) & MASK64
            state = h
        else:
            state = int(seed_material) & MASK64
        self._state = state

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def next_unit(self) -> float:
        # 53-bit mantissa, interval [0, 1)
        return (self.next_u64() >> 11) / float(1 << 53)

    def next_signed(self) -> float:
        # Symmetric interval (-1, 1), zero excluded deterministically.
        return self.next_unit() * 2.0 - 1.0

    def next_gaussian(self) -> float:
        # Box-Muller with a guarded log; the unit value never reaches 1.0.
        u1 = max(self.next_unit(), 1e-12)
        u2 = self.next_unit()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def noise(self, scheme: str) -> float:
        if scheme == "uniform":
            return self.next_signed()
        if scheme == "gaussian":
            return self.next_gaussian()
        if scheme == "laplacian":
            u = self.next_unit()
            # Inverse CDF of a standard Laplace on (-inf, inf) via u in (0,1)
            e = max(min(u, 1 - 1e-12), 1e-12)
            if e < 0.5:
                return math.log(2 * e)
            return -math.log(2 * (1 - e))
        raise ValueError(f"unknown noise scheme: {scheme!r}")
