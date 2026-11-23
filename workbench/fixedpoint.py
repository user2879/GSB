"""Deterministic fixed-point arithmetic helpers.

All scoring math runs on int64 values scaled by ``2 ** frac_bits`` and snapped
to a quantization grid of ``quantum`` integer units.  Every operation is
platform independent (no floating point in the decision path).
"""
from __future__ import annotations

import numpy as np


class FixedPoint:
    def __init__(self, frac_bits: int, quantum: int):
        if frac_bits <= 0:
            raise ValueError("frac_bits must be positive")
        if quantum <= 0:
            raise ValueError("quantum must be positive")
        self.frac_bits = int(frac_bits)
        self.quantum = int(quantum)
        self.scale = 1 << self.frac_bits

    def snap(self, ints: np.ndarray) -> np.ndarray:
        """Snap int64 fixed-point values to the quantum grid (round-half-away-from-zero)."""
        ints = np.asarray(ints, dtype=np.int64)
        q = self.quantum
        snapped = np.sign(ints) * ((np.abs(ints) + q // 2) // q) * q
        return snapped.astype(np.int64)

    def quantize(self, values) -> np.ndarray:
        """Convert floats to snapped fixed-point integers."""
        ints = np.rint(np.asarray(values, dtype=np.float64) * self.scale).astype(np.int64)
        return self.snap(ints)

    def quantize_scalar(self, value: float) -> int:
        return int(self.quantize(np.asarray([value]))[0])

    def score(self, weights, features) -> np.ndarray:
        """Weighted sum of quantized feature planes, rescaled and snapped.

        ``weights`` are plain integers, ``features`` are fixed-point planes.
        The accumulation order is the caller's order and is part of the spec.
        """
        acc = np.zeros_like(features[0], dtype=np.int64)
        for weight, feat in zip(weights, features):
            acc = acc + int(weight) * feat
        return self.snap(acc)

    def params_fingerprint(self) -> str:
        return f"fp{self.frac_bits}q{self.quantum}"
