"""Fixed-point quantization and integer-domain normalized similarity."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .config import DiarConfig

Q16_SHIFT = 16
Q16_ONE = 1 << Q16_SHIFT


@dataclass
class QuantizedWindow:
    index: int
    word_bits: int
    frac_bits: int
    step: float
    integers: Tuple[int, ...]
    residuals: Tuple[float, ...]
    dequantized: Tuple[float, ...]
    max_residual_dim: int
    max_residual_abs: float
    quant_key: str


def _clamp_bounds(word_bits: int) -> Tuple[int, int]:
    # One sign bit plus (word_bits - 1) magnitude bits.
    bound = (1 << (word_bits - 1)) - 1
    return -bound, bound


def quantize_embedding(index: int, embedding: np.ndarray, cfg: DiarConfig,
                       quant_key: str) -> QuantizedWindow:
    step = 2.0 ** (-cfg.frac_bits)
    lower, upper = _clamp_bounds(cfg.word_bits)
    integers: List[int] = []
    residuals: List[float] = []
    dequantized: List[float] = []
    for value in embedding.astype(np.float64):
        scaled = value / step
        candidate = int(math.floor(scaled + 0.5))
        clipped = max(lower, min(upper, candidate))
        back = clipped * step
        integers.append(clipped)
        dequantized.append(back)
        residuals.append(float(value) - back)
    abs_res = [abs(r) for r in residuals]
    max_dim = int(np.argmax(abs_res)) if abs_res else 0
    return QuantizedWindow(
        index=index, word_bits=cfg.word_bits, frac_bits=cfg.frac_bits,
        step=step, integers=tuple(integers), residuals=tuple(residuals),
        dequantized=tuple(dequantized), max_residual_dim=max_dim,
        max_residual_abs=abs_res[max_dim] if abs_res else 0.0,
        quant_key=quant_key,
    )


def _isqrt(value: int) -> int:
    if value <= 0:
        return 0
    return math.isqrt(value)


def integer_similarity(left: Sequence[int], right: Sequence[int]) -> int:
    """Cosine similarity computed entirely with arbitrary-precision ints.

    Returns a Q16 fixed point value in [0, 2^16]. Dimension order is the
    natural ascending index, matching the on-disk cache ordering.
    """
    dot = 0
    norm_l = 0
    norm_r = 0
    for i in range(len(left)):
        a = int(left[i])
        b = int(right[i])
        dot += a * b
        norm_l += a * a
        norm_r += b * b
    if norm_l == 0 or norm_r == 0:
        return 0
    scaled_num = dot * Q16_ONE
    root = _isqrt(norm_l * norm_r)
    if root == 0:
        return 0
    # Integer division truncation is deterministic and keeps the result in the
    # fixed-point domain for exact threshold comparison.
    return int(scaled_num // root)


def threshold_to_q16(threshold: float) -> int:
    value = int(math.floor(threshold * Q16_ONE + 0.5))
    return max(0, min(Q16_ONE, value))


def q16_to_float(value: int) -> float:
    return value / float(Q16_ONE)


def float_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
