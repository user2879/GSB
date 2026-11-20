"""Deterministic fixed-point integer arithmetic and quantization.

All quantized pipeline data is a signed 64-bit integer carrying a fixed number
of fractional bits (``frac_bits``).  Real value ``v`` is represented as
``round(v * 2**frac_bits)``.  Quantization to a configurable number of levels
is implemented on top of this representation so that both the fractional bit
width and the quantization level count are tracked in the artifact metadata.

No floating point is used anywhere after the initial quantization step; all
layer accumulation, material mapping and weighted blending operate on int64
arrays, which guarantees pixel-identical results across runs and platforms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROUND_MODES = ("half_up", "half_even", "truncate", "floor")


def _check_round_mode(round_mode: str) -> None:
    if round_mode not in ROUND_MODES:
        raise ValueError(f"unsupported round_mode={round_mode!r}, expect one of {ROUND_MODES}")


def floor_div(a: np.ndarray, b: int) -> np.ndarray:
    """Floor division for signed integer operands (Python-style, sign-safe)."""
    a = np.asarray(a, dtype=np.int64)
    q = a // np.int64(b)
    return q


def trunc_div(a: np.ndarray, b: int) -> np.ndarray:
    """Division rounded toward zero for signed integer operands."""
    a = np.asarray(a, dtype=np.int64)
    sign = np.sign(a)
    return (np.abs(a) // np.int64(b)) * sign


def _step_for_domain(lo: float, hi: float, levels: int, frac_bits: int) -> int:
    span = hi - lo
    real_step = span / levels
    return max(1, int(round(real_step * (1 << frac_bits))))


@dataclass(frozen=True)
class FixedSpec:
    """Quantization specification for one numerical domain.

    ``frac_bits`` is the fixed-point fractional bit width and ``levels`` the
    number of quantization steps across the closed domain ``[lo, hi]``.
    """

    frac_bits: int
    levels: int
    lo: float
    hi: float

    @property
    def scale(self) -> int:
        return 1 << self.frac_bits

    @property
    def step(self) -> int:
        return _step_for_domain(self.lo, self.hi, self.levels, self.frac_bits)

    def clamp_ints(self, values: np.ndarray) -> np.ndarray:
        lo = int(round(self.lo * self.scale))
        hi = int(round(self.hi * self.scale))
        return np.clip(values, lo, hi).astype(np.int64)


def quantize_float(values: np.ndarray, spec: FixedSpec, round_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a float array to fixed-point integer levels.

    Returns ``(quantized_int64, boundary_mask)`` where ``boundary_mask`` is a
    boolean array marking pixels whose raw value sat exactly on a rounding
    decision boundary (a half-step tie, or a domain endpoint after rounding).
    """
    _check_round_mode(round_mode)
    scale = spec.scale
    step = spec.step
    lo_i = int(round(spec.lo * scale))
    hi_i = int(round(spec.hi * scale))

    scaled = np.asarray(values, dtype=np.float64) * scale
    index = scaled / step
    if round_mode == "truncate":
        idx = np.trunc(index).astype(np.int64)
        tie = np.abs(index - np.trunc(index)) == 0.0
    elif round_mode == "floor":
        idx = np.floor(index).astype(np.int64)
        tie = np.abs(index - np.floor(index)) == 0.0
    else:
        floor_idx = np.floor(index).astype(np.int64)
        frac = index - np.floor(index)
        tie = np.isclose(frac, 0.5, rtol=0.0, atol=1e-12)
        if round_mode == "half_up":
            bump = (frac >= 0.5).astype(np.int64)
            idx = floor_idx + bump
        else:  # half_even: ties resolve toward the even multiple index
            bump = np.where(tie, (floor_idx % 2).astype(np.int64), (frac > 0.5).astype(np.int64))
            idx = floor_idx + bump
    q = idx * np.int64(step) + np.int64(lo_i)
    q = np.clip(q, lo_i, hi_i).astype(np.int64)
    boundary = tie | (q == lo_i) | (q == hi_i)
    return q, boundary


def round_div_int(a: np.ndarray, b: int, round_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Signed integer ``a / b`` with a selectable rounding rule.

    Returns the quotient and a boolean tie mask (exact half-step remainder),
    which callers use to annotate boundary samples.  ``b`` must be positive.
    """
    _check_round_mode(round_mode)
    a = np.asarray(a, dtype=np.int64)
    b = int(b)
    if b <= 0:
        raise ValueError("divisor must be positive")
    abs_a = np.abs(a)
    sign = np.sign(a).astype(np.int64)
    q_abs = abs_a // np.int64(b)
    r_abs = abs_a - q_abs * np.int64(b)
    half = b // 2
    tie = (2 * r_abs) == b if b % 2 == 0 else np.zeros_like(a, dtype=bool)
    if round_mode == "truncate":
        q = q_abs * sign
    elif round_mode == "floor":
        q = a // np.int64(b)
    elif round_mode == "half_up":
        bump = (r_abs * 2 >= b).astype(np.int64)
        q = (q_abs + bump) * sign
    else:  # half_even
        tie_bump = tie & ((q_abs % 2) == 1)
        bump = np.where(tie_bump, 1, (r_abs * 2 > b).astype(np.int64))
        q = (q_abs + bump) * sign
    return q.astype(np.int64), tie


def weighted_blend(
    terms: list[tuple[np.ndarray, int]],
    spec: FixedSpec,
    round_mode: str,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-point weighted blend of integer arrays.

    ``terms`` is a list of ``(int64_array, weight)``.  When ``normalize`` is
    true the weighted sum is divided by the (positive) total weight
    (``blend_mode='normalized_weight'``); otherwise weights are literal
    coefficients (``blend_mode='weighted_sum'``) and the result is clamped to
    the domain of ``spec``.  Returns the integer result and a boundary mask.
    """
    if not terms:
        raise ValueError("weighted_blend requires at least one term")
    total_weight = sum(w for _, w in terms)
    if total_weight <= 0:
        raise ValueError("sum of weights must be positive")
    acc = np.zeros_like(terms[0][0], dtype=np.int64)
    for arr, weight in terms:
        acc = acc + arr.astype(np.int64) * np.int64(weight)
    if normalize:
        out, tie = round_div_int(acc, total_weight, round_mode)
        out = spec.clamp_ints(out)
    else:
        out = spec.clamp_ints(acc)
        tie = np.zeros_like(acc, dtype=bool)
    return out, tie


def fixed_to_uint8(values: np.ndarray, spec: FixedSpec) -> np.ndarray:
    """Map fixed-point integers of ``spec`` onto a uint8 preview image."""
    lo_i = int(round(spec.lo * spec.scale))
    hi_i = int(round(spec.hi * spec.scale))
    shifted = np.asarray(values, dtype=np.int64) - lo_i
    span = max(1, hi_i - lo_i)
    out = np.rint(shifted.astype(np.float64) * 255.0 / span)
    return np.clip(out, 0, 255).astype(np.uint8)
