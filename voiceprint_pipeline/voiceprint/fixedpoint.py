"""Deterministic fixed-point integer arithmetic (Qm.n style, numpy int64)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import math

import numpy as np

from .util import dround_array, stable_seed


@dataclass(frozen=True)
class FixedPoint:
    bits: int  # total fractional width n (Qn scale = 2^n)

    @property
    def scale(self) -> int:
        return 1 << self.bits

    def quantize(self, values: np.ndarray) -> np.ndarray:
        """Quantize float64 values to signed integer representation at this Q width."""
        return dround_array(np.asarray(values, dtype=np.float64) * self.scale)

    def dequantize(self, ints: np.ndarray) -> np.ndarray:
        return np.asarray(ints, dtype=np.float64) / self.scale

    def reduce_scale(self, ints: np.ndarray, target_bits: int) -> np.ndarray:
        """Convert integers from this Q width to a smaller Q width with round-half-away."""
        if target_bits > self.bits:
            raise ValueError("reduce_scale can only reduce width")
        shift = self.bits - target_bits
        if shift == 0:
            return np.asarray(ints, dtype=np.int64).copy()
        values = np.asarray(ints, dtype=np.float64) / (1 << shift)
        return dround_array(values)


def deterministic_rng(seed_text: str) -> np.random.Generator:
    """Seed a numpy PCG64 generator purely from deterministic text."""
    return np.random.default_rng(stable_seed(seed_text))


def integer_cosine(vec_a: np.ndarray, vec_b: np.ndarray, q_bits: int) -> dict[str, Any]:
    """Cosine similarity on integer embeddings with deterministic integer norms.

    Returns intermediate integers plus the rounded Q30 similarity value.
    """
    a = np.asarray(vec_a, dtype=np.int64)
    b = np.asarray(vec_b, dtype=np.int64)
    dot = int(np.dot(a, b))
    norm_a_sq = int(np.dot(a, a))
    norm_b_sq = int(np.dot(b, b))
    norm_a = _isqrt64(norm_a_sq)
    norm_b = _isqrt64(norm_b_sq)
    if norm_a_sq == 0 or norm_b_sq == 0:
        return {
            "dot": dot,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "similarity_q30": 0,
            "similarity": 0.0,
        }
    # Keep squared norms to avoid sqrt truncation; the denominator product stays
    # exact and the single rounded division yields a similarity within [-1, 1].
    import math

    denom = math.isqrt(norm_a_sq * norm_b_sq)
    numerator = dot * (1 << 30)
    sim_q30 = _div_round_half_away(numerator, denom)
    sim_q30 = max(-(1 << 30), min(1 << 30, sim_q30))
    return {
        "dot": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "similarity_q30": int(sim_q30),
        "similarity": sim_q30 / (1 << 30),
        "q_bits": int(q_bits),
    }


def _isqrt64(value: int) -> int:
    import math

    return math.isqrt(max(0, int(value)))


def _div_round_half_away(numerator: int, denominator: int) -> int:
    import math

    if numerator == 0:
        return 0
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    num = abs(numerator)
    den = abs(denominator)
    if den == 0:
        return 0
    quotient, remainder = divmod(num, den)
    if remainder * 2 >= den:
        quotient += 1
    return sign * quotient


def quantize_symmetric(values: np.ndarray, levels: int, fp: FixedPoint) -> dict[str, Any]:
    """Signed uniform quantization into a fixed code range anchored at zero.

    Inputs are expected on the normalized calibration domain [-1, 1]. Components
    are mapped onto integer codes in ``[-(levels-1), levels-1]`` and then onto the
    fixed-point Q scale.
    """
    x = np.asarray(values, dtype=np.float64)
    codes = np.clip(dround_array(x * (levels - 1)), -(levels - 1),
                    levels - 1).astype(np.int64)
    step = 1.0 / (levels - 1)
    reconstructed = codes.astype(np.float64) * step
    q_ints = fp.quantize(reconstructed)
    return {
        "codes": codes,
        "step": step,
        "scale": step,
        "lo": -1.0,
        "hi": 1.0,
        "reconstructed": reconstructed,
        "q_ints": q_ints,
        "levels": int(levels),
        "bits": fp.bits,
    }


def calibration_scales(feature_names: list[str], config_scope_hash: str) -> np.ndarray:
    """Deterministic per-dimension calibration scales (no runtime data dependency).

    Scales follow fixed physical-unit rules keyed by the feature-name prefix and
    statistic suffix, then jittered by a config-seeded deterministic choice so the
    normalization remains fully reproducible and safe under incremental replacement.
    """
    scales = np.empty(len(feature_names), dtype=np.float64)
    for idx, name in enumerate(feature_names):
        rng = deterministic_rng(f"calibration|{config_scope_hash}|{idx}|{name}")
        base = _nominal_scale(name)
        jitter = float(rng.choice(np.array([0.75, 1.0, 1.25, 1.5], dtype=np.float64)))
        scales[idx] = base * jitter
    return scales


def _nominal_scale(name: str) -> float:
    if name.startswith("f0"):
        if name.endswith(".var"):
            return 4096.0
        return 512.0
    if name.startswith("energy_db"):
        if name.endswith(".var"):
            return 8.0
        return 32.0
    if name.startswith("cep"):
        if name.endswith(".var"):
            return 8.0
        return 16.0
    return 16.0


def integer_l2_normalize(embedding_qn: np.ndarray, fp: FixedPoint) -> dict[str, Any]:
    """Deterministic integer L2 normalization keeping the embedding on the Qn scale.

    Returns the unit-norm integer embedding plus the integer norm intermediates.
    """
    x = np.asarray(embedding_qn, dtype=np.int64)
    norm_sq = int(np.dot(x, x))
    norm = math.isqrt(norm_sq)
    if norm == 0:
        return {
            "embedding_int": x.copy(),
            "norm_sq": norm_sq,
            "norm": 0,
            "embedding_float": fp.dequantize(x),
        }
    scaled = dround_array(x.astype(np.float64) * fp.scale / norm)
    return {
        "embedding_int": scaled.astype(np.int64),
        "norm_sq": norm_sq,
        "norm": norm,
        "embedding_float": fp.dequantize(scaled.astype(np.int64)),
    }
