"""Deterministic fixed-point helpers.

Scores are accumulated in Python floats using a fixed operation order and are
quantized to a fixed-point integer representation (``round-half-up``) before
they are compared with the confidence bands.  The un-rounded reference value
is retained so drift can be reported.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np


def quantize_score(raw_score: float, scale: int) -> int:
    """Return ``raw_score * 10**scale`` rounded half-up as a fixed-point int."""
    if scale < 0:
        raise ValueError("fixed_point_scale must be non-negative")
    decimal_value = Decimal(repr(float(raw_score))) * Decimal(10) ** scale
    return int(decimal_value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def quantize_threshold(threshold: float, scale: int) -> int:
    """Quantize a configured band threshold with the same rounding rule."""
    return quantize_score(threshold, scale)


def classify_band(quantized: int, thresholds_q: list[int], labels: list[str]) -> str:
    """Classify a quantized score into a confidence band.

    Thresholds are assumed monotonically increasing.  A score exactly equal to a
    threshold is decided into the *higher* band (the band starting at that
    threshold).  ``labels`` has one more element than ``thresholds_q``.
    """
    band_index = 0
    for index, threshold in enumerate(thresholds_q):
        if quantized >= threshold:
            band_index = index + 1
        else:
            break
    return labels[band_index]


def is_boundary(quantized: int, thresholds_q: list[int]) -> bool:
    """True when the quantized score lands exactly on a band threshold."""
    return quantized in thresholds_q


def drift(raw_score: float, quantized: int, scale: int) -> float:
    """Signed difference between the quantized value and the raw reference."""
    raw_q = np.float64(raw_score) * np.float64(10.0**scale)
    return float(np.float64(quantized) - raw_q)
