"""Integer-domain fixed-point similarity and quantization."""

from __future__ import annotations


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (2 * numerator + denominator) // (2 * denominator)


def quantize_units(raw_units: int, scale: int, levels: int) -> int:
    """Quantize fixed-point units onto ``levels + 1`` deterministic rungs."""
    if scale <= 0 or levels <= 0:
        raise ValueError("scale and levels must be positive")
    rung = _round_half_up_ratio(raw_units * levels, scale)
    rung = max(0, min(levels, rung))
    return _round_half_up_ratio(rung * scale, levels)


def similarity_units(matches: int, dimension: int, scale: int, levels: int) -> int:
    if dimension <= 0:
        raise ValueError("signature dimension must be positive")
    raw = _round_half_up_ratio(matches * scale, dimension)
    return quantize_units(raw, scale, levels)


def format_fixed(units: int, decimal_places: int) -> str:
    scale = 10**decimal_places
    whole, fraction = divmod(units, scale)
    return f"{whole}.{fraction:0{decimal_places}d}"

