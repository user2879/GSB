"""Explicit cross-version compatibility table."""

from __future__ import annotations

from typing import Any


VERSION_COMPAT: dict[int, dict[str, Any]] = {
    1: {
        "name": "v1-fixed-width-text",
        "score_scale": 1_000,
        "total_field": "permille-integer",
        "level_field": "code",
        "frame_summaries": False,
        "bypass": False,
    },
    2: {
        "name": "v2-json",
        "score_scale": 1_000_000,
        "total_field": "float-6dp",
        "level_field": "name",
        "frame_summaries": False,
        "bypass": True,
    },
    3: {
        "name": "v3-binary-partition",
        "score_scale_base": 10,
        "total_field": "fixed-integer",
        "level_field": "code",
        "frame_summaries": True,
        "bypass": True,
    },
}


FIXED_SCORE_DECIMALS = 6


def fixed_factor(bits: int) -> int:
    """Micro-score units represented by one v3 fixed integer unit."""

    if not 0 <= bits <= FIXED_SCORE_DECIMALS:
        raise ValueError("fixed_decimal_bits must be in 0..6")
    return 10 ** (FIXED_SCORE_DECIMALS - bits)


def micro_to_fixed(micro: int, bits: int) -> int:
    factor = fixed_factor(bits)
    return (micro + factor // 2) // factor


def fixed_to_micro(fixed: int, bits: int) -> int:
    return fixed * fixed_factor(bits)

