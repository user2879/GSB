"""Deterministic bit-width ladders."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    ENCODING_WIDTHS,
    FIXED_POINT_WIDTHS,
    QUANTIZATION_LEVELS,
    Strategy,
    width_gear_fingerprint,
)


@dataclass(frozen=True)
class Gear:
    gear_id: str
    encoding_width: int
    fixed_point_width: int
    quantization_levels: int
    fingerprint: str


def make_gear(
    encoding_width: int,
    fixed_point_width: int,
    quantization_levels: int,
) -> Gear:
    return Gear(
        gear_id=f"e{encoding_width}_f{fixed_point_width}_q{quantization_levels}",
        encoding_width=int(encoding_width),
        fixed_point_width=int(fixed_point_width),
        quantization_levels=int(quantization_levels),
        fingerprint=width_gear_fingerprint(
            encoding_width, fixed_point_width, quantization_levels
        ),
    )


def encoding_ladder(strategy: Strategy) -> list[Gear]:
    widths = set(ENCODING_WIDTHS)
    widths.add(strategy.encoding_width)
    return [
        make_gear(width, strategy.fixed_point_width, strategy.quantization_levels)
        for width in sorted(widths)
    ]


def quantization_ladder(strategy: Strategy) -> list[Gear]:
    pairs = {(8, 8), (16, 16), (32, 32)}
    pairs.add((strategy.fixed_point_width, strategy.quantization_levels))
    valid_pairs = sorted(
        (width, levels)
        for width, levels in pairs
        if width in FIXED_POINT_WIDTHS and levels in QUANTIZATION_LEVELS
    )
    return [
        make_gear(strategy.encoding_width, width, levels)
        for width, levels in valid_pairs
    ]


def adjacent_pairs(gears: list[Gear]) -> list[tuple[Gear, Gear]]:
    return list(zip(gears, gears[1:]))
