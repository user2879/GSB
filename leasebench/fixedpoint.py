"""Integer fixed-point watermark arithmetic and explicit error bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Ratio:
    numerator: int
    denominator: int
    scale: int
    fixed: int
    quantization_error_fp: int
    borrow_return_error_fp: int

    @property
    def single_value_error_fp(self) -> int:
        return max(1, self.quantization_error_fp)

    @property
    def comparison_error_fp(self) -> int:
        return 2

    @property
    def total_comparison_error_fp(self) -> int:
        return self.comparison_error_fp + self.borrow_return_error_fp

    @property
    def dominant_error_segment(self) -> str:
        if self.borrow_return_error_fp > self.comparison_error_fp:
            return "BORROW_RETURN_DELAY"
        if self.borrow_return_error_fp:
            return "TIE_QUANTIZATION_AND_BORROW_DELAY"
        return "QUANTIZATION"


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return -((-numerator) // denominator)


def quantize_ratio(
    numerator: int,
    denominator: int,
    scale: int,
    *,
    near_borrow_units: int = 0,
) -> Ratio:
    """Quantize numerator/denominator with FLOOR_TOWARD_ZERO.

    `quantization_error_fp` is the conservative one-unit upper bound on a
    single value.  For comparison, each side contributes one unit, hence the
    fixed comparison bound of two units.  Near-deadline borrowed units add a
    separate worst-case segment measured in the same fixed-point units.
    """

    if numerator < 0 or denominator < 0 or near_borrow_units < 0:
        raise ValueError("ratio inputs must be non-negative")
    if scale <= 0:
        raise ValueError("scale must be positive")
    effective_denominator = max(1, denominator)
    product = int(numerator) * int(scale)
    if product > np.iinfo(np.int64).max:
        raise ValueError("fixed-point product exceeds int64 range")
    fixed = int(np.floor_divide(np.int64(product), np.int64(effective_denominator)))
    borrow_error = _ceil_div(
        int(near_borrow_units) * int(scale), effective_denominator
    )
    return Ratio(
        numerator=int(numerator),
        denominator=int(denominator),
        scale=int(scale),
        fixed=fixed,
        quantization_error_fp=1,
        borrow_return_error_fp=borrow_error,
    )


@dataclass(frozen=True)
class RankedRatio:
    key: str
    ratio: Ratio
    affected_by_error: bool


def rank_ratios(values: Iterable[tuple[str, Ratio]]) -> list[RankedRatio]:
    """Stable ordering: fixed-point value descending, then key ascending.

    Adjacent values whose difference is within the combined error bound are
    still ordered by the stable key and marked as error-affected.
    """

    ordered = sorted(values, key=lambda item: (-item[1].fixed, item[0]))
    result: list[RankedRatio] = []
    for index, (key, ratio) in enumerate(ordered):
        affected = False
        neighbours = []
        if index:
            neighbours.append(ordered[index - 1][1])
        if index + 1 < len(ordered):
            neighbours.append(ordered[index + 1][1])
        for other in neighbours:
            bound = ratio.total_comparison_error_fp + other.total_comparison_error_fp
            if abs(ratio.fixed - other.fixed) <= bound:
                affected = True
                break
        result.append(RankedRatio(key=key, ratio=ratio, affected_by_error=affected))
    return result

