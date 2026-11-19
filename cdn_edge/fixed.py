from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .errors import PipelineError

TRUNCATION_DIRECTION = "floor_toward_negative_infinity"
MAX_GRID_POWER = 12


@dataclass(frozen=True, order=True)
class FixedGrid:
    """One common decimal fixed-point grid, S = 10**power.

    All non-negative metrics are quantized to floor(value * S).  Multiplication
    and addition remain integer operations.  Signed multiplication is avoided
    intentionally so that the stated one-sided truncation bound is valid.
    """

    power: int

    def __post_init__(self) -> None:
        validate_grid_power(self.power)

    @property
    def scale(self) -> int:
        return 10 ** self.power

    @property
    def truncation(self) -> str:
        return TRUNCATION_DIRECTION

    def as_dict(self) -> dict[str, int | str]:
        return {
            "power": self.power,
            "scale": self.scale,
            "truncation": self.truncation,
        }

    def q_ratio(self, numerator: int, denominator: int) -> int:
        if denominator <= 0:
            raise ValueError("denominator must be positive")
        if numerator < 0:
            raise ValueError("negative metrics require a signed quantization scheme")
        return (numerator * self.scale) // denominator

    def q_ratio_fraction(self, value: Fraction) -> int:
        if value < 0:
            raise ValueError("negative metrics require a signed quantization scheme")
        return (value.numerator * self.scale) // value.denominator

    def q_average_ms(self, total_ms: int, sample_count: int) -> int:
        """Quantize an average duration in seconds from an integer millisecond sum."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if total_ms < 0:
            raise ValueError("total_ms must be non-negative")
        return (total_ms * self.scale) // (1000 * sample_count)

    def quantized(self, value: Fraction) -> Fraction:
        return Fraction(self.q_ratio_fraction(value), self.scale)

    @property
    def single_value_error_bound_units(self) -> int:
        # floor error is in [0, 1/S), represented as at most one grid unit.
        return 1

    @property
    def ratio_comparison_error_bound_units(self) -> int:
        # |qa(S*x)-qb(S*y)| < 2 for arbitrary real x,y.
        return 2

    def compare_with_bound(self, left_units: int, right_units: int) -> dict[str, object]:
        delta = left_units - right_units
        within = abs(delta) <= self.ratio_comparison_error_bound_units
        return {
            "delta_units": delta,
            "abs_delta_units": abs(delta),
            "bound_units": self.ratio_comparison_error_bound_units,
            "within_error_bound": within,
            "ordering": "equal" if within else ("left_greater" if delta > 0 else "right_greater"),
        }

    def error_units_from_fraction(self, error: Fraction) -> int:
        """Ceiling of a non-negative real error, expressed in grid units."""
        if error < 0:
            raise ValueError("error must be non-negative")
        scaled = error * self.scale
        return (scaled.numerator + scaled.denominator - 1) // scaled.denominator


def validate_grid_power(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PipelineError(
            "E_GRID",
            "grid power must be an integer exponent of 10",
            path="grid",
            field="power",
            details={"received": value},
        )
    if value < 0 or value > MAX_GRID_POWER:
        raise PipelineError(
            "E_GRID",
            f"grid power must be between 0 and {MAX_GRID_POWER}",
            path="grid",
            field="power",
            details={"received": value, "min": 0, "max": MAX_GRID_POWER},
        )
    return value


def parse_grid_power(text: str) -> int:
    try:
        value = int(text, 10)
    except (TypeError, ValueError) as exc:
        raise PipelineError(
            "E_GRID",
            "grid must be a base-10 integer exponent",
            path="grid",
            field="power",
            details={"received": text},
        ) from exc
    return validate_grid_power(value)


def fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def stable_descending_rank(
    values: Iterable[tuple[str, int]], grid: FixedGrid, *, tie_key: str = "name"
) -> list[dict[str, object]]:
    """Error-aware, stable ranking for equal-kind fixed-point statistics.

    Values whose pairwise distance is at most the comparison bound form a
    tie group.  Ties are resolved by the item name, never by input order.
    """
    items = sorted(values, key=lambda item: (-item[1], item[0]))
    groups: list[list[tuple[str, int]]] = []
    for item in items:
        if not groups:
            groups.append([item])
            continue
        candidate_values = [entry[1] for entry in groups[-1]] + [item[1]]
        if max(candidate_values) - min(candidate_values) <= grid.ratio_comparison_error_bound_units:
            groups[-1].append(item)
        else:
            groups.append([item])
    ranked: list[dict[str, object]] = []
    rank = 1
    for group in groups:
        group_names = sorted(group, key=lambda item: item[0])
        group_max = max(value for _, value in group)
        group_min = min(value for _, value in group)
        error_dominated = (
            len(group) > 1
            and (group_max - group_min) <= grid.ratio_comparison_error_bound_units
        )
        for name, value in group_names:
            ranked.append(
                {
                    "rank": rank,
                    tie_key: name,
                    "value_units": value,
                    "error_bound_dominated": error_dominated,
                    "tie_group_size": len(group),
                }
            )
        rank += 1
    return ranked
