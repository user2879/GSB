from __future__ import annotations

from dataclasses import dataclass


# Diagnostic fixed-point words use a signed 64-bit saturation domain. Python
# integers remain exact intermediate values; only the declared word is clamped.
I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)

# Non-negative millisecond quantities are divided with floor division. For
# non-negative values this is truncation toward zero; negative defensive paths
# use mathematical floor (toward -infinity), as documented in docs/DESIGN.md.
TRUNCATION_DIRECTION = "floor_non_negative_toward_zero"
OVERFLOW_RULE = "saturate_to_signed_int64"


def saturate_i64(value: int) -> int:
    if value > I64_MAX:
        return I64_MAX
    if value < I64_MIN:
        return I64_MIN
    return value


def saturated_add(left: int, right: int) -> int:
    return saturate_i64(left + right)


def saturated_mul(left: int, right: int) -> int:
    return saturate_i64(left * right)


@dataclass(frozen=True)
class ScaledComponent:
    name: str
    milliseconds: int
    local_units: int
    units: int
    remainder_numerator: int
    source_scale_factor: int
    session_scale: int
    saturated: bool

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "name": self.name,
            "milliseconds": self.milliseconds,
            "local_units": self.local_units,
            "units": self.units,
            "remainder_numerator": self.remainder_numerator,
            "source_scale_factor": self.source_scale_factor,
            "session_scale": self.session_scale,
            "saturated": self.saturated,
        }


def quantize_component(
    name: str,
    milliseconds: int,
    source_scale_factor: int,
    session_scale: int,
) -> ScaledComponent:
    if milliseconds < 0:
        raise ValueError("millisecond quantities must be non-negative")
    if source_scale_factor <= 0 or session_scale <= 0:
        raise ValueError("scale factors must be positive")

    local_units = saturated_mul(milliseconds, source_scale_factor)
    numerator = saturated_mul(local_units, session_scale)
    units = saturate_i64(numerator // source_scale_factor)
    remainder_numerator = numerator % source_scale_factor
    saturated = (
        local_units == I64_MAX
        or numerator == I64_MAX
        or units == I64_MAX
    ) and milliseconds * source_scale_factor * session_scale > I64_MAX
    return ScaledComponent(
        name=name,
        milliseconds=milliseconds,
        local_units=local_units,
        units=units,
        remainder_numerator=remainder_numerator,
        source_scale_factor=source_scale_factor,
        session_scale=session_scale,
        saturated=saturated,
    )


def quantization_tick_bound_units(source_scale_factor: int, session_scale: int) -> int:
    """Conservative one-tick bound in session fixed-point units."""

    return saturate_i64((session_scale + source_scale_factor - 1) // source_scale_factor)


def apportion_basis_points(weights: list[int]) -> list[int]:
    """Apportion 10000 bp exactly, breaking ties by stable earlier position."""

    if not weights:
        return []
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    total = sum(weights)
    if total == 0:
        base = [0 for _ in weights]
        # Zero weights still receive zero share; callers handle indistinguishability.
        return base

    raw = [weight * 10_000 for weight in weights]
    shares = [value // total for value in raw]
    remainder = 10_000 - sum(shares)
    fractions = [value % total for value in raw]
    order = sorted(
        range(len(weights)),
        key=lambda index: (-fractions[index], index),
    )
    for index in order[:remainder]:
        shares[index] += 1
    return shares
