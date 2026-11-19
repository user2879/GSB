"""Fixed-point arithmetic and deterministic error-bound propagation.

All multiplication and addition for derived configuration items are performed on
Python integers against a shared grid.  Truncation is always *toward zero*.

Error bounds are kept as exact :class:`fractions.Fraction` values so the
authoritative replay can reproduce them bit-for-bit; only the final rendered
report turns them into decimals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from .grid import Grid, _to_fraction


@dataclass(frozen=True)
class Fixed:
    """A real value stored as ``mantissa`` on a grid with scale ``10**q``."""

    q: int
    mantissa: int

    @property
    def scale(self) -> int:
        return 10**self.q

    @property
    def real(self) -> Fraction:
        return Fraction(self.mantissa, self.scale)

    def to_dict(self) -> dict[str, int]:
        return {"q": self.q, "mantissa": self.mantissa}

    def decimal_text(self) -> str:
        return f"{float(self.real):.{self.q}f}"


@dataclass(frozen=True)
class Term:
    upstream: str
    coefficient: Fixed  # coefficient quantized on the same grid


@dataclass
class QuantizedValue:
    fixed: Fixed
    trunc_error: Fraction  # exact quantization error bound of a leaf input


@dataclass
class Contribution:
    source: str           # immediate upstream key, or a local-trunc marker
    amount: Fraction
    kind: str             # input | local-trunc | offset


@dataclass
class PathContribution:
    path: list[str]       # leaf ... -> this key, propagation route
    leaf: str
    amount: Fraction
    level: int
    share_pct: Fraction


@dataclass
class ErrorResult:
    key: str
    total: Fraction
    immediate: list[Contribution] = field(default_factory=list)
    paths: list[PathContribution] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "total_bound": f"{self.total}",
            "total_decimal": format_fraction(self.total, 12),
            "immediate": [
                {
                    "source": c.source,
                    "kind": c.kind,
                    "amount": f"{c.amount}",
                    "share_pct": f"{_share(c.amount, self.total)}",
                }
                for c in self.immediate
            ],
            "paths": [
                {
                    "path": path_contrib.path,
                    "leaf": path_contrib.leaf,
                    "level": path_contrib.level,
                    "amount": f"{path_contrib.amount}",
                    "share_pct": f"{path_contrib.share_pct}",
                }
                for path_contrib in self.paths
            ],
        }


def _share(amount: Fraction, total: Fraction) -> Fraction:
    if total == 0:
        return Fraction(0)
    return amount * 100 / total


def format_fraction(value: Fraction, digits: int = 12) -> str:
    """Render an exact fraction as a finite, rounded decimal string."""
    scale = 10**digits
    rounded = (value.numerator * scale) // value.denominator  # >= 0 here
    whole, frac = divmod(rounded, scale)
    return f"{whole}.{str(frac).zfill(digits)}"


def quantize_value(number: int | float | str | Fraction, grid: Grid) -> QuantizedValue:
    exact = _to_fraction(number)
    mantissa = grid.quantize_int(exact)
    return QuantizedValue(
        fixed=Fixed(q=grid.q, mantissa=mantissa),
        trunc_error=abs(exact - Fraction(mantissa, grid.scale)),
    )


def quantize_coefficient(number: int | float | str | Fraction, grid: Grid) -> Fixed:
    return Fixed(q=grid.q, mantissa=grid.quantize(_to_fraction(number)))


def _trunc_div_toward_zero(numerator: int, denominator: int) -> tuple[int, int]:
    """Integer division toward zero; returns (quotient, absolute remainder)."""
    quotient = abs(numerator) // denominator
    remainder = abs(numerator) - quotient * denominator
    if numerator < 0:
        quotient = -quotient
    return quotient, remainder


def derive_with_values(
    key: str,
    ordered_upstream: list[str],
    values: dict[str, Fixed],
    coefficients: dict[str, Fixed],
    offset: int | float | str | Fraction,
    grid: Grid,
) -> tuple[Fixed, list[Contribution]]:
    """Integer-only linear derivation with exact local truncation errors."""
    scale = grid.scale
    accum = 0
    local: list[Contribution] = []

    offset_exact = _to_fraction(offset)
    offset_mantissa = grid.quantize_int(offset_exact)
    offset_error = abs(offset_exact - Fraction(offset_mantissa, scale))
    if offset_error != 0:
        local.append(Contribution(f"{key}#offset", offset_error, "offset"))
    accum += offset_mantissa

    for upstream in ordered_upstream:
        coeff = coefficients[upstream]
        value = values[upstream]
        raw = coeff.mantissa * value.mantissa
        product_mantissa, remainder = _trunc_div_toward_zero(raw, scale)
        accum += product_mantissa
        if remainder != 0:
            # remainder / scale^2 is the exact real truncation error.
            local.append(
                Contribution(
                    f"{key}#mult:{upstream}",
                    Fraction(remainder, scale * scale),
                    "local-trunc",
                )
            )

    return Fixed(q=grid.q, mantissa=accum), local


def synthesize_error(
    key: str,
    ordered_upstream: list[str],
    coefficients: dict[str, Fixed],
    upstream_results: dict[str, "ErrorResult"],
    leaf_errors: dict[str, Fraction],
    local: list[Contribution],
) -> ErrorResult:
    """Weighted error-bound synthesis along one derivation node.

    The immediate bound is ``sum(|coeff_i| * B_i)`` plus this node's own
    truncation terms.  Multi-level paths are flattened recursively so each leaf
    contribution carries its full propagation route, depth level and share.
    """
    immediate: list[Contribution] = list(local)
    for upstream in ordered_upstream:
        weight = abs(coefficients[upstream].real)
        bound = upstream_results[upstream].total
        immediate.append(Contribution(upstream, weight * bound, "input"))

    # Leaf input quantization errors attach directly to non-derived keys.
    for leaf, bound in leaf_errors.items():
        if bound != 0 and leaf not in upstream_results:
            immediate.append(Contribution(leaf, bound, "input"))

    total = sum((c.amount for c in immediate), Fraction(0))

    paths: list[PathContribution] = []
    for upstream in ordered_upstream:
        weight = abs(coefficients[upstream].real)
        upstream_result = upstream_results[upstream]
        if upstream_result.paths:
            for sub in upstream_result.paths:
                amount = weight * sub.amount
                paths.append(
                    PathContribution(
                        path=sub.path + [key],
                        leaf=sub.leaf,
                        amount=amount,
                        level=sub.level + 1,
                        share_pct=_share(amount, total),
                    )
                )
        else:
            amount = weight * upstream_result.total
            paths.append(
                PathContribution(
                    path=[upstream, key],
                    leaf=upstream,
                    amount=amount,
                    level=1,
                    share_pct=_share(amount, total),
                )
            )
    for contribution in local:
        paths.append(
            PathContribution(
                path=[contribution.source, key],
                leaf=contribution.source,
                amount=contribution.amount,
                level=0,
                share_pct=_share(contribution.amount, total),
            )
        )

    paths.sort(key=lambda p: (-p.amount, p.leaf, tuple(p.path)))
    immediate.sort(key=lambda c: (-c.amount, c.source))
    return ErrorResult(key=key, total=total, immediate=immediate, paths=paths)
