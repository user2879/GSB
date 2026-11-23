"""Exact request-local normalisation and fixed-point quantisation."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

from .canonical import decimal_text, fixed_from_real, fraction_text
from .models import FeatureCatalog, RawCandidate


def _decimal_sqrt(value: Fraction) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 80
        numerator = Decimal(value.numerator)
        denominator = Decimal(value.denominator)
        if numerator <= 0:
            return Decimal(0)
        return (numerator / denominator).sqrt()


def _normalised_fraction(
    value: Fraction,
    method: str,
    minimum: Fraction,
    maximum: Fraction,
    mean: Fraction,
    variance: Fraction,
) -> tuple[Fraction, bool]:
    if method == "minmax":
        denominator = maximum - minimum
        if denominator == 0:
            return Fraction(0), True
        return (value - minimum) / denominator, False
    if variance == 0:
        return Fraction(0), True
    with localcontext() as ctx:
        ctx.prec = 80
        numerator = Decimal((value - mean).numerator) / Decimal((value - mean).denominator)
        std = _decimal_sqrt(variance)
        result = numerator / std
        return Fraction(result), False
    raise AssertionError("unreachable")


def compute_normalization(
    candidates: tuple[RawCandidate, ...],
    catalog: FeatureCatalog,
    method: str,
    fixed_decimal_places: int,
    quantization_step: int,
) -> dict[str, Any]:
    """Compute stats and per-candidate fixed vectors for one request.

    ``candidates`` must already be sorted by item_id.  The returned object is
    cacheable and contains no command history or timing information.
    """

    vectors = {
        candidate.item_id: candidate.feature_vector(catalog) for candidate in candidates
    }
    item_ids = [candidate.item_id for candidate in candidates]
    feature_stats: dict[str, dict[str, str]] = {}
    zero_denominators: list[dict[str, Any]] = []
    normalised: dict[str, list[int]] = {item_id: [] for item_id in item_ids}

    for feature_index, feature_name in enumerate(catalog.feature_order):
        values = [Fraction(vector[feature_index]) for vector in vectors.values()]
        minimum = min(values) if values else Fraction(0)
        maximum = max(values) if values else Fraction(0)
        total = sum(values, Fraction(0))
        count = len(values)
        mean = total / count if count else Fraction(0)
        variance = (
            sum(((value - mean) ** 2 for value in values), Fraction(0)) / count
            if count
            else Fraction(0)
        )
        stats: dict[str, str] = {
            "group": catalog.group_for(feature_name) or "",
            "min": fraction_text(minimum),
            "max": fraction_text(maximum),
            "mean": fraction_text(mean),
            "variance": fraction_text(variance),
        }
        feature_stats[feature_name] = stats
        for item_id, vector in vectors.items():
            value = Fraction(vector[feature_index])
            normalised_value, zero = _normalised_fraction(
                value, method, minimum, maximum, mean, variance
            )
            if zero:
                zero_denominators.append(
                    {
                        "feature": feature_name,
                        "rule": "zero_denominator_to_zero"
                        if method == "minmax"
                        else "zero_variance_to_zero",
                    }
                )
            normalised[item_id].append(
                fixed_from_real(normalised_value, fixed_decimal_places, quantization_step)
            )

    dedup_zero: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in zero_denominators:
        key = (item["feature"], item["rule"])
        if key not in seen:
            seen.add(key)
            dedup_zero.append(item)

    return {
        "method": method,
        "feature_order": list(catalog.feature_order),
        "feature_dimension": catalog.dimension,
        "statistics": feature_stats,
        "zero_denominators": dedup_zero,
        "normalized_fixed": {
            item_id: normalised[item_id] for item_id in sorted(normalised)
        },
    }


def normalization_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Small human-readable projection used by command responses."""

    return {
        "method": entry["method"],
        "feature_dimension": entry["feature_dimension"],
        "zero_denominators": entry["zero_denominators"],
        "zero_denominator_count": len(entry["zero_denominators"]),
    }


def decimal_stat(value: str) -> str:
    numerator, denominator = value.split("/", 1)
    return decimal_text(Fraction(int(numerator), int(denominator)))
