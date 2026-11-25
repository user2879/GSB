from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

from .canonical import ceil_div

BP_SCALE = 10_000
I64_MAX = (1 << 63) - 1

# Declared arithmetic contract:
# 1. All inputs are non-negative integers.
# 2. Division uses floor toward -infinity, which is floor for this domain.
# 3. Seats lost to floor division are distributed by descending remainder and
#    then ascending stable identifier.
# 4. Multiplication/addition are saturated to signed int64 before persistence.
ROUNDING_RULE = "floor_nonnegative_then_descending_remainder_then_id"
SATURATION_RULE = "saturate_to_int64_max"

K = TypeVar("K")


def sat_i64(value: int) -> int:
    if not isinstance(value, int):
        raise TypeError("fixed-point value must be an integer")
    if value > I64_MAX:
        return I64_MAX
    if value < -I64_MAX - 1:
        return -I64_MAX - 1
    return value


def floor_bp(numerator: int) -> int:
    """Floor ``numerator / 10000`` for non-negative integer numerators."""

    if numerator < 0:
        raise ValueError("bp fixed-point operation expects non-negative input")
    return numerator // BP_SCALE


def allocate_integer(total: int, weights: Mapping[K, int], order: Sequence[K] | None = None) -> dict[K, int]:
    """Allocate exactly ``total`` integer seats by the declared stable rule."""

    if total < 0:
        raise ValueError("total must be non-negative")
    ids = list(order) if order is not None else sorted(weights)
    missing = [key for key in ids if key not in weights]
    if missing:
        raise KeyError(f"missing weights for ids: {missing!r}")
    if len(ids) != len(set(ids)):
        raise ValueError("allocation ids must be unique")

    safe_weights = {key: max(0, int(weights[key])) for key in ids}
    weight_sum = sat_i64(sum(safe_weights.values()))
    if weight_sum <= 0:
        if total:
            raise ValueError("cannot allocate positive total with zero total weight")
        return {key: 0 for key in ids}

    products = {key: sat_i64(total * safe_weights[key]) for key in ids}
    bases = {key: products[key] // weight_sum for key in ids}
    remainders = {key: products[key] % weight_sum for key in ids}
    allocated = sat_i64(sum(bases.values()))
    seats = total - allocated
    if seats < 0:
        raise AssertionError("floor allocation exceeded total")

    ranked = sorted(ids, key=lambda key: (-remainders[key], str(key)))
    if seats > len(ids):
        raise AssertionError("remaining seats cannot exceed number of buckets")
    for key in ranked[:seats]:
        bases[key] += 1
    if sum(bases.values()) != total:
        raise AssertionError("allocation did not preserve total")
    return bases


def weights_to_target_bp(weights_bp: Mapping[str, int], domain_ids: Sequence[str]) -> dict[str, int]:
    """Convert relative integer weights to basis points summing to 10000."""

    return allocate_integer(BP_SCALE, weights_bp, list(domain_ids))


def counts_to_actual_bp(counts: Mapping[str, int], domain_ids: Sequence[str]) -> dict[str, int]:
    total = sum(max(0, int(counts.get(domain, 0))) for domain in domain_ids)
    if total == 0:
        return {domain: 0 for domain in domain_ids}
    return allocate_integer(BP_SCALE, {d: max(0, int(counts.get(d, 0))) for d in domain_ids}, list(domain_ids))


def integer_bp_fraction(intersection: int, union: int) -> int:
    if union <= 0:
        return 0
    return sat_i64(BP_SCALE * max(0, intersection)) // union


def ceil_bp_error_from_count(delta_count: int, total_count: int) -> int:
    if total_count <= 0:
        return BP_SCALE
    return ceil_div(BP_SCALE * max(0, delta_count), total_count)
