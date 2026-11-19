"""SHA-256 purpose-separated seeds and a versioned deterministic PRNG."""

from __future__ import annotations

import hashlib
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")

MASK64 = (1 << 64) - 1


def derive_seed(master_seed: int, purpose: str, *parts: object, offset: int = 0) -> int:
    """Derive a 64-bit seed without process-global or dictionary state."""

    fields = ["level-workbench/seed-v1", str(master_seed), purpose]
    fields.extend(str(part) for part in parts)
    fields.append(f"offset={offset}")
    digest = hashlib.sha256("|".join(fields).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def seed_record(master_seed: int, purpose: str, seed: int, offset: int = 0, **parts: object) -> dict[str, object]:
    record: dict[str, object] = {
        "purpose": purpose,
        "master_seed": master_seed,
        "offset": offset,
        "seed_u64": seed,
        "seed_hex": f"{seed:016x}",
    }
    if parts:
        record["parts"] = {key: parts[key] for key in sorted(parts)}
    return record


class DeterministicRNG:
    """SplitMix64 with unbiased bounded sampling and canonical shuffling.

    Python's ``random`` module is intentionally not used: this keeps output
    stable across Python implementations and independent of hashing order.
    """

    def __init__(self, seed: int):
        self._state = seed & MASK64

    def next_u64(self) -> int:
        self._state = (self._state + 0x9E3779B97F4A7C15) & MASK64
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return (value ^ (value >> 31)) & MASK64

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper bound must be positive")
        if upper == 1:
            return 0
        limit = (1 << 64) - ((1 << 64) % upper)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % upper

    def choice(self, values: Sequence[T]) -> T:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return values[self.randbelow(len(values))]

    def shuffle(self, values: list[T]) -> None:
        for index in range(len(values) - 1, 0, -1):
            other = self.randbelow(index + 1)
            values[index], values[other] = values[other], values[index]

    def weighted_index(self, weights: Sequence[float]) -> int:
        units = [_weight_units(weight) for weight in weights]
        total = sum(units)
        if total <= 0:
            raise ValueError("weights have no positive mass")
        needle = self.randbelow(total)
        running = 0
        for index, unit in enumerate(units):
            running += unit
            if needle < running:
                return index
        return len(units) - 1


def _weight_units(weight: float) -> int:
    if weight < 0:
        raise ValueError("negative weight")
    return int((Decimal(str(weight)) * Decimal(1_000_000_000)).to_integral_value(rounding=ROUND_HALF_UP))


def deterministic_weighted_order(
    names: Iterable[str], weights: dict[str, float], rng: DeterministicRNG
) -> list[str]:
    """Return a random order while respecting relative integer weight mass."""

    ordered = sorted(names)
    remaining = ordered[:]
    result: list[str] = []
    while remaining:
        local_weights = [weights[name] for name in remaining]
        index = rng.weighted_index(local_weights)
        result.append(remaining.pop(index))
    return result
