from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constants import WATERMARK_RULE_METADATA


@dataclass(frozen=True)
class Watermark:
    capacity_slots: int
    soft_permille: int
    hard_permille: int
    soft_limit: int
    hard_limit: int
    capacity_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "capacity_version": self.capacity_version,
            "capacity_slots": self.capacity_slots,
            "soft_permille": self.soft_permille,
            "hard_permille": self.hard_permille,
            "soft_limit": self.soft_limit,
            "hard_limit": self.hard_limit,
            "rounding": WATERMARK_RULE_METADATA["threshold_rounding"],
            "boundary_rule": WATERMARK_RULE_METADATA["boundary_rule"],
        }


def rounded_permille_limit(capacity_slots: int, permille: int) -> int:
    """Round ``slots * permille / 1000`` with deterministic half-up rounding."""
    if isinstance(capacity_slots, bool) or not isinstance(capacity_slots, int):
        raise TypeError("capacity_slots must be an integer")
    if isinstance(permille, bool) or not isinstance(permille, int):
        raise TypeError("permille must be an integer")
    if capacity_slots <= 0:
        raise ValueError("capacity_slots must be positive")
    if not 0 <= permille <= 1000:
        raise ValueError("permille must be in [0, 1000]")

    # Python integers keep arbitrary-size JSON inputs exact; the result is a
    # fixed-point permille threshold and never uses floating point.
    scaled = capacity_slots * permille
    quotient, remainder = divmod(int(scaled), 1000)
    if 2 * remainder >= 1000:
        quotient += 1
    return int(quotient)


def build_watermark(
    *,
    capacity_slots: int,
    capacity_version: int,
    soft_permille: int,
    hard_permille: int,
) -> Watermark:
    if soft_permille > hard_permille:
        raise ValueError("soft_permille must not exceed hard_permille")
    soft_limit = rounded_permille_limit(capacity_slots, soft_permille)
    hard_limit = rounded_permille_limit(capacity_slots, hard_permille)
    return Watermark(
        capacity_slots=capacity_slots,
        soft_permille=soft_permille,
        hard_permille=hard_permille,
        soft_limit=soft_limit,
        hard_limit=hard_limit,
        capacity_version=capacity_version,
    )


def watermark_state(occupancy: int, watermark: Watermark) -> str:
    if occupancy >= watermark.hard_limit:
        return "hard"
    if occupancy >= watermark.soft_limit:
        return "soft_warning"
    return "normal"


def projected_hard_rejected(occupancy: int, requested_slots: int, hard_limit: int) -> bool:
    return occupancy + requested_slots > hard_limit
