"""Per-keypoint and global residual-evaluation accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import BudgetExceeded, E_POINT_QUOTA_EXHAUSTED, Failure


class ResidualCounter:
    """A monotonic per-keypoint quota counter.

    The counter applies to every active-observation reprojection evaluation,
    including the unquantized robust chain and both fixed-point diagnostic
    chains.  A request that would exceed the quota is rejected before the
    evaluation is performed.
    """

    def __init__(self, quota: int, *, keypoint_id: str, start_used: int = 0):
        self.quota = int(quota)
        self.keypoint_id = keypoint_id
        self.used = int(start_used)
        if self.used < 0 or self.used > self.quota:
            raise ValueError("start_used must lie in [0, quota]")

    @property
    def remaining(self) -> int:
        return self.quota - self.used

    def charge(self, amount: int, *, stage: str, details: dict[str, Any] | None = None) -> int:
        amount = int(amount)
        if amount < 0:
            raise ValueError("charge amount must be non-negative")
        if self.used + amount > self.quota:
            payload = {
                "used_before": self.used,
                "requested": amount,
                "quota": self.quota,
                "stage": stage,
            }
            if details:
                payload.update(details)
            raise BudgetExceeded(
                Failure(
                    code=E_POINT_QUOTA_EXHAUSTED,
                    keypoint_id=self.keypoint_id,
                    stage=stage,
                    message="per-keypoint residual evaluation quota would be exceeded",
                    details=payload,
                ),
                {
                    "quota": self.quota,
                    "used": self.used,
                    "requested": amount,
                },
            )
        self.used += amount
        return self.used


@dataclass
class BudgetLedger:
    initial: int
    remaining: int
    entries: list[dict[str, Any]] = field(default_factory=list)
    exhausted: bool = False

    @classmethod
    def create(cls, initial: int) -> "BudgetLedger":
        return cls(initial=int(initial), remaining=int(initial), entries=[], exhausted=False)

    def can_charge(self, amount: int) -> bool:
        return int(amount) <= self.remaining

    def charge(self, keypoint_id: str, amount: int, *, purpose: str = "point_solve") -> dict[str, Any]:
        amount = int(amount)
        if amount < 0:
            raise ValueError("amount must be non-negative")
        if amount > self.remaining:
            self.exhausted = True
            raise ValueError("global residual budget exhausted")
        before = self.remaining
        self.remaining -= amount
        entry = {
            "sequence": len(self.entries) + 1,
            "keypoint_id": keypoint_id,
            "purpose": purpose,
            "before": before,
            "amount": amount,
            "after": self.remaining,
        }
        self.entries.append(entry)
        if self.remaining == 0:
            self.exhausted = True
        return entry

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": "reprojection_evaluation",
            "initial": self.initial,
            "consumed": self.initial - self.remaining,
            "remaining": self.remaining,
            "exhausted": self.exhausted,
            "ledger": self.entries,
        }
