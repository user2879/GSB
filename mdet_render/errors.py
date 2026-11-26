"""Business errors with stable, machine-readable codes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 2

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_LEASE_EXPIRED = "E_LEASE_EXPIRED"
E_LEASE_GENERATION = "E_LEASE_GENERATION"
