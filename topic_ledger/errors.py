from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_INVALID_INPUT = "E_INVALID_INPUT"
E_DUPLICATE_DELIVERY = "E_DUPLICATE_DELIVERY"
E_STALE_DIGEST = "E_STALE_DIGEST"
E_LEASE_GENERATION = "E_LEASE_GENERATION"
E_LEASE_EXPIRED = "E_LEASE_EXPIRED"
E_NODE_BUSY = "E_NODE_BUSY"
E_NODE_LEASE_LIMIT = "E_NODE_LEASE_LIMIT"
E_BUDGET_PENDING = "E_BUDGET_PENDING"
E_ROUND_LIMIT = "E_ROUND_LIMIT"
E_LEDGER_MISMATCH = "E_LEDGER_MISMATCH"
E_LEDGER_FROZEN = "E_LEDGER_FROZEN"
E_STALE_SUMMARY = "E_STALE_SUMMARY"
E_SNAPSHOT_VERSION = "E_SNAPSHOT_VERSION"
E_WORKER_PROTOCOL = "E_WORKER_PROTOCOL"


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    field: str | None = None
    batch_id: str | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        prefix = f"{self.code}: {self.message}"
        if self.field:
            prefix += f" (field={self.field})"
        if self.batch_id:
            prefix += f" (batch_id={self.batch_id})"
        return prefix

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.field is not None:
            result["field"] = self.field
        if self.batch_id is not None:
            result["batch_id"] = self.batch_id
        if self.details:
            result["details"] = self.details
        return result


def invalid_input(message: str, field: str, **details: Any) -> BusinessError:
    return BusinessError(E_INVALID_INPUT, message, field=field, details=details or None)
