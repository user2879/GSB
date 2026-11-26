"""Business error codes and compact error objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

MISSING = "__MISSING__"

E_CONFIG = "E_CONFIG"
E_DUP_WORKER_ID = "E_DUP_WORKER_ID"
E_DUP_PORT = "E_DUP_PORT"
E_BAD_QUOTA = "E_BAD_QUOTA"
E_BAD_LEASE_LIMIT = "E_BAD_LEASE_LIMIT"
E_UNKNOWN_LAYER = "E_UNKNOWN_LAYER"
E_BAD_DIMENSION = "E_BAD_DIMENSION"
E_BAD_GROUP_SIZE = "E_BAD_GROUP_SIZE"
E_GROUP_NOT_DIVISIBLE = "E_GROUP_NOT_DIVISIBLE"
E_NONFINITE_WEIGHT = "E_NONFINITE_WEIGHT"
E_NONFINITE_INPUT = "E_NONFINITE_INPUT"
E_BAD_QUANT_LEVEL = "E_BAD_QUANT_LEVEL"
E_BAD_FRAC_BITS = "E_BAD_FRAC_BITS"
E_DUP_BATCH = "E_DUP_BATCH"
E_BAD_BATCH = "E_BAD_BATCH"
E_BAD_SAMPLE = "E_BAD_SAMPLE"
E_BAD_COMMAND = "E_BAD_COMMAND"
E_BAD_RECEIPT = "E_BAD_RECEIPT"
E_PORT_IN_USE = "E_PORT_IN_USE"
E_PROTOCOL = "E_PROTOCOL"
E_WORKER_START = "E_WORKER_START"
E_LEASE_CAPACITY = "E_LEASE_CAPACITY"
E_LEASE_NOT_FOUND = "E_LEASE_NOT_FOUND"
E_STALE_GENERATION = "E_STALE_GENERATION"
E_LEASE_EXPIRED = "E_LEASE_EXPIRED"
E_RECONCILE_PENDING = "E_RECONCILE_PENDING"
E_DIMENSION = "E_DIMENSION"
E_QUANT_LEVEL = "E_QUANT_LEVEL"
E_LEDGER_CORRUPT = "E_LEDGER_CORRUPT"
E_LEDGER_VERSION = "E_LEDGER_VERSION"
E_COMMAND = "E_COMMAND"
E_WORKER_UNAVAILABLE = "E_WORKER_UNAVAILABLE"


@dataclass(frozen=True)
class BusinessIssue:
    """One validation or runtime issue. ``item`` identifies the skipped item."""

    code: str
    item: str
    path: str
    message: str
    skipped: bool = True
    injection_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if result["injection_id"] is None:
            result.pop("injection_id")
        return result


class QLabError(Exception):
    """Exception carrying a stable business code for protocol boundaries."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "code": self.code, "message": self.message, **self.details}


def issue(code: str, item: str, path: str, message: str, **kwargs: Any) -> BusinessIssue:
    return BusinessIssue(code=code, item=item, path=path, message=message, **kwargs)

