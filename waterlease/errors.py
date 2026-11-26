from __future__ import annotations

from typing import Any


class WaterLeaseError(RuntimeError):
    """Structured error carrying one of the project error codes."""

    code = "E_INTERNAL"

    def __init__(
        self,
        message: str,
        *,
        path: str = "$",
        details: dict[str, Any] | None = None,
        code_override: str | None = None,
    ) -> None:
        code = code_override or self.code
        self.code = code
        super().__init__(f"{code} {path}: {message}")
        self.message = message
        self.path = path
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "details": self.details,
        }


class SpecError(WaterLeaseError):
    code = "E_SPEC"


class PortInUseError(WaterLeaseError):
    code = "E_PORT"


class RpcTimeoutError(WaterLeaseError):
    code = "E_TIMEOUT"


class LeaseConflictError(WaterLeaseError):
    code = "E_LEASE"


class DivergenceError(WaterLeaseError):
    code = "E_DIVERGE"


class CapacityError(WaterLeaseError):
    code = "E_CAPACITY"


class LedgerError(WaterLeaseError):
    code = "E_LEDGER"
