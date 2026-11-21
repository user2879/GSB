"""Domain errors with stable machine-readable codes and field paths."""

from __future__ import annotations

from typing import Any


class RepairError(Exception):
    """Base class for all expected domain failures."""

    code = "E_STATE"

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        details: dict[str, Any] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.details = details or {}
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            error["path"] = self.path
        if self.details:
            error["details"] = self.details
        return {"ok": False, "error": error}


class FormatError(RepairError):
    code = "E_FORMAT"


class BlockError(RepairError):
    code = "E_BLOCK"


class IntegrityError(RepairError):
    code = "E_INTEGRITY"


class BudgetError(RepairError):
    code = "E_BUDGET"


class ReplayError(RepairError):
    code = "E_REPLAY"


class StateError(RepairError):
    code = "E_STATE"


class NetworkError(RepairError):
    code = "E_NETWORK"


class EnvironmentError(RepairError):
    code = "E_ENV"


ERROR_TYPES = {
    "E_FORMAT": FormatError,
    "E_BLOCK": BlockError,
    "E_INTEGRITY": IntegrityError,
    "E_BUDGET": BudgetError,
    "E_REPLAY": ReplayError,
    "E_STATE": StateError,
    "E_NETWORK": NetworkError,
    "E_ENV": EnvironmentError,
}


def error_from_dict(payload: dict[str, Any]) -> RepairError:
    code = str(payload.get("code", "E_STATE"))
    cls = ERROR_TYPES.get(code, StateError)
    return cls(
        str(payload.get("message", "remote error")),
        path=payload.get("path"),
        details=payload.get("details") if isinstance(payload.get("details"), dict) else None,
        code=code,
    )
