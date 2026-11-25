from __future__ import annotations

from typing import Any


class ReconError(Exception):
    """Base class for structured ReconFlow failures."""

    code = "E_RECON"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        payload.update(self.details)
        return payload


class SpecError(ReconError):
    code = "E_SPEC"

    def __init__(
        self,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        field: str | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(message, path=path, line=line, field=field, value=value)
        self.path = path
        self.line = line
        self.field = field


class DivergeError(ReconError):
    code = "E_DIVERGE"


class ReplayError(ReconError):
    code = "E_REPLAY"


class PortBusyError(ReconError):
    code = "E_PORT"


class SocketTimeoutFailure(ReconError):
    code = "E_TIMEOUT"


class ReceiptFieldError(ReconError):
    code = "E_FIELD"


class BudgetError(ReconError):
    code = "E_BUDGET"
