from __future__ import annotations

from typing import Any


class HopLedgerError(Exception):
    """Base error carrying the stable external error code."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, code: str | None = None, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        self.details = details
        self.field = details.get("field")

    def to_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        payload.update(self.details)
        return payload


class SpecError(HopLedgerError):
    code = "E_SPEC"

    def __init__(
        self,
        message: str,
        *,
        hop_id: str | None = None,
        field: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message, hop_id=hop_id, field=field, **details)
        self.hop_id = hop_id
        self.field = field


class PortInUseError(HopLedgerError):
    code = "E_PORT"


class TimeoutError(HopLedgerError):
    code = "E_TIMEOUT"


class FieldError(HopLedgerError):
    code = "E_FIELD"


class ReplayError(HopLedgerError):
    code = "E_REPLAY"


class ForwardInjectedError(HopLedgerError):
    code = "E_FORWARD_INJECTED"
