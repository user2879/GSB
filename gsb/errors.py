from __future__ import annotations

from typing import Any


class GSBError(Exception):
    code = "E_GSB"

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        position: str | int | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.position = position
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "error": self.message,
        }
        if self.field is not None:
            payload["field"] = self.field
        if self.position is not None:
            payload["position"] = self.position
        payload.update(self.details)
        return payload


class SpecError(GSBError):
    code = "E_SPEC"


class PortError(GSBError):
    code = "E_PORT"


class ProtocolTimeout(GSBError):
    code = "E_TIMEOUT"


class ProtocolError(GSBError):
    code = "E_PROTOCOL"


class LeaseError(GSBError):
    code = "E_LEASE"


class DivergeError(GSBError):
    code = "E_DIVERGE"


class VerifyError(GSBError):
    code = "E_VERIFY"


class NodeUnavailable(GSBError):
    code = "E_NODE"
