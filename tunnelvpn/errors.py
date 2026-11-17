"""Structured business errors shared by every component."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    OK = "OK"
    E_CONFIG = "E_CONFIG"
    E_ENCAP = "E_ENCAP"
    E_FRAG = "E_FRAG"
    E_PEER = "E_PEER"
    E_TIMEOUT = "E_TIMEOUT"
    E_DIGEST = "E_DIGEST"
    E_REPLAY = "E_REPLAY"
    E_STATE = "E_STATE"


class TunnelError(Exception):
    """A business error with machine-readable location and context.

    ``offset`` is always the first offending byte offset when a byte stream is
    involved.  ``field`` is the JSON path for configuration errors.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        field: str | None = None,
        tunnel_id: str | None = None,
        session_id: str | None = None,
        offset: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = ErrorCode(code).value
        self.message = message
        self.field = field
        self.tunnel_id = tunnel_id
        self.session_id = session_id
        self.offset = offset
        self.details = details or {}

    def to_dict(self, *, include_empty: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        values = {
            "field": self.field,
            "tunnel_id": self.tunnel_id,
            "session_id": self.session_id,
            "offset": self.offset,
        }
        for key, value in values.items():
            if include_empty or value is not None:
                payload[key] = value
        if include_empty or self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        bits = [self.code, self.message]
        if self.field:
            bits.append(f"field={self.field}")
        if self.tunnel_id:
            bits.append(f"tunnel={self.tunnel_id}")
        if self.session_id:
            bits.append(f"session={self.session_id}")
        if self.offset is not None:
            bits.append(f"offset={self.offset}")
        return " | ".join(bits)


def as_error_dict(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, TunnelError):
        return exc.to_dict()
    return {"code": ErrorCode.E_STATE.value, "message": str(exc)}
