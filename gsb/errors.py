"""Typed errors with stable machine-readable codes."""

from __future__ import annotations

from typing import Any


E_SPEC = "E_SPEC"
E_FALSE_SKIP = "E_FALSE_SKIP"
E_REPLAY = "E_REPLAY"
E_TRANSPORT = "E_TRANSPORT"
E_TIMEOUT = "E_TIMEOUT"


class GSBError(Exception):
    """Base exception carrying a serializable diagnostic object."""

    code = "E_GSB"

    def __init__(self, message: str = "", **details: Any) -> None:
        super().__init__(message or self.code)
        self.message = message or self.code
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        payload = {"code": self.code, "message": self.message}
        payload.update(self.details)
        return payload


class SpecError(GSBError):
    code = E_SPEC

    def __init__(
        self,
        message: str,
        *,
        block_group_id: str | None = None,
        field: str | None = None,
        value: Any = None,
        **details: Any,
    ) -> None:
        payload = dict(details)
        payload.update(block_group_id=block_group_id, field=field, value=value)
        super().__init__(message, **payload)
        self.block_group_id = block_group_id
        self.field = field


SpecValidationError = SpecError
ESpecError = SpecError


class FalseSkipError(GSBError):
    code = E_FALSE_SKIP


EFalseSkipError = FalseSkipError


class ReplayError(GSBError):
    code = E_REPLAY


class TransportError(GSBError):
    code = E_TRANSPORT


class SocketTimeoutError(GSBError):
    code = E_TIMEOUT
