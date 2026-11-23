"""Typed diagnostics raised by the gateway application."""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    """An error with a stable machine-readable code and location."""

    default_code = "E_GATEWAY"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        entry: str | None = None,
        field: str | None = None,
        request_id: int | None = None,
        backend_id: str | None = None,
        port: int | None = None,
        batch_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.entry = entry
        self.field = field
        self.request_id = request_id
        self.backend_id = backend_id
        self.port = port
        self.batch_id = batch_id
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        for key in (
            "entry",
            "field",
            "request_id",
            "backend_id",
            "port",
            "batch_id",
        ):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        if self.details:
            result["details"] = self.details
        return result


class SpecError(GatewayError):
    default_code = "E_SPEC"


class PortError(GatewayError):
    default_code = "E_PORT"


class HandshakeError(GatewayError):
    default_code = "E_HANDSHAKE"


class ReceiptError(GatewayError):
    default_code = "E_RECEIPT"


class PreviewError(GatewayError):
    default_code = "E_PREVIEW"


class ReplayError(GatewayError):
    default_code = "E_REPLAY"


class SchedulerError(GatewayError):
    default_code = "E_SCHEDULER"
