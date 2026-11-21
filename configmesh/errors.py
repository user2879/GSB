"""Business error codes and the unified BusinessError exception.

All platform-level failures carry a stable business code so that the CLI,
structured log chain and tests can classify them deterministically.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class ErrorCode(str, Enum):
    E_MANIFEST = "E_MANIFEST"      # manifest / registry validation failure
    E_CYCLE = "E_CYCLE"            # derived-from dependency cycle
    E_GRID = "E_GRID"              # illegal fixed-point precision parameters
    E_PORT = "E_PORT"              # node port cannot be bound / reached
    E_HANDSHAKE = "E_HANDSHAKE"    # TCP handshake failure or timeout
    E_REPLY = "E_REPLY"            # malformed / missing receipt fields
    E_CONVERGE = "E_CONVERGE"      # receipt diverges from authoritative recompute
    E_PREVIEW = "E_PREVIEW"        # UI preview cache diverges from authority
    E_REPLAY = "E_REPLAY"          # snapshot replay mismatch
    E_INTERNAL = "E_INTERNAL"      # unexpected platform error


class BusinessError(Exception):
    """Exception carrying a stable business code and structured details."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        details: Iterable[Any] | None = None,
    ) -> None:
        self.code = ErrorCode(code) if not isinstance(code, ErrorCode) else code
        self.message = message
        self.details: list[Any] = list(details or [])
        super().__init__(f"{self.code.value}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


def raise_business(code: ErrorCode | str, message: str, details: Iterable[Any] | None = None):
    raise BusinessError(code, message, details)
