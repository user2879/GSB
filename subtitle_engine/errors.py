"""Business errors with stable E_xxx codes."""

from __future__ import annotations

from typing import Any


class EngineError(Exception):
    """An expected, machine-readable engine failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        repaired: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.repaired = repaired

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class BlockComputationCancelled(Exception):
    """Internal cancellation used when another block fails."""
