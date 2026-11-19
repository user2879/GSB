"""Typed business errors used across stage boundaries."""

from __future__ import annotations

from typing import Any


class VoicecalError(Exception):
    """An error that must be reported with a stable business code."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class InjectedFault(VoicecalError):
    """Raised by an enabled deterministic fault injection."""

    def __init__(self, injection_id: str, message: str, stage: str, turn_id: str) -> None:
        super().__init__(
            "E_INJECTED_FAULT",
            message,
            injection_id=injection_id,
            stage=stage,
            turn_id=turn_id,
        )
        self.injection_id = injection_id
