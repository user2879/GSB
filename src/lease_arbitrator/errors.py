from __future__ import annotations

from typing import Any


class LeaseArbitratorError(Exception):
    code = "E_INTERNAL"

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
        if code:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        result = {"ok": False, "error": {"code": self.code, "message": self.message}}
        if self.path is not None:
            result["error"]["path"] = self.path
        if self.details:
            result["error"]["details"] = self.details
        return result


class SpecError(LeaseArbitratorError):
    code = "E_SPEC"


class StaleGenerationError(LeaseArbitratorError):
    code = "E_STALE"


class GenerationChainError(LeaseArbitratorError):
    code = "E_GEN"


class StateTransitionError(LeaseArbitratorError):
    code = "E_STATE"


class VerificationError(LeaseArbitratorError):
    code = "E_VERIFY"


class NodeProtocolError(LeaseArbitratorError):
    code = "E_PROTOCOL"


class NodeUnavailableError(LeaseArbitratorError):
    code = "E_NODE"
