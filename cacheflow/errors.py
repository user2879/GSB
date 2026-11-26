from __future__ import annotations

from typing import Any


class CacheFlowError(Exception):
    code = "E_CACHEFLOW"

    def __init__(
        self,
        message: str,
        *,
        entry: str | int | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.entry = entry
        self.field = field
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.entry is not None:
            payload["entry"] = self.entry
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload


class SpecError(CacheFlowError):
    code = "E_SPEC"


class PortInUseError(CacheFlowError):
    code = "E_PORT"


class ProtocolTimeout(CacheFlowError):
    code = "E_TIMEOUT"


class FieldError(CacheFlowError):
    code = "E_FIELD"


class StaleGenerationError(CacheFlowError):
    code = "E_STALE"

    def __init__(self, message: str, *, current_generation: int, **kwargs: Any) -> None:
        super().__init__(message, details={"current_generation": current_generation}, **kwargs)
        self.current_generation = current_generation


class GenerationError(CacheFlowError):
    code = "E_GEN"


class DivergenceError(CacheFlowError):
    code = "E_DIVERGE"


class VerifyError(CacheFlowError):
    code = "E_VERIFY"


class NodeLifecycleError(CacheFlowError):
    pass


class UnknownRequestError(CacheFlowError):
    code = "E_UNKNOWN"
