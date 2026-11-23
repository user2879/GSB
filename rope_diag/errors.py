from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ErrorCode:
    CONFIG_INVALID = "E_CONFIG_INVALID"
    SAMPLE_INVALID = "E_SAMPLE_INVALID"
    NO_EXTRAPOLATION_SAMPLE = "E_NO_EXTRAPOLATION_SAMPLE"
    POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
    INJECTED_FAULT = "E_INJECTED_FAULT"
    IO_ERROR = "E_IO_ERROR"
    CACHE_CORRUPT = "E_CACHE_CORRUPT"
    INTERNAL_ERROR = "E_INTERNAL_ERROR"


@dataclass(frozen=True)
class Failure:
    """A deterministic, JSON-serializable validation or execution failure."""

    code: str
    message: str
    sample_id: str = ""
    stage: str = ""
    line: int | None = None
    injection_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_id": self.sample_id,
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
        }
        if self.line is not None:
            payload["line"] = self.line
        if self.injection_id:
            payload["injection_id"] = self.injection_id
        if self.details:
            payload["details"] = self.details
        return payload

    @property
    def sort_key(self) -> tuple[str, str, str, int, str]:
        return (
            self.sample_id,
            self.code,
            self.stage,
            self.line if self.line is not None else -1,
            self.message,
        )


class DiagnosticError(Exception):
    """Internal exception carrying a stable diagnostic error code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        sample_id: str = "",
        stage: str = "",
        injection_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.sample_id = sample_id
        self.stage = stage
        self.injection_id = injection_id
        self.details = details or {}

    def as_failure(self, *, line: int | None = None) -> Failure:
        return Failure(
            code=self.code,
            message=self.message,
            sample_id=self.sample_id,
            stage=self.stage,
            line=line,
            injection_id=self.injection_id,
            details=self.details,
        )


class EncodingBoundaryError(DiagnosticError):
    """Raised when a parameter scan reaches a non-finite/truncated encoding."""

