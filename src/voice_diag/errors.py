from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_SCHEMA = "E_SCHEMA"
E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_QUOTA_EXCEEDED = "E_QUOTA_EXCEEDED"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"

BUSINESS_CODES = (
    E_SCHEMA,
    E_AUDIO_FORMAT,
    E_FIELD_TRUNCATED,
    E_QUOTA_EXCEEDED,
    E_POINT_BUDGET_PENDING,
)


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    where: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.where is not None:
            result["where"] = self.where
        if self.details:
            result["details"] = self.details
        return result


class DiagnosticError(Exception):
    """A validation/business-code failure with stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        where: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.where = where
        self.details = details or {}

    def as_issue(self) -> Issue:
        return Issue(self.code, self.message, self.where, dict(self.details))

    def as_dict(self) -> dict[str, Any]:
        return self.as_issue().as_dict()


class SchemaError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        where: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_SCHEMA, message, where=where, details=details)


class AudioFormatError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        where: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_AUDIO_FORMAT, message, where=where, details=details)


class FieldTruncatedError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        where: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_FIELD_TRUNCATED, message, where=where, details=details)


class QuotaExceededError(DiagnosticError):
    def __init__(
        self,
        message: str,
        *,
        where: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_QUOTA_EXCEEDED, message, where=where, details=details)
