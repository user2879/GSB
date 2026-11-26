"""Structured exceptions and validation issue helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    source: str = ""
    sample_id: str | None = None
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "sample_id": self.sample_id,
            "injection_id": self.injection_id,
            "details": self.details,
        }


class GsbError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        source: str = "",
        sample_id: str | None = None,
        injection_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.source = source
        self.sample_id = sample_id
        self.injection_id = injection_id
        self.details = details or {}

    def issue(self) -> ValidationIssue:
        return ValidationIssue(
            code=self.code,
            message=self.message,
            source=self.source,
            sample_id=self.sample_id,
            injection_id=self.injection_id,
            details=self.details,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.issue().to_dict()


def sort_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return sorted(
        issues,
        key=lambda item: (
            item.sample_id or "",
            item.source,
            item.code,
            item.message,
            item.injection_id or "",
        ),
    )
