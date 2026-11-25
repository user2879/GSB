"""Business codes and validation error types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_FORMAT_INVALID = "E_FORMAT_INVALID"
E_DOC_QUOTA_EXCEEDED = "E_DOC_QUOTA_EXCEEDED"
E_MATCH_BUDGET_PENDING = "E_MATCH_BUDGET_PENDING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_VERSION_DECREASED = "E_VERSION_DECREASED"
E_REVISION_CHAIN = "E_REVISION_CHAIN"
E_VALIDATION = "E_VALIDATION"
E_SIDECAR_MISMATCH = "E_SIDECAR_MISMATCH"
E_RULE_INVALID = "E_RULE_INVALID"


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    line: int | None = None
    field: str | None = None
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.line is not None:
            data["line"] = self.line
        if self.field is not None:
            data["field"] = self.field
        if self.path is not None:
            data["path"] = self.path
        return data


class PipelineValidationError(Exception):
    """Raised when one or more declarations or exchange records are invalid."""

    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(f"{issue.code}:{issue.field or issue.path or ''}" for issue in issues))


def issue_dicts(issues: list[Issue]) -> list[dict[str, Any]]:
    return [issue.as_dict() for issue in issues]
