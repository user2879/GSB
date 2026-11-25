"""Business error codes and structured validation issues.

Every loader/parser failure is captured as an :class:`Issue` instead of being
raised, so a single bad entry never aborts the processing of the remaining
entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Code(str, Enum):
    """Stable business error codes used across loaders, scheduler and reports."""

    E_SCHEMA = "E_SCHEMA"
    E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
    E_MISSING_FIELD = "E_MISSING_FIELD"
    E_DUPLICATE = "E_DUPLICATE"
    E_CYCLE = "E_CYCLE"
    E_REF = "E_REF"
    E_QUOTA = "E_OPERATOR_QUOTA_EXCEEDED"
    E_BUDGET = "E_POINT_BUDGET_PENDING"


@dataclass(frozen=True)
class Issue:
    """A single non-fatal diagnostic attached to a concrete entry."""

    code: Code
    location: str
    message: str
    entry: dict | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code.value,
            "location": self.location,
            "message": self.message,
            "entry": self.entry,
        }


@dataclass
class IssueLog:
    """Collects issues without interrupting sibling entries."""

    issues: list[Issue] = field(default_factory=list)

    def add(self, code: Code, location: str, message: str, entry: dict | None = None) -> Issue:
        issue = Issue(code=code, location=location, message=message, entry=entry)
        self.issues.append(issue)
        return issue

    def count_by_code(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.code.value] = counts.get(issue.code.value, 0) + 1
        return counts

    def by_code(self, code: Code) -> list[Issue]:
        return [issue for issue in self.issues if issue.code == code]

    def to_dict(self) -> dict:
        return {
            "total": len(self.issues),
            "count_by_code": self.count_by_code(),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def extend(self, other: "IssueLog") -> None:
        self.issues.extend(other.issues)
