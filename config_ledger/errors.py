"""Business error types used by the local diagnostics console."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ErrorIssue:
    """One validation or replay issue with a stable location."""

    node_id: str
    field_path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {
            "node_id": self.node_id,
            "field_path": self.field_path,
            "message": self.message,
        }


class LedgerError(Exception):
    """Base class for errors carrying a stable business code."""

    code = "E_LEDGER"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class SpecValidationError(LedgerError):
    code = "E_SPEC"

    def __init__(self, issues: list[ErrorIssue], message: str | None = None):
        self.issues = tuple(issues)
        super().__init__(
            message or f"specification rejected: {len(issues)} validation issue(s)",
            details={"issues": [issue.as_dict() for issue in issues]},
        )

    def __str__(self) -> str:
        if not self.issues:
            return self.message
        details = "; ".join(
            f"{issue.node_id} {issue.field_path}: {issue.message}"
            for issue in self.issues[:5]
        )
        if len(self.issues) > 5:
            details += f"; ... and {len(self.issues) - 5} more"
        return f"{self.message}: {details}"


class SnapshotError(LedgerError):
    code = "E_SNAPSHOT"


class ReplayError(LedgerError):
    code = "E_REPLAY"

    def __init__(
        self,
        message: str,
        *,
        first_diff: dict[str, Any] | None = None,
        details: dict[str, Any] | None = None,
    ):
        merged = dict(details or {})
        if first_diff is not None:
            merged["first_diff"] = first_diff
        super().__init__(message, details=merged)
        self.first_diff = first_diff


@dataclass
class RecoveryInfo:
    """State recovered when a previous process did not commit a round."""

    unknown_attempts: list[dict[str, Any]] = field(default_factory=list)
    invalidated_current: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "unknown_attempts": self.unknown_attempts,
            "invalidated_current": self.invalidated_current,
        }
