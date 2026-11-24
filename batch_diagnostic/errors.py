"""Business error codes used across the batch attribution diagnostic component."""

from __future__ import annotations


class DiagnosticError(Exception):
    """Base error carrying a stable business code and structured context."""

    code = "E_INTERNAL"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }


class SpecError(DiagnosticError):
    """Invalid request stream / worker declaration / budget specification."""

    code = "E_SPEC"


class FieldError(DiagnosticError):
    """Malformed receipt field or unsupported receipt format version."""

    code = "E_FIELD"


class ChainError(DiagnosticError):
    """Broken attribution dependency chain, e.g. dangling duplicate reference."""

    code = "E_CHAIN"


class ReplayError(DiagnosticError):
    """Recomputed replay diverges from the committed ledger."""

    code = "E_REPLAY"


class SnapshotError(DiagnosticError):
    """Snapshot header / content digest mismatch."""

    code = "E_SNAPSHOT"


class VerifyError(DiagnosticError):
    """Independent verifier disagrees with the committed ledger."""

    code = "E_VERIFY"


class TimeoutError_(DiagnosticError):
    """A single socket round trip exceeded its deadline."""

    code = "E_TIMEOUT"
