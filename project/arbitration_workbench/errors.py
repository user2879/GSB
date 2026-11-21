"""Typed workbench errors with stable business codes."""

from __future__ import annotations


class WorkbenchError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class RuleValidationError(WorkbenchError):
    pass


class ConfigValidationError(WorkbenchError):
    pass


class VersionValidationError(WorkbenchError):
    pass


class SnapshotStaleError(WorkbenchError):
    pass


class SnapshotInvalidError(WorkbenchError):
    pass


class InteractionPending(WorkbenchError):
    pass


class StopRequested(Exception):
    """Raised internally when a graceful stop prevents a queued ticket task."""

