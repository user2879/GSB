from __future__ import annotations


class WorkbenchError(Exception):
    """Base class carrying a stable business code."""

    def __init__(self, code: str, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ValidationError(WorkbenchError):
    """A fatal input validation error."""


class SimulatedCrash(RuntimeError):
    """Raised deliberately by the crash-recovery test hook."""

