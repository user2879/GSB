"""Domain error codes for the elastic-scaling workbench.

Every failure path raises :class:`WorkbenchError` with a stable error code so
that the CLI, the GUI and the structured log files can classify diagnostics
deterministically.
"""

from __future__ import annotations


class WorkbenchError(Exception):
    """Base class carrying a stable machine-readable error code."""

    code = "E_UNKNOWN"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class SchemaError(WorkbenchError):
    """The declaration JSON fails strong validation."""

    code = "E_SCHEMA"


class LengthError(WorkbenchError):
    """Per-service load series do not share the same length."""

    code = "E_LENGTH"


class DivergenceError(WorkbenchError):
    """Full and incremental estimators produced different bytes."""

    code = "E_DIVERGE"


class PreviewError(WorkbenchError):
    """Preview cache disagrees with the authoritative ledger."""

    code = "E_PREVIEW"


class LatencyError(WorkbenchError):
    """A query exceeded the declared latency budget."""

    code = "E_LATENCY"


class ReplayError(WorkbenchError):
    """Recovered replay disagrees with the committed ledger."""

    code = "E_REPLAY"


class SnapshotError(WorkbenchError):
    """A stored snapshot digest does not match its payload."""

    code = "E_SNAPSHOT"


ERROR_CODES = (
    "E_SCHEMA",
    "E_LENGTH",
    "E_DIVERGE",
    "E_PREVIEW",
    "E_LATENCY",
    "E_REPLAY",
    "E_SNAPSHOT",
)
