"""Domain errors carrying stable business codes."""

from __future__ import annotations

from typing import Any


class WorkbenchError(Exception):
    """Base class for errors that must be reported with a business code."""

    code = "E_WORKBENCH"

    def __init__(
        self,
        message: str,
        *,
        seq: int | None = None,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.seq = seq
        self.field = field
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "error_seq": self.seq,
            "field": self.field,
            "details": self.details,
        }


class LogValidationError(WorkbenchError):
    code = "E_LOG"


class ParameterError(WorkbenchError):
    code = "E_PARAM"


class SnapshotError(WorkbenchError):
    code = "E_SNAPSHOT"


class ReplayError(WorkbenchError):
    code = "E_REPLAY"


class RecoveryError(WorkbenchError):
    code = "E_RECOVERY"
