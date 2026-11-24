"""Structured diagnostic exceptions used by both the CLI and the workbench."""

from __future__ import annotations

from typing import Any


class DiagnosticError(Exception):
    """Base class for errors that have a stable machine-readable code."""

    code = "E_DIAGNOSTIC"

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        if code is not None:
            self.code = code

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class SpecError(DiagnosticError):
    code = "E_SPEC"


class OrderError(DiagnosticError):
    code = "E_ORDER"


class BudgetError(DiagnosticError):
    code = "E_BUDGET"


class WatermarkError(DiagnosticError):
    code = "E_WATERMARK"


class ReplayError(DiagnosticError):
    code = "E_REPLAY"


class LedgerError(DiagnosticError):
    code = "E_LEDGER"


class SnapshotError(DiagnosticError):
    code = "E_SNAPSHOT"


class SchedulerError(DiagnosticError):
    code = "E_SCHEDULER"


class ProbeError(DiagnosticError):
    code = "E_PROBE"


class UiError(DiagnosticError):
    code = "E_UI"
