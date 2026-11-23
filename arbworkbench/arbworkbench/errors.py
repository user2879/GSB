"""Structured error hierarchy for the arbitration workbench.

Error codes
-----------
E_SPEC    : input specification (environments / observations / views) is invalid.
E_GRID    : arbitration scale / rule grid is out of the supported integer range.
E_VERSION : persisted snapshot or ledger version or content digest does not match.
E_REPLAY  : re-arbitration after recovery disagrees with a persisted conclusion.
"""
from __future__ import annotations


class WorkbenchError(Exception):
    code = "E_BASE"

    def __init__(self, message: str, location: str | None = None, detail: dict | None = None):
        super().__init__(message)
        self.message = message
        self.location = location
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "detail": self.detail,
        }


class SpecError(WorkbenchError):
    code = "E_SPEC"


class GridError(WorkbenchError):
    code = "E_GRID"


class VersionError(WorkbenchError):
    code = "E_VERSION"


class ReplayError(WorkbenchError):
    code = "E_REPLAY"
