"""Business errors and stable failure records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_RECORD_INVALID = "E_RECORD_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_SNAPSHOT_INVALID = "E_SNAPSHOT_INVALID"
E_INPUT_NOT_FOUND = "E_INPUT_NOT_FOUND"


@dataclass(frozen=True)
class FailureRecord:
    code: str
    source: str
    line: int | None
    field: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "source": self.source,
            "line": self.line,
            "field": self.field,
            "message": self.message,
        }

    @property
    def stable_key(self) -> tuple[str, int, str, str, str]:
        return (
            self.source,
            -1 if self.line is None else self.line,
            self.field,
            self.message,
            self.code,
        )


class WorkbenchError(Exception):
    """Base class for errors that carry a business error code."""

    code = "E_WORKBENCH"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ConfigInvalid(WorkbenchError):
    code = E_CONFIG_INVALID


class ConfigMismatch(WorkbenchError):
    code = E_CONFIG_MISMATCH


class SnapshotInvalid(WorkbenchError):
    code = E_SNAPSHOT_INVALID


class InputNotFound(WorkbenchError):
    code = E_INPUT_NOT_FOUND


def sorted_failures(failures: list[FailureRecord]) -> list[FailureRecord]:
    return sorted(failures, key=lambda failure: failure.stable_key)
