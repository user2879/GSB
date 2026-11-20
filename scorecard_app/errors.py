from __future__ import annotations

from dataclasses import dataclass
from typing import Any


E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SNAPSHOT_IO = "E_SNAPSHOT_IO"
E_COLUMN_BINNING_FAILED = "E_COLUMN_BINNING_FAILED"
E_COLUMN_WOE_FAILED = "E_COLUMN_WOE_FAILED"
E_MONOTONIC_FAILED = "E_MONOTONIC_FAILED"
E_MONOTONIC_UNRESOLVED = "E_MONOTONIC_UNRESOLVED"
E_EVALUATION_FAILED = "E_EVALUATION_FAILED"


@dataclass(frozen=True)
class Failure:
    column: str
    stage: str
    code: str
    exception_type: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "stage": self.stage,
            "code": self.code,
            "exception_type": self.exception_type,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Failure":
        return cls(
            column=str(value.get("column", "")),
            stage=str(value.get("stage", "")),
            code=str(value.get("code", "")),
            exception_type=(str(value["exception_type"]) if value.get("exception_type") else None),
            message=str(value.get("message", "")),
        )


def failure_sort_key(failure: Failure) -> tuple[str, str, str, str, str]:
    return (
        failure.column,
        failure.stage,
        failure.code,
        failure.exception_type or "",
        failure.message,
    )


def sort_failures(failures: list[Failure]) -> list[Failure]:
    # A failure can be discovered by more than one pass after a partial restart.
    unique: dict[tuple[str, str, str, str | None, str], Failure] = {}
    for item in failures:
        unique[(item.column, item.stage, item.code, item.exception_type, item.message)] = item
    return sorted(unique.values(), key=failure_sort_key)


class ScorecardError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str = "runtime",
        column: str = "",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.column = column

    def as_failure(self) -> Failure:
        return Failure(
            column=self.column,
            stage=self.stage,
            code=self.code,
            exception_type=type(self).__name__ if type(self) is not ScorecardError else None,
            message=self.message,
        )


class SnapshotError(ScorecardError):
    def __init__(self, message: str) -> None:
        super().__init__(E_SNAPSHOT_STALE, message, stage="snapshot")
