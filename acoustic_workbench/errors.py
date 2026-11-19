"""Business errors used across the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class ErrorCode(str, Enum):
    AUDIO_FORMAT = "E_AUDIO_FORMAT"
    SYNC_MARK = "E_SYNC_MARK"
    DELAY_BIAS = "E_DELAY_BIAS"
    CALIB_MISMATCH = "E_CALIB_MISMATCH"
    SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    BAND_INVALID = "E_BAND_INVALID"
    FILTER_LEN = "E_FILTER_LEN"


@dataclass(frozen=True)
class ErrorEvent:
    code: ErrorCode
    stage: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "stage": self.stage,
            "message": self.message,
            "details": dict(self.details),
        }


class WorkbenchError(Exception):
    """A deterministic, reportable business error."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        stage: str = "unknown",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        self.message = message
        self.stage = stage
        self.details = dict(details or {})
        super().__init__(f"{self.code.value}: {message}")

    def to_event(self) -> ErrorEvent:
        return ErrorEvent(self.code, self.stage, self.message, self.details)

    def to_dict(self) -> dict[str, Any]:
        return self.to_event().to_dict()


def fail(
    code: ErrorCode | str,
    message: str,
    *,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    raise WorkbenchError(code, message, stage=stage, details=details)
