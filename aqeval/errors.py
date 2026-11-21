"""Business errors and stable failure records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_EVAL_FORMAT = "E_EVAL_FORMAT"
E_LEVEL_UNMAPPED = "E_LEVEL_UNMAPPED"
E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_AUDIO_MISSING = "E_AUDIO_MISSING"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_FRAME_FEATURE = "E_FRAME_FEATURE"
E_SCORE = "E_SCORE"
E_SERIALIZE = "E_SERIALIZE"


STAGE_ORDER = {
    "parse": 0,
    "audio": 1,
    "features": 2,
    "score": 3,
    "grade": 4,
    "serialize": 5,
    "conversion": 6,
    "snapshot": 7,
}


class AQevalError(Exception):
    """Base exception carrying a business code."""

    code = "E_AQEVAL"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class FormatError(AQevalError):
    code = E_EVAL_FORMAT


class LevelMapError(AQevalError):
    code = E_LEVEL_UNMAPPED


class AudioFormatError(AQevalError):
    code = E_AUDIO_FORMAT


class SnapshotStaleError(AQevalError):
    code = E_SNAPSHOT_STALE


class InterruptedRun(AQevalError):
    """Raised after a requested interruption point has been committed."""

    code = "E_RUN_INTERRUPTED"


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    file_id: str = ""
    version: int | None = None
    stage: str = "parse"
    location: str = ""
    task_key: str = ""
    frame_index: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.file_id or "\uffff",
            STAGE_ORDER.get(self.stage, 99),
            self.frame_index if self.frame_index is not None else -1,
            self.task_key,
            self.version if self.version is not None else 0,
            self.code,
            self.location,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "file_id": self.file_id,
            "stage": self.stage,
            "location": self.location,
            "task_key": self.task_key,
        }
        if self.version is not None:
            result["version"] = self.version
        if self.frame_index is not None:
            result["frame_index"] = self.frame_index
        if self.detail:
            result["detail"] = self.detail
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Failure":
        return cls(
            code=str(value["code"]),
            message=str(value["message"]),
            file_id=str(value.get("file_id", "")),
            version=value.get("version"),
            stage=str(value.get("stage", "parse")),
            location=str(value.get("location", "")),
            task_key=str(value.get("task_key", "")),
            frame_index=value.get("frame_index"),
            detail=dict(value.get("detail", {})),
        )


def sorted_failures(failures: list[Failure]) -> list[Failure]:
    return sorted(failures, key=Failure.sort_key)

