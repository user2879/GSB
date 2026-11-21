"""Business error types and stable failure ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_SEGMENT_OVERFLOW = "E_SEGMENT_OVERFLOW"
E_FRAME_BUDGET_EXHAUSTED = "E_FRAME_BUDGET_EXHAUSTED"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_SEGMENT_FAILURE = "E_SEGMENT_FAILURE"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_STATS_EMPTY = "E_STATS_EMPTY"


@dataclass(slots=True)
class AudioFormatError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(slots=True)
class StageError(RuntimeError):
    stage: str
    code: str
    exception_type: str
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(slots=True)
class Failure:
    code: str
    stage: str
    detail: str
    segment_id: str | None = None
    segment_index: int | None = None
    path: str | None = None
    command_seq: int | None = None
    exception_type: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "detail": self.detail,
        }
        if self.segment_id is not None:
            result["segment_id"] = self.segment_id
        if self.segment_index is not None:
            result["segment_index"] = self.segment_index
        if self.path is not None:
            result["path"] = self.path
        if self.command_seq is not None:
            result["command_seq"] = self.command_seq
        if self.exception_type is not None:
            result["exception_type"] = self.exception_type
        result.update(self.extra)
        return result


_STAGE_ORDER = {
    "audio": 0,
    "analysis": 1,
    "stats": 2,
    "scheduler": 3,
    "mapping": 4,
    "synthesis": 5,
    "assembly": 6,
    "commands": 7,
    "snapshot": 8,
}


def failure_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Stable canonical order independent of worker completion order."""

    segment_index = item.get("segment_index")
    return (
        1 if segment_index is None else 0,
        -1 if segment_index is None else int(segment_index),
        _STAGE_ORDER.get(str(item.get("stage", "")), 99),
        str(item.get("path", "")),
        str(item.get("code", "")),
        "" if item.get("command_seq") is None else f"{int(item['command_seq']):020d}",
        str(item.get("segment_id", "")),
        str(item.get("detail", "")),
        str(item.get("exception_type", "")),
    )


def sorted_failures(failures: list[Failure] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    encoded = [f.as_dict() if isinstance(f, Failure) else dict(f) for f in failures]
    unique: dict[str, dict[str, Any]] = {}
    for item in encoded:
        import json

        key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        unique[key] = item
    return sorted(unique.values(), key=failure_sort_key)
