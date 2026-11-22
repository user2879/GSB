from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DiagnosticError(Exception):
    code: str
    message: str
    stage: str = ""
    format_version: str = ""
    clip_id: str = ""
    track: str = ""
    segment_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    @property
    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.stage,
            self.format_version,
            self.clip_id,
            self.segment_index if self.segment_index is not None else -1,
            self.track,
            self.code,
            self.message,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "format_version": self.format_version,
            "clip_id": self.clip_id,
            "track": self.track,
            "sort_key": list(self.sort_key),
        }
        if self.segment_index is not None:
            payload["segment_index"] = self.segment_index
        if self.details:
            payload["details"] = self.details
        return payload


def sort_errors(errors: list[DiagnosticError]) -> list[DiagnosticError]:
    return sorted(errors, key=lambda item: item.sort_key)


def error_to_dict(error: DiagnosticError | dict[str, Any]) -> dict[str, Any]:
    if isinstance(error, DiagnosticError):
        return error.to_dict()
    return dict(error)
