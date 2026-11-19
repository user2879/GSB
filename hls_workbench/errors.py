"""Stable business error codes used by every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ERROR_CODES: dict[str, str] = {
    "E_BUDGET_DENIED": "A task admission was denied by a resource budget; it is deferred, never dropped.",
    "E_PTS_GAP": "A video/audio PTS sequence is not continuous at a segment boundary.",
    "E_TASK_DUPLICATE": "The same (profile, segment) task was submitted or committed more than once.",
    "E_RETRY_EXHAUSTED": "Retries for a transcode unit were exhausted before a healthy reassignment.",
    "E_BUFFER_UNDERRUN": "Client playback exhausted its media buffer.",
    "E_MEDIA_FORMAT": "A media container, codec, geometry, or sample format is invalid.",
    "E_DETERMINISM_MISMATCH": "Two runs with identical inputs produced different bytes.",
}


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown workbench error code: {self.code}")
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


def require(condition: bool, code: str, message: str, **details: Any) -> None:
    if not condition:
        raise WorkbenchError(code, message, details)
