"""Typed errors used by the CLI, index and GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AudioWorkbenchError(Exception):
    """Base error with a stable machine-readable code and JSON details."""

    message: str
    code: str = "E_WORKBENCH"
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 1

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


class AudioFormatError(AudioWorkbenchError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "E_AUDIO_FORMAT", details or {}, 2)


class FormatDriftError(AudioWorkbenchError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "E_FORMAT_DRIFT", details or {}, 3)


class DedupMergeError(AudioWorkbenchError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "E_DEDUP_MERGE", details or {}, 4)


class IndexVersionError(AudioWorkbenchError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "E_INDEX_VERSION", details or {}, 5)


class EncoderError(AudioWorkbenchError):
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, "E_ENCODER", details or {}, 6)
